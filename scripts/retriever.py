import os
import sys
import math
import re
import unicodedata
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification


# Agregar el directorio padre al sys.path para poder importar preprocess_docs
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

from preprocess_docs import try_decode, clean_text, segment_text

def load_tokenizer_robust(model_name, local_files_only):
    """Carga tokenizer con fallbacks en cascada.

    1. AutoTokenizer (caso normal).
    2. PreTrainedTokenizerFast desde tokenizer.json local (cubre e5-small y
       paraphrase-MiniLM, cuyo tokenizer_config.json no declara
       tokenizer_class y el env no tiene sentencepiece).
    3. XLMRobertaTokenizerFast / BertTokenizerFast explícitos.
    """
    try:
        return AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
    except ValueError as e:
        if "TokenizersBackend" not in str(e):
            raise
        print(f"[WARNING] tokenizer_config inválido en '{model_name}', probando fallbacks...")
    if os.path.isdir(model_name):
        tok_json = os.path.join(model_name, "tokenizer.json")
        if os.path.exists(tok_json):
            try:
                from transformers import PreTrainedTokenizerFast
                tok = PreTrainedTokenizerFast(tokenizer_file=tok_json)
                print(f"[INFO] Tokenizer cargado desde tokenizer.json de '{model_name}'.")
                return tok
            except Exception as e:
                print(f"[WARNING] Fallback tokenizer.json falló: {e}")
    from transformers import XLMRobertaTokenizerFast, BertTokenizerFast
    for cls in (XLMRobertaTokenizerFast, BertTokenizerFast):
        try:
            return cls.from_pretrained(model_name, local_files_only=local_files_only)
        except Exception:
            continue
    raise ValueError(f"No se pudo cargar tokenizer de '{model_name}' con ninguna clase conocida.")

def tokenize(text):
    """
    Tokeniza el texto en minúsculas, elimina acentos y signos de puntuación,
    y filtra stopwords en portugués.
    """
    PORTUGUESE_STOPWORDS = {
        "a", "ao", "aos", "aquela", "aquelas", "aquele", "aqueles", "aquilo", "as", "até", "com", "como", "da", "das",
        "de", "dela", "delas", "dele", "deles", "depois", "do", "dos", "e", "ela", "elas", "ele", "eles", "em", "entre",
        "era", "eram", "essa", "essas", "esse", "esses", "esta", "estamos", "estas", "estava", "estavam", "este", "esteja",
        "estejam", "estejamos", "estes", "esteve", "estive", "estiver", "estivera", "estiveram", "estiverem", "estiveremos",
        "estiveria", "estiveriam", "estiveríamos", "estiverso", "estiveste", "estivestes", "estivemos", "estivéssemos",
        "estou", "está", "estávamos", "estão", "eu", "foi", "fomos", "for", "fora", "foram", "forem", "foremos", "foria",
        "foriam", "foríamos", "forse", "fosse", "fossem", "fôssemos", "foste", "fostes", "fui", "fôramos", "há", "hão",
        "houve", "houvemos", "houver", "houvera", "houveram", "houverei", "houverem", "houveremos", "houveria", "houveriam",
        "houveríamos", "houverá", "houverão", "houveríamos", "houvesse", "houvessem", "houvéssemos", "houveste", "houvestes",
        "houvéramos", "isso", "isto", "já", "lhe", "lhes", "mais", "mas", "me", "mesmo", "meu", "meus", "minha", "minhas",
        "muito", "na", "nas", "nem", "no", "nos", "nossa", "nossas", "nosso", "nossos", "num", "numa", "nós", "o", "os",
        "ou", "para", "pela", "pelas", "pelo", "pelos", "por", "qual", "quando", "que", "quem", "se", "seja", "sejam",
        "sejamos", "sem", "ser", "serei", "seria", "seriam", "seríamos", "será", "serão", "seu", "seus", "somos", "sou",
        "sua", "suas", "são", "só", "também", "te", "tem", "temos", "tenha", "tenham", "tenhamos", "tenho", "terei",
        "terá", "terão", "teríamos", "teve", "tinha", "tinham", "tínhamos", "tive", "tiver", "tivera", "tiveram", "tiverem",
        "tiveremos", "tiveria", "tiveriam", "tiveríamos", "tivesse", "tivessem", "tivéssemos", "tiveste", "tivestes",
        "tivemos", "tivéramos", "tu", "tua", "tuas", "tém", "tínhamos", "um", "uma", "umas", "uns", "vos", "você", "vocês",
        "vós", "à", "às", "é", "éramos"
    }
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    words = re.findall(r'\b\w+\b', text.lower())
    return [w for w in words if w not in PORTUGUESE_STOPWORDS]

class BM25Retriever:
    def __init__(self, k1=1.2, b=0.60):
        self.k1 = k1
        self.b = b
        self.documents = []  # Lista de dicts: {"context": ..., "source": ..., "subsection": ...}
        self.corpus_size = 0
        self.avg_doc_len = 0.0
        self.doc_lens = []
        self.doc_term_freqs = []  # Lista de dicts con frecuencias de términos para cada doc
        self.idf = {}  # Term -> IDF

    def fit_from_info_dir(self, info_dir=None):
        """
        Segmenta y lee todos los archivos de la carpeta data/documents/ y entrena el modelo BM25.
        """
        if info_dir is None:
            info_dir = os.path.join(PARENT_DIR, "data", "documents")
            
        print(f"Indexando documentos de la carpeta '{info_dir}'...")
        raw_documents = []
        
        for root, dirs, files in os.walk(info_dir):
            for file in files:
                if file.endswith('.txt'):
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, info_dir)
                    
                    content, enc = try_decode(file_path)
                    cleaned = clean_text(content)
                    segments = segment_text(file, cleaned)
                    
                    for seg in segments:
                        raw_documents.append({
                            "context": seg["context"].strip(),
                            "source": rel_path,
                            "subsection": seg["subsection"]
                        })
                        
        print(f"Total de fragmentos segmentados para RAG: {len(raw_documents)}")
        self.fit(raw_documents)

    def fit(self, documents):
        """
        Entrena el modelo BM25 con una lista de diccionarios de documentos.
        """
        self.documents = documents
        self.corpus_size = len(documents)
        
        # Tokenizar todos los documentos
        tokenized_docs = [tokenize(doc["context"]) for doc in documents]
        self.doc_lens = [len(doc) for doc in tokenized_docs]
        self.avg_doc_len = sum(self.doc_lens) / max(1, self.corpus_size)
        
        # Calcular frecuencias de términos en cada documento y la cantidad de documentos por término
        doc_counts = {}  # Term -> cantidad de documentos que lo contienen
        
        for doc_tokens in tokenized_docs:
            term_freq = {}
            for token in doc_tokens:
                term_freq[token] = term_freq.get(token, 0) + 1
            self.doc_term_freqs.append(term_freq)
            
            for token in term_freq:
                doc_counts[token] = doc_counts.get(token, 0) + 1
                
        # Calcular IDF para cada término usando la fórmula BM25 standard
        for term, count in doc_counts.items():
            # Evitar IDF negativo limitando con un mínimo de 0.0001
            val = (self.corpus_size - count + 0.5) / (count + 0.5)
            self.idf[term] = math.log(max(0.0001, val) + 1.0)
            
        print("Indexación completada con éxito.")

    def retrieve(self, query, k=3):
        """
        Calcula las puntuaciones de BM25 para la consulta dada y retorna los top-K documentos.
        """
        query_tokens = tokenize(query)
        scores = []
        
        for idx in range(self.corpus_size):
            score = 0.0
            doc_len = self.doc_lens[idx]
            term_freq = self.doc_term_freqs[idx]
            
            for token in query_tokens:
                if token in term_freq:
                    tf = term_freq[token]
                    idf_val = self.idf.get(token, 0.0)
                    
                    # Fórmula de score de término en BM25
                    numerator = tf * (self.k1 + 1)
                    denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / max(1.0, self.avg_doc_len))
                    score += idf_val * (numerator / denominator)
                    
            scores.append((score, idx))
            
        # Ordenar por score descendente
        scores.sort(key=lambda x: x[0], reverse=True)
        
        results = []
        for i in range(min(k, len(scores))):
            score, idx = scores[i]
            doc = self.documents[idx].copy()
            doc["score"] = score
            results.append(doc)
            
        return results

class DenseRetriever:
    def __init__(self, model_name="intfloat/multilingual-e5-small", device=None):
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        print(f"Loading Dense Retriever model '{model_name}' on {self.device}...")
        
        # Check if model_name is a local path
        is_local = os.path.exists(model_name)
        self.tokenizer = load_tokenizer_robust(model_name, local_files_only=is_local)
        self.model = AutoModel.from_pretrained(model_name, local_files_only=is_local).to(self.device)
        self.model.eval()
        self.doc_embeddings = None
        self.documents = []
        self.is_e5 = "e5" in model_name.lower()
        
    def fit(self, documents):
        self.documents = documents
        
        # Intentar cargar desde la caché para fomentar Green AI (evitar cómputos redundantes)
        import hashlib
        # Crear firma única del corpus basada en los textos y cantidad de documentos
        corpus_signature = "".join([doc["context"] for doc in documents])
        corpus_hash = hashlib.md5(corpus_signature.encode("utf-8")).hexdigest()
        clean_model_name = "".join([c if c.isalnum() else "_" for c in self.model_name])
        
        cache_dir = os.path.join(PARENT_DIR, "embeddings_cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f"dense_embeddings_{clean_model_name}_{corpus_hash}.pt")
        
        if os.path.exists(cache_file):
            print(f"Cargando embeddings indexados desde la caché en: {cache_file} (Green AI)")
            try:
                cached_data = torch.load(cache_file, map_location="cpu")
                self.doc_embeddings = cached_data["embeddings"].to(self.device)
                self.documents = cached_data["documents"]
                print("Indexación de embeddings completada desde caché.")
                return
            except Exception as e:
                print(f"[WARNING] No se pudo cargar la caché de embeddings: {e}. Re-calculando...")
        
        embeddings = []
        batch_size = 32
        
        # Codificar todos los documentos
        for i in range(0, len(documents), batch_size):
            if self.is_e5:
                batch = ["passage: " + doc["context"] for doc in documents[i:i+batch_size]]
            else:
                batch = [doc["context"] for doc in documents[i:i+batch_size]]
            inputs = self.tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Mean pooling
                attention_mask = inputs["attention_mask"]
                token_embeddings = outputs.last_hidden_state
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                batch_embeddings = sum_embeddings / sum_mask
                embeddings.append(batch_embeddings.cpu())
                
        self.doc_embeddings = torch.cat(embeddings, dim=0).to(self.device)
        # Normalizar para similitud de coseno
        self.doc_embeddings = self.doc_embeddings / self.doc_embeddings.norm(dim=1, keepdim=True)
        print("Dense document indexing completed.")
        
        # Guardar en caché para futuras ejecuciones
        try:
            torch.save({
                "embeddings": self.doc_embeddings.cpu(),
                "documents": self.documents
            }, cache_file)
            print(f"Embeddings indexados y guardados en caché local: {cache_file} (Green AI)")
        except Exception as e:
            print(f"[WARNING] No se pudo guardar la caché de embeddings: {e}")
        
    def retrieve(self, query, k=3):
        q_text = "query: " + query if self.is_e5 else query
        inputs = self.tokenizer([q_text], padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Mean pooling
            attention_mask = inputs["attention_mask"]
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            query_embedding = sum_embeddings / sum_mask
            query_embedding = query_embedding / query_embedding.norm(dim=1, keepdim=True)
            
        # Similitud de coseno
        scores = torch.mm(self.doc_embeddings, query_embedding.T).squeeze().tolist()
        if not isinstance(scores, list):
            scores = [scores]
            
        results = []
        for idx, score in enumerate(scores):
            results.append((score, idx))
        results.sort(key=lambda x: x[0], reverse=True)
        
        ret_docs = []
        for i in range(min(k, len(results))):
            score, idx = results[i]
            doc = self.documents[idx].copy()
            doc["score"] = score
            ret_docs.append(doc)
        return ret_docs

class HybridRetriever:
    def __init__(self, bm25_retriever, dense_retriever, alpha=0.5):
        self.bm25 = bm25_retriever
        self.dense = dense_retriever
        self.alpha = alpha
        self.documents = []
        
    def fit_from_info_dir(self, info_dir=None):
        self.bm25.fit_from_info_dir(info_dir)
        self.dense.fit(self.bm25.documents)
        self.documents = self.bm25.documents
        
    def fit(self, documents):
        self.bm25.fit(documents)
        self.dense.fit(documents)
        self.documents = documents
        
    def retrieve(self, query, k=3):
        bm25_res = self.bm25.retrieve(query, k=max(50, k*3))
        dense_res = self.dense.retrieve(query, k=max(50, k*3))
        
        bm25_max = max([doc["score"] for doc in bm25_res]) if bm25_res else 1.0
        bm25_min = min([doc["score"] for doc in bm25_res]) if bm25_res else 0.0
        dense_max = max([doc["score"] for doc in dense_res]) if dense_res else 1.0
        dense_min = min([doc["score"] for doc in dense_res]) if dense_res else 0.0
        
        bm25_dict = {}
        for doc in bm25_res:
            norm_score = (doc["score"] - bm25_min) / max(1e-9, bm25_max - bm25_min)
            bm25_dict[doc["context"]] = (norm_score, doc)
            
        dense_dict = {}
        for doc in dense_res:
            norm_score = (doc["score"] - dense_min) / max(1e-9, dense_max - dense_min)
            dense_dict[doc["context"]] = (norm_score, doc)
            
        combined_scores = []
        all_contexts = set(list(bm25_dict.keys()) + list(dense_dict.keys()))
        for ctx in all_contexts:
            bm25_score, doc_obj = bm25_dict.get(ctx, (0.0, None))
            dense_score, dense_doc = dense_dict.get(ctx, (0.0, None))
            if doc_obj is None:
                doc_obj = dense_doc
            combined_score = self.alpha * bm25_score + (1.0 - self.alpha) * dense_score
            combined_scores.append((combined_score, doc_obj))
            
        combined_scores.sort(key=lambda x: x[0], reverse=True)
        
        results = []
        for i in range(min(k, len(combined_scores))):
            score, doc = combined_scores[i]
            doc = doc.copy()
            doc["score"] = score
            results.append(doc)
        return results


class AdvancedRetriever:
    def __init__(self, base_retriever, device=None, reranker_model_path=None, spacy_model="pt_core_news_sm",
                 use_hyde=False, use_multi_query=False, use_reranking=False, use_graph=False, generate_fn=None):
        self.base = base_retriever
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        self.documents = base_retriever.documents
        self.use_hyde = use_hyde
        self.use_multi_query = use_multi_query
        self.use_reranking = use_reranking
        self.use_graph = use_graph
        self.generate_fn = generate_fn
        
        # Load Cross-Encoder if available
        self.reranker_model = None
        self.reranker_tokenizer = None
        if reranker_model_path:
            is_local = os.path.exists(reranker_model_path)
            try:
                print(f"Loading Cross-Encoder Reranker from '{reranker_model_path}' (local={is_local})...")
                self.reranker_tokenizer = load_tokenizer_robust(reranker_model_path, local_files_only=is_local)
                self.reranker_model = AutoModelForSequenceClassification.from_pretrained(reranker_model_path, local_files_only=is_local).to(self.device)
                self.reranker_model.eval()
            except Exception as e:
                print(f"[WARNING] Could not load Cross-Encoder: {e}")
                
        # Load spaCy and networkx for Knowledge Graph
        self.graph = None
        self.spacy_nlp = None
        try:
            import networkx as nx
            import spacy
            self.spacy_nlp = spacy.load(spacy_model)
            print("spaCy and NetworkX loaded for Knowledge Graph.")
        except Exception as e:
            print(f"[INFO] spaCy/NetworkX not available. Knowledge Graph will be skipped.")
            
        # Build Knowledge Graph immediately if base documents exist and graph technique is active
        if self.documents and self.spacy_nlp is not None:
            self.fit_graph(self.documents)
            
    def fit_from_info_dir(self, info_dir=None):
        self.base.fit_from_info_dir(info_dir)
        self.documents = self.base.documents
        if self.spacy_nlp is not None:
            self.fit_graph(self.documents)
            
    def fit(self, documents):
        self.base.fit(documents)
        self.documents = self.base.documents
        if self.spacy_nlp is not None:
            self.fit_graph(documents)

    def fit_graph(self, documents):
        try:
            import networkx as nx
            self.graph = nx.Graph()
            print("Building Knowledge Graph from documents...")
            for doc in documents:
                text = doc["context"]
                spacy_doc = self.spacy_nlp(text)
                # Extraer entidades como nodos
                entities = [ent.text.strip().lower() for ent in spacy_doc.ents if len(ent.text.strip()) > 2]
                # Limitar a entidades únicas para este documento
                entities = list(set(entities))
                
                # Añadir nodos y aristas entre entidades que coocurren en el mismo documento
                for i in range(len(entities)):
                    ent1 = entities[i]
                    if not self.graph.has_node(ent1):
                        self.graph.add_node(ent1, documents=[])
                    self.graph.nodes[ent1]["documents"].append(doc)
                    
                    for j in range(i + 1, len(entities)):
                        ent2 = entities[j]
                        if self.graph.has_edge(ent1, ent2):
                            self.graph[ent1][ent2]["weight"] += 1
                        else:
                            self.graph.add_edge(ent1, ent2, weight=1)
            print(f"Knowledge Graph built with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges.")
        except Exception as e:
            print(f"[WARNING] Error fitting Knowledge Graph: {e}")

    def query_graph(self, query):
        if self.graph is None or self.spacy_nlp is None:
            return ""
        try:
            spacy_query = self.spacy_nlp(query)
            query_entities = [ent.text.strip().lower() for ent in spacy_query.ents if len(ent.text.strip()) > 2]
            if not query_entities:
                # Fallback simple por palabras clave si no hay entidades reconocidas
                query_entities = [token.text.strip().lower() for token in spacy_query if token.pos_ in ["PROPN", "NOUN"] and len(token.text.strip()) > 2]
                
            related_entities = set()
            matched_nodes = []
            for ent in query_entities:
                # Búsqueda exacta de entidad en los nodos del grafo
                best_match = None
                if ent in self.graph:
                    best_match = ent
                else:
                    # Búsqueda difusa simple
                    for node in self.graph.nodes:
                        if ent in node or node in ent:
                            best_match = node
                            break
                if best_match:
                    matched_nodes.append(best_match)
                    related_entities.add(best_match)
                    # Traer vecinos de primer grado
                    for neighbor in self.graph.neighbors(best_match):
                        related_entities.add(neighbor)
                        
            if not related_entities:
                return ""
                
            # Recopilar información del grafo de las entidades relacionadas
            graph_snippets = []
            seen_edges = set()
            for ent in matched_nodes:
                # Conexiones en el grafo
                for neighbor in self.graph.neighbors(ent):
                    edge = tuple(sorted([ent, neighbor]))
                    if edge not in seen_edges:
                        seen_edges.add(edge)
                        weight = self.graph[ent][neighbor]["weight"]
                        graph_snippets.append(f"Relação: {ent.title()} <-> {neighbor.title()} (frequência: {weight})")
                        
            if graph_snippets:
                context_str = "\nContexto de Grafo de Conhecimento Relacionado:\n" + "\n".join(graph_snippets[:5])
                return context_str
        except Exception as e:
            print(f"[WARNING] Error querying Knowledge Graph: {e}")
        return ""

    def retrieve(self, query, k=3, use_hyde=None, use_multi_query=None, use_reranking=None, use_graph=None, generate_fn=None):
        # Resolver flags con fallback a los de instancia
        u_hyde = use_hyde if use_hyde is not None else self.use_hyde
        u_mq = use_multi_query if use_multi_query is not None else self.use_multi_query
        u_rerank = use_reranking if use_reranking is not None else self.use_reranking
        u_graph = use_graph if use_graph is not None else self.use_graph
        gen_fn = generate_fn if generate_fn is not None else self.generate_fn
        
        search_q = query
        
        # 1. HyDE
        if u_hyde and gen_fn:
            prompt = f"Escreva um parágrafo de 3 frases em português que responderia: {query}\nApenas o parágrafo:"
            try:
                hypothetical = gen_fn(prompt)
                if hypothetical.strip():
                    search_q = hypothetical
                    print(f"[HyDE] Usando query hipotética: '{search_q[:80]}...'")
            except Exception as e:
                print(f"[WARNING] Error running HyDE query generation: {e}")
                
        # 2. Multi-Query
        candidate_docs = []
        if u_mq and gen_fn:
            try:
                variations_prompt = f"""Gere 3 variações diferentes desta pergunta em português para busca em documentos.
Retorne apenas as 3 perguntas, uma por linha, sem numeração:
Pergunta original: {search_q}"""
                variations_text = gen_fn(variations_prompt)
                queries = [search_q] + [v.strip() for v in variations_text.split("\n") if v.strip()][:3]
                print(f"[Multi-Query] Queries geradas: {queries}")
                
                seen = set()
                retrieve_k = max(k * 2, 8)
                for q in queries:
                    docs = self.base.retrieve(q, k=retrieve_k)
                    for doc in docs:
                        key = doc["context"][:80]
                        if key not in seen:
                            seen.add(key)
                            candidate_docs.append(doc)
            except Exception as e:
                print(f"[WARNING] Error during Multi-Query: {e}")
                candidate_docs = self.base.retrieve(search_q, k=max(k * 4, 16) if u_rerank else k)
        else:
            candidate_docs = self.base.retrieve(search_q, k=max(k * 4, 16) if u_rerank else k)
            
        # 3. Reranking
        final_docs = []
        if u_rerank and self.reranker_model is not None and candidate_docs:
            try:
                pairs = [[query, doc["context"]] for doc in candidate_docs]
                inputs = self.reranker_tokenizer(pairs, padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self.reranker_model(**inputs)
                    logits = outputs.logits
                    if logits.shape[1] == 1:
                        scores = logits.squeeze(-1).tolist()
                    else:
                        scores = logits[:, 1].tolist()
                        
                if not isinstance(scores, list):
                    scores = [scores]
                    
                ranked = sorted(zip(scores, candidate_docs), key=lambda x: x[0], reverse=True)
                final_docs = [doc for _, doc in ranked[:k]]
                print(f"[Reranking] Reranked {len(candidate_docs)} docs to top-{k}.")
            except Exception as e:
                print(f"[WARNING] Error during reranking: {e}")
                final_docs = candidate_docs[:k]
        else:
            final_docs = candidate_docs[:k]
            
        # 4. Graph Context
        graph_context = ""
        if u_graph and self.graph is not None:
            graph_context = self.query_graph(query)
            
        class DocumentList(list):
            def __init__(self, items, graph_context=""):
                super().__init__(items)
                self.graph_context = graph_context
                
        return DocumentList(final_docs, graph_context)


if __name__ == "__main__":
    # Prueba rápida de funcionamiento
    retriever = BM25Retriever()
    info_path = os.path.join(PARENT_DIR, "data", "documents")
    if os.path.exists(info_path):
        retriever.fit_from_info_dir(info_path)
        q = "Qual é o período de inscrições para alunos especiais?"
        res = retriever.retrieve(q, k=2)
        print("\nPrueba de consulta:")
        for idx, doc in enumerate(res):
            print(f"[{idx+1}] Score: {doc['score']:.4f} | Fuente: {doc['source']} | Contexto (resumen): {doc['context'][:120]}...")
    else:
        print(f"No se pudo encontrar la carpeta 'data/documents' en la ruta especificada: {info_path}")
