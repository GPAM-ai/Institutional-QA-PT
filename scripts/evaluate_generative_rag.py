import os
import sys
import json
import time
import argparse
import random
from pathlib import Path

# Semilla global para reproducibilidad
SEED = 42

# Obtener rutas relativas
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from retriever import BM25Retriever, DenseRetriever, HybridRetriever
from train_unifesp_qa import compute_exact, compute_f1, compute_rouge, compute_containment, compute_token_recall, compute_k_precision, list_to_hf_dataset

MODEL_MAPPING = {
    "tucano": "TucanoBR/tucano-1b1-instruct",
    "qwen_1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen_0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen_3b": "Qwen/Qwen2.5-3B-Instruct",
    "llama_1b": "meta-llama/Llama-3.2-1B-Instruct",
    "llama_3b": "meta-llama/Llama-3.2-3B-Instruct",
    "gemma_2b": "google/gemma-2-2b-it",
    "phi_4_mini": "microsoft/Phi-4-mini-instruct",
    "smollm2": "HuggingFaceTB/SmolLM2-1.7B-Instruct"
}

# Configurar token de Hugging Face
HF_TOKEN = os.environ.get("HF_TOKEN", "")

def _maybe_patch_partial_rotary(model):
    """Parche para Phi-4-mini en transformers 4.48: su config usa
    partial_rotary_factor=0.75 (rota 96 de 128 dims), pero el attention de
    Phi3 en 4.48 aplica rotary al head completo -> mismatch 128 vs 96.
    Se rota el prefijo y el resto pasa intacto (convencion HF). Idempotente."""
    cfg = getattr(model, "config", None)
    if cfg is None or getattr(cfg, "model_type", "") != "phi3":
        return model
    try:
        prf = float(getattr(cfg, "partial_rotary_factor", 1.0) or 1.0)
    except (TypeError, ValueError):
        return model
    if prf >= 1.0:
        return model
    import transformers.models.phi3.modeling_phi3 as phi3_mod
    if getattr(phi3_mod, "_partial_rotary_patched", False):
        return model
    import inspect as _inspect
    _orig_apply = phi3_mod.apply_rotary_pos_emb
    try:
        _takes_pid = "position_ids" in _inspect.signature(_orig_apply).parameters
    except (TypeError, ValueError):
        _takes_pid = True

    def _partial_apply(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        rotary_ndims = cos.shape[-1]
        qr, kr = q[..., :rotary_ndims], k[..., :rotary_ndims]
        if _takes_pid:
            q_emb, k_emb = _orig_apply(qr, kr, cos, sin, position_ids, unsqueeze_dim)
        else:
            q_emb, k_emb = _orig_apply(qr, kr, cos, sin, unsqueeze_dim)
        q_out = torch.cat([q_emb, q[..., rotary_ndims:]], dim=-1)
        k_out = torch.cat([k_emb, k[..., rotary_ndims:]], dim=-1)
        return q_out, k_out

    phi3_mod.apply_rotary_pos_emb = _partial_apply
    phi3_mod._partial_rotary_patched = True
    print("[INFO] Partial rotary patch aplicado a modeling_phi3 (Phi-4-mini).")
    return model


def load_causal_lm_robust(model_name, torch_dtype, device_map, token, **extra):
    """Carga Causal LM tolerando rope_scaling incompatible (phi-4-mini + transformers 4.48).

    El config cacheado de Phi-4-mini-instruct trae longrope con short_factor
    de 48 y el Phi3Config del env exige 64. Como el kwarg rope_scaling=None no
    sobrescribe el archivo en esta versión, se reconstruye el config desde el
    config.json cacheado sin rope_scaling (para contextos RAG cortos el rope
    por defecto basta) y se carga el modelo con config explícito.
    """
    try:
        return _maybe_patch_partial_rotary(AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch_dtype, device_map=device_map, token=token, **extra))
    except ValueError as e:
        if "rope_scaling" not in str(e):
            raise
        print(f"[WARNING] rope_scaling incompatible en '{model_name}'. Reconstruyendo config sin rope_scaling...")
        import json as _json
        from huggingface_hub import hf_hub_download
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING
        raw_path = hf_hub_download(model_name, "config.json", token=token, local_files_only=True)
        with open(raw_path) as _f:
            cfg_dict = _json.load(_f)
        cfg_dict.pop("rope_scaling", None)
        config = CONFIG_MAPPING[cfg_dict["model_type"]].from_dict(cfg_dict)
        return _maybe_patch_partial_rotary(AutoModelForCausalLM.from_pretrained(
            model_name, config=config, torch_dtype=torch_dtype, device_map=device_map,
            token=token, **extra))

def clean_llm_answer(answer_text):
    import re
    # Eliminar prefijos de respuesta comunes y aclaraciones típicas en portugués/español
    prefixes = [
        r"^a resposta literal é\s*:",
        r"^o trecho literal do contexto é\s*:",
        r"^a resposta é\s*:",
        r"^resposta\s*:",
        r"^o trecho correspondente é\s*:",
        r"^segundo o contexto\s*,",
        r"^de acordo com o contexto\s*,"
    ]
    cleaned = answer_text.strip()
    for pattern in prefixes:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()
    
    # Quitar comillas iniciales y finales si el modelo las agregó
    if (cleaned.startswith('"') and cleaned.endswith('"')) or (cleaned.startswith("'") and cleaned.endswith("'")):
        cleaned = cleaned[1:-1].strip()
    return cleaned

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluación de pipeline RAG Generativo (LLMs de Bajo Costo).")
    parser.add_argument("--model_type", type=str, required=True, choices=list(MODEL_MAPPING.keys()),
                        help="Modelo LLM de lectura (tucano, qwen, llama_1b, llama_3b o gemma_2b).")
    parser.add_argument("--k", type=int, default=1, help="Número de documentos a recuperar (Top-K).")
    parser.add_argument("--no_few_shot", action="store_true", help="Desactiva ejemplos few-shot para la inferencia.")
    parser.add_argument("--context_max_chars", type=int, default=3000, help="Longitud máxima en caracteres del contexto.")
    parser.add_argument("--retriever_type", type=str, default="hybrid", choices=["bm25", "dense", "hybrid"],
                        help="Tipo de recuperador a usar.")
    parser.add_argument("--dense_model", type=str, default="models/multilingual-e5-small",
                        help="Ruta o nombre del modelo de embedding para Dense/Hybrid retriever.")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Alpha para el recuperador híbrido (alpha * BM25 + (1-alpha) * Dense).")
    parser.add_argument("--quantization", type=str, default="none", choices=["none", "4bit", "8bit"],
                        help="Tipo de cuantización para el LLM (none, 4bit, 8bit) - Green AI.")
    parser.add_argument("--dataset_file", type=str, default=None, help="Ruta del archivo de dataset personalizado.")
    parser.add_argument("--eval_full", action="store_true", help="Evaluar sobre todo el dataset sin split.")
    parser.add_argument("--only_main", action="store_true", help="Ignorar variantes de preguntas, usar solo la principal.")
    parser.add_argument("--use_hyde", action="store_true", help="Usar HyDE para recuperación avanzada.")
    parser.add_argument("--use_multi_query", action="store_true", help="Usar Multi-Query para recuperación avanzada.")
    parser.add_argument("--use_reranking", action="store_true", help="Usar Cross-Encoder Reranking.")
    parser.add_argument("--use_graph", action="store_true", help="Usar Knowledge Graph.")
    parser.add_argument("--peft_model_path", type=str, default=None, help="Ruta al adaptador PEFT (LoRA/QLoRA) para cargar sobre el LLM.")
    parser.add_argument("--gold_context", action="store_true",
                        help="Modo oracle: el lector recibe el pasaje gold del record en vez del contexto recuperado. El retriever se sigue ejecutando en la seccion [1/2] para medir Hit@k diagnostico, pero sus pasajes no alimentan al lector.")
    parser.add_argument("--prompt_style", type=str, default="literal", choices=["literal", "natural"],
                        help="literal: exige trecho exacto (comportamiento extractivo); natural: respuesta libre y concisa con base en el contexto.")
    parser.add_argument("--output_dir", type=str, default=None, help="Diretório para salvar uma nova avaliação sem sobrescrever resultados históricos.")
    return parser.parse_args()

def main():
    args = parse_args()
    
    random.seed(SEED)
    torch.manual_seed(SEED)

    # Green AI: VRAM base antes de cargar modelos (el pico de inferencia se
    # calcula al final como max_memory_allocated - initial_vram)
    initial_vram = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    
    model_name = MODEL_MAPPING[args.model_type]
    print("=" * 60)
    print(f"EVALUACIÓN RAG GENERATIVO: Lector: {args.model_type} ({model_name}) | Top-K: {args.k}")
    print("=" * 60)

    # 1. Cargar el dataset de prueba (el mismo 20% que el script de entrenamiento o el dataset personalizado)
    dataset_file = args.dataset_file if args.dataset_file else os.path.join(PARENT_DIR, "data", "processed", "corpus_completo.json")
        
    with open(dataset_file, "r", encoding="utf-8") as f:
        qa_data = json.load(f)
        
    def expand_variants(item_list):
        expanded = []
        for item in item_list:
            # Pregunta principal
            expanded.append({
                "id": item["id"],
                "section": item.get("section", ""),
                "subsection": item.get("subsection", ""),
                "source": item.get("source", ""),
                "context": item.get("context", "") or "",
                "question": item["question"],
                "answers": item["answers"]
            })
            # Preguntas variantes
            if not args.only_main:
                for idx, variant in enumerate(item.get("question_variants", [])):
                    if not variant.strip():
                        continue
                    expanded.append({
                        "id": f"{item['id']}_var{idx}",
                        "section": item.get("section", ""),
                        "subsection": item.get("subsection", ""),
                        "source": item.get("source", ""),
                        "context": item.get("context", "") or "",
                        "question": variant,
                        "answers": item["answers"]
                    })
        return expanded

    # Re-fijar la semilla JUSTO antes del shuffle para garantizar el mismo split siempre
    if args.eval_full:
        test_list_items = qa_data.copy()
    else:
        random.seed(SEED)
        shuffled_data = qa_data.copy()
        random.shuffle(shuffled_data)
        split_idx = int(len(shuffled_data) * 0.8)
        test_list_items = shuffled_data[split_idx:]  # 20% de test por ítems
    
    print(f"Número de ítems de prueba (sin fugas): {len(test_list_items)}")
    
    test_list = expand_variants(test_list_items)
    print(f"Número total de preguntas en el Test Set (con variantes): {len(test_list)}")

    # 2. Cargar el recuperador y entrenar con data/documents/
    info_dir = os.path.join(PARENT_DIR, "data", "documents")
    if not os.path.exists(info_dir):
        print(f"[ERROR] No se encontró la carpeta de información '{info_dir}' para RAG.")
        sys.exit(1)
        
    if args.retriever_type == "bm25":
        base_retriever = BM25Retriever()
        base_retriever.fit_from_info_dir(info_dir)
    elif args.retriever_type == "dense":
        local_model_path = os.path.join(PARENT_DIR, args.dense_model)
        retriever_model_name = local_model_path if os.path.exists(local_model_path) else args.dense_model
        base_retriever = DenseRetriever(model_name=retriever_model_name)
        
        # Necesitamos un BM25 temporal para procesar y segmentar los documentos de info_dir
        bm25_helper = BM25Retriever()
        bm25_helper.fit_from_info_dir(info_dir)
        base_retriever.fit(bm25_helper.documents)
    else: # hybrid
        bm25 = BM25Retriever()
        local_model_path = os.path.join(PARENT_DIR, args.dense_model)
        retriever_model_name = local_model_path if os.path.exists(local_model_path) else args.dense_model
        dense = DenseRetriever(model_name=retriever_model_name)
        base_retriever = HybridRetriever(bm25, dense, alpha=args.alpha)
        base_retriever.fit_from_info_dir(info_dir)

    # Wrap in AdvancedRetriever if requested
    use_advanced = args.use_hyde or args.use_multi_query or args.use_reranking or args.use_graph
    if use_advanced:
        from retriever import AdvancedRetriever
        reranker_path = os.path.join(PARENT_DIR, "models/ms-marco-MiniLM-L-6-v2")
        retriever = AdvancedRetriever(
            base_retriever,
            reranker_model_path=reranker_path if args.use_reranking else None,
            use_hyde=args.use_hyde,
            use_multi_query=args.use_multi_query,
            use_reranking=args.use_reranking,
            use_graph=args.use_graph
        )
    else:
        retriever = base_retriever

    # 3. Cargar el Tokenizador y el Modelo Causal LM (LLM)
    print(f"Cargando tokenizador para {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print(f"Cargando modelo Causal LM desde {model_name}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Configuración de cuantización opcional (Green AI)
    quant_kwargs = {}
    if args.quantization == "4bit":
        try:
            from transformers import BitsAndBytesConfig
            quant_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16 if device == "cuda" else torch.float32
            )
            print("Cargando modelo con cuantización de 4-bit (Green AI)")
        except ImportError:
            print("[WARNING] bitsandbytes no está disponible. Cargando sin cuantización de 4-bit.")
    elif args.quantization == "8bit":
        try:
            from transformers import BitsAndBytesConfig
            quant_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True
            )
            print("Cargando modelo con cuantización de 8-bit (Green AI)")
        except ImportError:
            print("[WARNING] bitsandbytes no está disponible. Cargando sin cuantización de 8-bit.")
            
    model = load_causal_lm_robust(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        token=HF_TOKEN,
        **quant_kwargs
    )

    if args.peft_model_path and os.path.exists(args.peft_model_path):
        print(f"Inyectando adaptador PEFT desde: {args.peft_model_path}")
        model = PeftModel.from_pretrained(model, args.peft_model_path)

    if device == "cpu":
        model = model.to(device)
    model.eval()

    # Definir generate_fn para HyDE y Multi-Query
    def generate_text(prompt_text):
        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            )
        input_len = inputs["input_ids"].shape[1]
        return tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()

    if use_advanced:
        retriever.generate_fn = generate_text

    # 4. Evaluación del Retriever (Hit Rate)
    print(f"\n[1/2] Evaluando precisión del Retriever BM25 (Top-{args.k})...")
    hit_at_1 = 0
    hit_at_3 = 0
    
    for record in test_list:
        gold_context = record.get("context", "")
        gold_context = gold_context.strip() if gold_context else ""
        gold_source = record.get("source", "")
        gold_base = os.path.basename(gold_source).lower() if gold_source else ""
        
        question = record["question"]
        
        retrieved = retriever.retrieve(question, k=max(3, args.k))
        retrieved_contexts = [r["context"].strip() for r in retrieved]
        retrieved_sources = [r.get("source", "") for r in retrieved]
        
        is_hit_1 = False
        is_hit_3 = False
        
        if retrieved:
            # Check hit @ 1
            r0_ctx = retrieved_contexts[0]
            r0_src = retrieved_sources[0]
            r0_base = os.path.basename(r0_src).lower() if r0_src else ""
            
            if (gold_context and gold_context in r0_ctx) or (gold_base and gold_base == r0_base):
                is_hit_1 = True
                is_hit_3 = True
            else:
                # Check hit @ 3
                for r_ctx, r_src in zip(retrieved_contexts[:3], retrieved_sources[:3]):
                    r_base = os.path.basename(r_src).lower() if r_src else ""
                    if (gold_context and gold_context in r_ctx) or (gold_base and gold_base == r_base):
                        is_hit_3 = True
                        break
                        
        if is_hit_1:
            hit_at_1 += 1
        if is_hit_3:
            hit_at_3 += 1
            
    retriever_hit_1 = (hit_at_1 / len(test_list)) * 100
    retriever_hit_3 = (hit_at_3 / len(test_list)) * 100
    print(f"  Hit Rate @ 1: {retriever_hit_1:.2f}%")
    print(f"  Hit Rate @ 3: {retriever_hit_3:.2f}%")

    # 5. Inferencia Generativa y Evaluación RAG
    print(f"\n[2/2] Evaluando RAG Generativo (Top-{args.k})...")
    
    total_rag_em = 0.0
    total_rag_f1 = 0.0
    total_rag_rouge1 = 0.0
    total_rag_rouge2 = 0.0
    total_rag_rougeL = 0.0
    total_rag_containment = 0.0
    total_rag_token_recall = 0.0
    total_rag_k_precision = 0.0
    latencies = []
    
    # Prompt de extracción literal usado en zero_shot_llm/prompts.py
    # (fallback sin chat template; con prompt_style=natural se usa instrucción libre)
    def build_prompt(context, question):
        if args.prompt_style == "natural":
            return f"""Responda à pergunta de forma natural, concisa e em português, com base no contexto.
Use suas próprias palavras e não invente informações.

Contexto:
{context}

Pergunta:
{question}

Resposta:"""
        return f"""Responda exatamente com um trecho literal do contexto.
Não explique, não resuma, não invente palavras.
Se a resposta não estiver explicitamente no contexto, responda com vazio.

Exemplo 1:
Contexto:
2026/01: Seleção de Alunos em Regime Especial (disciplinas isoladas)
Link inscrições: https://lnkd.in/dx5G_fGC
Início das aulas: 02/03/2026

Pergunta:
Qual é o link para fazer a inscrição no regime especial?

Resposta:
https://lnkd.in/dx5G_fGC

Exemplo 2:
Contexto:
Orientadores
Sistemas Inteligentes
Prof. Dr. Antonio Augusto Chaves
    Metaheurísticas
Prof. Dr. Elbert E. N. Macau
    Sistemas Complexos

Pergunta:
Qual é a especialidade do Prof. Dr. Antonio Augusto Chaves?

Resposta:
Metaheurísticas

Agora é sua vez:
Contexto:
{context}

Pergunta:
{question}

Resposta:"""

    # Exemplos Few-Shot para o template de chat
    few_shot_examples = [
        {
            "context": "2026/01: Seleção de Alunos em Regime Especial (disciplinas isoladas)\nLink inscrições: https://lnkd.in/dx5G_fGC\nInício das aulas: 02/03/2026",
            "question": "Qual é o link para fazer a inscrição no regime especial?",
            "answer": "https://lnkd.in/dx5G_fGC"
        },
        {
            "context": "Orientadores\nSistemas Inteligentes\nProf. Dr. Antonio Augusto Chaves\n    Metaheurísticas\nProf. Dr. Elbert E. N. Macau\n    Sistemas Complexos",
            "question": "Qual é a especialidade do Prof. Dr. Antonio Augusto Chaves?",
            "answer": "Metaheurísticas"
        }
    ]

    prediction_records = []  # para BERTScore (gold + respuesta por pregunta)
    for record in test_list:
        question = record["question"]
        answers = record["answers"]
        t0 = time.time()

        # 1. Contexto: oracle (gold) o recuperado (RAG estándar)
        if args.gold_context:
            gold_ctx = record.get("context", "") or ""
            context = gold_ctx.strip() if gold_ctx.strip() else "Sem contexto disponível."
        else:
            # 1. Recuperar contextos (Top-K)
            retrieved_docs = retriever.retrieve(question, k=args.k)
            if retrieved_docs:
                contexts_list = [doc["context"].strip() for doc in retrieved_docs]
                context = "\n---\n".join(contexts_list)
                # Agregar contexto de Grafo de Conocimiento si existe
                if hasattr(retrieved_docs, "graph_context") and retrieved_docs.graph_context:
                    context += "\n" + retrieved_docs.graph_context
            else:
                context = "Sem contexto disponível."
            
        # 2. Construir Prompt y Tokenizar usando chat template si está disponible
        if args.prompt_style == "natural":
            system_instruction = (
                "Você é um assistente de perguntas e respostas. Com base no contexto fornecido, "
                "responda à pergunta de forma natural, concisa e em português, usando suas próprias palavras. "
                "Não invente informações: se a resposta não estiver no contexto, diga que não encontrou."
            )
        elif args.k == 1:
            system_instruction = "Responda exatamente com um trecho literal do contexto fornecido. Não explique, não resuma, não invente palavras. Se a resposta não estiver explicitamente no contexto, responda apenas com vazio."
        else:
            system_instruction = (
                "Você é um extrator de informações preciso. O contexto fornecido pode conter múltiplos fragmentos separados por '---'. "
                "Seu objetivo é localizar o fragmento correto, extrair a resposta exata e literal para a pergunta e responder APENAS com esse trecho literal. "
                "Não explique, não resuma, não invente palavras. Se a resposta não estiver explicitamente no contexto, responda apenas com vazio."
            )
        
        messages = [
            {"role": "system", "content": system_instruction}
        ]
        
        # Inyectar ejemplos few-shot si no están desactivados
        if not args.no_few_shot:
            for ex in few_shot_examples:
                messages.append({"role": "user", "content": f"Contexto:\n{ex['context']}\n\nPergunta:\n{ex['question']}"})
                messages.append({"role": "assistant", "content": ex['answer']})
            
        # Añadir la consulta actual (prompt_context es el texto exacto enviado al modelo;
        # se reutiliza para K-Precision para medir anclaje contra lo realmente visto)
        prompt_context = context[:args.context_max_chars]
        messages.append({"role": "user", "content": f"Contexto:\n{prompt_context}\n\nPergunta:\n{question}"})

        try:
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            prompt = build_prompt(prompt_context, question)
            
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1536
        ).to(device)
        
        # 3. Generación
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            )
            
        # 4. Decodificar sólo el contenido generado
        input_len = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_len:]
        pred_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        
        # Quedarse con la primera línea de respuesta para evitar divagaciones
        pred_text = pred_text.split("\n")[0].strip()
        
        # Limpiar la respuesta para extraer el span literal de forma robusta
        pred_text = clean_llm_answer(pred_text)
        
        t1 = time.time()
        latencies.append(t1 - t0)
        
        # Evaluar contra la verdad de referencia
        if not answers["text"]:
            em = 1.0 if pred_text == "" else 0.0
            f1 = 1.0 if pred_text == "" else 0.0
            r1, r2, rl = (1.0, 1.0, 1.0) if pred_text == "" else (0.0, 0.0, 0.0)
            cont = 1.0 if pred_text == "" else 0.0
            trec = 1.0 if pred_text == "" else 0.0
            kprec = 1.0 if pred_text == "" else 0.0
        else:
            gold_text = answers["text"][0]
            em = compute_exact(gold_text, pred_text)
            f1 = compute_f1(gold_text, pred_text)
            r1, r2, rl = compute_rouge(gold_text, pred_text)
            cont = compute_containment(gold_text, pred_text)
            trec = compute_token_recall(gold_text, pred_text)
            kprec = compute_k_precision(pred_text, prompt_context)

        total_rag_em += em
        total_rag_f1 += f1
        total_rag_rouge1 += r1
        total_rag_rouge2 += r2
        total_rag_rougeL += rl
        total_rag_containment += cont
        total_rag_token_recall += trec
        total_rag_k_precision += kprec
        gold_text_all = answers["text"][0] if answers["text"] else ""
        prediction_records.append({
            "id": record.get("id", len(prediction_records)),
            "question": question,
            "gold_answer": gold_text_all,
            "llm_response": pred_text,
        })
        
    rag_em = (total_rag_em / len(test_list)) * 100
    rag_f1 = (total_rag_f1 / len(test_list)) * 100
    rag_r1 = (total_rag_rouge1 / len(test_list)) * 100
    rag_r2 = (total_rag_rouge2 / len(test_list)) * 100
    rag_rl = (total_rag_rougeL / len(test_list)) * 100
    rag_cont = (total_rag_containment / len(test_list)) * 100
    rag_trec = (total_rag_token_recall / len(test_list)) * 100
    rag_kprec = (total_rag_k_precision / len(test_list)) * 100
    avg_latency = sum(latencies) / len(latencies) * 1000  # en ms
    peak_vram = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    infer_vram_mb = (peak_vram - initial_vram) / (1024 * 1024)

    print(f"  RAG Generativo Exact Match: {rag_em:.2f}%")
    print(f"  RAG Generativo F1 Score   : {rag_f1:.2f}%")
    print(f"  RAG Generativo ROUGE-1    : {rag_r1:.2f}%")
    print(f"  RAG Generativo ROUGE-2    : {rag_r2:.2f}%")
    print(f"  RAG Generativo ROUGE-L    : {rag_rl:.2f}%")
    print(f"  RAG Generativo Contención (gold⊂resp): {rag_cont:.2f}%")
    print(f"  RAG Generativo Token-Recall       : {rag_trec:.2f}%")
    print(f"  RAG Generativo K-Precision        : {rag_kprec:.2f}%")
    print(f"  Pico VRAM inferencia      : {infer_vram_mb:.1f} MB")
    print(f"  Latencia media            : {avg_latency:.2f} ms por consulta")
 
    # Guardar los resultados en la carpeta results/
    suffix = "_natural" if args.prompt_style == "natural" else ""
    output_dir = args.output_dir or os.path.join(PARENT_DIR, "results", f"{args.model_type}_rag{suffix}")
    os.makedirs(output_dir, exist_ok=True)
    
    rag_metrics = {
        "retriever_hit_at_1": retriever_hit_1,
        "retriever_hit_at_3": retriever_hit_3,
        "reader_exact_match": None,
        "reader_f1": None,
        "rag_exact_match": rag_em,
        "rag_f1": rag_f1,
        "rag_rouge1": rag_r1,
        "rag_rouge2": rag_r2,
        "rag_rougeL": rag_rl,
        "rag_containment": rag_cont,
        "rag_token_recall": rag_trec,
        "rag_k_precision": rag_kprec,
        "peak_vram_mb": infer_vram_mb,
        "average_latency_ms": avg_latency
    }
    
    output_file = os.path.join(output_dir, "rag_evaluation.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(rag_metrics, f, indent=2)

    pred_file = os.path.join(output_dir, "predictions.json")
    with open(pred_file, "w", encoding="utf-8") as f:
        json.dump(prediction_records, f, ensure_ascii=False, indent=2)
        
    print(f"\nMétricas del pipeline RAG Generativo exportadas a {output_file}")
    print("=" * 60)

if __name__ == "__main__":
    main()
