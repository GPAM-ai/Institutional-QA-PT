import os
import sys
import json
import time
import argparse
import random
import numpy as np
from pathlib import Path

SEED = 42

# Obtener rutas relativas
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AutoModelForCausalLM
from peft import PeftModel, PeftConfig

from retriever import BM25Retriever, DenseRetriever, HybridRetriever
from train_unifesp_qa import compute_exact, compute_f1, compute_rouge, compute_containment, compute_token_recall, compute_k_precision, list_to_hf_dataset

# Mappings
BERT_MODEL_MAPPING = {
    "mbert": "bert-base-multilingual-cased",
    "xlmr": "xlm-roberta-base",
    "bertimbau_base": "pierreguillou/bert-base-cased-squad-v1.1-portuguese",
    "bertimbau_large": "pierreguillou/bert-large-cased-squad-v1.1-portuguese",
    "bertugues": "ricardoz/BERTugues-base-portuguese-cased"
}

LLM_MODEL_MAPPING = {
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

HF_TOKEN = os.environ.get("HF_TOKEN", "")

def load_peft_base_model_name(adapter_path, fallback=None):
    """Lee base_model_name_or_path del adapter_config.json con json plano.

    Evita PeftConfig.from_pretrained, que falla si el adaptador se entrenó
    con un peft más nuevo (ej. clave 'alora_invocation_tokens' de peft 0.18.x
    frente al peft 0.15.2 del env).
    """
    try:
        with open(os.path.join(adapter_path, "adapter_config.json")) as f:
            return json.load(f).get("base_model_name_or_path", fallback)
    except Exception as e:
        print(f"[WARNING] No se pudo leer adapter_config.json: {e}. Usando fallback.")
        return fallback

def load_bert_with_adapter(bert_model_path, base_model_id):
    """Carga modelo BERT + adaptador LoRA tolerando peft nuevo en checkpoint.

    Intenta PeftModel.from_pretrained y, si la versión del env no entiende
    el adapter_config, hace carga manual (get_peft_model + state_dict).
    """
    base_model = AutoModelForQuestionAnswering.from_pretrained(base_model_id)
    try:
        return PeftModel.from_pretrained(base_model, bert_model_path)
    except TypeError as e:
        print(f"[WARNING] PeftModel.from_pretrained falló ({e}). Carga manual del adaptador...")
    import inspect as _inspect
    import glob as _glob
    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
    with open(os.path.join(bert_model_path, "adapter_config.json")) as f:
        raw = json.load(f)
    allowed = set(_inspect.signature(LoraConfig.__init__).parameters)
    lora_cfg = LoraConfig(**{k: v for k, v in raw.items() if k in allowed})
    model = get_peft_model(base_model, lora_cfg)
    sft_files = _glob.glob(os.path.join(bert_model_path, "*.safetensors"))
    if sft_files:
        from safetensors.torch import load_file as _load_sft
        state = _load_sft(sft_files[0], device="cpu")
    else:
        state = torch.load(os.path.join(bert_model_path, "pytorch_model.bin"), map_location="cpu")
    set_peft_model_state_dict(model, state)
    return model

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
    if (cleaned.startswith('"') and cleaned.endswith('"')) or (cleaned.startswith("'") and cleaned.endswith("'")):
        cleaned = cleaned[1:-1].strip()
    return cleaned

def extract_span_with_bert(question, contexts_list, model, tokenizer, device):
    best_overall_span = ""
    best_overall_score = -float('inf')
    best_overall_margin = -float('inf')
    
    for context in contexts_list:
        encoding = tokenizer(
            question,
            context,
            max_length=512,
            truncation="only_second",
            return_offsets_mapping=True
        )
        
        input_ids = torch.tensor([encoding["input_ids"]]).to(device)
        attention_mask = torch.tensor([encoding["attention_mask"]]).to(device)
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
        start_logits = outputs.start_logits[0].cpu().numpy()
        end_logits = outputs.end_logits[0].cpu().numpy()
        
        offsets = encoding["offset_mapping"]
        sequence_ids = encoding.sequence_ids(0)
        
        context_tokens = [i for i, sid in enumerate(sequence_ids) if sid == 1]
        if not context_tokens:
            continue
            
        start_idx = context_tokens[0]
        end_idx = context_tokens[-1]
        
        best_start = start_idx
        best_end = start_idx
        best_score = -float('inf')
        
        n_best = 20
        start_candidates = np.argsort(start_logits)[::-1][:n_best]
        end_candidates = np.argsort(end_logits)[::-1][:n_best]
        
        for s in start_candidates:
            if s < start_idx or s > end_idx:
                continue
            for e in end_candidates:
                if e < start_idx or e > end_idx:
                    continue
                if e < s or (e - s + 1) > 40:  # longitud máxima de respuesta
                    continue
                score = start_logits[s] + end_logits[e]
                if score > best_score:
                    best_score = score
                    best_start = s
                    best_end = e
                    
        # Calcular si este chunk contiene una respuesta mejor que lo que tenemos hasta ahora
        null_score = start_logits[0] + end_logits[0]
        
        # Si la mejor respuesta de este-chunk supera al CLS (null) de este mismo chunk
        if best_score > null_score:
            # Y si supera a la mejor puntuación de respuesta encontrada en otros chunks
            if best_score > best_overall_score:
                start_char = offsets[best_start][0]
                end_char = offsets[best_end][1]
                best_overall_span = context[start_char:end_char].strip()
                best_overall_score = best_score
                best_overall_margin = best_score - null_score

    return best_overall_span, float(best_overall_margin)

def build_free_prompt(context, question):
    """Generación libre desde contexto, sin span (rama de baja confianza).

    Evita el anclaje del extractor cuando BERT es incierto: el LLM responde
    directamente del contexto en lenguaje natural, mismo estilo que la rama
    truth pero sin evidencia candidata. Patrón adaptive-gating (Self-RAG).
    """
    return f"""Você é um assistente de pós-graduação da UNIFESP.
Responda à pergunta de forma direta, completa e educada em português,
usando estritamente os fatos do contexto fornecido. Não adicione informações externas.
Se a informação não estiver no contexto, diga que não foi encontrada.

Contexto:
{context}

Pergunta:
{question}

Resposta natural em português:"""

def build_hybrid_prompt(context, question, extracted_span, mode="truth"):
    """Prompt del híbrido en dos modos (ver HYBGRAG/Self-RAG: validar antes de usar).

    truth  (default, comportamiento histórico): el span BERT se presenta como
             "base da verdade" que el LLM debe reescribir en lenguaje natural.
    verify : el span se presenta como CANDIDATO a verificar. El LLM debe
             comprobar si el contexto lo respalda; si sí, lo reescribe; si no,
             responde solo con lo que el contexto respalde. Evita el anclaje
             cuando BERT extrae un span erróneo (lector 43 F1 con retriever).
    """
    if not extracted_span:
        return f"""Você é um assistente de pós-graduação da UNIFESP.
Responda de forma direta e natural em português que a informação solicitada não foi encontrada no contexto.

Contexto:
{context}

Pergunta:
{question}

Resposta natural:"""

    if mode == "verify":
        return f"""Você é um assistente de pós-graduação da UNIFESP que verifica evidências antes de responder.
Recebeu um contexto, uma pergunta e uma resposta curta CANDIDATA extraída por outro modelo.
Passos: (1) verifique se a candidata está respaldada literalmente pelo contexto; (2) se sim, escreva-a em linguagem natural direta, completa e educada em português; (3) se não, ignore a candidata e responda apenas com o que o contexto respaldar, ou diga que a informação não foi encontrada.
Use apenas os fatos fornecidos. Não adicione informações externas.

Contexto:
{context}

Pergunta:
{question}

Resposta curta candidata (verifique no contexto, não assuma que está correta):
{extracted_span}

Resposta natural em português:"""

    return f"""Com base no contexto e na resposta curta extraída, escreva uma resposta em linguagem natural que seja direta, completa e educada em português.
Use apenas os fatos fornecidos. Não adicione informações externas.

Contexto:
{context}

Pergunta:
{question}

Resposta curta extraída (use como base da verdade):
{extracted_span}

Resposta natural em português:"""

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluación de pipeline RAG Híbrido (BERT + LLM).")
    parser.add_argument("--bert_model_type", type=str, default="bertimbau_large", choices=list(BERT_MODEL_MAPPING.keys()),
                        help="Modelo QA BERT para extracción.")
    parser.add_argument("--bert_tuning_type", type=str, default="lora", choices=["none", "full", "lora", "qlora"],
                        help="Tipo de ajuste del modelo BERT.")
    parser.add_argument("--bert_bs", type=int, default=64, help="Batch size usado en entrenamiento del modelo BERT.")
    parser.add_argument("--bert_lr", type=float, default=1e-4, help="Learning rate del modelo BERT.")
    parser.add_argument("--bert_epochs", type=float, default=15.0, help="Épocas del modelo BERT.")
    
    parser.add_argument("--llm_model_type", type=str, default="phi_4_mini", choices=list(LLM_MODEL_MAPPING.keys()),
                        help="Modelo LLM para reescritura natural.")
    parser.add_argument("--llm_adapter_path", type=str, default=None, help="Ruta al adaptador Peft/QLoRA del LLM fine-tuneado.")
    parser.add_argument("--k", type=int, default=1, help="Número de documentos a recuperar (Top-K).")
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
    parser.add_argument("--best_rag", action="store_true",
                        help="Usar la configuración híbrida recomendada: Top-5 para el lector extractivo.")
    parser.add_argument("--router_mode", type=str, default="always_llm",
                        choices=["always_llm", "evidence_guarded", "confidence_router", "bert_only", "llm_verify", "confidence_gated"],
                        help="Cómo seleccionar la salida final: Qwen, router por confianza, BERT, "
                              "llm_verify (el LLM verifica el span y ramifica el prompt), o "
                              "confidence_gated (truth si margen BERT >= umbral, generación libre "
                              "sin span en caso contrario; evita anclaje adaptativo).")
    parser.add_argument("--span_guidance", type=str, default="truth",
                        choices=["truth", "verify"],
                        help="Cómo presentar el span BERT al LLM: 'truth' (base da verdade, "
                             "histórico) o 'verify' (candidato a verificar contra el contexto; "
                             "evita anclaje con spans erróneos).")
    parser.add_argument("--confidence_threshold", type=float, default=0.0,
                        help="Margen mínimo span-null de BERT para seleccionar BERT.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directorio donde guardar métricas y predicciones.")
    return parser.parse_args()

def main():
    args = parse_args()

    if args.best_rag:
        args.k = max(args.k, 5)
    
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Green AI: VRAM base antes de cargar modelos (el pico de inferencia se
    # calcula al final como max_memory_allocated - initial_vram)
    initial_vram = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Rutas de modelos
    bert_exp_id = f"{args.bert_model_type}_{args.bert_tuning_type}_bs{args.bert_bs}_lr{args.bert_lr}_ep{args.bert_epochs}"
    if args.bert_tuning_type == "none":
        bert_model_path = BERT_MODEL_MAPPING[args.bert_model_type]
    else:
        bert_model_path = os.path.join(PARENT_DIR, "results", bert_exp_id, "best_model")
        
    llm_model_name = LLM_MODEL_MAPPING[args.llm_model_type]
    
    print("=" * 70)
    print(f"EVALUACIÓN RAG HÍBRIDO (BERT + LLM)")
    print(f"  Extractor BERT : {args.bert_model_type} ({args.bert_tuning_type})")
    print(f"  Generador LLM  : {args.llm_model_type} ({llm_model_name})")
    print(f"  Top-K          : {args.k}")
    if args.best_rag:
        print("  Configuración  : LECTOR PRINCIPAL BERTIMBAU (Top-5; Phi opcional)")
    print("=" * 70)

    # 2. Cargar el dataset de prueba (el mismo 20% que el script de entrenamiento)
    dataset_file = args.dataset_file if args.dataset_file else os.path.join(PARENT_DIR, "data", "processed", "corpus_completo.json")
        
    with open(dataset_file, "r", encoding="utf-8") as f:
        qa_data = json.load(f)
        
    def expand_variants(item_list):
        expanded = []
        for item in item_list:
            expanded.append({
                "id": item["id"],
                "section": item.get("section", ""),
                "subsection": item.get("subsection", ""),
                "source": item.get("source", ""),
                "context": item.get("context", "") or "",
                "question": item["question"],
                "answers": item["answers"]
            })
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
        test_list_items = shuffled_data[split_idx:]
    test_list = expand_variants(test_list_items)
    
    print(f"Número total de preguntas en el Test Set: {len(test_list)}")

    # 3. Cargar el recuperador y entrenar
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

    # 4. Cargar el Extractor BERT
    print(f"\nCargando tokenizador y modelo BERT desde {bert_model_path}...")
    bert_tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_MAPPING[args.bert_model_type])
    
    if args.bert_tuning_type == "none":
        bert_model = AutoModelForQuestionAnswering.from_pretrained(bert_model_path)
    elif os.path.exists(os.path.join(bert_model_path, "adapter_config.json")):
        base_model_id = load_peft_base_model_name(
            bert_model_path, BERT_MODEL_MAPPING[args.bert_model_type])
        bert_model = load_bert_with_adapter(bert_model_path, base_model_id)
    else:
        bert_model = AutoModelForQuestionAnswering.from_pretrained(bert_model_path)
        
    bert_model = bert_model.to(device)
    bert_model.eval()

    # 5. Cargar el Generador LLM
    print(f"Cargando tokenizador y modelo LLM desde {llm_model_name}...")
    llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name, token=HF_TOKEN)
    llm_tokenizer.pad_token = llm_tokenizer.eos_token
    llm_tokenizer.padding_side = "left"
    
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
            
    llm_model = load_causal_lm_robust(
        llm_model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        token=HF_TOKEN,
        **quant_kwargs
    )
    if args.llm_adapter_path and os.path.exists(args.llm_adapter_path):
        print(f"Cargando adaptador Peft/QLoRA desde {args.llm_adapter_path}...")
        llm_model = PeftModel.from_pretrained(llm_model, args.llm_adapter_path)

    if device == "cpu":
        llm_model = llm_model.to(device)
    llm_model.eval()

    # Definir generate_fn para HyDE y Multi-Query
    def generate_text(prompt_text):
        inputs = llm_tokenizer(prompt_text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = llm_model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                pad_token_id=llm_tokenizer.pad_token_id if llm_tokenizer.pad_token_id is not None else llm_tokenizer.eos_token_id
            )
        input_len = inputs["input_ids"].shape[1]
        return llm_tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()

    if use_advanced:
        retriever.generate_fn = generate_text

    # 6. Bucle de Inferencia y Evaluación
    print(f"\nEjecutando inferencia híbrida sobre el dataset de test...")
    
    total_bert_em = 0.0
    total_bert_f1 = 0.0
    total_hybrid_em = 0.0
    total_hybrid_f1 = 0.0
    total_hybrid_rouge1 = 0.0
    total_hybrid_rouge2 = 0.0
    total_hybrid_rougeL = 0.0
    total_hybrid_containment = 0.0
    total_hybrid_token_recall = 0.0
    total_hybrid_k_precision = 0.0
    total_retention = 0.0
    total_verify_sim = 0.0
    total_gated_truth = 0.0
    total_selected_em = 0.0
    total_selected_f1 = 0.0
    total_selected_rouge1 = 0.0
    total_selected_rouge2 = 0.0
    total_selected_rougeL = 0.0
    total_selected_containment = 0.0
    total_selected_token_recall = 0.0
    total_selected_k_precision = 0.0
    prediction_records = []
    
    latencies = []
    
    few_shot_examples = [
        {
            "context": "2026/01: Seleção de Alunos em Regime Especial (disciplinas isoladas)\nLink inscrições: https://lnkd.in/dx5G_fGC\nInício das aulas: 02/03/2026",
            "question": "Qual é o link para fazer a inscrição no regime especial?",
            "extracted_span": "https://lnkd.in/dx5G_fGC",
            "answer": "O link para realizar a inscrição no regime especial é https://lnkd.in/dx5G_fGC."
        },
        {
            "context": "Orientadores\nSistemas Inteligentes\nProf. Dr. Antonio Augusto Chaves\n    Metaheurísticas\nProf. Dr. Elbert E. N. Macau\n    Sistemas Complexos",
            "question": "Qual é a especialidade do Prof. Dr. Antonio Augusto Chaves?",
            "extracted_span": "Metaheurísticas",
            "answer": "A especialidade de pesquisa do Prof. Dr. Antonio Augusto Chaves é Metaheurísticas."
        }
    ]

    for record_idx, record in enumerate(test_list):
        question = record["question"]
        answers = record["answers"]
        
        t0 = time.time()
        
        # Paso 1: Recuperación
        retrieved_docs = retriever.retrieve(question, k=args.k)
        if retrieved_docs:
            contexts_list = [doc["context"].strip() for doc in retrieved_docs]
            retrieved_context = "\n---\n".join(contexts_list)
            # Agregar contexto de Grafo de Conocimiento si existe
            if hasattr(retrieved_docs, "graph_context") and retrieved_docs.graph_context:
                retrieved_context += "\n" + retrieved_docs.graph_context
        else:
            retrieved_context = "Sem contexto disponível."
            
        # Paso 2: Extracción con BERT
        contexts_list = [doc["context"].strip() for doc in retrieved_docs] if retrieved_docs else ["Sem contexto disponível."]
        bert_extracted_span, bert_confidence = extract_span_with_bert(
            question, contexts_list, bert_model, bert_tokenizer, device
        )

        # Contexto truncado exacto que verá el LLM (también base de K-Precision)
        prompt_ctx = retrieved_context[:3000]

        # Paso 2b: Verificación del span con el propio LLM (solo router llm_verify).
        # Rama SIM -> humaniza el span (truth); rama NÃO/EMPTY -> extracción
        # directa del contexto sin span (evita el anclaje). Un solo paso extra
        # corto de generación (max 8 tokens).
        verify_verdict = "N/A"
        use_truth_branch = True
        if args.router_mode == "llm_verify":
            if not bert_extracted_span:
                verify_verdict = "EMPTY"
                use_truth_branch = False
            else:
                verify_messages = [
                    {"role": "system", "content": "Você é um verificador factual. Responda apenas SIM ou NÃO."},
                    {"role": "user", "content": (
                        f"Contexto:\n{prompt_ctx}\n\nPergunta:\n{question}\n\n"
                        f"Resposta candidata:\n{bert_extracted_span}\n\n"
                        "A resposta candidata está correta e literalmente respaldada pelo contexto? "
                        "Responda apenas SIM ou NÃO.")}
                ]
                try:
                    verify_prompt = llm_tokenizer.apply_chat_template(verify_messages, tokenize=False, add_generation_prompt=True)
                except Exception:
                    verify_prompt = verify_messages[-1]["content"]
                verify_inputs = llm_tokenizer(verify_prompt, return_tensors="pt", truncation=True, max_length=1536).to(device)
                with torch.no_grad():
                    verify_outputs = llm_model.generate(
                        **verify_inputs,
                        max_new_tokens=8,
                        do_sample=False,
                        pad_token_id=llm_tokenizer.pad_token_id if llm_tokenizer.pad_token_id is not None else llm_tokenizer.eos_token_id
                    )
                verify_len = verify_inputs["input_ids"].shape[1]
                verify_text = llm_tokenizer.decode(verify_outputs[0][verify_len:], skip_special_tokens=True).strip().upper()
                verify_verdict = "SIM" if verify_text.startswith("SIM") else "NÃO"
                use_truth_branch = (verify_verdict == "SIM")
            total_verify_sim += 1.0 if verify_verdict == "SIM" else 0.0

        # Paso 3: Generación con LLM (rama según verificación / span_guidance)
        # confidence_gated (adaptive, estilo Self-RAG): span como verdad solo si
        # BERT es confiable (margen >= umbral); si no, generación libre sin span.
        gated_use_truth = True
        if args.router_mode == "confidence_gated":
            gated_use_truth = bool(bert_extracted_span) and float(bert_confidence) >= args.confidence_threshold
            total_gated_truth += 1.0 if gated_use_truth else 0.0
        if (args.router_mode == "llm_verify" and not use_truth_branch) or (
                args.router_mode == "confidence_gated" and not gated_use_truth):
            # Generación libre del contexto, sin span (evita el anclaje).
            system_instruction = "Você é um assistente que responde perguntas de forma completa, direta e natural em português, baseando-se estritamente no contexto fornecido."
            messages = [
                {"role": "system", "content": system_instruction}
            ]
            for ex in few_shot_examples:
                messages.append({"role": "user", "content": f"Contexto:\n{ex['context']}\n\nPergunta:\n{ex['question']}"})
                messages.append({"role": "assistant", "content": ex['answer']})
            current_user_content = f"Contexto:\n{prompt_ctx}\n\nPergunta:\n{question}"
            messages.append({"role": "user", "content": current_user_content})
            try:
                prompt = llm_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                prompt = build_free_prompt(prompt_ctx, question)
        elif args.span_guidance == "verify":
            system_instruction = "Você é um assistente que verifica uma resposta curta candidata contra o contexto e a reescreve em frases completas, diretas e naturais em português, usando estritamente os fatos do contexto fornecido."
            span_label = "Resposta curta candidata (verifique no contexto)"
            messages = [
                {"role": "system", "content": system_instruction}
            ]

            for ex in few_shot_examples:
                user_content = f"Contexto:\n{ex['context']}\n\nPergunta:\n{ex['question']}\n\n{span_label}:\n{ex['extracted_span']}"
                messages.append({"role": "user", "content": user_content})
                messages.append({"role": "assistant", "content": ex['answer']})

            current_user_content = f"Contexto:\n{prompt_ctx}\n\nPergunta:\n{question}\n\n{span_label}:\n{bert_extracted_span}"
            messages.append({"role": "user", "content": current_user_content})

            try:
                prompt = llm_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                prompt = build_hybrid_prompt(prompt_ctx, question, bert_extracted_span, mode=args.span_guidance)
        else:
            system_instruction = "Você é um assistente que reescreve respostas curtas em frases completas, diretas e naturais em português, baseando-se estritamente no contexto fornecido."
            span_label = "Resposta curta extraída"
            messages = [
                {"role": "system", "content": system_instruction}
            ]

            for ex in few_shot_examples:
                user_content = f"Contexto:\n{ex['context']}\n\nPergunta:\n{ex['question']}\n\n{span_label}:\n{ex['extracted_span']}"
                messages.append({"role": "user", "content": user_content})
                messages.append({"role": "assistant", "content": ex['answer']})

            current_user_content = f"Contexto:\n{prompt_ctx}\n\nPergunta:\n{question}\n\n{span_label}:\n{bert_extracted_span}"
            messages.append({"role": "user", "content": current_user_content})

            try:
                prompt = llm_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                prompt = build_hybrid_prompt(prompt_ctx, question, bert_extracted_span, mode=args.span_guidance)
            
        inputs = llm_tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1536
        ).to(device)
        
        with torch.no_grad():
            outputs = llm_model.generate(
                **inputs,
                max_new_tokens=48,
                do_sample=False,
                pad_token_id=llm_tokenizer.pad_token_id if llm_tokenizer.pad_token_id is not None else llm_tokenizer.eos_token_id
            )
            
        input_len = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_len:]
        llm_response = llm_tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        llm_response = llm_response.split("\n")[0].strip()
        llm_response = clean_llm_answer(llm_response)
        
        t1 = time.time()
        latencies.append(t1 - t0)
        
        # Paso 4: Evaluar métricas
        
        # A) Evaluación de Extracción BERT
        if not answers["text"]:
            b_em = 1.0 if bert_extracted_span == "" else 0.0
            b_f1 = 1.0 if bert_extracted_span == "" else 0.0
        else:
            gold_text = answers["text"][0]
            b_em = compute_exact(gold_text, bert_extracted_span)
            b_f1 = compute_f1(gold_text, bert_extracted_span)
            
        total_bert_em += b_em
        total_bert_f1 += b_f1
        
        # B) Evaluación de Generación Híbrida (contra gold answers)
        if not answers["text"]:
            h_em = 1.0 if llm_response == "" else 0.0
            h_f1 = 1.0 if llm_response == "" else 0.0
            h_r1, h_r2, h_rl = (1.0, 1.0, 1.0) if llm_response == "" else (0.0, 0.0, 0.0)
            h_cont = 1.0 if llm_response == "" else 0.0
            h_trec = 1.0 if llm_response == "" else 0.0
            h_kprec = 1.0 if llm_response == "" else 0.0
        else:
            gold_text = answers["text"][0]
            h_em = compute_exact(gold_text, llm_response)
            h_f1 = compute_f1(gold_text, llm_response)
            h_r1, h_r2, h_rl = compute_rouge(gold_text, llm_response)
            h_cont = compute_containment(gold_text, llm_response)
            h_trec = compute_token_recall(gold_text, llm_response)
            h_kprec = compute_k_precision(llm_response, prompt_ctx)

        total_hybrid_em += h_em
        total_hybrid_f1 += h_f1
        total_hybrid_rouge1 += h_r1
        total_hybrid_rouge2 += h_r2
        total_hybrid_rougeL += h_rl
        total_hybrid_containment += h_cont
        total_hybrid_token_recall += h_trec
        total_hybrid_k_precision += h_kprec
        
        # C) Evaluación de Retención (¿la extracción de BERT está contenida en la respuesta de Phi-4?)
        if not bert_extracted_span:
            retention = 1.0  # Si BERT no extrajo nada y no debía, retención es correcta
        else:
            # Normalizar para comparación robusta
            norm_span = bert_extracted_span.lower().strip()
            norm_llm = llm_response.lower().strip()
            # Quitar signos de puntuación finales comunes del span para evitar fallos de matching
            norm_span_clean = norm_span.rstrip(".,?!:;")
            if norm_span_clean in norm_llm:
                retention = 1.0
            else:
                retention = 0.0
                
        total_retention += retention

        # NOTA llm_verify: la decisión ya se tomó en el Paso 2b (rama del prompt),
        # por eso cae al else y selected_response = llm_response (salida verificada).
        if args.router_mode == "bert_only":
            selected_response = bert_extracted_span
        elif args.router_mode == "confidence_router" and bert_extracted_span and bert_confidence >= args.confidence_threshold:
            selected_response = bert_extracted_span
        elif args.router_mode == "evidence_guarded" and bert_extracted_span and not retention:
            selected_response = bert_extracted_span
        else:
            selected_response = llm_response

        if not answers["text"]:
            selected_em = 1.0 if selected_response == "" else 0.0
            selected_f1 = selected_em
            selected_r1, selected_r2, selected_rl = (1.0, 1.0, 1.0) if selected_em else (0.0, 0.0, 0.0)
            selected_cont = 1.0 if selected_response == "" else 0.0
            selected_trec = 1.0 if selected_response == "" else 0.0
            selected_kprec = 1.0 if selected_response == "" else 0.0
        else:
            gold_text = answers["text"][0]
            selected_em = compute_exact(gold_text, selected_response)
            selected_f1 = compute_f1(gold_text, selected_response)
            selected_r1, selected_r2, selected_rl = compute_rouge(gold_text, selected_response)
            selected_cont = compute_containment(gold_text, selected_response)
            selected_trec = compute_token_recall(gold_text, selected_response)
            selected_kprec = compute_k_precision(selected_response, prompt_ctx)

        total_selected_em += selected_em
        total_selected_f1 += selected_f1
        total_selected_rouge1 += selected_r1
        total_selected_rouge2 += selected_r2
        total_selected_rougeL += selected_rl
        total_selected_containment += selected_cont
        total_selected_token_recall += selected_trec
        total_selected_k_precision += selected_kprec

        prediction_records.append({
            "id": record["id"],
            "question": question,
            "gold_answer": answers["text"][0] if answers["text"] else "",
            "bert_span": bert_extracted_span,
            "bert_confidence": float(bert_confidence),
            "verify_verdict": verify_verdict,
            "llm_response": llm_response,
            "selected_response": selected_response,
            "router_mode": args.router_mode,
            "bert_f1": float(b_f1) * 100,
            "llm_f1": float(h_f1) * 100,
            "llm_containment": float(h_cont) * 100,
            "llm_token_recall": float(h_trec) * 100,
            "llm_k_precision": float(h_kprec) * 100,
            "selected_f1": float(selected_f1) * 100,
            "selected_containment": float(selected_cont) * 100,
            "selected_token_recall": float(selected_trec) * 100,
            "selected_k_precision": float(selected_kprec) * 100,
            "span_retained": bool(retention),
            "retrieved_context": retrieved_context
        })
        
        if (record_idx + 1) % 20 == 0:
            print(f" Procesadas {record_idx + 1}/{len(test_list)} preguntas...")

    # Promediar métricas
    avg_bert_em = (total_bert_em / len(test_list)) * 100
    avg_bert_f1 = (total_bert_f1 / len(test_list)) * 100
    avg_hybrid_em = (total_hybrid_em / len(test_list)) * 100
    avg_hybrid_f1 = (total_hybrid_f1 / len(test_list)) * 100
    avg_hybrid_r1 = (total_hybrid_rouge1 / len(test_list)) * 100
    avg_hybrid_r2 = (total_hybrid_rouge2 / len(test_list)) * 100
    avg_hybrid_rl = (total_hybrid_rougeL / len(test_list)) * 100
    avg_hybrid_cont = (total_hybrid_containment / len(test_list)) * 100
    avg_hybrid_trec = (total_hybrid_token_recall / len(test_list)) * 100
    avg_hybrid_kprec = (total_hybrid_k_precision / len(test_list)) * 100
    avg_retention = (total_retention / len(test_list)) * 100
    avg_verify_sim = (total_verify_sim / len(test_list)) * 100
    avg_gated_truth = (total_gated_truth / len(test_list)) * 100
    avg_selected_em = (total_selected_em / len(test_list)) * 100
    avg_selected_f1 = (total_selected_f1 / len(test_list)) * 100
    avg_selected_r1 = (total_selected_rouge1 / len(test_list)) * 100
    avg_selected_r2 = (total_selected_rouge2 / len(test_list)) * 100
    avg_selected_rl = (total_selected_rougeL / len(test_list)) * 100
    avg_selected_cont = (total_selected_containment / len(test_list)) * 100
    avg_selected_trec = (total_selected_token_recall / len(test_list)) * 100
    avg_selected_kprec = (total_selected_k_precision / len(test_list)) * 100
    avg_latency = (sum(latencies) / len(latencies)) * 1000
    peak_vram = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    infer_vram_mb = (peak_vram - initial_vram) / (1024 * 1024)
    
    print("\n" + "="*50)
    print("RESULTADOS DEL PIPELINE RAG HÍBRIDO (BERT + LLM):")
    print(f"  BERT Extractor F1 Score       : {avg_bert_f1:.2f}%")
    print(f"  BERT Extractor Exact Match    : {avg_bert_em:.2f}%")
    print(f"  LLM Hybrid F1 Score (Literal)  : {avg_hybrid_f1:.2f}%")
    print(f"  LLM Hybrid Exact Match         : {avg_hybrid_em:.2f}%")
    print(f"  LLM Hybrid ROUGE-1             : {avg_hybrid_r1:.2f}%")
    print(f"  LLM Hybrid ROUGE-2             : {avg_hybrid_r2:.2f}%")
    print(f"  LLM Hybrid ROUGE-L             : {avg_hybrid_rl:.2f}%")
    print(f"  LLM Hybrid Contención (gold⊂resp): {avg_hybrid_cont:.2f}%")
    print(f"  LLM Hybrid Token-Recall          : {avg_hybrid_trec:.2f}%")
    print(f"  LLM Hybrid K-Precision           : {avg_hybrid_kprec:.2f}%")
    print(f"  Tasa de Retención del Span (%) : {avg_retention:.2f}%")
    print(f"  Tasa de veredictos SIM (%)     : {avg_verify_sim:.2f}%")
    print(f"  Tasa rama truth en gated (%)   : {avg_gated_truth:.2f}%")
    print(f"  Salida seleccionada EM         : {avg_selected_em:.2f}%")
    print(f"  Salida seleccionada F1         : {avg_selected_f1:.2f}%")
    print(f"  Salida seleccionada ROUGE-L    : {avg_selected_rl:.2f}%")
    print(f"  Salida seleccionada Contención : {avg_selected_cont:.2f}%")
    print(f"  Salida seleccionada Token-Recall: {avg_selected_trec:.2f}%")
    print(f"  Salida seleccionada K-Precision : {avg_selected_kprec:.2f}%")
    print(f"  Pico VRAM inferencia (BERT+LLM): {infer_vram_mb:.1f} MB")
    print(f"  Latencia media por consulta    : {avg_latency:.2f} ms")
    print("="*50 + "\n")

    # Guardar resultados en results/hybrid_rag/
    output_dir = args.output_dir or os.path.join(PARENT_DIR, "results", "hybrid_rag")
    os.makedirs(output_dir, exist_ok=True)
    
    metrics = {
        "bert_model_type": args.bert_model_type,
        "bert_tuning_type": args.bert_tuning_type,
        "bert_exp_id": bert_exp_id,
        "llm_model_type": args.llm_model_type,
        "k": args.k,
        "retriever_type": args.retriever_type,
        "dense_model": args.dense_model,
        "alpha": args.alpha,
        "retriever_hit_at_1": None,  # se llenará dinámicamente si se calcula
        "retriever_hit_at_3": None,
        "reader_exact_match": avg_bert_em,
        "reader_f1": avg_bert_f1,
        "rag_exact_match": avg_hybrid_em,
        "rag_f1": avg_hybrid_f1,
        "rag_rouge1": avg_hybrid_r1,
        "rag_rouge2": avg_hybrid_r2,
        "rag_rougeL": avg_hybrid_rl,
        "rag_containment": avg_hybrid_cont,
        "rag_token_recall": avg_hybrid_trec,
        "rag_k_precision": avg_hybrid_kprec,
        "span_retention_rate": avg_retention,
        "verify_sim_rate": avg_verify_sim,
        "gated_truth_rate": avg_gated_truth,
        "router_mode": args.router_mode,
        "confidence_threshold": args.confidence_threshold,
        "span_guidance": args.span_guidance,
        "selected_exact_match": avg_selected_em,
        "selected_f1": avg_selected_f1,
        "selected_rouge1": avg_selected_r1,
        "selected_rouge2": avg_selected_r2,
        "selected_rougeL": avg_selected_rl,
        "selected_containment": avg_selected_cont,
        "selected_token_recall": avg_selected_trec,
        "selected_k_precision": avg_selected_kprec,
        "peak_vram_mb": infer_vram_mb,
        "average_latency_ms": avg_latency
    }
    
    # Evaluar precisión del retriever para guardar
    hit_at_1 = 0
    hit_at_3 = 0
    for record in test_list:
        gold_context = record.get("context", "")
        gold_context = gold_context.strip() if gold_context else ""
        gold_source = record.get("source", "")
        gold_base = os.path.basename(gold_source).lower() if gold_source else ""
        
        question = record["question"]
        retrieved = retriever.retrieve(question, k=3)
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
            
    metrics["retriever_hit_at_1"] = (hit_at_1 / len(test_list)) * 100
    metrics["retriever_hit_at_3"] = (hit_at_3 / len(test_list)) * 100
    
    output_file = os.path.join(output_dir, "rag_evaluation.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=float)

    predictions_file = os.path.join(output_dir, "hybrid_predictions.json")
    with open(predictions_file, "w", encoding="utf-8") as f:
        json.dump(prediction_records, f, ensure_ascii=False, indent=2, default=float)
        
    print(f"Métricas del pipeline RAG Híbrido escritas en: {output_file}")
    print(f"Predicciones duales escritas en: {predictions_file}")

if __name__ == "__main__":
    main()
