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
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

# Mapas de nombres de modelos
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

HF_TOKEN = os.environ.get("HF_TOKEN", "")

def expand_variants(item_list):
    """
    Expande cada item creando copias del contexto y respuesta
    para cada una de sus variantes de preguntas.
    """
    expanded = []
    for item in item_list:
        expanded.append({
            "id": item["id"],
            "context": item.get("context", "") or "",
            "question": item["question"],
            "answers": item["answers"]
        })
        for idx, variant in enumerate(item.get("question_variants", [])):
            if not variant.strip():
                continue
            expanded.append({
                "id": f"{item['id']}_var{idx}",
                "context": item.get("context", "") or "",
                "question": variant,
                "answers": item["answers"]
            })
    return expanded

def format_prompt_and_completion(context, question, gold_answer):
    """
    Crea la estructura de diálogo (System, User, Assistant) para Fine-Tuning.
    """
    system_instruction = (
        "Você é um extrator de informações preciso. "
        "Responda diretamente e exatamente com o trecho literal do contexto fornecido, sem explicações adicionais."
    )
    user_content = f"Contexto:\n{context.strip()}\n\nPergunta:\n{question.strip()}"
    assistant_content = gold_answer.strip()
    return system_instruction, user_content, assistant_content

def parse_args():
    parser = argparse.ArgumentParser(description="Supervised Fine-Tuning (SFT) de LLMs causales para RAG Generativo.")
    parser.add_argument("--model_type", type=str, required=True, choices=list(MODEL_MAPPING.keys()),
                        help="Modelo LLM a entrenar.")
    parser.add_argument("--tuning_type", type=str, default="qlora", choices=["lora", "qlora"],
                        help="Estrategia PEFT.")
    parser.add_argument("--epochs", type=float, default=5.0, help="Número de épocas.")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=4, help="Tamaño de lote por dispositivo.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Pasos de acumulación de gradiente.")
    parser.add_argument("--max_seq_length", type=int, default=1024, help="Longitud máxima de secuencia.")
    parser.add_argument("--dataset_file", type=str, default=None, help="Ruta al dataset JSON.")
    return parser.parse_args()

def main():
    args = parse_args()
    
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        
    model_name = MODEL_MAPPING[args.model_type]
    print("=" * 60)
    print(f"FINE-TUNING LLM RAG: Modelo: {args.model_type} ({model_name}) | PEFT: {args.tuning_type}")
    print("=" * 60)

    # 1. Cargar el dataset
    dataset_file = args.dataset_file if args.dataset_file else os.path.join(PARENT_DIR, "data", "processed", "corpus_completo.json")
    if not os.path.exists(dataset_file):
        print(f"[ERROR] No se encontró el dataset en '{dataset_file}'")
        sys.exit(1)
        
    with open(dataset_file, "r", encoding="utf-8") as f:
        qa_data = json.load(f)
        
    # Split 80/20 idéntico al de BERT para evitar fugas
    random.seed(SEED)
    shuffled_data = qa_data.copy()
    random.shuffle(shuffled_data)
    split_idx = int(len(shuffled_data) * 0.8)
    train_list = shuffled_data[:split_idx]
    val_list = shuffled_data[split_idx:]
    
    train_expanded = expand_variants(train_list)
    val_expanded = expand_variants(val_list)
    
    print(f"Ejemplos expandidos de Entrenamiento : {len(train_expanded)}")
    print(f"Ejemplos expandidos de Validación    : {len(val_expanded)}")

    # 2. Tokenizador y Modelo Causal
    print(f"Cargando tokenizador y modelo desde {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model_kwargs = {}
    if args.tuning_type == "qlora" and device == "cuda":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
        model_kwargs["quantization_config"] = bnb_config
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["torch_dtype"] = torch.float16 if device == "cuda" else torch.float32
        if device == "cuda":
            model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_name, token=HF_TOKEN, **model_kwargs)
    
    if args.tuning_type == "qlora" and device == "cuda":
        model = prepare_model_for_kbit_training(model)

    # Configuración de LoRA para LLMs CausalLM
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=target_modules
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 3. Función de procesamiento e tokenización con máscaras de loss
    def preprocess_record(record):
        ctx = record.get("context", "")
        q = record["question"]
        ans = record["answers"]["text"][0] if record["answers"].get("text") else ""
        
        sys_inst, user_c, asst_c = format_prompt_and_completion(ctx, q, ans)
        
        messages = [
            {"role": "system", "content": sys_inst},
            {"role": "user", "content": user_c},
            {"role": "assistant", "content": asst_c}
        ]
        
        try:
            full_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
        except Exception:
            full_prompt = f"System: {sys_inst}\nUser: {user_c}\nAssistant: {asst_c}"

        tokenized = tokenizer(
            full_prompt,
            truncation=True,
            max_length=args.max_seq_length,
            padding=False
        )
        
        input_ids = tokenized["input_ids"]
        labels = list(input_ids)
        
        # Enmascarar la parte del prompt para calcular la pérdida sólo sobre la respuesta
        try:
            prompt_only_messages = [
                {"role": "system", "content": sys_inst},
                {"role": "user", "content": user_c}
            ]
            try:
                prompt_text = tokenizer.apply_chat_template(prompt_only_messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                prompt_text = f"System: {sys_inst}\nUser: {user_c}\nAssistant:"
                
            prompt_tokenized = tokenizer(prompt_text, truncation=True, max_length=args.max_seq_length, padding=False)
            prompt_len = len(prompt_tokenized["input_ids"])
            
            for i in range(min(prompt_len, len(labels))):
                labels[i] = -100
        except Exception:
            pass
            
        tokenized["labels"] = labels
        return tokenized

    train_ds = Dataset.from_list([preprocess_record(r) for r in train_expanded])
    val_ds = Dataset.from_list([preprocess_record(r) for r in val_expanded])

    exp_id = f"{args.model_type}_sft_{args.tuning_type}_ep{args.epochs}"
    output_dir = os.path.join(PARENT_DIR, "results", exp_id)
    os.makedirs(output_dir, exist_ok=True)

    initial_vram = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    t0 = time.time()

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        fp16=torch.cuda.is_available(),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        save_total_limit=1,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt")
    )

    print("Iniciando entrenamiento SFT del LLM...")
    trainer.train()
    t1 = time.time()

    elapsed = t1 - t0
    peak_vram = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    used_vram_mb = (peak_vram - initial_vram) / (1024 * 1024)

    best_model_dir = os.path.join(PARENT_DIR, "models", f"{args.model_type}_lora_rag")
    os.makedirs(best_model_dir, exist_ok=True)
    model.save_pretrained(best_model_dir)
    tokenizer.save_pretrained(best_model_dir)

    print(f"\nEntrenamiento SFT completado en {elapsed:.2f} segundos ({elapsed/60:.2f} min).")
    print(f"Adaptador LoRA guardado en: {best_model_dir}")
    print(f"Pico de VRAM utilizado: {used_vram_mb:.2f} MB")
    print("=" * 60)

if __name__ == "__main__":
    main()
