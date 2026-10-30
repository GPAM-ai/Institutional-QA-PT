import os
import sys
import json
import time
import argparse
import random
import string
import re
import collections
from pathlib import Path

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
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
    default_data_collator,
    BitsAndBytesConfig,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

# Semilla global para reproducibilidad
SEED = 42

# Importar funciones de ayuda locales
from qa_utils import prepare_train_features, postprocess_qa_predictions

# Mapas de nombres de modelos
MODEL_MAPPING = {
    "mbert": "bert-base-multilingual-cased",
    "xlmr": "xlm-roberta-base",
    "bertimbau_base": "pierreguillou/bert-base-cased-squad-v1.1-portuguese",
    "bertimbau_large": "pierreguillou/bert-large-cased-squad-v1.1-portuguese",
    "bertugues": "ricardoz/BERTugues-base-portuguese-cased"
}

def normalize_answer(s):
    """Normaliza texto para cálculo de EM y F1 (remueve puntuación, minúsculas, etc.)"""
    def remove_articles(text):
        return re.sub(r'\b(o|a|os|as|um|uma|uns|umas|the|a|an)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))
def compute_f1(a_gold, a_pred):
    gold_toks = normalize_answer(a_gold).split()
    pred_toks = normalize_answer(a_pred).split()
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def normalize_containment(s):
    """Normalización agresiva para métrica de contención: minúsculas, sin
    acentos, sin puntuación, sin artículos, espacios colapsados."""
    import unicodedata
    s = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode('utf-8')
    s = s.lower()
    s = ''.join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r'\b(o|a|os|as|um|uma|uns|umas|de|do|da|dos|das|no|na|nos|nas|the|a|an)\b', ' ', s)
    return ' '.join(s.split())

def compute_containment(a_gold, a_pred):
    """Contención de respuesta (0.0/1.0): ¿la respuesta gold aparece, normalizada,
    dentro de la respuesta predicha?

    No castiga respuestas humanizadas: gold='sao jose dos campos' dentro de
    pred='la universidad esta ubicada en sao jose dos campos' -> 1.0,
    aunque el Exact Match sea 0.
    """
    norm_gold = normalize_containment(a_gold)
    norm_pred = normalize_containment(a_pred)
    if not norm_gold:
        return 1.0 if not norm_pred else 0.0
    return 1.0 if norm_gold in norm_pred else 0.0

def compute_token_recall(a_gold, a_pred):
    """Recall a nivel de tokens: fracción de tokens del gold presentes en la predicción.

    No penaliza respuestas verbosas/humanizadas mientras contengan la respuesta.
    Adlakha et al. (2024) muestran que es la métrica léxica con mayor correlación
    humana para correctness en QA con modelos instruction-following.
    """
    gold_toks = normalize_answer(a_gold).split()
    pred_toks = normalize_answer(a_pred).split()
    if len(gold_toks) == 0:
        return int(len(pred_toks) == 0)
    if len(pred_toks) == 0:
        return 0.0
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    return sum(common.values()) / len(gold_toks)

def compute_k_precision(a_pred, context):
    """K-Precision (fidelidad): fracción de tokens de la respuesta presentes en
    el contexto recuperado. Mide anclaje/factualidad sin penalizar el largo del
    contexto (asimétrico, a diferencia de K-F1). Adlakha et al. (2024).
    """
    pred_toks = normalize_answer(a_pred).split()
    ctx_toks = normalize_answer(context).split()
    if len(pred_toks) == 0:
        return 1.0
    if len(ctx_toks) == 0:
        return 0.0
    common = collections.Counter(pred_toks) & collections.Counter(ctx_toks)
    return sum(common.values()) / len(pred_toks)

def compute_rouge(a_gold, a_pred):
    """
    Computes ROUGE-1, ROUGE-2, and ROUGE-L F1 scores.
    Returns: (rouge_1_f1, rouge_2_f1, rouge_l_f1) as floats between 0.0 and 1.0.
    """
    # Use the same normalisation as SQuAD metrics for consistency
    norm_gold = normalize_answer(a_gold)
    norm_pred = normalize_answer(a_pred)
    
    gold_toks = norm_gold.split()
    pred_toks = norm_pred.split()
    
    if len(gold_toks) == 0 and len(pred_toks) == 0:
        return 1.0, 1.0, 1.0
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return 0.0, 0.0, 0.0

    # ROUGE-1
    common1 = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same1 = sum(common1.values())
    p1 = num_same1 / len(pred_toks)
    r1 = num_same1 / len(gold_toks)
    f1_1 = (2 * p1 * r1) / (p1 + r1) if (p1 + r1) > 0 else 0.0

    # ROUGE-2
    gold_bigrams = [tuple(gold_toks[i:i+2]) for i in range(len(gold_toks) - 1)]
    pred_bigrams = [tuple(pred_toks[i:i+2]) for i in range(len(pred_toks) - 1)]
    
    if len(gold_bigrams) == 0 or len(pred_bigrams) == 0:
        f1_2 = 0.0
    else:
        common2 = collections.Counter(gold_bigrams) & collections.Counter(pred_bigrams)
        num_same2 = sum(common2.values())
        p2 = num_same2 / len(pred_bigrams)
        r2 = num_same2 / len(gold_bigrams)
        f1_2 = (2 * p2 * r2) / (p2 + r2) if (p2 + r2) > 0 else 0.0

    # ROUGE-L (LCS-based)
    m = len(gold_toks)
    n = len(pred_toks)
    L = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                L[i][j] = 0
            elif gold_toks[i-1] == pred_toks[j-1]:
                L[i][j] = L[i-1][j-1] + 1
            else:
                L[i][j] = max(L[i-1][j], L[i][j-1])
    lcs_len = L[m][n]
    pl = lcs_len / len(pred_toks)
    rl = lcs_len / len(gold_toks)
    f1_l = (2 * pl * rl) / (pl + rl) if (pl + rl) > 0 else 0.0

    return f1_1, f1_2, f1_l

def list_to_hf_dataset(data_list):
    """Convierte una lista de diccionarios a un Dataset de Hugging Face"""
    columns = {}
    for record in data_list:
        for key, val in record.items():
            if key not in columns:
                columns[key] = []
            columns[key].append(val)
    return Dataset.from_dict(columns)

def get_reference_context(source_path, gold_answer, info_dir):
    if not source_path:
        return "", -1
    
    rel_path = source_path.replace("\\", "/")
    if rel_path.startswith("data/documents/"):
        rel_path = rel_path[len("data/documents/"):]
    if rel_path.startswith("info/"):
        rel_path = rel_path[5:]
    elif rel_path.startswith("info\\"):
        rel_path = rel_path[5:]
        
    full_path = os.path.join(info_dir, rel_path)
    if not os.path.exists(full_path):
        base_name = os.path.basename(rel_path)
        found = False
        for root, dirs, files in os.walk(info_dir):
            if base_name in files:
                full_path = os.path.join(root, base_name)
                found = True
                break
        if not found:
            return "", -1
            
    try:
        from preprocess_docs import try_decode, clean_text, segment_text
        content, enc = try_decode(full_path)
        cleaned = clean_text(content)
        segments = segment_text(os.path.basename(full_path), cleaned)
        
        if not segments:
            return "", -1
            
        if gold_answer:
            gold_answer_norm = gold_answer.lower().strip()
            for seg in segments:
                seg_ctx = seg["context"].strip()
                if gold_answer_norm in seg_ctx.lower():
                    start_idx = seg_ctx.lower().find(gold_answer_norm)
                    return seg_ctx, start_idx
                    
        return segments[0]["context"].strip(), -1
    except Exception as e:
        print(f"[WARNING] Falló la reconstrucción dinámica de contexto para {source_path}: {e}")
        return "", -1

def expand_variants(item_list):
    """
    Expande cada item creando copias del contexto y respuesta
    para cada una de sus variantes de preguntas. Multiplica las muestras por ~4
    manteniendo la consistencia de los datos.
    """
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

def parse_args():
    parser = argparse.ArgumentParser(description="Entrenamiento de modelos BERT para QA de la UNIFESP.")
    parser.add_argument("--model_type", type=str, required=True, choices=list(MODEL_MAPPING.keys()),
                        help="Modelo a entrenar.")
    parser.add_argument("--tuning_type", type=str, default="lora", choices=["full", "lora", "qlora"],
                        help="Estrategia de ajuste fino.")
    parser.add_argument("--epochs", type=float, default=10.0, help="Número de épocas.")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=16, help="Tamaño de lote (batch size).")
    parser.add_argument("--dataset_file", type=str, default=None, help="Ruta al archivo de dataset JSON para entrenamiento.")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Establecer semillas para reproducibilidad (SEED es constante global = 42)
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        
    print("=" * 60)
    print(f"EXPERIMENTO: Modelo: {args.model_type} | PEFT: {args.tuning_type}")
    print("=" * 60)

    # Cargar el dataset usando la ruta especificada o la por defecto
    dataset_file = args.dataset_file if args.dataset_file else os.path.join(PARENT_DIR, "data", "processed", "corpus_completo.json")
        
    if not os.path.exists(dataset_file):
        print(f"[ERROR] No se encontró el dataset en '{dataset_file}'")
        sys.exit(1)
        
    with open(dataset_file, "r", encoding="utf-8") as f:
        qa_data = json.load(f)
        
    print(f"Total de registros cargados: {len(qa_data)}")

    # Reconstrucción dinámica de contextos de referencia si están ausentes
    info_dir = os.path.join(PARENT_DIR, "data", "documents")
    reconstructed_count = 0
    for record in qa_data:
        if not record.get("context"):
            gold_ans = record["answers"]["text"][0] if record.get("answers") and record["answers"].get("text") else ""
            ctx, start = get_reference_context(record.get("source", ""), gold_ans, info_dir)
            record["context"] = ctx
            
            # Asegurar estructura de answers para SQuAD format
            if "answers" not in record:
                record["answers"] = {"text": [], "answer_start": []}
            if not record["answers"].get("text") and gold_ans:
                record["answers"]["text"] = [gold_ans]
            
            if start != -1:
                record["answers"]["answer_start"] = [start]
            else:
                record["answers"]["answer_start"] = [0]
            reconstructed_count += 1
            
    if reconstructed_count > 0:
        print(f"Reconstruidos dinámicamente {reconstructed_count} contextos de referencia desde la carpeta 'data/documents/'.")

    # Mezclar y realizar partición Train/Val (80/20) por ITEMS para evitar fuga de contextos
    # Re-fijar la semilla JUSTO antes del shuffle para garantizar el mismo split siempre
    random.seed(SEED)
    shuffled_data = qa_data.copy()
    random.shuffle(shuffled_data)
    split_idx = int(len(shuffled_data) * 0.8)
    train_list = shuffled_data[:split_idx]
    val_list = shuffled_data[split_idx:]
    
    print(f"Ítems únicos de Entrenamiento (sin fugas) : {len(train_list)}")
    print(f"Ítems únicos de Validación (sin fugas)    : {len(val_list)}")

    # Expandir con las variantes de preguntas después de separar para evitar fuga de datos
    global train_expanded, val_expanded
    train_expanded = expand_variants(train_list)
    val_expanded = expand_variants(val_list)
    
    print(f"Ejemplos expandidos de Entrenamiento      : {len(train_expanded)}")
    print(f"Ejemplos expandidos de Validación         : {len(val_expanded)}")

    # Convertir a objetos Dataset de Hugging Face
    raw_train_ds = list_to_hf_dataset(train_expanded)
    raw_val_ds = list_to_hf_dataset(val_expanded)

    # Cargar Tokenizador y Modelo
    model_name = MODEL_MAPPING[args.model_type]
    print(f"Cargando tokenizador y modelo para {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Preparar tokenización
    tokenized_train_ds = raw_train_ds.map(
        lambda x: prepare_train_features(x, tokenizer),
        batched=True,
        remove_columns=raw_train_ds.column_names
    )
    
    tokenized_val_ds = raw_val_ds.map(
        lambda x: prepare_train_features(x, tokenizer),
        batched=True,
        remove_columns=raw_val_ds.column_names
    )

    # Configuración de Carga y Quantización
    tuning_mode = args.tuning_type
    if tuning_mode == "qlora" and not torch.cuda.is_available():
        print("WARNING: CUDA no está disponible. Cambiando de QLoRA a LoRA sin cuantización de 4-bits.")
        tuning_mode = "lora"

    model_kwargs = {}
    if tuning_mode == "qlora":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            llm_int8_skip_modules=["qa_outputs"]
        )
        model_kwargs["quantization_config"] = bnb_config
        model_kwargs["device_map"] = "auto"
    else:
        if torch.cuda.is_available():
            model_kwargs["device_map"] = "auto"

    model = AutoModelForQuestionAnswering.from_pretrained(model_name, **model_kwargs)

    # Aplicar adaptadores PEFT si es necesario
    if tuning_mode in ["lora", "qlora"]:
        print("Inyectando adaptadores LoRA...")
        if tuning_mode == "qlora":
            model = prepare_model_for_kbit_training(model)
            
        target_modules = ["query", "value"]
        
        lora_config = LoraConfig(
            task_type=TaskType.QUESTION_ANS,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            bias="none",
            target_modules=target_modules,
            modules_to_save=["qa_outputs"]
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # Definir la función de cálculo de métricas para la evaluación interna de HF
    def compute_metrics(p):
        raw_predictions = p.predictions
        # Post-procesar logits usando la versión expandida de validación
        val_predictions = postprocess_qa_predictions(
            val_expanded,
            tokenized_val_ds,
            raw_predictions
        )
        
        # Calcular Exact Match y F1
        total_f1 = 0.0
        total_em = 0.0
        
        for record in val_expanded:
            rec_id = record["id"]
            pred_text = val_predictions.get(rec_id, "")
            answers = record["answers"]
            
            if not answers["text"]:
                # Pregunta imposible
                em = 1.0 if pred_text == "" else 0.0
                f1 = 1.0 if pred_text == "" else 0.0
            else:
                gold_text = answers["text"][0]
                em = compute_exact(gold_text, pred_text)
                f1 = compute_f1(gold_text, pred_text)
                
            total_f1 += f1
            total_em += em
            
        mean_f1 = (total_f1 / len(val_expanded)) * 100
        mean_em = (total_em / len(val_expanded)) * 100
        
        return {"exact_match": mean_em, "f1": mean_f1}

    # Carpeta de salida relativa al directorio padre
    exp_id = f"{args.model_type}_{args.tuning_type}_bs{args.batch_size}_lr{args.lr}_ep{args.epochs}"
    output_dir = os.path.join(PARENT_DIR, "results", exp_id)
    os.makedirs(os.path.join(PARENT_DIR, "results"), exist_ok=True)

    # Medir memoria GPU inicial
    initial_vram = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=5,
        fp16=torch.cuda.is_available(),
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        save_total_limit=1,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_ds,
        eval_dataset=tokenized_val_ds,
        processing_class=tokenizer,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    # Entrenamiento y medición de eficiencia
    print("Iniciando entrenamiento...")
    start_time = time.time()
    trainer.train()
    end_time = time.time()
    
    elapsed_time = end_time - start_time
    print(f"Entrenamiento completado en {elapsed_time:.2f} segundos.")

    # Evaluar el mejor modelo
    print("Evaluando mejor modelo en el Test Set interno...")
    eval_results = trainer.evaluate()
    print("Resultados finales de validación:")
    print(f"  Exact Match : {eval_results['eval_exact_match']:.2f}%")
    print(f"  F1 Score    : {eval_results['eval_f1']:.2f}%")

    # Medir memoria GPU pico
    peak_vram = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    used_vram_mb = (peak_vram - initial_vram) / (1024 * 1024)

    # Guardar modelo e información de peso en disco en el directorio padre
    best_model_dir = os.path.join(output_dir, "best_model")
    trainer.save_model(best_model_dir)
    
    total_size_bytes = sum(f.stat().st_size for f in Path(best_model_dir).glob('**/*') if f.is_file())
    size_mb = total_size_bytes / (1024 * 1024)

    # Consolidar métricas
    metrics = {
        "model_type": args.model_type,
        "tuning_type": args.tuning_type,
        "train_epochs": args.epochs,
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "train_time_seconds": elapsed_time,
        "train_time_minutes": elapsed_time / 60,
        "peak_vram_mb": used_vram_mb,
        "disk_size_mb": size_mb,
        "validation_exact_match": eval_results["eval_exact_match"],
        "validation_f1": eval_results["eval_f1"]
    }

    # Guardar métricas en un JSON
    metrics_file = os.path.join(output_dir, "metrics.json")
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
        
    print(f"Métricas guardadas en {metrics_file}")
    
    # Guardar historial de épocas en CSV para graficación
    epoch_logs = []
    for entry in trainer.state.log_history:
        if "eval_loss" in entry and "epoch" in entry:
            epoch = entry["epoch"]
            eval_loss = entry["eval_loss"]
            eval_f1 = entry.get("eval_f1", 0.0)
            eval_em = entry.get("eval_exact_match", 0.0)
            
            # Buscar la pérdida de entrenamiento más cercana a esta época
            train_loss = None
            closest_epoch_diff = float("inf")
            for t_entry in trainer.state.log_history:
                if "loss" in t_entry and "epoch" in t_entry:
                    diff = abs(t_entry["epoch"] - epoch)
                    if diff < closest_epoch_diff:
                        closest_epoch_diff = diff
                        train_loss = t_entry["loss"]
            
            epoch_logs.append({
                "epoch": epoch,
                "train_loss": train_loss if train_loss is not None else "N/A",
                "val_loss": eval_loss,
                "val_f1": eval_f1,
                "val_em": eval_em
            })
            
    if epoch_logs:
        import csv
        epoch_history_file = os.path.join(output_dir, "epoch_history.csv")
        with open(epoch_history_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "val_f1", "val_em"])
            writer.writeheader()
            writer.writerows(epoch_logs)
        print(f"Historial de épocas guardado en {epoch_history_file}")

    print(f"Modelo guardado en {best_model_dir} (Tamaño: {size_mb:.2f} MB)")
    print(f"Pico de VRAM utilizada: {used_vram_mb:.2f} MB")
    print("=" * 60)

if __name__ == "__main__":
    main()
