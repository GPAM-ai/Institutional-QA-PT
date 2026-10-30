"""Gera seis gráficos em português exclusivamente com métricas auditadas."""
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="qa-mpl-"))
os.environ.setdefault("XDG_CACHE_HOME", tempfile.mkdtemp(prefix="qa-font-cache-"))
# Neste cluster, importar NumPy antes de pyplot evita problemas de inicialização.
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
AUDIT = json.loads((ROOT / "reproducibility/verified_results.json").read_text())
ROWS = AUDIT["extractive_best_validation"]
PRED = AUDIT["prediction_audit"]
BY_KEY = {(r["model_type"], r["tuning_type"]): r for r in ROWS}
NAVY, TEAL, BRASS = "#2F3E4E", "#6A9A8B", "#C2A878"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "axes.titleweight": "bold"})
FOOTNOTE = "Validação reutilizada; resultados exploratórios, sem teste independente."


def save(fig, name, note=FOOTNOTE):
    fig.text(.5, .015, note, ha="center", va="bottom", fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, .07, 1, 1))
    for suffix in ["pdf", "png"]:
        fig.savefig(OUT / f"{name}.{suffix}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Gerado: figures/{name}.pdf e .png")


# 1. Leitores extrativos com contexto de referência.
models = ["mbert", "xlmr", "bertugues", "bertimbau_base", "bertimbau_large"]
tunings = ["full", "lora", "qlora"]
x = np.arange(len(models))
fig, ax = plt.subplots(figsize=(9, 5.5))
for i, tuning in enumerate(tunings):
    values = [BY_KEY[m, tuning]["validation_f1"] for m in models]
    bars = ax.bar(x + (i - 1) * .26, values, .26,
                  color=[NAVY, TEAL, BRASS][i], label=["Ajuste completo", "LoRA", "QLoRA"][i])
    ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)
ax.set(xticks=x, xticklabels=["mBERT", "XLM-R", "BERTugues", "BERTimbau Base", "BERTimbau Large"],
       ylabel="F1 (%)", ylim=(0, 85), title="Leitores extrativos com contexto de referência")
ax.grid(axis="y", alpha=.2)
ax.set_axisbelow(True)
ax.legend(loc="upper left", ncol=3, frameon=False)
save(fig, "extractive_f1")

# 2. Incremento de alocação de treinamento, e não pico total de memória.
fig, ax = plt.subplots(figsize=(9, 5.5))
for model, color, label in [("bertimbau_base", BRASS, "Base"), ("bertimbau_large", NAVY, "Large")]:
    selected = [BY_KEY[model, t] for t in tunings]
    xs = [r["reported_memory_increment_gib"] for r in selected]
    ys = [r["validation_f1"] for r in selected]
    ax.plot(xs, ys, color=color, alpha=.4)
    for i, row in enumerate(selected):
        ax.scatter(xs[i], ys[i], marker=["o", "s", "^"][i], color=color, s=100)
        strategy = ["Completo", "LoRA", "QLoRA"][i]
        label_text = f"{label} {strategy}\n{xs[i]:.2f} GiB"
        ax.annotate(label_text, (xs[i], ys[i]), xytext=(0, 12 if i != 1 else -30),
                    textcoords="offset points", ha="center", fontsize=8)
ax.set(xlabel="Incremento de alocação no treinamento (GiB)", ylabel="F1 (%)",
       title="Memória e F1: BERTimbau com contexto de referência", xlim=(-1, 42), ylim=(62, 78))
ax.grid(alpha=.2)
handles = [Line2D([], [], color=BRASS, label="BERTimbau Base"),
           Line2D([], [], color=NAVY, label="BERTimbau Large")]
handles += [Line2D([], [], color="gray", marker=m, linestyle="", label=t)
            for m, t in zip(["o", "s", "^"], ["Completo", "LoRA", "QLoRA"])]
ax.legend(handles=handles, fontsize=8, loc="upper right", ncol=2)
save(fig, "memory_f1")


def grouped_scores(keys, labels, title, name, note=FOOTNOTE):
    fig, ax = plt.subplots(figsize=(10, 6))
    positions = np.arange(len(keys))
    for i, (metric, color) in enumerate([("f1", NAVY), ("em", TEAL)]):
        values = [PRED[key][metric] for key in keys]
        bars = ax.barh(positions + (i - .5) * .34, values, .34, color=color, label=metric.upper())
        ax.bar_label(bars, fmt="%.2f", padding=4, fontsize=9)
    ax.set(yticks=positions, yticklabels=labels, xlabel="Pontuação (%)", xlim=(0, 100), title=title)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=.2)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=False)
    save(fig, name, note)


# 3. Comparação apenas entre geradores sem SFT no domínio, com contexto recuperado.
grouped_scores(["llama3b", "qwen15b", "llama1b", "smollm2", "tucano"],
    ["Llama-3.2-3B", "Qwen2.5-1.5B", "Llama-3.2-1B", "SmolLM2-1.7B", "Tucano-1B1"],
    "Geradores sem adaptação supervisionada no domínio", "generators_retrieved",
    "Contexto recuperado; exemplos no prompt habilitados. " + FOOTNOTE)

# 4. Contextos apresentados em painéis separados: os protocolos BERT diferem.
fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), sharey=True)
for ax, title, f1, em in [
    (axes[0], "Phi-4-Mini adaptado", [PRED["phi_gold"]["f1"], PRED["phi_retrieved"]["f1"]],
     [PRED["phi_gold"]["em"], PRED["phi_retrieved"]["em"]]),
    (axes[1], "BERTimbau Large LoRA", [BY_KEY["bertimbau_large", "lora"]["validation_f1"], PRED["retrieved_extractor"]["f1"]],
     [BY_KEY["bertimbau_large", "lora"]["validation_exact_match"], PRED["retrieved_extractor"]["em"]])]:
    x = np.arange(2)
    for i, (label, values, color) in enumerate([("F1", f1, NAVY), ("EM", em, TEAL)]):
        bars = ax.bar(x + (i - .5) * .34, values, .34, label=label, color=color)
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
    ax.set(xticks=x, xticklabels=["Referência", "Recuperado"], ylim=(0, 100), title=title)
    ax.grid(axis="y", alpha=.2)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
axes[0].set_ylabel("Pontuação (%)")
save(fig, "context_comparison", "BERT usa decodificadores diferentes entre regimes; comparação descritiva. " + FOOTNOTE)

# 5. Identificação explícita de adaptação, orientação e roteamento.
grouped_scores(["retrieved_extractor", "qwen_zs_guided", "qwen_guided", "qwen_gated8", "qwen_verify", "qwen_verdict", "qwen_guarded"],
    ["Extrator recuperado", "Orientado sem SFT", "Orientado com SFT", "Roteamento com SFT (limiar 8)",
     "Verificação de candidato sem SFT", "Verificador SIM/NÃO sem SFT", "Fallback ao span sem SFT"],
    "Variantes híbridas com Qwen2.5-1.5B", "hybrid_variants",
    "Recuperação 0,5/0,5; limiar escolhido no holdout. Ganhos híbridos inconclusivos.")

# 6. Intervalos de diferenças pareadas: não confundir com barras de F1 absoluto.
contrasts = AUDIT["paired_cluster_bootstrap"]
keys = ["phi_gold minus phi_retrieved", "qwen_gated8 minus qwen_guided", "qwen_guided minus retrieved_extractor"]
labels = ["Phi: referência − recuperado", "Roteamento − sempre orientado (SFT)", "Sempre orientado (SFT) − extrator"]
fig, ax = plt.subplots(figsize=(11, 4.6))
for i, key in enumerate(keys):
    row = contrasts[key]
    value = row["question_macro_difference"]
    lo, hi = row["ci95"]
    ax.errorbar(value, i, xerr=[[value - lo], [hi - value]], fmt="o", capsize=5,
                color=[NAVY, TEAL, BRASS][i], markersize=7)
    ax.annotate(f"{value:+.2f} [{lo:.2f}; {hi:.2f}]", (hi, i), xytext=(8, 0),
                textcoords="offset points", va="center", fontsize=9)
ax.axvline(0, color="#555555", linestyle="--", linewidth=1)
ax.set(yticks=np.arange(len(keys)), yticklabels=labels,
       xlabel="Diferença de F1 Q-macro (pontos percentuais)",
       title="Diferenças pareadas e intervalos bootstrap de 95%", xlim=(-7, 46), ylim=(2.6, -.6))
ax.grid(axis="x", alpha=.2)
save(fig, "paired_differences", "10.000 reamostragens por pergunta original; intervalos não corrigem o viés de seleção.")
