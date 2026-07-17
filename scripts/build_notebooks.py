"""
Generatore dei notebook gemelli (IT / EN) per l'analisi della tesi.
Tiene allineate le due versioni: il CODICE e' identico, cambiano solo i testi
markdown. Lo script viene esteso uno step per volta.

Uso:
    python build_notebooks.py
Produce:
    analisi_tesi_IT.ipynb
    analysis_thesis_EN.ipynb
"""
import nbformat as nbf

# ---------------------------------------------------------------------------
# CODICE (condiviso tra le due versioni)
# ---------------------------------------------------------------------------

CODE_SETUP = '''\
# --- Setup ---
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="notebook")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 160)

# Data paths (relative to the Benchmark_ML folder)
DATA_DIR    = Path("Data")
BENCH_PATH  = DATA_DIR / "benchmark_results.csv"   # real measurements (vLLM)
SIZING_PATH = DATA_DIR / "sizing_results.csv"      # theoretical model (llm-sizing-tool)

# Experimental design factors and metrics of interest
DESIGN_FACTORS = ["model", "gpu_type", "tensor_parallelism",
                  "input_length", "output_length", "concurrent_users"]
METRICS = ["ttft_avg_ms", "itl_avg_ms", "request_latency_avg_ms",
           "output_token_throughput", "power_w"]
print("Setup OK")'''

CODE_LOAD = '''\
bench  = pd.read_csv(BENCH_PATH)
sizing = pd.read_csv(SIZING_PATH)

print(f"benchmark_results.csv : {bench.shape[0]:>3} rows x {bench.shape[1]} cols  (real measurements)")
print(f"sizing_results.csv    : {sizing.shape[0]:>3} rows x {sizing.shape[1]} cols  (theoretical model)")
print(f"\\nIdentical column schema: {list(bench.columns) == list(sizing.columns)}")
bench.head(3)'''

CODE_FILTER = '''\
# --- Data consistency: keep only the GPUs present in the benchmark ---
# The real benchmark covers only H100 PCIe and H200 NVLink (SXM). We restrict
# the theoretical file to the same two GPU types so the comparison is fair.
def is_benchmark_gpu(g):
    g = str(g)
    if "H100" in g and "PCIe" in g:      # H100 PCIe
        return True
    if "H200" in g and "PCIe" not in g:  # H200 NVLink (SXM)
        return True
    return False

n_before = len(sizing)
sizing = sizing[sizing["gpu_type"].map(is_benchmark_gpu)].reset_index(drop=True)
print(f"Theoretical filtered to H100 PCIe / H200 NVLink: {n_before} -> {len(sizing)} rows")
print("Raw labels, real        :", sorted(bench["gpu_type"].unique()))
print("Raw labels, theoretical :", sorted(sizing["gpu_type"].unique()))

# Normalize gpu_type to the GPU IDENTITY in both files. The raw labels differ
# only in the "Nx" prefix: in the real file it is the node size (8 GPUs
# installed; a TP=k run uses k of them), in the theoretical file it is the TP
# itself. Either way the GPUs actually used = tensor_parallelism, which stays
# a design factor of its own, so the prefix carries no extra information and
# would otherwise show up in the plots as 8 pseudo-GPU series.
bench["gpu_type"]  = np.where(bench["gpu_type"].str.contains("H100"),  "H100 PCIe", "H200 NVLink")
sizing["gpu_type"] = np.where(sizing["gpu_type"].str.contains("H100"), "H100 PCIe", "H200 NVLink")
print("Normalized gpu_type     :", sorted(bench["gpu_type"].unique()), "(same in both files)")'''

CODE_COMPLETENESS = '''\
# Non-null cells per column, in both datasets
completeness = pd.DataFrame({
    "bench_non_null":  bench.notna().sum(),
    "sizing_non_null": sizing.notna().sum(),
})
completeness["bench_%"]  = (completeness["bench_non_null"]  / len(bench)  * 100).round(0)
completeness["sizing_%"] = (completeness["sizing_non_null"] / len(sizing) * 100).round(0)
completeness'''

CODE_EMPTY = '''\
empty_sizing = [c for c in sizing.columns if sizing[c].isna().all()]
empty_bench  = [c for c in bench.columns  if bench[c].isna().all()]
print("Fully empty columns - THEORETICAL :", empty_sizing)
print("Fully empty columns - REAL        :", empty_bench)

# Comparable metrics = populated in BOTH files
comparable_metrics = [m for m in METRICS if bench[m].notna().any() and sizing[m].notna().any()]
print("\\nComparable metrics (real vs theoretical):", comparable_metrics)'''

CODE_COVERAGE = '''\
# Unique levels of each design factor, in both datasets
def factor_levels(df):
    return {f: sorted(df[f].dropna().unique().tolist(), key=str) for f in DESIGN_FACTORS}

lev_b, lev_s = factor_levels(bench), factor_levels(sizing)
for f in DESIGN_FACTORS:
    print(f"\\n{f}")
    print(f"  REAL        ({len(lev_b[f])}): {lev_b[f]}")
    print(f"  THEORETICAL ({len(lev_s[f])}): {lev_s[f]}")'''

CODE_COUNTS = '''\
# Number of runs per model in each dataset
counts = pd.concat([
    bench.groupby("model").size().rename("bench"),
    sizing.groupby("model").size().rename("sizing"),
], axis=1)
counts'''

CODE_DESCRIBE = '''\
print("REAL - descriptive statistics of metrics")
display(bench[METRICS].describe().T.round(2))
print("\\nTHEORETICAL - descriptive statistics of metrics")
display(sizing[comparable_metrics].describe().T.round(2))'''

# ---- Step 2: exploratory visualizations -----------------------------------

CODE_DIST = '''\
# Distribution of the comparable metrics in the REAL benchmark, split by GPU
fig, axes = plt.subplots(1, len(METRICS), figsize=(4.2 * len(METRICS), 4.2))
for ax, m in zip(axes, METRICS):
    sns.boxplot(data=bench, x="model", y=m, hue="gpu_type", ax=ax)
    ax.set_title(m)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=90)
    ax.legend_.remove() if ax.get_legend() else None
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper right", ncol=2, title="gpu_type")
fig.suptitle("REAL benchmark - metric distributions by model and GPU", y=1.04)
plt.tight_layout()
plt.show()'''

CODE_TREND_HELPER = '''\
def trend(df, metric, title):
    """Line plot of `metric` vs concurrency, one panel per model.
    Colour = TP (distinct colors, fixed order), dash/marker = GPU identity:
    the same 2 GPU x 4 TP series in the real and theoretical plots."""
    d = df.assign(TP=df["tensor_parallelism"].astype(int).astype(str))
    g = sns.relplot(
        data=d, x="concurrent_users", y=metric,
        col="model", col_wrap=3,
        hue="TP", hue_order=["1", "2", "4", "8"], palette="tab10",
        style="gpu_type", markers=True, dashes=True,
        kind="line", height=3, aspect=1.25,
        facet_kws={"sharey": False},
    )
    g.set_titles("{col_name}")
    g.figure.suptitle(title, y=1.02)
    for ax in g.axes.flat:
        ax.set_xlabel("concurrent users")
        ax.grid(True, alpha=0.3)
    plt.show()

def bars(df, metric, title):
    """Bar plot of `metric` by GPU and TP, one panel per model, averaged over
    the concurrency levels (error bars = std across concurrency: tiny when the
    metric is ~flat in C). Colour = TP, same palette/order as trend()."""
    d = df.assign(TP=df["tensor_parallelism"].astype(int).astype(str))
    g = sns.catplot(
        data=d, x="gpu_type", y=metric,
        col="model", col_wrap=3,
        hue="TP", hue_order=["1", "2", "4", "8"], palette="tab10",
        kind="bar", errorbar="sd", height=3, aspect=1.25,
    )
    g.set_titles("{col_name}")
    g.figure.suptitle(title, y=1.02)
    for ax in g.axes.flat:
        ax.set_xlabel("")
        ax.grid(True, axis="y", alpha=0.3)
    plt.show()'''

CODE_TREND_TP = '''\
# Headline metric: throughput. Watch the OPPOSITE direction vs concurrency.
trend(bench,  "output_token_throughput", "REAL - output token throughput vs concurrency")
trend(sizing, "output_token_throughput", "THEORETICAL - output token throughput vs concurrency")'''

CODE_TREND_LAT = '''\
trend(bench,  "request_latency_avg_ms", "REAL - request latency (ms) vs concurrency")
trend(sizing, "request_latency_avg_ms", "THEORETICAL - request latency (ms) vs concurrency")'''

CODE_TREND_TTFT = '''\
trend(bench,  "ttft_avg_ms", "REAL - TTFT (ms) vs concurrency")
trend(sizing, "ttft_avg_ms", "THEORETICAL - TTFT (ms) vs concurrency")'''

CODE_TREND_ITL = '''\
trend(bench,  "itl_avg_ms", "REAL - ITL (ms) vs concurrency")
trend(sizing, "itl_avg_ms", "THEORETICAL - ITL (ms) vs concurrency")'''

CODE_TREND_PW = '''\
# Power is ~flat vs concurrency, so lines add nothing: bar chart by GPU and TP,
# averaged over concurrency (error bars = std across C, indeed tiny). Note the
# gap between the measured draw and the theoretical nameplate (tdp x n_gpu),
# and how the REAL per-GPU draw is sub-linear in TP.
bars(bench,  "power_w", "REAL - measured power draw (W) by GPU and TP")
bars(sizing, "power_w", "THEORETICAL - power (W) = tdp x n_gpu")'''

# ---- Step 3: alignment / join ---------------------------------------------

CODE_KEYS = '''\
# gpu_type was normalized to the GPU identity in step 1.0; in BOTH files the
# GPUs actually used = tensor_parallelism (confirmed: a real TP=k run uses k
# of the node's 8 GPUs). We decompose the identity into architecture and
# interconnect and align on the effective GPU count n_gpu = TP.
def add_keys(df):
    df = df.copy()
    g = df["gpu_type"].astype(str)
    df["gpu_arch"]     = np.where(g.str.contains("H100"), "H100", "H200")
    df["interconnect"] = np.where(g.str.contains("PCIe"), "PCIe", "NVLink")
    df["n_gpu"]        = df["tensor_parallelism"].astype(int)
    return df

KEYS = ["model", "gpu_arch", "interconnect", "n_gpu", "concurrent_users"]
bench_k, sizing_k = add_keys(bench), add_keys(sizing)
bench_k[["model", "gpu_type", "gpu_arch", "interconnect", "n_gpu", "concurrent_users"]].head()'''

CODE_AGG = '''\
# A config is fully identified by KEYS (input/output are fixed at 512->128).
# Collapse any repeated runs by averaging the comparable metrics
# (deterministic / a no-op for the theoretical file).
dup_real = bench_k.duplicated(KEYS).sum()
dup_theo = sizing_k.duplicated(KEYS).sum()
print(f"Duplicate-key rows  ->  real: {dup_real}  |  theoretical: {dup_theo}")

real_agg = bench_k.groupby(KEYS, as_index=False)[comparable_metrics].mean()
theo_agg = sizing_k.groupby(KEYS, as_index=False)[comparable_metrics].mean()
print(f"Distinct configs    ->  real: {len(real_agg)}  |  theoretical: {len(theo_agg)}")'''

CODE_MERGE = '''\
# Inner join: paired real <-> theoretical for the same config.
paired = real_agg.merge(theo_agg, on=KEYS, suffixes=("_real", "_theo"))
print(f"Paired configs (inner join): {len(paired)}")
paired.head()'''

CODE_COVERAGE3 = '''\
# What did NOT pair, and why.
real_only = real_agg.merge(theo_agg[KEYS], on=KEYS, how="left", indicator=True)
real_only = real_only[real_only["_merge"] == "left_only"]
theo_only = theo_agg.merge(real_agg[KEYS], on=KEYS, how="left", indicator=True)
theo_only = theo_only[theo_only["_merge"] == "left_only"]

print(f"Real configs WITHOUT theoretical match : {len(real_only)}")
print("   by n_gpu (TP)        :", real_only["n_gpu"].value_counts().sort_index().to_dict())
print("   by concurrent_users  :", real_only["concurrent_users"].value_counts().sort_index().to_dict())
print(f"\\nTheoretical configs WITHOUT real match : {len(theo_only)}")
print("   by n_gpu (TP)        :", theo_only["n_gpu"].value_counts().sort_index().to_dict())
print("   by concurrent_users  :", theo_only["concurrent_users"].value_counts().sort_index().to_dict())

print("\\nPaired set coverage across design factors:")
for k in ["model", "gpu_arch", "interconnect", "n_gpu", "concurrent_users"]:
    print(f"   {k:18s}: {sorted(paired[k].unique().tolist())}")'''

# ---- Step 4: comparison real vs theoretical -------------------------------

CODE_ERR_SETUP = '''\
# Error convention: error = theoretical - real  (positive => theory OVERestimates).
err = paired.copy()
for m in comparable_metrics:
    err[f"{m}__err"] = err[f"{m}_theo"] - err[f"{m}_real"]
    err[f"{m}__pe"]  = (err[f"{m}_theo"] - err[f"{m}_real"]) / err[f"{m}_real"] * 100

def error_summary(df, metrics):
    out = []
    for m in metrics:
        pe = df[f"{m}__pe"]; ae = df[f"{m}__err"].abs()
        out.append({"metric": m, "n": int(df[f"{m}_real"].notna().sum()),
                    "MAE": ae.mean(), "MAPE_%": pe.abs().mean(), "bias_%": pe.mean()})
    return pd.DataFrame(out).set_index("metric").round(2)

def scatter_vs(df, metrics, title, hue="concurrent_users", log=False, savepath=None):
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.3 * len(metrics), 4))
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, metrics):
        sns.scatterplot(data=df, x=f"{m}_real", y=f"{m}_theo", hue=hue,
                        palette="viridis" if hue == "concurrent_users" else "tab10",
                        s=40, ax=ax, legend=(ax is axes[-1]))
        lo = float(np.nanmin([df[f"{m}_real"].min(), df[f"{m}_theo"].min()]))
        hi = float(np.nanmax([df[f"{m}_real"].max(), df[f"{m}_theo"].max()]))
        if log:
            lo = max(lo, 1e-2)
            ax.set_xscale("log"); ax.set_yscale("log")
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.6)
        ax.set(title=m, xlabel="real (measured)", ylabel="theoretical (predicted)")
    fig.suptitle(title, y=1.04)
    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.show()'''

CODE_ASIS_TABLE = '''\
print("AS-IS: theoretical vs real on ALL comparable metrics (paired configs)")
print("bias_% > 0 => theory overestimates. Huge MAPE on throughput/latency = different quantities.\\n")
error_summary(err, comparable_metrics)'''

CODE_ASIS_SCATTER = '''\
scatter_vs(err, comparable_metrics, "AS-IS: theoretical vs real  (dashed = perfect agreement)")'''

CODE_APPROACH_B = '''\
# Approach (b): convert REAL aggregate throughput to per-request, matching the
# theoretical definition (N_out / single-request latency).
bdf = err.copy()
bdf["output_token_throughput_real"] = bdf["output_token_throughput_real"] / bdf["concurrent_users"]
bdf["output_token_throughput__pe"]  = ((bdf["output_token_throughput_theo"] - bdf["output_token_throughput_real"])
                                       / bdf["output_token_throughput_real"] * 100)

asis  = err["output_token_throughput__pe"].abs().mean()
normb = bdf["output_token_throughput__pe"].abs().mean()
print(f"Throughput MAPE  as-is (real = aggregate)   : {asis:8.1f} %")
print(f"Throughput MAPE  approach (b) per-request   : {normb:8.1f} %")
scatter_vs(bdf, ["output_token_throughput"],
           "Approach (b): theoretical vs REAL per-request throughput")'''

CODE_C_TABLE = '''\
clean = ["ttft_avg_ms", "itl_avg_ms"]
print("Approach (c) - quantitative comparison on clean per-token metrics\\n")
print("All paired configs:")
display(error_summary(err, clean))
print("Concurrency = 1 only (definitions align best):")
display(error_summary(err[err["concurrent_users"] == 1], clean))'''

CODE_C_BREAKDOWN = '''\
def mape_by(df, by):
    cols = ["ttft_avg_ms__pe", "itl_avg_ms__pe"]
    g = df.groupby(by)[cols].apply(lambda d: pd.Series({
        "TTFT_MAPE_%": d["ttft_avg_ms__pe"].abs().mean(),
        "ITL_MAPE_%":  d["itl_avg_ms__pe"].abs().mean(),
        "n": len(d),
    }))
    g["n"] = g["n"].astype(int)
    return g.round(1)

print("TTFT/ITL error by concurrency:");  display(mape_by(err, "concurrent_users"))
print("by interconnect (H100/PCIe vs H200/NVLink):"); display(mape_by(err, "interconnect"))
print("by model:");                       display(mape_by(err, "model"))'''

CODE_C_SCATTER = '''\
scatter_vs(err[err["concurrent_users"] == 1], ["ttft_avg_ms", "itl_avg_ms"],
           "Approach (c): TTFT & ITL at concurrency = 1")'''

# ---- Step 5.1: spec table + feature engineering ---------------------------

CODE_SPECS = '''\
# Physical specs FROZEN from the tool's own model resolution, so the features are
# consistent with the theoretical baseline. (gpt-oss is MoE: weights use total
# params, compute/bandwidth timings use ACTIVE params.)
N_IN, N_OUT = 512, 128

# PERF_MODEL constants of the CALIBRATED roofline model (fitted on this same
# benchmark with llm-sizing-tool/backend/calibrate_perf_model.py).
MFU, MBU = 0.413, 0.643               # effective FLOPs / bandwidth utilization
ETA = {"NVLink": 0.361, "PCIe": 0.372}  # TP scaling efficiency (1.0 when n_gpu == 1)
# NOTE (2026-06-11): parameters re-fitted on the 130 DENSE paired configs only.
# gpt-oss-120b (MoE) is excluded from the fit: with the model-card active params
# (5.1B, arXiv:2508.10925) the roofline underestimates it ~5x -> outside the
# closed-form validity domain (effective serving cost ~5x nominal active params).

MODEL_SPECS = {
    #                                params      active      is_moe layers hidden heads kv_heads
    "Llama-3.1-8B-Instruct":        dict(params=8.03e9,   active=8.03e9,   is_moe=0, layers=32, hidden=4096, heads=32, kv_heads=8),
    "Qwen2.5-7B-Instruct":          dict(params=7.616e9,  active=7.616e9,  is_moe=0, layers=28, hidden=3584, heads=28, kv_heads=4),
    "Qwen2.5-32B-Instruct":         dict(params=3.276e10, active=3.276e10, is_moe=0, layers=64, hidden=5120, heads=40, kv_heads=8),
    "DeepSeek-R1-Distill-Qwen-32B": dict(params=3.276e10, active=3.276e10, is_moe=0, layers=64, hidden=5120, heads=40, kv_heads=8),
    "Llama-3.3-70B-Instruct":       dict(params=7.055e10, active=7.055e10, is_moe=0, layers=80, hidden=8192, heads=64, kv_heads=8),
    # gpt-oss specs pinned from the model card (arXiv:2508.10925): 116.8B total, 5.1B active.
    # (Earlier: 120.4B/27.09B from HF safetensors + the 80%-experts heuristic of the tool.)
    "gpt-oss-120b":                 dict(params=1.168e11, active=5.1e9,    is_moe=1, layers=36, hidden=2880, heads=64, kv_heads=8),
}
# Dense-TFLOPS specs of the ACTUAL benchmark hardware: H100 = PCIe profile,
# H200 = SXM/NVLink (the same GPU profiles the tool uses).
GPU_SPECS_FE = {
    "H100": dict(mem_bw=2000, fp16_tflops=756, vram_gb=80,  tdp_w=350),
    "H200": dict(mem_bw=4800, fp16_tflops=989, vram_gb=141, tdp_w=700),
}
print("Model specs (frozen):")
display(pd.DataFrame(MODEL_SPECS).T)
print("GPU specs (frozen):")
display(pd.DataFrame(GPU_SPECS_FE).T)
print("Note: DeepSeek-R1-Distill-Qwen-32B and Qwen2.5-32B share identical specs "
      "(the distill is based on Qwen2.5-32B) -> indistinguishable in feature space.")'''

CODE_FEATS = '''\
def build_features(df):
    """Add physics-based features replicating the CALIBRATED roofline model
    (gpu_calc.py + PERF_MODEL). Works on any df with columns
    model, gpu_arch, interconnect, n_gpu, concurrent_users."""
    df = df.copy()
    for k in ["params", "active", "is_moe", "layers", "hidden", "heads", "kv_heads"]:
        df[k if k != "active" else "active_params"] = df["model"].map(lambda m: MODEL_SPECS[m][k])
    for k in ["mem_bw", "fp16_tflops", "vram_gb", "tdp_w"]:
        df[k] = df["gpu_arch"].map(lambda g: GPU_SPECS_FE[g][k])
    df["nvlink"] = (df["interconnect"] == "NVLink").astype(int)
    df["params_per_layer"] = df["params"] / df["layers"]

    # Effective aggregate resources of the TP group (calibrated roofline)
    n, C = df["n_gpu"], df["concurrent_users"]
    df["tp_eff"]          = np.where(n == 1, 1.0, df["interconnect"].map(ETA))
    df["agg_tflops_eff"]  = n * df["fp16_tflops"] * MFU * df["tp_eff"]
    df["agg_bw_eff"]      = n * df["mem_bw"] * MBU * df["tp_eff"]
    df["queue_factor"]    = (C + 1) / 2                  # closed-loop TTFT queueing
    df["arith_intensity"] = df["fp16_tflops"] / df["mem_bw"]

    Pb_act, byt = df["active_params"] / 1e9, 2           # FP16 = 2 bytes/param
    df["ttft_core"] = (2 * Pb_act / df["agg_tflops_eff"]) * N_IN   # single-request prefill, ms

    # Decode roofline terms: weights read once per step + whole-batch KV at the
    # average generation context, vs batched compute. head_dim = hidden/heads.
    head_dim = df["hidden"] / df["heads"]
    df["kv_per_token_gb"]   = 2 * df["layers"] * df["kv_heads"] * head_dim * byt / 1e9
    df["kv_read_gb"]        = df["kv_per_token_gb"] * (N_IN + N_OUT / 2) * C
    df["weights_gb"]        = df["params"] * byt / 1e9   # resident (total params)
    df["weights_active_gb"] = Pb_act * byt               # read per decode step (active)
    df["itl_mem_core"]      = (df["weights_active_gb"] + df["kv_read_gb"]) / df["agg_bw_eff"] * 1000
    df["itl_compute_core"]  = (2 * Pb_act / df["agg_tflops_eff"]) * C
    df["vram_total_gb"]     = df["vram_gb"] * df["n_gpu"]
    df["vram_pressure"]     = (df["weights_gb"] + df["kv_per_token_gb"] * (N_IN + N_OUT) * C) / df["vram_total_gb"]
    return df

FEATURES = [
    "params", "active_params", "is_moe", "layers", "hidden", "heads", "kv_heads", "params_per_layer",
    "mem_bw", "fp16_tflops", "vram_gb", "tdp_w", "nvlink",
    "n_gpu", "concurrent_users", "queue_factor", "tp_eff",
    "agg_tflops_eff", "agg_bw_eff", "arith_intensity",
    "ttft_core", "itl_mem_core", "itl_compute_core",
    "kv_per_token_gb", "kv_read_gb", "weights_gb", "weights_active_gb", "vram_total_gb", "vram_pressure",
]
TARGETS = ["ttft_avg_ms", "itl_avg_ms", "power_w"]   # power_w: ML corrects the tdp*n nameplate
BIN_FEATURES = ["is_moe", "nvlink"]                            # pass through (binary)
LOG_FEATURES = [f for f in FEATURES if f not in BIN_FEATURES]  # strictly positive -> log-transform

feat = build_features(paired)
print(f"Feature matrix: {feat.shape[0]} configs x {len(FEATURES)} features  |  targets: {TARGETS}")
feat[["model", "gpu_arch", "n_gpu", "concurrent_users"] + FEATURES].head()'''

CODE_FE_SANITY = '''\
# Sanity check: rebuild the THEORETICAL TTFT/ITL/power from the engineered features
# and compare to the tool's output. We check the RELATIVE error (the meaningful one:
# TTFT spans ms..1e5 ms, so an absolute threshold would be misleading).
ttft_rebuilt  = feat["ttft_core"] * feat["queue_factor"]
itl_rebuilt   = np.maximum(feat["itl_mem_core"], feat["itl_compute_core"])
power_rebuilt = feat["tdp_w"] * feat["n_gpu"]
def max_rel_err(rebuilt, tool_val):
    return (np.abs(rebuilt - tool_val) / tool_val.abs()).max() * 100
rel = {
    "TTFT":  max_rel_err(ttft_rebuilt,  feat["ttft_avg_ms_theo"]),
    "ITL":   max_rel_err(itl_rebuilt,   feat["itl_avg_ms_theo"]),
    "power": max_rel_err(power_rebuilt,  feat["power_w_theo"]),
}
for k, v in rel.items():
    print(f"Max relative error  {k:6s}: {v:.4f} %")
print("=> physics features reproduce the theoretical model "
      "(<0.5% residuals = CSV 2-decimal rounding + the tool's seconds<->ms conversion)."
      if max(rel.values()) < 0.5 else "=> MISMATCH: check the feature formulas.")'''

# ---- Step 5.1b: feature analysis (correlation, PCA, t-SNE) -----------------

CODE_FEAT_PREP = '''\
from sklearn.preprocessing import StandardScaler

# Modeling space: log-transform positive features (binary ones left as-is).
const_feats = [c for c in FEATURES if feat[c].nunique() == 1]
nonconst    = [c for c in FEATURES if feat[c].nunique() >  1]
print("Constant (zero-variance) features in this dataset:", const_feats)
print(f"-> dead for prediction. {len(nonconst)} features actually vary.")

Xlog = feat[FEATURES].copy()
Xlog[LOG_FEATURES] = np.log(Xlog[LOG_FEATURES])
Xstd = StandardScaler().fit_transform(Xlog)   # standardized matrix reused by PCA / t-SNE

MODELS_LIST = sorted(feat["model"].unique())  # used by the colour-by-model plots and LOMO'''

CODE_CORR = '''\
# Correlation matrix (modeling space, non-constant features) -> multicollinearity.
corr = Xlog[nonconst].corr()
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(corr, cmap="coolwarm", center=0, vmin=-1, vmax=1, square=True,
            cbar_kws={"shrink": 0.7}, ax=ax)
ax.set_title("Feature correlation matrix (non-constant features, log space)")
plt.tight_layout()
plt.show()

# Quick list of strongly collinear pairs (|r| > 0.9)
import itertools
strong = [(a, b, round(corr.loc[a, b], 2))
          for a, b in itertools.combinations(nonconst, 2) if abs(corr.loc[a, b]) > 0.9]
print(f"{len(strong)} feature pairs with |r| > 0.9 (collinear) -> motivates Ridge / dim. reduction")'''

CODE_PCA = '''\
from sklearn.decomposition import PCA

pca = PCA().fit(Xstd)
evr = pca.explained_variance_ratio_
n95 = int(np.argmax(np.cumsum(evr) >= 0.95) + 1)
print(f"Variance explained by PC1..PC5: {(evr[:5] * 100).round(1)} %")
print(f"PCs needed for 95% variance: {n95} of {len(FEATURES)} features "
      f"-> effective dimensionality is much smaller.")

Z = pca.transform(Xstd)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].plot(np.arange(1, len(evr) + 1), np.cumsum(evr) * 100, "o-")
axes[0].axhline(95, ls="--", c="grey")
axes[0].set(xlabel="number of PCs", ylabel="cumulative variance %", title="PCA scree (cumulative)")
for m in MODELS_LIST:
    idx = (feat["model"] == m).values
    axes[1].scatter(Z[idx, 0], Z[idx, 1], label=m, s=35)
axes[1].set(xlabel="PC1", ylabel="PC2", title="PCA projection (color = model)")
axes[1].legend(fontsize=7, loc="best")
plt.tight_layout()
plt.show()'''

CODE_TSNE = '''\
from sklearn.manifold import TSNE

Z2 = TSNE(n_components=2, perplexity=15, init="pca", random_state=0).fit_transform(Xstd)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for m in MODELS_LIST:
    idx = (feat["model"] == m).values
    axes[0].scatter(Z2[idx, 0], Z2[idx, 1], label=m, s=35)
axes[0].set(title="t-SNE (color = model)")
axes[0].legend(fontsize=7, loc="best")
sc = axes[1].scatter(Z2[:, 0], Z2[:, 1], c=feat["n_gpu"], cmap="viridis", s=35)
axes[1].set(title="t-SNE (color = n_gpu / TP)")
plt.colorbar(sc, ax=axes[1], label="n_gpu")
plt.tight_layout()
plt.show()'''

# ---- Step 5.2: leave-one-model-out validation harness ---------------------

CODE_LOMO_SETUP = '''\
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# BIN_FEATURES / LOG_FEATURES were defined in 5.1 (binary pass-through vs log-transform).
def make_pipeline(estimator):
    """log-transform positive features, standardize, then fit the estimator.
    All steps are fit INSIDE each fold (no leakage)."""
    pre = ColumnTransformer([
        ("log", FunctionTransformer(np.log, validate=False), LOG_FEATURES),
        ("bin", "passthrough", BIN_FEATURES),
    ])
    return Pipeline([("pre", pre), ("scale", StandardScaler()), ("est", estimator)])'''

CODE_LOMO_FUNC = '''\
def lomo_eval(estimator_factory, target, framing):
    """Leave-one-model-out. framing 'A' = predict log(real); 'B' = predict
    log(real/theoretical) and reconstruct real = theoretical * exp(pred).
    Returns one row per held-out config with real / theoretical / predicted."""
    rows = []
    for held in MODELS_LIST:
        tr = feat[feat["model"] != held]
        te = feat[feat["model"] == held]
        if len(te) == 0:
            continue
        real_tr, theo_tr = tr[f"{target}_real"], tr[f"{target}_theo"]
        real_te, theo_te = te[f"{target}_real"].values, te[f"{target}_theo"].values
        y_tr = np.log(real_tr) if framing == "A" else np.log(real_tr / theo_tr)

        pipe = estimator_factory()
        pipe.fit(tr[FEATURES], y_tr)
        pred = pipe.predict(te[FEATURES])
        pred_real = np.exp(pred) if framing == "A" else theo_te * np.exp(pred)

        sub = te[["model", "gpu_arch", "interconnect", "n_gpu", "concurrent_users"]].copy()
        sub.insert(0, "framing", framing); sub.insert(0, "target", target)
        sub["y_real"], sub["y_theo"], sub["y_pred"] = real_te, theo_te, pred_real
        rows.append(sub)
    out = pd.concat(rows, ignore_index=True)
    out["ape_pred"] = (out["y_pred"] - out["y_real"]).abs() / out["y_real"] * 100
    out["ape_theo"] = (out["y_theo"] - out["y_real"]).abs() / out["y_real"] * 100
    return out'''

CODE_LOMO_RUN = '''\
# Run the harness with Ridge for the 3 targets x 2 framings, and SAVE per-fold predictions.
import os
ridge_factory = lambda: make_pipeline(Ridge(alpha=1.0))
preds = pd.concat([lomo_eval(ridge_factory, t, fr) for t in TARGETS for fr in ["A", "B"]],
                  ignore_index=True)

os.makedirs("results", exist_ok=True)
preds.to_csv("results/lomo_predictions.csv", index=False)
print(f"Saved {len(preds)} per-fold predictions -> results/lomo_predictions.csv\\n")

summary = (preds.groupby(["target", "framing"])
           .agg(MAPE_pred=("ape_pred", "mean"),  medAPE_pred=("ape_pred", "median"),
                MAPE_theo=("ape_theo", "mean"),  medAPE_theo=("ape_theo", "median"),
                n=("ape_pred", "size")).round(1))
print("LOMO error: ML (Ridge) vs theoretical baseline  [lower is better]")
summary'''

# ---- Step 5.3: model comparison + final results ---------------------------

CODE_ESTIMATORS = '''\
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

# Trees are invariant to the monotone log+scale transforms, so the same pipeline is fine.
ESTIMATORS = {
    "Ridge": lambda: make_pipeline(Ridge(alpha=1.0)),
    "GBM":   lambda: make_pipeline(GradientBoostingRegressor(
                 n_estimators=300, max_depth=2, learning_rate=0.05, random_state=0)),
    "RF":    lambda: make_pipeline(RandomForestRegressor(
                 n_estimators=400, min_samples_leaf=2, random_state=0, n_jobs=-1)),
}

all_preds = {}
for name, fac in ESTIMATORS.items():
    p = pd.concat([lomo_eval(fac, t, fr) for t in TARGETS for fr in ["A", "B"]], ignore_index=True)
    p["estimator"] = name
    all_preds[name] = p
comp = pd.concat(all_preds.values(), ignore_index=True)

baseline = (comp.drop_duplicates(["target", "model", "gpu_arch", "interconnect", "n_gpu", "concurrent_users"])
            .groupby("target").agg(MAPE_theo=("ape_theo", "mean"),
                                   medAPE_theo=("ape_theo", "median")).round(1))
table = (comp.groupby(["target", "estimator", "framing"])
         .agg(MAPE=("ape_pred", "mean"), medAPE=("ape_pred", "median")).round(1))
print("Theoretical baseline (per target):")
display(baseline)
print("ML models, LOMO  [MAPE / median-APE, %]:")
table'''

CODE_BEST = '''\
# Best (estimator, framing) per target by median-APE, then per-model breakdown.
best_keys = {}
for t in TARGETS:
    sub = comp[comp["target"] == t]
    g = sub.groupby(["estimator", "framing"])["ape_pred"].median()
    name, fr = g.idxmin()
    best_keys[t] = (name, fr)
    print(f"{t:14s} best: {name:5s}-{fr}  medAPE {g.min():5.1f}%   (baseline {sub['ape_theo'].median():.1f}%)")

best = pd.concat([comp[(comp.target == t) & (comp.estimator == nm) & (comp.framing == fr)]
                  for t, (nm, fr) in best_keys.items()], ignore_index=True)
print("\\nPer-model breakdown (best config per target):")
pm = best.groupby(["target", "model"]).agg(
        medAPE_ml=("ape_pred", "median"), MAPE_ml=("ape_pred", "mean"),
        medAPE_theo=("ape_theo", "median")).round(1)
pm'''

CODE_FINAL_PLOTS = '''\
# Predicted vs real (ML and theoretical), log-log, one panel per target.
fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
for ax, t in zip(axes, TARGETS):
    s = best[best.target == t]
    ax.scatter(s.y_real, s.y_theo, s=25, alpha=0.45, color="tab:red",  label="theoretical")
    ax.scatter(s.y_real, s.y_pred, s=25, alpha=0.75, color="tab:blue", label="ML")
    lo = max(min(s.y_real.min(), s.y_pred.min(), s.y_theo.min()), 1e-2)
    hi = max(s.y_real.max(), s.y_pred.max(), s.y_theo.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set(xscale="log", yscale="log", xlabel="real (measured)", ylabel="predicted",
           title=f"{t}  [{best_keys[t][0]}-{best_keys[t][1]}]")
    ax.legend(fontsize=8)
fig.suptitle("ML vs theoretical (LOMO, dashed = perfect agreement)", y=1.03)
plt.tight_layout()
plt.show()

# Median APE: ML vs baseline per target (log y, ML is tiny for power).
agg = best.groupby("target").agg(ML=("ape_pred", "median"), Theoretical=("ape_theo", "median"))
ax = agg.plot(kind="bar", figsize=(7, 4), color=["tab:blue", "tab:red"])
ax.set(ylabel="median APE (%)", title="Median error: ML vs theoretical (LOMO)", yscale="log")
for c in ax.containers:
    ax.bar_label(c, fmt="%.0f", fontsize=8)
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()'''

CODE_SAVE53 = '''\
# Save all artifacts for the thesis.
comp.to_csv("results/lomo_predictions_all_estimators.csv", index=False)
table.to_csv("results/lomo_model_comparison.csv")
pm.to_csv("results/lomo_per_model.csv")
print("Saved -> results/: lomo_predictions_all_estimators.csv, lomo_model_comparison.csv, lomo_per_model.csv")'''

# ---- Step 6: thesis synthesis ---------------------------------------------

CODE_SUMMARY = '''\
# Paper-ready summary: best ML config vs theoretical baseline, per target.
summary_rows = []
for t in TARGETS:
    nm, fr = best_keys[t]
    s = best[best.target == t]
    summary_rows.append({
        "target": t, "best_ML": f"{nm}-{fr}",
        "medAPE_ML": s["ape_pred"].median(), "medAPE_theo": s["ape_theo"].median(),
        "MAPE_ML": s["ape_pred"].mean(), "MAPE_theo": s["ape_theo"].mean(),
        "improvement_x": s["ape_theo"].median() / s["ape_pred"].median(),
    })
summary_table = pd.DataFrame(summary_rows).set_index("target").round(1)
summary_table.to_csv("results/summary_table.csv")
print("HEADLINE - ML vs theoretical (LOMO, median-APE), saved to results/summary_table.csv")
summary_table'''

CODE_EXPORT = '''\
# Export the key figures as high-resolution PNG for the thesis / white paper.
import os
os.makedirs("results/figures", exist_ok=True)
DPI = 150

# Fig 1: predicted vs real (ML and theoretical), log-log
fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
for ax, t in zip(axes, TARGETS):
    s = best[best.target == t]
    ax.scatter(s.y_real, s.y_theo, s=25, alpha=0.45, color="tab:red",  label="theoretical")
    ax.scatter(s.y_real, s.y_pred, s=25, alpha=0.75, color="tab:blue", label="ML")
    lo = max(min(s.y_real.min(), s.y_pred.min(), s.y_theo.min()), 1e-2)
    hi = max(s.y_real.max(), s.y_pred.max(), s.y_theo.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set(xscale="log", yscale="log", xlabel="real (measured)", ylabel="predicted",
           title=f"{t} [{best_keys[t][0]}-{best_keys[t][1]}]")
    ax.legend(fontsize=8)
fig.suptitle("ML vs theoretical (LOMO, dashed = perfect agreement)", y=1.03)
fig.tight_layout()
fig.savefig("results/figures/ml_vs_theoretical_scatter.png", dpi=DPI, bbox_inches="tight")
plt.close(fig)

# Fig 2: median APE bars, ML vs baseline
agg = best.groupby("target").agg(ML=("ape_pred", "median"), Theoretical=("ape_theo", "median"))
fig, ax = plt.subplots(figsize=(7, 4))
agg.plot(kind="bar", ax=ax, color=["tab:blue", "tab:red"])
ax.set(ylabel="median APE (%)", yscale="log", title="Median error: ML vs theoretical (LOMO)")
for c in ax.containers:
    ax.bar_label(c, fmt="%.0f", fontsize=8)
plt.xticks(rotation=0)
fig.tight_layout()
fig.savefig("results/figures/median_ape_bars.png", dpi=DPI, bbox_inches="tight")
plt.close(fig)

# Fig 3: per-model median APE (ML vs theoretical), shown inline too
pm2 = best.groupby(["target", "model"]).agg(
        ML=("ape_pred", "median"), Theoretical=("ape_theo", "median")).reset_index()
fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
for ax, t in zip(axes, TARGETS):
    d = pm2[pm2.target == t].set_index("model")[["ML", "Theoretical"]]
    d.plot(kind="bar", ax=ax, color=["tab:blue", "tab:red"], legend=(ax is axes[0]))
    ax.set(title=t, ylabel="median APE (%)", yscale="log")
    ax.tick_params(axis="x", rotation=90)
fig.suptitle("Per-model median APE: ML vs theoretical (LOMO)", y=1.02)
fig.tight_layout()
fig.savefig("results/figures/per_model_ape.png", dpi=DPI, bbox_inches="tight")
plt.show()

# Fig 4 (thesis F8/F9): predicted vs measured TTFT & ITL, all paired configs, log-log
scatter_vs(err, ["ttft_avg_ms", "itl_avg_ms"],
           "Calibrated roofline: predicted vs measured (all paired configs, log-log)",
           hue="interconnect", log=True,
           savepath="results/figures/pred_vs_real_ttft_itl.png")

print("Saved figures -> results/figures/:", sorted(os.listdir("results/figures")))'''

CODE_ENERGY = '''\
# Energy per token (thesis Part III): rate = power / AGGREGATE throughput.
# NOTE: the per-request formula power*latency/tokens would count the system
# power once per in-flight request -> overestimates by ~C at concurrency C.
en = bench_k[bench_k.output_token_throughput > 0].copy()
en["gpu"] = np.where(en.gpu_arch == "H100", "H100 PCIe", "H200 NVLink")
J_PER_KWH = 3.6e6
en["kwh_per_mtok"]     = en.power_w / en.output_token_throughput * 1e6 / J_PER_KWH
en["tdp_power_w"]      = en.gpu_arch.map(lambda a: GPU_SPECS_FE[a]["tdp_w"]) * en.n_gpu
en["kwh_per_mtok_tdp"] = en.tdp_power_w / en.output_token_throughput * 1e6 / J_PER_KWH

piv = en.pivot_table(values=["kwh_per_mtok", "kwh_per_mtok_tdp"],
                     index="concurrent_users", columns="gpu", aggfunc="median").round(3)
print("median kWh per 1M output tokens (measured power vs TDP nameplate):")
display(piv)
ratio = (en.kwh_per_mtok_tdp / en.kwh_per_mtok).median()
print(f"median nameplate/measured ratio: {ratio:.2f}x | "
      f"overall median measured: {en.kwh_per_mtok.median():.3f} kWh/Mtok")

(en.groupby(["gpu", "n_gpu", "concurrent_users"])[["kwh_per_mtok", "kwh_per_mtok_tdp"]]
   .median().round(4).to_csv("results/energy_per_mtok.csv"))

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
for ax, gpu in zip(axes, ["H100 PCIe", "H200 NVLink"]):
    d = en[en.gpu == gpu]
    for col, color, label in [("kwh_per_mtok", "tab:blue", "measured (pynvml)"),
                              ("kwh_per_mtok_tdp", "tab:red", "TDP x n_gpu estimate")]:
        g = d.groupby("concurrent_users")[col]
        ax.plot(g.median().index, g.median().values, "o-", color=color, label=label)
        ax.fill_between(g.median().index, g.quantile(.25), g.quantile(.75),
                        color=color, alpha=0.15)
    ax.set(xscale="log", yscale="log", title=gpu, xlabel="concurrent users")
    ax.set_xticks([1, 10, 50, 100]); ax.set_xticklabels([1, 10, 50, 100])
    ax.grid(alpha=0.3)
axes[0].set_ylabel("kWh per 1M output tokens")
axes[0].legend()
fig.suptitle("Energy per token: batching amortizes power; the TDP nameplate overestimates", y=1.02)
fig.tight_layout()
fig.savefig("results/figures/energy_per_mtok.png", dpi=DPI, bbox_inches="tight")
plt.show()'''

# ---------------------------------------------------------------------------
# TESTI markdown per lingua
# ---------------------------------------------------------------------------

MD = {
    "IT": {
        "title": """# Analisi benchmark LLM: reale vs teorico

**Obiettivo (tesi).** Questo notebook (1) offre una visione completa dei dati nei due CSV in `Benchmark_ML/Data/`, (2) confronta i risultati misurati con quelli del modello teorico di sizing (`llm-sizing-tool`), e (3) sviluppa una piccola pipeline di ML per provare a predire le metriche in modo piu accurato del modello teorico.

I due dataset:
- **`benchmark_results.csv`** - misure reali ottenute con vLLM (la *verita di campo*).
- **`sizing_results.csv`** - stime del modello teorico di sizing (la *predizione da battere*).

> Procediamo per step. Questo e lo **Step 1: caricamento e panoramica strutturale**.""",
        "load": "## 1. Caricamento dei dati\n\nCarichiamo i due file e verifichiamo che condividano lo stesso schema di colonne.",
        "filter": "### 1.0 Coerenza dei dati: solo H100 PCIe e H200 NVLink, etichette GPU normalizzate\n\nIl benchmark reale e stato eseguito **solo** su due GPU: **H100 PCIe** e **H200 NVLink (SXM)**. Il file teorico, invece, contiene anche H100 SXM e H200 PCIe. Per un confronto corretto, restringiamo da subito il file teorico alle stesse due GPU del benchmark.\n\nNormalizziamo inoltre `gpu_type` alla **identita della GPU** (`H100 PCIe` / `H200 NVLink`) in entrambi i file: il prefisso `Nx` delle etichette grezze indica il **nodo** (8 GPU installate) nel reale e il **TP** nel teorico, ma in entrambi i casi le GPU effettivamente usate sono `tensor_parallelism` (TP=1 -> 1 GPU, TP=2 -> 2 GPU, ...), che resta il fattore sperimentale esplicito. Cosi i due dataset sono rappresentati allo stesso modo: **2 GPU x 4 TP**, non 8 pseudo-GPU. Tutto il resto del notebook lavora su questo sottoinsieme.",
        "completeness": "### 1.1 Completezza delle colonne\n\nLo schema e identico, ma il file teorico **non popola** tutte le colonne (non ha min/max ne `request_throughput`). Vediamo quante celle non nulle ci sono per colonna in ciascun dataset.",
        "empty": "Da qui ricaviamo l'insieme delle **metriche confrontabili**: quelle popolate in *entrambi* i file. Saranno la base del confronto reale vs teorico (Step 3-4).",
        "coverage": "### 1.2 Copertura dello spazio sperimentale\n\nQuali livelli assume ogni fattore sperimentale nei due dataset? Dopo la normalizzazione di 1.0, `gpu_type` ha gli **stessi 2 livelli** nei due file; il disallineamento residuo sono le config mancanti da un lato o dall'altro (il teorico salta quelle che non entrano in VRAM, il benchmark non ha eseguito alcune celle della griglia) - le quantificheremo nello Step 3.",
        "counts": "Numero di run per modello: utile per capire quanto e bilanciato lo spazio campionato in ciascun file.",
        "describe": "### 1.3 Statistiche descrittive delle metriche\n\nDistribuzione (media, deviazione, quartili) delle metriche chiave nei due dataset, per avere un primo ordine di grandezza dei valori e degli outlier.",
        "s2_intro": "## 2. Visualizzazioni esplorative\n\nGuardiamo i due dataset **separatamente**: prima capiamo ciascuno per conto suo, poi (Step 3-4) li allineeremo e confronteremo. Gia mettendo a fianco le due serie di grafici emerge la differenza di *forma* tra reale e teorico - in particolare sul throughput.",
        "s2_dist": "### 2.1 Distribuzioni delle metriche (reale)\n\nBoxplot delle 5 metriche confrontabili nel benchmark reale, per modello e GPU: serve a vedere range, mediana e outlier prima di guardare gli andamenti.",
        "s2_trends": "### 2.2 Andamenti rispetto alla concurrency\n\nPer ogni metrica tracciamo l'andamento al crescere degli utenti concorrenti, un pannello per modello (**colore = TP**, **tratteggio/marker = GPU**; stesse serie 2 GPU x 4 TP nei due dataset, grazie alla normalizzazione di 1.0). Mostriamo **reale e teorico separatamente**: confrontare i due grafici e gia un primo risultato. `power_w` fa eccezione: essendo quasi piatta rispetto alla concurrency, la mostriamo come **grafico a barre** per GPU e TP (media sulla concurrency, barre d'errore = deviazione standard), dove emergono i gradini per TP, la sub-linearita del consumo reale per-GPU e il divario con la targa teorica (`tdp x n_gpu`).\n\n> **Da osservare:** sul throughput le due curve vanno in **direzioni opposte** - il reale *cresce* con la concurrency (e' il throughput **aggregato** del sistema, che beneficia del batching di vLLM), il teorico *cala* (e' `N_out / latenza` della **singola richiesta**, la cui latenza cresce con `C` per coda e contention). Non e' un errore del modello ma una differenza di definizione: per questo nello Step 4 il confronto quantitativo usera TTFT/ITL (approccio c) e tratteremo il throughput a parte.",
        "s3_intro": "## 3. Allineamento dei dataset\n\nPer confrontare reale e teorico servono **chiavi comuni**. I nomi modello sono gia identici (sistemati alla fonte) e `gpu_type` e gia stato normalizzato all'identita della GPU (1.0). Scomponiamo l'identita in **architettura** (H100/H200) e **interconnessione** (PCIe/NVLink) e allineiamo sul **numero effettivo di GPU = TP** (in entrambi i file le GPU usate sono `tensor_parallelism`).\n\nChiave di join: `(model, gpu_arch, interconnect, n_gpu, concurrent_users)`. Input/output sono fissi (512->128), quindi non entrano nella chiave.",
        "s3_agg": "### 3.1 Chiavi e aggregazione\n\nUna configurazione e identificata univocamente dalle chiavi. I run ripetuti lato reale (4 righe duplicate) vengono mediati -> 156 config distinte; lato teorico l'operazione e un no-op (una riga per config).",
        "s3_merge": "### 3.2 Dataset appaiato (inner join)\n\nUniamo le due tabelle config-per-config. Il risultato (`paired`) ha le chiavi piu ogni metrica nelle due varianti `_real` / `_theo`: e la base per il confronto dello Step 4 e per la pipeline ML.",
        "s3_coverage": "### 3.3 Copertura: cosa non si appaia (e perche)\n\nL'inner join scarta le config presenti in un solo file. Lato reale restano fuori 13 config che il tool considera **infeasible per VRAM** (`SKIP_INFEASIBLE`): pesi FP16 che non entrano a TP basso (es. Llama-70B a TP=1, gpt-oss-120b che in realta gira quantizzato MXFP4 mentre il tool lo modella FP16) o KV cache che non entra ad alta concurrency. Lato teorico restano fuori 12 config TP=8 che il benchmark non ha eseguito. Quantifichiamo gli scarti per non scoprirli a sorpresa nel confronto.",
        "s4_method": "## 4. Confronto reale vs teorico\n\n### 4.0 Come il modello teorico calcola le metriche (box metodologico)\n\nIl `llm-sizing-tool` (`backend/gpu_calc.py` + `config.py:PERF_MODEL`) stima le metriche con un modello **roofline calibrato**. Con `P_att` = parametri attivi (miliardi; per i MoE solo gli expert attivati), `BW` = banda memoria (GB/s), `FLOPS` = TFLOPS FP16 *dense*, `C` = utenti concorrenti, `N_in/N_out` = token, `n` = numero GPU (= TP):\n\n```\neta        = 1 se n=1, altrimenti 0.361 (NVLink) / 0.372 (PCIe)   # efficienza scaling TP\nFLOPS_eff  = n x FLOPS x MFU x eta          con MFU = 0.413\nBW_eff     = n x BW x MBU x eta             con MBU = 0.643\n\nTTFT = 2 P_att N_in / FLOPS_eff x (C+1)/2               # coda closed-loop media\nITL  = max( (pesi_att + KV_batch) / BW_eff ,            # banda: pesi + KV di tutto il batch\n            2 P_att C / FLOPS_eff )                     # compute del batch\nrequest_latency_avg_ms  = TTFT + N_out x ITL\noutput_token_throughput = N_out / (request_latency_avg_ms / 1000)   # per SINGOLA richiesta\n```\n\nLe **stesse formule valgono per entrambi gli interconnect**: cambiano solo `eta` e le spec GPU (H100 = profilo *PCIe*: 756 TFLOPS dense, 2.0 TB/s; H200 = SXM/NVLink: 989 TFLOPS, 4.8 TB/s). `KV_batch` e la KV cache di tutto il batch al contesto medio (`N_in + N_out/2`).\n\n> **Caveat di calibrazione (da dichiarare in tesi).** I 4 parametri liberi (`MFU`, `MBU`, `eta` x2) sono stati **fittati su questo stesso dataset** (`calibrate_perf_model.py`, minimizzazione del log-errore su TTFT+ITL, **soli modelli densi**: gpt-oss e escluso dal fit perche il roofline con i 5.1B attivi della model card lo sottostima ~5x, vedi il box MoE in 6.1). Gli errori dello Step 4 sono quindi residui *in-sample*: misurano quanto la *struttura* del modello spiega i dati una volta rimosso il bias, non la generalizzazione.\n\n**Due conseguenze semantiche da tenere a mente nel confronto:**\n1. **`output_token_throughput`** teorico e *per singola richiesta* ed e collineare con la latenza (`= N_out/latenza`); quello reale e *aggregato di sistema*. Crescono in direzioni opposte con la concurrency.\n2. **Concurrency**: il modello la tratta con il fattore di coda `(C+1)/2` sul TTFT (validato dal fit: esponente ottimo ~0.97) e con la lettura KV del batch nell'ITL; il fit esteso indica pero che la **contention reale del decode e ~4x piu forte** del termine KV puro - residuo strutturale noto.\n\nProcediamo in tre parti: **(4.1)** confronto *as-is* su tutte le metriche, **(4.2)** approccio *(b)* per il throughput, **(4.3)** confronto quantitativo *(c)* sulle metriche pulite TTFT/ITL.",
        "s4_asis": "### 4.1 Confronto \"as-is\" (didattico)\n\nConfrontiamo direttamente predetto-vs-misurato su **tutte** le metriche, *senza* correzioni. Convenzione: `errore = teorico - reale` (positivo => il teorico sovrastima). Sulle metriche per-token (TTFT/ITL) e sulla latenza l'errore e ora dello stesso ordine (~60-75% MAPE, residuo post-calibrazione); restano fuorvianti il **throughput** (definizioni diverse, vedi 4.0: bias ~-64% perche il per-richiesta teorico e confrontato con l'aggregato reale) e **power_w** (targa `tdp x n` vs consumo misurato, MAPE ~186%).",
        "s4_b": "### 4.2 Approccio (b): throughput per-richiesta\n\nIl throughput reale e aggregato; quello teorico e per-richiesta. Per renderli confrontabili dividiamo il reale per il numero di utenti (`throughput_reale / C`). Risultato: la MAPE **resta ~80%** (cambia il segno e la struttura dell'errore, non la grandezza complessiva) - la differenza tra le due quantita non e un semplice fattore di scala, perche il batching reale di vLLM serve C utenti quasi al costo di uno mentre il modello teorico ripartisce coda e contention sulla singola richiesta. La lezione didattica resta: throughput aggregato e per-richiesta sono mondi diversi, e il confronto pulito va fatto su TTFT/ITL.",
        "s4_c": "### 4.3 Confronto quantitativo (c): TTFT e ITL\n\nQuesto e il confronto *legittimo*: le metriche per-token (TTFT, ITL) hanno definizione coerente tra i due file. Misuriamo MAE, MAPE e bias, prima su tutte le config appaiate poi a `C = 1`, e scomponiamo l'errore per concurrency, interconnessione e modello. Aspettativa post-calibrazione: errore **senza trend esplosivi in C** (coda e contention sono modellate) e **simile sui due interconnect**; il residuo e varianza per-modello, non bias sistematico.",
        "s4_end": "---\n\n**Fine Step 4.** Quantificato il residuo del modello calibrato: TTFT/ITL ~50% medAPE distribuito tra i modelli (in-sample), niente regimi patologici; throughput e power restano grandezze non confrontabili as-is. Margine residuo per l'ML: varianza per-modello e effetti di batching non catturati dalla struttura roofline.",
        "s5_intro": "## 5. Pipeline di Machine Learning\n\nObiettivo: imparare dai dati reali un modello che **corregga o batta** il teorico nel predire **TTFT, ITL e power_w**. Impostazione (decisa insieme):\n- **Target** (scala log): `ttft_avg_ms`, `itl_avg_ms`, `power_w`.\n- **Due framing**: **(A)** predizione da zero; **(B)** *residual* sul teorico (`log(reale/teorico)`, ricostruisce `reale = teorico x exp(pred)`).\n- **Baseline da battere**: il valore del tool **roofline calibrato** (per `power_w`: `tdp x n_gpu`).\n- **Validazione**: leave-one-model-out (6 fold) - test onesto sulla generalizzazione a modelli mai visti.\n- **Feature**: fisiche (modello + GPU) + workload + termini del roofline calibrato. **Niente upsampling sintetico** (rischio di punti non-fisici e leakage).\n\n> **Asimmetria dichiarata a favore della baseline**: i 4 parametri del roofline sono fittati *in-sample* su tutto il dataset, mentre l'ML e valutato *out-of-fold* (LOMO, il modello held-out non e mai visto in training ne in calibrazione). Se l'ML batte comunque la baseline, il risultato e conservativo.\n\nQui (5.1) costruiamo le feature; poi la harness LOMO (5.2) e i modelli A/B con i risultati (5.3).",
        "s5_specs": "### 5.1 Spec table e feature engineering\n\nCongeliamo i parametri fisici dei 6 modelli (risolti dal tool stesso, quindi coerenti con la baseline teorica) e le specs delle 2 GPU **reali del benchmark** (H100 = profilo PCIe, H200 = SXM; TFLOPS dense). Da questi costruiamo le feature, inclusi i **termini del roofline calibrato** (`ttft_core`, `itl_mem_core`/`itl_compute_core`, `queue_factor`, `tp_eff`, risorse effettive aggregate, `kv_read_gb`, `vram_pressure`...). Le feature fisiche **sostituiscono l'encoding categorico** del modello: e cio che rende possibile il leave-one-model-out (un modello mai visto e solo un nuovo punto numerico). Il sanity check finale ricostruisce TTFT/ITL/power teorici dalle feature e li confronta col tool: se combaciano, la fisica e replicata correttamente.",
        "s51b_intro": "### 5.1b Analisi delle feature (correlazione, PCA, t-SNE)\n\nPrima di modellare, guardiamo la struttura delle 29 feature.\n- **Matrice di correlazione**: ci aspettiamo forte collinearita (params, active, weights, ttft_core... legate alla taglia del modello; e le spec GPU, che variano tra H100 PCIe e H200 ma restano perfettamente legate all'identita della GPU). Giustifica l'uso di Ridge e mostra che la dimensionalita effettiva e ben minore di 29.\n- **PCA**: quante componenti bastano a spiegare la varianza, e la proiezione 2D mostra che i dati **si raggruppano per modello** -> ecco perche il leave-one-model-out e un'estrapolazione (si tiene fuori un intero cluster).\n- **t-SNE**: complemento esplorativo per visualizzare i cluster (meno rigoroso della PCA su N=143).",
        "s51_end": "---\n\n**Fine Step 5.1.** Abbiamo le feature fisiche, la conferma che riproducono il modello teorico, e ne abbiamo capito la struttura (collinearita, dimensionalita effettiva, clustering per modello).",
        "s52_intro": "### 5.2 Harness di validazione (leave-one-model-out)\n\nValutiamo con **LOMO**: a turno teniamo fuori un intero modello, alleniamo sugli altri 5 e prediciamo sul modello mai visto. E' il test piu onesto di generalizzazione (e l'unico sensato qui, dato che le feature fisiche sostituiscono l'encoding del modello).\n\nDettagli della harness:\n- **Pipeline per fold** (niente leakage): log-transform delle feature positive -> standardizzazione -> stimatore, tutto *fittato dentro il fold*.\n- **Due framing**: **A** predice `log(reale)`; **B** predice `log(reale/teorico)` e ricostruisce `reale = teorico x exp(pred)`.\n- **Baseline**: il valore teorico del tool, valutato sugli stessi fold.\n- **Metriche**: MAPE e **mediana-APE** (robusta alle code lunghe viste nello Step 4).\n- Stimatore di questo blocco: **Ridge** (lineare regolarizzato, estrapola bene). I modelli ad albero e il confronto finale sono nello Step 5.3.\n- Le **predizioni per-fold** vengono salvate in `results/lomo_predictions.csv` per la tesi.",
        "s52_end": "---\n\n**Fine Step 5.2.** Harness LOMO pronta, predizioni per-fold salvate, primo confronto Ridge vs baseline.",
        "s53_intro": "### 5.3 Confronto modelli e risultati finali\n\nConfrontiamo **Ridge** (lineare regolarizzato), **Gradient Boosting** e **Random Forest** sui 3 target x 2 framing (A/B), sempre in leave-one-model-out. Poi scegliamo la configurazione migliore per target (per **median-APE**, robusta alle code), guardiamo il **breakdown per modello** e produciamo i grafici finali.\n\n> *Caveat dichiarato:* gli alberi (GBM/RF) **non estrapolano** oltre il range visto in training; nel LOMO, sul modello held-out piu grande o piu piccolo possono andare peggio di Ridge. Niente nested-CV (dataset piccolo): iperparametri ragionevoli fissi.",
        "s53_findings": "**Lettura dei risultati (baseline = roofline calibrato).** `power_w` resta il caso piu netto: l'ML scende sotto l'1% di errore mediano contro ~168% della targa `tdp x n`, imparando la curva di utilizzo per-GPU. Su `TTFT` il best e RF-B (medAPE ~40% vs ~55% della baseline) e su `ITL` GBM-B (~29% vs ~51%): guadagni di **1.4-1.8x** su una baseline fisica gia solida. L'ML pero **non vince su ogni fold**: sul TTFT di `DeepSeek-R1-Distill` la baseline calibrata fa meglio (75 vs 57 medAPE) - le feature fisiche non distinguono i *gemelli architetturali* (specs identiche a Qwen2.5-32B, comportamento reale diverso) - onesto dichiararlo, insieme al fatto che la baseline e fittata in-sample (vedi 5. intro) quindi il confronto e *conservativo* per l'ML. Il framing **B (residual)** vince quasi ovunque sui target di latenza, segno che la baseline e ormai informativa; **A** vince su power, dove il teorico e fuori scala. Tutti gli artefatti sono in `results/` per tabelle e figure della tesi.",
        "s5_end": "---\n\n**Fine Step 5.** Pipeline ML completa: feature fisiche, validazione LOMO onesta, confronto modelli e baseline, artefatti salvati.",
        "s6_intro": "## 6. Sintesi per la tesi\n\nImpacchettiamo i risultati: una **tabella riassuntiva** 'da paper' (ML vs teorico per target), l'**esportazione delle figure chiave** in `results/figures/` (PNG ad alta risoluzione, pronte da inserire in tesi/white paper), e una nota su **limiti e lavoro futuro**.",
        "s6_energy": "### 6.0 Energia per token (kWh/Mtok)\n\nConvertiamo le misure di potenza in **energia per token**: `E = power_w / throughput_aggregato` (J/token), riportata come **kWh per 1M token** di output (numericamente identica a Wh per 1k token). Due note metodologiche: (1) la formula per-richiesta `power x latenza / token` conterebbe la potenza di sistema una volta per ogni richiesta in volo -> sovrastima di ~C a concurrency C; si usa il throughput aggregato. (2) per la stima a targa usiamo `TDP x n_gpu` diviso per lo **stesso** throughput misurato: cosi il confronto isola l'errore sulla potenza, senza mescolarlo con quello sul throughput. Messaggi attesi: l'energia per token **cala di ~20x con la concurrency** (il batching ammortizza una potenza quasi costante) e la stima a targa **sovrastima di ~2.6x** (mediana).",
        "s6_limits": "### 6.1 Limiti e lavoro futuro\n\n**Limiti (da dichiarare in tesi):**\n- **N piccolo**: 143 configurazioni appaiate, 6 modelli. Il leave-one-model-out e severo e le stime hanno varianza non trascurabile -> risultati *indicativi*, non definitivi.\n- **Baseline calibrata in-sample**: i 4 parametri del roofline sono fittati su questo stesso dataset, mentre l'ML e out-of-fold -> il confronto e conservativo per l'ML, ma gli errori \"teorici\" dello Step 4 non misurano la generalizzazione del tool a hardware/workload nuovi.\n- **Input/output fissi** (512->128): nessuna generalizzazione su lunghezze di sequenza diverse; il carico varia solo tramite la concurrency.\n- **Collinearita GPU/interconnessione**: nei dati H100 e sempre PCIe e H200 sempre NVLink, quindi l'effetto dell'architettura e dell'interconnessione non e separabile (le spec GPU variano, ma in blocco con l'identita della GPU).\n- **Un solo MoE** (gpt-oss): la generalizzazione ai MoE non e davvero testabile. Con le spec della model card (116.8B totali / 5.1B attivi, arXiv:2508.10925) il roofline lo sottostima ~5x -> e **escluso dal fit di calibrazione** (fuori dominio): il costo effettivo di serving di un MoE sparso e molto sopra i parametri attivi nominali (kernel grouped-GEMM, traffico pesi per-batch; il vecchio valore euristico di 27.09B attivi era di fatto un buon 'costo effettivo'). Resta il fold piu debole per l'ML sui target di latenza. Inoltre gpt-oss reale gira quantizzato (MXFP4) mentre il tool lo modella FP16.\n- **Baseline teorica** = il modello roofline calibrato del tool; le conclusioni sono legate a quella implementazione. `power_w` reale e una misura puntuale (sensibile a temperatura/throttling).\n\n**Lavoro futuro:**\n- Piu benchmark reali (altre GPU, piu MoE, interconnessioni miste) per ampliare N e rompere le collinearita.\n- Variare le lunghezze input/output per testare la generalizzazione.\n- Termine di contention del decode nel modello fisico (il fit esteso indica che la lettura KV del batch pesa ~4x il termine teorico puro).\n- Calibrazione per-fold (rifittare i 4 parametri dentro ogni fold LOMO) per un confronto fisico-vs-ML del tutto simmetrico.\n- Stima dell'incertezza sulle predizioni (intervalli di confidenza).",
        "s7_tp_intro": "## 7 · Errore vs numero di GPU (TP) a concurrency fissa\n\nDomanda: *dove* si concentra l'errore del modello fisico al crescere del parallelismo, e il correttivo ML lo assorbe? A concurrency fissa (C=1, il regime più pulito: niente coda) confrontiamo il medAPE della predizione teorica e di quella ML (miglior stimatore per target, predizioni out-of-fold LOMO) per livello di tensor parallelism. Nota chiave: a TP=1 non c'è comunicazione inter-GPU (η=1), quindi l'errore residuo lì misura la pura qualità di MFU/MBU; da TP≥2 entra in gioco il termine η.",
        "s7_tp_end": "### Lettura\n\n- **TTFT**: a TP=1 il roofline è quasi esatto (medAPE ~8%) e l'ML *non* lo migliora (~16%): dove la fisica basta, il correttivo non ha nulla da correggere. Da TP≥2 l'errore fisico sale (~40–60%) e il vantaggio ML cresce con il parallelismo, fino a ~3× a TP=8 (22% vs 62%).\n- **ITL**: il roofline ha il suo regime peggiore a TP=2 (medAPE ~160%); l'ML lo tiene ≤34% a ogni livello di TP.\n- **Potenza**: l'errore della targa TDP×n **cresce monotonicamente con il TP** (11% → 331% da 1 a 8 GPU): è la firma della sub-linearità della potenza, che la targa ignora per costruzione. L'ML resta ≤2% ovunque.\n- Robustezza: alle altre concurrency il pattern si conserva — il vantaggio ML si concentra ai TP alti.\n\n**Messaggio**: l'errore del modello fisico non è distribuito uniformemente — si accumula dove entrano la comunicazione multi-GPU e la sub-linearità della potenza. Il correttivo ML impara esattamente questi regimi ed è lì che serve; a TP=1 la fisica calibrata è già sufficiente.",
        "next": "---\n\n**Notebook completo.** Panoramica dati, confronto reale-vs-teorico, e pipeline ML (feature fisiche, LOMO, modelli A/B) con artefatti e figure pronti per la stesura della tesi in `results/`.",
    },
    "EN": {
        "title": """# LLM benchmark analysis: real vs theoretical

**Goal (thesis).** This notebook (1) gives a complete view of the data in the two CSVs in `Benchmark_ML/Data/`, (2) compares the measured results against the theoretical sizing model (`llm-sizing-tool`), and (3) builds a small ML pipeline to try to predict the metrics more accurately than the theoretical model.

The two datasets:
- **`benchmark_results.csv`** - real measurements obtained with vLLM (the *ground truth*).
- **`sizing_results.csv`** - estimates from the theoretical sizing model (the *prediction to beat*).

> We proceed step by step. This is **Step 1: loading and structural overview**.""",
        "load": "## 1. Loading the data\n\nWe load the two files and check they share the same column schema.",
        "filter": "### 1.0 Data consistency: only H100 PCIe and H200 NVLink, normalized GPU labels\n\nThe real benchmark was run on **only** two GPUs: **H100 PCIe** and **H200 NVLink (SXM)**. The theoretical file, however, also contains H100 SXM and H200 PCIe. For a fair comparison, we restrict the theoretical file to the same two GPUs from the start.\n\nWe also normalize `gpu_type` to the **GPU identity** (`H100 PCIe` / `H200 NVLink`) in both files: the `Nx` prefix of the raw labels is the **node** size (8 GPUs installed) in the real file and the **TP** in the theoretical one, but either way the GPUs actually used are `tensor_parallelism` (TP=1 -> 1 GPU, TP=2 -> 2 GPUs, ...), which remains the explicit design factor. This way both datasets are represented the same way: **2 GPUs x 4 TP**, not 8 pseudo-GPUs. The rest of the notebook works on this subset.",
        "completeness": "### 1.1 Column completeness\n\nThe schema is identical, but the theoretical file does **not** populate every column (no min/max, no `request_throughput`). Let's see how many non-null cells each column has in each dataset.",
        "empty": "From this we derive the set of **comparable metrics**: those populated in *both* files. They are the basis of the real-vs-theoretical comparison (Steps 3-4).",
        "coverage": "### 1.2 Coverage of the experimental space\n\nWhich levels does each design factor take in the two datasets? After the 1.0 normalization, `gpu_type` has the **same 2 levels** in both files; the remaining mismatch is the configs missing on either side (the theoretical file skips VRAM-infeasible configs, the benchmark did not run some cells of the grid) - quantified in Step 3.",
        "counts": "Number of runs per model: useful to gauge how balanced the sampled space is in each file.",
        "describe": "### 1.3 Descriptive statistics of the metrics\n\nDistribution (mean, std, quartiles) of the key metrics in the two datasets, to get a first sense of magnitudes and outliers.",
        "s2_intro": "## 2. Exploratory visualizations\n\nWe look at the two datasets **separately**: first understand each on its own, then (Steps 3-4) align and compare them. Just putting the two sets of plots side by side already reveals the difference in *shape* between real and theoretical - especially on throughput.",
        "s2_dist": "### 2.1 Metric distributions (real)\n\nBoxplots of the 5 comparable metrics in the real benchmark, by model and GPU: useful to see ranges, medians and outliers before looking at trends.",
        "s2_trends": "### 2.2 Trends with respect to concurrency\n\nFor each metric we plot the trend as concurrent users grow, one panel per model (**color = TP**, **dash/marker = GPU**; the same 2 GPU x 4 TP series in both datasets, thanks to the 1.0 normalization). We show **real and theoretical separately**: comparing the two plots is already a first result. `power_w` is the exception: being ~flat vs concurrency, we show it as a **bar chart** by GPU and TP (mean over concurrency, error bars = standard deviation), which surfaces the TP steps, the sub-linearity of the real per-GPU draw and the gap with the theoretical nameplate (`tdp x n_gpu`).\n\n> **Watch:** on throughput the two curves go in **opposite directions** - the real one *grows* with concurrency (it is the **aggregate** system throughput, which benefits from vLLM batching), the theoretical one *falls* (it is `N_out / latency` of a **single request**, whose latency grows with `C` through queueing and contention). This is not a model error but a definition mismatch: which is why in Step 4 the quantitative comparison will use TTFT/ITL (approach c) and treat throughput separately.",
        "s3_intro": "## 3. Aligning the datasets\n\nTo compare real and theoretical we need **common keys**. Model names already match (fixed at the source) and `gpu_type` was already normalized to the GPU identity (1.0). We decompose the identity into **architecture** (H100/H200) and **interconnect** (PCIe/NVLink) and align on the **effective number of GPUs = TP** (in both files the GPUs used are `tensor_parallelism`).\n\nJoin key: `(model, gpu_arch, interconnect, n_gpu, concurrent_users)`. Input/output are fixed (512->128), so they don't enter the key.",
        "s3_agg": "### 3.1 Keys and aggregation\n\nA configuration is uniquely identified by the keys. Repeated runs on the real side (4 duplicate rows) are averaged -> 156 distinct configs; on the theoretical side this is a no-op (one row per config).",
        "s3_merge": "### 3.2 Paired dataset (inner join)\n\nWe join the two tables config-by-config. The result (`paired`) has the keys plus each metric in its `_real` / `_theo` variants: it is the basis for the Step 4 comparison and for the ML pipeline.",
        "s3_coverage": "### 3.3 Coverage: what does not pair (and why)\n\nThe inner join drops configs present in only one file. On the real side, 13 configs are left out because the tool deems them **VRAM-infeasible** (`SKIP_INFEASIBLE`): FP16 weights that don't fit at low TP (e.g. Llama-70B at TP=1, gpt-oss-120b which actually runs MXFP4-quantized while the tool models it as FP16) or KV cache that doesn't fit at high concurrency. On the theoretical side, 12 TP=8 configs that the benchmark did not run. We quantify the drops so they don't surprise us during the comparison.",
        "s4_method": "## 4. Real vs theoretical comparison\n\n### 4.0 How the theoretical model computes the metrics (methodological box)\n\nThe `llm-sizing-tool` (`backend/gpu_calc.py` + `config.py:PERF_MODEL`) estimates the metrics with a **calibrated roofline** model. With `P_act` = active params (billions; for MoE only the activated experts), `BW` = memory bandwidth (GB/s), `FLOPS` = *dense* FP16 TFLOPS, `C` = concurrent users, `N_in/N_out` = tokens, `n` = number of GPUs (= TP):\n\n```\neta        = 1 if n=1, else 0.361 (NVLink) / 0.372 (PCIe)   # TP scaling efficiency\nFLOPS_eff  = n x FLOPS x MFU x eta          with MFU = 0.413\nBW_eff     = n x BW x MBU x eta             with MBU = 0.643\n\nTTFT = 2 P_act N_in / FLOPS_eff x (C+1)/2               # avg closed-loop queueing\nITL  = max( (act_weights + KV_batch) / BW_eff ,         # memory: weights + whole-batch KV\n            2 P_act C / FLOPS_eff )                     # batched compute\nrequest_latency_avg_ms  = TTFT + N_out x ITL\noutput_token_throughput = N_out / (request_latency_avg_ms / 1000)   # per SINGLE request\n```\n\nThe **same formulas hold for both interconnects**: only `eta` and the GPU specs differ (H100 = *PCIe* profile: 756 dense TFLOPS, 2.0 TB/s; H200 = SXM/NVLink: 989 TFLOPS, 4.8 TB/s). `KV_batch` is the whole batch's KV cache at the average generation context (`N_in + N_out/2`).\n\n> **Calibration caveat (to state in the thesis).** The 4 free parameters (`MFU`, `MBU`, `eta` x2) were **fitted on this very dataset** (`calibrate_perf_model.py`, mean abs log-error on TTFT+ITL, **dense models only**: gpt-oss is excluded from the fit because the roofline with the model card's 5.1B active params underestimates it ~5x, see the MoE box in 6.1). The Step 4 errors are therefore *in-sample* residuals: they measure how much of the data the model's *structure* explains once the bias is removed, not generalization.\n\n**Two semantic consequences to keep in mind:**\n1. The theoretical **`output_token_throughput`** is *per single request* and collinear with latency (`= N_out/latency`); the real one is *aggregate system* throughput. They move in opposite directions with concurrency.\n2. **Concurrency**: the model handles it via the `(C+1)/2` queue factor on TTFT (validated by the fit: optimal exponent ~0.97) and the batch KV read in ITL; the extended fit however indicates that **real decode contention is ~4x stronger** than the pure KV term - a known structural residual.\n\nWe proceed in three parts: **(4.1)** the *as-is* comparison on all metrics, **(4.2)** approach *(b)* for throughput, **(4.3)** the quantitative comparison *(c)* on the clean TTFT/ITL metrics.",
        "s4_asis": "### 4.1 \"As-is\" comparison (didactic)\n\nWe directly compare predicted-vs-measured on **all** metrics, *without* corrections. Convention: `error = theoretical - real` (positive => theory overestimates). On the per-token metrics (TTFT/ITL) and latency the error is now of the same order (~60-75% MAPE, post-calibration residual); **throughput** remains misleading (different definitions, see 4.0: bias ~-64% because the theoretical per-request value is compared to the real aggregate) and so does **power_w** (nameplate `tdp x n` vs measured draw, MAPE ~186%).",
        "s4_b": "### 4.2 Approach (b): per-request throughput\n\nThe real throughput is aggregate; the theoretical one is per-request. To make them comparable we divide the real one by the number of users (`real_throughput / C`). Result: the MAPE **stays at ~80%** (the sign and structure of the error change, not its overall size) - the difference between the two quantities is not a simple scaling factor, because real vLLM batching serves C users at nearly the cost of one, while the theoretical model charges queueing and contention to the single request. The didactic lesson stands: aggregate and per-request throughput are different worlds, and the clean comparison belongs to TTFT/ITL.",
        "s4_c": "### 4.3 Quantitative comparison (c): TTFT and ITL\n\nThis is the *legitimate* comparison: the per-token metrics (TTFT, ITL) have a consistent definition across the two files. We measure MAE, MAPE and bias, first on all paired configs then at `C = 1`, and we break the error down by concurrency, interconnect and model. Post-calibration expectation: error **without explosive trends in C** (queueing and contention are modelled) and **similar across the two interconnects**; the residual is per-model variance, not systematic bias.",
        "s4_end": "---\n\n**End of Step 4.** The calibrated model's residual is quantified: TTFT/ITL ~50% medAPE spread across models (in-sample), no pathological regimes; throughput and power remain non-comparable as-is. Remaining headroom for ML: per-model variance and batching effects not captured by the roofline structure.",
        "s5_intro": "## 5. Machine Learning pipeline\n\nGoal: learn from the real data a model that **corrects or beats** the theoretical one at predicting **TTFT, ITL and power_w**. Setup (decided together):\n- **Targets** (log scale): `ttft_avg_ms`, `itl_avg_ms`, `power_w`.\n- **Two framings**: **(A)** from-scratch prediction; **(B)** *residual* over the theoretical (`log(real/theoretical)`, reconstructs `real = theoretical x exp(pred)`).\n- **Baseline to beat**: the **calibrated roofline** tool value (for `power_w`: `tdp x n_gpu`).\n- **Validation**: leave-one-model-out (6 folds) - an honest test of generalization to unseen models.\n- **Features**: physical (model + GPU) + workload + calibrated-roofline terms. **No synthetic upsampling** (risk of non-physical points and leakage).\n\n> **Declared asymmetry in the baseline's favour**: the roofline's 4 parameters are fitted *in-sample* on the whole dataset, while ML is evaluated *out-of-fold* (LOMO: the held-out model is never seen in training nor in calibration). If ML still beats the baseline, the result is conservative.\n\nHere (5.1) we build the features; then the LOMO harness (5.2) and the A/B models with results (5.3).",
        "s5_specs": "### 5.1 Spec table and feature engineering\n\nWe freeze the physical parameters of the 6 models (resolved by the tool itself, hence consistent with the theoretical baseline) and the specs of the 2 **actual benchmark GPUs** (H100 = PCIe profile, H200 = SXM; dense TFLOPS). From these we build the features, including the **calibrated-roofline terms** (`ttft_core`, `itl_mem_core`/`itl_compute_core`, `queue_factor`, `tp_eff`, effective aggregate resources, `kv_read_gb`, `vram_pressure`...). The physical features **replace the categorical encoding** of the model: this is what makes leave-one-model-out possible (an unseen model is just a new numeric point). The final sanity check rebuilds the theoretical TTFT/ITL/power from the features and compares them to the tool: if they match, the physics is replicated correctly.",
        "s51b_intro": "### 5.1b Feature analysis (correlation, PCA, t-SNE)\n\nBefore modeling, we inspect the structure of the 29 features.\n- **Correlation matrix**: we expect strong collinearity (params, active, weights, ttft_core... all tied to model size; and the GPU specs, which differ between H100 PCIe and H200 but remain perfectly tied to the GPU identity). It justifies using Ridge and shows the effective dimensionality is well below 29.\n- **PCA**: how many components are needed to explain the variance, and the 2D projection shows the data **clusters by model** -> which is why leave-one-model-out is an extrapolation (you hold out a whole cluster).\n- **t-SNE**: an exploratory complement to visualize the clusters (less rigorous than PCA at N=143).",
        "s51_end": "---\n\n**End of Step 5.1.** We have the physical features, confirmation that they reproduce the theoretical model, and an understanding of their structure (collinearity, effective dimensionality, model-wise clustering).",
        "s52_intro": "### 5.2 Validation harness (leave-one-model-out)\n\nWe evaluate with **LOMO**: in turn we hold out a whole model, train on the other 5 and predict on the unseen model. It is the most honest test of generalization (and the only sensible one here, since the physical features replace the model's categorical encoding).\n\nHarness details:\n- **Per-fold pipeline** (no leakage): log-transform the positive features -> standardize -> estimator, all *fit inside the fold*.\n- **Two framings**: **A** predicts `log(real)`; **B** predicts `log(real/theoretical)` and reconstructs `real = theoretical x exp(pred)`.\n- **Baseline**: the tool's theoretical value, evaluated on the same folds.\n- **Metrics**: MAPE and **median-APE** (robust to the long tails seen in Step 4).\n- Estimator for this block: **Ridge** (regularized linear, extrapolates well). Tree models and the final comparison are in Step 5.3.\n- The **per-fold predictions** are saved to `results/lomo_predictions.csv` for the thesis.",
        "s52_end": "---\n\n**End of Step 5.2.** LOMO harness ready, per-fold predictions saved, first Ridge-vs-baseline comparison.",
        "s53_intro": "### 5.3 Model comparison and final results\n\nWe compare **Ridge** (regularized linear), **Gradient Boosting** and **Random Forest** across the 3 targets x 2 framings (A/B), always leave-one-model-out. Then we pick the best config per target (by **median-APE**, robust to tails), look at the **per-model breakdown** and produce the final plots.\n\n> *Stated caveat:* trees (GBM/RF) **do not extrapolate** beyond the training range; under LOMO, on the largest or smallest held-out model they may do worse than Ridge. No nested-CV (small dataset): reasonable fixed hyperparameters.",
        "s53_findings": "**Reading the results (baseline = calibrated roofline).** `power_w` remains the clearest case: ML drops below 1% median error against ~168% of the `tdp x n` nameplate, learning the per-GPU utilization curve. On `TTFT` the best is RF-B (medAPE ~40% vs ~55% baseline) and on `ITL` GBM-B (~29% vs ~51%): gains of **1.4-1.8x** over an already solid physical baseline. ML however does **not win on every fold**: on `DeepSeek-R1-Distill` TTFT the calibrated baseline does better (75 vs 57 medAPE) - the physical features cannot tell *architectural twins* apart (identical specs to Qwen2.5-32B, different real behavior) - honest to state, together with the fact that the baseline is fitted in-sample (see the Step 5 intro), making the comparison *conservative* for ML. The **B (residual)** framing wins almost everywhere on the latency targets, a sign the baseline is now informative; **A** wins on power, where the theoretical value is off-scale. All artifacts are in `results/` for the thesis tables and figures.",
        "s5_end": "---\n\n**End of Step 5.** Full ML pipeline: physical features, honest LOMO validation, model and baseline comparison, saved artifacts.",
        "s6_intro": "## 6. Thesis synthesis\n\nWe package the results: a **paper-ready summary table** (ML vs theoretical per target), the **export of the key figures** to `results/figures/` (high-resolution PNG, ready to drop into the thesis/white paper), and a note on **limitations and future work**.",
        "s6_energy": "### 6.0 Energy per token (kWh/Mtok)\n\nWe convert the power measurements into **energy per token**: `E = power_w / aggregate_throughput` (J/token), reported as **kWh per 1M output tokens** (numerically identical to Wh per 1k tokens). Two methodological notes: (1) the per-request formula `power x latency / tokens` would count the system power once per in-flight request -> overestimates by ~C at concurrency C; we use the aggregate throughput. (2) for the nameplate estimate we divide `TDP x n_gpu` by the **same** measured throughput, so the comparison isolates the power error without mixing in the throughput error. Expected messages: energy per token **drops ~20x with concurrency** (batching amortizes a nearly constant power draw) and the nameplate estimate **overestimates by ~2.6x** (median).",
        "s6_limits": "### 6.1 Limitations and future work\n\n**Limitations (to state in the thesis):**\n- **Small N**: 143 paired configurations, 6 models. Leave-one-model-out is harsh and the estimates carry non-negligible variance -> results are *indicative*, not definitive.\n- **In-sample-calibrated baseline**: the roofline's 4 parameters are fitted on this very dataset, while ML is out-of-fold -> the comparison is conservative for ML, but the Step 4 \"theoretical\" errors do not measure the tool's generalization to new hardware/workloads.\n- **Fixed input/output** (512->128): no generalization across sequence lengths; the load varies only through concurrency.\n- **GPU/interconnect collinearity**: in the data H100 is always PCIe and H200 always NVLink, so the architecture and interconnect effects cannot be separated (the GPU specs vary, but in lock-step with the GPU identity).\n- **A single MoE** (gpt-oss): MoE generalization is not really testable. With the model-card specs (116.8B total / 5.1B active, arXiv:2508.10925) the roofline underestimates it ~5x -> it is **excluded from the calibration fit** (out of domain): the effective serving cost of a sparse MoE sits far above its nominal active params (grouped-GEMM kernels, per-batch expert weight traffic; the old heuristic value of 27.09B active was in effect a good 'effective cost'). It remains the weakest fold for ML on the latency targets. Moreover, the real gpt-oss runs quantized (MXFP4) while the tool models it as FP16.\n- **Theoretical baseline** = the tool's calibrated roofline model; conclusions are tied to that implementation. Real `power_w` is a point measurement (sensitive to temperature/throttling).\n\n**Future work:**\n- More real benchmarks (other GPUs, more MoE, mixed interconnects) to enlarge N and break the collinearities.\n- Vary input/output lengths to test generalization.\n- A decode-contention term in the physical model (the extended fit indicates the batch KV read weighs ~4x the pure theoretical term).\n- Per-fold calibration (refit the 4 parameters inside each LOMO fold) for a fully symmetric physics-vs-ML comparison.\n- Uncertainty estimation on the predictions (confidence intervals).",
        "s7_tp_intro": "## 7 · Error vs GPU count (TP) at fixed concurrency\n\nQuestion: *where* does the physical model's error concentrate as parallelism grows, and does the ML corrector absorb it? At fixed concurrency (C=1, the cleanest regime: no queueing) we compare the medAPE of the theoretical and the ML prediction (best estimator per target, out-of-fold LOMO predictions) per tensor-parallelism level. Key note: at TP=1 there is no inter-GPU communication (η=1), so the residual error there measures pure MFU/MBU quality; from TP≥2 the η term kicks in.",
        "s7_tp_end": "### Reading\n\n- **TTFT**: at TP=1 the roofline is nearly exact (medAPE ~8%) and ML does *not* improve it (~16%): where the physics suffices, the corrector has nothing to correct. From TP≥2 the physical error rises (~40–60%) and the ML advantage grows with parallelism, up to ~3× at TP=8 (22% vs 62%).\n- **ITL**: the roofline has its worst regime at TP=2 (medAPE ~160%); ML keeps it ≤34% at every TP level.\n- **Power**: the TDP×n nameplate error **grows monotonically with TP** (11% → 331% from 1 to 8 GPUs): the signature of power sub-linearity, which the nameplate ignores by construction. ML stays ≤2% everywhere.\n- Robustness: at the other concurrency levels the pattern holds — the ML advantage concentrates at high TP.\n\n**Message**: the physical model's error is not uniformly distributed — it accumulates where multi-GPU communication and power sub-linearity enter. The ML corrector learns exactly these regimes, and that is where it is needed; at TP=1 the calibrated physics is already sufficient.",
        "next": "---\n\n**Notebook complete.** Data overview, real-vs-theoretical comparison, and ML pipeline (physical features, LOMO, A/B models) with artifacts and figures ready for thesis writing in `results/`.",
    },
}


CODE_ERR_TP = r'''# Section 7 - Error vs GPU count (TP) at fixed concurrency.
# Where does the physical model's error concentrate as parallelism grows,
# and does the ML corrector absorb it? Per-config LOMO predictions,
# best estimator per target (Step 5.3). TP=1 has no inter-GPU communication
# (eta=1): the residual error there measures pure MFU/MBU quality.
try:
    ep = comp.copy()                     # in-memory from Step 5.3
except NameError:
    ep = pd.read_csv("results/lomo_predictions_all_estimators.csv")

BEST_TP = {"ttft_avg_ms": ("RF", "B"), "itl_avg_ms": ("GBM", "B"), "power_w": ("RF", "A")}
C_FIX = 1

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
rows = []
for ax, (tgt, (est, fr)) in zip(axes, BEST_TP.items()):
    sub = ep[(ep.target == tgt) & (ep.estimator == est) & (ep.framing == fr)
             & (ep.concurrent_users == C_FIX)]
    g = sub.groupby("n_gpu").agg(n=("ape_theo", "size"),
                                 theo=("ape_theo", "median"),
                                 ml=("ape_pred", "median"))
    x = g.index.astype(str)
    lbl = "TDP nameplate" if tgt == "power_w" else "calibrated roofline"
    ax.plot(x, g.theo, "o--", color="#C44E52", label=lbl)
    ax.plot(x, g.ml, "o-", color="#55A868", label=f"ML corrector ({est}-{fr})")
    ax.set(title=tgt.replace("_avg_ms", "").upper().replace("POWER_W", "power (W)"),
           xlabel="n GPU (= TP)", ylabel="medAPE %")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    for tp, r in g.iterrows():
        rows.append({"target": tgt, "n_gpu": int(tp), "n_config": int(r.n),
                     "medAPE_theo": round(r.theo, 1), "medAPE_ml": round(r.ml, 1)})
fig.suptitle(f"Prediction error vs GPU count at fixed concurrency (C={C_FIX}): physics vs ML corrector", y=1.03)
fig.tight_layout()
fig.savefig("results/figures/error_vs_tp.png", dpi=150, bbox_inches="tight")
plt.show()

err_tp = pd.DataFrame(rows)
err_tp.to_csv("results/error_vs_tp.csv", index=False)
print(err_tp.to_string(index=False))

# Robustness: same view at the other concurrency levels (TTFT, best = RF-B)
for c in [10, 50, 100]:
    s = ep[(ep.target == "ttft_avg_ms") & (ep.estimator == "RF") & (ep.framing == "B")
           & (ep.concurrent_users == c)]
    piv = s.groupby("n_gpu")[["ape_theo", "ape_pred"]].median().round(0).astype(int)
    print(f"TTFT @ C={c}: " + " | ".join(f"TP{k}: roofline {v.ape_theo}% vs ML {v.ape_pred}%"
                                          for k, v in piv.iterrows()))'''


def build(lang):
    t = MD[lang]
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(t["title"]),
        nbf.v4.new_code_cell(CODE_SETUP),
        nbf.v4.new_markdown_cell(t["load"]),
        nbf.v4.new_code_cell(CODE_LOAD),
        nbf.v4.new_markdown_cell(t["filter"]),
        nbf.v4.new_code_cell(CODE_FILTER),
        nbf.v4.new_markdown_cell(t["completeness"]),
        nbf.v4.new_code_cell(CODE_COMPLETENESS),
        nbf.v4.new_code_cell(CODE_EMPTY),
        nbf.v4.new_markdown_cell(t["empty"]),
        nbf.v4.new_markdown_cell(t["coverage"]),
        nbf.v4.new_code_cell(CODE_COVERAGE),
        nbf.v4.new_code_cell(CODE_COUNTS),
        nbf.v4.new_markdown_cell(t["describe"]),
        nbf.v4.new_code_cell(CODE_DESCRIBE),
        nbf.v4.new_markdown_cell(t["s2_intro"]),
        nbf.v4.new_markdown_cell(t["s2_dist"]),
        nbf.v4.new_code_cell(CODE_DIST),
        nbf.v4.new_markdown_cell(t["s2_trends"]),
        nbf.v4.new_code_cell(CODE_TREND_HELPER),
        nbf.v4.new_code_cell(CODE_TREND_TP),
        nbf.v4.new_code_cell(CODE_TREND_LAT),
        nbf.v4.new_code_cell(CODE_TREND_TTFT),
        nbf.v4.new_code_cell(CODE_TREND_ITL),
        nbf.v4.new_code_cell(CODE_TREND_PW),
        nbf.v4.new_markdown_cell(t["s3_intro"]),
        nbf.v4.new_markdown_cell(t["s3_agg"]),
        nbf.v4.new_code_cell(CODE_KEYS),
        nbf.v4.new_code_cell(CODE_AGG),
        nbf.v4.new_markdown_cell(t["s3_merge"]),
        nbf.v4.new_code_cell(CODE_MERGE),
        nbf.v4.new_markdown_cell(t["s3_coverage"]),
        nbf.v4.new_code_cell(CODE_COVERAGE3),
        nbf.v4.new_markdown_cell(t["s4_method"]),
        nbf.v4.new_code_cell(CODE_ERR_SETUP),
        nbf.v4.new_markdown_cell(t["s4_asis"]),
        nbf.v4.new_code_cell(CODE_ASIS_TABLE),
        nbf.v4.new_code_cell(CODE_ASIS_SCATTER),
        nbf.v4.new_markdown_cell(t["s4_b"]),
        nbf.v4.new_code_cell(CODE_APPROACH_B),
        nbf.v4.new_markdown_cell(t["s4_c"]),
        nbf.v4.new_code_cell(CODE_C_TABLE),
        nbf.v4.new_code_cell(CODE_C_BREAKDOWN),
        nbf.v4.new_code_cell(CODE_C_SCATTER),
        nbf.v4.new_markdown_cell(t["s4_end"]),
        nbf.v4.new_markdown_cell(t["s5_intro"]),
        nbf.v4.new_markdown_cell(t["s5_specs"]),
        nbf.v4.new_code_cell(CODE_SPECS),
        nbf.v4.new_code_cell(CODE_FEATS),
        nbf.v4.new_code_cell(CODE_FE_SANITY),
        nbf.v4.new_markdown_cell(t["s51b_intro"]),
        nbf.v4.new_code_cell(CODE_FEAT_PREP),
        nbf.v4.new_code_cell(CODE_CORR),
        nbf.v4.new_code_cell(CODE_PCA),
        nbf.v4.new_code_cell(CODE_TSNE),
        nbf.v4.new_markdown_cell(t["s51_end"]),
        nbf.v4.new_markdown_cell(t["s52_intro"]),
        nbf.v4.new_code_cell(CODE_LOMO_SETUP),
        nbf.v4.new_code_cell(CODE_LOMO_FUNC),
        nbf.v4.new_code_cell(CODE_LOMO_RUN),
        nbf.v4.new_markdown_cell(t["s52_end"]),
        nbf.v4.new_markdown_cell(t["s53_intro"]),
        nbf.v4.new_code_cell(CODE_ESTIMATORS),
        nbf.v4.new_code_cell(CODE_BEST),
        nbf.v4.new_code_cell(CODE_FINAL_PLOTS),
        nbf.v4.new_code_cell(CODE_SAVE53),
        nbf.v4.new_markdown_cell(t["s53_findings"]),
        nbf.v4.new_markdown_cell(t["s5_end"]),
        nbf.v4.new_markdown_cell(t["s6_intro"]),
        nbf.v4.new_code_cell(CODE_SUMMARY),
        nbf.v4.new_code_cell(CODE_EXPORT),
        nbf.v4.new_markdown_cell(t["s6_energy"]),
        nbf.v4.new_code_cell(CODE_ENERGY),
        nbf.v4.new_markdown_cell(t["s6_limits"]),
        nbf.v4.new_markdown_cell(t["s7_tp_intro"]),
        nbf.v4.new_code_cell(CODE_ERR_TP),
        nbf.v4.new_markdown_cell(t["s7_tp_end"]),
        nbf.v4.new_markdown_cell(t["next"]),
    ]
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    return nb


if __name__ == "__main__":
    for lang, fname in [("IT", "analisi_tesi_IT.ipynb"), ("EN", "analysis_thesis_EN.ipynb")]:
        with open(fname, "w", encoding="utf-8") as fh:
            nbf.write(build(lang), fh)
        print("written", fname)
