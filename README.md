# LLM ML Benchmarking — Validation & LOMO Analysis

Analysis repository of the **LLM Sizing Challenge** framework: it holds the benchmark dataset, the theoretical-vs-measured validation, and the **leave-one-model-out (LOMO) machine-learning corrector**.

## What's here

| Folder | Content |
|---|---|
| `data/` | `benchmark_results.csv` — the **160-run measured dataset** (AIPerf + per-run power; 6 models 7B→120B, H100 PCIe vs H200 NVLink, TP 1–8, concurrency 1–100, unified 24-column schema). `sizing_results.csv` — the paired predictions of the calibrated sizing tool (plus `_prefit` and `_legacy` generations for the accuracy-ladder comparison) |
| `notebooks/` | `analysis_thesis_EN.ipynb` (English) and `analisi_tesi_IT.ipynb` (Italian): full validation — pairing, error decomposition, power analysis, energy/TCO. `analisi_tesi_IT_lomo.ipynb`: LOMO re-calibration of the roofline. `confronto_sizing_old_new.ipynb`: the three generations of the predictive model |
| `scripts/` | `build_notebooks.py` / `build_confronto_old_new.py` — deterministic notebook builders (the notebooks are generated, not hand-edited) |
| `results/` | Summary tables, LOMO per-fold predictions, and the paper figures |

## Key results

- **Three-step accuracy ladder** (median APE on TTFT): heuristic 87% → calibrated roofline 53% → ML corrector **40%** (under LOMO extrapolation to unseen models)
- **Roofline LOMO re-calibration**: 53.4% in-sample vs 52.1% out-of-fold (TTFT) — the calibration captures transferable structure, making the physics-vs-ML comparison symmetric
- **ML corrector** (29 physical features, no model identity): TTFT 39.6% (RF, residual framing), ITL 29.2% (GBM, residual framing), **power 0.9%** (RF, direct framing) vs 167.8% for the TDP nameplate
- **Energy**: batching is the dominant lever (~20× on kWh/Mtok from C=1 to C=100); the TDP nameplate overestimates real power by ~2.6× in the median

## Reproducing

```bash
git clone https://github.com/VRscience/LLM_ML_benchmarking.git
cd LLM_ML_benchmarking
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
python scripts/build_notebooks.py     # regenerates the analysis notebooks from data/
jupyter lab notebooks/
```

## Related repositories

- [LLM_Infra_Sizing_Tool](https://github.com/VRscience/LLM_Infra_Sizing_Tool) — the analytical sizing model whose predictions are validated here
- [LLM_benchmarking](https://github.com/VRscience/LLM_benchmarking) — the measurement pipeline that produced `benchmark_results.csv`
- [Applied_AI_Thesis](https://github.com/VRscience/Applied_AI_Thesis) — thesis and whitepaper
