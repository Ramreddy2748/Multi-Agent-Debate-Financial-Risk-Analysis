# Macro-Economic Risk Agent: fine-tuned Phi-4-mini (LoRA)

What it does: given one sector ETF and a snapshot of macro indicators (no dates), it forecasts the sector's annualized
realized volatility over the next 60 trading days and returns JSON:
  {"expected_vol_pct": 17.3, "risk_label": "MODERATE", "macro_risk_score": 6.1}
Labels: LOW <= 13.0%, MODERATE 13.0-18.5%, HIGH > 18.5%. Score 0 (calm) to 10 (turbulent).

## Files
- predictions_all.csv            forecasts + actual outcomes for 2019-2026 (val / test / post_cutoff). Easiest for a dashboard.
- latest_forecast_per_sector.json  most recent forecast for each of the 11 sectors.
- metrics.json                   Spearman, F1, MAE on held-out 2022-2026 rows; train_log.csv has the loss curve.
- phi4_mini_macro_lora.pt        LoRA weights as a PyTorch state dict (needs the base model microsoft/Phi-4-mini-instruct).
- phi4_mini_macro_lora/          the same adapter in Hugging Face format (adapter_config.json, safetensors, tokenizer).
- prompt_config.json             exact prompt template, feature list, thresholds, sector list.
- inference.py                   MacroAgent class: load base + adapter and call predict(sector, indicators).
- label_thresholds.json, training_config.json

## Notes for the dashboard
- split column: val = 2019-2021 (used for model selection), test = 2022-2024, post_cutoff = 2025-2026.
  Rows with has_label = False are the latest months where the 60-day outcome is not known yet.
- All sectors share the same macro snapshot in a given month, so differences between sectors come from their sensitivities.
- The model is a sector-level systematic risk signal, not a stock picker. Held-out Spearman about 0.68 pooled,
  weaker in 2025-2026. The model under-predicts volatility by about 3 points on average.
- Loading the model needs a GPU for speed; predictions_all.csv needs none.
