# Run the fine-tuned Phi-4-mini macro agent.  pip install -r requirements_inference.txt
#
#   from inference import MacroAgent
#   agent = MacroAgent(".")                     # folder containing this file
#   agent.predict("Energy", {"spy_beta": 0.95, "beta_r_oil": 0.04, "VIXCLS": 16.0, "DGS10": 5.26})
#   -> {"expected_vol_pct": 21.4, "risk_label": "HIGH", "macro_risk_score": 7.3}
#
# The dict uses the same column names as eval_set.parquet. Missing / NaN values are left out of the prompt,
# exactly as in training. A GPU is recommended (about 0.3 s per answer); CPU works but is slow.
import json, math, re
from pathlib import Path


class MacroAgent:
    def __init__(self, folder=".", device=None, adapter="phi4_mini_macro_lora"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        self.folder = Path(folder)
        self.cfg = json.loads((self.folder / "prompt_config.json").read_text())
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.tok = AutoTokenizer.from_pretrained(self.folder / adapter)
        base = AutoModelForCausalLM.from_pretrained(self.cfg["base_model"], torch_dtype=dtype)
        self.model = PeftModel.from_pretrained(base, self.folder / adapter).to(self.device).eval()

    def build_prompt(self, sector, row):
        def ok(v):
            return v is not None and not (isinstance(v, float) and math.isnan(v))
        lines = [self.cfg["header"], "", f"SECTOR: {sector}",
                 "SECTOR SENSITIVITIES (% sector return per move, net of market):"]
        for s in self.cfg["sensitivities"]:
            v = row.get(s["column"])
            if ok(v):
                lines.append(f"{s['key']}: {round(float(v) * s['scale'], 3)}")
        lines.append("INDICATORS:")
        for c in self.cfg["features"]:
            v = row.get(c)
            if ok(v):
                lines.append(f"{c}: {round(float(v), 3)}")
        return "\n".join(lines)

    def predict_raw(self, prompt):
        torch = self.torch
        text = self.tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                            add_generation_prompt=True)
        enc = self.tok(text, return_tensors="pt", add_special_tokens=False).to(self.device)
        with torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=self.cfg["generation"]["max_new_tokens"], do_sample=False,
                                      pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id)
        return self.tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)

    @staticmethod
    def parse(text):
        m = re.search(r"\{.*?\}", text, flags=re.S)
        try:
            o = json.loads(m.group(0)) if m else None
        except Exception:
            return None
        if not isinstance(o, dict) or str(o.get("risk_label", "")).upper() not in ("LOW", "MODERATE", "HIGH"):
            return None
        o["risk_label"] = o["risk_label"].upper()
        return o

    def predict(self, sector, row):
        return self.parse(self.predict_raw(self.build_prompt(sector, row)))
