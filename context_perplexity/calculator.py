"""Reader-centric context log-probability (PerpC) feature helper."""
from __future__ import annotations

import pandas as pd


class ContextPerplexityCalculator:
    """Compute the context-probability signals used as PerpC features.

    The returned values are average conditional token log-probabilities, matching
    the quantity consumed by the original ECIR analysis code. The historical
    feature names are retained for result compatibility.
    """

    def __init__(
        self,
        text_loader,
        model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        device: str | None = None,
        model=None,
        tokenizer=None,
    ):
        self.text_loader = text_loader
        if tokenizer is None or model is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            model = model or AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
            device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or str(next(model.parameters()).device)
        self.model.eval()

    def _conditional_avg_log_prob(self, prefix: str, continuation: str, max_length: int) -> float:
        import torch
        import torch.nn.functional as F

        full = prefix + continuation
        full_inputs = self.tokenizer(
            full,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        prefix_inputs = self.tokenizer(
            prefix,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        prefix_len = int(prefix_inputs["input_ids"].shape[1])
        input_ids = full_inputs["input_ids"].to(self.device)
        attention = full_inputs["attention_mask"].to(self.device)

        with torch.no_grad():
            logits = self.model(input_ids=input_ids, attention_mask=attention).logits
        shift_logits = logits[:, :-1, :]
        labels = input_ids[:, 1:]
        token_log_probs = F.log_softmax(shift_logits, dim=-1).gather(
            -1, labels.unsqueeze(-1)
        ).squeeze(-1)

        # shifted position j predicts original token j+1
        start = max(0, prefix_len - 1)
        valid = attention[:, 1:].bool()
        positions = torch.arange(token_log_probs.shape[1], device=self.device)[None, :]
        mask = valid & (positions >= start)
        selected = token_log_probs[mask]
        if selected.numel() == 0:
            return float("nan")
        return float(selected.mean().detach().cpu())

    def compute(self, retrieval: pd.DataFrame, k: int) -> pd.DataFrame:
        rows = []
        max_length = 1024 + 256 * max(0, k - 5)

        for qid, group in retrieval.groupby("qid", sort=False):
            selected = group[group["rank"] < k]
            loaded = self.text_loader(selected)
            query = str(group["query"].iloc[0])
            prefix = f"Q: {query}\nA: "
            texts = [str(x) for x in loaded["text"].values]

            individual = [
                self._conditional_avg_log_prob(prefix, text, max_length=512)
                for text in texts
            ]
            series = pd.Series(individual, dtype=float)
            integrated = self._conditional_avg_log_prob(
                prefix, "".join(texts), max_length=max_length
            )
            rows.append(
                {
                    "qid": str(qid),
                    "max(perpC)": series.max(skipna=True),
                    "min(perpC)": series.min(skipna=True),
                    "avg(perpC)": series.mean(skipna=True),
                    "itg(perpC)": integrated,
                }
            )
        return pd.DataFrame(rows)
