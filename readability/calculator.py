"""Readability feature helper used by the unified feature pipeline."""
from __future__ import annotations

import importlib.util
import os
import sys
from importlib.metadata import distribution, version
from pathlib import Path

import pandas as pd


def _load_external_readability_class():
    """Load ``py-readability-metrics`` despite the local package name collision.

    This repository intentionally exposes its feature helper as the local
    ``readability`` package, while ``py-readability-metrics`` also installs a
    package named ``readability``.  Most of the external package uses relative
    imports, but some scorer modules use absolute imports such as
    ``readability.exceptions``.  Loading it only under an alias therefore is
    insufficient.

    During initialisation we temporarily bind the external package to its
    canonical name so those absolute imports resolve against the installed
    distribution.  The repository's local ``readability`` package is restored
    immediately afterwards.
    """

    package_dir = Path(
        distribution("py-readability-metrics").locate_file("readability")
    )
    init_file = package_dir / "__init__.py"
    module_name = "_rag_external_readability"
    module = sys.modules.get(module_name)
    if module is None:
        spec = importlib.util.spec_from_file_location(
            module_name,
            init_file,
            submodule_search_locations=[str(package_dir)],
        )
        if spec is None or spec.loader is None:
            raise ImportError("Could not load py-readability-metrics.")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module

        local_readability = sys.modules.get("readability")
        sys.modules["readability"] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(module_name, None)
            raise
        finally:
            if local_readability is None:
                sys.modules.pop("readability", None)
            else:
                sys.modules["readability"] = local_readability

    return module.Readability


def _ensure_nltk_data() -> None:
    """Provide clean tokenizer data without relying on a user's NLTK cache.

    Cluster/home NLTK directories are sometimes partially extracted or contain
    resources produced by a different NLTK release.  We therefore prepend a
    small dedicated cache and verify resources only inside that cache.  This
    avoids a corrupt ``~/nltk_data`` shadowing otherwise valid downloads.
    """
    import nltk

    cache_dir = Path(
        os.environ.get(
            "RAG_NLTK_DATA",
            Path.home() / ".cache" / "rag_prediction" / "nltk_data",
        )
    ).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = str(cache_dir)
    if cache not in nltk.data.path:
        nltk.data.path.insert(0, cache)

    resources = [("tokenizers/punkt", "punkt")]

    # punkt_tab is used by newer NLTK releases.  Older releases can interpret
    # the name incorrectly (for example as ``punkt/PY3_tab``), so do not probe
    # it there.
    try:
        nltk_version = tuple(int(part) for part in version("nltk").split(".")[:3])
    except Exception:
        nltk_version = (0, 0, 0)
    if nltk_version >= (3, 8, 2):
        resources.append(("tokenizers/punkt_tab", "punkt_tab"))

    for resource, download_name in resources:
        try:
            nltk.data.find(resource, paths=[cache])
        except (LookupError, OSError):
            ok = nltk.download(download_name, download_dir=cache, quiet=True, force=True)
            if not ok:
                raise RuntimeError(f"Failed to download NLTK resource '{download_name}'.")
            try:
                nltk.data.find(resource, paths=[cache])
            except (LookupError, OSError) as exc:
                raise RuntimeError(
                    f"NLTK resource '{download_name}' could not be loaded from {cache}."
                ) from exc


class ReadabilityCalculator:
    METRICS = {
        "Dale Chall": "dale_chall",
        "Spache": "spache",
        "Flesch-Kincaid": "flesch_kincaid",
        "Flesch": "flesch",
        "Gunning Fog": "gunning_fog",
        "Coleman Liau": "coleman_liau",
        "ARI": "ari",
        "Linsear Write": "linsear_write",
        "SMOG": "smog",
    }

    def __init__(self, text_loader):
        _ensure_nltk_data()
        self.text_loader = text_loader
        self.readability_class = _load_external_readability_class()

    def _scores(self, text: str) -> dict[str, float]:
        readability = self.readability_class(text)
        values = {}
        for name, method_name in self.METRICS.items():
            try:
                values[name] = float(getattr(readability, method_name)().score)
            except Exception:
                values[name] = float("nan")
        return values

    def compute(self, retrieval: pd.DataFrame, k: int) -> pd.DataFrame:
        rows = []
        for qid, group in retrieval.groupby("qid", sort=False):
            selected = group[group["rank"] < k]
            loaded = self.text_loader(selected)
            individual = [self._scores(str(text)) for text in loaded["text"].values]
            integrated = self._scores("".join(str(x) for x in loaded["text"].values))
            row = {"qid": str(qid)}
            for metric in self.METRICS:
                vals = pd.Series([x[metric] for x in individual], dtype=float)
                row[f"max({metric})"] = vals.max(skipna=True)
                row[f"min({metric})"] = vals.min(skipna=True)
                row[f"avg({metric})"] = vals.mean(skipna=True)
                row[f"itg({metric})"] = integrated[metric]
            rows.append(row)
        return pd.DataFrame(rows)
