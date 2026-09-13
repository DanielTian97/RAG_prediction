import pandas as pd

from analyse import combination_correlations, single_correlations


def test_combination_correlations_handles_all_nan_single_correlations():
    frame = pd.DataFrame(
        {
            "qid": ["1", "2", "3", "4"],
            "nqc": [0.5, 0.5, 0.5, 0.5],
            "f1": [0.1, 0.3, 0.6, 0.9],
            "utility": [0.0, 0.2, 0.1, 0.4],
        }
    )
    singles = single_correlations(frame, ["nqc"])
    groups = {
        "qpp": ["nqc"],
        "context_perplexity": [],
        "qualt5": [],
        "posteriors": [],
        "readability": [],
    }

    result = combination_correlations(
        frame,
        frame,
        groups,
        singles,
        task="dl",
        retriever="e5",
        k=3,
    )

    assert len(result) == 2
    assert result["Best Single Signal"].isna().all()
    assert result["Best Single Rho"].isna().all()
    assert result["Best Single Tau"].isna().all()
