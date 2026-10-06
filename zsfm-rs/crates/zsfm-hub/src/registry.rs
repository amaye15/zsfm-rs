/// Canonical HuggingFace repo per friendly model name.
/// Single source of truth for CLI `--model` defaults and Python `convert()`
/// dispatch, so the two cannot drift when defaults change.
pub const MODEL_REPOS: &[(&str, &str)] = &[
    ("toto", "Datadog/Toto-2.0-2.5B"),
    ("chronos", "amazon/chronos-2"),
    ("timesfm", "google/timesfm-2.5-200m-pytorch"),
    ("sundial", "thuml/sundial-base-128m"),
    ("ttm", "ibm-granite/granite-timeseries-ttm-r2"),
    ("lag_llama", "time-series-foundation-models/Lag-Llama"),
    ("lag-llama", "time-series-foundation-models/Lag-Llama"),
    ("moment", "AutonLab/MOMENT-1-large"),
    ("moirai", "Salesforce/moirai-1.0-R-large"),
    ("moirai2", "Salesforce/moirai-2.0-R-small"),
    ("moirai-2", "Salesforce/moirai-2.0-R-small"),
    ("flowstate", "ibm-granite/granite-timeseries-flowstate-r1"),
    ("tirex", "NX-AI/TiRex"),
    ("mitra-classifier", "autogluon/mitra-classifier"),
    ("mitra-regression", "autogluon/mitra-regressor"),
    ("tabdpt", "Layer6/TabDPT"),
    ("tabicl", "jingang/TabICL"),
    ("tabpfn", "Prior-Labs/tabpfn_3"),
    ("tabfm", "google/tabfm-1.0.0-pytorch"),
];

/// Look up the default repo for a friendly model name.
pub fn default_repo_for(model: &str) -> Option<&'static str> {
    MODEL_REPOS
        .iter()
        .find(|(name, _)| *name == model)
        .map(|(_, repo)| *repo)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_model_has_repo() {
        for m in [
            "toto",
            "chronos",
            "timesfm",
            "sundial",
            "ttm",
            "moment",
            "moirai",
            "moirai2",
            "flowstate",
            "tirex",
            "tabdpt",
            "tabicl",
            "tabpfn",
            "tabfm",
        ] {
            assert!(default_repo_for(m).is_some(), "missing repo for {m}");
        }
    }
}
