/// Inference engine selector. Default is candle until each model passes its
/// Burn parity gate (see `benchmark/burn_migration.md`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Engine {
    Candle,
    Burn,
}

impl Engine {
    pub fn name(self) -> &'static str {
        match self {
            Engine::Candle => "candle",
            Engine::Burn => "burn",
        }
    }
}

impl std::str::FromStr for Engine {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "candle" => Ok(Engine::Candle),
            "burn" => Ok(Engine::Burn),
            other => anyhow::bail!("unknown engine {other:?}: expected candle|burn"),
        }
    }
}

/// Read the engine from `ZSFM_ENGINE` (set by `zsfm --engine`). Defaults to
/// candle so existing scripts keep working during the migration.
pub fn engine_from_env() -> Engine {
    std::env::var("ZSFM_ENGINE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(Engine::Candle)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses() {
        assert_eq!("burn".parse::<Engine>().unwrap(), Engine::Burn);
        assert_eq!("CANDLE".parse::<Engine>().unwrap(), Engine::Candle);
        assert!("gpu".parse::<Engine>().is_err());
    }
}
