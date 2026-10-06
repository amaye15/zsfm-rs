/// Typed validation errors for request parsing. Callers at the CLI edge
/// convert these to `anyhow` with context; library code can match on variants.
#[derive(Debug, thiserror::Error)]
pub enum ValidationError {
    #[error("context must be a non-empty array")]
    EmptyContext,
    #[error("context[{0}][{1}] must not be empty")]
    EmptySeries(usize, usize),
    #[error("context too large — got {0} values, max is {1}")]
    ContextTooLarge(usize, usize),
    #[error("context[{0}][{1}] must contain only finite numbers")]
    NonFinite(usize, usize),
    #[error("horizon must be a positive integer")]
    BadHorizon,
    #[error("horizon too large — got {0}, max is {1}")]
    HorizonTooLarge(usize, usize),
    #[error("stdin input too large — max is {0} bytes")]
    StdinTooLarge(u64),
    #[error("matrix must be a non-empty array")]
    EmptyMatrix,
    #[error("row {0} has inconsistent width")]
    RaggedMatrix(usize),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn displays() {
        assert_eq!(
            ValidationError::HorizonTooLarge(5000, 4096).to_string(),
            "horizon too large — got 5000, max is 4096"
        );
    }
}
