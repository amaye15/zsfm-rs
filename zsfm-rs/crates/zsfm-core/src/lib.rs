mod cfgval;
mod error;
mod forecast;
mod input;
mod json;

pub use cfgval::{json_bool, json_f64, json_usize};
pub use error::ValidationError;
pub use forecast::{Forecaster, QuantileMatrix};
pub use input::{
    parse_horizon, parse_matrix, parse_mv_contexts, read_stdin_limited, softmax, validate_contexts,
    validate_horizon, MAX_CONTEXT_VALUES, MAX_HORIZON, MAX_STDIN_BYTES,
};
pub use json::{
    forecast_response_json, quantile_matrix_to_output, ForecastOutput, VariateForecast,
};
