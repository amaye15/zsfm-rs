use zsfm_core::{
    parse_horizon, parse_matrix, parse_mv_contexts, softmax, validate_horizon, MAX_HORIZON,
};

#[test]
fn cli_forecast_envelope_roundtrip() {
    let contexts = parse_mv_contexts(serde_json::json!([1.0, 2.0, 3.0])).unwrap();
    assert_eq!(contexts.len(), 1);
    let horizon = parse_horizon(&serde_json::json!({"horizon": 4})).unwrap();
    assert_eq!(horizon, 4);
    assert!(validate_horizon(MAX_HORIZON).is_ok());
    assert!(validate_horizon(MAX_HORIZON + 1).is_err());
    assert!(parse_horizon(&serde_json::json!({"horizon": 0})).is_err());
}

#[test]
fn cli_rejects_bad_shapes() {
    assert!(parse_mv_contexts(serde_json::json!([])).is_err());
    assert!(parse_mv_contexts(serde_json::json!([[]])).is_err());
    assert!(parse_mv_contexts(serde_json::json!([[[]]])).is_err());
}

#[test]
fn tabular_matrix_parity_with_cli() {
    let m = parse_matrix(&serde_json::json!([[1.0, 2.0], [3.0, 4.0]])).unwrap();
    assert_eq!(m.len(), 2);
    assert!(parse_matrix(&serde_json::json!([])).is_err());
    assert!(parse_matrix(&serde_json::json!([[1.0], [2.0, 3.0]])).is_err());
    let p = softmax(&[1.0, 2.0, 3.0]);
    assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-5);
}

#[test]
fn quantile_output_shapes() {
    let qmat = vec![vec![vec![1.0, 2.0]], vec![vec![0.5, 1.5]]];
    let out = zsfm_core::quantile_matrix_to_output(&qmat, &[0.5, 0.9], 0);
    match out {
        zsfm_core::ForecastOutput::Univariate { point, .. } => assert_eq!(point, vec![1.0, 2.0]),
        _ => panic!("expected univariate"),
    }
    let json = zsfm_core::forecast_response_json(
        "test-model",
        3,
        2,
        vec![zsfm_core::quantile_matrix_to_output(&qmat, &[0.5, 0.9], 0)],
    )
    .unwrap();
    assert!(json.contains("test-model"));
}
