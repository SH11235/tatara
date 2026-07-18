//! GPU/CUDA toolkitなしの通常CIでも、versioned benchmark report fixtureをJSON Schemaへ
//! 通してschema自体と代表documentのdriftを検出する。

#[test]
fn native_cuda_benchmark_v1_fixture_conforms_to_schema() {
    let schema: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../docs/schemas/native-cuda-benchmark-v1.schema.json"
    ))
    .expect("benchmark schema must be valid JSON");
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../docs/schemas/fixtures/native-cuda-benchmark-v1.json"
    ))
    .expect("benchmark fixture must be valid JSON");
    let validator = jsonschema::validator_for(&schema).expect("benchmark schema must compile");
    let errors = validator
        .iter_errors(&fixture)
        .map(|error| error.to_string())
        .collect::<Vec<_>>();
    assert!(errors.is_empty(), "schema validation failed: {errors:#?}");
}
