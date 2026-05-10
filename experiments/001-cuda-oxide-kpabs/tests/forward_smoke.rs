//! Forward kernel の reference CPU 実装に対する smoke test。
//!
//! 手計算可能な小さな入力で `forward_cpu` の出力が期待値と一致することを
//! 確認する。GPU 実機 (cuda-oxide PTX) との bit-equivalent 検証は
//! Stage 1-9 (#13) で host loop が組まれた段階で別途追加する。

use exp_001_cuda_oxide_kpabs::kernels::forward::forward_cpu;

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[test]
fn small_known_input_matches_hand_calculation() {
    // weights = [0.5, -0.3, 0.2, 0.0], n_pos = 3, max_inds = 2
    // pos 0: indices [0, 1] → w0 + w1 = 0.5 + (-0.3) = 0.2
    // pos 1: indices [-1, 2] → -1 skip, w2 = 0.2
    // pos 2: indices [0, -1] → w0 = 0.5, -1 skip
    let weights = vec![0.5_f32, -0.3, 0.2, 0.0];
    let n_pos = 3;
    let max_inds = 2;
    let indices = vec![0_i32, 1, -1, 2, 0, -1];

    let got = forward_cpu(&indices, &weights, n_pos, max_inds);

    let expected = [sigmoid(0.2), sigmoid(0.2), sigmoid(0.5)];
    for i in 0..n_pos {
        let diff = (got[i] - expected[i]).abs();
        assert!(
            diff < 1e-6,
            "pos {i}: got {got_v}, expected {exp_v}, diff {diff}",
            got_v = got[i],
            exp_v = expected[i],
        );
    }
}

#[test]
fn all_padding_yields_sigmoid_of_zero_equals_half() {
    // 全 index が padding (-1) → z = 0 → sigmoid(0) = 0.5 (厳密)
    let weights = vec![1.0_f32, 2.0, 3.0];
    let indices = vec![-1_i32; 5];
    let got = forward_cpu(&indices, &weights, 1, 5);
    assert_eq!(got.len(), 1);
    assert!(
        (got[0] - 0.5).abs() < 1e-6,
        "padding-only should give sigmoid(0)=0.5, got {}",
        got[0]
    );
}

#[test]
fn nontrivial_j_ordering_with_multiple_distinct_weights() {
    // weights = [1, 2, 4, 8], indices = [3, 1, 0, -1]
    // → z = w3 + w1 + w0 + (skip) = 8 + 2 + 1 = 11
    // sigmoid(11) ≈ 0.999983...
    // 重み index が単調でない順 + 異 weight 値の和 + padding 混在で、
    // 将来 kernel が誤って sort/dedup するような変更を入れたら検出する。
    let weights = vec![1.0_f32, 2.0, 4.0, 8.0];
    let indices = vec![3_i32, 1, 0, -1];
    let got = forward_cpu(&indices, &weights, 1, 4);
    let expected = sigmoid(11.0);
    assert!(
        (got[0] - expected).abs() < 1e-6,
        "non-trivial j-ordering: got {}, expected {} (sigmoid(11))",
        got[0],
        expected
    );
}

#[test]
fn empty_position_set_yields_empty() {
    // n_pos = 0 なら出力も空
    let got = forward_cpu(&[], &[1.0_f32, 2.0], 0, 4);
    assert!(got.is_empty());
}

#[test]
fn negative_weight_drives_sigmoid_below_half() {
    // 大きな負の z → sigmoid → 0 に近づく
    let weights = vec![-10.0_f32];
    // 1 position, max_inds = 3, 全部 weight idx 0 を参照 → z = -30
    let indices = vec![0_i32, 0, 0];
    let got = forward_cpu(&indices, &weights, 1, 3);
    assert!(got[0] < 0.001, "expected near 0, got {}", got[0]);
}
