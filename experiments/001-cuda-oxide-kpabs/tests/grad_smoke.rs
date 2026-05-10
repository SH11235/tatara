//! Grad kernel の reference CPU 実装に対する smoke test。
//!
//! 手計算可能な小さな入力で `grad_cpu` の loss 累積 / gradient scatter /
//! prediction histogram が期待値と一致することを確認する。GPU 実機
//! (cuda-oxide PTX) との bit-equivalent 検証は Stage 1-9 (#13) で host loop
//! が組まれた段階で別途追加する。

use exp_001_cuda_oxide_kpabs::kernels::grad::grad_cpu;

#[test]
fn single_position_known_input_matches_hand_calculation() {
    // 1 position, 3 weights, max_inds = 3, indices = [0, 2, -1]
    // p = 0.5, y = 0.0, norm = 1.0
    //   err = 0.5
    //   gscale = 2 * 0.5 * 0.5 * 0.5 * 1.0 = 0.25
    //   grad[0] += 0.25, grad[2] += 0.25, idx=-1 skip
    //   loss += 0.25 (= err^2)
    //   bin = (int)(0.5 * 8) = 4 → hist[4] += 1
    let indices = vec![0_i32, 2, -1];
    let preds = vec![0.5_f32];
    let targets = vec![0.0_f32];
    let per_pos_norm = vec![1.0_f32];
    let mut grad = vec![0.0_f32; 3];
    let mut loss_acc = 0.0_f64;
    let mut hist = [0_u64; 8];

    grad_cpu(
        &indices,
        &preds,
        &targets,
        &per_pos_norm,
        &mut grad,
        &mut loss_acc,
        &mut hist,
        1,
        3,
    );

    assert!((grad[0] - 0.25).abs() < 1e-6, "grad[0] = {}", grad[0]);
    assert!(grad[1].abs() < 1e-6, "grad[1] should be 0, got {}", grad[1]);
    assert!((grad[2] - 0.25).abs() < 1e-6, "grad[2] = {}", grad[2]);
    assert!(
        (loss_acc - 0.25).abs() < 1e-12,
        "loss_acc = {} expected 0.25",
        loss_acc
    );
    assert_eq!(hist, [0, 0, 0, 0, 1, 0, 0, 0], "hist = {:?}", hist);
}

#[test]
fn padding_positions_do_not_touch_grad() {
    // 全 index が padding (-1) → gscale 計算は走るが scatter は no-op
    // p = 0.7, y = 1.0 → err = -0.3
    //   gscale = 2 * (-0.3) * 0.7 * 0.3 * 1.0 = -0.126
    //   grad に変更なし
    //   loss += 0.09
    //   bin = (int)(0.7 * 8) = (int)5.6 = 5 → hist[5] += 1
    let indices = vec![-1_i32; 4];
    let preds = vec![0.7_f32];
    let targets = vec![1.0_f32];
    let per_pos_norm = vec![1.0_f32];
    let mut grad = vec![0.0_f32; 5];
    let mut loss_acc = 0.0_f64;
    let mut hist = [0_u64; 8];

    grad_cpu(
        &indices,
        &preds,
        &targets,
        &per_pos_norm,
        &mut grad,
        &mut loss_acc,
        &mut hist,
        1,
        4,
    );

    for (i, &g) in grad.iter().enumerate() {
        assert!(
            g.abs() < 1e-6,
            "grad[{i}] should be 0 (all padding), got {g}"
        );
    }
    // err は f32 計算 (0.7_f32 - 1.0_f32 ≈ -0.30000001) のため、loss = (err as f64)^2 で expected を組む。
    // 直接 0.09 と比較すると f32 精度由来の誤差で fail する。
    let err = 0.7_f32 - 1.0_f32;
    let expected_loss: f64 = (err as f64) * (err as f64);
    assert!(
        (loss_acc - expected_loss).abs() < 1e-12,
        "loss_acc = {} expected {}",
        loss_acc,
        expected_loss
    );
    assert_eq!(hist, [0, 0, 0, 0, 0, 1, 0, 0], "hist = {:?}", hist);
}

#[test]
fn multi_position_grad_accumulates_across_positions() {
    // 同じ weight index が 2 つの position から書かれるケース。
    // n_pos = 2, max_inds = 2, 4 weights
    // pos 0: indices [1, 3], p = 0.5, y = 0.0, norm = 1.0
    //   err = 0.5, gscale = 2 * 0.5 * 0.5 * 0.5 * 1.0 = 0.25
    //   grad[1] += 0.25, grad[3] += 0.25
    //   loss += 0.25, bin = 4
    // pos 1: indices [1, 0], p = 0.25, y = 0.0, norm = 1.0
    //   err = 0.25, gscale = 2 * 0.25 * 0.25 * 0.75 * 1.0 = 0.09375
    //   grad[1] += 0.09375, grad[0] += 0.09375
    //   loss += 0.0625, bin = (int)(0.25 * 8) = (int)2.0 = 2
    let indices = vec![1_i32, 3, 1, 0];
    let preds = vec![0.5_f32, 0.25];
    let targets = vec![0.0_f32, 0.0];
    let per_pos_norm = vec![1.0_f32, 1.0];
    let mut grad = vec![0.0_f32; 4];
    let mut loss_acc = 0.0_f64;
    let mut hist = [0_u64; 8];

    grad_cpu(
        &indices,
        &preds,
        &targets,
        &per_pos_norm,
        &mut grad,
        &mut loss_acc,
        &mut hist,
        2,
        2,
    );

    assert!((grad[0] - 0.09375).abs() < 1e-6, "grad[0] = {}", grad[0]);
    assert!(
        (grad[1] - (0.25 + 0.09375)).abs() < 1e-6,
        "grad[1] = {} expected {}",
        grad[1],
        0.25 + 0.09375
    );
    assert!(grad[2].abs() < 1e-6, "grad[2] should be 0, got {}", grad[2]);
    assert!((grad[3] - 0.25).abs() < 1e-6, "grad[3] = {}", grad[3]);

    let expected_loss: f64 = 0.25 + 0.0625;
    assert!(
        (loss_acc - expected_loss).abs() < 1e-9,
        "loss_acc = {} expected {}",
        loss_acc,
        expected_loss
    );
    assert_eq!(hist, [0, 0, 1, 0, 1, 0, 0, 0], "hist = {:?}", hist);
}

#[test]
fn histogram_bin_clamps_at_boundaries() {
    // p = 0.0 → bin = 0
    // p = 1.0 → (int)(1.0 * 8.0) = 8、clamp で 7
    // p > 1.0 (異常入力だが C++ 上流も clamp する) → bin 7
    // p < 0.0 (sigmoid 出力としてはあり得ないが clamp 検証用) → bin 0
    let indices = vec![-1_i32; 4];
    let preds = vec![0.0_f32, 1.0, 1.5, -0.5];
    let targets = vec![0.0_f32, 0.0, 0.0, 0.0];
    let per_pos_norm = vec![1.0_f32; 4];
    let mut grad = vec![0.0_f32; 1];
    let mut loss_acc = 0.0_f64;
    let mut hist = [0_u64; 8];

    grad_cpu(
        &indices,
        &preds,
        &targets,
        &per_pos_norm,
        &mut grad,
        &mut loss_acc,
        &mut hist,
        4,
        1,
    );

    // p=0.0 → bin 0, p=1.0 → 7 (8 を clamp), p=1.5 → 7 (12 を clamp), p=-0.5 → 0 (-4 を clamp)
    assert_eq!(
        hist[0], 2,
        "hist[0] = {} (p=0.0 と p=-0.5 が落ちる想定)",
        hist[0]
    );
    assert_eq!(
        hist[7], 2,
        "hist[7] = {} (p=1.0 と p=1.5 が落ちる想定)",
        hist[7]
    );
    let other_sum: u64 = (1..7).map(|i| hist[i]).sum();
    assert_eq!(other_sum, 0, "中間 bin は 0 のはず: {:?}", hist);
}

#[test]
fn negative_error_yields_negative_gscale() {
    // p < y → err < 0 → gscale < 0 → grad は減少方向
    // p = 0.2, y = 0.8 → err = -0.6
    // gscale = 2 * (-0.6) * 0.2 * 0.8 * 1.0 = -0.192
    let indices = vec![0_i32];
    let preds = vec![0.2_f32];
    let targets = vec![0.8_f32];
    let per_pos_norm = vec![1.0_f32];
    let mut grad = vec![0.0_f32; 1];
    let mut loss_acc = 0.0_f64;
    let mut hist = [0_u64; 8];

    grad_cpu(
        &indices,
        &preds,
        &targets,
        &per_pos_norm,
        &mut grad,
        &mut loss_acc,
        &mut hist,
        1,
        1,
    );

    assert!(
        (grad[0] - (-0.192)).abs() < 1e-6,
        "grad[0] = {} expected -0.192",
        grad[0]
    );
    // err は f32 計算 (0.2_f32 - 0.8_f32) のため f32 精度誤差込みで expected を組む。
    let err = 0.2_f32 - 0.8_f32;
    let expected_loss: f64 = (err as f64) * (err as f64);
    assert!(
        (loss_acc - expected_loss).abs() < 1e-12,
        "loss_acc = {} expected {}",
        loss_acc,
        expected_loss
    );
    // bin = (int)(0.2 * 8) = (int)1.6 = 1
    assert_eq!(hist, [0, 1, 0, 0, 0, 0, 0, 0], "hist = {:?}", hist);
}

#[test]
fn per_pos_norm_scales_gradient() {
    // norm = 0.5 → gscale が半分になる、loss は norm に依存しない
    // p = 0.5, y = 0.0, norm = 0.5 → err = 0.5
    //   gscale = 2 * 0.5 * 0.5 * 0.5 * 0.5 = 0.125
    //   loss += 0.25 (norm に依存しない、loss は raw error^2)
    let indices = vec![0_i32];
    let preds = vec![0.5_f32];
    let targets = vec![0.0_f32];
    let per_pos_norm = vec![0.5_f32];
    let mut grad = vec![0.0_f32; 1];
    let mut loss_acc = 0.0_f64;
    let mut hist = [0_u64; 8];

    grad_cpu(
        &indices,
        &preds,
        &targets,
        &per_pos_norm,
        &mut grad,
        &mut loss_acc,
        &mut hist,
        1,
        1,
    );

    assert!((grad[0] - 0.125).abs() < 1e-6, "grad[0] = {}", grad[0]);
    assert!(
        (loss_acc - 0.25).abs() < 1e-9,
        "loss_acc = {} expected 0.25 (norm に非依存)",
        loss_acc
    );
}
