//! Eval kernel の reference CPU 実装に対する smoke test。
//!
//! 手計算可能な小さな入力で `eval_cpu` の loss 累積 / histogram が期待値と
//! 一致することを確認する。GPU 実機 (cuda-oxide PTX) との bit-equivalent
//! 検証は Stage 1-9 (#13) で host loop が組まれた段階で別途追加する。
//!
//! `eval` は `grad` の loss/hist 部分のサブセットなので、同じ入力に対して
//! grad_cpu と loss/hist 出力が一致するはずという不変条件もテスト末尾で確認。

use exp_001_cuda_oxide_kpabs::kernels::eval::eval_cpu;
use exp_001_cuda_oxide_kpabs::kernels::grad::grad_cpu;

#[test]
fn single_position_known_input() {
    // p = 0.5, y = 0.0 → err = 0.5 → loss += 0.25
    // bin = (int)(0.5 * 8) = 4 → hist[4] += 1
    let preds = vec![0.5_f32];
    let targets = vec![0.0_f32];
    let mut loss_acc = 0.0_f64;
    let mut hist = [0_u64; 8];
    eval_cpu(&preds, &targets, &mut loss_acc, &mut hist, 1);

    assert!(
        (loss_acc - 0.25).abs() < 1e-12,
        "loss_acc = {} expected 0.25",
        loss_acc
    );
    assert_eq!(hist, [0, 0, 0, 0, 1, 0, 0, 0]);
}

#[test]
fn perfect_prediction_zero_loss() {
    // preds == targets で error なし → loss 0、histogram は preds 分布
    let preds = vec![0.1_f32, 0.5, 0.9];
    let targets = preds.clone();
    let mut loss_acc = 0.0_f64;
    let mut hist = [0_u64; 8];
    eval_cpu(&preds, &targets, &mut loss_acc, &mut hist, 3);

    assert!(
        loss_acc.abs() < 1e-12,
        "loss should be ~0, got {}",
        loss_acc
    );
    // bin 0 (0.1*8=0.8 → 0)、bin 4 (0.5*8=4)、bin 7 (0.9*8=7.2 → 7)
    assert_eq!(hist[0], 1);
    assert_eq!(hist[4], 1);
    assert_eq!(hist[7], 1);
    assert_eq!(hist[1] + hist[2] + hist[3] + hist[5] + hist[6], 0);
}

#[test]
fn histogram_clamp_at_boundaries() {
    let preds = vec![0.0_f32, 1.0, 1.5, -0.5];
    let targets = vec![0.0_f32; 4];
    let mut loss_acc = 0.0_f64;
    let mut hist = [0_u64; 8];
    eval_cpu(&preds, &targets, &mut loss_acc, &mut hist, 4);

    // 0.0 → bin 0、1.0 → 8 を clamp して 7、1.5 → 12 を clamp して 7、-0.5 → -4 を clamp して 0
    assert_eq!(hist[0], 2, "p=0.0 と p=-0.5 が落ちる想定: {:?}", hist);
    assert_eq!(hist[7], 2, "p=1.0 と p=1.5 が落ちる想定: {:?}", hist);
    assert_eq!(hist.iter().sum::<u64>(), 4);
}

#[test]
fn loss_accumulates_across_batch() {
    // 3 positions: err = 0.5, -0.3, 0.0 → loss = 0.25 + 0.09 + 0 = 0.34
    let preds = vec![0.5_f32, 0.2, 0.7];
    let targets = vec![0.0_f32, 0.5, 0.7];
    let mut loss_acc = 0.0_f64;
    let mut hist = [0_u64; 8];
    eval_cpu(&preds, &targets, &mut loss_acc, &mut hist, 3);

    let err1 = 0.5_f32 - 0.0_f32;
    let err2 = 0.2_f32 - 0.5_f32;
    let err3 = 0.7_f32 - 0.7_f32;
    let expected = (err1 as f64) * (err1 as f64)
        + (err2 as f64) * (err2 as f64)
        + (err3 as f64) * (err3 as f64);
    assert!(
        (loss_acc - expected).abs() < 1e-12,
        "loss_acc = {} expected {}",
        loss_acc,
        expected
    );
}

#[test]
fn eval_output_matches_grad_loss_hist_subset() {
    // eval は grad の loss/hist 部分のサブセット。同じ入力で両者の loss_acc / hist
    // が一致することを確認 (grad 側は scatter / norm が追加で動くが、loss/hist は
    // 同じ累積を行うため、preds/targets/n_pos が同一なら loss と hist は一致する)。
    let preds = vec![0.1_f32, 0.4, 0.6, 0.95];
    let targets = vec![0.0_f32, 0.5, 0.5, 1.0];
    let n_pos = 4;

    let mut eval_loss = 0.0_f64;
    let mut eval_hist = [0_u64; 8];
    eval_cpu(&preds, &targets, &mut eval_loss, &mut eval_hist, n_pos);

    // grad_cpu は indices / per_pos_norm / grad buffer を要求する。indices は全 padding (-1)
    // にすれば scatter は no-op になる。per_pos_norm は loss に影響しない。
    let indices = vec![-1_i32; n_pos]; // max_inds = 1, 全 padding
    let per_pos_norm = vec![1.0_f32; n_pos];
    let mut grad_buf = vec![0.0_f32; 1]; // 1 weight (使われない)
    let mut grad_loss = 0.0_f64;
    let mut grad_hist = [0_u64; 8];
    grad_cpu(
        &indices,
        &preds,
        &targets,
        &per_pos_norm,
        &mut grad_buf,
        &mut grad_loss,
        &mut grad_hist,
        n_pos,
        1,
    );

    assert!(
        (eval_loss - grad_loss).abs() < 1e-12,
        "eval_loss = {} vs grad_loss = {}",
        eval_loss,
        grad_loss
    );
    assert_eq!(eval_hist, grad_hist, "histograms must match exactly");
}
