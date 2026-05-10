//! Adam step kernel の reference CPU 実装に対する smoke test。
//!
//! 手計算可能な小さな入力で `adam_step_cpu` の m / v / weights / grad の
//! in-place 更新が期待値と一致することを確認する。GPU 実機 (cuda-oxide PTX)
//! との bit-equivalent 検証は Stage 1-9 (#13) で host loop が組まれた段階
//! で別途追加する。

use exp_001_cuda_oxide_kpabs::kernels::adam_step::adam_step_cpu;

const BETA1: f32 = 0.9;
const BETA2: f32 = 0.999;
const EPS: f32 = 1e-8;

/// 1 step 目に固定した bc1 / bc2 (`1 - beta^1 = 1 - beta`)。
fn bc_step1() -> (f32, f32) {
    (1.0 - BETA1, 1.0 - BETA2)
}

#[test]
fn first_step_zero_state_matches_hand_calculation() {
    // n = 1, m = v = 0, grad = 0.1, weight = 1.0, lr = 0.01
    // step 1 (bc1 = 0.1, bc2 = 0.001):
    //   mi = 0.9 * 0 + 0.1 * 0.1 = 0.01
    //   vi = 0.999 * 0 + 0.001 * 0.01 = 1e-5
    //   m_hat = 0.01 / max(0.1, 1e-30) = 0.01 / 0.1 = 0.1
    //   v_hat = 1e-5 / max(0.001, 1e-30) = 1e-5 / 0.001 = 0.01
    //   weights[0] -= 0.01 * 0.1 / (sqrt(0.01) + 1e-8) = 0.01 * 0.1 / (0.1 + 1e-8) ≈ 0.01
    //   grad[0] = 0
    let mut weights = vec![1.0_f32];
    let mut m = vec![0.0_f32];
    let mut v = vec![0.0_f32];
    let mut grad = vec![0.1_f32];
    let (bc1, bc2) = bc_step1();
    adam_step_cpu(
        &mut weights,
        &mut m,
        &mut v,
        &mut grad,
        0.01,
        BETA1,
        BETA2,
        EPS,
        bc1,
        bc2,
        1,
    );

    // f32 精度: BETA1/BETA2 が exact float でないため expected を f32 計算で組む。
    let mi_expected = (1.0_f32 - BETA1) * 0.1_f32;
    let vi_expected = (1.0_f32 - BETA2) * 0.1_f32 * 0.1_f32;
    assert!((m[0] - mi_expected).abs() < 1e-7, "m[0] = {}", m[0]);
    assert!((v[0] - vi_expected).abs() < 1e-9, "v[0] = {}", v[0]);
    // weights ≈ 1.0 - 0.01 (within ~1e-6 due to eps in denominator)
    assert!(
        (weights[0] - (1.0 - 0.01)).abs() < 1e-5,
        "weights[0] = {} expected ≈ 0.99",
        weights[0]
    );
    assert_eq!(grad[0], 0.0, "grad must be reset to 0");
}

#[test]
fn zero_grad_only_decays_moments() {
    // grad = 0 → mi = 0.9 * m, vi = 0.999 * v, weight 更新は 0/(sqrt(...)+eps)≈0 だが
    // m_hat / v_hat の分母 bc が小さいので weight が動く可能性に注意。
    // 初期 m = v = 0 で grad = 0 なら mi = vi = 0 → weight 不変、grad reset のみ。
    let mut weights = vec![5.0_f32];
    let mut m = vec![0.0_f32];
    let mut v = vec![0.0_f32];
    let mut grad = vec![0.0_f32];
    let (bc1, bc2) = bc_step1();
    adam_step_cpu(
        &mut weights,
        &mut m,
        &mut v,
        &mut grad,
        0.01,
        BETA1,
        BETA2,
        EPS,
        bc1,
        bc2,
        1,
    );

    assert_eq!(m[0], 0.0, "m must remain 0");
    assert_eq!(v[0], 0.0, "v must remain 0");
    // m_hat = 0 / bc1 = 0, v_hat = 0, sqrt(0) = 0, lr * 0 / eps = 0 → weight 不変
    assert!(
        (weights[0] - 5.0).abs() < 1e-7,
        "weights[0] = {}",
        weights[0]
    );
    assert_eq!(grad[0], 0.0);
}

#[test]
fn nonzero_initial_moment_decays() {
    // m=0.5, v=0.25, grad=0 → mi = 0.45, vi = 0.249750
    // m_hat = 0.45/0.1 = 4.5, v_hat = 0.249750/0.001 ≈ 249.75
    // delta = lr * 4.5 / (sqrt(249.75) + eps) = lr * 4.5 / 15.8035... ≈ lr * 0.2848
    let mut weights = vec![10.0_f32];
    let mut m = vec![0.5_f32];
    let mut v = vec![0.25_f32];
    let mut grad = vec![0.0_f32];
    let lr = 0.01_f32;
    let (bc1, bc2) = bc_step1();
    adam_step_cpu(
        &mut weights,
        &mut m,
        &mut v,
        &mut grad,
        lr,
        BETA1,
        BETA2,
        EPS,
        bc1,
        bc2,
        1,
    );

    let mi_expected = BETA1 * 0.5;
    let vi_expected = BETA2 * 0.25;
    assert!((m[0] - mi_expected).abs() < 1e-6, "m[0] = {}", m[0]);
    assert!((v[0] - vi_expected).abs() < 1e-6, "v[0] = {}", v[0]);

    let m_hat = mi_expected / bc1;
    let v_hat = vi_expected / bc2;
    let expected_weight = 10.0 - lr * m_hat / (v_hat.sqrt() + EPS);
    assert!(
        (weights[0] - expected_weight).abs() < 1e-4,
        "weights[0] = {} expected {}",
        weights[0],
        expected_weight
    );
    assert_eq!(grad[0], 0.0);
}

#[test]
fn negative_grad_increases_weight() {
    // grad = -0.1 → mi = -0.01, weights は + 方向に更新
    let mut weights = vec![1.0_f32];
    let mut m = vec![0.0_f32];
    let mut v = vec![0.0_f32];
    let mut grad = vec![-0.1_f32];
    let (bc1, bc2) = bc_step1();
    adam_step_cpu(
        &mut weights,
        &mut m,
        &mut v,
        &mut grad,
        0.01,
        BETA1,
        BETA2,
        EPS,
        bc1,
        bc2,
        1,
    );
    assert!(
        weights[0] > 1.0,
        "weights should increase, got {}",
        weights[0]
    );
    assert!(
        (weights[0] - (1.0 + 0.01)).abs() < 1e-5,
        "weights[0] = {} expected ≈ 1.01",
        weights[0]
    );
}

#[test]
fn multi_weight_independent_updates() {
    // n = 4: 各 weight が異なる grad に対し独立に更新される
    let mut weights = vec![1.0_f32, 2.0, 3.0, 4.0];
    let mut m = vec![0.0_f32; 4];
    let mut v = vec![0.0_f32; 4];
    let mut grad = vec![0.1_f32, -0.1, 0.0, 0.5];
    let (bc1, bc2) = bc_step1();
    adam_step_cpu(
        &mut weights,
        &mut m,
        &mut v,
        &mut grad,
        0.01,
        BETA1,
        BETA2,
        EPS,
        bc1,
        bc2,
        4,
    );

    // grad=0.1 → weight 減少
    assert!(weights[0] < 1.0);
    // grad=-0.1 → weight 増加
    assert!(weights[1] > 2.0);
    // grad=0 で m=v=0 → weight 不変
    assert!((weights[2] - 3.0).abs() < 1e-7);
    // grad=0.5 → 大きく減少 (符号方向で正しいことのみ確認)
    assert!(weights[3] < 4.0);

    // grad は全部 reset
    for &g in &grad {
        assert_eq!(g, 0.0);
    }
}

#[test]
fn bias_correction_floor_prevents_divide_by_zero() {
    // bc1 = 0 / bc2 = 0 を渡した場合、1e-30 floor でクラッシュせずに結果が出る。
    // (実用上は host 側で bc = 1 - beta^t を必ず > 0 にするが、kernel 自身は floor で守る)
    let mut weights = vec![1.0_f32];
    let mut m = vec![0.01_f32];
    let mut v = vec![1e-5_f32];
    let mut grad = vec![0.0_f32]; // grad=0 で moment が劣化しないようにする
    adam_step_cpu(
        &mut weights,
        &mut m,
        &mut v,
        &mut grad,
        0.01,
        BETA1,
        BETA2,
        EPS,
        0.0,
        0.0,
        1,
    );
    // bc=0 でも 1e-30 floor → m_hat = 0.009 / 1e-30 = 9e27 という巨大な値になり
    // weight は壊れるが NaN/inf にはならず finite に収まることを確認 (無効化試験)
    assert!(
        weights[0].is_finite(),
        "weights must be finite even with bc=0"
    );
}

#[test]
fn ten_step_loop_converges_toward_zero_grad() {
    // 同じ grad = 0.1 を 10 step 連続適用、weight が単調減少することを確認。
    // step ごとに bc = 1 - beta^t を更新する点に注意。
    let mut weights = vec![1.0_f32];
    let mut m = vec![0.0_f32];
    let mut v = vec![0.0_f32];
    let lr = 0.01;

    let mut prev_w = weights[0];
    for t in 1..=10 {
        let mut grad = vec![0.1_f32];
        let bc1 = 1.0 - BETA1.powi(t);
        let bc2 = 1.0 - BETA2.powi(t);
        adam_step_cpu(
            &mut weights,
            &mut m,
            &mut v,
            &mut grad,
            lr,
            BETA1,
            BETA2,
            EPS,
            bc1,
            bc2,
            1,
        );
        assert!(
            weights[0] < prev_w,
            "step {t}: weight {} should be < prev {}",
            weights[0],
            prev_w
        );
        prev_w = weights[0];
    }
    // 10 step でおよそ 0.9 (1.0 - 10 * lr) 周辺になる (Adam は per-step ≈ lr の更新量)
    assert!(
        (weights[0] - 0.9).abs() < 0.02,
        "after 10 steps weights[0] = {}, expected near 0.9",
        weights[0]
    );
}
