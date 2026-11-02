//! GPU テストが同時に確保する device memory を見積り (GiB) の予算で制限する。
//! libtest は既定で CPU 数ぶんの thread でテストを並列に走らせ、全テストの trainer
//! buffer が同じ primary context 上に積み上がるため、制限しないと 12 GB GPU でも
//! `CUDA_ERROR_OUT_OF_MEMORY` で落ちるテストが出る。

use std::{
    cell::Cell,
    ops::Deref,
    sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock, PoisonError},
};

use gpu_runtime::CudaContext;

/// 同時に予約できる見積りの合計 (GiB)。12 GB GPU で表示用 process と CUDA context
/// の分を残す値。VRAM の小さい GPU では `TATARA_TEST_GPU_BUDGET_GIB` で下げる。
const DEFAULT_BUDGET_GIB: usize = 8;
const BUDGET_ENV: &str = "TATARA_TEST_GPU_BUDGET_GIB";

/// 予約は ticket 順に通す。空きだけで判定すると、大きい予約が小さい予約の
/// 後続に追い越され続けて最後まで待たされ、スイートの所要時間が延びる。
struct BudgetState {
    reserved_gib: usize,
    next_ticket: u64,
    now_serving: u64,
}

static STATE: Mutex<BudgetState> = Mutex::new(BudgetState {
    reserved_gib: 0,
    next_ticket: 0,
    now_serving: 0,
});
static RELEASED: Condvar = Condvar::new();

thread_local! {
    /// 同じテスト thread 内で helper が context を作り直しても二重に予約しない
    /// (二重予約は自分の予約が解放されるのを待って deadlock し得る)。
    static HELD_GIB: Cell<usize> = const { Cell::new(0) };
}

fn budget_gib() -> usize {
    static BUDGET: OnceLock<usize> = OnceLock::new();
    *BUDGET.get_or_init(|| {
        std::env::var(BUDGET_ENV)
            .ok()
            .and_then(|value| value.parse().ok())
            .filter(|&gib| gib > 0)
            .unwrap_or(DEFAULT_BUDGET_GIB)
    })
}

fn lock_state() -> MutexGuard<'static, BudgetState> {
    STATE.lock().unwrap_or_else(PoisonError::into_inner)
}

struct Reservation {
    gib: usize,
}

impl Reservation {
    fn acquire(requested_gib: usize) -> Self {
        let held = HELD_GIB.get();
        if held > 0 {
            assert!(
                requested_gib <= held,
                "nested GPU test context requests {requested_gib} GiB but the test reserved only {held} GiB"
            );
            return Self { gib: 0 };
        }
        // 予算より大きい見積りは予算いっぱいで単独実行させる (永久に待たせない)。
        let gib = requested_gib.clamp(1, budget_gib());
        let mut state = lock_state();
        let ticket = state.next_ticket;
        state.next_ticket += 1;
        while state.now_serving != ticket || state.reserved_gib + gib > budget_gib() {
            state = RELEASED.wait(state).unwrap_or_else(PoisonError::into_inner);
        }
        state.now_serving += 1;
        state.reserved_gib += gib;
        drop(state);
        RELEASED.notify_all();
        HELD_GIB.set(gib);
        Self { gib }
    }
}

impl Drop for Reservation {
    fn drop(&mut self) {
        if self.gib == 0 {
            return;
        }
        HELD_GIB.set(0);
        lock_state().reserved_gib -= self.gib;
        RELEASED.notify_all();
    }
}

/// 予約を保持したままの CUDA context。テスト内で後から作る trainer / buffer は
/// この値より先に drop されるため、予約は確保した memory の解放後に返る。
pub(crate) struct TestCudaContext {
    // field は宣言順に drop される: context を先に手放してから予約を返す。
    context: Arc<CudaContext>,
    _reservation: Reservation,
}

impl Deref for TestCudaContext {
    type Target = Arc<CudaContext>;

    fn deref(&self) -> &Self::Target {
        &self.context
    }
}

/// peak 1 GiB 以下のテスト用。
pub(crate) fn test_cuda_context() -> Result<TestCudaContext, Box<dyn std::error::Error>> {
    test_cuda_context_reserving_gib(1)
}

/// 単独実行時の device memory peak (切り上げ GiB) を予約してから context を作る。
pub(crate) fn test_cuda_context_reserving_gib(
    gib: usize,
) -> Result<TestCudaContext, Box<dyn std::error::Error>> {
    let reservation = Reservation::acquire(gib);
    Ok(TestCudaContext {
        context: CudaContext::new(0)?,
        _reservation: reservation,
    })
}
