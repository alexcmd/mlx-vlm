"""The intermediate-states GDN kernel must index batch rows by StateT.

``_make_gated_delta_with_states_kernel`` writes per-step states into a
``(B, StateT, Hv, Dv, Dk)`` buffer. The row base address used to be computed
with the full sequence length ``T`` instead of ``StateT``; whenever
``StateT < T`` (a verify step that only keeps the states preceding the last
token) rows >= 1 were shifted by ``(T - StateT) * Hv * Dv * Dk`` elements and
the last row wrote past the end of the buffer. Rows read back garbage after a
speculative rollback (token id 0 loops) or the GPU faulted.
"""

import pytest

import mlx.core as mx


def _run_kernel(kernel, q, k, v, g, beta, state, state_steps, mask=None):
    B, T, Hk, Dk = k.shape
    Hv, Dv = v.shape[2:]
    inputs = [q, k, v, g, beta, state, T]
    if mask is not None:
        inputs.append(mask)
    return kernel(
        inputs=inputs,
        template=[
            ("InT", q.dtype),
            ("StT", state.dtype),
            ("Dk", Dk),
            ("Dv", Dv),
            ("Hk", Hk),
            ("Hv", Hv),
            ("StateT", state_steps),
        ],
        grid=(32, Dv, B * Hv),
        threadgroup=(32, 4, 1),
        output_shapes=[(B, T, Hv, Dv), state.shape, (B, state_steps, Hv, Dv, Dk)],
        output_dtypes=[q.dtype, state.dtype, state.dtype],
    )


@pytest.mark.skipif(not mx.metal.is_available(), reason="Metal kernel only")
@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("batch", [2, 3])
def test_states_rows_match_ops_path_when_state_steps_below_t(masked, batch):
    from mlx_vlm.models.qwen3_5 import gated_delta as gd

    mx.random.seed(0)
    B, T, Hk, Hv, Dk, Dv = batch, 4, 2, 4, 64, 64
    q = mx.random.normal((B, T, Hk, Dk)).astype(mx.bfloat16)
    k = mx.random.normal((B, T, Hk, Dk)).astype(mx.bfloat16)
    v = mx.random.normal((B, T, Hv, Dv)).astype(mx.bfloat16)
    g = mx.random.uniform(shape=(B, T, Hv)).astype(mx.float32)
    beta = mx.random.uniform(shape=(B, T, Hv)).astype(mx.float32)
    state = mx.random.normal((B, Hv, Dv, Dk)).astype(mx.float32)
    mask = (mx.random.uniform(shape=(B, T)) > 0.3) if masked else None

    kernel = gd._gated_delta_with_states_kernel_masked if masked else gd._gated_delta_with_states_kernel
    state_steps = T - 1
    y_k, final_k, states_k = _run_kernel(kernel, q, k, v, g, beta, state, state_steps, mask)
    y_r, final_r, states_r = gd._gated_delta_with_states_ops(q, k, v, g, beta, state, mask)
    mx.eval(y_k, final_k, states_k, y_r, final_r, states_r)

    assert float(mx.abs(final_k - final_r).max()) < 1e-2
    assert float(mx.abs(y_k.astype(mx.float32) - y_r.astype(mx.float32)).max()) < 5e-2
    # every batch row, not just row 0, must hold the first state_steps states
    for b in range(B):
        diff = float(mx.abs(states_k[b] - states_r[b, :state_steps]).max())
        assert diff < 1e-2, f"row {b}: max abs diff {diff}"
