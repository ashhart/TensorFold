//! Optional R12 resource setup, kept separate from the shared engine's load and forward contracts.
const cuda = @import("cuda");
const engine = @import("cuda_engine.zig");
const modelopt = @import("cuda_modelopt.zig");

pub fn requested() !bool {
    _ = try cuda.fp8_prefill.requested();
    if (try cuda.fp8_prefill_tiles.requested() != .baseline and !cuda.fp8_prefill.enabled()) return error.Fp8TileNeedsFastPrefill;
    if (try cuda.fp8_prefill.affine_packing.requested() and !cuda.fp8_prefill.enabled()) return error.AffinePreparedNeedsFastPrefill;
    const cublas_requested = try cuda.cublaslt_prefill.requested();
    _ = try @import("cuda_prefill_paths.zig").requested();
    return cublas_requested;
}

pub fn additionalModelOpt(e: *engine.Engine, capability: u32, sms: usize, cublas_requested: bool) !void {
    if (e.mo == null and ((try @import("cuda_nvfp4_mlp.zig").directory()) != null or cuda.fp8_prefill.enabled() or cublas_requested))
        e.mo = try modelopt.Kernels.load(e.d, capability, sms, e.g.moe());
}

pub fn tapState(e: *engine.Engine) void {
    e.rec_budget = .{};
    e.tap_capacity = engine.tap_rows;
    e.prompt_taps = .{};
}

pub fn preparePaths(e: *engine.Engine, capability: u32) !void {
    e.prefill_paths = try @import("cuda_prefill_paths.zig").Paths.load(e, capability);
}

pub fn freePaths(e: *engine.Engine) void {
    cuda.completion.drain(e.d, e.stream, "prefill initialization did not drain");
    e.prefill_paths.deinit();
}

pub fn shutdown(e: *engine.Engine) void {
    cuda.completion.drain(e.d, e.stream, "prefill shutdown did not drain");
    e.prompt_taps.deinit();
    e.prefill_paths.deinit();
}
