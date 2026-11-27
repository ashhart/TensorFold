//! BF16 prompt-tile loading and selection beside the lane matmul implementation.
const q = @import("qmmf.zig");
const Driver = @import("driver.zig").Driver;
const Function = @import("module.zig").Function;
const abi = @import("abi.zig");

/// The BF16 prompt tile on the current device if it is sm_12x; only families that timed their shapes call this.
pub fn loadPrompt(l: *q.Lane, d: *const Driver) !void {
    if ((try d.currentCapability()) / 10 != 12) return;
    var prompt: [2][2]Function = undefined;
    inline for (.{ q.Out.bf16, q.Out.f32 }, 0..) |out, oi| inline for (.{ false, true }, 0..) |fused, fi| {
        const f = try l.mod.function(q.promptSymbol(out, fused));
        try f.allowDynamicShared(q.prompt_shared);
        prompt[oi][fi] = f;
    };
    var k32: [2]Function = undefined;
    inline for (.{ q.Out.bf16, q.Out.f32 }, 0..) |out, oi| {
        const f = try l.mod.function(q.promptK32Symbol(out));
        try f.allowDynamicShared(q.prompt_k32_shared);
        k32[oi] = f;
    }
    var dev: abi.Device = 0;
    try d.check(d.api.cuCtxGetDevice(&dev), "cuCtxGetDevice");
    var sms: c_int = 0;
    try d.check(d.api.cuDeviceGetAttribute(&sms, .multiprocessor_count, dev), "cuDeviceGetAttribute");
    l.prompt = prompt;
    l.prompt_k32 = k32;
    l.prompt_sms = @intCast(@max(sms, 1));
}

/// The actual prompt selection: loaded tiles, the pinned slice plan, and blocks over half the SMs.
pub fn usesPrompt(l: q.Lane, m: usize, w: q.Weight) bool {
    const p = q.planFor(m, w, l.clusters);
    return q.promptEligible(m, w) and !p.cluster and !p.reduce and l.prompt != null and
        2 * q.promptBlocks(m, w.n) > l.prompt_sms;
}
