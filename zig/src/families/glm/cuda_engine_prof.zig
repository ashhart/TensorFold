//! GLM profiler lifetime operation used by unwind and close.
const Engine = @import("cuda_engine.zig").Engine;
pub fn freeProfs(e: *Engine) void {
    for (&e.profs) |*p| {
        if (p.*) |v| v.deinit();
        p.* = null;
    }
    for (&e.prompt_profs) |*p| {
        if (p.*) |v| v.deinit();
        p.* = null;
    }
}
