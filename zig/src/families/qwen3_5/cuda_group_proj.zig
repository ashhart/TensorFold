//! TF_Q35_GROUP_PROJ=1: same-input ModelOpt projections in one lane launch, each with its own launch's bits (opt-in).
const cuda = @import("cuda");
const platform = @import("sys");
const engine = @import("cuda_engine.zig");

const Engine = engine.Engine;
const Proj = engine.Proj;
const group = cuda.qmmf_group;

/// The switch, read once when the ModelOpt kernels load; default off.
pub fn on() !bool {
    return platform.environment.read(bool, "TF_Q35_GROUP_PROJ", false);
}

/// The grouped kernels when the switch is on and slices meet in clusters (sm_90 on), else null.
pub fn load(lane: *const cuda.qmmf.Lane) !?group.Group {
    if (!lane.clusters or !try on()) return null;
    return try group.Group.load(lane);
}

/// The lane weight a ModelOpt projection decodes through (NVFP4, FP8, BF16), else null.
fn laneWeight(p: Proj) ?cuda.qmmf.Weight {
    if (p != .ql) return null;
    return switch (p.ql) {
        .nvfp4, .fp8, .bf16 => |q| q,
        else => null,
    };
}

/// `ls` over x into `outs` as compatible grouped launches, or false: the caller runs separate launches.
pub fn run(e: *Engine, ls: []const Proj, x: u64, outs: []const u64, rows: usize, prompt: bool) !bool {
    if (@import("cuda_launch_trace.zig").active != null) return false; // the trace keeps counting separate launches
    const k = if (e.mo) |*m| m else return false;
    const g = k.group orelse return false;
    if (prompt and rows > 16) return false; // modelopt.project's prompt GEMM
    if (ls.len > group.most) return false;
    var ws: [group.most]cuda.qmmf.Weight = undefined;
    for (ls, ws[0..ls.len]) |l, *w| w.* = laneWeight(l) orelse return false;
    for (ws[0..ls.len]) |w| if (w.k != ws[0].k) return false;
    for (outs, 0..) |out, i| for (outs[i + 1 ..]) |other| if (out == other) return false;
    var shapes: [2]group.Shape = undefined;
    var count: usize = 0;
    var at: usize = 0;
    while (at < ls.len) {
        var end = at + 1;
        while (end < ls.len and ws[end].mode == ws[at].mode) : (end += 1) {}
        shapes[count] = group.plan(k.lane, ws[at..end], outs[at..end], rows) orelse return false;
        count += 1;
        at = end;
    }
    for (shapes[0..count]) |sh| try g.launch(e.stream, x, rows, ws[0].k, sh, k.lane.pdl);
    return true;
}
