//! NVFP4 experts' prompt tiles with the decode form's sums (tf_nvfp4_fact_*); experts.zig re-exports it.
const launch_ = @import("launch.zig");
const Stream = @import("stream.zig").Stream;
const Plan = @import("grouped.zig").Plan;
const experts = @import("experts.zig");
const Nvfp4Experts = experts.Nvfp4Experts;
const Epi = experts.Epi;
const Rows = experts.Rows;
const Layer = experts.Layer;

/// SwiGLU gate-up (.up) or fp32 down (.down_f32) on promptTile items: run's bits at every row, weights read once.
pub fn runFact(e: *const Nvfp4Experts, s: Stream, epi: Epi, in: Rows, l: Layer, p: Plan, out: u64, items: usize, skip: i32) !void {
    const gate = epi == .up;
    if (epi == .down_bf16 or (gate and l.relu2)) return error.EpilogueNotBuilt;
    const nb = (if (gate) l.width else l.dims) / Nvfp4Experts.cols;
    if (items < 1) return;
    var a: launch_.Args = .{};
    a.add(in.x);
    a.add(@as(i32, @intCast(in.stride)));
    a.add(@as(i32, @intCast(if (gate) in.slots else 0)));
    a.add(if (gate) l.up else l.down);
    a.add(if (gate) l.up_scale else l.down_scale);
    a.add(@as(i32, @intCast((if (gate) l.dims else l.width) / 32)));
    a.add(@as(i32, @intCast(nb)));
    for ([_]u64{ p.items, p.counts, p.members, out }) |v| a.add(v);
    a.add(@as(i32, @intCast(if (gate) l.width else l.dims)));
    a.add(if (gate) l.limit else @as(f32, 0.0));
    a.add(skip);
    const fi: usize = if (gate) 0 else 1;
    const wn = Nvfp4Experts.fact_wn[fi];
    const cfg: launch_.Config = .{ .grid = .{ .x = @intCast(items * ((nb + wn - 1) / wn)), .y = 1, .z = 1 }, .block = .{ .x = @intCast(wn * 32), .y = 1, .z = 1 }, .shared = Nvfp4Experts.promptSmem(if (gate) 2 else 1, wn) };
    try launch_.launch(e.fact_fns[fi], cfg, s, &a);
}
