//! Two GLM prompt allocations on one NVFP4 grouped plan, with 128 pairs an item.

const experts = @import("experts.zig");
const layout = @import("experts_prompt_layout.zig");
const Plan = @import("grouped.zig").Plan;
const launch_ = @import("launch.zig");
const Stream = @import("stream.zig").Stream;

/// Member IDs below `split` select A, the rest select B; each pair keeps its original arithmetic.
pub fn run(e: *const experts.Nvfp4Experts, s: Stream, epi: experts.Epi, a: experts.Rows, b: experts.Rows, l: experts.Layer, p: Plan, out_a: u64, out_b: u64, split: usize, items: usize, skip: i32) !void {
    if (epi == .down_f32 or (epi == .up and l.relu2)) return error.EpilogueNotBuilt;
    const gate = epi == .up;
    const parts: usize = if (gate) 2 else 1;
    const nb = (if (gate) l.width else l.dims) / experts.Nvfp4Experts.cols;
    if (items < 1) return;
    const fi: usize = @intFromBool(!gate);
    const wn = experts.Nvfp4Experts.pair_prompt_wn[fi];
    var args: launch_.Args = .{};
    args.add(a.x);
    args.add(@as(i32, @intCast(a.stride)));
    args.add(@as(i32, @intCast(if (gate) a.slots else 0)));
    args.add(if (gate) l.up else l.down);
    args.add(if (gate) l.up_scale else l.down_scale);
    args.add(@as(i32, @intCast((if (gate) l.dims else l.width) / 32)));
    args.add(@as(i32, @intCast(nb)));
    for ([_]u64{ p.items, p.counts, p.members, out_a }) |v| args.add(v);
    args.add(@as(i32, @intCast(if (gate) l.width else l.dims)));
    args.add(if (gate) l.limit else @as(f32, 0.0));
    args.add(skip);
    args.add(b.x);
    args.add(out_b);
    args.add(@as(i32, @intCast(split)));
    const cfg: launch_.Config = .{ .grid = .{ .x = @intCast(items * ((nb + wn - 1) / wn)), .y = 1, .z = 1 }, .block = .{ .x = @intCast(2 * wn * 32), .y = 1, .z = 1 }, .shared = layout.pairPromptSmem(parts, wn) };
    try launch_.launch(e.pair_fns[fi], cfg, s, &args);
}
