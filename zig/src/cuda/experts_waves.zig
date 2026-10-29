//! NVFP4 experts' decode form in equal waves, or two warps a unit when every half unit fits the grid: run's bits.
const std = @import("std");
const launch_ = @import("launch.zig");
const Stream = @import("stream.zig").Stream;
const Plan = @import("grouped.zig").Plan;
const experts = @import("experts.zig");
const Nvfp4Experts = experts.Nvfp4Experts;
const Epi = experts.Epi;
const Rows = experts.Rows;
const Layer = experts.Layer;

/// nvfp4_expert_waves<2, 2, 4> (SwiGLU gate-up), <1, 0, 4> (fp32 down), nvfp4_expert_half<2, 2, 4>: fns 4, 5, 6.
pub const symbols = [3][:0]const u8{
    "_ZN16tf_nvfp4_experts18nvfp4_expert_wavesILi2ELi2ELi4EEEvPK13__nv_bfloat16iiPK5uint4PKfiiPKiSA_SA_Pvifi",
    "_ZN16tf_nvfp4_experts18nvfp4_expert_wavesILi1ELi0ELi4EEEvPK13__nv_bfloat16iiPK5uint4PKfiiPKiSA_SA_Pvifi",
    "_ZN16tf_nvfp4_experts17nvfp4_expert_halfILi2ELi2ELi4EEEvPK13__nv_bfloat16iiPK5uint4PKfiiPKiSA_SA_Pvifi",
};
const first = 4;

/// The entry: half while its units fit `half` warps, waves past one wave of the classic kernel's `classic`, else null.
pub fn pick(gate: bool, items: usize, nb: usize, half: usize, classic: usize) ?usize {
    if (gate and 2 * items * nb <= half) return first + 2;
    if (items * nb <= classic) return null;
    return first + @as(usize, if (gate) 0 else 1);
}

/// Nvfp4Experts.run's outputs, bit for bit: SwiGLU gate-up and fp32 down here, other epilogues through run.
pub fn run(e: *const Nvfp4Experts, s: Stream, epi: Epi, in: Rows, l: Layer, p: Plan, out: u64, items: usize, skip: i32) !void {
    const gate = epi == .up;
    if ((gate and l.relu2) or (!gate and epi != .down_f32)) return e.run(s, epi, in, l, p, out, items, skip);
    const nb = (if (gate) l.width else l.dims) / Nvfp4Experts.cols;
    const classic = e.resident[if (gate) 0 else 1] * e.warps;
    const fi = pick(gate, items, nb, e.resident[first + 2] * e.warps, classic) orelse return e.run(s, epi, in, l, p, out, items, skip);
    return entry(e, fi, s, in, l, p, out, items, skip);
}

/// Entry `fi` (4 waves gate-up, 5 waves down, 6 half gate-up) on a 16-pair plan of at most `items` items.
pub fn entry(e: *const Nvfp4Experts, fi: usize, s: Stream, in: Rows, l: Layer, p: Plan, out: u64, items: usize, skip: i32) !void {
    const gate = fi != first + 1;
    const kg = (if (gate) l.dims else l.width) / 32;
    const nb = (if (gate) l.width else l.dims) / Nvfp4Experts.cols;
    const units = items * nb * @as(usize, if (fi == first + 2) 2 else 1);
    const grid = @min((units + e.warps - 1) / e.warps, e.resident[fi]);
    if (grid < 1) return;
    var a: launch_.Args = .{};
    a.add(in.x);
    a.add(@as(i32, @intCast(in.stride)));
    a.add(@as(i32, @intCast(if (gate) in.slots else 0)));
    a.add(if (gate) l.up else l.down);
    a.add(if (gate) l.up_scale else l.down_scale);
    a.add(@as(i32, @intCast(kg)));
    a.add(@as(i32, @intCast(nb)));
    for ([_]u64{ p.items, p.counts, p.members, out }) |v| a.add(v);
    a.add(@as(i32, @intCast(if (gate) l.width else l.dims)));
    a.add(if (gate) l.limit else @as(f32, 0.0));
    a.add(skip);
    try launch_.launch(e.fns[fi], .{ .grid = .{ .x = @intCast(grid), .y = 1, .z = 1 }, .block = .{ .x = @intCast(e.warps * 32), .y = 1, .z = 1 } }, s, &a);
}

test "two warps a unit only while every half unit fits the resident warps" {
    // Flash Next gate-up on GB10: 20 column blocks, 768 resident warps; one row's 11 items fit, two rows' 23 do not
    try std.testing.expectEqual(@as(?usize, 6), pick(true, 19, 20, 768, 768));
    try std.testing.expectEqual(@as(?usize, 4), pick(true, 46, 20, 768, 768));
    try std.testing.expectEqual(@as(?usize, 5), pick(false, 11, 80, 768, 768));
    // one wave and no half fit: the classic kernel (rows 2 and 3 of Flash Next: 23 and 35 items of 20 blocks)
    try std.testing.expectEqual(@as(?usize, null), pick(true, 23, 20, 768, 768));
    try std.testing.expectEqual(@as(?usize, null), pick(true, 35, 20, 768, 768));
    try std.testing.expectEqual(@as(?usize, null), pick(false, 1, 1, 768, 768));
}

test "the one-wave test uses the replaced classic kernel's resident warps, not the half kernel's" {
    // 1 warp: four-row gate-up's 780 units fit half's ~1,000 warps but not classic's 768, so waves, not classic
    try std.testing.expectEqual(@as(?usize, 4), pick(true, 39, 20, 1000, 768));
    // 4 warps: one-row down's 880 units pass half's 768 warps but fit one wave of a classic down holding 960
    try std.testing.expectEqual(@as(?usize, null), pick(false, 11, 80, 768, 960));
    // the half test reads half's warps alone: 760 half units fit 768, 800 do not and one classic wave holds them
    try std.testing.expectEqual(@as(?usize, 6), pick(true, 19, 20, 768, 960));
    try std.testing.expectEqual(@as(?usize, null), pick(true, 20, 20, 768, 960));
}
