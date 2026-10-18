//! The common EXL3 prompt path, after @BobClawblaw's #976 design and MiaAI-Lab's transposed kernels.
const std = @import("std");
const ex = @import("exl3_experts.zig");
const launch = @import("launch.zig");
const grouped = @import("grouped.zig");
const Module = @import("module.zig").Module;
const Function = @import("module.zig").Function;
const Stream = @import("stream.zig").Stream;

pub const tile = 64;
pub const Scratch = struct { xg: u64, xu: u64, xd: u64, y: u64, rows: usize, slots: usize };
pub fn eligible(l: ex.Layer) bool {
    return l.codebook == .mcg and l.uniform_k2 == 8 and l.gu_stride == 0 and l.dims > 0 and l.width > 0 and l.dims % 256 == 0 and l.width % 128 == 0 and (l.act_mode == 0 or l.act_mode == 1);
}
fn mul(a: usize, b: usize) !usize {
    return std.math.mul(usize, a, b) catch error.Exl3Shape;
}
fn add(a: usize, b: usize) !usize {
    return std.math.add(usize, a, b) catch error.Exl3Shape;
}
fn aligned(n: usize) !usize {
    return (try add(n, 255)) / 256 * 256;
}
fn sizes(l: ex.Layer, rows: usize, slots: usize) ![4]usize {
    if (!eligible(l) or rows == 0 or slots == 0 or slots > 32 or l.experts == 0 or l.experts > grouped.max_experts) return error.Exl3Shape;
    const pairs = try mul(rows, slots);
    if (pairs > std.math.maxInt(i32) or l.dims > std.math.maxInt(i32) or l.width > std.math.maxInt(i32)) return error.Exl3Shape;
    const rotated = try aligned(try mul(try mul(if (l.shared_input) rows else pairs, l.dims), 2));
    return .{ rotated, if (l.shared_input) 0 else rotated, try aligned(try mul(try mul(pairs, l.width), 2)), try aligned(try mul(try mul(pairs, l.dims), 4)) };
}
pub fn bytes(l: ex.Layer, rows: usize, slots: usize) !usize {
    const ns = try sizes(l, rows, slots);
    var sum: usize = 0;
    for (ns) |n| sum = try add(sum, n);
    return sum;
}
pub fn scratch(base: u64, l: ex.Layer, rows: usize, slots: usize) !Scratch {
    const ns = try sizes(l, rows, slots);
    const total = try bytes(l, rows, slots);
    _ = std.math.add(u64, base, total) catch return error.Exl3Shape;
    return .{ .xg = base, .xu = if (l.shared_input) base else base + ns[0], .xd = base + ns[0] + ns[1], .y = base + ns[0] + ns[1] + ns[2], .rows = rows, .slots = slots };
}
fn go(f: Function, s: Stream, grid: [3]usize, threads: u32, values: anytype) !void {
    var args: launch.Args = .{};
    inline for (values) |v| args.add(v);
    try launch.launch(f, .{ .grid = .{ .x = @intCast(grid[0]), .y = @intCast(grid[1]), .z = @intCast(grid[2]) }, .block = .{ .x = threads } }, s, &args);
}
fn i(n: usize) i32 {
    return @intCast(n);
}
pub const Kernels = struct {
    rotate: Function,
    gateup: Function,
    down: Function,
    pub fn resolve(mod: Module) !Kernels {
        return .{ .rotate = try mod.function("tf_exl3_prompt_rotate"), .gateup = try mod.function("tf_exl3_prompt_gu"), .down = try mod.function("tf_exl3_prompt_down") };
    }
    pub fn run(k: Kernels, e: *const ex.Experts, router: grouped.Router, s: Stream, l: ex.Layer, x: u64, stride: usize, picks: u64, rows: usize, sc: Scratch, p: grouped.Plan) !void {
        if (!eligible(l) or rows == 0 or rows > sc.rows or stride < l.dims or stride > std.math.maxInt(i32)) return error.Exl3Shape;
        _ = try bytes(l, rows, sc.slots);
        const pairs = rows * sc.slots;
        try router.route(s, picks, pairs, l.experts, tile, p);
        if (l.shared_input) {
            try go(k.rotate, s, .{ rows, l.dims / 128, 1 }, 32, .{ x, i(stride), l.suh_g, sc.xg, i(l.dims) });
        } else {
            var args: launch.Args = .{};
            inline for (.{ x, i(stride), @as(i32, @backingInt(ex.DType.bf16)), picks, l.suh_g, l.suh_u, sc.xg, sc.xu, i(l.dims), i(sc.slots), i(l.experts) }) |v| args.add(v);
            try launch.launch(e.rot_in, .{ .grid = .{ .x = @intCast(pairs), .y = @intCast(l.dims / 128), .z = 2 }, .block = .{ .x = 32 } }, s, &args);
        }
        const items = grouped.maxItems(pairs, l.experts, tile);
        try go(k.gateup, s, .{ l.width / 128, items, 1 }, 256, .{ sc.xg, sc.xu, l.gate, l.up, p.items, p.counts, p.members, l.svh_g, l.svh_u, l.suh_d, sc.xd, i(l.dims), i(l.width), i(if (l.shared_input) sc.slots else 1), l.limit, l.act_mode });
        try go(k.down, s, .{ l.dims / 256, items, 1 }, 256, .{ sc.xd, l.down, p.items, p.counts, p.members, l.svh_d, sc.y, i(l.width), i(l.dims), i(sc.slots) });
    }
};
