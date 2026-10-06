//! Speed-up mode's split projections on one Mac's GPU: recorded lane kernels over a tile map (fp32 partials over an input-group range), the DeltaNet step over half the heads.
const std = @import("std");
const mtl = @import("metal");
const replay = @import("replay.zig");
const Run = replay.Run;
const Buf = replay.Buf;

/// TP: a recorded lane projection over this Mac's output tiles only, in one dispatch: a copy of the kernel whose tile index goes through `ranges` ({first tile, count} pairs), with `sk` K slices; `groups` ({first, count}) cuts the sum to those input groups and writes fp32 partials. The recorded SK and no group cut keep its sums.
pub fn laneTiles(r: *Run, role: []const u8, ins: []const Buf, y: Buf, ranges: []const [2]usize, sk: usize, groups: ?[2]usize) !void {
    if (r.skip & Run.class(role) != 0) return;
    if (r.lane_new) { // fz_lane over the same tiles and K slices: the same sums
        const s = try replay.dense.recorded(r, role);
        return replay.dense.project(r, .{ .n = s.n, .k = s.k, .sk = sk, .ranges = ranges, .groups = groups, .pf = r.lane_pf }, ins[0], .{ .wq = ins[2], .sbt = ins[3] }, ins[4], y);
    }
    var key: [96]u8 = undefined;
    const s = r.roles.get(try std.fmt.bufPrint(&key, "{s}|{d}", .{ role, r.rows })) orelse return error.NoSite;
    var h = std.hash.Wyhash.init(@intFromPtr(s.v));
    h.update(std.mem.sliceAsBytes(ranges));
    h.update(std.mem.asBytes(&sk));
    if (groups) |g| h.update(std.mem.asBytes(&g));
    var v = s.v.*;
    v.pipe = r.tp_lane.get(h.final()) orelse blk: {
        var text: []const u8 = try Run.variantText(r.arena, s.v);
        const a = r.arena;
        const sk_at = std.mem.indexOf(u8, text, "constexpr int SK = ") orelse return error.LanePatch;
        const sk_end = sk_at + (std.mem.indexOfScalar(u8, text[sk_at..], ';') orelse return error.LanePatch);
        text = try std.fmt.allocPrint(a, "{s}constexpr int SK = {d}{s}", .{ text[0..sk_at], sk, text[sk_end..] });
        text = try swap(a, text, "threadgroup_position_in_grid.x", "tp_tile", 2);
        text = try swap(a, text, "const int n0 = ", "const int tp_tile = tp_map(int(threadgroup_position_in_grid.x));\n  const int n0 = ", 1);
        var map: std.ArrayList(u8) = .empty;
        try map.appendSlice(a, "inline int tp_map(int j) {\n");
        var at: usize = 0;
        for (ranges) |c| {
            try map.print(a, "  if (j < {d}) return {d} + j - {d};\n", .{ at + c[1], c[0], at });
            at += c[1];
        }
        try map.appendSlice(a, "  return 0;\n}\n[[kernel]]");
        text = try swap(a, text, "[[kernel]]", map.items, 1);
        if (groups) |g| {
            text = try swap(a, text, "const int g_begin = (sg * KG) / SK;", try std.fmt.allocPrint(a, "const int g_begin = {d} + (sg * {d}) / SK;", .{ g[0], g[1] }), 1);
            text = try swap(a, text, "const int g_end = ((sg + 1) * KG) / SK;", try std.fmt.allocPrint(a, "const int g_end = {d} + ((sg + 1) * {d}) / SK;", .{ g[0], g[1] }), 1);
            text = try swap(a, text, "device bfloat16_t* Y [[buffer(5)]]", "device float* Y [[buffer(5)]]", 1);
            text = try swap(a, text, "static_cast<bfloat>(C[t][f * 8 + r * 4 + j])", "C[t][f * 8 + r * 4 + j]", 1);
        }
        const lib = try mtl.Library.fromSource(r.device, text, mtl.CompileOptions.mlx());
        const pipe = try mtl.Pipeline.init(r.device, lib, try std.fmt.allocPrintSentinel(a, "{s}", .{s.v.name}, 0), false);
        try r.tp_lane.put(a, h.final(), pipe);
        break :blk pipe;
    };
    try r.bindV(&v, ins, &.{y});
    var tiles: usize = 0;
    for (ranges) |c| tiles += c[1];
    r.enc.dispatchThreads(mtl.Size.of(tiles * 32 * sk, s.grid.height, s.grid.depth), mtl.Size.of(32 * sk, 1, 1));
    if (!r.serial) r.enc.barrier();
}

fn swap(a: std.mem.Allocator, text: []const u8, from: []const u8, to: []const u8, count: usize) ![]const u8 {
    if (std.mem.count(u8, text, from) != count) return error.LanePatch;
    return std.mem.replaceOwned(u8, a, text, from, to);
}

/// TP: the recorded DeltaNet step over value heads [24 rank, 24 rank + 24) only (rank 1 runs a copy of the kernel whose head index starts at 24); every other index the kernel derives from the head, at the full layout.
pub fn gdnHeads(r: *Run, role: []const u8, as_rows: usize, ins: []const Buf, outs: []const Buf, rank: u32) !void {
    if (r.skip & Run.class(role) != 0) return;
    if (r.gdn_pipe) |pipe| return replay.gdn_step.step(r, pipe, ins, outs, 24 * rank, 24);
    var key: [96]u8 = undefined;
    const s = r.roles.get(try std.fmt.bufPrint(&key, "{s}|{d}", .{ role, as_rows })) orelse return error.NoSite;
    var v = s.v.*;
    if (rank == 1) v.pipe = r.tp_gdn.get(s.v) orelse blk: {
        const from = "const int hv = int(threadgroup_position_in_grid.x);";
        const text = try Run.variantText(r.arena, s.v);
        if (std.mem.count(u8, text, from) != 1) return error.GdnPatch;
        const patched = try std.mem.replaceOwned(u8, r.arena, text, from, "const int hv = int(threadgroup_position_in_grid.x) + 24;");
        const lib = try mtl.Library.fromSource(r.device, patched, mtl.CompileOptions.mlx());
        const pipe = try mtl.Pipeline.init(r.device, lib, try std.fmt.allocPrintSentinel(r.arena, "{s}", .{s.v.name}, 0), false);
        try r.tp_gdn.put(r.arena, s.v, pipe);
        break :blk pipe;
    };
    try r.bindV(&v, ins, outs);
    r.enc.dispatchThreads(mtl.Size.of(s.grid.width / 2, s.grid.height, s.grid.depth), s.tg);
    if (!r.serial) r.enc.barrier();
}
