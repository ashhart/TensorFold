//! Score MTP chains and level-1 runners-up at every position against a plain greedy reply.

const std = @import("std");
const cuda = @import("cuda");
const nemotron = @import("nemotron");
const check = @import("check.zig");

const depth = 8; // levels a branch drafts
const branches = 4; // the level-1 pick and its three runners-up
const candidates = 28; // the greedy draft's top-k candidate list (sampler.count(20, n))

/// At a 16k window, score COUNT greedy tokens after IDS_FILE; OUT records the reply and every branch.
pub fn run(gpu: check.Gpu, model: []const u8, ids_path: []const u8, count_text: []const u8, out_path: []const u8) !void {
    const gpa = gpu.gpa;
    const io = gpu.io;
    const text = try std.Io.Dir.cwd().readFileAlloc(io, ids_path, gpa, .limited(64 << 20));
    defer gpa.free(text);
    var prompt: std.ArrayList(u32) = .empty;
    defer prompt.deinit(gpa);
    var it = std.mem.tokenizeAny(u8, text, ", \n");
    while (it.next()) |t| try prompt.append(gpa, try std.fmt.parseInt(u32, t, 10));
    const count = try std.fmt.parseInt(usize, count_text, 10);
    const e = try nemotron.Engine.init(gpa, io, gpu.ctx, model, null, .{ .context = 16384, .mtp = true, .graphs = true, .sampling = null, .segments = 1 });
    defer e.deinit();
    const h = try nemotron.Head.init(e);
    defer h.deinit();
    try h.capture();
    const D: u64 = e.c.hidden;
    var level1 = try cuda.DeviceBuffer.alloc(e.ctx.d, D * 2);
    defer level1.free();
    var truth: std.ArrayList(u32) = .empty;
    defer truth.deinit(gpa);
    var drafts: std.ArrayList(u32) = .empty; // per position: branches x depth draft ids, the pick's chain first
    defer drafts.deinit(gpa);
    var pending = try e.prefill(prompt.items, null, h);
    try truth.append(gpa, pending);
    const last = (prompt.items.len - 1) % nemotron.state.prefill_rows;
    try e.ops().copy(e.b.hidden, e.b.p_hidden + last * D * 2, D * 2);
    try e.ops().fill32(e.b.sampled, pending, 1);
    var vals: [candidates]f32 = undefined;
    var wide: [candidates]i64 = undefined; // the id lookup writes int64, as torch indexes
    var cand: [candidates]u32 = undefined;
    for (0..count) |_| {
        try h.begin(1);
        try h.launch(1);
        try e.ops().copy(level1.ptr, h.out, D * 2);
        try e.ops().download(std.mem.sliceAsBytes(&vals), h.vals);
        try e.ops().download(std.mem.sliceAsBytes(&wide), h.cand);
        for (2..depth + 1) |j| try h.launch(j);
        try e.stream.synchronize();
        for (&cand, wide) |*c, w| c.* = @intCast(w);
        const base = drafts.items.len;
        try drafts.appendSlice(gpa, h.drafts()[0..depth]);
        const order = ranked(&vals, &cand);
        var b: usize = 1;
        var k: usize = 0;
        while (b < branches and k < candidates) : (k += 1) {
            const c = cand[order[k]];
            if (taken(drafts.items[base..], depth, c)) continue;
            try e.ops().copy(h.out, level1.ptr, D * 2);
            try e.ops().fill32(e.b.ids + 4, c, 1);
            for (2..depth + 1) |j| try h.launch(j);
            try e.stream.synchronize();
            try drafts.append(gpa, c);
            try drafts.appendSlice(gpa, h.drafts()[1..depth]);
            b += 1;
        }
        while (b < branches) : (b += 1) try drafts.appendNTimes(gpa, std.math.maxInt(u32), depth);
        pending = try e.step(pending, null);
        try truth.append(gpa, pending);
    }
    var out: std.Io.Writer.Allocating = .init(gpa);
    defer out.deinit();
    try out.writer.print("{{\"depth\": {d}, \"branches\": {d}, \"truth\": [", .{ depth, branches });
    for (truth.items, 0..) |t, i| try out.writer.print("{s}{d}", .{ if (i > 0) "," else "", t });
    try out.writer.writeAll("], \"drafts\": [");
    for (drafts.items, 0..) |t, i| try out.writer.print("{s}{d}", .{ if (i > 0) "," else "", if (t == std.math.maxInt(u32)) -1 else @as(i64, t) });
    try out.writer.writeAll("]}\n");
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = out_path, .data = out.written() });
    std.debug.print("RESULT tree-accept: {d} positions, {d} branches of {d} drafts each, written to {s}\n", .{ count, branches, depth, out_path });
}

/// Candidate indices by value, highest first, the lower id first on a tie (the keyed greedy draft's order).
fn ranked(vals: *const [candidates]f32, cand: *const [candidates]u32) [candidates]usize {
    var order: [candidates]usize = undefined;
    for (&order, 0..) |*o, i| o.* = i;
    const Ctx = struct { v: *const [candidates]f32, c: *const [candidates]u32 };
    std.mem.sort(usize, &order, Ctx{ .v = vals, .c = cand }, struct {
        fn lt(x: Ctx, a: usize, b: usize) bool {
            return x.v[a] > x.v[b] or (x.v[a] == x.v[b] and x.c[a] < x.c[b]);
        }
    }.lt);
    return order;
}

/// True when a branch already recorded at this position (the pick's chain first) starts with `c`.
fn taken(rows: []const u32, d: usize, c: u32) bool {
    var i: usize = 0;
    while (i + d <= rows.len) : (i += d) if (rows[i] == c) return true;
    return false;
}
