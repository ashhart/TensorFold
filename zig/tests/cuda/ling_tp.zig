//! `ling-tp DIR OUT [--layers a,b] [--steps N] [--seed S]`: a prompt, then N greedy rows at TP=1 or TP=2.

// TP=2 by TF_LING_RANK, TF_LING_WORLD and NCCL_COMM_ID; rank 1 runs tp.follow (the serving loop) on rank 0's steps.

// OUT (rank 0): "TFLTP001", u32 world, vocab, rows (the prompt's last row, then N), u32 tokens[rows], bf16 logits.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const ling = @import("ling_cuda");

const engine = ling.engine;

/// The clock step() times drafts by (run sets it).
var clock: std.Io = undefined;

pub fn run(gpu: check.Gpu, rest: []const [:0]const u8) !void {
    clock = gpu.io;
    if (rest.len < 2) return error.MissingArgument;
    var arena = std.heap.ArenaAllocator.init(gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var layers: ?[]u32 = null;
    var steps: usize = 64;
    var seed: u64 = 0;
    var cap: usize = 512;
    var P: usize = 40;
    var resume_at: ?u32 = null;
    var win: usize = 1; // --window W: each greedy step verifies W rows (the token, then W - 1 made-up drafts), keeps 1
    var k: usize = 2;
    while (k < rest.len) : (k += 1) {
        if (std.mem.eql(u8, rest[k], "--layers") and k + 1 < rest.len) {
            k += 1;
            var list: std.ArrayList(u32) = .empty;
            var it = std.mem.splitScalar(u8, rest[k], ',');
            while (it.next()) |x| try list.append(a, try std.fmt.parseInt(u32, x, 10));
            layers = list.items;
        } else if (std.mem.eql(u8, rest[k], "--seed") and k + 1 < rest.len) {
            k += 1;
            seed = try std.fmt.parseInt(u64, rest[k], 10);
        } else if (std.mem.eql(u8, rest[k], "--steps") and k + 1 < rest.len) {
            k += 1;
            steps = try std.fmt.parseInt(usize, rest[k], 10);
        } else if (std.mem.eql(u8, rest[k], "--prompt") and k + 1 < rest.len) {
            k += 1;
            P = try std.fmt.parseInt(usize, rest[k], 10);
        } else if (std.mem.eql(u8, rest[k], "--capacity") and k + 1 < rest.len) {
            k += 1;
            cap = try std.fmt.parseInt(usize, rest[k], 10);
        } else if (std.mem.eql(u8, rest[k], "--resume") and k + 1 < rest.len) {
            k += 1;
            resume_at = try std.fmt.parseInt(u32, rest[k], 10);
        } else if (std.mem.eql(u8, rest[k], "--window") and k + 1 < rest.len) {
            k += 1;
            win = try std.fmt.parseInt(usize, rest[k], 10);
        } else return error.BadArgument;
    }
    const p = try ling.tp.place(ling.tp.env);
    var group: ?cuda.tp_group.Group = null;
    if (p.world > 1) {
        try cuda.tp_watch.start(gpu.io, p.rank, p.world);
        group = try ling.tp.group(gpu.io, p);
    }
    defer if (group) |*g| g.close();
    const e = try engine.Engine.init(gpu.gpa, gpu.io, gpu.d, gpu.ctx, rest[0], .{ .capacity = cap, .layers = layers, .group = if (group) |*g| g else null });
    defer e.deinit();
    var link: ?ling.tp.Link = null;
    defer if (link) |*l| l.deinit();
    if (group) |*g| {
        try ling.tp.roce(g, gpu.gpa, gpu.io, gpu.d, e.stream);
        link = try ling.tp.Link.init(gpu.gpa, gpu.d, &g.lib, g.comm, e.stream, cap);
        e.link = &link.?;
        e.lead = p.rank == 0;
        var dg: [32]u8 = @splat(0);
        std.crypto.hash.sha2.Sha256.hash(rest[0], &dg, .{});
        try cuda.tp_watch.ready(gpu.io, dg);
        if (p.rank != 0) {
            try ling.tp.follow(e, gpu.io);
            check.pass("ling-tp rank {d}: followed rank 0's steps to its end", .{p.rank});
            return;
        }
        e.asleep = cuda.tp_watch.open();
    }
    const V: usize = e.c.vocab;
    const ids = try a.alloc(u32, P);
    var rng = std.Random.DefaultPrng.init(0x6c696e67 +% seed);
    for (ids) |*t| t.* = rng.random().uintLessThan(u32, @intCast(@min(V, 30000)));
    const rows = steps + 1;
    const logits = try a.alloc(u16, rows * V);
    const toks = try a.alloc(u32, rows);
    var keep: Keep = .{ .e = e };
    const at = [_]u32{resume_at orelse 0};
    const marks: ?engine.Marks = if (resume_at != null) .{ .at = &at, .ctx = &keep, .call = Keep.call } else null;
    var all = std.crypto.hash.sha2.Sha256.init(.{});
    try greedy(e, ids, null, marks, toks, logits, win, &all);
    if (keep.err) |err| return err;
    var same: usize = rows;
    if (resume_at) |m| { // resumed == fresh: the same prompt again from the state kept at m, every row's bytes compared
        const toks2 = try a.alloc(u32, rows);
        const logits2 = try a.alloc(u16, rows * V);
        var all2 = std.crypto.hash.sha2.Sha256.init(.{});
        try greedy(e, ids, .{ .id = keep.id orelse return error.NoKeptState, .at = m }, null, toks2, logits2, win, &all2);
        same = 0;
        for (0..rows) |r| same += @intFromBool(toks[r] == toks2[r] and std.mem.eql(u16, logits[r * V ..][0..V], logits2[r * V ..][0..V]));
        std.debug.print("RESULT ling-tp resumed at {d} == fresh: {d}/{d} rows byte-equal\n", .{ m, same, rows });
    }
    var dg: [32]u8 = undefined;
    all.final(&dg);
    std.debug.print("RESULT ling-tp windows of {d}: every row's logits and draws sha256 {x}\n", .{ win, dg[0..8] });
    cuda.tp_watch.ended(); // rank 1 leaves on the end step: its line closing from here is expected
    try e.end();
    var buf: std.ArrayList(u8) = .empty;
    try buf.appendSlice(a, "TFLTP001");
    for ([_]u32{ p.world, @intCast(V), @intCast(rows) }) |v| try buf.appendSlice(a, std.mem.asBytes(&v));
    try buf.appendSlice(a, std.mem.sliceAsBytes(toks));
    try buf.appendSlice(a, std.mem.sliceAsBytes(logits));
    try std.Io.Dir.cwd().writeFile(gpu.io, .{ .sub_path = rest[1], .data = buf.items });
    std.debug.print("RESULT ling-tp world {d}: {d} prompt ids, {d} greedy rows; tokens {any}\n", .{ p.world, P, steps, toks[0..@min(rows, 16)] });
    if (same != rows) return error.ResumedDiffers;
    check.pass("ling-tp world {d}: rows saved in {s}", .{ p.world, rest[1] });
}

/// The pass's mark: the state kept there (an error kept for after the pass).
const Keep = struct {
    e: *engine.Engine,
    id: ?u64 = null,
    err: ?anyerror = null,

    fn call(ctx: *anyopaque, at: u32) void {
        const kp: *Keep = @ptrCast(@alignCast(ctx));
        _ = at;
        kp.id = kp.e.keepState() catch |err| {
            kp.err = err;
            return;
        };
    }
};

/// A fresh (or resumed) sequence: prompt, then greedy windows of `win` rows; every row's logits into `all`, 1 kept.
fn greedy(e: *engine.Engine, ids: []const u32, from: ?engine.Resume, marks: ?engine.Marks, toks: []u32, logits: []u16, win: usize, all: *std.crypto.hash.sha2.Sha256) !void {
    const V: usize = e.c.vocab;
    try e.bind(0);
    try e.begin(null);
    const t0 = check.now(clock);
    toks[0] = try e.prefill(ids, from, null, marks);
    try e.stream.synchronize();
    std.debug.print("RESULT ling-tp prompt: {d} ids from {d} in {d:.1} ms\n", .{ ids.len, if (from) |f| f.at else 0, @as(f64, @floatFromInt(check.now(clock) - t0)) / 1e6 });
    try e.bp.arena.buf.download(e.bp.logits - e.bp.arena.buf.ptr, std.mem.sliceAsBytes(logits[0..V]));
    const row = try std.heap.page_allocator.alloc(u16, 16 * V);
    defer std.heap.page_allocator.free(row);
    for (1..toks.len) |s| {
        var out: [16]u32 = undefined;
        const w = try step(e, toks[s - 1], win, &out);
        const n = @min(w * V, row.len);
        try e.bd.arena.buf.download(e.bd.logits - e.bd.arena.buf.ptr, std.mem.sliceAsBytes(row[0..n]));
        @memcpy(logits[s * V ..][0..V], row[0..V]);
        all.update(std.mem.sliceAsBytes(row[0..n]));
        all.update(std.mem.sliceAsBytes(out[0..w]));
        toks[s] = out[0];
    }
}

/// One verify of `win` rows from `tok` (made-up drafts after it), the first kept; rows verified.
fn step(e: *engine.Engine, tok: u32, win: usize, out: *[16]u32) !usize {
    const w = @min(@max(win, 1), 16);
    var ids: [16]u32 = undefined;
    ids[0] = tok;
    for (1..w) |j| ids[j] = @intCast((@as(u64, tok) * 31 + j * 7919) % @min(e.c.vocab, 30000));
    try e.verify(ids[0..w], out[0..w]);
    try e.keep(1);
    return w;
}
