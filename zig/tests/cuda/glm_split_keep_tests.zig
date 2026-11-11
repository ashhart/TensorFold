//! TF_GLM_KV_SPLIT's kept states and twins on GPUs: twinned, resumed at home and carried passes against fresh twins.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const lanes_tests = @import("glm_lanes_tests.zig");
const Gpu = check.Gpu;

const capacity = 32_832; // a window of 32,768 and a verify window, as cuda_native.capacity makes it
const len = 9000; // past the dense pass and four chunks
const mark = 4096; // a kept state at a chunk start
const paged = 1; // the stream under test, on pages of an owner class
const whole = 3; // past the paged sequences: a whole window, as without TF_KV_PAGES, every row on every rank

/// `glm-kv-split-keep-check`: TF_GLM_KV_SPLIT=1's twinned pass, a kept state resumed at home and carried both ways.
pub fn keepCheck(gpu: Gpu, checkpoint: []const u8, rank: c_int, world: c_int, id_path: []const u8) !void {
    const glm = @import("glm_cuda");
    var arena = std.heap.ArenaAllocator.init(gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var lib = try cuda.nccl.Library.open();
    defer lib.close();
    const comm = try lanes_tests.openComm(gpu, &lib, rank, world, id_path);
    defer _ = lib.api.ncclCommDestroy(comm);
    var problem: []const u8 = "";
    var set = glm.ranks.Settings.read(&glm.ranks.environment, &problem) catch |err| {
        std.debug.print("REFUSE {s}\n", .{problem});
        return err;
    };
    set.pages = true;
    set.kv_split = try glm.start.splitting(set.kv_split, @intCast(world), true, set.kv_host != 0);
    if (!set.kv_split) {
        std.debug.print("REFUSE glm-kv-split-keep-check needs TF_GLM_KV_SPLIT=1 on every rank\n", .{});
        return error.TestFailed;
    }
    const e = try glm.engine.Engine.init(gpu.gpa, gpu.io, gpu.d, gpu.ctx, checkpoint, &lib, comm, @intCast(rank), @intCast(world), .{ .capacity = capacity, .mtp = true, .settings = set });
    defer e.deinit();
    var link = try glm.follow.Link.init(gpu.gpa, gpu.d, &lib, comm, e.stream, capacity);
    defer link.deinit();
    if (rank != 0) {
        try glm.follow.follow(e, &link);
        check.pass("rank {d} kept, read and carried rank 0's states through every step", .{rank});
        return;
    }
    const x: glm.follow.Leader = .{ .e = e, .link = &link };
    defer x.end() catch {};
    const P = glm.pages;
    const l = try P.layout(gpu.gpa, e, 0);
    const count = l.classes * (2 * P.tableLen(capacity) + 2); // two windows a class: a stream and its twin
    try x.pool(2, count, l.pageBytes());
    e.pages = try P.Pages.init(gpu.gpa, e, l, 2, count, true);
    const pages = e.pages.?;
    pages.poison = true; // a row read before its write, or carried to the wrong place, turns the reply to NaN
    try x.snapshots(2 << 30);
    const V = e.share.vocab.len();
    const ids = try a.alloc(u32, len);
    for (ids, 0..) |*u, i| u.* = @intCast(100 + (i * 7919) % 5000);
    // the reference: the prompt from 0 on a whole window, its first token's logits and four windows
    var ref: Run = .{ .logits = try zeros(a, 5 * 4 * V) };
    try x.bind(whole);
    try x.begin(null);
    try ref.go(x, e, ids, null, null, V);
    try x.release(whole);
    var failed = false;
    var kept: [2]u64 = undefined; // the state at `mark`, kept on a stream of owner 0 and of owner 1
    const paths = [_]struct { name: []const u8, owner: u32, twin: bool, from: ?u32, mark: bool = true }{
        .{ .name = "control: owner-computed on owner 0, nothing kept", .owner = 0, .twin = false, .from = null, .mark = false },
        .{ .name = "twinned, kept on owner 1", .owner = 1, .twin = true, .from = null },
        .{ .name = "owner-computed, kept on owner 0", .owner = 0, .twin = false, .from = null },
        .{ .name = "resumed at home 1", .owner = 1, .twin = false, .from = 1 },
        .{ .name = "resumed off home 1 (carried to 0)", .owner = 0, .twin = false, .from = 1 },
        .{ .name = "resumed off home 0 (carried to 1)", .owner = 1, .twin = false, .from = 0 },
    };
    for (paths) |path| {
        try x.bind(paged);
        const held = if (path.twin) try pages.openTwinned(paged, len, 64, path.owner) else try pages.openOn(paged, len, 64, path.owner);
        if (path.twin and !glm.pages_twin.twinned(pages, paged)) {
            std.debug.print("FAIL keep: the {s} pass found no twin pages\n", .{path.name});
            failed = true;
        }
        try x.beginOn(null, held, 0);
        var keeper: Keeper = .{ .e = e };
        var run: Run = .{ .logits = try zeros(a, 5 * 4 * V) };
        const from: ?glm.engine.Resume = if (path.from) |h| .{ .id = kept[h], .at = mark } else null;
        try run.go(x, e, ids, from, if (from == null and path.mark) keeper.marks() else null, V);
        if (from == null and path.mark) kept[path.owner] = keeper.id orelse {
            std.debug.print("FAIL keep: the {s} pass kept no state at {d}\n", .{ path.name, mark });
            return error.TestFailed;
        };
        if (from == null and path.mark and e.homeOf(kept[path.owner]) != path.owner) {
            std.debug.print("FAIL keep: the {s} pass's state is not homed on its owner\n", .{path.name});
            failed = true;
        }
        if (run.first != ref.first or !std.mem.eql(u16, run.logits, ref.logits) or !std.mem.eql(u32, &run.outs, &ref.outs)) {
            std.debug.print("FAIL keep: the {s} pass's tokens or logits differ from a fresh whole window's\n", .{path.name});
            report(&run, &ref, V);
            failed = true;
        } else std.debug.print("  {s}: first token, 4 windows and their logits equal a fresh whole window's\n", .{path.name});
        try x.release(paged);
    }
    const c = pages.pool.census();
    if (!pages.pool.idle() or c.free != c.total) {
        std.debug.print("FAIL keep: the pool is not idle after every stream ended (free {d} of {d})\n", .{ c.free, c.total });
        failed = true;
    }
    if (failed) return error.TestFailed;
    check.pass("kv split keeps at TP={d}: a twinned and an owner-computed pass, a state kept on each owner, resumed at home and carried each way, equal a fresh whole window byte for byte; the pool idle after", .{world});
}

/// One pass: the prompt (resumed or from 0), its first token's logits, then four windows of 1 to 4 rows.
const Run = struct {
    first: u32 = 0,
    outs: [10]u32 = @splat(0),
    logits: []u16,

    fn go(r: *Run, x: anytype, e: anytype, ids: []const u32, from: anytype, marks: anytype, V: usize) !void {
        r.first = try x.prefill(ids, from, null, marks); // a resume restores on every rank before its first chunk
        try cuda.memory.download(e.d, e.bp.logits, std.mem.sliceAsBytes(r.logits[0..V]));
        var at: usize = 0;
        for (1..5) |rows| {
            var w: [4]u32 = undefined;
            for (w[0..rows], 0..) |*u, k| u.* = @intCast(100 + (rows * 31 + k * 13) % 5000);
            try x.verify(w[0..rows], r.outs[at..][0..rows], 0);
            try cuda.memory.download(e.d, e.bd.logits, std.mem.sliceAsBytes(r.logits[rows * 4 * V ..][0 .. rows * V]));
            try x.keep(rows);
            at += rows;
        }
    }
};

/// Rank 0's mark at `mark`: the state kept there on every rank (its gate takes it to the others).
const Keeper = struct {
    e: *@import("glm_cuda").engine.Engine,
    id: ?u64 = null,
    const at = [_]u32{mark};

    fn marks(k: *Keeper) @import("glm_cuda").engine.Marks {
        return .{ .at = &at, .ctx = k, .call = call };
    }
    fn call(ctx: *anyopaque, _: u32) void {
        const k: *Keeper = @ptrCast(@alignCast(ctx));
        k.id = k.e.keepState() catch null;
    }
};

/// Where a pass parts from the reference: its first token and logits, then each window's logits and tokens.
fn report(run: *const Run, ref: *const Run, V: usize) void {
    const f = run.logits[0..V];
    std.debug.print("    first token {d} (reference {d}); its logits: {d} of {d} differ, {d} NaN\n", .{ run.first, ref.first, differ(f, ref.logits[0..V]), V, nans(f) });
    var at: usize = 0;
    for (1..5) |rows| {
        const w = run.logits[rows * 4 * V ..][0 .. rows * V];
        std.debug.print("    window of {d}: logits {d} of {d} differ, {d} NaN; tokens {any} (reference {any})\n", .{ rows, differ(w, ref.logits[rows * 4 * V ..][0 .. rows * V]), rows * V, nans(w), run.outs[at..][0..rows], ref.outs[at..][0..rows] });
        at += rows;
    }
}

/// The passes' logits: windows fill only their own rows, so the rest must start equal.
fn zeros(a: std.mem.Allocator, n: usize) ![]u16 {
    const l = try a.alloc(u16, n);
    @memset(l, 0);
    return l;
}

fn differ(a: []const u16, b: []const u16) usize {
    var n: usize = 0;
    for (a, b) |x, y| n += @intFromBool(x != y);
    return n;
}

/// bf16 NaNs: what a poisoned row read before its write turns a logit to.
fn nans(a: []const u16) usize {
    var n: usize = 0;
    for (a) |x| n += @intFromBool(x & 0x7f80 == 0x7f80 and x & 0x7f != 0);
    return n;
}
