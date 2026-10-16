//! Flash Next's verdict kernels under thinking budgets on a fake target: plain and drafted against the lane core.
const std = @import("std");
const mtl = @import("metal");
const lanes = @import("lanes");
const fz = @import("replay.zig");
const verdict = @import("verdict.zig");
const think = @import("think.zig");

const EOS = 999;
const END = 901;
const close = [_]u32{ 900, END, 902 }; // "\n", the think end, "\n\n"
const HIST = 8192;

/// The fake target's next token after `seq`: a hash of the whole prefix; now and then a newline, the think end or EOS.
fn target(seq: []const u32) u32 {
    const x = std.hash.Wyhash.hash(7, std.mem.sliceAsBytes(seq));
    return switch (x % 128) {
        0 => END,
        1, 2, 3 => close[0],
        4 => EOS,
        else => @intCast((x >> 8) % 40),
    };
}

/// A wrong draft, sometimes one of the close's tokens.
fn stray(rng: std.Random) u32 {
    const specials = [_]u32{ close[0], END, close[2], EOS };
    const k = rng.uintLessThan(u32, 48);
    return if (k < 40) k else specials[k % specials.len];
}

/// The lane core's reply, one token a step (stream.zig: the budget's cut, its forced close, the think end before it).
fn reference(gpa: std.mem.Allocator, prompt: []const u32, max: u32, budget: u32) ![]u32 {
    var s = try lanes.Stream.init(gpa, .{ .id = "ref", .prompt = prompt, .max_new = max, .eos = &.{EOS}, .think_budget = budget, .think_close = &close, .think_end = END });
    defer s.deinit(gpa);
    while (!s.finished) {
        const t = s.popForce() orelse if (s.cutsNext()) try s.startClose(gpa) else target(s.context.items);
        _ = try s.commit(gpa, &.{t}, &.{});
    }
    return gpa.dupe(u32, s.emitted());
}

/// fz_accept, fz_force and fz_span, and the buffers one reply's rounds use.
pub const Rig = struct {
    lib: mtl.Library,
    q: mtl.Queue,
    accept: mtl.Pipeline,
    force: mtl.Pipeline,
    span: mtl.Pipeline,
    bufs: [6]mtl.Buffer, // window ids, picks, arena, ring, n-gram history, token history

    pub fn init(dev: mtl.Device) !Rig {
        const lib = try mtl.Library.fromSource(dev, verdict.source, mtl.CompileOptions.mlx());
        var bufs: [6]mtl.Buffer = undefined;
        for (&bufs, [_]usize{ 64, 64, 4 * fz.AR_WORDS, 4 * fz.RING_WORDS * 512, 8 * 39, 4 * HIST }) |*b, n| b.* = try dev.buffer(n, fz.opts);
        return .{ .lib = lib, .q = try dev.queue(), .accept = try mtl.Pipeline.init(dev, lib, "fz_accept", false), .force = try mtl.Pipeline.init(dev, lib, "fz_force", false), .span = try mtl.Pipeline.init(dev, lib, "fz_span", false), .bufs = bufs };
    }

    pub fn deinit(g: *Rig) void {
        for (g.bufs) |b| b.deinit();
        g.accept.deinit();
        g.force.deinit();
        g.span.deinit();
        g.q.deinit();
        g.lib.deinit();
    }

    /// One kernel on one thread with buffers at 0.. and `bytes` at `at`, waited for.
    fn once(g: *Rig, pipe: mtl.Pipeline, bufs: []const mtl.Buffer, bytes: []const u8, at: usize, last: ?mtl.Buffer) !void {
        const cb = g.q.commandBuffer();
        const enc = cb.compute(.serial);
        enc.setPipeline(pipe);
        for (bufs, 0..) |b, i| enc.setBuffer(b, 0, i);
        enc.setBytes(bytes, at);
        if (last) |b| enc.setBuffer(b, 0, at + 1);
        enc.dispatchThreads(mtl.Size.of(1, 1, 1), mtl.Size.of(1, 1, 1));
        enc.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |msg| {
            std.log.err("command buffer failed: {s}", .{msg});
            return error.GpuFailed;
        }
    }
};

/// One reply through the kernels: windows of 1..16 rows when `wide` (drafts right three times in four), else 1.
fn rounds(gpa: std.mem.Allocator, g: *Rig, rng: std.Random, prompt: []const u32, max: u32, budget: u32, wide: bool) ![]u32 {
    var seq: std.ArrayList(u32) = .empty; // the prompt and every landed token
    defer seq.deinit(gpa);
    var reply: std.ArrayList(u32) = .empty;
    errdefer reply.deinit(gpa);
    var watch = try think.Watch.init(.{ .budget = budget, .close = &close, .end = END });
    const pick = watch.first(target(prompt));
    try seq.appendSlice(gpa, prompt);
    try seq.append(gpa, pick);
    try reply.append(gpa, pick);
    if (pick == EOS or max <= 1) return reply.toOwnedSlice(gpa);
    const wids = g.bufs[0].slice(u32, 16);
    const picks = g.bufs[1].slice(u32, 16);
    const ar = g.bufs[2].slice(i32, fz.AR_WORDS);
    const ring = g.bufs[3].slice(u32, fz.RING_WORDS * 512);
    const hist = g.bufs[5].slice(u32, HIST);
    @memset(ar, 0);
    ar[1] = @intCast(prompt.len);
    ar[fz.AR_HLEN] = @intCast(seq.items.len);
    @memcpy(hist[0..seq.items.len], seq.items);
    watch.arm(ar, prompt.len);
    wids[0] = pick;
    var round: usize = 0;
    while (true) : (round += 1) {
        const w: usize = if (wide) rng.intRangeAtMost(usize, 1, 16) else 1;
        const at = seq.items.len;
        for (1..w) |j| {
            wids[j] = if (rng.uintLessThan(u32, 4) != 0) target(seq.items) else stray(rng);
            try seq.append(gpa, wids[j]);
        }
        seq.shrinkRetainingCapacity(at);
        const span = watch.armed or rng.boolean(); // the engine's view lags the GPU's: span kernels past its end are inert
        const c = [2]u32{ @intCast(w), think.AR_SPAN };
        if (span) try g.once(g.force, &.{ g.bufs[0], g.bufs[2], g.bufs[5] }, std.mem.asBytes(&c), 3, null);
        for (0..w) |i| { // the target's token after each row, its input wids[i]
            picks[i] = target(seq.items);
            if (i + 1 < w) try seq.append(gpa, wids[i + 1]);
        }
        seq.shrinkRetainingCapacity(at);
        if (span) try g.once(g.span, &.{ g.bufs[1], g.bufs[2] }, std.mem.asBytes(&c), 2, null);
        const cfg = [4]u32{ @intCast(w), 16, HIST, 0 };
        try g.once(g.accept, g.bufs[0..5], std.mem.asBytes(&cfg), 5, g.bufs[5]);
        const slot = (round % 512) * fz.RING_WORDS;
        const got = ring[slot + 3 ..][0..ring[slot]];
        _ = watch.read(got);
        try seq.appendSlice(gpa, got);
        if (wids[0] != got[got.len - 1] or ar[1] + 1 != @as(i64, @intCast(seq.items.len)) or ar[fz.AR_HLEN] != @as(i64, @intCast(seq.items.len))) return error.ArenaOutOfStep;
        for (got) |t| {
            try reply.append(gpa, t);
            if (t != EOS and reply.items.len < max) continue;
            if (!std.mem.eql(u32, hist[0..seq.items.len], seq.items)) return error.HistoryOutOfStep;
            return reply.toOwnedSlice(gpa);
        }
    }
}

/// Whether the budget closed `reply`: no think end before its last token, then the close from there on.
fn closedAt(reply: []const u32, budget: u32) bool {
    if (budget == 0 or reply.len < budget) return false;
    if (std.mem.indexOfScalar(u32, reply[0 .. budget - 1], END) != null) return false;
    const n = @min(close.len, reply.len - (budget - 1));
    return std.mem.eql(u32, reply[budget - 1 ..][0..n], close[0..n]);
}

pub const Summary = struct { cases: usize = 0, closed: usize = 0, ended_first: usize = 0 };

/// `cases` random replies, each plain and drafted, against the lane core's; error.Mismatch at the first that differs.
pub fn run(gpa: std.mem.Allocator, g: *Rig, cases: usize, seed: u64) !Summary {
    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();
    var sum: Summary = .{};
    const budgets = [_]u32{ 0, 1, 2, 3, 5, 9, 17, 40, 100, 300 };
    const maxes = [_]u32{ 1, 2, 8, 60, 200, 400 };
    for (0..cases) |c| {
        var buf: [24]u32 = undefined;
        const prompt = buf[0..rng.intRangeAtMost(usize, 2, buf.len)];
        for (prompt) |*t| t.* = rng.uintLessThan(u32, 40);
        const budget = budgets[rng.uintLessThan(usize, budgets.len)];
        const max = maxes[rng.uintLessThan(usize, maxes.len)];
        const want = try reference(gpa, prompt, max, budget);
        defer gpa.free(want);
        const thought = want[0..@min(want.len, budget -| 1)];
        const ended = std.mem.indexOfScalar(u32, thought, END) != null or std.mem.indexOfScalar(u32, thought, EOS) != null;
        if (budget > 0 and want.len >= budget and !ended and !closedAt(want, budget)) return error.NotClosed;
        for ([_]bool{ false, true }) |wide| {
            const got = try rounds(gpa, g, rng, prompt, max, budget, wide);
            defer gpa.free(got);
            if (std.mem.eql(u32, want, got)) continue;
            std.log.err("case {d} (budget {d}, max {d}, {s}): lane core {any}, kernels {any}", .{ c, budget, max, if (wide) "drafted" else "plain", want, got });
            return error.Mismatch;
        }
        sum.cases += 1;
        if (closedAt(want, budget)) sum.closed += 1;
        if (budget > 0 and std.mem.indexOfScalar(u32, thought, END) != null) sum.ended_first += 1;
    }
    return sum;
}

test "the verdict kernels close the thinking at the budget, plain and drafted, as the lane core does" {
    const dev = mtl.Device.init() catch return error.SkipZigTest;
    defer dev.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    var g = try Rig.init(dev);
    defer g.deinit();
    const sum = try run(std.testing.allocator, &g, 80, 11);
    try std.testing.expect(sum.closed >= 10 and sum.ended_first >= 1);
}
