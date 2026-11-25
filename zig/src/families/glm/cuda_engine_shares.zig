//! Shares on fewer ranks (full GLM-5.3 at TP=1): a process runs each share it holds as an in-process rank, same bits.
const std = @import("std");
const cuda = @import("cuda");
const cuda_pdl = @import("cuda_pdl.zig");
const fw = @import("cuda_forward.zig");
const weights = @import("cuda_weights.zig");
const shard = @import("cuda_shard.zig");
const sampler = @import("cuda_sampler.zig");
const chance = @import("cuda_mtp_chance.zig");
const config = @import("config.zig");
const Seq = @import("cuda_engine_types.zig").Seq;

/// The most shares one process holds (full GLM-5.3's three at TP=1).
pub const max_held = 3;
const pair_bytes = 16;

/// A share past the process's first: its weights and buffers (each sequence's state for it is in Seq.more).
pub const Local = struct { w: weights.Weights, bp: fw.Buffers, bd: fw.Buffers, bm: fw.Buffers };

/// Shares [first, first + count) of `total` that rank `rank` of `world` holds.
pub const Held = struct { first: usize, count: usize, total: usize };

/// Config.shares cut evenly over the ranks; 0 shares: one a rank (GLM-5.3-Flash, its sums at each world's own cut).
pub fn held(c: *const config.Config, rank: usize, world: usize) !Held {
    const total: usize = if (c.shares == 0) world else c.shares;
    if (world == 0 or total % world != 0 or total / world > max_held) return error.SharesNotOnRanks;
    const n = total / world;
    return .{ .first = rank * n, .count = n, .total = total };
}

/// The settings an in-process share cannot take yet: each keeps or pairs one sequence's rows of one rank.
pub fn refused(e: anytype, drafter: bool, vision_or_packs: bool) !void {
    if (e.settings.pages or drafter or vision_or_packs) return error.SharesOneRankEach;
}

/// The other shares' weights (`how`: the engine's tiler, checkpoint, layers, MTP) and buffers, in share order.
pub fn load(e: anytype, h: Held, how: anytype) !void {
    if (h.count <= 1) return;
    const out = try e.gpa.alloc(Local, h.count - 1);
    var made: usize = 0;
    errdefer {
        for (out[0..made]) |*l| deinitOne(l);
        e.gpa.free(out);
    }
    for (out, 1..) |*l, j| {
        const s = try shard.Share.of(&e.c, h.first + j, h.total);
        const w = try weights.load(e.gpa, e.io, e.d, e.stream, e.m.tools(), how.checkpoint, e.c, s, how.layers, how.mtp);
        l.w = try @import("cuda_load_owner.zig").finish(w, how.tiler, @TypeOf(how.tiler).apply);
        errdefer l.w.deinit();
        l.bp = try fw.Buffers.init(e.d, &l.w, e.bp.rows, true, e.capacity);
        errdefer l.bp.deinit();
        l.bd = try fw.Buffers.init(e.d, &l.w, fw.window_rows, false, e.capacity);
        errdefer l.bd.deinit();
        l.bm = try fw.Buffers.init(e.d, &l.w, fw.window_rows, false, e.capacity);
        made += 1;
    }
    e.local = out;
}

fn deinitOne(l: *Local) void {
    l.bm.deinit();
    l.bd.deinit();
    l.bp.deinit();
    l.w.deinit();
}

pub fn deinit(e: anytype) void {
    for (e.local) |*l| deinitOne(l);
    e.gpa.free(e.local);
    e.local = &.{};
}

/// A new sequence's state for each other share.
pub fn states(e: anytype) ![]fw.State {
    const out = try e.gpa.alloc(fw.State, e.local.len);
    var made: usize = 0;
    errdefer {
        for (out[0..made]) |*s| s.deinit();
        e.gpa.free(out);
    }
    for (out, e.local) |*s, *l| {
        s.* = try fw.State.initFormat(e.gpa, e.d, &l.w, e.capacity, e.settings.kv8);
        made += 1;
    }
    return out;
}

/// Each sequence made before the shares loaded gets its state for each of them.
pub fn fill(e: anytype) !void {
    for (e.seqs.items) |q| if (q) |x| if (x.more.len < e.local.len) {
        x.more = try states(e);
    };
}

/// A new sequence: its state on the engine's own share, and at TP=1 each other share's.
pub fn seq(e: anytype) !Seq {
    var st = try fw.State.initFormat(e.gpa, e.d, &e.w, e.capacity, e.settings.kv8);
    errdefer st.deinit();
    return .{ .st = st, .more = if (e.local.len > 0) try states(e) else &.{} };
}

/// Bytes sequence `q` takes on this rank: its state arena and each other share's.
pub fn seqBytes(q: *const Seq) u64 {
    var n: u64 = q.st.arena.buf.len;
    for (q.more) |*st| n += st.arena.buf.len;
    return n;
}

pub fn freeStates(gpa: std.mem.Allocator, more: []fw.State) void {
    for (more) |*s| s.deinit();
    gpa.free(more);
}

/// The other shares' buffers of the kind `b` is (the prompt's, a window's or the MTP head's).
fn peer(e: anytype, b: *const fw.Buffers, l: *Local) !*fw.Buffers {
    if (b == &e.bp) return &l.bp;
    if (b == &e.bd) return &l.bd;
    if (b == &e.bm) return &l.bm;
    return error.SharesOneRankEach; // a paired, shared or prompt-head pass's buffers
}

/// Every held share as a rank of the forward on `b`: the engine's own first, the others in share order.
pub fn ranks(e: anytype, b: *fw.Buffers) []const fw.Rank {
    e.rank_buf[0] = .{ .w = &e.w, .st = &e.cur.st, .b = b };
    for (e.local, e.cur.more, 1..) |*l, *st, j| e.rank_buf[j] = .{ .w = &l.w, .st = st, .b = peer(e, b, l) catch @panic("a pass on buffers only rank 0 holds") };
    return e.rank_buf[0 .. e.local.len + 1];
}

/// Every share's MTP row count (State.setMtpLen), the engine's own included.
pub fn setMtpLen(e: anytype, n: usize) !void {
    try e.cur.st.setMtpLen(e.stream, n);
    for (e.cur.more) |*st| try st.setMtpLen(e.stream, n);
}

/// Every share's state reset (State.reset).
pub fn reset(e: anytype) !void {
    try e.cur.st.reset(e.stream);
    for (e.cur.more) |*st| try st.reset(e.stream);
}

/// The logits rows at `logits` (in one of the engine's own buffers) as each held share's: row offsets by its vocab.
fn mine(e: anytype, logits: u64, out: *[max_held]sampler.Local) ![]const sampler.Local {
    out[0] = .{ .rank = e.share.rank, .logits = logits };
    const v0 = e.share.vocab.len() * 2;
    for (e.local, 1..) |*l, j| {
        const own, const other = for ([_]*const fw.Buffers{ &e.bp, &e.bd, &e.bm }, [_]*const fw.Buffers{ &l.bp, &l.bd, &l.bm }) |b, o| {
            const rows = if (b == &e.bp) @import("cuda_state.zig").prompt_heads else b.rows; // its logits rows alone
            if (logits >= b.logits and (logits - b.logits) % v0 == 0 and logits - b.logits < rows * v0) break .{ b, o };
        } else return error.SharesOneRankEach;
        out[j] = .{ .rank = l.w.share.rank, .logits = other.logits + (logits - own.logits) / v0 * l.w.share.vocab.len() * 2 };
    }
    return out[0 .. e.local.len + 1];
}

/// Engine.draw over the held shares: sampled through the sampler's local placement, greedy by every share's top-1.
pub fn draw(e: anytype, logits: u64, rows: usize, first: u64, out: []u32) !void {
    var buf: [max_held]sampler.Local = undefined;
    const ls = try mine(e, logits, &buf);
    if (e.cur.sampling) |s| return e.smp.draw(e.stream, .local, e.spans, ls, rows, first, sampler.rule(s), out);
    var shares: [max_held]shard.Share = undefined;
    shares[0] = e.share;
    for (e.local, 1..) |*l, j| shares[j] = l.w.share;
    for (0..rows) |r| {
        for (ls, shares[0..ls.len]) |l, share| {
            var a: cuda.Args = .{};
            a.add(l.logits + r * share.vocab.len() * 2);
            a.add(@as(i64, @intCast(share.vocab.len())));
            a.add(@as(u32, @intCast(share.vocab.lo)));
            a.add(e.pairs.ptr + (r * share.world + share.rank) * pair_bytes);
            try cuda_pdl.launch(e.m.glue.top1, .{ .grid = .{ .x = 1 }, .block = .{ .x = 1024 } }, e.stream, &a);
        }
    }
    try e.stream.synchronize();
    const world = e.share.world;
    var all: [max_held * fw.window_rows][4]u32 = undefined;
    try cuda.memory.download(e.d, e.pairs.ptr, std.mem.asBytes(&all)[0 .. rows * world * pair_bytes]);
    for (0..rows) |r| out[r] = chance.pick(all[r * world ..][0..world], false).token;
}

test "full GLM-5.3's three shares: all on one rank, one each on three; two ranks and GLM-5.3-Flash's own cut" {
    const dsa: config.Config = .{ .shares = 3 };
    try std.testing.expectEqual(Held{ .first = 0, .count = 3, .total = 3 }, try held(&dsa, 0, 1));
    try std.testing.expectEqual(Held{ .first = 2, .count = 1, .total = 3 }, try held(&dsa, 2, 3));
    try std.testing.expectError(error.SharesNotOnRanks, held(&dsa, 0, 2));
    try std.testing.expectEqual(Held{ .first = 1, .count = 1, .total = 2 }, try held(&config.Config{}, 1, 2));
}
