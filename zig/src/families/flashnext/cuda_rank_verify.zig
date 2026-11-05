//! Control carries a shared verify pass: each window's sequence and rows, then their ids; both ranks run one pass.
const std = @import("std");
const prompt = @import("cuda_prompt.zig");
const wire = @import("cuda_rank_wire.zig");
const wt = @import("cuda_prompt_wt.zig");
const mtp = @import("cuda_mtp.zig");
const Session = @import("cuda_rank_engine.zig").Session;

/// Windows a shared pass takes at most (a round's streams).
pub const max_windows = 32;
/// The bytes a frame can hold: every window's header, then a shared pass's rows of ids.
pub const max_bytes = 12 * max_windows + 4 * mtp.round_rows;

/// A decoded frame: window w is sequence ids[w], rows[w] token ids from `tokens` in window order.
pub const Frame = struct {
    count: usize,
    ids: [max_windows]u64 = undefined,
    rows: [max_windows]u32 = undefined,
    tokens: []const u8,

    /// Window w's token ids, each read from the frame's little-endian words.
    pub fn window(f: *const Frame, w: usize, out: *[mtp.max_rows]u32) []const u32 {
        var at: usize = 0;
        for (f.rows[0..w]) |r| at += r;
        for (out[0..f.rows[w]], 0..) |*v, i| v.* = std.mem.readInt(u32, f.tokens[(at + i) * 4 ..][0..4], .little);
        return out[0..f.rows[w]];
    }
};

/// `count` windows from `payload`: exact length, 1 to max_rows rows each, at most `limit` rows, distinct sequences.
pub fn decode(payload: []const u8, count: usize, limit: usize) !Frame {
    if (count == 0 or count > max_windows or payload.len < 12 * count or payload.len > max_bytes) return error.BadRankFrame;
    var f: Frame = .{ .count = count, .tokens = payload[12 * count ..] };
    var rows: usize = 0;
    for (0..count) |w| {
        f.ids[w] = std.mem.readInt(u64, payload[w * 12 ..][0..8], .little);
        f.rows[w] = std.mem.readInt(u32, payload[w * 12 + 8 ..][0..4], .little);
        if (f.ids[w] == 0 or f.rows[w] == 0 or f.rows[w] > mtp.max_rows) return error.BadRankFrame;
        for (f.ids[0..w]) |prior| if (prior == f.ids[w]) return error.DuplicateRankSequence;
        rows += f.rows[w];
    }
    if (rows > limit or f.tokens.len != 4 * rows) return error.BadRankFrame;
    for (0..rows) |i| if (std.mem.readInt(u32, f.tokens[i * 4 ..][0..4], .little) >= wt.vocab) return error.InvalidToken;
    return f;
}

/// The frame for `ids[w]`'s window `idss[w]` into `out`; refuses what `decode` refuses at the same `limit`.
pub fn encode(ids: []const u64, idss: []const []const u32, limit: usize, out: []u8) ![]const u8 {
    if (ids.len != idss.len) return error.BadRankFrame;
    var rows: usize = 0;
    for (idss) |w| rows += w.len;
    const n = 12 * ids.len + 4 * rows;
    if (n > out.len) return error.BadRankFrame;
    var at = 12 * ids.len;
    for (ids, idss, 0..) |id, w, i| {
        std.mem.writeInt(u64, out[i * 12 ..][0..8], id, .little);
        std.mem.writeInt(u32, out[i * 12 + 8 ..][0..4], @intCast(w.len), .little);
        for (w) |token| {
            std.mem.writeInt(u32, out[at..][0..4], token, .little);
            at += 4;
        }
    }
    _ = try decode(out[0..n], ids.len, limit);
    return out[0..n];
}

/// Every key in `map`, missing ones by `make(ctx, i)`, before any getPtr; a failed make's key stays absent.
pub fn ensure(map: anytype, gpa: std.mem.Allocator, keys: []const u64, ctx: anytype, comptime make: anytype) !void {
    for (keys, 0..) |key, i| {
        const item = try map.getOrPut(gpa, key);
        if (item.found_existing) continue;
        errdefer _ = map.remove(key);
        item.value_ptr.* = try make(ctx, i);
    }
}

/// Rank 1's window state for a sequence: one draft row, the head's absorb rows (Session.window's form).
pub fn makeWindow(s: *Session, q: *const prompt.Seq) !mtp.Draft {
    return mtp.Draft.init(s.gpa, s.engine.driver, s.engine.stream.*, q.cap, 1, s.engine.chunk, s.engine.heads);
}

fn limitOf(s: *const Session) usize {
    return @min(mtp.round_rows, s.engine.chunk);
}

/// Rank 0: the frame to rank 1, the same pass here, then rank 1's ack; a failure after the send poisons the session.
pub fn send(s: *Session, comptime Tri: type, tri: Tri, seqs: []const *prompt.Seq, ds: []const *mtp.Draft, idss: []const []const u32, out: u64) !void {
    if (!s.drafting or seqs.len == 0 or seqs.len > max_windows or seqs.len != ds.len or seqs.len != idss.len) return error.UnsupportedRankOperation;
    var ids: [max_windows]u64 = undefined;
    for (seqs, idss, 0..) |q, w, i| {
        if (q.rank_id == 0 or q.fed + w.len > q.cap) return error.UnsupportedRankOperation;
        ids[i] = q.rank_id;
    }
    var raw: [max_bytes]u8 = undefined;
    const bytes = try encode(ids[0..seqs.len], idss, limitOf(s), &raw);
    const h = try s.command(.{ .kind = .verify_many, .arg = seqs.len, .bytes = @intCast(bytes.len) }, bytes);
    errdefer s.poison();
    const ops: mtp.Ops(Tri) = .{ .e = s.engine, .tri = tri, .h = undefined };
    try ops.verifyMany(seqs, ds, idss, out);
    try s.ack(h);
}

/// Rank 1: every check before any state moves, every window draft made before any is borrowed, then the pass.
pub fn receive(s: *Session, comptime Tri: type, tri: Tri, h: wire.Header, payload: []const u8, logits: u64) !void {
    if (!s.drafting) return error.UnsupportedRankOperation;
    const f = try decode(payload, @intCast(h.arg), limitOf(s));
    var seqs: [max_windows]*prompt.Seq = undefined;
    for (f.ids[0..f.count], f.rows[0..f.count], 0..) |id, rows, w| {
        seqs[w] = s.seqs.get(id) orelse return error.UnknownRankSequence;
        if (seqs[w].fed + rows > seqs[w].cap) return error.PromptTooLong;
    }
    const Make = struct {
        s: *Session,
        seqs: []const *prompt.Seq,
        fn make(ctx: @This(), i: usize) !mtp.Draft {
            return makeWindow(ctx.s, ctx.seqs[i]);
        }
    };
    try ensure(&s.windows, s.gpa, f.ids[0..f.count], Make{ .s = s, .seqs = seqs[0..f.count] }, Make.make);
    var ds: [max_windows]*mtp.Draft = undefined;
    var tokens: [max_windows][mtp.max_rows]u32 = undefined;
    var idss: [max_windows][]const u32 = undefined;
    for (0..f.count) |w| {
        ds[w] = s.windows.getPtr(f.ids[w]).?;
        idss[w] = f.window(w, &tokens[w]);
    }
    const ops: mtp.Ops(Tri) = .{ .e = s.engine, .tri = tri, .h = undefined };
    try ops.verifyMany(seqs[0..f.count], ds[0..f.count], idss[0..f.count], logits);
}

test {
    _ = @import("cuda_rank_verify_test.zig");
}
