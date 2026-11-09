//! The dense engine's draws: a ring of drawn tokens behind handles, greedy picks or keyed samples of bf16 logits.
const std = @import("std");
const cuda = @import("cuda");
const kern = @import("kernels.zig");
const mdl = @import("model.zig");

const Pick = cuda.tree_round.Pick;
const round_rows = mdl.round_rows;

/// Drawn tokens a handle names (a ring; read before the round loop could wrap it).
pub const ring = 1024;

pub const Draws = struct {
    d: *const cuda.Driver,
    k: *const kern.Kernels,
    s: cuda.Stream,
    vocab: usize,
    tok: cuda.DeviceBuffer, // drawn tokens (u32): the handle ring, then a round's rows
    picks: cuda.DeviceBuffer, // argmax picks (Pick), the same entries
    greedy: []bool, // entry e of the ring (and the round's rows) is a pick, not a draw
    meta: cuda.DeviceBuffer, // a draw's first position minus one, a window each
    next: u64 = 0,

    /// The device bytes `init` allocates: the token and pick entries and the metas.
    pub fn openBytes() usize {
        return (ring + round_rows) * (4 + @sizeOf(Pick)) + round_rows * 16;
    }

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, k: *const kern.Kernels, s: cuda.Stream, vocab: usize) !Draws {
        const greedy = try gpa.alloc(bool, ring + round_rows);
        errdefer gpa.free(greedy);
        @memset(greedy, true);
        var tok = try cuda.DeviceBuffer.alloc(d, (ring + round_rows) * 4);
        errdefer tok.free();
        var picks = try cuda.DeviceBuffer.alloc(d, (ring + round_rows) * @sizeOf(Pick));
        errdefer picks.free();
        const meta = try cuda.DeviceBuffer.alloc(d, round_rows * 16);
        return .{ .d = d, .k = k, .s = s, .vocab = vocab, .tok = tok, .picks = picks, .greedy = greedy, .meta = meta };
    }

    pub fn deinit(x: *Draws, gpa: std.mem.Allocator) void {
        gpa.free(x.greedy);
        x.tok.free();
        x.picks.free();
        x.meta.free();
    }

    /// The next handle and its ring entry.
    pub fn handle(x: *Draws) struct { h: u64, at: usize } {
        const h = x.next;
        x.next += 1;
        return .{ .h = h, .at = @intCast(h % ring) };
    }

    /// Rows [row0, row0 + rows) of bf16 `logits` into entries [at, at + rows), row r at `position0 + r`.
    pub fn rows(x: *Draws, sampled: bool, rule: u64, logits: u64, row0: usize, n: usize, position0: u64, at: usize, meta_at: usize) !void {
        try x.pick(sampled, logits, row0, n, at);
        if (!sampled) return;
        try x.finite(at, n);
        try x.draw(rule, logits, row0, n, position0, at, meta_at);
    }

    /// Every row's argmax pick into entries [at, at + n): a greedy row's token, and any row's NaN or infinity flag.
    pub fn pick(x: *Draws, sampled: bool, logits: u64, row0: usize, n: usize, at: usize) !void {
        const v = x.vocab;
        @memset(x.greedy[at..][0..n], !sampled);
        try x.k.round.argmax(x.s, logits + row0 * v * 2, x.picks.ptr + at * @sizeOf(Pick), n, v, v);
    }

    /// Entries [at, at + n)'s picks read back: a row of NaN or infinite logits is refused before any draw reads it.
    pub fn finite(x: *Draws, at: usize, n: usize) !void {
        try x.s.synchronize();
        var picks: [round_rows]Pick = undefined;
        try x.picks.download(at * @sizeOf(Pick), std.mem.sliceAsBytes(picks[0..n]));
        for (picks[0..n]) |p| if (p.nonfinite != 0) return error.NonFiniteLogits;
    }

    /// Keyed samples of rows whose picks `finite` passed (tf_draw faults on a row with no finite logit).
    pub fn draw(x: *Draws, rule: u64, logits: u64, row0: usize, n: usize, position0: u64, at: usize, meta_at: usize) !void {
        const v = x.vocab;
        const meta = [4]i32{ @intCast(position0 - 1), 0, 0, 0 };
        const mp = x.meta.ptr + meta_at * 16;
        try mdl.put(x.d, x.s, mp, std.mem.asBytes(&meta));
        try x.k.draws.draw(x.s, logits + row0 * v * 2, v, rule, mp, 0, x.tok.ptr + at * 4, n, null, null);
    }

    /// Entries [at, at + out.len) as tokens after the stream's work; non-finite logits are an error, drawn or picked.
    pub fn tokens(x: *Draws, at: usize, out: []u32) !void {
        try x.s.synchronize();
        var picks: [round_rows]Pick = undefined;
        try x.tok.download(at * 4, std.mem.sliceAsBytes(out));
        try x.picks.download(at * @sizeOf(Pick), std.mem.sliceAsBytes(picks[0..out.len]));
        for (out, picks[0..out.len], x.greedy[at..][0..out.len]) |*t, p, g| {
            if (p.nonfinite != 0) return error.NonFiniteLogits;
            if (g) t.* = p.token;
        }
    }

    /// The token handle `h` names, once drawn; a handle the ring has passed is refused.
    pub fn read(x: *Draws, h: u64) !u32 {
        if (h >= x.next or x.next - h > ring) return error.NoSuchToken;
        var out: [1]u32 = undefined;
        try x.tokens(@intCast(h % ring), &out);
        return out[0];
    }
};
