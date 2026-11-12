//! The joint schema head's device memory: each readout stream's state and the scratch one finish shares, by layout.
const std = @import("std");
const Config = @import("config.zig").Config;

/// Query rows an attention block scores at once: the FP32 score block is heads x 64 x keys.
pub const block = 64;
const alignment = 256;

/// The largest readout the engine admits: prompt rows, questions, options, and a piece's rows.
pub const Bounds = struct { rows: usize, questions: usize, options: usize, piece: usize };

/// Byte offsets into one buffer, each region 256-aligned.
const Bump = struct {
    at: usize = 0,

    fn take(b: *Bump, len: usize) usize {
        const here = b.at;
        b.at += std.mem.alignForward(usize, len, alignment);
        return here;
    }
};

/// A stream's state: the memory rows M, the FP32 span sums (questions then options), the last row g, and its ints.
pub const Lane = struct {
    memory: usize,
    sums: usize,
    g: usize,
    spans: usize, // [start, end) pairs, questions then options
    ids: usize, // the prompt, for the options' lexical rows
    owner: usize, // each option's question
    first: usize, // each question's first option, then the option count
    total: usize,

    pub fn of(c: Config, b: Bounds) Lane {
        var p: Bump = .{};
        const spans = b.questions + b.options;
        return .{
            .memory = p.take(b.rows * c.width * 2),
            .sums = p.take(spans * c.hidden_size * 4),
            .g = p.take(c.hidden_size * 2),
            .spans = p.take(spans * 8),
            .ids = p.take(b.rows * 4),
            .owner = p.take(b.options * 4),
            .first = p.take((b.questions + 1) * 4),
            .total = p.at,
        };
    }
};

/// The scratch every consume and finish shares (one runs at a time on the engine's stream).
pub const Scratch = struct {
    piece: usize, // a piece's normed rows (consume only; finish reuses the bytes)
    qv: usize,
    cv: usize,
    ev: usize,
    lex: usize, // FP32 sums of the options' lm_head rows
    routed: usize,
    tmp: usize,
    normed: usize,
    att: usize,
    ffh: usize,
    mn: usize,
    qp: usize,
    kp: usize,
    vp: usize,
    vf: usize,
    scores: usize,
    ctx32: usize,
    ctx: usize,
    fields: usize,
    summary: usize,
    weights: usize, // step 5's FP32 option weights, a slot an option
    gp: usize,
    opts: usize,
    feat: usize,
    hid: usize,
    res: usize,
    pc: usize, // FP32 prior then cosine
    logits: usize,
    total: usize,

    pub fn of(c: Config, b: Bounds) Scratch {
        const H = c.hidden_size;
        const W = c.width;
        const T = b.options;
        const Q = b.questions;
        const R = @max(T, Q); // the most query rows an attention or feed-forward takes
        const K = @max(b.rows, Q); // the most keys: the memory's rows, or the fields' in self-attention
        var p: Bump = .{};
        var s: Scratch = undefined;
        s.piece = 0;
        s.qv = p.take(Q * H * 2);
        s.cv = p.take(T * H * 2);
        s.ev = p.take(T * H * 2);
        s.lex = p.take(T * H * 4);
        s.routed = p.take(T * W * 2);
        s.tmp = p.take(R * W * 2);
        s.normed = p.take(R * W * 2);
        s.att = p.take(R * W * 2);
        s.ffh = p.take(R * c.feedforward * 2);
        s.mn = p.take(b.rows * W * 2);
        s.qp = p.take(R * W * 2);
        s.kp = p.take(K * W * 2);
        s.vp = p.take(K * W * 2);
        s.vf = p.take(K * W * 4);
        s.scores = p.take(c.heads * block * K * 4);
        s.ctx32 = p.take(block * W * 4);
        s.ctx = p.take(R * W * 2);
        s.fields = p.take(Q * W * 2);
        s.summary = p.take(Q * W * 2);
        s.weights = p.take(T * 4);
        s.gp = p.take(W * 2);
        s.opts = p.take(T * W * 2);
        s.feat = p.take(T * 4 * W * 2);
        s.hid = p.take(T * W * 2);
        s.res = p.take(T * 2);
        s.pc = p.take(T * 8);
        s.logits = p.take(T * 2);
        s.total = @max(p.at, b.piece * H * 2);
        return s;
    }
};

/// Device bytes beside the weights: `streams` states and one scratch.
pub fn bytes(c: Config, b: Bounds, streams: usize) usize {
    return streams * Lane.of(c, b).total + Scratch.of(c, b).total;
}

test "joint schema plan: Clef-Flash's state is 49 MiB a stream at 16,384 rows, the scratch under 400 MiB" {
    const b: Bounds = .{ .rows = 16384, .questions = 64, .options = 1024, .piece = 1024 };
    const lane = Lane.of(Config.clef_flash, b);
    try std.testing.expect(lane.total >= 49 << 20 and lane.total < 50 << 20);
    const s = Scratch.of(Config.clef_flash, b);
    try std.testing.expect(s.total > 256 << 20 and s.total < 400 << 20);
    try std.testing.expectEqual(lane.total * 3 + s.total, bytes(Config.clef_flash, b, 3));
}

test "joint schema plan: a window shorter than the questions still holds the fields' self-attention keys" {
    const b: Bounds = .{ .rows = 32, .questions = 64, .options = 1024, .piece = 32 };
    const s = Scratch.of(Config.clef_flash, b);
    const W = Config.clef_flash.width;
    try std.testing.expect(s.vp - s.kp >= b.questions * W * 2);
    try std.testing.expect(s.vf - s.vp >= b.questions * W * 2);
    try std.testing.expect(s.scores - s.vf >= b.questions * W * 4);
    try std.testing.expect(s.ctx32 - s.scores >= Config.clef_flash.heads * block * b.questions * 4);
}
