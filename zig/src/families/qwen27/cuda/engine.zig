//! The 27B's CUDA engine for one stream: the weights, kernels and scratch, prefill and greedy serial decode.

const std = @import("std");
const cuda = @import("cuda");
const c = @import("shape.zig");
const kern = @import("kernels.zig");
const tri = @import("triton.zig");
const st = @import("state.zig");
const wts = @import("weights.zig");
const Forward = @import("forward.zig").Forward;
const Part = @import("forward.zig").Part;

/// prefill.CHUNK: a prompt chunk's rows (prompt_rows on a GPU of 80 GB or more).
pub const prompt_rows = 4096;

/// prefill.chunks: even chunks of at most `size` rows of [start, end); any bounds give the same bits.
pub fn chunkBounds(start: usize, end: usize, size: usize, j: usize) [2]usize {
    const n = (end - start + size - 1) / size;
    return .{ start + (end - start) * j / n, start + (end - start) * (j + 1) / n };
}

pub const Options = struct { context: usize };

pub const Engine = struct {
    gpa: std.mem.Allocator,
    ctx: *const cuda.Context,
    s: cuda.Stream,
    k: kern.Kernels,
    w: wts.Weights,
    scratch: st.Scratch,
    context: usize,

    pub fn init(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, kernels: []const u8, o: Options) !*Engine {
        const cfg = try c.Config.read(gpa, io, dir);
        const g = cfg.g;
        const e = try gpa.create(Engine);
        errdefer gpa.destroy(e);
        e.gpa = gpa;
        e.ctx = ctx;
        e.context = o.context;
        e.s = try cuda.Stream.init(ctx.d, true);
        errdefer e.s.deinit();
        e.k = try kern.Kernels.load(gpa, io, ctx, kernels, g.linear_v_heads);
        errdefer e.k.deinit();
        e.w = try wts.load(gpa, io, e.ops(), dir, cfg);
        errdefer e.w.deinit();
        e.scratch = try st.Scratch.init(gpa, ctx.d, g, prompt_rows, o.context);
        errdefer e.scratch.deinit();
        try e.s.synchronize();
        return e;
    }

    pub fn deinit(e: *Engine) void {
        e.scratch.deinit();
        e.w.deinit();
        e.k.deinit();
        e.s.deinit();
        e.gpa.destroy(e);
    }

    pub fn ops(e: *Engine) kern.Ops {
        return .{ .k = &e.k, .s = e.s };
    }

    pub fn forward(e: *Engine) Forward {
        return .{ .gpa = e.gpa, .w = &e.w, .ops = e.ops(), .t = .{ .set = &e.k.triton, .s = e.s }, .s = &e.scratch };
    }

    /// A stream's state for `capacity` positions (its caches, recurrences and conv windows), empty.
    pub fn sequence(e: *Engine, capacity: usize) !st.Seq {
        return st.Seq.init(e.gpa, e.ops(), e.w.g, capacity);
    }

    /// prefill_state from an empty state: the prompt in even chunks of at most `prompt_rows`, its last row normed.
    pub fn prefill(e: *Engine, seq: *st.Seq, prompt: []const u32) !void {
        if (prompt.len == 0) return error.EmptyPrompt;
        if (prompt.len >= seq.capacity) return error.PromptTooLong;
        try seq.reset(e.ops(), e.w.g);
        var f = e.forward();
        const n = (prompt.len + prompt_rows - 1) / prompt_rows;
        for (0..n) |j| {
            const b = chunkBounds(0, prompt.len, prompt_rows, j);
            try f.chunk(prompt[b[0]..b[1]], seq, j == n - 1);
        }
    }

    /// decode.serial_decode, greedy: one-row windows from `pending` until `count` tokens or an end token.
    pub fn decode(e: *Engine, seq: *st.Seq, pending: u32, count: usize, out: *std.ArrayList(u32)) !void {
        var f = e.forward();
        try out.append(e.gpa, pending);
        var pick: [1]u32 = undefined;
        while (out.items.len < count and !e.w.isEos(out.items[out.items.len - 1])) {
            const last = out.items[out.items.len - 1];
            const part = [_]Part{.{ .seq = seq, .tokens = &.{last} }};
            try f.round(&part);
            try f.picks(1, &pick);
            try f.commit(&part, &.{&.{0}});
            try out.append(e.gpa, pick[0]);
        }
    }

    /// The prompt, then up to `max_tokens` greedy tokens (the first included), as generate(draft=False) gives them.
    pub fn generate(e: *Engine, seq: *st.Seq, prompt: []const u32, max_tokens: usize, out: *std.ArrayList(u32)) !void {
        const room = @min(max_tokens, seq.capacity - prompt.len);
        try e.prefill(seq, prompt);
        var f = e.forward();
        try f.head(seq);
        var pick: [1]u32 = undefined;
        try f.picks(1, &pick);
        try e.decode(seq, pick[0], @max(1, room), out);
    }
};

test "prefill.chunks splits evenly" {
    try std.testing.expectEqual([2]usize{ 0, 2539 }, chunkBounds(0, 5078, 4096, 0));
    try std.testing.expectEqual([2]usize{ 2539, 5078 }, chunkBounds(0, 5078, 4096, 1));
    try std.testing.expectEqual([2]usize{ 0, 40 }, chunkBounds(0, 40, 4096, 0));
}
