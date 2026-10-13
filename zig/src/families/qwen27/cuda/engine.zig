//! The 27B's CUDA engine for one stream: the weights, kernels and scratch, prefill and greedy serial decode.

const std = @import("std");
const cuda = @import("cuda");
const c = @import("shape.zig");
const kern = @import("kernels.zig");
const tri = @import("triton.zig");
const st = @import("state.zig");
const vision = @import("vision.zig");
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

/// A prompt's chunks from `from`: even splits of the whole prompt (0.6.6's PrefillPlan), or under a prompt cache the
/// zero-anchored grid, so a pass resumed or kept at a grid point cuts where a fresh pass of the same prompt does.
pub const Chunks = struct {
    len: usize,
    grid: bool,
    at: usize,
    j: usize = 0,

    pub fn init(len: usize, from: usize, grid: bool) Chunks {
        return .{ .len = len, .grid = grid, .at = from };
    }

    pub fn next(x: *Chunks) ?[2]usize {
        if (x.at >= x.len) return null;
        const b: [2]usize = if (x.grid) .{ x.at, @min(x.len, x.at - x.at % prompt_rows + prompt_rows) } else chunkBounds(0, x.len, prompt_rows, x.j);
        x.at = b[1];
        x.j += 1;
        return b;
    }
};

/// A state the cache may resume: a grid point inside the prompt.
pub fn resumable(from: usize, len: usize) bool {
    return from > 0 and from < len and from % prompt_rows == 0;
}

/// Where a prompt pass stops to let the cache copy its state (ascending grid points), and who copies it.
pub const Keep = struct {
    at: []const u32,
    ctx: *anyopaque,
    call: *const fn (ctx: *anyopaque, at: u32) void,

    /// Call the keep for a chunk that ended at `end`.
    pub fn after(k: ?Keep, end: usize) void {
        const x = k orelse return;
        for (x.at) |m| if (m == end) x.call(x.ctx, m);
    }
};

pub const Options = struct { context: usize, kv8: bool = false };

pub const Engine = struct {
    gpa: std.mem.Allocator,
    ctx: *const cuda.Context,
    s: cuda.Stream,
    k: kern.Kernels,
    w: wts.Weights,
    scratch: st.Scratch,
    context: usize,
    kv8: bool, // --kv-cache fp8: every stream's attention caches e4m3 with a scale a row
    grid: bool = false, // a prompt cache: prompts in zero-anchored chunks (Chunks), not even splits

    pub fn init(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, kernels: []const u8, o: Options) !*Engine {
        if (o.kv8 and kernels.len == 0) return error.QwenNeedsCapturedKernels;
        const cfg = try c.Config.read(gpa, io, dir);
        const g = cfg.g;
        const e = try gpa.create(Engine);
        errdefer gpa.destroy(e);
        e.gpa = gpa;
        e.ctx = ctx;
        e.context = o.context;
        e.kv8 = o.kv8;
        e.s = try cuda.Stream.init(ctx.d, true);
        errdefer e.s.deinit();
        e.k = try kern.Kernels.load(gpa, io, ctx, kernels, g.linear_v_heads);
        errdefer e.k.deinit();
        e.w = try wts.load(gpa, io, e.ops(), dir, cfg);
        errdefer e.w.deinit();
        e.scratch = try st.Scratch.initForKv(gpa, ctx.d, g, prompt_rows, o.context, o.kv8);
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
        return .{ .gpa = e.gpa, .w = &e.w, .ops = e.ops(), .t = .{ .g = &e.k.glue, .s = e.s, .set = if (e.k.glue.set) |*s| s else null }, .s = &e.scratch };
    }

    /// A stream's state for `capacity` positions (its caches, recurrences and conv windows), empty.
    pub fn sequence(e: *Engine, capacity: usize) !st.Seq {
        return st.Seq.init(e.gpa, e.ops(), e.w.g, capacity, e.kv8);
    }

    /// Attention's partials for `streams` streams of the engine's context in one round.
    pub fn fit(e: *Engine, streams: usize) !void {
        try e.s.synchronize();
        try e.scratch.fitPartials(e.ctx.d, e.w.g, streams * st.Scratch.streamPartRows(e.context));
    }

    /// prefill_state from an empty state: the prompt in even chunks of at most `prompt_rows`, its last row normed.
    pub fn prefill(e: *Engine, seq: *st.Seq, prompt: []const u32) !void {
        return e.prefillMedia(seq, prompt, null);
    }

    /// prefill with an image prompt's rows and positions (null: text); its decode offset stays on the stream.
    pub fn prefillMedia(e: *Engine, seq: *st.Seq, prompt: []const u32, media: ?*const vision.Media) !void {
        return e.prefillFrom(seq, prompt, 0, media, null);
    }

    /// The prompt from `from` (0, or a grid point the stream's state was restored at), keeping states at `keep`'s marks.
    pub fn prefillFrom(e: *Engine, seq: *st.Seq, prompt: []const u32, from: usize, media: ?*const vision.Media, keep: ?Keep) !void {
        if (prompt.len == 0) return error.EmptyPrompt;
        if (prompt.len >= seq.capacity) return error.PromptTooLong;
        if (from == 0) try seq.reset(e.ops(), e.w.g) else if (from != seq.pos or !resumable(from, prompt.len)) return error.BadReuse;
        seq.media = media;
        defer seq.media = null;
        var f = e.forward();
        var it = Chunks.init(prompt.len, from, e.grid);
        while (it.next()) |b| {
            try f.chunk(prompt[b[0]..b[1]], seq, b[1] == prompt.len);
            Keep.after(keep, b[1]);
        }
        if (media) |m| seq.rope_delta = m.delta;
    }

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

test "a cached prompt cuts the zero-anchored grid, resumed or not" {
    var it = Chunks.init(9000, 0, true);
    try std.testing.expectEqual([2]usize{ 0, 4096 }, it.next().?);
    try std.testing.expectEqual([2]usize{ 4096, 8192 }, it.next().?);
    try std.testing.expectEqual([2]usize{ 8192, 9000 }, it.next().?);
    try std.testing.expectEqual(@as(?[2]usize, null), it.next());
    var resumed = Chunks.init(9000, 4096, true);
    try std.testing.expectEqual([2]usize{ 4096, 8192 }, resumed.next().?);
    var even = Chunks.init(5078, 0, false);
    try std.testing.expectEqual([2]usize{ 0, 2539 }, even.next().?);
    try std.testing.expectEqual([2]usize{ 2539, 5078 }, even.next().?);
    try std.testing.expect(resumable(4096, 5000) and !resumable(4096, 4096) and !resumable(3000, 5000) and !resumable(0, 5000));
}

test "prefill.chunks splits evenly" {
    try std.testing.expectEqual([2]usize{ 0, 2539 }, chunkBounds(0, 5078, 4096, 0));
    try std.testing.expectEqual([2]usize{ 2539, 5078 }, chunkBounds(0, 5078, 4096, 1));
    try std.testing.expectEqual([2]usize{ 0, 40 }, chunkBounds(0, 40, 4096, 0));
}
