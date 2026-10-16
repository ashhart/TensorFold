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
const pocket = @import("pocket.zig");
const hooks = @import("hooks.zig");

/// prefill.CHUNK: a prompt chunk's rows (prompt_rows on a GPU of 80 GB or more).
pub const prompt_rows = 4096;

/// prefill.chunks: even chunks of at most `size` rows of [start, end); any bounds give the same bits.
pub fn chunkBounds(start: usize, end: usize, size: usize, j: usize) [2]usize {
    const n = (end - start + size - 1) / size;
    return .{ start + (end - start) * j / n, start + (end - start) * (j + 1) / n };
}

/// Chunks from `from`: even splits, or with a prompt cache the zero-anchored grid so resumed passes cut alike.
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
    pockets: pocket.Pockets, // Living Weights' neurons beside each layer's MLP (none until a folder or a lesson has them)
    dir: []const u8 = "", // the checkpoint folder (the trainer reads its MLX copies there)

    pub fn init(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, kernels: []const u8, o: Options) !*Engine {
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
        e.dir = try gpa.dupe(u8, dir);
        errdefer gpa.free(e.dir);
        e.pockets = try pocket.Pockets.init(gpa, ctx.d, g.hidden, g.layers);
        errdefer e.pockets.deinit();
        if (std.c.getenv("TF_Q27_POCKET_DECOUPLE") != null and e.w.pocket.len > 0) {
            const norms = try gpa.alloc([]f32, g.layers);
            defer gpa.free(norms);
            const raw = try gpa.alloc(u16, g.hidden);
            defer gpa.free(raw);
            for (norms, 0..) |*n, l| {
                n.* = try gpa.alloc(f32, g.hidden);
                try ctx.d.check(ctx.d.api.cuMemcpyDtoH_v2(raw.ptr, e.w.layers[l].post_norm, g.hidden * 2), "cuMemcpyDtoH");
                for (n.*, raw) |*dst, b| dst.* = @bitCast(@as(u32, b) << 16);
            }
            defer for (norms) |n| gpa.free(n);
            var mean: f64 = 0;
            for (norms[20]) |v| mean += v;
            const share = try pocket.decouple(gpa, g.hidden, e.w.pocket, norms);
            std.log.info("qwen27: pocket diagnostic: outputs decoupled from later neurons' reads ({d:.1}% of their squared size removed; layer 20's post norm mean {d:.3})", .{ 100 * share, mean / @as(f64, @floatFromInt(g.hidden)) });
        }
        const learned = try e.pockets.loadAll(e.s, e.w.pocket);
        e.w.dropPocket();
        if (learned > 0) std.log.info("qwen27: the folder's Living Weights pocket: {d} learned neurons beside the MLPs", .{learned});
        e.scratch = try st.Scratch.init(gpa, ctx.d, g, prompt_rows, o.context);
        errdefer e.scratch.deinit();
        try e.s.synchronize();
        return e;
    }

    pub fn deinit(e: *Engine) void {
        e.pockets.deinit();
        e.gpa.free(e.dir);
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
        const mlp: ?hooks.MlpAdd = if (e.pockets.any()) .{ .ctx = e, .f = servePocket } else null;
        return .{ .gpa = e.gpa, .w = &e.w, .ops = e.ops(), .t = .{ .set = &e.k.triton, .s = e.s }, .s = &e.scratch, .mlp = mlp, .taught_rows = e.pockets.rowsBuffer() };
    }

    /// The serving forward's MLP hook: each layer's pocket added to its MLP's output.
    fn servePocket(ctx: *anyopaque, layer: usize, y: u64, out: u64, rows: usize, taught: u64) anyerror!void {
        const e: *Engine = @ptrCast(@alignCast(ctx));
        try e.pockets.add(e.s, layer, y, out, rows, taught);
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

    /// The prompt from `from` (0, or a grid point its state was restored at), keeping states at `keep`'s marks.
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
