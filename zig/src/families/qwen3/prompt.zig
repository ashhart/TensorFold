//! The lane backend of a prompt-only Qwen3 engine: a stream's whole prompt in one step, pooled rows until release.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api");
const Engine = @import("engine.zig").Engine;
const engine_max_inputs = @import("engine.zig").max_inputs;
const score = @import("score.zig");

const be = lanes.backend;
const Stream = lanes.Stream;
const Allocator = std.mem.Allocator;

/// What a prompt-only engine refuses: drawing, verifying or drafting tokens.
pub const Refusal = error{PromptOnly};

pub const Prompt = struct {
    gpa: Allocator,
    engine: *Engine,
    /// The checkpoint's Normalize module: unit L2 rows.
    normalize: bool,
    /// Each stream's pooled fp32 rows, from its prompt pass to its release.
    slots: std.AutoHashMapUnmanaged(*Stream, []f32) = .empty,
    /// The reranker's head rows, when the checkpoint scores labels (`scoreLabels` is then in `table`).
    heads: ?score.Heads = null,
    table: be.Backend.VTable = .{
        .embedding = embedding,
        .prefill = prefill,
        .prefill_step = prefillStep,
        .first = first,
        .queue = queue,
        .read = read,
        .verify = verify,
        .keep = keep,
        .draft = draft,
        .release = release,
    },

    pub fn init(gpa: Allocator, engine: *Engine, normalize: bool) Prompt {
        return .{ .gpa = gpa, .engine = engine, .normalize = normalize };
    }

    pub fn deinit(p: *Prompt) void {
        var it = p.slots.valueIterator();
        while (it.next()) |rows| p.gpa.free(rows.*);
        p.slots.deinit(p.gpa);
        if (p.heads) |*h| h.deinit(p.gpa);
    }

    /// The score hook: one prompt's last row through the engine, its label logits in f64; no mass.
    pub fn scoreLabels(ptr: *anyopaque, ids: []const u32, labels: []const u32, logits: []f64, mass: bool) anyerror!f64 {
        const p = of(ptr);
        const heads = if (p.heads) |*h| h else return error.NoScoreHead;
        const e = p.engine;
        try score.check(ids, labels, logits, mass, e.window, e.m.spec.vocab, heads.count);
        const row = try p.gpa.alloc(f32, e.m.spec.hidden);
        defer p.gpa.free(row);
        try e.run(ids, &.{@intCast(ids.len)}, row);
        try score.labelLogits(p.gpa, heads, row, labels, logits);
        return 0;
    }

    /// The decide hook: branches packed into few runs; a last row keeps its prompt's bits, so scoreLabels's logits.
    pub fn decideLabels(ptr: *anyopaque, trunk: []const u32, branches: []const []const u32, counts: []const usize, labels: []const u32, logits: []f64) anyerror!void {
        const p = of(ptr);
        const heads = if (p.heads) |*h| h else return error.NoScoreHead;
        const e = p.engine;
        try score.checkDecision(trunk, branches, counts, labels, logits, e.window, e.m.spec.vocab, heads.count);
        const Runner = struct {
            e: *Engine,
            hidden: usize,
            pub fn run(r: @This(), tokens: []const u32, ends: []const u32, rows: []f32) !void {
                try r.e.run(tokens, ends, rows);
            }
        };
        try score.decidePacked(p.gpa, Runner{ .e = e, .hidden = e.m.spec.hidden }, heads, trunk, branches, counts, labels, logits, e.window, engine_max_inputs);
    }

    /// The backend over this engine; heads add their hooks to `table` before it serves.
    pub fn backend(p: *Prompt) be.Backend {
        return .{ .ptr = p, .vtable = &p.table, .chunk_invariant = true };
    }

    fn of(ptr: *anyopaque) *Prompt {
        return @ptrCast(@alignCast(ptr));
    }

    /// The whole prompt in one step: every input's last row, kept for `embedding`.
    fn prefillStep(ptr: *anyopaque, s: *Stream, _: u32) anyerror!bool {
        const p = of(ptr);
        const batch = s.embedding orelse return error.PromptOnly;
        const hidden = p.engine.m.spec.hidden;
        const rows = try p.gpa.alloc(f32, batch.ends.len * hidden);
        errdefer p.gpa.free(rows);
        try p.engine.run(s.prompt(), batch.ends, rows);
        const slot = try p.slots.getOrPut(p.gpa, s);
        if (slot.found_existing) p.gpa.free(slot.value_ptr.*);
        slot.value_ptr.* = rows;
        return true;
    }

    fn prefill(ptr: *anyopaque, s: *Stream) anyerror!void {
        _ = try prefillStep(ptr, s, 0);
    }

    /// The stream's pooled rows into `out`, each unit-normed when the checkpoint normalizes.
    fn embedding(ptr: *anyopaque, s: *Stream, out: []f32) anyerror!void {
        const p = of(ptr);
        const rows = p.slots.get(s) orelse return error.EmbeddingNotFilled;
        if (rows.len != out.len) return error.EmbeddingShape;
        @memcpy(out, rows);
        if (!p.normalize) return;
        const hidden = p.engine.m.spec.hidden;
        var at: usize = 0;
        while (at < out.len) : (at += hidden) try api.embedding_vector.normalize(out[at..][0..hidden]);
    }

    fn release(ptr: *anyopaque, s: *Stream) void {
        const p = of(ptr);
        if (p.slots.fetchRemove(s)) |kv| p.gpa.free(kv.value);
    }

    fn first(_: *anyopaque, _: *Stream, _: u64) anyerror!u64 {
        return error.PromptOnly;
    }
    fn queue(_: *anyopaque, _: *Stream, _: be.Feed, _: u64) anyerror!u64 {
        return error.PromptOnly;
    }
    fn read(_: *anyopaque, _: u64) anyerror!u32 {
        return error.PromptOnly;
    }
    fn verify(_: *anyopaque, _: []const be.Window, _: []be.Verified) anyerror!void {
        return error.PromptOnly;
    }
    fn keep(_: *anyopaque, _: []const be.Window, _: []const []const u32) anyerror!void {
        return error.PromptOnly;
    }
    fn draft(_: *anyopaque, _: []const be.DraftRequest) anyerror!void {
        return error.PromptOnly;
    }
};
