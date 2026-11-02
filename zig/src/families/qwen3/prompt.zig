//! The lane backend of a prompt-only Qwen3 engine: a stream's whole prompt in one step, pooled rows until release.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api");
const Engine = @import("engine.zig").Engine;

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
