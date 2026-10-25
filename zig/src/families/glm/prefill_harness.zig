//! The fake engine and event sink shared by GLM's prompt-pass tests; the driver's boundaries match the real backend.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("../../core/engine_api.zig");
const Backend = @import("backend.zig").Backend;
const mac = @import("builtin").os.tag == .macos;
const fill = @import("prefill.zig");
const a = std.testing.allocator;
const Stream = lanes.Stream;

pub const Fake = struct {
    gpa: std.mem.Allocator = a,
    target: lanes.fake.Fake = .{ .gpa = a },
    filling: std.AutoHashMapUnmanaged(*Stream, fill.State) = .empty,
    table: lanes.backend.Backend.VTable = undefined,
    barrier: bool = false,
    ready: std.atomic.Value(bool) = .init(false),
    go: std.atomic.Value(bool) = .init(false),
    chunks: usize = 0,
    first_round: u64 = 0,
    last_round: u64 = 0,
    max_rows: u32 = 0,
    begins: usize = 0,
    ends: usize = 0,
    restored_at: u32 = 0,
    fail_chunk: bool = false,
    chunk_hook: ?*const fn (*anyopaque, *Stream) void = null,
    chunk_ctx: ?*anyopaque = null,
    invariant: bool = false, // the harness backend's chunk_invariant declaration (real families: false)
    grid: u32 = 0, // when set, chunk starts must ride this grid or the fake raises the CUDA engine's BadChunk
    decode_shrink: bool = true, // false: chunks keep the engine's full cuts beside decoders (GLM CUDA's shape)
    starts: std.ArrayListUnmanaged(u32) = .empty, // chunk starts the grid guard saw
    fn from(ptr: *anyopaque) *Fake {
        const target: *lanes.fake.Fake = @ptrCast(@alignCast(ptr));
        return @fieldParentPtr("target", target);
    }
    pub fn backend(f: *Fake) lanes.backend.Backend {
        const stepped = if (comptime mac) blk: {
            var real: Backend = undefined;
            break :blk real.backend().vtable.prefill_step != null;
        } else true;
        const b = f.target.backend();
        f.table = b.vtable.*;
        f.table.prefill = prefill;
        f.table.prefill_step = if (stepped) step else null;
        f.table.release = release;
        f.table.verify = verify;
        return .{ .ptr = &f.target, .vtable = &f.table, .chunk_invariant = f.invariant };
    }
    pub fn deinit(f: *Fake) void {
        f.filling.deinit(a);
        f.starts.deinit(a);
        f.target.deinit();
    }
    fn prefill(ptr: *anyopaque, s: *Stream) anyerror!void {
        while (!try fill.step(from(ptr), s, 0)) {}
    }
    fn step(ptr: *anyopaque, s: *Stream, cap_rows: u32) anyerror!bool {
        return fill.step(from(ptr), s, cap_rows);
    }
    fn release(ptr: *anyopaque, s: *Stream) void {
        from(ptr).abortPrompt(s);
    }
    fn verify(ptr: *anyopaque, wins: []const lanes.backend.Window, out: []lanes.backend.Verified) anyerror!void {
        const f = from(ptr);
        try f.target.backend().vtable.verify(ptr, wins, out);
        if (f.barrier and !f.ready.load(.acquire)) {
            f.ready.store(true, .release);
            while (!f.go.load(.acquire)) std.Thread.yield() catch {};
        }
    }
    pub fn beginPrompt(f: *Fake, s: *Stream) !fill.State {
        f.begins += 1;
        f.target.backend().release(s);
        const got = try f.target.lanes.getOrPut(a, s);
        got.value_ptr.* = .{};
        errdefer f.target.backend().release(s);
        const at = if (s.reuse.saved != null) s.reuse.at else f.restored_at;
        if (at > s.prompt_len) return error.SnapshotNotPrefix;
        if (s.reuse.saved) |snap| {
            const kept: *std.ArrayList(u32) = @ptrCast(@alignCast(snap));
            if (kept.items.len != at) return error.SnapshotOutOfStep;
            try got.value_ptr.history.appendSlice(a, kept.items);
        } else try got.value_ptr.history.appendSlice(a, s.prompt()[0..at]);
        s.cached = at;
        s.fill_at.store(at, .monotonic);
        return .{ .stream = s, .slot = 0, .at = at };
    }
    pub fn decoding(f: *const Fake) bool {
        return f.decode_shrink and f.target.lanes.count() > f.filling.count();
    }
    pub fn promptRows(_: *Fake, at: u32, end: u32) u32 {
        return @min(16384, end - at);
    }
    pub fn chunkPrompt(f: *Fake, s: *Stream, _: u32, at: u32, n: u32) !void {
        if (f.fail_chunk) return error.GpuFailed;
        const held = &f.target.lanes.getPtr(s).?.history;
        if (held.items.len != at) return error.PositionMismatch;
        if (f.grid != 0 and s.prompt_len > f.grid) { // the CUDA engine's guard (cuda_engine.zig): starts ride the grid
            try f.starts.append(a, at);
            if (@mod(at, f.grid) != 0) return error.BadChunk;
        }
        try held.appendSlice(a, s.prompt()[at..][0..n]);
        if (s.prompt_len == 102400) {
            if (f.chunks == 0) f.first_round = f.target.rounds;
            f.last_round = f.target.rounds;
            f.chunks += 1;
            f.max_rows = @max(f.max_rows, n);
        }
        if (f.chunk_hook) |hook| hook(f.chunk_ctx.?, s);
    }
    pub fn finishPrompt(f: *Fake, s: *Stream) !void {
        if (f.target.lanes.get(s).?.history.items.len != s.prompt_len) return error.PromptIncomplete;
        f.ends += 1;
    }
    pub fn abortPrompt(f: *Fake, s: *Stream) void {
        _ = f.filling.remove(s);
        f.target.backend().release(s);
    }
};

pub const Box = struct {
    tokens: [2]std.ArrayList(u32) = @splat(.empty),
    done: std.atomic.Value(u32) = .init(0),
    bad: bool = false,
    pub fn event(ptr: *anyopaque, id: api.Id, e: *const api.Event) void {
        const b: *Box = @ptrCast(@alignCast(ptr));
        switch (e.*) {
            .tokens => |ts| b.tokens[id - 1].appendSlice(a, ts) catch {
                b.bad = true;
            },
            .finished => |end| {
                if (end.reason == .failed) b.bad = true;
                _ = b.done.fetchAdd(1, .release);
            },
            else => {},
        }
    }
    pub fn deinit(b: *Box) void {
        for (&b.tokens) |*ts| ts.deinit(a);
    }
};
