//! A recording engine at the production prompt driver's boundaries, with history-dependent target tokens.
const std = @import("std");
const lanes = @import("lanes");
const Stream = lanes.Stream;
const fill = @import("prefill.zig");
const Backend = @import("backend.zig").Backend;
const a = std.testing.allocator;
const Pair = struct { row: usize, hidden: u64, token: u32 };
const Owner = struct {
    pairs: std.ArrayList(Pair) = .empty,
    hash: u64 = 0,
    last: u64 = 0,
    normalized: bool = true,
    serial: usize = 0,
    image_hash: u64 = 0,
};
pub const Summary = struct { hash: u64, head: u64, last: u64, pairs: usize, image_hash: u64 = 0 };
fn mix(h: u64, value: u64) u64 {
    return h *% 0x100000001b3 +% value +% 1;
}
pub fn imageHash(images: []const lanes.stream.Image) u64 {
    var h: u64 = 0;
    for (images) |image| h = mix(mix(mix(h, image.hash), image.at), image.tokens);
    return h;
}
pub fn reference(prompt: []const u32) Summary {
    return referenceImages(prompt, &.{});
}
pub fn referenceImages(prompt: []const u32, images: []const lanes.stream.Image) Summary {
    var h: u64 = imageHash(images);
    var head: u64 = 0;
    for (prompt, 0..) |token, row| {
        h = mix(h, token);
        if (row + 1 < prompt.len) head = mix(mix(head, h), prompt[row + 1]);
    }
    return .{ .hash = h, .head = head, .last = h, .pairs = prompt.len - 1, .image_hash = imageHash(images) };
}

pub const Fake = struct {
    gpa: std.mem.Allocator = a,
    target: lanes.fake.Fake = .{ .gpa = a },
    filling: std.AutoHashMapUnmanaged(*Stream, fill.State) = .empty,
    complete: std.AutoHashMapUnmanaged(*Stream, Summary) = .empty,
    table: lanes.backend.Backend.VTable = undefined,
    barrier: bool = false,
    ready: std.atomic.Value(bool) = .init(false),
    go: std.atomic.Value(bool) = .init(false),
    chunks: usize = 0,
    busy_chunks: usize = 0,
    first_round: u64 = 0,
    last_round: u64 = 0,
    max_busy_rows: u32 = 0,
    begins: usize = 0,
    ends: usize = 0,
    owners: usize = 0,
    freed: usize = 0,
    max_slots: usize = 4,
    restored_at: u32 = 0,
    fail_image: bool = false,
    fail_chunk: bool = false,
    fail_completion: bool = false,
    fail_mark: bool = false,
    drop_lookahead: bool = false,
    pending: bool = false,
    shared_owner: ?*Stream = null,
    deferred_owner: ?*Stream = null,
    settled: usize = 0,
    bound: ?*Stream = null,
    chunk_hook: ?*const fn (*anyopaque, *Stream) void = null,
    chunk_ctx: ?*anyopaque = null,

    pub fn init(gpa: std.mem.Allocator) Fake {
        return .{ .gpa = gpa, .target = .{ .gpa = gpa } };
    }
    pub fn self(ptr: *anyopaque) *Fake {
        const target: *lanes.fake.Fake = @ptrCast(@alignCast(ptr));
        return @fieldParentPtr("target", target);
    }
    pub fn promptPool(_: *Fake) typePool {
        return .{};
    }
    const typePool = struct {
        pub fn pop(_: @This()) void {}
    };
    pub fn backend(f: *Fake) lanes.backend.Backend {
        var real: Backend = undefined;
        const stepped = if (comptime @import("builtin").os.tag == .macos)
            real.backend().vtable.prefill_step == fill.Callbacks(Backend).one
        else
            true;
        const b = f.target.backend();
        f.table = b.vtable.*;
        f.table.prefill = fill.Callbacks(Fake).whole;
        f.table.prefill_step = if (stepped) fill.Callbacks(Fake).one else null;
        f.table.release = release;
        f.table.verify = verify;
        f.table.keep = keep;
        return .{ .ptr = &f.target, .vtable = &f.table };
    }
    pub fn deinit(f: *Fake) void {
        while (f.filling.count() > 0) {
            var it = f.filling.keyIterator();
            f.abortPrompt(it.next().?.*);
        }
        f.filling.deinit(f.gpa);
        f.complete.deinit(f.gpa);
        f.target.deinit();
    }
    fn release(ptr: *anyopaque, s: *Stream) void {
        self(ptr).abortPrompt(s);
    }
    fn verify(ptr: *anyopaque, wins: []const lanes.backend.Window, out: []lanes.backend.Verified) anyerror!void {
        const f = self(ptr);
        if (wins.len > 0) try f.settlePrompt(wins[0].stream);
        try f.target.backend().vtable.verify(ptr, wins, out);
        if (wins.len > 0) {
            f.shared_owner = wins[0].stream;
            f.pending = true;
            f.deferred_owner = f.shared_owner;
        }
        if (f.barrier and !f.ready.load(.acquire)) {
            f.ready.store(true, .release);
            while (!f.go.load(.acquire)) std.Thread.yield() catch {};
        }
    }
    fn keep(ptr: *anyopaque, windows: []const lanes.backend.Window, paths: []const []const u32) anyerror!void {
        const f = self(ptr);
        try f.target.backend().vtable.keep(ptr, windows, paths);
        f.pending = true;
        f.deferred_owner = f.shared_owner;
    }
    pub fn beginPrompt(f: *Fake, s: *Stream, preferred: ?usize) !fill.State {
        if (f.target.lanes.contains(s)) return error.PromptAlreadyStarted;
        if (f.target.lanes.count() >= f.max_slots) return error.LaneMemory;
        if (s.prompt_len == 0) return error.EmptyPrompt;
        f.begins += 1;
        const o = try f.gpa.create(Owner);
        o.* = .{ .serial = f.begins, .image_hash = imageHash(s.images), .hash = imageHash(s.images) };
        errdefer {
            o.pairs.deinit(f.gpa);
            f.gpa.destroy(o);
        }
        const got = try f.target.lanes.getOrPut(f.gpa, s);
        got.value_ptr.* = .{};
        errdefer f.target.backend().release(s);
        if (f.fail_image and s.images.len > 0) return error.ImageFailed;
        const asked = if (s.reuse.saved != null) s.reuse.at else f.restored_at;
        const at = if (asked < s.prompt_len) asked else 0;
        if (asked >= s.prompt_len) s.reuse_failed = true;
        try got.value_ptr.history.appendSlice(f.gpa, s.prompt()[0..at]);
        for (s.prompt()[0..at], 0..) |token, row| {
            o.hash = mix(o.hash, token);
            try o.pairs.append(f.gpa, .{ .row = row, .hidden = o.hash, .token = s.prompt()[row + 1] });
        }
        s.cached = at;
        s.fill_at.store(at, .monotonic);
        f.owners += 1;
        return .{ .stream = s, .slot = preferred orelse f.target.lanes.count() - 1, .at = at, .owner = o };
    }
    pub fn decoding(f: *const Fake) bool {
        var it = f.target.lanes.keyIterator();
        while (it.next()) |key| if (!f.filling.contains(key.*) and !key.*.finished and !key.*.paused) return true;
        return false;
    }
    pub fn settlePrompt(f: *Fake, _: *Stream) !void {
        if (f.pending) {
            if (f.shared_owner != f.deferred_owner) return error.SharedInputsOverwritten;
            f.pending = false;
            f.settled += 1;
        }
    }
    pub fn promptRows(_: *Fake, at: u32, end: u32) u32 {
        return @min(1024, end - at);
    }
    fn owner(f: *Fake, s: *Stream) *Owner {
        return @ptrCast(@alignCast(f.filling.get(s).?.owner.?));
    }
    pub fn chunkPrompt(f: *Fake, s: *Stream, _: usize, at: u32, n: u32) !void {
        if (f.pending) return error.SharedInputsOverwritten;
        if (f.fail_chunk) return error.GpuFailed;
        const history = &f.target.lanes.getPtr(s).?.history;
        if (history.items.len != at) return error.PositionMismatch;
        const o = f.owner(s);
        if (!o.normalized) return error.WrongRecurrentRow;
        if (o.image_hash != imageHash(s.images)) return error.WrongImageOwner;
        f.shared_owner = s;
        f.bound = s;
        o.normalized = false;
        try history.appendSlice(f.gpa, s.prompt()[at..][0..n]);
        const parts: usize = if (n > 1) 2 else 1;
        for (0..parts) |k| {
            const span = try fill.piece(at, n, parts, k, s.prompt_len);
            for (0..span.rows) |j| {
                const row = span.start + j;
                o.hash = mix(o.hash, s.prompt()[row]);
                const paired = j < span.head_rows;
                if (paired and !(f.drop_lookahead and j + 1 == span.rows))
                    try o.pairs.append(f.gpa, .{ .row = row, .hidden = o.hash, .token = s.prompt()[row + 1] });
            }
            if (span.final) o.last = o.hash;
        }
        f.chunks += 1;
        if (s.prompt_len == 102400 and f.decoding()) {
            if (f.busy_chunks == 0) f.first_round = f.target.rounds;
            f.last_round = f.target.rounds;
            f.busy_chunks += 1;
            f.max_busy_rows = @max(f.max_busy_rows, n);
        }
        if (f.chunk_hook) |hook| hook(f.chunk_ctx.?, s);
    }
    pub fn finishChunk(f: *Fake, s: *Stream) !void {
        if (f.fail_completion) return error.GpuFailed;
        if (f.bound != s) return error.WrongSession;
        const o = f.owner(s);
        const end = f.target.lanes.get(s).?.history.items.len;
        if (o.pairs.items.len != @min(end, s.prompt_len - 1)) return error.LostPromptLookahead;
        o.normalized = true;
    }
    pub fn markPrompt(f: *Fake, s: *Stream, at: u32) !void {
        if (f.fail_mark) return error.MarkFailed;
        if (f.bound != s or !f.owner(s).normalized) return error.WrongSession;
        if (f.target.lanes.get(s).?.history.items.len != at) return error.PositionMismatch;
        if (s.reuse.hook) |hook| hook.at(hook.ptr, s, at);
    }
    pub fn finishPrompt(f: *Fake, s: *Stream) !void {
        const o = f.owner(s);
        if (!o.normalized) return error.WrongRecurrentRow;
        var head: u64 = 0;
        for (o.pairs.items, 0..) |pair, row| {
            if (row != pair.row) return error.HeadOutOfStep;
            head = mix(mix(head, pair.hidden), pair.token);
        }
        const result: Summary = .{ .hash = o.hash, .head = head, .last = o.last, .pairs = o.pairs.items.len, .image_hash = o.image_hash };
        const want = referenceImages(s.prompt(), s.images);
        if (!std.meta.eql(want, result)) return error.PromptStateMismatch;
        try f.complete.put(f.gpa, s, result);
        f.filling.getPtr(s).?.owner = null;
        f.dropOwner(o);
        f.ends += 1;
    }
    fn dropOwner(f: *Fake, o: *Owner) void {
        o.pairs.deinit(f.gpa);
        f.gpa.destroy(o);
        f.freed += 1;
    }
    pub fn abortPrompt(f: *Fake, s: *Stream) void {
        if (f.filling.fetchRemove(s)) |entry| if (entry.value.owner) |p| f.dropOwner(@ptrCast(@alignCast(p)));
        f.target.backend().release(s);
    }
};
