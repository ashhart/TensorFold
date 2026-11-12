//! LightOnOCR's HIP lane adapter: shared chain forwards, request-owned image spans and positional KV rollback.
const std = @import("std");
const hip = @import("hip");
const core = @import("core");
const lanes = @import("lanes");
const serve = @import("hip_serve");
const pc = @import("engine_api").prompt_cache;
const cfg = @import("../../config.zig");
const text = @import("text.zig");
const tower = @import("tower.zig");
const vision = @import("vision.zig");
const snapshot = @import("snapshot.zig");
const CancelCheck = @import("lanes").stream.CancelCheck;
const cancels = @import("hip_serve").cancel;
const Allocator = std.mem.Allocator;
const be = lanes.backend;
const Stream = lanes.Stream;

pub const max_window = 16;
pub const round_rows = 128;
pub const prompt_rows = 128;
pub const Vision = vision.For(tower.Tower);

const Lane = struct {
    slot: usize,
    len: usize = 0,
    base: usize = 0,
    verified: usize = 0,
    fill: ?usize = 0,
    images: []vision.Span,
};

test "LightOnOCR HIP opening adapts request cancellation before touching a slot or feature buffer" {
    const Check = struct {
        fn check(ptr: *anyopaque) bool {
            const cancelled: *bool = @ptrCast(@alignCast(ptr));
            return cancelled.*;
        }
    };
    var cancelled = true;
    var s = try Stream.init(std.testing.allocator, .{
        .id = "cancelled-before-opening",
        .prompt = &.{1},
        .max_new = 1,
        .cancel_check = .{ .ptr = &cancelled, .check = Check.check },
    });
    defer s.deinit(std.testing.allocator);
    var h: Hip = undefined;
    try std.testing.expectError(error.Cancelled, h.opening(&s));
    cancelled = false;
    try cancels.poll(Hip.cancellation(&s));
    s.cancel_check = null;
    try std.testing.expect(Hip.cancellation(&s) == null);
}

fn sampleScratch(c: cfg.Config) usize {
    return round_rows * ((@as(usize, c.vocab) + 1023) / 1024) * 8 + 512;
}

pub fn fixedBytes(c: cfg.Config) usize {
    return serve.draw.Drawer.deviceBytes(round_rows) + sampleScratch(c);
}

pub fn slotBytes(c: cfg.Config, limits: ?vision.Limits) usize {
    return @as(usize, c.vocab) * 2 + if (limits) |v| @as(usize, v.image_tokens) * c.hidden * 2 else @as(usize, 0);
}

pub const Hip = struct {
    gpa: Allocator,
    ctx: *const hip.Context,
    m: *text.Text,
    shared: *const hip.Launcher,
    vision: ?*Vision,
    window: usize,
    free: []bool,
    live: std.AutoHashMapUnmanaged(*const Stream, Lane) = .empty,
    drawer: serve.draw.Drawer,
    sample_arena: hip.Arena,
    drawn: serve.backend.Drawn = .{},
    last: hip.DeviceBuffer,
    image_rows: hip.DeviceBuffer,

    pub fn init(gpa: Allocator, ctx: *const hip.Context, m: *text.Text, shared: *const hip.Launcher, v: ?*Vision, window: usize) !Hip {
        const free = try gpa.alloc(bool, m.slots);
        errdefer gpa.free(free);
        @memset(free, true);
        var drawer = try serve.draw.Drawer.init(gpa, m.s.r, .bf16, m.c.vocab, round_rows);
        errdefer drawer.deinit();
        var sample_arena = try hip.Arena.init(m.s.r, sampleScratch(m.c));
        errdefer sample_arena.deinit();
        var last = try hip.DeviceBuffer.alloc(m.s.r, m.slots * m.c.vocab * 2);
        errdefer last.free();
        const image_bytes = if (v) |val| m.slots * @as(usize, val.limits.image_tokens) * m.c.hidden * 2 else 0;
        const image_rows = try hip.DeviceBuffer.alloc(m.s.r, image_bytes);
        return .{ .gpa = gpa, .ctx = ctx, .m = m, .shared = shared, .vision = v, .window = window, .free = free, .drawer = drawer, .sample_arena = sample_arena, .last = last, .image_rows = image_rows };
    }

    pub fn deinit(h: *Hip) void {
        snapshot.drain(h.m.s);
        var it = h.live.valueIterator();
        while (it.next()) |lane| h.gpa.free(lane.images);
        h.live.deinit(h.gpa);
        h.image_rows.free();
        h.last.free();
        h.sample_arena.deinit();
        h.drawer.deinit();
        h.gpa.free(h.free);
        h.* = undefined;
    }

    pub fn backend(h: *Hip) be.Backend {
        return .{ .ptr = h, .vtable = &vtable, .chunk_invariant = true };
    }

    pub fn facts(h: *const Hip) lanes.Model {
        return .{ .exact_width = max_window, .gpu_tokens = false, .mtp = false, .speculate = false, .hidden_rows = true, .batch_rows = round_rows, .max_streams = @intCast(h.free.len) };
    }

    const vtable: be.Backend.VTable = .{ .prefill = prefill, .prefill_step = prefillStep, .first = first, .first_masked = firstMasked, .queue = queue, .read = read, .verify = verify, .keep = keep, .draft = draft, .release = release, .reports_fill = true };

    fn of(ptr: *anyopaque) !*Hip {
        const h: *Hip = @ptrCast(@alignCast(ptr));
        try h.ctx.makeCurrent();
        return h;
    }

    fn lastAt(h: *const Hip, slot: usize) u64 {
        return h.last.base() + slot * h.m.c.vocab * 2;
    }

    fn cancellation(s: *const Stream) ?CancelCheck {
        return s.cancel_check;
    }

    fn opening(h: *Hip, s: *Stream) !*Lane {
        const cancel = cancellation(s);
        try cancels.poll(cancel);
        const prompt = s.prompt();
        if (prompt.len == 0) return error.EmptyPrompt;
        if (prompt.len > h.window or s.max_new > h.window - prompt.len) return error.PastContext;
        if (h.live.contains(s)) return error.AlreadyPrefilled;
        const slot = std.mem.indexOfScalar(bool, h.free, true) orelse return error.NoFreeSlot;
        if (s.images.len > 0 and h.vision == null) return error.VisionDisabled;
        if (h.vision) |v| _ = try vision.validate(v.c, v.limits, s.images, prompt.len);
        const spans = try h.gpa.alloc(vision.Span, s.images.len);
        errdefer h.gpa.free(spans);
        if (s.images.len > 0) {
            const v = h.vision.?;
            const at = h.image_rows.base() + slot * @as(usize, v.limits.image_tokens) * h.m.c.hidden * 2;
            try v.encodeCancelable(s.images, prompt.len, at, spans, cancel);
        }
        try h.live.put(h.gpa, s, .{ .slot = slot, .images = spans });
        h.free[slot] = false;
        const lane = h.live.getPtr(s).?;
        s.cached = 0;
        if (s.reuse.saved) |saved| {
            const kept: *const snapshot.Saved = @ptrCast(@alignCast(saved));
            if (kept.at != s.reuse.at or kept.at >= prompt.len) {
                s.reuse_failed = true;
            } else if (snapshot.restore(h.m, slot, kept)) |_| {
                lane.len = kept.at;
                lane.fill = kept.at;
                s.cached = kept.at;
            } else |_| s.reuse_failed = true;
        }
        s.fill_at.store(@intCast(lane.fill.?), .monotonic);
        return lane;
    }

    fn chunk(h: *Hip, s: *Stream, lane: *Lane, max_rows: u32) !bool {
        if (s.isCancelled()) return error.Cancelled;
        const prompt = s.prompt();
        const at = lane.fill orelse return true;
        const count = if (max_rows == 0) prompt_rows else @min(max_rows, prompt_rows);
        var end = @min(prompt.len, at + count);
        for (s.reuse.marks) |mark| if (mark > at and mark < end) {
            end = mark;
        };
        var images: std.ArrayList(text.ImageRows) = .empty;
        defer images.deinit(h.gpa);
        for (lane.images) |span| if (vision.intersect(span, @intCast(at), @intCast(end), 0, h.m.c.hidden)) |rows| {
            try images.append(h.gpa, .{ .first = rows.first, .rows = rows.rows, .features = rows.features });
        };
        errdefer snapshot.drain(h.m.s);
        try h.m.forwardCancelable(prompt[at..end], &.{.{ .slot = @intCast(lane.slot), .first = 0, .rows = @intCast(end - at), .committed = @intCast(at) }}, images.items, cancellation(s));
        if (end == prompt.len) try hip.runtime.check(h.m.s.r.api.hipMemcpyDtoDAsync(@ptrFromInt(h.lastAt(lane.slot)), @ptrFromInt(h.m.logits + (end - at - 1) * h.m.c.vocab * 2), h.m.c.vocab * 2, h.m.s.handle));
        try h.m.s.synchronize();
        lane.len = end;
        lane.fill = if (end == prompt.len) null else end;
        s.fill_at.store(@intCast(end), .monotonic);
        if (s.reuse.hook) |hook| if (std.mem.indexOfScalar(u32, s.reuse.marks, @intCast(end)) != null)
            hook.at(hook.ptr, s, @intCast(end));
        return end == prompt.len;
    }

    fn prefill(ptr: *anyopaque, s: *Stream) anyerror!void {
        const h = try of(ptr);
        const lane = try h.opening(s);
        errdefer h.drop(s);
        while (!try h.chunk(s, lane, 0)) {}
    }

    fn prefillStep(ptr: *anyopaque, s: *Stream, rows: u32) anyerror!bool {
        const h = try of(ptr);
        const lane = h.live.getPtr(s) orelse try h.opening(s);
        errdefer h.drop(s);
        return h.chunk(s, lane, rows);
    }

    fn draw(h: *Hip, logits: u64, requests: []const serve.draw.Request, out: []u32) !void {
        h.sample_arena.reset();
        const o: hip.ops.Ops = .{ .l = h.shared, .bf16 = true, .stream = h.m.s.handle, .arena = &h.sample_arena, .window = true };
        errdefer snapshot.drain(h.m.s);
        try h.drawer.draw(o, h.m.s, .{ .ptr = logits, .kind = .bf16 }, requests, out, false);
    }

    fn firstMasked(ptr: *anyopaque, s: *Stream, position: u64, mask: []const u32) anyerror!u64 {
        const h = try of(ptr);
        const lane = h.live.getPtr(s) orelse return error.UnknownStream;
        if (position != lane.len or lane.fill != null) return error.PositionMismatch;
        var token: [1]u32 = undefined;
        try h.draw(h.lastAt(lane.slot), &.{.{ .sampling = s.sampling, .position = position, .mask = mask }}, &token);
        return h.drawn.take(token[0]);
    }

    fn first(ptr: *anyopaque, s: *Stream, position: u64) anyerror!u64 {
        return firstMasked(ptr, s, position, &.{});
    }

    fn queue(ptr: *anyopaque, s: *Stream, feed: be.Feed, position: u64) anyerror!u64 {
        const h = try of(ptr);
        const lane = h.live.getPtr(s) orelse return error.UnknownStream;
        if (position != lane.len + 1 or position > h.window) return error.PositionMismatch;
        const token = switch (feed) {
            .value => |v| v,
            .handle => |v| try h.drawn.read(v),
        };
        try h.m.forward(&.{token}, &.{.{ .slot = @intCast(lane.slot), .first = 0, .rows = 1, .committed = @intCast(lane.len) }}, &.{});
        var out: [1]u32 = undefined;
        try h.draw(h.m.logits, &.{.{ .sampling = s.sampling, .position = position }}, &out);
        lane.len += 1;
        return h.drawn.take(out[0]);
    }

    fn read(ptr: *anyopaque, handle: u64) anyerror!u32 {
        return (try of(ptr)).drawn.read(handle);
    }

    fn verify(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
        const h = try of(ptr);
        if (windows.len != out.len or windows.len > round_rows) return error.RoundTooWide;
        var ids: [round_rows]u32 = undefined;
        var spans: [round_rows]text.Span = undefined;
        var requests: [round_rows]serve.draw.Request = undefined;
        var total: usize = 0;
        for (windows, out, 0..) |w, result, i| {
            const lane = h.live.getPtr(w.stream) orelse return error.UnknownStream;
            if (w.held != 0) return error.NoHeldDrafts;
            if (w.parents != null) return error.ChainsOnly;
            const n = w.rows();
            if (n > max_window or total + n > round_rows or w.positions.len != n or result.sampled.len < n or result.drafts.len < n - 1)
                return error.WindowTooWide;
            if (lane.len + n > h.m.context or lane.fill != null) return error.PastContext;
            if (w.masks.len != 0 and w.masks.len != n * ((h.m.c.vocab + 31) / 32)) return error.MaskShape;
            for (w.positions, 0..) |position, r| if (position != lane.len + r + 1) return error.PositionMismatch;
            ids[total] = w.pending;
            @memcpy(ids[total + 1 ..][0..w.tokens.len], w.tokens);
            spans[i] = .{ .slot = @intCast(lane.slot), .first = @intCast(total), .rows = @intCast(n), .committed = @intCast(lane.len) };
            serve.backend.requests(w, requests[total..][0..n]);
            lane.base = lane.len;
            lane.verified = n;
            total += n;
        }
        if (total == 0) return;
        try h.m.forward(ids[0..total], spans[0..windows.len], &.{});
        var tokens: [round_rows]u32 = undefined;
        try h.draw(h.m.logits, requests[0..total], tokens[0..total]);
        total = 0;
        for (windows, out) |w, result| {
            const lane = h.live.getPtr(w.stream).?;
            const n = w.rows();
            @memcpy(result.sampled[0..n], tokens[total..][0..n]);
            @memcpy(result.drafts[0 .. n - 1], w.tokens);
            lane.len = lane.base + n;
            total += n;
        }
    }

    fn keep(ptr: *anyopaque, windows: []const be.Window, paths: []const []const u32) anyerror!void {
        const h = try of(ptr);
        if (windows.len != paths.len) return error.BadKeep;
        for (windows, paths) |w, path| {
            const lane = h.live.getPtr(w.stream) orelse return error.UnknownStream;
            if (path.len == 0 or path.len > lane.verified) return error.EmptyPath;
            for (path, 0..) |row, i| if (row != i) return error.ChainsOnly;
            lane.len = lane.base + path.len;
        }
    }

    fn draft(_: *anyopaque, requests: []const be.DraftRequest) anyerror!void {
        for (requests) |request| if (request.depth != 0 or request.lanes != null) return error.NoDraftHead;
    }

    fn drop(h: *Hip, s: *const Stream) void {
        if (h.live.fetchRemove(s)) |entry| {
            h.free[entry.value.slot] = true;
            h.gpa.free(entry.value.images);
        }
    }

    fn release(ptr: *anyopaque, s: *Stream) void {
        const h: *Hip = @ptrCast(@alignCast(ptr));
        h.drop(s);
    }

    pub fn snapshots(h: *Hip) pc.Snapshots {
        return .{ .ptr = h, .vtable = &.{ .bytes = snapBytes, .save = snapSave, .restore = snapRestore, .drop = snapDrop } };
    }

    fn snapBytes(ptr: *anyopaque, at: u32) u64 {
        const h: *Hip = @ptrCast(@alignCast(ptr));
        return snapshot.bytes(h.m.c, at);
    }

    fn snapSave(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!*anyopaque {
        const h = try of(ptr);
        const s: *const Stream = @ptrCast(@alignCast(owner orelse return error.NoOwner));
        const lane = h.live.getPtr(s) orelse return error.UnknownStream;
        if ((lane.fill orelse lane.len) != at) return error.NotAtMark;
        return @ptrCast(try snapshot.save(h.gpa, h.m, lane.slot, at));
    }

    fn snapRestore(ptr: *anyopaque, owner: ?*anyopaque, saved: *anyopaque) anyerror!void {
        const h = try of(ptr);
        const s: *const Stream = @ptrCast(@alignCast(owner orelse return error.NoOwner));
        const lane = h.live.getPtr(s) orelse return error.UnknownStream;
        const kept: *const snapshot.Saved = @ptrCast(@alignCast(saved));
        try snapshot.restore(h.m, lane.slot, kept);
        lane.len = kept.at;
    }

    fn snapDrop(ptr: *anyopaque, saved: *anyopaque) void {
        const h: *Hip = @ptrCast(@alignCast(ptr));
        h.ctx.makeCurrent() catch {
            std.log.err("cannot select the LightOnOCR HIP context before snapshot release", .{});
            std.c._exit(1);
        };
        snapshot.drop(h.gpa, h.m.s, @ptrCast(@alignCast(saved)));
    }
};
