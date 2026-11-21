//! Inkling 2 Mini lane execution and snapshot adapter.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const engine = @import("loop_engine.zig");
const ops = @import("ops.zig");

const Model = engine.Model;
const Kernels = engine.Kernels;
const prompt_rows = engine.prompt_rows;
const window_rows = engine.window_rows;
const token_ring = 1024;

const Lane = struct {
    len: usize = 0,
    fill: ?usize = null,
    base: usize = 0,
    rows: std.ArrayList(u32) = .empty,
    last: u64,
    rule: u64,
    sampled: bool = false,
};

pub const Backend = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    m: *Model,
    k: *const Kernels,
    s: cuda.Stream,
    lane: ?Lane = null,
    stream: ?*lanes.Stream = null,
    tokens: cuda.DeviceBuffer,
    logits16: cuda.DeviceBuffer,
    meta: cuda.DeviceBuffer,
    slot: [2]cuda.DeviceBuffer,
    next: u64 = 0,

    const be = lanes.backend;

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, m: *Model, k: *const Kernels, s: cuda.Stream) !Backend {
        const v = m.w.c.text.unpadded_vocab;
        var tokens = try cuda.DeviceBuffer.alloc(d, (token_ring + window_rows) * 4);
        errdefer tokens.free();
        var logits16 = try cuda.DeviceBuffer.alloc(d, window_rows * v * 2);
        errdefer logits16.free();
        var meta = try cuda.DeviceBuffer.alloc(d, 16);
        errdefer meta.free();
        var last = try cuda.DeviceBuffer.alloc(d, v * 4);
        errdefer last.free();
        var rule = try cuda.DeviceBuffer.alloc(d, @sizeOf(cuda.sample.Rule));
        errdefer rule.free();
        return .{ .gpa = gpa, .d = d, .m = m, .k = k, .s = s, .tokens = tokens, .logits16 = logits16, .meta = meta, .slot = .{ last, rule } };
    }

    pub fn deinit(b: *Backend) void {
        if (b.lane) |*l| l.rows.deinit(b.gpa);
        for (&b.slot) |*x| x.free();
        b.meta.free();
        b.logits16.free();
        b.tokens.free();
        b.* = undefined;
    }

    pub fn backend(b: *Backend) be.Backend {
        return .{ .ptr = b, .vtable = &vtable };
    }

    pub fn facts(_: *const Backend) lanes.Model {
        return .{ .exact_width = window_rows, .gpu_tokens = false, .mtp = false, .speculate = false, .hidden_rows = true, .batch_rows = prompt_rows, .max_streams = 1, .streams_exact = true };
    }

    pub fn admits(b: *const Backend, prompt: usize, max_new: usize) bool {
        return b.stream == null and prompt <= b.m.context and max_new <= b.m.context - prompt;
    }

    fn self(ptr: *anyopaque) *Backend {
        return @ptrCast(@alignCast(ptr));
    }

    fn upload(b: *Backend, dst: u64, bytes: []const u8) !void {
        try cuda.copy_source.enqueue(b.d, dst, bytes, b.s);
    }

    fn laneOf(b: *Backend, s: *const lanes.Stream) !*Lane {
        if (b.stream != s) return error.UnknownStream;
        return if (b.lane) |*l| l else error.UnknownStream;
    }

    fn resumeSaved(b: *Backend, s: *lanes.Stream) usize {
        const saved = s.reuse.saved orelse return 0;
        const k: *const Model.Saved = @ptrCast(@alignCast(saved));
        if (k.at != s.reuse.at or k.at == 0 or k.at >= s.prompt().len) {
            s.reuse_failed = true;
            return 0;
        }
        b.m.restore(b.s, k) catch {
            s.reuse_failed = true;
            return 0;
        };
        s.cached = k.at;
        return k.at;
    }

    fn opening(b: *Backend, s: *lanes.Stream) !*Lane {
        const prompt = s.prompt();
        if (prompt.len == 0) return error.EmptyPrompt;
        if (prompt.len > b.m.context or s.max_new > b.m.context - prompt.len) return error.PastContext;
        if (b.stream != null and b.stream != s) return error.NoFreeSlot;
        if (b.stream == null) {
            b.lane = .{ .last = b.slot[0].ptr, .rule = b.slot[1].ptr };
            b.stream = s;
        }
        const l = try b.laneOf(s);
        l.sampled = false;
        if (cuda.sample.ruleFor(s.sampling)) |r| {
            try b.upload(l.rule, std.mem.asBytes(&r));
            l.sampled = true;
        }
        l.len = 0;
        try b.m.reset(b.s);
        const at = b.resumeSaved(s);
        l.len = at;
        l.fill = at;
        s.fill_at.store(@intCast(at), .monotonic);
        return l;
    }

    fn chunk(b: *Backend, s: *lanes.Stream, l: *Lane) !bool {
        const prompt = s.prompt();
        const at = l.fill.?;
        if (s.isCancelled()) return error.Cancelled;
        var end = @min(prompt.len, (at / prompt_rows + 1) * prompt_rows);
        for (s.reuse.marks) |mark| {
            if (mark > at and mark < end) end = mark;
        }
        const done = end == prompt.len;
        try b.m.forward(b.k, b.s, prompt[at..end], at, end - at - 1, @intFromBool(done), false);
        if (done) try b.d.check(b.d.api.cuMemcpyDtoDAsync_v2(l.last, b.m.s.logits, b.m.w.c.text.unpadded_vocab * 4, b.s.handle), "cuMemcpyDtoDAsync");
        l.len = end;
        l.fill = if (done) null else end;
        s.fill_at.store(@intCast(end), .monotonic);
        if (s.reuse.hook) |hook| if (std.mem.indexOfScalar(u32, s.reuse.marks, @intCast(end)) != null)
            hook.at(hook.ptr, s, @intCast(end));
        return done;
    }

    fn handle(b: *Backend) usize {
        const h = b.next;
        b.next += 1;
        return @intCast(h % token_ring);
    }

    fn drawRows(b: *Backend, l: *const Lane, logits: u64, rows: usize, position0: u64, at: usize) !void {
        const v = b.m.w.c.text.unpadded_vocab;
        if (!l.sampled) return ops.top1(&b.k.glue, b.s, logits, rows, v, v, 0, b.m.s.t + at * 4, b.tokens.ptr + at * 4);
        try ops.tobf(&b.k.glue, b.s, logits, rows, v, v, b.logits16.ptr, v, 0, v);
        const metadata = [4]i32{ @intCast(position0 - 1), 0, 0, 0 };
        try b.upload(b.meta.ptr, std.mem.asBytes(&metadata));
        try b.k.draws.draw(b.s, b.logits16.ptr, v, l.rule, b.meta.ptr, 0, b.tokens.ptr + at * 4, rows, null, null);
    }

    fn read(ptr: *anyopaque, h: u64) anyerror!u32 {
        const b = self(ptr);
        if (h >= b.next or b.next - h > token_ring) return error.NoSuchToken;
        var out: [1]u32 = undefined;
        try b.s.synchronize();
        try b.tokens.download(@intCast((h % token_ring) * 4), std.mem.sliceAsBytes(&out));
        return out[0];
    }

    const vtable: be.Backend.VTable = .{
        .prefill = prefill,
        .prefill_step = prefillStep,
        .first = first,
        .first_masked = firstMasked,
        .queue = queue,
        .read = read,
        .verify = verify,
        .keep = keep,
        .draft = draft,
        .release = release,
        .reports_fill = true,
    };

    fn prefill(ptr: *anyopaque, s: *lanes.Stream) anyerror!void {
        const b = self(ptr);
        const l = try b.opening(s);
        errdefer release(ptr, s);
        while (!try b.chunk(s, l)) {}
    }

    fn prefillStep(ptr: *anyopaque, s: *lanes.Stream, _: u32) anyerror!bool {
        const b = self(ptr);
        const going = b.stream == s and b.lane != null and b.lane.?.fill != null;
        const l = if (going) try b.laneOf(s) else try b.opening(s);
        errdefer release(ptr, s);
        return b.chunk(s, l);
    }

    fn first(ptr: *anyopaque, s: *lanes.Stream, position: u64) anyerror!u64 {
        const b = self(ptr);
        const l = try b.laneOf(s);
        if (position != l.len or l.fill != null) return error.PositionMismatch;
        const h = b.next;
        try b.drawRows(l, l.last, 1, position, b.handle());
        return h;
    }

    fn firstMasked(_: *anyopaque, _: *lanes.Stream, _: u64, _: []const u32) anyerror!u64 {
        return error.StructuresUnsupported;
    }

    fn feed(b: *Backend, value: be.Feed) !u32 {
        return switch (value) {
            .value => |v| v,
            .handle => |h| read(b, h),
        };
    }

    fn queue(ptr: *anyopaque, s: *lanes.Stream, value: be.Feed, position: u64) anyerror!u64 {
        const b = self(ptr);
        const l = try b.laneOf(s);
        const token = try b.feed(value);
        if (position != l.len + 1) return error.PositionMismatch;
        try b.m.forward(b.k, b.s, &.{token}, l.len, 0, 1, false);
        l.len += 1;
        const h = b.next;
        try b.drawRows(l, b.m.s.logits, 1, position, b.handle());
        return h;
    }

    fn verify(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
        const b = self(ptr);
        if (windows.len != 1 or out.len != 1) return error.OneStreamOnly;
        const w = windows[0];
        if (w.held != 0) return error.NoHeldDrafts;
        if (w.parents != null) return error.ChainsOnly;
        if (w.masks.len != 0) return error.StructuresUnsupported;
        const l = try b.laneOf(w.stream);
        const n = w.rows();
        if (n == 0 or n > window_rows) return error.WindowTooWide;
        l.base = l.len;
        l.rows.clearRetainingCapacity();
        try l.rows.append(b.gpa, w.pending);
        try l.rows.appendSlice(b.gpa, w.tokens);
        for (w.positions, 0..) |position, i| if (position != l.base + i + 1) return error.PositionMismatch;
        try b.m.forward(b.k, b.s, l.rows.items, l.base, 0, n, true);
        try b.drawRows(l, b.m.s.logits, n, w.positions[0], token_ring);
        var got: [window_rows]u32 = undefined;
        try b.s.synchronize();
        try b.tokens.download(token_ring * 4, std.mem.sliceAsBytes(got[0..n]));
        @memcpy(out[0].sampled[0..n], got[0..n]);
        @memcpy(out[0].drafts[0 .. n - 1], l.rows.items[1..n]);
        l.len = l.base + n;
    }

    fn keep(ptr: *anyopaque, windows: []const be.Window, paths: []const []const u32) anyerror!void {
        const b = self(ptr);
        if (windows.len != 1 or paths.len != 1) return error.OneStreamOnly;
        const l = try b.laneOf(windows[0].stream);
        const path = paths[0];
        if (path.len == 0 or path.len > l.rows.items.len) return error.EmptyPath;
        for (path, 0..) |row, i| if (row != i) return error.ChainsOnly;
        try b.m.rewind(b.k, b.s, path.len);
        l.len = l.base + path.len;
    }

    fn draft(_: *anyopaque, requests: []const be.DraftRequest) anyerror!void {
        for (requests) |r| if (r.depth != 0 or r.lanes != null) return error.NoDraftHead;
    }

    pub fn saveState(b: *Backend, owner: ?*anyopaque, at: u32) !*Model.Saved {
        const s: *const lanes.Stream = @ptrCast(@alignCast(owner orelse return error.NoOwner));
        const l = try b.laneOf(s);
        if ((l.fill orelse l.len) != at) return error.NotAtMark;
        return b.m.save(b.s, at);
    }

    pub fn restoreState(b: *Backend, owner: ?*anyopaque, saved: *const Model.Saved) !void {
        const s: *const lanes.Stream = @ptrCast(@alignCast(owner orelse return error.NoOwner));
        const l = try b.laneOf(s);
        try b.m.restore(b.s, saved);
        l.len = saved.at;
    }

    pub fn dropState(b: *Backend, saved: *Model.Saved) void {
        b.m.drop(b.s, saved);
    }

    fn release(ptr: *anyopaque, s: *lanes.Stream) void {
        const b = self(ptr);
        if (b.stream != s) return;
        if (b.lane) |*l| l.rows.deinit(b.gpa);
        b.lane = null;
        b.stream = null;
    }
};
