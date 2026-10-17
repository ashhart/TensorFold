//! Own copy sources through completion. Failed disposal moves them to context/stream quarantine.
const std = @import("std");
const abi = @import("abi.zig");
const Driver = @import("driver.zig").Driver;
const memory = @import("memory.zig");
const streams = @import("stream.zig");
const budget = @import("copy_source_budget.zig");
const durable = std.heap.page_allocator;
const capture_scope = @import("copy_source_capture.zig");
threadlocal var current_context: abi.Context = null;
var mutex: std.c.pthread_mutex_t = .{};
var changed: std.c.pthread_cond_t = .{};
var head: ?*Node = null;
var next_ticket: u64 = 1;
const Node = struct {
    charge: usize = @sizeOf(Node),
    next: ?*Node = null,
    work_next: ?*Node = null,
    busy: bool = false,
    ticket: u64 = 0,
    driver: usize = 0,
    context: abi.Context = null,
    stream: abi.Stream = null,
    failed: bool = false,
    data: union(enum) {
        none,
        cpu: []align(16) u8,
        pinned: struct {
            bytes: []align(16) u8,
            free: @FieldType(abi.Api, "cuMemFreeHost"),
            event: ?abi.Event,
            destroy: ?@FieldType(abi.Api, "cuEventDestroy_v2"),
            query: ?@FieldType(abi.Api, "cuEventQuery"),
        },
    } = .none,
    fn releaseData(n: *Node) void {
        const a = durable;
        switch (n.data) {
            .none => {},
            .cpu => |bytes| a.free(bytes),
            .pinned => |p| {
                if (p.destroy) |drop| _ = drop(p.event.?);
                _ = p.free(p.bytes.ptr);
                memory.releaseRetainedHost(p.bytes.len);
            },
        }
    }
    fn destroy(n: *Node) void {
        const charge = n.charge;
        durable.destroy(n);
        budget.release(charge, 1);
    }
    fn release(n: *Node) void {
        n.releaseData();
        n.destroy();
    }
};
fn newNode() !*Node {
    try budget.acquire(@sizeOf(Node), 1);
    errdefer budget.release(@sizeOf(Node), 1);
    const n = try durable.create(Node);
    n.* = .{};
    return n;
}
fn lock() void {
    _ = std.c.pthread_mutex_lock(&mutex);
}
fn unlock() void {
    _ = std.c.pthread_mutex_unlock(&mutex);
}
pub fn setCurrent(context: abi.Context) void {
    current_context = context;
}
pub fn current() abi.Context {
    return current_context;
}
fn cutoff() u64 {
    lock();
    defer unlock();
    return next_ticket;
}
fn publish(n: *Node) void {
    lock();
    defer unlock();
    n.ticket = next_ticket;
    next_ticket +%= 1;
    n.next = head;
    head = n;
}
// Busy owners stay registered until every CUDA callback has returned.
fn releaseTracked(n: *Node) void {
    n.releaseData();
    lock();
    var at = &head;
    while (at.* != n) at = &at.*.?.next;
    at.* = n.next;
    _ = std.c.pthread_cond_broadcast(&changed);
    unlock();
    n.destroy();
}
fn completed(d: *const Driver, context: abi.Context, stream: abi.Stream, all: bool, before: u64) void {
    var released: ?*Node = null;
    lock();
    var at = head;
    while (at) |n| {
        if (!n.busy and n.driver == @intFromPtr(d) and n.context == context and (all or n.stream == stream) and n.ticket < before) {
            n.busy = true;
            n.work_next = released;
            released = n;
        }
        at = n.next;
    }
    unlock();
    while (released) |n| {
        released = n.work_next;
        releaseTracked(n);
    }
}
fn hasRetained(d: *const Driver, context: abi.Context) bool {
    lock();
    defer unlock();
    var at = head;
    while (at) |n| {
        if (n.driver == @intFromPtr(d) and n.context == context and n.failed) return true;
        at = n.next;
    }
    return false;
}
pub fn captureBegin(stream: streams.Stream, mode: abi.CaptureMode) !void {
    try capture_scope.begin(stream, current_context, mode);
}
pub fn captureEnd(stream: streams.Stream) void {
    capture_scope.end(stream, current_context);
}
pub fn keepDriver(d: *const Driver) bool {
    lock();
    defer unlock();
    while (true) {
        var kept = false;
        var busy = false;
        var at = head;
        while (at) |n| {
            if (n.driver == @intFromPtr(d)) {
                kept = true;
                busy = busy or n.busy;
            }
            at = n.next;
        }
        if (!busy) return kept;
        _ = std.c.pthread_cond_wait(&changed, &mutex);
    }
}
pub fn retained() usize {
    lock();
    defer unlock();
    var n: usize = 0;
    var at = head;
    while (at) |p| {
        n += 1;
        at = p.next;
    }
    return n;
}
pub fn synchronizeStream(d: *const Driver, stream: abi.Stream) @import("driver.zig").Error!void {
    const before = cutoff();
    const context = current_context;
    d.check(d.api.cuStreamSynchronize(stream), "cuStreamSynchronize") catch |e| {
        failed(d, context, stream, false, before);
        return e;
    };
    completed(d, context, stream, false, before);
}
pub fn synchronizeContext(d: *const Driver) @import("driver.zig").Error!void {
    const before = cutoff();
    const context = current_context;
    d.check(d.api.cuCtxSynchronize(), "cuCtxSynchronize") catch |e| {
        failed(d, context, null, true, before);
        return e;
    };
    completed(d, context, null, true, before);
}
pub fn streamDone(d: *const Driver, stream: abi.Stream) @import("driver.zig").Error!bool {
    const before = cutoff();
    const context = current_context;
    d.check(d.api.cuStreamQuery(stream), "cuStreamQuery") catch |e| switch (e) {
        error.NotReady => return false,
        else => return e,
    };
    completed(d, context, stream, false, before);
    return true;
}
/// The allocator argument is a compatibility hint; escaped custody never borrows it.
pub fn synchronousCopy(d: *const Driver, dst: u64, bytes: []const u8, gpa: std.mem.Allocator) @import("driver.zig").Error!void {
    if (capture_scope.active(d, current_context)) return error.UnownedCaptureSource;
    var at: usize = 0;
    while (at < bytes.len) {
        const n = @min(bytes.len - at, 64 << 20);
        try syncChunk(d, std.math.add(u64, dst, at) catch return error.Invalid, bytes[at..][0..n], gpa);
        at += n;
    }
}
fn syncChunk(d: *const Driver, dst: u64, bytes: []const u8, _: std.mem.Allocator) @import("driver.zig").Error!void {
    if (bytes.len == 0) return;
    if (hasRetained(d, current_context)) try synchronizeContext(d);
    const n = try newNode();
    var transferred = false;
    defer if (!transferred) n.release();
    try budget.acquire(bytes.len, 0);
    n.charge += bytes.len;
    const owned = try durable.alignedAlloc(u8, .@"16", bytes.len);
    n.driver = @intFromPtr(d);
    n.context = current_context;
    n.data = .{ .cpu = owned };
    @memcpy(owned, bytes);
    const result = d.api.cuMemcpyHtoD_v2(dst, owned.ptr, owned.len);
    synchronizeStream(d, null) catch |e| {
        transferred = true;
        n.failed = true;
        publish(n);
        return e;
    };
    try d.check(result, "cuMemcpyHtoD");
}

/// Reusable pinned storage. take waits before mutation; close waits before release; deinit quarantines on failure.
pub const Stage = struct {
    // Compatibility hint only: custody storage always uses the durable core allocator.
    gpa: std.mem.Allocator = std.heap.page_allocator,
    buf: ?memory.HostBuffer = null,
    ev: ?streams.Event = null,
    pending: bool = false,
    stream: ?streams.Stream = null,
    context: abi.Context = null,
    event_valid: bool = false,
    used: usize = 0,
    holder: ?*Node = null,
    /// A different current context leaves the saved owner pending for retry or quarantine.
    pub fn wait(s: *Stage) !void {
        if (!s.pending) return;
        if (s.context != current_context) return error.StageBusy;
        if (s.event_valid) try s.ev.?.synchronize() else try s.stream.?.synchronize();
        s.pending = false;
        s.event_valid = false;
    }
    pub fn take(s: *Stage, d: *const Driver, bytes: usize) ![]u8 {
        try s.wait();
        if (s.holder == null) {
            s.holder = try newNode();
        }
        if (s.buf == null or s.buf.?.bytes.len < bytes) {
            if (s.buf) |*b| {
                const len = b.bytes.len;
                b.free();
                budget.release(len, 0);
                s.holder.?.charge -= len;
            }
            s.buf = null;
            const len = @max(bytes, 4096);
            try budget.acquire(len, 0);
            errdefer budget.release(len, 0);
            s.buf = try memory.HostBuffer.alloc(d, len);
            s.holder.?.charge += len;
        }
        s.used = bytes;
        return s.buf.?.bytes[0..bytes];
    }
    pub fn issued(s: *Stage, d: *const Driver, stream: streams.Stream) !void {
        if (s.holder == null) return error.StageUnprepared;
        if (s.pending and s.context != current_context) return error.StageBusy;
        s.stream = stream;
        s.context = current_context;
        s.pending = true;
        s.event_valid = false;
        if (s.ev == null) s.ev = try streams.Event.init(d, false);
        try s.ev.?.record(stream);
        s.event_valid = true;
    }
    pub fn begin(s: *Stage, stream: streams.Stream) !void {
        if (s.buf == null or s.holder == null) return error.StageUnprepared;
        if (s.pending and (s.stream.?.d != stream.d or s.stream.?.handle != stream.handle or s.context != current_context)) try s.wait();
        s.stream = stream;
        s.context = current_context;
        s.pending = true;
        s.event_valid = false;
    }
    pub fn copyRangeTo(s: *Stage, destination: memory.DeviceBuffer, offset: usize, from: usize, len: usize, stream: streams.Stream) !void {
        if (s.buf == null or from > s.used or len > s.used - from) return error.Invalid;
        if (len == 0) return;
        try s.begin(stream);
        const dst = try destination.at(offset);
        if (len > destination.len - offset) return error.Invalid;
        try destination.d.check(destination.d.api.cuMemcpyHtoDAsync_v2(dst, s.buf.?.bytes[from..].ptr, len, stream.handle), "cuMemcpyHtoDAsync");
        try s.issued(destination.d, stream);
    }
    pub fn copyTo(s: *Stage, destination: memory.DeviceBuffer, offset: usize, stream: streams.Stream) !void {
        try s.copyRangeTo(destination, offset, 0, s.used, stream);
    }
    pub fn close(s: *Stage) !void {
        try s.wait();
        if (s.ev) |*e| e.deinit();
        if (s.buf) |*b| b.free();
        if (s.holder) |n| n.release();
        s.* = .{};
    }
    pub fn abandon(s: *Stage) void {
        s.retire(false);
    }
    fn retire(s: *Stage, bad: bool) void {
        if (!s.pending) {
            s.close() catch unreachable;
            return;
        }
        const n = s.holder.?;
        const b = s.buf.?;
        n.* = .{ .charge = n.charge, .failed = bad, .driver = @intFromPtr(s.stream.?.d), .context = s.context, .stream = s.stream.?.handle, .data = .{ .pinned = .{
            .bytes = b.bytes,
            .free = b.d.api.cuMemFreeHost,
            .event = if (s.ev) |e| e.handle else null,
            .destroy = if (s.ev) |e| e.d.api.cuEventDestroy_v2 else null,
            .query = if (s.event_valid) s.ev.?.d.api.cuEventQuery else null,
        } } };
        publish(n);
        s.* = .{};
    }
    pub fn deinit(s: *Stage) void {
        s.close() catch s.retire(true);
    }
};
/// Stable pinned storage. Graph owners keep it until graph disposal and mark each replay before release.
pub const Buffer = struct {
    d: *const Driver,
    bytes: []align(16) u8,
    stage: Stage,
    pub fn alloc(d: *const Driver, len: usize) !Buffer {
        return allocKind(d, len, false);
    }
    pub fn allocMapped(d: *const Driver, len: usize) !Buffer {
        return allocKind(d, len, true);
    }
    fn allocKind(d: *const Driver, len: usize, mapped: bool) !Buffer {
        var s: Stage = .{};
        errdefer s.deinit();
        s.holder = try newNode();
        try budget.acquire(len, 0);
        errdefer budget.release(len, 0);
        s.buf = if (mapped) try memory.HostBuffer.allocMapped(d, len) else try memory.HostBuffer.alloc(d, len);
        s.holder.?.charge += len;
        s.used = len;
        return .{ .d = d, .bytes = s.buf.?.bytes, .stage = s };
    }
    pub fn slice(b: Buffer, comptime T: type) []T {
        return std.mem.bytesAsSlice(T, b.bytes[0 .. b.bytes.len / @sizeOf(T) * @sizeOf(T)]);
    }
    pub fn device(b: Buffer) !abi.DevicePtr {
        return b.stage.buf.?.device();
    }
    pub fn wait(b: *Buffer) !void {
        try b.stage.wait();
    }
    pub fn mark(b: *Buffer, stream: streams.Stream) !void {
        try b.stage.begin(stream);
    }
    pub fn copyTo(b: *Buffer, destination: memory.DeviceBuffer, offset: usize, bytes: []const u8, stream: streams.Stream) !void {
        const start = @intFromPtr(bytes.ptr);
        const base = @intFromPtr(b.bytes.ptr);
        if (start < base or start - base > b.bytes.len or bytes.len > b.bytes.len - (start - base)) return error.Invalid;
        try b.stage.copyRangeTo(destination, offset, start - base, bytes.len, stream);
    }
    pub fn free(b: *Buffer) void {
        b.stage.deinit();
        b.* = undefined;
    }
};

/// Temporary inputs transfer ownership to quarantine immediately; stream completion releases the private copy.
pub fn enqueue(d: *const Driver, dst: u64, bytes: []const u8, stream: streams.Stream) @import("driver.zig").Error!void {
    var at: usize = 0;
    while (at < bytes.len) {
        const n = @min(bytes.len - at, 64 << 20);
        try asyncChunk(d, std.math.add(u64, dst, at) catch return error.Invalid, bytes[at..][0..n], stream);
        at += n;
        if (at < bytes.len) try stream.synchronize();
    }
}
fn asyncChunk(d: *const Driver, dst: u64, bytes: []const u8, stream: streams.Stream) @import("driver.zig").Error!void {
    if (bytes.len == 0) return;
    if (capture_scope.active(d, current_context)) return error.UnownedCaptureSource;
    reap(d);
    if (hasRetained(d, current_context)) try synchronizeContext(d);
    var s: Stage = .{};
    defer s.deinit();
    @memcpy(try s.take(d, bytes.len), bytes);
    s.copyTo(.{ .d = d, .ptr = dst, .len = bytes.len }, 0, stream) catch |e| {
        if (s.pending) s.retire(true);
        return e;
    };
    s.abandon();
}
fn failed(d: *const Driver, context: abi.Context, stream: abi.Stream, all: bool, before: u64) void {
    lock();
    defer unlock();
    var at = head;
    while (at) |n| {
        if (n.driver == @intFromPtr(d) and n.context == context and (all or n.stream == stream) and n.ticket < before)
            n.failed = true;
        at = n.next;
    }
}
fn reap(d: *const Driver) void {
    var pending: ?*Node = null;
    lock();
    var at = head;
    while (at) |n| {
        if (!n.busy and n.driver == @intFromPtr(d) and n.context == current_context and n.data == .pinned and n.data.pinned.query != null) {
            n.busy = true;
            n.work_next = pending;
            pending = n;
        }
        at = n.next;
    }
    unlock();
    while (pending) |n| {
        pending = n.work_next;
        const p = n.data.pinned;
        const result = p.query.?(p.event.?);
        if (result == 0) releaseTracked(n) else {
            lock();
            if (result != abi.error_not_ready) n.failed = true;
            n.busy = false;
            _ = std.c.pthread_cond_broadcast(&changed);
            unlock();
        }
    }
}

test {
    _ = @import("copy_source_test.zig");
}

test "driver close waits through reaper query, event destruction and host release" {
    try @import("copy_source_race_test.zig").run(reap, false);
}
test "driver close waits through completion release callbacks" {
    try @import("copy_source_race_test.zig").run(reap, true);
}
test {
    _ = @import("copy_source_context_test.zig");
}

test {
    _ = @import("copy_source_limits_test.zig");
}

test {
    _ = @import("copy_source_capture_test.zig");
}
