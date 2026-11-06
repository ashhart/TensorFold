//! Opt-in, thread-local pass receipts; scopes partition host time without splitting GPU command buffers.
const std = @import("std");
pub const Phase = enum { planning, drafter, selector, verify, sampling, keep };
pub const Bucket = struct { ms: f64 = 0, wait_ms: f64 = 0, gpu_ms: f64 = 0, waits: u64 = 0, commits: u64 = 0, allocations: u64 = 0, bytes: u64 = 0, buffers: u64 = 0 };
threadlocal var active: bool = false;
threadlocal var clock: ?*const fn () f64 = null;
threadlocal var buckets: [6]Bucket = @splat(.{});
threadlocal var current: Phase = .planning;
threadlocal var changed: f64 = 0;
threadlocal var started: f64 = 0;
threadlocal var mode: []const u8 = "";
threadlocal var width: u32 = 0;
threadlocal var verified_rows: u32 = 0;
pub fn rows(n: usize) void {
    if (active) verified_rows = @intCast(n);
}
pub fn time() f64 {
    return if (active) clock.?() else 0;
}
fn charge() void {
    if (active) {
        const now = time();
        buckets[@backingInt(current)].ms += (now - changed) * 1000;
        changed = now;
    }
}
pub fn begin(enabled: bool, label: []const u8, window: u32, now: *const fn () f64) void {
    if (!enabled) return;
    if (active) end();
    active = true;
    clock = now;
    buckets = @splat(.{});
    current = .planning;
    mode = label;
    width = window;
    verified_rows = 0;
    started = time();
    changed = started;
}
pub const Scope = struct {
    previous: ?Phase = null,
    pub fn end(s: Scope) void {
        const previous = s.previous orelse return;
        if (active) {
            charge();
            current = previous;
        }
    }
};
pub fn phase(p: Phase) Scope {
    if (!active) return .{};
    const old = current;
    if (active) {
        charge();
        current = p;
    }
    return .{ .previous = old };
}
pub fn wait(start: f64, gpu_seconds: f64) void {
    if (!active) return;
    const b = &buckets[@backingInt(current)];
    b.waits += 1;
    b.wait_ms += (time() - start) * 1000;
    b.gpu_ms += gpu_seconds * 1000;
}
pub fn commit() void {
    if (active) buckets[@backingInt(current)].commits += 1;
}
pub fn buffer() void {
    if (active) buckets[@backingInt(current)].buffers += 1;
}
pub fn allocation(bytes: usize) void {
    if (active) {
        const b = &buckets[@backingInt(current)];
        b.allocations += 1;
        b.bytes += bytes;
    }
}
pub fn end() void {
    if (!active) return;
    charge();
    const total = (time() - started) * 1000;
    active = false;
    for (buckets, 0..) |b, i| std.log.info("q27_pass loop={s} width={d} rows={d} phase={s} ms={d:.6} wait_ms={d:.6} gpu_ms={d:.6} waits={d} commits={d} allocations={d} bytes={d} buffers={d} total_ms={d:.6}", .{ mode, width, verified_rows, @tagName(@as(Phase, @fromBackingInt(@intCast(i)))), b.ms, b.wait_ms, b.gpu_ms, b.waits, b.commits, b.allocations, b.bytes, b.buffers, total });
}
/// The host owns this forwarding allocator for the whole serving lifetime, including escaped round plans.
pub const Counting = struct {
    parent: std.mem.Allocator,
    pub fn allocator(c: *Counting) std.mem.Allocator {
        return .{ .ptr = c, .vtable = &.{ .alloc = alloc, .resize = resize, .remap = remap, .free = free } };
    }
    fn self(p: *anyopaque) *Counting {
        return @ptrCast(@alignCast(p));
    }
    fn alloc(p: *anyopaque, n: usize, a: std.mem.Alignment, ret: usize) ?[*]u8 {
        const out = self(p).parent.rawAlloc(n, a, ret);
        if (out != null) allocation(n);
        return out;
    }
    fn resize(p: *anyopaque, b: []u8, a: std.mem.Alignment, n: usize, ret: usize) bool {
        return self(p).parent.rawResize(b, a, n, ret);
    }
    fn remap(p: *anyopaque, b: []u8, a: std.mem.Alignment, n: usize, ret: usize) ?[*]u8 {
        const out = self(p).parent.rawRemap(b, a, n, ret);
        if (out != null) allocation(n);
        return out;
    }
    fn free(p: *anyopaque, b: []u8, a: std.mem.Alignment, ret: usize) void {
        self(p).parent.rawFree(b, a, ret);
    }
};

var test_time: f64 = 1;
fn testClock() f64 {
    return test_time;
}
test "pass metrics: nested scopes count actual allocations and partition waits without extra synchronization" {
    var counted = Counting{ .parent = std.testing.allocator };
    const a = counted.allocator();
    begin(true, "test", 16, testClock);
    const b = try a.alloc(u8, 17);
    defer a.free(b);
    test_time += 0.001;
    const scope = phase(.verify);
    commit();
    buffer();
    const start = time();
    test_time += 0.003;
    wait(start, 0.002);
    scope.end();
    test_time += 0.001;
    charge();
    try std.testing.expectEqual(@as(u64, 1), buckets[0].allocations);
    try std.testing.expectEqual(@as(u64, 17), buckets[0].bytes);
    const verify = buckets[@backingInt(Phase.verify)];
    try std.testing.expectEqual(@as(u64, 1), verify.waits);
    try std.testing.expectEqual(@as(u64, 1), verify.commits);
    try std.testing.expectEqual(@as(u64, 1), verify.buffers);
    try std.testing.expectApproxEqAbs(@as(f64, 3), verify.wait_ms, 1e-8);
    try std.testing.expectApproxEqAbs(@as(f64, 2), verify.gpu_ms, 1e-8);
    try std.testing.expectApproxEqAbs(@as(f64, 2), buckets[0].ms, 1e-8);
    end();
}
test "pass metrics: disabled scopes neither call the clock nor count work" {
    const previous = buckets;
    begin(false, "off", 16, testClock);
    const scope = phase(.verify);
    allocation(17);
    commit();
    buffer();
    wait(0, 1);
    scope.end();
    end();
    try std.testing.expectEqualDeep(previous, buckets);
}

test "pass metrics: a deferred keep may close the previous pass before the next proposal" {
    begin(true, "first", 16, testClock);
    allocation(9);
    begin(true, "next", 32, testClock);
    try std.testing.expectEqual(@as(u32, 32), width);
    try std.testing.expectEqual(@as(u64, 0), buckets[0].allocations);
    end();
}

test "pass metrics: a disabled scope cannot charge or redirect a later recording" {
    const idle = phase(.keep);
    begin(true, "later", 16, testClock);
    const verify = phase(.verify);
    test_time += 0.001;
    idle.end();
    test_time += 0.002;
    verify.end();
    try std.testing.expectApproxEqAbs(@as(f64, 3), buckets[@backingInt(Phase.verify)].ms, 1e-8);
    try std.testing.expectEqual(@as(f64, 0), buckets[@backingInt(Phase.keep)].ms);
    end();
}
