//! TF_DSV41_AOT_TRACE=<file>: an AOT set's distinct kernel lookups written there; the set owns and destroys the trace.
const std = @import("std");
const cuda = @import("cuda");
const aot = cuda.aot;

/// The trace file TF_DSV41_AOT_TRACE names, or null when unset or empty.
pub fn pathFromEnv() ?[]const u8 {
    const p = std.mem.span(std.c.getenv("TF_DSV41_AOT_TRACE") orelse return null);
    return if (p.len == 0) null else p;
}

/// A trace to `path` on `set`, which destroys it at deinit; no path attaches none.
pub fn attach(gpa: std.mem.Allocator, io: std.Io, set: *aot.Set, path: ?[]const u8) !void {
    const p = path orelse return;
    if (set.trace != null) return error.TraceAttached;
    set.trace = try aot.Trace.create(gpa, io, p);
}

fn emptySet(gpa: std.mem.Allocator) !aot.Set {
    return .{ .parsed = try aot.parseSpecs(gpa, "{\"kernels\": []}"), .variants = &.{}, .gpa = gpa };
}

test "an attached trace is the set's: deinit destroys it, a second attach is refused and leaks nothing" {
    const t = std.testing;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmp.dir.realPathFileAlloc(t.io, ".", t.allocator);
    defer t.allocator.free(dir);
    const path = try std.fs.path.join(t.allocator, &.{ dir, "trace.jsonl" });
    defer t.allocator.free(path);
    var set = try emptySet(t.allocator);
    try attach(t.allocator, t.io, &set, path);
    try t.expect(set.trace != null);
    try t.expectEqualStrings(path, set.trace.?.path);
    try t.expectError(error.TraceAttached, attach(t.allocator, t.io, &set, path));
    set.deinit(); // the testing allocator fails the test if the trace or its path outlives the set
}

test "a failed attach leaves the set as it was: no trace, nothing leaked, the set still deinits" {
    const t = std.testing;
    var set = try emptySet(t.allocator);
    defer set.deinit();
    for ([_]usize{ 0, 1 }) |at| { // the trace itself, then its path's copy
        var failing = t.FailingAllocator.init(t.allocator, .{ .fail_index = at });
        try t.expectError(error.OutOfMemory, attach(failing.allocator(), t.io, &set, "trace.jsonl"));
        try t.expect(set.trace == null);
        try t.expectEqual(failing.allocated_bytes, failing.freed_bytes);
    }
}

test "no trace path attaches nothing" {
    const t = std.testing;
    var set = try emptySet(t.allocator);
    defer set.deinit();
    try attach(t.allocator, t.io, &set, null);
    try t.expect(set.trace == null);
}
