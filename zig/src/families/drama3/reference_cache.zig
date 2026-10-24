//! One live KV prefix per lane, identified exactly through the tower's next-row dependency.
const std = @import("std");
const Frame = @import("model.zig").Frame;

/// Stop on an existing chunk boundary with the tower lookahead inside the invariant prompt, keeping uncached cuts.
pub fn prefillRows(total: usize, reference_rows: usize, chunk: usize) usize {
    if (reference_rows < 2 or reference_rows >= total) return 0;
    std.debug.assert(chunk > 0);
    return (reference_rows - 1) / chunk * chunk;
}

pub const Prefix = struct {
    rows: std.ArrayList(Frame) = .empty, // saved rows plus one tower lookahead row; never borrows a request
    valid: bool = false,

    pub fn deinit(p: *Prefix, a: std.mem.Allocator) void {
        p.rows.deinit(a);
        p.* = .{};
    }

    pub fn invalidate(p: *Prefix) void {
        p.valid = false;
    }

    /// Called before reset/prefill overwrites live KV; a miss invalidates even on allocation failure or cancel.
    pub fn begin(p: *Prefix, a: std.mem.Allocator, frames: []const Frame, saved_rows: usize) !bool {
        if (saved_rows > 0) {
            std.debug.assert(saved_rows < frames.len);
            const identity = frames[0 .. saved_rows + 1];
            if (p.valid and std.mem.eql(u8, std.mem.sliceAsBytes(p.rows.items), std.mem.sliceAsBytes(identity))) return true;
            p.invalidate();
            try p.rows.resize(a, identity.len);
            @memcpy(p.rows.items, identity);
        } else {
            p.invalidate();
            p.rows.clearRetainingCapacity();
        }
        return false;
    }

    /// Only after both the prefix prefill and recurrent-state copy have completed successfully.
    pub fn saved(p: *Prefix) void {
        std.debug.assert(p.rows.items.len >= 2);
        p.valid = true;
    }
};

/// Recurrent state only; KV stays in its lane. Buffers are GPU-owned, reused, and freed on close even if partial.
pub const Snapshot = struct {
    buffers: [2]u64 = .{ 0, 0 },

    pub fn capture(s: *Snapshot, g: anytype, live: [2]u64, bytes: [2]usize) !void {
        try g.flush();
        for (0..2) |i| if (s.buffers[i] == 0) {
            s.buffers[i] = try g.alloc(bytes[i]);
        };
        for (0..2) |i| g.copy(live[i], s.buffers[i], bytes[i]);
        try g.flush();
    }

    pub fn restore(s: *const Snapshot, g: anytype, live: [2]u64, bytes: [2]usize) !void {
        for (0..2) |i| {
            std.debug.assert(s.buffers[i] != 0);
            g.copy(s.buffers[i], live[i], bytes[i]);
        }
        try g.flush();
    }
};

test "reference cache: boundary excludes tower lookahead and preserves original chunk cuts" {
    const t = std.testing;
    try t.expectEqual(@as(usize, 9), prefillRows(20, 10, 1));
    try t.expectEqual(@as(usize, 8), prefillRows(20, 10, 4));
    try t.expectEqual(@as(usize, 0), prefillRows(20, 4, 4));
    for ([_]usize{ 0, 1, 20, 21, std.math.maxInt(usize) }) |boundary|
        try t.expectEqual(@as(usize, 0), prefillRows(20, boundary, 1));
    for ([_]usize{ 1, 2, 4, 16 }) |chunk| {
        for (2..20) |boundary| {
            const rows = prefillRows(20, boundary, chunk);
            try t.expect(rows < boundary);
            try t.expectEqual(@as(usize, 0), rows % chunk);
        }
    }
}

test "reference cache: exact owned identity covers every prefix and tower lookahead cell" {
    const t = std.testing;
    var p: Prefix = .{};
    defer p.deinit(t.allocator);
    const frames: [6]Frame = @splat(@splat(7));
    try t.expect(!try p.begin(t.allocator, &frames, 3));
    p.saved();
    var changed = frames;
    changed[4][0] = 42; // request text beyond the lookahead does not change the saved state
    try t.expect(try p.begin(t.allocator, &changed, 3));
    try t.expect(try p.begin(t.allocator, changed[0..5], 3));
    for (0..4) |row| {
        for (0..11) |col| {
            changed = frames;
            changed[row][col] += 1;
            try t.expect(!try p.begin(t.allocator, &changed, 3));
            p.saved();
            changed[row][col] = frames[row][col]; // the identity must not borrow the caller's buffer
            try t.expect(!try p.begin(t.allocator, &frames, 3));
            p.saved();
        }
    }
    try t.expect(!try p.begin(t.allocator, &frames, 2)); // equal contents, different prefix length
}

test "reference cache: default, partial replacement and allocation errors invalidate before overwrite" {
    const t = std.testing;
    var p: Prefix = .{};
    defer p.deinit(t.allocator);
    const a: [5]Frame = @splat(@splat(1));
    const b: [5]Frame = @splat(@splat(2));
    _ = try p.begin(t.allocator, &a, 3);
    p.saved();
    try t.expect(try p.begin(t.allocator, &a, 3));
    // Cancelling a same-prefix suffix leaves prefix KV and the saved recurrence untouched.
    try t.expect(try p.begin(t.allocator, &a, 3));
    try t.expect(!try p.begin(t.allocator, &b, 0)); // cancellation during uncached/default prefill
    try t.expect(!p.valid); // even before a replacement snapshot can be reached
    try t.expect(!try p.begin(t.allocator, &a, 3));
    p.saved();
    try t.expect(!try p.begin(t.allocator, &b, 3)); // cancelled/error before a replacement snapshot
    try t.expect(!p.valid);
    try t.expect(!try p.begin(t.allocator, &a, 3));
    p.saved();
    const larger: [100]Frame = @splat(@splat(3));
    var failing = std.testing.FailingAllocator.init(t.allocator, .{ .fail_index = 0, .resize_fail_index = 0 });
    try t.expectError(error.OutOfMemory, p.begin(failing.allocator(), &larger, 99));
    try t.expect(!p.valid);
    try t.expect(!try p.begin(t.allocator, &a, 3));
    p.saved();
    p.invalidate(); // an execution error after a hit cannot publish a reusable prefix
    try t.expect(!try p.begin(t.allocator, &a, 3));
}

/// Small CPU buffers exercise the same capture/restore and partial-allocation ownership as Metal.
const TestGpu = struct {
    buffers: std.ArrayList([]u8) = .empty,
    fail_at: ?usize = null,

    pub fn alloc(g: *TestGpu, bytes: usize) !u64 {
        if (g.fail_at == g.buffers.items.len) return error.OutOfMemory;
        const buffer = try std.testing.allocator.alloc(u8, bytes);
        errdefer std.testing.allocator.free(buffer);
        @memset(buffer, 0);
        try g.buffers.append(std.testing.allocator, buffer);
        return @intCast(g.buffers.items.len);
    }

    pub fn flush(_: *TestGpu) !void {}

    pub fn copy(g: *TestGpu, from: u64, to: u64, bytes: usize) void {
        @memcpy(g.buffers.items[@intCast(to - 1)][0..bytes], g.buffers.items[@intCast(from - 1)][0..bytes]);
    }

    fn deinit(g: *TestGpu) void {
        for (g.buffers.items) |b| std.testing.allocator.free(b);
        g.buffers.deinit(std.testing.allocator);
    }
};

test "reference cache: switches reuse one snapshot pair and restore the latest recurrent state" {
    const t = std.testing;
    var g: TestGpu = .{};
    defer g.deinit();
    const bytes: [2]usize = .{ 4, 8 };
    const live: [2]u64 = .{ try g.alloc(bytes[0]), try g.alloc(bytes[1]) };
    var snapshot: Snapshot = .{};
    for (1..33) |voice| {
        const value: u8 = @intCast(voice);
        for (g.buffers.items[0..2]) |buffer| @memset(buffer, value);
        try snapshot.capture(&g, live, bytes);
        try t.expectEqual(@as(usize, 4), g.buffers.items.len);
        for (g.buffers.items[0..2]) |buffer| @memset(buffer, 0);
        try snapshot.restore(&g, live, bytes);
        for (g.buffers.items[0..2]) |buffer| for (buffer) |b| try t.expectEqual(value, b);
    }
}

test "reference cache: partial snapshot allocation stays owned and is reused or freed at close" {
    const t = std.testing;
    for ([_]bool{ false, true }) |retry| {
        var g: TestGpu = .{};
        defer g.deinit();
        const bytes: [2]usize = .{ 4, 8 };
        const live: [2]u64 = .{ try g.alloc(bytes[0]), try g.alloc(bytes[1]) };
        var snapshot: Snapshot = .{};
        g.fail_at = 3;
        try t.expectError(error.OutOfMemory, snapshot.capture(&g, live, bytes));
        try t.expectEqual(@as(usize, 3), g.buffers.items.len);
        try t.expectEqual(@as(u64, 3), snapshot.buffers[0]);
        try t.expectEqual(@as(u64, 0), snapshot.buffers[1]);
        if (retry) {
            g.fail_at = null;
            try snapshot.capture(&g, live, bytes);
            try t.expectEqual(@as(usize, 4), g.buffers.items.len);
            try t.expectEqual(@as(u64, 3), snapshot.buffers[0]);
        }
    }
}
