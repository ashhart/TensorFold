//! The actual snapshot allocator, reference counts, byte mapper and drop path against recording Metal.
const std = @import("std");
const mtl = @import("metal");
const snap = @import("families/qwen27/snapshot.zig");
const pages = snap.page_store;
pub const qwen27 = struct {
    pub const snapshot = snap;
    pub const residency = @import("families/qwen27/residency.zig");
};
const a = std.testing.allocator;
const shape: pages.Layout = .{ .fixed = 16, .groups = 4, .row = 4, .tokens = 256 };
const Follow = struct {
    live: usize = 0,
    fn join(p: *anyopaque, _: mtl.Buffer) void {
        const f: *Follow = @ptrCast(@alignCast(p));
        f.live += 1;
    }
    fn leave(p: *anyopaque, _: mtl.Buffer) void {
        const f: *Follow = @ptrCast(@alignCast(p));
        f.live -= 1;
    }
    fn api(f: *Follow) mtl.ResidencySet.Follow {
        return .{ .ctx = f, .join = join, .leave = leave };
    }
};
fn image(device: mtl.Device, at: u32) !mtl.Buffer {
    const b = try device.buffer(16 + at * 16 + 32, mtl.ResourceOptions.private);
    @memset(b.data, @truncate(at));
    for (0..4) |g| for (0..at) |t| {
        const off = 16 + (g * at + t) * 4;
        std.mem.writeInt(u32, b.data[off..][0..4], @intCast(g * 10000 + t), .little);
    };
    return b;
}
fn state(device: mtl.Device, p: *pages.Store, old: ?*snap.State, at: u32) !*snap.State {
    const s = try a.create(snap.State);
    errdefer a.destroy(s);
    s.* = .{ .buf = undefined, .at = at, .bytes = 16 + at * 16 + 32, .draft_end = at, .residency = p.follow };
    try s.allocatePages(device, p, old, shape, 48);
    return s;
}

test "qwen27 snapshot: production paged image survives earlier eviction with exact private state and disk ranges" {
    for ([_][2]u32{ .{ 1, 2 }, .{ 255, 256 }, .{ 256, 257 }, .{ 260, 520 }, .{ 512, 520 } }) |pair| {
        var allocations: mtl.Allocations = .{};
        const d: mtl.Device = .{ .allocations = &allocations };
        var follow: Follow = .{};
        const p = try pages.Store.init(a, 32 * shape.pageBytes(), shape, follow.api());
        defer p.deinit();
        const old = try state(d, p, null, pair[0]);
        var old_live = true;
        defer if (old_live) snap.drop(a, old);
        const first = try image(d, pair[0]);
        defer first.deinit();
        var copied: usize = 0;
        const e: mtl.BlitEncoder = .{ .copied = &copied };
        try old.transfer(e, 0, first, 0, first.length(), true);
        var pending: mtl.Submission = .{};
        old.pending = .{ .submission = &pending };
        const next = try state(.{ .allocations = &allocations, .before = &pending }, p, old, pair[1]);
        defer snap.drop(a, next);
        try std.testing.expectEqual(@as(usize, 1), pending.waits);
        try std.testing.expectEqual(@as(usize, 1), pending.releases);
        const current = try image(d, pair[1]);
        defer current.deinit();
        const shared = next.paged.?.image.shared_pages * shape.pageBytes();
        copied = 0;
        try next.transfer(e, 0, current, 0, current.length(), true);
        try std.testing.expectEqual(current.length() - shared, copied);
        const output = try d.buffer(current.length(), 0);
        defer output.deinit();
        @memset(output.data, 0);
        snap.drop(a, old);
        old_live = false;
        try next.transfer(e, 0, output, 0, output.length(), false);
        try std.testing.expectEqualSlices(u8, current.data, output.data);
        // Disk gather uses these logical ranges too, including ranges spanning page and head boundaries.
        var off: usize = 0;
        while (off < current.length()) : (off += 4) {
            const len = @min(28, current.length() - off);
            @memset(output.data, 0);
            try next.transfer(e, off, output, 0, len, false);
            try std.testing.expectEqualSlices(u8, current.data[off..][0..len], output.data[0..len]);
        }
        try std.testing.expectEqual(@as(usize, next.paged.?.image.ids.len + 1), follow.live);
    }
}

test "qwen27 snapshot: every device allocation refusal returns new buffers and shared references" {
    for (0..3) |failure| {
        var allocations: mtl.Allocations = .{};
        const d: mtl.Device = .{ .allocations = &allocations };
        var follow: Follow = .{};
        const p = try pages.Store.init(a, 32 * shape.pageBytes(), shape, follow.api());
        defer p.deinit();
        const old = try state(d, p, null, 260);
        defer snap.drop(a, old);
        const before = allocations.bytes;
        const resident = follow.live;
        const free = p.pages.pool.available();
        allocations.fail = allocations.calls + failure;
        try std.testing.expectError(error.NoBuffer, state(d, p, old, 520));
        try std.testing.expectEqual(before, allocations.bytes);
        try std.testing.expectEqual(resident, follow.live);
        try std.testing.expectEqual(free, p.pages.pool.available());
        for (old.paged.?.image.ids) |id| try std.testing.expectEqual(@as(u32, 1), p.pages.pool.holders(id));
    }
}

test "qwen27 snapshot: a failed parent copy refuses sharing before any page or buffer allocation" {
    var allocations: mtl.Allocations = .{};
    const d: mtl.Device = .{ .allocations = &allocations };
    const p = try pages.Store.init(a, 32 * shape.pageBytes(), shape, null);
    defer p.deinit();
    const old = try state(d, p, null, 260);
    defer snap.drop(a, old);
    var pending: mtl.Submission = .{ .failed = true };
    old.pending = .{ .submission = &pending };
    const calls = allocations.calls;
    try std.testing.expectError(error.SnapshotGpuFailure, state(d, p, old, 520));
    try std.testing.expectEqual(calls, allocations.calls);
    try std.testing.expectEqual(@as(usize, 1), pending.waits);
    try std.testing.expectEqual(@as(usize, 1), pending.releases);
}

test "qwen27 snapshot: native disk gather reads a paged state across boundaries after parent eviction" {
    var allocations: mtl.Allocations = .{};
    const device: mtl.Device = .{ .allocations = &allocations };
    const store = try pages.Store.init(a, 32 * shape.pageBytes(), shape, null);
    defer store.deinit();
    const first = try state(device, store, null, 260);
    var first_live = true;
    defer if (first_live) snap.drop(a, first);
    const original = try image(device, 260);
    defer original.deinit();
    var copied: usize = 0;
    const encoder: mtl.BlitEncoder = .{ .copied = &copied };
    try first.transfer(encoder, 0, original, 0, original.length(), true);
    const next = try state(device, store, first, 520);
    defer snap.drop(a, next);
    const current = try image(device, 520);
    defer current.deinit();
    try next.transfer(encoder, 0, current, 0, current.length(), true);
    snap.drop(a, first);
    first_live = false;
    var saved: mtl.Submission = .{};
    next.pending = .{ .submission = &saved };
    var submission: mtl.Submission = .{};
    const Resident = struct {
        pub fn add(_: *@This(), _: []const mtl.Buffer) void {}
        pub fn remove(_: *@This(), _: mtl.Buffer) void {}
    };
    var model = .{ .device = device, .queue = mtl.Queue{ .submission = &submission, .copied = &copied, .before = &saved } };
    var owner = .{ .model = &model, .disk_stage = @as(?mtl.Buffer, try device.buffer(128, 0)), .resident = @as(?Resident, null) };
    defer owner.disk_stage.?.deinit();
    @memset(owner.disk_stage.?.data, 0x44);
    const Range = struct { off: u64, len: u64 };
    const ranges = [_]Range{
        .{ .off = 0, .len = 8 },
        .{ .off = 8 + 16 + 256 * 4 - 8, .len = 28 },
        .{ .off = 8 + 16 + 520 * 4 - 12, .len = 28 },
        .{ .off = 8 + 16 + 520 * 16 + 4, .len = 28 },
    };
    var out: [92]u8 = undefined;
    try @import("native/qwen27_disk.zig").gatherFrom(&owner, next, &ranges, &out);
    try std.testing.expectEqual(@as(usize, 1), saved.waits);
    try std.testing.expectEqual(@as(usize, 1), submission.commits);
    try std.testing.expectEqual(@as(usize, 1), submission.waits);
    try std.testing.expectEqualSlices(u8, std.mem.asBytes(&next.draft_end), out[0..8]);
    var at: usize = 8;
    for (ranges[1..]) |range| {
        const off: usize = @intCast(range.off - 8);
        const len: usize = @intCast(range.len);
        try std.testing.expectEqualSlices(u8, current.data[off..][0..len], out[at..][0..len]);
        at += len;
    }
}
