//! The vision tower's residency refusal, through the production owner's bounded join over a counting fake set.
const std = @import("std");
const metal = @import("metal");
const residency = @import("vision_residency.zig");

const GiB = metal.ResidencySet.GiB;

const FakeBuffer = struct {
    id: *const u8,
    len: u64,

    pub fn length(b: FakeBuffer) u64 {
        return b.len;
    }
};

/// Count buffers admitted to the set and committed rounds.
const Record = struct {
    adds: usize = 0,
    commits: usize = 0,
};

const Set = struct {
    record: *Record,

    pub fn add(s: Set, _: FakeBuffer) void {
        s.record.adds += 1;
    }

    pub fn remove(_: Set, _: FakeBuffer) void {}

    pub fn commit(s: Set) void {
        s.record.commits += 1;
    }
};

const Fake = struct {
    pub const ResidencySet = Set;
    pub const Buffer = FakeBuffer;
    pub const Queue = struct {};
};

const Owner = metal.resident.Owner(Fake);
const Held = metal.resident.Held(Set, FakeBuffer);

/// A served model holding `weights` and its scratch, 256 GiB of RAM: the set may hold 70% of it.
fn serving(record: *Record, ids: []const u8, weights: u64) !Owner {
    const held = (try Held.init(std.testing.allocator, std.testing.io, .{ .record = record }, &.{.{ .id = &ids[0], .len = weights }}, .{ .id = &ids[1], .len = 64 }, "lfm3", 256 * GiB)).?;
    return .{ .held = held, .io = std.testing.io, .work = .{}, .scratch = .{ .id = &ids[1], .len = 64 } };
}

test "LFM3 vision residency: scratch and rows that do not fit the set refuse the open and stay out" {
    const log_level = std.testing.log_level;
    std.testing.log_level = .err;
    defer std.testing.log_level = log_level;
    const ids: [8]u8 = .{ 0, 1, 2, 3, 4, 5, 6, 7 };
    var record: Record = .{};
    var owner = try serving(&record, &ids, 100 * GiB);
    defer owner.held.deinit();
    const before = owner.held.bytes;
    try std.testing.expectEqual(@as(usize, 2), record.adds);
    const scratch = [_]FakeBuffer{ .{ .id = &ids[2], .len = 30 * GiB }, .{ .id = &ids[3], .len = 30 * GiB } };
    var problem: []const u8 = "";
    try std.testing.expectError(error.Lfm3VisionResidencyBudget, residency.admit(std.testing.allocator, &owner, &scratch, FakeBuffer{ .id = &ids[4], .len = 30 * GiB }, &problem));
    // nothing joined: the set's bytes, adds and commits are what the model alone left
    try std.testing.expectEqual(before, owner.held.bytes);
    try std.testing.expectEqual(@as(usize, 2), record.adds);
    try std.testing.expectEqual(@as(usize, 1), record.commits);
}

test "LFM3 vision residency: scratch and rows that fit join together in one committed round" {
    const log_level = std.testing.log_level;
    std.testing.log_level = .err;
    defer std.testing.log_level = log_level;
    const ids: [8]u8 = .{ 0, 1, 2, 3, 4, 5, 6, 7 };
    var record: Record = .{};
    var owner = try serving(&record, &ids, 100 * GiB);
    defer owner.held.deinit();
    const before = owner.held.bytes;
    const scratch = [_]FakeBuffer{ .{ .id = &ids[2], .len = 10 * GiB }, .{ .id = &ids[3], .len = 5 * GiB } };
    var problem: []const u8 = "";
    try residency.admit(std.testing.allocator, &owner, &scratch, FakeBuffer{ .id = &ids[4], .len = GiB }, &problem);
    try std.testing.expectEqual(before + 16 * GiB, owner.held.bytes);
    try std.testing.expectEqual(@as(usize, 5), record.adds);
    try std.testing.expectEqual(@as(usize, 2), record.commits);
}
