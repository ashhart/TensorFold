const std = @import("std");
const Config = @import("config.zig").Config;

pub const ObjectId = u64;
pub const Slot = struct { bucket: usize, slot: usize, epoch: u64 };

pub const Slots = struct {
    allocator: std.mem.Allocator,
    config: Config,
    ids: []?ObjectId,
    epochs: []u64,

    pub fn init(allocator: std.mem.Allocator, config: Config, capacity: usize) !Slots {
        const count = try config.buckets(capacity);
        const ids = try allocator.alloc(?ObjectId, capacity);
        errdefer allocator.free(ids);
        const epochs = try allocator.alloc(u64, count);
        @memset(ids, null);
        @memset(epochs, 0);
        return .{ .allocator = allocator, .config = config, .ids = ids, .epochs = epochs };
    }

    pub fn deinit(self: *Slots) void {
        self.allocator.free(self.ids);
        self.allocator.free(self.epochs);
        self.* = undefined;
    }

    pub fn get(self: Slots, id: ObjectId) ?Slot {
        for (self.ids, 0..) |value, i| if (value == id) return self.at(i);
        return null;
    }

    fn at(self: Slots, i: usize) Slot {
        const bucket = i / self.config.bucket_size;
        return .{ .bucket = bucket, .slot = i % self.config.bucket_size, .epoch = self.epochs[bucket] };
    }

    fn bump(self: *Slots, bucket: usize) !void {
        self.epochs[bucket] = std.math.add(u64, self.epochs[bucket], 1) catch return error.EpochOverflow;
    }

    pub fn add(self: *Slots, id: ObjectId) !Slot {
        if (self.get(id)) |slot| return slot;
        for (self.ids, 0..) |value, i| if (value == null) {
            try self.bump(i / self.config.bucket_size);
            self.ids[i] = id;
            return self.at(i);
        };
        return error.TooManyObjects;
    }

    pub fn remove(self: *Slots, id: ObjectId) !Slot {
        const slot = self.get(id) orelse return error.UnknownObject;
        try self.bump(slot.bucket);
        self.ids[slot.bucket * self.config.bucket_size + slot.slot] = null;
        return .{ .bucket = slot.bucket, .slot = slot.slot, .epoch = self.epochs[slot.bucket] };
    }

    pub fn current(self: Slots, slot: Slot) bool {
        if (slot.bucket >= self.epochs.len or slot.slot >= self.config.bucket_size) return false;
        const i = slot.bucket * self.config.bucket_size + slot.slot;
        return i < self.ids.len and self.ids[i] != null and self.epochs[slot.bucket] == slot.epoch;
    }
};

test "SAM multiplex assigns stable IDs across buckets and invalidates removed-slot memory" {
    var slots = try Slots.init(std.testing.allocator, Config.of(.sam31), 17);
    defer slots.deinit();
    for (0..17) |id| {
        const slot = try slots.add(id);
        try std.testing.expectEqual(id / 16, slot.bucket);
        try std.testing.expectEqual(id % 16, slot.slot);
    }
    const last = slots.get(16).?;
    const first = slots.get(0).?;
    try std.testing.expect(slots.current(last));
    _ = try slots.remove(0);
    const replacement = try slots.add(99);
    try std.testing.expectEqual(first.slot, replacement.slot);
    try std.testing.expect(!slots.current(first));
    try std.testing.expect(slots.current(last));
    try std.testing.expectError(error.TooManyObjects, slots.add(100));
    try std.testing.expectError(error.UnknownObject, slots.remove(100));
}

test "SAM3 keeps each tracked object in an independent memory bucket" {
    var slots = try Slots.init(std.testing.allocator, Config.of(.sam3), 2);
    defer slots.deinit();
    const first = try slots.add(123);
    const second = try slots.add(456);
    try std.testing.expectEqual(@as(usize, 0), first.bucket);
    try std.testing.expectEqual(@as(usize, 1), second.bucket);
    try std.testing.expect(slots.current(first));
    try std.testing.expectEqual(second, try slots.add(456));
}
