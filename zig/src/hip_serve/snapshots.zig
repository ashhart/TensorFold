//! Snapshot slots: device buffers holding a stream's state at a prompt cut, in the slots the prompt cache names.

const std = @import("std");
const hip = @import("hip");

pub const Slots = struct {
    gpa: std.mem.Allocator,
    d: *const hip.Runtime,
    /// Each slot's buffer, made on first use and kept for the next snapshot that takes the slot.
    slots: std.ArrayList(?hip.DeviceBuffer) = .empty,
    /// Slots a kept state names (rank 0's bookkeeping; a follower takes the slots rank 0 names).
    used: std.ArrayList(bool) = .empty,

    pub fn init(gpa: std.mem.Allocator, d: *const hip.Runtime) Slots {
        return .{ .gpa = gpa, .d = d };
    }

    pub fn deinit(s: *Slots) void {
        for (s.slots.items) |*b| if (b.*) |*x| x.free();
        s.slots.deinit(s.gpa);
        s.used.deinit(s.gpa);
    }

    /// The lowest free slot, now in use (a new one past the end when every slot holds a state).
    pub fn claim(s: *Slots) !u32 {
        for (s.used.items, 0..) |u, i| if (!u) {
            s.used.items[i] = true;
            return @intCast(i);
        };
        try s.used.append(s.gpa, true);
        return @intCast(s.used.items.len - 1);
    }

    /// Slot `slot` free again; its buffer stays for the next state.
    pub fn drop(s: *Slots, slot: u32) void {
        if (slot < s.used.items.len) s.used.items[slot] = false;
    }

    /// Slot `slot`'s buffer of `bytes`, made on first use.
    pub fn buffer(s: *Slots, slot: u32, bytes: usize) !hip.DeviceBuffer {
        while (s.slots.items.len <= slot) try s.slots.append(s.gpa, null);
        const entry = &s.slots.items[slot];
        if (entry.* == null) entry.* = try hip.DeviceBuffer.alloc(s.d, bytes);
        return entry.*.?;
    }

    /// Slot `slot`'s buffer, which a state was taken into before.
    pub fn get(s: *const Slots, slot: u32) !hip.DeviceBuffer {
        return (if (slot < s.slots.items.len) s.slots.items[slot] else null) orelse error.NoSnapshot;
    }

    /// Frees every slot's buffer (the cache keeps no snapshots).
    pub fn clear(s: *Slots) void {
        for (s.slots.items) |*b| if (b.*) |*x| x.free();
        s.slots.clearRetainingCapacity();
        @memset(s.used.items, false);
    }

    /// Bytes the slots hold.
    pub fn held(s: *const Slots) usize {
        var n: usize = 0;
        for (s.slots.items) |b| n += if (b) |x| x.len else 0;
        return n;
    }
};

test "the lowest free slot is claimed, and a dropped slot is claimed again before a new one" {
    var r: hip.Runtime = undefined;
    var s = Slots.init(std.testing.allocator, &r);
    defer s.deinit();
    try std.testing.expectEqual(@as(u32, 0), try s.claim());
    try std.testing.expectEqual(@as(u32, 1), try s.claim());
    s.drop(0);
    try std.testing.expectEqual(@as(u32, 0), try s.claim());
    try std.testing.expectEqual(@as(u32, 2), try s.claim());
    try std.testing.expectError(error.NoSnapshot, s.get(5));
    try std.testing.expectEqual(@as(usize, 0), s.held());
}
