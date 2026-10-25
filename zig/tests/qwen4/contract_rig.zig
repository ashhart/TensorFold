//! Guarded CPU buffers used only by source-body contract tests.
const std = @import("std");
const a = std.testing.allocator;
pub const HostRig = struct {
    const Slot = struct { bytes: []u8, output: bool };
    slots: std.ArrayList(Slot) = .empty,
    seen: std.ArrayList(u8) = .empty,
    pub fn reset(r: *HostRig) void {
        for (r.slots.items) |slot| a.free(slot.bytes);
        r.slots.clearRetainingCapacity();
    }
    pub fn deinit(r: *HostRig) void {
        r.reset();
        r.slots.deinit(a);
        r.seen.deinit(a);
    }
    pub fn add(r: *HostRig, input: ?[]const u8, len: usize) ![]u8 {
        const all = try a.alloc(u8, len + 128);
        errdefer a.free(all);
        @memset(all, 0xa5);
        const payload = all[64..][0..len];
        if (input) |data| @memcpy(payload, data) else @memset(payload, 0xff);
        try r.slots.append(a, .{ .bytes = all, .output = input == null });
        return payload;
    }
    pub fn inputBytes(r: *HostRig) ![]const u8 {
        r.seen.clearRetainingCapacity();
        for (r.slots.items) |slot| if (!slot.output) try r.seen.appendSlice(a, slot.bytes[64 .. slot.bytes.len - 64]);
        return r.seen.items;
    }
    pub fn outputBytes(r: *HostRig) ![]const u8 {
        const bytes = r.slots.items[r.slots.items.len - 1].bytes;
        return bytes[64 .. bytes.len - 64];
    }
    pub fn intact(r: *HostRig) !bool {
        for (r.slots.items) |slot| {
            for (slot.bytes[0..64]) |byte| if (byte != 0xa5) return false;
            for (slot.bytes[slot.bytes.len - 64 ..]) |byte| if (byte != 0xa5) return false;
        }
        return true;
    }
    pub fn overlap(_: *HostRig) bool {
        return false;
    }
};
