//! The contract's host buffers: inputs copied in, outputs poisoned, every payload between guard bytes.
const std = @import("std");

pub const guard = 64;
const pattern = 0xa5;

const Kind = enum { input, output, scratch };
const Slot = struct { all: []align(guard) u8, len: usize, kind: Kind };

/// One call's buffers on the host: each payload its own allocation, guarded on both sides.
pub const Rig = struct {
    a: std.mem.Allocator,
    slots: std.ArrayList(Slot) = .empty,
    seen: std.ArrayList(u8) = .empty,
    outs: std.ArrayList(u8) = .empty,

    pub fn deinit(r: *Rig) void {
        r.reset();
        r.slots.deinit(r.a);
        r.seen.deinit(r.a);
        r.outs.deinit(r.a);
    }

    pub fn reset(r: *Rig) void {
        for (r.slots.items) |s| r.a.free(s.all);
        r.slots.clearRetainingCapacity();
    }

    fn slot(r: *Rig, comptime T: type, count: usize, kind: Kind) ![]T {
        const len = count * @sizeOf(T);
        const all = try r.a.alignedAlloc(u8, .fromByteUnits(guard), len + 2 * guard);
        errdefer r.a.free(all);
        @memset(all, pattern);
        try r.slots.append(r.a, .{ .all = all, .len = len, .kind = kind });
        return @alignCast(std.mem.bytesAsSlice(T, all[guard..][0..len]));
    }

    /// An input holding `data`.
    pub fn input(r: *Rig, comptime T: type, data: []const T) ![]T {
        const payload = try r.slot(T, data.len, .input);
        @memcpy(payload, data);
        return payload;
    }

    /// An output of `count` elements, every byte 0xff: an element no run writes reads as NaN.
    pub fn output(r: *Rig, comptime T: type, count: usize) ![]T {
        const payload = try r.slot(T, count, .output);
        @memset(std.mem.sliceAsBytes(payload), 0xff);
        return payload;
    }

    /// Working memory a call fills before its sums: neither an input nor an output.
    pub fn scratch(r: *Rig, comptime T: type, count: usize) ![]T {
        return r.slot(T, count, .scratch);
    }

    fn gather(r: *Rig, kind: Kind, out: *std.ArrayList(u8)) ![]const u8 {
        out.clearRetainingCapacity();
        for (r.slots.items) |s| if (s.kind == kind) try out.appendSlice(r.a, s.all[guard..][0..s.len]);
        return out.items;
    }

    pub fn outputBytes(r: *Rig) ![]const u8 {
        return r.gather(.output, &r.outs);
    }

    pub fn inputBytes(r: *Rig) ![]const u8 {
        return r.gather(.input, &r.seen);
    }

    /// Every guard of every payload still holds its pattern.
    pub fn intact(r: *Rig) !bool {
        for (r.slots.items) |s| {
            for (s.all[0..guard]) |v| if (v != pattern) return false;
            for (s.all[guard + s.len ..]) |v| if (v != pattern) return false;
        }
        return true;
    }

    /// Each payload has an allocation of its own, so an overlap means one allocation listed as two kinds.
    pub fn overlap(r: *Rig) bool {
        for (r.slots.items, 0..) |s, i| for (r.slots.items[i + 1 ..]) |t| {
            if (s.all.ptr == t.all.ptr and s.kind != t.kind) return true;
        };
        return false;
    }
};

test "a write past a payload breaks its guard, and the payloads come back in order" {
    var r: Rig = .{ .a = std.testing.allocator };
    defer r.deinit();
    const x = try r.input(u16, &.{ 1, 2, 3 });
    const y = try r.output(u32, 2);
    const z = try r.input(u8, &.{9});
    try std.testing.expectEqualSlices(u8, &.{ 1, 0, 2, 0, 3, 0, 9 }, try r.inputBytes());
    try std.testing.expectEqualSlices(u8, &.{ 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff }, try r.outputBytes());
    try std.testing.expect(try r.intact());
    try std.testing.expect(!r.overlap());
    const past: [*]u8 = @ptrCast(y.ptr);
    past[2 * @sizeOf(u32)] = 7;
    try std.testing.expect(!try r.intact());
    _ = x;
    _ = z;
}
