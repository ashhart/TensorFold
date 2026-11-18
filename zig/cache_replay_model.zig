//! A byte-accounted host snapshot family; live stream state is copied, never transferred to the cache.
const std = @import("std");
const pc = @import("cache");
pub const Model = struct {
    padded: bool = true,
    pub const recurrent_bytes: u64 = 48 * (48 * 128 * 128 * 4 + (2 * 16 * 128 + 48 * 128) * 3 * 2);
    pub const kv_bytes_per_token: u64 = 16 * 2 * 4 * 256 * 2;
    pub fn bytes(m: Model, at: u32) u64 {
        const raw = recurrent_bytes + kv_bytes_per_token * at;
        return if (m.padded) std.mem.alignForward(u64, raw + raw / 8, 32 << 20) else raw;
    }
};
pub const State = struct {
    at: u32 = 0,
    sum: u64 = 0,
    pub fn advance(s: *State, tokens: []const u32, until: u32) void {
        for (tokens[s.at..until]) |v| s.sum = s.sum *% 31 +% v;
        s.at = until;
    }
};
pub const Family = struct {
    gpa: std.mem.Allocator,
    model: Model,
    live: u64 = 0,
    pub fn snapshots(f: *Family) pc.Snapshots {
        return .{ .ptr = f, .vtable = &.{ .bytes = bytes, .save = save, .restore = restore, .drop = drop } };
    }
    fn of(ptr: *anyopaque) *Family {
        return @ptrCast(@alignCast(ptr));
    }
    fn bytes(ptr: *anyopaque, at: u32) u64 {
        return of(ptr).model.bytes(at);
    }
    fn save(ptr: *anyopaque, owner: ?*anyopaque, at: u32) !pc.Saved {
        const own: *State = @ptrCast(@alignCast(owner.?));
        if (own.at != at) return error.NotAtMark;
        const f = of(ptr);
        const copy = try f.gpa.create(State);
        copy.* = own.*;
        f.live += 1;
        return copy;
    }
    fn restore(_: *anyopaque, owner: ?*anyopaque, saved: pc.Saved) !void {
        const own: *State = @ptrCast(@alignCast(owner.?));
        const copy: *const State = @ptrCast(@alignCast(saved));
        own.* = copy.*;
    }
    fn drop(ptr: *anyopaque, saved: pc.Saved) void {
        const f = of(ptr);
        f.gpa.destroy(@as(*State, @ptrCast(@alignCast(saved))));
        f.live -= 1;
    }
};
