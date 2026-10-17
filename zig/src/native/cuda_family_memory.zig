//! Family-aware checkpoint and stream admission; rank coordination runs before the host admits any stream.
const std = @import("std");
const cuda = @import("cuda");
const api = @import("engine_api");

pub fn weights(comptime F: type, a: std.mem.Allocator, io: std.Io, o: api.Open, comptime fallback: anytype) !u64 {
    if (@hasDecl(F, "placementWeightBytes")) return F.placementWeightBytes(a, io, o.dir, o.tp, o.rank, o.drafts);
    if (@hasDecl(F, "deviceWeightBytes")) return F.deviceWeightBytes(a, io, o.dir);
    return fallback(io, o.dir);
}

pub fn room(comptime F: type, ptr: *anyopaque, local: u64) !u64 {
    if (@hasDecl(F, "rankRoom")) return F.rankRoom(ptr, local);
    return local;
}

pub fn read(_: ?*anyopaque, reset_peak: bool) ?api.Memory {
    const u = cuda.usage(reset_peak);
    return .{ .active = u.device, .cache = 0, .peak = u.peak, .pinned = u.host };
}

pub fn learnBytes(loaded: anytype) u64 {
    return if (@hasField(@TypeOf(loaded), "learn_bytes")) loaded.learn_bytes else 0;
}

pub fn toGib(bytes: u64) f64 {
    return @as(f64, @floatFromInt(bytes)) / @as(f64, 1 << 30);
}

test "rank room is agreed before stream admission and preserves a single-rank family's pool" {
    const Single = struct {};
    const Pair = struct {
        pub fn rankRoom(_: *anyopaque, local: u64) !u64 {
            return @min(local, 19);
        }
    };
    var context: u8 = 0;
    try std.testing.expectEqual(@as(u64, 30), try room(Single, &context, 30));
    try std.testing.expectEqual(@as(u64, 19), try room(Pair, &context, 30));
    try std.testing.expectEqual(@as(u64, 7), try room(Pair, &context, 7));
}
