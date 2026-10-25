//! Streams a Metal family serves under the load limit: kv_pool's plan, a stream's contiguous caches its fixed part.
const std = @import("std");
const mtl = @import("metal");
const api = @import("engine_api");
const kv_pool = @import("tensorfold").kv_pool;
const cache_fit = @import("cache_fit.zig");

/// What the load limit (70% of RAM, at most the working set) leaves past the footprint or the device's buffers.
pub fn room(device: mtl.Device) u64 {
    const limit = (cache_fit.ceiling(cache_fit.ram() orelse 0, device.maxWorkingSet(), null) catch unreachable).bytes;
    return limit -| @max(cache_fit.footprint() orelse 0, device.allocated());
}

/// A plan's layout for streams without pages: all of a stream is fixed, and a one-byte page keeps the arithmetic whole.
pub fn layout(stream_bytes: u64) kv_pool.Layout {
    return .{ .fixed_bytes = stream_bytes, .token_bytes = 1, .page_tokens = 1, .snap_bytes = 0 };
}

/// Streams of `stream_bytes` that fit `left`, --parallel's count at most; null: `problem` says why none or too few.
pub fn streams(a: std.mem.Allocator, family: []const u8, stream_bytes: u64, left: u64, o: api.Open, window: u64, problem: *[]const u8) !?u32 {
    const p = kv_pool.plan(layout(stream_bytes), .{ .room = left, .engine = stream_bytes, .lanes = o.lanes, .fixed = o.lanes_fixed }) catch |e| {
        problem.* = switch (e) {
            error.NoStream => try std.fmt.allocPrint(a, "{s}: one stream at a {d}-token window takes {d:.2} GiB, more than the {d:.2} GiB left under the load limit; lower --context", .{ family, window, gib(stream_bytes), gib(left) }),
            error.TooMany => try std.fmt.allocPrint(a, "{s}: --parallel {d} at a {d}-token window needs {d:.2} GiB ({d:.2} GiB a stream), more than the {d:.2} GiB left under the load limit; {d} fit: serve fewer, or lower --context", .{ family, o.lanes, window, gib(stream_bytes * o.lanes), gib(stream_bytes), gib(left), fit(stream_bytes, left) }),
        };
        return null;
    };
    if (p.lanes < o.lanes) std.log.info("{s}: {d} streams at a {d}-token window fit the {d:.2} GiB left under the load limit ({d:.2} GiB a stream), not {d}", .{ family, p.lanes, window, gib(left), gib(stream_bytes), o.lanes });
    return p.lanes;
}

/// Streams the plan would serve from `left`, for the refusal's count.
fn fit(stream_bytes: u64, left: u64) u32 {
    const p = kv_pool.plan(layout(stream_bytes), .{ .room = left, .engine = stream_bytes, .lanes = std.math.maxInt(u16) }) catch return 0;
    return p.lanes;
}

fn gib(bytes: u64) f64 {
    return @as(f64, @floatFromInt(bytes)) / (1 << 30);
}

const GiB: u64 = 1 << 30;

test "streams that fit are served, an asked count past them refused, and fewer served when the count was not named" {
    const a = std.testing.allocator;
    var why: []const u8 = "";
    const asked: api.Open = .{ .dir = "", .model_type = "t", .lanes = 4, .lanes_fixed = true };
    try std.testing.expectEqual(@as(?u32, 4), try streams(a, "t", 3 * GiB, 13 * GiB, asked, 4096, &why));
    try std.testing.expectEqual(@as(?u32, null), try streams(a, "t", 3 * GiB, 10 * GiB, asked, 4096, &why));
    defer a.free(why);
    try std.testing.expect(std.mem.indexOf(u8, why, "--parallel 4") != null and std.mem.indexOf(u8, why, "; 3 fit") != null);
    const auto: api.Open = .{ .dir = "", .model_type = "t", .lanes = 8 };
    try std.testing.expectEqual(@as(?u32, 3), try streams(a, "t", 3 * GiB, 10 * GiB, auto, 4096, &why));
}

test "a window that fits no stream is refused with lower --context" {
    const a = std.testing.allocator;
    var why: []const u8 = "";
    try std.testing.expectEqual(@as(?u32, null), try streams(a, "t", 3 * GiB, 2 * GiB, .{ .dir = "", .model_type = "t", .lanes = 1 }, 262144, &why));
    defer a.free(why);
    try std.testing.expect(std.mem.indexOf(u8, why, "lower --context") != null);
}
