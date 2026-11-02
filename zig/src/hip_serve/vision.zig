//! A family-side helper: a typed image tower (`c.image_token`, `limits`, `prepare`, `explain`) as an api.Vision.
const std = @import("std");
const api = @import("engine_api");

/// `tower` is an optional pointer to the tower; null gives null (text only). The host never calls this.
pub fn offer(tower: anytype) ?api.Vision {
    const x = tower orelse return null;
    const T = @typeInfo(@TypeOf(x)).pointer.child;
    const prepare = struct {
        fn f(ctx: *anyopaque, a: std.mem.Allocator, bytes: []const u8, max_tokens: u32) anyerror!api.PreparedImage {
            const t: *T = @ptrCast(@alignCast(ctx));
            const p = try t.prepare(a, bytes, max_tokens);
            return .{ .pixels = p.pixels, .gh = p.gh, .gw = p.gw, .tokens = p.tokens, .hash = p.hash };
        }
    }.f;
    var out: api.Vision = .{
        .ctx = x,
        .prepare = prepare,
        .image_token = x.c.image_token,
        .image_tokens = x.limits.image_tokens,
        .max_images = x.limits.max_images,
        .explain = T.explain,
    };
    if (comptime @hasField(T, "markers")) {
        out.markers = &x.markers;
    } else {
        // A marker-free template still reserves its image placeholder against literal request text.
        out.markers = @as(*const [1]u32, @ptrCast(&x.c.image_token));
    }
    if (comptime @hasDecl(T, "part_type")) out.part_type = T.part_type;
    return out;
}

test "a marker-free image offer reserves its placeholder without adding wrapper tokens" {
    const Tower = struct {
        pub const part_type = "image";
        c: struct { image_token: u32 } = .{ .image_token = 57 },
        limits: struct { image_tokens: u32, max_images: u32 } = .{ .image_tokens = 64, .max_images = 2 },
        pub fn prepare(_: *@This(), _: std.mem.Allocator, _: []const u8, _: u32) !api.PreparedImage {
            return error.Unused;
        }
        pub fn explain(_: anyerror) ?[]const u8 {
            return null;
        }
    };
    var tower: Tower = .{};
    const v = offer(@as(?*Tower, &tower)).?;
    try std.testing.expectEqualSlices(u32, &.{57}, v.markers);
    try std.testing.expectEqual(@as(u32, 57), v.image_token);
    try std.testing.expectEqualStrings("image", v.part_type);
    try std.testing.expect(v.wrap == null);
    try std.testing.expect(v.markers.ptr == @as([*]const u32, @ptrCast(&tower.c.image_token)));
    try std.testing.expect(offer(@as(?*Tower, null)) == null);
}

test "a tower with markers offers them, and its limits reach the host" {
    const Tower = struct {
        c: struct { image_token: u32 } = .{ .image_token = 9 },
        limits: struct { image_tokens: u32, max_images: u32 } = .{ .image_tokens = 256, .max_images = 3 },
        markers: [3]u32 = .{ 8, 9, 10 },
        pub fn prepare(_: *@This(), _: std.mem.Allocator, _: []const u8, _: u32) !api.PreparedImage {
            return error.Unused;
        }
        pub fn explain(_: anyerror) ?[]const u8 {
            return null;
        }
    };
    var tower: Tower = .{};
    const v = offer(@as(?*Tower, &tower)).?;
    try std.testing.expectEqualSlices(u32, &.{ 8, 9, 10 }, v.markers);
    try std.testing.expectEqual(@as(u32, 256), v.image_tokens);
    try std.testing.expectEqual(@as(u32, 3), v.max_images);
}
