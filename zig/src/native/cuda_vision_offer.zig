//! What a CUDA family with an image tower offers the server: its processor, image token, markers and limits.
const std = @import("std");
const api = @import("engine_api");
const qwen35 = @import("qwen35");
const Allocator = std.mem.Allocator;

/// Info.vision for a family's tower (Nemotron Omni's, Qwen's or Flash Next's); null without --vision.
pub fn offer(v: anytype) ?api.Vision {
    const x = v orelse return null;
    const T = @typeInfo(@TypeOf(x)).pointer.child;
    const prepare = struct {
        fn f(ctx: *anyopaque, a: Allocator, bytes: []const u8, max_tokens: u32) anyerror!api.PreparedImage {
            const t: *T = @ptrCast(@alignCast(ctx));
            const p = try t.prepare(a, bytes, max_tokens);
            return .{ .pixels = p.pixels, .gh = p.gh, .gw = p.gw, .tokens = p.tokens, .hash = p.hash };
        }
    }.f;
    if (comptime T == qwen35.native.vision.Vision) {
        // the template writes Qwen's vision_start and vision_end around the image token itself
        const image = x.shared.image_token;
        return .{ .ctx = x, .prepare = prepare, .image_token = image, .markers = &x.markers, .image_tokens = x.limits.image_tokens, .max_images = x.limits.max_images, .explain = T.explain };
    }
    // a template that writes each image's begin and end itself: the server expands the image token alone
    const wrap: ?[2]u32 = if (@hasDecl(T, "template_wraps")) null else .{ x.marks.open, x.marks.close };
    return .{ .ctx = x, .prepare = prepare, .image_token = x.marks.image, .markers = &x.mark_ids, .wrap = wrap, .image_tokens = x.limits.image_tokens, .max_images = x.limits.max_images, .explain = T.explain };
}

/// The device bytes a family's tower takes beside its text weights on this rank, 0 without --vision.
pub fn weightBytes(comptime F: type, a: Allocator, io: std.Io, o: api.Open) !u64 {
    if (comptime @hasDecl(F, "visionBytes")) if (o.vision != null and o.rank == 0) return F.visionBytes(a, io, o.dir);
    return 0;
}

test "a tower whose template writes its own markers gets no wrapper; other towers get theirs" {
    const Marks = struct { open: u32 = 1, image: u32 = 2, close: u32 = 3 };
    const Limits = struct { image_tokens: u32 = 4, max_images: u32 = 2 };
    const Plain = struct {
        marks: Marks = .{},
        mark_ids: [3]u32 = .{ 1, 2, 3 },
        limits: Limits = .{},
        fn prepare(_: *@This(), _: Allocator, _: []const u8, _: u32) !api.PreparedImage {
            return error.Unused;
        }
        fn explain(_: anyerror) ?[]const u8 {
            return null;
        }
    };
    const Wraps = struct {
        pub const template_wraps = true;
        marks: Marks = .{},
        mark_ids: [3]u32 = .{ 1, 2, 3 },
        limits: Limits = .{},
        fn prepare(_: *@This(), _: Allocator, _: []const u8, _: u32) !api.PreparedImage {
            return error.Unused;
        }
        fn explain(_: anyerror) ?[]const u8 {
            return null;
        }
    };
    var plain: Plain = .{};
    var own: Wraps = .{};
    try std.testing.expectEqual([2]u32{ 1, 3 }, offer(@as(?*Plain, &plain)).?.wrap.?);
    try std.testing.expect(offer(@as(?*Wraps, &own)).?.wrap == null);
    try std.testing.expectEqual(@as(u32, 2), offer(@as(?*Wraps, &own)).?.image_token);
}
