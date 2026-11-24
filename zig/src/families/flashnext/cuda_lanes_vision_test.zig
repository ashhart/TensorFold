//! --vision at Flash Next's CUDA open: refused by name by core's gate before the loader touches the driver.
const std = @import("std");
const cuda = @import("cuda");

pub fn Cases(comptime F: type) type {
    return struct {
        test "--vision on a checkpoint without a vision tower is refused by name before a stream, a module or a buffer" {
            const a = std.testing.allocator;
            var tmp = std.testing.tmpDir(.{});
            defer tmp.cleanup();
            try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "config.json", .data = "{\"model_type\":\"qwen3_5_moe\"}" });
            const dir = try tmp.dir.realPathFileAlloc(std.testing.io, ".", a);
            defer a.free(dir);
            // every driver entry point undefined: the refusal must come before the first call
            var driver: cuda.Driver = undefined;
            const ctx: cuda.Context = .{ .d = &driver, .device = 0, .handle = null };
            const o: F.Options = .{ .vision = .{ .image_tokens = 4096, .max_images = 4 }, .context = 4096, .drafts = true };
            try std.testing.expectError(error.VisionWithoutTower, F.open(a, std.testing.io, &ctx, dir, null, o));
            const words = @import("cuda_lanes_explain.zig").open(error.VisionWithoutTower).?;
            try std.testing.expectEqualStrings("--vision given but this Flash Next checkpoint has no vision tower", words);
            try std.testing.expectEqual(@as(u64, 0), try @import("cuda_lanes_bytes.zig").visionBytes(a, std.testing.io, dir, 0, 1));
        }
    };
}
