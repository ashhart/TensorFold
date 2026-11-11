const std = @import("std");

pub const Version = enum {
    sam3,
    sam31,

    pub fn parse(value: []const u8) !Version {
        if (std.mem.eql(u8, value, "sam3")) return .sam3;
        if (std.mem.eql(u8, value, "sam3.1")) return .sam31;
        return error.UnsupportedSamVersion;
    }

    pub fn name(self: Version) []const u8 {
        return if (self == .sam3) "sam3" else "sam3.1";
    }
};

pub const Config = struct {
    version: Version,
    image_size: usize = 1008,
    patch_size: usize = 14,
    vision_width: usize = 1024,
    vision_layers: usize = 32,
    vision_heads: usize = 16,
    vision_mlp: usize = 4736,
    window: usize = 24,
    hidden: usize = 256,
    memory_width: usize,
    tracker_heads: usize,
    bucket_size: usize,
    mask_tokens: usize,
    memory_frames: usize = 7,
    pointer_frames: usize = 16,

    pub fn of(version: Version) Config {
        return switch (version) {
            .sam3 => .{ .version = version, .memory_width = 64, .tracker_heads = 1, .bucket_size = 1, .mask_tokens = 4 },
            .sam31 => .{ .version = version, .memory_width = 256, .tracker_heads = 8, .bucket_size = 16, .mask_tokens = 3 },
        };
    }

    pub fn featureSide(self: Config) usize {
        return self.image_size / self.patch_size;
    }

    pub fn trackerPrefix(self: Config) []const u8 {
        return if (self.version == .sam3) "tracker" else "tracker.model";
    }

    pub fn buckets(self: Config, objects: usize) !usize {
        if (objects == 0) return error.NoObjects;
        return std.math.divCeil(usize, objects, self.bucket_size) catch error.TooManyObjects;
    }

    pub fn decoderPrefix(self: Config, interactive: bool) []const u8 {
        return if (self.version == .sam31 and interactive)
            "tracker.model.interactive_sam_mask_decoder"
        else if (self.version == .sam3) "tracker.sam_mask_decoder" else "tracker.model.sam_mask_decoder";
    }

    pub fn promptPrefix(self: Config) []const u8 {
        return if (self.version == .sam31)
            "tracker.model.interactive_sam_prompt_encoder"
        else
            "tracker.sam_prompt_encoder";
    }

    pub fn pyramidPrefix(self: Config, interactive: bool) []const u8 {
        if (self.version == .sam3) return "sam2_convs";
        return if (interactive) "interactive_convs" else "propagation_convs";
    }
};

test "SAM versions keep the older tracker separate from multiplex geometry" {
    const a = Config.of(.sam3);
    const b = Config.of(.sam31);
    try std.testing.expectEqual(@as(usize, 72), a.featureSide());
    try std.testing.expectEqual(@as(usize, 64), a.memory_width);
    try std.testing.expectEqual(@as(usize, 256), b.memory_width);
    try std.testing.expectEqual(@as(usize, 256), a.hidden / a.tracker_heads);
    try std.testing.expectEqual(@as(usize, 32), b.hidden / b.tracker_heads);
    try std.testing.expectEqual(@as(usize, 17), try a.buckets(17));
    try std.testing.expectEqual(@as(usize, 2), try b.buckets(17));
    try std.testing.expectEqualStrings(a.decoderPrefix(false), a.decoderPrefix(true));
    try std.testing.expect(!std.mem.eql(u8, b.decoderPrefix(false), b.decoderPrefix(true)));
    try std.testing.expectError(error.NoObjects, b.buckets(0));
    try std.testing.expectError(error.UnsupportedSamVersion, Version.parse("sam2"));
}
