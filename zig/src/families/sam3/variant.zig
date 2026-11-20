const std = @import("std");

pub const Version = enum {
    sam3,
    sam31,

    pub fn parse(text: []const u8) error{UnsupportedSamVersion}!Version {
        if (std.mem.eql(u8, text, "sam3")) return .sam3;
        if (std.mem.eql(u8, text, "sam3.1")) return .sam31;
        return error.UnsupportedSamVersion;
    }

    pub fn name(self: Version) []const u8 {
        return switch (self) {
            .sam3 => "sam3",
            .sam31 => "sam3.1",
        };
    }
};

pub const LevelSet = enum { detector, prompt, tracking };

pub const Roots = struct {
    trunk: []const u8,
    levels: []const u8,
    text: []const u8,
    detector: []const u8,
    tracker: []const u8,
    prompt_encoder: []const u8,
    prompt_head: []const u8,
    grouped_head: ?[]const u8,

    fn of(version: Version) Roots {
        const sam31 = version == .sam31;
        return .{
            .trunk = "detector.backbone.vision_backbone.trunk",
            .levels = "detector.backbone.vision_backbone",
            .text = "detector.backbone.language_backbone",
            .detector = "detector",
            .tracker = if (sam31) "tracker.model" else "tracker",
            .prompt_encoder = if (sam31) "tracker.model.interactive_sam_prompt_encoder" else "tracker.sam_prompt_encoder",
            .prompt_head = if (sam31) "tracker.model.interactive_sam_mask_decoder" else "tracker.sam_mask_decoder",
            .grouped_head = if (sam31) "tracker.model.sam_mask_decoder" else null,
        };
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
    tile_side: usize = 24,
    hidden: usize = 256,
    memory_width: usize,
    tracker_heads: usize,
    bucket_size: usize,
    mask_tokens: usize,
    memory_frames: usize = 7,
    pointer_frames: usize = 16,
    roots: Roots,

    pub fn of(version: Version) Config {
        return switch (version) {
            .sam3 => .{
                .version = .sam3,
                .memory_width = 64,
                .tracker_heads = 1,
                .bucket_size = 1,
                .mask_tokens = 4,
                .roots = Roots.of(.sam3),
            },
            .sam31 => .{
                .version = .sam31,
                .memory_width = 256,
                .tracker_heads = 8,
                .bucket_size = 16,
                .mask_tokens = 3,
                .roots = Roots.of(.sam31),
            },
        };
    }

    pub fn featureSide(self: Config) usize {
        return self.image_size / self.patch_size;
    }

    pub fn maskSide(self: Config) usize {
        return 4 * self.featureSide();
    }

    pub fn grouped(self: Config) bool {
        return self.bucket_size > 1;
    }

    pub fn buckets(self: Config, objects: usize) error{ NoObjects, TooManyObjects }!usize {
        if (objects == 0) return error.NoObjects;
        const rounded = std.math.add(usize, objects, self.bucket_size - 1) catch return error.TooManyObjects;
        return rounded / self.bucket_size;
    }

    pub fn levelRoot(self: Config, set: LevelSet) []const u8 {
        return switch (set) {
            .detector => "convs",
            .prompt => if (self.version == .sam31) "interactive_convs" else "sam2_convs",
            .tracking => if (self.version == .sam31) "propagation_convs" else "sam2_convs",
        };
    }
};

test "SAM version names round trip and refuse unknown text" {
    for ([_]Version{ .sam3, .sam31 }) |version| try std.testing.expectEqual(version, try Version.parse(version.name()));
    try std.testing.expectEqualStrings("sam3.1", Version.sam31.name());
    try std.testing.expectError(error.UnsupportedSamVersion, Version.parse("sam31"));
    try std.testing.expectError(error.UnsupportedSamVersion, Version.parse(""));
}

test "SAM configuration values per version" {
    const a = Config.of(.sam3);
    const b = Config.of(.sam31);
    try std.testing.expectEqual(@as(usize, 64), a.memory_width);
    try std.testing.expectEqual(@as(usize, 1), a.tracker_heads);
    try std.testing.expectEqual(@as(usize, 1), a.bucket_size);
    try std.testing.expectEqual(@as(usize, 4), a.mask_tokens);
    try std.testing.expectEqual(@as(usize, 256), b.memory_width);
    try std.testing.expectEqual(@as(usize, 8), b.tracker_heads);
    try std.testing.expectEqual(@as(usize, 16), b.bucket_size);
    try std.testing.expectEqual(@as(usize, 3), b.mask_tokens);
    for ([_]Config{ a, b }) |c| {
        try std.testing.expectEqual(@as(usize, 1008), c.image_size);
        try std.testing.expectEqual(@as(usize, 14), c.patch_size);
        try std.testing.expectEqual(@as(usize, 1024), c.vision_width);
        try std.testing.expectEqual(@as(usize, 32), c.vision_layers);
        try std.testing.expectEqual(@as(usize, 16), c.vision_heads);
        try std.testing.expectEqual(@as(usize, 4736), c.vision_mlp);
        try std.testing.expectEqual(@as(usize, 24), c.tile_side);
        try std.testing.expectEqual(@as(usize, 256), c.hidden);
        try std.testing.expectEqual(@as(usize, 7), c.memory_frames);
        try std.testing.expectEqual(@as(usize, 16), c.pointer_frames);
        try std.testing.expectEqual(@as(usize, 72), c.featureSide());
        try std.testing.expectEqual(@as(usize, 288), c.maskSide());
    }
    try std.testing.expect(!a.grouped());
    try std.testing.expect(b.grouped());
}

test "SAM group counts round up and refuse empty or overflowing requests" {
    const a = Config.of(.sam3);
    const b = Config.of(.sam31);
    try std.testing.expectEqual(@as(usize, 5), try a.buckets(5));
    try std.testing.expectEqual(@as(usize, 1), try b.buckets(16));
    try std.testing.expectEqual(@as(usize, 3), try b.buckets(33));
    try std.testing.expectEqual(@as(usize, 1), try b.buckets(1));
    try std.testing.expectError(error.NoObjects, a.buckets(0));
    try std.testing.expectError(error.NoObjects, b.buckets(0));
    try std.testing.expectError(error.TooManyObjects, b.buckets(std.math.maxInt(usize)));
    try std.testing.expectEqual(std.math.maxInt(usize), try a.buckets(std.math.maxInt(usize)));
}

test "SAM level-set and checkpoint roots per version" {
    const a = Config.of(.sam3);
    const b = Config.of(.sam31);
    try std.testing.expectEqualStrings("convs", a.levelRoot(.detector));
    try std.testing.expectEqualStrings("sam2_convs", a.levelRoot(.prompt));
    try std.testing.expectEqualStrings("sam2_convs", a.levelRoot(.tracking));
    try std.testing.expectEqualStrings("convs", b.levelRoot(.detector));
    try std.testing.expectEqualStrings("interactive_convs", b.levelRoot(.prompt));
    try std.testing.expectEqualStrings("propagation_convs", b.levelRoot(.tracking));
    for ([_]Config{ a, b }) |c| {
        try std.testing.expectEqualStrings("detector.backbone.vision_backbone.trunk", c.roots.trunk);
        try std.testing.expectEqualStrings("detector.backbone.vision_backbone", c.roots.levels);
        try std.testing.expectEqualStrings("detector.backbone.language_backbone", c.roots.text);
        try std.testing.expectEqualStrings("detector", c.roots.detector);
    }
    try std.testing.expectEqualStrings("tracker", a.roots.tracker);
    try std.testing.expectEqualStrings("tracker.sam_prompt_encoder", a.roots.prompt_encoder);
    try std.testing.expectEqualStrings("tracker.sam_mask_decoder", a.roots.prompt_head);
    try std.testing.expect(a.roots.grouped_head == null);
    try std.testing.expectEqualStrings("tracker.model", b.roots.tracker);
    try std.testing.expectEqualStrings("tracker.model.interactive_sam_prompt_encoder", b.roots.prompt_encoder);
    try std.testing.expectEqualStrings("tracker.model.interactive_sam_mask_decoder", b.roots.prompt_head);
    try std.testing.expectEqualStrings("tracker.model.sam_mask_decoder", b.roots.grouped_head.?);
}
