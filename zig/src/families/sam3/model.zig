const std = @import("std");
const gpu = @import("tensor_metal");
const tokenizer = @import("tokenizer");
const ops = @import("ops.zig");
const vision = @import("vision.zig");
const decoder = @import("decoder.zig");
const prompt_encoder = @import("prompt_encoder.zig");
const prompts = @import("prompts.zig");
const Config = @import("config.zig").Config;
const Store = @import("checkpoint.zig").Store;

pub const LoadOptions = struct {
    tokenizer_path: ?[]const u8 = null,
    version: ?@import("config.zig").Version = null,
    precision: enum { checkpoint, float32 } = .checkpoint,
    compact_gemm: bool = true,
    split_attention: bool = false,
    device_projections: bool = false,
    device_attention: bool = false,
    compensated_norm: bool = false,
    scratch_cache_bytes: usize = 2 << 30,
};

pub const Mask = @import("mask.zig").Mask;

pub const Prediction = struct {
    allocator: std.mem.Allocator,
    masks: []Mask,

    pub fn deinit(self: *Prediction) void {
        for (self.masks) |mask| self.allocator.free(mask.pixels);
        self.allocator.free(self.masks);
        self.* = undefined;
    }
};

pub const Model = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    config: Config,
    store: Store,
    gpu: *gpu.Gpu,
    tokenizer: ?tokenizer.Tokenizer = null,
    precision: @FieldType(ops.Runtime, "precision"),
    split_attention: bool = false,
    device_projections: bool = false,
    device_attention: bool = false,
    compensated_norm: bool = false,
    mutex: std.Io.Mutex = .init,

    pub fn load(allocator: std.mem.Allocator, io: std.Io, path: []const u8, options: LoadOptions) !*Model {
        var store = try Store.open(allocator, io, path);
        errdefer store.deinit();
        const config = try store.config();
        if (options.version) |version| if (version != config.version) return error.CheckpointVersionMismatch;
        const weights = try store.get("detector.backbone.vision_backbone.trunk.patch_embed.proj.weight");
        const device = try allocator.create(gpu.Gpu);
        errdefer allocator.destroy(device);
        device.* = try gpu.Gpu.init(allocator, @import("sam3_kernels").source);
        errdefer device.deinit();
        device.cache_weights = true;
        device.map_weights = true;
        device.compact_gemm = options.compact_gemm;
        device.buffer_cache_limit = options.scratch_cache_bytes;
        var text: ?tokenizer.Tokenizer = null;
        if (options.tokenizer_path) |file| text = try tokenizer.loadTokenizer(io, allocator, file);
        errdefer if (text) |*t| t.deinit();
        const self = try allocator.create(Model);
        self.* = .{
            .allocator = allocator,
            .io = io,
            .config = config,
            .store = store,
            .gpu = device,
            .tokenizer = text,
            .precision = if (options.precision == .float32 or weights.dtype != .f16) .fp32 else .mixed,
            .split_attention = options.split_attention and device.device.tensorUnits(),
            .device_projections = options.device_projections and device.device.tensorUnits() and (options.precision == .float32 or weights.dtype != .f16),
            .device_attention = options.device_attention and device.device.tensorUnits() and (options.precision == .float32 or weights.dtype != .f16),
            .compensated_norm = options.compensated_norm and device.device.tensorUnits() and (options.precision == .float32 or weights.dtype != .f16),
        };
        return self;
    }

    pub fn deinit(self: *Model) void {
        if (self.tokenizer) |*text| text.deinit();
        self.gpu.deinit();
        self.allocator.destroy(self.gpu);
        self.store.deinit();
        self.allocator.destroy(self);
    }

    pub fn runtime(self: *Model, arena: std.mem.Allocator) ops.Runtime {
        return .{ .g = self.gpu, .store = &self.store, .a = arena, .precision = self.precision, .config = self.config, .split_attention = self.split_attention, .device_projections = self.device_projections, .device_attention = self.device_attention, .compensated_norm = self.compensated_norm };
    }

    pub fn tokenize(self: *const Model, allocator: std.mem.Allocator, text: []const u8) ![]u32 {
        const t = self.tokenizer orelse return error.TokenizerNotLoaded;
        const encoded = try t.encode(allocator, text);
        defer allocator.free(encoded);
        if (encoded.len > 30) return error.TextPromptTooLong;
        const tokens = try allocator.alloc(u32, encoded.len + 2);
        tokens[0] = 49406;
        @memcpy(tokens[1..][0..encoded.len], encoded);
        tokens[tokens.len - 1] = 49407;
        return tokens;
    }

    pub fn predictText(self: *Model, allocator: std.mem.Allocator, image: prompts.Image, text: []const u8, options: vision.DetectionOptions) !Prediction {
        const tokens = try self.tokenize(allocator, text);
        defer allocator.free(tokens);
        return self.predictTokens(allocator, image, tokens, options);
    }

    pub fn predictTokens(self: *Model, allocator: std.mem.Allocator, image: prompts.Image, tokens: []const u32, options: vision.DetectionOptions) !Prediction {
        try self.mutex.lock(self.io);
        defer self.mutex.unlock(self.io);
        var arena = std.heap.ArenaAllocator.init(self.allocator);
        defer arena.deinit();
        var r = self.runtime(arena.allocator());
        defer r.deinit();
        const features = try encodeImage(&r, image, "convs");
        const detections = try vision.detect(&r, features, try vision.text(&r, tokens), options);
        const masks = try allocator.alloc(Mask, detections.len);
        var ready: usize = 0;
        errdefer {
            for (masks[0..ready]) |mask| allocator.free(mask.pixels);
            allocator.free(masks);
        }
        for (detections, masks) |detection, *mask| {
            mask.* = try readMask(&r, allocator, detection.mask, image.size, detection.score);
            for (&mask.box, detection.box, 0..) |*coordinate, value, axis| {
                const extent = if (axis % 2 == 0) image.size.width else image.size.height;
                coordinate.* = std.math.clamp(value, 0, 1) * @as(f32, @floatFromInt(extent));
            }
            ready += 1;
        }
        return .{ .allocator = allocator, .masks = masks };
    }

    pub fn predictVisual(self: *Model, allocator: std.mem.Allocator, image: prompts.Image, prompt: prompts.Prompt, multimask: bool) !Prediction {
        try image.validate();
        try prompt.validate(image.size);
        try self.mutex.lock(self.io);
        defer self.mutex.unlock(self.io);
        var arena = std.heap.ArenaAllocator.init(self.allocator);
        defer arena.deinit();
        var r = self.runtime(arena.allocator());
        defer r.deinit();
        const features = try encodeImage(&r, image, self.config.pyramidPrefix(true));
        const encoded = try prompt_encoder.encode(&r, prompt, image.size);
        const image_features = try r.add(features[2].tensor, try r.weight(try r.name("{s}.{s}", .{
            self.config.trackerPrefix(), if (self.config.version == .sam31) "interactivity_no_mem_embed" else "no_mem_embed",
        })));
        const decoded = try decoder.decode(&r, features, image_features, .{ .dense = encoded.dense, .sparse = encoded.sparse, .multimask = multimask });
        const masks = try allocator.alloc(Mask, 1);
        errdefer allocator.free(masks);
        masks[0] = try readMask(&r, allocator, decoded.masks, image.size, decoded.ious[0]);
        return .{ .allocator = allocator, .masks = masks };
    }
};

pub fn encodeImage(r: *ops.Runtime, image: prompts.Image, pyramid: []const u8) ![3]ops.Feature {
    return vision.pyramid(r, try encodeTrunk(r, image), pyramid);
}

pub fn encodeTrunk(r: *ops.Runtime, image: prompts.Image) !ops.Feature {
    return vision.trunk(r, try prepareImage(r, image));
}

pub fn prepareImage(r: *ops.Runtime, image: prompts.Image) !ops.Feature {
    try image.validate();
    const stride = try image.stride();
    const pixels = try image.size.pixels();
    const tensor = try r.alloc(try std.math.mul(usize, pixels, 3));
    for (0..image.size.height) |y| {
        for (0..image.size.width * 3) |x| {
            tensor.slice(f32, tensor.count)[y * image.size.width * 3 + x] =
                @as(f32, @floatFromInt(image.rgb[y * stride + x])) / 127.5 - 1;
        }
    }
    try r.observe("input.normalized_rgb", tensor, &.{ 1, image.size.height, image.size.width, 3 });
    return r.resize(ops.feature(tensor, image.size.height, image.size.width, 3), 1008, 1008, true);
}

pub fn readMask(r: *ops.Runtime, allocator: std.mem.Allocator, mask: ops.Feature, size: prompts.Size, score: f32) !Mask {
    const start = r.mark();
    defer r.trim(start, &.{});
    const full = try r.resize(mask, size.height, size.width, true);
    return @import("mask.zig").fromLogits(allocator, full.tensor.slice(f32, full.tensor.count), size, score);
}
