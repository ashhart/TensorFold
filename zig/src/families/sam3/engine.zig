const std = @import("std");
const gpu = @import("tensor_metal");
const st = @import("safetensors");
const tokenizer = @import("tokenizer");
const Store = @import("checkpoint.zig").Store;
const variant = @import("variant.zig");
const runtime_module = @import("runtime.zig");
const still_image = @import("still_image.zig");
const Config = variant.Config;
const Version = variant.Version;
const Precision = runtime_module.Precision;
const Routes = runtime_module.Routes;
const Runtime = runtime_module.Runtime;

pub const LoadOptions = struct {
    tokenizer_path: ?[]const u8 = null,
    version: ?Version = null,
    precision: enum { checkpoint, float32 } = .checkpoint,
    compact_gemm: bool = true,
    split_attention: bool = false,
    device_projections: bool = false,
    device_attention: bool = false,
    compensated_norm: bool = false,
    scratch_cache_bytes: usize = 2 << 30,
};

pub const Model = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    config: Config,
    store: Store,
    gpu: *gpu.Gpu,
    tokenizer: ?tokenizer.Tokenizer = null,
    precision: Precision,
    split_attention: bool = false,
    device_projections: bool = false,
    device_attention: bool = false,
    compensated_norm: bool = false,
    mutex: std.Io.Mutex = .init,

    pub fn load(allocator: std.mem.Allocator, io: std.Io, path: []const u8, options: LoadOptions) !*Model {
        var store = try Store.open(allocator, io, path);
        errdefer store.deinit();
        const config = try store.config();
        if (options.version) |wanted| if (wanted != config.version) return error.CheckpointVersionMismatch;
        try validate(config, try readShapes(&store, config));
        var name: [256]u8 = undefined;
        const patch = try store.get(try std.fmt.bufPrint(&name, "{s}.patch_embed.proj.weight", .{config.roots.trunk}));
        const device = try allocator.create(gpu.Gpu);
        errdefer allocator.destroy(device);
        device.* = try gpu.Gpu.init(allocator, @import("sam3_kernels").source);
        errdefer device.deinit();
        device.cache_weights = true;
        device.map_weights = true;
        device.compact_gemm = options.compact_gemm;
        device.buffer_cache_limit = options.scratch_cache_bytes;
        var text: ?tokenizer.Tokenizer = null;
        if (options.tokenizer_path) |tokenizer_path| text = try tokenizer.loadTokenizer(io, allocator, tokenizer_path);
        errdefer if (text) |*loaded| loaded.deinit();
        const model = try allocator.create(Model);
        const flags = deriveFlags(options, patch.dtype, device.device.tensorUnits());
        model.* = .{
            .allocator = allocator,
            .io = io,
            .config = config,
            .store = store,
            .gpu = device,
            .tokenizer = text,
            .precision = flags.precision,
            .split_attention = flags.split_attention,
            .device_projections = flags.device_projections,
            .device_attention = flags.device_attention,
            .compensated_norm = flags.compensated_norm,
        };
        return model;
    }

    pub fn deinit(self: *Model) void {
        if (self.tokenizer) |*loaded| loaded.deinit();
        // Cached weights may be no-copy views of the store's file mapping, so the GPU goes first.
        self.gpu.deinit();
        self.allocator.destroy(self.gpu);
        self.store.deinit();
        const allocator = self.allocator;
        allocator.destroy(self);
    }

    pub fn routes(self: *const Model) Routes {
        return .{
            .precision = self.precision,
            .split_attention = self.split_attention,
            .device_projections = self.device_projections,
            .device_attention = self.device_attention,
            .compensated_norm = self.compensated_norm,
        };
    }

    pub fn runtime(self: *Model, scratch: std.mem.Allocator) Runtime {
        return .{ .gpu = self.gpu, .store = &self.store, .scratch = scratch, .config = self.config, .routes = self.routes() };
    }

    pub const predictText = still_image.predictText;
    pub const predictTokens = still_image.predictTokens;
    pub const predictVisual = still_image.predictVisual;
};

fn deriveFlags(options: LoadOptions, patch: st.DType, tensor_units: bool) Routes {
    const precision: Precision = if (options.precision == .float32 or patch != .f16) .fp32 else .mixed;
    const exact_device = tensor_units and precision == .fp32;
    return .{
        .precision = precision,
        .split_attention = options.split_attention and tensor_units,
        .device_projections = options.device_projections and exact_device,
        .device_attention = options.device_attention and exact_device,
        .compensated_norm = options.compensated_norm and exact_device,
    };
}

const Shapes = struct {
    output_valid: ?st.Tensor,
    mask_tokens: st.Tensor,
    time_rows: st.Tensor,
    out_proj: ?st.Tensor,
    pix_feat_proj: st.Tensor,
};

const Derived = struct { slots: usize, candidates: usize, memory_frames: usize, memory_width: usize };

fn readShapes(store: *Store, config: Config) !Shapes {
    const tracker = config.roots.tracker;
    const head = config.roots.grouped_head orelse config.roots.prompt_head;
    var name: [256]u8 = undefined;
    return .{
        .output_valid = try optional(store, try std.fmt.bufPrint(&name, "{s}.output_valid_embed", .{tracker})),
        .mask_tokens = try store.get(try std.fmt.bufPrint(&name, "{s}.mask_tokens.weight", .{head})),
        .time_rows = try store.get(try std.fmt.bufPrint(&name, "{s}.maskmem_tpos_enc", .{tracker})),
        .out_proj = try optional(store, try std.fmt.bufPrint(&name, "{s}.maskmem_backbone.out_proj.weight", .{tracker})),
        .pix_feat_proj = try store.get(try std.fmt.bufPrint(&name, "{s}.maskmem_backbone.pix_feat_proj.weight", .{tracker})),
    };
}

fn optional(store: *Store, name: []const u8) !?st.Tensor {
    if (!store.has(name)) return null;
    return try store.get(name);
}

fn derive(shapes: Shapes) error{UnsupportedSamTracker}!Derived {
    const slots = if (shapes.output_valid) |rows| rows.dim(0) else 1;
    const mask_rows = shapes.mask_tokens.dim(0);
    if (slots == 0 or mask_rows % slots != 0) return error.UnsupportedSamTracker;
    const time_rank = shapes.time_rows.rank;
    if (time_rank == 0) return error.UnsupportedSamTracker;
    const width = if (shapes.out_proj) |projection| projection.dim(0) else shapes.pix_feat_proj.dim(0);
    if (width != shapes.time_rows.dim(time_rank - 1)) return error.UnsupportedSamTracker;
    return .{
        .slots = slots,
        .candidates = mask_rows / slots,
        .memory_frames = shapes.time_rows.dim(0),
        .memory_width = width,
    };
}

fn validate(config: Config, shapes: Shapes) error{UnsupportedSamTracker}!void {
    const derived = try derive(shapes);
    if (derived.slots != config.bucket_size or derived.candidates != config.mask_tokens or
        derived.memory_frames != config.memory_frames or derived.memory_width != config.memory_width)
        return error.UnsupportedSamTracker;
}

test "SAM load flags follow precision, request and tensor units" {
    const half_patch = deriveFlags(.{}, .f16, true);
    try std.testing.expectEqual(Precision.mixed, half_patch.precision);
    try std.testing.expectEqual(Precision.fp32, deriveFlags(.{}, .f32, true).precision);
    try std.testing.expectEqual(Precision.fp32, deriveFlags(.{}, .bf16, true).precision);
    try std.testing.expectEqual(Precision.fp32, deriveFlags(.{ .precision = .float32 }, .f16, true).precision);
    for ([_]st.DType{ .f16, .f32 }) |patch| for ([_]bool{ false, true }) |units| for (0..32) |bits| {
        const options = LoadOptions{
            .precision = if (bits & 1 != 0) .float32 else .checkpoint,
            .split_attention = bits & 2 != 0,
            .device_projections = bits & 4 != 0,
            .device_attention = bits & 8 != 0,
            .compensated_norm = bits & 16 != 0,
        };
        const flags = deriveFlags(options, patch, units);
        const fp32 = options.precision == .float32 or patch != .f16;
        try std.testing.expectEqual(if (fp32) Precision.fp32 else Precision.mixed, flags.precision);
        try std.testing.expectEqual(options.split_attention and units, flags.split_attention);
        try std.testing.expectEqual(options.device_projections and units and fp32, flags.device_projections);
        try std.testing.expectEqual(options.device_attention and units and fp32, flags.device_attention);
        try std.testing.expectEqual(options.compensated_norm and units and fp32, flags.compensated_norm);
    };
}

fn shaped(dims: []const usize) st.Tensor {
    var tensor = st.Tensor{ .dtype = .f32, .rank = @intCast(dims.len), .shape = @splat(1), .bytes = &.{} };
    @memcpy(tensor.shape[0..dims.len], dims);
    return tensor;
}

test "SAM tracker shapes derive the grouped configuration and refuse foreign memory" {
    var shapes = Shapes{
        .output_valid = shaped(&.{ 16, 256 }),
        .mask_tokens = shaped(&.{ 48, 256 }),
        .time_rows = shaped(&.{ 7, 1, 1, 256 }),
        .out_proj = null,
        .pix_feat_proj = shaped(&.{ 256, 256, 1, 1 }),
    };
    const derived = try derive(shapes);
    try std.testing.expectEqual(Derived{ .slots = 16, .candidates = 3, .memory_frames = 7, .memory_width = 256 }, derived);
    try validate(Config.of(.sam31), shapes);
    try std.testing.expectError(error.UnsupportedSamTracker, validate(Config.of(.sam3), shapes));
    shapes.time_rows = shaped(&.{ 5, 1, 1, 256 });
    try std.testing.expectError(error.UnsupportedSamTracker, validate(Config.of(.sam31), shapes));
    shapes.time_rows = shaped(&.{ 7, 1, 1, 256 });
    shapes.out_proj = shaped(&.{ 128, 256, 1, 1 });
    try std.testing.expectError(error.UnsupportedSamTracker, validate(Config.of(.sam31), shapes));
}

test "SAM tracker shapes without grouped slots derive one slot per group" {
    const shapes = Shapes{
        .output_valid = null,
        .mask_tokens = shaped(&.{ 4, 256 }),
        .time_rows = shaped(&.{ 7, 1, 1, 64 }),
        .out_proj = shaped(&.{ 64, 256, 1, 1 }),
        .pix_feat_proj = shaped(&.{ 256, 256, 1, 1 }),
    };
    try std.testing.expectEqual(Derived{ .slots = 1, .candidates = 4, .memory_frames = 7, .memory_width = 64 }, try derive(shapes));
    try validate(Config.of(.sam3), shapes);
    var uneven = shapes;
    uneven.output_valid = shaped(&.{ 3, 256 });
    try std.testing.expectError(error.UnsupportedSamTracker, derive(uneven));
}
