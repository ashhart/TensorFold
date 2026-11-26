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
    tensor_units: bool,
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
        device.compact_gemm = true;
        device.buffer_cache_limit = options.scratch_cache_bytes;
        var text: ?tokenizer.Tokenizer = null;
        if (options.tokenizer_path) |tokenizer_path| text = try tokenizer.loadTokenizer(io, allocator, tokenizer_path);
        errdefer if (text) |*loaded| loaded.deinit();
        const derived = try deriveRoutes(options, patch.dtype, device.device.tensorUnits());
        const model = try allocator.create(Model);
        model.* = .{
            .allocator = allocator,
            .io = io,
            .config = config,
            .store = store,
            .gpu = device,
            .tokenizer = text,
            .precision = derived.precision,
            .tensor_units = derived.tensor_units,
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
        return .{ .precision = self.precision, .tensor_units = self.tensor_units };
    }

    pub fn runtime(self: *Model, scratch: std.mem.Allocator) Runtime {
        return .{ .gpu = self.gpu, .store = &self.store, .scratch = scratch, .config = self.config, .routes = self.routes() };
    }

    pub const predictText = still_image.predictText;
    pub const predictTokens = still_image.predictTokens;
    pub const predictVisual = still_image.predictVisual;
};

fn deriveRoutes(options: LoadOptions, patch: st.DType, tensor_units: bool) error{UnsupportedSamPrecision}!Routes {
    if (patch == .f16 and options.precision != .float32) return error.UnsupportedSamPrecision;
    return .{ .precision = .fp32, .tensor_units = tensor_units };
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

test "SAM load routes follow precision, request and tensor units" {
    try std.testing.expectError(error.UnsupportedSamPrecision, deriveRoutes(.{}, .f16, true));
    try std.testing.expectError(error.UnsupportedSamPrecision, deriveRoutes(.{}, .f16, false));
    try std.testing.expectEqual(Precision.fp32, (try deriveRoutes(.{}, .bf16, true)).precision);
    try std.testing.expectEqual(try deriveRoutes(.{ .precision = .float32 }, .f16, true), try deriveRoutes(.{}, .f32, true));
    for ([_]st.DType{ .f16, .f32 }) |patch| for ([_]bool{ false, true }) |units| {
        const routes = try deriveRoutes(.{ .precision = .float32 }, patch, units);
        try std.testing.expectEqual(Precision.fp32, routes.precision);
        try std.testing.expectEqual(units, routes.tensor_units);
        try std.testing.expectEqual(units, routes.exactDevice());
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
