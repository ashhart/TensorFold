//! CUDA launches for Hy4-flash's absorbed MLA projections.
const cuda = @import("cuda");
const weights = @import("mla_weights.zig");
const attention = @import("mla_attention.zig");
const rotary = @import("rope.zig");

pub const Projections = struct {
    image: cuda.Module,
    absorb_kernel: cuda.Function,
    expand_kernel: cuda.Function,
    table_kernel: cuda.Function,
    rotate_kernel: cuda.Function,
    cache_kernel: cuda.Function,
    gate_kernel: cuda.Function,

    pub fn load(driver: *const cuda.Driver) !Projections {
        var image = try cuda.Module.load(driver, cuda.kernels.hy4_mla_project);
        errdefer image.unload();
        return .{
            .image = image,
            .absorb_kernel = try image.function("tf_hy4_absorb_q"),
            .expand_kernel = try image.function("tf_hy4_expand_v"),
            .table_kernel = try image.function("tf_hy4_rotary_table"),
            .rotate_kernel = try image.function("tf_hy4_rotate"),
            .cache_kernel = try image.function("tf_hy4_cache_keys"),
            .gate_kernel = try image.function("tf_hy4_gate_values"),
        };
    }

    pub fn deinit(self: *Projections) void {
        self.image.unload();
        self.* = undefined;
    }

    fn launch(kernel: cuda.Function, stream: cuda.Stream, input: u64, coefficients: u64, output: u64, rows: u32, heads: u32, column_tiles: u32) !void {
        if (rows == 0 or heads == 0) return;
        const rows_per_block = @min(rows, 64);
        var args: cuda.Args = .{};
        inline for (.{ input, coefficients, output, @as(i64, rows), @as(i64, heads), @as(i64, rows_per_block) }) |x| args.add(x);
        try cuda.launch.launch(kernel, .{
            .grid = .{ .x = heads, .y = column_tiles, .z = (rows - 1) / rows_per_block + 1 },
            .block = .{ .x = 128 },
        }, stream, &args);
    }

    pub fn absorb(self: Projections, stream: cuda.Stream, query: u64, wk: u64, output: u64, rows: u32, heads: u32) !void {
        try launch(self.absorb_kernel, stream, query, wk, output, rows, heads, weights.latent / 32);
    }

    pub fn expand(self: Projections, stream: cuda.Stream, latent: u64, wv: u64, output: u64, rows: u32, heads: u32) !void {
        try launch(self.expand_kernel, stream, latent, wv, output, rows, heads, weights.value_width / 16);
    }

    /// Positions [row][3] i32, table [row][32][cos,sin] BF16.
    pub fn rotaryTable(self: Projections, s: cuda.Stream, positions: u64, table: u64, rows: u32, p: rotary.Parameters) !void {
        if (rows == 0) return;
        var args: cuda.Args = .{};
        inline for (.{ positions, table, p }) |x| args.add(x);
        try cuda.launch.launch(self.table_kernel, .{ .grid = .{ .x = rows }, .block = .{ .x = rotary.pairs } }, s, &args);
    }

    pub fn rotate(self: Projections, s: cuda.Stream, input: u64, table: u64, output: u64, shape: rotary.View) !void {
        try shape.check();
        var args: cuda.Args = .{};
        inline for (.{ input, table, output, shape }) |x| args.add(x);
        try cuda.launch.launch(self.rotate_kernel, .{
            .grid = .{ .x = shape.rows, .y = shape.heads },
            .block = .{ .x = rotary.pairs },
        }, s, &args);
    }

    /// Normalize and rotate KV projection rows directly into the lane's assigned latent slots.
    pub fn cacheKeys(self: Projections, s: cuda.Stream, input: u64, gain: u64, table: u64, slots: u64, cache: u64, shape: attention.CacheWrite) !void {
        try shape.check();
        var args: cuda.Args = .{};
        inline for (.{ input, gain, table, slots, cache, shape }) |x| args.add(x);
        try cuda.launch.launch(self.cache_kernel, .{ .grid = .{ .x = shape.rows }, .block = .{ .x = 128 } }, s, &args);
    }

    pub fn gate(self: Projections, s: cuda.Stream, values: u64, logits: u64, rows: u32, heads: u32) !void {
        if (rows == 0) return;
        if (rows > 65535 or heads == 0 or heads > 64) return error.BadHy4GateShape;
        const columns = heads * weights.value_width;
        var args: cuda.Args = .{};
        inline for (.{ values, logits, columns }) |x| args.add(x);
        try cuda.launch.launch(self.gate_kernel, .{ .grid = .{ .x = columns / 128, .y = rows }, .block = .{ .x = 128 } }, s, &args);
    }
};

pub const Attention = struct {
    image: cuda.Module,
    chunks: cuda.Function,
    merge: cuda.Function,

    const head_tile = cuda.mla.heads_tile;
    const shared_bytes = cuda.mla.chunksSmem(head_tile);

    pub const Buffers = struct {
        query: u64,
        cache: u64,
        slots: u64,
        counts: u64,
        sinks: u64,
        scratch: u64,
        scratch_bytes: usize,
        output: u64,
    };

    pub fn load(driver: *const cuda.Driver) !Attention {
        var image = try cuda.Module.load(driver, cuda.kernels.hy4_mla_attention);
        errdefer image.unload();
        const chunks = try image.function("tf_hy4_attend_chunks");
        try chunks.allowDynamicShared(shared_bytes);
        return .{ .image = image, .chunks = chunks, .merge = try image.function("tf_hy4_attend_merge") };
    }

    pub fn deinit(self: *Attention) void {
        self.image.unload();
        self.* = undefined;
    }

    /// Counts must fit key_stride and all active key slots must address the passed cache.
    pub fn attend(self: Attention, s: cuda.Stream, b: Buffers, shape: attention.Shape) !void {
        try shape.check();
        if (b.scratch_bytes < shape.scratchBytes()) return error.ShortHy4AttentionScratch;
        const partials = shape.partialHeads() * @sizeOf(f32);
        const maxima = b.scratch + partials * weights.latent;
        const mass = maxima + partials;
        const rows: i64 = shape.rows;
        const heads: i64 = shape.heads;
        var args: cuda.Args = .{};
        inline for (.{ b.query, b.cache, b.slots, b.counts, b.scratch, maxima, mass, rows, heads, @as(i64, shape.key_stride), attention.score_scale }) |x|
            args.add(x);
        try cuda.launch.launch(self.chunks, .{
            .grid = .{ .x = shape.rows, .y = (shape.heads + head_tile - 1) / head_tile, .z = shape.chunks },
            .block = .{ .x = 256 },
            .shared = shared_bytes,
        }, s, &args);
        var merge: cuda.Args = .{};
        inline for (.{ b.scratch, maxima, mass, b.sinks, b.output, rows, heads, @as(i32, @intCast(shape.chunks)) }) |x| merge.add(x);
        try cuda.launch.launch(self.merge, .{
            .grid = .{ .x = shape.rows, .y = shape.heads },
            .block = .{ .x = 128 },
        }, s, &merge);
    }
};

pub const KvInput = struct {
    projection: u64,
    gain: u64,
    write_slots: u64,
    capacity: u32,
    epsilon: f32,
};

pub const MlaPass = struct {
    query: u64,
    key_weights: u64,
    value_weights: u64,
    value_logits: u64,
    positions: u64,
    rotary_table: u64,
    kv: ?KvInput = null,
    attention: Attention.Buffers,
    values: u64,
};

/// The affine projections feed this pass. The caller owns the KV leases and retained physical key list.
pub fn forward(project: Projections, attend: Attention, s: cuda.Stream, b: MlaPass, shape: attention.Shape, p: rotary.Parameters) !void {
    try shape.check();
    if (b.kv) |kv| try (attention.CacheWrite{ .rows = shape.rows, .capacity = kv.capacity, .epsilon = kv.epsilon }).check();
    try project.rotaryTable(s, b.positions, b.rotary_table, shape.rows, p);
    try project.absorb(s, b.query, b.key_weights, b.attention.query, shape.rows, shape.heads);
    try project.rotate(s, b.query + weights.nope * 2, b.rotary_table, b.attention.query + weights.latent * 2, .{
        .rows = shape.rows,
        .heads = shape.heads,
        .input_stride = weights.query_stride,
        .output_stride = weights.absorbed_stride,
    });
    if (b.kv) |kv| try project.cacheKeys(s, kv.projection, kv.gain, b.rotary_table, kv.write_slots, b.attention.cache, .{
        .rows = shape.rows,
        .capacity = kv.capacity,
        .epsilon = kv.epsilon,
    });
    try attend.attend(s, b.attention, shape);
    try project.expand(s, b.attention.output, b.value_weights, b.values, shape.rows, shape.heads);
    try project.gate(s, b.values, b.value_logits, shape.rows, shape.heads);
}
