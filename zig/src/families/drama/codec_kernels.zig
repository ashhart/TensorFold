//! drama_codec.cu's kernels: loaded from `<kernel_dir>/drama_codec.fatbin`, launched with their exact C argument types.

const std = @import("std");
const cuda = @import("cuda");

pub const fatbin_name = "drama_codec.fatbin";

pub const BiasMode = enum(i32) {
    none = 0,
    /// cuBLASLt's bias epilogue: h(acc + b).
    fused = 1,
    /// cuDNN then add_: h(h(acc) + b).
    separate = 2,
};

pub const Act = enum(i32) { none = 0, gelu = 1 };

/// drama_codec.cu's GemmParams, field for field.
const GemmParams = extern struct {
    x: u64,
    w: u64,
    bias: u64,
    scale: u64,
    residual: u64,
    y: u64,
    m: i64,
    n: i32,
    cin: i32,
    taps: i32,
    dilation: i32,
    ldx: i32,
    ldy: i32,
    ldr: i32,
    period: i32,
    bias_mode: i32,
    act: i32,
};

/// One drama_codec_conv_gemm call: m rows of n columns from taps shifted reads of cin channels (ld in elements).
pub const Gemm = struct {
    x: u64,
    w: u64,
    y: u64,
    m: usize,
    n: usize,
    cin: usize,
    ldx: usize,
    ldy: usize,
    taps: usize = 1,
    dilation: usize = 1,
    bias: u64 = 0,
    bias_mode: BiasMode = .none,
    /// Bias and scale index n % period; 0 means n.
    period: usize = 0,
    act: Act = .none,
    scale: u64 = 0,
    residual: u64 = 0,
    ldr: usize = 0,
};

fn small(v: usize) !i32 {
    if (v > std.math.maxInt(i32)) return error.Invalid;
    return @intCast(v);
}

fn blocks(n: usize, per: usize) !u32 {
    const b = (n + per - 1) / per;
    if (b == 0 or b > std.math.maxInt(i32)) return error.Invalid;
    return @intCast(b);
}

pub const Kernels = struct {
    module: cuda.Module,
    rvq: cuda.Function,
    rms_norm: cuda.Function,
    rope: cuda.Function,
    attention: cuda.Function,
    silu_mul: cuda.Function,
    layer_norm: cuda.Function,
    dwconv7: cuda.Function,
    snake_fn: cuda.Function,
    out_tanh: cuda.Function,
    gemm_fn: cuda.Function,

    pub fn load(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, kernel_dir: []const u8) !Kernels {
        var module = if (kernel_dir.len == 0) try cuda.Module.load(d, cuda.kernels.drama_codec) else blk: {
            const path = try std.fs.path.join(gpa, &.{ kernel_dir, fatbin_name });
            defer gpa.free(path);
            const image = std.Io.Dir.cwd().readFileAllocOptions(io, path, gpa, .limited(1 << 28), .@"16", null) catch |e| {
                std.log.err("drama codec kernels: cannot read {s}: {s}", .{ path, @errorName(e) });
                return e;
            };
            defer gpa.free(image);
            break :blk try cuda.Module.load(d, image);
        };
        errdefer module.unload();
        return .{
            .module = module,
            .rvq = try module.function("drama_codec_rvq_latent"),
            .rms_norm = try module.function("drama_codec_rms_norm"),
            .rope = try module.function("drama_codec_rope"),
            .attention = try module.function("drama_codec_window_attention"),
            .silu_mul = try module.function("drama_codec_silu_mul"),
            .layer_norm = try module.function("drama_codec_layer_norm"),
            .dwconv7 = try module.function("drama_codec_dwconv7"),
            .snake_fn = try module.function("drama_codec_snake"),
            .out_tanh = try module.function("drama_codec_conv_out_tanh"),
            .gemm_fn = try module.function("drama_codec_conv_gemm"),
        };
    }

    pub fn unload(k: *Kernels) void {
        k.module.unload();
        k.* = undefined;
    }

    fn run(f: cuda.Function, s: cuda.Stream, grid: cuda.Dim3, threads: u32, a: *cuda.Args) !void {
        try cuda.launch.launch(f, .{ .grid = grid, .block = .{ .x = threads } }, s, a);
    }

    /// codes [frames][10] u32 -> z [frames][1024].
    pub fn rvqLatent(k: *const Kernels, s: cuda.Stream, codes: u64, semantic_book: u64, residual_books: u64, out_w: u64, out_b: u64, z: u64, frames: usize) !void {
        var a: cuda.Args = .{};
        a.add(codes);
        a.add(semantic_book);
        a.add(residual_books);
        a.add(out_w);
        a.add(out_b);
        a.add(z);
        try run(k.rvq, s, .{ .x = try blocks(frames, 1) }, 256, &a);
    }

    pub fn rmsNorm(k: *const Kernels, s: cuda.Stream, x: u64, w: u64, y: u64, rows: usize, width: usize, eps: f32) !void {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(w);
        a.add(y);
        a.add(try small(width));
        a.add(eps);
        try run(k.rms_norm, s, .{ .x = try blocks(rows, 1) }, 256, &a);
    }

    /// In place on the q and k thirds of qkv [frames][3 * width].
    pub fn ropeQk(k: *const Kernels, s: cuda.Stream, qkv: u64, table: u64, frames: usize, width: usize) !void {
        var a: cuda.Args = .{};
        a.add(qkv);
        a.add(table);
        a.add(try small(width));
        a.add(try small(3 * width));
        try run(k.rope, s, .{ .x = try blocks(frames, 1) }, 256, &a);
    }

    pub fn windowAttention(k: *const Kernels, s: cuda.Stream, qkv: u64, out: u64, frames: usize, heads: usize, window: usize, scale: f32) !void {
        if (window == 0 or window > 128) return error.Invalid;
        var a: cuda.Args = .{};
        a.add(qkv);
        a.add(out);
        a.add(try small(heads));
        a.add(try small(window));
        a.add(scale);
        try run(k.attention, s, .{ .x = try blocks(frames, 1), .y = @intCast(heads) }, 128, &a);
    }

    pub fn siluMul(k: *const Kernels, s: cuda.Stream, gu: u64, out: u64, rows: usize, width: usize) !void {
        var a: cuda.Args = .{};
        a.add(gu);
        a.add(out);
        a.add(@as(i64, @intCast(rows)));
        a.add(try small(width));
        try run(k.silu_mul, s, .{ .x = try blocks(rows * width, 256) }, 256, &a);
    }

    pub fn layerNorm(k: *const Kernels, s: cuda.Stream, x: u64, w: u64, b: u64, y: u64, rows: usize, width: usize, eps: f32) !void {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(w);
        a.add(b);
        a.add(y);
        a.add(try small(width));
        a.add(eps);
        try run(k.layer_norm, s, .{ .x = try blocks(rows, 1) }, 256, &a);
    }

    pub fn dwconv(k: *const Kernels, s: cuda.Stream, x: u64, w: u64, b: u64, y: u64, rows: usize, width: usize) !void {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(w);
        a.add(b);
        a.add(y);
        a.add(@as(i64, @intCast(rows)));
        a.add(try small(width));
        try run(k.dwconv7, s, .{ .x = try blocks(rows * width, 256) }, 256, &a);
    }

    /// Snake over [rows][width]; x and y may be the same buffer.
    pub fn snake(k: *const Kernels, s: cuda.Stream, x: u64, alpha: u64, inv_alpha: u64, y: u64, rows: usize, width: usize) !void {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(alpha);
        a.add(inv_alpha);
        a.add(y);
        a.add(@as(i64, @intCast(rows * width)));
        a.add(try small(width));
        try run(k.snake_fn, s, .{ .x = try blocks(rows * width, 256) }, 256, &a);
    }

    /// The final conv and tanh into FP32 PCM [rows].
    pub fn convOutTanh(k: *const Kernels, s: cuda.Stream, x: u64, w: u64, b: u64, y: u64, rows: usize, width: usize) !void {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(w);
        a.add(b);
        a.add(y);
        a.add(@as(i64, @intCast(rows)));
        a.add(try small(width));
        try run(k.out_tanh, s, .{ .x = try blocks(rows, 256) }, 256, &a);
    }

    pub fn gemm(k: *const Kernels, s: cuda.Stream, g: Gemm) !void {
        const ktot = g.taps * g.cin;
        if (g.m == 0 or g.n == 0 or g.taps == 0 or g.cin == 0) return error.Invalid;
        if (g.cin % 8 != 0 or g.ldx % 8 != 0 or g.ldx < g.cin or g.ldy < g.n) return error.Invalid;
        if (g.x % 16 != 0 or g.w % 16 != 0) return error.Misaligned;
        if ((g.bias_mode != .none) != (g.bias != 0)) return error.Invalid;
        if (g.residual != 0 and g.ldr < g.n) return error.Invalid;
        const p: GemmParams = .{
            .x = g.x,
            .w = g.w,
            .bias = g.bias,
            .scale = g.scale,
            .residual = g.residual,
            .y = g.y,
            .m = @intCast(g.m),
            .n = try small(g.n),
            .cin = try small(g.cin),
            .taps = try small(g.taps),
            .dilation = try small(g.dilation),
            .ldx = try small(g.ldx),
            .ldy = try small(g.ldy),
            .ldr = try small(g.ldr),
            .period = try small(if (g.period == 0) g.n else g.period),
            .bias_mode = @backingInt(g.bias_mode),
            .act = @backingInt(g.act),
        };
        _ = try small(ktot);
        var a: cuda.Args = .{};
        a.add(p);
        try run(k.gemm_fn, s, .{ .x = try blocks(g.m, 128), .y = try blocks(g.n, 128) }, 256, &a);
    }
};
