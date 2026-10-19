//! GLM-5.3-Flash's image tower on CUDA as glm5_next/vision.py: patches to the language model's rows (rank 0 encodes).

const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const formats = @import("cuda_formats.zig");

const H: i64 = 1024; // hidden
const HEADS: i64 = 16;
const HD: i64 = 64; // a head's width
const MLP: i64 = 4096;
const PATCH: i64 = 1176; // 3 channels x 2 frames x 14 x 14
const MERGE: i64 = 2;
const OUT: i64 = 4096; // the merged rows' width (the language model's)
const PROJ: i64 = 10240; // the merger's SwiGLU width
pub const DEPTH = 24;
const CHUNK: i64 = 2048; // query rows a scores product covers
const ROWS: i64 = 4096; // rows a linear's fp32 product covers (the scratch's bound)
const EPS: f32 = 1e-5;
const LIMIT: f32 = 10.0;

/// A projection's weight: the checkpoint's bf16 for cuBLASLt, or MXFP8 or NVFP4 as stored for qlinear's prompt rows.
const Lin = union(enum) { bf16: u64, q: cuda.qlinear.Weight };

const Block = struct { n1: u64, n2: u64, qkv_w: Lin, qkv_b: u64, qn: u64, kn: u64, proj_w: Lin, proj_b: u64, g_w: Lin, g_b: u64, u_w: Lin, u_b: u64, d_w: Lin, d_b: u64 };

/// A tensor's module and the MIXED_PRECISION recipe that quantizes it.
const Quant = struct { module: []const u8, recipe: core.modelopt.Recipe };

/// `name`'s module and recipe when the checkpoint's recipes quantize it: its weight and scales are packed together.
fn quantOf(recipes: *const ?core.modelopt.Recipes, name: []const u8) ?Quant {
    const r = if (recipes.*) |*rs| rs else return null;
    inline for (.{ ".weight", ".weight_scale", ".weight_scale_2", ".input_scale" }) |suffix| {
        if (std.mem.endsWith(u8, name, suffix)) {
            const module = name[0 .. name.len - suffix.len];
            return if (r.of(module)) |recipe| .{ .module = module, .recipe = recipe } else null;
        }
    }
    return null;
}

/// A quantized projection as packed: its stored format, shape, packed bytes and NVFP4's global scale.
const Plan = struct { module: []const u8, format: formats.Format, n: usize, k: usize, codes: u64, scales: u64, global: f32 = 0 };

/// `q`'s weight classified against its recipe (formats.of; any other format refused by name) and its packed bytes.
fn plan(ck: *core.Checkpoint, q: Quant) !Plan {
    var nb: [160]u8 = undefined;
    const w = try ck.get(try std.fmt.bufPrint(&nb, "{s}.weight", .{q.module}));
    const scale_name = try std.fmt.bufPrint(&nb, "{s}.weight_scale", .{q.module});
    const s: ?core.checkpoint.Tensor = if (ck.has(scale_name)) try ck.get(scale_name) else null;
    const declared: formats.Declared = .{ .mixed = q.recipe };
    const stored: formats.Stored = .{ .weight = formats.metaOf(&w), .scale = if (s) |*x| formats.metaOf(x) else null };
    const got = formats.of(declared, stored) catch |e| return refuse(q.module, declared, stored, e);
    const n = w.shape[0];
    const k_in = if (got == .nvfp4) 2 * w.shape[1] else w.shape[1];
    if (k_in % 64 != 0) return refuse(q.module, declared, stored, error.BadVisionShape); // the packers' 64-input tiles
    switch (got) {
        .mxfp8 => {
            const k = w.shape[1];
            const npad = cuda.mxfp8.padded(n);
            return .{ .module = q.module, .format = got, .n = n, .k = k, .codes = npad * k, .scales = npad * (k / cuda.mxfp8.group) };
        },
        .nvfp4 => {
            const k = 2 * w.shape[1];
            const npad = cuda.nvfp4.padded(n);
            const g = try ck.get(try std.fmt.bufPrint(&nb, "{s}.weight_scale_2", .{q.module}));
            if (g.dtype != .f32 or g.numel() != 1) return refuse(q.module, declared, stored, error.NoGlobalScale);
            return .{ .module = q.module, .format = got, .n = n, .k = k, .codes = npad / 64 * (k / 64) * 512 * 4, .scales = npad * (k / 16), .global = std.mem.bytesToValue(f32, g.bytes[0..4]) };
        },
        else => return refuse(q.module, declared, stored, error.FormatMismatch),
    }
}

/// The refusal, by the module's name: what it declares, what it stores and why the tower cannot read it.
fn refuse(module: []const u8, declared: formats.Declared, stored: formats.Stored, e: anyerror) error{UnreadableFormat} {
    var why: [320]u8 = undefined;
    std.log.err("{s} ({t}); the image tower reads MXFP8 and W4A16_NVFP4 projections", .{ formats.describe(&why, module, declared, stored), e });
    return error.UnreadableFormat;
}

const LinKey = struct { m: u64, n: u64, k: u64, f32_out: bool };

/// glm_vision's kernels and the stream they run on.
pub const Run = struct {
    stream: cuda.Stream,
    f: Funcs,
};

pub const Funcs = struct {
    to_bf16: cuda.Function,
    bias: cuda.Function,
    embed32: cuda.Function,
    rmsnorm: cuda.Function,
    qkv: cuda.Function,
    softmax: cuda.Function,
    head_out: cuda.Function,
    swiglu: cuda.Function,
    ln_gelu: cuda.Function,
    inject: cuda.Function,

    pub fn load(m: cuda.Module) !Funcs {
        var f: Funcs = undefined;
        inline for (@typeInfo(Funcs).@"struct".field_names) |name| @field(f, name) = try m.function("glm_vis_" ++ name);
        return f;
    }
};

fn launch(r: *const Run, f: cuda.Function, grid: cuda.Dim3, threads: u32, a: *cuda.Args) !void {
    try cuda.launch.launch(f, .{ .grid = grid, .block = .{ .x = threads } }, r.stream, a);
}

pub const Tower = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    lt: cuda.cublaslt.Library,
    mem: cuda.DeviceBuffer,
    blocks: [DEPTH]Block = undefined,
    patch_w: u64 = 0,
    patch_b: u64 = 0,
    post: u64 = 0,
    down_w: u64 = 0, // [4096, 2, 2, 1024]: the conv's taps in the merge block's (row, col) order, then the channels
    down_b: u64 = 0,
    m_proj: Lin = .{ .bf16 = 0 },
    m_ln_w: u64 = 0,
    m_ln_b: u64 = 0,
    m_gate: Lin = .{ .bf16 = 0 },
    m_up: Lin = .{ .bf16 = 0 },
    m_down: Lin = .{ .bf16 = 0 },
    ql: cuda.qlinear.Linear, // the engine's lane and prompt GEMM, for the projections stored MXFP8 or NVFP4
    inv_freq: cuda.DeviceBuffer,
    linears: std.AutoHashMap(LinKey, cuda.cublaslt.Linear),
    ws: cuda.DeviceBuffer,
    ws_len: usize = 64 << 20,
    bytes: u64 = 0,
    stages: ?[]f32 = null, // checks: the fp32 stream after the patch embedding and after every block, read back

    /// The tower's tensors (model.visual.*) in one allocation: bf16 as stored, MXFP8 and NVFP4 packed for ``ql``.
    pub fn load(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, dir: []const u8, ql: cuda.qlinear.Linear) !Tower {
        var ck = try core.Checkpoint.openModelPrefix(gpa, io, dir, "model.visual.");
        defer ck.close();
        var recipes = try formats.readRecipes(gpa, io, dir);
        defer if (recipes) |*r| r.deinit();
        var plans: std.ArrayList(Plan) = .empty;
        defer plans.deinit(gpa);
        var total: u64 = 0;
        for (ck.files.items) |f| for (f.names.keys()) |name| if (std.mem.startsWith(u8, name, "model.visual.")) {
            if (quantOf(&recipes, name)) |q| {
                if (!std.mem.endsWith(u8, name, ".weight")) continue; // its scales are packed with its weight
                const p = try plan(&ck, q);
                try plans.append(gpa, p);
                total += std.mem.alignForward(u64, p.codes, 256) + std.mem.alignForward(u64, p.scales, 256);
            } else total += std.mem.alignForward(u64, (try ck.get(name)).bytes.len, 256);
        };
        var t: Tower = .{ .gpa = gpa, .d = d, .lt = try cuda.cublaslt.Library.open(), .mem = try cuda.DeviceBuffer.alloc(d, total), .ql = ql, .inv_freq = try cuda.DeviceBuffer.alloc(d, 16 * 4), .linears = .init(gpa), .ws = undefined, .bytes = total };
        errdefer t.deinit();
        t.ws = try cuda.DeviceBuffer.alloc(d, t.ws_len);
        var at: u64 = 0;
        var names = std.StringHashMap(u64).init(gpa);
        defer names.deinit();
        var quant = std.StringHashMap(cuda.qlinear.Weight).init(gpa);
        defer quant.deinit();
        for (plans.items) |p| try quant.put(p.module, try t.pack(&ck, p, &at));
        for (ck.files.items) |f| for (f.names.keys()) |name| if (std.mem.startsWith(u8, name, "model.visual.") and quantOf(&recipes, name) == null) {
            const ten = try ck.get(name);
            if (ten.dtype != .bf16) return error.UnsupportedVisionDType;
            var bytes = ten.bytes;
            var perm: []u16 = &.{};
            defer gpa.free(perm);
            if (std.mem.eql(u8, name, "model.visual.downsample.weight")) { // [o, c, 2, 2] -> [o, 2, 2, c]
                if (ten.rank != 4 or ten.dim(0) != OUT or ten.dim(1) != H) return error.BadVisionShape;
                const src = std.mem.bytesAsSlice(u16, @as([]align(1) const u8, bytes));
                perm = try gpa.alloc(u16, src.len);
                const c_n: usize = @intCast(H);
                for (0..@intCast(OUT)) |o| for (0..c_n) |c| for (0..2) |kh| for (0..2) |kw| {
                    perm[((o * 2 + kh) * 2 + kw) * c_n + c] = src[((o * c_n + c) * 2 + kh) * 2 + kw];
                };
                bytes = std.mem.sliceAsBytes(perm);
            }
            const ptr = t.mem.ptr + at;
            try cuda.copy_source.synchronousCopy(d, ptr, bytes, std.heap.page_allocator);
            try names.put(name, ptr);
            at += std.mem.alignForward(u64, bytes.len, 256);
        };
        const get = struct {
            fn f(m: *const std.StringHashMap(u64), comptime fmt: []const u8, args: anytype) !u64 {
                var nb: [128]u8 = undefined;
                const name = try std.fmt.bufPrint(&nb, "model.visual." ++ fmt, args);
                return m.get(name) orelse {
                    std.log.err("glm vision: no tensor {s}", .{name});
                    return error.MissingVisionTensor;
                };
            }
        }.f;
        const lin = struct {
            fn f(m: *const std.StringHashMap(u64), qm: *const std.StringHashMap(cuda.qlinear.Weight), comptime fmt: []const u8, args: anytype) !Lin {
                var nb: [128]u8 = undefined;
                const module = try std.fmt.bufPrint(&nb, "model.visual." ++ fmt, args);
                if (qm.get(module)) |q| return .{ .q = q };
                var wb: [136]u8 = undefined;
                const name = try std.fmt.bufPrint(&wb, "{s}.weight", .{module});
                return .{ .bf16 = m.get(name) orelse {
                    std.log.err("glm vision: no tensor {s}", .{name});
                    return error.MissingVisionTensor;
                } };
            }
        }.f;
        t.patch_w = try get(&names, "patch_embed.proj.weight", .{});
        t.patch_b = try get(&names, "patch_embed.proj.bias", .{});
        t.post = try get(&names, "post_layernorm.weight", .{});
        t.down_w = try get(&names, "downsample.weight", .{});
        t.down_b = try get(&names, "downsample.bias", .{});
        t.m_proj = try lin(&names, &quant, "merger.proj", .{});
        t.m_ln_w = try get(&names, "merger.post_projection_norm.weight", .{});
        t.m_ln_b = try get(&names, "merger.post_projection_norm.bias", .{});
        t.m_gate = try lin(&names, &quant, "merger.gate_proj", .{});
        t.m_up = try lin(&names, &quant, "merger.up_proj", .{});
        t.m_down = try lin(&names, &quant, "merger.down_proj", .{});
        for (&t.blocks, 0..) |*bl, i| {
            bl.* = .{
                .n1 = try get(&names, "blocks.{d}.norm1.weight", .{i}),
                .n2 = try get(&names, "blocks.{d}.norm2.weight", .{i}),
                .qkv_w = try lin(&names, &quant, "blocks.{d}.attn.qkv", .{i}),
                .qkv_b = try get(&names, "blocks.{d}.attn.qkv.bias", .{i}),
                .qn = try get(&names, "blocks.{d}.attn.q_norm.weight", .{i}),
                .kn = try get(&names, "blocks.{d}.attn.k_norm.weight", .{i}),
                .proj_w = try lin(&names, &quant, "blocks.{d}.attn.proj", .{i}),
                .proj_b = try get(&names, "blocks.{d}.attn.proj.bias", .{i}),
                .g_w = try lin(&names, &quant, "blocks.{d}.mlp.gate_proj", .{i}),
                .g_b = try get(&names, "blocks.{d}.mlp.gate_proj.bias", .{i}),
                .u_w = try lin(&names, &quant, "blocks.{d}.mlp.up_proj", .{i}),
                .u_b = try get(&names, "blocks.{d}.mlp.up_proj.bias", .{i}),
                .d_w = try lin(&names, &quant, "blocks.{d}.mlp.down_proj", .{i}),
                .d_b = try get(&names, "blocks.{d}.mlp.down_proj.bias", .{i}),
            };
        }
        // the rotary's frequencies: 1 / 10000^(i / 16), i = 0, 2 .. 30 over 32 (head width / 2)
        var inv: [16]f32 = undefined;
        for (&inv, 0..) |*v, i| v.* = 1.0 / std.math.pow(f32, 10000.0, @as(f32, @floatFromInt(2 * i)) / 32.0);
        try t.inv_freq.upload(0, std.mem.sliceAsBytes(&inv));
        return t;
    }

    pub fn deinit(t: *Tower) void {
        var it = t.linears.valueIterator();
        while (it.next()) |l| l.deinit();
        t.linears.deinit();
        t.mem.free();
        t.inv_freq.free();
        t.ws.free();
        t.lt.close();
    }

    /// Stage `i`'s fp32 stream (N rows) into `dst` (checks only: it waits for the GPU).
    fn stage(t: *Tower, ru: *const Run, dst: []f32, i: usize, x32: u64, N: i64) !void {
        const n: usize = @intCast(N * H);
        if ((i + 1) * n > dst.len) return;
        try ru.stream.synchronize();
        try t.d.check(t.d.api.cuMemcpyDtoH_v2(@ptrCast(dst[i * n ..].ptr), x32, n * 4), "vision stage");
    }

    /// D[m, n] = X[m, k] . W[n, k]^T (fp32 or bf16 out), a plan per shape kept for the next image.
    fn linear(t: *Tower, ru: *const Run, x: u64, w: u64, d: u64, m: i64, n: i64, kk: i64, f32_out: bool) !void {
        const key: LinKey = .{ .m = @intCast(m), .n = @intCast(n), .k = @intCast(kk), .f32_out = f32_out };
        const gop = try t.linears.getOrPut(key);
        if (!gop.found_existing) {
            gop.value_ptr.* = cuda.cublaslt.Linear.init(&t.lt, key.m, key.n, key.k, if (f32_out) .f32 else .bf16, t.ws_len) catch |e| {
                _ = t.linears.remove(key);
                return e;
            };
        }
        try gop.value_ptr.run(x, w, d, t.ws.ptr, t.ws_len, ru.stream.handle);
    }

    /// D[m, n] = X[m, k] . W^T for a projection's weight: cuBLASLt for bf16, qlinear's prompt rows as stored otherwise.
    fn project(t: *Tower, ru: *const Run, x: u64, w: Lin, d: u64, m: i64, n: i64, kk: i64, f32_out: bool) !void {
        switch (w) {
            .bf16 => |p| return t.linear(ru, x, p, d, m, n, kk, f32_out),
            .q => |q| {
                if (q.outputs() != n) return error.BadVisionShape;
                const in: cuda.qlinear.Rows = .{ .x = x, .ldx = @intCast(kk) };
                const rows: usize = @intCast(m);
                return if (f32_out) t.ql.promptF32(ru.stream, q, in, d, rows) else t.ql.prompt(ru.stream, q, in, d, rows);
            },
        }
    }

    /// `p`'s weight packed into ``t.mem`` at ``at`` as qlinear reads it: MXFP8 codes and scales, or NVFP4's.
    fn pack(t: *Tower, ck: *core.Checkpoint, p: Plan, at: *u64) !cuda.qlinear.Weight {
        var nb: [160]u8 = undefined;
        const w = try ck.get(try std.fmt.bufPrint(&nb, "{s}.weight", .{p.module}));
        const s = try ck.get(try std.fmt.bufPrint(&nb, "{s}.weight_scale", .{p.module}));
        const scales = try t.gpa.alloc(u8, p.scales);
        defer t.gpa.free(scales);
        switch (p.format) {
            .mxfp8 => {
                const codes = try t.gpa.alloc(u8, p.codes);
                defer t.gpa.free(codes);
                cuda.mxfp8.packCodes(codes, w.bytes, p.n, p.k);
                cuda.mxfp8.packScales(scales, s.bytes, p.n, p.k);
                const c = try t.put(codes, at);
                return .{ .mxfp8 = cuda.mxfp8.weight(c, try t.put(scales, at), p.n, p.k) };
            },
            .nvfp4 => {
                const words = try t.gpa.alloc(u32, p.codes / 4);
                defer t.gpa.free(words);
                cuda.nvfp4.packWords(words, w.bytes, p.n, p.k);
                cuda.nvfp4.packScales(scales, s.bytes, p.n, p.k);
                const c = try t.put(std.mem.sliceAsBytes(words), at);
                return .{ .nvfp4 = cuda.nvfp4.weight(c, try t.put(scales, at), p.global, p.n, p.k) };
            },
            else => unreachable, // plan refused every other format
        }
    }

    /// `bytes` copied into ``t.mem`` at ``at``, which moves past them to the next 256-byte boundary.
    fn put(t: *Tower, bytes: []const u8, at: *u64) !u64 {
        const ptr = t.mem.ptr + at.*;
        try cuda.copy_source.synchronousCopy(t.d, ptr, bytes, std.heap.page_allocator);
        at.* += std.mem.alignForward(u64, bytes.len, 256);
        return ptr;
    }

    fn grid1(n: i64) cuda.Dim3 {
        return .{ .x = @intCast(@min(std.math.divCeil(i64, n, 256) catch unreachable, 65535)) };
    }

    /// out = bf16(x . W^T + bias) (bias 0: none), in row chunks through ``acc``; with ``x32``, x32 += that instead.
    fn dense(t: *Tower, ru: *const Run, x: u64, w: Lin, bias: u64, out: u64, x32: u64, acc: u64, m: i64, n: i64, kk: i64) !void {
        var r: i64 = 0;
        while (r < m) : (r += ROWS) {
            const rows = @min(ROWS, m - r);
            try t.project(ru, x + @as(u64, @intCast(r * kk * 2)), w, acc, rows, n, kk, true);
            var a: cuda.Args = .{};
            a.add(acc);
            a.add(bias);
            a.add(if (out != 0) out + @as(u64, @intCast(r * n * 2)) else 0);
            a.add(if (x32 != 0) x32 + @as(u64, @intCast(r * n * 4)) else 0);
            a.add(@as(i64, rows * n));
            a.add(@as(i32, @intCast(n)));
            try launch(ru, ru.f.bias, grid1(rows * n), 256, &a);
        }
    }

    /// The limited SwiGLU of two linears (biases 0: none) over x [m, kk] -> out [m, n] bf16, in row chunks.
    fn swiglu(t: *Tower, ru: *const Run, x: u64, gw: Lin, gb: u64, uw: Lin, ub: u64, out: u64, acc: u64, acc2: u64, m: i64, n: i64, kk: i64) !void {
        var r: i64 = 0;
        const step = @divFloor(ROWS * MLP, n); // the fp32 products' rows within the scratch
        while (r < m) : (r += step) {
            const rows = @min(step, m - r);
            const xr = x + @as(u64, @intCast(r * kk * 2));
            try t.project(ru, xr, gw, acc, rows, n, kk, true);
            try t.project(ru, xr, uw, acc2, rows, n, kk, true);
            var a: cuda.Args = .{};
            a.add(acc);
            a.add(acc2);
            a.add(gb);
            a.add(ub);
            a.add(out + @as(u64, @intCast(r * n * 2)));
            a.add(@as(i64, rows * n));
            a.add(@as(i32, @intCast(n)));
            a.add(LIMIT);
            try launch(ru, ru.f.swiglu, grid1(rows * n), 256, &a);
        }
    }

    fn rmsnorm(ru: *const Run, x32: u64, w: u64, y: u64, rows: i64, cols: i64) !void {
        var a: cuda.Args = .{};
        a.add(x32);
        a.add(w);
        a.add(y);
        a.add(@as(i32, @intCast(cols)));
        a.add(EPS);
        try launch(ru, ru.f.rmsnorm, .{ .x = @intCast(rows) }, 256, &a);
    }

    /// Scratch bytes ``encode`` takes for ``patches`` patches in segments of ``seg``.
    pub fn scratchBytes(patches: i64, seg: i64) u64 {
        const N = patches;
        const q = @min(CHUNK, seg);
        return @intCast(N * PATCH * 4 + N * H * 4 + N * H * 2 * 5 + N * MLP * 2 + 2 * ROWS * MLP * 4 + q * seg * 6 + q * HD * 4 + q * 4 + N * 8 + 16 * 1024);
    }

    /// Features of an image or video (``grid`` [t, h, w]): fp32 patches [t h w, 1176] -> ``out`` [t h w / 4, 4096].
    pub fn encode(t: *Tower, ru: *const Run, patches: []const f32, grid: [3]i64, out: u64) !void {
        const d = t.d;
        const gt = grid[0];
        const gh = grid[1];
        const gw = grid[2];
        if (gt < 1 or gh < MERGE or gw < MERGE or @mod(gh, MERGE) != 0 or @mod(gw, MERGE) != 0) return error.BadImageGrid;
        const seg = gh * gw;
        const N = gt * seg;
        if (@as(i64, @intCast(patches.len)) != N * PATCH) return error.BadImagePatches;
        // each patch's (row, col) in merge-block order, every frame group alike
        const pos = try t.gpa.alloc(i32, @intCast(N * 2));
        defer t.gpa.free(pos);
        var i: i64 = 0;
        while (i < N) : (i += 1) {
            const j = @mod(i, seg);
            const in_col = @mod(j, MERGE);
            const in_row = @mod(@divTrunc(j, MERGE), MERGE);
            const block_col = @mod(@divTrunc(j, MERGE * MERGE), @divTrunc(gw, MERGE));
            const block_row = @divTrunc(j, MERGE * MERGE * @divTrunc(gw, MERGE));
            pos[@intCast(i * 2)] = @intCast(block_row * MERGE + in_row);
            pos[@intCast(i * 2 + 1)] = @intCast(block_col * MERGE + in_col);
        }
        var scratch = try cuda.DeviceBuffer.alloc(d, scratchBytes(N, seg));
        defer scratch.free();
        var at: u64 = scratch.ptr;
        const take = struct {
            fn f(p: *u64, n: i64) u64 {
                const r = p.*;
                p.* += std.mem.alignForward(u64, @intCast(n), 256);
                return r;
            }
        }.f;
        const q_rows = @min(CHUNK, seg);
        const p32 = take(&at, N * PATCH * 4);
        const x32 = take(&at, N * H * 4);
        const xn = take(&at, N * H * 2);
        const qh = take(&at, N * H * 2);
        const kh = take(&at, N * H * 2);
        const vt = take(&at, N * H * 2);
        const att = take(&at, N * H * 2);
        const big = take(&at, N * MLP * 2);
        const acc = take(&at, ROWS * MLP * 4);
        const acc2 = take(&at, ROWS * MLP * 4);
        const sc = take(&at, q_rows * seg * 4);
        const pr = take(&at, q_rows * seg * 2);
        const oh = take(&at, q_rows * HD * 4);
        const sums = take(&at, q_rows * 4);
        const dpos = take(&at, N * 8);
        try cuda.copy_source.enqueue(d, dpos, std.mem.sliceAsBytes(pos), ru.stream);
        try cuda.copy_source.enqueue(d, p32, std.mem.sliceAsBytes(patches), ru.stream);
        { // the patch embedding: the patches as bf16, then the conv as a linear (with its bias) into the fp32 stream
            var a: cuda.Args = .{};
            a.add(p32);
            a.add(big);
            a.add(@as(i64, N * PATCH));
            try launch(ru, ru.f.to_bf16, grid1(N * PATCH), 256, &a);
            var r: i64 = 0;
            while (r < N) : (r += ROWS) {
                const rows = @min(ROWS, N - r);
                try t.linear(ru, big + @as(u64, @intCast(r * PATCH * 2)), t.patch_w, acc, rows, H, PATCH, true);
                var b: cuda.Args = .{};
                b.add(acc);
                b.add(t.patch_b);
                b.add(x32 + @as(u64, @intCast(r * H * 4)));
                b.add(@as(i64, rows * H));
                b.add(@as(i32, @intCast(H)));
                try launch(ru, ru.f.embed32, grid1(rows * H), 256, &b);
            }
        }
        const scale: f32 = 1.0 / @sqrt(@as(f32, HD));
        if (t.stages) |dst| try t.stage(ru, dst, 0, x32, N);
        for (t.blocks, 1..) |bl, bi| {
            defer if (t.stages) |dst| t.stage(ru, dst, bi, x32, N) catch {};
            try rmsnorm(ru, x32, bl.n1, xn, N, H);
            try t.dense(ru, xn, bl.qkv_w, bl.qkv_b, big, 0, acc, N, 3 * H, H);
            {
                var a: cuda.Args = .{};
                a.add(big);
                a.add(bl.qn);
                a.add(bl.kn);
                a.add(dpos);
                a.add(t.inv_freq.ptr);
                a.add(qh);
                a.add(kh);
                a.add(vt);
                a.add(@as(i32, @intCast(N)));
                a.add(@as(i32, @intCast(seg)));
                a.add(@as(i32, @intCast(HEADS)));
                a.add(EPS);
                try launch(ru, ru.f.qkv, .{ .x = @intCast(std.math.divCeil(i64, N * HEADS, 8) catch unreachable) }, 256, &a);
            }
            // each frame group attends within itself, per head, in query chunks (q, k [t, H, seg, 64]; v transposed)
            var g: i64 = 0;
            while (g < gt) : (g += 1) {
                var h: i64 = 0;
                while (h < HEADS) : (h += 1) {
                    const blk: u64 = @intCast((g * HEADS + h) * seg * HD * 2);
                    var c: i64 = 0;
                    while (c < seg) : (c += CHUNK) {
                        const rows = @min(CHUNK, seg - c);
                        try t.linear(ru, qh + blk + @as(u64, @intCast(c * HD * 2)), kh + blk, sc, rows, seg, HD, true);
                        var a: cuda.Args = .{};
                        a.add(sc);
                        a.add(pr);
                        a.add(sums);
                        a.add(@as(i32, @intCast(seg)));
                        a.add(scale);
                        try launch(ru, ru.f.softmax, .{ .x = @intCast(rows) }, 256, &a);
                        try t.linear(ru, pr, vt + blk, oh, rows, HD, seg, true);
                        var b: cuda.Args = .{};
                        b.add(oh);
                        b.add(sums);
                        b.add(att + @as(u64, @intCast((g * seg + c) * H * 2)));
                        b.add(@as(i32, @intCast(rows)));
                        b.add(@as(i32, @intCast(HEADS)));
                        b.add(@as(i32, @intCast(h)));
                        try launch(ru, ru.f.head_out, grid1(rows * HD), 256, &b);
                    }
                }
            }
            try t.dense(ru, att, bl.proj_w, bl.proj_b, 0, x32, acc, N, H, H);
            try rmsnorm(ru, x32, bl.n2, xn, N, H);
            try t.swiglu(ru, xn, bl.g_w, bl.g_b, bl.u_w, bl.u_b, big, acc, acc2, N, MLP, H);
            try t.dense(ru, big, bl.d_w, bl.d_b, 0, x32, acc, N, H, MLP);
        }
        // the post-norm, then each merge block's four patches (adjacent rows) through the 2 x 2 conv as one linear
        try rmsnorm(ru, x32, t.post, xn, N, H);
        const M = @divExact(N, MERGE * MERGE);
        try t.dense(ru, xn, .{ .bf16 = t.down_w }, t.down_b, att, 0, acc, M, OUT, 4 * H);
        // the merger: projection, LayerNorm + GELU, the limited SwiGLU to 10240 and back
        var r: i64 = 0;
        while (r < M) : (r += ROWS) {
            const rows = @min(ROWS, M - r);
            try t.project(ru, att + @as(u64, @intCast(r * OUT * 2)), t.m_proj, acc, rows, OUT, OUT, true);
            var a: cuda.Args = .{};
            a.add(acc);
            a.add(t.m_ln_w);
            a.add(t.m_ln_b);
            a.add(xn + @as(u64, @intCast(r * OUT * 2)));
            a.add(@as(i32, @intCast(OUT)));
            a.add(EPS);
            try launch(ru, ru.f.ln_gelu, .{ .x = @intCast(rows) }, 256, &a);
        }
        r = 0;
        const step = @divFloor(ROWS * MLP, PROJ);
        while (r < M) : (r += step) {
            const rows = @min(step, M - r);
            try t.swiglu(ru, xn + @as(u64, @intCast(r * OUT * 2)), t.m_gate, 0, t.m_up, 0, big, acc, acc2, rows, PROJ, OUT);
            try t.project(ru, big, t.m_down, out + @as(u64, @intCast(r * OUT * 2)), rows, OUT, PROJ, false);
        }
        try ru.stream.synchronize(); // the scratch is freed on return
    }
};

test "a projection its recipe quantizes is packed with its scales; biases, norms and bf16 ones read as stored" {
    var r: ?core.modelopt.Recipes = .{ .arena = .init(std.testing.allocator) };
    defer r.?.deinit();
    try r.?.modules.put(r.?.arena.allocator(), "model.visual.blocks.0.attn.qkv", .mxfp8);
    try r.?.modules.put(r.?.arena.allocator(), "model.visual.merger.down_proj", .w4a16_nvfp4);
    const q = quantOf(&r, "model.visual.blocks.0.attn.qkv.weight_scale").?;
    try std.testing.expectEqualStrings("model.visual.blocks.0.attn.qkv", q.module);
    try std.testing.expectEqual(core.modelopt.Recipe.mxfp8, q.recipe);
    try std.testing.expectEqual(core.modelopt.Recipe.w4a16_nvfp4, quantOf(&r, "model.visual.merger.down_proj.weight_scale_2").?.recipe);
    try std.testing.expect(quantOf(&r, "model.visual.blocks.0.attn.qkv.bias") == null);
    try std.testing.expect(quantOf(&r, "model.visual.blocks.0.norm1.weight") == null);
    try std.testing.expect(quantOf(&r, "model.visual.patch_embed.proj.weight") == null);
    const none: ?core.modelopt.Recipes = null;
    try std.testing.expect(quantOf(&none, "model.visual.blocks.0.attn.qkv.weight") == null);
}
