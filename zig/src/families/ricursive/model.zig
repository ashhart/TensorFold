//! A native Ripple model on Metal: config, the checkpoint in Metal buffers, its matrices and every pipeline it runs.
const std = @import("std");
const mtl = @import("metal");
const ks = @import("kernel_sources");
const cfg = @import("config.zig");
const ck = @import("../../core/checkpoint_metal.zig");
const shards = @import("../../core/checkpoint.zig");
const frags = @import("../../core/frags.zig");
const weights = @import("weights.zig");

/// The family's own glue (glue.metal in kernels/metal/ricursive).
pub const glue_names = [_][:0]const u8{ "rc_embed", "rc_add_norm", "rc_prep", "rc_gate", "rc_silu_clamp", "rc_decode_vec" };
pub const Glue = enum { embed, add_norm, prep, gate, silu_clamp, decode_vec };
/// Llama's kernels it runs as they are: bf16 row projections, the decode attention's merge and the greedy draw.
pub const llama_names = [_][:0]const u8{ "ll_matvec_1", "ll_matvec_2", "ll_matvec_4", "ll_matvec_8", "ll_matvec_16", "ll_matvec_32", "ll_merge_vec", "ll_argmax" };
pub const Llama = enum { matvec_1, matvec_2, matvec_4, matvec_8, matvec_16, matvec_32, merge_vec, argmax };

/// The NVFP4 MLP's row kernels (1 to 32 rows) and its prompt dequantization (fp4.metal).
pub const fp4_names = [_][:0]const u8{ "rc_fp4_matvec_1", "rc_fp4_matvec_2", "rc_fp4_matvec_4", "rc_fp4_matvec_8", "rc_fp4_matvec_16", "rc_fp4_matvec_32", "rc_fp4_dequant" };

pub const Model = struct {
    gpa: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    config: cfg.Config,
    checkpoint: ck.Checkpoint,
    weights: weights.Weights,
    glue: [glue_names.len]mtl.Pipeline,
    llama: [llama_names.len]mtl.Pipeline,
    /// Prompt attention on the tensor units over a window and a ring; decode attention is rc_decode_vec.
    prompt_attn: mtl.Pipeline,
    /// A sampled step's keyed draw over the whole vocabulary (host twin: lanes' gpu_full).
    sample_full: mtl.Pipeline,
    /// Prompt chunks' bf16 GEMM on the tensor units, by whole M and N tiles (llama's).
    gemm: [2][2]mtl.Pipeline,
    /// The NVFP4 release's MLP kernels; null for the bf16 release.
    fp4: ?Fp4 = null,
    /// RoPE's inverse frequencies: fp32 [head_dim / 2].
    inv: mtl.Buffer,
    /// A drafter's taps: each layer's place in its features (-1 untapped), their count, and the copy (dflash2.metal).
    tap_at: [cfg.max_layers]i8 = @splat(-1),
    taps: u32 = 0,
    tap: ?mtl.Pipeline = null,

    pub fn load(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !*Model {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const m = try gpa.create(Model);
        errdefer gpa.destroy(m);
        m.gpa = gpa;
        m.tap_at = @splat(-1);
        m.taps = 0;
        m.tap = null;
        m.config = try cfg.Config.read(gpa, io, dir);
        m.device = try mtl.Device.init();
        errdefer m.device.deinit();
        // prompt chunks' GEMM and attention run on Metal 4 tensor units: an M5 or later
        if (!m.device.tensorUnits()) return error.RicursiveNeedsTensorUnits;
        m.queue = try m.device.queue();
        errdefer m.queue.deinit();
        m.checkpoint = ck.Checkpoint.init(gpa);
        errdefer m.checkpoint.deinit();
        const files = try shards.shardFiles(gpa, io, dir);
        defer shards.freeShardFiles(gpa, files);
        // the release's 2-byte value_res_lambda scalars leave later tensors 2-14 bytes off 8: each placed 256-aligned
        for (files) |path| try m.checkpoint.addFileSplit(m.device, path, "", null, keepAll);
        m.weights = try weights.load(gpa, &m.checkpoint, &m.config);
        errdefer m.weights.deinit(gpa);
        try pipelines(m.device, gpa, "#define RC_DH {d}\n{s}", ks.ricursive_glue, &glue_names, &m.glue);
        errdefer for (m.glue) |p| p.deinit();
        try pipelines(m.device, gpa, "#define LL_DH {d}\n{s}", ks.llama_glue, &llama_names, &m.llama);
        errdefer for (m.llama) |p| p.deinit();
        {
            const src = try attentionSource(m.device, gpa, &m.config);
            defer gpa.free(src);
            const lib = try mtl.Library.fromSource(m.device, src, mtl.CompileOptions.mlx());
            defer lib.deinit();
            m.prompt_attn = try mtl.Pipeline.init(m.device, lib, "rc_prompt_attention", false);
        }
        errdefer m.prompt_attn.deinit();
        {
            const lib = try mtl.Library.fromSource(m.device, ks.nemotron_sample, mtl.CompileOptions.mlx());
            defer lib.deinit();
            m.sample_full = try mtl.Pipeline.init(m.device, lib, "tf_sample_full", false);
        }
        errdefer m.sample_full.deinit();
        {
            const body = try frags.source(m.device, gpa, ks.llama_gemm);
            defer gpa.free(body);
            const lib = try mtl.Library.fromSource(m.device, body, mtl.CompileOptions.mlx());
            defer lib.deinit();
            var made: usize = 0;
            errdefer for (0..made) |i| m.gemm[i / 2][i % 2].deinit();
            for (0..4) |i| {
                var nm: [16]u8 = undefined;
                const full = try std.fmt.bufPrintSentinel(&nm, "ll_gemm_{d}{d}", .{ i / 2, i % 2 }, 0);
                m.gemm[i / 2][i % 2] = try mtl.Pipeline.init(m.device, lib, full, false);
                made += 1;
            }
        }
        errdefer for (m.gemm) |row| for (row) |p| p.deinit();
        m.fp4 = if (m.config.fp4_mlp) try Fp4.init(m.device, gpa) else null;
        errdefer if (m.fp4) |*f| f.deinit();
        m.inv = try m.device.buffer(cfg.head_dim / 2 * 4, mtl.ResourceOptions.shared);
        m.config.inverseFrequencies(m.inv.slice(f32, cfg.head_dim / 2));
        return m;
    }

    pub fn pipe(m: *const Model, g: Glue) mtl.Pipeline {
        return m.glue[@backingInt(g)];
    }

    pub fn ll(m: *const Model, g: Llama) mtl.Pipeline {
        return m.llama[@backingInt(g)];
    }

    /// The drafter's layers, in the order its features concatenate them; the copy compiled once.
    pub fn setTaps(m: *Model, layers: []const u32) !void {
        if (layers.len == 0 or layers.len > 16) return error.UnsupportedRicursiveTaps;
        var at: [cfg.max_layers]i8 = @splat(-1);
        for (layers, 0..) |l, i| {
            if (l >= m.config.layers or at[l] >= 0) return error.UnsupportedRicursiveTaps;
            at[l] = @intCast(i);
        }
        if (m.tap == null) {
            const lib = try mtl.Library.fromSource(m.device, ks.ricursive_dflash2, .{ .language = mtl.CompileOptions.mlx().language, .math_functions = .precise });
            defer lib.deinit();
            m.tap = try mtl.Pipeline.init(m.device, lib, "rc_tap", false);
        }
        m.tap_at = at;
        m.taps = @intCast(layers.len);
    }

    pub fn deinit(m: *Model) void {
        if (m.tap) |p| p.deinit();
        m.inv.deinit();
        if (m.fp4) |*f| f.deinit();
        for (m.gemm) |row| for (row) |p| p.deinit();
        m.sample_full.deinit();
        m.prompt_attn.deinit();
        for (m.llama) |p| p.deinit();
        for (m.glue) |p| p.deinit();
        m.weights.deinit(m.gpa);
        m.checkpoint.deinit();
        m.queue.deinit();
        m.device.deinit();
        m.gpa.destroy(m);
    }
};

/// The NVFP4 pipelines: the row kernels and dequantization (precise math) and the tensor-unit GEMM with its global.
pub const Fp4 = struct {
    rows: [fp4_names.len]mtl.Pipeline,
    gemm: [2][2]mtl.Pipeline,

    fn init(device: mtl.Device, gpa: std.mem.Allocator) !Fp4 {
        var f: Fp4 = undefined;
        {
            const lib = try mtl.Library.fromSource(device, ks.ricursive_fp4, .{ .language = mtl.CompileOptions.mlx().language, .math_functions = .precise });
            defer lib.deinit();
            var made: usize = 0;
            errdefer for (f.rows[0..made]) |p| p.deinit();
            for (fp4_names, 0..) |n, i| {
                f.rows[i] = try mtl.Pipeline.init(device, lib, n, false);
                made += 1;
            }
        }
        errdefer for (f.rows) |p| p.deinit();
        const body = try frags.source(device, gpa, ks.ricursive_fp4_gemm);
        defer gpa.free(body);
        const lib = try mtl.Library.fromSource(device, body, mtl.CompileOptions.mlx());
        defer lib.deinit();
        var made: usize = 0;
        errdefer for (0..made) |i| f.gemm[i / 2][i % 2].deinit();
        for (0..4) |i| {
            var nm: [24]u8 = undefined;
            const full = try std.fmt.bufPrintSentinel(&nm, "rc_fp4_gemm_{d}{d}", .{ i / 2, i % 2 }, 0);
            f.gemm[i / 2][i % 2] = try mtl.Pipeline.init(device, lib, full, false);
            made += 1;
        }
        return f;
    }

    fn deinit(f: *Fp4) void {
        for (f.gemm) |row| for (row) |p| p.deinit();
        for (f.rows) |p| p.deinit();
    }
};

/// No tensor stays in its file: every one is read into the shard's buffer at its own aligned place.
fn keepAll(_: []const u8, _: usize) bool {
    return false;
}

/// The glue text with its head-size define, as the loader compiles it (precise math functions).
pub fn glueSource(a: std.mem.Allocator, comptime fmt: []const u8, body: []const u8) ![]u8 {
    return std.fmt.allocPrint(a, fmt, .{ cfg.head_dim, body });
}

/// The prompt attention's text for a geometry (nax.h inlined for this GPU).
pub fn attentionSource(device: mtl.Device, a: std.mem.Allocator, c: *const cfg.Config) ![]u8 {
    const body = try frags.source(device, a, ks.ricursive_attention_nax);
    defer a.free(body);
    return std.fmt.allocPrint(a, "#define DH {d}\n#define NQ {d}\n#define G {d}\n#define KVH {d}\n#define SCALE {e}f\n{s}", .{ cfg.head_dim, c.heads, c.group(), c.kv_heads, c.scale(), body });
}

fn pipelines(device: mtl.Device, gpa: std.mem.Allocator, comptime fmt: []const u8, body: []const u8, names: []const [:0]const u8, out: []mtl.Pipeline) !void {
    const src = try glueSource(gpa, fmt, body);
    defer gpa.free(src);
    const lib = try mtl.Library.fromSource(device, src, .{ .language = mtl.CompileOptions.mlx().language, .math_functions = .precise });
    defer lib.deinit();
    var made: usize = 0;
    errdefer for (out[0..made]) |p| p.deinit();
    for (names, 0..) |n, i| {
        out[i] = try mtl.Pipeline.init(device, lib, n, false);
        made += 1;
    }
}
