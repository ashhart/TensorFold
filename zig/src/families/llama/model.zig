//! A native Llama model on Metal: config, the checkpoint in Metal buffers, its matrices and every pipeline it runs.
const std = @import("std");
const mtl = @import("metal");
const ks = @import("kernel_sources");
const cfg = @import("config.zig");
const ck = @import("../../core/checkpoint_metal.zig");
const shards = @import("../../core/checkpoint.zig");
const frags = @import("../../core/frags.zig");
const weights = @import("weights.zig");

pub const glue_names = [_][:0]const u8{ "ll_embed", "ll_norm", "ll_add", "ll_add_norm", "ll_prep", "ll_silu_gu", "ll_decode_vec", "ll_merge_vec", "ll_argmax", "ll_matvec_1", "ll_matvec_2", "ll_matvec_4", "ll_matvec_8", "ll_matvec_16", "ll_matvec_32" };
pub const Glue = enum { embed, norm, add, add_norm, prep, silu_gu, decode_vec, merge_vec, argmax, matvec_1, matvec_2, matvec_4, matvec_8, matvec_16, matvec_32 };

pub const Model = struct {
    gpa: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    config: cfg.Config,
    checkpoint: ck.Checkpoint,
    weights: weights.Weights,
    glue: [glue_names.len]mtl.Pipeline,
    /// Prompt attention on the tensor units; decode attention is glue's ll_decode_vec.
    prompt_attn: mtl.Pipeline,
    /// A sampled step's keyed draw over the whole vocabulary (host twin: lanes' gpu_full).
    sample_full: mtl.Pipeline,
    /// Prompt chunks' bf16 GEMM on the tensor units, by whole M and N tiles.
    gemm: [2][2]mtl.Pipeline,
    /// RoPE's inverse frequencies: fp32 [head_dim / 2].
    inv: mtl.Buffer,

    pub fn load(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !*Model {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const m = try gpa.create(Model);
        errdefer gpa.destroy(m);
        m.gpa = gpa;
        m.config = try cfg.Config.read(gpa, io, dir);
        m.device = try mtl.Device.init();
        errdefer m.device.deinit();
        // prompt chunks' GEMM and attention run on Metal 4 tensor units: an M5 or later
        if (!m.device.tensorUnits()) return error.LlamaNeedsTensorUnits;
        m.queue = try m.device.queue();
        errdefer m.queue.deinit();
        m.checkpoint = ck.Checkpoint.init(gpa);
        errdefer m.checkpoint.deinit();
        const files = try shards.shardFiles(gpa, io, dir);
        defer shards.freeShardFiles(gpa, files);
        for (files) |path| try m.checkpoint.addFile(m.device, path, "");
        m.weights = try weights.load(gpa, &m.checkpoint, &m.config);
        errdefer m.weights.deinit(gpa);
        {
            const src = try std.fmt.allocPrint(gpa, "#define LL_DH {d}\n{s}", .{ cfg.head_dim, ks.llama_glue });
            defer gpa.free(src);
            const lib = try mtl.Library.fromSource(m.device, src, .{ .language = mtl.CompileOptions.mlx().language, .math_functions = .precise });
            defer lib.deinit();
            var made: usize = 0;
            errdefer for (m.glue[0..made]) |p| p.deinit();
            for (glue_names, 0..) |n, i| {
                m.glue[i] = try mtl.Pipeline.init(m.device, lib, n, false);
                made += 1;
            }
        }
        errdefer for (m.glue) |p| p.deinit();
        {
            const body = try frags.source(m.device, gpa, ks.llama_attention_nax);
            defer gpa.free(body);
            const c = m.config;
            const src = try std.fmt.allocPrint(gpa, "#define DH {d}\n#define NQ {d}\n#define G {d}\n#define KVH {d}\n#define SCALE {e}f\n{s}", .{ cfg.head_dim, c.heads, c.group(), c.kv_heads, c.scale(), body });
            defer gpa.free(src);
            const lib = try mtl.Library.fromSource(m.device, src, mtl.CompileOptions.mlx());
            defer lib.deinit();
            m.prompt_attn = try mtl.Pipeline.init(m.device, lib, "ll_prompt_attention", false);
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
        m.inv = try m.device.buffer(cfg.head_dim / 2 * 4, mtl.ResourceOptions.shared);
        m.config.inverseFrequencies(m.inv.slice(f32, cfg.head_dim / 2));
        return m;
    }

    pub fn pipe(m: *const Model, g: Glue) mtl.Pipeline {
        return m.glue[@backingInt(g)];
    }

    pub fn deinit(m: *Model) void {
        m.inv.deinit();
        for (m.gemm) |row| for (row) |p| p.deinit();
        m.sample_full.deinit();
        m.prompt_attn.deinit();
        for (m.glue) |p| p.deinit();
        m.weights.deinit(m.gpa);
        m.checkpoint.deinit();
        m.queue.deinit();
        m.device.deinit();
        m.gpa.destroy(m);
    }
};
