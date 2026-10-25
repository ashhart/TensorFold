//! A native Spark2.5 model on Metal: config, the checkpoint in Metal buffers, its matrices and every pipeline it runs.
const std = @import("std");
const mtl = @import("metal");
const ks = @import("kernel_sources");
const cfg = @import("config.zig");
const ck = @import("../../core/checkpoint_metal.zig");
const shards = @import("../../core/checkpoint.zig");
const rows = @import("../../core/row_projection.zig");
const affine_mm = @import("../../core/affine_mm.zig");
const frags = @import("../../core/frags.zig");
const st = @import("metal_state.zig");
const weights = @import("metal_weights.zig");

const geo = cfg.spark4b;

/// Nemotron's generated keyed sampler, compiled at its 131,072 tokens: Spark's vocabulary.
fn sampler() ?ks.nemotron.Kernel {
    for (ks.nemotron.all) |k| if (std.mem.eql(u8, k.key, "sample")) return k;
    return null;
}

pub const glue_names = [_][:0]const u8{ "sp_embed", "sp_norm", "sp_add", "sp_add_norm", "sp_prep", "sp_gate", "sp_gelu_gu", "sp_decode_vec", "sp_merge_vec", "sp_argmax" };
pub const Glue = enum { embed, norm, add, add_norm, prep, gate, gelu_gu, decode_vec, merge_vec, argmax };

pub const Model = struct {
    gpa: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    config: cfg.Config,
    checkpoint: ck.Checkpoint,
    weights: weights.Weights,
    glue: [glue_names.len]mtl.Pipeline,
    rows: rows.Pipelines,
    /// Prompt attention on the tensor units by layer kind (sliding, full); decode attention is glue's sp_decode_vec.
    prompt_attn: [2]mtl.Pipeline,
    /// A pipelined step's keyed draws (host twins: lanes' gpu_rule, gpu_full): top-k candidates, whole vocabulary.
    sample: mtl.Pipeline,
    sample_full: mtl.Pipeline,
    /// The checkpoint's one width as prompt chunks' tensor-unit matmuls.
    mm: affine_mm.Pipes,
    /// RoPE's inverse frequencies by layer kind (sliding, full): fp32 [head_dim / 2], zero past the rotated dims.
    inv: [2]mtl.Buffer,

    pub fn load(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !*Model {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const m = try gpa.create(Model);
        errdefer gpa.destroy(m);
        m.gpa = gpa;
        m.config = try cfg.Config.read(gpa, io, dir);
        m.device = try mtl.Device.init();
        errdefer m.device.deinit();
        m.queue = try m.device.queue();
        errdefer m.queue.deinit();
        m.checkpoint = ck.Checkpoint.init(gpa);
        errdefer m.checkpoint.deinit();
        const files = try shards.shardFiles(gpa, io, dir);
        defer shards.freeShardFiles(gpa, files);
        for (files) |path| try m.checkpoint.addFile(m.device, path, "");
        m.weights = try weights.load(gpa, m.device, &m.checkpoint, &m.config);
        errdefer m.weights.deinit(gpa);
        {
            const lib = try mtl.Library.fromSource(m.device, ks.spark25_glue, .{ .language = mtl.CompileOptions.mlx().language, .math_functions = .precise });
            defer lib.deinit();
            var made: usize = 0;
            errdefer for (m.glue[0..made]) |p| p.deinit();
            for (glue_names, 0..) |n, i| {
                m.glue[i] = try mtl.Pipeline.init(m.device, lib, n, false);
                made += 1;
            }
        }
        errdefer for (m.glue) |p| p.deinit();
        m.rows = try rows.Pipelines.load(m.device);
        errdefer m.rows.deinit();
        {
            const body = try frags.source(m.device, gpa, ks.spark25_attention_nax);
            defer gpa.free(body);
            var made: usize = 0;
            errdefer for (m.prompt_attn[0..made]) |p| p.deinit();
            for ([_]bool{ true, false }, 0..) |sliding, k| {
                const src = try std.fmt.allocPrint(gpa, "#define DH {d}\n#define NQ {d}\n#define G {d}\n#define KVH {d}\n#define WINDOW {d}\n#define RING {d}\n#define SCALE {e}f\n{s}", .{ cfg.head_dim, geo.query_heads, geo.query_heads / geo.kv_heads, geo.kv_heads, if (sliding) @as(usize, cfg.window) else 0, if (sliding) st.sliding_ring else 0, cfg.scale, body });
                defer gpa.free(src);
                const lib = try mtl.Library.fromSource(m.device, src, mtl.CompileOptions.mlx());
                defer lib.deinit();
                m.prompt_attn[k] = try mtl.Pipeline.init(m.device, lib, "sp_prompt_attention", false);
                made += 1;
            }
        }
        errdefer for (m.prompt_attn) |p| p.deinit();
        {
            const generated = sampler() orelse return error.SparkSamplerMissing;
            if (std.mem.indexOf(u8, generated.source, std.fmt.comptimePrint("constexpr int V = {d};", .{cfg.vocab})) == null) return error.SparkSamplerMissing;
            const lib = try mtl.Library.fromSource(m.device, generated.source, mtl.CompileOptions.mlx());
            defer lib.deinit();
            m.sample = try mtl.Pipeline.init(m.device, lib, generated.function, false);
        }
        errdefer m.sample.deinit();
        {
            const lib = try mtl.Library.fromSource(m.device, ks.nemotron_sample, mtl.CompileOptions.mlx());
            defer lib.deinit();
            m.sample_full = try mtl.Pipeline.init(m.device, lib, "tf_sample_full", false);
        }
        errdefer m.sample_full.deinit();
        {
            const text = try affine_mm.source(m.device, gpa, .{ .bits = m.config.bits, .group = cfg.group });
            defer gpa.free(text);
            const lib = try mtl.Library.fromSource(m.device, text, mtl.CompileOptions.mlx());
            defer lib.deinit();
            var made: usize = 0;
            errdefer for (m.mm[0..made]) |p| p.deinit();
            for (affine_mm.names, 0..) |n, i| {
                m.mm[i] = try mtl.Pipeline.init(m.device, lib, n, false);
                made += 1;
            }
        }
        errdefer for (m.mm) |p| p.deinit();
        var made: usize = 0;
        errdefer for (m.inv[0..made]) |b| b.deinit();
        for ([_]bool{ false, true }, 0..) |full, k| {
            m.inv[k] = try m.device.buffer(cfg.head_dim / 2 * 4, mtl.ResourceOptions.shared);
            made += 1;
            cfg.inverseFrequencies(full, m.inv[k].slice(f32, cfg.head_dim / 2));
        }
        return m;
    }

    pub fn pipe(m: *const Model, g: Glue) mtl.Pipeline {
        return m.glue[@backingInt(g)];
    }

    pub fn deinit(m: *Model) void {
        for (m.inv) |b| b.deinit();
        for (m.mm) |p| p.deinit();
        for (m.prompt_attn) |p| p.deinit();
        m.sample.deinit();
        m.sample_full.deinit();
        m.rows.deinit();
        for (m.glue) |p| p.deinit();
        m.weights.deinit(m.gpa);
        m.checkpoint.deinit();
        m.queue.deinit();
        m.device.deinit();
        m.gpa.destroy(m);
    }
};
