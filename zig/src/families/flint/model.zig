//! A native Flint model: config, the checkpoint in Metal buffers, its matrices and every pipeline the forward runs.
const std = @import("std");
const mtl = @import("metal");
const ks = @import("kernel_sources");
const cfg = @import("config.zig");
const ck = @import("../../core/checkpoint_metal.zig");
const shards = @import("../../core/checkpoint.zig");
const rows = @import("../../core/row_projection.zig");
const route = @import("../../core/moe_route.zig");
const affine_mm = @import("../../core/affine_mm.zig");
const frags = @import("../../core/frags.zig");
const st = @import("state.zig");
const weights = @import("weights.zig");

/// Nemotron's generated keyed sampler (its vocabulary compiled in; Flint's replaces it).
fn sampler() ?ks.nemotron.Kernel {
    for (ks.nemotron.all) |k| if (std.mem.eql(u8, k.key, "sample")) return k;
    return null;
}

/// A prompt chunk's tensor-unit matmul for one (bits, group) the checkpoint uses.
pub const Mm = struct { bits: u8, group: usize, pipes: affine_mm.Pipes };

pub const glue_names = [_][:0]const u8{ "fl_embed", "fl_norm", "fl_swiglu", "fl_add", "fl_moe_add", "fl_sort", "fl_rows_by_order", "fl_route_pick", "fl_decode_vec", "fl_merge_vec", "fl_add_norm", "fl_moe_add_norm", "fl_prep", "fl_swiglu_gu", "fl_argmax" };
pub const Glue = enum { embed, norm, swiglu, add, moe_add, sort, rows_by_order, route_pick, decode_vec, merge_vec, add_norm, moe_add_norm, prep, swiglu_gu, argmax };

pub const Model = struct {
    gpa: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    config: cfg.Config,
    checkpoint: ck.Checkpoint,
    weights: weights.Weights,
    glue: [glue_names.len]mtl.Pipeline,
    rows: rows.Pipelines,
    route: mtl.Pipeline, // the router's logits (moe_route's first kernel); glue.metal's fl_route_pick takes the top-k
    /// Prompt attention on the tensor units by layer kind (sliding, full); decode attention is glue's fl_decode_vec.
    prompt_attn: [2]mtl.Pipeline,
    /// A pipelined step's keyed draws (host twins: lanes' gpu_rule, gpu_full): top-k candidates, whole vocabulary.
    sample: mtl.Pipeline,
    sample_full: mtl.Pipeline,
    mm: [4]Mm = undefined,
    mms: usize = 0,

    pub fn load(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !*Model {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const m = try gpa.create(Model);
        errdefer gpa.destroy(m);
        m.gpa = gpa;
        m.mms = 0;
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
        const c = &m.config;
        {
            const lib = try mtl.Library.fromSource(m.device, ks.flint_glue, .{ .language = mtl.CompileOptions.mlx().language, .math_functions = .precise });
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
            const text = try route.source(gpa, .{ .hidden = c.hidden, .experts = c.experts, .topk = c.topk, .round_logits = true });
            defer gpa.free(text);
            const lib = try mtl.Library.fromSource(m.device, text, .{ .language = mtl.CompileOptions.mlx().language, .math_functions = .precise });
            defer lib.deinit();
            m.route = try mtl.Pipeline.init(m.device, lib, route.names[0], false);
        }
        errdefer m.route.deinit();
        if (c.head_dim != 128 or c.heads / c.kv_heads > 16 or 16 % (c.heads / c.kv_heads) != 0) return error.UnsupportedFlintAttention;
        {
            const body = try frags.source(m.device, gpa, ks.flint_attention_nax);
            defer gpa.free(body);
            var made: usize = 0;
            errdefer for (0..made) |k| {
                m.prompt_attn[k].deinit();
            };
            for ([_]bool{ true, false }, 0..) |sliding, k| {
                const src = try std.fmt.allocPrint(gpa, "#define DH {d}\n#define NQ {d}\n#define G {d}\n#define KVH {d}\n#define WINDOW {d}\n#define RING {d}\n#define SCALE {e}f\n{s}", .{ c.head_dim, c.heads, c.heads / c.kv_heads, c.kv_heads, if (sliding) c.window else 0, if (sliding) st.slidingRing(c) else 0, 1.0 / @sqrt(@as(f64, @floatFromInt(c.head_dim))), body });
                defer gpa.free(src);
                const lib = try mtl.Library.fromSource(m.device, src, mtl.CompileOptions.mlx());
                defer lib.deinit();
                m.prompt_attn[k] = try mtl.Pipeline.init(m.device, lib, "fl_prompt_attention", false);
                made += 1;
            }
        }
        errdefer {
            for (m.prompt_attn) |p| p.deinit();
        }
        {
            const generated = sampler() orelse return error.FlintSamplerMissing;
            const v = try std.fmt.allocPrint(gpa, "constexpr int V = {d};", .{c.vocab});
            defer gpa.free(v);
            const text = try std.mem.replaceOwned(u8, gpa, generated.source, "constexpr int V = 131072;", v);
            defer gpa.free(text);
            if (std.mem.indexOf(u8, text, v) == null) return error.FlintSamplerMissing;
            const lib = try mtl.Library.fromSource(m.device, text, mtl.CompileOptions.mlx());
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
        errdefer for (m.mm[0..m.mms]) |f| for (f.pipes) |p| p.deinit();
        try m.addMm(m.weights.head);
        for (m.weights.layers) |l| {
            try m.addMm(l.attn.qkv);
            switch (l.ffn) {
                .dense => |d| try m.addMm(d.gu),
                .moe => |e| {
                    try m.addMm(e.gate.one);
                    try m.addMm(e.down.one);
                    try m.addMm(e.shared.gu);
                },
            }
        }
        return m;
    }

    /// Pipelines for `w`'s format, compiled once per format (at most 4).
    fn addMm(m: *Model, w: rows.Weights) !void {
        if (m.mmFor(w) != null) return;
        if (m.mms == m.mm.len) return error.TooManyFlintFormats;
        const text = try affine_mm.source(m.device, m.gpa, .{ .bits = w.bits, .group = @intCast(w.group) });
        defer m.gpa.free(text);
        const lib = try mtl.Library.fromSource(m.device, text, mtl.CompileOptions.mlx());
        defer lib.deinit();
        var f = Mm{ .bits = w.bits, .group = w.group, .pipes = undefined };
        var made: usize = 0;
        errdefer for (f.pipes[0..made]) |p| p.deinit();
        for (affine_mm.names, 0..) |n, i| {
            f.pipes[i] = try mtl.Pipeline.init(m.device, lib, n, false);
            made += 1;
        }
        m.mm[m.mms] = f;
        m.mms += 1;
    }

    /// The tensor-unit pipelines for `w`'s format, if loaded.
    pub fn mmFor(m: *const Model, w: rows.Weights) ?affine_mm.Pipes {
        for (m.mm[0..m.mms]) |f| if (f.bits == w.bits and f.group == w.group) return f.pipes;
        return null;
    }

    pub fn pipe(m: *const Model, g: Glue) mtl.Pipeline {
        return m.glue[@backingInt(g)];
    }

    pub fn deinit(m: *Model) void {
        for (m.mm[0..m.mms]) |f| for (f.pipes) |p| p.deinit();
        for (m.prompt_attn) |p| p.deinit();
        m.sample.deinit();
        m.sample_full.deinit();
        m.route.deinit();
        m.rows.deinit();
        for (m.glue) |p| p.deinit();
        m.weights.deinit(m.gpa);
        m.checkpoint.deinit();
        m.queue.deinit();
        m.device.deinit();
        m.gpa.destroy(m);
    }
};
