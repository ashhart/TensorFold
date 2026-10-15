//! A native Step-5 model: config, the checkpoint in Metal buffers, its matrices and every pipeline the forward runs.
const std = @import("std");
const mtl = @import("metal");
const ks = @import("kernel_sources");
const cfg = @import("config.zig");
const ck = @import("../../core/checkpoint_metal.zig");
const shards = @import("../../core/checkpoint.zig");
const rows = @import("../../core/row_projection.zig");
const route = @import("../../core/moe_route.zig");
const weights = @import("weights.zig");

pub const glue_names = [_][:0]const u8{ "s5_embed", "s5_norm", "s5_final_norm", "s5_qk", "s5_copy", "s5_gate", "s5_swiglu", "s5_add", "s5_moe_add", "s5_index_weights", "s5_index_norm", "s5_summarize", "s5_score", "s5_select", "s5_attend" };
pub const Glue = enum { embed, norm, final_norm, qk, copy, gate, swiglu, add, moe_add, index_weights, index_norm, summarize, score, select, attend };

pub const Model = struct {
    gpa: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    config: cfg.Config,
    checkpoint: ck.Checkpoint,
    weights: weights.Weights,
    glue: [glue_names.len]mtl.Pipeline,
    rows: rows.Pipelines,
    route: [2]mtl.Pipeline,

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
        const c = &m.config;
        {
            const lib = try mtl.Library.fromSource(m.device, ks.step5_glue ++ ks.step5_sparse, .{ .language = mtl.CompileOptions.mlx().language, .math_functions = .precise });
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
            const text = try route.source(gpa, .{ .hidden = c.hidden, .experts = c.experts, .topk = c.topk });
            defer gpa.free(text);
            const lib = try mtl.Library.fromSource(m.device, text, .{ .language = mtl.CompileOptions.mlx().language, .math_functions = .precise });
            defer lib.deinit();
            m.route[0] = try mtl.Pipeline.init(m.device, lib, route.names[0], false);
            errdefer m.route[0].deinit();
            m.route[1] = try mtl.Pipeline.init(m.device, lib, route.names[1], false);
        }
        return m;
    }

    pub fn pipe(m: *const Model, g: Glue) mtl.Pipeline {
        return m.glue[@intFromEnum(g)];
    }

    pub fn deinit(m: *Model) void {
        for (m.route) |p| p.deinit();
        m.rows.deinit();
        for (m.glue) |p| p.deinit();
        m.weights.deinit(m.gpa);
        m.checkpoint.deinit();
        m.queue.deinit();
        m.device.deinit();
        m.gpa.destroy(m);
    }
};
