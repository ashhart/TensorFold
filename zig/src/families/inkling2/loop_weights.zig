//! Inkling 2 Mini's text checkpoint on one CUDA rank. Every scheduled visit shares these physical layer weights.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const config = @import("config.zig");
const plan_ = @import("plan.zig");
const modelopt = @import("modelopt.zig");
const weights = @import("weights.zig");

pub const Controller = weights.Controller;
pub const Attention = weights.Attention;

/// Controller order is q, k, v and MLP, matching loop_math_cuda.messages' message layout.
pub const Layer = struct {
    attn: Attention,
    attn_conv: u64,
    mlp_norm: u64,
    mlp_conv: u64,
    router: u64,
    router_bias: u64,
    experts: modelopt.Layer,
    tfq3_experts: ?cuda.tfq3.MoeLayer = null,
    controllers: [4]Controller,
};

pub const Weights = struct {
    gpa: std.mem.Allocator,
    c: config.Config,
    p: plan_.Plan,
    embed: u64 = 0,
    embed_norm: u64 = 0,
    norm: u64 = 0,
    unembed: u64 = 0,
    output_query: u64 = 0,
    attn_queries: []u64 = &.{},
    mlp_queries: []u64 = &.{},
    cross_qm: u64 = 0,
    cross_kv: u64 = 0,
    layers: []Layer = &.{},
    buffers: std.ArrayList(cuda.DeviceBuffer) = .empty,
    tfq3_tables: ?cuda.DeviceBuffer = null,
    tfq4_tables: ?cuda.tfq4.UploadedTables = null,
    tfq3_layers: std.ArrayList(cuda.tfq3.OwnedMoeLayer) = .empty,
    bytes: usize = 0,

    pub fn hasTfq3(w: *const Weights) bool {
        return w.tfq3_tables != null or w.tfq4_tables != null or w.tfq3_layers.items.len > 0;
    }

    pub fn deinit(w: *Weights) void {
        for (w.tfq3_layers.items) |*l| l.deinit();
        w.tfq3_layers.deinit(w.gpa);
        if (w.tfq4_tables) |*t| t.free();
        if (w.tfq3_tables) |*t| t.free();
        for (w.buffers.items) |*b| b.free();
        w.buffers.deinit(w.gpa);
        w.gpa.free(w.layers);
        w.gpa.free(w.attn_queries);
        w.gpa.free(w.mlp_queries);
        w.* = undefined;
    }
};

/// All text tensors occupy one aligned device arena; expert uploads use this same bump allocator.
const Loader = struct {
    w: *Weights,
    arena: cuda.DeviceBuffer,
    used: usize = 0,
    d: *const cuda.Driver,
    io: ?std.Io,
    dir: ?[]const u8,

    pub fn experts(l: *Loader, ck: *core.Checkpoint, c: *const config.Config, p: plan_.Plan, i: u32, out: *Layer) !void {
        out.tfq3_experts = null;
        if (p.layers[i].precision == .tfq3) {
            out.experts = std.mem.zeroes(modelopt.Layer);
            var fname_buf: [32]u8 = undefined;
            const fname = try std.fmt.bufPrint(&fname_buf, "layer_{d:0>2}.tffpw", .{i});
            const layer_path = try std.fs.path.join(l.w.gpa, &.{ l.dir.?, fname });
            defer l.w.gpa.free(layer_path);
            const t3_ptr: u64 = if (l.w.tfq3_tables) |t| t.ptr else 0;
            const t4_ptr: u64 = if (l.w.tfq4_tables) |t| t.lutPtr() else 0;
            const t4_kind: u32 = if (l.w.tfq4_tables) |t| t.table_kind else 0;
            var owned = try cuda.tfq3.uploadTfpwLayerHybrid(
                l.io.?,
                l.d,
                layer_path,
                t3_ptr,
                t4_ptr,
                t4_kind,
                c.text.experts,
                c.text.hidden,
                c.text.expert_inter,
                i,
            );
            errdefer owned.deinit();
            const layer_bytes = owned.w_buf.len + owned.s_buf.len + owned.m_buf.len;
            try l.w.tfq3_layers.append(l.w.gpa, owned);
            out.tfq3_experts = owned.layer;
            l.w.bytes += layer_bytes;
        } else {
            var prefix: [96]u8 = undefined;
            out.experts = try modelopt.load(l, ck, try std.fmt.bufPrint(&prefix, "model.llm.layers.{d}.mlp.experts.", .{i}), if (p.layers[i].precision == .nvfp4) .nvfp4 else .mxfp8, c.text.experts, c.text.hidden, c.text.expert_inter);
        }
    }

    pub fn alloc(l: *Loader, len: usize) !u64 {
        const at = std.mem.alignForward(usize, l.used, 256);
        const end = try std.math.add(usize, at, len);
        if (end > l.arena.len) return error.LoopWeightsOverflow;
        l.used = end;
        l.w.bytes += len;
        return l.arena.ptr + at;
    }

    pub fn put(l: *Loader, bytes: []const u8) !u64 {
        const p = try l.alloc(bytes.len);
        try l.arena.upload(p - l.arena.ptr, bytes);
        return p;
    }

    pub fn tensor(l: *Loader, ck: *core.Checkpoint, comptime fmt: []const u8, args: anytype, dtype: core.safetensors.DType, dims: []const usize) !u64 {
        var buf: [256]u8 = undefined;
        const name = try std.fmt.bufPrint(&buf, fmt, args);
        return l.put((try ck.expect(name, dtype, dims)).bytes);
    }
};

const checkpointBytes = weights.checkpointBytes;

/// Load every text tensor, including all four controller triplets and both cross-pass query matrices.
pub fn load(gpa: std.mem.Allocator, d: *const cuda.Driver, ck: *core.Checkpoint, c: *const config.Config, p: plan_.Plan) !Weights {
    return loadImpl(gpa, null, d, ck, c, p, null);
}

/// Load every text tensor plus any directory-backed TFQ3 (`TFPW` v3 + `pw_v3_tables.bin`) expert layers.
pub fn loadWith(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, ck: *core.Checkpoint, c: *const config.Config, p: plan_.Plan, dir: []const u8) !Weights {
    return loadImpl(gpa, io, d, ck, c, p, dir);
}

fn loadImpl(gpa: std.mem.Allocator, io_opt: ?std.Io, d: *const cuda.Driver, ck: *core.Checkpoint, c: *const config.Config, p: plan_.Plan, dir_opt: ?[]const u8) !Weights {
    if (c.text.loop == null) return error.Inkling2LoopRequired;
    var any_tfq3 = false;
    for (p.layers[0..c.text.layers]) |layer| {
        switch (layer.precision) {
            .nvfp4, .mxfp8 => {},
            .tfq3 => any_tfq3 = true,
            .exl3 => return error.UnsupportedExpertFormat,
        }
    }
    if (any_tfq3 and (io_opt == null or dir_opt == null)) return error.UnsupportedExpertFormat;
    const total = try checkpointBytes(ck);
    var arena = try cuda.DeviceBuffer.alloc(d, @max(total, 256));
    var w: Weights = .{ .gpa = gpa, .c = c.*, .p = p };
    errdefer w.deinit();
    w.buffers.append(gpa, arena) catch |err| {
        arena.free();
        return err;
    };
    if (any_tfq3) {
        if (cuda.tfq3_format.hasTables(gpa, io_opt.?, dir_opt.?)) {
            const tables = try cuda.tfq3.uploadTables(gpa, io_opt.?, d, dir_opt.?);
            w.tfq3_tables = tables;
            w.bytes += tables.len;
        }
        if (try cuda.tfq4.uploadTablesOptional(gpa, io_opt.?, d, dir_opt.?)) |t4| {
            w.tfq4_tables = t4;
            w.bytes += t4.buf.len;
        }
        if (w.tfq3_tables == null and w.tfq4_tables == null) return error.MissingTfq3Tables;
    }
    w.layers = try gpa.alloc(Layer, c.text.layers);
    w.attn_queries = try gpa.alloc(u64, c.text.layers);
    w.mlp_queries = try gpa.alloc(u64, c.text.layers);
    var l: Loader = .{ .w = &w, .arena = arena, .d = d, .io = io_opt, .dir = dir_opt };
    try weights.fill(&l, ck, c, p);
    return w;
}
