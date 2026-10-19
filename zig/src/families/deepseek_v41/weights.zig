//! Strict pinned backbone schema and borrowed resident bindings; only expert descriptor tables are owned.
const std = @import("std");
const mtl = @import("metal");
const cp = @import("../../core/checkpoint_metal.zig");
const st = @import("../../core/safetensors.zig");
const exl3 = @import("../../core/exl3_projection.zig");
const load = @import("metal_load.zig");

pub const Tensor = cp.Tensor;
pub const Matrix = exl3.Matrix;

/// A kept-dense repack stores small matrices as a bf16 weight beside the rotation scales.
pub const Dense = struct { weight: Tensor, suh: Tensor, svh: Tensor, k: usize, n: usize };
pub const AnyMatrix = union(enum) { trellis: Matrix, dense: Dense };
pub const Hc = struct { projection: Tensor, scale: Tensor, base: Tensor };
pub const Compressor = struct { wkv: AnyMatrix, wgate: ?AnyMatrix, norm: Tensor };
pub const Indexer = struct { wq_b: AnyMatrix, wk: ?AnyMatrix, weights_proj: Tensor, k_norm: ?Tensor };
pub const Attention = struct {
    wq_a: Matrix,
    wq_b: Matrix,
    wkv: Matrix,
    wo_b: Matrix,
    wo_a: [8]Matrix,
    /// The eight wo_a slices as expert descriptors: one indexed product serves every slice.
    wo_a_descriptors: mtl.Buffer,
    sink: Tensor,
    q_norm: Tensor,
    kv_norm: Tensor,
    compressor: ?Compressor,
    indexer: ?Indexer,
};
/// Projection order is w1 (gate), w3 (up), w2 (down), for shared and routed experts alike.
pub const Moe = struct {
    gate: Tensor,
    bias: Tensor,
    shared: [3]Matrix,
    descriptors: [3]mtl.Buffer,
};
/// `table`/`scales` are null when the tables are streamed from their shards (load.Engram.streamed).
pub const Engram = struct { table: ?Tensor, scales: ?Tensor, q_weight: Tensor, k_weight: Tensor, wkv: Matrix };
pub const Layer = struct {
    attn_norm: Tensor,
    ffn_norm: Tensor,
    hc_attn: Hc,
    hc_ffn: Hc,
    attn: Attention,
    moe: Moe,
    engram: ?Engram,
};
pub const Model = struct {
    embed: Tensor,
    head: Matrix,
    norm: Tensor,
    layers: [40]Layer,

    /// Checkpoint buffers must outlive this borrowed model and every submitted command using it.
    pub fn deinit(self: *Model) void {
        for (&self.layers) |*layer| {
            for (layer.moe.descriptors) |buffer| buffer.deinit();
            layer.attn.wo_a_descriptors.deinit();
        }
        self.* = undefined;
    }
    pub fn registerResidency(self: *const Model, set: mtl.ResidencySet) void {
        for (&self.layers) |*layer| {
            for (layer.moe.descriptors) |buffer| set.add(buffer);
            set.add(layer.attn.wo_a_descriptors);
        }
    }
    pub fn residentBytes(_: *const Model) usize {
        return sizeofDescriptors();
    }
};

/// Admission adds these bytes to load.Plan.bytes; no dequantized weights or duplicate projections are allocated.
pub fn sizeofDescriptors() usize {
    return 40 * (3 * 385 + 8) * @sizeOf(exl3.ExpertDescriptor);
}

/// Every executed tensor is checked using headers only, before checkpoint materialization or model allocation.
pub fn validate(plan: *const load.Plan) !void {
    try populate(false, plan, null, plan.engram_files != null);
}

/// Bind resident tensors and marker payloads; route descriptors replace redundant host matrix metadata.
pub fn bind(device: mtl.Device, checkpoint: *const cp.Checkpoint, engram: load.Engram) !Model {
    var model: Model = undefined;
    try populate(true, checkpoint, &model, engram == .streamed);
    var owned: usize = 0;
    errdefer for (0..owned) |i| model.layers[i / 3].moe.descriptors[i % 3].deinit();
    var slices: usize = 0;
    errdefer for (model.layers[0..slices]) |*layer| layer.attn.wo_a_descriptors.deinit();
    for (&model.layers) |*layer| {
        const buffer = try device.buffer(8 * @sizeOf(exl3.ExpertDescriptor), mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        layer.attn.wo_a_descriptors = buffer;
        slices += 1;
        const out: [*]exl3.ExpertDescriptor = @ptrCast(@alignCast(buffer.contents()));
        for (layer.attn.wo_a, 0..) |m, slice| out[slice] = try exl3.descriptor(m);
    }
    var prefix: [128]u8 = undefined;
    for (&model.layers, 0..) |*layer, layer_id| {
        inline for (.{ "w1", "w3", "w2" }, 0..) |part, projection| {
            // Entry 384 is the shared expert, so merged products can route it as a seventh row.
            const bytes = 385 * @sizeOf(exl3.ExpertDescriptor);
            const buffer = try device.buffer(bytes, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
            layer.moe.descriptors[projection] = buffer;
            owned += 1;
            const out: [*]exl3.ExpertDescriptor = @ptrCast(@alignCast(buffer.contents()));
            for (0..384) |expert| {
                var m: Matrix = undefined;
                const k: usize = if (projection == 2) 2304 else 5120;
                const n: usize = if (projection == 2) 5120 else 2304;
                try matrix(true, checkpoint, try named(&prefix, "layers.{d}.ffn.experts.{d}.{s}", .{ layer_id, expert, part }), k, n, &m);
                out[expert] = try exl3.descriptor(m);
            }
            out[384] = try exl3.descriptor(layer.moe.shared[projection]);
        }
    }
    return model;
}

/// A borrowed CPU/GPU matrix view for checkpoint qualification, using the same strict storage contract.
pub fn lookup(checkpoint: *const cp.Checkpoint, prefix: []const u8, k: usize, n: usize) !Matrix {
    var out: Matrix = undefined;
    try matrix(true, checkpoint, prefix, k, n, &out);
    return out;
}

fn named(buffer: []u8, comptime fmt: []const u8, args: anytype) ![]const u8 {
    return std.fmt.bufPrint(buffer, fmt, args);
}

/// A plain tensor the encoders decode as f32, f16 or bf16; the shape must match exactly.
fn plainEither(comptime binding: bool, source: anytype, name: []const u8, shape: []const usize, destination: ?*Tensor) !void {
    const tensor = try source.get(name);
    if ((tensor.dtype != .f32 and tensor.dtype != .f16 and tensor.dtype != .bf16) or tensor.rank != shape.len or
        !std.mem.eql(usize, tensor.shape[0..tensor.rank], shape))
    {
        std.log.err("unsupported tensor {s}: {s} {any}", .{ name, @tagName(tensor.dtype), tensor.shape[0..tensor.rank] });
        return error.UnsupportedTensor;
    }
    if (binding) {
        var count: usize = 1;
        for (shape) |dim| count = try std.math.mul(usize, count, dim);
        if (tensor.bytes != try std.math.mul(usize, count, tensor.dtype.size())) return error.UnsupportedTensor;
        if (tensor.offset % tensor.dtype.size() != 0 or tensor.offset > tensor.buffer.length() or
            tensor.bytes > tensor.buffer.length() - tensor.offset) return error.InvalidTensorView;
        destination.?.* = tensor;
    } else if (tensor.begin % 4 != 0) return error.InvalidTensorView;
}

fn plain(comptime binding: bool, source: anytype, name: []const u8, dtype: st.DType, shape: []const usize, destination: ?*Tensor) !void {
    const tensor = try source.get(name);
    if (tensor.dtype != dtype or tensor.rank != shape.len or
        !std.mem.eql(usize, tensor.shape[0..tensor.rank], shape))
    {
        std.log.err("unsupported backbone tensor {s}: {s} {any}", .{ name, @tagName(tensor.dtype), tensor.shape[0..tensor.rank] });
        return error.UnsupportedTensor;
    }
    const offset = if (binding) tensor.offset else tensor.begin;
    if (offset % dtype.size() != 0) return error.InvalidTensorView;
    if (binding) {
        var count: usize = 1;
        for (shape) |dim| count = try std.math.mul(usize, count, dim);
        if (tensor.bytes != try std.math.mul(usize, count, dtype.size())) return error.UnsupportedTensor;
        if (tensor.offset % dtype.size() != 0 or tensor.offset > tensor.buffer.length() or
            tensor.bytes > tensor.buffer.length() - tensor.offset) return error.InvalidTensorView;
        destination.?.* = tensor;
    }
}

fn has(source: anytype, name: []const u8) bool {
    return source.has(name);
}

const Scale = struct { name: []const u8, format: exl3.ScaleFormat };
fn scale(comptime binding: bool, source: anytype, prefix: []const u8, size: usize, comptime fp: []const u8, comptime signs: []const u8, storage: []u8) !Scale {
    var alternative: [160]u8 = undefined;
    const f = try named(storage, "{s}.{s}", .{ prefix, fp });
    const s = try named(&alternative, "{s}.{s}", .{ prefix, signs });
    const have_fp = has(source, f);
    if (have_fp == has(source, s)) return error.MissingOrAmbiguousExl3Scale;
    const name = if (have_fp) f else try named(storage, "{s}.{s}", .{ prefix, signs });
    const t = try source.get(name);
    if (t.rank != 1 or t.shape[0] != (if (have_fp) size else size / 16) or
        t.dtype != (if (have_fp) st.DType.f16 else st.DType.i16)) return error.UnsupportedExl3Scale;
    const offset = if (binding) t.offset else t.begin;
    if (offset % 2 != 0) return error.InvalidTensorView;
    if (binding) {
        const bytes = (if (have_fp) size else size / 16) * 2;
        if (t.bytes != bytes or t.offset % 2 != 0 or t.offset > t.buffer.length() or
            bytes > t.buffer.length() - t.offset) return error.InvalidTensorView;
    }
    return .{ .name = name, .format = if (have_fp) .f16 else .packed_signs };
}

/// Compressor/indexer matrices: quantized in release packs, dense beside rotation scales in rank-split repacks.
fn matrixAny(comptime binding: bool, source: anytype, prefix: []const u8, k: usize, n: usize, destination: ?*AnyMatrix) !void {
    var name_buf: [160]u8 = undefined;
    if (has(source, try named(&name_buf, "{s}.weight", .{prefix}))) {
        const weight_name = try named(&name_buf, "{s}.weight", .{prefix});
        const w = try source.get(weight_name);
        if (w.dtype != .bf16 or w.rank != 2 or w.shape[0] != n or w.shape[1] != k) {
            std.log.err("unsupported dense tensor {s}: {s} {any}", .{ weight_name, @tagName(w.dtype), w.shape[0..w.rank] });
            return error.UnsupportedTensor;
        }
        const su = try source.get(try named(&name_buf, "{s}.suh", .{prefix}));
        const sv = try source.get(try named(&name_buf, "{s}.svh", .{prefix}));
        if (su.dtype != .f16 or su.rank != 1 or su.shape[0] != k or
            sv.dtype != .f16 or sv.rank != 1 or sv.shape[0] != n) return error.UnsupportedExl3Scale;
        if (binding) {
            if (w.offset % 2 != 0) return error.InvalidTensorView;
            destination.?.* = .{ .dense = .{ .weight = w, .suh = su, .svh = sv, .k = k, .n = n } };
        }
        return;
    }
    var m: Matrix = undefined;
    try matrix(binding, source, prefix, k, n, if (binding) &m else null);
    if (binding) destination.?.* = .{ .trellis = m };
}

fn matrix(comptime binding: bool, source: anytype, prefix: []const u8, k: usize, n: usize, destination: ?*Matrix) !void {
    var name: [160]u8 = undefined;
    const trellis = try source.get(try named(&name, "{s}.trellis", .{prefix}));
    if (trellis.dtype != .i16 or trellis.rank != 3 or trellis.shape[0] != k / 16 or
        trellis.shape[1] != n / 16 or trellis.shape[2] % 8 != 0) return error.UnsupportedExl3Shape;
    if ((if (binding) trellis.offset else trellis.begin) % 2 != 0) return error.InvalidTensorView;
    const halves = trellis.shape[2] / 8;
    if (halves < 2 or halves > 16 or (halves % 2 != 0 and halves > 7)) return error.UnsupportedExl3Width;
    // Pinned checkpoint uses mul1 throughout; missing markers cannot silently select the legacy codebook.
    const marker = try source.get(try named(&name, "{s}.mul1", .{prefix}));
    if (marker.dtype != .i32 or marker.rank != 0) return error.UnsupportedExl3Marker;
    if ((if (binding) marker.offset else marker.begin) % 4 != 0) return error.InvalidTensorView;
    if (has(source, try named(&name, "{s}.mcg", .{prefix})) or
        has(source, try named(&name, "{s}.3inst", .{prefix})) or
        has(source, try named(&name, "{s}.bias", .{prefix})))
    {
        std.log.err("unexpected exl3 parts beside {s}", .{prefix});
        return error.UnsupportedExl3Parts;
    }
    var su_name: [160]u8 = undefined;
    var sv_name: [160]u8 = undefined;
    const su = try scale(binding, source, prefix, k, "suh", "su", &su_name);
    const sv = try scale(binding, source, prefix, n, "svh", "sv", &sv_name);
    if (binding) {
        const expected = k / 16 * (n / 16) * halves * 16;
        if (trellis.bytes != expected or trellis.offset % 2 != 0 or
            trellis.offset > trellis.buffer.length() or expected > trellis.buffer.length() - trellis.offset)
            return error.InvalidTensorView;
        if (marker.bytes != 4 or marker.offset % 4 != 0 or marker.offset > marker.buffer.length() or
            4 > marker.buffer.length() - marker.offset) return error.InvalidTensorView;
        const marker_bytes = marker.buffer.contents()[marker.offset..][0..4];
        if (std.mem.readInt(u32, marker_bytes, .little) != exl3.format.Mul1Mul) return error.UnsupportedExl3Marker;
        const suh = try source.get(su.name);
        const svh = try source.get(sv.name);
        destination.?.* = .{
            .trellis = .{ .buffer = trellis.buffer, .offset = trellis.offset },
            .suh = .{ .buffer = suh.buffer, .offset = suh.offset },
            .svh = .{ .buffer = svh.buffer, .offset = svh.offset },
            .k = k,
            .n = n,
            .width = .{ .halves = @intCast(halves) },
            .codebook = .mul1,
            .suh_format = su.format,
            .svh_format = sv.format,
        };
        try destination.?.validate();
    }
}

fn populate(comptime binding: bool, source: anytype, model: ?*Model, streamed: bool) !void {
    try plain(binding, source, "embed.weight", .bf16, &.{ 129280, 5120 }, if (binding) &model.?.embed else null);
    try plain(binding, source, "norm.weight", .bf16, &.{5120}, if (binding) &model.?.norm else null);
    try matrix(binding, source, "head", 5120, 129280, if (binding) &model.?.head else null);
    var name: [160]u8 = undefined;
    var prefix: [128]u8 = undefined;
    for (0..40) |layer| {
        const out: ?*Layer = if (binding) &model.?.layers[layer] else null;
        try plain(binding, source, try named(&name, "layers.{d}.attn_norm.weight", .{layer}), .bf16, &.{5120}, if (binding) &out.?.attn_norm else null);
        try plain(binding, source, try named(&name, "layers.{d}.ffn_norm.weight", .{layer}), .bf16, &.{5120}, if (binding) &out.?.ffn_norm else null);
        inline for (.{ "attn", "ffn" }) |kind| {
            const hc: ?*Hc = if (binding) (if (std.mem.eql(u8, kind, "attn")) &out.?.hc_attn else &out.?.hc_ffn) else null;
            try plain(binding, source, try named(&name, "layers.{d}.hc_{s}_fn", .{ layer, kind }), .f32, &.{ 24, 20480 }, if (binding) &hc.?.projection else null);
            try plain(binding, source, try named(&name, "layers.{d}.hc_{s}_scale", .{ layer, kind }), .f32, &.{3}, if (binding) &hc.?.scale else null);
            try plain(binding, source, try named(&name, "layers.{d}.hc_{s}_base", .{ layer, kind }), .f32, &.{24}, if (binding) &hc.?.base else null);
        }
        const attn: ?*Attention = if (binding) &out.?.attn else null;
        inline for (.{ "wq_a", "wq_b", "wkv", "wo_b" }, .{ 5120, 1280, 5120, 8192 }, .{ 1280, 32768, 512, 5120 }) |part, k, n| {
            try matrix(binding, source, try named(&prefix, "layers.{d}.attn.{s}", .{ layer, part }), k, n, if (binding) &@field(attn.?, part) else null);
        }
        for (0..8) |slice| try matrix(binding, source, try named(&prefix, "layers.{d}.attn.wo_a.slice.{d}", .{ layer, slice }), 4096, 1024, if (binding) &attn.?.wo_a[slice] else null);
        try plain(binding, source, try named(&name, "layers.{d}.attn.attn_sink", .{layer}), .f32, &.{64}, if (binding) &attn.?.sink else null);
        try plain(binding, source, try named(&name, "layers.{d}.attn.q_norm.weight", .{layer}), .bf16, &.{1280}, if (binding) &attn.?.q_norm else null);
        try plain(binding, source, try named(&name, "layers.{d}.attn.kv_norm.weight", .{layer}), .bf16, &.{512}, if (binding) &attn.?.kv_norm else null);
        const owner = layer == 2 or layer == 8 or layer == 14 or layer == 20;
        if (binding) {
            attn.?.compressor = if (owner) @as(Compressor, undefined) else null;
            attn.?.indexer = if (owner or layer == 24 or layer == 28 or layer == 32 or layer == 36)
                @as(Indexer, undefined)
            else
                null;
        }
        if (owner) {
            const compressor: ?*Compressor = if (binding) &attn.?.compressor.? else null;
            try matrixAny(binding, source, try named(&prefix, "layers.{d}.attn.compressor.wkv", .{layer}), 5120, 512, if (binding) &compressor.?.wkv else null);
            try plain(binding, source, try named(&name, "layers.{d}.attn.compressor.norm.weight", .{layer}), .bf16, &.{512}, if (binding) &compressor.?.norm else null);
            if (binding) compressor.?.wgate = if (layer < 20) @as(AnyMatrix, undefined) else null;
            if (layer < 20) try matrixAny(binding, source, try named(&prefix, "layers.{d}.attn.compressor.wgate", .{layer}), 5120, 512, if (binding) &compressor.?.wgate.? else null);
        }
        if (owner or layer == 24 or layer == 28 or layer == 32 or layer == 36) {
            const indexer: ?*Indexer = if (binding) &attn.?.indexer.? else null;
            try matrixAny(binding, source, try named(&prefix, "layers.{d}.attn.indexer.wq_b", .{layer}), 1280, 4096, if (binding) &indexer.?.wq_b else null);
            try plainEither(binding, source, try named(&name, "layers.{d}.attn.indexer.weights_proj.weight", .{layer}), &.{ 32, 5120 }, if (binding) &indexer.?.weights_proj else null);
            if (binding) {
                indexer.?.wk = if (owner) @as(AnyMatrix, undefined) else null;
                indexer.?.k_norm = if (owner) @as(Tensor, undefined) else null;
            }
            if (owner) {
                try matrixAny(binding, source, try named(&prefix, "layers.{d}.attn.indexer.wk", .{layer}), 512, 128, if (binding) &indexer.?.wk.? else null);
                try plain(binding, source, try named(&name, "layers.{d}.attn.indexer.k_norm.weight", .{layer}), .bf16, &.{128}, if (binding) &indexer.?.k_norm.? else null);
            }
        }
        const moe: ?*Moe = if (binding) &out.?.moe else null;
        // The released checkpoints store the router gate f16; rank-split repacks store it bf16.
        if (binding) {
            const tensor = try source.get(try named(&name, "layers.{d}.ffn.gate.weight", .{layer}));
            if ((tensor.dtype != .f16 and tensor.dtype != .bf16) or tensor.rank != 2 or
                tensor.shape[0] != 384 or tensor.shape[1] != 5120)
            {
                std.log.err("unsupported router gate {s}: {s} {any}", .{ name, @tagName(tensor.dtype), tensor.shape[0..tensor.rank] });
                return error.UnsupportedTensor;
            }
            moe.?.gate = tensor;
        } else {
            const tensor = try source.get(try named(&name, "layers.{d}.ffn.gate.weight", .{layer}));
            if ((tensor.dtype != .f16 and tensor.dtype != .bf16) or tensor.rank != 2 or
                tensor.shape[0] != 384 or tensor.shape[1] != 5120) return error.UnsupportedTensor;
        }
        // The bias rides the same three dtypes the route kernel decodes (f32, f16, bf16).
        if (binding) {
            const tensor = try source.get(try named(&name, "layers.{d}.ffn.gate.bias", .{layer}));
            if ((tensor.dtype != .f16 and tensor.dtype != .bf16 and tensor.dtype != .f32) or tensor.rank != 1 or tensor.shape[0] != 384)
                return error.UnsupportedTensor;
            moe.?.bias = tensor;
        } else {
            const tensor = try source.get(try named(&name, "layers.{d}.ffn.gate.bias", .{layer}));
            if ((tensor.dtype != .f16 and tensor.dtype != .bf16 and tensor.dtype != .f32) or tensor.rank != 1 or tensor.shape[0] != 384)
                return error.UnsupportedTensor;
        }
        inline for (.{ "w1", "w3", "w2" }, 0..) |part, projection| {
            const k: usize = if (projection == 2) 2304 else 5120;
            const n: usize = if (projection == 2) 5120 else 2304;
            try matrix(binding, source, try named(&prefix, "layers.{d}.ffn.shared_experts.{s}", .{ layer, part }), k, n, if (binding) &moe.?.shared[projection] else null);
            if (!binding) for (0..384) |expert| try matrix(false, source, try named(&prefix, "layers.{d}.ffn.experts.{d}.{s}", .{ layer, expert, part }), k, n, null);
        }
        const engram = layer == 1 or layer == 14;
        if (binding) out.?.engram = if (engram) @as(Engram, undefined) else null;
        if (engram) {
            const e: ?*Engram = if (binding) &out.?.engram.? else null;
            const rows: usize = if (layer == 1) 384006168 else 384016682;
            if (streamed) {
                // load.Plan checked the streamed tables' headers; the tables are not checkpoint tensors.
                if (binding) {
                    e.?.table = null;
                    e.?.scales = null;
                }
            } else {
                var table: Tensor = undefined;
                var scales: Tensor = undefined;
                try plain(binding, source, try named(&name, "layers.{d}.engram.embed.weight", .{layer}), .f8_e4m3, &.{ rows, 256 }, if (binding) &table else null);
                try plain(binding, source, try named(&name, "layers.{d}.engram.embed.scale", .{layer}), .f8_e8m0, &.{ rows, 8 }, if (binding) &scales else null);
                if (binding) {
                    e.?.table = table;
                    e.?.scales = scales;
                }
            }
            try plainEither(binding, source, try named(&name, "layers.{d}.engram.q_weight", .{layer}), &.{ 4, 5120 }, if (binding) &e.?.q_weight else null);
            try plainEither(binding, source, try named(&name, "layers.{d}.engram.k_weight", .{layer}), &.{ 4, 5120 }, if (binding) &e.?.k_weight else null);
            try matrix(binding, source, try named(&prefix, "layers.{d}.engram.wkv", .{layer}), 6144, 25600, if (binding) &e.?.wkv else null);
        }
    }
}
