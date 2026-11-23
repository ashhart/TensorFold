//! A Hy4-flash pass's per-row scratch, sized once at the widest layer.
const std = @import("std");
const mtl = @import("metal");
const affine = @import("../../core/row_projection.zig");
const cfg = @import("config.zig");
const ihc = @import("ihc.zig");
const ihc_gpu = @import("ihc_metal.zig");
const mla_gpu = @import("mla_metal.zig");
const mha = @import("mla_attention.zig");
const shape_ = @import("mla_weights.zig");
const gpu_qkv = @import("metal_tensor_qkv.zig");
const lane = @import("../../core/lane_projection.zig");
const dense = @import("metal_dense.zig");
const Buffers = dense.Buffers;
const Ref = dense.Ref;
const Attention = dense.Attention;
const withAllocation = dense.withAllocation;

pub const Scratch = struct {
    own: Buffers,
    capacity: mha.Shape,
    shape: mha.Shape,
    inner_capacity: u32,
    merged: Ref,
    ihc_partials: Ref,
    /// A count and a generation a row for the fused seams' in-kernel barrier (zeroed; each launch leaves count 0).
    seam_barriers: Ref,
    qkv: ?Ref = null,
    post: Ref,
    normalized: Ref,
    query_a: Ref,
    query_norm: Ref,
    query: Ref,
    kv: Ref,
    value_logits: Ref,
    values: Ref,
    absorbed: Ref,
    rotary: Ref,
    latent: Ref,
    attention: Ref,
    key_scores: Ref,
    flash_parts: ?Ref = null,
    branch: Ref,
    gate: Ref,
    up: Ref,
    inner: Ref,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, layer: *const Attention, inner: u32, keys: u32, rows: u32) !Scratch {
        return initForModel(a, device, layer, inner, keys, rows, false);
    }

    pub fn initForModel(a: std.mem.Allocator, device: mtl.Device, layer: *const Attention, inner: u32, keys: u32, rows: u32, use_tensors: bool) !Scratch {
        var b: Buffers = .{ .a = a, .d = device };
        errdefer b.clear();
        const r = rows;
        const h = layer.hidden;
        const heads = layer.heads;
        const shape = try mha.Shape.init(r, heads, keys);
        const out: Scratch = .{
            .own = undefined,
            .capacity = shape,
            .shape = shape,
            .inner_capacity = inner,
            .merged = try b.rows(r, h),
            .ihc_partials = .{ .buffer = try b.alloc(@as(usize, r) * ihc_gpu.partial_floats * @sizeOf(f32)) },
            .seam_barriers = blk: {
                const bars = try b.alloc(ihc_gpu.Kernels.seam_rows * 8);
                @memset(bars.slice(u8, ihc_gpu.Kernels.seam_rows * 8), 0);
                break :blk .{ .buffer = bars };
            },
            .qkv = if (use_tensors) try b.rows(r, layer.query_rank + shape_.absorbed_stride + heads * shape_.value_width) else null,
            .post = .{ .buffer = try b.alloc(@as(usize, r) * ihc.streams * @sizeOf(f32)) },
            .normalized = try b.rows(r, h),
            .query_a = try b.rows(r, layer.query_rank),
            .query_norm = try b.rows(r, layer.query_rank),
            .query = try b.rows(r, heads * shape_.query_stride),
            .kv = try b.rows(r, shape_.absorbed_stride),
            .value_logits = try b.rows(r, heads * shape_.value_width),
            .values = try b.rows(r, heads * shape_.value_width),
            .absorbed = try b.rows(r, heads * shape_.absorbed_stride),
            .rotary = try b.rows(r, shape_.rope),
            .latent = try b.rows(r, heads * shape_.latent),
            .attention = .{ .buffer = try b.alloc(shape.scratchBytes()) },
            .key_scores = .{ .buffer = try b.alloc(@as(usize, r) * heads * keys * @sizeOf(f32)) },
            .flash_parts = if (use_tensors) blk: {
                const parts = try b.alloc(mla_gpu.Nax.partBytes(heads));
                @memset(parts.slice(u8, mla_gpu.Nax.partBytes(heads)), 0);
                break :blk .{ .buffer = parts };
            } else null,
            .branch = try b.rows(r, h),
            .gate = try b.rows(r, inner),
            .up = try b.rows(r, inner),
            .inner = try b.rows(r, inner),
        };
        return withAllocation(out, b);
    }

    /// Keep one allocation at the widest layer and bind the current lane batch and head count before encoding.
    pub fn bind(self: *Scratch, rows: u32, heads: u32, keys: u32) !void {
        if (rows > self.capacity.rows or heads > self.capacity.heads or keys > self.capacity.key_stride)
            return error.ShortHy4Scratch;
        self.shape = try mha.Shape.init(rows, heads, keys);
    }

    pub fn deinit(self: *Scratch) void {
        self.own.clear();
        self.* = undefined;
    }
};
