//! Metal Engram weights and apply pass. The large lookup table stays in the checkpoint mapping.
const std = @import("std");
const core = @import("core");
const mtl = @import("metal");
const affine = @import("row_projection");
const sources = @import("kernel_sources");
const common = @import("metal_dense.zig");
const engram = @import("engram.zig");
const row_table = @import("engram_rows.zig");
const weights = @import("model_weights.zig");
const ihc = @import("ihc.zig");

const Ref = common.Ref;
const width = 96;

pub const Layer = struct {
    own: common.Buffers,
    table: row_table.Table,
    lookup_gamma: mtl.Buffer,
    key: [ihc.streams]affine.Weights,
    key_bias: mtl.Buffer,
    key_gain: mtl.Buffer,
    query_gain: mtl.Buffer,
    delta_gain: mtl.Buffer,
    value: affine.Weights,
    value_bias: mtl.Buffer,
    value_gain: mtl.Buffer,
    fp8: bool,
    hidden: u32,

    pub fn upload(a: std.mem.Allocator, d: mtl.Device, w: *const weights.Engram, hidden: u32, fp8: bool) !Layer {
        if (w.table.width != width or hidden == 0) return error.BadHy4EngramShape;
        var b: common.Buffers = .{ .a = a, .d = d };
        errdefer b.clear();
        var out: Layer = .{
            .own = undefined,
            .table = w.table,
            .lookup_gamma = try concatenate(&b, &w.lookup_norm),
            .key = undefined,
            .key_bias = undefined,
            .key_gain = undefined,
            .query_gain = undefined,
            .delta_gain = undefined,
            .value = try b.matrix(w.value),
            .value_bias = try b.upload(w.value_bias.bytes),
            .value_gain = try b.upload(w.pre_gamma.bytes),
            .fp8 = fp8,
            .hidden = hidden,
        };
        for (&out.key, w.key) |*dst, src| dst.* = try b.matrix(src.key);
        inline for (.{ "key_bias", "key_gain", "query_gain", "delta_gain" }, .{ "bias", "key_norm", "query_norm", "residual_scale" }) |dst, field| {
            var tensors: [ihc.streams]core.checkpoint.Tensor = undefined;
            for (&tensors, w.key) |*tensor, src| tensor.* = @field(src, field);
            @field(out, dst) = try concatenate(&b, &tensors);
        }
        out.own = b;
        return out;
    }

    pub fn deinit(self: *Layer) void {
        self.own.clear();
        self.* = undefined;
    }
};

fn concatenate(b: *common.Buffers, weights_: []const core.checkpoint.Tensor) !mtl.Buffer {
    const size = weights_[0].bytes.len;
    const result = try b.alloc(size * weights_.len);
    const bytes = result.slice(u8, size * weights_.len);
    for (weights_, 0..) |tensor, i| @memcpy(bytes[i * size ..][0..size], tensor.bytes);
    return result;
}

pub const Scratch = struct {
    own: common.Buffers,
    capacity: u32,
    raw: Ref,
    normalized: Ref,
    keys: Ref,
    value: Ref,
    media_mask: Ref,

    pub fn init(a: std.mem.Allocator, d: mtl.Device, capacity: u32, hidden: u32) !Scratch {
        if (capacity == 0 or hidden == 0) return error.BadHy4EngramShape;
        var b: common.Buffers = .{ .a = a, .d = d };
        errdefer b.clear();
        var out: Scratch = .{
            .own = undefined,
            .capacity = capacity,
            .raw = try b.rows(capacity, engram.columns * width),
            .normalized = try b.rows(capacity, engram.columns * width),
            .keys = try b.rows(capacity, ihc.streams * hidden),
            .value = try b.rows(capacity, hidden),
            .media_mask = .{ .buffer = try b.alloc(@as(usize, capacity) * @sizeOf(u32)) },
        };
        out.own = b;
        return out;
    }

    pub fn deinit(self: *Scratch) void {
        self.own.clear();
        self.* = undefined;
    }

    /// Windows come from committed or speculative lane token history. A media row leaves its hidden streams untouched.
    pub fn stage(self: *const Scratch, a: std.mem.Allocator, layer: *const Layer, windows: []const [engram.order]u32, media: []const u32) !void {
        if (windows.len == 0 or windows.len > self.capacity or windows.len != media.len) return error.BadHy4EngramRows;
        const cells = windows.len * engram.columns * width;
        try layer.table.gather(a, windows, self.raw.buffer.slice(u16, cells));
        @memcpy(self.media_mask.buffer.slice(u32, media.len), media);
    }
};

pub const Kernels = struct {
    lookup: mtl.Pipeline,
    apply: mtl.Pipeline,

    pub fn load(d: mtl.Device) !Kernels {
        const library = try mtl.Library.fromSource(d, sources.hy4_flash_engram, mtl.CompileOptions.mlx());
        defer library.deinit();
        var lookup = try mtl.Pipeline.init(d, library, "tf_hy4_engram_lookup", false);
        errdefer lookup.deinit();
        return .{ .lookup = lookup, .apply = try mtl.Pipeline.init(d, library, "tf_hy4_engram_apply", false) };
    }

    pub fn deinit(self: *Kernels) void {
        self.lookup.deinit();
        self.apply.deinit();
    }

    pub fn forward(self: Kernels, ops: *const common.Kernels, e: mtl.ComputeEncoder, w: *const Layer, s: *const Scratch, x: Ref, output: Ref, rows: u32) !void {
        if (rows == 0 or rows > s.capacity) return error.BadHy4EngramRows;
        e.setPipeline(self.lookup);
        e.setBuffer(s.raw.buffer, 0, 0);
        e.setBuffer(w.lookup_gamma, 0, 1);
        e.setBuffer(s.normalized.buffer, 0, 2);
        e.setValue(@as(u32, @intFromBool(w.fp8)), 3);
        e.dispatchGroups(mtl.Size.of(engram.columns, rows, 1), mtl.Size.of(32, 1, 1));
        e.barrier();
        try ops.project(e, w.value, s.normalized, s.value, rows);
        for (w.key, 0..) |projection, i| {
            const dst: Ref = .{ .buffer = s.keys.buffer, .offset = i * rows * w.hidden * @sizeOf(u16) };
            try ops.project(e, projection, s.normalized, dst, rows);
        }
        e.barrier();
        e.setPipeline(self.apply);
        const buffers = [_]Ref{
            x,                           s.keys,  .{ .buffer = w.key_bias },   .{ .buffer = w.key_gain },   .{ .buffer = w.query_gain },
            .{ .buffer = w.delta_gain }, s.value, .{ .buffer = w.value_bias }, .{ .buffer = w.value_gain }, s.media_mask,
            output,
        };
        for (buffers, 0..) |b, i| e.setBuffer(b.buffer, b.offset, i);
        e.setValue([2]u32{ rows, w.hidden }, buffers.len);
        e.dispatchGroups(mtl.Size.of(ihc.streams, rows, 1), mtl.Size.of(128, 1, 1));
    }
};
