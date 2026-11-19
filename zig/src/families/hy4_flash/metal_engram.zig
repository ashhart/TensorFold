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

/// Where the n-gram tables live (TF_HY4_ENGRAM): nvme reads the checkpoint's mapping (pages on demand, read
/// ahead a step at a time); ram copies the tables into resident memory for the host's gather; gpu keeps them in GPU
/// memory and gathers there, the host only hashing the windows. All three stage the same bits.
pub const Placement = enum { nvme, ram, gpu };

pub fn placement() Placement {
    const v = std.mem.span(std.c.getenv("TF_HY4_ENGRAM") orelse return .nvme);
    return std.meta.stringToEnum(Placement, v) orelse .nvme;
}

pub const Layer = struct {
    own: common.Buffers,
    table: row_table.Table,
    /// The tables in memory of ours (ram or gpu placement).
    resident: ?struct { words: mtl.Buffer, scales: mtl.Buffer, biases: mtl.Buffer, gpu: bool } = null,
    lookup_gamma: mtl.Buffer,
    key: [ihc.streams]common.Linear,
    key_bias: mtl.Buffer,
    key_gain: mtl.Buffer,
    query_gain: mtl.Buffer,
    delta_gain: mtl.Buffer,
    value: common.Linear,
    value_bias: mtl.Buffer,
    value_gain: mtl.Buffer,
    fp8: bool,
    hidden: u32,

    pub fn upload(a: std.mem.Allocator, d: mtl.Device, w: *const weights.Engram, hidden: u32, fp8: bool) !Layer {
        return uploadForModel(a, d, w, hidden, fp8, false);
    }

    pub fn uploadForModel(a: std.mem.Allocator, d: mtl.Device, w: *const weights.Engram, hidden: u32, fp8: bool, use_tensors: bool) !Layer {
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
            .value = try b.linear(w.value, use_tensors),
            .value_bias = try b.upload(w.value_bias.bytes),
            .value_gain = try b.upload(w.pre_gamma.bytes),
            .fp8 = fp8,
            .hidden = hidden,
        };
        switch (placement()) {
            .nvme => {},
            .ram, .gpu => |where| {
                const t = &out.table.affine;
                const words = try b.upload(t.words.bytes);
                const scales = try b.upload(t.scales.bytes);
                const biases = try b.upload(t.biases.bytes);
                // the host's gather reads the copies from here on
                t.words.bytes = words.slice(u8, t.words.bytes.len);
                t.scales.bytes = scales.slice(u8, t.scales.bytes.len);
                t.biases.bytes = biases.slice(u8, t.biases.bytes.len);
                out.resident = .{ .words = words, .scales = scales, .biases = biases, .gpu = where == .gpu };
            },
        }
        for (&out.key, w.key) |*dst, src| dst.* = try b.linear(src.key, use_tensors);
        inline for (.{ "key_bias", "key_gain", "query_gain", "delta_gain" }, .{ "bias", "key_norm", "query_norm", "residual_scale" }) |dst, field| {
            var tensors: [ihc.streams]core.checkpoint_host.Tensor = undefined;
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

fn concatenate(b: *common.Buffers, weights_: []const core.checkpoint_host.Tensor) !mtl.Buffer {
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
    /// Table rows a token and column, for the GPU gather.
    table_rows: Ref,

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
            .table_rows = .{ .buffer = try b.alloc(@as(usize, capacity) * engram.columns * @sizeOf(u32)) },
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
        if (layer.resident) |r| if (r.gpu) {
            const out = self.table_rows.buffer.slice(u32, windows.len * engram.columns);
            for (windows, 0..) |window, t| out[t * engram.columns ..][0..engram.columns].* = layer.table.windowRows(window);
            @memcpy(self.media_mask.buffer.slice(u32, media.len), media);
            return;
        };
        try layer.table.gather(a, windows, self.raw.buffer.slice(u16, cells));
        @memcpy(self.media_mask.buffer.slice(u32, media.len), media);
    }
};

pub const Kernels = struct {
    lookup: mtl.Pipeline,
    apply: mtl.Pipeline,
    gather: mtl.Pipeline,

    pub fn load(d: mtl.Device) !Kernels {
        const library = try mtl.Library.fromSource(d, sources.hy4_flash_engram, mtl.CompileOptions.mlx());
        defer library.deinit();
        var lookup = try mtl.Pipeline.init(d, library, "tf_hy4_engram_lookup", false);
        errdefer lookup.deinit();
        var apply = try mtl.Pipeline.init(d, library, "tf_hy4_engram_apply", false);
        errdefer apply.deinit();
        return .{ .lookup = lookup, .apply = apply, .gather = try mtl.Pipeline.init(d, library, "tf_hy4_engram_gather", false) };
    }

    pub fn deinit(self: *Kernels) void {
        self.lookup.deinit();
        self.apply.deinit();
        self.gather.deinit();
    }

    pub fn forward(self: Kernels, ops: *const common.Kernels, e: mtl.ComputeEncoder, w: *const Layer, s: *const Scratch, x: Ref, output: Ref, rows: u32) !void {
        if (rows == 0 or rows > s.capacity) return error.BadHy4EngramRows;
        if (w.resident) |r| if (r.gpu) {
            const cells: u32 = rows * engram.columns * width;
            e.setPipeline(self.gather);
            e.setBuffer(r.words, 0, 0);
            e.setBuffer(r.scales, 0, 1);
            e.setBuffer(r.biases, 0, 2);
            e.setBuffer(s.table_rows.buffer, 0, 3);
            e.setBuffer(s.raw.buffer, 0, 4);
            e.setValue([2]u32{ w.table.affine.bits, cells }, 5);
            e.dispatchThreads(mtl.Size.of(cells, 1, 1), mtl.Size.of(256, 1, 1));
            e.barrier();
        };
        e.setPipeline(self.lookup);
        e.setBuffer(s.raw.buffer, 0, 0);
        e.setBuffer(w.lookup_gamma, 0, 1);
        e.setBuffer(s.normalized.buffer, 0, 2);
        e.setValue(@as(u32, @intFromBool(w.fp8)), 3);
        e.dispatchGroups(mtl.Size.of(engram.columns, rows, 1), mtl.Size.of(32, 1, 1));
        e.barrier();
        var outs: [1 + ihc.streams]common.Kernels.Out = undefined;
        outs[0] = .{ .w = w.value, .y = s.value };
        for (w.key, 0..) |projection, i| {
            outs[1 + i] = .{ .w = projection, .y = .{ .buffer = s.keys.buffer, .offset = i * rows * w.hidden * @sizeOf(u16) } };
        }
        try ops.linearsFrom(e, s.normalized, &outs, rows);
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
