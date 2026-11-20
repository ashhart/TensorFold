//! GLM's DFlash2 drafter with --drafter-bits 4: each bf16 linear quantized at load, then read only through its copy.
const std = @import("std");
const mtl = @import("metal");
const core = @import("core");
const op = @import("../qwen27/dflash/operators.zig");
const ck = @import("../qwen27/checkpoint.zig");
const Tiled = @import("../qwen27/dflash/runtime_backend.zig").Tiled;
const simd = core.row_projection.simd;

/// The draft's linears: its rank-2 weights, which a 4-bit drafter reads only through their prepared copies.
pub fn linear(name: []const u8, rank: usize) bool {
    return rank == 2 and std.mem.endsWith(u8, name, ".weight");
}

/// Every linear's 4-bit copy, keyed by its bf16 view's address: lane adapters with tensor units, else SIMD tiles.
pub const Copies = struct {
    gpa: std.mem.Allocator,
    adapters: std.AutoHashMapUnmanaged(usize, core.affine4_lane.Adapter) = .empty,
    tiled: std.AutoHashMapUnmanaged(usize, Tiled) = .empty,
    simd: ?simd.Pipelines = null,
    opt: ?@import("metal_dflash_opt.zig").Opt = null,

    pub fn deinit(c: *Copies) void {
        if (c.opt) |*o| o.deinit();
        var adapters = c.adapters.valueIterator();
        while (adapters.next()) |a| a.deinit();
        c.adapters.deinit(c.gpa);
        var tiled = c.tiled.valueIterator();
        while (tiled.next()) |t| t.deinit();
        c.tiled.deinit(c.gpa);
        if (c.simd) |*p| p.deinit();
    }

    /// Device bytes the copies hold.
    pub fn bytes(c: *const Copies) usize {
        var n: usize = 0;
        if (c.opt) |*o| n += o.bytes();
        var adapters = c.adapters.valueIterator();
        while (adapters.next()) |a| n += a.words.length() + a.metadata.length();
        var tiled = c.tiled.valueIterator();
        while (tiled.next()) |t| n += t.w.length() + t.scales.length() + t.biases.length();
        return n;
    }

    pub fn appendBuffers(c: *const Copies, out: *std.ArrayList(mtl.Buffer)) !void {
        if (c.opt) |*o| try o.appendBuffers(out);
        var adapters = c.adapters.valueIterator();
        while (adapters.next()) |a| try out.appendSlice(c.gpa, &.{ a.words, a.metadata });
        var tiled = c.tiled.valueIterator();
        while (tiled.next()) |t| try out.appendSlice(c.gpa, &.{ t.w, t.scales, t.biases });
    }

    pub fn prepareOpt(c: *Copies, device: mtl.Device, checkpoint: *core.checkpoint_metal.Checkpoint, cfg: @import("../qwen27/dflash/config.zig").Config) !void {
        c.opt = try @import("metal_dflash_opt.zig").Opt.init(c.gpa, device, checkpoint, cfg);
    }

    /// The copies of `checkpoint`'s linears, `linear_count` of them, on `device`; any other count is refused.
    pub fn prepare(gpa: std.mem.Allocator, device: mtl.Device, checkpoint: *core.checkpoint_metal.Checkpoint, linear_count: usize) !Copies {
        var c: Copies = .{ .gpa = gpa };
        errdefer c.deinit();
        const units = device.tensorUnits();
        const order = if (units) try core.lane_projection.RegOrder.init(device) else null;
        defer if (order) |o| o.deinit();
        if (!units) c.simd = try simd.Pipelines.load(device);
        var rows: std.ArrayList(u16) = .empty;
        defer rows.deinit(gpa);
        var kept = checkpoint.tensors.iterator();
        while (kept.next()) |entry| if (linear(entry.key_ptr.*, entry.value_ptr.rank)) {
            const t = entry.value_ptr.*;
            try c.one(device, order, @intFromPtr(t.buffer.contents() + t.offset), t.dtype, t.shape, t.host(u16));
        };
        var later = checkpoint.deferred.iterator();
        while (later.next()) |entry| if (linear(entry.key_ptr.*, entry.value_ptr.rank)) {
            const d = entry.value_ptr.*;
            try rows.resize(gpa, d.bytes / 2);
            try checkpoint.read(entry.key_ptr.*, 0, std.mem.sliceAsBytes(rows.items));
            try c.one(device, order, @intFromPtr((try checkpoint.view(entry.key_ptr.*)).ptr), d.dtype, d.shape, rows.items);
        };
        if ((if (units) c.adapters.count() else c.tiled.count()) != linear_count) return error.DraftTensorCoverage;
        return c;
    }

    /// One bf16 [n][k] linear's 4-bit copy under `key`: a lane adapter with `order`, else the same bits on tiles.
    fn one(c: *Copies, device: mtl.Device, order: ?core.lane_projection.RegOrder, key: usize, dtype: core.checkpoint_metal.DType, shape: @FieldType(core.checkpoint_metal.Tensor, "shape"), rows: []const u16) !void {
        if (dtype != .bf16 or shape[0] % 32 != 0 or shape[1] % 64 != 0) return error.DraftPreparationUnqualified;
        var words = try core.affine4.prepare(c.gpa, rows, shape[0], shape[1]);
        defer words.deinit();
        if (order) |o| {
            const adapter = try core.affine4_lane.Adapter.init(c.gpa, device, words, try @import("../qwen27/projection.zig").split(shape[0], shape[1]), o);
            errdefer adapter.deinit();
            if (try c.adapters.fetchPut(c.gpa, key, adapter)) |old| old.value.deinit();
        } else {
            const t = try Tiled.init(device, words);
            errdefer t.deinit();
            if (try c.tiled.fetchPut(c.gpa, key, t)) |old| old.value.deinit();
        }
    }

    /// `x` through `w`'s copy into `y` (bf16 rows), then a barrier; a linear without one is refused.
    pub fn project(c: *const Copies, e: mtl.ComputeEncoder, w: ck.Tensor, x: op.View, input: core.draft_ops.Ref, y: op.View, output: core.draft_ops.Ref) !void {
        if (c.tiled.get(@intFromPtr(w.bytes.ptr))) |t| {
            if (x.stride != x.width or y.stride != y.width) return error.DraftPreparationUnqualified;
            simd.encode(e, try simd.call(&c.simd.?, t.n, t.k, x.rows, false, false, 0), t.matrix(), input.buf, input.off, output.buf, output.off);
        } else {
            const adapter = c.adapters.get(@intFromPtr(w.bytes.ptr)) orelse return error.DraftPreparationUnqualified;
            // the reg kernel takes up to 128 rows (a batch's blocks) in one weight read, a row's bits the same
            const most = adapter.projection.max_rows;
            var first: u32 = 0;
            while (first < x.rows) : (first += most) {
                try adapter.encode(e, .{ .buf = input.buf, .off = input.off + @as(usize, first) * x.stride * 2 }, .{ .buf = output.buf, .off = output.off + @as(usize, first) * y.stride * 2 }, @min(x.rows - first, most));
            }
        }
        e.barrier();
    }
};

test "only rank-2 weights are linears" {
    try std.testing.expect(linear("layers.0.self_attn.q_proj.weight", 2));
    try std.testing.expect(!linear("layers.0.input_layernorm.weight", 1));
    try std.testing.expect(!linear("layers.0.attention_conv.base_kernel", 3));
}
