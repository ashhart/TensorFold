//! Lossless device packing and row-stable CUDA projections for the two published affine widths.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");

pub const max_rows = 16;

pub const Owned = struct {
    a: std.mem.Allocator,
    driver: *const cuda.Driver,
    buffers: std.ArrayList(cuda.DeviceBuffer) = .empty,

    pub fn alloc(self: *Owned, bytes: usize) !u64 {
        var b = try cuda.DeviceBuffer.alloc(self.driver, bytes);
        errdefer b.free();
        try self.buffers.append(self.a, b);
        return b.ptr;
    }

    pub fn upload(self: *Owned, bytes: []const u8) !u64 {
        var b = try cuda.DeviceBuffer.alloc(self.driver, bytes.len);
        errdefer b.free();
        try b.upload(0, bytes);
        try self.buffers.append(self.a, b);
        return b.ptr;
    }

    fn rollback(self: *Owned, start: usize) void {
        for (self.buffers.items[start..]) |*b| b.free();
        self.buffers.shrinkRetainingCapacity(start);
    }

    pub fn deinit(self: *Owned) void {
        self.rollback(0);
        self.buffers.deinit(self.a);
    }
};

pub const Weight = union(enum) {
    q4: cuda.qlinear.Affine4,
    q8: cuda.qlinear.AffineN,

    pub fn inputs(self: Weight) usize {
        return switch (self) {
            inline else => |w| w.k,
        };
    }
};

pub const Kernels = struct {
    modules: [6]cuda.Module,
    q4: cuda.qlinear.Affine4Kernels,
    q8: cuda.qlinear.AffineNKernels,
    act: cuda.act_rows.ActRows,

    pub fn load(ctx: *const cuda.Context) !Kernels {
        const k = cuda.kernels;
        if (!k.available) return error.BuiltWithoutKernels;
        var out: Kernels = undefined;
        var count: usize = 0;
        errdefer for (out.modules[0..count]) |*m| m.unload();
        for (&out.modules, [_][]const u8{ k.qmm_group, k.qmm_prefill, k.lane_gemv, k.affine4_pack, k.affine8, k.affine_prefill }) |*slot, image| {
            slot.* = try cuda.Module.load(ctx.d, image);
            count += 1;
        }
        const sms = @as(usize, @intCast(try ctx.attribute(.multiprocessor_count)));
        const gb10 = try ctx.capability() == 121;
        const m = out.modules;
        out.q4 = try cuda.qlinear.Affine4Kernels.resolve(ctx.d, m[0], m[1], m[2], m[3], sms, gb10);
        errdefer out.q4.deinit();
        out.q8 = try cuda.qlinear.AffineNKernels.resolve(ctx.d, m[4], m[5], m[3], sms, gb10);
        errdefer out.q8.deinit();
        out.act = try cuda.act_rows.ActRows.load(ctx.d);
        return out;
    }

    pub fn deinit(self: *Kernels) void {
        self.act.unload();
        self.q8.deinit();
        self.q4.deinit();
        for (&self.modules) |*m| m.unload();
    }

    /// Adapt the storage layout without requantizing the checkpoint.
    pub fn pack(self: *const Kernels, owner: *Owned, s: cuda.Stream, host: core.quant.mlx.Host) !Weight {
        if (host.group != 64 or (host.bits != 4 and host.bits != 8)) return error.BadHy4ProjectionWidth;
        const dims = try core.quant.mlx.geometry.shape(host.words, host.scales, host.biases, host.bits, host.group, false);
        const n = host.words.shape[0];
        const k = host.scales.shape[1] * host.group;
        if (dims.elements == 0 or n == 0 or k == 0) return error.BadHy4ProjectionWidth;
        const start = owner.buffers.items.len;
        errdefer owner.rollback(start);
        var raw: Owned = .{ .a = owner.a, .driver = owner.driver };
        defer raw.deinit();
        const codes = try raw.upload(host.words.bytes);
        const scales = try raw.upload(host.scales.bytes);
        const zeros = try raw.upload(host.biases.bytes);
        const size = cuda.qlinear.AffineN.layout(n, k, host.bits);
        const w = try owner.alloc(size.words);
        const scale = try owner.alloc(size.scales);
        const bias = try owner.alloc(size.scales);
        errdefer s.synchronize() catch {};
        const packed_: Weight = switch (host.bits) {
            4 => blk: {
                const q: cuda.qlinear.Affine4 = .{ .w = w, .s = scale, .b = bias, .n = n, .k = k, .npad = size.npad };
                try self.q4.pack(s, codes, scales, zeros, q);
                break :blk .{ .q4 = q };
            },
            8 => blk: {
                try self.q8.packWords(s, codes, 8, n, k, w);
                try self.q8.packTable(s, scales, n, k, size.npad, scale);
                try self.q8.packTable(s, zeros, n, k, size.npad, bias);
                break :blk .{ .q8 = .{ .w = w, .s = scale, .b = bias, .n = n, .k = k, .npad = size.npad, .bits = 8 } };
            },
            else => unreachable,
        };
        try s.synchronize();
        return packed_;
    }

    pub fn run(self: *const Kernels, s: cuda.Stream, w: Weight, input: u64, sums: u64, output: u64, rows: usize) !void {
        if (rows == 0 or rows > max_rows or sums == 0) return error.BadHy4ProjectionRows;
        const width = w.inputs();
        try self.act.groupSums(s, input, width, sums, rows, width);
        switch (w) {
            .q4 => |q| try self.q4.decode(s, input, sums, q, output, rows),
            .q8 => |q| try self.q8.decode(s, input, sums, q, output, rows),
        }
    }
};
