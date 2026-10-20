//! GLM's linears tiled at load for glm_mm: bf16, or packed to 4, 6 or 8 bits in place (TF_GLM_DENSE_BITS, _HEAD_BITS).
const std = @import("std");
const cuda = @import("cuda");
const weights = @import("cuda_weights.zig");
const mm_mod = @import("cuda_mm.zig");
const qpack = @import("cuda_qpack.zig");
const projection = @import("cuda_proj.zig");
const Weights = weights.Weights;
const Layer = weights.Layer;

/// A linear b16.matmul reads: the field holding its tiles' pointer, and its shape [n][k].
const Lin = struct { field: *u64, n: usize, k: usize };

fn add(l: *std.ArrayList(Lin), gpa: std.mem.Allocator, field: *u64, n: usize, k: usize) !void {
    if (field.* != 0) try l.append(gpa, .{ .field = field, .n = n, .k = k });
}

/// A projection as stored: its bf16 rows like any linear's; FP8 kinds stay the lane's tiles, never read as bf16 here.
fn addProj(l: *std.ArrayList(Lin), gpa: std.mem.Allocator, p: *projection.Proj, n: usize, k: usize) !void {
    switch (p.*) {
        .bf16 => |*w| try add(l, gpa, w, n, k),
        .fp8g, .mxfp8 => {},
    }
}

fn layerLinears(gpa: std.mem.Allocator, l: *Layer, list: *std.ArrayList(Lin)) !void {
    const kd = @import("cuda_kda.zig").dk;
    switch (l.mixer) {
        .kda => |*x| {
            const n = x.heads * kd;
            if (x.tiledInput()) {
                switch (x.proj) { // a split stack's MXFP8 [q | k | v] is the lane's; its bf16 [f_a | g_a | b] tiles
                    .bf16 => |*p| try add(list, gpa, p, x.width(), x.dims),
                    .split => |*sp| try add(list, gpa, &sp.rest, x.width() - 3 * n, x.dims),
                }
                try add(list, gpa, &x.fb, n, kd);
                try add(list, gpa, &x.gb, n, kd);
            }
            try addProj(list, gpa, &x.o, x.dims, n);
        },
        .mla => |*x| {
            try addProj(list, gpa, &x.proj, x.q_lora + x.lw, x.dims);
            try addProj(list, gpa, &x.q_b, x.heads * x.qk, x.q_lora);
            try addProj(list, gpa, &x.index.kw, x.index.d + x.index.heads, x.dims);
            try addProj(list, gpa, &x.index.qb, x.index.heads * x.index.d, x.q_lora);
            try addProj(list, gpa, &x.o, x.dims, x.heads * x.dv);
        },
    }
    switch (l.ffn) {
        .dense => |*x| switch (x.w) {
            .bf16 => |*b| {
                try add(list, gpa, &b.gu, 2 * x.width, x.dims);
                try add(list, gpa, &b.down, x.dims, x.width);
            },
            .nvfp4, .mxfp8 => {},
        },
        .moe => |*x| {
            try addProj(list, gpa, &x.shared_gu, 2 * x.shared, x.routed.dims);
            try addProj(list, gpa, &x.shared_down, x.routed.dims, x.shared);
        },
    }
}

/// Every linear b16.matmul reads, as the fields holding them; the head last, when it is bf16.
fn linears(w: *Weights, list: *std.ArrayList(Lin)) !void {
    for (w.layers) |*l| try layerLinears(w.gpa, l, list);
    if (w.mtp) |*m| {
        try layerLinears(w.gpa, &m.layer, list);
        try add(list, w.gpa, &m.eh, w.config.hidden, 2 * w.config.hidden);
    }
    try addProj(list, w.gpa, &w.head, w.share.vocab.len(), w.config.hidden);
}

/// A packed linear moved into a buffer its own size, its bf16 one freed; the bytes freed (0: it shares a buffer).
fn compact(w: *Weights, d: *const cuda.Driver, s: cuda.Stream, field: *u64, bytes: usize) !usize {
    const old = field.*;
    const at = for (w.buffers.items, 0..) |b, i| {
        if (b.ptr == old) break i;
    } else return 0;
    const was = w.buffers.items[at].len;
    var nb = try cuda.DeviceBuffer.alloc(d, bytes);
    errdefer nb.free(); // the weights own it only once it replaces the old buffer
    try d.check(d.api.cuMemcpyDtoDAsync_v2(nb.ptr, old, bytes, s.handle), "cuMemcpyDtoDAsync");
    try s.synchronize();
    w.buffers.items[at].free();
    w.buffers.items[at] = nb;
    for (w.named.items) |*n| if (n.ptr == old) {
        n.ptr = nb.ptr;
        n.len = nb.len;
    };
    field.* = nb.ptr;
    return was -| nb.len;
}

/// Eligible linears tiled in place; uneven KDA inputs and their views stay bf16 for the row-major kernels.
pub fn tileLinears(gpa: std.mem.Allocator, w: *Weights, mk: *const mm_mod.Kernels, s: cuda.Stream, d: *const cuda.Driver, dense_bits: u8, head_bits: u8) !mm_mod.Packed {
    var list: std.ArrayList(Lin) = .empty;
    defer list.deinit(w.gpa);
    try linears(w, &list);
    var most: usize = 0;
    for (list.items) |e| most = @max(most, e.n * e.k * 2);
    var tmp = try cuda.DeviceBuffer.alloc(d, most);
    defer tmp.free();
    errdefer @import("cuda_load_owner.zig").drain(s, d);
    var host: std.ArrayList(u16) = .empty; // a packed linear's bf16 rows, then its codes and scales, on the host
    defer host.deinit(gpa);
    var words: std.ArrayList(u32) = .empty;
    defer words.deinit(gpa);
    var scales: std.ArrayList(u16) = .empty;
    defer scales.deinit(gpa);
    var packed_ptrs: std.ArrayList(u64) = .empty;
    errdefer packed_ptrs.deinit(gpa);
    var packed_bits: std.ArrayList(u8) = .empty;
    errdefer packed_bits.deinit(gpa);
    var freed: usize = 0;
    for (list.items, 0..) |e, i| {
        const n = e.n;
        const k = e.k;
        const ptr = e.field.*;
        const bits = if (i + 1 == list.items.len and w.head == .bf16) head_bits else dense_bits; // an MXFP8 head stays
        if (bits == 16) {
            try mm_mod.tile(mk, s, ptr, tmp.ptr, n, k);
            try d.check(d.api.cuMemcpyDtoDAsync_v2(ptr, tmp.ptr, n * k * 2, s.handle), "cuMemcpyDtoDAsync");
            continue;
        }
        try host.resize(gpa, n * k);
        try s.synchronize();
        try d.check(d.api.cuMemcpyDtoH_v2(@ptrCast(host.items.ptr), ptr, n * k * 2), "cuMemcpyDtoH");
        try words.resize(gpa, n * k * bits / 32);
        try scales.resize(gpa, n * (k / 64));
        qpack.quantize(bits, host.items, n, k, words.items, scales.items);
        const sizes = mm_mod.packedBytes(n, k, bits); // codes then scales, inside the bf16 tiles' bytes
        try tmp.upload(0, std.mem.sliceAsBytes(words.items));
        try tmp.upload(sizes[0], std.mem.sliceAsBytes(scales.items));
        try mm_mod.tilePacked(mk, s, bits, tmp.ptr, tmp.ptr + sizes[0], ptr, ptr + sizes[0], n, k);
        try s.synchronize(); // tmp is the next linear's
        freed += try compact(w, d, s, e.field, sizes[0] + sizes[1]); // its bf16 bytes back to the device
        try packed_ptrs.append(gpa, e.field.*);
        try packed_bits.append(gpa, bits);
    }
    try s.synchronize();
    if (packed_ptrs.items.len > 0) std.log.info("glm: {d} linears packed ({d}/{d} bits), {d:.2} GiB freed", .{ packed_ptrs.items.len, dense_bits, head_bits, @as(f64, @floatFromInt(freed)) / (1 << 30) });
    // sorted by pointer, the bits beside them (Packed.tiled searches)
    const ptrs = try packed_ptrs.toOwnedSlice(gpa);
    errdefer gpa.free(ptrs);
    const bits_out = try packed_bits.toOwnedSlice(gpa);
    const Ctx = struct {
        p: []u64,
        b: []u8,
        pub fn lessThan(c: @This(), a: usize, b: usize) bool {
            return c.p[a] < c.p[b];
        }
        pub fn swap(c: @This(), a: usize, b: usize) void {
            std.mem.swap(u64, &c.p[a], &c.p[b]);
            std.mem.swap(u8, &c.b[a], &c.b[b]);
        }
    };
    std.sort.pdqContext(0, ptrs.len, Ctx{ .p = ptrs, .b = bits_out });
    return .{ .ptrs = ptrs, .bits = bits_out };
}

test "GLM TP3: the tiler keeps unsupported KDA input and view weights in row-major order" {
    const c = @import("config.zig").Config{};
    const Share = @import("cuda_shard.zig").Share;
    const a = std.testing.allocator;
    for ([_]usize{ 1, 2, 3, 4, 5, 8 }) |world| for (0..world) |rank| {
        const share = try Share.of(&c, rank, world);
        for ([_]bool{ false, true }) |split| { // a split stack's bf16 rest tiles or stays row-major with the stack
            const in: @import("cuda_kda_block.zig").In = if (split) .{ .split = .{ .qkv = undefined, .rest = 1 } } else .{ .bf16 = 1 };
            var l: Layer = undefined;
            l.mixer = .{ .kda = .{ .proj = in, .fb = 2, .gb = 3, .o = .{ .bf16 = 4 }, .conv_w = 0, .a_log = 0, .dt_bias = 0, .norm_w = 0, .eps = 1e-6, .lower = -5, .heads = share.kda.len(), .dims = c.hidden } };
            l.ffn = .{ .dense = .{ .w = .{ .nvfp4 = undefined }, .width = share.dense.len(), .dims = c.hidden, .limit = 0 } };
            var list: std.ArrayList(Lin) = .empty;
            defer list.deinit(a);
            try layerLinears(a, &l, &list);
            const aligned = share.kda.len() % 8 == 0;
            try std.testing.expectEqual(@as(usize, if (aligned) 4 else 1), list.items.len);
            try std.testing.expectEqual(&l.mixer.kda.o.bf16, list.items[list.items.len - 1].field);
            if (aligned) {
                for (list.items, [_]u64{ 1, 2, 3, 4 }) |linear, ptr| try std.testing.expectEqual(ptr, linear.field.*);
            }
        }
    };
}
