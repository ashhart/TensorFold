//! GLM's linears tiled at load for glm_mm: bf16, or packed to 4, 6 or 8 bits in place (TF_GLM_DENSE_BITS, _HEAD_BITS).
const std = @import("std");
const cuda = @import("cuda");
const weights = @import("cuda_weights.zig");
const mm_mod = @import("cuda_mm.zig");
const qpack = @import("cuda_qpack.zig");
const Weights = weights.Weights;
const Layer = weights.Layer;

/// Every linear b16.matmul reads, as (pointer, n, k); the head last.
fn linears(w: *const Weights, list: *std.ArrayList([3]usize)) !void {
    const add = struct {
        fn f(l: *std.ArrayList([3]usize), gpa: std.mem.Allocator, ptr: u64, n: usize, k: usize) !void {
            if (ptr != 0) try l.append(gpa, .{ ptr, n, k });
        }
    }.f;
    const kd = @import("cuda_kda.zig").dk;
    var layers: std.ArrayList(Layer) = .empty;
    defer layers.deinit(w.gpa);
    try layers.appendSlice(w.gpa, w.layers);
    if (w.mtp) |m| {
        try layers.append(w.gpa, m.layer);
        try add(list, w.gpa, m.eh, w.config.hidden, 2 * w.config.hidden);
    }
    for (layers.items) |l| {
        switch (l.mixer) {
            .kda => |x| {
                const n = x.heads * kd;
                try add(list, w.gpa, x.proj, x.width(), x.dims);
                try add(list, w.gpa, x.fb, n, kd);
                try add(list, w.gpa, x.gb, n, kd);
                try add(list, w.gpa, x.o, x.dims, n);
            },
            .mla => |x| {
                try add(list, w.gpa, x.proj, x.q_lora + x.lw, x.dims);
                try add(list, w.gpa, x.q_b, x.heads * x.qk, x.q_lora);
                try add(list, w.gpa, x.index.kw, x.index.d + x.index.heads, x.dims);
                try add(list, w.gpa, x.index.qb, x.index.heads * x.index.d, x.q_lora);
                try add(list, w.gpa, x.o, x.dims, x.heads * x.dv);
            },
        }
        switch (l.ffn) {
            .dense => |x| switch (x.w) {
                .bf16 => |b| {
                    try add(list, w.gpa, b.gu, 2 * x.width, x.dims);
                    try add(list, w.gpa, b.down, x.dims, x.width);
                },
                .nvfp4 => {},
            },
            .moe => |x| {
                try add(list, w.gpa, x.shared_gu, 2 * x.shared, x.routed.dims);
                try add(list, w.gpa, x.shared_down, x.routed.dims, x.shared);
            },
        }
    }
    try add(list, w.gpa, w.head, w.share.vocab.len(), w.config.hidden);
}

/// Every linear retiled in place: dense at `dense_bits`, the head at `head_bits` (16: bf16); the packed ones' table.
pub fn tileLinears(gpa: std.mem.Allocator, w: *const Weights, mk: *const mm_mod.Kernels, s: cuda.Stream, d: *const cuda.Driver, dense_bits: u8, head_bits: u8) !mm_mod.Packed {
    var list: std.ArrayList([3]usize) = .empty;
    defer list.deinit(w.gpa);
    try linears(w, &list);
    var most: usize = 0;
    for (list.items) |e| most = @max(most, e[1] * e[2] * 2);
    var tmp = try cuda.DeviceBuffer.alloc(d, most);
    defer tmp.free();
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
    for (list.items, 0..) |e, i| {
        const n = e[1];
        const k = e[2];
        const bits = if (i + 1 == list.items.len) head_bits else dense_bits;
        if (bits == 16) {
            try mm_mod.tile(mk, s, e[0], tmp.ptr, n, k);
            try d.check(d.api.cuMemcpyDtoDAsync_v2(e[0], tmp.ptr, n * k * 2, s.handle), "cuMemcpyDtoDAsync");
            continue;
        }
        try host.resize(gpa, n * k);
        try s.synchronize();
        try d.check(d.api.cuMemcpyDtoH_v2(@ptrCast(host.items.ptr), e[0], n * k * 2), "cuMemcpyDtoH");
        try words.resize(gpa, n * k * bits / 32);
        try scales.resize(gpa, n * (k / 64));
        qpack.quantize(bits, host.items, n, k, words.items, scales.items);
        const sizes = mm_mod.packedBytes(n, k, bits); // codes then scales, inside the bf16 tiles' bytes
        try tmp.upload(0, std.mem.sliceAsBytes(words.items));
        try tmp.upload(sizes[0], std.mem.sliceAsBytes(scales.items));
        try mm_mod.tilePacked(mk, s, bits, tmp.ptr, tmp.ptr + sizes[0], e[0], e[0] + sizes[0], n, k);
        try s.synchronize(); // tmp is the next linear's
        try packed_ptrs.append(gpa, e[0]);
        try packed_bits.append(gpa, bits);
    }
    try s.synchronize();
    // sorted by pointer, the bits beside them (Packed.tiled searches)
    const ptrs = try packed_ptrs.toOwnedSlice(gpa);
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
