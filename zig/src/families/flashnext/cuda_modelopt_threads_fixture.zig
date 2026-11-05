//! The Flash Next expert loader tests' fake driver, checkpoint fixture and the one-expert loop's words.
const std = @import("std");
const cuda = @import("cuda");
const ex = cuda.experts.Nvfp4Experts;

pub const abi = cuda.abi;
pub const wt = @import("cuda_prompt_wt.zig");
pub const experts = @import("cuda_modelopt_experts.zig");
pub const io = std.testing.io;
pub const ta = std.testing.allocator;
pub const env = @import("sys").env;

/// A driver whose device memory is host memory, counting calls and live allocations; cuda.memory's limit refuses.
pub const Fake = struct {
    pub const Kind = enum { pinned, device, copy };
    pub var calls: [3]usize = @splat(0);
    pub var live_pinned: usize = 0;
    pub var live_device: usize = 0;
    var spans: [256][]u8 = undefined;
    var nspans: usize = 0;

    pub fn reset() void {
        calls = @splat(0);
    }
    fn tally(k: Kind) void {
        calls[@backingInt(k)] += 1;
    }
    fn take(n: usize) ?[]u8 {
        if (nspans == spans.len) @panic("the fake driver holds 256 allocations at most");
        const m = std.heap.page_allocator.alloc(u8, @max(n, 1)) catch return null;
        spans[nspans] = m;
        nspans += 1;
        return m;
    }
    fn give(p: usize) void {
        for (spans[0..nspans], 0..) |m, i| if (@intFromPtr(m.ptr) == p) {
            std.heap.page_allocator.free(m);
            nspans -= 1;
            spans[i] = spans[nspans];
            return;
        };
        @panic("freed memory the fake never gave");
    }
    fn hostAlloc(p: *?*anyopaque, n: usize, _: c_uint) callconv(.c) abi.Result {
        tally(.pinned);
        const m = take(n) orelse return 2;
        p.* = m.ptr;
        live_pinned += 1;
        return abi.success;
    }
    fn hostFree(p: ?*anyopaque) callconv(.c) abi.Result {
        give(@intFromPtr(p));
        live_pinned -= 1;
        return abi.success;
    }
    fn devAlloc(p: *abi.DevicePtr, n: usize) callconv(.c) abi.Result {
        tally(.device);
        const m = take(n) orelse return 2;
        p.* = @intFromPtr(m.ptr);
        live_device += 1;
        return abi.success;
    }
    pub fn devFree(p: abi.DevicePtr) callconv(.c) abi.Result {
        give(p);
        live_device -= 1;
        return abi.success;
    }
    fn copy(dst: abi.DevicePtr, src: ?*const anyopaque, n: usize) callconv(.c) abi.Result {
        tally(.copy);
        @memcpy(@as([*]u8, @ptrFromInt(dst))[0..n], @as([*]const u8, @ptrCast(src.?))[0..n]);
        return abi.success;
    }
    fn sync(_: abi.Stream) callconv(.c) abi.Result {
        return abi.success;
    }
    fn syncCtx() callconv(.c) abi.Result {
        return abi.success;
    }
    fn name(_: abi.Result, s: *?[*:0]const u8) callconv(.c) abi.Result {
        s.* = null;
        return 1;
    }
    pub fn driver() cuda.Driver {
        var d: cuda.Driver = undefined;
        d.func_name = null;
        d.api.cuMemHostAlloc = hostAlloc;
        d.api.cuMemFreeHost = hostFree;
        d.api.cuMemAlloc_v2 = devAlloc;
        d.api.cuMemFree_v2 = devFree;
        d.api.cuMemcpyHtoD_v2 = copy;
        d.api.cuStreamSynchronize = sync;
        d.api.cuCtxSynchronize = syncCtx;
        d.api.cuGetErrorName = name;
        d.api.cuGetErrorString = name;
        return d;
    }
};

pub const layer_w = 64; // the experts' intermediate width
pub const layer_d = 64; // the model width
pub const layer_n = 12; // experts a layer, each projection in its own shard: 36 shards

/// One expert projection's planes: FP4 codes, row-major E4M3 scales a 16 (decoded, for CSF), the global.
pub const Planes = struct { codes: []u8, scales: []u8, global: f32 };

pub const Layer = struct { index: usize, csf: bool = false };

/// A CSF plane of `scales` [rows, cols], as core/nvfp4_csf_test.zig builds one: row bases, offsets, exceptions.
pub fn csfPlanes(scales: []u8, rows: usize, cols: usize, fixed: *std.ArrayList(u8), exceptions: *std.ArrayList(u8)) !void {
    try fixed.appendNTimes(ta, 0, rows + rows * cols / 2);
    for (0..rows) |row| {
        const base: u8 = 0x30;
        const slab = row / 16 * (16 + cols * 8);
        fixed.items[slab + row % 16] = base;
        for (0..cols) |col| {
            const i = row * cols + col;
            const delta: u8 = if (scales[i] >= base and scales[i] - base < 16) scales[i] - base else 0;
            fixed.items[slab + 16 + row % 16 * (cols / 2) + col / 2] |= delta << @as(u3, if (col % 2 == 0) 0 else 4);
            if (base + delta != scales[i]) try exceptions.appendSlice(ta, &std.mem.toBytes(std.mem.nativeToLittle(u32, @as(u32, @intCast(i)) | (@as(u32, scales[i]) << 24))));
        }
    }
}

/// `layers` with every expert projection in its own shard; `missing` drops one down projection's global.
pub fn fixture(dir: std.Io.Dir, layers: []const Layer, missing: ?struct { layer: usize, expert: usize }, planes: []Planes) !void {
    var index: std.Io.Writer.Allocating = .init(ta);
    defer index.deinit();
    try index.writer.writeAll("{\"weight_map\":{");
    var prng = std.Random.DefaultPrng.init(0xe1);
    const r = prng.random();
    var first = true;
    var at: usize = 0;
    for (layers) |layer| for (0..layer_n) |e| for ([_][]const u8{ "gate_proj", "up_proj", "down_proj" }, 0..) |proj, pi| {
        const n: usize = if (pi == 2) layer_d else layer_w;
        const k: usize = if (pi == 2) layer_w else layer_d;
        const p = &planes[at];
        at += 1;
        p.* = .{ .codes = try ta.alloc(u8, n * k / 2), .scales = try ta.alloc(u8, n * k / 16), .global = 0.5 + 0.01 * @as(f32, @floatFromInt(e)) };
        r.bytes(p.codes);
        // finite E4M3 bytes; one in 9 off the CSF slab's 16-byte window (an exception)
        for (p.scales) |*s| s.* = if (r.uintLessThan(u8, 9) == 0) 0x10 + r.uintLessThan(u8, 16) else 0x30 + r.uintLessThan(u8, 16);
        const base = try std.fmt.allocPrint(ta, "language_model.model.layers.{d}.mlp.experts.{d}.{s}", .{ layer.index, e, proj });
        defer ta.free(base);
        const file = try std.fmt.allocPrint(ta, "l{d}-e{d}-{s}.safetensors", .{ layer.index, e, proj });
        defer ta.free(file);
        const drop = if (missing) |m| m.layer == layer.index and m.expert == e and pi == 2 else false;
        var header: std.Io.Writer.Allocating = .init(ta);
        defer header.deinit();
        var data: std.ArrayList(u8) = .empty;
        defer data.deinit(ta);
        var parts: [4][]const u8 = undefined;
        var nparts: usize = 0;
        try data.appendSlice(ta, p.codes);
        try header.writer.print("{{\"{s}.weight\":{{\"dtype\":\"U8\",\"shape\":[{d},{d}],\"data_offsets\":[0,{d}]}}", .{ base, n, k / 2, data.items.len });
        parts[nparts] = "weight";
        nparts += 1;
        if (layer.csf) {
            var fixed: std.ArrayList(u8) = .empty;
            defer fixed.deinit(ta);
            var exc: std.ArrayList(u8) = .empty;
            defer exc.deinit(ta);
            try csfPlanes(p.scales, n, k / 16, &fixed, &exc);
            const f0 = data.items.len;
            try data.appendSlice(ta, fixed.items);
            try header.writer.print(",\"{s}.weight_scale.nvfp4_csf_fixed\":{{\"dtype\":\"U8\",\"shape\":[{d},{d}],\"data_offsets\":[{d},{d}]}}", .{ base, n / 16, 16 + 8 * (k / 16), f0, data.items.len });
            const x0 = data.items.len;
            try data.appendSlice(ta, exc.items);
            try header.writer.print(",\"{s}.weight_scale.nvfp4_csf_exceptions\":{{\"dtype\":\"U32\",\"shape\":[{d}],\"data_offsets\":[{d},{d}]}}", .{ base, exc.items.len / 4, x0, data.items.len });
            parts[nparts] = "weight_scale.nvfp4_csf_fixed";
            parts[nparts + 1] = "weight_scale.nvfp4_csf_exceptions";
            nparts += 2;
        } else {
            const s0 = data.items.len;
            try data.appendSlice(ta, p.scales);
            try header.writer.print(",\"{s}.weight_scale\":{{\"dtype\":\"F8_E4M3\",\"shape\":[{d},{d}],\"data_offsets\":[{d},{d}]}}", .{ base, n, k / 16, s0, data.items.len });
            parts[nparts] = "weight_scale";
            nparts += 1;
        }
        if (!drop) {
            const g0 = data.items.len;
            try data.appendSlice(ta, &std.mem.toBytes(@as(u32, @bitCast(p.global))));
            try header.writer.print(",\"{s}.weight_scale_2\":{{\"dtype\":\"F32\",\"shape\":[],\"data_offsets\":[{d},{d}]}}", .{ base, g0, data.items.len });
            parts[nparts] = "weight_scale_2";
            nparts += 1;
        }
        try header.writer.writeByte('}');
        var image: std.Io.Writer.Allocating = .init(ta);
        defer image.deinit();
        try image.writer.writeAll(&std.mem.toBytes(std.mem.nativeToLittle(u64, header.written().len)));
        try image.writer.writeAll(header.written());
        try image.writer.writeAll(data.items);
        try dir.writeFile(io, .{ .sub_path = file, .data = image.written() });
        for (parts[0..nparts]) |part| {
            if (!first) try index.writer.writeByte(',');
            first = false;
            try index.writer.print("\"{s}.{s}\":\"{s}\"", .{ base, part, file });
        }
    };
    try index.writer.writeAll("}}");
    try dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = index.written() });
}

pub fn freePlanes(planes: []Planes) void {
    for (planes) |p| {
        ta.free(p.codes);
        ta.free(p.scales);
    }
}

pub fn prefixOf(buf: []u8, layer: usize) ![]const u8 {
    return std.fmt.bufPrint(buf, "language_model.model.layers.{d}.mlp", .{layer});
}

/// The words and globals the one-expert loop packs for a layer's planes, against the device buffers.
pub fn expectLoopWords(owned: experts.Owned, planes: []const Planes) !void {
    const up_words = ex.bytes(1, layer_w, layer_d, 2) / 4;
    const down_words = ex.bytes(1, layer_d, layer_w, 1) / 4;
    const want_up = try ta.alloc(u32, layer_n * up_words);
    defer ta.free(want_up);
    const want_down = try ta.alloc(u32, layer_n * down_words);
    defer ta.free(want_down);
    for (0..layer_n) |i| {
        const g = planes[3 * i];
        const u = planes[3 * i + 1];
        const o = planes[3 * i + 2];
        ex.packGateUp(want_up[i * up_words ..][0..up_words], .{ g.codes, g.scales }, .{ u.codes, u.scales }, layer_w, layer_d);
        ex.packOne(want_down[i * down_words ..][0..down_words], o.codes, o.scales, layer_d, layer_w);
    }
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(want_up), @as([*]const u8, @ptrFromInt(owned.up.ptr))[0..owned.up.len]);
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(want_down), @as([*]const u8, @ptrFromInt(owned.down.ptr))[0..owned.down.len]);
    const ug: [*]const f32 = @ptrFromInt(owned.up_scale.ptr);
    const dg: [*]const f32 = @ptrFromInt(owned.down_scale.ptr);
    for (0..layer_n) |i| {
        try std.testing.expectEqual(planes[3 * i].global, ug[2 * i]);
        try std.testing.expectEqual(planes[3 * i + 1].global, ug[2 * i + 1]);
        try std.testing.expectEqual(planes[3 * i + 2].global, dg[i]);
    }
}
