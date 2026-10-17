//! kvsplit.cu against kv/split.zig's HostKernels formulas on seeded data (dense, row mode, gather), byte for byte.

const std = @import("std");
const cuda = @import("cuda");
const dsv41 = @import("dsv41_kernels");
const check = @import("../check.zig");

const Gpu = check.Gpu;
const DenseArgs = dsv41.ops.kvsplit.DenseArgs;

const Case = struct { rows: u32, k: u32, slots: u32, pages: u32, psh: u5, row_bytes: u32, world: u32, stacked: bool };

pub fn run(gpu: Gpu, k: *const dsv41.Kernels) !void {
    const cases = [_]Case{
        .{ .rows = 1, .k = 512, .slots = 1, .pages = 64, .psh = 7, .row_bytes = 584, .world = 2, .stacked = false },
        .{ .rows = 16, .k = 512, .slots = 5, .pages = 40, .psh = 8, .row_bytes = 584, .world = 2, .stacked = true },
        .{ .rows = 3, .k = 1024, .slots = 3, .pages = 33, .psh = 7, .row_bytes = 136, .world = 3, .stacked = true },
    };
    var stream = try cuda.Stream.init(gpu.d, false); // blocking: ordered after the uploads and memsets on the default stream
    defer stream.deinit();
    const o = k.others(stream);
    var prng = std.Random.DefaultPrng.init(4141);
    const rnd = prng.random();
    for (cases) |c| {
        const gpa = gpu.gpa;
        const n: usize = @as(usize, c.rows) * c.k;
        const local_pages = c.pages; // rows a family holds locally: pages << psh
        const base_rows = @as(usize, local_pages) << c.psh;
        const base = try gpa.alloc(u8, base_rows * c.row_bytes);
        defer gpa.free(base);
        rnd.bytes(base);
        const pts: u32 = c.pages * 2; // logical pages a slot
        const table = try gpa.alloc(i32, @as(usize, c.slots) * pts);
        defer gpa.free(table);
        for (table) |*t| t.* = rnd.intRangeLessThan(i32, 0, @intCast(local_pages));
        const rslot = try gpa.alloc(i32, c.rows);
        defer gpa.free(rslot);
        for (rslot) |*s| s.* = if (c.stacked) rnd.intRangeLessThan(i32, 0, @intCast(c.slots)) else 0;
        const sel = try gpa.alloc(i32, n);
        defer gpa.free(sel);
        const logical_rows: i32 = @intCast(@as(u32, pts) << c.psh);
        for (sel) |*s| s.* = if (rnd.uintLessThan(u32, 8) == 0) -1 else rnd.intRangeLessThan(i32, 0, logical_rows);

        // HostKernels.dense
        const want = try gpa.alloc(u8, n * c.row_bytes);
        defer gpa.free(want);
        const want_tok = try gpa.alloc(i32, n);
        defer gpa.free(want_tok);
        const mask = (@as(u32, 1) << c.psh) - 1;
        for (0..n) |i| {
            const t: u32 = @intCast(@max(sel[i], 0));
            const s: usize = if (c.stacked) @intCast(rslot[i / c.k]) else 0;
            const local: u32 = @intCast(table[s * pts + (t >> c.psh)]);
            const phys = (@as(usize, local) << c.psh) | (t & mask);
            @memcpy(want[i * c.row_bytes ..][0..c.row_bytes], base[phys * c.row_bytes ..][0..c.row_bytes]);
            want_tok[i] = if (sel[i] < 0) -1 else @intCast(((t >> c.psh) % c.world) * n + i);
        }

        var d_base = try cuda.DeviceBuffer.fromHost(gpu.d, base);
        defer d_base.free();
        var d_table = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(table));
        defer d_table.free();
        var d_rslot = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(rslot));
        defer d_rslot.free();
        var d_sel = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(sel));
        defer d_sel.free();
        var d_send = try cuda.DeviceBuffer.alloc(gpu.d, n * c.row_bytes);
        defer d_send.free();
        var d_tok = try cuda.DeviceBuffer.alloc(gpu.d, n * 4);
        defer d_tok.free();
        try o.kvDense(.{ .sel = d_sel.ptr, .rows = c.rows, .k = c.k, .table = d_table.ptr, .pts = pts, .rslot = if (c.stacked) d_rslot.ptr else 0, .psh = c.psh, .base = d_base.ptr, .row_bytes = c.row_bytes, .world = c.world, .send = d_send.ptr, .tok = d_tok.ptr });
        try stream.synchronize();
        const got = try check.download(gpu, d_send);
        defer gpa.free(got);
        try check.sameBytes("kvsplit dense rows", got, want);
        const got_tok = try check.download(gpu, d_tok);
        defer gpa.free(got_tok);
        try check.sameBytes("kvsplit dense tokens", got_tok, std.mem.sliceAsBytes(want_tok));

        // HostKernels.gather: a row list of the local rows
        const m: u32 = 300;
        const phys = try gpa.alloc(u32, m);
        defer gpa.free(phys);
        for (phys) |*p| p.* = rnd.uintLessThan(u32, @intCast(base_rows));
        for (0..m) |i| @memcpy(want[i * c.row_bytes ..][0..c.row_bytes], base[@as(usize, phys[i]) * c.row_bytes ..][0..c.row_bytes]);
        var d_phys = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(phys));
        defer d_phys.free();
        try o.kvGather(d_base.ptr, c.row_bytes, d_phys.ptr, m, d_send.ptr);
        try stream.synchronize();
        const got2 = try check.download(gpu, d_send);
        defer gpa.free(got2);
        try check.sameBytes("kvsplit gather", got2[0 .. m * c.row_bytes], want[0 .. m * c.row_bytes]);
        check.pass("BITEXACT kvsplit dense + gather == kv/split.zig HostKernels: R {d} x K {d}, {d} B rows, psh {d}, world {d}, {s}", .{ c.rows, c.k, c.row_bytes, c.psh, c.world, if (c.stacked) "row mode" else "one table" });
    }
}
