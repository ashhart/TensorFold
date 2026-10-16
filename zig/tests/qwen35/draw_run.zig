//! `draw`: the device argmax and top-k kernels against the host's (sample.zig) on tie-heavy rows (GPU).

const std = @import("std");
const hip = @import("hip");
const qwen35 = @import("qwen3_5");

const sample = qwen35.sample;

const Order = struct {
    row: []const u16,
    dtype: sample.Dtype,

    fn less(c: Order, a: u32, b: u32) bool {
        const va = sample.widen(c.row, c.dtype, a);
        const vb = sample.widen(c.row, c.dtype, b);
        return va > vb or (va == vb and a < b);
    }
};

pub fn run(gpa: std.mem.Allocator) !void {
    var d = try hip.Runtime.open();
    defer d.close();
    var ctx = try hip.Context.init(&d, 0);
    defer ctx.deinit();
    const caps = try hip.Device.kernelCaps(&d, 0);
    var notes: hip.Policy.Notes = .{};
    var lib = try hip.Launcher.load(&d, (try hip.Policy.resolve("", .current, &notes)).choices(), hip.kernels.images);
    defer lib.unload();
    var stream = try hip.Stream.init(&d);
    defer stream.deinit();
    var arena = try hip.Arena.init(&d, 1 << 20);
    defer arena.deinit();
    const o: hip.ops.Ops = .{ .l = &lib, .bf16 = caps.act == .bf16, .stream = stream.handle, .arena = &arena };
    const dtype: sample.Dtype = if (caps.act == .f16) .f16 else .bf16;
    const kind: hip.ops.Kind = if (caps.act == .f16) .f16 else .bf16;
    var prng = std.Random.DefaultPrng.init(7);
    const rnd = prng.random();
    const rows = 5;
    for ([_]usize{ 1000, 4099, 248320 }) |vocab| {
        // few distinct values (both zeros among them) so the k-th value is shared by many ids
        const palette: [8]u16 = if (dtype == .f16)
            .{ 0x0000, 0x8000, 0x3c00, 0xbc00, 0x4000, 0x4400, 0xc000, 0x3800 }
        else
            .{ 0x0000, 0x8000, 0x3f80, 0xbf80, 0x4000, 0x4080, 0xc000, 0x3f00 };
        const host = try gpa.alloc(u16, rows * vocab);
        defer gpa.free(host);
        for (host, 0..) |*v, i| v.* = if (i / vocab % 2 == 0) palette[rnd.uintLessThan(usize, palette.len)] else rnd.int(u16) & 0xbfff;
        var logits = try hip.DeviceBuffer.alloc(&d, host.len * 2);
        defer logits.free();
        try logits.upload(0, std.mem.sliceAsBytes(host));
        const ks = [rows]i32{ 1, 8, 28, 100, 0 };
        const stride = 100;
        var ks_dev = try hip.DeviceBuffer.alloc(&d, rows * 4);
        defer ks_dev.free();
        try ks_dev.upload(0, std.mem.sliceAsBytes(&ks));
        var arg_dev = try hip.DeviceBuffer.alloc(&d, rows * 4);
        defer arg_dev.free();
        var ids_dev = try hip.DeviceBuffer.alloc(&d, rows * stride * 4);
        defer ids_dev.free();
        var vals_dev = try hip.DeviceBuffer.alloc(&d, rows * stride * 2);
        defer vals_dev.free();
        const t: hip.ops.Tensor = .{ .ptr = logits.base(), .kind = kind };
        try o.argmaxRows(t, rows, vocab, arg_dev.base());
        try o.topkRows(t, rows, vocab, ks_dev.base(), stride, ids_dev.base(), vals_dev.base());
        try stream.synchronize();
        var arg: [rows]i32 = undefined;
        try arg_dev.download(0, std.mem.sliceAsBytes(&arg));
        const ids = try gpa.alloc(i32, rows * stride);
        defer gpa.free(ids);
        try ids_dev.download(0, std.mem.sliceAsBytes(ids));
        const vals = try gpa.alloc(u16, rows * stride);
        defer gpa.free(vals);
        try vals_dev.download(0, std.mem.sliceAsBytes(vals));
        for (0..rows) |r| {
            const row = host[r * vocab ..][0..vocab];
            if (sample.argmax(row, dtype) != arg[r]) {
                std.debug.print("argmax row {d} vocab {d}: device {d}, host {d}\n", .{ r, vocab, arg[r], sample.argmax(row, dtype) });
                return error.Mismatch;
            }
            const k: usize = @intCast(ks[r]);
            if (k == 0) continue;
            // the host's selection: sort every id by (value desc, id asc) and take k
            const order = try gpa.alloc(u32, vocab);
            defer gpa.free(order);
            for (order, 0..) |*x, i| x.* = @intCast(i);
            std.mem.sort(u32, order, Order{ .row = row, .dtype = dtype }, Order.less);
            var want: std.AutoHashMapUnmanaged(u32, void) = .empty;
            defer want.deinit(gpa);
            for (order[0..k]) |id| try want.put(gpa, id, {});
            for (0..k) |j| {
                const id: u32 = @intCast(ids[r * stride + j]);
                if (!want.remove(id) or vals[r * stride + j] != row[id]) {
                    std.debug.print("top-{d} row {d} vocab {d}: id {d} is not the host's pick\n", .{ k, r, vocab, id });
                    return error.Mismatch;
                }
            }
        }
        std.debug.print("vocab {d}: argmax and top-k of {d} rows equal the host's\n", .{ vocab, rows });
    }
    std.debug.print("OK\n", .{});
}
