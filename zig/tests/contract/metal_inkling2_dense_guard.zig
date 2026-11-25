//! The dense-MMA tail reads no row past its input; every row of a partial tile still equals the matvec it replaces.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");
const tf = @import("tensorfold").inkling2_metal_host;
const dense = tf.dense;
const rig_ = @import("rig.zig");

const n: u32 = 16;
const k: u32 = 3072;
/// Under one tile, a full tile, a partial tile and the sizes the issue named around 45.
const cases = [_]u32{ 1, 7, 8, 9, 16, 17, 45, 46, 47 };

const MatvecArgs = extern struct { rows: u32, n: u32, k: u32, ldy: u32 };

fn hr(r: rig_.Ref) dense.Ref {
    return .{ .buf = r.buf, .off = r.off };
}

/// The kernels it replaces: `dense`'s launches of up to 32 rows.
fn replaced(device: mtl.Device, r: *rig_.Rig, x: rig_.Ref, w: rig_.Ref, out: rig_.Ref, rows: u32) !void {
    const lib = try mtl.Library.fromSource(device, sources.spark25_glue, .{ .language = mtl.CompileOptions.mlx().language, .math_functions = .precise });
    defer lib.deinit();
    var pipes: [6]mtl.Pipeline = undefined;
    var made: usize = 0;
    defer for (pipes[0..made]) |p| p.deinit();
    for ([_]u32{ 1, 2, 4, 8, 16, 32 }, 0..) |take, i| {
        var name: [32]u8 = undefined;
        pipes[i] = try mtl.Pipeline.init(device, lib, try std.fmt.bufPrintSentinel(&name, "sp_matvec_{d}", .{take}, 0), false);
        made += 1;
    }
    const pass = r.begin();
    var base: u32 = 0;
    while (base < rows) {
        const left = rows - base;
        const take: u32 = if (left >= 32) 32 else if (left >= 16) 16 else if (left >= 8) 8 else if (left >= 4) 4 else if (left >= 2) 2 else 1;
        const variant: usize = if (take > 16) 5 else if (take > 8) 4 else if (take > 4) 3 else if (take > 2) 2 else if (take > 1) 1 else 0;
        pass.e.setPipeline(pipes[variant]);
        pass.e.setBuffer(x.buf, x.off + base * k * 2, 0);
        pass.e.setBuffer(w.buf, w.off, 1);
        pass.e.setValue(MatvecArgs{ .rows = take, .n = n, .k = k, .ldy = n }, 2);
        pass.e.setBuffer(out.buf, out.off + base * n * 2, 3);
        pass.e.dispatchGroups(mtl.Size.of((n + 7) / 8, 1, 1), mtl.Size.of(256, 1, 1));
        base += take;
    }
    try r.finish(pass);
}

test "metal inkling2 mma tf_ink2_dense_mma reads no row past its input" {
    // Metal tolerates a read past a buffer's end; only its shader validation faults, and the named build step sets it.
    if (std.c.getenv("MTL_SHADER_VALIDATION") == null) return error.SkipZigTest;
    const device = mtl.Device.init() catch |err| switch (err) {
        error.NoDevice => return error.SkipZigTest,
        else => return err,
    };
    defer device.deinit();
    if (device.tensorUnits()) return error.SkipZigTest; // the units kernels take the projections there
    const queue = try device.queue();
    defer queue.deinit();
    const mma = try dense.Mma.load(device) orelse return error.SkipZigTest;
    defer mma.deinit();

    var rig: rig_.Rig = .{ .a = std.testing.allocator, .device = device, .queue = queue };
    defer rig.deinit();

    var prng = std.Random.DefaultPrng.init(0x64656e73);
    const rng = prng.random();

    const page = std.heap.pageSize();
    for (cases) |rows| {
        defer rig.reset();
        const x_bytes = @as(usize, rows) * k * 2;
        const y_bytes = @as(usize, rows) * n * 2;
        // The input ends at a guard page: a read one row past the payload faults, and no other page is slack.
        const valid = std.mem.alignForward(usize, x_bytes, page);
        const mem = try std.posix.mmap(null, valid + page, .{ .READ = true, .WRITE = true }, .{ .TYPE = .PRIVATE, .ANONYMOUS = true }, -1, 0);
        defer std.posix.munmap(mem);
        if (std.c.mprotect(@ptrCast(@alignCast(mem[valid..].ptr)), page, .{}) != 0) return error.MprotectFailed;
        const off = valid - x_bytes;
        const x = mem[off..][0..x_bytes];
        const xv = std.mem.bytesAsSlice(u16, x);
        for (xv) |*v| v.* = @truncate(@as(u32, @bitCast(rng.float(f32) * 2 - 1)) >> 16);
        const w = try rig.a.alloc(u16, @as(usize, n) * k);
        defer rig.a.free(w);
        for (w) |*v| v.* = @truncate(@as(u32, @bitCast(rng.float(f32) * 2 - 1)) >> 16);
        // Huge terms that cancel leave the small terms' sum less what each order of adding lost.
        for ([_]usize{ 37, 2100 }) |i| {
            for (0..rows) |row| xv[row * k + i] = 0x4980; // 2^20
            for (0..rows) |row| xv[row * k + i + 163] = 0x4980;
            for (0..n) |o| {
                w[o * k + i] = 0x3f80; // 1
                w[o * k + i + 163] = 0xbf80; // -1
            }
        }
        const xbuf = try device.bufferNoCopy(@ptrCast(mem.ptr), valid, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        defer xbuf.deinit();
        const xin = try rig.input(x);
        const win = try rig.input(std.mem.sliceAsBytes(w));
        const y = try rig.output(y_bytes);
        const want = try rig.scratch(y_bytes);

        const pass = rig.begin();
        mma.launch(pass.e, .{ .buf = xbuf, .off = off }, hr(win), hr(y), .{ .rows = rows, .n = n, .k = k, .ldy = n });
        try rig.finish(pass);
        try replaced(device, &rig, xin, win, want, rows);

        const got = try rig.outputBytes();
        const want_bytes = rig.bytes(want, y_bytes);
        if (!std.mem.eql(u8, got, want_bytes)) {
            for (0..rows * n) |i| {
                if (got[2 * i] == want_bytes[2 * i] and got[2 * i + 1] == want_bytes[2 * i + 1]) continue;
                std.debug.print("rows {d}: output {d} differs: {x}{x} against the matvec's {x}{x}\n", .{ rows, i, got[2 * i + 1], got[2 * i], want_bytes[2 * i + 1], want_bytes[2 * i] });
                break;
            }
            return error.BitsDiffer;
        }
        try std.testing.expect(try rig.intact());
    }
}
