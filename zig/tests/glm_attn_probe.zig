//! GLM-5.3-Flash's dense latent attention (forward.attendDense) on given inputs, for a bit check against a reference.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");

const usage =
    \\tf-glm-attn-probe IN OUT. IN: u32 cases, then a case: u32 n, q [64, 512] bf16 (scaled), keys [n, 512] bf16.
    \\OUT: a case's scores [64, n], probabilities [64, n] and outputs [64, 512], bf16.
;
const glm = tf.glm;

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const arena = init.arena.allocator();
    const args = try init.minimal.args.toSlice(arena);
    if (args.len != 3) {
        std.debug.print("usage: {s}\n", .{usage});
        std.process.exit(2);
    }
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const in = try mtl.MappedFile.open(try std.fmt.allocPrintSentinel(arena, "{s}", .{args[1]}, 0));
    defer in.deinit();
    const bytes = in.bytes[0..in.size];
    const device = try mtl.Device.init();
    const queue = try device.queue();
    const k = try glm.kernels.load(gpa, device);
    const c: glm.config.Config = .{};
    var x: glm.forward.Ctx = .{ .k = k, .c = &c, .w = undefined, .s = undefined, .sc = undefined };
    const opts = mtl.ResourceOptions.shared;
    const cases = std.mem.readInt(u32, bytes[0..4], .little);
    var at: usize = 4;
    var out: std.ArrayList(u8) = .empty;
    for (0..cases) |_| {
        const n: usize = std.mem.readInt(u32, bytes[at..][0..4], .little);
        at += 4;
        const qb = 64 * 512 * 2;
        const kb = n * 512 * 2;
        const bufs = [_]mtl.Buffer{ try device.buffer(qb, opts), try device.buffer(@max(kb, 16), opts), try device.buffer(64 * n * 2, opts), try device.buffer(64 * n * 2, opts), try device.buffer(qb, opts) };
        defer for (bufs) |b| b.deinit();
        @memcpy(bufs[0].contents()[0..qb], bytes[at..][0..qb]);
        at += qb;
        @memcpy(bufs[1].contents()[0..kb], bytes[at..][0..kb]);
        at += kb;
        const cb = queue.commandBuffer();
        const enc = cb.compute(.serial);
        const R = glm.weights.Ref;
        glm.forward.attendDense(&x, enc, R{ .buf = bufs[1] }, R{ .buf = bufs[0] }, R{ .buf = bufs[2] }, R{ .buf = bufs[3] }, R{ .buf = bufs[4] }, @intCast(n));
        enc.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |msg| {
            std.debug.print("n {d}: command buffer failed: {s}\n", .{ n, msg });
            std.process.exit(1);
        }
        try out.appendSlice(arena, bufs[2].contents()[0 .. 64 * n * 2]);
        try out.appendSlice(arena, bufs[3].contents()[0 .. 64 * n * 2]);
        try out.appendSlice(arena, bufs[4].contents()[0..qb]);
    }
    const file = std.c.fopen(try std.fmt.allocPrintSentinel(arena, "{s}", .{args[2]}, 0), "wb") orelse return error.OpenFailed;
    defer _ = std.c.fclose(file);
    if (std.c.fwrite(out.items.ptr, 1, out.items.len, file) != out.items.len) return error.WriteFailed;
    std.debug.print("{d} cases\n", .{cases});
}
