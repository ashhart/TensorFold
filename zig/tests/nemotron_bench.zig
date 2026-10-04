//! One-row lane projection kernels timed alone at chosen tile counts (how many threadgroups a GPU wave holds).
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");

const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;

fn time(m: *tf.nemotron.Model, p: mtl.Pipeline, lin: tf.nemotron.weights.Linear, x: mtl.Buffer, xs: mtl.Buffer, y: mtl.Buffer, tiles: usize, sk: usize, reps: usize) !f64 {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const cb = m.queue.commandBuffer();
    const enc = cb.compute(.serial);
    const mdims = [8]i32{ 1, 16, 0, 0, 0, 0, 0, 0 };
    for (0..reps) |_| {
        enc.setPipeline(p);
        enc.setBuffer(x, 0, 0);
        enc.setBuffer(xs, 0, 1);
        enc.setBuffer(lin.w, 0, 2);
        enc.setBuffer(lin.sbt, 0, 3);
        enc.setBytes(std.mem.asBytes(&mdims), 4);
        enc.setBuffer(y, 0, 5);
        enc.dispatchThreads(mtl.Size.of(tiles * 64 * sk, 1, 1), mtl.Size.of(64 * sk, 1, 1));
    }
    enc.end();
    cb.commit();
    cb.wait();
    return cb.gpuSeconds() * 1e6 / @as(f64, @floatFromInt(reps));
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const m = try tf.nemotron.Model.load(gpa, init.io, args[1], false);
    defer m.deinit();
    const x = try m.device.buffer(4096 * 2, opts);
    const xs = try m.device.buffer(64 * 16 * 4, opts);
    const y = try m.device.buffer(131072 * 2, opts);
    for (x.slice(u16, 4096), 0..) |*v, i| v.* = @truncate(0x3c00 + i % 64);
    @memset(xs.slice(f32, 64 * 16), 0.5);
    const cases = .{
        .{ "coop_in_1_0", "backbone.layers.0.mixer.in_proj", 4, [_]usize{ 1, 20, 40, 41, 80, 81, 120, 121, 160, 161 } },
        .{ "coop_out_1_0", "backbone.layers.0.mixer.out_proj", 8, [_]usize{ 1, 10, 20, 21, 30, 40, 41, 42, 42, 42 } },
        .{ "coop_down_1_0", "backbone.layers.1.mixer.shared_experts.down_proj", 4, [_]usize{ 1, 10, 20, 21, 30, 40, 41, 42, 42, 42 } },
    };
    inline for (cases) |c| {
        const lin = m.weights.linears.get(c[1]).?;
        const p = m.kernels.get(c[0]);
        _ = try time(m, p, lin, x, xs, y, lin.n / 64, c[2], 20);
        std.debug.print("{s} (N {d}, {d} tiles):", .{ c[0], lin.n, lin.n / 64 });
        for (c[3]) |t| {
            std.debug.print(" {d}:{d:.1}", .{ t, try time(m, p, lin, x, xs, y, t, c[2], 50) });
        }
        std.debug.print(" us\n", .{});
    }
}
