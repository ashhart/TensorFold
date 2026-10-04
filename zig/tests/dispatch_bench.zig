//! Dependent dispatch boundaries through serial and concurrent encoders and indirect command buffers; run under the GPU lock.
const std = @import("std");
const mtl = @import("metal");
const common = @import("common.zig");

const Gpu = common.Gpu;
const Size = mtl.Size;
const counts = [_]usize{ 1, 16, 64, 256, 1024 };
const max_count = counts[counts.len - 1];
const reps = 15;

const Shape = struct { name: []const u8, groups: usize, threads: usize };
const shapes = [_]Shape{
    .{ .name = "tiny: 1 threadgroup x 32", .groups = 1, .threads = 32 },
    .{ .name = "wide: 1024 threadgroups x 64", .groups = 1024, .threads = 64 },
};

const Mode = enum { serial, concurrent_barrier, icb_barrier, icb_serial_encoder, concurrent_no_dependency };

const Bench = struct {
    gpu: Gpu,
    step: mtl.Pipeline,
    x: mtl.Buffer,
    icb: mtl.IndirectCommandBuffer,
    shape: Shape,

    fn words(self: Bench) usize {
        return self.shape.groups * self.shape.threads;
    }
};

/// tf_step's value after `times` steps from 0: v -> 3 v + 1, wrapping.
fn expected(times: usize) u32 {
    var v: u32 = 0;
    for (0..times) |_| v = v *% 3 +% 1;
    return v;
}

/// One run's GPU, host-encode and wall seconds, and whether every word took exactly its steps in order.
fn run(b: Bench, mode: Mode, count: usize) !struct { gpu: f64, encode: f64, wall: f64, exact: bool } {
    const independent = mode == .concurrent_no_dependency;
    const words = if (independent) b.words() * count else b.words();
    @memset(b.x.slice(u32, words), 0);
    const cb = b.gpu.queue.commandBuffer();
    const groups = Size.of(b.shape.groups, 1, 1);
    const threads = Size.of(b.shape.threads, 1, 1);
    const t0 = mtl.clock.seconds();
    switch (mode) {
        .serial, .concurrent_barrier, .concurrent_no_dependency => {
            const enc = cb.compute(if (mode == .serial) .serial else .concurrent);
            enc.setPipeline(b.step);
            for (0..count) |k| {
                enc.setBuffer(b.x, if (independent) k * b.words() * @sizeOf(u32) else 0, 0);
                enc.dispatchGroups(groups, threads);
                if (mode == .concurrent_barrier and k + 1 < count) enc.barrier();
            }
            enc.end();
        },
        .icb_barrier, .icb_serial_encoder => {
            const enc = cb.compute(if (mode == .icb_barrier) .concurrent else .serial);
            enc.useResource(b.x, mtl.ResourceUsage.read | mtl.ResourceUsage.write);
            enc.execute(b.icb, 0, count);
            enc.end();
        },
    }
    const encode = mtl.clock.seconds() - t0;
    try common.run(cb);
    const wall = mtl.clock.seconds() - t0;
    const want = expected(if (independent) 1 else count);
    var exact = true;
    for (b.x.slice(u32, words)) |v| exact = exact and v == want;
    return .{ .gpu = cb.gpuSeconds(), .encode = encode, .wall = wall, .exact = exact };
}

/// Least-squares slope of y over x.
fn slope(xs: []const f64, ys: []const f64) f64 {
    var mx: f64 = 0;
    var my: f64 = 0;
    for (xs, ys) |x, y| {
        mx += x;
        my += y;
    }
    mx /= @floatFromInt(xs.len);
    my /= @floatFromInt(xs.len);
    var num: f64 = 0;
    var den: f64 = 0;
    for (xs, ys) |x, y| {
        num += (x - mx) * (y - my);
        den += (x - mx) * (x - mx);
    }
    return num / den;
}

fn bench(gpu: Gpu, step: mtl.Pipeline, step_icb: mtl.Pipeline, x: mtl.Buffer, shape: Shape) !void {
    // the indirect command buffer is encoded once; a run replays its first N commands
    const t_build = mtl.clock.seconds();
    const icb = try mtl.IndirectCommandBuffer.init(gpu.device, max_count, 1);
    defer icb.deinit();
    for (0..max_count) |k| {
        const cmd = icb.command(k);
        cmd.setPipeline(step_icb);
        cmd.setBuffer(x, 0, 0);
        cmd.dispatchGroups(Size.of(shape.groups, 1, 1), Size.of(shape.threads, 1, 1));
        cmd.setBarrier();
    }
    const build_us = (mtl.clock.seconds() - t_build) * 1e6 / @as(f64, @floatFromInt(max_count));
    const b = Bench{ .gpu = gpu, .step = step, .x = x, .icb = icb, .shape = shape };

    const modes = comptime std.enums.values(Mode);
    var gpu_s: [modes.len][counts.len][reps]f64 = undefined;
    var enc_s: [modes.len][counts.len][reps]f64 = undefined;
    var wall_s: [modes.len][counts.len][reps]f64 = undefined;
    var exact: [modes.len]bool = @splat(true);
    // warm the GPU clock, then interleave modes and sizes so drift hits them alike
    for (0..20) |_| _ = try run(b, .serial, max_count);
    for (0..reps) |r| {
        for (modes, 0..) |mode, m| {
            for (counts, 0..) |count, c| {
                const inner = mtl.objc.Pool.push();
                defer inner.pop();
                const res = try run(b, mode, count);
                gpu_s[m][c][r] = res.gpu;
                enc_s[m][c][r] = res.encode;
                wall_s[m][c][r] = res.wall;
                exact[m] = exact[m] and res.exact;
            }
        }
    }

    std.debug.print("\n{s} (ICB encoded once at {d:.2} us a command)\n", .{ shape.name, build_us });
    std.debug.print("{s:<26}", .{"mode / GPU us at N ="});
    for (counts) |count| std.debug.print("{d:>9}", .{count});
    std.debug.print("{s:>14}{s:>14}{s:>16}{s:>8}\n", .{ "GPU us/bound", "wall us/bound", "host us/disp", "exact" });
    for (modes, 0..) |mode, m| {
        var xs: [counts.len]f64 = undefined;
        var ys: [counts.len]f64 = undefined;
        var walls: [counts.len]f64 = undefined;
        std.debug.print("{s:<26}", .{@tagName(mode)});
        for (counts, 0..) |count, c| {
            xs[c] = @floatFromInt(count);
            ys[c] = common.median(&gpu_s[m][c]) * 1e6;
            walls[c] = common.median(&wall_s[m][c]) * 1e6;
            std.debug.print("{d:>9.1}", .{ys[c]});
        }
        const host = common.median(&enc_s[m][counts.len - 1]) * 1e6 / xs[counts.len - 1];
        std.debug.print("{d:>14.3}{d:>14.3}{d:>16.3}{s:>8}\n", .{ slope(&xs, &ys), slope(&xs, &walls), host, if (exact[m]) "yes" else "NO" });
    }
}

pub fn main() !void {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const gpu = try Gpu.init();
    defer gpu.deinit();
    const step = try gpu.pipeline("tf_step", false);
    defer step.deinit();
    const step_icb = try gpu.pipeline("tf_step", true);
    defer step_icb.deinit();
    var most: usize = 0;
    for (shapes) |s| most = @max(most, s.groups * s.threads * max_count);
    const x = try gpu.buffer(u32, most);
    defer x.deinit();
    std.debug.print("device: {s}; GPU time per command buffer, median of {d} runs; slope = us per dependent boundary\n", .{ gpu.device.name(), reps });
    for (shapes) |shape| try bench(gpu, step, step_icb, x, shape);
}
