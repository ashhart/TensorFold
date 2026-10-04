//! The command encoder the Nemotron forward writes into, and a profiler that runs each dispatch alone.
const std = @import("std");
const mtl = @import("metal");
const kern = @import("kernels.zig");

const Buffer = mtl.Buffer;

/// A compute encoder that dispatches as mx.fast.metal_kernel does: dispatchThreads, group = min(threadgroup, grid).
pub const Enc = struct {
    e: mtl.ComputeEncoder,
    dispatches: usize = 0,
    concurrent: bool = false, // a concurrent encoder: a barrier before every dispatch not marked `alongside`
    skip: bool = false,
    prof: ?*Profiler = null, // times each dispatch in a command buffer of its own
    drop: []const mtl.objc.Id = &.{}, // pipelines whose dispatches a cost probe leaves out (their outputs go stale)
    dropping: bool = false,
    gate: ?*Gate = null, // dispatches whose threadgroup counts the GPU writes (a GPU-side round's head levels)

    /// A level's dispatches as whole threadgroups, their counts read from `live` (zeros: skipped) or recorded into `tmpl`.
    pub const Gate = struct {
        live: Buffer,
        base: usize, // the 3-u32 entry this level's first dispatch reads
        tmpl: ?[]u32 = null, // recording: each dispatch's counts also go here (for the GPU to copy to later levels)
        at: usize = 0,
    };

    /// The next dispatch reads nothing the dispatches since the last barrier write: no barrier before it.
    pub fn alongside(self: *Enc) void {
        self.skip = true;
    }

    pub fn pipe(self: *Enc, p: mtl.Pipeline) void {
        self.dropping = std.mem.indexOfScalar(mtl.objc.Id, self.drop, p.id) != null;
        if (self.dropping) return;
        self.e.setPipeline(p);
        if (self.prof) |pr| pr.pick(p);
    }

    pub fn buf(self: *Enc, b: Buffer, offset: usize, index: usize) void {
        self.e.setBuffer(b, offset, index);
    }

    pub fn bytes(self: *Enc, value: anytype, index: usize) void {
        self.e.setBytes(std.mem.asBytes(&value), index);
    }

    pub fn run(self: *Enc, grid: [3]usize, group: [3]usize) void {
        if (self.dropping) {
            self.skip = false;
            return;
        }
        if (self.concurrent and !self.skip and self.dispatches > 0) self.e.barrier();
        self.skip = false;
        if (self.gate) |g| {
            // whole threadgroups: a group that does not divide its grid halves until it does (these kernels index by thread)
            var size: [3]usize = undefined;
            var n: [3]u32 = undefined;
            for (0..3) |d| {
                size[d] = @min(group[d], grid[d]);
                while (grid[d] % size[d] != 0) size[d] /= 2;
                n[d] = @intCast(grid[d] / size[d]);
            }
            const threads = mtl.Size.of(size[0], size[1], size[2]);
            if (g.tmpl) |t| t[3 * g.at ..][0..3].* = n;
            self.e.dispatchIndirect(g.live, (g.base + g.at) * 12, threads);
            g.at += 1;
        } else self.e.dispatchThreads(mtl.Size.of(grid[0], grid[1], grid[2]), mtl.Size.of(@min(group[0], grid[0]), @min(group[1], grid[1]), @min(group[2], grid[2])));
        self.dispatches += 1;
        if (self.prof) |pr| self.e = pr.cut(self.e);
    }
};

/// GPU time by kernel, each dispatch alone in a command buffer (the step's kernels without their boundaries).
pub const Profiler = struct {
    queue: mtl.Queue,
    kernels: *const kern.Kernels,
    cb: ?mtl.CommandBuffer = null,
    current: usize = 0,
    ms: [kern.total]f64 = @splat(0),
    calls: [kern.total]usize = @splat(0),

    fn pick(self: *Profiler, p: mtl.Pipeline) void {
        for (self.kernels.pipelines, 0..) |q, i| if (q.id == p.id) {
            self.current = i;
        };
    }

    /// The encoder for the next dispatch, after the last one ran alone.
    pub fn cut(self: *Profiler, enc: mtl.ComputeEncoder) mtl.ComputeEncoder {
        enc.end();
        const cb = self.cb.?;
        cb.commit();
        cb.wait();
        self.ms[self.current] += cb.gpuSeconds() * 1e3;
        self.calls[self.current] += 1;
        return self.begin();
    }

    pub fn begin(self: *Profiler) mtl.ComputeEncoder {
        self.cb = self.queue.commandBuffer();
        return self.cb.?.compute(.serial);
    }

    pub fn report(self: *const Profiler, steps: usize) void {
        var total: f64 = 0;
        for (self.ms) |v| total += v;
        std.debug.print("kernels alone, ms a step ({d} steps): total {d:.3}\n", .{ steps, total / @as(f64, @floatFromInt(steps)) });
        for (self.ms, self.calls, 0..) |v, n, i| {
            if (n == 0) continue;
            std.debug.print("  {s:<24} {d:>4} calls a step {d:>8.3} ms {d:>7.2} us a call\n", .{ kern.keyOf(i), n / steps, v / @as(f64, @floatFromInt(steps)), v * 1e3 / @as(f64, @floatFromInt(n)) });
        }
    }
};
