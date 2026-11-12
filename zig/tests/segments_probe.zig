//! tf-segments-probe: a two-segment chunk behind a held first queue times out without the probe and passes with it.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const segments = tf.segments;

pub const std_options: std.Options = .{ .logFn = log };
var failure: [256]u8 = undefined; // the last error line logged, which segments.run writes for a failed lane
var failure_len: usize = 0;

fn log(comptime level: std.log.Level, comptime scope: @EnumLiteral(), comptime format: []const u8, args: anytype) void {
    std.log.defaultLog(level, scope, format, args);
    if (level != .err) return;
    failure_len = if (std.fmt.bufPrint(&failure, format, args)) |line| line.len else |_| failure.len;
}

const source =
    \\#include <metal_stdlib>
    \\using namespace metal;
    \\kernel void spin(device uint* out [[buffer(0)]], constant uint& n [[buffer(1)]], uint t [[thread_position_in_grid]]) {
    \\  uint x = t + 1;
    \\  for (uint i = 0; i < n; i++) x = x * 1664525u + 1013904223u;
    \\  out[t] = x;
    \\}
;

const slot = 16; // bytes of `out` each writer owns, so no two queues write the same word
const spin_s = 1.5; // each spin command buffer stays well under the GPU watchdog itself
const busy_buffers = 5; // together well past it: spin_s * busy_buffers ahead of the chunk, also the hold

/// One layer of a little work a segment: segment 1 runs its pre, then waits on the GPU for segment 0's event.
const Fam = struct {
    first: mtl.Queue,
    scratch: mtl.Buffer,
    pipe: mtl.Pipeline,
    out: mtl.Buffer,
    with_probe: bool,

    pub fn begin(_: *Fam, _: *segments.Lane) !void {}
    pub fn wait(_: *Fam, _: usize) segments.Wait {
        return .mixer;
    }
    pub fn handoff(_: *Fam, _: *segments.Lane, _: usize) !void {}
    pub fn pre(f: *Fam, l: *segments.Lane, _: usize) !void {
        work(l.enc, f.pipe, f.out, slot * (l.k + 1), 1000);
    }
    pub fn mixer(f: *Fam, l: *segments.Lane, _: usize) !void {
        work(l.enc, f.pipe, f.out, slot * (l.k + 1), 1000);
    }
    pub fn post(_: *Fam, _: *segments.Lane, _: usize) !void {}
    pub fn finish(_: *Fam, _: *segments.Lane) !void {}
    pub fn probe(f: *Fam) ?mtl.CommandBuffer {
        if (!f.with_probe) return null;
        const t: mtl.keepalive.Target = .{ .queue = f.first, .scratch = f.scratch };
        return t.probe();
    }
};

fn work(enc: mtl.ComputeEncoder, pipe: mtl.Pipeline, out: mtl.Buffer, at: usize, n: u32) void {
    enc.setPipeline(pipe);
    enc.setBuffer(out, at, 0);
    enc.setBytes(std.mem.asBytes(&n), 1);
    enc.dispatchThreads(mtl.Size.of(1, 1, 1), mtl.Size.of(1, 1, 1));
}

fn spin(queue: mtl.Queue, pipe: mtl.Pipeline, out: mtl.Buffer, n: u32) mtl.CommandBuffer {
    const cb = queue.commandBuffer();
    const enc = cb.compute(.serial);
    work(enc, pipe, out, 0, n);
    enc.end();
    cb.commit();
    return cb;
}

/// Iterations for one spin command buffer of about `spin_s` on this GPU.
fn calibrate(queue: mtl.Queue, pipe: mtl.Pipeline, out: mtl.Buffer) !u32 {
    var n: u32 = 1 << 22;
    for (0..4) |_| {
        const cb = spin(queue, pipe, out, n);
        cb.wait();
        if (cb.failure()) |msg| {
            std.debug.print("calibration failed: {s}\n", .{msg});
            return error.GpuFailed;
        }
        const s = cb.gpuSeconds();
        if (s > 0.9 * spin_s and s < 1.1 * spin_s) break;
        // at most 16x a step, so a tiny first sample cannot overshoot into one long command buffer
        n = @intFromFloat(@min(@as(f64, @floatFromInt(n)) * @min(spin_s / @max(s, 1e-6), 16.0), 4.0e9));
    }
    return n;
}

/// Releases the held first queue one blocker every `each_ms`: the wait on a value not yet set is the sleep.
fn release(ev: mtl.SharedEvent, each_ms: u64) void {
    for (1..busy_buffers + 1) |v| {
        _ = ev.wait(busy_buffers + 1, each_ms);
        ev.set(v);
    }
}

pub fn main(init: std.process.Init) !void {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const busy_mode = args.len > 1 and std.mem.eql(u8, args[1], "busy"); // spin kernels ahead: reported, not judged
    const device = try mtl.Device.init();
    const first = try device.queue();
    const lib = try mtl.Library.fromSource(device, source, mtl.CompileOptions.mlx());
    const pipe = try mtl.Pipeline.init(device, lib, "spin", false);
    const out = try device.buffer(64, mtl.ResourceOptions.shared);
    const scratch = try device.buffer(64, mtl.ResourceOptions.shared);
    const n = if (busy_mode) try calibrate(first, pipe, out) else 0;
    const hold_ms: u64 = @intFromFloat(spin_s * busy_buffers * 1e3);
    var wrong: usize = 0;
    for ([_]bool{ true, false }) |with_probe| { // the probe first: after a timeout macOS ignores this process's work
        var ahead: [busy_buffers]mtl.CommandBuffer = undefined;
        var n_ahead: usize = 0;
        var thread: ?std.Thread = null;
        if (busy_mode) {
            for (&ahead) |*cb| cb.* = spin(first, pipe, out, n);
            n_ahead = busy_buffers;
        } else { // the first queue held with the GPU idle, as when segment 0's commit re-wires
            const ev = try device.sharedEvent();
            for (&ahead, 1..) |*cb, v| {
                cb.* = first.commandBuffer();
                cb.*.waitFor(ev, v);
                cb.*.commit();
            }
            n_ahead = busy_buffers;
            thread = try std.Thread.spawn(.{}, release, .{ ev, @as(u64, @intFromFloat(spin_s * 1e3)) });
        }
        const queues = [_]mtl.Queue{ first, try device.queue() };
        var fam: Fam = .{ .first = first, .scratch = scratch, .pipe = pipe, .out = out, .with_probe = with_probe };
        failure_len = 0;
        const t0 = mtl.clock.seconds();
        const got = segments.run(device, &queues, 1, .serial, &fam);
        const chunk_s = mtl.clock.seconds() - t0;
        if (thread) |t| t.join();
        var ahead_s: f64 = if (busy_mode) 0 else @as(f64, @floatFromInt(hold_ms)) / 1e3;
        for (ahead[0..n_ahead]) |cb| {
            cb.wait();
            if (busy_mode) ahead_s += cb.gpuSeconds();
            if (cb.failure()) |msg| { // only the chunk may fail: a failed blocker would void the comparison
                std.debug.print("a command buffer ahead of the chunk failed: {s}\n", .{msg});
                wrong += 1;
            }
        }
        const label = if (with_probe) "with the probe" else "without the probe";
        const what = if (busy_mode) "of spin kernels (GPU time)" else "held by the host (nominal), GPU idle";
        if (got) |_| {
            std.debug.print("RESULT {s}: the chunk passed after {d:.2} s, behind {d:.2} s {s}\n", .{ label, chunk_s, ahead_s, what });
            if (!with_probe and !busy_mode) wrong += 1;
            if (with_probe and !busy_mode and chunk_s < 0.9 * ahead_s) { // the pass must come from waiting out the hold
                std.debug.print("the chunk passed before the hold ended: the first queue was not held\n", .{});
                wrong += 1;
            }
        } else |err| {
            std.debug.print("RESULT {s}: the chunk failed ({s}) after {d:.2} s, behind {d:.2} s {s}\n", .{ label, @errorName(err), chunk_s, ahead_s, what });
            // only the watchdog's timeout counts as the failure this tool is about
            const timeout = std.mem.indexOf(u8, failure[0..failure_len], "CallbackErrorTimeout") != null;
            if (!busy_mode and (with_probe or err != error.GpuFailed or !timeout)) wrong += 1;
        }
    }
    if (busy_mode) return;
    if (wrong > 0) {
        std.debug.print("FAIL {d} result(s) other than expected: a pass with the probe, a timeout without it\n", .{wrong});
        std.process.exit(1);
    }
    std.debug.print("PASS with the probe the chunk passed; without it segment 1 timed out in its wait\n", .{});
}
