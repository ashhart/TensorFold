//! Is a GPU-wide barrier inside one persistent dispatch cheaper than a dependent relaunch? N phases of a small
//! dependent step (each reads what the previous phase wrote), as N serial dispatches and as one persistent dispatch.
const std = @import("std");
const mtl = @import("metal");

const source =
    \\#include <metal_stdlib>
    \\using namespace metal;
    \\// one phase: every thread adds its neighbour's previous value (a dependency on other threadgroups' writes)
    \\kernel void step(device float* a [[buffer(0)]], device float* b [[buffer(1)]], constant uint& n [[buffer(2)]],
    \\    uint i [[thread_position_in_grid]]) {
    \\  if (i < n) b[i] = a[i] + a[(i + 977) % n] * 0.5f;
    \\}
    \\inline void grid_sync(device atomic_uint* ctr, uint target, uint t) {
    \\  threadgroup_barrier(mem_flags::mem_device);
    \\  if (t == 0) {
    \\    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    \\    atomic_fetch_add_explicit(ctr, 1u, memory_order_relaxed);
    \\    while (atomic_load_explicit(ctr, memory_order_relaxed) < target) {}
    \\    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    \\  }
    \\  threadgroup_barrier(mem_flags::mem_device);
    \\}
    \\kernel void persistent(coherent(device) device float* a [[buffer(0)]], coherent(device) device float* b [[buffer(1)]],
    \\    constant uint& n [[buffer(2)]], constant uint& phases [[buffer(3)]], device atomic_uint* ctr [[buffer(4)]],
    \\    uint t [[thread_index_in_threadgroup]], uint g [[threadgroup_position_in_grid]], uint ng [[threadgroups_per_grid]],
    \\    uint tpg [[threads_per_threadgroup]]) {
    \\  for (uint p = 0; p < phases; p++) {
    \\    coherent(device) device float* src = (p & 1) ? b : a;
    \\    coherent(device) device float* dst = (p & 1) ? a : b;
    \\    for (uint i = g * tpg + t; i < n; i += ng * tpg) dst[i] = src[i] + src[(i + 977) % n] * 0.5f;
    \\    grid_sync(ctr, (p + 1) * ng, t);
    \\  }
    \\}
;

pub fn main() !void {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    const queue = try device.queue();
    const lib = try mtl.Library.fromSource(device, source, mtl.CompileOptions.mlx());
    const step = try mtl.Pipeline.init(device, lib, "step", false);
    const pers = try mtl.Pipeline.init(device, lib, "persistent", false);
    const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;
    const n: u32 = 65536;
    const a = try device.buffer(n * 4, opts);
    const b = try device.buffer(n * 4, opts);
    const ctr = try device.buffer(64, opts);
    const phases: u32 = 600;
    for (a.slice(f32, n), 0..) |*x, i| x.* = @floatFromInt(i % 7);
    // reference: N serial dispatches
    var best_serial: f64 = 1e9;
    for (0..3) |_| {
        const cb = queue.commandBuffer();
        const enc = cb.compute(.serial);
        enc.setPipeline(step);
        for (0..phases) |p| {
            enc.setBuffer(if (p % 2 == 0) a else b, 0, 0);
            enc.setBuffer(if (p % 2 == 0) b else a, 0, 1);
            enc.setBytes(std.mem.asBytes(&n), 2);
            enc.dispatchThreads(mtl.Size.of(n, 1, 1), mtl.Size.of(256, 1, 1));
        }
        enc.end();
        cb.commit();
        cb.wait();
        best_serial = @min(best_serial, cb.gpuSeconds());
    }
    const ref = a.slice(f32, n)[12345];
    for (a.slice(f32, n), 0..) |*x, i| x.* = @floatFromInt(i % 7);
    var best_pers: f64 = 1e9;
    var groups: usize = 16;
    while (groups <= 64) : (groups *= 2) {
        for (0..3) |_| {
            for (a.slice(f32, n), 0..) |*x, i| x.* = @floatFromInt(i % 7);
            @memset(ctr.contents()[0..64], 0);
            const cb = queue.commandBuffer();
            const enc = cb.compute(.serial);
            enc.setPipeline(pers);
            enc.setBuffer(a, 0, 0);
            enc.setBuffer(b, 0, 1);
            enc.setBytes(std.mem.asBytes(&n), 2);
            enc.setBytes(std.mem.asBytes(&phases), 3);
            enc.setBuffer(ctr, 0, 4);
            enc.dispatchThreads(mtl.Size.of(groups * 1024, 1, 1), mtl.Size.of(1024, 1, 1));
            enc.end();
            cb.commit();
            cb.wait();
            const ok = a.slice(f32, n)[12345] == ref;
            std.debug.print("persistent {d} groups: {d:.3} ms for {d} phases ({d:.2} us a phase), result {s}\n", .{ groups, cb.gpuSeconds() * 1e3, phases, cb.gpuSeconds() * 1e6 / @as(f64, @floatFromInt(phases)), if (ok) "equal" else "DIFFERENT" });
            best_pers = @min(best_pers, cb.gpuSeconds());
        }
    }
    std.debug.print("serial dispatches: {d:.3} ms for {d} phases ({d:.2} us a phase)\n", .{ best_serial * 1e3, phases, best_serial * 1e6 / @as(f64, @floatFromInt(phases)) });
}
