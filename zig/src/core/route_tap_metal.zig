//! Metal capture: blit each layer's picks into a shared-storage ring; the host reads it at the existing wait.
const std = @import("std");
const tap = @import("route_tap.zig");
const mtl = @import("metal");

const copy_src =
    \\kernel void tf_route_tap_copy(const device uint* src [[buffer(0)]], device uint* dst [[buffer(1)]],
    \\                              constant uint& n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    \\  if (i < n) dst[i] = src[i];
    \\}
;

var ring: ?mtl.Buffer = null;
var copy: ?mtl.Pipeline = null;
var lib: ?mtl.Library = null;

/// Shared-storage ring the size of the host tap ring (ids then weights). No-op when the tap is off.
pub fn attach(device: mtl.Device) void {
    if (!tap.live() or ring != null) return;
    const bytes = tap.ring_slots * tap.max_ids * 8;
    ring = device.buffer(bytes, mtl.ResourceOptions.shared) catch return;
    const compiled = mtl.Library.fromSource(device, copy_src, mtl.CompileOptions.mlx()) catch {
        if (ring) |b| b.deinit();
        ring = null;
        return;
    };
    lib = compiled;
    copy = mtl.Pipeline.init(device, compiled, "tf_route_tap_copy", false) catch {
        compiled.deinit();
        lib = null;
        if (ring) |b| b.deinit();
        ring = null;
        return;
    };
}

pub fn detach() void {
    if (copy) |p| p.deinit();
    copy = null;
    if (lib) |l| l.deinit();
    lib = null;
    if (ring) |b| b.deinit();
    ring = null;
}

fn dest() ?mtl.Buffer {
    return ring;
}

pub fn live() bool {
    return tap.live();
}

/// Copy `n` u32 picks (and the same count of fp32 weights) into the ring inside this command buffer.
pub fn blit(enc: mtl.ComputeEncoder, pipe: mtl.Pipeline, pick: mtl.Buffer, pick_off: usize, wts: mtl.Buffer, wts_off: usize, n: u32) void {
    if (!tap.live()) return;
    copyInto(enc, pipe, pick, pick_off, wts, wts_off, n);
}

/// Same as `blit` using the tap's own copy kernel (families that have no u32 copy pipeline).
pub fn blitPicks(enc: mtl.ComputeEncoder, pick: mtl.Buffer, pick_off: usize, wts: mtl.Buffer, wts_off: usize, n: u32) void {
    if (!tap.live()) return;
    const pipe = copy orelse return;
    copyInto(enc, pipe, pick, pick_off, wts, wts_off, n);
}

fn copyInto(enc: mtl.ComputeEncoder, pipe: mtl.Pipeline, pick: mtl.Buffer, pick_off: usize, wts: mtl.Buffer, wts_off: usize, n: u32) void {
    const dest_buf = dest() orelse return;
    if (n == 0) return;
    const sl = tap.reserve(@intCast(n)) orelse return;
    const off = tap.sharedOff(sl.index);
    enc.setPipeline(pipe);
    enc.setBuffer(pick, pick_off, 0);
    enc.setBuffer(dest_buf, off, 1);
    enc.setValue(n, 2);
    enc.dispatchThreads(mtl.Size.of(n, 1, 1), mtl.Size.of(@min(n, 256), 1, 1));
    enc.setPipeline(pipe);
    enc.setBuffer(wts, wts_off, 0);
    enc.setBuffer(dest_buf, off + tap.ring_slots * tap.max_ids * 4, 1);
    enc.setValue(n, 2);
    enc.dispatchThreads(mtl.Size.of(n, 1, 1), mtl.Size.of(@min(n, 256), 1, 1));
}

/// After the command buffer wait the step already does: read the shared ring, then publish.
pub fn afterWait() void {
    afterLand(tap.reservedCount());
}

/// Same as `afterWait`, but only slots reserved by this command buffer (pipelined submits).
pub fn afterLand(to: u32) void {
    const dest_buf = dest() orelse {
        tap.publishTo(to);
        return;
    };
    const words = tap.ring_slots * tap.max_ids;
    const bytes = dest_buf.contents()[0 .. words * 8];
    const ids = std.mem.bytesAsSlice(i32, bytes[0 .. words * 4]);
    const wts = std.mem.bytesAsSlice(f32, bytes[words * 4 ..]);
    tap.absorbSharedTo(ids, wts, to);
    tap.publishTo(to);
}
