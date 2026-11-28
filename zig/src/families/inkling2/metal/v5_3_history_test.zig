//! Byte checks for poisoned compact taps and each committed convolution role.
const std = @import("std");
const mtl = @import("metal");
const st = @import("state.zig");

pub fn poison(queue: mtl.Queue, taps: mtl.Buffer) !void {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const cb = queue.commandBuffer();
    const e = cb.blit();
    e.fill(taps, 0, st.taps_bytes, 0xa5);
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure() != null) return error.GpuFailed;
}

pub fn expectPending(queue: mtl.Queue, full: mtl.Buffer, compact: mtl.Buffer, a: mtl.Buffer, b: mtl.Buffer, rows: usize) !void {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const cb = queue.commandBuffer();
    const e = cb.blit();
    for (0..st.visit_slots) |slot| for (0..st.conv_states) |cs| {
        const from = st.convByteOffset(slot, cs, 3, 3 + st.max_rows);
        const to = st.convByteOffset(slot, cs, 0, st.window_rows);
        e.copy(full, from, a, to, rows * st.convWidth(cs) * 2);
    };
    e.copy(compact, 0, b, 0, st.taps_bytes);
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure() != null) return error.GpuFailed;
    for (0..st.visit_slots) |slot| for (0..st.conv_states) |cs| {
        const off = st.convByteOffset(slot, cs, 0, st.window_rows);
        const bytes = rows * st.convWidth(cs) * 2;
        try std.testing.expectEqualSlices(u8, a.contents()[off..][0..bytes], b.contents()[off..][0..bytes]);
    };
}

pub fn expectCommitted(a: mtl.Buffer, b: mtl.Buffer) !void {
    for (0..st.visit_slots) |slot| for (0..st.conv_states) |cs| {
        const off = st.convByteOffset(slot, cs, 0, 3);
        const bytes = 3 * st.convWidth(cs) * 2;
        try std.testing.expectEqualSlices(u8, a.contents()[off..][0..bytes], b.contents()[off..][0..bytes]);
    };
}
