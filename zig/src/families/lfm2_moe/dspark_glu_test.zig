//! Consumer-visible BF16 activations, including the hold's scalar/MMA boundary and offset guards.
const std = @import("std");
const mtl = @import("metal");
const rig_ = @import("dspark_hold_test.zig");
const bf16_ops = @import("dspark_bf16.zig");
const quant = @import("dspark_quant.zig");

const Args = extern struct { rows: u32, n: u32, k: u32 };
const Ref = struct { buf: mtl.Buffer, off: usize };
const guard = 16;
const sentinel: u16 = 0x55aa;

fn projection(r: *rig_.Rig, e: mtl.ComputeEncoder, x: Ref, w: mtl.Buffer, out: mtl.Buffer, p: Args) void {
    e.setPipeline(if (p.rows <= 16) r.k.get("lfm2_bf16_rows") else r.k.get("lfm2_bf16_mma"));
    e.setBuffer(x.buf, x.off, 0);
    e.setBuffer(w, 0, 1);
    e.setValue(p, 2);
    e.setBuffer(out, 0, 3);
    if (p.rows <= 16) {
        e.dispatchGroups(mtl.Size.of((p.n + 7) / 8, (p.rows + 15) / 16, 1), mtl.Size.of(256, 1, 1));
    } else e.dispatchGroups(mtl.Size.of((p.n + 63) / 64, (p.rows + 63) / 64, 1), mtl.Size.of(128, 1, 1));
}

const Fill = enum { finite, extreme, rounding, nonfinite };

fn fill(mode: Fill, x: []u16, gate: []u16, up: []u16, n: usize, k: usize) void {
    if (mode == .finite or mode == .nonfinite) {
        rig_.noise(71, x, 2);
        rig_.noise(72, gate, 0.125);
        rig_.noise(73, up, 0.25);
        if (mode == .nonfinite) {
            x[0] = 0x7fc0;
            x[k] = 0x7f80;
            x[2 * k] = 0xff80;
        }
        return;
    }
    @memset(x, 0);
    @memset(gate, 0);
    @memset(up, 0);
    for (0..x.len / k) |row| {
        x[row * k] = quant.toBf16(1);
        x[row * k + 1] = quant.toBf16(1);
        x[row * k + 2] = quant.toBf16(@as(f32, @floatFromInt(row % 3)) - 1);
    }
    const extremes = [_]f32{ -120, -88, -16, -0.0001, 0, 0.0001, 16, 88, 120 };
    for (0..n) |col| {
        const sign: f32 = if (col % 2 == 0) 1 else -1;
        gate[col * k] = quant.toBf16(if (mode == .extreme) extremes[col % extremes.len] else sign);
        up[col * k] = quant.toBf16(if (mode == .extreme) sign * 1e10 else sign);
        if (mode == .rounding) {
            gate[col * k + 1] = quant.toBf16(sign * 0x1p-8);
            gate[col * k + 2] = quant.toBf16(sign * 0x1p-16);
            up[col * k + 1] = quant.toBf16(-sign * 0x1p-9);
            up[col * k + 2] = quant.toBf16(sign * 0x1p-17);
        }
    }
}

test "lfm2 dspark: fused BF16 gate/up gives exact SwiGLU bytes across 16 rows and preserves output guards" {
    var r = try rig_.Rig.init();
    defer r.deinit();
    const n = 72; // non-power-of-two output width
    const k = 512; // two scalar lane strides
    const max_rows = 16;
    const input = try r.buffer(u16, guard + max_rows * k + guard);
    defer input.deinit();
    const gate = try r.buffer(u16, n * k);
    defer gate.deinit();
    const up = try r.buffer(u16, n * k);
    defer up.deinit();
    const g = try r.buffer(u16, max_rows * n);
    defer g.deinit();
    const u = try r.buffer(u16, max_rows * n);
    defer u.deinit();
    const want = try r.buffer(u16, guard + max_rows * n + guard);
    defer want.deinit();
    const got = try r.buffer(u16, guard + max_rows * n + guard);
    defer got.deinit();
    const x = Ref{ .buf = input, .off = guard * 2 };
    for (std.enums.values(Fill)) |mode| {
        @memset(input.slice(u16, guard + max_rows * k + guard), sentinel);
        fill(mode, input.slice(u16, guard + max_rows * k)[guard..], gate.slice(u16, n * k), up.slice(u16, n * k), n, k);
        for ([_]u32{ 1, 4, 8, 16 }) |rows| {
            @memset(want.slice(u16, guard + max_rows * n + guard), sentinel);
            @memset(got.slice(u16, guard + max_rows * n + guard), sentinel);
            const cb = r.queue.commandBuffer();
            const e = cb.compute(.concurrent);
            const p = Args{ .rows = rows, .n = n, .k = k };
            projection(&r, e, x, gate, g, p);
            projection(&r, e, x, up, u, p);
            e.barrier();
            e.setPipeline(r.k.get("lfm2_swiglu"));
            e.setBuffer(g, 0, 0);
            e.setBuffer(u, 0, 1);
            e.setBuffer(want, guard * 2, 2);
            e.setValue(rows * n, 3);
            e.dispatchThreads(mtl.Size.of(rows * n / 8, 1, 1), mtl.Size.of(256, 1, 1));
            bf16_ops.glu(e, &r.k, .{ .w = gate, .n = n, .k = k }, .{ .w = up, .n = n, .k = k }, x, Ref{ .buf = got, .off = guard * 2 }, rows);
            e.barrier();
            e.end();
            cb.commit();
            cb.wait();
            if (cb.failure()) |_| return error.GpuFailed;
            const all = got.slice(u16, guard + max_rows * n + guard);
            for (want.slice(u16, all.len), all) |expected, actual| {
                if ((expected & 0x7f80) == 0x7f80) {
                    try std.testing.expect((actual & 0x7f80) == 0x7f80);
                } else try std.testing.expectEqual(expected, actual);
            }
            for (all[0..guard]) |v| try std.testing.expectEqual(sentinel, v);
            for (all[guard + rows * n ..]) |v| try std.testing.expectEqual(sentinel, v);
        }
    }
}
