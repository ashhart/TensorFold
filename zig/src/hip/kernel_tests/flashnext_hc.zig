//! Flash Next's merged hyper-connection and MoE-combine launches against the launches they replace, byte for byte.
const std = @import("std");
const rig = @import("rig.zig");
const fx = @import("flashnext.zig");
const Rig = rig.Rig;
const gpa = rig.gpa;
const at = rig.at;
const DeviceBuffer = @import("../memory.zig").DeviceBuffer;
const ptr = fx.ptr;
const acts = fx.acts;
const value = fx.value;

/// A random MLX 4-bit g32 matrix (n, k) with bf16 tables, on the device.
const Packed = struct {
    dev: [3]DeviceBuffer,

    fn init(t: *Rig, rng: *rig.Rng, n: usize, k: usize) !Packed {
        const words = try gpa.alloc(u32, n * k / 8);
        defer gpa.free(words);
        const scale = try gpa.alloc(u16, n * k / 32);
        defer gpa.free(scale);
        const bias = try gpa.alloc(u16, scale.len);
        defer gpa.free(bias);
        for (words) |*w| w.* = @truncate(rng.next() >> 16);
        for (scale) |*v| v.* = rig.bf16Bits((rng.unit() + 1.0) / 64.0);
        for (bias) |*v| v.* = rig.bf16Bits(-rng.unit() / 8.0);
        return .{ .dev = .{ try t.upload(words), try t.upload(scale), try t.upload(bias) } };
    }

    fn deinit(p: *Packed) void {
        for (&p.dev) |*b| b.free();
    }

    fn affine(p: Packed, n: usize, k: usize) @import("../ops/types.zig").Affine {
        return .{ .words = at(p.dev[0]), .scale = at(p.dev[1]), .bias = at(p.dev[2]), .tables = .bf16, .n = @intCast(n), .k = @intCast(k), .bits = 4, .group = 32 };
    }
};

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

test "hyper-connection gates, up and mix in one launch: the chain's values, each row alone bit for bit, and its time" {
    const t = try Rig.open(64 << 20);
    defer t.close();
    if (!t.bf16) return; // compiled for bf16 activations
    var rng: rig.Rng = .{ .state = 0x4C3 };
    const rows = 11; // a block of 8 rows and one of 3
    const s = 4;
    const d = 2560;
    const w = s * d;
    const low = 320;
    var up = try Packed.init(t, &rng, w, low);
    defer up.deinit();
    const hx = try acts(t, &rng, rows * w, 1.5);
    defer gpa.free(hx);
    var x = try t.upload(hx);
    defer x.free();
    const k = t.kind();
    const st = t.stream.handle;
    for ([_]bool{ true, false }) |with_inject| {
        const n: usize = if (with_inject) low + s else low;
        const zh = try gpa.alloc(f32, rows * n);
        defer gpa.free(zh);
        for (zh) |*v| v.* = rng.unit() * 6.0;
        var z = try t.upload(zh);
        defer z.free();
        // the chain: gates, the up product (fp32), the mix
        var act = try t.alloc(rows * low * 2);
        defer act.free();
        var inj = [_]DeviceBuffer{ try t.alloc(rows * s * 2), try t.alloc(rows * s * 2), try t.alloc(rows * s * 2) };
        defer for (&inj) |*b| b.free();
        var mixed = [_]DeviceBuffer{ try t.alloc(rows * d * 2), try t.alloc(rows * d * 2), try t.alloc(rows * d * 2) };
        defer for (&mixed) |*b| b.free();
        t.arena.reset();
        try t.on.fn_hc_gates(ptr(?[*]const f32, z), ptr(?*anyopaque, act), if (with_inject) ptr(?*anyopaque, inj[0]) else null, k, rows, low, s, st);
        const g = try t.ops(&t.on).affine(.{ .ptr = at(act), .kind = .bf16 }, up.affine(w, low), rows, true);
        try t.on.fn_hc_mix(@ptrFromInt(g.ptr), 0, ptr(?*const anyopaque, x), ptr(?*anyopaque, mixed[0]), k, rows, s, d, st);
        // the mix of g first rounded to the activation kind (tf_cast, as the dense tile's bf16 output): the same bytes
        {
            var g16 = try t.alloc(rows * s * d * 2);
            defer g16.free();
            var mixed16 = try t.alloc(rows * d * 2);
            defer mixed16.free();
            try t.on.tf_cast(@ptrFromInt(g.ptr), 0, ptr(?*anyopaque, g16), k, @intCast(rows * s * d), st);
            try t.on.fn_hc_mix(ptr(?*const anyopaque, g16), k, ptr(?*const anyopaque, x), ptr(?*anyopaque, mixed16), k, rows, s, d, st);
            try t.stream.synchronize();
            const want = try rig.download(u16, mixed[0]);
            defer gpa.free(want);
            const got16 = try rig.download(u16, mixed16);
            defer gpa.free(got16);
            try std.testing.expectEqualSlices(u16, want, got16);
        }
        // one launch for all rows, then one a row
        const one = struct {
            fn go(tt: *Rig, nn: usize, zz: DeviceBuffer, u: Packed, xx: DeviceBuffer, mm: DeviceBuffer, ii: DeviceBuffer, r0: usize, m: usize, wi: bool) !void {
                try tt.on.fn_hc_up_mix(
                    @ptrFromInt(at(zz) + r0 * nn * 4),
                    @intCast(nn),
                    @ptrFromInt(at(u.dev[0])),
                    @ptrFromInt(at(u.dev[1])),
                    @ptrFromInt(at(u.dev[2])),
                    @ptrFromInt(at(xx) + r0 * 10240 * 2),
                    @ptrFromInt(at(mm) + r0 * 2560 * 2),
                    if (wi) @ptrFromInt(at(ii) + r0 * 4 * 2) else null,
                    @intCast(m),
                    tt.stream.handle,
                );
            }
        }.go;
        try one(t, n, z, up, x, mixed[1], inj[1], 0, rows, with_inject);
        for (0..rows) |r| try one(t, n, z, up, x, mixed[2], inj[2], r, 1, with_inject);
        try t.stream.synchronize();
        var got: [3][2][]u16 = undefined;
        for (&got, mixed, inj) |*gg, mb, ib| gg.* = .{ try rig.download(u16, mb), try rig.download(u16, ib) };
        defer for (got) |gg| for (gg) |v| gpa.free(v);
        try std.testing.expectEqualSlices(u16, got[1][0], got[2][0]);
        if (with_inject) {
            try std.testing.expectEqualSlices(u16, got[1][1], got[2][1]);
            try std.testing.expectEqualSlices(u16, got[0][1], got[1][1]); // the gates: the same roundings
        }
        // a mixed value is a mean of four terms of about unit size: within 4 bf16 steps at that scale
        var differ: usize = 0;
        for (got[0][0], got[1][0], 0..) |a, b, i| {
            const want = value(t, a);
            differ += @intFromBool(a != b);
            try rig.expect(@abs(value(t, b) - want) <= 4.0 / 128.0 * @max(@abs(want), 0.5), "hc up mix {d}: {d} vs {d}", .{ i, value(t, b), want });
        }
        // back to back: the one launch against gates, product and mix
        var us: [2]f64 = undefined;
        for (0..2) |which| {
            const reps = 200;
            var t0: u64 = 0;
            for (0..reps + 10) |i| {
                if (i == 10) {
                    try t.stream.synchronize();
                    t0 = nowNs();
                }
                if (which == 0) try one(t, n, z, up, x, mixed[1], inj[1], 0, 1, with_inject) else {
                    t.arena.reset();
                    try t.on.fn_hc_gates(ptr(?[*]const f32, z), ptr(?*anyopaque, act), if (with_inject) ptr(?*anyopaque, inj[0]) else null, k, 1, low, s, st);
                    const g1 = try t.ops(&t.on).affine(.{ .ptr = at(act), .kind = .bf16 }, up.affine(w, low), 1, true);
                    try t.on.fn_hc_mix(@ptrFromInt(g1.ptr), 0, ptr(?*const anyopaque, x), ptr(?*anyopaque, mixed[0]), k, 1, s, d, st);
                }
            }
            try t.stream.synchronize();
            us[which] = @as(f64, @floatFromInt(nowNs() - t0)) / reps / 1e3;
        }
        std.debug.print("  hc up mix (inject {}): {d} of {d} mixed values differ from the chain's; 1 row {d:.1} us, the chain's three launches {d:.1} us (back to back)\n", .{ with_inject, differ, got[0][0].len, us[0], us[1] });
    }
}

test "the MoE combine inside the HC norm: moe_combine_kernel's then tf_fn_hc_norm's bytes, 1 to 8 rows" {
    const t = try Rig.open(64 << 20);
    defer t.close();
    var rng: rig.Rng = .{ .state = 0x4C7 };
    const s = 4;
    const d = 2560;
    const slots = 11; // Flash Next: 10 routed experts and the shared one
    const k = t.kind();
    const st = t.stream.handle;
    for ([_]usize{ 1, 3, 4, 8 }) |rows| {
        const yh = try gpa.alloc(f32, rows * slots * d);
        defer gpa.free(yh);
        for (yh) |*v| v.* = rng.unit() * 2.0;
        const wh = try gpa.alloc(f32, rows * slots);
        defer gpa.free(wh);
        for (wh) |*v| v.* = rng.unit() * 0.5;
        const hh = try acts(t, &rng, rows * s * d, 3.0);
        defer gpa.free(hh);
        const ih = try acts(t, &rng, rows * s, 2.0);
        defer gpa.free(ih);
        const sh = try gpa.alloc(f32, s * d);
        defer gpa.free(sh);
        for (sh) |*v| v.* = 1.0 + rng.unit() * 0.25;
        var y = try t.upload(yh);
        defer y.free();
        var w = try t.upload(wh);
        defer w.free();
        var inject = try t.upload(ih);
        defer inject.free();
        var scale = try t.upload(sh);
        defer scale.free();
        var h = [_]DeviceBuffer{ try t.upload(hh), try t.upload(hh) };
        defer for (&h) |*b| b.free();
        var x = [_]DeviceBuffer{ try t.alloc(rows * s * d * 2), try t.alloc(rows * s * d * 2) };
        defer for (&x) |*b| b.free();
        var branch = try t.alloc(rows * d * 2);
        defer branch.free();
        // the chain: the combine into the activation kind, then the norm with it as the branch
        try t.ops(&t.on).moeCombine(at(y), at(w), .{ .ptr = at(branch), .kind = if (k == 2) .bf16 else .f16 }, rows, slots, d);
        try t.on.fn_hc_norm(@ptrFromInt(at(h[0])), @ptrFromInt(at(branch)), @ptrFromInt(at(inject)), @ptrFromInt(at(h[0])), @ptrFromInt(at(x[0])), @ptrFromInt(at(scale)), k, @intCast(rows), s, d, 1e-6, st);
        try t.on.fn_hc_norm_moe(@ptrFromInt(at(h[1])), @ptrFromInt(at(y)), @ptrFromInt(at(w)), @ptrFromInt(at(inject)), @ptrFromInt(at(h[1])), @ptrFromInt(at(x[1])), @ptrFromInt(at(scale)), k, @intCast(rows), s, d, 1e-6, slots, st);
        try t.stream.synchronize();
        for ([_][2]DeviceBuffer{ h, x }) |pair| {
            const a = try rig.download(u16, pair[0]);
            defer gpa.free(a);
            const b = try rig.download(u16, pair[1]);
            defer gpa.free(b);
            try std.testing.expectEqualSlices(u16, a, b);
        }
    }
}

test "a prompt's MoE combine inside the HC norm, a block a row: moe_combine_kernel's then tf_fn_hc_norm_v's bytes" {
    const t = try Rig.open(256 << 20);
    defer t.close();
    if (!t.bf16) return; // the 16-byte kernels are bf16 only
    var rng: rig.Rng = .{ .state = 0x4D1 };
    const s = 4;
    const d = 2560;
    const slots = 11; // Flash Next: 10 routed experts and the shared one
    const st = t.stream.handle;
    for ([_]usize{ 1, 7, 64, 513 }) |rows| {
        const yh = try gpa.alloc(f32, rows * slots * d);
        defer gpa.free(yh);
        for (yh) |*v| v.* = rng.unit() * 2.0;
        // a few slot values huge, tiny or zero: the sums' rounding and the norm's range, not only the middle
        for (0..rows) |r| yh[(r * slots + r % slots) * d + r % d] = if (r % 3 == 0) 3.0e4 else if (r % 3 == 1) 1.0e-30 else 0.0;
        const wh = try gpa.alloc(f32, rows * slots);
        defer gpa.free(wh);
        for (wh) |*v| v.* = rng.unit() * 0.5;
        const hh = try acts(t, &rng, rows * s * d, 3.0);
        defer gpa.free(hh);
        const ih = try acts(t, &rng, rows * s, 2.0);
        defer gpa.free(ih);
        const sh = try gpa.alloc(f32, s * d);
        defer gpa.free(sh);
        for (sh) |*v| v.* = 1.0 + rng.unit() * 0.25;
        var y = try t.upload(yh);
        defer y.free();
        var w = try t.upload(wh);
        defer w.free();
        var inject = try t.upload(ih);
        defer inject.free();
        var scale = try t.upload(sh);
        defer scale.free();
        var h = [_]DeviceBuffer{ try t.upload(hh), try t.upload(hh) };
        defer for (&h) |*b| b.free();
        var x = [_]DeviceBuffer{ try t.alloc(rows * s * d * 2), try t.alloc(rows * s * d * 2) };
        defer for (&x) |*b| b.free();
        var branch = try t.alloc(rows * d * 2);
        defer branch.free();
        // the chain: the combine into bf16, then the norm (tf_fn_hc_norm_v at bf16 2,560) with it as the branch
        try t.ops(&t.on).moeCombine(at(y), at(w), .{ .ptr = at(branch), .kind = .bf16 }, rows, slots, d);
        try t.on.fn_hc_norm(@ptrFromInt(at(h[0])), @ptrFromInt(at(branch)), @ptrFromInt(at(inject)), @ptrFromInt(at(h[0])), @ptrFromInt(at(x[0])), @ptrFromInt(at(scale)), 2, @intCast(rows), s, d, 1e-6, st);
        try t.on.fn_hc_norm_moe_v(@ptrFromInt(at(h[1])), @ptrFromInt(at(y)), @ptrFromInt(at(w)), @ptrFromInt(at(inject)), @ptrFromInt(at(h[1])), @ptrFromInt(at(x[1])), @ptrFromInt(at(scale)), @intCast(rows), s, 1e-6, slots, st);
        try t.stream.synchronize();
        for ([_][2]DeviceBuffer{ h, x }) |pair| {
            const a = try rig.download(u16, pair[0]);
            defer gpa.free(a);
            const b = try rig.download(u16, pair[1]);
            defer gpa.free(b);
            try std.testing.expectEqualSlices(u16, a, b);
        }
    }
}

test "hyper-connection norm and mix, 16-byte kernels: the per-element kernels' bytes at 1 to 300 rows" {
    const t = try Rig.open(64 << 20);
    defer t.close();
    if (!t.bf16) return; // the 16-byte kernels are bf16 only
    var rng: rig.Rng = .{ .state = 0x4CA };
    const s = 4;
    const d = 2560;
    const w = s * d;
    const k = t.kind();
    const st = t.stream.handle;
    const scale = try gpa.alloc(f32, w);
    defer gpa.free(scale);
    for (scale) |*x| x.* = 1.0 + rng.unit() * 0.5;
    var sc = try t.upload(scale);
    defer sc.free();
    for ([_]usize{ 1, 3, 4, 8, 300 }) |rows| {
        const hh = try acts(t, &rng, rows * w, 2.0);
        defer gpa.free(hh);
        const hb = try acts(t, &rng, rows * d, 1.0);
        defer gpa.free(hb);
        const hi = try acts(t, &rng, rows * s, 1.0);
        defer gpa.free(hi);
        // the branch as one fp32 share holds the same values: the shares path is the 1,024-thread kernel
        const fb = try gpa.alloc(f32, rows * d);
        defer gpa.free(fb);
        for (fb, hb) |*f, b| f.* = value(t, b);
        var h = try t.upload(hh);
        defer h.free();
        var branch = try t.upload(hb);
        defer branch.free();
        var share = try t.upload(fb);
        defer share.free();
        var inject = try t.upload(hi);
        defer inject.free();
        for ([_]bool{ true, false }) |with_branch| {
            var outs = [_]DeviceBuffer{ try t.alloc(rows * w * 2), try t.alloc(rows * w * 2), try t.alloc(rows * w * 2), try t.alloc(rows * w * 2) };
            defer for (&outs) |*b| b.free();
            try t.on.fn_hc_norm(ptr(?*const anyopaque, h), if (with_branch) ptr(?*const anyopaque, branch) else null, ptr(?*const anyopaque, inject), ptr(?*anyopaque, outs[0]), ptr(?*anyopaque, outs[1]), ptr(?[*]const f32, sc), k, @intCast(rows), s, d, 1e-6, st);
            try t.on.fn_hc_norm_shares(ptr(?*const anyopaque, h), if (with_branch) ptr(?*const anyopaque, share) else null, ptr(?*const anyopaque, inject), ptr(?*anyopaque, outs[2]), ptr(?*anyopaque, outs[3]), ptr(?[*]const f32, sc), k, @intCast(rows), s, d, 1e-6, 1, 0, st);
            try t.stream.synchronize();
            for ([_][2]usize{ .{ 0, 2 }, .{ 1, 3 } }) |pair| {
                const a = try rig.download(u16, outs[pair[0]]);
                defer gpa.free(a);
                const b = try rig.download(u16, outs[pair[1]]);
                defer gpa.free(b);
                try std.testing.expectEqualSlices(u16, b, a);
            }
        }
        // the mix over fp32 and bf16 g against the per-element kernel
        const gh = try gpa.alloc(f32, rows * w);
        defer gpa.free(gh);
        for (gh) |*x| x.* = rng.unit() * 6.0;
        const g16 = try acts(t, &rng, rows * w, 6.0);
        defer gpa.free(g16);
        var g32 = try t.upload(gh);
        defer g32.free();
        var gb = try t.upload(g16);
        defer gb.free();
        var x = try t.upload(hh);
        defer x.free();
        for ([_]c_int{ 0, k }) |gkind| {
            var mixed = [_]DeviceBuffer{ try t.alloc(rows * d * 2), try t.alloc(rows * d * 2) };
            defer for (&mixed) |*b| b.free();
            const gp = if (gkind == 0) ptr(?*const anyopaque, g32) else ptr(?*const anyopaque, gb);
            try t.on.fn_hc_mix(gp, gkind, ptr(?*const anyopaque, x), ptr(?*anyopaque, mixed[0]), k, @intCast(rows), s, d, st);
            try t.on.fn_hc_mix_scalar(gp, gkind, ptr(?*const anyopaque, x), ptr(?*anyopaque, mixed[1]), k, @intCast(rows), s, d, st);
            try t.stream.synchronize();
            const a = try rig.download(u16, mixed[0]);
            defer gpa.free(a);
            const b = try rig.download(u16, mixed[1]);
            defer gpa.free(b);
            try std.testing.expectEqualSlices(u16, b, a);
        }
    }
}

test "hc up with the HC mix in the dense tile's epilogue: the dense product's bf16 g then tf_fn_hc_mix, byte for byte" {
    const t = try Rig.open(64 << 20);
    defer t.close();
    if (!t.bf16) return;
    var o = t.ops(&t.on);
    o.prefill = true;
    var rng: rig.Rng = .{ .state = 0x4CB };
    const s = 4;
    const d = 2560;
    const w = s * d;
    const low = 320;
    var up = try Packed.init(t, &rng, w, low);
    defer up.deinit();
    var ran: usize = 0;
    for ([_]usize{ 129, 300, 1000 }) |rows| {
        t.arena.reset();
        const ah = try acts(t, &rng, rows * low, 1.0);
        defer gpa.free(ah);
        const xh = try acts(t, &rng, rows * w, 2.0);
        defer gpa.free(xh);
        var act = try t.upload(ah);
        defer act.free();
        var x = try t.upload(xh);
        defer x.free();
        const at_act: @import("../ops/types.zig").Tensor = .{ .ptr = at(act), .kind = .bf16 };
        const at_x: @import("../ops/types.zig").Tensor = .{ .ptr = at(x), .kind = .bf16 };
        // null: the registry does not take the dense tile at this row count
        const fused = (try o.hcUpMixAffine(at_act, up.affine(w, low), at_x, rows)) orelse continue;
        ran += 1;
        const g = try o.affine(at_act, up.affine(w, low), rows, false);
        try std.testing.expectEqual(@import("../ops/types.zig").Kind.bf16, g.kind);
        var want = try t.alloc(rows * d * 2);
        defer want.free();
        try t.on.fn_hc_mix(@ptrFromInt(g.ptr), 2, ptr(?*const anyopaque, x), ptr(?*anyopaque, want), t.kind(), @intCast(rows), s, d, t.stream.handle);
        try t.stream.synchronize();
        const a = try rig.download(u16, want);
        defer gpa.free(a);
        const b = try gpa.alloc(u16, rows * d);
        defer gpa.free(b);
        try @import("../runtime.zig").check(t.r.api.hipMemcpyDtoH(std.mem.sliceAsBytes(b).ptr, @ptrFromInt(fused.ptr), b.len * 2));
        try std.testing.expectEqualSlices(u16, a, b);
    }
    try std.testing.expect(ran > 0);
}
