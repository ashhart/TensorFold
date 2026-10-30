//! Flash Next's hyper-connection launches against the launches they replace, byte for byte.
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
