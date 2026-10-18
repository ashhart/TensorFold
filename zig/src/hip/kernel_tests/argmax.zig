//! The argmax of a wide row in parts against the one-block argmax: the same pick, ties, NaNs and every width.

const std = @import("std");
const rig = @import("rig.zig");
const DeviceBuffer = @import("../memory.zig").DeviceBuffer;

const Rig = rig.Rig;
const gpa = rig.gpa;
const at = rig.at;

fn ptr(b: DeviceBuffer) ?*anyopaque {
    return @ptrFromInt(at(b));
}

/// Logits of `kind` (1 fp16, 2 bf16): `levels` > 0 draws from that many values so many tie; `nans` puts a few NaNs in.
fn logits(kind: c_int, rng: *rig.Rng, n: usize, levels: u32, nans: bool) ![]u16 {
    const out = try gpa.alloc(u16, n);
    for (out, 0..) |*v, i| {
        const f: f32 = if (levels > 0) @floatFromInt(rng.next() % levels) else @floatCast(4 * rng.normal());
        v.* = if (kind == 1) rig.f16Bits(f) else rig.bf16Bits(f);
        if (nans and i % 7919 == 1234) v.* = if (kind == 1) 0x7e00 else 0x7fc0;
    }
    return out;
}

test "the argmax in parts picks the one-block argmax's id: random, tied, NaN and every width" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    var rng: rig.Rng = .{ .state = 0xA4A };
    var cases: usize = 0;
    for ([_]c_int{ 1, 2 }) |kind| for ([_]usize{ 1, 3, 4 }) |rows| for (
        [_]usize{ 2048, 49152, 131072, 131073, 5000 },
    ) |n| {
        for ([_]u32{ 0, 2, 50 }) |levels| for ([_]bool{ false, true }) |nans| {
            const host = try logits(kind, &rng, rows * n, levels, nans);
            defer gpa.free(host);
            var x = try t.upload(host);
            defer x.free();
            var outs: [2]DeviceBuffer = .{ try t.alloc(rows * 4), try t.alloc(rows * 4) };
            defer for (&outs) |*o| o.free();
            const parts = (n + 1023) / 1024;
            var v = try t.alloc(rows * parts * 4);
            defer v.free();
            var ix = try t.alloc(rows * parts * 4);
            defer ix.free();
            const s = t.stream.handle;
            try t.on.tf_argmax_rows(ptr(x), kind, @intCast(rows), @intCast(n), @ptrFromInt(at(outs[0])), s);
            try t.on.tf_argmax_parts(
                ptr(x),
                kind,
                @intCast(rows),
                @intCast(n),
                1024,
                @intCast(parts),
                ptr(v),
                @ptrFromInt(at(ix)),
                @ptrFromInt(at(outs[1])),
                s,
            );
            try t.stream.synchronize();
            const a = try rig.download(i32, outs[0]);
            defer gpa.free(a);
            const b = try rig.download(i32, outs[1]);
            defer gpa.free(b);
            try rig.expect(
                std.mem.eql(i32, a, b),
                "argmax kind {d}, {d} rows of {d}, {d} levels, NaNs {}: {any} against {any}",
                .{ kind, rows, n, levels, nans, b, a },
            );
            cases += 1;
        };
    };
    std.debug.print("  argmax in parts: {d} cases, the one-block argmax's ids\n", .{cases});
}
