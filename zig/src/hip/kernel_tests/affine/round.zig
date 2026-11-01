//! A prompt tile given out16 writes its sum there rounded as tf_cast rounds the fp32 one, and leaves out untouched.

const std = @import("std");
const base = @import("../rig.zig");
const Buffer = base.Buffer;
const cases = @import("cases.zig");
const prod = @import("product.zig");
const Product = prod.Product;
const Rig = prod.Rig;

fn dense(name: []const u8, rows: usize, group: u16, bits: u8) cases.Case {
    return .{ .name = name, .path = .prefill, .rows = rows, .n = 1001, .k = 2112, .group = group, .bits = bits };
}

/// Row counts on every prompt tile (the K-parallel ones, then the 128-row tiles and their edges), groups and widths.
const shapes = [_]cases.Case{
    dense("1 row", 1, 64, 4),      dense("2 rows", 2, 64, 4),       dense("3 rows", 3, 64, 4),
    dense("4 rows", 4, 64, 4),     dense("17 rows", 17, 64, 4),     dense("32 rows", 32, 64, 4),
    dense("33 rows", 33, 64, 4),   dense("100 rows", 100, 64, 4),   dense("300 rows", 300, 64, 4),
    dense("group 32", 100, 32, 4), dense("group 128", 100, 128, 4), dense("8-bit", 100, 64, 8),
};

fn digest(buf: Buffer, len: usize) !u64 {
    return prod.digestOf(buf, len);
}

test "a prompt entry with out16 set writes tf_cast's bytes of its fp32 sum and leaves out untouched" {
    const b = try base.Rig.open(1 << 20);
    defer b.close();
    var rig = Rig.open(b);
    var checked: usize = 0;
    for (shapes) |c| {
        var p = try Product.init(&rig, c);
        defer p.deinit();
        const n = p.out_rows * c.n;
        var wide = try Buffer.alloc(rig.base, n * 4);
        defer wide.free();
        var want = try Buffer.alloc(rig.base, n * 2);
        defer want.free();
        var got = try Buffer.alloc(rig.base, n * 2);
        defer got.free();
        var spare = try Buffer.alloc(rig.base, n * 4);
        defer spare.free();
        try spare.fill8(0xA5, null);
        const untouched = try digest(spare, n * 4);
        for (rig.kernels.reg.entries) |*e| {
            if (e.op != .project or e.path != .prefill or !prod.takes(&p, e)) continue;
            try wide.fill8(0xA5, null);
            try prod.launchEntry(&p, e, wide, false, 0, 0);
            try rig.base.on.tf_cast(@ptrFromInt(wide.ptr), 0, @ptrFromInt(want.ptr), rig.base.kind(), @intCast(n), rig.base.stream.handle);
            try got.fill8(0x5A, null);
            var a = p.arg(spare.ptr, 0);
            a.out16 = got.ptr;
            try e.launch(rig.kernels, .{ .r = &rig.base.r, .s = rig.base.stream.handle, .arg = a, .items = @intCast(p.items), .partial = 0, .parts = 1 });
            try rig.base.stream.synchronize();
            try base.expect(try digest(got, n * 2) == try digest(want, n * 2), "round {s}: {s} with out16 writes other bytes than tf_cast of its fp32 sum", .{ c.name, e.id });
            try base.expect(try digest(spare, n * 4) == untouched, "round {s}: {s} with out16 also writes out", .{ c.name, e.id });
            checked += 1;
        }
    }
    std.debug.print("ROUND {d} prompt launches write tf_cast's bytes into out16\n", .{checked});
    try base.expect(checked >= shapes.len, "round: only {d} prompt launches ran", .{checked});
}
