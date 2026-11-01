//! The matrix tile's banded block order keeps each tile's bits: its bytes equal the dot2 GEMM tile's on served shapes.

const std = @import("std");
const base = @import("../rig.zig");
const Buffer = base.Buffer;
const cases = @import("cases.zig");
const prod = @import("product.zig");
const Product = prod.Product;
const Rig = prod.Rig;
const Entry = @import("../../launches/affine.zig").Entry;
const Case = cases.Case;

const routed: Case = .{
    .name = "35b routed gate_up",
    .path = .prefill,
    .rows = 2048,
    .n = 1024,
    .k = 2048,
    .experts = 256,
    .slots = 8,
    .item_rows = 128,
};

fn dense(name: []const u8, rows: usize, n: usize, k: usize) Case {
    return .{ .name = name, .path = .prefill, .rows = rows, .n = n, .k = k };
}

/// The 27B's prompt projections at 2048 rows, then rows off the 128 tile and the band, and one row or column block.
const shapes = [_]Case{
    dense("27b gate_up", 2048, 17408, 5120),
    dense("27b down", 2048, 5120, 17408),
    dense("27b linear qkv", 2048, 10240, 5120),
    dense("27b z", 2048, 6144, 5120),
    dense("27b out_proj", 2048, 5120, 6144),
    dense("27b q gate", 2048, 12288, 5120),
    dense("27b kv", 2048, 1024, 5120),
    dense("27b b a", 2048, 48, 5120),
    dense("edge rows", 1000, 1024, 5120),
    dense("edge band", 1300, 6144, 5120),
    dense("edge both", 300, 1001, 2112),
    dense("edge column", 2048, 130, 2048),
    dense("one row block", 100, 1001, 2112),
    dense("one column block", 300, 48, 2112),
    routed,
    .{ .name = "8-bit", .path = .prefill, .rows = 1300, .n = 4096, .k = 2048, .bits = 8 },
};

fn find(p: *const Product, id: []const u8) ?*const Entry {
    for (p.rig.kernels.reg.entries) |*e| if (std.mem.endsWith(u8, e.id, id) and prod.takes(p, e)) return e;
    return null;
}

fn digest(p: *const Product, e: *const Entry, out: Buffer) !u64 {
    try out.fill8(0xA5, null);
    try prod.launchEntry(p, e, out, false, 0, 0);
    try p.rig.base.stream.synchronize();
    return prod.digestOf(out, p.outBytes(false));
}

test "the matrix tile's banded order writes the dot2 GEMM tile's bytes on the 27B's prompt shapes and their edges" {
    const b = try base.Rig.open(1 << 20);
    defer b.close();
    var rig = Rig.open(b);
    if (!rig.kernels.env().matrix) return error.SkipZigTest;
    for (shapes) |c| {
        var p = try Product.init(&rig, c);
        defer p.deinit();
        const matrix = find(&p, "matrix") orelse return error.TestUnexpectedResult;
        const gemm = find(&p, "gemm") orelse return error.TestUnexpectedResult;
        var out = try Buffer.alloc(rig.base, p.outBytes(false));
        defer out.free();
        const want = try digest(&p, gemm, out);
        const got = try digest(&p, matrix, out);
        std.debug.print("ORDER {s} m{d} n{d} k{d}: {x:0>16} {x:0>16}\n", .{ c.name, c.rows, c.n, c.k, got, want });
        try base.expect(got == want, "order {s}: the matrix tile writes other bytes than the dot2 tile", .{c.name});
    }
}
