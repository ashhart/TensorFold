//! The 64-row routed prompt tile's K-parallel passes write the matrix path's bytes for an item of few rows, at any cut.

const std = @import("std");
const base = @import("../rig.zig");
const Buffer = base.Buffer;
const gpa = base.gpa;
const cases = @import("cases.zig");
const prod = @import("product.zig");
const Product = prod.Product;
const Rig = prod.Rig;
const Entry = @import("../../launches/affine.zig").Entry;
const item = @import("../../launches/matrix_item.zig");

/// Experts on both sides of every cut, of a row tile, of a block and of an item, then three of few rows at the end.
const loads = [_]u16{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 17, 63, 64, 65, 128, 129, 200, 5, 8, 3 };
const pairs = blk: {
    var n: usize = 0;
    for (loads) |l| n += l;
    break :blk n;
};

/// The 35B's gate and up, its down, and a ragged product: a 72-column last block and a last round of one group.
const shapes = [_]cases.Case{
    .{ .name = "35b gate_up", .path = .prefill, .rows = pairs, .n = 1024, .k = 2048, .experts = loads.len, .slots = 1, .item_rows = 128, .loads = &loads },
    .{ .name = "35b down", .path = .prefill, .rows = pairs, .n = 2048, .k = 512, .experts = loads.len, .slots = 1, .item_rows = 128, .loads = &loads, .down = true },
    .{ .name = "ragged", .path = .prefill, .rows = pairs, .n = 200, .k = 2080, .group = 32, .experts = loads.len, .slots = 1, .item_rows = 128, .loads = &loads },
};

/// What a run of outputs holds when no launch wrote it.
const unwritten: [16]u8 = @splat(0xa5);

/// The tile's output on `p` with items of at most `few` rows on the K-parallel passes.
fn run(p: *const Product, e: *const Entry, out: Buffer, few: c_int) ![]u8 {
    var a = p.arg(out.ptr, 0);
    a.route.few = few;
    try out.fill8(unwritten[0], null);
    try e.launch(p.rig.kernels, .{ .r = &p.rig.base.r, .s = p.rig.base.stream.handle, .arg = a, .items = @intCast(p.items) });
    try p.rig.base.stream.synchronize();
    const bytes = try gpa.alloc(u8, p.outBytes(false));
    errdefer gpa.free(bytes);
    try out.download(0, bytes);
    return bytes;
}

test "the routed item tile writes the matrix path's bytes through its K-parallel passes, at every cut and width" {
    const b = try base.Rig.open(1 << 20);
    defer b.close();
    var rig = Rig.open(b);
    if (!rig.kernels.env().matrix) return error.SkipZigTest;
    for (cases.widths) |bits| for (shapes) |shape| {
        var c = shape;
        c.bits = bits;
        var p = try Product.init(&rig, c);
        defer p.deinit();
        const e = for (rig.kernels.reg.entries) |*x| {
            if (std.mem.endsWith(u8, x.id, ".routed.prefill.item64") and prod.takes(&p, x)) break x;
        } else return error.NoItemTile;
        var out = try Buffer.alloc(rig.base, p.outBytes(false));
        defer out.free();
        const matrix = try run(&p, e, out, 0);
        defer gpa.free(matrix);
        // every pair's output is written: no byte of the fill is left where an item of few rows sits
        try base.expect(std.mem.indexOf(u8, matrix, &unwritten) == null, "{s} b{d}: the matrix path left outputs unwritten", .{ c.name, bits });
        // a cut below, at and past the rows a pass takes (the tile clamps the last)
        for ([_]c_int{ 1, 3, item.few_rows, 64 }) |few| {
            const passes = try run(&p, e, out, few);
            defer gpa.free(passes);
            const at = std.mem.indexOfDiff(u8, matrix, passes);
            try base.expect(at == null, "{s} b{d} cut {d}: byte {?d} differs from the matrix path", .{ c.name, bits, few, at });
        }
    };
}
