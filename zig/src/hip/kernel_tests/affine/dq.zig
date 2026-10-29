//! The prompt tile over dequantized weights against float64 on a Flash Next prompt's shapes, beside the folding tile.

const std = @import("std");
const base = @import("../rig.zig");
const affine = @import("../../launches/affine.zig");
const Launcher = @import("../../launches.zig").Launcher;
const kernels = @import("../../kernels.zig");
const Buffer = base.Buffer;
const gpa = base.gpa;
const prod = @import("product.zig");
const cases = @import("cases.zig");
const Product = prod.Product;
const Rig = prod.Rig;
const Entry = affine.Entry;

/// Hidden 2560 and expert width 640 at K 2560 and 640, group 32: a Flash Next layer's prompt products.
const shapes = [_]cases.Case{
    .{ .name = "flashnext hidden", .path = .prefill, .rows = 4096, .n = 2560, .k = 2560, .group = 32 },
    .{ .name = "flashnext expert gate_up", .path = .prefill, .rows = 4096, .n = 640, .k = 2560, .group = 32 },
    .{ .name = "flashnext expert down", .path = .prefill, .rows = 4096, .n = 2560, .k = 640, .group = 32 },
};

/// The row counts the two tiles are compared at: a tile's, its edges, and a prompt's.
const rows = [_]usize{ 16, 17, 128, 129, 256, 1024, 4096 };
/// The columns a row's products are checked at when the case has many (`n` above: the tail is the last).
const tail_n = [_]cases.Case{
    .{ .name = "flashnext hidden tail", .path = .prefill, .rows = 129, .n = 2561, .k = 2560, .group = 32 },
    .{ .name = "flashnext expert down tail", .path = .prefill, .rows = 17, .n = 641, .k = 640, .group = 32 },
};

/// The most |y - ref| a sum of the terms' magnitudes may leave: fp32 roundings a term, and each weight's bf16 rounding.
fn bound(k: usize, mids: f64) f64 {
    return 0x1p-24 * (8 + 2 * @sqrt(@as(f64, @floatFromInt(k)))) + mids * 0x1p-9;
}

const rows_sampled = 4;

fn at(i: usize, count: usize, n: usize) usize {
    return if (count <= 1 or n <= count) @min(i, n - 1) else i * (n - 1) / (count - 1);
}

const Errors = struct { worst: f64, column: usize };

/// One product: the folding tile's and the dequantizing one's error per column, against the float64 product.
fn compare(rig: *Rig, c: cases.Case, tot: *f64) !void {
    var p = try Product.init(rig, c);
    defer p.deinit();
    const matrix = find(rig, "mlx.project.prefill.matrix") orelse return error.NotRegistered;
    const dq = find(rig, "mlx.project.prefill.matrix_dq") orelse return error.NotRegistered;
    var label_buf: [160]u8 = undefined;
    const label = std.fmt.bufPrint(&label_buf, "{s} m{d} n{d} k{d}", .{ c.name, c.rows, c.n, c.k }) catch "?";

    var out = try Buffer.alloc(rig.base, p.outBytes(false));
    defer out.free();
    const row_f32 = try gpa.alloc(f32, c.n);
    defer gpa.free(row_f32);
    var errors = [2]Errors{ .{ .worst = 0, .column = 0 }, .{ .worst = 0, .column = 0 } };
    for ([_]*const Entry{ matrix, dq }, &errors) |e, *worst| {
        try out.fill8(0xA5, null);
        try prod.launchEntry(&p, e, out, false, 0, 0);
        try rig.base.stream.synchronize();
        for (0..@min(rows_sampled, c.rows)) |i| {
            const row = at(i, rows_sampled, c.rows);
            try out.download(row * c.n * 4, std.mem.sliceAsBytes(row_f32));
            for (0..c.n) |col| {
                const v = p.reference(row, col);
                const q = @abs(@as(f64, row_f32[col]) - v.y) / (v.norm * bound(c.k, 0) + 0x1p-149);
                if (q > worst.worst) worst.* = .{ .worst = q, .column = col };
            }
        }
    }
    const folding = errors[0].worst * bound(c.k, 0);
    const dequantized = errors[1].worst * bound(c.k, 0);
    tot.* = @max(tot.*, errors[1].worst);
    std.debug.print("  {s}: folding {e:.3} of the terms (column {d}), dequantized {e:.3} (column {d}), {d:.2}x\n", .{
        label, folding, errors[0].column, dequantized, errors[1].column, errors[1].worst / errors[0].worst,
    });
    // the folding tile keeps its own bound; the dequantized one adds each weight's single bf16 rounding
    const why = "{s}: the folding tile is {d:.1} times the error an fp32 sum may have";
    try base.expect(errors[0].worst <= 1, why, .{ label, errors[0].worst });
    const allow = bound(c.k, 1) / bound(c.k, 0);
    const past = "{s}: the dequantizing tile is {d:.1} times the terms, past the {d:.1} a bf16 weight allows";
    try base.expect(errors[1].worst <= allow, past, .{ label, errors[1].worst, allow });
}

fn find(rig: *Rig, id: []const u8) ?*const Entry {
    for (rig.kernels.reg.entries) |*e| if (std.mem.eql(u8, e.id, id)) return e;
    return null;
}

/// Both tiles on each shape and row count, and on the shapes whose last column is a tail.
pub fn sweep(rig: *Rig) !void {
    var tot: f64 = 0;
    for (shapes) |shape| for (rows) |m| {
        if (m > shape.rows) continue;
        var c = shape;
        c.rows = m;
        try compare(rig, c, &tot);
    };
    for (tail_n) |c| try compare(rig, c, &tot);
    const why = "  worst dequantizing tile error, in the folding tile's bound: {d:.3} (each cell allows its own)\n";
    std.debug.print(why, .{tot});
}

test "the dequantizing prompt tile against float64 on a Flash Next prompt's shapes, beside the folding tile" {
    const b = try base.Rig.open(1 << 20);
    defer b.close();
    // both tiles launch from a launcher that resolves the dequantizing one; the folding tile is there too
    var l = try Launcher.load(&b.r, .{ .products = .{ .dq = true } }, kernels.images);
    defer l.unload();
    var rig = Rig.open(b);
    rig.kernels = &l.affine;
    try sweep(&rig);
}
