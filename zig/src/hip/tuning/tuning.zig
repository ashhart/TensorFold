//! The registry's costs on each GPU: the .zon tables beside this file, read at build time.

const std = @import("std");
const tuning = @import("core").tuning;
const registry = @import("core").registry;

pub const Row = tuning.Row;
pub const Table = tuning.Table;
pub const find = tuning.find;

pub const gfx1030 = tuning.build(@import("gfx1030.zon"));
pub const gfx1100 = tuning.build(@import("gfx1100.zon"));
pub const gfx1151 = tuning.build(@import("gfx1151.zon"));
pub const gfx1201 = tuning.build(@import("gfx1201.zon"));

/// The tables' documents, for the fields a table sets beside its rows.
const docs = .{ @import("gfx1030.zon"), @import("gfx1100.zon"), @import("gfx1151.zon"), @import("gfx1201.zon") };

/// The fewest rows a prompt's routed plan cuts at the GEMM tile on `chip`: its own table's, null where none sets it.
pub fn promptBlockFrom(chip: []const u8) ?usize {
    inline for (docs) |doc| {
        // a generation's other chips share its costs, not this measured start
        if (@hasField(@TypeOf(doc), "prompt_block_from") and std.mem.eql(u8, chip, doc.gfx)) return doc.prompt_block_from;
    }
    return null;
}

test "only gfx1151 starts a prompt's routed GEMM tile below 64 rows, and only on that chip" {
    try std.testing.expectEqual(@as(?usize, 33), promptBlockFrom("gfx1151"));
    // the generation's other chips and the other tables keep the default; so does a build without kernels
    for ([_][]const u8{ "gfx1150", "gfx1152", "gfx1153", "gfx1030", "gfx1100", "gfx1101", "gfx1201", "" }) |chip| {
        try std.testing.expectEqual(@as(?usize, null), promptBlockFrom(chip));
    }
    // a table's start lies past the decode tile's 8 rows and at most at the default's 64
    inline for (docs) |doc| if (@hasField(@TypeOf(doc), "prompt_block_from")) {
        try std.testing.expect(doc.prompt_block_from > 8 and doc.prompt_block_from <= 64);
    };
}

test "only a routed prompt row bounds its pairs per item, by a finite number from 0 to 256" {
    for ([_]*const Table{ &gfx1030, &gfx1100, &gfx1151, &gfx1201 }) |t| for (t.rows) |r| {
        try std.testing.expect(std.math.isFinite(r.pairs_per_item) and r.pairs_per_item >= 0 and r.pairs_per_item <= 256);
        if (r.pairs_per_item != 0) try std.testing.expect(std.mem.startsWith(u8, r.id, "mlx.routed.prefill."));
    };
}

test "only the routed prompt's item tile cuts at few rows, at no more than its K-parallel passes take" {
    const most = @import("../launches/matrix_item.zig").few_rows;
    for ([_]*const Table{ &gfx1030, &gfx1100, &gfx1151, &gfx1201 }) |t| for (t.rows) |r| {
        try std.testing.expect(r.few_rows <= most);
        if (r.few_rows != 0) try std.testing.expectEqualStrings("mlx.routed.prefill.item64", r.id);
    };
    // the cut is live where the receipts were taken: a table that lost it would pass the loop above
    try std.testing.expect(find(&gfx1151, "mlx.routed.prefill.item64").?.few_rows != 0);
}

test "a row's width bound is a width the codes come in" {
    for ([_]*const Table{ &gfx1030, &gfx1100, &gfx1151, &gfx1201 }) |t| for (t.rows) |r| {
        if (r.max_bits != 0) try std.testing.expect(std.mem.indexOfScalar(u8, &registry.widths, r.max_bits) != null);
    };
    // the bound is live where it was measured: a table that lost it would pass the loop above
    try std.testing.expectEqual(@as(u8, 3), find(&gfx1151, "mlx.project.prefill.kp0").?.max_bits);
}

test "the tables name each entry once" {
    for ([_]*const Table{ &gfx1030, &gfx1100, &gfx1151, &gfx1201 }) |t| {
        for (t.rows, 0..) |r, i| {
            for (t.rows[i + 1 ..]) |o| try std.testing.expect(!std.mem.eql(u8, r.id, o.id));
        }
    }
    try std.testing.expect(find(&gfx1100, "mlx.project.decode.matrix") != null);
    try std.testing.expect(find(&gfx1030, "mlx.project.decode.matrix") == null);
}
