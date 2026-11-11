//! The registry's costs per GPU, read at build time from a backend's .zon files; no autotuning, so ranks agree.

const std = @import("std");

/// One entry's cost and the products it takes (the file's header says what each field means).
pub const Row = struct {
    id: []const u8,
    cost: f32,
    rows: u32 = 0,
    rows_matrix: u32 = 0,
    min_rows: u32 = 0,
    items: u32 = 0,
    /// The most pairs a routed plan holds per item: more than `pairs_per_item * items` is refused (0: any).
    pairs_per_item: f32 = 0,
    /// A routed prompt item of at most this many rows takes its kernel's K-parallel passes (0: none does).
    few_rows: u32 = 0,
    /// The most bits a weight code holds (0: any); a bound that is no code width takes no product.
    max_bits: u8 = 0,
    plain: bool = true,
    routed: bool = true,
};

pub const Table = struct { gfx: []const u8, rows: []const Row };

/// The widths a format's codes come in.
pub const widths = [_]u8{ 2, 3, 4, 5, 6, 8 };

/// The table a .zon document of `gfx` and `entries` describes.
pub fn build(comptime zon: anytype) Table {
    const rows = comptime blk: {
        // each row compares every field's name: the default quota stops near 30 rows
        @setEvalBranchQuota(1000 + 100 * zon.entries.len * @typeInfo(Row).@"struct".field_names.len);
        var out: [zon.entries.len]Row = undefined;
        for (zon.entries, 0..) |e, i| {
            // a field read by name is dropped when misspelled, and a dropped bound takes every product
            if (stray(e)) |name| @compileError("tuning: " ++ e.id ++ " has a field no row has: " ++ name);
            var row: Row = .{ .id = e.id, .cost = e.cost };
            for (@typeInfo(Row).@"struct".field_names) |name| {
                if (std.mem.eql(u8, name, "id") or std.mem.eql(u8, name, "cost")) continue;
                if (@hasField(@TypeOf(e), name)) @field(row, name) = @field(e, name);
            }
            // a bound that is no code width would take no product at all: it is reported here, where it is written
            if (!widthBound(row.max_bits)) @compileError(std.fmt.comptimePrint("tuning: {s} bounds its codes at {d} bits, which is no width they come in", .{ e.id, row.max_bits }));
            out[i] = row;
        }
        break :blk out;
    };
    return .{ .gfx = zon.gfx, .rows = &rows };
}

/// The first field of a table entry that the row type lacks, or null.
pub fn stray(comptime e: anytype) ?[]const u8 {
    for (@typeInfo(@TypeOf(e)).@"struct".field_names) |name| {
        if (!@hasField(Row, name)) return name;
    }
    return null;
}

/// Whether `max_bits` is a bound a row may carry: 0 for any width, or a width the codes come in.
pub fn widthBound(max_bits: u8) bool {
    return max_bits == 0 or std.mem.indexOfScalar(u8, &widths, max_bits) != null;
}

pub fn find(t: *const Table, id: []const u8) ?Row {
    for (t.rows) |r| if (std.mem.eql(u8, r.id, id)) return r;
    return null;
}

test "a table is found by an entry's id" {
    const zon = .{ .gfx = "toy", .entries = .{ .{ .id = "a", .cost = 1 }, .{ .id = "b", .cost = 2, .rows = 8 } } };
    const t = comptime build(zon);
    try std.testing.expectEqual(@as(u32, 8), find(&t, "b").?.rows);
    try std.testing.expect(find(&t, "c") == null);
}

test "a field the row type lacks is found in an entry, so a misspelled bound stops the build instead of being dropped" {
    try std.testing.expect(comptime stray(.{ .id = "a", .cost = 1, .rows = 8, .max_bits = 3 }) == null);
    const misspelled = comptime stray(.{ .id = "a", .cost = 1, .max_bit = 3 });
    try std.testing.expectEqualStrings("max_bit", misspelled orelse "");
    const t = comptime build(.{ .gfx = "toy", .entries = .{.{ .id = "a", .cost = 1, .max_bits = 3 }} });
    try std.testing.expectEqual(@as(u8, 3), find(&t, "a").?.max_bits);
}

test "a width bound that is no code width is told apart, so a table that writes one stops the build instead of taking nothing" {
    try std.testing.expect(widthBound(0));
    for (widths) |bits| try std.testing.expect(widthBound(bits));
    for ([_]u8{ 1, 7, 9, 255 }) |bits| try std.testing.expect(!widthBound(bits));
    // every width is a bound a table may write
    inline for (widths) |bits| {
        const t = comptime build(.{ .gfx = "toy", .entries = .{.{ .id = "a", .cost = 1, .max_bits = bits }} });
        try std.testing.expectEqual(@as(u8, bits), find(&t, "a").?.max_bits);
    }
}
