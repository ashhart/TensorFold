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
    plain: bool = true,
    routed: bool = true,
};

pub const Table = struct { gfx: []const u8, rows: []const Row };

/// The table a .zon document of `gfx` and `entries` describes.
pub fn build(comptime zon: anytype) Table {
    const rows = comptime blk: {
        var out: [zon.entries.len]Row = undefined;
        for (zon.entries, 0..) |e, i| {
            var row: Row = .{ .id = e.id, .cost = e.cost };
            for (@typeInfo(Row).@"struct".field_names) |name| {
                if (std.mem.eql(u8, name, "id") or std.mem.eql(u8, name, "cost")) continue;
                if (@hasField(@TypeOf(e), name)) @field(row, name) = @field(e, name);
            }
            out[i] = row;
        }
        break :blk out;
    };
    return .{ .gfx = zon.gfx, .rows = &rows };
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
