//! The registry's costs on each GPU: the .zon tables beside this file, read at build time.

const std = @import("std");
const tuning = @import("core").tuning;

pub const Row = tuning.Row;
pub const Table = tuning.Table;
pub const find = tuning.find;

pub const gfx1030 = tuning.build(@import("gfx1030.zon"));
pub const gfx1100 = tuning.build(@import("gfx1100.zon"));

test "the tables name each entry once" {
    for ([_]*const Table{ &gfx1030, &gfx1100 }) |t| {
        for (t.rows, 0..) |r, i| {
            for (t.rows[i + 1 ..]) |o| try std.testing.expect(!std.mem.eql(u8, r.id, o.id));
        }
    }
    try std.testing.expect(find(&gfx1100, "mlx.project.decode.matrix") != null);
    try std.testing.expect(find(&gfx1030, "mlx.project.decode.matrix") == null);
}
