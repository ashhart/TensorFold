//! The decode layer picks its FP8 kernel by row count: the row kernel below the tile's rows, the MMA tile from them.
const std = @import("std");
const mtl = @import("metal");
const pkg = @import("dsv41_metal");
const fixture = @import("fixture.zig");
const recorder = @import("recorder.zig");

const testing = std.testing;
const tile = pkg.pipelines.fp8_tile;

/// Dispatches of one layer's encode at `rows` rows: on the tile, and on the row kernel's one-launch pipelines.
fn count(r: *fixture.Rig, rows: u32) ![2]usize {
    var rec: recorder.Recording = .{ .a = testing.allocator };
    defer rec.deinit();
    const x = r.ctx(&r.slot, 1101, rows);
    const s = r.streams();
    const pr = r.pres();
    const e = recorder.start(&rec);
    try pkg.layer.encode(&x, e, &r.layers[1], .{ .x = s[0], .pre = pr[0], .x_out = s[1], .pre_out = pr[1] });
    recorder.stop(e);
    var out: [2]usize = .{ 0, 0 };
    var pipe: ?mtl.objc.Id = null;
    for (rec.ops.items) |op| switch (op) {
        .pipe => |p| pipe = p,
        .groups, .threads => {
            for (r.p.tile.p) |t| for (t) |cb| for (cb) |q| {
                out[0] += @intFromBool(pipe == q.id);
            };
            for (r.p.fp8.whole) |k| for (k) |t| for (t) |q| {
                out[1] += @intFromBool(pipe == q.id);
            };
        },
        else => {},
    };
    return out;
}

test "dsv41 metal: a layer's FP8 projections run on the row kernel below the tile's rows and on the tile from them" {
    const r = try fixture.Rig.init(testing.allocator, testing.io, 1280, 44);
    defer r.deinit();
    for (1..tile.max_rows + 1) |rows| {
        const n = try count(r, @intCast(rows));
        if (rows < tile.min_rows) {
            try testing.expect(n[0] == 0 and n[1] > 0);
        } else {
            try testing.expect(n[0] > 0 and n[1] == 0);
        }
    }
    std.debug.print("PASS dsv41 metal FP8 kernel by rows: row kernel at 1-{d} rows, the MMA tile at {d}-16\n", .{ tile.min_rows - 1, tile.min_rows });
}
