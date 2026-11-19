//! The decode layer picks its FP8 kernel by rows and columns: the row kernel below 5 rows, the MMA tile for wide ones.
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

test "dsv41 metal: a layer's FP8 projections run on the row kernel below 5 rows and its wide ones on the tile from 5" {
    const r = try fixture.Rig.init(testing.allocator, testing.io, 1280, 44);
    defer r.deinit();
    // the fixture's wq_b (2048 columns) and index wq_b (4096) are wide at K 128; the rest stay on the row kernel
    try testing.expect(tile.minRows(.{ .k = 128, .n = 2048, .rows = 1 }, 1) == 5 and tile.minRows(.{ .k = 256, .n = 256, .rows = 1 }, 1) > tile.max_rows);
    for (1..tile.max_rows + 1) |rows| {
        const n = try count(r, @intCast(rows));
        try testing.expect(n[1] > 0 and (n[0] > 0) == (rows >= 5));
    }
    std.debug.print("PASS dsv41 metal FP8 kernel by rows: row kernel at 1-4 rows, the MMA tile for wide projections at 5-16\n", .{});
}
