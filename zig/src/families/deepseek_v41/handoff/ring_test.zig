//! The release layout's logical rings: one shorter than the window is refused before landing, the window holds every row.
const std = @import("std");
const g = @import("geometry.zig");
const c = @import("codec.zig");
const plan = @import("plan.zig");
const ex = @import("export.zig");
const metal = @import("metal.zig");
const im = @import("import.zig");
const s = @import("synth.zig");

const geo = g.release;

/// Release buffers sized for the engine's 144 window slots and 133 drafter rows, filled with markers.
fn release(arena: std.mem.Allocator, upto: u64, window_slots: u32, ds_window: u32) !metal.Release {
    const layers = try arena.alloc(metal.Release.Layer, geo.layers);
    for (layers, 0..) |*l, i| {
        l.* = .{ .window = try arena.alloc(u8, 2 * 144 * 1024) };
        @memset(l.window, 0xEE);
        if (geo.sourceIndex(@intCast(i))) |si| {
            const rows: usize = @intCast(geo.rowsOf(si, upto) + 1);
            l.pool = try arena.alloc(u8, rows * 1024);
            l.keys = try arena.alloc(u8, rows * 68);
            l.carry = try arena.alloc(u8, 4 * 2048);
        }
    }
    const drafter = try arena.alloc([]u8, geo.mtp_layers);
    for (drafter) |*d| {
        d.* = try arena.alloc(u8, 133 * 1024);
        @memset(d.*, 0x33);
    }
    return .{ .window_slots = window_slots, .ds_window = ds_window, .layers = layers, .drafter = drafter };
}

fn untouched(r: metal.Release) bool {
    for (r.layers) |l| if (!std.mem.allEqual(u8, l.window, 0xEE)) return false;
    for (r.drafter) |d| if (!std.mem.allEqual(u8, d, 0x33)) return false;
    return true;
}

/// `rows` rows of 512 bf16 values, pairwise different, so a ring that overwrote one would show it.
fn distinct(rows: []const u16, n: usize) !void {
    for (0..n) |i| for (i + 1..n) |j|
        try std.testing.expect(!std.mem.eql(u16, rows[i * 512 ..][0..512], rows[j * 512 ..][0..512]));
}

test "a release window or drafter ring shorter than the window is refused before any row lands" {
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const shape: plan.Shape = .{ .upto = 300, .ds_rows = 128 };
    for ([_][2]u32{ .{ 1, 128 }, .{ geo.window - 1, 128 }, .{ 144, 1 }, .{ 144, geo.window - 1 } }) |size| {
        const r = try release(arena, shape.upto, size[0], size[1]);
        try std.testing.expectError(error.RingTooSmall, im.Import.init(arena, shape, .{ .release = r }, .{}));
        try std.testing.expect(untouched(r));
    }
}

test "rings exactly the window long hold all 128 distinct window and drafter rows, each in its own slot" {
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const shape: plan.Shape = .{ .upto = 300, .ds_rows = 128 };
    const w = try s.make(arena, .{ .upto = shape.upto, .ds_rows = shape.ds_rows, .seed = 41 });
    const r = try release(arena, shape.upto, geo.window, geo.window);
    var imp = try im.Import.init(arena, shape, .{ .release = r }, .{});
    var buf: [40]u8 = undefined;
    for (try ex.build(arena, shape, w.cuda)) |p| try imp.put(0, p.spec.layer, p.spec.name(&buf), p.spec.dtype, p.spec.rows, p.spec.cols, p.bytes);
    const res = try imp.complete();
    try std.testing.expectEqual(shape.upto - 128, res.ds_first);
    var want: [1024]u8 = undefined;
    for (r.layers, 0..) |l, li| {
        try distinct(w.truth.swa[li], geo.window);
        for (0..geo.window) |i| {
            const pos = shape.upto - geo.window + i;
            c.storeRow(512, w.truth.swa[li][i * 512 ..][0..512], &want);
            for ([_]u64{ pos % geo.window, pos % geo.window + geo.window }) |slot|
                try std.testing.expectEqualSlices(u8, &want, l.window[@intCast(slot * 1024)..][0..1024]);
        }
    }
    for (r.drafter, 0..) |d, stage| {
        try distinct(w.truth.ds[stage], 128);
        for (0..128) |i| {
            const pos = shape.upto - 128 + i;
            c.storeRow(512, w.truth.ds[stage][i * 512 ..][0..512], &want);
            try std.testing.expectEqualSlices(u8, &want, d[@intCast((pos % geo.window) * 1024)..][0..1024]);
        }
    }
}
