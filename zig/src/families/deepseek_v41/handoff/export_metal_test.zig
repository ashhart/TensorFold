//! Metal -> neutral: a Packed slot exported and re-imported reproduces the slot; parts are `dsv41_unpack` rows.
const std = @import("std");
const g = @import("geometry.zig");
const c = @import("codec.zig");
const fp4 = @import("fp4.zig");
const plan = @import("plan.zig");
const ex = @import("export.zig");
const metal = @import("metal.zig");
const im = @import("import.zig");
const xm = @import("export_metal.zig");
const s = @import("synth.zig");

const capacity: u32 = 1024;

fn blankPacked(arena: std.mem.Allocator) !metal.Packed {
    const pk: metal.Packed = .{ .live = &.{}, .storage = &.{}, .layers = 40, .capacity = capacity };
    return .{ .live = try arena.alloc(u8, pk.liveBytes()), .storage = try arena.alloc(u8, metal.Packed.storageBytes(capacity)), .layers = 40, .capacity = capacity };
}

fn land(arena: std.mem.Allocator, shape: plan.Shape, parts: []const ex.Part, pk: metal.Packed) !void {
    var imp = try im.Import.init(arena, shape, .{ .@"packed" = pk }, .{});
    var buf: [40]u8 = undefined;
    for (parts) |p| try imp.put(0, p.spec.layer, p.spec.name(&buf), p.spec.dtype, p.spec.rows, p.spec.cols, p.bytes);
    _ = try imp.complete();
}

/// Per `block` values, every difference within `rel` of the block's largest magnitude.
fn near(a: []const u16, b: []const u16, block: usize, rel: f32) !void {
    var at: usize = 0;
    while (at < a.len) : (at += block) {
        var peak: f32 = 0;
        for (a[at..][0..block]) |v| peak = @max(peak, @abs(c.bf16ToF32(v)));
        for (a[at..][0..block], b[at..][0..block]) |x, y| try std.testing.expect(@abs(c.bf16ToF32(x) - c.bf16ToF32(y)) <= rel * peak);
    }
}

test "a Packed slot exported and imported again comes back within one requantisation step (even, odd and short prefixes)" {
    for ([_]u64{ 1, 2, 131, 300, 301 }) |upto| {
        var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
        defer arena_state.deinit();
        const arena = arena_state.allocator();
        const shape: plan.Shape = .{ .upto = upto };
        const w = try s.make(arena, .{ .upto = upto, .seed = upto });
        const first = try blankPacked(arena);
        @memset(first.storage, 0);
        try land(arena, shape, try ex.build(arena, shape, w.cuda), first);

        const parts = try xm.build(arena, shape, first);
        const second = try blankPacked(arena);
        @memset(second.storage, 0);
        try land(arena, shape, parts, second);
        // requantising a dequantised row moves a value by at most a step of its block: bounded, never a different row
        for (0..40) |layer| for (0..@min(upto, 128)) |k| {
            const pos = upto - 1 - k;
            const at = (layer * 128 + @as(usize, @intCast(pos % 128))) * metal.Packed.ring_stride;
            var a: [512]u16 = undefined;
            var b: [512]u16 = undefined;
            fp4.unpackWindow(first.live[at..][0..metal.Packed.ring_stride], &a);
            fp4.unpackWindow(second.live[at..][0..metal.Packed.ring_stride], &b);
            try near(&a, &b, 32, 0.07);
        };
        for (g.release.sources, 0..) |src, i| {
            const rows: usize = @intCast(g.release.rowsOf(@intCast(i), upto));
            const off = first.mainOff(i);
            const ioff = off + @as(usize, capacity / src.ratio) * metal.Packed.main_stride;
            for (0..rows) |r| {
                var a: [512]u16 = undefined;
                var b: [512]u16 = undefined;
                fp4.unpackMain(first.storage[off + r * metal.Packed.main_stride ..][0..metal.Packed.main_stride], &a);
                fp4.unpackMain(second.storage[off + r * metal.Packed.main_stride ..][0..metal.Packed.main_stride], &b);
                try near(&a, &b, 16, 0.34);
                var ia: [128]u16 = undefined;
                var ib: [128]u16 = undefined;
                fp4.unpackIndex(first.storage[ioff + r * metal.Packed.index_stride ..][0..metal.Packed.index_stride], &ia);
                fp4.unpackIndex(second.storage[ioff + r * metal.Packed.index_stride ..][0..metal.Packed.index_stride], &ib);
                try near(&ia, &ib, 32, 0.34);
            }
        }
        // the parts are Metal's own dequantised rows: the window of the last position
        var want: [512]u16 = undefined;
        const at = (@as(usize, 7) * 128 + @as(usize, @intCast((upto - 1) % 128))) * metal.Packed.ring_stride;
        fp4.unpackWindow(first.live[at..][0..metal.Packed.ring_stride], &want);
        for (parts) |p| if (p.spec.layer == 7 and p.spec.kind == .swa) {
            const last = c.loadRow(512, p.bytes[(p.spec.rows - 1) * 1024 ..]);
            try std.testing.expectEqualSlices(u16, &want, &last);
        };
    }
}

test "the Metal exporter holds no drafter rings and refuses a slot too small for the prefix" {
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const pk = try blankPacked(arena);
    try std.testing.expectError(error.NoDrafterRings, xm.build(arena, .{ .upto = 200, .ds_rows = 4 }, pk));
    try std.testing.expectError(error.TargetTooSmall, xm.build(arena, .{ .upto = capacity + 1 }, pk));
}
