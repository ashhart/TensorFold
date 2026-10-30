//! Routed and plain products against the launcher's pick: the same bytes.
const std = @import("std");
const rig = @import("rig.zig");
const Rig = rig.Rig;
const gpa = rig.gpa;
const at = rig.at;
const affine = @import("../launches/affine.zig");
const ref = @import("affine/reference.zig");
const Matrix = @import("decode.zig").Matrix;

/// A verify window's routed gate+up or down at every columns-a-lane count, and as single-pair items: the same bytes.
fn windowBytes(t: *Rig, rng: *rig.Rng, down: bool, rows: usize, unique_in: ?usize) !void {
    const n: usize = if (down) 2560 else 2 * 640;
    const k: usize = if (down) 640 else 2560;
    const experts = 64;
    const slots = 11;
    var m = try Matrix.init(t, rng, n, k, 4, 32, experts);
    defer m.deinit();
    const kernels = &t.on.affine;
    defer kernels.stream_cb_force = 0;
    const pairs = rows * slots;
    const unique = unique_in orelse slots + (slots * (rows - 1)) / 2;
    const x_rows = if (down) pairs else rows;
    const hx = try gpa.alloc(u16, x_rows * k);
    defer gpa.free(hx);
    for (hx) |*v| v.* = t.bits(rng.unit());
    var x = try t.upload(hx);
    defer x.free();
    const dead = 9;
    const items = try gpa.alloc(i32, (unique + dead) * 3);
    defer gpa.free(items);
    const members = try gpa.alloc(i32, pairs);
    defer gpa.free(members);
    // the same pairs, every one its own item (members in the same order), then the dead items
    const single = try gpa.alloc(i32, (pairs + dead) * 3);
    defer gpa.free(single);
    var first: usize = 0;
    var most: usize = 0;
    for (0..unique) |i| {
        var count: usize = 0;
        var j = i;
        while (j < pairs) : (j += unique) {
            members[first + count] = @intCast(j);
            single[3 * (first + count) ..][0..3].* = .{ @intCast((i * 7) % experts), @intCast(first + count), 1 };
            count += 1;
        }
        items[3 * i ..][0..3].* = .{ @intCast((i * 7) % experts), @intCast(first), @intCast(count) };
        first += count;
        most = @max(most, count);
    }
    for (0..dead) |d| {
        items[3 * (unique + d) ..][0..3].* = .{ experts - 1, @intCast(pairs), 0 };
        single[3 * (pairs + d) ..][0..3].* = .{ experts - 1, @intCast(pairs), 0 };
    }
    var dev_items = try t.upload(items);
    defer dev_items.free();
    var dev_single = try t.upload(single);
    defer dev_single.free();
    var dev_members = try t.upload(members);
    defer dev_members.free();
    const out_bytes = if (down) pairs * n * 4 else pairs * (n / 2) * 2;
    var want: ?[]u8 = null;
    defer if (want) |w| gpa.free(w);
    // each CB on the grouped plan, the rule's CB on single items (a.m 1), the grouped plan at a.m = rows as the engine
    for ([_]c_int{ 0, 1, 2, 4, 8, -1, -2 }) |cb| {
        var out = try t.alloc(out_bytes);
        defer out.free();
        const split = cb == -1;
        var a = m.arg(t, at(x), if (split) 1 else if (cb == -2) rows else most, if (down) at(out) else 0);
        if (!down) a.out16 = at(out);
        a.route = .{ .items = at(if (split) dev_single else dev_items), .members = at(dev_members), .x_div = if (down) 1 else slots };
        const n_items: c_int = @intCast(if (split) pairs + dead else unique + dead);
        kernels.stream_cb_force = @max(cb, 0);
        if (down) try kernels.routed(&t.r, a, n_items, t.stream.handle) else try std.testing.expect(try kernels.pairRun(&t.r, a, 0, n_items, t.stream.handle));
        try t.stream.synchronize();
        const got = try rig.download(u8, out);
        if (want) |w| {
            defer gpa.free(got);
            try std.testing.expectEqualSlices(u8, w, got);
        } else want = got;
    }
}

test "the routed gate+up and down write the same bytes at every columns a lane and as single-pair items (Flash Next, 1 to 8 rows)" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    var rng: rig.Rng = .{ .state = 0x9E3779B97F4A7C15 };
    for ([_]bool{ false, true }) |down| {
        for ([_]usize{ 1, 2, 4, 5, 8 }) |rows| try windowBytes(t, &rng, down, rows, null);
        // 4 rows an expert: the 4-row tile
        try windowBytes(t, &rng, down, 4, 11);
        try windowBytes(t, &rng, down, 8, 22);
    }
}

/// A plain product (one matrix, fp32 out) at every columns-a-lane count against the launcher's pick: the same bytes.
fn plainBytes(t: *Rig, rng: *rig.Rng, n: usize, k: usize, rows: usize) !void {
    var m = try Matrix.init(t, rng, n, k, 4, 32, 1);
    defer m.deinit();
    const kernels = &t.on.affine;
    defer kernels.stream_cb_force = 0;
    const hx = try gpa.alloc(u16, rows * k);
    defer gpa.free(hx);
    for (hx) |*v| v.* = t.bits(rng.unit());
    var x = try t.upload(hx);
    defer x.free();
    var want: ?[]u8 = null;
    defer if (want) |w| gpa.free(w);
    for ([_]c_int{ 0, 1, 2, 4, 8 }) |cb| {
        var out = try t.alloc(rows * n * 4);
        defer out.free();
        const a = m.arg(t, at(x), rows, at(out));
        kernels.stream_cb_force = cb;
        try kernels.streamGo(&t.r, a, .{}, a.n, 1, t.stream.handle);
        try t.stream.synchronize();
        const got = try rig.download(u8, out);
        if (want) |w| {
            defer gpa.free(got);
            try std.testing.expectEqualSlices(u8, w, got);
        } else want = got;
    }
}

test "a plain product writes the same bytes at every columns a lane (Flash Next's in-proj K, out-proj, a ragged N; 1 to 4 rows)" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    var rng: rig.Rng = .{ .state = 0x9E3779B97F4A7C15 };
    for ([_]usize{ 1, 2, 4 }) |rows| {
        try plainBytes(t, &rng, 4096, 2560, rows);
        try plainBytes(t, &rng, 2560, 6144, rows);
        try plainBytes(t, &rng, 4099, 2560, rows);
    }
}
