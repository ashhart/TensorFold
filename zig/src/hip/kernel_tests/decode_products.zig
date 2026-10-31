//! Routed, plain and prompt products against the launcher's pick and the matrix tile: the same bytes.
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

/// Flash Next's prompt expert tiles against the 128-row matrix tile: the same bytes.
fn promptExperts(t: *Rig, rng: *rig.Rng, down: bool, rows: usize, experts: usize, skew: bool, limit: f32) !void {
    const kernels = &t.on.affine;
    const sh = if (down) affine.Kernels.moe_down_shape else affine.Kernels.moe_gu_shape;
    const n: usize = @intCast(sh.n);
    const k: usize = @intCast(sh.k);
    const slots = 11;
    const pairs = rows * slots;
    const width = if (down) n else n / 2;
    var m = try Matrix.init(t, rng, n, k, @intCast(sh.bits), @intCast(sh.group), experts);
    defer m.deinit();
    const x_rows = if (down) pairs else rows;
    const hx = try gpa.alloc(u16, x_rows * k);
    defer gpa.free(hx);
    for (hx) |*v| v.* = t.bits(rng.unit());
    var x = try t.upload(hx);
    defer x.free();
    // each pair draws an expert; sorted by expert (stable) into items of at most 128 pairs, as moe_route plans a prompt
    const pick = try gpa.alloc(u32, pairs);
    defer gpa.free(pick);
    for (pick) |*e| {
        const r = rng.next();
        e.* = @intCast(if (skew and r % 4 != 0) r % 3 else r % experts);
    }
    const members = try gpa.alloc(i32, pairs);
    defer gpa.free(members);
    var n_members: usize = 0;
    const items = try gpa.alloc(i32, 3 * (pairs + experts + 8));
    defer gpa.free(items);
    @memset(items, 0);
    var n_items: usize = 0;
    for (0..experts) |e| {
        var run: usize = 0;
        for (0..pairs) |pi| {
            if (pick[pi] != e) continue;
            if (run == 0) {
                items[3 * n_items ..][0..3].* = .{ @intCast(e), @intCast(n_members), 0 };
                n_items += 1;
            }
            members[n_members] = @intCast(pi);
            n_members += 1;
            items[3 * (n_items - 1) + 2] += 1;
            run = if (run + 1 == 128) 0 else run + 1;
        }
    }
    // a plan's capacity holds empty items past the used ones
    const count = n_items + 8;
    var dev_items = try t.upload(items[0 .. 3 * count]);
    defer dev_items.free();
    var dev_members = try t.upload(members);
    defer dev_members.free();
    var sums = [2]@import("../memory.zig").DeviceBuffer{ try t.alloc(pairs * n * 4), try t.alloc(pairs * n * 4) };
    defer for (&sums) |*b| b.free();
    var acts: [2]@import("../memory.zig").DeviceBuffer = .{ try t.alloc(pairs * width * 2), try t.alloc(pairs * width * 2) };
    defer for (&acts) |*b| b.free();
    try sums[1].fill8(0xA5);
    var a = m.arg(t, at(x), 128, at(sums[0]));
    a.route = .{ .items = at(dev_items), .members = at(dev_members), .x_div = if (down) 1 else slots };
    try kernels.blockGo(&t.r, a, t.stream.handle, @intCast(count), .matrix);
    if (down) {
        a.out = at(sums[1]);
        try std.testing.expect(try kernels.moeDown(&t.r, a, @intCast(count), t.stream.handle));
    } else {
        try t.off.tf_moe_act(@ptrFromInt(at(sums[0])), @ptrFromInt(at(acts[0])), t.kind(), @intCast(pairs), @intCast(width), limit, t.stream.handle);
        a.out = 0;
        a.out16 = at(acts[1]);
        try std.testing.expect(try kernels.moeGateUp(&t.r, a, @intCast(count), limit, t.stream.handle));
    }
    try t.stream.synchronize();
    const cmp = if (down) sums else acts;
    const want = try rig.download(u8, cmp[0]);
    defer gpa.free(want);
    const got = try rig.download(u8, cmp[1]);
    defer gpa.free(got);
    const size: usize = if (down) 4 else 2;
    var differ: usize = 0;
    for (0..want.len / size) |e| if (!std.mem.eql(u8, want[e * size ..][0..size], got[e * size ..][0..size])) {
        if (differ < 4) std.debug.print("  prompt experts down {} rows {d} skew {}: pair {d} column {d} differs\n", .{ down, rows, skew, e / width, e % width });
        differ += 1;
    };
    try std.testing.expectEqual(@as(usize, 0), differ);
    // and the matrix tile's sums are float64's within its bound
    const prob = m.problem(t, hx);
    const ref_sums = try rig.download(f32, sums[0]);
    defer gpa.free(ref_sums);
    for (0..8) |s| {
        const pi = s * (pairs - 1) / 7;
        const col = (s * 97) % width;
        const r = ref.reference(prob, if (down) pi else pi / slots, pick[pi], col);
        try std.testing.expect(@abs(@as(f64, ref_sums[pi * n + col]) - r.y) <= 1e-5 * r.norm + 1e-7);
    }
}

test "Flash Next's prompt gate and up with its activation in one launch writes the bytes of the matrix tile and moe_act" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    if (t.on.affine.moe_gu == null) return error.SkipZigTest;
    var rng: rig.Rng = .{ .state = 0x9E3779B97F4A7C15 };
    try promptExperts(t, &rng, false, 300, 24, false, 0);
    try promptExperts(t, &rng, false, 2048, 24, true, 0);
    try promptExperts(t, &rng, false, 64, 24, false, 0);
    try promptExperts(t, &rng, false, 512, 24, false, 7.0);
}

test "Flash Next's prompt expert down writes the bytes of the matrix tile" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    if (t.on.affine.moe_down == null) return error.SkipZigTest;
    var rng: rig.Rng = .{ .state = 0x2545F4914F6CDD1D };
    try promptExperts(t, &rng, true, 300, 24, false, 0);
    try promptExperts(t, &rng, true, 2048, 24, true, 0);
    try promptExperts(t, &rng, true, 64, 24, false, 0);
}

/// A prompt's product on its K's dense tile against the matrix tile: the same fp32 bytes, and tf_cast's bf16 bytes.
fn promptDense(t: *Rig, rng: *rig.Rng, rows: usize, n: usize, k: usize) !void {
    const kernels = &t.on.affine;
    var m = try Matrix.init(t, rng, n, k, 4, 32, 1);
    defer m.deinit();
    const hx = try gpa.alloc(u16, rows * k);
    defer gpa.free(hx);
    for (hx) |*v| v.* = t.bits(rng.unit());
    var x = try t.upload(hx);
    defer x.free();
    var outs = [2]@import("../memory.zig").DeviceBuffer{ try t.alloc(rows * n * 4), try t.alloc(rows * n * 4) };
    defer for (&outs) |*b| b.free();
    try outs[1].fill8(0xA5);
    var a = m.arg(t, at(x), rows, at(outs[0]));
    try kernels.blockGo(&t.r, a, t.stream.handle, 1, .matrix);
    a.out = at(outs[1]);
    const e = kernels.choose(.project, .prefill, affine.Kernels.shapeOf(a, 1)) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqualStrings("mlx.project.prefill.dense", e.id);
    try kernels.prefillLaunch(&t.r, a, t.stream.handle, 1, 0);
    try t.stream.synchronize();
    const want = try rig.download(u8, outs[0]);
    defer gpa.free(want);
    const got = try rig.download(u8, outs[1]);
    defer gpa.free(got);
    var differ: usize = 0;
    for (0..want.len / 4) |i| if (!std.mem.eql(u8, want[i * 4 ..][0..4], got[i * 4 ..][0..4])) {
        if (differ < 4) std.debug.print("  prompt dense {d}x{d} rows {d}: row {d} column {d} differs\n", .{ n, k, rows, i / n, i % n });
        differ += 1;
    };
    try std.testing.expectEqual(@as(usize, 0), differ);
    // the BF16 output: the cast of the fp32 sums
    var halves = [2]@import("../memory.zig").DeviceBuffer{ try t.alloc(rows * n * 2), try t.alloc(rows * n * 2) };
    defer for (&halves) |*b| b.free();
    try halves[1].fill8(0xA5);
    try t.on.tf_cast(@ptrFromInt(at(outs[0])), 0, @ptrFromInt(at(halves[0])), t.kind(), @intCast(rows * n), t.stream.handle);
    a.out = 0;
    a.out16 = at(halves[1]);
    try kernels.prefillLaunch(&t.r, a, t.stream.handle, 1, 0);
    try t.stream.synchronize();
    const want16 = try rig.download(u8, halves[0]);
    defer gpa.free(want16);
    const got16 = try rig.download(u8, halves[1]);
    defer gpa.free(got16);
    for (0..want16.len / 2) |i| if (!std.mem.eql(u8, want16[i * 2 ..][0..2], got16[i * 2 ..][0..2])) {
        if (differ < 4) std.debug.print("  prompt dense bf16 {d}x{d} rows {d}: row {d} column {d} differs\n", .{ n, k, rows, i / n, i % n });
        differ += 1;
    };
    try std.testing.expectEqual(@as(usize, 0), differ);
    const prob = m.problem(t, hx);
    const sums = try rig.download(f32, outs[1]);
    defer gpa.free(sums);
    for (0..8) |s| {
        const r = s * (rows - 1) / 7;
        const col = (s * 97) % n;
        const y = ref.reference(prob, r, 0, col);
        try std.testing.expect(@abs(@as(f64, sums[r * n + col]) - y.y) <= 1e-5 * y.norm + 1e-7);
    }
}

test "a prompt's dense products at Flash Next's K write the bytes of the matrix tile, and of tf_cast's rounding in bf16" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    if (t.on.affine.dense[0] == null) return error.SkipZigTest;
    var rng: rig.Rng = .{ .state = 0x6A09E667F3BCC908 };
    try promptDense(t, &rng, 300, 640, 2560);
    try promptDense(t, &rng, 130, 200, 2560);
    try promptDense(t, &rng, 257, 2560, 6144);
    try promptDense(t, &rng, 200, 324, 10240);
    try promptDense(t, &rng, 383, 10240, 320);
    try promptDense(t, &rng, 1024, 1024, 2560);
}
