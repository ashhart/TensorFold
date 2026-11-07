//! The combine operator's FP64 boundary.

const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const ops = tf.deepseek_v41.ops;
const math = @import("math.zig");
const h = @import("operator_csa2_fp64.zig");
const View = ops.View;

const slots = 6;
const experts = 384;

fn view(v: anytype) View {
    return .{ .buffer = v.buffer, .offset = v.offset };
}

/// ULP distance between two BF16-coded values, through their exact FP32 values.
fn bf16Unequal(a: u16, b: u16) bool {
    return a != b;
}

const Case = struct {
    name: []const u8,
    rows: usize,
    width: usize,
    seed: u64,
    invalid: bool = false,
};

fn combineCase(device: mtl.Device, queue: mtl.Queue, p: *const ops.Pipelines, case: Case) !struct { unequal: usize, cells: usize, worst_relative: f64, nan_cells: usize } {
    var b = h.Buffers{ .device = device };
    defer b.deinit();
    var rng = h.Lcg{ .state = case.seed };
    const routed_row_stride = slots * case.width;
    const routed = try b.get(u16, case.rows * routed_row_stride);
    const shared = try b.get(u16, case.rows * case.width);
    const ids = try b.get(u32, case.rows * slots);
    const y = try b.get(u16, case.rows * case.width);
    for (routed.slice(u16, case.rows * routed_row_stride)) |*cell| cell.* = math.b(rng.scale(0.5));
    for (shared.slice(u16, case.rows * case.width)) |*cell| cell.* = math.b(rng.scale(0.5));
    // Distinct ids per row, in a scrambled order, so the kernel's sort has work to do.
    for (0..case.rows) |row| {
        var used: [slots]u32 = undefined;
        for (0..slots) |s| {
            var pick: u32 = undefined;
            var ok = false;
            while (!ok) {
                pick = if (case.invalid and s == slots - 1) @as(u32, experts + 1) else @intCast(rng.next() % experts);
                ok = true;
                for (used[0..s]) |u| {
                    if (u == pick) ok = false;
                }
            }
            used[s] = pick;
        }
        // A fixed scramble keeps the ids reproducible without depending on the RNG's draw order.
        const order = [_]usize{ 3, 0, 5, 1, 4, 2 };
        for (&order, 0..) |slot, at| ids.slice(u32, case.rows * slots)[row * slots + at] = used[slot];
    }
    const cb = queue.commandBuffer();
    const enc = cb.compute(.serial);
    ops.encodeCombine(enc, p, view(h.view(routed)), view(h.view(shared)), view(h.view(ids)), view(h.view(y)), .{
        .rows = @intCast(case.rows),
        .width = @intCast(case.width),
        .routed_row_stride = @intCast(routed_row_stride),
        .slot_stride = @intCast(case.width),
        .shared_stride = @intCast(case.width),
        .y_stride = @intCast(case.width),
    });
    enc.end();
    cb.commit();
    cb.wait();
    if (cb.failure() != null) return error.CombineGpuFailure;

    const routed_cells = routed.slice(u16, case.rows * routed_row_stride);
    const shared_cells = shared.slice(u16, case.rows * case.width);
    const id_cells = ids.slice(u32, case.rows * slots);
    const out = y.slice(u16, case.rows * case.width);
    var unequal: usize = 0;
    var nan_cells: usize = 0;
    var worst_relative: f64 = 0;
    for (0..case.rows) |row| {
        var rank: [slots]usize = undefined;
        for (&rank, 0..) |*r, i| r.* = i;
        // The kernel's own insertion sort of the six slot indices by ascending expert ID.
        var a: usize = 1;
        while (a < slots) : (a += 1) {
            const slot = rank[a];
            var j = a;
            while (j > 0 and id_cells[row * slots + rank[j - 1]] > id_cells[row * slots + slot]) : (j -= 1) rank[j] = rank[j - 1];
            rank[j] = slot;
        }
        const invalid_row = case.invalid;
        for (0..case.width) |i| {
            if (invalid_row) {
                if (std.math.isNan(math.f(out[row * case.width + i])) or out[row * case.width + i] == math.b(std.math.nan(f32))) nan_cells += 1;
                continue;
            }
            // The documented order in FP32: the six routed cells by ascending expert ID, then shared.
            var ordered: f32 = 0;
            var ideal: f64 = 0;
            for (rank) |slot| {
                const cell: f32 = math.f(routed_cells[row * routed_row_stride + slot * case.width + i]);
                ordered += cell;
                ideal += @as(f64, cell);
            }
            const total = ordered + math.f(shared_cells[row * case.width + i]);
            ideal += @as(f64, math.f(shared_cells[row * case.width + i]));
            const expected = math.b(total);
            if (bf16Unequal(out[row * case.width + i], expected)) unequal += 1;
            const scale = @abs(ideal);
            if (scale > 1e-20) worst_relative = @max(worst_relative, @abs(@as(f64, math.f(expected)) - ideal) / scale);
        }
    }
    return .{ .unequal = unequal, .cells = case.rows * case.width, .worst_relative = worst_relative, .nan_cells = nan_cells };
}

test "the combine sums six slots by ascending expert ID and rounds once, against an FP64 ideal" {
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const queue = try device.queue();
    defer queue.deinit();
    var p = try ops.Pipelines.load(device);
    defer p.deinit();
    var total_unequal: usize = 0;
    var total_cells: usize = 0;
    var worst: f64 = 0;
    const cases = [_]Case{
        .{ .name = "one_row", .rows = 1, .width = 5120, .seed = 0xc0_1e01 },
        .{ .name = "four_rows", .rows = 4, .width = 5120, .seed = 0xc0_1e02 },
        .{ .name = "narrow", .rows = 3, .width = 1024, .seed = 0xc0_1e03 },
        .{ .name = "wide", .rows = 2, .width = 8192, .seed = 0xc0_1e04 },
    };
    for (cases) |case| {
        const r = try combineCase(device, queue, &p, case);
        total_unequal += r.unequal;
        total_cells += r.cells;
        worst = @max(worst, r.worst_relative);
        std.debug.print("COMBINE {{\"case\":\"{s}\",\"rows\":{d},\"width\":{d},\"cells\":{d},\"unequal\":{d},\"worst_relative\":{e}}}\n", .{ case.name, case.rows, case.width, r.cells, r.unequal, r.worst_relative });
        try std.testing.expectEqual(@as(usize, 0), r.unequal);
        try std.testing.expect(r.worst_relative < 1e-2);
    }
    // The invalid-ID contract: an expert ID past the table poisons the whole row with NaN.
    const bad = try combineCase(device, queue, &p, .{ .name = "invalid_id", .rows = 2, .width = 1024, .seed = 0xc0_1e05, .invalid = true });
    std.debug.print("COMBINE {{\"case\":\"invalid_id\",\"rows\":2,\"width\":1024,\"nan_cells\":{d},\"cells\":{d}}}\n", .{ bad.nan_cells, bad.cells });
    try std.testing.expectEqual(bad.cells, bad.nan_cells);
    std.debug.print("COMBINE GATE {{\"cells\":{d},\"unequal\":{d},\"worst_relative\":{e}}}\n", .{ total_cells, total_unequal, worst });
}
