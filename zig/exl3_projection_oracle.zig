//! Bounded scalar-format decoding and FP64 dots; no Metal decoder, butterfly mirror or full weight matrix.
const std = @import("std");
const ex = @import("exl3_projection");

fn pointer(comptime T: type, v: ex.View) [*]const T {
    return @ptrCast(@alignCast(v.buffer.contents() + v.offset));
}
fn scale(v: ex.View, kind: ex.ScaleFormat, i: usize) f64 {
    if (kind == .packed_signs) return if ((pointer(u16, v)[i / 16] >> @intCast(i % 16)) & 1 == 1) -1 else 1;
    return @floatCast(pointer(f16, v)[i]);
}
fn output(v: ex.View, kind: ex.Output, i: usize) f64 {
    return if (kind == .f32) pointer(f32, v)[i] else @as(f32, @bitCast(@as(u32, pointer(u16, v)[i]) << 16));
}

fn rotateInput(m: ex.Matrix, x: ex.View, row: usize, stride: usize, rotated: []f16) void {
    for (0..m.k / 128) |b| {
        var block: [128]f64 = undefined;
        for (&block, 0..) |*v, i| {
            const j = b * 128 + i;
            const input: f32 = @bitCast(@as(u32, pointer(u16, x)[row * stride + j]) << 16);
            v.* = @as(f64, input) * scale(m.suh, m.suh_format, j);
        }
        ex.format.rotate(&block, 1, 128, 1);
        for (block, 0..) |v, i| rotated[b * 128 + i] = @floatCast(v);
    }
}

fn referenceBlock(m: ex.Matrix, rotated: []const f16, base: usize, out: *[128]f64) !void {
    @memset(out, 0);
    var placement: [256]ex.format.Placement = undefined;
    ex.format.tilePlacement(&placement);
    const k_stride = try m.kStride();
    for (0..m.k / 16) |tk| {
        for (0..8) |tn| {
            const bytes = tk * k_stride + (base / 16 + tn) * m.nStride();
            const tile: [*]const i16 = @ptrCast(@alignCast(m.trellis.buffer.contents() + m.trellis.offset + bytes));
            var states: [256]u32 = undefined;
            ex.format.tileStates(m.width, tile[0..m.width.tileWords()], &states);
            for (states, placement) |state, at| {
                const weight: f64 = @floatCast(ex.format.codebookValue(m.codebook, @intCast(state)));
                out[tn * 16 + at.col] += @as(f64, rotated[tk * 16 + at.row]) * weight;
            }
        }
    }
    ex.format.rotate(out, 1, 128, 1);
    for (out, 0..) |*v, i| {
        v.* *= scale(m.svh, m.svh_format, base + i);
        if (m.bias) |bias| v.* += @as(f64, pointer(f16, bias)[base + i]);
    }
}

const Stats = struct {
    squared_error: f64 = 0,
    squared_reference: f64 = 0,
    max_abs: f64 = 0,
    count: usize = 0,

    fn add(s: *Stats, got: f64, want: f64) !void {
        if (!std.math.isFinite(got) or !std.math.isFinite(want)) return error.NonFiniteOracleOutput;
        const difference = got - want;
        s.squared_error += difference * difference;
        s.squared_reference += want * want;
        s.max_abs = @max(s.max_abs, @abs(difference));
        s.count += 1;
    }
    fn nrms(s: Stats) f64 {
        if (s.squared_reference == 0) return if (s.squared_error == 0) 0 else std.math.inf(f64);
        return std.math.sqrt(s.squared_error / s.squared_reference);
    }
};

pub fn verify(gpa: std.mem.Allocator, matrices: []const ex.Matrix, x: ex.View, ys: [2]ex.View, rows: ex.Rows) !void {
    const n = matrices[0].n;
    const rotated = try gpa.alloc(f16, matrices[0].k);
    defer gpa.free(rotated);
    var aggregate = [_]Stats{ .{}, .{} };
    var worst_abs: f64 = 0;
    var worst_column: usize = 0;
    var worst_nrms: f64 = 0;
    var worst_nrms_column: usize = 0;
    var baseline_worst_abs: f64 = 0;
    var baseline_worst_nrms: f64 = 0;
    // Compact fixtures check every row and column; large shapes sample three full rotation blocks and two rows.
    const blocks: usize = @min(n / 128, 3);
    const checked_rows = if (n == 128) rows.count else @min(rows.count, 2);
    for (0..blocks) |sample| {
        const base = (if (blocks == 1) 0 else sample * (n / 128 - 1) / (blocks - 1)) * 128;
        var columns: [2][128]Stats = .{ @splat(.{}), @splat(.{}) };
        for (0..checked_rows) |r| {
            const expert = if (rows.routes) |routes| pointer(u32, routes.ids)[r] else 0;
            if (expert >= matrices.len) return error.InvalidOracleExpert;
            const m = matrices[expert];
            const source = if (rows.input_ids) |ids| pointer(u32, ids)[r] else r / (if (rows.routes) |routes| routes.repeat else @as(usize, 1));
            const dest = if (rows.output_ids) |ids| pointer(u32, ids)[r] else r;
            rotateInput(m, x, source, rows.x_stride, rotated);
            var expected: [128]f64 = undefined;
            try referenceBlock(m, rotated, base, &expected);
            for (expected, 0..) |want, c| {
                for (0..2) |variant| {
                    const got = output(ys[variant], rows.output, dest * rows.y_stride + base + c);
                    try aggregate[variant].add(got, want);
                    try columns[variant][c].add(got, want);
                }
            }
        }
        for (0..128) |c| {
            const before = columns[0][c];
            const after = columns[1][c];
            if (after.squared_error > before.squared_error or after.max_abs > before.max_abs)
                return error.WorseOracleColumn;
            baseline_worst_abs = @max(baseline_worst_abs, before.max_abs);
            baseline_worst_nrms = @max(baseline_worst_nrms, before.nrms());
            if (after.max_abs > worst_abs) {
                worst_abs = after.max_abs;
                worst_column = base + c;
            }
            if (after.nrms() > worst_nrms) {
                worst_nrms = after.nrms();
                worst_nrms_column = base + c;
            }
        }
    }
    std.debug.print("EXL3 oracle=scalar_format_fp64_dot_fp16_rotated_input k={d} n={d} " ++
        "book={s} halves={d} output={s} sampled_rows={d} sampled_columns={d} values={d} " ++
        "baseline_nrms={d:.9} candidate_nrms={d:.9} worst_sampled_column={d} worst_sampled_abs={d:.9} " ++
        "worst_sampled_column_nrms_index={d} worst_sampled_column_nrms={d:.9} " ++
        "baseline_worst_sampled_abs={d:.9} baseline_worst_sampled_column_nrms={d:.9}\n", .{ matrices[0].k, n, @tagName(matrices[0].codebook), matrices[0].width.halves, @tagName(rows.output), checked_rows, blocks * 128, aggregate[1].count, aggregate[0].nrms(), aggregate[1].nrms(), worst_column, worst_abs, worst_nrms_column, worst_nrms, baseline_worst_abs, baseline_worst_nrms });
    if (aggregate[1].nrms() > 0.01) return error.OracleNrmsFailure;
}

test "oracle error statistics retain cancellation columns instead of averaging them away" {
    var stats: Stats = .{};
    try stats.add(1.5, 1);
    try stats.add(-0.5, -1);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), stats.nrms(), 1e-15);
    try std.testing.expectEqual(@as(f64, 0.5), stats.max_abs);
    try std.testing.expectError(error.NonFiniteOracleOutput, stats.add(std.math.nan(f64), 1));
}
