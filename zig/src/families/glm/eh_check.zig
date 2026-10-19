//! The tensor-unit eh_proj stays no worse than the row kernel on an fp64 reference, per column and in aggregate.

//! Decode (1 row) and prompt (2048 and 8192 rows) are separate shapes on identical input rows.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");
const wide = @import("wide.zig");

const n: usize = 4096; // D, kept for every shape
const k: usize = 8192;
const max_rows: usize = 8192;
const col_step: usize = 16;
const few: f64 = 2; // attn_check's pair bar, and the ~2-step fp64 sanity bound
// Floor under the 2-step bar: an fp32 reduction of this k can move a small column by more than one step.
const acc_floor: f64 = 0.1;

fn bf16(v: f32) u16 {
    const u: u32 = @bitCast(v);
    return @intCast((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

fn f32of(h: u16) f32 {
    return @bitCast(@as(u32, h) << 16);
}

fn limit(scale: f64) f64 {
    return @max(few * scale / 128.0, acc_floor);
}

// At the cast floor the pair may differ by a few bf16 steps; otherwise nax must not exceed the row kernel.
fn pointOk(es: f64, en: f64, pair_abs: f64, scale: f64) bool {
    const step = scale / 128.0;
    if (es > limit(scale) or en > limit(scale)) return false;
    if (es <= @max(step, acc_floor)) return pair_abs <= limit(scale);
    return en <= es;
}

fn addRow(list: []usize, len: usize, r: usize, rows: usize) usize {
    if (r >= rows) return len;
    for (list[0..len]) |e| if (e == r) return len;
    list[len] = r;
    return len + 1;
}

// 32 rows across the shape, plus the 4 corner rows; a short shape keeps the rows it has.
fn sampledRows(rows: usize, list: *[36]usize) usize {
    var len: usize = 0;
    const take = @min(rows, 32);
    if (take <= 1) {
        len = addRow(list, len, 0, rows);
    } else {
        const span = rows - 1;
        for (0..take) |i| len = addRow(list, len, i * span / (take - 1), rows);
    }
    len = addRow(list, len, 0, rows);
    if (rows >= 2) len = addRow(list, len, 1, rows);
    if (rows >= 3) len = addRow(list, len, rows - 2, rows);
    if (rows >= 1) len = addRow(list, len, rows - 1, rows);
    return len;
}

fn dot(x: []const u16, w: []const u16, r: usize, j: usize) f64 {
    var s: f64 = 0;
    for (x[r * k ..][0..k], w[j * k ..][0..k]) |a, b| s += @as(f64, f32of(a)) * f32of(b);
    return s;
}

fn ehCheck() !void {
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const queue = try device.queue();
    defer queue.deinit();
    const a = std.testing.allocator;
    // glm_rows.metal is not standalone: wide.zig's prefix compiles it. Both kernels come from that one file.
    const rows_src = try std.mem.concat(a, u8, &.{
        sources.core_row_projection,
        wide.glue_math,
        sources.glm_rows,
    });
    defer a.free(rows_src);
    const lib = try mtl.Library.fromSource(device, rows_src, mtl.CompileOptions.mlx());
    defer lib.deinit();
    const row_pipe = try mtl.Pipeline.init(device, lib, "glm_rows_bf16", false);
    defer row_pipe.deinit();
    const nax_pipe = try mtl.Pipeline.init(device, lib, "glm_rows_bf16_nax", false);
    defer nax_pipe.deinit();
    try std.testing.expect(row_pipe.maxThreads() >= 64);
    try std.testing.expect(nax_pipe.maxThreads() >= 8 * 32);

    const opts = mtl.ResourceOptions.shared;
    const bufs = [_]mtl.Buffer{
        try device.buffer(max_rows * k * 2, opts),
        try device.buffer(n * k * 2, opts),
        try device.buffer(max_rows * n * 2, opts),
        try device.buffer(max_rows * n * 2, opts),
    };
    defer for (bufs) |b| b.deinit();
    var prng = std.Random.DefaultPrng.init(23);
    const rnd = prng.random();
    const x = bufs[0].slice(u16, max_rows * k);
    const w = bufs[1].slice(u16, n * k);
    for (w) |*v| v.* = bf16(rnd.float(f32) * 2 - 1);
    for (x) |*v| v.* = bf16(rnd.float(f32) * 2 - 1);

    // Decode is the 1-row shape; prompt chunks are 2048 and 8192. Same leading rows, separate launches.
    for ([_]usize{ 1, 2048, 8192 }) |rows| {
        @memset(bufs[2].slice(u16, rows * n), 0);
        @memset(bufs[3].slice(u16, rows * n), 0);
        const dims = wide.RowDims{
            .k = @intCast(k),
            .n = @intCast(n),
            .group = 0,
            .rows = @intCast(rows),
            .ldx = @intCast(k),
            .ldo = @intCast(n),
        };
        const cb = queue.commandBuffer();
        const enc = cb.compute(.serial);
        enc.setPipeline(row_pipe);
        enc.setBuffer(bufs[0], 0, 0);
        enc.setBuffer(bufs[1], 0, 1);
        enc.setValue(dims, 4);
        enc.setBuffer(bufs[2], 0, 5);
        enc.dispatchGroups(mtl.Size.of((n + 7) / 8, 1, 1), mtl.Size.of(64, 1, 1));
        enc.setPipeline(nax_pipe);
        enc.setBuffer(bufs[0], 0, 0);
        enc.setBuffer(bufs[1], 0, 1);
        enc.setValue(dims, 4);
        enc.setBuffer(bufs[3], 0, 5);
        enc.dispatchGroups(mtl.Size.of((n + 63) / 64, (rows + 7) / 8, 1), mtl.Size.of(8 * 32, 1, 1));
        enc.end();
        cb.commit();
        cb.wait();
        try std.testing.expect(cb.failure() == null);

        const row_out = bufs[2].slice(u16, rows * n);
        const nax_out = bufs[3].slice(u16, rows * n);
        var worst_steps: f64 = 0;
        var worst_pair: f64 = 0;
        var pair_ok = true;
        var pair_i: usize = 0;
        for (row_out, nax_out, 0..) |sv, nv, i| {
            const scal: f64 = f32of(sv);
            const got: f64 = f32of(nv);
            const diff = @abs(scal - got);
            const scale = @max(@abs(scal), @abs(got));
            const step = scale / 128.0;
            worst_pair = @max(worst_pair, diff);
            if (few * step >= acc_floor and step != 0) worst_steps = @max(worst_steps, diff / step);
            if (pair_ok and diff > limit(scale)) {
                pair_ok = false;
                pair_i = i;
            }
        }

        var list: [36]usize = undefined;
        const n_list = sampledRows(rows, &list);
        try std.testing.expect(n_list > 0);
        var worst_scal: f64 = 0;
        var worst_nax: f64 = 0;
        var ref_ok = true;
        var fail_r: usize = 0;
        var fail_c: usize = 0;
        var fail_es: f64 = 0;
        var fail_en: f64 = 0;
        var fail_scale: f64 = 0;
        for (list[0..n_list]) |r| {
            var j: usize = 0;
            while (j < n) : (j += col_step) {
                const ref = dot(x, w, r, j);
                const scal: f64 = f32of(row_out[r * n + j]);
                const got: f64 = f32of(nax_out[r * n + j]);
                const es = @abs(scal - ref);
                const en = @abs(got - ref);
                const scale = @max(@abs(ref), @max(@abs(scal), @abs(got)));
                worst_scal = @max(worst_scal, es);
                worst_nax = @max(worst_nax, en);
                if (ref_ok and !pointOk(es, en, @abs(scal - got), scale)) {
                    ref_ok = false;
                    fail_r = r;
                    fail_c = j;
                    fail_es = es;
                    fail_en = en;
                    fail_scale = scale;
                }
            }
        }
        std.debug.print(
            "eh nax vs scal rows={d} k={d} n={d}: worst-col scal={d:.2} nax={d:.2} max={d:.2} pair-abs={d:.3} fp64-abs bf16-steps={d:.2} floor-cols; sampled fp64: cols step 16 x 32 rows + 4 corner rows (used {d} rows x {d} cols)\n",
            .{
                rows,
                k,
                n,
                worst_scal,
                worst_nax,
                @max(worst_scal, worst_nax),
                worst_pair,
                worst_steps,
                n_list,
                n / col_step,
            },
        );
        if (!pair_ok) std.debug.print(
            "eh nax pair fail rows={d} r={d} c={d} scal={d:.4} nax={d:.4} fp64-abs\n",
            .{ rows, pair_i / n, pair_i % n, f32of(row_out[pair_i]), f32of(nax_out[pair_i]) },
        );
        if (!ref_ok) std.debug.print(
            "eh nax ref fail rows={d} r={d} c={d} scal={d:.4} nax={d:.4} scale={d:.4} fp64-abs\n",
            .{ rows, fail_r, fail_c, fail_es, fail_en, fail_scale },
        );
        try std.testing.expect(pair_ok);
        try std.testing.expect(ref_ok);
    }
}

test "eh_proj on the tensor units: fp64 error no worse than the row kernel, decode and prompt apart" {
    try ehCheck();
}
