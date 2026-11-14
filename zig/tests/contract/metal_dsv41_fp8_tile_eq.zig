//! The FP8 tile against the row kernel on the model's own shapes: rows at once == one row a launch, every form.
const std = @import("std");
const tf = @import("tensorfold");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const ref = @import("dsv41_fp8_ref.zig");

const c = contract.spec;
const tile = tf.dsv41_metal.fp8_tile;
const rows_k = tf.dsv41_metal.fp8_rows;

/// K x N of every dense FP8 projection the decode forward runs (wo_a once a group, the drafter's main_proj last).
pub const shapes = [_][2]u32{
    .{ 5120, 1280 }, .{ 5120, 512 },  .{ 1280, 32768 }, .{ 4096, 1024 },  .{ 8192, 5120 },
    .{ 5120, 2304 }, .{ 2304, 5120 }, .{ 1280, 4096 },  .{ 15360, 5120 },
};
const forms = [_]tile.Form{
    .{ .nj = 1, .sgs = 1 },  .{ .nj = 1, .sgs = 2 }, .{ .nj = 1, .sgs = 4 }, .{ .nj = 1, .sgs = 8 },
    .{ .nj = 1, .sgs = 16 }, .{ .nj = 2, .sgs = 1 }, .{ .nj = 2, .sgs = 4 }, .{ .nj = 2, .sgs = 8 },
};
const all = 17;

fn at(r: rig_.Ref, extra: usize) tile.Ref {
    return .{ .buf = r.buf, .off = r.off + extra };
}

fn rat(r: rig_.Ref, extra: usize) rows_k.Ref {
    return .{ .buf = r.buf, .off = r.off + extra };
}

/// Non-finite inputs and weights: NaN and both infinities in four rows, NaN codes, a NaN scale and a huge one.
fn plant(f: ref.Fp8) void {
    f.fill(.nonfinite, 0);
    const k = f.shape.k;
    const groups = k / 32;
    f.codes[3 * k + 7] = 0x7f;
    f.codes[40 * k + k / 2] = 0xff;
    f.scales[groups - 1] = 255;
    f.scales[f.scales.len / 2] = 250;
    f.x[9 * k + 5] = c.toBf16(std.math.inf(f32));
    f.x[14 * k + k - 3] = c.toBf16(std.math.nan(f32));
}

fn validForm(k: u32, n: u32, f: tile.Form) bool {
    const pad = std.math.ceilPowerOfTwo(u32, k / 32) catch unreachable;
    return n % (8 * f.nj) == 0 and f.sgs * f.nj <= tile.max_lanes and f.sgs <= pad and pad / f.sgs <= 512;
}

/// Returns the launches whose bytes differ from the row kernel's (NaN payloads included).
pub fn run(a: std.mem.Allocator, rig: *rig_.Rig, pipes: *const tile.Pipelines, rows: *const rows_k.Pipelines, seed: u64) !usize {
    _ = seed;
    var bad: usize = 0;
    var launches: usize = 0;
    var nan_outputs: usize = 0;
    for (shapes) |kn| for ([_]tile.Out{ .f32, .bf16 }) |out| {
        rig.reset();
        const k = kn[0];
        const n = kn[1];
        const f = try ref.Fp8.init(a, .{ .rows = all, .n = n, .k = k });
        defer f.deinit();
        plant(f);
        const es: usize = if (out == .f32) 4 else 2;
        const bytes = all * @as(usize, n) * es;
        const x = try rig.input(std.mem.sliceAsBytes(f.x));
        const w = try rig.input(f.codes);
        const s = try rig.input(f.scales);
        const want = try rig.output(bytes);
        const got = try rig.output(bytes);
        const one: rows_k.Call = .{ .k = k, .n = n, .rows = 1, .split = rows_k.splitOf(k), .out = if (out == .f32) .f32 else .bf16 };
        {
            const pass = rig.begin();
            for (0..all) |r| try rows_k.encodeWhole(rows, pass.e, rat(x, 2 * r * k), rat(w, 0), rat(s, 0), rat(want, es * r * n), one, .{});
            try rig.finish(pass);
        }
        const base = rig.bytes(want, bytes);
        for (0..all * @as(usize, n)) |i| {
            if (out == .f32 and std.math.isNan(std.mem.bytesToValue(f32, base[4 * i ..][0..4]))) nan_outputs += 1;
        }
        // every row count and form from row 0; then 1-15 rows from row 1 (a later segment's rows)
        for ([_]usize{ 0, 1 }) |first| for (1..tile.max_rows + 1 - first) |r| for (forms) |fm| {
            if (!validForm(k, n, fm)) continue;
            @memset(rig.bytes(got, bytes), 0xff);
            const pass = rig.begin();
            const cl: tile.Call = .{ .k = k, .n = n, .rows = @intCast(r), .out = out };
            try tile.encodeAs(pipes, pass.e, at(x, 2 * first * k), at(w, 0), at(s, 0), at(got, es * first * n), cl, .{}, fm);
            try rig.finish(pass);
            launches += 1;
            const span = es * n;
            if (!std.mem.eql(u8, base[first * span ..][0 .. r * span], rig.bytes(got, bytes)[first * span ..][0 .. r * span])) {
                bad += 1;
                if (bad <= 8) std.debug.print("FAIL dsv41_fp8_tile equality {d}x{d} {s}: rows {d}..{d} form c{d}s{d}\n", .{ k, n, @tagName(out), first, first + r, 8 * fm.nj, fm.sgs });
            }
        };
        // the 16/17 seam: 16 rows on the tile, the 17th on the row kernel, as the forward splits a 17-row window
        {
            @memset(rig.bytes(got, bytes), 0xff);
            const pass = rig.begin();
            try tile.encode(pipes, pass.e, at(x, 0), at(w, 0), at(s, 0), at(got, 0), .{ .k = k, .n = n, .rows = 16, .out = out }, .{});
            try rows_k.encodeWhole(rows, pass.e, rat(x, 2 * 16 * k), rat(w, 0), rat(s, 0), rat(got, es * 16 * n), one, .{});
            try rig.finish(pass);
            launches += 1;
            if (!std.mem.eql(u8, base, rig.bytes(got, bytes))) bad += 1;
        }
        // a batch in grid y: three launches of 5 rows over the same weights, rows 0-4, 6-10 and 12-16
        {
            @memset(rig.bytes(got, bytes), 0xff);
            const pass = rig.begin();
            const b: tile.Batch = .{ .n = 3, .xs = @intCast(2 * 6 * k), .ys = @intCast(es * 6 * n) };
            try tile.encode(pipes, pass.e, at(x, 0), at(w, 0), at(s, 0), at(got, 0), .{ .k = k, .n = n, .rows = 5, .out = out }, b);
            try rig.finish(pass);
            launches += 1;
            for ([_]usize{ 0, 6, 12 }) |r0| {
                const span = es * n;
                if (!std.mem.eql(u8, base[r0 * span ..][0 .. 5 * span], rig.bytes(got, bytes)[r0 * span ..][0 .. 5 * span])) bad += 1;
            }
        }
    };
    const word = if (bad == 0) "PASS" else "FAIL";
    std.debug.print("{s} dsv41_fp8_tile equality: {d} model shapes x f32/bf16, rows 1-16 from row 0 and 1-15 from row 1 at every form, the 16/17 seam and a 3-launch batch ({d} launches, NaN/Inf planted, {d} NaN outputs in the f32 baselines) byte-equal to fp8_rows one row a launch: {d} differ\n", .{ word, shapes.len, launches, nan_outputs, bad });
    return bad;
}
