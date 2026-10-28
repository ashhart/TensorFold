//! exl3-prefill dense: the prompt tiles of a dense EXL3 projection, bit-equal to the decode kernel at every row count.

const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("../../check.zig");
const exl3fmt = core.exl3_format;
const exl3 = cuda.exl3;
const Gpu = check.Gpu;
const packs = @import("exl3_prefill_pack.zig");

const Pack = packs.Pack;
const Timer = packs.Timer;
const bf16Bits = packs.bf16Bits;
const bfValue = packs.bfValue;

// ---------------------------------------------------------------------------------------------------- dense

const DenseCall = struct {
    de: *exl3.Dense,
    s: cuda.Stream,
    w: exl3.Weight,
    x: u64,
    y: u64,
    rows: usize,
    exact: bool = false,
    pub fn go(c: DenseCall) anyerror!void {
        if (c.exact) return c.de.promptExact(c.s, c.w, c.x, c.rows, c.y, .bf16);
        try c.de.prompt(c.s, c.w, c.x, .bf16, c.rows, c.y, .bf16);
    }
};

/// y [rows, N] = x [rows, K] . diag(suh) H Wq H diag(svh) in float64.
fn exact(a: std.mem.Allocator, trellis: []const i16, suh: []const f16, svh: []const f16, tk: usize, tn: usize, halves: u16, x: []const f64, rows: usize, y: []f64) !void {
    const k = tk * 16;
    const n = tn * 16;
    const wq = try a.alloc(f16, k * n);
    defer a.free(wq);
    exl3fmt.unpack(.{ .halves = halves }, .mul1, trellis, tk, tn, wq);
    const xs = try a.alloc(f64, rows * k);
    defer a.free(xs);
    for (0..rows) |r| for (0..k) |i| {
        xs[r * k + i] = x[r * k + i] * @as(f64, @floatCast(suh[i]));
    };
    exl3fmt.rotate(xs, rows, k, 1);
    @memset(y, 0);
    for (0..k) |ki| {
        const row_w = wq[ki * n ..][0..n];
        for (0..rows) |r| {
            const xv = xs[r * k + ki];
            const yr = y[r * n ..][0..n];
            for (yr, row_w) |*o, w| o.* += xv * @as(f64, @floatCast(w));
        }
    }
    exl3fmt.rotate(y, rows, n, 1);
    for (0..rows) |r| for (0..n) |c| {
        y[r * n + c] *= @as(f64, @floatCast(svh[c]));
    };
}

fn copyAs(comptime T: type, a: std.mem.Allocator, bytes: []const u8) ![]T {
    const out = try a.alloc(T, bytes.len / @sizeOf(T));
    @memcpy(std.mem.sliceAsBytes(out), bytes[0 .. out.len * @sizeOf(T)]);
    return out;
}

pub fn denseRun(gpu: Gpu, pack: *const Pack, s: cuda.Stream) !void {
    const a = gpu.gpa;
    var de = try exl3.Dense.load(gpu.d);
    defer de.unload();
    var t = try Timer.init(gpu.d);
    defer t.deinit();
    const names = [_][]const u8{
        "model.language_model.layers.0.linear_attn.in_proj_qkv",
        "model.language_model.layers.0.linear_attn.in_proj_z",
        "model.language_model.layers.0.linear_attn.out_proj",
        "model.language_model.layers.3.self_attn.q_proj",
        "model.language_model.layers.3.self_attn.k_proj",
        "model.language_model.layers.3.self_attn.o_proj",
        "model.language_model.layers.3.mlp.shared_expert.gate_proj",
        "model.language_model.layers.3.mlp.shared_expert.down_proj",
    };
    const row_counts = [_]usize{ 16, 63, 64, 65, 128, 256, 1024, 2048 };
    const max_rows: usize = 2048;
    var prng = std.Random.DefaultPrng.init(0xde45e);
    const rng = prng.random();
    var sum_ref: [row_counts.len]f64 = @splat(0);
    var sum_tiles: [row_counts.len]f64 = @splat(0);
    for (names) |name| {
        const tr = try pack.get("{s}.trellis", .{name});
        const su = try pack.get("{s}.suh", .{name});
        const sv = try pack.get("{s}.svh", .{name});
        const tk = tr.dim(0);
        const tn = tr.dim(1);
        const halves: u16 = @intCast(tr.dim(2) / 8);
        const k = tk * 16;
        const n = tn * 16;
        const trellis = try copyAs(i16, a, tr.bytes);
        defer a.free(trellis);
        const suh = try copyAs(f16, a, su.bytes);
        defer a.free(suh);
        const svh = try copyAs(f16, a, sv.bytes);
        defer a.free(svh);
        var trd = try cuda.DeviceBuffer.fromHost(gpu.d, tr.bytes);
        defer trd.free();
        var sud = try cuda.DeviceBuffer.fromHost(gpu.d, su.bytes);
        defer sud.free();
        var svd = try cuda.DeviceBuffer.fromHost(gpu.d, sv.bytes);
        defer svd.free();
        const strides = exl3.strides(k, n, halves, false);
        const w: exl3.Weight = .{ .trellis = trd.ptr, .suh = sud.ptr, .svh = svd.ptr, .k = @intCast(k), .n = @intCast(n), .halves = halves, .codebook = .mul1, .stride_k = strides.k, .stride_nb = strides.nb };
        const tiles_ok = if (exl3.Dense.candidateWeight(w)) |_| true else |_| false;
        const xh = try a.alloc(u16, max_rows * k);
        defer a.free(xh);
        for (xh) |*b| b.* = bf16Bits((rng.float(f32) - 0.5) * 2.0);
        var x = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(xh));
        defer x.free();
        var yr = try cuda.DeviceBuffer.alloc(gpu.d, max_rows * n * 2);
        defer yr.free();
        var yt = try cuda.DeviceBuffer.alloc(gpu.d, max_rows * n * 2);
        defer yt.free();
        // f64 of rows 0 and 1
        const xf = try a.alloc(f64, 2 * k);
        defer a.free(xf);
        for (xf, xh[0 .. 2 * k]) |*o, b| o.* = bfValue(b);
        const want = try a.alloc(f64, 2 * n);
        defer a.free(want);
        try exact(a, trellis, suh, svh, tk, tn, halves, xf, 2, want);
        std.debug.print("  {s} K={d} N={d} halves={d}{s}\n", .{ name, k, n, halves, if (tiles_ok) "" else " (tiles refuse it)" });
        for (row_counts, 0..) |rows, ri| {
            de.implementation = .reference;
            const rc: DenseCall = .{ .de = &de, .s = s, .w = w, .x = x.ptr, .y = yr.ptr, .rows = rows };
            const ms_ref = try t.best(s, 3, rc, DenseCall.go);
            sum_ref[ri] += ms_ref;
            var ms_tiles: f64 = std.math.nan(f64);
            var diff: f64 = std.math.nan(f64);
            var err_ref: f64 = 0;
            var err_tiles: f64 = std.math.nan(f64);
            var peak: f64 = 0;
            const got_r = try a.alloc(u16, rows * n);
            defer a.free(got_r);
            try yr.download(0, std.mem.sliceAsBytes(got_r));
            for (got_r[0 .. 2 * n], want) |g, wv| {
                err_ref = @max(err_ref, @abs(bfValue(g) - wv));
                peak = @max(peak, @abs(wv));
            }
            if (tiles_ok) {
                try de.candidates(w.k);
                const tc: DenseCall = .{ .de = &de, .s = s, .w = w, .x = x.ptr, .y = yt.ptr, .rows = rows, .exact = true };
                ms_tiles = try t.best(s, 3, tc, DenseCall.go);
                sum_tiles[ri] += ms_tiles;
                const got_t = try a.alloc(u16, rows * n);
                defer a.free(got_t);
                try yt.download(0, std.mem.sliceAsBytes(got_t));
                var unequal: usize = 0;
                for (got_r, got_t) |g, h| unequal += @intFromBool(g != h);
                diff = @floatFromInt(unequal);
                err_tiles = 0;
                for (got_t[0 .. 2 * n], want) |g, wv| err_tiles = @max(err_tiles, @abs(bfValue(g) - wv));
                try check.expect(unequal == 0, "{s}: the prompt tiles' bytes are the decode kernel's at {d} rows ({d} of {d} outputs differ)", .{ name, rows, unequal, got_r.len });
            } else sum_tiles[ri] += ms_ref;
            std.debug.print("    rows {d:5}: reference {d:8.4} ms ({d:6.2} us/row)  tiles {d:8.4} ms ({d:6.2} us/row)  outputs differing from the decode kernel's {d:.0}  rows 0-1 vs f64: ref {e:.2} tiles {e:.2} (rel to peak {d:.3})\n", .{ rows, ms_ref, 1e3 * ms_ref / @as(f64, @floatFromInt(rows)), ms_tiles, 1e3 * ms_tiles / @as(f64, @floatFromInt(rows)), diff, err_ref / peak, err_tiles / peak, peak });
        }
    }
    std.debug.print("  sum over the {d} shapes (ms): ", .{names.len});
    for (row_counts, sum_ref, sum_tiles) |r, x, y| std.debug.print("rows {d}: reference {d:.3} best {d:.3}; ", .{ r, x, y });
    std.debug.print("\n", .{});
    check.pass("dense prompt tiles are the decode kernel's bytes at {d} row counts, timed", .{row_counts.len});
}
