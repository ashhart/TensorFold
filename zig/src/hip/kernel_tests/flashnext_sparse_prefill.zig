//! Flash Next's sparse prefill attention, the scalar and the WMMA kernel, against fp64, on idx_select's lists.
const std = @import("std");
const rig = @import("rig.zig");
const fx = @import("flashnext.zig");
const Rig = rig.Rig;
const gpa = rig.gpa;
const ptr = fx.ptr;

test "sparse prefill: the scalar and the WMMA kernels match fp64 on full and tail slices" {
    const t = try Rig.open(1 << 22);
    defer t.close();
    // 12 query heads a kv head over a full slice, and 6 over a slice of rows that is not a multiple of 4
    try case(t, 24, 2, 64);
    try case(t, 6, 1, 17); // rows % 4 != 0: a tile of two rows, and group 6 pads its 16-row tile
}

fn case(t: *Rig, comptime heads: usize, comptime kv_heads: usize, comptime rows: usize) !void {
    var rng: rig.Rng = .{ .state = 0x5EED + heads + rows };
    const dims = 256;
    const ratio = 4;
    const parts = 16;
    const cap = 1024;
    const stride = 520; // id_stride: more than the deepest row's count
    const scale: f32 = 1.0 / @sqrt(@as(f32, dims));
    const group = heads / kv_heads;

    const hk = try fx.acts(t, &rng, kv_heads * cap * dims, 1.0);
    defer gpa.free(hk);
    const hv = try fx.acts(t, &rng, kv_heads * cap * dims, 1.0);
    defer gpa.free(hv);
    const hq = try gpa.alloc(f32, rows * heads * dims);
    defer gpa.free(hq);
    for (hq) |*x| x.* = rng.unit() * 4.0;

    // Ends march forward and neighbours share blocks; blocks are distinct, a repeat would count twice in one kernel.
    var ids = try gpa.alloc(i32, rows * stride);
    defer gpa.free(ids);
    @memset(ids, 0);
    const count = try gpa.alloc(i32, rows);
    defer gpa.free(count);
    const sparse = try gpa.alloc(i32, rows);
    defer gpa.free(sparse);
    const ends = try gpa.alloc(i32, rows);
    defer gpa.free(ends);
    for (0..rows) |r| {
        // ends is not a multiple of ratio on three rows in four, so the incomplete tail is 1..3 tokens
        ends[r] = @as(i32, @intCast(200 + r * 12 + (r % 4)));
        const nb: usize = @as(usize, @intCast(ends[r])) / ratio; // the blocks the row may select
        if (r == 0) { // dense: the first ends[0] keys, unlisted
            sparse[r] = 0;
            count[r] = ends[r];
            continue;
        }
        sparse[r] = 1;
        const want: usize = @min(if (r < 8) @as(usize, 5) else 120, nb - 8);
        const start: usize = if (nb > want) (r * 3) % (nb - want) else 0; // neighbours differ by three blocks
        var w: usize = 0;
        for (0..want) |i| {
            const b = start + i; // distinct, increasing, all below nb
            for (0..ratio) |o| ids[r * stride + w + o] = @intCast(b * ratio + o);
            w += ratio;
        }
        var k = nb * ratio; // the incomplete tail past every complete block
        while (k < @as(usize, @intCast(ends[r]))) : (k += 1) {
            ids[r * stride + w] = @intCast(k);
            w += 1;
        }
        count[r] = @intCast(w);
    }

    var dk = try t.upload(hk);
    defer dk.free();
    var dv = try t.upload(hv);
    defer dv.free();
    var dq = try t.upload(hq);
    defer dq.free();
    var di = try t.upload(ids);
    defer di.free();
    var dc = try t.upload(count);
    defer dc.free();
    var ds = try t.upload(sparse);
    defer ds.free();
    var de = try t.upload(ends);
    defer de.free();
    var po = try t.alloc(rows * heads * parts * dims * 4);
    defer po.free();
    var pm = try t.alloc(rows * heads * parts * 2 * 4);
    defer pm.free();
    var out = try t.alloc(rows * heads * dims * 4);
    defer out.free();
    const ck: c_int = if (t.bf16) 2 else 1;

    var err: [2]f64 = undefined;
    for ([_]bool{ false, true }, 0..) |wmma, which| {
        t.on.sparse_wmma = wmma;
        try t.on.fn_sparse_attention(
            ptr(?[*]const f32, dq), ptr(?*const anyopaque, dk), ptr(?*const anyopaque, dv), ck,
            ptr(?[*]const i32, di), stride, ptr(?[*]const i32, dc), ptr(?[*]const i32, ds),
            ptr(?[*]const i32, de), ptr(?[*]f32, po), ptr(?[*]f32, pm), ptr(?[*]f32, out),
            @intCast(rows), @intCast(heads), @intCast(kv_heads), @intCast(dims), @intCast(cap), scale,
            @intCast(parts), t.stream.handle,
        );
        try t.stream.synchronize();
        const got = try rig.download(f32, out);
        defer gpa.free(got);
        err[which] = 0;
        for (0..rows) |r| {
            for (0..heads) |h| {
                const kvh = h / group;
                const listed = sparse[r] != 0;
                const n: usize = @intCast(count[r]);
                const end: usize = @intCast(ends[r]);
                const w = try gpa.alloc(f64, n);
                defer gpa.free(w);
                var top: f64 = -std.math.inf(f64);
                for (0..n) |j| {
                    const key: usize = if (listed) @intCast(ids[r * stride + j]) else j;
                    var sc: f64 = 0;
                    for (0..dims) |d| sc += @as(f64, hq[(r * heads + h) * dims + d] * scale) *
                        t.value(hk[(kvh * cap + key) * dims + d]);
                    w[j] = if (key < end) sc else -std.math.inf(f64);
                    top = @max(top, w[j]);
                }
                var total: f64 = 0;
                for (w) |*x| {
                    x.* = @exp(x.* - top);
                    total += x.*;
                }
                for (0..dims) |d| {
                    var acc: f64 = 0;
                    for (0..n) |j| {
                        const key: usize = if (listed) @intCast(ids[r * stride + j]) else j;
                        acc += w[j] * t.value(hv[(kvh * cap + key) * dims + d]);
                    }
                    const e = @abs(@as(f64, got[(r * heads + h) * dims + d]) - acc / total);
                    err[which] = @max(err[which], e);
                }
            }
        }
    }
    std.debug.print("sparse prefill heads={d} rows={d}: |err| scalar {e} wmma {e}\n", .{ heads, rows, err[0], err[1] });
    // the scalar kernel sums in fp32 over the cache's values; the tile rounds q and p to 16 bits for its fragments
    try rig.expect(err[0] <= 1e-5, "the scalar kernel drifts {e} from fp64", .{err[0]});
    try rig.expect(err[1] <= 5e-3, "the WMMA tile drifts {e} from fp64", .{err[1]});
}
