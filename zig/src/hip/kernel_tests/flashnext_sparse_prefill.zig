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
    try case(t, 24, 2, 64, false);
    try case(t, 6, 1, 17, false); // rows % 4 != 0: a tile of two rows, and group 6 pads its 16-row tile
}

test "sparse prefill: a deep cache with the indexer's full budget, the two kernels' time and their difference" {
    const t = try Rig.open(1 << 27);
    defer t.close();
    try case(t, 24, 2, 64, true);
}

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

fn case(t: *Rig, comptime heads: usize, comptime kv_heads: usize, comptime rows: usize, comptime deep: bool) !void {
    var rng: rig.Rng = .{ .state = 0x5EED + heads + rows };
    const dims = 256;
    const ratio = 4;
    const parts = 16;
    const cap = if (deep) 16384 else 1024;
    const stride = if (deep) 2064 else 520; // id_stride: more than the deepest row's count
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
        ends[r] = @as(i32, @intCast((if (deep) cap - rows * 12 else 200) + r * 12 + (r % 4)));
        const nb: usize = @as(usize, @intCast(ends[r])) / ratio; // the blocks the row may select
        if (r == 0) { // dense: the first ends[0] keys, unlisted
            sparse[r] = 0;
            count[r] = ends[r];
            continue;
        }
        sparse[r] = 1;
        const want: usize = @min(if (deep) 512 else if (r < 8) @as(usize, 5) else 120, nb - 8);
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
    var ms: [2]f64 = undefined; // the kernel pair's time a launch
    var dev: [2]f64 = undefined; // vs the scalar kernel's output, the largest |difference|
    var scalar_out: []f32 = &.{};
    defer gpa.free(scalar_out);
    for (0..2) |which| {
        t.on.sparse_wmma = which != 0;
        try launch(t, dq, dk, dv, ck, di, stride, dc, ds, de, po, pm, out, rows, heads, kv_heads, dims, cap, scale, parts);
        try t.stream.synchronize();
        const t0 = nowNs();
        for (0..10) |_| try launch(t, dq, dk, dv, ck, di, stride, dc, ds, de, po, pm, out, rows, heads, kv_heads, dims, cap, scale, parts);
        try t.stream.synchronize();
        ms[which] = @as(f64, @floatFromInt(nowNs() - t0)) / 10e6;
        const got = try rig.download(f32, out);
        defer if (which != 0) gpa.free(got);
        if (which == 0) scalar_out = got;
        dev[which] = 0;
        for (got, scalar_out) |a, b| dev[which] = @max(dev[which], @abs(@as(f64, a) - b));
        err[which] = 0;
        if (deep) continue; // the fp64 reference over 2,048 keys a row is slow; the shallow cases hold the error
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
    std.debug.print("sparse prefill {d} keys a row, ms a launch: scalar {d:.3} wmma {d:.3}\n", .{ if (deep) 2048 else 480, ms[0], ms[1] });
    if (deep) {
        std.debug.print("sparse prefill deep: wmma differs {e} from scalar\n", .{dev[1]});
        try rig.expect(dev[1] <= 1e-5, "the WMMA tile differs {e} from the scalar kernel", .{dev[1]});
        return;
    }
    std.debug.print("sparse prefill heads={d} rows={d}: |err| fp64 scalar {e} wmma {e}; wmma vs scalar {e}\n", .{ heads, rows, err[0], err[1], dev[1] });
    // both sum in fp32 over the cache's values; the tile carries q and p as three 16-bit parts, so it is held as tight
    try rig.expect(err[0] <= 1e-5, "the scalar kernel drifts {e} from fp64", .{err[0]});
    try rig.expect(err[1] <= 1e-5, "the WMMA tile drifts {e} from fp64", .{err[1]});
}

fn launch(
    t: *Rig,
    dq: anytype,
    dk: anytype,
    dv: anytype,
    ck: c_int,
    di: anytype,
    stride: usize,
    dc: anytype,
    ds: anytype,
    de: anytype,
    po: anytype,
    pm: anytype,
    out: anytype,
    rows: usize,
    heads: usize,
    kv_heads: usize,
    dims: usize,
    cap: usize,
    scale: f32,
    parts: usize,
) !void {
    try t.on.fn_sparse_attention(
        ptr(?[*]const f32, dq),
        ptr(?*const anyopaque, dk),
        ptr(?*const anyopaque, dv),
        ck,
        ptr(?[*]const i32, di),
        @intCast(stride),
        ptr(?[*]const i32, dc),
        ptr(?[*]const i32, ds),
        ptr(?[*]const i32, de),
        ptr(?[*]f32, po),
        ptr(?[*]f32, pm),
        ptr(?[*]f32, out),
        @intCast(rows),
        @intCast(heads),
        @intCast(kv_heads),
        @intCast(dims),
        @intCast(cap),
        scale,
        @intCast(parts),
        t.stream.handle,
    );
}
