//! tf-cuda-test glm-kcheck (kernels vs float64, row invariance) and glm-batch (alone = batched = drafted = resumed).

const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const gw = @import("glm_weights");
const kern = gw.kern;
const fwd = gw.forward;
const dec = gw.decode;
const Gpu = check.Gpu;
const run = @import("glm_forward_run.zig");
pub const teacher = run.teacher;
pub const prof = run.prof;

fn bf(x: u16) f32 {
    return @bitCast(@as(u32, x) << 16);
}

fn tobf(v: f32) u16 { // round to nearest even
    const u: u32 = @bitCast(v);
    return @truncate((u + 0x7FFF + ((u >> 16) & 1)) >> 16);
}

fn randBf(rnd: std.Random, out: []u16, scale: f32) void {
    for (out) |*h| h.* = tobf(rnd.floatNorm(f32) * scale);
}

pub fn ms(io: std.Io, t0: std.Io.Timestamp) f64 {
    return @as(f64, @floatFromInt(t0.durationTo(std.Io.Timestamp.now(io, .awake)).toNanoseconds())) / 1e6;
}

fn up(d: *const cuda.Driver, bytes: []const u8) !cuda.DeviceBuffer {
    const b = try cuda.DeviceBuffer.alloc(d, @max(bytes.len, 16));
    try b.upload(0, bytes);
    return b;
}

fn down(gpa: std.mem.Allocator, d: *const cuda.Driver, comptime T: type, b: cuda.DeviceBuffer, n: usize) ![]T {
    const out = try gpa.alloc(T, n);
    try d.check(d.api.cuMemcpyDtoH_v2(@ptrCast(out.ptr), b.ptr, n * @sizeOf(T)), "download");
    return out;
}

const Stats = struct {
    worst: f64 = 0,
    scale: f64 = 0,
    fn add(s: *Stats, got: f64, want: f64) void {
        s.worst = @max(s.worst, @abs(got - want));
        s.scale = @max(s.scale, @abs(want));
    }
    fn rel(s: Stats) f64 {
        return s.worst / @max(s.scale, 1e-30);
    }
};

// -- glm-kcheck ------------------------------------------------------------------------------------------------
fn checkMm(gpu: Gpu, k: *kern.K, rnd: std.Random) !void {
    const gpa = gpu.gpa;
    const d = gpu.d;
    const shapes = [_][2]usize{ .{ 1024, 2048 }, .{ 160, 4096 }, .{ 12576, 512 } };
    for (shapes) |sh| {
        const n = sh[0];
        const kk = sh[1];
        const M: usize = 128;
        const w16 = try gpa.alloc(u16, n * kk);
        defer gpa.free(w16);
        randBf(rnd, w16, 0.02);
        const x16 = try gpa.alloc(u16, M * kk);
        defer gpa.free(x16);
        randBf(rnd, x16, 1.0);
        var xb = try up(d, std.mem.sliceAsBytes(x16));
        defer xb.free();
        var ob = try cuda.DeviceBuffer.alloc(d, M * n * 4);
        defer ob.free();
        var part = try cuda.DeviceBuffer.alloc(d, M * 16384 * 4);
        defer part.free();
        for ([_]u8{ 0, 8, 6, 4 }) |bits| {
            // the loader's row-major form, then tiled on the device as forward's load does
            var src: cuda.DeviceBuffer = undefined;
            var ss: ?cuda.DeviceBuffer = null;
            const deq = try gpa.alloc(f64, n * kk);
            defer gpa.free(deq);
            if (bits == 0) {
                src = try up(d, std.mem.sliceAsBytes(w16));
                for (w16, deq) |h, *v| v.* = bf(h);
            } else {
                const words = try gpa.alloc(u32, n * kk * bits / 32);
                defer gpa.free(words);
                const scales = try gpa.alloc(u16, n * kk / 64);
                defer gpa.free(scales);
                gw.quantize(bits, w16, n, kk, words, scales);
                src = try up(d, std.mem.sliceAsBytes(words));
                ss = try up(d, std.mem.sliceAsBytes(scales));
                const half: i32 = @as(i32, 1) << @intCast(bits - 1);
                const wpc: usize = bits / 2;
                for (0..n * kk / 16) |c| {
                    const p = words[c * wpc ..][0..wpc];
                    for (0..16) |i| {
                        const raw: u32 = switch (bits) {
                            4 => (p[i / 8] >> @intCast(4 * (i % 8))) & 15,
                            8 => (p[i / 4] >> @intCast(8 * (i % 4))) & 255,
                            else => if (i < 15) (p[i / 5] >> @intCast(6 * (i % 5))) & 63 else (p[0] >> 30) | ((p[1] >> 30) << 2) | ((p[2] >> 30) << 4),
                        };
                        const sc: f64 = @as(f16, @bitCast(scales[c / 4]));
                        deq[c * 16 + i] = @as(f64, @floatFromInt(@as(i32, @intCast(raw)) - half)) * sc;
                    }
                }
            }
            defer src.free();
            defer if (ss) |*x| x.free();
            const wbytes = n * kk * @as(usize, if (bits == 0) 16 else bits) / 8;
            var tw = try cuda.DeviceBuffer.alloc(d, wbytes);
            defer tw.free();
            var ts = try cuda.DeviceBuffer.alloc(d, n * kk / 64 * 2 + 16);
            defer ts.free();
            try kern.mmTile(k, bits, src.ptr, if (ss) |x| x.ptr else 0, tw.ptr, ts.ptr, @intCast(n), @intCast(kk));
            const w: kern.B16 = .{ .w = tw.ptr, .n = @intCast(n), .k = @intCast(kk), .s = if (bits == 0) 0 else ts.ptr, .bits = bits };
            try kern.mm(k, xb.ptr, @intCast(kk), w, ob.ptr, true, @intCast(M), part.ptr);
            try k.stream.synchronize();
            const full = try down(gpa, d, f32, ob, M * n);
            defer gpa.free(full);
            var st: Stats = .{};
            for (0..M) |r| {
                if (r % 17 != 0 and r != M - 1) continue; // a sample of rows against float64
                for (0..n) |c| {
                    var acc: f64 = 0;
                    for (0..kk) |i| acc += @as(f64, bf(x16[r * kk + i])) * deq[c * kk + i];
                    st.add(full[r * n + c], acc);
                }
            }
            // a row alone and in narrower windows: the same bits as in the 128-row one
            var same = true;
            for ([_]usize{ 1, 3, 16, 17, 33, 64, 100 }) |m| {
                try kern.mm(k, xb.ptr, @intCast(kk), w, ob.ptr, true, @intCast(m), part.ptr);
                try k.stream.synchronize();
                const part_rows = try down(gpa, d, f32, ob, m * n);
                defer gpa.free(part_rows);
                if (!std.mem.eql(u32, @ptrCast(part_rows), @as([]const u32, @ptrCast(full[0 .. m * n])))) same = false;
            }
            std.debug.print("mm {d}x{d} bits {d}: rel err {e:.2} vs float64; rows alone == in windows: {}\n", .{ n, kk, if (bits == 0) @as(u8, 16) else bits, st.rel(), same });
            try check.expect(st.rel() < 1e-5 and same, "mm {d}x{d} bits {d}", .{ n, kk, bits });
        }
    }
}

/// Latent attention: two streams' rows, dense and sparse, against float64; each row alone equals it in the window.
fn checkAttention(gpu: Gpu, k: *kern.K, rnd: std.Random) !void {
    const gpa = gpu.gpa;
    const d = gpu.d;
    const H: usize = 32;
    const LW: usize = 512;
    const L: usize = 9000; // cache rows a stream
    const caches = [2][]u16{ try gpa.alloc(u16, L * LW), try gpa.alloc(u16, L * LW) };
    for (caches) |c| randBf(rnd, c, 0.5);
    var cb: [2]cuda.DeviceBuffer = .{ try up(d, std.mem.sliceAsBytes(caches[0])), try up(d, std.mem.sliceAsBytes(caches[1])) };
    defer for (&cb) |*b| b.free();
    // rows: stream 0 at positions 5, 600, 2050 (dense), stream 1 at 1999, 2051, 8000 (sparse past the limit)
    const pos = [_]i32{ 5, 600, 2050, 1999, 2051, 8000 };
    const which = [_]usize{ 0, 0, 0, 1, 1, 1 };
    const R = pos.len;
    const q16 = try gpa.alloc(u16, R * H * LW);
    randBf(rnd, q16, 0.2);
    // sparse rows' tokens: 2048 distinct ascending tokens <= pos, then the 3 tail slots (-1)
    const tok = try gpa.alloc(i32, R * 2051);
    var cnt: [R]i32 = @splat(0);
    for (0..R) |r| {
        @memset(tok[r * 2051 ..][0..2051], -1);
        if (pos[r] < 2051) continue;
        var t: i32 = 0;
        var j: usize = 0;
        while (j < 2048) : (t += 1) {
            if (rnd.float(f32) < 2048.0 / @as(f32, @floatFromInt(pos[r] + 1)) or @as(i32, @intCast(2048 - j)) > pos[r] - t) {
                tok[r * 2051 + j] = t;
                j += 1;
            }
        }
        cnt[r] = 2048;
    }
    var lc: [R]u64 = undefined;
    for (0..R) |r| lc[r] = cb[which[r]].ptr;
    // the tables as forward.prepare lays them out (positions, then lc for cache 0; one cache here)
    var qb = try up(d, std.mem.sliceAsBytes(q16));
    defer qb.free();
    var tb = try up(d, std.mem.sliceAsBytes(tok));
    defer tb.free();
    var nb = try up(d, std.mem.sliceAsBytes(&cnt));
    defer nb.free();
    var po = try cuda.DeviceBuffer.alloc(d, 5 * R * H * LW * 4);
    defer po.free();
    var pm = try cuda.DeviceBuffer.alloc(d, 5 * R * H * 4);
    defer pm.free();
    var pl = try cuda.DeviceBuffer.alloc(d, 5 * R * H * 4);
    defer pl.free();
    var ob = try cuda.DeviceBuffer.alloc(d, R * H * LW * 2);
    defer ob.free();
    const run_rows = struct {
        fn f(kk: *kern.K, dd: *const cuda.Driver, q: u64, rows: []const usize, posv: []const i32, lcv: []const u64, tokv: u64, cntv: []const i32, b: [3]u64, out: u64) !void {
            var p: [8]i32 = undefined;
            var l: [8]u64 = undefined;
            var c: [8]i32 = undefined;
            for (rows, 0..) |r, i| {
                p[i] = posv[r];
                l[i] = lcv[r];
                c[i] = cntv[r];
            }
            const n = rows.len;
            const lo = std.mem.alignForward(usize, n * 4, 16); // the pointer table 8-aligned, as forward.prepare's is
            var tab = try cuda.DeviceBuffer.alloc(dd, lo + n * 8);
            defer tab.free();
            try tab.upload(0, std.mem.sliceAsBytes(p[0..n]));
            try tab.upload(lo, std.mem.sliceAsBytes(l[0..n]));
            var cn = try up(dd, std.mem.sliceAsBytes(c[0..n]));
            defer cn.free();
            var tk = try cuda.DeviceBuffer.alloc(dd, n * 2051 * 4);
            defer tk.free();
            for (rows, 0..) |r, i| try kk.copy(tk.ptr + i * 2051 * 4, tokv + r * 2051 * 4, 2051 * 4);
            var qq = try cuda.DeviceBuffer.alloc(dd, n * 32 * 512 * 2);
            defer qq.free();
            for (rows, 0..) |r, i| try kk.copy(qq.ptr + i * 32 * 512 * 2, q + r * 32 * 512 * 2, 32 * 512 * 2);
            const t: kern.Rows = .{ .pos = tab.ptr, .lc = tab.ptr + lo, .ik = 0, .ig = 0, .pk = 0, .cap = @intCast(n) };
            try kern.denseAttention(kk, qq.ptr, t, 0, b[0], b[1], b[2], out, @intCast(n), kern.DENSE_CHUNKS);
            try kern.sparseAttention(kk, qq.ptr, t, 0, tk.ptr, cn.ptr, b[0], b[1], b[2], out, @intCast(n));
            try kk.stream.synchronize();
        }
    }.f;
    const all = [_]usize{ 0, 1, 2, 3, 4, 5 };
    try run_rows(k, d, qb.ptr, &all, &pos, &lc, tb.ptr, &cnt, .{ po.ptr, pm.ptr, pl.ptr }, ob.ptr);
    const mixed = try down(gpa, d, u16, ob, R * H * LW);
    var st: Stats = .{};
    for (0..R) |r| {
        const cache = caches[which[r]];
        for (0..H) |h| {
            const q = q16[(r * H + h) * LW ..][0..LW];
            var keys: std.ArrayList(usize) = .empty;
            defer keys.deinit(gpa);
            if (pos[r] < 2051) {
                for (0..@intCast(pos[r] + 1)) |j| try keys.append(gpa, j);
            } else for (tok[r * 2051 ..][0..@intCast(cnt[r])]) |t| try keys.append(gpa, @intCast(t));
            var m: f64 = -std.math.inf(f64);
            const sc = try gpa.alloc(f64, keys.items.len);
            defer gpa.free(sc);
            for (keys.items, 0..) |j, i| {
                var a: f64 = 0;
                for (0..LW) |c| a += @as(f64, bf(q[c])) * bf(cache[j * LW + c]);
                sc[i] = a * 0.0625;
                m = @max(m, sc[i]);
            }
            var l: f64 = 0;
            for (sc) |*s| {
                s.* = @exp(s.* - m);
                l += s.*;
            }
            for (0..LW) |c| {
                var o: f64 = 0;
                for (keys.items, 0..) |j, i| o += sc[i] * bf(cache[j * LW + c]);
                st.add(bf(mixed[(r * H + h) * LW + c]), o / l);
            }
        }
    }
    // each row alone, and the stream-1 rows on their own
    var same = true;
    for (0..R) |r| {
        try run_rows(k, d, qb.ptr, &.{r}, &pos, &lc, tb.ptr, &cnt, .{ po.ptr, pm.ptr, pl.ptr }, ob.ptr);
        const one = try down(gpa, d, u16, ob, H * LW);
        defer gpa.free(one);
        if (!std.mem.eql(u16, one, mixed[r * H * LW ..][0 .. H * LW])) same = false;
    }
    std.debug.print("latent attention (dense and sparse, two streams): rel err {e:.2} vs float64; rows alone == mixed: {}\n", .{ st.rel(), same });
    try check.expect(st.rel() < 2e-2 and same, "latent attention", .{});
}

/// The indexer: pool scores against float64, the 512 best pools against a host pick over the kernel's scores.
fn checkIndex(gpu: Gpu, k: *kern.K, rnd: std.Random) !void {
    const gpa = gpu.gpa;
    const d = gpu.d;
    const NP: usize = 4096;
    const pk16 = try gpa.alloc(u16, NP * 128);
    randBf(rnd, pk16, 0.3);
    for (1000..1300) |p| @memcpy(pk16[p * 128 ..][0..128], pk16[999 * 128 ..][0..128]); // ties: 301 equal pools
    var pkb = try up(d, std.mem.sliceAsBytes(pk16));
    defer pkb.free();
    const pos = [_]i32{ 2051, 5000, 16383, 9999 };
    const R = pos.len;
    const qi = try gpa.alloc(u16, R * 32 * 128);
    randBf(rnd, qi, 0.5);
    const wv = try gpa.alloc(u16, R * 160);
    randBf(rnd, wv, 1.0);
    var qb = try up(d, std.mem.sliceAsBytes(qi));
    defer qb.free();
    var wb = try up(d, std.mem.sliceAsBytes(wv));
    defer wb.free();
    var tab = try cuda.DeviceBuffer.alloc(d, R * 4 + R * 8 * 4);
    defer tab.free();
    try tab.upload(0, std.mem.sliceAsBytes(&pos));
    var pkp: [R]u64 = @splat(pkb.ptr);
    try tab.upload(R * 4 + 3 * R * 8, std.mem.sliceAsBytes(&pkp)); // the pk table (lc, ik, ig unused)
    const t: kern.Rows = .{ .pos = tab.ptr, .lc = tab.ptr + R * 4, .ik = tab.ptr + R * 4 + R * 8, .ig = tab.ptr + R * 4 + 2 * R * 8, .pk = tab.ptr + R * 4 + 3 * R * 8, .cap = R };
    var sb = try cuda.DeviceBuffer.alloc(d, R * NP * 4);
    defer sb.free();
    var ob = try cuda.DeviceBuffer.alloc(d, R * 512 * 8);
    defer ob.free();
    try kern.scores(k, qb.ptr, wb.ptr, 160, t, 0, sb.ptr, R, NP);
    try kern.selectRows(k, sb.ptr, ob.ptr, NP, t, R);
    try k.stream.synchronize();
    const sc = try down(gpa, d, f32, sb, R * NP);
    const sel = try down(gpa, d, i64, ob, R * 512);
    var st: Stats = .{};
    var picks_ok = true;
    for (0..R) |r| {
        const npool: usize = @intCast(@divFloor(pos[r] + 1, 4));
        for (0..NP) |p| {
            if (p >= npool) {
                if (sc[r * NP + p] != -std.math.inf(f32)) picks_ok = false;
                continue;
            }
            var s: f64 = 0;
            for (0..32) |h| {
                var dot: f64 = 0;
                for (0..128) |c| dot += @as(f64, bf(qi[(r * 32 + h) * 128 + c])) * bf(pk16[p * 128 + c]);
                s += @as(f64, bf(wv[r * 160 + h])) / 5.656854249492381 * @max(dot * 0.08838834764831845, 0);
            }
            st.add(sc[r * NP + p], s);
        }
        // the host's 512: by score descending, the lower pool first on ties, then ascending
        const idx = try gpa.alloc(usize, @min(NP, @max(512, npool)));
        defer gpa.free(idx);
        for (idx, 0..) |*x, i| x.* = i;
        const Ctx = struct {
            s: []const f32,
            fn lt(c: @This(), a: usize, b: usize) bool {
                const x = c.s[a] + 0.0;
                const y = c.s[b] + 0.0;
                return x > y or (x == y and a < b);
            }
        };
        std.mem.sort(usize, idx, Ctx{ .s = sc[r * NP ..][0..NP] }, Ctx.lt);
        std.mem.sort(usize, idx[0..512], {}, std.sort.asc(usize));
        for (0..512) |i| if (sel[r * 512 + i] != idx[i]) {
            picks_ok = false;
        };
    }
    std.debug.print("indexer: scores rel err {e:.2} vs float64; the 512 pools equal the host's selection: {}\n", .{ st.rel(), picks_ok });
    try check.expect(st.rel() < 1e-4 and picks_ok, "indexer", .{});
}

pub fn kcheck(gpu: Gpu, _: []const [:0]const u8) !void {
    var stream = try cuda.Stream.init(gpu.d, true);
    defer stream.deinit();
    var kernels = try gw.api.Kernels.load(gpu.d);
    defer kernels.deinit();
    var k = try kern.K.init(gpu.gpa, gpu.d, gpu.ctx, stream, &kernels.ext);
    var prng = std.Random.DefaultPrng.init(11);
    try checkMm(gpu, &k, prng.random());
    try checkAttention(gpu, &k, prng.random());
    try checkIndex(gpu, &k, prng.random());
    std.debug.print("glm-kcheck: all passed\n", .{});
}

// -- glm-batch -------------------------------------------------------------------------------------------------
const Stream = struct { s: *fwd.Seq, out: std.ArrayList(u32) = .empty, pending: u32 = 0, held: [7]u32 = undefined, nheld: usize = 0, done: bool = false };

/// Greedy decode of every stream to ``n`` tokens: pending token plus ``depth`` drafts, the agreeing prefix kept.
fn decodeAll(e: *fwd.Engine, ss: []Stream, n: usize, depth: usize) !void {
    var picks: [fwd.STEP_ROWS]dec.Pick = undefined;
    var wins: [fwd.MAX_STREAMS]dec.Win = undefined;
    var ids: [fwd.MAX_STREAMS][8]u32 = undefined;
    var map: [fwd.MAX_STREAMS]usize = undefined;
    var probs: [fwd.MAX_STREAMS][7]f64 = undefined;
    var nexts: [fwd.MAX_STREAMS][8]u32 = undefined;
    while (true) {
        var nw: usize = 0;
        for (ss, 0..) |*x, i| {
            if (x.done) continue;
            ids[nw][0] = x.pending;
            @memcpy(ids[nw][1..][0..x.nheld], x.held[0..x.nheld]);
            wins[nw] = .{ .s = x.s, .ids = ids[nw][0 .. 1 + x.nheld] };
            map[nw] = i;
            nw += 1;
        }
        if (nw == 0) return;
        try dec.verify(e, wins[0..nw], &picks);
        var reqs: [fwd.MAX_STREAMS]dec.DraftReq = undefined;
        var nr: usize = 0;
        var at: usize = 0;
        for (0..nw) |wi| {
            const x = &ss[map[wi]];
            const rows = wins[wi].ids.len;
            var kept: usize = 1;
            while (kept < rows and wins[wi].ids[kept] == picks[at + kept - 1].tok) kept += 1;
            try dec.keep(e, x.s, @intCast(rows), @intCast(kept));
            for (0..kept) |r| {
                try x.out.append(e.gpa, picks[at + r].tok);
                nexts[wi][r] = picks[at + r].tok;
            }
            x.pending = picks[at + kept - 1].tok;
            x.nheld = 0;
            if (x.out.items.len >= n) x.done = true else if (depth > 0) {
                reqs[nr] = .{ .s = x.s, .hidden = x.s.hidden, .next = nexts[wi][0..kept], .count = depth, .confidence = 0.7, .out = x.held[0..depth], .probs = probs[nr][0..depth] };
                nr += 1;
            }
            at += rows;
        }
        if (nr > 0) {
            try dec.draft(e, reqs[0..nr]);
            var ri: usize = 0;
            for (0..nw) |wi| {
                const x = &ss[map[wi]];
                if (x.done) continue;
                x.nheld = reqs[ri].got;
                ri += 1;
            }
        }
    }
}

pub fn batch(gpu: Gpu, args: []const [:0]const u8) !void {
    const gpa = gpu.gpa;
    const io = gpu.io;
    if (args.len < 4) return error.Usage;
    const rank = try std.fmt.parseInt(u8, args[0], 10);
    const port = try std.fmt.parseInt(u16, args[2], 10);
    const ntok: usize = if (args.len > 4) try std.fmt.parseInt(usize, args[4], 10) else 48;
    var link = if (rank == 0) try cuda.tp_link.Link.lead(port, 600_000) else try cuda.tp_link.Link.follow(io, try std.Io.net.IpAddress.parse(args[1], port), 600_000);
    defer link.close();
    var nccl = try cuda.nccl.Library.open();
    defer nccl.close();
    const comm = try cuda.tp_link.communicator(&nccl, link);
    // the communicator's threads stop before the library unloads (an unloaded NCCL under live threads crashes the exit)
    defer _ = nccl.api.ncclCommAbort(comm);
    var stream = try cuda.Stream.init(gpu.d, true);
    defer stream.deinit();
    {
        const warm = try cuda.DeviceBuffer.alloc(gpu.d, 4096);
        try nccl.check(nccl.api.ncclAllGather(warm.ptr, warm.ptr + 2048, 256, .i32, comm, stream.handle), "warm gather");
        try stream.synchronize();
    }
    const t0 = std.Io.Timestamp.now(io, .awake);
    var kernels = try gw.api.Kernels.load(gpu.d);
    var ctx: gw.api.Ctx = .{ .d = gpu.d, .ctx = gpu.ctx, .stream = stream, .nccl = &nccl, .comm = comm, .rank = rank };
    var store = try gw.load(gpa, io, &ctx, args[3], rank);
    defer store.deinit();
    const e = try fwd.init(gpa, io, &ctx, &kernels, &store, .{ .context = 65536 });
    e.k.sync_each = args.len > 5 and std.mem.eql(u8, args[5], "sync");
    e.sync_each = e.k.sync_each;
    std.debug.print("rank {d}: engine up in {d:.1} s\n", .{ rank, ms(io, t0) / 1000 });
    const S = fwd.MAX_STREAMS;
    var prompts: [S][]u32 = undefined;
    var prng = std.Random.DefaultPrng.init(5);
    for (&prompts, 0..) |*p, i| {
        p.* = try gpa.alloc(u32, 40 + i * 37);
        for (p.*) |*t| t.* = 1000 + prng.random().uintLessThan(u32, 60000);
    }
    var solo: [S][]u32 = undefined;
    // each stream alone, serial (one-row windows, no drafts)
    var ts = std.Io.Timestamp.now(io, .awake);
    for (0..S) |i| {
        var one = [_]Stream{.{ .s = try fwd.newSeq(e, 4096) }};
        one[0].pending = try dec.prefill(e, one[0].s, prompts[i], 0, 0);
        try one[0].out.append(gpa, one[0].pending);
        try decodeAll(e, &one, ntok, 0);
        solo[i] = one[0].out.items;
        fwd.freeSeq(e, one[0].s);
    }
    const solo_ms = ms(io, ts);
    std.debug.print("rank {d}: 16 streams one at a time, serial: {d:.1} tok/s\n", .{ rank, @as(f64, @floatFromInt(S * ntok)) / solo_ms * 1000 });
    // all together, serial, then drafted; and one stream alone drafted
    for ([_]usize{ 0, 3 }) |depth| {
        var all: [S]Stream = undefined;
        for (&all, 0..) |*x, i| {
            x.* = .{ .s = try fwd.newSeq(e, 4096) };
            x.pending = try dec.prefill(e, x.s, prompts[i], 0, if (depth > 0) prompts[i].len - 1 else 0);
            try x.out.append(gpa, x.pending);
        }
        ts = std.Io.Timestamp.now(io, .awake);
        try decodeAll(e, &all, ntok, depth);
        const t_ms = ms(io, ts);
        var equal: usize = 0;
        for (all, 0..) |x, i| equal += @intFromBool(std.mem.eql(u32, x.out.items[0..ntok], solo[i][0..ntok]));
        std.debug.print("rank {d}: 16 streams together, {s}: {d} of 16 replies equal to alone, {d:.1} tok/s aggregate\n", .{ rank, if (depth > 0) "drafted" else "serial", equal, @as(f64, @floatFromInt(S * (ntok - 1))) / t_ms * 1000 });
        try check.expect(equal == S, "16 streams together equal alone", .{});
        if (depth > 0) { // resumed: stream 0's kept state at its prompt's end - 1, a longer prompt extending it
            const ext = try std.mem.concat(gpa, u32, &.{ prompts[0], solo[0][0..8] });
            var fresh = [_]Stream{.{ .s = try fwd.newSeq(e, 4096) }};
            fresh[0].pending = try dec.prefill(e, fresh[0].s, ext, 0, 0);
            try fresh[0].out.append(gpa, fresh[0].pending);
            try decodeAll(e, &fresh, 16, 0);
            var res = [_]Stream{.{ .s = all[0].s }};
            res[0].pending = try dec.prefill(e, res[0].s, ext, prompts[0].len - 1, 0);
            try res[0].out.append(gpa, res[0].pending);
            try decodeAll(e, &res, 16, 0);
            const ok = std.mem.eql(u32, res[0].out.items[0..16], fresh[0].out.items[0..16]);
            std.debug.print("rank {d}: resumed from the kept state == fresh: {}\n", .{ rank, ok });
            try check.expect(ok, "resumed equals fresh", .{});
            fwd.freeSeq(e, fresh[0].s);
        }
        for (all) |x| fwd.freeSeq(e, x.s);
    }
    {
        var one = [_]Stream{.{ .s = try fwd.newSeq(e, 4096) }};
        one[0].pending = try dec.prefill(e, one[0].s, prompts[3], 0, prompts[3].len - 1);
        try one[0].out.append(gpa, one[0].pending);
        ts = std.Io.Timestamp.now(io, .awake);
        try decodeAll(e, &one, 4 * ntok, 3);
        const t_ms = ms(io, ts);
        std.debug.print("rank {d}: one stream drafted: {d:.1} tok/s (graphs {d})\n", .{ rank, @as(f64, @floatFromInt(4 * ntok - 1)) / t_ms * 1000, e.captures });
        fwd.freeSeq(e, one[0].s);
    }
    std.debug.print("rank {d}: glm-batch passed\n", .{rank});
    std.process.exit(0); // a test binary: no teardown (NCCL's service threads under the library's unload hang it)
}
