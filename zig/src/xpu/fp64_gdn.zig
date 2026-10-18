//! The GDN state recurrence against FP64 (XPU_FP64_REF): the prompt's chunked form and the serial decode, same rows.
const std = @import("std");
const q = @import("qwen27.zig");
const ze = q.ze;
const Dev = q.Dev;
const Acts = q.Acts;
const alloc = q.alloc;

/// GDN layers measured (the attention layers are those with (i + 1) % 4 == 0).
const GDN_ON = [_]usize{ 0, 1, 30, 62 };
const GPATHS = [_][]const u8{ "prompt (chunked)", "decode (serial)", "fp32 sequential", "bf16(FP64)" };

/// One V head's recurrence over T rows from a zero state: y [T, 128] and the final state [128, 128], in F.
fn recur(comptime F: type, qh: []const f32, kh: []const f32, vh: []const f32, g: []const f32, b: []const f32, T: usize, y: []F, st: []F) void {
    @memset(st, 0);
    for (0..T) |t| {
        const qt = qh[t * 128 ..][0..128];
        const kt = kh[t * 128 ..][0..128];
        const gt: F = g[t];
        for (0..128) |v| {
            const s = st[v * 128 ..][0..128];
            var mem: F = 0;
            for (s, kt) |*x, kv| {
                x.* = x.* * gt;
                mem = mem + x.* * @as(F, kv);
            }
            const u = (@as(F, vh[t * 128 + v]) - mem) * @as(F, b[t]);
            var acc: F = 0;
            for (s, kt, qt) |*x, kv, qv| {
                x.* = x.* + @as(F, kv) * u;
                acc = acc + x.* * @as(F, qv);
            }
            y[t * 128 + v] = acc;
        }
    }
}

const GdnJob = struct {
    q: []const f32,
    k: []const f32,
    v: []const f32,
    g: []const f32,
    b: []const f32,
    T: usize,
    h0: usize,
    h1: usize,
    y64: []f64,
    s64: []f64,
    y32: []f32,
    s32: []f32,

    fn run(j: *GdnJob) void {
        const T = j.T;
        const qh = alloc.alloc(f32, T * 128) catch return;
        defer alloc.free(qh);
        const kh = alloc.alloc(f32, T * 128) catch return;
        defer alloc.free(kh);
        const vh = alloc.alloc(f32, T * 128) catch return;
        defer alloc.free(vh);
        const gh = alloc.alloc(f32, T) catch return;
        defer alloc.free(gh);
        const bh = alloc.alloc(f32, T) catch return;
        defer alloc.free(bh);
        const y64 = alloc.alloc(f64, T * 128) catch return;
        defer alloc.free(y64);
        const y32 = alloc.alloc(f32, T * 128) catch return;
        defer alloc.free(y32);
        for (j.h0..j.h1) |h| {
            const kq = h / 3;
            for (0..T) |t| {
                @memcpy(qh[t * 128 ..][0..128], j.q[(t * 16 + kq) * 128 ..][0..128]);
                @memcpy(kh[t * 128 ..][0..128], j.k[(t * 16 + kq) * 128 ..][0..128]);
                @memcpy(vh[t * 128 ..][0..128], j.v[(t * 48 + h) * 128 ..][0..128]);
                gh[t] = j.g[t * 48 + h];
                bh[t] = j.b[t * 48 + h];
            }
            recur(f64, qh, kh, vh, gh, bh, T, y64, j.s64[h * 16384 ..][0..16384]);
            recur(f32, qh, kh, vh, gh, bh, T, y32, j.s32[h * 16384 ..][0..16384]);
            for (0..T) |t| {
                @memcpy(j.y64[(t * 48 + h) * 128 ..][0..128], y64[t * 128 ..][0..128]);
                @memcpy(j.y32[(t * 48 + h) * 128 ..][0..128], y32[t * 128 ..][0..128]);
            }
        }
    }
};

fn bfToF32(x: u16) f32 {
    return @bitCast(@as(u32, x) << 16);
}

fn f32ToBf(x: f32) u16 {
    const b: u32 = @bitCast(x);
    return @intCast((b + 0x7fff + ((b >> 16) & 1)) >> 16); // round to nearest even
}

fn bfOrdered(h: u16) i32 {
    return if (h & 0x8000 != 0) -@as(i32, h & 0x7fff) else @as(i32, h);
}

const GStats = struct {
    sq_err: f64 = 0,
    sq_ref: f64 = 0,
    sq_rn: f64 = 0,
    tail_err: f64 = 0, // the last 64 rows: what the recurrence accumulated
    tail_ref: f64 = 0,
    max_abs: f64 = 0,
    ulp1: usize = 0,
    n: usize = 0,
    head_err: [48]f64 = @splat(0),
    head_ref: [48]f64 = @splat(0),

    fn add(s: *GStats, out: f64, ref: f64, h: usize, tail: bool) void {
        const e = out - ref;
        s.sq_err += e * e;
        s.sq_ref += ref * ref;
        s.head_err[h] += e * e;
        s.head_ref[h] += ref * ref;
        if (tail) {
            s.tail_err += e * e;
            s.tail_ref += ref * ref;
        }
        s.max_abs = @max(s.max_abs, @abs(e));
        const rn = f32ToBf(@floatCast(ref));
        const er = out - @as(f64, bfToF32(rn));
        s.sq_rn += er * er;
        if (@abs(bfOrdered(f32ToBf(@floatCast(out))) - bfOrdered(rn)) <= 1) s.ulp1 += 1;
        s.n += 1;
    }

    fn worst(s: GStats) struct { f64, usize } {
        var w: f64 = 0;
        var wh: usize = 0;
        for (s.head_err, s.head_ref, 0..) |e, r, h| if (r > 0 and @sqrt(e / r) > w) {
            w = @sqrt(e / r);
            wh = h;
        };
        return .{ w, wh };
    }
};

fn readBuf(d: *Dev, comptime E: type, src: *anyopaque, n: usize) ![]E {
    const out = try alloc.alloc(E, n);
    try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, out.ptr, src, n * @sizeOf(E), null, 0, null));
    try d.sync();
    return out;
}

/// The GDN probe: the prompt's chunked recurrence and the serial decode one on the same rows, against FP64.
pub fn probe(d: *Dev, a: *Acts, state: *anyopaque, T: u32, layer: usize) !void {
    if (std.mem.indexOfScalar(usize, &GDN_ON, layer) == null) return;
    const t0 = q.now();
    try d.sync();
    const n = @as(usize, T);
    const qb = try readBuf(d, u16, a.q, n * 2048);
    defer alloc.free(qb);
    const kb = try readBuf(d, u16, a.k, n * 2048);
    defer alloc.free(kb);
    const vb = try readBuf(d, u16, a.v, n * 6144);
    defer alloc.free(vb);
    const g = try readBuf(d, f32, a.g, n * 48);
    defer alloc.free(g);
    const b = try readBuf(d, f32, a.beta, n * 48);
    defer alloc.free(b);
    const yp = try readBuf(d, u16, a.y, n * 6144);
    defer alloc.free(yp);
    const sp = try readBuf(d, f32, state, 48 * 16384);
    defer alloc.free(sp);
    // the decode kernel over the same rows from a zero state
    const s0 = try d.g.deviceAlloc(48 * 16384 * 4);
    defer d.g.free(s0);
    const s1 = try d.g.deviceAlloc(48 * 16384 * 4);
    defer d.g.free(s1);
    const yd_dev = try d.g.deviceAlloc(n * 6144 * 2);
    defer d.g.free(yd_dev);
    const zeros = try alloc.alloc(f32, 48 * 16384);
    defer alloc.free(zeros);
    @memset(zeros, 0);
    try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, s0, zeros.ptr, zeros.len * 4, null, 0, null));
    try d.sync();
    const nul: ?*anyopaque = null;
    try d.launch(d.k.gdn_step, .{ .x = 16, .y = 48, .z = 1 }, .{ a.q, a.k, a.v, a.g, a.beta, s0, s1, yd_dev, @as(i32, @intCast(T)), nul, nul, a.sl, nul });
    try d.sync();
    const yd = try readBuf(d, u16, yd_dev, n * 6144);
    defer alloc.free(yd);
    const sd = try readBuf(d, f32, s1, 48 * 16384);
    defer alloc.free(sd);
    // host references
    const qf = try alloc.alloc(f32, qb.len);
    defer alloc.free(qf);
    const kf = try alloc.alloc(f32, kb.len);
    defer alloc.free(kf);
    const vf = try alloc.alloc(f32, vb.len);
    defer alloc.free(vf);
    for (qf, qb) |*o, x| o.* = bfToF32(x);
    for (kf, kb) |*o, x| o.* = bfToF32(x);
    for (vf, vb) |*o, x| o.* = bfToF32(x);
    const y64 = try alloc.alloc(f64, n * 6144);
    defer alloc.free(y64);
    const s64 = try alloc.alloc(f64, 48 * 16384);
    defer alloc.free(s64);
    const y32 = try alloc.alloc(f32, n * 6144);
    defer alloc.free(y32);
    const s32 = try alloc.alloc(f32, 48 * 16384);
    defer alloc.free(s32);
    var jobs: [48]GdnJob = undefined;
    var threads: [48]?std.Thread = @splat(null);
    for (0..48) |h| {
        jobs[h] = .{ .q = qf, .k = kf, .v = vf, .g = g, .b = b, .T = n, .h0 = h, .h1 = h + 1, .y64 = y64, .s64 = s64, .y32 = y32, .s32 = s32 };
        threads[h] = std.Thread.spawn(.{}, GdnJob.run, .{&jobs[h]}) catch null;
        if (threads[h] == null) jobs[h].run();
    }
    for (threads) |t| if (t) |th| th.join();
    // outputs y (bf16 on the device) and final state S (fp32), by path
    var ys: [4]GStats = @splat(.{});
    var ss: [3]GStats = @splat(.{});
    for (0..n) |t| for (0..48) |h| for (0..128) |v| {
        const i = (t * 48 + h) * 128 + v;
        const r = y64[i];
        const tail = t + 64 >= n;
        ys[0].add(bfToF32(yp[i]), r, h, tail);
        ys[1].add(bfToF32(yd[i]), r, h, tail);
        ys[2].add(y32[i], r, h, tail);
        ys[3].add(bfToF32(f32ToBf(@floatCast(r))), r, h, tail);
    };
    for (0..48) |h| for (0..16384) |i| {
        const r = s64[h * 16384 + i];
        ss[0].add(sp[h * 16384 + i], r, h, false);
        ss[1].add(sd[h * 16384 + i], r, h, false);
        ss[2].add(s32[h * 16384 + i], r, h, false);
    };
    for (ys, 0..) |s, p| {
        const w = s.worst();
        std.debug.print("| GDN | {d} | y | {s} | {d} | {e:.2} | {e:.2} | {e:.2} | {e:.2} | {e:.2} (head {d}) | {d:.1} % |\n", .{
            layer,                                                      GPATHS[p], s.n,  @sqrt(s.sq_err / s.sq_ref),                                           @sqrt(s.sq_rn / s.sq_ref), @sqrt(s.tail_err / s.tail_ref),
            s.max_abs / @sqrt(s.sq_ref / @as(f64, @floatFromInt(s.n))), w[0],      w[1], 100 * @as(f64, @floatFromInt(s.ulp1)) / @as(f64, @floatFromInt(s.n)),
        });
    }
    for (ss, 0..) |s, p| {
        const w = s.worst();
        std.debug.print("| GDN | {d} | final state | {s} | {d} | {e:.2} | – | – | {e:.2} | {e:.2} (head {d}) | – |\n", .{
            layer, GPATHS[p], s.n, @sqrt(s.sq_err / s.sq_ref), s.max_abs / @sqrt(s.sq_ref / @as(f64, @floatFromInt(s.n))), w[0], w[1],
        });
    }
    std.debug.print("fp64 ref: GDN layer {d} ({d} rows) in {d:.1} s\n", .{ layer, T, q.now() - t0 });
}
