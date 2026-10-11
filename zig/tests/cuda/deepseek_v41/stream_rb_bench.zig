//! Stream top-k twins (`_stream_pf`, `_stream_rb2` / `_rb4`) against `_stream`: keys slot for slot, then timed.

const std = @import("std");
const cuda = @import("cuda");
const check = @import("../check.zig");

const aot = cuda.aot;
const Gpu = check.Gpu;

const H = 32;
const D = 128;
const BP = 64;
const RATIO = 2;
const SPLIT = 16384;
const K = 512;
const CAP = 1024;
const PSH = 7;
const KEY_NONE: i64 = std.math.minInt(i64);

const starts = [_]i64{ 32768, 131072 - 2048 };
const row_counts = [_]i64{ 2048, 2047, 300 };

const Rng = struct {
    s: u64,
    fn next(r: *Rng) u64 {
        r.s ^= r.s << 13;
        r.s ^= r.s >> 7;
        r.s ^= r.s << 17;
        return r.s;
    }
    fn unit(r: *Rng) f32 {
        return @as(f32, @floatFromInt(r.next() >> 40)) / @as(f32, 1 << 24);
    }
};

fn twinName(rb: u32) []const u8 {
    return switch (rb) {
        1 => "_stream_pf",
        2 => "_stream_rb2",
        else => "_stream_rb4",
    };
}

fn median(xs: []f32) f32 {
    std.mem.sort(f32, xs, {}, std.sort.asc(f32));
    return xs[xs.len / 2];
}

/// The emitter's constexprs (block_prefill.streamSelect): WS / SCALE as f64 1 / sqrt narrowed to fp32.
fn common(c: *std.ArrayList(aot.Const), a: std.mem.Allocator) !void {
    try c.appendSlice(a, &.{
        aot.ci("RATIO", RATIO),                             aot.ci("H", H),
        aot.ci("D", D),                                     aot.ci("BP", BP),
        aot.cf("WS", @floatCast(1.0 / @sqrt(@as(f64, H)))), aot.cf("SCALE", @floatCast(1.0 / @sqrt(@as(f64, D)))),
        aot.ci("SPLIT", SPLIT),                             aot.ci("K", K),
        aot.ci("CAP", CAP),                                 aot.ci("PSH", PSH),
        aot.ci("KFP8", 1),
    });
}

const Case = struct {
    set: *const aot.Set,
    stream: cuda.Stream,
    qi: u64,
    w: u64,
    ik: u64,
    pos: u64,
    pt: u64,
    rows: i64,
    nk: i64,
    ns: i64,

    /// `_stream` (rb 0), `_stream_pf` (rb 1) or `_stream_rbN` into `buf`.
    fn launch(c: *const Case, a: std.mem.Allocator, rb: u32, buf: u64) !void {
        var args: std.ArrayList(aot.Arg) = .empty;
        var consts: std.ArrayList(aot.Const) = .empty;
        try args.appendSlice(a, &.{
            aot.ptr("QI", "*bf16", c.qi),      aot.ptr("W", "*fp32", c.w),
            aot.int("w_stride", H),            aot.ptr("IK", "*u8", c.ik),
            aot.ptr("POS", "*i32", c.pos),     aot.ptr("BUF", "*i64", buf),
            aot.int("nsplit", @intCast(c.ns)), aot.ptr("PT", "*i32", c.pt),
        });
        try common(&consts, a);
        const ns: u32 = @intCast(c.ns);
        if (rb == 0) {
            try args.appendSlice(a, &.{ aot.ptr("KEYS", "*i64", buf), aot.int("k_stride", 0), aot.int("NK", @intCast(c.nk)) });
            try consts.appendSlice(a, &.{ aot.ci("MODE", 0), aot.ci("BS", 8) });
            return c.set.run(c.stream, "_stream", .{ @intCast(c.rows), ns, 1 }, args.items, consts.items);
        }
        try args.append(a, aot.int("R", @intCast(c.rows)));
        const blocks: u32 = @intCast(@divFloor(c.rows + rb - 1, rb));
        return c.set.run(c.stream, twinName(rb), .{ blocks, ns, 1 }, args.items, consts.items);
    }

    /// The median ms of `reps` launches after 2 warm-ups.
    fn time(c: *const Case, a: std.mem.Allocator, d: *const cuda.Driver, rb: u32, buf: u64, reps: usize) !f32 {
        var t0 = try cuda.Event.init(d, true);
        defer t0.deinit();
        var t1 = try cuda.Event.init(d, true);
        defer t1.deinit();
        for (0..2) |_| try c.launch(a, rb, buf);
        var ms: [64]f32 = undefined;
        const n = @max(1, @min(reps, ms.len));
        for (0..n) |i| {
            try t0.record(c.stream);
            try c.launch(a, rb, buf);
            try t1.record(c.stream);
            try t1.synchronize();
            ms[i] = try cuda.Event.elapsedMs(t0, t1);
        }
        return median(ms[0..n]);
    }
};

/// One poisoned run into `buf`, downloaded (null: the variant is not in the set).
fn once(c: *const Case, a: std.mem.Allocator, rb: u32, buf: cuda.DeviceBuffer, poison: u8, out: []i64) !bool {
    try buf.fill8(poison, c.stream.handle);
    c.launch(a, rb, buf.ptr) catch |e| switch (e) {
        error.MissingTritonVariant => return false,
        else => return e,
    };
    try c.stream.synchronize();
    try buf.download(0, std.mem.sliceAsBytes(out));
    return true;
}

/// Every (row, split)'s first K keys equal; prints the first difference.
fn compare(name: []const u8, got: []const i64, want: []const i64, rows: i64, ns: i64) bool {
    var bad: usize = 0;
    var first: ?[3]usize = null;
    for (0..@intCast(rows)) |r| for (0..@intCast(ns)) |c| {
        const o = (r * @as(usize, @intCast(ns)) + c) * CAP;
        if (std.mem.indexOfDiff(i64, got[o..][0..K], want[o..][0..K])) |s| {
            bad += 1;
            if (first == null) first = .{ r, c, s };
        }
    };
    if (first) |f| {
        const o = (f[0] * @as(usize, @intCast(ns)) + f[1]) * CAP + f[2];
        std.debug.print("  bits {s}: FAIL ({d} row/splits differ; first row {d} split {d} slot {d}: {d} vs _stream {d})\n", .{ name, bad, f[0], f[1], f[2], got[o], want[o] });
        return false;
    }
    std.debug.print("  bits {s}: PASS\n", .{name});
    return true;
}

pub fn run(gpu: Gpu, rest: []const [:0]const u8) !void {
    const gpa = gpu.gpa;
    const d = gpu.d;
    if (rest.len < 1) return error.MissingArgument;
    const dir = rest[0];
    var reps: usize = 5;
    var twins_buf: [3]u32 = .{ 1, 2, 4 };
    var twins: []const u32 = &twins_buf;
    var i: usize = 1;
    while (i < rest.len) : (i += 1) {
        if (std.mem.eql(u8, rest[i], "--reps") and i + 1 < rest.len) {
            i += 1;
            reps = try std.fmt.parseInt(usize, rest[i], 10);
        } else if (std.mem.eql(u8, rest[i], "--twins") and i + 1 < rest.len) {
            i += 1;
            var it = std.mem.tokenizeScalar(u8, rest[i], ',');
            var nt: usize = 0;
            while (it.next()) |x| {
                if (nt == twins_buf.len) return error.BadArgument;
                const v = try std.fmt.parseInt(u32, x, 10);
                if (v != 1 and v != 2 and v != 4) return error.BadArgument;
                twins_buf[nt] = v;
                nt += 1;
            }
            twins = twins_buf[0..nt];
        } else return error.BadArgument;
    }
    var set = try aot.Set.load(gpa, gpu.io, d, gpu.ctx.device, dir);
    defer set.deinit();
    var stream = try cuda.Stream.init(d, false);
    defer stream.deinit();
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    var rng: Rng = .{ .s = 0x9e3779b97f4a7c15 };

    const max_rows: usize = @intCast(std.mem.max(i64, &row_counts));
    const max_keys: usize = @intCast(@divFloor(std.mem.max(i64, &starts) + @as(i64, @intCast(max_rows)), RATIO));
    const pages = (max_keys >> PSH) + 2;
    const key_rows = pages << PSH;
    // queries bf16 [rows, H, D] in (-1, 1) (fp32 truncated), weights fp32 [rows, H] signed
    const qh = try gpa.alloc(u16, max_rows * H * D);
    defer gpa.free(qh);
    for (qh) |*v| v.* = @truncate(@as(u32, @bitCast(2.0 * rng.unit() - 1.0)) >> 16);
    var qi = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(qh));
    defer qi.free();
    const wh = try gpa.alloc(f32, max_rows * H);
    defer gpa.free(wh);
    for (wh) |*v| v.* = 2.0 * rng.unit() - 1.0;
    var w = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(wh));
    defer w.free();
    // FP8 key rows: D e4m3 bytes (no NaN 0x7F / 0xFF), then the fp32 scale
    const ikh = try gpa.alloc(u8, key_rows * (D + 4));
    defer gpa.free(ikh);
    const scale: [4]u8 = @bitCast(@as(f32, 0.0625));
    for (0..key_rows) |r| {
        const row = ikh[r * (D + 4) ..][0 .. D + 4];
        for (row[0..D]) |*b| {
            b.* = @truncate(rng.next() >> 32);
            while (b.* & 0x7F == 0x7F) b.* = @truncate(rng.next() >> 32);
        }
        @memcpy(row[D..], &scale);
    }
    var ik = try cuda.DeviceBuffer.fromHost(d, ikh);
    defer ik.free();
    const pth = try gpa.alloc(i32, pages);
    defer gpa.free(pth);
    for (pth, 0..) |*p, j| p.* = @intCast(j);
    var pt = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(pth));
    defer pt.free();
    var pos = try cuda.DeviceBuffer.alloc(d, 4);
    defer pos.free();
    std.debug.print("stream-rb: H {d}, D {d}, BP {d}, ratio {d}, SPLIT {d}, K {d}, CAP {d}, FP8 keys paged (PSH {d}), {d} reps (median), aot {s}\n", .{ H, D, BP, RATIO, SPLIT, K, CAP, PSH, reps, dir });

    var failed: usize = 0;
    var skipped: usize = 0;
    for (starts) |start| for (row_counts) |R| {
        _ = arena_state.reset(.retain_capacity);
        const nk = @max(@divFloor(start + R, RATIO), 1);
        const ns = @divFloor(nk + SPLIT - 1, SPLIT);
        const start32: i32 = @intCast(start);
        try pos.upload(0, std.mem.asBytes(&start32));
        const n: usize = @intCast(R * ns * CAP);
        var bref = try cuda.DeviceBuffer.alloc(d, 8 * n);
        defer bref.free();
        var brb = try cuda.DeviceBuffer.alloc(d, 8 * n);
        defer brb.free();
        const want = try gpa.alloc(i64, n);
        defer gpa.free(want);
        const got = try gpa.alloc(i64, n);
        defer gpa.free(got);
        const c: Case = .{ .set = &set, .stream = stream, .qi = qi.ptr, .w = w.ptr, .ik = ik.ptr, .pos = pos.ptr, .pt = pt.ptr, .rows = R, .nk = nk, .ns = ns };
        std.debug.print("start {d}, rows {d}: {d} keys, {d} splits\n", .{ start, R, nk, ns });

        const have_ref = try once(&c, a, 0, bref, 0x00, want);
        var t_ref: f32 = 0;
        if (have_ref) {
            t_ref = try c.time(a, d, 0, bref.ptr, reps);
            std.debug.print("  _stream {d:.3} ms\n", .{t_ref});
        } else {
            std.debug.print("  _stream SKIPPED (no AOT variant for these args, see above)\n", .{});
            skipped += 1;
        }
        for (twins) |rb| {
            const name = twinName(rb);
            const short = if (rb == 1) "pf" else if (rb == 2) "rb2" else "rb4";
            if (!try once(&c, a, rb, brb, 0xA5, got)) {
                std.debug.print("  {s} SKIPPED (no AOT variant for these args, see above)\n  bits {s}: SKIPPED\n", .{ name, short });
                skipped += 1;
                continue;
            }
            const ok = if (have_ref) compare(short, got, want, R, ns) else blk: {
                std.debug.print("  bits {s}: SKIPPED (no _stream)\n", .{short});
                break :blk true;
            };
            if (!ok) failed += 1;
            const t = try c.time(a, d, rb, brb.ptr, reps);
            if (have_ref) {
                std.debug.print("  {s} {d:.3} ms (x{d:.2})\n", .{ name, t, t_ref / t });
            } else std.debug.print("  {s} {d:.3} ms\n", .{ name, t });
        }
    };
    if (failed == 0 and skipped == 0) {
        std.debug.print("stream-rb: PASS\n", .{});
        return;
    }
    std.debug.print("stream-rb: FAIL ({d} compares failed, {d} skipped)\n", .{ failed, skipped });
    return error.TestFailed;
}
