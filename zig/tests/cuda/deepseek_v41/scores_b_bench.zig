//! Row-bounded index scores (TF_DSV41_INDEX_BOUND) against the dense `_scores`: bit-exact, then timed (scores-b <aot>).

const std = @import("std");
const cuda = @import("cuda");
const check = @import("../check.zig");

const aot = cuda.aot;
const Gpu = check.Gpu;

const H = 32;
const D = 128;
const BP = 64;
/// graphs.Settings.bucket: a served window's keys end at a multiple of it (the served variants' div16 NK / strides).
const bucket: i64 = 2048;
const LONG: i64 = 154_183; // the bench's 131K prompt (long-131072: 154,183 prompt tokens)

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

fn median(xs: []f32) f32 {
    std.mem.sort(f32, xs, {}, std.sort.asc(f32));
    return xs[xs.len / 2];
}

const Shape = struct { ratio: i64, psh: u5 };
const shapes = [_]Shape{ .{ .ratio = 2, .psh = 7 }, .{ .ratio = 1, .psh = 8 } };

const Case = struct {
    name: []const u8,
    rows_mode: bool,
    /// each row's (slot, position)
    rows: []const [2]i64,
};

const mixed = blk: {
    var r: [24][2]i64 = undefined;
    for (0..20) |i| r[i] = .{ @as(i64, @intCast(i % 3)), 4_000 + @as(i64, @intCast(i / 3)) * 37 + @as(i64, @intCast(i % 3)) * 211 };
    for (20..24) |i| r[i] = .{ 3, LONG + @as(i64, @intCast(i - 20)) };
    break :blk r;
};
const all_long = blk: {
    var r: [24][2]i64 = undefined;
    for (0..24) |i| r[i] = .{ @as(i64, @intCast(i % 4)), LONG - 100 + @as(i64, @intCast(i)) };
    break :blk r;
};
const one_slot = blk: {
    var r: [16][2]i64 = undefined;
    for (0..16) |i| r[i] = .{ 0, 150_000 + @as(i64, @intCast(i)) };
    break :blk r;
};
const cases = [_]Case{
    .{ .name = "mixed 20 short + 4 at 154K (row mode)", .rows_mode = true, .rows = &mixed },
    .{ .name = "24 rows at 154K (row mode)", .rows_mode = true, .rows = &all_long },
    .{ .name = "one slot, 16 rows at 150K", .rows_mode = false, .rows = &one_slot },
};

const Bufs = struct { qi: u64, w: u64, ik: u64, pos: u64, sl: u64, pt: u64, pts: i64 };

fn launch(set: *const aot.Set, stream: cuda.Stream, a: std.mem.Allocator, bounded: bool, b: Bufs, s: Shape, c: Case, nk: i64, out: u64) !void {
    var args: std.ArrayList(aot.Arg) = .empty;
    try args.appendSlice(a, &.{
        aot.ptr("QI", "*bf16", b.qi),  aot.ptr("W", "*bf16", b.w),
        aot.int("w_stride", H),        aot.ptr("IK", "*u8", b.ik),
        aot.ptr("OUT", "*fp32", out),  aot.ptr("POS", if (c.rows_mode) "*i64" else "*i32", b.pos),
        aot.ptr("KEYS", "*fp32", out), aot.int("k_stride", @intCast(nk)),
        aot.int("NK", @intCast(nk)),   aot.int("o_stride", @intCast(nk)),
        aot.ptr("PT", "*i32", b.pt),   aot.int("PTS", @intCast(b.pts)),
    });
    if (c.rows_mode) try args.append(a, aot.ptr("SL", "*i64", b.sl));
    var consts: std.ArrayList(aot.Const) = .empty;
    try consts.appendSlice(a, &.{
        aot.ci("RATIO", s.ratio),                           aot.ci("H", H),
        aot.ci("D", D),                                     aot.ci("BP", BP),
        aot.cf("WS", @floatCast(1.0 / @sqrt(@as(f64, H)))), aot.cf("SCALE", @floatCast(1.0 / @sqrt(@as(f64, D)))),
        aot.ci("GATHER", 0),                                aot.ci("PSH", s.psh),
        aot.ci("KFP8", 1),                                  aot.ci("ROWS", @intFromBool(c.rows_mode)),
        aot.ci("CBS", 0),
    });
    const grid = [3]u32{ @intCast(c.rows.len), @intCast(@divFloor(nk + BP - 1, BP)), 1 };
    return set.run(stream, if (bounded) "_scores_b" else "_scores", grid, args.items, consts.items);
}

pub fn run(gpu: Gpu, rest: []const [:0]const u8) !void {
    const gpa = gpu.gpa;
    const d = gpu.d;
    if (rest.len < 1) return error.MissingArgument;
    const dir = rest[0];
    var reps: usize = 20;
    var i: usize = 1;
    while (i < rest.len) : (i += 1) {
        if (std.mem.eql(u8, rest[i], "--reps") and i + 1 < rest.len) {
            i += 1;
            reps = try std.fmt.parseInt(usize, rest[i], 10);
        } else return error.BadArgument;
    }
    var set = try aot.Set.load(gpa, gpu.io, d, gpu.ctx.device, dir);
    defer set.deinit();
    var stream = try cuda.Stream.init(d, false);
    defer stream.deinit();
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    var rng: Rng = .{ .s = 0x2545f4914f6cdd1d };

    const max_rows = 24;
    // queries bf16 [rows, H, D] in (-1, 1), weights bf16 [rows, H] signed
    const qh = try gpa.alloc(u16, max_rows * H * D);
    defer gpa.free(qh);
    for (qh) |*v| v.* = @truncate(@as(u32, @bitCast(2.0 * rng.unit() - 1.0)) >> 16);
    var qi = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(qh));
    defer qi.free();
    const wh = try gpa.alloc(u16, max_rows * H);
    defer gpa.free(wh);
    for (wh) |*v| v.* = @truncate(@as(u32, @bitCast(2.0 * rng.unit() - 1.0)) >> 16);
    var w = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(wh));
    defer w.free();
    // one key pool sized for ratio 1 at LONG + 64; FP8 rows: D e4m3 bytes (no NaN), then the fp32 scale
    const pool_pages: usize = ((@as(usize, @intCast(LONG)) + 64) >> 7) + 2;
    const key_rows: usize = pool_pages << 7;
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
    std.debug.print("scores-b: H {d}, D {d}, BP {d}, FP8 keys paged, {d} reps (median), aot {s}\n", .{ H, D, BP, reps, dir });

    var failed: usize = 0;
    var skipped: usize = 0;
    for (shapes) |s| for (cases) |c| {
        _ = arena_state.reset(.retain_capacity);
        const a = arena_state.allocator();
        // page tables: 4 slots, slot k starts k x 7 pages in; pages a slot a multiple of 16 as in every served variant
        const pages_a_slot: usize = std.mem.alignForward(usize, (@as(usize, @intCast(@divFloor(std.mem.alignForward(i64, LONG + 64, bucket), s.ratio))) >> s.psh) + 2, 16);
        const pool_pages_s = key_rows >> s.psh;
        const pth = try a.alloc(i32, 4 * pages_a_slot);
        for (0..4) |k| for (0..pages_a_slot) |p| {
            pth[k * pages_a_slot + p] = @intCast((p + 7 * k) % pool_pages_s);
        };
        var pt = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(pth));
        defer pt.free();
        var last: i64 = 0;
        for (c.rows) |r| last = @max(last, r[1]);
        // NK: the context bucket's end over the ratio; rows past their own visible keys are -inf in both kernels
        const nk = @divFloor(std.mem.alignForward(i64, last + 1, bucket), s.ratio);
        const R = c.rows.len;
        var pos = try cuda.DeviceBuffer.alloc(d, 8 * R);
        defer pos.free();
        var sl = try cuda.DeviceBuffer.alloc(d, 8 * R);
        defer sl.free();
        if (c.rows_mode) {
            const ph = try a.alloc(i64, R);
            const sh = try a.alloc(i64, R);
            for (c.rows, ph, sh) |r, *p, *x| {
                p.* = r[1];
                x.* = r[0];
            }
            try pos.upload(0, std.mem.sliceAsBytes(ph));
            try sl.upload(0, std.mem.sliceAsBytes(sh));
        } else {
            // one slot: q = POS[0] + r (int32)
            const p0: i32 = @intCast(c.rows[0][1]);
            try pos.upload(0, std.mem.asBytes(&p0));
        }
        const n: usize = R * @as(usize, @intCast(nk));
        var oref = try cuda.DeviceBuffer.alloc(d, 4 * n);
        defer oref.free();
        var ob = try cuda.DeviceBuffer.alloc(d, 4 * n);
        defer ob.free();
        const want = try gpa.alloc(u32, n);
        defer gpa.free(want);
        const got = try gpa.alloc(u32, n);
        defer gpa.free(got);
        const b: Bufs = .{ .qi = qi.ptr, .w = w.ptr, .ik = ik.ptr, .pos = pos.ptr, .sl = sl.ptr, .pt = pt.ptr, .pts = @intCast(pages_a_slot) };
        std.debug.print("ratio {d} (PSH {d}), {s}: {d} keys\n", .{ s.ratio, s.psh, c.name, nk });

        var have: [2]bool = undefined;
        for ([_]bool{ false, true }, [_]cuda.DeviceBuffer{ oref, ob }, [_][]u32{ want, got }, [_]u8{ 0x00, 0xA5 }, 0..) |bounded, buf, host, poison, j| {
            try buf.fill8(poison, stream.handle);
            launch(&set, stream, a, bounded, b, s, c, nk, buf.ptr) catch |e| switch (e) {
                error.MissingTritonVariant => {
                    have[j] = false;
                    std.debug.print("  {s} SKIPPED (no AOT variant for these args)\n", .{if (bounded) "_scores_b" else "_scores"});
                    skipped += 1;
                    continue;
                },
                else => return e,
            };
            try stream.synchronize();
            try buf.download(0, std.mem.sliceAsBytes(host));
            have[j] = true;
        }
        if (!have[0] or !have[1]) continue;
        if (std.mem.indexOfDiff(u32, got, want)) |at| {
            var bad: usize = 0;
            for (got, want) |x, y| bad += @intFromBool(x != y);
            std.debug.print("  bits: FAIL ({d} of {d} differ; first row {d} key {d}: {x} vs _scores {x})\n", .{ bad, n, at / @as(usize, @intCast(nk)), at % @as(usize, @intCast(nk)), got[at], want[at] });
            failed += 1;
        } else std.debug.print("  bits: PASS ({d} scores)\n", .{n});
        var ms: [2]f32 = undefined;
        for ([_]bool{ false, true }, [_]cuda.DeviceBuffer{ oref, ob }, 0..) |bounded, buf, j| {
            var t0 = try cuda.Event.init(d, true);
            defer t0.deinit();
            var t1 = try cuda.Event.init(d, true);
            defer t1.deinit();
            for (0..2) |_| try launch(&set, stream, a, bounded, b, s, c, nk, buf.ptr);
            var ts: [256]f32 = undefined;
            const m = @max(1, @min(reps, ts.len));
            for (0..m) |k| {
                try t0.record(stream);
                try launch(&set, stream, a, bounded, b, s, c, nk, buf.ptr);
                try t1.record(stream);
                try t1.synchronize();
                ts[k] = try cuda.Event.elapsedMs(t0, t1);
            }
            ms[j] = median(ts[0..m]);
        }
        std.debug.print("  _scores {d:.3} ms, _scores_b {d:.3} ms (x{d:.2})\n", .{ ms[0], ms[1], ms[0] / ms[1] });
    };
    if (failed == 0 and skipped == 0) {
        std.debug.print("scores-b: PASS\n", .{});
        return;
    }
    std.debug.print("scores-b: FAIL ({d} compares failed, {d} skipped)\n", .{ failed, skipped });
    return error.TestFailed;
}
