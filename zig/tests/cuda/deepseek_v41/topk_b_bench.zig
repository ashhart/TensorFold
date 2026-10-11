//! Row-bounded decode top-k (TF_DSV41_INDEX_BOUND, CUDA and Triton) against the served kernels: bit-exact, timed.

const std = @import("std");
const cuda = @import("cuda");
const dsv41 = @import("dsv41_kernels");
const check = @import("../check.zig");

const aot = cuda.aot;
const Gpu = check.Gpu;
const tk = dsv41.ops.topk;

const K: i64 = 512;
const CB: i64 = 2048;
const BS: i64 = 8;
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

const Kind = enum { cuda_select, cuda_both, cuda_blocks, dtopk_select, dtopk_blocks };

const Shape = struct { name: []const u8, kind: Kind, ratio: i64, long: i64 };
const shapes = [_]Shape{
    .{ .name = "CUDA select, ratio 2", .kind = .cuda_select, .ratio = 2, .long = LONG },
    .{ .name = "CUDA select + blocks, ratio 1", .kind = .cuda_both, .ratio = 1, .long = 120_000 },
    .{ .name = "CUDA blocks only, ratio 1", .kind = .cuda_blocks, .ratio = 1, .long = LONG },
    .{ .name = "_dtopk select (mode 0), ratio 1", .kind = .dtopk_select, .ratio = 1, .long = LONG },
    .{ .name = "_dtopk blocks (mode 2), ratio 1", .kind = .dtopk_blocks, .ratio = 1, .long = 1_040_000 },
};

const Set = enum { mixed, all_long, one_slot };
const sets = [_]Set{ .mixed, .all_long, .one_slot };

fn positions(a: std.mem.Allocator, s: Set, long: i64) ![]i64 {
    switch (s) {
        .mixed => {
            const p = try a.alloc(i64, 24);
            const short = [_]i64{ 100, 700, 1023 }; // nvis < K at ratio 1 / 2
            for (0..20) |i| p[i] = if (i < short.len) short[i] else 4_000 + @as(i64, @intCast(i / 3)) * 37 + @as(i64, @intCast(i % 3)) * 211;
            for (20..24) |i| p[i] = long + @as(i64, @intCast(i - 20));
            return p;
        },
        .all_long => {
            const p = try a.alloc(i64, 24);
            for (p, 0..) |*x, i| x.* = long - 100 + @as(i64, @intCast(i));
            return p;
        },
        .one_slot => {
            const p = try a.alloc(i64, 16);
            for (p, 0..) |*x, i| x.* = long - 16 + @as(i64, @intCast(i));
            return p;
        },
    }
}

/// block.zig's topkPlan: (cluster CTAs, entries a thread) for `nk` entries.
fn plan(nk: i64, cl_in: ?i64) [2]i64 {
    const cl = cl_in orelse @max(1, @min(8, std.math.divCeil(i64, nk, 4096) catch 1));
    var ept = @max(1, std.math.divCeil(i64, nk, cl * 512) catch 1);
    ept += 1 - @mod(ept, 2);
    return .{ cl, ept };
}

/// Index scores as `_scores` leaves them: row r's first nvis entries patterned, -inf after.
fn fillScores(rng: *Rng, out: []f32, pos: []const i64, ratio: i64, nk: usize) void {
    for (pos, 0..) |q, r| {
        const nvis: usize = @intCast(@divFloor(q + 1, ratio));
        const row = out[r * nk ..][0..nk];
        for (row, 0..) |*v, j| {
            if (j >= nvis) {
                v.* = -std.math.inf(f32);
                continue;
            }
            const u = rng.unit();
            v.* = switch (r % 4) {
                0 => switch (rng.next() % 16) {
                    0 => 0.0,
                    1 => -0.0,
                    else => @floor(u * 8.0) / 8.0,
                },
                1 => 4.0 * u - 1.0,
                2 => switch (rng.next() % 61) {
                    0 => -std.math.inf(f32),
                    1 => std.math.inf(f32),
                    2 => std.math.nan(f32),
                    else => 4.0 * u - 1.0,
                },
                else => 0.5,
            };
        }
    }
}

const Bufs = struct { s: u64, ss: i64, nk: i64, pos: u64, pos64: bool, R: usize, out0: u64, cnt0: u64, out1: u64 };

fn launch(k: *const dsv41.Kernels, set: *const aot.Set, stream: cuda.Stream, a: std.mem.Allocator, bounded: bool, sh: Shape, b: Bufs) !void {
    const o = k.others(stream);
    const nb = std.math.divCeil(i64, b.nk, BS) catch unreachable;
    switch (sh.kind) {
        .cuda_select, .cuda_both, .cuda_blocks => {
            const sel: tk.Job = .{ .s = b.s, .ss = b.ss, .out = b.out0, .os = K, .cnt = b.cnt0, .nk = @intCast(b.nk), .k = @intCast(K), .mode = 0, .ept = 0 };
            const pl = plan(b.nk, null);
            var args: tk.Args = .{ .job = undefined, .pos = b.pos, .pos64 = @intFromBool(b.pos64), .ratio = @intCast(sh.ratio), .bs = 1 };
            var jobs: usize = 1;
            var cl: usize = @intCast(pl[0]);
            var ept: usize = @intCast(pl[1]);
            switch (sh.kind) {
                .cuda_select => {
                    args.job = .{ sel, sel };
                    args.job[0].ept = @intCast(pl[1]);
                    args.job[1] = args.job[0];
                },
                .cuda_both => {
                    args.bs = @intCast(BS);
                    jobs = 2;
                    var s0 = sel;
                    s0.ept = @intCast(pl[1]);
                    const e1 = plan(nb, pl[0])[1];
                    const s1: tk.Job = .{ .s = b.s, .ss = b.ss, .out = b.out1, .os = CB, .cnt = 0, .nk = @intCast(nb), .k = @intCast(CB), .mode = 2, .ept = @intCast(e1) };
                    args.job = .{ s0, s1 };
                    ept = @intCast(@max(pl[1], e1));
                },
                else => {
                    args.bs = @intCast(BS);
                    const bp = plan(nb, null);
                    const s1: tk.Job = .{ .s = b.s, .ss = b.ss, .out = b.out1, .os = CB, .cnt = 0, .nk = @intCast(nb), .k = @intCast(CB), .mode = 2, .ept = @intCast(bp[1]) };
                    args.job = .{ s1, s1 };
                    cl = @intCast(bp[0]);
                    ept = @intCast(bp[1]);
                },
            }
            return if (bounded) o.topKB(args, jobs, b.R, cl, ept) else o.topK(args, jobs, b.R, cl, ept);
        },
        .dtopk_select, .dtopk_blocks => {
            const blocks = sh.kind == .dtopk_blocks;
            const kk: i64 = if (blocks) CB else K;
            const out = if (blocks) b.out1 else b.out0;
            var args: std.ArrayList(aot.Arg) = .empty;
            try args.appendSlice(a, &.{
                aot.ptr("S", "*fp32", b.s),                             aot.int("s_stride", @intCast(b.ss)),
                aot.ptr("P", "*fp32", b.s),                             aot.int("p_stride", 0),
                aot.ptr("POS", if (b.pos64) "*i64" else "*i32", b.pos), aot.int("NK", @intCast(if (blocks) nb else b.nk)),
                aot.ptr("OUT", "*i32", out),                            aot.int("o_stride", @intCast(kk)),
                aot.ptr("CNT", "*i32", if (blocks) out else b.cnt0),
            });
            const consts = [_]aot.Const{
                aot.ci("RATIO", sh.ratio),              aot.ci("K", kk),    aot.ci("MODE", if (blocks) 2 else 0),
                aot.ci("ROWS", @intFromBool(b.pos64)),  aot.ci("T", 4096),  aot.ci("BS", BS),
                aot.ci("RB", 8),                        aot.ci("NPASS", 8), aot.ci("SORT", 0),
                aot.ci("COUNT", @intFromBool(!blocks)), aot.ci("KP", kk),
            };
            return set.run(stream, if (bounded) "_dtopk_b" else "_dtopk", .{ @intCast(b.R), 1, 1 }, args.items, &consts);
        },
    }
}

pub fn run(gpu: Gpu, k: *const dsv41.Kernels, rest: []const [:0]const u8) !void {
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
    var rng: Rng = .{ .s = 0x9e3779b97f4a7c15 };
    std.debug.print("topk-b: K {d}, candidate blocks {d} x {d}, {d} reps (median), aot {s}\n", .{ K, CB, BS, reps, dir });

    var failed: usize = 0;
    var skipped: usize = 0;
    for (shapes) |sh| for (sets) |st| {
        _ = arena_state.reset(.retain_capacity);
        const a = arena_state.allocator();
        const pos = try positions(a, st, sh.long);
        const R = pos.len;
        var last: i64 = 0;
        for (pos) |q| last = @max(last, q);
        // NK: the context bucket's end over the ratio (a multiple of 16); keys past a row's visible end are -inf
        const nk: i64 = @divFloor(std.mem.alignForward(i64, last + 1, bucket), sh.ratio);
        const nku: usize = @intCast(nk);
        const host = try gpa.alloc(f32, R * nku);
        defer gpa.free(host);
        fillScores(&rng, host, pos, sh.ratio, nku);
        var s = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(host));
        defer s.free();
        const row_mode = st != .one_slot;
        var pbuf = try cuda.DeviceBuffer.alloc(d, 8 * R);
        defer pbuf.free();
        if (row_mode) try pbuf.upload(0, std.mem.sliceAsBytes(pos)) else {
            const p0: i32 = @intCast(pos[0]);
            try pbuf.upload(0, std.mem.asBytes(&p0));
        }
        // outputs, one set a kernel: selection [R, K] | counts [R] | blocks [R, CB], as one buffer
        const words: usize = R * @as(usize, @intCast(K)) + R + R * @as(usize, @intCast(CB));
        var outs: [2]cuda.DeviceBuffer = .{ try cuda.DeviceBuffer.alloc(d, 4 * words), try cuda.DeviceBuffer.alloc(d, 4 * words) };
        defer for (&outs) |*x| x.free();
        const hosts: [2][]u32 = .{ try a.alloc(u32, words), try a.alloc(u32, words) };
        var bufs: [2]Bufs = undefined;
        for (&bufs, outs) |*b, ob| b.* = .{
            .s = s.ptr,
            .ss = nk,
            .nk = nk,
            .pos = pbuf.ptr,
            .pos64 = row_mode,
            .R = R,
            .out0 = ob.ptr,
            .cnt0 = ob.ptr + 4 * R * @as(usize, @intCast(K)),
            .out1 = ob.ptr + 4 * (R * @as(usize, @intCast(K)) + R),
        };
        const label = switch (st) {
            .mixed => "row window: 20 short + 4 long",
            .all_long => "row window: 24 long",
            .one_slot => "one slot: 16 rows",
        };
        std.debug.print("{s} at {d}, {s}: NK {d}\n", .{ sh.name, sh.long, label, nk });
        var have: [2]bool = .{ false, false };
        for ([_]bool{ false, true }, outs, hosts, [_]u8{ 0x00, 0xA5 }, 0..) |bounded, ob, h, poison, j| {
            try ob.fill8(poison, stream.handle);
            launch(k, &set, stream, a, bounded, sh, bufs[j]) catch |e| switch (e) {
                error.MissingTritonVariant => {
                    std.debug.print("  {s} SKIPPED (no AOT variant for these args)\n", .{if (bounded) "twin" else "served"});
                    skipped += 1;
                    continue;
                },
                else => return e,
            };
            try stream.synchronize();
            try ob.download(0, std.mem.sliceAsBytes(h));
            have[j] = true;
        }
        if (!have[0] or !have[1]) continue;
        // the poison differs, so a word neither kernel wrote differs too: every output word must be written alike
        const used: usize = switch (sh.kind) {
            .cuda_select, .dtopk_select => R * @as(usize, @intCast(K)) + R,
            .cuda_both => words,
            .cuda_blocks, .dtopk_blocks => words, // the selection / counts regions stay poisoned: compared below
        };
        const lo: usize = switch (sh.kind) {
            .cuda_blocks, .dtopk_blocks => R * @as(usize, @intCast(K)) + R,
            else => 0,
        };
        if (std.mem.indexOfDiff(u32, hosts[1][lo..used], hosts[0][lo..used])) |at| {
            var bad: usize = 0;
            for (hosts[1][lo..used], hosts[0][lo..used]) |x, y| bad += @intFromBool(x != y);
            std.debug.print("  bytes: FAIL ({d} of {d} words differ; first word {d}: {x} vs served {x})\n", .{ bad, used - lo, lo + at, hosts[1][lo + at], hosts[0][lo + at] });
            failed += 1;
        } else std.debug.print("  bytes: PASS ({d} output words)\n", .{used - lo});
        var ms: [2]f32 = undefined;
        for ([_]bool{ false, true }, 0..) |bounded, j| {
            var t0 = try cuda.Event.init(d, true);
            defer t0.deinit();
            var t1 = try cuda.Event.init(d, true);
            defer t1.deinit();
            for (0..2) |_| try launch(k, &set, stream, a, bounded, sh, bufs[j]);
            var ts: [256]f32 = undefined;
            const m = @max(1, @min(reps, ts.len));
            for (0..m) |x| {
                try t0.record(stream);
                try launch(k, &set, stream, a, bounded, sh, bufs[j]);
                try t1.record(stream);
                try t1.synchronize();
                ts[x] = try cuda.Event.elapsedMs(t0, t1);
            }
            ms[j] = median(ts[0..m]);
        }
        std.debug.print("  served {d:.3} ms, twin {d:.3} ms (x{d:.2})\n", .{ ms[0], ms[1], ms[0] / ms[1] });
    };
    if (failed == 0 and skipped == 0) {
        std.debug.print("topk-b: PASS\n", .{});
        return;
    }
    std.debug.print("topk-b: FAIL ({d} compares failed, {d} skipped)\n", .{ failed, skipped });
    return error.TestFailed;
}
