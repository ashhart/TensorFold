//! exl3-prefill engine and agree: whole prefills through the deep chunks, timed and compared with the batched pass.

const std = @import("std");
const cuda = @import("cuda");
const check = @import("../../check.zig");
const flash = @import("flashnext");
const Gpu = check.Gpu;

// ---------------------------------------------------------------------------------------------------- engine

const prompt = flash.prompt;
const xpass = flash.exl3.pass;
const xpre = flash.exl3.prefill;
const vocab = prompt.vocabulary;

fn promote(bits: u16) f32 {
    return @bitCast(@as(u32, bits) << 16);
}

const Eng = struct {
    gpu: Gpu,
    stream: cuda.Stream,
    glue: flash.Glue,
    e: *prompt.Engine,
    seq: prompt.Seq,
    attached: xpass.Attached,

    pub fn init(gpu: Gpu, dir: []const u8, lines: ?usize, cap: usize, deep_rows: usize) !*Eng {
        const g = try gpu.gpa.create(Eng);
        errdefer gpu.gpa.destroy(g);
        g.gpu = gpu;
        g.stream = try cuda.Stream.init(gpu.d, true);
        g.glue = try flash.Glue.load(gpu.d);
        const t0 = check.now(gpu.io);
        g.e = try prompt.Engine.initRank(gpu.gpa, gpu.io, @constCast(gpu.d), &g.stream, dir, 1, 0);
        var free0: usize = 0;
        var total: usize = 0;
        try gpu.d.check(gpu.d.api.cuMemGetInfo_v2(&free0, &total), "cuMemGetInfo");
        _ = try g.e.reserve(@min(cap, g.e.chunk));
        _ = try g.e.reserveScores(cap, @min(cap, g.e.chunk));
        var free1: usize = 0;
        try gpu.d.check(gpu.d.api.cuMemGetInfo_v2(&free1, &total), "cuMemGetInfo");
        g.seq = try g.e.newSeq(cap);
        var free2: usize = 0;
        try gpu.d.check(gpu.d.api.cuMemGetInfo_v2(&free2, &total), "cuMemGetInfo");
        if (deep_rows > 0) try xpre.reserve(gpu.gpa, gpu.d, g.e, deep_rows);
        var free3: usize = 0;
        try gpu.d.check(gpu.d.api.cuMemGetInfo_v2(&free3, &total), "cuMemGetInfo");
        var b = try xpass.budgetFromEnv(0);
        if (lines) |n| b.lines = n;
        g.attached = try xpass.attach(gpu.gpa, gpu.io, gpu.d, g.stream, g.e.layers[0..g.e.n_layers], null, g.e.chunk, b, null);
        var free4: usize = 0;
        try gpu.d.check(gpu.d.api.cuMemGetInfo_v2(&free4, &total), "cuMemGetInfo");
        const gib = struct {
            fn f(x: usize) f64 {
                return @as(f64, @floatFromInt(x)) / (1 << 30);
            }
        }.f;
        std.debug.print("engine: loaded in {d:.1} s; card {d:.2} GiB; after weights {d:.2} free; pass scratch (chunk {d}) {d:.2} GiB; caches for {d} tokens {d:.2} GiB; deep chunks of {d} rows {d:.2} GiB; pool {d} lines a layer {d:.2} GiB; {d:.2} GiB free after\n", .{ @as(f64, @floatFromInt(check.now(gpu.io) - t0)) / 1e9, gib(total), gib(free0), g.e.chunk, gib(free0 - free1), cap, gib(free1 - free2), deep_rows, gib(free2 - free3), g.attached.lines, gib(g.attached.device_bytes), gib(free4) });
        return g;
    }

    pub fn deinit(g: *Eng) void {
        xpre.release();
        g.seq.deinit(g.gpu.gpa);
        g.e.deinit();
        g.glue.deinit();
        g.stream.deinit();
        g.gpu.gpa.destroy(g);
    }

    fn tri(g: *Eng) flash.Tri {
        return .{ .own = &g.glue, .s = g.stream };
    }

    fn reset(g: *Eng) !void {
        const s = &g.seq;
        for ([_][]cuda.DeviceBuffer{ s.conv, s.state, s.kc, s.vc, s.ikc, s.pooled }) |bs| for (bs) |b| try b.fill8(0, g.stream.handle);
        try s.tail.fill8(0, g.stream.handle);
        try g.stream.synchronize();
        s.fed = 0;
        s.history = .{ g.e.eos, g.e.eos };
    }

    fn poolStats(g: *Eng) flash.exl3_pool.Stats {
        return xpass.poolStats(g.e.layers[0..g.e.n_layers], null);
    }
};

fn top5(row: []const u16) [5]u32 {
    var ids: [5]u32 = @splat(0);
    var vals: [5]f32 = @splat(-std.math.inf(f32));
    for (row, 0..) |b, i| {
        const v = promote(b);
        if (!(v > vals[4])) continue;
        var at: usize = 4;
        while (at > 0 and v > vals[at - 1]) : (at -= 1) {
            vals[at] = vals[at - 1];
            ids[at] = ids[at - 1];
        }
        vals[at] = v;
        ids[at] = @intCast(i);
    }
    return ids;
}

/// One prefill of `ids` from a fresh sequence: ms, and the last row's logits copied into `out`.
fn prefillOnce(g: *Eng, ids: []const u32, deep_on: bool, out: []u16) !f64 {
    var o = xpre.optionsFromEnv();
    o.enabled = deep_on;
    if (rows_override) |r| o.rows = r;
    xpre.state.options = o;
    try g.reset();
    const t0 = check.now(g.gpu.io);
    try g.e.forward(flash.Tri, g.tri(), &g.seq, ids, .{ .logits = .last });
    const ms_ = @as(f64, @floatFromInt(check.now(g.gpu.io) - t0)) / 1e6;
    @memcpy(std.mem.sliceAsBytes(out), g.seq.logits);
    return ms_;
}

/// Greedy decode rounds after each engine-mode prefill (TF_TEST_DECODE, default 32; 0: none).
var decode_steps: usize = 32;

fn decodeAfter(g: *Eng, n: usize) !void {
    var out = try cuda.DeviceBuffer.alloc(g.gpu.d, vocab * 2);
    defer out.free();
    var tok = prompt.argmax(g.seq.logits);
    const before = g.poolStats();
    const t0 = check.now(g.gpu.io);
    var first: f64 = 0;
    for (0..decode_steps) |i| {
        const q = [_]*prompt.Seq{&g.seq};
        try g.e.round(flash.Tri, g.tri(), &q, (&tok)[0..1], out.ptr);
        tok = prompt.argmax(g.seq.logits);
        if (i == 7) first = @as(f64, @floatFromInt(check.now(g.gpu.io) - t0)) / 1e6;
    }
    const t = @as(f64, @floatFromInt(check.now(g.gpu.io) - t0)) / 1e6;
    const after = g.poolStats();
    const hits = after.hit_slots - before.hit_slots;
    const all = hits + after.miss_slots - before.miss_slots;
    std.debug.print("        decode after {d} tokens: {d:.2} ms/token over {d} steps (first 8: {d:.2} ms/token; hit {d:.1}%)\n", .{ n, t / @as(f64, @floatFromInt(decode_steps)), decode_steps, first / 8, 100 * @as(f64, @floatFromInt(hits)) / @as(f64, @floatFromInt(@max(all, 1))) });
}

/// The deep chunk's rows for the next prefills (engine mode's ROWS list), else the environment's.
var rows_override: ?usize = null;

pub fn engineRun(gpu: Gpu, dir: []const u8, rest: []const [:0]const u8) !void {
    const a = gpu.gpa;
    if (rest.len < 2) return error.MissingArgument;
    const text = try std.Io.Dir.cwd().readFileAlloc(gpu.io, rest[0], a, .limited(1 << 28));
    defer a.free(text);
    const all_ids = try a.alloc(u32, text.len / 4);
    defer a.free(all_ids);
    for (all_ids, 0..) |*x, i| x.* = std.mem.readInt(u32, text[4 * i ..][0..4], .little);
    var ns: [16]usize = undefined;
    var n_ns: usize = 0;
    var it = std.mem.splitScalar(u8, rest[1], ',');
    while (it.next()) |tok| {
        ns[n_ns] = try std.fmt.parseInt(usize, tok, 10);
        n_ns += 1;
    }
    const lines: ?usize = if (rest.len > 2 and !std.mem.eql(u8, rest[2], "-")) try std.fmt.parseInt(usize, rest[2], 10) else null;
    const cmp = rest.len > 3 and std.mem.eql(u8, rest[3], "cmp");
    // ROWS: deep chunk sizes to sweep (rest[3] when not "cmp", or rest[4])
    var rs: [16]usize = undefined;
    var n_rs: usize = 0;
    const rows_arg: ?[]const u8 = if (rest.len > 3 and !cmp) rest[3] else if (rest.len > 4) rest[4] else null;
    if (rows_arg) |ra| {
        var rit = std.mem.splitScalar(u8, ra, ',');
        while (rit.next()) |tok| {
            rs[n_rs] = try std.fmt.parseInt(usize, tok, 10);
            n_rs += 1;
        }
    }
    var max_n: usize = 0;
    for (ns[0..n_ns]) |n| max_n = @max(max_n, n);
    if (max_n > all_ids.len) return error.NotEnoughIds;
    if (std.c.getenv("TF_TEST_DECODE")) |v| decode_steps = std.fmt.parseInt(usize, std.mem.span(v), 10) catch decode_steps;
    const o = xpre.optionsFromEnv();
    var max_r = o.rows;
    for (rs[0..n_rs]) |r| max_r = @max(max_r, r);
    if (n_rs > 0) max_r = 0;
    for (rs[0..n_rs]) |r| max_r = @max(max_r, r);
    const deep_rows = @min(max_r, (max_n + 255) / 256 * 256);
    const g = try Eng.init(gpu, dir, lines, max_n + 64, if (o.enabled) deep_rows else 0);
    defer g.deinit();
    const old = try a.alloc(u16, vocab);
    defer a.free(old);
    const new = try a.alloc(u16, vocab);
    defer a.free(new);
    // warm-up: kernels, the pool's first lines
    _ = try prefillOnce(g, all_ids[0..@min(max_n, 1024)], o.enabled, new);
    var all_ok = true;
    if (n_rs == 0) {
        rs[0] = 0;
        n_rs = 1;
    }
    for (rs[0..n_rs]) |rr| for (ns[0..n_ns]) |n| {
        rows_override = if (rr == 0) null else rr;
        if (rr != 0) std.debug.print("deep chunk rows {d}:\n", .{rr});
        const ids = all_ids[0..n];
        const p0 = g.poolStats();
        xpre.state.stats = .{};
        const t_new = try prefillOnce(g, ids, o.enabled, new);
        const p1 = g.poolStats();
        const xs_ = xpre.state.stats;
        std.debug.print("PREFILL {d:6} tokens ({s}): {d:9.1} ms = {d:7.1} rows/s; pool copies {d} ({d:.1} GiB, {d:.2} a layer per 1k rows), deep chunks {d}, passes {d}, prefetched {d}, members {d}; host: picks wait {d:.1} ms, plan {d:.1} ms\n", .{ n, if (o.enabled) "expert-major deep" else "batched pass", t_new, @as(f64, @floatFromInt(n)) / (t_new / 1e3), p1.misses - p0.misses, @as(f64, @floatFromInt(p1.misses - p0.misses)) * 1.758 / 1024, @as(f64, @floatFromInt(p1.misses - p0.misses)) / 48 / (@as(f64, @floatFromInt(n)) / 1000), xs_.chunks, xs_.passes, xs_.prefetched, xs_.members, @as(f64, @floatFromInt(xs_.host_wait_ns)) / 1e6, @as(f64, @floatFromInt(xs_.plan_ns)) / 1e6 });
        // decode after the prefill: the pools were left empty by the ring (or hold the batched pass's lines)
        if (decode_steps > 0) try decodeAfter(g, n);
        if (!cmp) continue;
        const q0 = g.poolStats();
        const t_old = try prefillOnce(g, ids, false, old);
        const q1 = g.poolStats();
        std.debug.print("        same prompt, batched pass: {d:9.1} ms = {d:7.1} rows/s; pool copies {d}\n", .{ t_old, @as(f64, @floatFromInt(n)) / (t_old / 1e3), q1.misses - q0.misses });
        if (decode_steps > 0) try decodeAfter(g, n);
        var dmax: f32 = 0;
        var peak: f32 = 0;
        var same: usize = 0;
        for (old, new) |x, y| {
            dmax = @max(dmax, @abs(promote(x) - promote(y)));
            peak = @max(peak, @abs(promote(x)));
            same += @intFromBool(x == y);
        }
        const to = top5(old);
        const tn = top5(new);
        var overlap: usize = 0;
        for (to) |x| for (tn) |y| {
            overlap += @intFromBool(x == y);
        };
        const margin_old = promote(old[to[0]]) - promote(old[to[1]]);
        std.debug.print("CMP {d:6} tokens: last-row logits max |d| {d:.4} (peak {d:.2}, {d:.1}% of logits bit-equal); top-1 {d} vs {d} ({s}, batched-pass margin {d:.4}); top-5 overlap {d}/5\n", .{ n, dmax, peak, 100 * @as(f64, @floatFromInt(same)) / vocab, to[0], tn[0], if (to[0] == tn[0]) "equal" else "DIFFERENT", margin_old, overlap });
        if (to[0] != tn[0] and margin_old > 0.25) all_ok = false;
    };
    const ps = g.poolStats();
    std.debug.print("pool overall: {d} rounds, {d} copies, hit {d:.1}%\n", .{ ps.rounds, ps.misses, 100 * @as(f64, @floatFromInt(ps.hit_slots)) / @as(f64, @floatFromInt(@max(1, ps.hit_slots + ps.miss_slots))) });
    try check.expect(all_ok, "top-1 equal wherever the batched pass's margin exceeds bf16 noise", .{});
    check.pass("engine prefill measured", .{});
}

fn readIds(gpu: Gpu, path: []const u8) ![]u32 {
    const a = gpu.gpa;
    const text = try std.Io.Dir.cwd().readFileAlloc(gpu.io, path, a, .limited(1 << 28));
    defer a.free(text);
    const ids = try a.alloc(u32, text.len / 4);
    for (ids, 0..) |*x, i| x.* = std.mem.readInt(u32, text[4 * i ..][0..4], .little);
    return ids;
}

pub fn agreeRun(gpu: Gpu, dir: []const u8, rest: []const [:0]const u8) !void {
    const a = gpu.gpa;
    if (rest.len < 4) return error.MissingArgument;
    const all_ids = try readIds(gpu, rest[0]);
    defer a.free(all_ids);
    const first = try std.fmt.parseInt(usize, rest[1], 10);
    const step = try std.fmt.parseInt(usize, rest[2], 10);
    const count = try std.fmt.parseInt(usize, rest[3], 10);
    const lines: ?usize = if (rest.len > 4) try std.fmt.parseInt(usize, rest[4], 10) else null;
    const max_n = first + step * (count - 1);
    if (max_n + 1 > all_ids.len) return error.NotEnoughIds;
    const o = xpre.optionsFromEnv();
    const g = try Eng.init(gpu, dir, lines, max_n + 64, @min(o.rows, (max_n + 255) / 256 * 256));
    defer g.deinit();
    const old = try a.alloc(u16, vocab);
    defer a.free(old);
    const new = try a.alloc(u16, vocab);
    defer a.free(new);
    var agree: usize = 0;
    var acc_old: usize = 0;
    var acc_new: usize = 0;
    var top5_same: usize = 0;
    var sum_d: f64 = 0;
    var max_d: f64 = 0;
    var n_d: usize = 0;
    var disagree_margin_max: f32 = 0;
    var t_old: f64 = 0;
    var t_new: f64 = 0;
    for (0..count) |i| {
        const n = first + step * i;
        const ids = all_ids[0..n];
        t_old += try prefillOnce(g, ids, false, old);
        t_new += try prefillOnce(g, ids, o.enabled, new);
        const to = top5(old);
        const tn = top5(new);
        const want = all_ids[n];
        agree += @intFromBool(to[0] == tn[0]);
        acc_old += @intFromBool(to[0] == want);
        acc_new += @intFromBool(tn[0] == want);
        top5_same += @intFromBool(std.mem.eql(u32, &to, &tn));
        for (to) |id| {
            const d = @abs(@as(f64, promote(old[id])) - promote(new[id]));
            sum_d += d;
            max_d = @max(max_d, d);
            n_d += 1;
        }
        if (to[0] != tn[0]) {
            const m = promote(old[to[0]]) - promote(old[to[1]]);
            disagree_margin_max = @max(disagree_margin_max, m);
            std.debug.print("  length {d}: top-1 {d} (batched) vs {d} (deep), batched margin {d:.4}, deep margin {d:.4}\n", .{ n, to[0], tn[0], m, promote(new[tn[0]]) - promote(new[tn[1]]) });
        }
    }
    std.debug.print("AGREE {d} prompt lengths {d}..{d}: top-1 equal {d}/{d}, top-5 identical in order {d}/{d}; next-token accuracy batched {d}/{d}, deep {d}/{d}; |d| on the batched top-5: mean {d:.4}, max {d:.4}; largest batched margin where top-1 differs {d:.4}; time batched {d:.1} s, deep {d:.1} s\n", .{ count, first, max_n, agree, count, top5_same, count, acc_old, count, acc_new, count, sum_d / @as(f64, @floatFromInt(n_d)), max_d, disagree_margin_max, t_old / 1e3, t_new / 1e3 });
    check.pass("agreement measured", .{});
}

/// Greedy tokens after the prefill that just ran: the next `out.len` ids, each fed back through a decode round.
fn greedy(g: *Eng, out: []u32) !void {
    var buf = try cuda.DeviceBuffer.alloc(g.gpu.d, vocab * 2);
    defer buf.free();
    var tok = prompt.argmax(g.seq.logits);
    for (out) |*o| {
        o.* = tok;
        const q = [_]*prompt.Seq{&g.seq};
        try g.e.round(flash.Tri, g.tri(), &q, (&tok)[0..1], buf.ptr);
        tok = prompt.argmax(g.seq.logits);
    }
}

/// One prompt through the batched pass and through the deep chunks at each chunk size: same logits bytes and tokens.
pub fn equalRun(gpu: Gpu, dir: []const u8, rest: []const [:0]const u8) !void {
    const a = gpu.gpa;
    if (rest.len < 3) return error.MissingArgument;
    const all_ids = try readIds(gpu, rest[0]);
    defer a.free(all_ids);
    const n = try std.fmt.parseInt(usize, rest[1], 10);
    if (n == 0 or n > all_ids.len) return error.NotEnoughIds;
    var sizes: [8]usize = undefined;
    var n_sizes: usize = 0;
    var it = std.mem.splitScalar(u8, rest[2], ',');
    while (it.next()) |tok| : (n_sizes += 1) sizes[n_sizes] = try std.fmt.parseInt(usize, tok, 10);
    const lines: ?usize = if (rest.len > 3 and !std.mem.eql(u8, rest[3], "-")) try std.fmt.parseInt(usize, rest[3], 10) else null;
    var max_rows: usize = 0;
    for (sizes[0..n_sizes]) |r| max_rows = @max(max_rows, r);
    const g = try Eng.init(gpu, dir, lines, n + 64, @min(max_rows, (n + 255) / 256 * 256));
    defer g.deinit();
    const ids = all_ids[0..n];
    const want = try a.alloc(u16, vocab);
    defer a.free(want);
    const got = try a.alloc(u16, vocab);
    defer a.free(got);
    var want_tokens: [24]u32 = undefined;
    var got_tokens: [24]u32 = undefined;
    // TF_EQUAL_FIRST_SIZE=1: the first chunk size is the reference (the batched pass is not the canonical reduction's)
    const first = std.c.getenv("TF_EQUAL_FIRST_SIZE") != null;
    rows_override = if (first) sizes[0] else null;
    _ = try prefillOnce(g, ids, first, want);
    try greedy(g, &want_tokens);
    var all_equal = true;
    for (sizes[if (first) 1 else 0..n_sizes]) |rows| {
        rows_override = rows;
        _ = try prefillOnce(g, ids, true, got);
        try greedy(g, &got_tokens);
        const logits_equal = std.mem.eql(u8, std.mem.sliceAsBytes(want), std.mem.sliceAsBytes(got));
        const tokens_equal = std.mem.eql(u32, &want_tokens, &got_tokens);
        std.debug.print("EQUAL {d} tokens, deep chunks of {d} rows vs the batched pass: last-row logits {s}, {d} greedy tokens {s}\n", .{ n, rows, if (logits_equal) "bit-equal" else "DIFFERENT", want_tokens.len, if (tokens_equal) "identical" else "DIFFERENT" });
        all_equal = all_equal and logits_equal and tokens_equal;
    }
    rows_override = null;
    try check.expect(all_equal, "deep prefill on (at {d} chunk sizes) and off give the same logits bytes and tokens", .{n_sizes});
    check.pass("deep prefill is the batched pass's bytes at {d} chunk sizes", .{n_sizes});
}
