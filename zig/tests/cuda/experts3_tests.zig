//! 3-bit affine experts (cuda/experts3.zig) against the affine-4 kernels on the same codes and an fp64 reference.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const Gpu = check.Gpu;

const A3 = cuda.experts3.Affine3Experts;
const A4 = cuda.experts.Affine4Experts;

fn bf16f(h: u16) f32 {
    return @bitCast(@as(u32, h) << 16);
}

fn f32bf16(x: f32) u16 {
    const u: u32 = @bitCast(x);
    return @intCast((u +% 0x7fff +% ((u >> 16) & 1)) >> 16);
}

fn randBf16(r: std.Random, scale: f32) u16 {
    return f32bf16((r.float(f32) * 2.0 - 1.0) * scale);
}

/// `count` experts of n by k as codes (one a byte, below 2^bits), scales and biases a group of 64.
pub const Codes = struct {
    q: []u8,
    scales: []u16,
    biases: []u16,
    count: usize,
    n: usize,
    k: usize,

    fn init(a: std.mem.Allocator, r: std.Random, count: usize, n: usize, k: usize, bits: u3) !Codes {
        const c: Codes = .{ .q = try a.alloc(u8, count * n * k), .scales = try a.alloc(u16, count * n * k / 64), .biases = try a.alloc(u16, count * n * k / 64), .count = count, .n = n, .k = k };
        for (c.q) |*v| v.* = r.uintLessThan(u8, @as(u8, 1) << bits);
        for (c.scales) |*v| v.* = f32bf16(0.02 + r.float(f32) * 0.03);
        for (c.biases) |*v| v.* = randBf16(r, 0.2);
        return c;
    }

    fn deinit(c: Codes, a: std.mem.Allocator) void {
        a.free(c.q);
        a.free(c.scales);
        a.free(c.biases);
    }

    /// MLX words at `bits`: each row's codes as an LSB-first bitstream of u32.
    fn words(c: Codes, a: std.mem.Allocator, bits: usize) ![]u32 {
        const w = try a.alloc(u32, c.count * c.n * c.k * bits / 32);
        @memset(w, 0);
        for (c.q, 0..) |v, i| {
            const at = i * bits;
            w[at / 32] |= @as(u32, v) << @intCast(at % 32);
            if (at % 32 + bits > 32) w[at / 32 + 1] |= @as(u32, v) >> @intCast(32 - at % 32);
        }
        return w;
    }

    fn weight(c: Codes, e: usize, j: usize, i: usize) f64 {
        const row = e * c.n + j;
        const g = row * (c.k / 64) + i / 64;
        return @as(f64, @floatFromInt(c.q[row * c.k + i])) * bf16f(c.scales[g]) + bf16f(c.biases[g]);
    }
};

fn upload(gpu: Gpu, bytes: []const u8) !cuda.DeviceBuffer {
    return cuda.DeviceBuffer.fromHost(gpu.d, bytes);
}

pub const Kernels = struct { a3: A3, a4: A4, router: cuda.grouped.Router };

/// Pack `routed` (3-bit codes) and `shared` (4-bit) both ways: one 4-bit table, or a 3-bit table and a 4-bit one.
const Tables = struct {
    all4: cuda.DeviceBuffer,
    t3: cuda.DeviceBuffer,
    shared4: cuda.DeviceBuffer,

    fn deinit(t: *Tables) void {
        t.all4.free();
        t.t3.free();
        t.shared4.free();
    }
};

pub fn packTable(gpu: Gpu, k: Kernels, s: cuda.Stream, c: Codes, bits: usize, out: u64, m: usize, mi: usize, at: usize) !void {
    const a = gpu.gpa;
    const w = try c.words(a, bits);
    defer a.free(w);
    var dw = try upload(gpu, std.mem.sliceAsBytes(w));
    defer dw.free();
    var ds = try upload(gpu, std.mem.sliceAsBytes(c.scales));
    defer ds.free();
    var db = try upload(gpu, std.mem.sliceAsBytes(c.biases));
    defer db.free();
    if (bits == 4) {
        try k.a4.packInto(s, dw.ptr, ds.ptr, db.ptr, out + at * A4.bytes(1, c.n, c.k) * m, c.count, c.n, c.k, m, mi);
    } else {
        const sb = A3.scratchBytes(c.count, c.n, c.k);
        var s0 = try cuda.DeviceBuffer.alloc(gpu.d, sb[0]);
        defer s0.free();
        var s1 = try cuda.DeviceBuffer.alloc(gpu.d, sb[1]);
        defer s1.free();
        try k.a3.packInto(&k.a4, s, dw.ptr, ds.ptr, db.ptr, .{ s0.ptr, s1.ptr }, out, c.count, c.n, c.k, m, mi);
    }
    try s.synchronize();
}

/// One projection class's tables: `routed` and `shared` side by side as m-of-`m` (gate 0 and up 1, or down alone).
fn tables(gpu: Gpu, k: Kernels, s: cuda.Stream, parts: []const [2]Codes) !Tables {
    const r = parts[0][0];
    const m = parts.len;
    var t: Tables = .{
        .all4 = try cuda.DeviceBuffer.alloc(gpu.d, m * A4.bytes(r.count + 1, r.n, r.k)),
        .t3 = try cuda.DeviceBuffer.alloc(gpu.d, m * A3.bytes(r.count, r.n, r.k)),
        .shared4 = try cuda.DeviceBuffer.alloc(gpu.d, m * A4.bytes(1, r.n, r.k)),
    };
    errdefer t.deinit();
    for (parts, 0..) |p, mi| {
        try packTable(gpu, k, s, p[0], 4, t.all4.ptr, m, mi, 0);
        try packTable(gpu, k, s, p[1], 4, t.all4.ptr, m, mi, r.count);
        try packTable(gpu, k, s, p[0], 3, t.t3.ptr, m, mi, 0);
        try packTable(gpu, k, s, p[1], 4, t.shared4.ptr, m, mi, 0);
    }
    return t;
}

const Shape = struct { experts: usize = 6, ni: usize = 128, dims: usize = 256, rows: usize = 40, slots: usize = 3 };
/// Qwen3.5-35B-A3B's expert widths at a verify window of 4 rows: 8 routed picks and the shared expert a row.
const wide: Shape = .{ .experts = 16, .ni = 512, .dims = 2048, .rows = 4, .slots = 9 };
/// 5 K groups: the split form's last round has fewer groups than warps.
const short: Shape = .{ .dims = 320 };

/// A plan for `picks` at `tile`, on buffers that live as long as the returned Plan's owner.
const PlanBufs = struct {
    picks: cuda.DeviceBuffer,
    members: cuda.DeviceBuffer,
    items: cuda.DeviceBuffer,
    counts: cuda.DeviceBuffer,
    items_n: usize,

    fn init(gpu: Gpu, k: Kernels, s: cuda.Stream, picks: []const i32, count: usize, tile: usize) !PlanBufs {
        const n = cuda.grouped.maxItems(picks.len, count, tile);
        var p: PlanBufs = .{ .picks = try upload(gpu, std.mem.sliceAsBytes(picks)), .members = try cuda.DeviceBuffer.alloc(gpu.d, picks.len * 4), .items = try cuda.DeviceBuffer.alloc(gpu.d, n * 12), .counts = try cuda.DeviceBuffer.alloc(gpu.d, 8), .items_n = n };
        try k.router.route(s, p.picks.ptr, picks.len, count, tile, p.plan());
        return p;
    }

    fn plan(p: PlanBufs) cuda.grouped.Plan {
        return .{ .members = p.members.ptr, .items = p.items.ptr, .counts = p.counts.ptr, .rank = 0, .hist = 0 };
    }

    fn deinit(p: *PlanBufs) void {
        p.picks.free();
        p.members.free();
        p.items.free();
        p.counts.free();
    }
};

/// The SwiGLU pair form and down, decode and prompt rows: the 3-bit tables give the 4-bit tables' bytes.
fn sameBits(gpu: Gpu, k: Kernels, s: cuda.Stream, r: std.Random, sh: Shape) !void {
    const a = gpu.gpa;
    const e = sh.experts;
    var cs: [3][2]Codes = undefined;
    for (&cs, [_][2]usize{ .{ sh.ni, sh.dims }, .{ sh.ni, sh.dims }, .{ sh.dims, sh.ni } }) |*c, nk| c.* = .{ try Codes.init(a, r, e, nk[0], nk[1], 3), try Codes.init(a, r, 1, nk[0], nk[1], 4) };
    defer for (cs) |c| for (c) |x| x.deinit(a);
    var pair = try tables(gpu, k, s, &.{ cs[0], cs[1] });
    defer pair.deinit();
    var down = try tables(gpu, k, s, &.{cs[2]});
    defer down.deinit();
    const pairs = sh.rows * sh.slots;
    const picks = try a.alloc(i32, pairs);
    defer a.free(picks);
    for (0..sh.rows) |row| for (0..sh.slots) |i| {
        // the shared expert in the last slot, as the engine places it; expert 0 in slot 1 of every fourth row
        const pick = if (i == sh.slots - 1) e else if (i == 1 and row % 4 == 0) 0 else r.uintLessThan(usize, e);
        picks[row * sh.slots + i] = @intCast(pick);
    };
    const x = try a.alloc(u16, sh.rows * sh.dims);
    defer a.free(x);
    for (x) |*v| v.* = randBf16(r, 1.0);
    var dx = try upload(gpu, std.mem.sliceAsBytes(x));
    defer dx.free();
    const l4: cuda.experts.Layer = .{ .format = .affine4, .up = pair.all4.ptr, .down = down.all4.ptr, .width = sh.ni, .dims = sh.dims, .experts = e + 1, .swiglu = true };
    const l3: cuda.experts.Layer = .{ .format = .affine3, .up = pair.t3.ptr, .down = down.t3.ptr, .width = sh.ni, .dims = sh.dims, .experts = e + 1, .swiglu = true, .shared = @intCast(e), .shared_up = pair.shared4.ptr, .shared_down = down.shared4.ptr };
    const g: cuda.experts.Grouped = .{ .affine4 = &k.a4, .affine3 = &k.a3 };
    for ([_]bool{ false, true }) |prompt| {
        var pb = try PlanBufs.init(gpu, k, s, picks, e + 1, if (prompt) 64 else 16);
        defer pb.deinit();
        var outs: [2][2][]u8 = undefined;
        for ([_]cuda.experts.Layer{ l4, l3 }, 0..) |l, li| {
            var act = try cuda.DeviceBuffer.alloc(gpu.d, pairs * sh.ni * 2);
            defer act.free();
            var y = try cuda.DeviceBuffer.alloc(gpu.d, pairs * sh.dims * 4);
            defer y.free();
            const rows_in: cuda.experts.Rows = .{ .x = dx.ptr, .stride = sh.dims, .slots = sh.slots };
            const act_in: cuda.experts.Rows = .{ .x = act.ptr, .stride = sh.ni };
            if (prompt) {
                try g.prompt(s, .up, rows_in, l, pb.plan(), act.ptr, pb.items_n, -1);
                try g.prompt(s, .down_bf16, act_in, l, pb.plan(), y.ptr, pb.items_n, -1);
            } else {
                try g.decode(s, .up, rows_in, l, pb.plan(), act.ptr, pb.items_n, -1);
                try g.decode(s, .down_f32, act_in, l, pb.plan(), y.ptr, pb.items_n, -1);
            }
            try s.synchronize();
            outs[li] = .{ try check.download(gpu, act), try check.download(gpu, y) };
        }
        defer for (outs) |o| for (o) |b| a.free(b);
        const form = if (prompt) "prompt" else "decode";
        try check.sameBytes(if (prompt) "prompt SwiGLU rows" else "decode SwiGLU rows", outs[1][0], outs[0][0]);
        const used = pairs * sh.dims * @as(usize, if (prompt) 2 else 4);
        try check.sameBytes(if (prompt) "prompt down sums" else "decode down sums", outs[1][1][0..used], outs[0][1][0..used]);
        const pair_form = if (prompt) "" else if (k.a3.split_below == 0) ", the pair a column block a warp" else ", the pair a column block a CTA";
        check.pass("experts3 {s} ({d} by {d}{s}): {d} pairs over {d} 3-bit experts and a 4-bit shared one, SwiGLU rows and down sums byte-equal to affine-4 on the same codes", .{ form, sh.ni, sh.dims, pair_form, pairs, e });
    }
}

/// Each projection class alone (the down form, any n by k) against fp64: worst column error, 3-bit next to 4-bit.
fn projections(gpu: Gpu, k: Kernels, s: cuda.Stream, r: std.Random) !void {
    const a = gpu.gpa;
    const sh: Shape = .{};
    const e = sh.experts;
    const classes = [_]struct { []const u8, usize, usize }{ .{ "gate", sh.ni, sh.dims }, .{ "up", sh.ni, sh.dims }, .{ "down", sh.dims, sh.ni } };
    for (classes) |cl| {
        const n = cl[1];
        const kk = cl[2];
        const c: [2]Codes = .{ try Codes.init(a, r, e, n, kk, 3), try Codes.init(a, r, 1, n, kk, 4) };
        defer for (c) |x| x.deinit(a);
        var t = try tables(gpu, k, s, &.{c});
        defer t.deinit();
        const pairs = 48;
        var picks: [pairs]i32 = undefined;
        for (&picks, 0..) |*p, i| p.* = @intCast(if (i % 6 == 5) e else i % e);
        const x = try a.alloc(u16, pairs * kk);
        defer a.free(x);
        for (x) |*v| v.* = randBf16(r, 1.0);
        var dx = try upload(gpu, std.mem.sliceAsBytes(x));
        defer dx.free();
        const want = try a.alloc(f64, pairs * n);
        defer a.free(want);
        for (0..pairs) |p| for (0..n) |j| {
            const ex: usize = @intCast(picks[p]);
            var acc: f64 = 0;
            for (0..kk) |i| acc += (if (ex == e) c[1].weight(0, j, i) else c[0].weight(ex, j, i)) * bf16f(x[p * kk + i]);
            want[p * n + j] = acc;
        };
        const l4: cuda.experts.Layer = .{ .format = .affine4, .up = 0, .down = t.all4.ptr, .width = kk, .dims = n, .experts = e + 1, .swiglu = true };
        const l3: cuda.experts.Layer = .{ .format = .affine3, .up = 0, .down = t.t3.ptr, .width = kk, .dims = n, .experts = e + 1, .swiglu = true, .shared = @intCast(e), .shared_down = t.shared4.ptr };
        const g: cuda.experts.Grouped = .{ .affine4 = &k.a4, .affine3 = &k.a3 };
        for ([_]bool{ false, true }) |prompt| {
            var pb = try PlanBufs.init(gpu, k, s, &picks, e + 1, if (prompt) 64 else 16);
            defer pb.deinit();
            var worst: [2]f64 = .{ 0, 0 };
            var sum: [2]f64 = .{ 0, 0 };
            var outs: [2][]u8 = undefined;
            for ([_]cuda.experts.Layer{ l4, l3 }, 0..) |l, li| {
                var y = try cuda.DeviceBuffer.alloc(gpu.d, pairs * n * 4);
                defer y.free();
                const in: cuda.experts.Rows = .{ .x = dx.ptr, .stride = kk };
                if (prompt) try g.prompt(s, .down_bf16, in, l, pb.plan(), y.ptr, pb.items_n, -1) else try g.decode(s, .down_f32, in, l, pb.plan(), y.ptr, pb.items_n, -1);
                try s.synchronize();
                outs[li] = try check.download(gpu, y);
                for (0..pairs * n) |i| {
                    const got: f64 = if (prompt) bf16f(std.mem.bytesAsSlice(u16, outs[li])[i]) else std.mem.bytesAsSlice(f32, outs[li])[i];
                    const d = @abs(got - want[i]);
                    worst[li] = @max(worst[li], d);
                    sum[li] += d * d;
                }
            }
            defer for (outs) |o| a.free(o);
            const used = pairs * n * @as(usize, if (prompt) 2 else 4);
            try check.sameBytes("projection sums", outs[1][0..used], outs[0][0..used]);
            try check.expect(worst[1] <= worst[0], "{s}: 3-bit worst {e} over 4-bit {e}", .{ cl[0], worst[1], worst[0] });
            const rms = @sqrt(sum[1] / @as(f64, @floatFromInt(pairs * n)));
            check.pass("experts3 {s} {s} ({d} by {d}): worst column error {e} against fp64 (affine-4 on the same codes {e}), rms {e}", .{ cl[0], if (prompt) "prompt" else "decode", n, kk, worst[1], worst[0], rms });
        }
    }
}

/// Decode up (both pair forms) and down at the wide shape over 64 experts, fresh picks a call: us a call at 1-4 rows.
fn timeDecode(gpu: Gpu, k: Kernels, s: cuda.Stream, r: std.Random, reps: usize) !void {
    const a = gpu.gpa;
    const sh: Shape = .{ .experts = 64, .ni = wide.ni, .dims = wide.dims, .slots = wide.slots };
    const e = sh.experts;
    var cs: [3][2]Codes = undefined;
    for (&cs, [_][2]usize{ .{ sh.ni, sh.dims }, .{ sh.ni, sh.dims }, .{ sh.dims, sh.ni } }) |*c, nk| c.* = .{ try Codes.init(a, r, e, nk[0], nk[1], 3), try Codes.init(a, r, 1, nk[0], nk[1], 4) };
    defer for (cs) |c| for (c) |x| x.deinit(a);
    var pair = try tables(gpu, k, s, &.{ cs[0], cs[1] });
    defer pair.deinit();
    var down = try tables(gpu, k, s, &.{cs[2]});
    defer down.deinit();
    const l3: cuda.experts.Layer = .{ .format = .affine3, .up = pair.t3.ptr, .down = down.t3.ptr, .width = sh.ni, .dims = sh.dims, .experts = e + 1, .swiglu = true, .shared = @intCast(e), .shared_up = pair.shared4.ptr, .shared_down = down.shared4.ptr };
    var a3 = k.a3;
    const g: cuda.experts.Grouped = .{ .affine3 = &a3 };
    const x = try a.alloc(u16, 4 * sh.dims);
    defer a.free(x);
    for (x) |*v| v.* = randBf16(r, 1.0);
    var dx = try upload(gpu, std.mem.sliceAsBytes(x));
    defer dx.free();
    var act = try cuda.DeviceBuffer.alloc(gpu.d, 4 * sh.slots * sh.ni * 2);
    defer act.free();
    var y = try cuda.DeviceBuffer.alloc(gpu.d, 4 * sh.slots * sh.dims * 4);
    defer y.free();
    var t0 = try cuda.Event.init(gpu.d, true);
    defer t0.deinit();
    var t1 = try cuda.Event.init(gpu.d, true);
    defer t1.deinit();
    for ([_]usize{ 1, 2, 3, 4 }) |rows| {
        var plans: [8]PlanBufs = undefined;
        var made: usize = 0;
        defer for (plans[0..made]) |*p| p.deinit();
        for (&plans) |*p| {
            var picks: [4 * 9]i32 = undefined;
            for (0..rows) |row| {
                var taken: [64]bool = @splat(false);
                for (0..sh.slots - 1) |i| {
                    var pick = r.uintLessThan(usize, e);
                    while (taken[pick]) pick = r.uintLessThan(usize, e);
                    taken[pick] = true;
                    picks[row * sh.slots + i] = @intCast(pick);
                }
                picks[row * sh.slots + sh.slots - 1] = @intCast(e);
            }
            p.* = try PlanBufs.init(gpu, k, s, picks[0 .. rows * sh.slots], e + 1, 16);
            made += 1;
        }
        try s.synchronize();
        const rows_in: cuda.experts.Rows = .{ .x = dx.ptr, .stride = sh.dims, .slots = sh.slots };
        const act_in: cuda.experts.Rows = .{ .x = act.ptr, .stride = sh.ni };
        var us: [3]f64 = undefined;
        for (0..3) |form| {
            a3.split_below = if (form == 1) std.math.maxInt(usize) else 0;
            // the calls replayed as one captured graph, as the engine's forwards run them
            try cuda.graph.beginCapture(s, .thread_local);
            for (0..reps) |i| {
                const p = plans[i % plans.len];
                if (form < 2) {
                    try g.decode(s, .up, rows_in, l3, p.plan(), act.ptr, p.items_n, -1);
                } else try g.decode(s, .down_f32, act_in, l3, p.plan(), y.ptr, p.items_n, -1);
            }
            var graph = try cuda.graph.endCapture(s);
            defer graph.deinit();
            var exec = try graph.instantiate();
            defer exec.deinit();
            for (0..2) |round| {
                try t0.record(s);
                try exec.launchOn(s);
                try t1.record(s);
                try t1.synchronize();
                if (round == 1) us[form] = @as(f64, try cuda.Event.elapsedMs(t0, t1)) * 1000 / @as(f64, @floatFromInt(reps));
            }
        }
        const auto = if (plans[0].items_n * (sh.ni / 32) < k.a3.split_below) "a block a CTA" else "a block a warp";
        std.debug.print("TIME experts3 decode ({d} by {d}, {d} picks a row) at {d} rows: up a block a warp {d:.1} us, a block a CTA {d:.1} us (taken: {s}); down {d:.1} us\n", .{ sh.ni, sh.dims, sh.slots, rows, us[0], us[1], auto, us[2] });
    }
}

pub fn run(gpu: Gpu, reps: usize) !void {
    var prng = std.Random.DefaultPrng.init(0x3b17);
    const r = prng.random();
    var mods: [5]cuda.Module = undefined;
    const images = [_][]const u8{ cuda.kernels.experts, cuda.kernels.experts_prefill, cuda.kernels.experts_pack, cuda.kernels.experts3, cuda.kernels.experts3_prefill };
    for (images, &mods, 0..) |img, *m, i| {
        errdefer for (mods[0..i]) |*o| o.unload();
        m.* = try cuda.Module.load(gpu.d, img);
    }
    defer for (&mods) |*m| m.unload();
    const sms: usize = @intCast(try gpu.ctx.attribute(.multiprocessor_count));
    const k: Kernels = .{ .a3 = try A3.resolve(mods[3], mods[4], sms), .a4 = try A4.resolve(mods[0], mods[1], mods[2], sms), .router = try cuda.grouped.Router.resolve(mods[0]) };
    var s = try cuda.Stream.init(gpu.d, true);
    defer s.deinit();
    // both decode pair forms: a column block a warp, and a column block a CTA
    for ([_]usize{ 0, std.math.maxInt(usize) }) |below| {
        var kf = k;
        kf.a3.split_below = below;
        try sameBits(gpu, kf, s, r, .{});
        try sameBits(gpu, kf, s, r, wide);
        try sameBits(gpu, kf, s, r, short);
    }
    try projections(gpu, k, s, r);
    if (reps > 0) try timeDecode(gpu, k, s, r, reps);
}
