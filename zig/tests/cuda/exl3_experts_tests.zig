//! EXL3 routed experts on CUDA: FP64 per pair, SwiGLU of the same rows, pairs alone vs grouped, decode vs prompt plans.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const t3 = @import("exl3_tests.zig");
const Gpu = check.Gpu;
const E = cuda.exl3_linear;
const X = cuda.exl3_experts;
const grouped = cuda.grouped;
const ex_ = cuda.experts;
const st = core.safetensors;

/// One layer's experts on the host: gate and up take D inputs to NI, down NI to D.
const Experts = struct {
    gate: []t3.Host,
    up: []t3.Host,
    down: []t3.Host,
    dims: usize,
    width: usize,
};

/// The layer on the device as exl3_experts.Stacks lays it out, and gate or up alone as one-matrix layers.
const Dev = struct {
    layer: ex_.Layer,
    gate: ex_.Layer,
    up: ex_.Layer,
};

fn stack(a: std.mem.Allocator, rig: *t3.Rig, hs: []const []const t3.Host) !struct { trellis: u64, halves: u64 } {
    var t: std.ArrayList(i16) = .empty;
    defer t.deinit(a);
    var h: std.ArrayList(f16) = .empty;
    defer h.deinit(a);
    for (0..hs[0].len) |e| for (hs) |set| {
        try t.appendSlice(a, set[e].trellis);
        try h.appendSlice(a, set[e].suh);
        try h.appendSlice(a, set[e].svh);
    };
    return .{ .trellis = try rig.put(t.items), .halves = try rig.put(h.items) };
}

fn upload(a: std.mem.Allocator, rig: *t3.Rig, x: Experts, limit: f32) !Dev {
    const up = try stack(a, rig, &.{ x.gate, x.up });
    const down = try stack(a, rig, &.{x.down});
    const g = try stack(a, rig, &.{x.gate});
    const u = try stack(a, rig, &.{x.up});
    const n = x.gate.len;
    const w: ex_.Layer = .{ .format = .exl3, .up = up.trellis, .down = down.trellis, .up_scale = up.halves, .down_scale = down.halves, .width = x.width, .dims = x.dims, .experts = n, .limit = limit, .exl3 = .{ .up = x.gate[0].bits, .down = x.down[0].bits, .codebook = x.gate[0].book } };
    // gate or up alone as a down layer: width is its inputs (D), dims its outputs (NI)
    const one: ex_.Layer = .{ .format = .exl3, .up = 0, .down = g.trellis, .down_scale = g.halves, .width = x.dims, .dims = x.width, .experts = n, .exl3 = .{ .down = x.gate[0].bits, .codebook = x.gate[0].book } };
    var other = one;
    other.down = u.trellis;
    other.down_scale = u.halves;
    return .{ .layer = w, .gate = one, .up = other };
}

/// The plan's scratch for `pairs` pairs at decode tile 16 (the widest plan), routed at `tile`.
const PlanBufs = struct {
    rig: t3.Rig,
    plan: grouped.Plan,
    items: usize,

    fn init(gpu: Gpu, r: grouped.Router, s: cuda.Stream, picks: []const i32, experts: usize, tile: usize) !PlanBufs {
        const pairs = picks.len;
        var p: PlanBufs = .{ .rig = .{ .gpu = gpu }, .plan = undefined, .items = grouped.maxItems(pairs, experts, tile) };
        const wide = pairs > grouped.small;
        const n_max = grouped.maxItems(pairs, experts, 16);
        p.plan = .{ .members = try p.rig.zeros(pairs * 4), .items = try p.rig.zeros(n_max * 12), .counts = try p.rig.zeros(8), .rank = try p.rig.zeros((if (wide) pairs else 1) * 4), .hist = try p.rig.zeros((if (wide) (pairs + 1023) / 1024 * experts else 1) * 4) };
        try r.route(s, try p.rig.put(picks), pairs, experts, tile, p.plan);
        return p;
    }
    fn deinit(p: *PlanBufs) void {
        p.rig.deinit();
    }
};

const Ctx = struct {
    gpu: Gpu,
    g: ex_.Grouped,
    router: grouped.Router,
    s: cuda.Stream,
};

/// up then down on one plan: act (pairs, NI) bf16 and y (pairs, D) fp32, through Grouped.decode or .prompt.
fn layerRun(c: Ctx, d: Dev, x: u64, slots: usize, picks: []const i32, prompt: bool, act: u64, y: u64, skip: i32) !void {
    const tile: usize = if (prompt) ex_.promptTile(.exl3) else 16;
    var p = try PlanBufs.init(c.gpu, c.router, c.s, picks, d.layer.experts, tile);
    defer p.deinit();
    const call = if (prompt) &ex_.Grouped.prompt else &ex_.Grouped.decode;
    try call(c.g, c.s, .up, .{ .x = x, .stride = d.layer.dims, .slots = slots }, d.layer, p.plan, act, p.items, skip);
    try call(c.g, c.s, .down_f32, .{ .x = act, .stride = d.layer.width }, d.layer, p.plan, y, p.items, skip);
    try c.s.synchronize();
}

/// One matrix alone over pair rows (each pair's own input row): fp32 (pairs, n), on either path.
fn oneRun(c: Ctx, l: ex_.Layer, xp: u64, picks: []const i32, y: u64, prompt: bool) !void {
    var p = try PlanBufs.init(c.gpu, c.router, c.s, picks, l.experts, if (prompt) ex_.promptTile(.exl3) else 16);
    defer p.deinit();
    const call = if (prompt) &ex_.Grouped.prompt else &ex_.Grouped.decode;
    try call(c.g, c.s, .down_f32, .{ .x = xp, .stride = l.width }, l, p.plan, y, p.items, -1);
    try c.s.synchronize();
}

/// Round to bf16, nearest even (the device's __float2bfloat16_rn on finite values).
fn bfRound(v: f32) u16 {
    const b: u32 = @bitCast(v);
    return @intCast((b +% 0x7FFF +% ((b >> 16) & 1)) >> 16);
}

fn bf(v: f32) f32 {
    return @bitCast(@as(u32, bfRound(v)) << 16);
}

fn swiglu(g: f32, u: f32, limit: f32) f32 {
    var gv = bf(g);
    var uv = bf(u);
    if (limit > 0) {
        gv = @min(gv, limit);
        uv = @min(@max(uv, -limit), limit);
    }
    return bf(gv / (1 + @exp(-gv))) * uv;
}

/// One layer on both paths: FP64 per pair, SwiGLU of the path's own gate and up, tokens alone, a skipped expert.
fn checkLayer(c: Ctx, what: []const u8, x: Experts, d: Dev, xs: []const u16, slots: usize, picks: []const i32, b: *t3.Bound) !void {
    const a = c.gpu.gpa;
    const pairs = picks.len;
    var rig: t3.Rig = .{ .gpu = c.gpu };
    defer rig.deinit();
    const xd = try rig.put(xs);
    // each pair's own input row, for gate or up alone and for the FP64 references
    const xp = try a.alloc(u16, pairs * x.dims);
    defer a.free(xp);
    for (0..pairs) |p| @memcpy(xp[p * x.dims ..][0..x.dims], xs[p / slots * x.dims ..][0..x.dims]);
    const xpd = try rig.put(xp);
    const act_h = try a.alloc(u16, pairs * x.width);
    defer a.free(act_h);
    const y_h = try a.alloc(f32, pairs * x.dims);
    defer a.free(y_h);
    const g_h = try a.alloc(f32, pairs * x.width);
    defer a.free(g_h);
    const u_h = try a.alloc(f32, pairs * x.width);
    defer a.free(u_h);
    const ref = try a.alloc(f64, @max(x.width, x.dims));
    defer a.free(ref);
    const bound = try a.alloc(f64, @max(x.width, x.dims));
    defer a.free(bound);
    var exact: [2]usize = .{ 0, 0 };
    var ulp: [2]usize = .{ 0, 0 };
    for ([_]bool{ false, true }) |prompt| {
        const pi: usize = @intFromBool(prompt);
        const act = try rig.zeros(pairs * x.width * 2);
        const y = try rig.zeros(pairs * x.dims * 4);
        try layerRun(c, d, xd, slots, picks, prompt, act, y, -1);
        try rig.get(act, std.mem.sliceAsBytes(act_h));
        try rig.get(y, std.mem.sliceAsBytes(y_h));
        const gy = try rig.zeros(pairs * x.width * 4);
        const uy = try rig.zeros(pairs * x.width * 4);
        try oneRun(c, d.gate, xpd, picks, gy, prompt);
        try oneRun(c, d.up, xpd, picks, uy, prompt);
        try rig.get(gy, std.mem.sliceAsBytes(g_h));
        try rig.get(uy, std.mem.sliceAsBytes(u_h));
        for (0..pairs) |p| {
            const e: usize = @intCast(picks[p]);
            for ([_][]const t3.Host{ x.gate, x.up }, [_][]const f32{ g_h, u_h }) |hs, got| {
                try t3.reference(a, hs[e], xp[p * x.dims ..][0..x.dims], 1, ref[0..x.width], bound[0..x.width], prompt);
                try t3.compare(what, got[p * x.width ..][0..x.width], ref[0..x.width], bound[0..x.width], 1, x.width, b);
            }
            // down's fp32 rows: the prompt path's bound covers its fp16 inputs and tensor-unit sums
            try t3.reference(a, x.down[e], act_h[p * x.width ..][0..x.width], 1, ref[0..x.dims], bound[0..x.dims], prompt);
            try t3.compare(what, y_h[p * x.dims ..][0..x.dims], ref[0..x.dims], bound[0..x.dims], 1, x.dims, b);
            for (0..x.width) |j| {
                const want = bfRound(swiglu(g_h[p * x.width + j], u_h[p * x.width + j], d.layer.limit));
                const got = act_h[p * x.width + j];
                const diff = @abs(@as(i32, got) - @as(i32, want));
                try check.expect(diff <= 1, "{s}: pair {d} column {d} SwiGLU {x} against {x}", .{ what, p, j, got, want });
                if (diff == 0) exact[pi] += 1 else ulp[pi] += 1;
            }
        }
        for (0..pairs / slots) |t| {
            const one_act = try rig.zeros(slots * x.width * 2);
            const one_y = try rig.zeros(slots * x.dims * 4);
            try layerRun(c, d, xd + t * x.dims * 2, slots, picks[t * slots ..][0..slots], prompt, one_act, one_y, -1);
            try sameRows(rig, one_act, act_h[t * slots * x.width ..][0 .. slots * x.width], one_y, y_h[t * slots * x.dims ..][0 .. slots * x.dims]);
        }
        if (prompt) continue;
        // a skipped expert leaves its pairs' rows as they were (zero here), every other row unchanged
        const act3 = try rig.zeros(pairs * x.width * 2);
        const y3 = try rig.zeros(pairs * x.dims * 4);
        try layerRun(c, d, xd, slots, picks, false, act3, y3, picks[0]);
        const skipped = try a.alloc(u16, pairs * x.width);
        defer a.free(skipped);
        try rig.get(act3, std.mem.sliceAsBytes(skipped));
        for (0..pairs) |p| {
            const row = skipped[p * x.width ..][0..x.width];
            if (picks[p] == picks[0]) {
                for (row) |v| try check.expect(v == 0, "{s}: skipped expert's pair {d} written", .{ what, p });
            } else try check.sameBytes("exl3 experts skip", std.mem.sliceAsBytes(row), std.mem.sliceAsBytes(act_h[p * x.width ..][0..x.width]));
        }
    }
    check.pass("{s}: {d} pairs, decode and prompt paths: gate, up and down within FP64 bounds; SwiGLU of each path's own gate and up {d}/{d} exact, {d}/{d} one bf16 step (expf); tokens alone byte-equal, a skipped expert untouched", .{ what, pairs, exact[0], exact[1], ulp[0], ulp[1] });
}

fn sameRows(rig: t3.Rig, act: u64, act_want: []const u16, y: u64, y_want: []const f32) !void {
    var r = rig;
    const a = r.gpu.gpa;
    const got_a = try a.alloc(u16, act_want.len);
    defer a.free(got_a);
    try r.get(act, std.mem.sliceAsBytes(got_a));
    try check.sameBytes("exl3 experts act", std.mem.sliceAsBytes(got_a), std.mem.sliceAsBytes(act_want));
    const got_y = try a.alloc(f32, y_want.len);
    defer a.free(got_y);
    try r.get(y, std.mem.sliceAsBytes(got_y));
    try check.sameBytes("exl3 experts down", std.mem.sliceAsBytes(got_y), std.mem.sliceAsBytes(y_want));
}

fn freeExperts(a: std.mem.Allocator, x: Experts) void {
    for ([_][]t3.Host{ x.gate, x.up, x.down }) |set| {
        for (set) |h| t3.freeHost(a, h);
        a.free(set);
    }
}

/// `tf-cuda-test exl3-experts [DIR]`: a synthetic layer at two widths and codebooks; DIR adds a real layer's experts.
pub fn run(gpu: Gpu, dir: ?[]const u8) !void {
    const a = gpu.gpa;
    const sms: usize = @intCast(try gpu.ctx.attribute(.multiprocessor_count));
    var ex = try X.Experts.load(gpu.d, sms);
    defer ex.unload();
    var mod = try cuda.Module.load(gpu.d, cuda.kernels.experts);
    defer mod.unload();
    var s = try cuda.Stream.init(gpu.d, false); // blocking: launches wait for fromHost's legacy-stream copies
    defer s.deinit();
    const c: Ctx = .{ .gpu = gpu, .g = .{ .exl3 = &ex }, .router = try grouped.Router.resolve(mod), .s = s };
    var b: t3.Bound = .{};
    for ([_]struct { n: usize, d: usize, ni: usize, up: u32, down: u32, book: E.Codebook, tokens: usize, slots: usize, limit: f32 }{
        .{ .n = 6, .d = 256, .ni = 384, .up = 4, .down = 5, .book = .mul1, .tokens = 7, .slots = 4, .limit = 10 },
        .{ .n = 9, .d = 384, .ni = 256, .up = 3, .down = 6, .book = .mcg, .tokens = 9, .slots = 3, .limit = 0 },
    }, 0..) |cs, i| {
        var prng = std.Random.DefaultPrng.init(0xE7 + i);
        const r = prng.random();
        var x: Experts = .{ .gate = try a.alloc(t3.Host, cs.n), .up = try a.alloc(t3.Host, cs.n), .down = try a.alloc(t3.Host, cs.n), .dims = cs.d, .width = cs.ni };
        defer freeExperts(a, x);
        for (0..cs.n) |e| {
            x.gate[e] = try t3.randomMatrix(a, r, .{ .k = cs.d, .n = cs.ni, .bits = cs.up, .book = cs.book });
            x.up[e] = try t3.randomMatrix(a, r, .{ .k = cs.d, .n = cs.ni, .bits = cs.up, .book = cs.book });
            x.down[e] = try t3.randomMatrix(a, r, .{ .k = cs.ni, .n = cs.d, .bits = cs.down, .book = cs.book });
        }
        // scales sized so SwiGLU sees values around its limit, not only near zero
        for (x.gate) |*h| for (@constCast(h.svh)) |*v| {
            v.* *= 40;
        };
        const xs = try t3.randomRows(a, r, cs.tokens, cs.d);
        defer a.free(xs);
        const picks = try a.alloc(i32, cs.tokens * cs.slots);
        defer a.free(picks);
        for (picks) |*p| p.* = @intCast(r.uintLessThan(usize, cs.n));
        var rig: t3.Rig = .{ .gpu = gpu };
        defer rig.deinit();
        const d = try upload(a, &rig, x, cs.limit);
        var name: [64]u8 = undefined;
        try checkLayer(c, try std.fmt.bufPrint(&name, "synthetic {d} experts {d}/{d} bits {t}", .{ cs.n, cs.up, cs.down, cs.book }), x, d, xs, cs.slots, picks, &b);
    }
    if (dir) |path| try real(c, path, &b);
    check.pass("exl3 experts: {d} FP64 values, max error {e}, max {d:.4} of the bound, worst column {e} of the largest output", .{ b.values, b.max_error, b.max_fraction, b.worst_column });
}

/// A real layer's first 16 experts: the checks above on 2 tokens of 8 pairs, then a decode step's time.
fn real(c: Ctx, dir: []const u8, b: *t3.Bound) !void {
    const a = c.gpu.gpa;
    const count = 16;
    var x: Experts = .{ .gate = try a.alloc(t3.Host, count), .up = try a.alloc(t3.Host, count), .down = try a.alloc(t3.Host, count), .dims = 0, .width = 0 };
    defer freeExperts(a, x);
    var shards = try t3.Shards.open(c.gpu, dir);
    defer shards.close();
    for (0..count) |e| for ([_][]t3.Host{ x.gate, x.up, x.down }, [_][]const u8{ "gate_proj", "up_proj", "down_proj" }) |set, p| {
        var name: [96]u8 = undefined;
        set[e] = try shards.host(try std.fmt.bufPrint(&name, "model.language_model.layers.3.mlp.experts.{d}.{s}", .{ e, p }));
    };
    x.dims = x.gate[0].k;
    x.width = x.gate[0].n;
    var prng = std.Random.DefaultPrng.init(0x3A11);
    const r = prng.random();
    const tokens = 2;
    const slots = 8;
    const xs = try t3.randomRows(a, r, tokens, x.dims);
    defer a.free(xs);
    var picks: [tokens * slots]i32 = undefined;
    for (0..tokens) |t| for (0..slots) |k| {
        picks[t * slots + k] = @intCast((t * 5 + k * 2) % count);
    };
    var rig: t3.Rig = .{ .gpu = c.gpu };
    defer rig.deinit();
    const d = try upload(a, &rig, x, 10);
    try checkLayer(c, "layer 3 experts 0-15", x, d, xs, slots, &picks, b);
    // a decode step's experts: one token's 8 pairs, up and down, timed as a whole
    const xd = try rig.put(xs);
    const act = try rig.zeros(slots * x.width * 2);
    const y = try rig.zeros(slots * x.dims * 4);
    var p = try PlanBufs.init(c.gpu, c.router, c.s, picks[0..slots], count, 16);
    defer p.deinit();
    const both = [_]ex_.Epi{ .up, .down_bf16 };
    var secs: [3]f64 = undefined;
    for ([_][]const ex_.Epi{ both[0..1], both[1..2], &both }, &secs) |epis, *sec| {
        var runs: [7]f64 = undefined;
        for (0..8) |rep| {
            const t0 = check.now(c.gpu.io);
            for (0..20) |_| for (epis) |epi| {
                const in: ex_.Rows = if (epi == .up) .{ .x = xd, .stride = x.dims, .slots = slots } else .{ .x = act, .stride = x.width };
                try c.g.decode(c.s, epi, in, d.layer, p.plan, if (epi == .up) act else y, p.items, -1);
            };
            try c.s.synchronize();
            if (rep > 0) runs[rep - 1] = @as(f64, @floatFromInt(check.now(c.gpu.io) - t0)) / 20e9;
        }
        sec.* = check.median(&runs);
    }
    const up_bytes = slots * 2 * x.gate[0].trellis.len * 2;
    const down_bytes = slots * x.down[0].trellis.len * 2;
    check.pass("layer 3: one token's 8 experts (up {d} bits, down {d} bits): up {d:.1} us ({d:.0} GB/s), down {d:.1} us ({d:.0} GB/s), both {d:.1} us ({d:.0} GB/s)", .{ x.gate[0].bits, x.down[0].bits, secs[0] * 1e6, gbs(up_bytes, secs[0]), secs[1] * 1e6, gbs(down_bytes, secs[1]), secs[2] * 1e6, gbs(up_bytes + down_bytes, secs[2]) });
    // a 512-token prompt, 8 of the 16 experts a token: up and down on the tensor units, the plan routed once
    const ptok = 512;
    const pxs = try t3.randomRows(a, r, ptok, x.dims);
    defer a.free(pxs);
    const ppicks = try a.alloc(i32, ptok * slots);
    defer a.free(ppicks);
    for (0..ptok) |t_| for (0..slots) |s_| {
        ppicks[t_ * slots + s_] = @intCast((t_ * 3 + s_ * 2) % count);
    };
    const pxd = try rig.put(pxs);
    const pact = try rig.zeros(ptok * slots * x.width * 2);
    const py = try rig.zeros(ptok * slots * x.dims * 2);
    var pp = try PlanBufs.init(c.gpu, c.router, c.s, ppicks, count, ex_.promptTile(.exl3));
    defer pp.deinit();
    var pruns: [5]f64 = undefined;
    for (0..6) |rep| {
        const t0 = check.now(c.gpu.io);
        try c.g.prompt(c.s, .up, .{ .x = pxd, .stride = x.dims, .slots = slots }, d.layer, pp.plan, pact, pp.items, -1);
        try c.g.prompt(c.s, .down_bf16, .{ .x = pact, .stride = x.width }, d.layer, pp.plan, py, pp.items, -1);
        try c.s.synchronize();
        if (rep > 0) pruns[rep - 1] = @as(f64, @floatFromInt(check.now(c.gpu.io) - t0)) / 1e9;
    }
    const psec = check.median(&pruns);
    check.pass("layer 3: a {d}-token prompt, 8 experts a token over {d}: up and down {d:.0} us, {d:.0} tokens/s", .{ ptok, count, psec * 1e6, ptok / psec });
}

fn gbs(bytes: usize, sec: f64) f64 {
    return @as(f64, @floatFromInt(bytes)) / 1e9 / sec;
}
