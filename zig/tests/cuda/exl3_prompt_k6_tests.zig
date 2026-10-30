//! The common EXL3 prompt kernels at 3 and 4 bits (mcg) at GLM's shapes: runSlots, float64, row bytes and their time.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const qr = @import("quant_ref_tests.zig");
const Gpu = check.Gpu;
const ex = cuda.exl3_experts;
const fmt = core.exl3_format;
const h = qr.helpers;

pub const D = 4096; // model width
pub const N = 1024; // one rank's expert width (TP 2 of GLM-5.3-Flash's 2048)
pub const E = 288; // routed experts
pub const K = 8; // picks a row
pub const limit: f32 = 10.0; // GLM-5.3-Flash's swiglu_limit
const chunk = 1024; // rows a call in GLM's dispatch (cuda_exl3.chunk_rows)
pub const tiles = (D / 16) * (N / 16); // 16x16 tiles of one matrix
/// int16 words of one matrix of k2 half-bits a value: 16 * bits a tile.
pub fn words(k2: u32) usize {
    return tiles * 8 * @as(usize, k2);
}

pub const Ctx = struct {
    gpu: Gpu,
    a: std.mem.Allocator, // an arena
    s: cuda.Stream,
    exp: *const ex.Experts,
    router: cuda.grouped.Router,
    worst: f64 = 0, // the float64 checks' largest difference as a fraction of its tolerance
};

/// Device buffers freed together.
pub const Bufs = struct {
    d: *const cuda.Driver,
    gpa: std.mem.Allocator,
    list: std.ArrayList(cuda.DeviceBuffer) = .empty,
    pub fn alloc(b: *Bufs, n: usize) !u64 {
        const buf = try cuda.DeviceBuffer.alloc(b.d, @max(n, 16));
        try b.list.append(b.gpa, buf);
        try buf.fill8(0xff, null); // a launch that writes nothing leaves NaN bytes
        return buf.ptr;
    }
    pub fn put(b: *Bufs, bytes: []const u8) !u64 {
        const buf = try cuda.DeviceBuffer.fromHost(b.d, bytes);
        try b.list.append(b.gpa, buf);
        return buf.ptr;
    }
    pub fn deinit(b: *Bufs) void {
        for (b.list.items) |*buf| buf.free();
        b.list.deinit(b.gpa);
    }
};

/// One layer's tensors: trellis pointer and width tables on the device, the same words and scales on the host.
pub const Set = struct {
    gp: u64, // int64 [E] trellis pointers
    up: u64,
    dp: u64,
    k2: [2]u64, // int32 [E] of 6, and of 8
    suh_g: u64, // fp16 [E, D]
    suh_u: u64,
    svh_g: u64, // fp16 [E, N]
    svh_u: u64,
    suh_d: u64, // fp16 [E, N]
    svh_d: u64, // fp16 [E, D]
    shared: bool, // one input scale vector for every gate and up
    hg: []const []const i16, // each expert's words (at least those of the widths run)
    hu: []const []const i16,
    hd: []const []const i16,
    sg: []const f16,
    su: []const f16,
    vg: []const f16,
    vu: []const f16,
    sd: []const f16,
    vd: []const f16,

    pub fn layer(s: *const Set, gu: u32, d: u32, shared: bool) ex.Layer {
        return .{ .gate = s.gp, .up = s.up, .down = s.dp, .gate_k2 = s.k2[(gu - 6) / 2], .up_k2 = s.k2[(gu - 6) / 2], .down_k2 = s.k2[(d - 6) / 2], .suh_g = s.suh_g, .suh_u = s.suh_u, .svh_g = s.svh_g, .svh_u = s.svh_u, .suh_d = s.suh_d, .svh_d = s.svh_d, .experts = E, .dims = D, .width = N, .codebook = .mcg, .uniform_k2_gu = gu, .uniform_k2_d = d, .shared_input = shared, .limit = limit, .act_mode = 0 };
    }
};

/// The int32 tables of 6 and 8 for every expert.
pub fn widthTables(b: *Bufs) ![2]u64 {
    var out: [2]u64 = undefined;
    for ([_]i32{ 6, 8 }, 0..) |v, i| {
        const t: [E]i32 = @splat(v);
        out[i] = try b.put(std.mem.sliceAsBytes(&t));
    }
    return out;
}

fn scales(c: *Ctx, rng: std.Random, n: usize, scale: f32) ![]f16 {
    const out = try c.a.alloc(f16, n);
    for (out) |*v| v.* = @floatCast((if (rng.boolean()) scale else -scale) * (0.5 + rng.float(f32)));
    return out;
}

/// Random words in 4-bit slots shared by both widths; `one` shares one input scale, `own` has each expert's.
fn synthetic(c: *Ctx, b: *Bufs, rng: std.Random) ![2]Set {
    const slot = words(8);
    var ptrs: [3]u64 = undefined;
    var host: [3][]const []const i16 = undefined;
    for (0..3) |m| {
        const all = try c.a.alloc(i16, E * slot);
        rng.bytes(std.mem.sliceAsBytes(all));
        const base = try b.put(std.mem.sliceAsBytes(all));
        const table = try c.a.alloc(u64, E);
        const each = try c.a.alloc([]const i16, E);
        for (0..E) |e| {
            table[e] = base + e * slot * 2;
            each[e] = all[e * slot ..][0..slot];
        }
        ptrs[m] = try b.put(std.mem.sliceAsBytes(table));
        host[m] = each;
    }
    const k2 = try widthTables(b);
    const one_suh = try scales(c, rng, D, 1.0);
    const sg1 = try c.a.alloc(f16, E * D);
    for (0..E) |e| @memcpy(sg1[e * D ..][0..D], one_suh);
    const sg = try scales(c, rng, E * D, 1.0);
    const su = try scales(c, rng, E * D, 1.0);
    const vg = try scales(c, rng, E * N, 0.02);
    const vu = try scales(c, rng, E * N, 0.02);
    const sd = try scales(c, rng, E * N, 1.0);
    const vd = try scales(c, rng, E * D, 0.02);
    var one: Set = .{ .gp = ptrs[0], .up = ptrs[1], .dp = ptrs[2], .k2 = k2, .suh_g = try b.put(std.mem.sliceAsBytes(sg1)), .suh_u = 0, .svh_g = try b.put(std.mem.sliceAsBytes(vg)), .svh_u = try b.put(std.mem.sliceAsBytes(vu)), .suh_d = try b.put(std.mem.sliceAsBytes(sd)), .svh_d = try b.put(std.mem.sliceAsBytes(vd)), .shared = true, .hg = host[0], .hu = host[1], .hd = host[2], .sg = sg1, .su = sg1, .vg = vg, .vu = vu, .sd = sd, .vd = vd };
    one.suh_u = one.suh_g;
    var own = one;
    own.shared = false;
    own.suh_g = try b.put(std.mem.sliceAsBytes(sg));
    own.suh_u = try b.put(std.mem.sliceAsBytes(su));
    own.sg = sg;
    own.su = su;
    return .{ one, own };
}

/// bf16 rows in [-1, 1) and each row's distinct picks; expert 0 in the first row's and expert 1 in the last row's.
pub fn inputs(c: *Ctx, rng: std.Random, rows: usize) !struct { x: []u16, pk: []i32 } {
    const x = try c.a.alloc(u16, rows * D);
    for (x) |*v| v.* = h.bf16Of((rng.float(f32) * 2 - 1));
    const pk = try c.a.alloc(i32, rows * K);
    for (0..rows) |row| {
        var seen: [E]bool = @splat(false);
        for (0..K) |j| {
            var e = rng.uintLessThan(usize, E);
            while (seen[e]) e = rng.uintLessThan(usize, E);
            seen[e] = true;
            pk[row * K + j] = @intCast(e);
        }
    }
    for ([_][2]usize{ .{ 0, 0 }, .{ rows - 1, 1 } }) |want| {
        const r = pk[want[0] * K ..][0..K];
        if (std.mem.indexOfScalar(i32, r, @intCast(want[1])) == null) r[K - 1] = @intCast(want[1]);
    }
    return .{ .x = x, .pk = pk };
}

/// The plan's scratch for `pairs` pairs in items of the prompt tile (as glm_exl3_tests sizes it).
fn plan(b: *Bufs, pairs: usize) !cuda.grouped.Plan {
    const wide = pairs > cuda.grouped.small;
    return .{ .members = try b.alloc(pairs * 4), .items = try b.alloc(cuda.grouped.maxItems(pairs, E, ex.prompt.tile) * 12), .counts = try b.alloc(8), .rank = try b.alloc((if (wide) pairs else 1) * 4), .hist = try b.alloc((if (wide) (pairs + 1023) / 1024 * E else 1) * 4) };
}

fn download(c: *Ctx, ptr: u64, out: []f32) !void {
    try c.s.synchronize();
    const buf: cuda.DeviceBuffer = .{ .d = c.gpu.d, .ptr = ptr, .len = out.len * 4 };
    try buf.download(0, std.mem.sliceAsBytes(out));
}

/// The prompt path over `rows` rows, `per` rows a call; go() downloads y [rows * K, D] when given somewhere to put it.
pub const Prompt = struct {
    c: *Ctx,
    l: ex.Layer,
    x: u64,
    pk: u64,
    rows: usize,
    per: usize,
    b: Bufs,
    sc: ex.prompt.Scratch,
    p: cuda.grouped.Plan,
    pub fn init(c: *Ctx, l: ex.Layer, x: u64, pk: u64, rows: usize, per: usize) !Prompt {
        var b: Bufs = .{ .d = c.gpu.d, .gpa = c.gpu.gpa };
        errdefer b.deinit();
        const cap: usize = @min(rows, per);
        const sc = try ex.prompt.scratch(try b.alloc(try ex.prompt.bytes(l, cap, K)), l, cap, K);
        const p = try plan(&b, cap * K);
        return .{ .c = c, .l = l, .x = x, .pk = pk, .rows = rows, .per = per, .b = b, .sc = sc, .p = p };
    }
    pub fn deinit(p: *Prompt) void {
        p.b.deinit();
    }
    pub fn go(p: *const Prompt, out: ?[]f32) !void {
        var at: usize = 0;
        while (at < p.rows) : (at += p.per) {
            const n: usize = @min(p.per, p.rows - at);
            try p.c.exp.prompt.run(p.c.exp, p.c.router, p.c.s, p.l, p.x + at * D * 2, D, p.pk + at * K * 4, n, p.sc, p.p);
            if (out) |o| try download(p.c, p.sc.y, o[at * K * D ..][0 .. n * K * D]);
        }
    }
    /// The gate/up (which 0) or down (1) launch of the last call again, as run() makes it.
    pub fn kernel(p: *const Prompt, which: u1) !void {
        const k = p.c.exp.prompt;
        const w = ex.prompt.widths(p.l);
        const n: usize = @min(p.per, p.rows);
        const items = cuda.grouped.maxItems(n * K, E, ex.prompt.tile);
        var a: cuda.Args = .{};
        if (which == 0) {
            inline for (.{ p.sc.xg, p.sc.xu, p.l.gate, p.l.up, p.p.items, p.p.counts, p.p.members, p.l.svh_g, p.l.svh_u, p.l.suh_d, p.sc.xd, @as(i32, D), @as(i32, N), @as(i32, if (p.l.shared_input) K else 1), p.l.limit, p.l.act_mode }) |v| a.add(v);
            try cuda.launch.launch(if (w.gu == 6) k.gateup_k6 else k.gateup, .{ .grid = .{ .x = N / 128, .y = @intCast(items), .z = 1 }, .block = .{ .x = 256 } }, p.c.s, &a);
        } else {
            inline for (.{ p.sc.xd, p.l.down, p.p.items, p.p.counts, p.p.members, p.l.svh_d, p.sc.y, @as(i32, N), @as(i32, D), @as(i32, K) }) |v| a.add(v);
            try cuda.launch.launch(if (w.d == 6) k.down_k6 else k.down, .{ .grid = .{ .x = D / 256, .y = @intCast(items), .z = 1 }, .block = .{ .x = 256 } }, p.c.s, &a);
        }
    }
};

/// The general grouped path (runSlots, any width) in GLM's 1024-row calls: the oracle, and the fallback's time.
pub const Slots = struct {
    c: *Ctx,
    l: ex.Layer,
    x: u64,
    pk: u64,
    rows: usize,
    b: Bufs,
    sc: ex.Scratch,
    pub fn init(c: *Ctx, l: ex.Layer, x: u64, pk: u64, rows: usize) !Slots {
        var b: Bufs = .{ .d = c.gpu.d, .gpa = c.gpu.gpa };
        errdefer b.deinit();
        const cap: usize = @min(rows, chunk);
        const sc = try ex.scratch(try b.alloc(try ex.scratchBytes(l, cap, K)), l, cap, K);
        return .{ .c = c, .l = l, .x = x, .pk = pk, .rows = rows, .b = b, .sc = sc };
    }
    pub fn deinit(s: *Slots) void {
        s.b.deinit();
    }
    pub fn go(s: *const Slots, out: ?[]f32) !void {
        var at: usize = 0;
        while (at < s.rows) : (at += chunk) {
            const n: usize = @min(chunk, s.rows - at);
            try s.c.exp.runSlots(s.c.s, s.l, s.x + at * D * 2, D, .bf16, s.pk + at * K * 4, n, s.sc);
            if (out) |o| try download(s.c, s.sc.y, o[at * K * D ..][0 .. n * K * D]);
        }
    }
};

/// Prompt against runSlots: the largest difference, absolute and over its slot row's RMS, and the equal values.
pub const Stats = struct { max_abs: f64 = 0, max_norm: f64 = 0, max_rel: f64 = 0, equal: usize = 0, n: usize = 0, dd: f64 = 0, ww: f64 = 0 };
pub fn compare(got: []const f32, want: []const f32) Stats {
    var st: Stats = .{ .n = got.len };
    var at: usize = 0;
    while (at < got.len) : (at += D) {
        var ss: f64 = 0;
        for (want[at..][0..D]) |v| ss += @as(f64, v) * v;
        st.ww += ss;
        const rms = @sqrt(ss / D) + 1e-30;
        for (got[at..][0..D], want[at..][0..D]) |g, w| {
            const d = @abs(@as(f64, g) - w);
            st.dd += d * d;
            if (@as(u32, @bitCast(g)) == @as(u32, @bitCast(w))) st.equal += 1;
            if (!(d <= st.max_abs)) st.max_abs = d; // a NaN sticks
            if (!(d / rms <= st.max_norm)) st.max_norm = d / rms;
            if (@abs(w) >= rms and !(d / @abs(w) <= st.max_rel)) st.max_rel = d / @abs(w);
        }
    }
    return st;
}
/// The 4-bit path's own distance from runSlots (fp32 sums reordered under bf16 roundings) is 2.7e-4 RMS, 2e-2 at most.
pub fn expectClose(name: []const u8, rows: usize, st: Stats) !void {
    std.debug.print("RESULT {s} ({d} rows) vs runSlots: max |diff| {e:.3}, max |diff|/rms {e:.3}, max rel {e:.3}, rms diff/rms {e:.3}, {d:.2}% equal\n", .{ name, rows, st.max_abs, st.max_norm, st.max_rel, @sqrt(st.dd / st.ww), 100.0 * @as(f64, @floatFromInt(st.equal)) / @as(f64, @floatFromInt(st.n)) });
    try check.expect(st.max_norm <= 1.0 / 16.0 and @sqrt(st.dd / st.ww) <= 1e-3, "{s} ({d} rows): {e} of a slot row's RMS from runSlots, {e} in all", .{ name, rows, st.max_norm, @sqrt(st.dd / st.ww) });
}

/// GLM's SwiGLU as the kernels round it (glm_exl3_tests' act).
fn act(g: f64, u: f64) f64 {
    const gg = @min(h.bfOf(h.bf16Of(@floatCast(g))), limit);
    const uu = @max(@min(h.bfOf(h.bf16Of(@floatCast(u))), limit), -limit);
    return h.bfOf(h.bf16Of(@floatCast(h.bfOf(h.bf16Of(@floatCast(gg / (1 + @exp(-gg))))) * uu)));
}

/// Float64 matrices of the first `F` experts, decoded once per projection and width.
pub const F = 2;
pub const Mats = struct {
    w: [F][3][2]?[]f64 = @splat(@splat(@splat(null))),
    fn get(m: *Mats, c: *Ctx, s: *const Set, e: usize, which: usize, k2: u32) ![]f64 {
        const slot = &m.w[e][which][(k2 - 6) / 2];
        if (slot.*) |w| return w;
        const out = try c.gpu.gpa.alloc(f64, D * N);
        const scratch = try c.gpu.gpa.alloc(f64, 2 * D * N);
        defer c.gpu.gpa.free(scratch);
        const width: fmt.Width = .{ .halves = @intCast(k2) };
        const tw = words(k2);
        switch (which) {
            0 => fmt.dequantize(width, .mcg, s.hg[e][0..tw], D / 16, N / 16, s.sg[e * D ..][0..D], s.vg[e * N ..][0..N], out, scratch),
            1 => fmt.dequantize(width, .mcg, s.hu[e][0..tw], D / 16, N / 16, s.su[e * D ..][0..D], s.vu[e * N ..][0..N], out, scratch),
            else => fmt.dequantize(width, .mcg, s.hd[e][0..tw], N / 16, D / 16, s.sd[e * N ..][0..N], s.vd[e * D ..][0..D], out, scratch),
        }
        slot.* = out;
        return out;
    }
    pub fn deinit(m: *Mats, gpa: std.mem.Allocator) void {
        for (&m.w) |*a| for (a) |*b| for (b) |*v| if (v.*) |w| gpa.free(w);
    }
};

/// `outs` on pairs of the first F experts (at most `most` each) against float64: glm_exl3_tests' |a w| / 64 + 1e-6.
pub fn reference(c: *Ctx, m: *Mats, s: *const Set, gu: u32, d: u32, x: []const u16, pk: []const i32, outs: []const []const f32, names: []const []const u8, most: usize) !void {
    const g = try c.a.alloc(f64, N);
    const u = try c.a.alloc(f64, N);
    const av = try c.a.alloc(f64, N);
    const want = try c.a.alloc(f64, D);
    const mag = try c.a.alloc(f64, D);
    var worst: [4]f64 = @splat(0);
    var pairs: usize = 0;
    for (0..F) |e| {
        var taken: usize = 0;
        for (pk, 0..) |pe, p| {
            if (pe != @as(i32, @intCast(e)) or taken == most) continue;
            taken += 1;
            const wg = try m.get(c, s, e, 0, gu);
            const wu = try m.get(c, s, e, 1, gu);
            const wd = try m.get(c, s, e, 2, d);
            const xr = x[(p / K) * D ..][0..D];
            @memset(g, 0);
            @memset(u, 0);
            for (0..D) |kk| {
                const xv = h.bfOf(xr[kk]);
                for (g, u, wg[kk * N ..][0..N], wu[kk * N ..][0..N]) |*gv, *uv, a, b| {
                    gv.* += xv * a;
                    uv.* += xv * b;
                }
            }
            for (av, g, u) |*o, gv, uv| o.* = act(gv, uv);
            @memset(want, 0);
            @memset(mag, 0);
            for (0..N) |kk| for (want, mag, wd[kk * D ..][0..D]) |*wv, *mv, wt| {
                wv.* += av[kk] * wt;
                mv.* += @abs(av[kk] * wt);
            };
            for (outs, 0..) |o, oi| for (0..D) |n| {
                const q = @abs(@as(f64, o[p * D + n]) - want[n]) / (mag[n] / 64.0 + 1e-6);
                if (!(q <= worst[oi])) worst[oi] = q;
            };
            pairs += 1;
        }
    }
    for (outs, 0..) |_, oi| {
        std.debug.print("RESULT {s}: {d} pairs of experts 0-{d} against float64, largest difference {d:.4} of its tolerance\n", .{ names[oi], pairs, F - 1, worst[oi] });
        c.worst = @max(c.worst, worst[oi]);
        try check.expect(worst[oi] <= 1 and pairs > 0, "{s}: {d} of its float64 tolerance over {d} pairs", .{ names[oi], worst[oi], pairs });
    }
}

/// Median milliseconds of `reps` runs of `f(ctx)` after two warm-ups.
pub fn timed(c: *Ctx, reps: usize, ctx: anytype, comptime f: anytype) !f64 {
    var t0 = try cuda.Event.init(c.gpu.d, true);
    defer t0.deinit();
    var t1 = try cuda.Event.init(c.gpu.d, true);
    defer t1.deinit();
    for (0..2) |_| try f(ctx);
    const ms = try c.a.alloc(f64, reps);
    for (ms) |*v| {
        try t0.record(c.s);
        try f(ctx);
        try t1.record(c.s);
        try t1.synchronize();
        v.* = try cuda.Event.elapsedMs(t0, t1);
    }
    return check.median(ms);
}
fn promptGo(p: *const Prompt) anyerror!void {
    return p.go(null);
}
fn slotsGo(s: *const Slots) anyerror!void {
    return s.go(null);
}
fn gateUp(p: *const Prompt) anyerror!void {
    return p.kernel(0);
}
fn downOnly(p: *const Prompt) anyerror!void {
    return p.kernel(1);
}

/// One layer's widths through the prompt path and runSlots at every row count, float64 on its sampled pairs.
pub fn widthsCase(c: *Ctx, m: *Mats, s: *const Set, gu: u32, d: u32, shared: bool, row_counts: []const usize, x: []const u16, pk: []const i32, dx: u64, dpk: u64) !void {
    const l = s.layer(gu, d, shared);
    var name_buf: [96]u8 = undefined;
    const name = try std.fmt.bufPrint(&name_buf, "exl3 prompt gate/up {d}-bit, down {d}-bit, {s}", .{ gu / 2, d / 2, if (shared) "one rotation" else "a rotation a pick" });
    for (row_counts) |rows| {
        const got = try c.gpu.gpa.alloc(f32, rows * K * D);
        defer c.gpu.gpa.free(got);
        const want = try c.gpu.gpa.alloc(f32, rows * K * D);
        defer c.gpu.gpa.free(want);
        {
            var p = try Prompt.init(c, l, dx, dpk, rows, rows);
            defer p.deinit();
            try p.go(got);
        }
        {
            var o = try Slots.init(c, l, dx, dpk, rows);
            defer o.deinit();
            try o.go(want);
        }
        try expectClose(name, rows, compare(got, want));
        var pn: [128]u8 = undefined;
        var on: [128]u8 = undefined;
        const names = [_][]const u8{ try std.fmt.bufPrint(&pn, "{s}, {d} rows, prompt", .{ name, rows }), try std.fmt.bufPrint(&on, "{s}, {d} rows, runSlots", .{ name, rows }) };
        try reference(c, m, s, gu, d, x[0 .. rows * D], pk[0 .. rows * K], &.{ got, want }, &names, 48);
    }
}

pub fn run(gpu: Gpu, dir: []const u8, base: ?[]const u8) !void {
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    var stream = try cuda.Stream.init(gpu.d, false);
    defer stream.deinit();
    var exp = try ex.Experts.load(gpu.d);
    defer exp.unload();
    var pm = try cuda.Module.load(gpu.d, cuda.kernels.experts);
    defer pm.unload();
    var c: Ctx = .{ .gpu = gpu, .a = arena.allocator(), .s = stream, .exp = &exp, .router = try cuda.grouped.Router.resolve(pm) };
    var prng = std.Random.DefaultPrng.init(0x3b17);
    const rng = prng.random();
    {
        var b: Bufs = .{ .d = gpu.d, .gpa = gpu.gpa };
        defer b.deinit();
        const sets = try synthetic(&c, &b, rng);
        var m: Mats = .{};
        defer m.deinit(gpu.gpa);
        const in = try inputs(&c, rng, 4096);
        const dx = try b.put(std.mem.sliceAsBytes(in.x));
        const dpk = try b.put(std.mem.sliceAsBytes(in.pk));
        const all = [_]usize{ 1, 64, 1000, 4096 };
        for ([_][2]u32{ .{ 6, 8 }, .{ 6, 6 }, .{ 8, 8 }, .{ 8, 6 } }) |w| try widthsCase(&c, &m, &sets[0], w[0], w[1], true, &all, in.x, in.pk, dx, dpk);
        var m2: Mats = .{};
        defer m2.deinit(gpu.gpa);
        for ([_][2]u32{ .{ 6, 8 }, .{ 6, 6 } }) |w| try widthsCase(&c, &m2, &sets[1], w[0], w[1], false, all[0..3], in.x, in.pk, dx, dpk);
        try invariance(&c, &sets[0], dx, dpk);
        if (base) |path| try baseline(&c, &sets[0], &sets[1], dx, dpk, path);
        try timing(&c, &sets[0], dx, dpk, &.{ .{ 6, 8 }, .{ 6, 6 }, .{ 8, 8 } }, true);
    }
    if (!std.mem.eql(u8, dir, "none")) try @import("exl3_prompt_k6_real.zig").run(&c, dir, rng);
    check.pass("exl3-prompt-k6: 3- and 4-bit prompt kernels within 1e-3 RMS (1/16 a slot row) of runSlots, {d:.4} of float64's tolerance", .{c.worst});
}

/// A row's bytes: one 4096-row call, GLM's 1024-row calls, and with a rotation a pick, all the same.
fn invariance(c: *Ctx, s: *const Set, dx: u64, dpk: u64) !void {
    const rows = 4096;
    const one = try c.gpu.gpa.alloc(f32, rows * K * D);
    defer c.gpu.gpa.free(one);
    const other = try c.gpu.gpa.alloc(f32, rows * K * D);
    defer c.gpu.gpa.free(other);
    for ([_][2]u32{ .{ 6, 8 }, .{ 6, 6 } }) |w| {
        {
            var p = try Prompt.init(c, s.layer(w[0], w[1], true), dx, dpk, rows, rows);
            defer p.deinit();
            try p.go(one);
        }
        {
            var p = try Prompt.init(c, s.layer(w[0], w[1], true), dx, dpk, rows, chunk);
            defer p.deinit();
            try p.go(other);
        }
        try check.sameBytes("exl3 prompt k6: 4096 rows in one call and in 1024-row calls", std.mem.sliceAsBytes(other), std.mem.sliceAsBytes(one));
        {
            var p = try Prompt.init(c, s.layer(w[0], w[1], false), dx, dpk, 1000, 1000);
            defer p.deinit();
            try p.go(other[0 .. 1000 * K * D]);
        }
        try check.sameBytes("exl3 prompt k6: one rotation a row and a rotation a pick", std.mem.sliceAsBytes(other[0 .. 1000 * K * D]), std.mem.sliceAsBytes(one[0 .. 1000 * K * D]));
        check.pass("exl3 prompt gate/up {d}-bit, down {d}-bit: a row's bytes in one call, in 1024-row calls and with a rotation a pick", .{ w[0] / 2, w[1] / 2 });
    }
}

/// An exl3_experts image built before a change (BASE): its 4-bit prompt kernels give this build's bytes.
fn baseline(c: *Ctx, one: *const Set, own: *const Set, dx: u64, dpk: u64, path: []const u8) !void {
    const image = try std.Io.Dir.cwd().readFileAlloc(c.gpu.io, path, c.gpu.gpa, .limited(1 << 28));
    defer c.gpu.gpa.free(image);
    var mod = try cuda.Module.load(c.gpu.d, image);
    defer mod.unload();
    var old = c.exp.*;
    old.rot_in = try mod.function("tf_exl3_experts_rot_in");
    old.prompt.rotate = try mod.function("tf_exl3_prompt_rotate");
    old.prompt.gateup = try mod.function("tf_exl3_prompt_gu");
    old.prompt.down = try mod.function("tf_exl3_prompt_down");
    var oc = c.*;
    oc.exp = &old;
    const got = try c.gpu.gpa.alloc(f32, 4096 * K * D);
    defer c.gpu.gpa.free(got);
    const want = try c.gpu.gpa.alloc(f32, 4096 * K * D);
    defer c.gpu.gpa.free(want);
    for ([_]struct { *const Set, bool, usize }{ .{ one, true, 4096 }, .{ own, false, 1000 }, .{ one, true, 1 } }) |case| {
        const l = case[0].layer(8, 8, case[1]);
        const n = case[2] * K * D;
        for ([_]*Ctx{ c, &oc }, [_][]f32{ got[0..n], want[0..n] }) |ctx, out| {
            var p = try Prompt.init(ctx, l, dx, dpk, case[2], case[2]);
            defer p.deinit();
            try p.go(out);
        }
        try check.sameBytes("exl3 prompt 4-bit: this build against the BASE image", std.mem.sliceAsBytes(got[0..n]), std.mem.sliceAsBytes(want[0..n]));
        check.pass("exl3 prompt gate/up 4-bit, down 4-bit, {s}, {d} rows: the BASE image's bytes", .{ if (case[1]) "one rotation" else "a rotation a pick", case[2] });
    }
}

/// One layer at 4096 rows by widths: prompt (one call, 1024-row calls), runSlots; rounds interleave (a shared device).
pub fn timing(c: *Ctx, s: *const Set, dx: u64, dpk: u64, cases: []const [2]u32, shared: bool) !void {
    const rows = 4096;
    const reps = 5;
    const rounds = 5;
    const ms = try c.a.alloc([5][rounds]f64, cases.len);
    for (0..rounds) |round| for (cases, ms) |w, *m| {
        const l = s.layer(w[0], w[1], shared);
        {
            var p = try Prompt.init(c, l, dx, dpk, rows, rows);
            defer p.deinit();
            m[0][round] = try timed(c, reps, &p, promptGo);
            m[1][round] = try timed(c, reps, &p, gateUp);
            m[2][round] = try timed(c, reps, &p, downOnly);
        }
        {
            var q = try Prompt.init(c, l, dx, dpk, rows, chunk);
            defer q.deinit();
            m[3][round] = try timed(c, reps, &q, promptGo);
        }
        var o = try Slots.init(c, l, dx, dpk, rows);
        defer o.deinit();
        m[4][round] = try timed(c, reps, &o, slotsGo);
    };
    std.debug.print("TIME one layer, {d} rows, top-{d} of {d} experts, D {d}, N {d}: ms, best / median of {d} rounds\n", .{ rows, K, E, D, N, rounds });
    std.debug.print("TIME widths (gate/up, down) | prompt, one call | its gate/up | its down | prompt, 1024-row calls | runSlots, 1024-row calls\n", .{});
    for (cases, ms) |w, *m| {
        var best: [5]f64 = undefined;
        var mid: [5]f64 = undefined;
        for (m, 0..) |*col, j| {
            mid[j] = check.median(col);
            best[j] = col[0];
        }
        std.debug.print("TIME {d}-bit, {d}-bit | {d:.3} / {d:.3} | {d:.3} / {d:.3} | {d:.3} / {d:.3} | {d:.3} / {d:.3} | {d:.3} / {d:.3}\n", .{ w[0] / 2, w[1] / 2, best[0], mid[0], best[1], mid[1], best[2], mid[2], best[3], mid[3], best[4], mid[4] });
    }
}
