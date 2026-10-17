//! glm_kda's paths on Ling-3.1 layer 0 (kda-real's fla fixture, q/k/v convs as one q|k|v): vs fla and float64.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const f64r = @import("glm_kda_f64.zig");
const rig = @import("glm_kda_rig.zig");
const ref = @import("glm_ref_tests.zig");
const Rig = rig.Rig;
const kda = rig.kda;
const dk = rig.dk;

fn take(comptime T: type, a: std.mem.Allocator, bytes: []const u8, at: *usize, count: usize) ![]T {
    const out = try a.alloc(T, count);
    @memcpy(std.mem.sliceAsBytes(out), bytes[at.*..][0 .. count * @sizeOf(T)]);
    at.* += count * @sizeOf(T);
    return out;
}

/// The layer's rows as GLM's chain takes them, fla's outputs and states, and per-row conv windows on the device.
const Layer = struct {
    P: usize,
    T: usize,
    H: usize,
    l: f64r.Layout,
    in: f64r.Inputs,
    c: kda.Chain,
    cs_all: u64, // [T][3][3 H 128]: row t's window, rows t - 3 .. t - 1 (zeros before the first)
    fla_y: []f64,
    fla_sp: []f64,
    fla_st: []f64,

    /// The chain over rows lo .. from state_in with row lo's conv window.
    fn at(l: Layer, lo: usize, state_in: u64) kda.Chain {
        const n = l.H * dk;
        var c = l.c;
        c.p += lo * l.l.p_stride * 2;
        c.a += lo * n * 2;
        c.g += lo * n * 2;
        c.conv_state = l.cs_all + lo * 9 * n * 2;
        c.state_in = state_in;
        return c;
    }
};

fn load(r: *Rig, path: []const u8) !?Layer {
    const a = r.a;
    const bytes = std.Io.Dir.cwd().readFileAlloc(r.gpu.io, path, a, .limited(1 << 30)) catch |e| switch (e) {
        error.FileNotFound => return null,
        else => return e,
    };
    if (bytes.len < 32 or !std.mem.eql(u8, bytes[0..8], "TFKDA001")) return error.BadFixture;
    const P: usize = std.mem.readInt(u32, bytes[8..12], .little);
    const N: usize = std.mem.readInt(u32, bytes[12..16], .little);
    const H: usize = std.mem.readInt(u32, bytes[16..20], .little);
    if (std.mem.readInt(u32, bytes[20..24], .little) != dk) return error.BadFixture;
    const lower: f32 = @bitCast(std.mem.readInt(u32, bytes[24..28], .little));
    const eps: f32 = @bitCast(std.mem.readInt(u32, bytes[28..32], .little));
    const T = P + N;
    const n = H * dk;
    var o: usize = 32;
    const qkv = try take(u16, a, bytes, &o, T * 3 * n);
    const f = try take(u16, a, bytes, &o, T * n);
    const b = try take(u16, a, bytes, &o, T * H);
    const gate = try take(u16, a, bytes, &o, T * n);
    const conv = try take(f32, a, bytes, &o, 3 * n * 4); // Ling's q_conv1d | k_conv1d | v_conv1d: GLM's q|k|v channels
    const a_log = try take(f32, a, bytes, &o, H);
    const dt_bias = try take(f32, a, bytes, &o, n);
    const norm32 = try take(f32, a, bytes, &o, dk);
    const fy = try take(u16, a, bytes, &o, T * n);
    const fsp = try take(f32, a, bytes, &o, n * dk);
    const fst = try take(f32, a, bytes, &o, n * dk);
    if (o != bytes.len) return error.BadFixture;
    const p_stride = 3 * n + H; // [q | k | v | beta logits]
    const p = try a.alloc(u16, T * p_stride);
    for (0..T) |t| {
        @memcpy(p[t * p_stride ..][0 .. 3 * n], qkv[t * 3 * n ..][0 .. 3 * n]);
        @memcpy(p[t * p_stride + 3 * n ..][0..H], b[t * H ..][0..H]);
    }
    const norm = try a.alloc(u16, dk); // the o_norm weight is bf16 in the checkpoint: exact
    for (norm, norm32) |*d, x| {
        d.* = @truncate(@as(u32, @bitCast(x)) >> 16);
        if (f64r.bf(d.*) != x) return error.NormNotBf16;
    }
    const zero = try a.alloc(f32, n * dk);
    @memset(zero, 0);
    const cs = try a.alloc(u16, T * 9 * n);
    @memset(cs, 0);
    for (0..T) |t| for (0..3) |j| if (t + j >= 3) @memcpy(cs[(t * 3 + j) * 3 * n ..][0 .. 3 * n], qkv[(t + j - 3) * 3 * n ..][0 .. 3 * n]);
    const in: f64r.Inputs = .{ .p = p, .a = f, .g = gate, .cs = cs[0 .. 9 * n], .cw = conv, .state = zero, .a_log = a_log, .dt_bias = dt_bias, .norm = norm };
    const B = std.mem.sliceAsBytes;
    const c: kda.Chain = .{ .p = try r.dev(B(p)), .p_stride = p_stride, .b_off = 3 * n, .a = try r.dev(B(f)), .a_stride = n, .g = try r.dev(B(gate)), .g_stride = n, .conv_state = 0, .conv_w = try r.dev(B(conv)), .state_in = try r.dev(B(zero)), .a_log = try r.dev(B(a_log)), .dt_bias = try r.dev(B(dt_bias)), .norm_w = try r.dev(B(norm)), .eps = eps, .lower = lower, .heads = H };
    const w64 = struct {
        fn of(al: std.mem.Allocator, x: anytype) ![]f64 {
            const out = try al.alloc(f64, x.len);
            for (out, x) |*d, v| d.* = if (@TypeOf(v) == u16) f64r.bf(v) else v;
            return out;
        }
    };
    return .{ .P = P, .T = T, .H = H, .l = .{ .heads = H, .p_stride = p_stride, .b_off = 3 * n, .lower = lower, .eps = eps }, .in = in, .c = c, .cs_all = try r.dev(B(cs)), .fla_y = try w64.of(a, fy), .fla_sp = try w64.of(a, fsp), .fla_st = try w64.of(a, fst) };
}

/// One run's outputs [T][H 128] and its states after the prompt (P rows) and at the end.
const Result = struct { y: []f64, sp: u64, st: u64 };

/// Rows lo .. T as decode rows (fused chain, one row a call) from `state`; each row's state into states[t] when given.
fn decode(r: *Rig, m: *const rig.Mods, l: Layer, lo: usize, state: u64, y: []f64, states: ?[]u64) !u64 {
    const n = l.H * dk;
    var s = state;
    for (lo..l.T) |t| {
        const o = try rig.Out.alloc(r, l.H, 1);
        try rig.run(m, r.s, l.at(t, s), 1, o, .fused);
        @memcpy(y[t * n ..][0..n], try rig.wide64(r, o.out, n, true));
        s = o.state;
        if (states) |ss| ss[t + 1] = s;
    }
    return s;
}

/// A prompt of P rows through `path`, then decode rows; `saves` returns the prompt's buffers.
fn promptThenDecode(r: *Rig, m: *const rig.Mods, l: Layer, path: rig.Path, saves: *rig.Out) !Result {
    const n = l.H * dk;
    const y = try r.a.alloc(f64, l.T * n);
    saves.* = try rig.Out.alloc(r, l.H, l.P);
    try rig.run(m, r.s, l.at(0, l.c.state_in), l.P, saves.*, path);
    @memcpy(y[0 .. l.P * n], try rig.wide64(r, saves.out, l.P * n, true));
    return .{ .y = y, .sp = saves.state, .st = try decode(r, m, l, l.P, saves.state, y, null) };
}

fn fracs(r: *Rig, l: Layer, res: Result, e2e_p: f64r.Want, e2e: f64r.Want) ![4]f64 {
    const n = l.H * dk;
    const sp = try rig.wide64(r, res.sp, n * dk, false);
    const st = try rig.wide64(r, res.st, n * dk, false);
    return .{ f64r.fraction(res.y, e2e.out, e2e.out_tol), f64r.fraction(sp, e2e_p.state, e2e_p.state_tol), f64r.fraction(st, e2e.state, e2e.state_tol), @max(f64r.relL2(sp, l.fla_sp), f64r.relL2(st, l.fla_st)) };
}

/// `glm-kda-ling FIXTURE`.
pub fn run(gpu: check.Gpu, path: []const u8) !void {
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var prng = std.Random.DefaultPrng.init(1);
    var r: Rig = .{ .gpu = gpu, .a = arena.allocator(), .s = s, .rng = prng.random() };
    defer {
        for (r.bufs.items) |*b| b.free();
        r.bufs.deinit(gpu.gpa);
    }
    const l = (try load(&r, path)) orelse return std.debug.print("SKIP glm-kda-ling: no fixture at {s}\n", .{path});
    var m = try rig.Mods.load(gpu.d);
    defer m.unload();
    const n = l.H * dk;
    const T = l.T;
    var name: [128]u8 = undefined;
    const P = std.fmt.bufPrint;
    // every row a decode row, each row's state kept
    const states = try r.a.alloc(u64, T + 1);
    states[0] = l.c.state_in;
    const yd = try r.a.alloc(f64, T * n);
    _ = try decode(&r, &m, l, 0, l.c.state_in, yd, states);
    const dec: Result = .{ .y = yd, .sp = states[l.P], .st = states[T] };
    // 16-row verify windows; each replayed whole (== its state) and over its first 5 rows (== the decode rows' state)
    const yw = try r.a.alloc(f64, T * n);
    var ws = l.c.state_in;
    var w0: usize = 0;
    while (w0 < T) : (w0 += 16) {
        const rows = @min(16, T - w0);
        const o = try rig.Out.alloc(&r, l.H, rows);
        try rig.run(&m, r.s, l.at(w0, ws), rows, o, .fused);
        @memcpy(yw[w0 * n ..][0 .. rows * n], try rig.wide64(&r, o.out, rows * n, true));
        const full = try r.zeros(n * dk * 4);
        const part = try r.zeros(n * dk * 4);
        try kda.replay(&m.ks, r.s, ws, o.k, o.v, o.g, o.b, rows, l.H, full);
        try kda.replay(&m.ks, r.s, ws, o.k, o.v, o.g, o.b, @min(5, rows), l.H, part);
        try check.sameBytes(try P(&name, "Ling window at {d}: replay == the window's state", .{w0}), try r.back(full, n * dk * 4), try r.back(o.state, n * dk * 4));
        try check.sameBytes(try P(&name, "Ling window at {d}: replay of 5 rows == 5 decode rows' state", .{w0}), try r.back(part, n * dk * 4), try r.back(states[w0 + @min(5, rows)], n * dk * 4));
        ws = o.state;
    }
    try check.sameBytes("Ling: 16-row windows' state == decode rows' state", try r.back(ws, n * dk * 4), try r.back(states[T], n * dk * 4));
    for (yw, yd) |a, b| if (a != b) return check.expect(false, "Ling: 16-row windows' outputs == decode rows' outputs", .{});
    const win: Result = .{ .y = yw, .sp = states[l.P], .st = ws };
    var wsv: rig.Out = undefined;
    const wide = try promptThenDecode(&r, &m, l, .wide, &wsv);
    for (wide.y, yd) |a, b| if (a != b) return check.expect(false, "Ling: the wide prompt's and decode outputs == every row a decode row's", .{});
    try ref.sameRows(&r, "Ling: the wide prompt's state == P decode rows' state", wide.sp, states[l.P], n * dk * 4);
    const e2e_p = try f64r.chain(r.a, l.l, l.in, l.P);
    const e2e = try f64r.chain(r.a, l.l, l.in, T);
    const rec = try f64r.recurrence(r.a, l.H, l.in.state, try rig.saved(&r, wsv, l.H), l.P);
    const recw = [2]f64{ f64r.fraction(try rig.wide64(&r, wsv.y, l.P * n, true), rec.out, rec.out_tol), f64r.fraction(try rig.wide64(&r, wsv.state, n * dk, false), rec.state, rec.state_tol) };
    var worst: f64 = @max(recw[0], recw[1]);
    var most_l2: f64 = 0;
    std.debug.print("RESULT glm-kda-ling: Ling-3.1 layer 0, {d} heads, {d} prompt + {d} decode rows; fla rel L2 of outputs (states after the prompt / end), worst fraction of float64's bound (out, state after prompt, at end)\n", .{ l.H, l.P, T - l.P });
    for ([_]Result{ dec, win, wide }, [_][]const u8{ "chain, every row a decode row", "chain, 16-row windows", "wide prompt + decode" }) |res, label| {
        const f = try fracs(&r, l, res, e2e_p, e2e);
        const sp = try rig.wide64(&r, res.sp, n * dk, false);
        const st = try rig.wide64(&r, res.st, n * dk, false);
        const yl2 = f64r.relL2(res.y, l.fla_y);
        std.debug.print("RESULT   {s}: rel L2 {e:.3} (states {e:.3} / {e:.3}); float64 {d:.3}, {d:.4}, {d:.4}\n", .{ label, yl2, f64r.relL2(sp, l.fla_sp), f64r.relL2(st, l.fla_st), f[0], f[1], f[2] });
        worst = @max(worst, @max(f[0], @max(f[1], f[2])));
        most_l2 = @max(most_l2, @max(yl2, f[3]));
    }
    const fla_f = f64r.fraction(l.fla_y, e2e.out, e2e.out_tol);
    std.debug.print("RESULT   fla's own outputs: float64 {d:.3}; recurrence from the wide prompt's saves, y {d:.3} state {d:.3}\n", .{ fla_f, recw[0], recw[1] });
    try check.expect(worst <= 1, "glm-kda-ling: every path within float64's bound (worst {d:.3})", .{worst});
    try check.expect(most_l2 < 1e-2, "glm-kda-ling: every path within 1 % of fla (worst rel L2 {e:.3})", .{most_l2});
    check.pass("glm-kda-ling: glm_kda serves Ling layer 0 as is; chain, windows and wide within float64's bound (worst {d:.3}) and fla (rel L2 <= {e:.3}); replays == chain bits", .{ worst, most_l2 });
}
