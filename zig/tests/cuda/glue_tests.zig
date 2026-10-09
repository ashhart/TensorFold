//! Our glue kernels (nemotron_*.cu) at Nemotron's shapes against their host references (glue_ref.zig): every byte.

const std = @import("std");
const cuda = @import("cuda");
const nemotron = @import("nemotron");
const check = @import("check.zig");
const Gpu = check.Gpu;
const ref = nemotron.glue_ref;
const gm = nemotron.glue_math;
const glue = nemotron.glue;

const D = 2688;
const E = 128;
const top_k = 6;
const NS = 8;
const eps: f32 = 1e-5;
const ms: ref.Shape = .{ .proj = 10304, .xd = 4096, .cd = 6144, .heads = 64, .dh = 64, .groups = 8, .rmax = 16 };
const at: ref.Attn = .{ .nqkv = 4608, .heads = 32, .kv_heads = 2, .dim = 128, .nch = 12 };

fn mshape() glue.Shape {
    return .{ .proj = ms.proj, .xd = ms.xd, .cd = ms.cd, .heads = ms.heads, .dh = ms.dh, .groups = ms.groups, .state = 128, .rmax = ms.rmax };
}

fn ashape() glue.Attn {
    return .{ .nqkv = at.nqkv, .heads = at.heads, .kv_heads = at.kv_heads, .dim = at.dim, .nch = at.nch };
}

const Rig = struct {
    gpu: Gpu,
    a: std.mem.Allocator, // host data, freed together at the end
    s: cuda.Stream,
    f: glue.Fns,
    rng: std.Random,
    bufs: std.ArrayList(cuda.DeviceBuffer) = .empty,
    bad: usize = 0,
    cases: usize = 0,

    fn g(r: *Rig) glue.Glue {
        return .{ .set = null, .f = &r.f, .s = r.s };
    }

    /// A device copy of host values, landed before any kernel on the rig's stream reads it.
    fn dev(r: *Rig, comptime T: type, host: []const T) !u64 {
        const b = try cuda.DeviceBuffer.fromHost(r.gpu.d, std.mem.sliceAsBytes(host));
        try r.bufs.append(r.gpu.gpa, b);
        try r.gpu.ctx.synchronize();
        return b.ptr;
    }

    fn zeros(r: *Rig, comptime T: type, n: usize) !u64 {
        const host = try r.a.alloc(T, n);
        @memset(std.mem.sliceAsBytes(host), 0);
        return r.dev(T, host);
    }

    fn bfs(r: *Rig, n: usize, scale: f32) ![]u16 {
        const out = try r.a.alloc(u16, n);
        for (out) |*v| v.* = gm.f32ToBf16((r.rng.float(f32) * 2 - 1) * scale);
        return out;
    }

    fn f32s(r: *Rig, n: usize, lo: f32, hi: f32) ![]f32 {
        const out = try r.a.alloc(f32, n);
        for (out) |*v| v.* = lo + r.rng.float(f32) * (hi - lo);
        return out;
    }

    /// The device bytes at `ptr` against `want`, element for element; a difference names its first element.
    fn same(r: *Rig, what: []const u8, comptime T: type, ptr: u64, want: []const T) !void {
        try r.s.synchronize();
        const got = try r.a.alloc(T, want.len);
        try r.gpu.d.check(r.gpu.d.api.cuMemcpyDtoH_v2(got.ptr, ptr, want.len * @sizeOf(T)), "cuMemcpyDtoH");
        r.cases += 1;
        if (std.mem.eql(u8, std.mem.sliceAsBytes(got), std.mem.sliceAsBytes(want))) return;
        r.bad += 1;
        var n: usize = 0;
        var first: ?usize = null;
        for (got, want, 0..) |x, y, i| if (!std.mem.eql(u8, std.mem.asBytes(&x), std.mem.asBytes(&y))) {
            n += 1;
            if (first == null) first = i;
        };
        std.debug.print("DIFFER {s}: {d} of {d} elements, first at {d}: got {any}, want {any}\n", .{ what, n, want.len, first.?, got[first.?], want[first.?] });
    }
};

fn norms(r: *Rig) !void {
    const gl = r.g();
    // token rows from a 64-token table
    const words = try r.a.alloc(u32, 64 * D / 8);
    for (words) |*w| w.* = r.rng.int(u32);
    const sc = try r.bfs(64 * D / 64, 0.05);
    const bi = try r.bfs(64 * D / 64, 0.05);
    const ids = try r.a.alloc(i32, 16);
    for (ids) |*i| i.* = r.rng.intRangeLessThan(i32, 0, 64);
    const emb = try r.a.alloc(u16, 16 * D);
    ref.embed(ids, words, sc, bi, emb, D);
    const emb_d = try r.zeros(u16, 16 * D);
    try gl.embed(try r.dev(i32, ids), try r.dev(u32, words), try r.dev(u16, sc), try r.dev(u16, bi), emb_d, 16, D);
    try r.same("embed", u16, emb_d, emb);
    for ([_]usize{ 1, 16, 300 }, [_]bool{ false, true, true }) |rows, with_r| {
        const x = try r.bfs(rows * D, 2.0);
        const res = try r.bfs(rows * D, 1.0);
        const w = try r.bfs(D, 1.0);
        const h = try r.a.alloc(u16, rows * D);
        const y = try r.a.alloc(u16, rows * D);
        const xs = try r.a.alloc(f32, rows * D / 64);
        try ref.addRmsnorm(r.a, x, if (with_r) res else null, w, h, y, xs, rows, D, eps);
        const hd = try r.zeros(u16, rows * D);
        const yd = try r.zeros(u16, rows * D);
        const xsd = try r.zeros(f32, rows * D / 64);
        const xd = try r.dev(u16, x);
        try gl.addRmsnorm(xd, if (with_r) try r.dev(u16, res) else null, try r.dev(u16, w), if (with_r) hd else xd, yd, xsd, rows, D, eps);
        if (with_r) try r.same("add_rmsnorm h", u16, hd, h);
        try r.same("add_rmsnorm y", u16, yd, y);
        try r.same("add_rmsnorm xs", f32, xsd, xs);
    }
    for ([_]usize{ 16, 300 }, [_]bool{ true, false }) |rows, f32_out| {
        const x = try r.bfs(rows * D, 2.0);
        const y32 = try r.f32s(rows * NS * D, -1, 1);
        const y16 = try r.bfs(rows * NS * D, 1.0);
        const wt = try r.f32s(rows * NS, 0, 0.5);
        const w = try r.bfs(D, 1.0);
        const hn = try r.a.alloc(u16, rows * D);
        const out = try r.a.alloc(u16, rows * D);
        const xs = try r.a.alloc(f32, rows * D / 64);
        try ref.addMoeNorm(r.a, x, if (f32_out) y32 else null, if (f32_out) null else y16, wt, w, hn, out, xs, rows, D, eps, top_k, NS);
        const hnd = try r.zeros(u16, rows * D);
        const outd = try r.zeros(u16, rows * D);
        const xsd = try r.zeros(f32, rows * D / 64);
        const yd = if (f32_out) try r.dev(f32, y32) else try r.dev(u16, y16);
        try gl.addMoeNorm(try r.dev(u16, x), yd, f32_out, try r.dev(f32, wt), try r.dev(u16, w), hnd, outd, xsd, rows, D, eps, top_k, NS);
        try r.same("add_moe_norm h", u16, hnd, hn);
        try r.same("add_moe_norm y", u16, outd, out);
        try r.same("add_moe_norm xs", f32, xsd, xs);
    }
    {
        const rows = 16;
        const e = try r.bfs(rows * D, 2.0);
        const hid = try r.bfs(rows * D, 3.0);
        const we = try r.bfs(D, 1.0);
        const wh = try r.bfs(D, 1.0);
        const out = try r.a.alloc(u16, rows * 2 * D);
        const xs = try r.a.alloc(f32, rows * 2 * D / 64);
        try ref.concatNorms(r.a, e, hid, we, wh, out, xs, rows, D, eps);
        const outd = try r.zeros(u16, rows * 2 * D);
        const xsd = try r.zeros(f32, rows * 2 * D / 64);
        try gl.concatNorms(try r.dev(u16, e), try r.dev(u16, hid), try r.dev(u16, we), try r.dev(u16, wh), outd, xsd, rows, D, eps);
        try r.same("concat_norms y", u16, outd, out);
        try r.same("concat_norms xs", f32, xsd, xs);
    }
    for ([_]usize{ 16, 300 }) |rows| {
        const x = try r.bfs(rows * ms.xd, 2.0);
        const w = try r.bfs(ms.xd, 1.0);
        const out = try r.a.alloc(u16, rows * ms.xd);
        const xs = try r.a.alloc(f32, rows * ms.xd / 64);
        try ref.groupRmsnorm(r.a, x, w, out, xs, rows, ms.xd, ms.groups, eps);
        const outd = try r.zeros(u16, rows * ms.xd);
        const xsd = try r.zeros(f32, rows * ms.xd / 64);
        try gl.groupRmsnorm(try r.dev(u16, x), try r.dev(u16, w), outd, xsd, rows, ms.xd, ms.groups, eps);
        try r.same("group_rmsnorm y", u16, outd, out);
        try r.same("group_rmsnorm xs", f32, xsd, xs);
    }
}

fn route(r: *Rig) !void {
    const sk = glue.routerShape(D).sk;
    for ([_]usize{ 1, 16, 300 }) |rows| {
        const x = try r.bfs(rows * D, 1.0);
        const w = try r.bfs(E * D, 0.05);
        const bias = try r.f32s(E, -0.05, 0.05);
        const part = try r.a.alloc(f32, sk * rows * E);
        const idx = try r.a.alloc(i32, rows * NS);
        const wt = try r.a.alloc(f32, rows * NS);
        ref.router(x, w, part, rows, D, E, sk);
        ref.topk(part, bias, idx, wt, rows, 2.5, E, sk, top_k, NS, true);
        const partd = try r.zeros(f32, sk * rows * E);
        const idxd = try r.zeros(i32, rows * NS);
        const wtd = try r.zeros(f32, rows * NS);
        try r.g().route(try r.dev(u16, x), try r.dev(u16, w), try r.dev(f32, bias), partd, idxd, wtd, rows, D, E, top_k, 2.5, true);
        try r.same("router partials", f32, partd, part);
        try r.same("top-k ids", i32, idxd, idx);
        try r.same("top-k weights", f32, wtd, wt);
    }
}

/// Two windows through conv and scan: the second replays three kept rows of the first, at the other parity.
fn mamba(r: *Rig, lo: f32, hi: f32) !void {
    const cd = ms.cd;
    const base = try r.bfs(3 * cd, 1.0);
    const raw = try r.a.alloc(u16, 2 * ms.rmax * cd);
    @memset(raw, 0);
    const xc = try r.a.alloc(u16, 2 * ms.rmax * cd);
    @memset(xc, 0);
    const cw = try r.f32s(4 * cd, -0.5, 0.5);
    const cb = try r.f32s(cd, -0.1, 0.1);
    const dt = try r.a.alloc(f32, 2 * ms.rmax * ms.heads);
    @memset(dt, 0);
    const st = try r.f32s(ms.heads * ms.dh * 128, -0.2, 0.2);
    const a = try r.f32s(ms.heads, -2.0, -0.05);
    const dsk = try r.f32s(ms.heads, 0.5, 1.5);
    const dtb = try r.f32s(ms.heads, -4.0, 1.0);
    const based = try r.dev(u16, base);
    const rawd = try r.dev(u16, raw);
    const xcd = try r.dev(u16, xc);
    const cwd = try r.dev(f32, cw);
    const cbd = try r.dev(f32, cb);
    const dtd = try r.dev(f32, dt);
    const std_ = try r.dev(f32, st);
    const ad = try r.dev(f32, a);
    const dskd = try r.dev(f32, dsk);
    const dtbd = try r.dev(f32, dtb);
    for ([_][3]usize{ .{ 0, 0, 5 }, .{ 1, 3, 7 }, .{ 0, 7, 16 } }) |w| {
        const parity = w[0];
        const pk = w[1];
        const rows = w[2];
        const p = try r.bfs(rows * ms.proj, 2.0);
        const pd = try r.dev(u16, p);
        const meta = try r.dev(i32, &.{ 100, @intCast(parity), @intCast(pk), 0 });
        ref.conv(p, base, raw, xc, cw, cb, parity, pk, rows, ms);
        try r.g().conv(pd, based, rawd, xcd, cwd, cbd, meta, rows, mshape());
        try r.same("conv xc", u16, xcd, xc);
        try r.same("conv raw", u16, rawd, raw);
        try r.same("conv base", u16, based, base);
        const y = try r.a.alloc(u16, rows * ms.xd);
        @memset(y, 0);
        const yd = try r.dev(u16, y);
        ref.scan(p, xc, dt, st, a, dsk, dtb, y, parity, pk, rows, lo, hi, ms);
        try r.g().scan(pd, xcd, dtd, std_, ad, dskd, dtbd, meta, yd, rows, lo, hi, mshape());
        try r.same("scan y", u16, yd, y);
        try r.same("scan dt", f32, dtd, dt);
        try r.same("scan state", f32, std_, st);
    }
    const rows = 300;
    const p = try r.bfs(rows * ms.proj, 2.0);
    const out = try r.a.alloc(u16, rows * cd);
    ref.convRows(p, base, out, cw, cb, rows, ms);
    const outd = try r.zeros(u16, rows * cd);
    try r.g().convRows(try r.dev(u16, p), based, outd, cwd, cbd, rows, mshape());
    try r.same("conv_rows xc", u16, outd, out);
    try r.same("conv_rows base", u16, based, base);
}

/// A window's keys written at `pos`, then attention over every key through each row's own position.
fn attention(r: *Rig, pos: usize, rows: usize) !void {
    const kvd = at.kv_heads * at.dim;
    const qd = at.heads * at.dim;
    const kc = try r.bfs((pos + rows) * kvd, 1.0);
    const vc = try r.bfs((pos + rows) * kvd, 1.0);
    const qkv = try r.bfs(rows * at.nqkv, 1.0);
    const kcd = try r.dev(u16, kc);
    const vcd = try r.dev(u16, vc);
    const qkvd = try r.dev(u16, qkv);
    const meta = try r.dev(i32, &.{ @intCast(pos), 0, 0, 0 });
    for (0..rows) |row| {
        @memcpy(kc[(pos + row) * kvd ..][0..kvd], qkv[row * at.nqkv + qd ..][0..kvd]);
        @memcpy(vc[(pos + row) * kvd ..][0..kvd], qkv[row * at.nqkv + qd + kvd ..][0..kvd]);
    }
    try r.g().kvWrite(qkvd, kcd, vcd, meta, rows, ashape());
    try r.same("kv_write k", u16, kcd, kc);
    try r.same("kv_write v", u16, vcd, vc);
    const po = try r.a.alloc(f32, rows * at.nch * qd);
    const pm = try r.a.alloc(f32, rows * at.nch * at.heads);
    const pl = try r.a.alloc(f32, rows * at.nch * at.heads);
    const out = try r.a.alloc(u16, rows * qd);
    const xs = try r.a.alloc(f32, rows * qd / 64);
    ref.attention(qkv, kc, vc, pos, rows, at, po, pm, pl, out, xs);
    const pod = try r.zeros(f32, rows * at.nch * qd);
    const pmd = try r.zeros(f32, rows * at.nch * at.heads);
    const pld = try r.zeros(f32, rows * at.nch * at.heads);
    const outd = try r.zeros(u16, rows * qd);
    const xsd = try r.zeros(f32, rows * qd / 64);
    try r.g().attention(qkvd, kcd, vcd, meta, pod, pmd, pld, outd, xsd, rows, ashape());
    // chunks past a row's position are never written: compare the merged rows, and the partials of the last row
    const last = rows - 1;
    const used = (pos + rows + 511) / 512;
    const tail = (last * at.nch) * at.heads;
    try r.same("attention partial m", f32, pmd + tail * 4, pm[tail..][0 .. used * at.heads]);
    try r.same("attention partial l", f32, pld + tail * 4, pl[tail..][0 .. used * at.heads]);
    try r.same("attention partial o", f32, pod + tail * at.dim * 4, po[tail * at.dim ..][0 .. used * qd]);
    try r.same("attention out", u16, outd, out);
    try r.same("attention xs", f32, xsd, xs);
}

fn keyed(r: *Rig) !void {
    const rows = 4;
    const count = 28;
    const vals = try r.a.alloc(f32, rows * count);
    for (vals) |*v| v.* = @floor(r.rng.float(f32) * 24) / 8 - 1; // eighths: plenty of ties
    const ids = try r.a.alloc(i64, rows * count);
    for (ids, 0..) |*i, n| i.* = @intCast((n * 7919 + 13) % 131072);
    const out = try r.a.alloc(i32, rows);
    const prob = try r.a.alloc(f32, rows);
    ref.keyedGreedy(vals, ids, out, prob, rows, count, 20);
    const outd = try r.zeros(i32, rows);
    const probd = try r.zeros(f32, rows);
    try r.g().keyed(try r.dev(f32, vals), try r.dev(i64, ids), 0, outd, 0, 0, probd, 0, rows, count, .{ .k = 20, .greedy = true });
    try r.same("keyed token", i32, outd, out);
    try r.same("keyed share", f32, probd, prob);
}

/// Every glue kernel on synthetic inputs at Nemotron's shapes, byte-compared with its host reference.
pub fn run(gpu: Gpu) !void {
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    const kk = cuda.kernels;
    var mods: [5]cuda.Module = undefined;
    var loaded: usize = 0;
    defer for (mods[0..loaded]) |*m| m.unload();
    for ([_][]const u8{ kk.nemotron_norms, kk.nemotron_route, kk.nemotron_mamba, kk.nemotron_attention, kk.nemotron_keyed }, 0..) |img, i| {
        mods[i] = try cuda.Module.load(gpu.d, img);
        loaded += 1;
    }
    var prng = std.Random.DefaultPrng.init(0x5eed_9a7e);
    var r: Rig = .{ .gpu = gpu, .a = arena.allocator(), .s = try cuda.Stream.init(gpu.d, true), .f = try glue.Fns.resolve(&mods), .rng = prng.random() };
    defer {
        r.s.deinit();
        for (r.bufs.items) |*b| b.free();
        r.bufs.deinit(gpu.gpa);
    }
    try norms(&r);
    try route(&r);
    try mamba(&r, 0, std.math.inf(f32));
    try mamba(&r, 0.001, 0.1);
    for ([_][2]usize{ .{ 0, 16 }, .{ 1000, 16 }, .{ 5000, 3 } }) |c| try attention(&r, c[0], c[1]);
    try keyed(&r);
    try check.expect(r.bad == 0, "glue: {d} of {d} comparisons differ from the host references", .{ r.bad, r.cases });
    check.pass("glue: {d} of {d} comparisons byte-equal to the host references (norms, router, top-k, conv, scan, attention, draw)", .{ r.cases, r.cases });
}
