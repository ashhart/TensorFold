//! glm.dsa.block: a real RED-SNOW DSA layer at TP=2 rank 0 vs the Python engine's dsa_block, bit for bit.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const quant_ref_tests = @import("quant_ref_tests.zig");
const exl3_tests = @import("exl3_tests.zig");
const ref = @import("glm_kda_ref.zig");
const lref = @import("glm_layer_ref.zig");
const dref = @import("glm_dsa_ref.zig");
const dsa = cuda.glm_dsa;
const exl3 = cuda.exl3;

const Run = exl3_tests.Run;
const Rig = quant_ref_tests.RigT;
const Obj = std.json.ObjectMap;
const id = "glm.dsa.block";

pub fn run(r: *Rig, state: *Run, dir: []const u8) !void {
    block(r, dir) catch |e| state.fail(e, id);
}

fn skip() void {
    std.debug.print("SKIP {s}\n", .{id});
}

fn file(r: *Rig, dir: []const u8, name: []const u8) ![]u8 {
    return ref.readFile(r.a, r.gpu.io, dir, name);
}

/// Device bytes back to the host 8-aligned (the f64 chain reads them as u16/f32/i32 slices).
fn back(r: *Rig, ptr: u64, len: usize) ![]u8 {
    const out = try r.a.alignedAlloc(u8, .@"8", len);
    try r.s.synchronize();
    const b: cuda.DeviceBuffer = .{ .d = r.gpu.d, .ptr = ptr, .len = len };
    try b.download(0, out);
    return out;
}

fn linW(r: *Rig, l: lref.Lin) !exl3.Weight {
    const st = exl3.strides(l.k, l.n, l.halves, true);
    return .{ .trellis = try r.dev(l.words), .suh = try r.dev(l.suh), .svh = try r.dev(l.svh), .k = @intCast(l.k), .n = @intCast(l.n), .halves = l.halves, .codebook = .mul1, .stride_k = st.k, .stride_nb = st.nb };
}

fn b16W(r: *Rig, m: lref.Mat) !cuda.glm_bmm.B16 {
    return .{ .w = try r.dev(m.w), .n = @intCast(m.n), .k = @intCast(m.k) };
}

/// The caches every window starts from: FILL tokens the real block wrote (kc fp8, scales, ik, ig, pk).
const names = [_][]const u8{ "kc", "scales", "ik", "ig", "pk" };
const row_bytes = [_]usize{ dsa.kv_lora, 4, dsa.index_dim * 2, dsa.index_dim * 2, dsa.index_dim * 2 };

fn block(r: *Rig, dir: []const u8) !void {
    if (std.mem.eql(u8, dir, "none")) return skip();
    var man = try ref.Manifest.open(r.gpu.gpa, r.gpu.io, dir);
    defer man.deinit();
    const o = (man.root().get("dsa") orelse return skip()).object;
    const p = o.get("params").?.object;
    const c: dsa.Config = .{
        .eps = @floatCast(ref.float(p, "eps")),
        .index_eps = @floatCast(ref.float(p, "index_eps")),
        .scale = @floatCast(ref.float(p, "scale")),
        .index_scale = @floatCast(ref.float(p, "index_scale")),
        .wscale = @floatCast(ref.float(p, "wscale")),
    };
    const p64: dref.Params = .{ .eps = c.eps, .index_eps = c.index_eps, .scale = c.scale, .index_scale = c.index_scale, .wscale = c.wscale };
    const cap: usize = @intCast(ref.int(p, "cap"));
    const host = try dref.Host.read(r.a, r.gpu.io, dir, o.get("weights").?.object);
    const w: dsa.DsaW = .{
        .q_a = try linW(r, host.q_a),
        .kv_a = try linW(r, host.kv_a),
        .q_b = try linW(r, host.q_b),
        .o = try linW(r, host.o),
        .q_norm = try r.dev(host.q_norm),
        .kv_norm = try r.dev(host.kv_norm),
        .wk = try r.dev(host.wk),
        .wv = try r.dev(host.wv),
        .index = .{ .kw = try b16W(r, host.kw), .qb = try b16W(r, host.qb), .gate = try r.dev(host.gate), .ln_w = try r.dev(host.ln_w), .ln_b = try r.dev(host.ln_b), .ape = try r.dev(host.ape) },
    };
    var pre: [names.len][]u8 = undefined;
    for (names, 0..) |n, i| pre[i] = try file(r, dir, ref.str(o.get("pre").?.object, n));
    var glue2 = try cuda.glm_glue2.Glue2.load(r.gpu.d);
    defer glue2.unload();
    var bmm = try cuda.glm_bmm.Bmm.load(r.gpu.d);
    defer bmm.unload();
    var dense = try exl3.Dense.load(r.gpu.d);
    defer dense.unload();
    var lat = try cuda.glm_latent.Latent.load(r.gpu.d);
    defer lat.unload();
    var sp = try cuda.glm_sparse.Sparse.load(r.gpu.d);
    defer sp.unload();
    const k: dsa.Kernels = .{ .glue2 = &glue2, .bmm = &bmm, .dense = &dense, .latent = &lat, .sparse = &sp };
    var worst: f64 = 0;
    const cases = o.get("cases").?.object;
    var it = cases.iterator();
    while (it.next()) |kv| {
        const t0 = check.now(r.gpu.io);
        // a window's host bytes live in its own arena
        var arena = std.heap.ArenaAllocator.init(r.gpu.gpa);
        defer arena.deinit();
        const saved = r.a;
        r.a = arena.allocator();
        defer r.a = saved;
        const q = try window(r, k, c, p64, &w, &host, &pre, cap, dir, kv.key_ptr.*, kv.value_ptr.object);
        worst = @max(worst, q);
        const dt = @as(f64, @floatFromInt(check.now(r.gpu.io) - t0)) / 1e9;
        std.debug.print("RESULT {s} {s}: bit-exact, f64 worst {d:.4} of tolerance ({d:.1} s)\n", .{ id, kv.key_ptr.*, q, dt });
    }
    std.debug.print("RESULT {s}: {d} windows bit-exact, f64 worst {d:.4} of tolerance\n", .{ id, cases.count(), worst });
    check.pass("{s}", .{id});
}

fn same(r: *Rig, tag: []const u8, what: []const u8, got: []const u8, want: []const u8) !void {
    try check.sameBytes(try std.fmt.allocPrint(r.a, "{s} {s} {s}", .{ id, tag, what }), got, want);
}

/// One window from the shared pre-state: every stage's bits, the fp32 partial, the caches after; then the f64 chain.
fn window(r: *Rig, k: dsa.Kernels, c: dsa.Config, p64: dref.Params, w: *const dsa.DsaW, host: *const dref.Host, pre: *const [names.len][]u8, cap: usize, dir: []const u8, tag: []const u8, e: Obj) !f64 {
    const rows: usize = @intCast(ref.int(e, "rows"));
    const pos: usize = @intCast(ref.int(e, "pos"));
    var sc = try dsa.Scratch.init(r.gpu.d, w, @intCast(ref.int(e, "buf_rows")), e.get("prefill").?.bool, cap);
    defer sc.deinit();
    try sc.buf.fill8(0xff, r.s.handle); // torch.empty's scratch: a read of an unwritten word shows
    var caches: [names.len]u64 = undefined;
    for (pre, 0..) |bytes, i| caches[i] = try r.dev(bytes);
    var pos_b: [4]u8 = undefined;
    std.mem.writeInt(i32, &pos_b, @intCast(pos), .little);
    const st: dsa.DsaState = .{ .lc = caches[0], .scales = caches[1], .ik = caches[2], .ig = caches[3], .pk = caches[4], .pos_dev = try r.dev(&pos_b), .cap = cap };
    const normed = try file(r, dir, ref.str(e, "normed"));
    const part = try r.zeros(rows * dsa.hidden * 4);
    try dsa.dsaBlock(k, &r.s, c, w, st, &sc, try r.dev(normed), part, rows, pos);
    // every stage in the block's order: the first that differs names the kernel
    const stages = [_]struct { []const u8, u64 }{
        .{ "dqa", sc.dqa }, .{ "dkv", sc.dkv },       .{ "qr", sc.qr },         .{ "lat", sc.lat },
        .{ "q", sc.q },     .{ "ikr", sc.ikr },       .{ "igr", sc.igr },       .{ "qa", sc.qa },
        .{ "qi", sc.qi },   .{ "tokens", sc.tokens }, .{ "counts", sc.counts }, .{ "ol", sc.ol },
        .{ "vn", sc.vn },   .{ "part", part },
    };
    var got: [stages.len]?[]const u8 = undefined;
    for (stages, 0..) |stg, i| {
        got[i] = null;
        const name = (e.get(stg[0]) orelse continue).string;
        const want = try file(r, dir, name);
        const g = try back(r, stg[1], want.len);
        try same(r, tag, stg[0], g, want);
        got[i] = g;
    }
    // the caches: the window's rows as Python wrote them, every other row as the pre-state left it
    var post: [names.len][]const u8 = undefined;
    for (names, 0..) |n, i| {
        const co = e.get(n).?.object;
        const lo: usize = @intCast(ref.int(co, "lo"));
        const hi: usize = @intCast(ref.int(co, "hi"));
        const want = try r.a.dupe(u8, pre[i]);
        const win = try file(r, dir, ref.str(co, "file"));
        try check.expect(win.len == (hi - lo) * row_bytes[i], "{s} {s} {s}: window bytes", .{ id, tag, n });
        @memcpy(want[lo * row_bytes[i] ..][0..win.len], win);
        post[i] = try back(r, caches[i], want.len);
        try same(r, tag, n, post[i], want);
    }
    const np = dsa.poolBucket(pos, rows, cap / 4);
    const last = (rows - 1) / dsa.select_rows * dsa.select_rows;
    const g: dref.Got = .{
        .normed = normed,
        .dqa = got[0].?,
        .dkv = got[1].?,
        .qr = got[2].?,
        .lat = got[3].?,
        .q = got[4].?,
        .ikr = got[5].?,
        .igr = got[6].?,
        .qa = got[7].?,
        .qi = got[8],
        .tokens = got[9],
        .counts = got[10],
        .ol = got[11].?,
        .vn = got[12].?,
        .part = got[13].?,
        .scores = if (got[8] != null) try back(r, sc.scores, (rows - last) * np * 4) else null,
        .np = np,
        .kc = post[0],
        .scales = post[1],
        .ik = post[2],
        .ig = post[3],
        .pk = post[4],
    };
    return dref.chain(r.a, host, p64, g, tag, rows, pos);
}
