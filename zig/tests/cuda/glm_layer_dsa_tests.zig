//! glm.layer.dsa: a real RED-SNOW DSA MoE layer at TP=2 rank 0 vs the Python engine's layer_forward, bit for bit.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const quant_ref_tests = @import("quant_ref_tests.zig");
const exl3_tests = @import("exl3_tests.zig");
const ref = @import("glm_kda_ref.zig");
const lref = @import("glm_layer_ref.zig");
const dref = @import("glm_dsa_ref.zig");
const ldref = @import("glm_layer_dsa_ref.zig");
const lt = @import("glm_layer_tests.zig");
const lay = cuda.glm_layer;
const dsa = cuda.glm_dsa;

const Run = exl3_tests.Run;
const Rig = quant_ref_tests.RigT;
const Obj = std.json.ObjectMap;
const Stage = lt.Stage;
const id = "glm.layer.dsa";

pub fn run(r: *Rig, state: *Run, dir: []const u8) !void {
    dsaLayer(r, dir) catch |e| state.fail(e, id);
}

fn skip() void {
    std.debug.print("SKIP {s}\n", .{id});
}

/// The caches every window starts from (kc fp8, scales, ik, ig, pk), as the real prefill of this layer left them.
const names = [_][]const u8{ "kc", "scales", "ik", "ig", "pk" };
const row_bytes = [_]usize{ dsa.kv_lora, 4, dsa.index_dim * 2, dsa.index_dim * 2, dsa.index_dim * 2 };

/// The fixed inputs of every window.
const Layer = struct {
    c: lay.Config,
    p64: dref.Params,
    cap: usize,
    host: lref.Host,
    dh: dref.Host,
    w: lay.LayerW,
    pre: [names.len][]u8,
};

fn dsaLayer(r: *Rig, dir: []const u8) !void {
    if (std.mem.eql(u8, dir, "none")) return skip();
    var man = try ref.Manifest.open(r.gpu.gpa, r.gpu.io, dir);
    defer man.deinit();
    const o = (man.root().get("layer_dsa") orelse return skip()).object;
    const p = o.get("params").?.object;
    var c = lt.config(p);
    c.dsa = .{
        .eps = c.eps,
        .index_eps = @floatCast(ref.float(p, "index_eps")),
        .scale = @floatCast(ref.float(p, "scale")),
        .index_scale = @floatCast(ref.float(p, "index_scale")),
        .wscale = @floatCast(ref.float(p, "wscale")),
    };
    const dc = c.dsa.?;
    const wo = o.get("weights").?.object;
    var l: Layer = .{
        .c = c,
        .p64 = .{ .eps = dc.eps, .index_eps = dc.index_eps, .scale = dc.scale, .index_scale = dc.index_scale, .wscale = dc.wscale },
        .cap = @intCast(ref.int(p, "cap")),
        .host = try lref.Host.read(r.a, r.gpu.io, dir, wo),
        .dh = try dref.Host.read(r.a, r.gpu.io, dir, wo.get("dsa").?.object),
        .w = undefined,
        .pre = undefined,
    };
    const h = &l.host;
    const dh = &l.dh;
    l.w = .{
        .attn_hc = .{ .fn_w = try r.dev(h.attn_hc[0]), .base = try r.dev(h.attn_hc[1]), .scale = try r.dev(h.attn_hc[2]) },
        .ffn_hc = .{ .fn_w = try r.dev(h.ffn_hc[0]), .base = try r.dev(h.ffn_hc[1]), .scale = try r.dev(h.ffn_hc[2]) },
        .in_norm = try r.dev(h.in_norm),
        .post_norm = try r.dev(h.post_norm),
        .mixer = .{ .dsa = .{
            .q_a = try lt.linW(r, dh.q_a),
            .kv_a = try lt.linW(r, dh.kv_a),
            .q_b = try lt.linW(r, dh.q_b),
            .o = try lt.linW(r, dh.o),
            .q_norm = try r.dev(dh.q_norm),
            .kv_norm = try r.dev(dh.kv_norm),
            .wk = try r.dev(dh.wk),
            .wv = try r.dev(dh.wv),
            .index = .{ .kw = try lt.b16W(r, dh.kw), .qb = try lt.b16W(r, dh.qb), .gate = try r.dev(dh.gate), .ln_w = try r.dev(dh.ln_w), .ln_b = try r.dev(dh.ln_b), .ape = try r.dev(dh.ape) },
        } },
        .ffn = .{ .moe = try lt.moeW(r, h, c) },
    };
    for (names, 0..) |n, i| l.pre[i] = try lt.file(r, dir, ref.str(o.get("pre").?.object, n));
    var k = try lay.Kernels.load(r.gpu.d);
    defer k.unload();
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
        const q = try window(r, &k, &l, dir, kv.key_ptr.*, kv.value_ptr.object);
        worst = @max(worst, q);
        const dt = @as(f64, @floatFromInt(check.now(r.gpu.io) - t0)) / 1e9;
        std.debug.print("RESULT {s} {s}: bit-exact, f64 worst {d:.4} of tolerance ({d:.1} s)\n", .{ id, kv.key_ptr.*, q, dt });
    }
    std.debug.print("RESULT {s}: {d} windows bit-exact, f64 worst {d:.4} of tolerance\n", .{ id, cases.count(), worst });
    check.pass("{s}", .{id});
}

/// Device bytes back to the host 8-aligned (the f64 chain reads them as u16/f32/i32 slices).
fn back(r: *Rig, ptr: u64, len: usize) ![]u8 {
    const out = try r.a.alignedAlloc(u8, .@"8", len);
    try r.s.synchronize();
    const b: cuda.DeviceBuffer = .{ .d = r.gpu.d, .ptr = ptr, .len = len };
    try b.download(0, out);
    return out;
}

/// The fixture's bytes 8-aligned.
fn file8(r: *Rig, dir: []const u8, name: []const u8) ![]u8 {
    const raw = try lt.file(r, dir, name);
    const out = try r.a.alignedAlloc(u8, .@"8", raw.len);
    @memcpy(out, raw);
    return out;
}

/// One window from the pre-state: forward with every stage's bits checked at the gathers, x and the caches after.
fn window(r: *Rig, k: *lay.Kernels, l: *const Layer, dir: []const u8, tag: []const u8, e: Obj) !f64 {
    const rows: usize = @intCast(ref.int(e, "rows"));
    const pos: usize = @intCast(ref.int(e, "pos"));
    const prefill = e.get("prefill").?.bool;
    const w = &l.w;
    var sc = try lay.Scratch.init(r.gpu.d, r.s, l.c, w, rows, 2);
    defer sc.deinit();
    var dsc = try dsa.Scratch.init(r.gpu.d, &w.mixer.dsa, @intCast(ref.int(e, "buf_rows")), prefill, l.cap);
    defer dsc.deinit();
    try dsc.buf.fill8(0xff, r.s.handle); // torch.empty's scratch: a read of an unwritten word shows
    var caches: [names.len]u64 = undefined;
    for (l.pre, 0..) |bytes, i| caches[i] = try r.dev(bytes);
    var pos_b: [4]u8 = undefined;
    std.mem.writeInt(i32, &pos_b, @intCast(pos), .little);
    const st: dsa.DsaState = .{ .lc = caches[0], .scales = caches[1], .ik = caches[2], .ig = caches[3], .pk = caches[4], .pos_dev = try r.dev(&pos_b), .cap = l.cap };
    const x = try lt.up(r, dir, ref.str(e, "x0"));
    // the DSA block's stages in its order: the first that differs names the kernel
    const attn = [_]Stage{
        .{ "dqa", dsc.dqa }, .{ "dkv", dsc.dkv },       .{ "qr", dsc.qr },         .{ "lat", dsc.lat },
        .{ "q", dsc.q },     .{ "ikr", dsc.ikr },       .{ "igr", dsc.igr },       .{ "qa", dsc.qa },
        .{ "qi", dsc.qi },   .{ "tokens", dsc.tokens }, .{ "counts", dsc.counts }, .{ "ol", dsc.ol },
        .{ "vn", dsc.vn },
    };
    var ctx: lt.Ctx = .{ .r = r, .id = id, .dir = dir, .tag = tag, .e = e, .sc = &sc, .attn = &attn, .x = x, .d = l.c.d };
    const g: lay.Gather = .{ .ctx = &ctx, .world = 2, .run = lt.Ctx.gather };
    try lay.forward(k, &r.s, l.c, w, .{ .dsa = .{ .st = st, .sc = &dsc, .pos = pos } }, &sc, x, rows, g);
    try lt.same(r, dir, try std.fmt.allocPrint(r.a, "{s} {s} x", .{ id, tag }), x, ref.str(e, "x"));
    // the caches: the window's rows as Python wrote them, every other row as the pre-state left it
    var post: [names.len][]const u8 = undefined;
    for (names, 0..) |n, i| {
        const co = e.get(n).?.object;
        const lo: usize = @intCast(ref.int(co, "lo"));
        const hi: usize = @intCast(ref.int(co, "hi"));
        const want = try r.a.dupe(u8, l.pre[i]);
        const win = try lt.file(r, dir, ref.str(co, "file"));
        try check.expect(win.len == (hi - lo) * row_bytes[i], "{s} {s} {s}: window bytes", .{ id, tag, n });
        @memcpy(want[lo * row_bytes[i] ..][0..win.len], win);
        post[i] = try back(r, caches[i], want.len);
        try check.sameBytes(try std.fmt.allocPrint(r.a, "{s} {s} {s}", .{ id, tag, n }), post[i], want);
    }
    // the oracle ran the chunk row-split too (HC_SPLIT=1, PREFILL_OVERLAP=2): own rows and every cache equal unsplit
    if (prefill) try check.expect(if (e.get("split_equal")) |v| v.bool else false, "{s} {s}: no split check", .{ id, tag });
    // the f64 chain reads the block's stages (bit-equal by now) from the fixture
    const s0 = e.get("s0").?.object;
    var got: [attn.len + 2]?[]const u8 = undefined;
    const keys = [_][]const u8{ "dqa", "dkv", "qr", "lat", "q", "ikr", "igr", "qa", "qi", "tokens", "counts", "ol", "vn", "normed", "part" };
    for (keys, 0..) |key, i| got[i] = if (s0.get(key)) |v| try file8(r, dir, v.string) else null;
    const np = dsa.poolBucket(pos, rows, l.cap / 4);
    const last = (rows - 1) / dsa.select_rows * dsa.select_rows;
    const dg: dref.Got = .{
        .normed = got[13].?,
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
        .part = got[14].?,
        .scores = if (got[8] != null) try back(r, dsc.scores, (rows - last) * np * 4) else null,
        .np = np,
        .kc = post[0],
        .scales = post[1],
        .ik = post[2],
        .ig = post[3],
        .pk = post[4],
    };
    return ldref.layerF64(r.gpu.gpa, r.gpu.io, dir, l.c, &l.host, &l.dh, l.p64, dg, tag, e);
}
