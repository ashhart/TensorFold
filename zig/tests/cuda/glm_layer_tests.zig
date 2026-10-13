//! glm.layer.kda (glm.layer.mlp: a dense layer): a real KDA layer at TP=2 rank 0 vs Python's layer_forward, bit-exact.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const quant_ref_tests = @import("quant_ref_tests.zig");
const exl3_tests = @import("exl3_tests.zig");
const ref = @import("glm_kda_ref.zig");
const lref = @import("glm_layer_ref.zig");
const lay = cuda.glm_layer;
const exl3 = cuda.exl3;
const exl3e = cuda.exl3_experts;

const Run = exl3_tests.Run;
const Rig = quant_ref_tests.RigT;
const Obj = std.json.ObjectMap;
const id = "glm.layer.kda";

pub fn run(r: *Rig, state: *Run, dir: []const u8) !void {
    kdaLayer(r, dir, "layer", id) catch |e| state.fail(e, id);
}

fn skip(lid: []const u8) void {
    std.debug.print("SKIP {s}\n", .{lid});
}

pub fn file(r: *Rig, dir: []const u8, name: []const u8) ![]u8 {
    return ref.readFile(r.a, r.gpu.io, dir, name);
}

pub fn up(r: *Rig, dir: []const u8, name: []const u8) !u64 {
    return r.dev(try file(r, dir, name));
}

pub fn uint(o: Obj, key: []const u8) u32 {
    return @intCast(ref.int(o, key));
}

/// The layer's constants (manifest params).
pub fn config(p: Obj) lay.Config {
    return .{
        .eps = @floatCast(ref.float(p, "eps")),
        .hc_eps = @floatCast(ref.float(p, "hc_eps")),
        .hc_iters = uint(p, "hc_iters"),
        .lower = @floatCast(ref.float(p, "lower")),
        .limit = @floatCast(ref.float(p, "limit")),
        .routed_scale = @floatCast(ref.float(p, "routed_scale")),
        .top_k = uint(p, "top_k"),
        .experts = uint(p, "experts"),
        .d = uint(p, "hidden"),
        .streams = uint(p, "streams"),
    };
}

/// The manifest section's KDA layer (MoE or dense MLP) through every window, reported as `lid`.
pub fn kdaLayer(r: *Rig, dir: []const u8, section: []const u8, lid: []const u8) !void {
    if (std.mem.eql(u8, dir, "none")) return skip(lid);
    var man = try ref.Manifest.open(r.gpu.gpa, r.gpu.io, dir);
    defer man.deinit();
    const o = (man.root().get(section) orelse return skip(lid)).object;
    const c = config(o.get("params").?.object);
    const host = try lref.Host.read(r.a, r.gpu.io, dir, o.get("weights").?.object);
    const w = try weights(r, &host, c);
    var k = try lay.Kernels.load(r.gpu.d);
    defer k.unload();
    var worst: f64 = 0;
    const cases = o.get("cases").?.object;
    var it = cases.iterator();
    while (it.next()) |kv| {
        const t0 = check.now(r.gpu.io);
        const tag = kv.key_ptr.*;
        const e = kv.value_ptr.object;
        try window(r, &k, dir, c, &w, tag, e, lid);
        const q = try lref.layerF64(r.a, r.gpu.io, dir, c, &host, tag, e, lid);
        worst = @max(worst, q);
        const dt = @as(f64, @floatFromInt(check.now(r.gpu.io) - t0)) / 1e9;
        std.debug.print("RESULT {s} {s}: bit-exact, f64 worst {d:.4} of tolerance ({d:.1} s)\n", .{ lid, tag, q, dt });
    }
    std.debug.print("RESULT {s}: {d} windows bit-exact, f64 worst {d:.4} of tolerance\n", .{ lid, cases.count(), worst });
    check.pass("{s}", .{lid});
}

pub fn linW(r: *Rig, l: lref.Lin) !exl3.Weight {
    const st = exl3.strides(l.k, l.n, l.halves, true);
    return .{
        .trellis = try r.dev(l.words),
        .suh = try r.dev(l.suh),
        .svh = try r.dev(l.svh),
        .k = @intCast(l.k),
        .n = @intCast(l.n),
        .halves = l.halves,
        .codebook = .mul1,
        .stride_k = st.k,
        .stride_nb = st.nb,
    };
}

pub fn b16W(r: *Rig, b: lref.Mat) !cuda.glm_bmm.B16 {
    return .{ .w = try r.dev(b.w), .n = @intCast(b.n), .k = @intCast(b.k) };
}

/// The MoE on the device as the engine holds it; the routed experts' trellises stay in one buffer each.
pub fn moeW(r: *Rig, h: *const lref.Host, c: lay.Config) !lay.MoeW {
    const ex = &h.ex;
    var tables: [3]u64 = undefined;
    for (ex.proj, ex.offs, 0..) |bytes, offs, j| {
        const base = try r.dev(bytes);
        const ptrs = try r.a.alloc(u64, ex.count);
        for (offs, 0..) |off, i| ptrs[i] = base + off;
        tables[j] = try r.dev(std.mem.sliceAsBytes(ptrs));
    }
    const experts: exl3e.Layer = .{
        .gate = tables[0],
        .up = tables[1],
        .down = tables[2],
        .gate_k2 = try r.dev(std.mem.sliceAsBytes(ex.k2[0])),
        .up_k2 = try r.dev(std.mem.sliceAsBytes(ex.k2[1])),
        .down_k2 = try r.dev(std.mem.sliceAsBytes(ex.k2[2])),
        .suh_g = try r.dev(ex.suh_g),
        .suh_u = try r.dev(ex.suh_u),
        .svh_g = try r.dev(ex.svh_g),
        .svh_u = try r.dev(ex.svh_u),
        .suh_d = try r.dev(ex.suh_d),
        .svh_d = try r.dev(ex.svh_d),
        .experts = ex.count,
        .dims = ex.dims,
        .width = ex.width,
        .codebook = .mul1,
        .limit = c.limit,
        .act_mode = 0,
    };
    return .{
        .router = try r.dev(h.router),
        .bias = try r.dev(h.bias),
        .experts = experts,
        .gate = try linW(r, h.sh[0]),
        .up = try linW(r, h.sh[1]),
        .down = try linW(r, h.sh[2]),
    };
}

/// Rank 0's tensors on the device as the engine holds them.
fn weights(r: *Rig, h: *const lref.Host, c: lay.Config) !lay.LayerW {
    const kd = h.kda orelse return error.GlmLayerKind;
    return .{
        .attn_hc = .{ .fn_w = try r.dev(h.attn_hc[0]), .base = try r.dev(h.attn_hc[1]), .scale = try r.dev(h.attn_hc[2]) },
        .ffn_hc = .{ .fn_w = try r.dev(h.ffn_hc[0]), .base = try r.dev(h.ffn_hc[1]), .scale = try r.dev(h.ffn_hc[2]) },
        .in_norm = try r.dev(h.in_norm),
        .post_norm = try r.dev(h.post_norm),
        .mixer = .{ .kda = .{
            .qkv = try linW(r, kd.qkv),
            .fab = try b16W(r, .{ .w = try std.mem.concat(r.a, u8, &.{ kd.fa.w, kd.ga.w, kd.b.w }), .n = kd.fa.n + kd.ga.n + kd.b.n, .k = kd.fa.k }),
            .fb = try b16W(r, kd.fb),
            .gb = try b16W(r, kd.gb),
            .o = try linW(r, kd.o),
            .conv = try r.dev(kd.conv),
            .a_log = try r.dev(kd.a_log),
            .dt_bias = try r.dev(kd.dt_bias),
            .norm = try r.dev(kd.norm),
            .heads = kd.heads,
            .fa_off = kd.fa_off,
            .ga_off = kd.ga_off,
            .b_off = kd.b_off,
            .p_stride = kd.p_stride,
        } },
        .ffn = if (h.dense) .{ .mlp = .{ .gate = try linW(r, h.sh[0]), .up = try linW(r, h.sh[1]), .down = try linW(r, h.sh[2]) } } else .{ .moe = try moeW(r, h, c) },
    };
}

pub fn same(r: *Rig, dir: []const u8, label: []const u8, ptr: u64, name: []const u8) !void {
    const want = try file(r, dir, name);
    try check.sameBytes(label, try r.back(ptr, want.len), want);
}

pub const Stage = struct { []const u8, u64 };

/// The test's all-gather: own part in slot 0, the fixture's rank-1 partial in slot 1; then the site's stage bits.
pub const Ctx = struct {
    r: *Rig,
    id: []const u8,
    dir: []const u8,
    tag: []const u8,
    e: Obj,
    sc: *const lay.Scratch,
    attn: []const Stage, // the mixer's stages
    x: u64,
    d: usize,

    pub fn gather(p: *anyopaque, s: cuda.Stream, site: lay.Site, part: u64, gath: u64, rows: usize) anyerror!void {
        const c: *Ctx = @ptrCast(@alignCast(p));
        try s.synchronize();
        const n = rows * c.d * 4;
        const so = c.e.get(if (site == .attn) "s0" else "s1").?.object;
        const g: cuda.DeviceBuffer = .{ .d = c.r.gpu.d, .ptr = gath, .len = 2 * n, .borrowed = true };
        try g.copyFrom(0, part, n, s.handle);
        try g.upload(n, try file(c.r, c.dir, ref.str(so, "peer")));
        const sc = c.sc;
        // s0 holds no x (the layer's input); s1's is the mixer's hc_post
        const hc = [_]Stage{ .{ "x", c.x }, .{ "normed", sc.normed }, .{ "post", sc.post }, .{ "comb", sc.comb } };
        const moe = [_]Stage{
            .{ "mlog", sc.mlog }, .{ "pick", sc.pick }, .{ "wts", sc.wts },   .{ "ey", sc.ey },
            .{ "smg", sc.smg },   .{ "smu", sc.smu },   .{ "sact", sc.sact }, .{ "sy", sc.sy },
        };
        const tail = [_]Stage{.{ "part", part }};
        for ([_][]const Stage{ &hc, if (site == .attn) c.attn else &moe, &tail }) |list| for (list) |st| {
            const name = (so.get(st[0]) orelse continue).string;
            const label = try std.fmt.allocPrint(c.r.a, "{s} {s} {s} {s}", .{ c.id, c.tag, @tagName(site), st[0] });
            try same(c.r, c.dir, label, st[1], name);
        };
    }
};

fn sameAt(r: *Rig, dir: []const u8, lid: []const u8, tag: []const u8, what: []const u8, ptr: u64, name: []const u8) !void {
    try same(r, dir, try std.fmt.allocPrint(r.a, "{s} {s} {s}", .{ lid, tag, what }), ptr, name);
}

/// One window from the fixture's state: forward (+ commit for decode), every stage's bits, x and the KDA state.
fn window(r: *Rig, k: *lay.Kernels, dir: []const u8, c: lay.Config, w: *const lay.LayerW, tag: []const u8, e: Obj, lid: []const u8) !void {
    const rows: usize = @intCast(ref.int(e, "rows"));
    const prefill = e.get("prefill").?.bool;
    var sc = try lay.Scratch.init(r.gpu.d, r.s, c, w, rows, 2);
    defer sc.deinit();
    const x = try up(r, dir, ref.str(e, "x0"));
    const rec0 = try file(r, dir, ref.str(e, "rec0"));
    const st: lay.KdaState = .{ .conv = try up(r, dir, ref.str(e, "conv0")), .rec_in = try r.dev(rec0), .rec_out = try r.zeros(rec0.len) };
    const attn = [_]Stage{
        .{ "p", sc.p },      .{ "ka", sc.ka },    .{ "kg", sc.kg },    .{ "kout", sc.kout },
        .{ "sk", sc.kda.k }, .{ "sv", sc.kda.v }, .{ "sg", sc.kda.g }, .{ "sb", sc.kda.b },
    };
    var ctx: Ctx = .{ .r = r, .id = lid, .dir = dir, .tag = tag, .e = e, .sc = &sc, .attn = &attn, .x = x, .d = c.d };
    const g: lay.Gather = .{ .ctx = &ctx, .world = 2, .run = Ctx.gather };
    const mode: lay.Mode = if (prefill) .{ .prompt = @intCast(ref.int(e, "pos")) } else .decode;
    try lay.forward(k, &r.s, c, w, .{ .kda = .{ .st = st, .mode = mode } }, &sc, x, rows, g);
    const kw = &w.mixer.kda;
    if (!prefill) try lay.commit(k, &r.s, kw, st, &sc, rows, rows);
    try sameAt(r, dir, lid, tag, "x", x, ref.str(e, "x"));
    try sameAt(r, dir, lid, tag, "rec", st.rec_out, ref.str(e, "rec"));
    try sameAt(r, dir, lid, tag, "conv", st.conv, ref.str(e, "conv"));
    // the oracle ran the chunk row-split too (HC_SPLIT=1, PREFILL_OVERLAP=2): own rows and both states equal unsplit
    if (prefill) try check.expect(if (e.get("split_equal")) |v| v.bool else false, "{s} {s}: no split check", .{ lid, tag });
    if (e.get("keep")) |kv| {
        const ko = kv.object;
        const kst: lay.KdaState = .{ .conv = try up(r, dir, ref.str(e, "conv0")), .rec_in = st.rec_in, .rec_out = try r.zeros(rec0.len) };
        try lay.commit(k, &r.s, kw, kst, &sc, rows, @intCast(ref.int(ko, "keep")));
        try sameAt(r, dir, lid, tag, "keep rec", kst.rec_out, ref.str(ko, "rec"));
        try sameAt(r, dir, lid, tag, "keep conv", kst.conv, ref.str(ko, "conv"));
    }
}
