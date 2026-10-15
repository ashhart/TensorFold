//! The Ling loader against its checkpoint: each projection's y = W x vs float64 from the file, small tensors as bytes.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const ling = @import("ling_cuda");
const check = @import("check.zig");
const qr = @import("quant_ref_tests.zig");

const Rig = qr.RigT;
const h = qr.helpers;
const W = ling.weights;
const Config = ling.config.Config;
const Quant = ling.config.Quant;
const Role = ling.names.Role;

/// The kernels a check runs: lane matmul and prompt GEMM, the packers, both grouped expert formats and the plan.
pub const Kit = struct {
    lane: cuda.qmmf.Lane,
    gemm: cuda.nvfp4.Prompt,
    pack: cuda.modelopt.Packer,
    x8: cuda.experts.Fp8Experts,
    x4: cuda.experts.Nvfp4Experts,
    mod: cuda.Module,
    router: cuda.grouped.Router,

    pub fn init(k: *Kit, gpu: check.Gpu) !void {
        const sms: usize = @intCast(try gpu.ctx.attribute(.multiprocessor_count));
        k.lane = try cuda.qmmf.Lane.load(gpu.d, (try gpu.ctx.capability()) / 10);
        k.gemm = try cuda.nvfp4.Prompt.load(gpu.d);
        k.pack = try cuda.modelopt.Packer.load(gpu.d);
        k.x8 = try cuda.experts.Fp8Experts.load(gpu.d, sms);
        k.x4 = try cuda.experts.Nvfp4Experts.load(gpu.d, sms);
        k.mod = try cuda.Module.load(gpu.d, cuda.kernels.experts);
        k.router = try cuda.grouped.Router.resolve(k.mod);
    }

    pub fn deinit(k: *Kit) void {
        k.lane.unload();
        k.gemm.unload();
        k.pack.unload();
        k.x8.unload();
        k.x4.unload();
        k.mod.unload();
    }

    pub fn linear(k: *const Kit) cuda.qlinear.Linear {
        return .{ .lane = &k.lane, .nvfp4_prompt = &k.gemm };
    }

    pub fn grouped(k: *const Kit) cuda.experts.Grouped {
        return .{ .fp8g = &k.x8, .nvfp4 = &k.x4 };
    }
};

/// A linear as its file stores it, read back to float64 by (row, col): the host's dequantisation.
pub const Mat = struct {
    f: Quant,
    n: usize,
    k: usize,
    w: []const u8,
    s: []const u8 = &.{},
    global: f64 = 1,

    pub fn read(ck: *core.Checkpoint, c: *const Config, module: []const u8, n: usize, k: usize) !Mat {
        var b: [256]u8 = undefined;
        const f = c.format(module);
        const w = try ck.get(try std.fmt.bufPrint(&b, "{s}.weight", .{module}));
        return switch (f) {
            .bf16 => .{ .f = f, .n = n, .k = k, .w = w.bytes },
            .fp8_block128 => .{ .f = f, .n = n, .k = k, .w = w.bytes, .s = (try ck.get(try std.fmt.bufPrint(&b, "{s}.weight_scale_inv", .{module}))).bytes },
            .nvfp4 => blk: {
                const l = try core.nvfp4.Linear.read(ck, module);
                break :blk .{ .f = f, .n = n, .k = k, .w = l.codes, .s = l.scales, .global = l.global };
            },
        };
    }

    pub fn at(m: Mat, row: usize, col: usize) f64 {
        return switch (m.f) {
            .bf16 => h.bfOf(std.mem.readInt(u16, m.w[2 * (row * m.k + col) ..][0..2], .little)),
            .fp8_block128 => h.e4m3Of(m.w[row * m.k + col]) * @as(f64, @as(f32, @bitCast(std.mem.readInt(u32, m.s[4 * ((row / 128) * (m.k / 128) + col / 128) ..][0..4], .little)))),
            .nvfp4 => blk: {
                const byte = m.w[row * (m.k / 2) + col / 2];
                break :blk h.e2m1Of(if (col & 1 == 0) byte & 0xF else byte >> 4) * h.e4m3Of(m.s[row * (m.k / 16) + col / 16]) * m.global;
            },
        };
    }
};

/// y = x W^T in float64 for `rows` bf16 rows, and the bound: 2^-7 of |y| (bf16 out) and 2^-12 of sum |x w| (fp32 sums).
pub fn reference(a: std.mem.Allocator, m: Mat, x: []const u16, rows: usize) !struct { want: []f64, tol: []f64 } {
    const want = try a.alloc(f64, rows * m.n);
    const tol = try a.alloc(f64, rows * m.n);
    @memset(want, 0);
    @memset(tol, 0);
    for (0..m.n) |j| for (0..m.k) |i| {
        const v = m.at(j, i);
        for (0..rows) |r| {
            const t = h.bfOf(x[r * m.k + i]) * v;
            want[r * m.n + j] += t;
            tol[r * m.n + j] += @abs(t);
        }
    };
    for (want, tol) |y, *t| t.* = @abs(y) * std.math.pow(f64, 2, -7) + t.* * std.math.pow(f64, 2, -12) + 1e-30;
    return .{ .want = want, .tol = tol };
}

/// The view's tag is the format the config gives the module.
fn tagged(w: W.Proj, f: Quant) bool {
    return switch (w) {
        .bf16 => f == .bf16,
        .fp8g => f == .fp8_block128,
        .nvfp4 => f == .nvfp4,
        else => false,
    };
}

/// One projection: its tag, then decode and prompt rows against the file's float64 product.
pub fn linear(r: *Rig, kit: *const Kit, ck: *core.Checkpoint, c: *const Config, module: []const u8, w: W.Proj, rows: usize) !void {
    const n = w.outputs();
    const k = w.inputs();
    try check.expect(tagged(w, c.format(module)), "{s}: loaded as {s}, the config says {t}", .{ module, @tagName(w), c.format(module) });
    const m = try Mat.read(ck, c, module, n, k);
    const x = try r.bfs(rows * k, 1.0);
    const ref = try reference(r.a, m, x, rows);
    const dx = try r.dev(std.mem.sliceAsBytes(x));
    const part = try r.zeros(8 * rows * n * 4);
    var name: [256]u8 = undefined;
    for ([_]bool{ false, true }) |prompt| {
        const out = try r.zeros(rows * n * 2);
        const in: cuda.qlinear.Rows = .{ .x = dx, .ldx = k, .part = part };
        if (prompt) try kit.linear().prompt(r.s, w, in, out, rows) else try kit.linear().decode(r.s, w, in, out, rows);
        try r.close(try std.fmt.bufPrint(&name, "{s} {t} {s} [{d}, {d}]", .{ module, c.format(module), if (prompt) "prompt" else "decode", n, k }), out, false, ref.want, ref.tol);
    }
}

pub const Item = struct { Role, W.Proj };

fn put(out: []Item, at: *usize, role: Role, p: ?W.Proj) void {
    const x = p orelse return;
    out[at.*] = .{ role, x };
    at.* += 1;
}

/// Every projection a layer holds, by role (routed experts apart).
pub fn projections(l: *const W.Layer, out: []Item) usize {
    var n: usize = 0;
    switch (l.mixer) {
        .kda => |a| for ([_]Role{ .kda_q, .kda_k, .kda_v, .kda_b, .kda_f, .kda_g, .kda_o }, [_]W.Proj{ a.q, a.k, a.v, a.b, a.f, a.g, a.o }) |role, p| put(out, &n, role, p),
        .mla => |a| for ([_]Role{ .mla_q, .mla_q_a, .mla_q_b, .mla_kv_a, .mla_kv_b, .mla_gate, .mla_o }, [_]?W.Proj{ a.q, a.q_a, a.q_b, a.kv_a, a.kv_b, a.gate, a.o }) |role, p| put(out, &n, role, p),
    }
    var roles = [3]Role{ .dense_gate, .dense_up, .dense_down };
    var m: ?W.Mlp = null;
    switch (l.ffn) {
        .dense => |d| m = d,
        .moe => |e| {
            m = e.shared;
            roles = .{ .shared_gate, .shared_up, .shared_down };
        },
    }
    if (m) |x| for (roles, [_]W.Proj{ x.gate, x.up, x.down }) |role, p| put(out, &n, role, p);
    return n;
}

/// The bytes at `ptr` equal the file's tensors (concatenated) as `to`.
fn sameAs(r: *Rig, ck: *core.Checkpoint, ptr: u64, tensors: []const []const u8, to: core.checkpoint.DType) !void {
    var want: std.ArrayList(u8) = .empty;
    for (tensors) |t| try W.proj.append(r.a, &want, try ck.get(t), to);
    try check.sameBytes(tensors[0], try r.back(ptr, want.items.len), want.items);
}

/// A layer's small tensors byte for byte: norms, the KDA convs (fp32 q|k|v), A_log, dt_bias, router and bias.
pub fn smalls(r: *Rig, ck: *core.Checkpoint, c: *const Config, l: *const W.Layer, w: *const W.Weights) !usize {
    var b: [8][192]u8 = undefined;
    const i = l.index;
    const p = "model.layers.";
    var n: usize = 2;
    try sameAs(r, ck, l.attn_norm, &.{try std.fmt.bufPrint(&b[0], p ++ "{d}.input_layernorm.weight", .{i})}, .bf16);
    try sameAs(r, ck, l.mlp_norm, &.{try std.fmt.bufPrint(&b[0], p ++ "{d}.post_attention_layernorm.weight", .{i})}, .bf16);
    switch (l.mixer) {
        .kda => |a| {
            var convs: [3][]const u8 = undefined;
            for (&convs, [_][]const u8{ "q", "k", "v" }, 0..) |*t, x, j| t.* = try std.fmt.bufPrint(&b[j], p ++ "{d}.attention.{s}_conv1d.weight", .{ i, x });
            try sameAs(r, ck, a.conv, &convs, .f32);
            try sameAs(r, ck, a.a_log, &.{try std.fmt.bufPrint(&b[3], p ++ "{d}.attention.A_log", .{i})}, .f32);
            try sameAs(r, ck, a.dt_bias, &.{try std.fmt.bufPrint(&b[3], p ++ "{d}.attention.dt_bias", .{i})}, .f32);
            try sameAs(r, ck, a.o_norm, &.{try std.fmt.bufPrint(&b[3], p ++ "{d}.attention.o_norm.weight", .{i})}, .bf16);
            n += 4;
        },
        .mla => |a| {
            try sameAs(r, ck, a.kv_a_norm, &.{try std.fmt.bufPrint(&b[0], p ++ "{d}.attention.kv_a_layernorm.weight", .{i})}, .bf16);
            n += 1;
            if (c.q_lora > 0) {
                try sameAs(r, ck, a.q_a_norm, &.{try std.fmt.bufPrint(&b[0], p ++ "{d}.attention.q_a_layernorm.weight", .{i})}, .bf16);
                n += 1;
            }
        },
    }
    if (l.ffn == .moe) {
        try sameAs(r, ck, l.ffn.moe.router, &.{try std.fmt.bufPrint(&b[0], p ++ "{d}.mlp.gate.weight", .{i})}, .bf16);
        if (c.expert_bias) try sameAs(r, ck, l.ffn.moe.bias, &.{try std.fmt.bufPrint(&b[0], p ++ "{d}.mlp.gate.expert_bias", .{i})}, .f32);
        n += 2;
    }
    if (i >= c.layers) if (w.mtp) |m| {
        for ([_]u64{ m.enorm, m.hnorm, m.norm }, [_][]const u8{ "enorm", "hnorm", "final_layernorm" }) |ptr, x| try sameAs(r, ck, ptr, &.{try std.fmt.bufPrint(&b[0], p ++ "{d}.{s}.weight", .{ i, x })}, .bf16);
        n += 3;
    };
    return n;
}

/// SwiGLU as the expert kernels round it: gate and up to bf16, silu to bf16, the product to bf16 (limit 0).
pub fn swiglu(g: f64, u: f64) f64 {
    const gv = h.bfOf(h.bf16Of(@floatCast(g)));
    const uv = h.bfOf(h.bf16Of(@floatCast(u)));
    return h.bfOf(h.bf16Of(@floatCast(gv / (1 + @exp(-gv))))) * uv;
}

/// Pair p's expert: row p / slots takes `slots` distinct experts, rotated a row.
pub fn pick(p: usize, slots: usize, count: usize) i32 {
    return @intCast((p % slots + p / slots) % count);
}

/// The grouped experts' gate-up (SwiGLU) and down on their own inputs, against float64 from the file.
pub fn experts(r: *Rig, kit: *const Kit, ck: *core.Checkpoint, c: *const Config, i: u32, l: cuda.experts.Layer, rows: usize) !void {
    const slots = @min(4, l.experts);
    const pairs = rows * slots;
    const ni = l.width;
    const d = l.dims;
    const pk = try r.a.alloc(i32, pairs);
    for (pk, 0..) |*e, p| e.* = pick(p, slots, l.experts);
    const plan: cuda.grouped.Plan = .{ .members = try r.zeros(pairs * 4), .items = try r.zeros(cuda.grouped.maxItems(pairs, l.experts, 16) * 12), .counts = try r.zeros(8), .rank = try r.zeros(4), .hist = try r.zeros(4) };
    try kit.router.route(r.s, try r.dev(std.mem.sliceAsBytes(pk)), pairs, l.experts, 16, plan);
    const items = cuda.grouped.maxItems(pairs, l.experts, 16);
    try check.expect(l.limit == ling.limits(c, i).routed, "layer {d}: routed limit {d}, the config's {d}", .{ i, l.limit, ling.limits(c, i).routed });
    var nolimit = l;
    nolimit.limit = 0;
    const x = try r.bfs(rows * d, 1.0);
    const act = try r.zeros(pairs * ni * 2);
    try kit.grouped().decode(r.s, .up, .{ .x = try r.dev(std.mem.sliceAsBytes(x)), .stride = d, .slots = slots }, nolimit, plan, act, items, -1);
    const xd = try r.bfs(pairs * ni, 1.0);
    const y = try r.zeros(pairs * d * 4);
    try kit.grouped().decode(r.s, .down_f32, .{ .x = try r.dev(std.mem.sliceAsBytes(xd)), .stride = ni }, nolimit, plan, y, items, -1);
    var b: [3][192]u8 = undefined;
    const want = try r.a.alloc(f64, pairs * ni);
    const tol = try r.a.alloc(f64, pairs * ni);
    const dwant = try r.a.alloc(f64, pairs * d);
    const dtol = try r.a.alloc(f64, pairs * d);
    for (0..pairs) |p| {
        const e: u32 = @intCast(pk[p]);
        const g = try Mat.read(ck, c, try ling.names.moduleName(&b[0], i, .expert_gate, e), ni, d);
        const u = try Mat.read(ck, c, try ling.names.moduleName(&b[1], i, .expert_up, e), ni, d);
        const dn = try Mat.read(ck, c, try ling.names.moduleName(&b[2], i, .expert_down, e), d, ni);
        const gr = try reference(r.a, g, x[(p / slots) * d ..][0..d], 1);
        const ur = try reference(r.a, u, x[(p / slots) * d ..][0..d], 1);
        for (0..ni) |j| {
            const g0 = gr.want[j];
            want[p * ni + j] = swiglu(g0, ur.want[j]);
            // the bf16 steps (2^-6 of |a|), then each product's bound through silu(g) u: silu' stays under 1.1
            tol[p * ni + j] = @abs(want[p * ni + j]) * std.math.pow(f64, 2, -6) + 1.1 * @abs(ur.want[j]) * gr.tol[j] + @abs(g0 / (1 + @exp(-g0))) * ur.tol[j] + 1e-30;
        }
        const dr = try reference(r.a, dn, xd[p * ni ..][0..ni], 1);
        for (0..d) |j| {
            dwant[p * d + j] = dr.want[j];
            dtol[p * d + j] = dr.tol[j] - @abs(dr.want[j]) * std.math.pow(f64, 2, -7); // fp32 out: no bf16 step
        }
    }
    var name: [128]u8 = undefined;
    try r.close(try std.fmt.bufPrint(&name, "layer {d} experts {t} gate-up, {d} pairs of {d}", .{ i, l.format, pairs, l.experts }), act, false, want, tol);
    try r.close(try std.fmt.bufPrint(&name, "layer {d} experts {t} down, {d} pairs", .{ i, l.format, pairs }), y, true, dwant, dtol);
}

/// Every loaded layer's projections, experts and small tensors; the globals when loaded.
pub fn layers(r: *Rig, kit: *const Kit, ck: *core.Checkpoint, w: *const W.Weights) !struct { proj: usize, small: usize } {
    const c = &w.config;
    var np: usize = 0;
    var ns: usize = 0;
    var b: [192]u8 = undefined;
    for (w.layers) |*l| {
        var list: [16]Item = undefined;
        for (list[0..projections(l, &list)]) |e| {
            try linear(r, kit, ck, c, try ling.names.moduleName(&b, l.index, e[0], null), e[1], 3);
            np += 1;
        }
        if (l.ffn == .moe) switch (l.ffn.moe.routed) {
            .grouped => |g| {
                try experts(r, kit, ck, c, l.index, g, 2);
                np += 3;
            },
            .lanes => |ls| for (ls[0..@min(ls.len, 3)], 0..) |m, e| {
                for ([_]Role{ .expert_gate, .expert_up, .expert_down }, [_]W.Proj{ m.gate, m.up, m.down }) |role, p| try linear(r, kit, ck, c, try ling.names.moduleName(&b, l.index, role, @intCast(e)), p, 3);
                np += 3;
            },
        };
        if (l.index >= c.layers) if (w.mtp) |m| {
            try linear(r, kit, ck, c, try ling.names.moduleName(&b, l.index, .mtp_eh, null), m.eh, 3);
            np += 1;
        };
        ns += try smalls(r, ck, c, l, w);
    }
    if (w.embed != 0) {
        try sameAs(r, ck, w.embed, &.{"model.word_embeddings.weight"}, .bf16);
        try sameAs(r, ck, w.norm, &.{"model.norm.weight"}, .bf16);
        ns += 2;
        if (w.head) |hd| {
            try linear(r, kit, ck, c, "lm_head", hd, 3);
            np += 1;
        }
    }
    return .{ .proj = np, .small = ns };
}

/// `ling-weights DIR`: every layer of a (synthetic) checkpoint loaded and checked.
pub fn run(gpu: check.Gpu, dir: []const u8) !void {
    var arena = std.heap.ArenaAllocator.init(gpu.gpa);
    defer arena.deinit();
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var prng = std.Random.DefaultPrng.init(0x11a6);
    var r: Rig = .{ .gpu = gpu, .a = arena.allocator(), .s = s, .rng = prng.random() };
    defer {
        for (r.bufs.items) |*x| x.free();
        r.bufs.deinit(gpu.gpa);
    }
    var kit: Kit = undefined;
    try kit.init(gpu);
    defer kit.deinit();
    var why: ling.config.Why = .{};
    var o = W.open(gpu.gpa, gpu.io, dir, &why) catch |e| {
        std.debug.print("{s}: {s}\n", .{ dir, why.text() });
        return e;
    };
    defer o.ck.close();
    const all = try r.a.alloc(u32, o.c.layers + o.c.mtp);
    for (all, 0..) |*i, j| i.* = @intCast(j);
    var w = try W.load(gpu.gpa, gpu.d, s, &kit.pack, &o.ck, o.c, all, .{});
    defer w.deinit();
    try check.expect(o.ck.unused() == 0, "{s}: every tensor read", .{dir});
    const n = try layers(&r, &kit, &o.ck, &w);
    check.pass("ling-weights {t}: {d} layers + MTP, {d} projections within float64 bounds (worst {d:.3}), {d} small tensors byte-equal", .{ o.c.quant, o.c.layers, n.proj, r.worst, n.small });
}
