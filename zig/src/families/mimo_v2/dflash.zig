//! MiMo-V2.6's DFlash drafter, step for step as the Python family's dflash.py; every rank runs it whole.

const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const load_ = @import("load.zig");
const fwd = @import("forward.zig");
const exl3 = cuda.exl3;
const aot = cuda.aot;
const Stream = cuda.Stream;

const p = aot.ptr;
fn ci(name: []const u8, v: usize) aot.Const {
    return aot.ci(name, @intCast(v));
}
fn u(x: usize) u32 {
    return @intCast(x);
}

const DConfig = struct {
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize = 128,
    partial_rotary_factor: f64 = 1.0,
    block_size: usize = 8,
    sliding_window: usize = 2048,
    rms_norm_eps: f64 = 1e-6,
    rope_theta: f64 = 10000.0,
    is_causal: bool = false,
    dflash_config: struct { target_layer_ids: []usize, mask_token_id: usize, tap_shift: usize = 1, block_size: ?usize = null },
};

const DLayer = struct {
    in_norm: u64,
    post_norm: u64,
    q_norm: u64,
    k_norm: u64,
    sinks: u64, // 0: none
    q: exl3.Linear,
    k: exl3.Linear,
    v: exl3.Linear,
    o: exl3.Linear,
    gate: exl3.Linear,
    up: exl3.Linear,
    down: exl3.Linear,
};

pub const Drafter = struct {
    m: *fwd.Model,
    d: *const cuda.Driver,
    module: cuda.Module,
    f: struct { embed: cuda.Function, tap: cuda.Function, norm_rope: cuda.Function, store: cuda.Function, attn: cuda.Function, take: cuda.Function, argmax: cuda.Function },
    arena: load_.Arena,
    gpa: std.mem.Allocator,
    hidden: usize,
    inter: usize,
    heads: usize,
    kv: usize,
    rot: usize,
    block: usize,
    window: usize,
    eps: f32,
    taps: []usize, // target layers whose residual feeds the context, in order
    ring: usize,
    scale: f32,
    fc: exl3.Linear,
    fc_norm: u64,
    layers: []DLayer,
    norm: u64,
    mask_emb: u64,
    kc: []u64,
    vc: []u64,
    cos: u64,
    sin: u64,
    streams: usize, // most streams a draft takes (the target's slots)
    // scratch
    tapbuf: u64, // f16 [R, NT * D]
    ctx32: u64, // f32 [R, D]
    ctx: u64, // f16 [R, D]
    kb: u64, // f16 [max(R, SB), KV * 128]
    vb: u64,
    xh: u64,
    z: u64,
    x32: u64, // f32 [SB, D]
    h: u64, // f16 [SB, D]
    qb: u64, // f16 [SB, H * 128]
    q32: u64, // f32 [SB, H * 128]
    a: u64, // f16 [SB, H * 128]
    o32: u64, // f32 [SB, D]
    g: u64, // f16 [SB, I]
    up: u64,
    mm: u64,
    tk: u64, // f16 [S(B-1), D]
    head: u64, // f32 [S(B-1), Vr]
    recv: u64, // f32 [world, S(B-1), Vr]
    am: u64, // i32 [S(B-1)]
    in_tok: u64, // i64 [S]
    in_pos0: u64, // i32 [S]
    in_slot: u64, // i32 [S]
    rpos: u64, // i32 [SB]
    rslot: u64, // i32 [SB]
    drafts: usize = 0,
    graphs: [17]?cuda.graph.Exec = @splat(null), // DraftRunner: one graph per stream count

    /// Drafter(model, dir): `dir` the pack's dflash/ (config.json + model.safetensors, mul1 EXL3).
    pub fn init(gpa: std.mem.Allocator, io: std.Io, m: *fwd.Model, dir: []const u8, s: Stream) !*Drafter {
        const d = m.d;
        var path_buf: [std.fs.max_path_bytes]u8 = undefined;
        const cfg_path = try std.fmt.bufPrint(&path_buf, "{s}/config.json", .{dir});
        const text = try std.Io.Dir.cwd().readFileAlloc(io, cfg_path, gpa, .limited(1 << 20));
        defer gpa.free(text);
        const parsed = try std.json.parseFromSlice(DConfig, gpa, text, .{ .ignore_unknown_fields = true });
        defer parsed.deinit();
        const c = parsed.value;
        if (c.is_causal) return error.CausalDrafter;
        if (c.hidden_size != m.w.cfg.hidden) return error.HiddenMismatch;
        if (c.head_dim != 128) return error.HeadDim;
        if (c.num_attention_heads / c.num_key_value_heads > 8) return error.GroupWiderThan8; // dr_attn: a warp per head

        const dr = try gpa.create(Drafter);
        errdefer gpa.destroy(dr);
        dr.* = undefined;
        dr.m = m;
        dr.d = d;
        dr.gpa = gpa;
        dr.drafts = 0;
        dr.graphs = @splat(null);
        dr.hidden = c.hidden_size;
        dr.inter = c.intermediate_size;
        dr.heads = c.num_attention_heads;
        dr.kv = c.num_key_value_heads;
        dr.rot = @intFromFloat(@as(f64, @floatFromInt(c.head_dim)) * c.partial_rotary_factor);
        dr.block = c.dflash_config.block_size orelse c.block_size;
        dr.window = c.sliding_window;
        dr.eps = @floatCast(c.rms_norm_eps);
        dr.taps = try gpa.alloc(usize, c.dflash_config.target_layer_ids.len);
        errdefer gpa.free(dr.taps);
        for (c.dflash_config.target_layer_ids, 0..) |t, i| {
            dr.taps[i] = t + c.dflash_config.tap_shift;
            if (dr.taps[i] >= m.w.layers.len) return error.TapPastLastLayer;
            if (i > 0 and dr.taps[i] <= dr.taps[i - 1]) return error.TapsOutOfOrder;
        }
        dr.ring = m.ring;
        if (dr.ring < dr.window + 2 * dr.block) return error.RingShorterThanWindow;
        dr.scale = @floatCast(std.math.pow(f64, @floatFromInt(c.head_dim), -0.5));
        dr.streams = m.slots;
        dr.module = try cuda.Module.load(d, cuda.kernels.mimo_dflash);
        errdefer dr.module.unload();
        dr.f = .{
            .embed = try dr.module.function("dr_block_embed"),
            .tap = try dr.module.function("dr_tap"),
            .norm_rope = try dr.module.function("dr_norm_rope"),
            .store = try dr.module.function("dr_store"),
            .attn = try dr.module.function("dr_attn"),
            .take = try dr.module.function("dr_take"),
            .argmax = try dr.module.function("dr_argmax"),
        };
        dr.arena = .{ .d = d, .gpa = gpa };
        errdefer dr.arena.deinit();
        const ar = &dr.arena;

        var ck = try core.Checkpoint.openModel(gpa, io, dir);
        defer ck.close();
        dr.fc = try load_.linear(gpa, ar, try load_.readMat(gpa, &ck, "fc"));
        dr.fc_norm = try f32Upload(gpa, ar, try ck.get("hidden_norm.weight"));
        dr.norm = try f32Upload(gpa, ar, try ck.get("norm.weight"));
        dr.mask_emb = try f32Upload(gpa, ar, try ck.get("mask_embedding"));
        dr.layers = try gpa.alloc(DLayer, c.num_hidden_layers);
        errdefer gpa.free(dr.layers);
        var nb: [256]u8 = undefined;
        for (dr.layers, 0..) |*L, i| {
            const P = struct {
                fn name(buf: []u8, li: usize, rest: []const u8) ![]const u8 {
                    return std.fmt.bufPrint(buf, "layers.{d}.{s}", .{ li, rest });
                }
            };
            L.in_norm = try f32Upload(gpa, ar, try ck.get(try P.name(&nb, i, "input_layernorm.weight")));
            L.post_norm = try f32Upload(gpa, ar, try ck.get(try P.name(&nb, i, "post_attention_layernorm.weight")));
            L.q_norm = try f32Upload(gpa, ar, try ck.get(try P.name(&nb, i, "self_attn.q_norm.weight")));
            L.k_norm = try f32Upload(gpa, ar, try ck.get(try P.name(&nb, i, "self_attn.k_norm.weight")));
            L.sinks = if (ck.get(try P.name(&nb, i, "self_attn.attention_sink_bias"))) |t| try f32Upload(gpa, ar, t) else |_| 0;
            inline for (.{ .{ "q", "self_attn.q_proj" }, .{ "k", "self_attn.k_proj" }, .{ "v", "self_attn.v_proj" }, .{ "o", "self_attn.o_proj" }, .{ "gate", "mlp.gate_proj" }, .{ "up", "mlp.up_proj" }, .{ "down", "mlp.down_proj" } }) |f|
                @field(L, f[0]) = try load_.linear(gpa, ar, try load_.readMat(gpa, &ck, try P.name(&nb, i, f[1])));
        }
        // caches [slots, ring, KV, 128] fp16, zeroed; rope tables [context + block, rot / 2]
        dr.kc = try gpa.alloc(u64, dr.layers.len);
        dr.vc = try gpa.alloc(u64, dr.layers.len);
        for (0..dr.layers.len) |i| {
            dr.kc[i] = try ar.zeros(m.slots * dr.ring * dr.kv * 128 * 2);
            dr.vc[i] = try ar.zeros(m.slots * dr.ring * dr.kv * 128 * 2);
        }
        const tlen = m.context + dr.block;
        dr.cos = try ar.alloc(tlen * (dr.rot / 2) * 4);
        dr.sin = try ar.alloc(tlen * (dr.rot / 2) * 4);
        {
            const rope = try m.glue.function("mimo_rope_tables");
            var a: cuda.Args = .{};
            a.add(dr.cos);
            a.add(dr.sin);
            a.add(@as(c_int, @intCast(tlen)));
            a.add(@as(f64, c.rope_theta));
            a.add(@as(c_int, @intCast(dr.rot)));
            try cuda.launch.launch(rope, .{ .grid = .{ .x = 256, .y = 1, .z = 1 }, .block = .{ .x = 256, .y = 1, .z = 1 } }, s, &a);
        }
        // scratch
        const R = m.rows;
        const SB = dr.streams * dr.block;
        const RR = @max(R, SB);
        const S1 = dr.streams * (dr.block - 1);
        const D = dr.hidden;
        const NT = dr.taps.len;
        const vr = m.w.head.n;
        var max_z: usize = zBytes(dr.fc, R);
        for (dr.layers) |L| inline for (.{ "q", "k", "v", "o", "gate", "up", "down" }) |f| {
            max_z = @max(max_z, zBytes(@field(L, f), RR));
        };
        max_z = @max(max_z, zBytes(m.w.head, S1));
        dr.tapbuf = try ar.zeros(R * NT * D * 2);
        dr.ctx32 = try ar.zeros(R * D * 4);
        dr.ctx = try ar.zeros(R * D * 2);
        dr.kb = try ar.zeros(RR * dr.kv * 128 * 2);
        dr.vb = try ar.zeros(RR * dr.kv * 128 * 2);
        dr.xh = try ar.zeros(RR * @max(NT * D, @max(dr.inter, dr.heads * 128)) * 2);
        dr.z = try ar.zeros(@max(max_z, 4));
        dr.x32 = try ar.zeros(SB * D * 4);
        dr.h = try ar.zeros(SB * D * 2);
        dr.qb = try ar.zeros(SB * dr.heads * 128 * 2);
        dr.q32 = try ar.zeros(SB * dr.heads * 128 * 4);
        dr.a = try ar.zeros(SB * dr.heads * 128 * 2);
        dr.o32 = try ar.zeros(SB * D * 4);
        dr.g = try ar.zeros(SB * dr.inter * 2);
        dr.up = try ar.zeros(SB * dr.inter * 2);
        dr.mm = try ar.zeros(SB * dr.inter * 2);
        dr.tk = try ar.zeros(S1 * D * 2);
        dr.head = try ar.zeros(S1 * vr * 4);
        dr.recv = try ar.zeros(m.w.world * S1 * vr * 4);
        dr.am = try ar.zeros(S1 * 4);
        dr.in_tok = try ar.zeros(dr.streams * 8);
        dr.in_pos0 = try ar.zeros(dr.streams * 4);
        dr.in_slot = try ar.zeros(dr.streams * 4);
        dr.rpos = try ar.zeros(SB * 4);
        dr.rslot = try ar.zeros(SB * 4);
        try s.synchronize();
        return dr;
    }

    pub fn deinit(dr: *Drafter) void {
        const gpa = dr.gpa;
        for (&dr.graphs) |*g| if (g.*) |*e| {
            e.deinit();
            g.* = null;
        };
        gpa.free(dr.taps);
        gpa.free(dr.layers);
        gpa.free(dr.kc);
        gpa.free(dr.vc);
        dr.module.unload();
        dr.arena.deinit();
        gpa.destroy(dr);
    }

    fn zBytes(l: exl3.Linear, R: usize) usize {
        return if (l.split[0] > 1) l.split[0] * R * l.n * 4 else 0;
    }

    fn proj(dr: *Drafter, s: Stream, l: exl3.Linear, x: u64, out: u64, out_f32: bool, n: usize) !void {
        try dr.m.k3.linear(s, l, x, .f16, dr.xh, out, if (out_f32) .f32 else .f16, if (l.split[0] > 1) dr.z else 0, n);
    }

    fn rms(dr: *Drafter, s: Stream, x: u64, w: u64, out: u64, n: usize) !void {
        const D = dr.hidden;
        try dr.m.set.run(s, "_rms", .{ u(n), 1, 1 }, &.{ p("X", "*fp32", x), p("W", "*fp32", w), p("OUT", "*fp16", out) }, &.{ ci("N", D), aot.cf("EPS", dr.eps), ci("BLOCK", std.math.ceilPowerOfTwo(usize, D) catch unreachable) });
    }

    fn launch(f: cuda.Function, s: Stream, gx: usize, gy: usize, threads: usize, shared: usize, a: *cuda.Args) !void {
        try cuda.launch.launch(f, .{ .grid = .{ .x = u(gx), .y = u(gy), .z = 1 }, .block = .{ .x = u(threads), .y = 1, .z = 1 }, .shared = u(shared) }, s, a);
    }

    /// The target forward's tap: residual rows after target layer `layer` (when tapped) into the tap buffer.
    pub fn tap(dr: *Drafter, s: Stream, layer: usize, res: u64, n: usize) !void {
        return dr.tapInto(s, layer, res, n, dr.tapbuf);
    }

    /// `tap` into another buffer of the same row layout (a prompt chunk's last TOKENS_PER_UPDATE rows).
    pub fn tapInto(dr: *Drafter, s: Stream, layer: usize, res: u64, n: usize, dst: u64) !void {
        const j = std.mem.indexOfScalar(usize, dr.taps, layer) orelse return;
        var a: cuda.Args = .{};
        a.add(res);
        a.add(dst);
        a.add(@as(c_int, @intCast(dr.hidden)));
        a.add(@as(c_int, @intCast(dr.taps.len)));
        a.add(@as(c_int, @intCast(j)));
        try launch(dr.f.tap, s, n, 1, 256, 0, &a);
    }

    /// Drafter.update: the target's `n` rows (positions / slots staged in the target's buffers) into the caches.
    pub fn update(dr: *Drafter, s: Stream, n: usize) !void {
        return dr.updateFrom(s, dr.tapbuf, dr.m.b.pos, dr.m.b.slot, n);
    }

    /// `update` from given taps, positions and slots (n <= the decode rows).
    pub fn updateFrom(dr: *Drafter, s: Stream, taps: u64, pos: u64, slot: u64, n: usize) !void {
        try dr.proj(s, dr.fc, taps, dr.ctx32, true, n);
        try dr.rms(s, dr.ctx32, dr.fc_norm, dr.ctx, n);
        for (dr.layers, 0..) |L, i| {
            try dr.proj(s, L.k, dr.ctx, dr.kb, false, n);
            try dr.normRope(s, dr.kb, L.k_norm, pos, slot, 0, dr.kc[i], dr.kv, n);
            try dr.proj(s, L.v, dr.ctx, dr.vb, false, n);
            try dr.store(s, dr.vb, pos, slot, dr.vc[i], n);
        }
    }

    fn normRope(dr: *Drafter, s: Stream, in: u64, w: u64, pos: u64, slot: u64, out32: u64, cache: u64, nh: usize, n: usize) !void {
        var a: cuda.Args = .{};
        a.add(in);
        a.add(w);
        a.add(dr.cos);
        a.add(dr.sin);
        a.add(pos);
        a.add(slot);
        a.add(out32);
        a.add(cache);
        a.add(@as(c_int, @intCast(nh)));
        a.add(@as(c_int, @intCast(dr.rot)));
        a.add(@as(c_int, @intCast(dr.ring)));
        a.add(dr.eps);
        try launch(dr.f.norm_rope, s, n, nh, 128, 0, &a);
    }

    fn store(dr: *Drafter, s: Stream, in: u64, pos: u64, slot: u64, cache: u64, n: usize) !void {
        var a: cuda.Args = .{};
        a.add(in);
        a.add(pos);
        a.add(slot);
        a.add(cache);
        a.add(@as(c_int, @intCast(dr.kv)));
        a.add(@as(c_int, @intCast(dr.ring)));
        try launch(dr.f.store, s, n, 1, 256, 0, &a);
    }

    /// Stage a draft: each stream's pending token, its position and slot (DraftRunner's fixed inputs).
    pub fn stage(dr: *Drafter, tokens: []const i64, pos0: []const u32, slots: []const u32) !void {
        const S = tokens.len;
        if (S == 0 or S > dr.streams) return error.TooManyStreams;
        var pz: [16]i32 = undefined;
        var sz: [16]i32 = undefined;
        var rp: [16 * 16]i32 = undefined;
        var rs: [16 * 16]i32 = undefined;
        for (0..S) |i| {
            if (pos0[i] + dr.block > dr.m.context + dr.block) return error.PastContext;
            pz[i] = @intCast(pos0[i]);
            sz[i] = @intCast(slots[i]);
            for (0..dr.block) |b| {
                rp[i * dr.block + b] = @intCast(pos0[i] + b);
                rs[i * dr.block + b] = @intCast(slots[i]);
            }
        }
        const d = dr.d;
        try d.check(d.api.cuMemcpyHtoD_v2(dr.in_tok, std.mem.sliceAsBytes(tokens).ptr, S * 8), "draft in");
        try d.check(d.api.cuMemcpyHtoD_v2(dr.in_pos0, @ptrCast(&pz), S * 4), "draft in");
        try d.check(d.api.cuMemcpyHtoD_v2(dr.in_slot, @ptrCast(&sz), S * 4), "draft in");
        try d.check(d.api.cuMemcpyHtoD_v2(dr.rpos, @ptrCast(&rp), S * dr.block * 4), "draft in");
        try d.check(d.api.cuMemcpyHtoD_v2(dr.rslot, @ptrCast(&rs), S * dr.block * 4), "draft in");
    }

    /// Drafter.draft on staged inputs for S streams: drafts [S, block - 1] (int32) left in `am`.
    pub fn run(dr: *Drafter, s: Stream, S: usize) !void {
        const m = dr.m;
        const B = dr.block;
        const n = S * B;
        const D = dr.hidden;
        const H = dr.heads;
        {
            var a: cuda.Args = .{};
            a.add(m.w.embed);
            a.add(dr.mask_emb);
            a.add(dr.in_tok);
            a.add(dr.x32);
            a.add(@as(c_int, @intCast(B)));
            a.add(@as(c_int, @intCast(D)));
            try launch(dr.f.embed, s, n, 1, 256, 0, &a);
        }
        for (dr.layers, 0..) |L, i| {
            try dr.rms(s, dr.x32, L.in_norm, dr.h, n);
            try dr.proj(s, L.q, dr.h, dr.qb, false, n);
            try dr.normRope(s, dr.qb, L.q_norm, dr.rpos, dr.rslot, dr.q32, 0, H, n);
            try dr.proj(s, L.k, dr.h, dr.kb, false, n);
            try dr.normRope(s, dr.kb, L.k_norm, dr.rpos, dr.rslot, 0, dr.kc[i], dr.kv, n);
            try dr.proj(s, L.v, dr.h, dr.vb, false, n);
            try dr.store(s, dr.vb, dr.rpos, dr.rslot, dr.vc[i], n);
            {
                var a: cuda.Args = .{};
                a.add(dr.q32);
                a.add(dr.kc[i]);
                a.add(dr.vc[i]);
                a.add(L.sinks);
                a.add(dr.in_pos0);
                a.add(dr.in_slot);
                a.add(dr.a);
                a.add(@as(c_int, @intCast(B)));
                a.add(@as(c_int, @intCast(dr.window)));
                a.add(@as(c_int, @intCast(dr.ring)));
                a.add(@as(c_int, @intCast(H)));
                a.add(@as(c_int, @intCast(dr.kv)));
                a.add(dr.scale);
                try launch(dr.f.attn, s, n, dr.kv, 256, 0, &a); // a block per (row, KV head), a warp per head of the group
            }
            try dr.proj(s, L.o, dr.a, dr.o32, true, n);
            try m.add2(s, dr.x32, dr.o32, dr.x32, n * D);
            try dr.rms(s, dr.x32, L.post_norm, dr.h, n);
            try dr.proj(s, L.gate, dr.h, dr.g, false, n);
            try dr.proj(s, L.up, dr.h, dr.up, false, n);
            try m.set.run(s, "_silu_mul", .{ u(n), u(dr.inter / 1024), 1 }, &.{ p("G", "*fp16", dr.g), p("U", "*fp16", dr.up), p("OUT", "*fp16", dr.mm) }, &.{ ci("N", dr.inter), ci("BLOCK", 1024) });
            try dr.proj(s, L.down, dr.mm, dr.o32, true, n);
            try m.add2(s, dr.x32, dr.o32, dr.x32, n * D);
        }
        try dr.rms(s, dr.x32, dr.norm, dr.h, n);
        const n1 = S * (B - 1);
        {
            var a: cuda.Args = .{};
            a.add(dr.h);
            a.add(dr.tk);
            a.add(@as(c_int, @intCast(B)));
            a.add(@as(c_int, @intCast(D)));
            try launch(dr.f.take, s, n1, 1, 256, 0, &a);
        }
        // model._head: this rank's vocab slice, gathered; argmax over [:vocab]
        try dr.m.k3.linear(s, m.w.head, dr.tk, .f16, dr.xh, dr.head, .f32, if (m.w.head.split[0] > 1) dr.z else 0, n1);
        const vr = m.w.head.n;
        const src = if (m.comm) |c| blk: {
            try c.allGather(dr.head, dr.recv, n1 * vr, .f32, s.handle);
            break :blk dr.recv;
        } else dr.head;
        var a: cuda.Args = .{};
        a.add(src);
        a.add(dr.am);
        a.add(@as(c_int, @intCast(m.w.world)));
        a.add(@as(c_int, @intCast(n1)));
        a.add(@as(c_int, @intCast(vr)));
        a.add(@as(c_int, @intCast(m.w.full.vocab)));
        try launch(dr.f.argmax, s, n1, 1, 256, 0, &a);
    }

    /// DraftRunner.__call__ (eager): drafts [S, block - 1] into `out` (row-major).
    pub fn draft(dr: *Drafter, s: Stream, tokens: []const i64, pos0: []const u32, slots: []const u32, out: []i32) !void {
        try dr.stage(tokens, pos0, slots);
        const S = tokens.len;
        if (S >= dr.graphs.len) {
            try dr.run(s, S);
        } else if (dr.graphs[S]) |e| {
            try e.launchOn(s);
        } else {
            try dr.run(s, S); // warm-up (as DraftRunner); writes the same cache rows the replay will
            try s.synchronize();
            try cuda.graph.beginCapture(s, .thread_local);
            dr.run(s, S) catch |err| {
                if (cuda.graph.endCapture(s)) |g| {
                    var gg = g;
                    gg.deinit();
                } else |_| {}
                return err;
            };
            var g = try cuda.graph.endCapture(s);
            defer g.deinit();
            dr.graphs[S] = try g.instantiate();
            try dr.graphs[S].?.launchOn(s);
        }
        try s.synchronize();
        const n1 = tokens.len * (dr.block - 1);
        try dr.d.check(dr.d.api.cuMemcpyDtoH_v2(@ptrCast(out.ptr), dr.am, n1 * 4), "drafts");
        dr.drafts += 1;
    }
};

fn f32Upload(gpa: std.mem.Allocator, ar: *load_.Arena, t: core.safetensors.Tensor) !u64 {
    const f = try load_.asF32(gpa, t);
    defer gpa.free(f);
    return ar.upload(std.mem.sliceAsBytes(f));
}
