//! Nemotron-H decode: weights on the device, per-layer state, one token through 52 layers to a greedy token id.

const std = @import("std");
const rt = @import("xpu").rt;
const cfgm = @import("xpu_config.zig");
const ld = @import("xpu").loader;

const spv_basic = @import("xpu").kernels.basic;
const spv_mamba = @import("xpu").kernels.mamba;
const spv_moe = @import("xpu").kernels.moe;
const spv_attn = @import("xpu").kernels.attn;
const spv_attn_dec = @import("xpu").kernels.nem_attn_dec;
const spv_glue = @import("xpu").kernels.glue;

const Buf = rt.Buffer;
const attn_chunk: u32 = 512; // the upstream split-K chunk
const min_chunks: u32 = 64; // split-K scratch holds at least this many chunks; a longer context (cap > 32768) grows it (Model.max_chunks)
const max_ctx: u32 = 1 << 20;
const argmax_parts: u32 = 64;
const top_k: u32 = 6;

/// Rejects checkpoints the kernels do not read (NVFP4 experts, FP8 Mamba projections); only MLX 4-bit affine is wired.
pub fn checkFormat(l: *const ld.Loader) !void {
    var it = l.map.keyIterator();
    while (it.next()) |k| {
        if (std.mem.endsWith(u8, k.*, ".weight_scale_2") or std.mem.endsWith(u8, k.*, ".input_scale")) {
            std.log.err("NVFP4 checkpoints are not supported on the XPU backend yet; use the MLX 4-bit checkpoint (found {s})", .{k.*});
            return error.UnsupportedFormat;
        }
    }
}

pub const Table = struct { w: Buf, s: Buf, b: Buf };
pub const Mamba = struct { in: Table, out: Table, conv_w: Buf, conv_b: Buf, a_log: Buf, d: Buf, dtb: Buf, gnorm: Buf, cstate: Buf, sstate: Buf };
pub const Moe = struct { gate: Buf, bias: Buf, fc1: Table, fc2: Table, shup: Table, shdn: Table };
pub const Attn = struct { q: Table, k: Table, v: Table, o: Table, kc: Buf, vc: Buf };
pub const Layer = struct { norm: Buf, mixer: union(cfgm.Kind) { mamba: Mamba, moe: Moe, attention: Attn } };

const Kernels = struct {
    embed: rt.Kernel,
    embed_tok: rt.Kernel,
    rms: rt.Kernel,
    add: rt.Kernel,
    add_rms: rt.Kernel,
    round: rt.Kernel,
    m_qmv: rt.Kernel,
    m_conv: rt.Kernel,
    m_ssm: rt.Kernel,
    m_norm: rt.Kernel,
    e_logits: rt.Kernel,
    e_route: rt.Kernel,
    e_up: rt.Kernel,
    e_down: rt.Kernel,
    e_rup: rt.Kernel,
    e_rdown: rt.Kernel,
    e_comb: rt.Kernel,
    e_comb_rms: rt.Kernel,
    a_qmv: rt.Kernel,
    a_qkv: rt.Kernel,
    a_part: rt.Kernel,
    a_merge: rt.Kernel,
    /// Matrix-engine decode attention (windows of up to 16 rows too); null: NEM_OLD_DEC=1 or another head geometry.
    a_dec: ?rt.Kernel,
    a_dmerge: ?rt.Kernel,
    head: rt.Kernel,
    am_part: rt.Kernel,
    am_fin: rt.Kernel,
};

/// Device scratch shared by all layers (one token at a time).
const Scratch = struct {
    tok: Buf,
    x: Buf,
    xn: Buf,
    delta: Buf,
    proj: Buf,
    xc: Buf,
    y: Buf,
    yn: Buf,
    r_logits: Buf,
    ids: Buf,
    wts: Buf,
    act: Buf,
    ey: Buf,
    sact: Buf,
    sy: Buf,
    zero: Buf,
    q: Buf,
    att: Buf,
    po: Buf,
    pm: Buf,
    pl: Buf,
    logits: Buf,
    am_scratch: Buf,
    am_out: Buf,
};

/// Sets every argument (Buffer, f32 or integer) in order and queues the kernel.
fn run(k: *rt.Kernel, groups: [3]u32, a: anytype) !void {
    inline for (a, 0..) |v, i| {
        const T = @TypeOf(v);
        if (T == Buf) {
            try k.setBuffer(i, v);
        } else if (T == f32) {
            try k.setF32(i, v);
        } else try k.setU32(i, @as(u32, v));
    }
    try k.launch(groups);
}

pub const Model = struct {
    r: *rt.Runtime,
    cfg: cfgm.Config,
    layers: []Layer,
    emb: Table,
    norm_f: Buf,
    head: Table,
    k: Kernels,
    s: Scratch,
    cap: u32,
    /// Split-K chunks the attention scratch (po / pm / pl, one row) holds: ceil(cap / 512), at least 64.
    max_chunks: u32,
    pos: u32 = 0,
    trace: bool = false,
    /// Round the lm_head logits to bf16 before argmax and readback, as upstream.
    bf16_logits: bool = false,

    fn table(l: *ld.Loader, prefix: []const u8) !Table {
        var buf: [160]u8 = undefined;
        return .{
            .w = try l.load(try std.fmt.bufPrint(&buf, "{s}.weight", .{prefix})),
            .s = try l.load(try std.fmt.bufPrint(&buf, "{s}.scales", .{prefix})),
            .b = try l.load(try std.fmt.bufPrint(&buf, "{s}.biases", .{prefix})),
        };
    }

    pub fn load(gpa: std.mem.Allocator, r: *rt.Runtime, l: *ld.Loader, cfg: cfgm.Config, cap: u32) !Model {
        try cfg.validate();
        try checkFormat(l);
        if (cap > max_ctx) return error.ContextTooLarge;
        const max_chunks = @max(min_chunks, (cap + attn_chunk - 1) / attn_chunk);
        var bm = try r.module(spv_basic);
        var mm = try r.module(spv_mamba);
        var em = try r.module(spv_moe);
        var am = try r.module(spv_attn);
        var gm = try r.module(spv_glue);
        const k: Kernels = .{
            .embed = try bm.kernel("embed4", .{ 64, 1, 1 }),
            .embed_tok = try bm.kernel("embed4_tok", .{ 64, 1, 1 }),
            .rms = try bm.kernel("rmsnorm", .{ 64, 1, 1 }),
            .add = try gm.kernel("add_bf16", .{ 64, 1, 1 }),
            .add_rms = try bm.kernel("add_rmsnorm", .{ 64, 1, 1 }),
            .round = try gm.kernel("round_bf16_f32", .{ 64, 1, 1 }),
            .m_qmv = try mm.kernel("qmv4_bf16", .{ 16, 1, 1 }),
            .m_conv = try mm.kernel("conv1d_step", .{ 64, 1, 1 }),
            .m_ssm = try mm.kernel("ssm_step", .{ 64, 1, 1 }),
            .m_norm = try mm.kernel("group_rmsnorm", .{ 64, 1, 1 }),
            .e_logits = try em.kernel("router_logits", .{ 16, 1, 1 }),
            .e_route = try em.kernel("moe_route", .{ 16, 1, 1 }),
            .e_up = try em.kernel("expert_up_relu2", .{ 16, 1, 1 }),
            .e_down = try em.kernel("expert_down_f32", .{ 16, 1, 1 }),
            .e_rup = try em.kernel("router_up_sh", .{ 16, 1, 1 }),
            .e_rdown = try em.kernel("route_down_sh", .{ 16, 1, 1 }),
            .e_comb = try em.kernel("moe_combine", .{ 64, 1, 1 }),
            .e_comb_rms = try em.kernel("add_comb_rmsnorm", .{ 64, 1, 1 }),
            .a_qmv = try am.kernel("qmv4_bf", .{ 64, 1, 1 }),
            .a_qkv = try am.kernel("qkv4_bf", .{ 64, 1, 1 }),
            .a_part = try am.kernel("attn_partial", .{ 256, 1, 1 }),
            .a_merge = try am.kernel("attn_merge", .{ 16, 1, 1 }),
            .a_dec = if (cfg.head_dim == 128 and cfg.num_attention_heads == 32 and cfg.num_key_value_heads == 2 and std.c.getenv("NEM_OLD_DEC") == null) blk: {
                var dm = try r.moduleWith(spv_attn_dec, "-cl-intel-256-GRF-per-thread");
                break :blk try dm.kernel("nem_attn_dec_partial", .{ 128, 1, 1 });
            } else null,
            .a_dmerge = if (cfg.head_dim == 128 and cfg.num_attention_heads == 32 and cfg.num_key_value_heads == 2 and std.c.getenv("NEM_OLD_DEC") == null) blk: {
                var dm = try r.moduleWith(spv_attn_dec, "-cl-intel-256-GRF-per-thread");
                break :blk try dm.kernel("nem_attn_dec_merge", .{ 256, 1, 1 });
            } else null,
            .head = try am.kernel("qmv4_f32", .{ 64, 1, 1 }),
            .am_part = try am.kernel("argmax_partial", .{ 256, 1, 1 }),
            .am_fin = try am.kernel("argmax_final", .{ 256, 1, 1 }),
        };
        const h: usize = cfg.hidden_size;
        const s: Scratch = .{
            .tok = try l.empty(4),
            .x = try l.empty(h * 2),
            .xn = try l.empty(h * 2),
            .delta = try l.empty(h * 2),
            .proj = try l.empty(cfg.projDim() * 2),
            .xc = try l.empty(cfg.convDim() * 2),
            .y = try l.empty(cfg.xd() * 2),
            .yn = try l.empty(cfg.xd() * 2),
            .r_logits = try l.empty(cfg.n_routed_experts * 2),
            .ids = try l.empty(top_k * 4),
            .wts = try l.empty(top_k * 4),
            .act = try l.empty(top_k * cfg.moe_intermediate_size * 2),
            .ey = try l.empty(top_k * h * 4),
            .sact = try l.empty(cfg.moe_shared_expert_intermediate_size * 2),
            .sy = try l.empty(h * 4),
            .zero = try l.zeros(4),
            .q = try l.empty(cfg.num_attention_heads * cfg.head_dim * 2),
            .att = try l.empty(cfg.num_attention_heads * cfg.head_dim * 2),
            .po = try l.empty(max_chunks * cfg.num_attention_heads * cfg.head_dim * 4),
            .pm = try l.empty(max_chunks * cfg.num_attention_heads * 4),
            .pl = try l.empty(max_chunks * cfg.num_attention_heads * 4),
            .logits = try l.empty(@as(usize, cfg.vocab_size) * 4),
            .am_scratch = try l.empty(argmax_parts * 8),
            .am_out = try l.empty(4),
        };
        const m: Model = .{
            .r = r,
            .cfg = cfg,
            .layers = try gpa.alloc(Layer, cfg.num_hidden_layers),
            .emb = try table(l, "backbone.embeddings"),
            .norm_f = try l.load("backbone.norm_f.weight"),
            .head = try table(l, "lm_head"),
            .k = k,
            .s = s,
            .cap = cap,
            .max_chunks = max_chunks,
        };
        var buf: [160]u8 = undefined;
        for (m.layers, 0..) |*ly, i| {
            ly.norm = try l.load(try std.fmt.bufPrint(&buf, "backbone.layers.{d}.norm.weight", .{i}));
            const p = try std.fmt.allocPrint(gpa, "backbone.layers.{d}.mixer", .{i});
            defer gpa.free(p);
            switch (cfg.kind(i)) {
                .mamba => ly.mixer = .{ .mamba = try loadMamba(gpa, l, p, cfg) },
                .moe => ly.mixer = .{ .moe = try loadMoe(gpa, l, p) },
                .attention => ly.mixer = .{ .attention = try loadAttn(gpa, l, p, cfg, cap) },
            }
            std.debug.print("\rloaded layer {d}/{d}, {d:.2} GB on device", .{ i + 1, cfg.num_hidden_layers, @as(f64, @floatFromInt(l.total)) / 1e9 });
        }
        std.debug.print("\n", .{});
        return m;
    }

    fn sub(gpa: std.mem.Allocator, p: []const u8, tail: []const u8) ![]u8 {
        return std.fmt.allocPrint(gpa, "{s}.{s}", .{ p, tail });
    }

    pub fn loadT(gpa: std.mem.Allocator, l: *ld.Loader, p: []const u8, tail: []const u8) !Table {
        const name = try sub(gpa, p, tail);
        defer gpa.free(name);
        return table(l, name);
    }

    fn loadB(gpa: std.mem.Allocator, l: *ld.Loader, p: []const u8, tail: []const u8, f32s: bool) !Buf {
        const name = try sub(gpa, p, tail);
        defer gpa.free(name);
        return if (f32s) l.loadF32(name) else l.load(name);
    }

    fn loadMamba(gpa: std.mem.Allocator, l: *ld.Loader, p: []const u8, cfg: cfgm.Config) !Mamba {
        return .{
            .in = try loadT(gpa, l, p, "in_proj"),
            .out = try loadT(gpa, l, p, "out_proj"),
            .conv_w = try loadB(gpa, l, p, "conv1d.weight", false),
            .conv_b = try loadB(gpa, l, p, "conv1d.bias", false),
            .a_log = try loadB(gpa, l, p, "A_log", true),
            .d = try loadB(gpa, l, p, "D", true),
            .dtb = try loadB(gpa, l, p, "dt_bias", true),
            .gnorm = try loadB(gpa, l, p, "norm.weight", false),
            .cstate = try l.zeros((cfg.conv_kernel - 1) * cfg.convDim() * 2),
            .sstate = try l.zeros(@as(usize, cfg.mamba_num_heads) * cfg.mamba_head_dim * cfg.ssm_state_size * 4),
        };
    }

    pub fn loadMoe(gpa: std.mem.Allocator, l: *ld.Loader, p: []const u8) !Moe {
        return .{
            .gate = try loadB(gpa, l, p, "gate.weight", false),
            .bias = try loadB(gpa, l, p, "gate.e_score_correction_bias", false),
            .fc1 = try loadT(gpa, l, p, "switch_mlp.fc1"),
            .fc2 = try loadT(gpa, l, p, "switch_mlp.fc2"),
            .shup = try loadT(gpa, l, p, "shared_experts.up_proj"),
            .shdn = try loadT(gpa, l, p, "shared_experts.down_proj"),
        };
    }

    pub fn loadAttn(gpa: std.mem.Allocator, l: *ld.Loader, p: []const u8, cfg: cfgm.Config, cap: u32) !Attn {
        const kv = @as(usize, cap) * cfg.num_key_value_heads * cfg.head_dim * 2;
        return .{
            .q = try loadT(gpa, l, p, "q_proj"),
            .k = try loadT(gpa, l, p, "k_proj"),
            .v = try loadT(gpa, l, p, "v_proj"),
            .o = try loadT(gpa, l, p, "o_proj"),
            .kc = try l.zeros(kv),
            .vc = try l.zeros(kv),
        };
    }

    fn rms(m: *Model, w: Buf, eps: f32) !void {
        try run(&m.k.rms, .{ 1, 1, 1 }, .{ m.s.x, w, m.s.xn, m.cfg.hidden_size, eps });
    }

    fn addResidual(m: *Model) !void {
        try run(&m.k.add, .{ (m.cfg.hidden_size + 63) / 64, 1, 1 }, .{ m.s.x, m.s.delta, m.cfg.hidden_size });
    }

    /// x = bf16(x + delta) then xn = rmsnorm(x) with weight w: the previous layer's residual add fused into the norm.
    fn addRms(m: *Model, w: Buf, eps: f32) !void {
        try run(&m.k.add_rms, .{ 1, 1, 1 }, .{ m.s.x, m.s.delta, w, m.s.xn, m.cfg.hidden_size, eps });
    }

    /// As addRms for a MoE block: the combine of its experts is part of the launch.
    fn combRms(m: *Model, w: Buf, eps: f32) !void {
        const s = m.s;
        try run(&m.k.e_comb_rms, .{ 1, 1, 1 }, .{ s.x, s.ey, s.wts, s.sy, w, s.xn, m.cfg.hidden_size, top_k, eps });
    }

    fn mamba(m: *Model, w: Mamba) !void {
        const c = m.cfg;
        const s = m.s;
        try run(&m.k.m_qmv, .{ c.projDim(), 1, 1 }, .{ w.in.w, w.in.s, w.in.b, s.xn, s.proj, c.hidden_size });
        try run(&m.k.m_conv, .{ c.convDim() / 64, 1, 1 }, .{ s.proj, c.xd(), w.cstate, w.conv_w, w.conv_b, s.xc, c.convDim() });
        const per_group = c.mamba_num_heads / c.n_groups;
        try run(&m.k.m_ssm, .{ c.mamba_num_heads, c.mamba_head_dim / 4, 1 }, .{
            s.proj,  c.xd() + c.convDim(), s.xc, w.sstate, w.a_log, w.d, w.dtb, s.y,
            c.mamba_head_dim, c.xd(), c.n_groups, per_group, @as(f32, 0.0), std.math.inf(f32),
        });
        try run(&m.k.m_norm, .{ c.n_groups, 1, 1 }, .{ s.y, w.gnorm, s.yn, c.xd() / c.n_groups, c.layer_norm_epsilon });
        try run(&m.k.m_qmv, .{ c.hidden_size, 1, 1 }, .{ w.out.w, w.out.s, w.out.b, s.yn, s.delta, c.xd() });
    }

    fn moe(m: *Model, w: Moe) !void {
        const c = m.cfg;
        const s = m.s;
        const wd = c.moe_intermediate_size;
        const sw = c.moe_shared_expert_intermediate_size;
        const scaling: u32 = @bitCast(c.routed_scaling_factor);
        try run(&m.k.e_rup, .{ c.n_routed_experts + sw, 1, 1 }, .{ s.xn, w.gate, s.r_logits, c.hidden_size, c.n_routed_experts, w.shup.w, w.shup.s, w.shup.b, s.sact });
        try run(&m.k.e_rdown, .{ 1 + c.hidden_size, 1, 1 }, .{ s.r_logits, w.bias, s.ids, s.wts, c.n_routed_experts, c.num_experts_per_tok, scaling, w.shdn.w, w.shdn.s, w.shdn.b, s.sact, s.sy, sw });
        try run(&m.k.e_up, .{ wd, top_k, 1 }, .{ w.fc1.w, w.fc1.s, w.fc1.b, s.xn, s.ids, s.act, c.hidden_size, wd, @as(u32, 0) });
        try run(&m.k.e_down, .{ c.hidden_size, top_k, 1 }, .{ w.fc2.w, w.fc2.s, w.fc2.b, s.act, s.ids, s.ey, wd, c.hidden_size, wd });
    }

    /// delta = the MoE block output (routed experts weighted + shared expert) once moe() has run.
    fn moeCombine(m: *Model) !void {
        const s = m.s;
        try run(&m.k.e_comb, .{ (m.cfg.hidden_size + 63) / 64, 1, 1 }, .{ s.ey, s.wts, s.sy, s.delta, m.cfg.hidden_size, top_k });
    }

    fn attention(m: *Model, w: Attn) !void {
        const c = m.cfg;
        const s = m.s;
        const q_dim = c.num_attention_heads * c.head_dim;
        const kv_dim = c.num_key_value_heads * c.head_dim;
        const len = m.pos + 1;
        const nch = (len + attn_chunk - 1) / attn_chunk;
        const none: u32 = 0;
        try run(&m.k.a_qkv, .{ q_dim / 4 + kv_dim / 2, 1, 1 }, .{ w.q.w, w.q.s, w.q.b, w.k.w, w.k.s, w.k.b, w.v.w, w.v.s, w.v.b, s.xn, s.q, w.kc, w.vc, c.hidden_size, m.pos * kv_dim, q_dim, kv_dim });
        const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(c.head_dim)));
        if (m.k.a_dec) |*kd| try run(kd, .{ c.num_key_value_heads * 2, nch, 1 }, .{ s.q, w.kc, w.vc, s.po, s.pm, s.pl, len, none, scale }) else try run(&m.k.a_part, .{ c.num_key_value_heads, nch, 1 }, .{ s.q, w.kc, w.vc, s.po, s.pm, s.pl, len, attn_chunk, c.num_key_value_heads, none, scale });
        if (m.k.a_dmerge) |*kd| try run(kd, .{ c.num_attention_heads, 1, 4 }, .{ s.po, s.pm, s.pl, s.att, len, none }) else try run(&m.k.a_merge, .{ c.num_attention_heads, 1, 1 }, .{ s.po, s.pm, s.pl, s.att, len, attn_chunk, c.num_attention_heads, none });
        try run(&m.k.a_qmv, .{ c.hidden_size / 4, 1, 1 }, .{ w.o.w, w.o.s, w.o.b, s.att, s.delta, q_dim, none, none, c.hidden_size });
    }

    /// Prints finiteness and ranges of the residual stream and the block output after a layer.
    fn traceLayer(m: *Model, i: usize) !void {
        var x: [2688]u16 = undefined;
        var d: [2688]u16 = undefined;
        try m.r.download(std.mem.sliceAsBytes(&x), m.s.x);
        try m.r.download(std.mem.sliceAsBytes(&d), m.s.delta);
        try m.r.sync();
        var bad: u32 = 0;
        var xmax: f32 = 0;
        var dmax: f32 = 0;
        var ss: f64 = 0;
        for (x, d) |xv, dv| {
            const xf: f32 = @bitCast(@as(u32, xv) << 16);
            const df: f32 = @bitCast(@as(u32, dv) << 16);
            if (!std.math.isFinite(xf) or !std.math.isFinite(df)) bad += 1 else {
                xmax = @max(xmax, @abs(xf));
                dmax = @max(dmax, @abs(df));
                ss += @as(f64, xf) * xf;
            }
        }
        std.debug.print("  layer {d:>2} {s:<9} x max {d:>10.3} rms {d:>9.3}  delta max {d:>10.3}  nonfinite {d}\n", .{ i, @tagName(m.cfg.kind(i)), xmax, @sqrt(ss / 2688.0), dmax, bad });
    }

    /// Queues one token through every layer (and the head when `logits`); the caller syncs. Advances the position.
    pub fn forward(m: *Model, token: u32, logits: bool) !void {
        if (m.pos >= m.cap) return error.ContextFull;
        const c = m.cfg;
        try run(&m.k.embed_tok, .{ 1, 1, 1 }, .{ m.emb.w, m.emb.s, m.emb.b, token, m.s.x, c.hidden_size }); // the id is a launch argument: no host slot to overwrite
        const Pend = enum { none, add, comb }; // the last layer's residual add (and for a MoE block its combine) is still to do: fused into the next norm
        var pending: Pend = .none;
        for (m.layers, 0..) |ly, i| {
            switch (pending) {
                .none => try m.rms(ly.norm, c.layer_norm_epsilon),
                .add => try m.addRms(ly.norm, c.layer_norm_epsilon),
                .comb => try m.combRms(ly.norm, c.layer_norm_epsilon),
            }
            pending = .add;
            switch (ly.mixer) {
                .mamba => |w| try m.mamba(w),
                .moe => |w| {
                    try m.moe(w);
                    pending = .comb;
                },
                .attention => |w| try m.attention(w),
            }
            if (m.trace) {
                if (pending == .comb) try m.moeCombine();
                try m.addResidual();
                pending = .none;
                try m.traceLayer(i);
            }
        }
        if (!logits and pending != .none) {
            if (pending == .comb) try m.moeCombine();
            try m.addResidual();
            pending = .none;
        }
        if (logits) {
            switch (pending) {
                .none => try m.rms(m.norm_f, c.layer_norm_epsilon),
                .add => try m.addRms(m.norm_f, c.layer_norm_epsilon),
                .comb => try m.combRms(m.norm_f, c.layer_norm_epsilon),
            }
            try run(&m.k.head, .{ c.vocab_size / 4, 1, 1 }, .{ m.head.w, m.head.s, m.head.b, m.s.xn, m.s.logits, c.hidden_size, @as(u32, 0), @as(u32, 0), c.vocab_size });
            if (m.bf16_logits) try run(&m.k.round, .{ (c.vocab_size + 63) / 64, 1, 1 }, .{ m.s.logits, c.vocab_size });
            const per = (c.vocab_size + argmax_parts - 1) / argmax_parts;
            try run(&m.k.am_part, .{ argmax_parts, 1, 1 }, .{ m.s.logits, m.s.am_scratch, c.vocab_size, per });
            try run(&m.k.am_fin, .{ 1, 1, 1 }, .{ m.s.am_scratch, m.s.am_out, argmax_parts });
        }
        m.pos += 1;
    }

    /// The greedy token of the last forward with logits (syncs).
    pub fn argmax(m: *Model) !u32 {
        var out: [1]i32 = undefined;
        try m.r.download(std.mem.sliceAsBytes(&out), m.s.am_out);
        try m.r.sync();
        return @intCast(out[0]);
    }

    /// Copies the fp32 logits of the last forward with logits to `out` (syncs).
    pub fn fetchLogits(m: *Model, out: []f32) !void {
        try m.r.download(std.mem.sliceAsBytes(out), m.s.logits);
        try m.r.sync();
    }
};
