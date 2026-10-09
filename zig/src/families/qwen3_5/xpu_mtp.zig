//! The model's own multi-token-prediction head as a speculative drafter; the target verifies, so output is greedy.

const std = @import("std");
const rt = @import("xpu").rt;
const ld = @import("xpu").loader;
const cq = @import("xpu_config.zig");
const qb = @import("xpu_blocks.zig");
const al = @import("xpu_attn_long.zig");
const ql = @import("xpu_load.zig");
const model = @import("xpu_model.zig");
const qw = @import("xpu_win.zig");
const exl3 = @import("xpu").exl3;

const Buf = qb.Buf;
const hidden = qb.hidden;

pub const Stats = struct {
    windows: u64 = 0,
    plain: u64 = 0,
    drafted: u64 = 0,
    accepted: u64 = 0,
    tokens: u64 = 0,
    win_ns: u64 = 0, // verify windows (forward, argmax, commit)
    mtp_ns: u64 = 0, // absorb + chained draft steps
    steps: u64 = 0, // MTP steps with the head (one draft each)
    absorbs: u64 = 0, // MTP steps without the head
    /// Per draft level j: windows reaching it, where draft j was right, where the second choice would have been.
    reach: [16]u64 = @splat(0),
    hit1: [16]u64 = @splat(0),
    hit2: [16]u64 = @splat(0),
};

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

fn runK(k: *rt.Kernel, groups: [3]u32, a: anytype) !void {
    qb.touch();
    inline for (a, 0..) |v, i| {
        const T = @TypeOf(v);
        if (T == Buf) try k.setBuffer(i, v) else if (T == f32) try k.setF32(i, v) else try k.setU32(i, @as(u32, v));
    }
    try k.launch(groups);
}

pub const Mtp = struct {
    ops: *qb.Ops,
    fc: qb.Table,
    attn: qb.AttnW,
    kvm: al.Mode = .bf16, // KV cache format of the block (al.mtpMode)
    mlp: qb.MlpW,
    norm: Buf,
    n_emb: Buf,
    n_hid: Buf,
    cat: Buf, // [embedding | hidden] normed, bf16
    /// Device-side RoPE table ([cap][64] fp32), step tokens (u32) and drafted tokens (i32): no host upload per step.
    rope_tab: Buf,
    tokdev: Buf,
    drafts: Buf,
    /// Reduced draft head: the leading `dvocab` rows of the EXL3 lm_head (ids are roughly by frequency), 0 = full head.
    dhead: ?qb.Table = null,
    dvocab: u32 = 0,
    /// Instrumentation (env MTP_TIME: time head / absorb steps; MTP_TOP2: keep the draft head's top two ids per level).
    timing: bool = false,
    t_head_ns: u64 = 0,
    n_head: u64 = 0,
    t_abs_ns: u64 = 0,
    n_abs: u64 = 0,
    top2_on: bool = false,
    top2: [16][2]u32 = undefined,
    hostl: []f32 = &.{},
    /// Own EXL3 engine when the target is not EXL3 (swapped into ops.ex for the head's steps only).
    own: ?*qb.Exl = null,
    /// Debug switches (env MTP_SWAP: hidden first in the concatenation; MTP_RAWCHAIN: not wired).
    swap: bool = false,

    /// Loads the head from an EXL3 checkpoint through `l` (the target's or a separate loader); `cap` is its KV length.
    pub fn load(gpa: std.mem.Allocator, m: *model.Model, l: *ld.Loader, cap: u32) !Mtp {
        const o = m.ops;
        if (l.gguf) return fromWeights(gpa, m, l, cap, try @import("xpu_mtp_gguf.zig").loadWeights(gpa, m, l, cap));
        var own: ?*qb.Exl = null;
        if (o.ex == null) {
            const mod = try gpa.create(rt.Module);
            mod.* = try o.r.module(@import("xpu").kernels.exl3_mul1);
            const shapes = [_][2]u32{ .{ 5120, 17408 }, .{ 17408, 5120 }, .{ 5120, 12288 }, .{ 5120, 1024 }, .{ 6144, 5120 }, .{ 5120, 10240 }, .{ 10240, 5120 } };
            var zn: u64 = 0;
            for (shapes) |sh| zn = @max(zn, exl3.zn(sh[0], sh[1]));
            own = try gpa.create(qb.Exl);
            own.?.* = .{ .eng = try exl3.Engine.init(o.r, mod), .scratch = try exl3.Scratch.init(o.r, 16, 17408, zn) };
        }
        const eng: *exl3.Engine = if (own) |e| &e.eng else &o.ex.?.eng;
        const p = "mtp.layers.0";
        const kv = al.cacheBytesFor(al.mtpMode(cap), cap);
        const inter = qb.inter;
        const dv: u32 = if (std.c.getenv("MTP_VOCAB")) |v| (std.fmt.parseInt(u32, std.mem.span(v), 10) catch 131072) else 131072;
        return .{
            .ops = o,
            .own = own,
            .dvocab = dv,
            .dhead = if (dv > 0) try ql.exl3ProjCut(gpa, eng, l, "lm_head", m.cfg.text_config.vocab_size, hidden, dv) else null,
            .fc = try ql.exl3ProjE(gpa, eng, l, "mtp.fc", hidden, 2 * hidden),
            .kvm = al.mtpMode(cap),
            .attn = .{
                .norm = try ql.normWs(gpa, true, l, p ++ ".input_layernorm.weight", hidden),
                .q = try ql.exl3ProjE(gpa, eng, l, p ++ ".self_attn.q_proj", 2 * qb.q_dim, hidden),
                .k = try ql.exl3ProjE(gpa, eng, l, p ++ ".self_attn.k_proj", qb.kv_dim, hidden),
                .v = try ql.exl3ProjE(gpa, eng, l, p ++ ".self_attn.v_proj", qb.kv_dim, hidden),
                .o = try ql.exl3ProjE(gpa, eng, l, p ++ ".self_attn.o_proj", hidden, qb.q_dim),
                .qn = try ql.normWs(gpa, true, l, p ++ ".self_attn.q_norm.weight", qb.head_dim),
                .kn = try ql.normWs(gpa, true, l, p ++ ".self_attn.k_norm.weight", qb.head_dim),
                .kc = try l.zeros(kv),
                .vc = try l.zeros(kv),
            },
            .mlp = .{
                .norm = try ql.normWs(gpa, true, l, p ++ ".post_attention_layernorm.weight", hidden),
                .gate = try ql.exl3ProjE(gpa, eng, l, p ++ ".mlp.gate_proj", inter, hidden),
                .up = try ql.exl3ProjE(gpa, eng, l, p ++ ".mlp.up_proj", inter, hidden),
                .down = try ql.exl3ProjE(gpa, eng, l, p ++ ".mlp.down_proj", hidden, inter),
            },
            .norm = try ql.normWs(gpa, true, l, "mtp.norm.weight", hidden),
            .n_emb = try ql.normWs(gpa, true, l, "mtp.pre_fc_norm_embedding.weight", hidden),
            .n_hid = try ql.normWs(gpa, true, l, "mtp.pre_fc_norm_hidden.weight", hidden),
            .cat = try l.empty(2 * hidden * 2 + 64),
            .rope_tab = try ropeTable(gpa, o, l, cap),
            .tokdev = try l.empty(16 * 4),
            .drafts = try l.empty(16 * 4),
            .swap = std.c.getenv("MTP_SWAP") != null,
            .timing = std.c.getenv("MTP_TIME") != null,
            .top2_on = std.c.getenv("MTP_TOP2") != null,
        };
    }

    /// The head of a GGUF target file (its own MTP block, loaded by qwen_mtp_gguf; the target's lm_head drafts).
    fn fromWeights(gpa: std.mem.Allocator, m: *model.Model, l: *ld.Loader, cap: u32, w: @import("xpu_mtp_gguf.zig").Weights) !Mtp {
        const o = m.ops;
        return .{
            .ops = o,
            .fc = w.fc,
            .attn = w.attn,
            .kvm = al.mtpMode(cap),
            .mlp = w.mlp,
            .norm = w.norm,
            .n_emb = w.n_emb,
            .n_hid = w.n_hid,
            .cat = try l.empty(2 * hidden * 2 + 64),
            .rope_tab = try ropeTable(gpa, o, l, cap),
            .tokdev = try l.empty(16 * 4),
            .drafts = try l.empty(16 * 4),
            .swap = std.c.getenv("MTP_SWAP") != null,
            .timing = std.c.getenv("MTP_TIME") != null,
            .top2_on = std.c.getenv("MTP_TOP2") != null,
        };
    }

    /// One MTP position `pos`: token at `tok` (u32), normed hidden row h (bf16); with `out` the draft (i32) goes there.
    pub fn step(self: *Mtp, m: *model.Model, tok: Buf, h: Buf, pos: u32, out: ?Buf) !void {
        const o = self.ops;
        const saved = .{ o.ex, o.s.rope, o.s.am_out };
        al.setOverride(self.kvm);
        defer al.setOverride(null);
        if (self.own) |e| o.ex = e;
        o.s.rope = exl3.at(self.rope_tab, @as(usize, pos) * 256);
        if (out) |b| o.s.am_out = b;
        defer {
            o.ex = saved[0];
            o.s.rope = saved[1];
            o.s.am_out = saved[2];
        }
        const t0: u64 = if (self.timing) blk: {
            try o.r.sync();
            break :blk nowNs();
        } else 0;
        try embedDev(o, m.emb, tok);
        const half: usize = hidden * 2;
        const e_off: usize = if (self.swap) half else 0;
        const h_off: usize = if (self.swap) 0 else half;
        try runK(&o.k.rms512, .{ 1, 1, 1 }, .{ o.s.x, self.n_emb, exl3.at(self.cat, e_off), hidden, o.eps });
        try runK(&o.k.rms512, .{ 1, 1, 1 }, .{ h, self.n_hid, exl3.at(self.cat, h_off), hidden, o.eps });
        try o.matvec(self.fc, self.cat, o.s.x, 0);
        try o.attention(self.attn, pos);
        try o.mlp(self.mlp);
        if (out != null) {
            if (self.dhead) |dh| try o.head(self.norm, dh, self.dvocab, m.bf16_logits) else try o.head(self.norm, m.head, m.cfg.text_config.vocab_size, m.bf16_logits);
        }
        if (self.timing) {
            try o.r.sync();
            if (out != null) {
                self.t_head_ns += nowNs() - t0;
                self.n_head += 1;
            } else {
                self.t_abs_ns += nowNs() - t0;
                self.n_abs += 1;
            }
        }
        if (self.top2_on) if (out) |b| try self.grabTop2(m, (@intFromPtr(b.ptr.?) - @intFromPtr(self.drafts.ptr.?)) / 4);
    }

    /// Top two logits of the draft head's last step (ties to the lowest id, as the argmax kernels) into top2[level].
    fn grabTop2(self: *Mtp, m: *model.Model, level: usize) !void {
        const n: usize = if (self.dhead != null) self.dvocab else m.cfg.text_config.vocab_size;
        if (self.hostl.len < n) self.hostl = try std.heap.page_allocator.alloc(f32, n);
        const o = self.ops;
        try o.r.sync();
        try o.r.download(std.mem.sliceAsBytes(self.hostl[0..n]), o.s.logits);
        try o.r.sync();
        var b1: usize = 0;
        for (self.hostl[0..n], 0..) |v, i| if (v > self.hostl[b1]) {
            b1 = i;
        };
        var b2: usize = if (b1 == 0) 1 else 0;
        for (self.hostl[0..n], 0..) |v, i| if (i != b1 and v > self.hostl[b2]) {
            b2 = i;
        };
        self.top2[level] = .{ @intCast(b1), @intCast(b2) };
    }

    /// Stages host tokens in the device token array (the queue must be idle: the copy reads the host array later).
    pub fn stage(self: *Mtp, toks: []const u32) !void {
        try self.ops.r.upload(self.tokdev, std.mem.sliceAsBytes(toks));
    }

    /// Chains drafts[from..k] on the previous draft (device side), then reads all k back (syncs).
    pub fn chain(self: *Mtp, m: *model.Model, from: usize, k: usize, pos_of_first: u32, out: []u32, st: *Stats) !void {
        var j = from;
        while (j < k) : (j += 1) {
            try self.step(m, exl3.at(self.drafts, (j - 1) * 4), self.ops.s.xn, pos_of_first + @as(u32, @intCast(j - from)), exl3.at(self.drafts, j * 4));
            st.steps += 1;
        }
        try self.ops.r.download(std.mem.sliceAsBytes(out[0..k]), self.drafts);
        try self.ops.r.sync();
    }
};

/// The embedding of the token at device address `tok` into s.x (Ops.embed without the host upload).
fn embedDev(o: *qb.Ops, t: qb.Table, tok: Buf) !void {
    o.pending = false;
    switch (t.format) {
        .mlx_affine4_g64 => try runK(&o.k.embed, .{ 1, 1, 1 }, .{ t.w, t.s, t.b, tok, o.s.x, hidden }),
        .bf16 => try runK(&o.k.embed16, .{ 1, 1, 1 }, .{ t.w, tok, o.s.x, hidden }),
        .f16, .exl3 => return error.Invalid,
        else => try o.gg.?.embedRow(qb.ggType(t.format).?, t.w, tok, o.s.x, hidden),
    }
}

/// cos/sin of pos * theta^(-i/32) for every position, as Ops.setPosition computes them.
fn ropeTable(gpa: std.mem.Allocator, o: *qb.Ops, l: *ld.Loader, cap: u32) !Buf {
    const host = try gpa.alloc(f32, @as(usize, cap) * 64);
    defer gpa.free(host);
    for (0..cap) |p| for (0..32) |i| {
        const inv: f32 = @floatCast(std.math.pow(f64, o.rope_theta, -@as(f64, @floatFromInt(i)) / 32.0));
        const ang: f32 = @as(f32, @floatFromInt(p)) * inv;
        host[p * 64 + i] = @cos(ang);
        host[p * 64 + 32 + i] = @sin(ang);
    };
    const b = try l.empty(host.len * 4);
    try o.r.upload(b, std.mem.sliceAsBytes(host));
    try o.r.sync();
    return b;
}

fn isEos(t: u32, eos: []const u32) bool {
    for (eos) |e| if (t == e) return true;
    return false;
}

/// Prefills in windows of win_rows rows (16: decode kernels, more: prefill GEMMs); the MTP head absorbs the rows.
pub fn prefill(m: *model.Model, mt: *Mtp, prompt: []const u32, k: usize, drafts: []u32, st: *Stats, win_rows: u32) !struct { first: u32, nd: usize } {
    const R: usize = @max(16, @min(win_rows, m.win.rcap));
    var t: usize = 0;
    var first: u32 = 0;
    var stg: [16]u32 = undefined;
    while (t < prompt.len) {
        const rows = @min(R, prompt.len - t);
        const last = t + rows == prompt.len;
        // the last window gets the logits of its last row; the others keep their normalised rows for the head
        try m.forwardRows(prompt[t .. t + rows], if (last) 1 else qw.keep_hidden, true);
        if (last) {
            var g: [1]i32 = undefined;
            try m.rowArgmax(&g);
            first = @intCast(g[0]);
        } else try m.r.sync(); // windows read host-written uploads; the staged tokens below as well
        const t0 = nowNs();
        var j0: usize = 0;
        while (j0 < rows) : (j0 += 16) {
            const n = @min(16, rows - j0);
            for (0..n) |j| stg[j] = if (t + j0 + j + 1 < prompt.len) prompt[t + j0 + j + 1] else first;
            try mt.stage(stg[0..n]);
            for (0..n) |j| {
                const is_last = last and j0 + j + 1 == rows and k > 0;
                try mt.step(m, exl3.at(mt.tokdev, j * 4), exl3.at(m.win.xn, (j0 + j) * hidden * 2), @intCast(t + j0 + j), if (is_last) mt.drafts else null);
                if (is_last) st.steps += 1 else st.absorbs += 1;
            }
            if (!(last and j0 + n == rows)) try m.r.sync(); // the staged tokens are overwritten by the next block
        }
        st.mtp_ns += nowNs() - t0;
        t += rows;
    }
    const t0 = nowNs();
    if (k > 0) try mt.chain(m, 1, k, @intCast(prompt.len), drafts, st);
    st.mtp_ns += nowNs() - t0;
    return .{ .first = first, .nd = k };
}

/// Generates n_new more tokens after `ctx` (m.pos == ctx.len - 1) with MTP drafts of up to k tokens per window.
pub fn run(m: *model.Model, mt: *Mtp, gpa: std.mem.Allocator, ctx: *std.ArrayList(u32), drafts_in: []const u32, n_new: usize, k: usize, eos: []const u32, ignore_eos: bool, st: *Stats) !void {
    var produced: usize = 0;
    var drafts: [16]u32 = undefined;
    var nd: usize = @min(drafts_in.len, 15);
    @memcpy(drafts[0..nd], drafts_in[0..nd]);
    var win: [16]u32 = undefined;
    var g: [16]i32 = undefined;
    while (produced < n_new) {
        if (m.pos + 20 > m.ops.cap) return error.ContextFull; // window + chained draft positions must fit the caches
        const last = ctx.items[ctx.items.len - 1];
        const room = @min(@min(k, 15), n_new - produced - 1);
        const kd = @min(nd, room);
        const pos0 = m.pos;
        const t0 = nowNs();
        var a: usize = 0;
        if (kd == 0) {
            try m.forward(last, true);
            g[0] = @intCast(try m.argmax());
            st.plain += 1;
        } else {
            win[0] = last;
            @memcpy(win[1 .. 1 + kd], drafts[0..kd]);
            try m.forwardRows(win[0 .. kd + 1], @intCast(kd + 1), false);
            try m.rowArgmax(&g);
            while (a < kd and win[1 + a] == @as(u32, @intCast(g[a]))) a += 1;
            try m.commitRows(@intCast(a + 1));
            st.windows += 1;
            st.drafted += kd;
            st.accepted += a;
            if (mt.top2_on) for (0..@min(a + 1, kd)) |j| {
                st.reach[j] += 1;
                const gt: u32 = @intCast(g[j]);
                if (j < a) st.hit1[j] += 1 else if (gt == mt.top2[j][1]) st.hit2[j] += 1;
            };
        }
        st.win_ns += nowNs() - t0;
        const t1 = nowNs();
        // absorb the kept rows (token after row i is g[i]); the last one drafts again, the chain follows on the device
        const hid: Buf = if (kd == 0) m.ops.s.xn else m.win.xn;
        const want = produced + a + 1 < n_new; // another window follows
        var stg: [16]u32 = undefined;
        for (0..a + 1) |i| stg[i] = @intCast(g[i]);
        try mt.stage(stg[0 .. a + 1]);
        for (0..a + 1) |i| {
            const h = if (kd == 0) hid else exl3.at(hid, i * hidden * 2);
            const is_last = i == a and want;
            try mt.step(m, exl3.at(mt.tokdev, i * 4), h, pos0 + @as(u32, @intCast(i)), if (is_last) mt.drafts else null);
            if (is_last) st.steps += 1 else st.absorbs += 1;
        }
        nd = 0;
        if (want) {
            nd = @min(k, 15);
            try mt.chain(m, 1, nd, pos0 + @as(u32, @intCast(a)) + 1, &drafts, st);
        }
        st.mtp_ns += nowNs() - t1;
        for (0..a + 1) |i| {
            const t: u32 = @intCast(g[i]);
            try ctx.append(gpa, t);
            produced += 1;
            st.tokens += 1;
            if (!ignore_eos and isEos(t, eos)) return;
        }
    }
}
