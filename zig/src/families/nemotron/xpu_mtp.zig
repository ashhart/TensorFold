//! Nemotron-H's own MTP head as a speculative drafter: eh_proj, attention (own KV), MoE, norm, lm_head; equals greedy.
const std = @import("std");
const rt = @import("xpu").rt;
const ld = @import("xpu").loader;
const model = @import("xpu_model.zig");
const nw = @import("xpu_win.zig");

const Buf = rt.Buffer;
const spv = @import("xpu").kernels.nem_rows;

fn at(b: Buf, off: usize) Buf {
    return .{ .rt = b.rt, .ptr = @ptrFromInt(@intFromPtr(b.ptr.?) + off), .len = b.len - off };
}

fn run(k: *rt.Kernel, groups: [3]u32, a: anytype) !void {
    inline for (a, 0..) |v, i| {
        const T = @TypeOf(v);
        if (T == Buf) try k.setBuffer(i, v) else if (T == f32) try k.setF32(i, v) else try k.setU32(i, @as(u32, v));
    }
    try k.launch(groups);
}

pub const Stats = struct { windows: u64 = 0, plain: u64 = 0, drafted: u64 = 0, accepted: u64 = 0, tokens: u64 = 0, win_ns: u64 = 0, mtp_ns: u64 = 0, steps: u64 = 0, absorbs: u64 = 0 };

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

pub const Mtp = struct {
    eh: model.Table,
    enorm: Buf,
    hnorm: Buf,
    anorm: Buf,
    attn: model.Attn,
    mnorm: Buf,
    moe: model.Moe,
    fnorm: Buf,
    cat_k: rt.Kernel,
    qmv: rt.Kernel,
    me: Buf, // embeddings of the next tokens [R][h]
    en: Buf,
    hn: Buf, // the target's final-normed hidden rows [R][h] (filled by `hidden`)
    hnr: Buf,
    cat: Buf, // [R][2h]
    x: Buf, // [R][h]
    xl: Buf, // the tail row [h]
    fo: Buf, // the final_layernorm output [h]
    pos: u32 = 0,

    pub fn load(gpa: std.mem.Allocator, m: *model.Model, l: *ld.Loader, dir: []const u8) !Mtp {
        try l.addShard(dir, "mtp-4bit.safetensors");
        var mod = try m.r.module(spv);
        const h: usize = m.cfg.hidden_size;
        const R: usize = nw.max_rows;
        const r = m.r;
        return .{
            .eh = try model.Model.loadT(gpa, l, "layers.0", "eh_proj"),
            .enorm = try l.load("layers.0.enorm.weight"),
            .hnorm = try l.load("layers.0.hnorm.weight"),
            .anorm = try l.load("layers.0.norm.weight"),
            .attn = try model.Model.loadAttn(gpa, l, "layers.0.mixer", m.cfg, m.cap),
            .mnorm = try l.load("layers.1.norm.weight"),
            .moe = try model.Model.loadMoe(gpa, l, "layers.1.mixer"),
            .fnorm = try l.load("layers.1.final_layernorm.weight"),
            .cat_k = try mod.kernel("concat_rows", .{ 64, 1, 1 }),
            .qmv = try mod.kernel("qmv4_bf16_r", .{ 16, 1, 1 }),
            .me = try r.alloc(R * h * 2),
            .en = try r.alloc(R * h * 2),
            .hn = try r.alloc(R * h * 2),
            .hnr = try r.alloc(R * h * 2),
            .cat = try r.alloc(R * 2 * h * 2),
            .x = try r.alloc(R * h * 2),
            .xl = try r.alloc(h * 2),
            .fo = try r.alloc(h * 2),
        };
    }

    /// One MTP pass over n rows (tokens[i] follows target row i, at pos0 + i); the last row runs MoE+head if `draft`.
    pub fn step(self: *Mtp, w: *nw.Win, m: *model.Model, tokens: []const u32, hid: Buf, pos0: u32, draft: bool) !?u32 {
        const n: u32 = @intCast(tokens.len);
        const c = m.cfg;
        const h = c.hidden_size;
        try w.uploadIds(m, tokens);
        try run(&m.k.embed, .{ n, 1, 1 }, .{ m.emb.w, m.emb.s, m.emb.b, w.ids, self.me, h });
        try run(&m.k.rms, .{ n, 1, 1 }, .{ self.me, self.enorm, self.en, h, c.layer_norm_epsilon });
        try run(&m.k.rms, .{ n, 1, 1 }, .{ hid, self.hnorm, self.hnr, h, c.layer_norm_epsilon });
        try run(&self.cat_k, .{ (h + 63) / 64, n, 1 }, .{ self.en, self.hnr, self.cat, h });
        try run(&self.qmv, .{ h, 1, 1 }, .{ self.eh.w, self.eh.s, self.eh.b, self.cat, self.x, 2 * h, h, h, @as(u32, 0), n });
        try run(&m.k.rms, .{ n, 1, 1 }, .{ self.x, self.anorm, w.xn, h, c.layer_norm_epsilon });
        try w.attention(m, self.attn, n, pos0);
        try run(&m.k.add, .{ (n * h + 63) / 64, 1, 1 }, .{ self.x, w.delta, n * h });
        if (!draft) return null;
        const off = @as(usize, n - 1) * h * 2;
        try run(&m.k.rms, .{ 1, 1, 1 }, .{ at(self.x, off), self.mnorm, w.xn, h, c.layer_norm_epsilon });
        try w.moe(m, self.moe, 1);
        try run(&m.k.add, .{ (h + 63) / 64, 1, 1 }, .{ at(self.x, off), w.delta, h });
        try run(&m.k.rms, .{ 1, 1, 1 }, .{ at(self.x, off), self.fnorm, self.fo, h, c.layer_norm_epsilon });
        try w.headOn(m, self.fo);
        var out: [1]i32 = undefined;
        try m.r.download(std.mem.sliceAsBytes(&out), w.am_out);
        try m.r.sync();
        return @intCast(out[0]);
    }
};

fn isEos(t: u32, eos: []const u32) bool {
    for (eos) |e| if (t == e) return true;
    return false;
}

/// Prompt in windows of 16 (in place), every row absorbed; returns the first generated token and first draft chain (k).
pub fn prefill(w: *nw.Win, mt: *Mtp, m: *model.Model, prompt: []const u32, k: usize, drafts: []u32, st: *Stats) !struct { first: u32, nd: usize } {
    var t: usize = 0;
    var first: u32 = 0;
    var stg: [16]u32 = undefined;
    while (t < prompt.len) {
        try nw.stopCheck(m.r);
        const rows = @min(16, prompt.len - t);
        const last = t + rows == prompt.len;
        try w.forward(m, prompt[t .. t + rows], true, if (last) 1 else 0);
        if (last) {
            var g: [1]i32 = undefined;
            try w.rowArgmax(m, &g);
            first = @intCast(g[0]);
        } else try m.r.sync();
        try w.normRows(m, @intCast(rows), mt.hn);
        const t0 = nowNs();
        for (0..rows) |j| stg[j] = if (t + j + 1 < prompt.len) prompt[t + j + 1] else first;
        const d = try mt.step(w, m, stg[0..rows], mt.hn, @intCast(t), last and k > 0);
        if (last and k > 0) {
            drafts[0] = d.?;
            st.steps += 1;
        } else st.absorbs += 1;
        st.mtp_ns += nowNs() - t0;
        t += rows;
    }
    var nd: usize = if (k > 0) 1 else 0;
    var pos: u32 = @intCast(prompt.len);
    const t0 = nowNs();
    while (nd < k) : (nd += 1) {
        const d = try mt.step(w, m, drafts[nd - 1 .. nd], mt.fo, pos, true);
        drafts[nd] = d.?;
        pos += 1;
        st.steps += 1;
    }
    st.mtp_ns += nowNs() - t0;
    return .{ .first = first, .nd = nd };
}

/// Generates n_new tokens after ctx (prompt + first token; m.pos == ctx.len - 1) with MTP drafts of up to k per window.
pub fn runSpec(w: *nw.Win, mt: *Mtp, m: *model.Model, gpa: std.mem.Allocator, ctx: *std.ArrayList(u32), drafts_in: []const u32, n_new: usize, k: usize, eos: []const u32, ignore_eos: bool, st: *Stats) !void {
    var produced: usize = 0;
    var drafts: [16]u32 = undefined;
    var nd: usize = @min(drafts_in.len, 15);
    @memcpy(drafts[0..nd], drafts_in[0..nd]);
    var win: [16]u32 = undefined;
    var g: [16]i32 = undefined;
    while (produced < n_new) {
        try nw.stopCheck(m.r);
        const last = ctx.items[ctx.items.len - 1];
        const room = @min(@min(k, 15), n_new - produced - 1);
        const kd = @min(nd, room);
        const pos0 = m.pos;
        const t0 = nowNs();
        win[0] = last;
        @memcpy(win[1 .. 1 + kd], drafts[0..kd]);
        try w.forward(m, win[0 .. kd + 1], false, @intCast(kd + 1));
        try w.rowArgmax(m, &g);
        var a: usize = 0;
        while (a < kd and win[1 + a] == @as(u32, @intCast(g[a]))) a += 1;
        try w.commit(m, @intCast(a + 1));
        if (kd == 0) st.plain += 1 else {
            st.windows += 1;
            st.drafted += kd;
            st.accepted += a;
        }
        st.win_ns += nowNs() - t0;
        const t1 = nowNs();
        const want = produced + a + 1 < n_new;
        var stg: [16]u32 = undefined;
        for (0..a + 1) |i| stg[i] = @intCast(g[i]);
        try w.normRows(m, @intCast(a + 1), mt.hn);
        const d = try mt.step(w, m, stg[0 .. a + 1], mt.hn, pos0, want);
        nd = 0;
        if (want) {
            drafts[0] = d.?;
            st.steps += 1;
            nd = 1;
            var pos = pos0 + @as(u32, @intCast(a)) + 1;
            while (nd < k and nd < 15) : (nd += 1) {
                drafts[nd] = (try mt.step(w, m, drafts[nd - 1 .. nd], mt.fo, pos, true)).?;
                pos += 1;
                st.steps += 1;
            }
        } else st.absorbs += 1;
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
