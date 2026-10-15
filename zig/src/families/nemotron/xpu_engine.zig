//! Nemotron-H serial decode on an Intel GPU as one engine value: device checkpoint, one sequence.

const std = @import("std");
const xpu = @import("xpu");
const cfgm = @import("xpu_config.zig");
const model = @import("xpu_model.zig");
const nw = @import("xpu_win.zig");
const mtp = @import("xpu_mtp.zig");

const ld = xpu.loader;

pub const Options = struct {
    /// Positions of the KV caches (prompt and generated tokens).
    context: usize = 4096,
    progress: bool = false,
    /// Rows of the prompt windows when more than 16 (`--prefill`): the prompt kernels.
    window_rows: u32 = 0,
    /// Load the MTP head (mtp-4bit.safetensors) for drafted rounds.
    mtp: bool = false,
};

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

fn median(v: []f64) f64 {
    std.mem.sort(f64, v, {}, std.sort.asc(f64));
    return v[v.len / 2];
}

/// One stream's sequence (Engine.bind puts it in the model): states, KV, verify scratch, the head's rows.
pub const Seq = struct {
    cstate: []xpu.rt.Buffer,
    sstate: []xpu.rt.Buffer,
    kc: []xpu.rt.Buffer,
    vc: []xpu.rt.Buffer,
    ms: []nw.MState,
    hkc: ?xpu.rt.Buffer = null,
    hvc: ?xpu.rt.Buffer = null,
    rows_h: ?xpu.rt.Buffer = null,
    last_h: ?xpu.rt.Buffer = null,
    pos: u32 = 0,
    win: [4]u32 = .{ 0, 1, 0, 0 }, // the window record: rows, inplace, pos0, head_rows
};

/// A sequence after `at` prompt tokens (the prompt cache's copy): Mamba states, KV rows [0, at), the head's too.
pub const Snap = struct { at: u32, cstate: []xpu.rt.Buffer, sstate: []xpu.rt.Buffer, kc: []xpu.rt.Buffer, vc: []xpu.rt.Buffer, hkc: ?xpu.rt.Buffer, hvc: ?xpu.rt.Buffer };

pub const Engine = struct {
    pub const Sequence = Seq;
    pub const Snapshot = Snap;
    gpa: std.mem.Allocator,
    ctx: *const xpu.Context,
    stream: xpu.Stream,
    r: xpu.rt.Runtime,
    loader: ld.Loader,
    cfg: cfgm.Config,
    m: model.Model,
    w: nw.Win,
    max_len: usize,
    load_seconds: f64,
    /// The MTP head and the prompt's last normed row (its first draft reads it), when loaded.
    head: ?mtp.Mtp = null,
    last_h: ?xpu.rt.Buffer = null,
    /// The sequence bound to the model (own: the model's buffers) and the prompt windows' scratch (rcap rows, shared).
    own: Seq = undefined,
    cur: *Seq = undefined,
    big_ms: []nw.MState = &.{},

    pub fn init(gpa: std.mem.Allocator, io: std.Io, ctx: *const xpu.Context, dir: []const u8, o: Options) !*Engine {
        const t0 = std.Io.Clock.awake.now(io);
        const e = try gpa.create(Engine);
        errdefer gpa.destroy(e);
        e.gpa = gpa;
        e.ctx = ctx;
        e.stream = try xpu.Stream.init(ctx);
        errdefer e.stream.deinit();
        e.r = try xpu.rt.Runtime.init(ctx, e.stream);
        errdefer e.r.deinit(); // drains the queue and forgets it, so the exit handler never calls into a closed driver
        e.loader = try ld.Loader.init(gpa, &e.r, dir);
        try model.checkFormat(&e.loader); // before the config: another format's config.json may not parse
        const text = try ld.readFile(gpa, dir, "config.json");
        e.cfg = (try cfgm.parse(gpa, text)).value;
        e.m = try model.Model.load(gpa, &e.r, &e.loader, e.cfg, @intCast(o.context));
        e.m.bf16_logits = true;
        if (o.window_rows > 16) nw.default_rows = o.window_rows;
        e.w = try nw.Win.init(gpa, &e.m);
        e.head = null;
        e.last_h = null;
        if (o.mtp) {
            e.head = try mtp.Mtp.load(gpa, &e.m, &e.loader, dir);
            e.last_h = try e.r.alloc(e.cfg.hidden_size * 2);
        }
        e.big_ms = e.w.ms;
        e.own = try e.ownSeq();
        e.cur = &e.own;
        e.max_len = o.context;
        e.load_seconds = xpu.decode.seconds(io, t0);
        return e;
    }

    /// The model's own buffers as a sequence (its verify scratch is the prompt windows').
    fn ownSeq(e: *Engine) !Seq {
        var s: Seq = .{ .cstate = &.{}, .sstate = &.{}, .kc = &.{}, .vc = &.{}, .ms = e.big_ms, .last_h = e.last_h };
        var cs: std.ArrayList(xpu.rt.Buffer) = .empty;
        var ss: std.ArrayList(xpu.rt.Buffer) = .empty;
        var ks: std.ArrayList(xpu.rt.Buffer) = .empty;
        var vs: std.ArrayList(xpu.rt.Buffer) = .empty;
        for (e.m.layers) |ly| switch (ly.mixer) {
            .mamba => |mw| {
                try cs.append(e.gpa, mw.cstate);
                try ss.append(e.gpa, mw.sstate);
            },
            .attention => |aw| {
                try ks.append(e.gpa, aw.kc);
                try vs.append(e.gpa, aw.vc);
            },
            else => {},
        };
        s.cstate = try cs.toOwnedSlice(e.gpa);
        s.sstate = try ss.toOwnedSlice(e.gpa);
        s.kc = try ks.toOwnedSlice(e.gpa);
        s.vc = try vs.toOwnedSlice(e.gpa);
        if (e.head) |h| {
            s.hkc = h.attn.kc;
            s.hvc = h.attn.vc;
            s.rows_h = try e.r.alloc(nw.max_rows * e.cfg.hidden_size * 2);
        }
        return s;
    }

    /// A new sequence for a stream: zeroed states, KV for the engine's context, its own verify scratch (16 rows).
    pub fn newSeq(e: *Engine) !*Seq {
        const c = e.cfg;
        const s = try e.gpa.create(Seq);
        errdefer e.gpa.destroy(s);
        const nm = e.own.cstate.len;
        const na = e.own.kc.len;
        s.* = .{ .cstate = try e.gpa.alloc(xpu.rt.Buffer, nm), .sstate = try e.gpa.alloc(xpu.rt.Buffer, nm), .kc = try e.gpa.alloc(xpu.rt.Buffer, na), .vc = try e.gpa.alloc(xpu.rt.Buffer, na), .ms = try e.gpa.alloc(nw.MState, nm) };
        const conv = (c.conv_kernel - 1) * c.convDim() * 2;
        const ssm = @as(usize, c.mamba_num_heads) * c.mamba_head_dim * c.ssm_state_size * 4;
        const kv = e.m.cap * @as(usize, c.num_key_value_heads) * c.head_dim * 2;
        const R: usize = nw.max_rows;
        for (0..nm) |i| {
            s.cstate[i] = try e.r.alloc(conv);
            s.sstate[i] = try e.r.alloc(ssm);
            s.ms[i] = .{ .proj = try e.r.alloc(R * c.projDim() * 2), .xc = try e.r.alloc(R * c.convDim() * 2), .cstate2 = try e.r.alloc(conv), .sstate2 = try e.r.alloc(ssm) };
        }
        for (0..na) |i| {
            s.kc[i] = try e.r.alloc(kv);
            s.vc[i] = try e.r.alloc(kv);
        }
        if (e.head != null) {
            s.hkc = try e.r.alloc(kv);
            s.hvc = try e.r.alloc(kv);
            s.rows_h = try e.r.alloc(R * c.hidden_size * 2);
            s.last_h = try e.r.alloc(c.hidden_size * 2);
        }
        return s;
    }

    /// Frees a sequence from newSeq (never the engine's own); binds the own sequence first if it was bound.
    pub fn freeSeq(e: *Engine, s: *Seq) void {
        if (s == &e.own) return;
        if (e.cur == s) e.bind(&e.own);
        e.r.sync() catch {};
        for (s.cstate, s.sstate, s.ms) |*a, *b, *ms| {
            a.free();
            b.free();
            ms.proj.free();
            ms.xc.free();
            ms.cstate2.free();
            ms.sstate2.free();
        }
        for (s.kc, s.vc) |*a, *b| {
            a.free();
            b.free();
        }
        inline for (.{ "hkc", "hvc", "rows_h", "last_h" }) |f| if (@field(s, f)) |*b| b.free();
        e.gpa.free(s.cstate);
        e.gpa.free(s.sstate);
        e.gpa.free(s.kc);
        e.gpa.free(s.vc);
        e.gpa.free(s.ms);
        e.gpa.destroy(s);
    }

    /// Puts sequence s in the model; the bound one keeps its state handles (commit swaps them) and position first.
    pub fn bind(e: *Engine, s: *Seq) void {
        if (e.cur == s) return;
        const o = e.cur;
        var mi: usize = 0;
        var ai: usize = 0;
        for (e.m.layers) |*ly| switch (ly.mixer) {
            .mamba => |*mw| {
                o.cstate[mi] = mw.cstate;
                o.sstate[mi] = mw.sstate;
                mw.cstate = s.cstate[mi];
                mw.sstate = s.sstate[mi];
                mi += 1;
            },
            .attention => |*aw| {
                aw.kc = s.kc[ai];
                aw.vc = s.vc[ai];
                ai += 1;
            },
            else => {},
        };
        if (e.head) |*h| {
            h.attn.kc = s.hkc.?;
            h.attn.vc = s.hvc.?;
        }
        o.pos = e.m.pos;
        o.win = .{ e.w.rows, @intFromBool(e.w.inplace), e.w.pos0, e.w.head_rows };
        e.m.pos = s.pos;
        e.w.ms = s.ms;
        e.w.rows = s.win[0];
        e.w.inplace = s.win[1] != 0;
        e.w.pos0 = s.win[2];
        e.w.head_rows = s.win[3];
        e.last_h = s.last_h;
        e.cur = s;
    }

    /// The process exits right after: device memory goes with it once the queue is drained.
    pub fn deinit(e: *Engine) void {
        e.r.deinit();
        e.stream.deinit();
        e.gpa.destroy(e);
    }

    pub fn deviceBytes(e: *const Engine) u64 {
        return e.loader.total;
    }

    pub fn vocab(e: *const Engine) usize {
        return e.cfg.vocab_size;
    }

    pub fn isEos(e: *const Engine, token: u32) bool {
        for (e.cfg.eos_token_id) |x| if (x == token) return true;
        return false;
    }

    pub fn reset(e: *Engine) !void {
        try e.w.reset(&e.m);
    }

    pub fn feed(e: *Engine, token: u32, logits: bool) !void {
        try nw.stopCheck(&e.r);
        try e.m.forward(token, logits);
    }

    pub fn sync(e: *Engine) !void {
        try e.r.sync();
    }

    pub fn argmax(e: *Engine) !u32 {
        return e.m.argmax();
    }

    pub fn fetchLogits(e: *Engine, out: []f32) !void {
        try e.m.fetchLogits(out);
    }

    // -- drafted rounds (the MTP head loaded): every round is a window, the head absorbs the kept rows --------------

    /// The prompt in windows of `rows`, in place; the head absorbs all rows but the last (last_h, for draftFirst).
    pub fn prefillHead(e: *Engine, prompt: []const u32, rows: u32) !void {
        try e.prefillSpan(prompt, 0, prompt.len, rows);
    }

    /// Prompt tokens [from, to) at m.pos == from, in windows of `rows`; the head absorbs as prefillHead does.
    pub fn prefillSpan(e: *Engine, prompt: []const u32, from: usize, to: usize, rows: u32) !void {
        const mt = &e.head.?;
        if (e.m.pos != from) return error.PositionMismatch;
        e.w.ms = e.big_ms; // prompt windows (in place) take the shared scratch; the sequence's own is for verify windows
        defer e.w.ms = e.cur.ms;
        var t: usize = from;
        while (t < to) {
            try nw.stopCheck(&e.r);
            const n = @min(rows, to - t);
            const last = t + n == prompt.len;
            try e.w.forward(&e.m, prompt[t .. t + n], true, if (last) 1 else 0);
            const absorb = if (last) n - 1 else n;
            var j: usize = 0;
            while (j < absorb) : (j += nw.max_rows) {
                const k = @min(nw.max_rows, absorb - j);
                try e.w.normRowsAt(&e.m, @intCast(j), @intCast(k), mt.hn);
                _ = try mt.step(&e.w, &e.m, prompt[t + j + 1 ..][0..k], mt.hn, @intCast(t + j), false);
            }
            if (last) try e.w.normRowsAt(&e.m, @intCast(n - 1), 1, e.last_h.?);
            t += n;
        }
    }

    /// Device bytes a snapshot after `at` prompt tokens takes.
    pub fn snapBytes(e: *const Engine, at: u32) u64 {
        const c = e.cfg;
        const conv = (c.conv_kernel - 1) * c.convDim() * 2;
        const ssm = @as(u64, c.mamba_num_heads) * c.mamba_head_dim * c.ssm_state_size * 4;
        const row = @as(u64, c.num_key_value_heads) * c.head_dim * 2;
        const kvs = e.own.kc.len + @as(u64, @intFromBool(e.head != null));
        return e.own.cstate.len * (conv + ssm) + 2 * kvs * at * row;
    }

    /// The bound sequence after `at` prompt tokens (no window pending) copied into new storage.
    pub fn snapSave(e: *Engine, at: u32) !*Snap {
        if (e.m.pos != at) return error.SnapshotOutOfStep;
        const c = e.cfg;
        const conv = (c.conv_kernel - 1) * c.convDim() * 2;
        const ssm = @as(usize, c.mamba_num_heads) * c.mamba_head_dim * c.ssm_state_size * 4;
        const kvb = @as(usize, at) * c.num_key_value_heads * c.head_dim * 2;
        const s = try e.gpa.create(Snap);
        const nm = e.own.cstate.len;
        const na = e.own.kc.len;
        s.* = .{ .at = at, .cstate = try e.gpa.alloc(xpu.rt.Buffer, nm), .sstate = try e.gpa.alloc(xpu.rt.Buffer, nm), .kc = try e.gpa.alloc(xpu.rt.Buffer, na), .vc = try e.gpa.alloc(xpu.rt.Buffer, na), .hkc = null, .hvc = null };
        var mi: usize = 0;
        var ai: usize = 0;
        for (e.m.layers) |ly| switch (ly.mixer) {
            .mamba => |mw| {
                s.cstate[mi] = try e.r.alloc(conv);
                s.sstate[mi] = try e.r.alloc(ssm);
                try e.r.copy(s.cstate[mi], mw.cstate, conv);
                try e.r.copy(s.sstate[mi], mw.sstate, ssm);
                mi += 1;
            },
            .attention => |aw| {
                s.kc[ai] = try e.r.alloc(@max(kvb, 4));
                s.vc[ai] = try e.r.alloc(@max(kvb, 4));
                if (kvb > 0) try e.r.copy(s.kc[ai], aw.kc, kvb);
                if (kvb > 0) try e.r.copy(s.vc[ai], aw.vc, kvb);
                ai += 1;
            },
            else => {},
        };
        if (e.head) |h| {
            s.hkc = try e.r.alloc(@max(kvb, 4));
            s.hvc = try e.r.alloc(@max(kvb, 4));
            if (kvb > 0) try e.r.copy(s.hkc.?, h.attn.kc, kvb);
            if (kvb > 0) try e.r.copy(s.hvc.?, h.attn.vc, kvb);
        }
        try e.r.sync();
        return s;
    }

    /// The bound sequence becomes the snapshot's: states and KV rows [0, at) copied in, m.pos = at.
    pub fn snapRestore(e: *Engine, s: *const Snap) !void {
        const c = e.cfg;
        const conv = (c.conv_kernel - 1) * c.convDim() * 2;
        const ssm = @as(usize, c.mamba_num_heads) * c.mamba_head_dim * c.ssm_state_size * 4;
        const kvb = @as(usize, s.at) * c.num_key_value_heads * c.head_dim * 2;
        var mi: usize = 0;
        var ai: usize = 0;
        for (e.m.layers) |ly| switch (ly.mixer) {
            .mamba => |mw| {
                try e.r.copy(mw.cstate, s.cstate[mi], conv);
                try e.r.copy(mw.sstate, s.sstate[mi], ssm);
                mi += 1;
            },
            .attention => |aw| {
                if (kvb > 0) try e.r.copy(aw.kc, s.kc[ai], kvb);
                if (kvb > 0) try e.r.copy(aw.vc, s.vc[ai], kvb);
                ai += 1;
            },
            else => {},
        };
        if (e.head) |h| if (kvb > 0) {
            try e.r.copy(h.attn.kc, s.hkc.?, kvb);
            try e.r.copy(h.attn.vc, s.hvc.?, kvb);
        };
        try e.r.sync();
        e.m.pos = s.at;
    }

    pub fn snapDrop(e: *Engine, s: *Snap) void {
        e.r.sync() catch {};
        for (s.cstate, s.sstate) |*a, *b| {
            a.free();
            b.free();
        }
        for (s.kc, s.vc) |*a, *b| {
            a.free();
            b.free();
        }
        if (s.hkc) |*b| b.free();
        if (s.hvc) |*b| b.free();
        e.gpa.free(s.cstate);
        e.gpa.free(s.sstate);
        e.gpa.free(s.kc);
        e.gpa.free(s.vc);
        e.gpa.destroy(s);
    }

    /// One window of ids at m.pos, not committed; every row gets logits (rowTokens / rowLogits).
    pub fn verifyRows(e: *Engine, ids: []const u32) !void {
        try nw.stopCheck(&e.r);
        try e.w.forward(&e.m, ids, false, @intCast(ids.len));
        if (e.cur.rows_h) |rh| try e.w.normRows(&e.m, @intCast(ids.len), rh);
    }

    /// One token on the decode step, in place, with logits (argmax / fetchLogits): the bits of a one-row window.
    pub fn stepRow(e: *Engine, token: u32) !void {
        try nw.stopCheck(&e.r);
        try e.m.forward(token, true);
        if (e.cur.rows_h) |rh| try e.m.normRow(rh);
    }

    /// Greedy token of each row of the last window (syncs).
    pub fn rowTokens(e: *Engine, out: []i32) !void {
        try e.w.rowArgmax(&e.m, out);
    }

    /// fp32 logits [rows][vocab] of the last window (syncs).
    pub fn rowLogits(e: *Engine, out: []f32) !void {
        try e.w.fetchRowLogits(&e.m, out);
    }

    /// Keep the first c rows of the last window.
    pub fn keepRows(e: *Engine, c: u32) !void {
        try e.w.commit(&e.m, c);
    }

    /// The head absorbs the kept rows of the last window (follow: the token after each) from `start`, then drafts.
    pub fn draftKept(e: *Engine, follow: []const u32, start: u32, depth: u32, out: []u32) !usize {
        const mt = &e.head.?;
        const d = try mt.step(&e.w, &e.m, follow, e.cur.rows_h.?, start, depth > 0);
        return e.chain(d, depth, out);
    }

    /// The head absorbs the prompt's last row with the first token, then drafts.
    pub fn draftFirst(e: *Engine, first: u32, depth: u32, out: []u32) !usize {
        const mt = &e.head.?;
        const d = try mt.step(&e.w, &e.m, &.{first}, e.last_h.?, @intCast(e.m.pos - 1), depth > 0);
        return e.chain(d, depth, out);
    }

    /// Drafts after the head's first: each from the head's last output row, at m.pos and on.
    fn chain(e: *Engine, first: ?u32, depth: u32, out: []u32) !usize {
        if (depth == 0) return 0;
        const n = @min(depth, out.len);
        out[0] = first.?;
        var pos: u32 = @intCast(e.m.pos);
        for (1..n) |i| {
            out[i] = (try e.head.?.step(&e.w, &e.m, out[i - 1 .. i], e.head.?.fo, pos, true)).?;
            pos += 1;
        }
        return n;
    }

    /// Times windows of 1 .. ms.len rows and one head step (ms, median of 5) on a scratch prompt; resets after.
    pub fn timeRounds(e: *Engine, ms: []f64) !f64 {
        var ids: [64]u32 = undefined;
        for (&ids, 0..) |*x, i| x.* = 1000 + @as(u32, @intCast(i)) * 37;
        try e.reset();
        try e.prefillHead(&ids, 64);
        var g: [nw.max_rows]i32 = undefined;
        var runs: [5]f64 = undefined;
        for (ms, 1..) |*out, width| {
            for (&runs) |*r| {
                const t0 = nowNs();
                if (width == 1) { // a one-row round is the decode step (lane: stepRow)
                    try e.stepRow(ids[0]);
                    _ = try e.argmax();
                } else {
                    try e.verifyRows(ids[0..width]);
                    try e.rowTokens(g[0..width]);
                }
                r.* = @as(f64, @floatFromInt(nowNs() - t0)) / 1e6;
            }
            out.* = median(&runs);
        }
        var d: [1]u32 = undefined;
        _ = try e.draftFirst(ids[0], 1, &d);
        for (&runs) |*r| {
            const t0 = nowNs();
            _ = try e.head.?.step(&e.w, &e.m, &d, e.head.?.fo, @intCast(e.m.pos), true);
            r.* = @as(f64, @floatFromInt(nowNs() - t0)) / 1e6;
        }
        try e.reset();
        return median(&runs);
    }

    /// The prompt head in windows of `rows` tokens (above 16: the prompt kernels), in place; last token fed as decode.
    pub fn prefillWindows(e: *Engine, head: []const u32, rows: u32) !void {
        var t: usize = 0;
        while (t < head.len) {
            try nw.stopCheck(&e.r);
            const n = @min(rows, head.len - t);
            try e.w.forward(&e.m, head[t .. t + n], true, 0);
            t += n;
        }
        try e.r.sync();
    }
};
