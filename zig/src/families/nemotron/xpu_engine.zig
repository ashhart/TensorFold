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

pub const Engine = struct {
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
        e.max_len = o.context;
        e.load_seconds = xpu.decode.seconds(io, t0);
        return e;
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
        const mt = &e.head.?;
        var t: usize = 0;
        while (t < prompt.len) {
            try nw.stopCheck(&e.r);
            const n = @min(rows, prompt.len - t);
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

    /// One window of ids at m.pos, not committed; every row gets logits (rowTokens / rowLogits).
    pub fn verifyRows(e: *Engine, ids: []const u32) !void {
        try nw.stopCheck(&e.r);
        try e.w.forward(&e.m, ids, false, @intCast(ids.len));
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
        try e.w.normRows(&e.m, @intCast(follow.len), mt.hn);
        const d = try mt.step(&e.w, &e.m, follow, mt.hn, start, depth > 0);
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
                try e.verifyRows(ids[0..width]);
                try e.rowTokens(g[0..width]);
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
