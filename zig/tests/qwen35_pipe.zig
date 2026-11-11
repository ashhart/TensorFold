//! `tf-qwen35-cuda <model> pipe <out> [fp8]`: greedy streams decoded in `k` groups, one engine or a stage ring.
const std = @import("std");
const cuda = @import("cuda");
const stage = @import("stage");
const q = @import("qwen35");
const E = q.engine;
const S = q.stage;
const st = @import("qwen35_stage.zig");

/// The run's shape, the same on every rank (TF_PIPE_*): followers size their slots from it.
const Setup = struct {
    streams: u32 = 4,
    /// Groups pipelined (rounds in flight); 0: one round at a time through `step`, as before pipelining.
    k: u32 = 1,
    prompt: u32 = 512,
    tokens: u32 = 64,
    /// Drafts a window from the stream's own history (prompt lookup), up to this many; 0: plain one-row steps.
    draft: u32 = 0,

    fn read() !Setup {
        var s: Setup = .{};
        inline for (.{ .{ "TF_PIPE_STREAMS", "streams" }, .{ "TF_PIPE_K", "k" }, .{ "TF_PIPE_PROMPT", "prompt" }, .{ "TF_PIPE_TOKENS", "tokens" }, .{ "TF_PIPE_DRAFT", "draft" } }) |f| {
            if (std.c.getenv(f[0])) |v| @field(s, f[1]) = try std.fmt.parseInt(u32, std.mem.span(v), 10);
        }
        if (s.streams == 0 or s.streams > 64 or s.k > s.streams or s.tokens == 0 or s.prompt == 0) return error.BadPipeSetup;
        if (s.draft > 7 or s.streams * (s.draft + 1) > E.most_verify) return error.BadPipeSetup;
        return s;
    }

    fn context(s: Setup) u32 {
        return s.prompt + s.tokens + s.draft + 16;
    }
};

pub fn run(init: std.process.Init, a: std.mem.Allocator, dir: []const u8, out_path: []const u8, kv8: bool) !u8 {
    const set = try Setup.read();
    const config = try std.Io.Dir.cwd().readFileAlloc(init.io, try std.fs.path.join(a, &.{ dir, "config.json" }), a, .limited(1 << 22));
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    if (std.c.getenv("TF_STAGE_RANK") == null) {
        const e = try E.Engine.init(init.gpa, init.io, &ctx, dir, kv8, null, E.max_rows);
        defer e.deinit();
        var lead = try S.Lead.init(init.gpa, e, null, set.streams, set.context(), 1);
        defer lead.deinit(init.gpa);
        return finish(init, a, e, &lead, set, out_path, "whole");
    }
    var ss: stage.config.Settings = undefined;
    try stage.config.parse(stage.config.Env{}, &ss);
    var ranges: [stage.config.most_ranks]stage.boot.Range = undefined;
    const layout = try st.parseLayout(std.mem.span(std.c.getenv("TF_STAGE_LAYOUT") orelse return error.MissingStageLayout), &ranges);
    if (layout.len != ss.world) return error.BadStageLayout;
    const mine = layout[ss.rank];
    const e = try E.Engine.initStage(init.gpa, init.io, &ctx, dir, kv8, null, E.max_rows, .{ .first = mine.first, .end = mine.end, .heads = ss.rank == 0 });
    defer e.deinit();
    var ring = try stage.Ring.open(ss.options(S.fingerprint(config, layout)));
    defer ring.close();
    defer ring.stop();
    const own = try S.describe(e, ss.rank, ss.world, mine, set.streams, set.context());
    const model: stage.boot.Model = .{ .version = S.version, .kv_dtype = own.kv_dtype, .layers = @intCast(e.g.model().layers), .boundary = own.boundary, .shapes = own.shapes, .taps = &.{} };
    const next = try S.bootRing(init.gpa, &ring, &own, &model, layout);
    if (ss.rank != 0) {
        try S.serve(init.gpa, e, &ring, set.streams, set.context(), next);
        st.report(ss.rank, "served");
        return 0;
    }
    var lead = try S.Lead.init(init.gpa, e, &ring, set.streams, set.context(), next);
    defer lead.deinit(init.gpa);
    return finish(init, a, e, &lead, set, out_path, "rank 0");
}

/// One stream's tokens: its prompt, then what greedy decoding kept; its unsettled window's rows.
const Stream = struct {
    hist: std.ArrayList(u32) = .empty,
    made: u32 = 0,
    rows: u32 = 0,
    kept: u32 = 0,
    window: [8]u32 = undefined,
};

fn now(io: std.Io) i96 {
    return std.Io.Clock.awake.now(io).toNanoseconds();
}

fn finish(init: std.process.Init, a: std.mem.Allocator, e: *E.Engine, lead: *S.Lead, set: Setup, out_path: []const u8, role: []const u8) !u8 {
    const streams = try a.alloc(Stream, set.streams);
    for (streams) |*x| x.* = .{};
    const t0 = now(init.io);
    for (streams, 0..) |*x, s| try prompt(a, lead, x, @intCast(s), set);
    const t1 = now(init.io);
    if (set.k > 0) try lead.pipe(init.io, set.k);
    var rounds: usize = 0;
    var drafted: usize = 0;
    var taken: usize = 0;
    if (set.k == 0) {
        while (try issue(lead, streams, set, 0, 1, true)) {
            taken += try land(a, e, streams, set, 0, 1, &drafted);
            rounds += 1;
        }
    } else {
        var order: [64]u32 = undefined;
        var n: usize = 0;
        for (0..set.k) |g| if (try issue(lead, streams, set, @intCast(g), set.k, false)) {
            order[n] = @intCast(g);
            n += 1;
        };
        while (n > 0) {
            const g = order[0];
            std.mem.copyForwards(u32, order[0 .. n - 1], order[1..n]);
            n -= 1;
            try lead.complete();
            taken += try land(a, e, streams, set, g, set.k, &drafted);
            rounds += 1;
            if (!try issue(lead, streams, set, g, set.k, false)) continue;
            order[n] = g;
            n += 1;
        }
    }
    const t2 = now(init.io);
    var closes: [64]S.Entry = undefined;
    for (closes[0..set.streams], 0..) |*c, s| c.* = .{ .id = @intCast(s + 1), .op = .close };
    try lead.step(closes[0..set.streams], &.{}, .window);
    var out: std.ArrayList(u8) = .empty;
    for (streams) |x| try out.appendSlice(a, std.mem.sliceAsBytes(x.hist.items[set.prompt..][0..set.tokens]));
    try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = out_path, .data = out.items });
    const ms = @as(f64, @floatFromInt(t2 - t1)) / 1e6;
    const made = @as(f64, @floatFromInt(set.streams * set.tokens));
    std.debug.print("pipe {s}: {{\"streams\":{d},\"k\":{d},\"draft\":{d},\"prompt\":{d},\"tokens\":{d},\"prompt_ms\":{d:.1},\"decode_ms\":{d:.1},\"tok_s\":{d:.2},\"rounds\":{d},\"drafted\":{d},\"kept_drafts\":{d}}}\n", .{ role, set.streams, set.k, set.draft, set.prompt, set.tokens, @as(f64, @floatFromInt(t1 - t0)) / 1e6, ms, made * 1e3 / ms, rounds, drafted, taken });
    st.report(0, role);
    return 0;
}

/// Stream `s`'s prompt (its own seed, so it is the same whatever the stream count), in prompt chunks.
fn prompt(a: std.mem.Allocator, lead: *S.Lead, x: *Stream, s: u32, set: Setup) !void {
    var prng = std.Random.DefaultPrng.init(1000 + s);
    const rnd = prng.random();
    try x.hist.resize(a, set.prompt);
    for (x.hist.items) |*t| t.* = rnd.intRangeLessThan(u32, 1000, 60000);
    var at: u32 = 0;
    while (at < set.prompt) {
        const n = @min(set.prompt - at, E.max_rows);
        const op: stage.plan.Op = if (at == 0) .open else .@"continue";
        try lead.step(&.{.{ .id = s + 1, .op = op, .rows = n }}, x.hist.items[at..][0..n], .prompt);
        at += n;
    }
    try x.hist.append(a, argmax(lead.cache(s + 1).?.logits));
    x.made = 1;
}

/// Group `g` of `k` (contiguous streams): `first` and the end.
fn span(n: u32, k: u32, g: u32) [2]u32 {
    const base = n / k;
    const extra = n % k;
    const first = g * base + @min(g, extra);
    return .{ first, first + base + @intFromBool(g < extra) };
}

/// Group `g`'s next round: each unfinished stream's last token and its drafts; false once all its streams are done.
fn issue(lead: *S.Lead, streams: []Stream, set: Setup, g: u32, k: u32, step: bool) !bool {
    var entries: [64]S.Entry = undefined;
    var ids: [64 * 8]u32 = undefined;
    var n: usize = 0;
    var rows: usize = 0;
    const r = span(set.streams, k, g);
    for (streams[r[0]..r[1]], r[0]..) |*x, s| {
        if (x.made >= set.tokens) continue;
        const h = x.hist.items;
        x.window[0] = h[h.len - 1];
        const d = lookup(h, x.window[1..][0..set.draft]);
        entries[n] = .{ .id = @intCast(s + 1), .op = .@"continue", .rows = @intCast(1 + d), .accepted = x.kept };
        x.rows = @intCast(1 + d);
        @memcpy(ids[rows..][0..x.rows], x.window[0..x.rows]);
        rows += x.rows;
        n += 1;
    }
    if (n == 0) return false;
    if (step) try lead.step(entries[0..n], ids[0..rows], .window) else try lead.issue(entries[0..n], ids[0..rows], .window);
    return true;
}

/// Group `g`'s logits after its round: each stream keeps its window up to the first draft greedy disagrees with.
fn land(a: std.mem.Allocator, e: *E.Engine, streams: []Stream, set: Setup, g: u32, k: u32, drafted: *usize) !usize {
    const r = span(set.streams, k, g);
    var row: usize = 0;
    var kept_drafts: usize = 0;
    for (streams[r[0]..r[1]]) |*x| {
        if (x.made >= set.tokens) continue;
        var take: u32 = 1;
        while (take < x.rows and argmax(e.rowLogits(row + take - 1)) == x.window[take]) take += 1;
        for (1..take) |i| try x.hist.append(a, x.window[i]);
        try x.hist.append(a, argmax(e.rowLogits(row + take - 1)));
        drafted.* += x.rows - 1;
        kept_drafts += take - 1;
        x.made += take;
        x.kept = take;
        row += x.rows;
    }
    return kept_drafts;
}

/// Prompt lookup: after the latest earlier match of the last token, the tokens that followed, at most `out.len`.
fn lookup(h: []const u32, out: []u32) usize {
    if (out.len == 0 or h.len < 2) return 0;
    const last = h[h.len - 1];
    var i = h.len - 1;
    while (i > 0) {
        i -= 1;
        if (h[i] != last) continue;
        const n = @min(out.len, h.len - (i + 1));
        @memcpy(out[0..n], h[i + 1 ..][0..n]);
        return n;
    }
    return 0;
}

fn argmax(row: []const u16) u32 {
    var best: usize = 0;
    var bv: f32 = -std.math.inf(f32);
    for (row, 0..) |h, i| {
        const v: f32 = @bitCast(@as(u32, h) << 16);
        if (v > bv) {
            bv = v;
            best = i;
        }
    }
    return @intCast(best);
}
