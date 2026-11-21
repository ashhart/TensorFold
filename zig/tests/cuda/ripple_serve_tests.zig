//! Ripple served through its lane backend, bit for bit (ripple-serve): shared rounds, drafts, kept states, logprobs.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const rc = @import("ricursive_cuda");
const check = @import("check.zig");

const fwd = rc.model;
const wts = rc.weights;
const be = rc.backend;
const snap = be.snap;
const Stream = lanes.Stream;
const Backend = lanes.backend.Backend;
const Window = lanes.backend.Window;
const Verified = lanes.backend.Verified;
const LogRow = lanes.LogprobRow;

const fixture = @import("ripple_forward_tests.zig"); // the release-width weights on 10 layers

const Spec = struct { name: []const u8, prompt: usize };

const specs = [_]Spec{
    .{ .name = "D", .prompt = 37 },
    .{ .name = "A", .prompt = 507 }, // decode rows across the 512-row page edge
    .{ .name = "E", .prompt = 4093 }, // a wrapped ring; decode rows across 4,096 keys
    .{ .name = "C", .prompt = 1019 }, // kept states at 510 and 514, about the first page edge
};
const D = 0;
const A = 1;
const E = 2;
const C = 3;
/// The three streams decoding together (checks 1 and 2).
const shared = [_]usize{ D, A, E };

/// Draws a stream makes: the first (the prompt's last row), then one a round.
const steps = 8;
const max_new = 64;
const slots = 4;

/// Each draw of a stream (the token at prompt length + i) and the bf16 logits it was drawn from.
const Trace = struct {
    tokens: [steps]u32 = undefined,
    logits: [steps][]u8 = undefined,

    fn init(gpa: std.mem.Allocator, vocab: usize) !Trace {
        var t: Trace = .{};
        var made: usize = 0;
        errdefer for (t.logits[0..made]) |l| gpa.free(l);
        for (&t.logits) |*l| {
            l.* = try gpa.alloc(u8, vocab * 2);
            made += 1;
        }
        return t;
    }

    fn deinit(t: *Trace, gpa: std.mem.Allocator) void {
        for (t.logits) |l| gpa.free(l);
    }
};

/// Draws and logits equal from step `from` on, or the first difference printed.
fn sameTrace(comptime what: []const u8, args: anytype, got: *const Trace, want: *const Trace, from: usize) bool {
    for (from..steps) |i| {
        if (got.tokens[i] == want.tokens[i] and std.mem.eql(u8, got.logits[i], want.logits[i])) continue;
        const g = std.mem.bytesAsSlice(u16, got.logits[i]);
        const w = std.mem.bytesAsSlice(u16, want.logits[i]);
        var diff: usize = 0;
        for (g, w) |a, b| diff += @intFromBool(a != b);
        std.debug.print("  DIFF ", .{});
        std.debug.print(what, args);
        std.debug.print(": step {d}: token {d} for {d}, {d} of {d} logits differ\n", .{ i, got.tokens[i], want.tokens[i], diff, g.len });
        return false;
    }
    return true;
}

const Rig = struct {
    gpa: std.mem.Allocator,
    m: *fwd.Model,
    x: *be.Ripple,
    b: Backend,

    fn stream(r: Rig, id: []const u8, p: []const u32, sampling: ?lanes.Sampling) !Stream {
        return Stream.init(r.gpa, .{ .id = id, .prompt = p, .max_new = max_new, .sampling = sampling, .drafts = false });
    }

    /// The prompt pass; the slot the stream took.
    fn open(r: Rig, s: *Stream) !usize {
        try r.b.prefill(s);
        return r.x.lanes.getPtr(s).?.slot;
    }

    /// The first draw and its logits (the prompt's last row, kept in the stream's slot).
    fn first(r: Rig, s: *Stream, t: *Trace) !u32 {
        const tok = try r.b.read(try r.b.first(s, s.prompt().len));
        t.tokens[0] = tok;
        try r.m.download(r.x.lanes.getPtr(s).?.last, t.logits[0]);
        return tok;
    }

    /// One-row rounds by queue from step `from` (fed `tok`, the draw before it), each draw and its logits into `t`.
    fn decode(r: Rig, s: *Stream, t: *Trace, from: usize, tok0: u32) !void {
        var tok = tok0;
        for (from..steps) |i| {
            tok = try r.b.read(try r.b.queue(s, .{ .value = tok }, s.prompt().len + i));
            t.tokens[i] = tok;
            try r.m.download(r.m.b.rows, t.logits[i]);
        }
    }

    /// Stream `s` alone from its prompt (or its kept state); the slot it took.
    fn alone(r: Rig, s: *Stream, t: *Trace) !usize {
        const slot = try r.open(s);
        defer r.b.release(s);
        try r.decode(s, t, 1, try r.first(s, t));
        return slot;
    }

    /// Streams `ss` together: their prompt passes, first draws, then one-row windows by verify and keep (no drafts).
    fn together(r: Rig, ss: []const *Stream, ts: []Trace) !void {
        const v2 = r.m.c.vocab * 2;
        var opened: usize = 0;
        defer for (ss[0..opened]) |s| r.b.release(s);
        for (ss) |s| {
            _ = try r.open(s);
            opened += 1;
        }
        var pending: [shared.len]u32 = undefined;
        for (ss, ts, 0..) |s, *t, j| pending[j] = try r.first(s, t);
        var windows: [shared.len]Window = undefined;
        var out: [shared.len]Verified = undefined;
        var pos: [shared.len][1]u64 = undefined;
        var sampled: [shared.len][1]u32 = undefined;
        var drafts: [shared.len][1]u32 = undefined;
        const row0 = [_]u32{0};
        const paths: [shared.len][]const u32 = @splat(&row0);
        const n = ss.len;
        for (1..steps) |i| {
            for (ss, 0..) |s, j| {
                pos[j] = .{s.prompt().len + i};
                windows[j] = .{ .stream = s, .pending = pending[j], .held = 0, .tokens = &.{}, .parents = null, .positions = &pos[j] };
                out[j] = .{ .sampled = &sampled[j], .drafts = drafts[j][0..0] };
            }
            try r.b.verify(windows[0..n], out[0..n]);
            for (ts[0..n], 0..) |*t, j| {
                try r.m.download(r.m.b.rows + j * v2, t.logits[i]);
                t.tokens[i] = sampled[j][0];
                pending[j] = sampled[j][0];
            }
            try r.b.keep(windows[0..n], paths[0..n]);
        }
    }
};

/// Check 1 and 2: three streams together == each alone (`sp` null: greedy), their slots 0 to 2 against slot 0 alone.
fn sharedRounds(r: Rig, prompts: []const []u32, alone: []const Trace, sp: ?lanes.Sampling, label: []const u8) !usize {
    var ss: [shared.len]Stream = undefined;
    var made: usize = 0;
    defer for (ss[0..made]) |*s| s.deinit(r.gpa);
    var ptrs: [shared.len]*Stream = undefined;
    for (shared, &ss, &ptrs) |j, *s, *p| {
        var each = sp;
        if (each) |*q| q.seed +%= j;
        s.* = try r.stream(specs[j].name, prompts[j], each);
        made += 1;
        p.* = s;
    }
    var ts: [shared.len]Trace = undefined;
    var traced: usize = 0;
    defer for (ts[0..traced]) |*t| t.deinit(r.gpa);
    for (&ts) |*t| {
        t.* = try Trace.init(r.gpa, r.m.c.vocab);
        traced += 1;
    }
    try r.together(&ptrs, &ts);
    var bad: usize = 0;
    for (shared, &ts) |j, *t| {
        if (!sameTrace("{s} stream {s} together vs alone", .{ label, specs[j].name }, t, &alone[j], 0)) bad += 1;
    }
    if (bad == 0) {
        check.pass("ripple-serve {s} shared rounds: streams D, A, E ({d}, {d}, {d} prompt tokens) together == each alone, {d} draws and their logits each", .{ label, specs[D].prompt, specs[A].prompt, specs[E].prompt, steps });
        return 0;
    }
    std.debug.print("FAIL ripple-serve {s} shared rounds: {d} of {d} streams differ from alone\n", .{ label, bad, shared.len });
    return 1;
}

/// Check 3: a 7-row window of copied drafts with a wrong fourth keeps 4 rows; draws then == plain decode.
fn copiedDrafts(r: Rig, p: []const u32, plain: *const Trace) !usize {
    const v = r.m.c.vocab;
    var s = try r.stream("copied", p, null);
    defer s.deinit(r.gpa);
    var t = try Trace.init(r.gpa, v);
    defer t.deinit(r.gpa);
    _ = try r.open(&s);
    defer r.b.release(&s);
    const n = p.len;
    const t0 = try r.first(&s, &t);
    const wrong = 4; // the draft for prompt length + 4
    var drafts: [6]u32 = undefined;
    for (&drafts, 1..) |*d, i| d.* = plain.tokens[i];
    drafts[wrong - 1] = @intCast((plain.tokens[wrong] + 1) % v);
    var pos: [7]u64 = undefined;
    for (&pos, 1..) |*q, i| q.* = n + i;
    var sampled: [7]u32 = undefined;
    var dr: [6]u32 = undefined;
    const w: Window = .{ .stream = &s, .pending = t0, .held = 0, .tokens = &drafts, .parents = null, .positions = &pos };
    var out = [_]Verified{.{ .sampled = &sampled, .drafts = &dr }};
    try r.b.verify(&.{w}, &out);
    // rows 0 to 3 read plain tokens: their draws are plain steps 1 to 4; row 3's is the bonus past the wrong draft
    for (0..wrong) |row| {
        t.tokens[row + 1] = sampled[row];
        try r.m.download(r.m.b.rows + row * v * 2, t.logits[row + 1]);
    }
    const kept = [_]u32{ 0, 1, 2, 3 };
    try r.b.keep(&.{w}, &.{&kept});
    try r.decode(&s, &t, wrong + 1, sampled[wrong - 1]);
    const ok = std.mem.eql(u32, &dr, &drafts) and sameTrace("copied drafts", .{}, &t, plain, 0);
    if (ok) {
        check.pass("ripple-serve copied drafts: stream A, a 7-row window (positions {d} to {d}) with a wrong 4th draft keeps 4 rows; draws and logits == plain decode, and the next 3 steps too", .{ n + 1, n + 7 });
        return 0;
    }
    std.debug.print("FAIL ripple-serve copied drafts: stream A's window or the steps after it differ from plain decode\n", .{});
    return 1;
}

/// The prompt cache's hook: a state at each mark the pass stands at.
const Keep = struct {
    x: *be.Ripple,
    saved: [2]?*snap.Saved = .{ null, null },
    n: usize = 0,

    fn hook(ptr: *anyopaque, s: *Stream, at: u32) void {
        const k: *Keep = @ptrCast(@alignCast(ptr));
        if (k.n == k.saved.len) return;
        k.saved[k.n] = k.x.saveState(s, at) catch |e| blk: {
            std.debug.print("  saveState at {d}: {s}\n", .{ at, @errorName(e) });
            break :blk null;
        };
        k.n += 1;
    }
};

/// Check 4: states kept at `ats`, each restored into another slot and continued == the uninterrupted `plain`.
fn keptStates(r: Rig, name: []const u8, p: []const u32, ats: []const u32, notes: []const []const u8, plain: *const Trace) !usize {
    var keep: Keep = .{ .x = r.x };
    var saver = try Stream.init(r.gpa, .{ .id = "saver", .prompt = p, .max_new = max_new, .drafts = false, .reuse = .{ .marks = ats, .hook = .{ .ptr = &keep, .at = Keep.hook } } });
    defer saver.deinit(r.gpa);
    const from = try r.open(&saver);
    defer r.b.release(&saver); // held while the restores run, so each lands in another slot
    defer for (keep.saved[0..keep.n]) |k| if (k) |saved| r.x.dropState(saved);
    var bad: usize = 0;
    for (ats, notes, 0..) |at, note, i| {
        const saved = (if (i < keep.n) keep.saved[i] else null) orelse {
            std.debug.print("FAIL ripple-serve kept state at {d} of stream {s}: none kept\n", .{ at, name });
            bad += 1;
            continue;
        };
        var s = try Stream.init(r.gpa, .{ .id = "restored", .prompt = p, .max_new = max_new, .drafts = false, .reuse = .{ .saved = saved, .at = at } });
        defer s.deinit(r.gpa);
        var t = try Trace.init(r.gpa, r.m.c.vocab);
        defer t.deinit(r.gpa);
        const into = try r.alone(&s, &t);
        const ok = !s.reuse_failed and s.cached == at and into != from and sameTrace("state at {d}", .{at}, &t, plain, 0);
        if (ok) {
            check.pass("ripple-serve kept state at {d} of stream {s} ({s}, {d:.1} MiB): saved from slot {d}, restored into slot {d}, {d} more prompt rows and {d} draws == uninterrupted", .{ at, name, note, @as(f64, @floatFromInt(snap.bytes(r.m.c, at))) / (1 << 20), from, into, p.len - at, steps });
        } else {
            std.debug.print("FAIL ripple-serve kept state at {d} of stream {s}: reuse_failed {s}, cached {d}, slots {d} -> {d}, or draws differ\n", .{ at, name, if (s.reuse_failed) "yes" else "no", s.cached, from, into });
            bad += 1;
        }
    }
    return bad;
}

/// The Nemotron CUDA logprobs' bound against a float64 log-softmax of the same logits (CHANGELOG, #572).
const tol: f64 = 2e-6;

/// A float64 log-softmax of one bf16 logits row.
const Host = struct {
    vals: []align(1) const u16,
    lse: f64,

    fn init(row: []const u8) Host {
        var h: Host = .{ .vals = std.mem.bytesAsSlice(u16, row), .lse = 0 };
        var most = -std.math.inf(f64);
        for (0..h.vals.len) |i| most = @max(most, h.at(i));
        var sum: f64 = 0;
        for (0..h.vals.len) |i| sum += @exp(h.at(i) - most);
        h.lse = most + @log(sum);
        return h;
    }

    fn at(h: Host, i: usize) f64 {
        const f: f32 = @bitCast(@as(u32, h.vals[i]) << 16);
        return f;
    }

    fn lp(h: Host, i: usize) f64 {
        return h.at(i) - h.lse;
    }

    /// The best ids, logit descending, ties by the lower id.
    fn best(h: Host, out: []u32) void {
        var n: usize = 0;
        for (0..h.vals.len) |i| {
            const v = h.at(i);
            var j = n;
            while (j > 0 and h.at(out[j - 1]) < v) j -= 1;
            if (j == out.len) continue;
            var q = @min(n, out.len - 1);
            while (q > j) : (q -= 1) out[q] = out[q - 1];
            out[j] = @intCast(i);
            n = @min(n + 1, out.len);
        }
    }

    /// vLLM's rank: the ids whose logit is at or above the pick's, itself included.
    fn rank(h: Host, pick: u32) u32 {
        var n: u32 = 0;
        for (0..h.vals.len) |i| n += @intFromBool(h.at(i) >= h.at(pick));
        return n;
    }
};

/// A backend row against the host's: the pick, its best ids exactly, their log probabilities within `tol`.
fn rowMatches(h: Host, row: LogRow, pick: u32, k: u8, ranked: bool, worst: *f64) bool {
    var ok = row.token == pick and row.count == k;
    const e0 = @abs(@as(f64, row.logprob) - h.lp(pick));
    worst.* = @max(worst.*, e0);
    ok = ok and e0 <= tol;
    var ids: [lanes.logprob.max_top]u32 = undefined;
    h.best(ids[0..k]);
    for (0..k) |i| {
        const e = @abs(@as(f64, row.logprobs[i]) - h.lp(ids[i]));
        worst.* = @max(worst.*, e);
        ok = ok and row.ids[i] == ids[i] and e <= tol;
    }
    if (ranked) ok = ok and row.rank == h.rank(pick);
    return ok;
}

fn verdict(ok: bool, comptime what: []const u8, args: anytype, worst: f64) usize {
    std.debug.print("{s} ripple-serve logprob rows: ", .{if (ok) "PASS" else "FAIL"});
    std.debug.print(what, args);
    if (ok) {
        std.debug.print(" == a float64 log-softmax (ids exact, worst {e:.2} <= {e:.0})\n", .{ worst, tol });
        return 0;
    }
    std.debug.print(" differ from a float64 log-softmax (worst {e:.2}, bound {e:.0})\n", .{ worst, tol });
    return 1;
}

/// Check 5: the first draw's row and a verify window's rows (20 best), and prompt rows with ranks (5 best).
fn logprobRows(r: Rig, pa: []const u32, plain: *const Trace, pe: []const u32) !usize {
    const v = r.m.c.vocab;
    const row_bytes = v * 2;
    const k: u8 = lanes.logprob.max_top;
    var bad: usize = 0;
    const buf = try r.gpa.alloc(u8, fwd.head_rows * row_bytes);
    defer r.gpa.free(buf);
    {
        var s = try Stream.init(r.gpa, .{ .id = "logprobs", .prompt = pa, .max_new = max_new, .drafts = false, .logprobs = k });
        defer s.deinit(r.gpa);
        var t = try Trace.init(r.gpa, v);
        defer t.deinit(r.gpa);
        _ = try r.open(&s);
        defer r.b.release(&s);
        const t0 = try r.first(&s, &t);
        var worst: f64 = 0;
        const row = try r.b.firstRow(&s);
        bad += verdict(rowMatches(Host.init(t.logits[0]), row, t0, k, false, &worst), "the first draw's row (stream A, {d} best)", .{k}, worst);

        // a 4-row window of plain drafts, every row scored
        const n = pa.len;
        const drafts = plain.tokens[1..4];
        var pos: [4]u64 = undefined;
        for (&pos, 1..) |*q, i| q.* = n + i;
        var sampled: [4]u32 = undefined;
        var dr: [3]u32 = undefined;
        var rows: [4]LogRow = undefined;
        const w: Window = .{ .stream = &s, .pending = t0, .held = 0, .tokens = drafts, .parents = null, .positions = &pos };
        var out = [_]Verified{.{ .sampled = &sampled, .drafts = &dr, .rows = &rows }};
        try r.b.verify(&.{w}, &out);
        try r.m.download(r.m.b.rows, buf[0 .. 4 * row_bytes]);
        worst = 0;
        var ok = true;
        for (rows, sampled, 0..) |lr, pick, i| ok = rowMatches(Host.init(buf[i * row_bytes ..][0..row_bytes]), lr, pick, k, false, &worst) and ok;
        bad += verdict(ok, "a 4-row verify window's rows (stream A at {d} to {d}, {d} best)", .{ n + 1, n + 4, k }, worst);
    }
    {
        // prompt rows: the reference logits from the same prompt in a slot no stream holds, head_rows at a time
        const kp: u8 = 5;
        const slot = slots - 1;
        const scored = pe.len - 1;
        const want = try r.gpa.alloc(u8, scored * row_bytes);
        defer r.gpa.free(want);
        try r.m.prompt(slot, 0, pe);
        var r0: usize = 0;
        while (r0 < scored) : (r0 += fwd.head_rows) {
            const rows = @min(fwd.head_rows, scored - r0);
            try r.m.headRows(r0, rows);
            try r.m.download(r.m.b.rows, want[r0 * row_bytes ..][0 .. rows * row_bytes]);
        }
        try r.m.reset(slot, pe.len);
        var s = try Stream.init(r.gpa, .{ .id = "prompt-logprobs", .prompt = pe, .max_new = max_new, .drafts = false, .prompt_logprobs = kp });
        defer s.deinit(r.gpa);
        try r.b.prefill(&s);
        r.b.release(&s);
        var worst: f64 = 0;
        var ok = s.prompt_rows.items.len == scored;
        if (ok) {
            for (s.prompt_rows.items, 0..) |lr, i| ok = rowMatches(Host.init(want[i * row_bytes ..][0..row_bytes]), lr, pe[i + 1], kp, true, &worst) and ok;
        }
        bad += verdict(ok, "{d} prompt rows with ranks ({d} best, {d}-row head batches)", .{ scored, kp, fwd.head_rows }, worst);
    }
    return bad;
}

pub fn run(gpu: check.Gpu) !void {
    const a = gpu.gpa;
    const c = fixture.config();
    try wts.gate(&c);
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    const t0 = check.now(gpu.io);
    var w = try fixture.build(gpu, c, 0x7269_7070);
    defer w.deinit();
    std.debug.print("ripple-serve: weights built in {d:.1} s\n", .{@as(f64, @floatFromInt(check.now(gpu.io) - t0)) / 1e9});
    var k = try fwd.Kernels.load(gpu.ctx);
    defer k.deinit();
    var m = try fwd.Model.init(a, gpu.d, s, &w, &k, c.context, slots, be.round_rows);
    defer m.deinit();
    var x = try be.Ripple.init(a, gpu.d, &m);
    defer x.deinit();
    const r: Rig = .{ .gpa = a, .m = &m, .x = &x, .b = x.backend() };

    var prompts: [specs.len][]u32 = undefined;
    var made: usize = 0;
    defer for (prompts[0..made]) |p| a.free(p);
    var prng = std.Random.DefaultPrng.init(0x7365_7276);
    for (specs, &prompts) |sp, *p| {
        p.* = try a.alloc(u32, sp.prompt);
        made += 1;
        for (p.*) |*t| t.* = prng.random().uintLessThan(u32, @intCast(c.vocab));
    }
    const sampling: lanes.Sampling = .{ .seed = 0x5eed, .temperature = 1.0, .top_p = 0.95, .top_k = 20 };

    // each stream alone in a fresh slot, greedy (all four) and sampled (the shared three, a seed each)
    var greedy: [specs.len]Trace = undefined;
    var sampled: [specs.len]Trace = undefined;
    var traced: usize = 0;
    defer for (greedy[0..traced], sampled[0..traced]) |*g, *q| {
        g.deinit(a);
        q.deinit(a);
    };
    for (&greedy, &sampled) |*g, *q| {
        g.* = try Trace.init(a, c.vocab);
        q.* = Trace.init(a, c.vocab) catch |e| {
            g.deinit(a);
            return e;
        };
        traced += 1;
    }
    for (specs, prompts, &greedy, 0..) |sp, p, *g, j| {
        var st = try r.stream(sp.name, p, null);
        defer st.deinit(a);
        _ = try r.alone(&st, g);
        if (std.mem.indexOfScalar(usize, &shared, j) == null) continue;
        var each = sampling;
        each.seed +%= j;
        var sq = try r.stream(sp.name, p, each);
        defer sq.deinit(a);
        _ = try r.alone(&sq, &sampled[j]);
    }

    var fails: usize = 0;
    fails += try sharedRounds(r, &prompts, &greedy, null, "greedy");
    fails += try sharedRounds(r, &prompts, &sampled, sampling, "sampled (temperature 1, top-p 0.95, top-k 20)");
    fails += try copiedDrafts(r, prompts[A], &greedy[A]);
    fails += try keptStates(r, "E", prompts[E], &.{3500}, &.{"past the 3,072-row ring's wrap"}, &greedy[E]);
    fails += try keptStates(r, "C", prompts[C], &.{ 510, 514 }, &.{ "two rows before a 512-row page edge", "two rows across a 512-row page edge" }, &greedy[C]);
    fails += try logprobRows(r, prompts[A], &greedy[A], prompts[E][0..300]);
    try check.expect(fails == 0, "ripple-serve: {d} checks differ", .{fails});
}
