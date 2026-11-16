//! #1594 step 2b's Mac half: a prompt state in hybrid_state's order restored into slot 0, its tail prefilled, logits.
const std = @import("std");
const mtl = @import("metal");
const q = @import("tensorfold").qwen27;
const host = @import("qwen27_served");
const hs = @import("fabric").hybrid_state;

const help =
    \\tf-qwen27-import MODEL IDS.json export AT OUT: this Mac's state after AT tokens, in hybrid_state's order, to OUT
    \\tf-qwen27-import MODEL IDS.json import FILE exact|close: FILE restored, the rest prefilled, logits against fresh
    \\tf-qwen27-import MODEL IDS.json link FILE exact|close: as import, FILE served and pulled over the fake MKVH link
    \\tf-qwen27-import MODEL IDS.json remote URL,LINK exact|close: as import, the GB10 server's export pulled over LINK
    \\tf-qwen27-import MODEL IDS.json greedy N OUT.json: the prompt and this Mac's N-token greedy reply, for `tail`
    \\tf-qwen27-import MODEL IDS.json floor ROWS.bf16 N: this Mac's N steps against the GB10's `tail` rows
    \\tf-qwen27-import MODEL IDS.json time FILE -: stage times, this Mac alone against FILE restored, 32 decode steps each
    \\tf-qwen27-import MODEL IDS.json agree file:FILE|remote:URL,LINK N OUT.json: Ash's token check, top-2 margins
    \\tf-qwen27-import MODEL PROMPTS.json multi DIR GEN: N streams alone, then restored from DIR/s<k>.tfhs, GEN tokens each
    \\                (DIR "-": the Mac-alone half only, a production control for another method's prompts)
;

fn read(a: std.mem.Allocator, r: *const q.decode_round.Runner, st: *q.snapshot.State) ![]u8 {
    try st.landed(); // a save's blit is not waited for: it lands before the copy reads it
    const buffer = try r.model.device.buffer(st.bytes, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    defer buffer.deinit();
    const cb = r.model.queue.commandBuffer();
    const e = cb.blit();
    e.copy(st.buf, 0, buffer, 0, st.bytes);
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure() != null) return error.SnapshotGpuFailure;
    return a.dupe(u8, buffer.contents()[0..st.bytes]);
}

pub fn logits(a: std.mem.Allocator, r: *const q.decode_round.Runner) ![]f32 {
    const ref = r.model.frame.get(.logits);
    const raw = ref.buffer.contents()[ref.offset..][0 .. r.model.config.vocab * 2];
    const out = try a.alloc(f32, r.model.config.vocab);
    for (out, 0..) |*x, i| x.* = @bitCast(@as(u32, std.mem.readInt(u16, raw[i * 2 ..][0..2], .little)) << 16);
    return out;
}

fn argmax(xs: []const f32) usize {
    var best: usize = 0;
    for (xs, 0..) |x, i| if (x > xs[best]) {
        best = i;
    };
    return best;
}

/// KL(p || q) of the two softmaxes, in nats.
fn kl(p: []const f32, qq: []const f32) f64 {
    var mp: f64 = -std.math.inf(f64);
    var mq: f64 = -std.math.inf(f64);
    for (p, qq) |x, y| {
        mp = @max(mp, x);
        mq = @max(mq, y);
    }
    var zp: f64 = 0;
    var zq: f64 = 0;
    for (p, qq) |x, y| {
        zp += @exp(@as(f64, x) - mp);
        zq += @exp(@as(f64, y) - mq);
    }
    var d: f64 = 0;
    for (p, qq) |x, y| {
        const lp = @as(f64, x) - mp - @log(zp);
        const lq = @as(f64, y) - mq - @log(zq);
        d += @exp(lp) * (lp - lq);
    }
    return d;
}

/// `n` steps from the slot's last logits, each kept; each feeds its argmax, or `forced`'s token (teacher forcing).
fn steps(a: std.mem.Allocator, r: *q.decode_round.Runner, n: usize, forced: ?[]const u32) !struct { toks: []u32, logits: [][]f32 } {
    const toks = try a.alloc(u32, n);
    const all = try a.alloc([]f32, n);
    for (toks, all, 0..) |*t, *l, i| {
        l.* = try logits(a, r);
        t.* = if (forced) |f| f[i] else @intCast(argmax(l.*));
        try (q.session.Session{ .runner = r }).promptChunk(&.{t.*}, true);
    }
    return .{ .toks = toks, .logits = all };
}

/// `ids` as prompt chunks of at most the frame's rows, the last one with logits if `last`.
pub fn feed(r: *q.decode_round.Runner, ids: []const u32, last: bool) !void {
    return feedSlot(r, 0, ids, last);
}

fn feedSlot(r: *q.decode_round.Runner, slot: u32, ids: []const u32, last: bool) !void {
    const cap = r.model.frame.capacity;
    var at: usize = 0;
    while (at < ids.len) {
        const end = @min(ids.len, at + cap);
        try (q.session.Session{ .runner = r, .slot = slot }).promptChunk(ids[at..end], last and end == ids.len);
        at = end;
    }
}

/// The most KL a restored state may cost at any of the steps: zeroed or swapped states cost far more (see the receipt).
const close_kl = 0.02;
const n_steps = 32;

/// The state's bytes in this order must be exactly what this engine snapshots, or the geometry is not this model's.
pub fn sized(r: *const q.decode_round.Runner, at: u32) !void {
    if (q.snapshot.slotBytes(r, null, at) != hs.qwen27.bytes(at)) return error.HybridGeometryDiffers;
}

fn exportState(a: std.mem.Allocator, io: std.Io, r: *q.decode_round.Runner, ids: []const u32, at: u32, out: []const u8) !void {
    try sized(r, at);
    try r.reset(0);
    try feed(r, ids[0..at], false);
    const st = try q.snapshot.saveSlot(a, r, 0, null, at, null);
    defer q.snapshot.drop(a, st);
    const body = try read(a, r, st);
    defer a.free(body);
    const header = hs.Header.of(hs.qwen27, at, ids);
    var f = try std.Io.Dir.cwd().createFile(io, out, .{});
    defer f.close(io);
    try f.writeStreamingAll(io, std.mem.asBytes(&header));
    try f.writeStreamingAll(io, body);
    std.debug.print("RESULT export: Metal state after {d} of {d} tokens, {d} bytes, to {s}\n", .{ at, ids.len, body.len, out });
}

/// A state file read whole in 1 GiB pieces: macOS refuses a single read over 2 GiB, which std's readFileAlloc makes.
pub fn readState(a: std.mem.Allocator, io: std.Io, path: []const u8) ![]u8 {
    const f = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer f.close(io);
    const len = try f.length(io);
    if (len > 16 << 30) return error.StateFileTooBig;
    const buf = try a.alloc(u8, @intCast(len));
    errdefer a.free(buf);
    var at: usize = 0;
    while (at < buf.len) {
        const got = try f.readPositional(io, &.{buf[at..@min(buf.len, at + (1 << 30))]}, at);
        if (got == 0) return error.StateFileShort;
        at += got;
    }
    return buf;
}

fn importState(a: std.mem.Allocator, io: std.Io, r: *q.decode_round.Runner, ids: []const u32, path: []const u8, exact: bool, link: bool) !void {
    const file = try readState(a, io, path);
    defer a.free(file);
    if (file.len < @sizeOf(hs.Header)) return error.NotAHybridState;
    var header: hs.Header = undefined;
    @memcpy(std.mem.asBytes(&header), file[0..@sizeOf(hs.Header)]);
    try header.check(hs.qwen27, ids);
    const body = file[@sizeOf(hs.Header)..];
    if (body.len != header.total) return error.HybridStateLength;
    try sized(r, header.at);
    // the Mac's own pass over the whole prompt: what the restored state must lead to
    try r.reset(0);
    try feed(r, ids, true);
    const want = try steps(a, r, n_steps, null); // the Mac's own greedy reply, every step's logits
    // a blank state filled with the file's bytes (or pulled over the fake link), restored over the prompt's slot
    const st = if (link) try @import("import_link.zig").pulled(a, io, r, ids, header.at, body) else try q.snapshot.blank(a, r, null, header.at);
    defer q.snapshot.drop(a, st);
    if (!link) @memcpy(st.buf.contents()[0..body.len], body);
    try q.snapshot.restoreSlot(a, r, 0, null, st);
    try feed(r, ids[header.at..], true);
    const got = try steps(a, r, n_steps, want.toks); // the same reply fed through the restored state
    return verdict(want.logits, got.logits, want.toks, header.at, ids.len - header.at, exact);
}

/// Every step's logits through the restored state against the Mac's own: bit-equal (exact) or within close_kl.
fn verdict(want: []const []f32, got: []const []f32, toks: []const u32, at: u32, prefilled: usize, exact: bool) !void {
    var same = true;
    var worst: f32 = 0;
    var d: f64 = 0; // the most KL at any step
    var sum: f64 = 0;
    var agree: usize = 0;
    for (want, got, toks) |x, y, t| {
        same = same and std.mem.eql(f32, x, y);
        for (x, y) |u, v| worst = @max(worst, @abs(u - v));
        const k = kl(x, y);
        d = @max(d, k);
        sum += k;
        agree += @intFromBool(argmax(y) == t);
    }
    const tw = toks[0];
    const tg = argmax(got[0]);
    std.debug.print("RESULT import: restored {d} tokens, prefilled {d}; over {d} teacher-forced steps logits {s}, worst |diff| {d:.4}, KL max {e:.3} mean {e:.3} nats, top-1 {d} of {d}; first token {d} vs {d}\n", .{ at, prefilled, n_steps, if (same) "bit-equal" else "differ", worst, d, sum / n_steps, agree, n_steps, tg, tw });
    if (exact and !same) return error.ImportNotExact;
    if (tw != tg) return error.ImportFirstTokenDiffers;
    if (d > close_kl) return error.ImportTooFar;
    std.debug.print("PASS import ({s})\n", .{if (exact) "exact" else "close: same first token, every step's KL within 0.02 nats"});
}

/// The real link: the GB10 server exports `ids` at the policy's cut, pulled over LINK and checked as `import` checks.
fn remoteState(a: std.mem.Allocator, io: std.Io, r: *q.decode_round.Runner, ids: []const u32, target: []const u8, dir: []const u8, exact: bool) !void {
    const comma = std.mem.indexOfScalar(u8, target, ',') orelse return error.Usage;
    try r.reset(0);
    try feed(r, ids, true);
    const want = try steps(a, r, n_steps, null); // the Mac's own greedy reply, every step's logits
    const p = try @import("import_link.zig").pulledRemote(a, io, r, ids, target[0..comma], target[comma + 1 ..], dir);
    defer q.snapshot.drop(a, p.st);
    try sized(r, p.at);
    try q.snapshot.restoreSlot(a, r, 0, null, p.st);
    try feed(r, ids[p.at..], true);
    const got = try steps(a, r, n_steps, want.toks);
    return verdict(want.logits, got.logits, want.toks, p.at, ids.len - p.at, exact);
}

/// The prompt and all but the last of the Mac's `n` greedy tokens, for the GB10's `tail` with k = n: the same n steps.
fn greedyTo(a: std.mem.Allocator, io: std.Io, r: *q.decode_round.Runner, ids: []const u32, n: usize, out: []const u8) !void {
    try r.reset(0);
    try feed(r, ids, true);
    const got = try steps(a, r, n, null);
    var list: std.ArrayList(u32) = .empty;
    try list.appendSlice(a, ids);
    try list.appendSlice(a, got.toks[0 .. n - 1]); // `tail` with k = n then feeds ids' last token and n - 1 replies
    var w: std.Io.Writer.Allocating = .init(a);
    try std.json.Stringify.value(list.items, .{}, &w.writer);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = out, .data = w.written() });
    std.debug.print("RESULT greedy: {d} prompt tokens and the Mac's {d}-token reply to {s}\n", .{ ids.len, n, out });
}

/// The backends' own gap: the Mac's steps against the GB10's bf16 rows for the same tokens, with no hand-off.
fn floor(a: std.mem.Allocator, io: std.Io, r: *q.decode_round.Runner, ids: []const u32, path: []const u8, n: usize) !void {
    const raw = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(4 << 30));
    const vocab = r.model.config.vocab;
    if (raw.len != n * vocab * 2) return error.TailRowsDiffer;
    try r.reset(0);
    try feed(r, ids, true);
    const want = try steps(a, r, n, null);
    var d: f64 = 0;
    var sum: f64 = 0;
    var agree: usize = 0;
    const row = try a.alloc(f32, vocab);
    for (want.logits, want.toks, 0..) |x, t, i| {
        for (row, 0..) |*y, j| y.* = @bitCast(@as(u32, std.mem.readInt(u16, raw[(i * vocab + j) * 2 ..][0..2], .little)) << 16);
        const k = kl(x, row);
        d = @max(d, k);
        sum += k;
        agree += @intFromBool(argmax(row) == t);
    }
    std.debug.print("RESULT floor: GB10-only against Mac-only over {d} steps, KL max {e:.3} mean {e:.3} nats, top-1 {d} of {d}\n", .{ n, d, sum / @as(f64, @floatFromInt(n)), agree, n });
}

/// One stream's stage times: this Mac alone, then a restored state's restore and tail, each with 32 decode steps.
fn timeRun(a: std.mem.Allocator, io: std.Io, r: *q.decode_round.Runner, ids: []const u32, path: []const u8) !void {
    const clock = std.Io.Clock.awake;
    const t_read = clock.now(io).toNanoseconds();
    const file = try readState(a, io, path);
    var header: hs.Header = undefined;
    if (file.len < @sizeOf(hs.Header)) return error.NotAHybridState;
    @memcpy(std.mem.asBytes(&header), file[0..@sizeOf(hs.Header)]);
    try header.check(hs.qwen27, ids);
    const body = file[@sizeOf(hs.Header)..];
    if (body.len != header.total) return error.HybridStateLength;
    try sized(r, header.at);
    const read_ns = clock.now(io).toNanoseconds() - t_read;
    const session = q.session.Session{ .runner = r };
    try r.reset(0); // a warm-up prompt first, so neither timed pass pays the first launch
    try feed(r, ids[0..@min(ids.len, 128)], true);
    try r.reset(0);
    const t0 = clock.now(io).toNanoseconds();
    try feed(r, ids, true);
    const t1 = clock.now(io).toNanoseconds();
    var tok: u32 = try session.greedy();
    const want_first = tok;
    for (0..n_steps) |_| {
        try session.step(tok);
        tok = try session.greedy();
    }
    const t2 = clock.now(io).toNanoseconds();
    const t3 = clock.now(io).toNanoseconds();
    const st = try q.snapshot.blank(a, r, null, header.at);
    defer q.snapshot.drop(a, st);
    @memcpy(st.buf.contents()[0..body.len], body);
    try q.snapshot.restoreSlot(a, r, 0, null, st);
    const t4 = clock.now(io).toNanoseconds();
    try feed(r, ids[header.at..], true);
    const t5 = clock.now(io).toNanoseconds();
    tok = try session.greedy();
    const got_first = tok;
    for (0..n_steps) |_| {
        try session.step(tok);
        tok = try session.greedy();
    }
    const t6 = clock.now(io).toNanoseconds();
    const ms = struct {
        fn of(ns: anytype) f64 {
            return @as(f64, @floatFromInt(ns)) / 1e6;
        }
    }.of;
    const rate = struct {
        fn of(ns: anytype) f64 {
            return @as(f64, @floatFromInt(n_steps)) / (@as(f64, @floatFromInt(ns)) / 1e9);
        }
    }.of;
    std.debug.print("RESULT time: {d} tokens. Mac alone: prefill {d:.1} ms, decode {d:.2} tok/s. Restored: state read {d:.1} ms (not counted), restore {d:.1} ms, tail of {d} tokens {d:.1} ms, decode {d:.2} tok/s; first token {d} vs {d}\n", .{ ids.len, ms(t1 - t0), rate(t2 - t1), ms(read_ns), ms(t4 - t3), ids.len - header.at, ms(t5 - t4), rate(t6 - t5), got_first, want_first });
}

/// `gen` one-row rounds shared by every slot from its pending token `toks[k]`: each slot's greedy tokens into `out`.
fn rounds(a: std.mem.Allocator, r: *q.decode_round.Runner, toks: []u32, gen: usize, out: [][]u32) !void {
    const vocab: u32 = @intCast(r.model.config.vocab);
    var inputs: [q.attention_plan.max_streams]q.round_plan.Input = undefined;
    var ids: [q.attention_plan.max_streams][1]u32 = undefined;
    const row: [1]u32 = .{0};
    var paths: [q.attention_plan.max_streams][]const u32 = undefined;
    for (0..gen) |g| {
        for (toks, 0..) |t, k| {
            ids[k] = .{t};
            inputs[k] = .{ .slot = @intCast(k), .start = r.offsets[k], .capacity = r.capacity, .ids = &ids[k] };
            paths[k] = &row;
        }
        var round = try q.round_plan.Round.init(a, inputs[0..toks.len], @intCast(r.model.config.conv_kernel), r.slots, vocab);
        defer round.deinit();
        try r.verifyHead(&round, .all);
        const logits_words = r.model.frame.get(.logits).buffer.slice(u16, (toks.len + 1) * vocab);
        for (toks, 0..) |*t, k| {
            const first = round.firsts[k];
            t.* = try q.session.argmax(logits_words[first * vocab ..][0..vocab]);
            out[k][g] = t.*;
        }
        try r.keep(&round, paths[0..toks.len]);
    }
}

/// N streams to `gen` tokens: this Mac alone, then each restored from DIR/s<k>.tfhs with its tail; the same rounds.
fn multi(a: std.mem.Allocator, io: std.Io, r: *q.decode_round.Runner, prompts: []const []const u32, dir: []const u8, gen: usize) !void {
    const clock = std.Io.Clock.awake;
    const n = prompts.len;
    const bodies = try a.alloc([]const u8, n);
    const cuts = try a.alloc(u32, n);
    const alone_only = std.mem.eql(u8, dir, "-"); // the production Mac-alone control only, no state files
    for (prompts, 0..) |p, k| {
        if (alone_only) break;
        const path = try std.fmt.allocPrint(a, "{s}/s{d}.tfhs", .{ dir, k });
        const file = try readState(a, io, path);
        if (file.len < @sizeOf(hs.Header)) return error.NotAHybridState;
        var header: hs.Header = undefined;
        @memcpy(std.mem.asBytes(&header), file[0..@sizeOf(hs.Header)]);
        try header.check(hs.qwen27, p);
        if (file.len - @sizeOf(hs.Header) != header.total) return error.HybridStateLength;
        try sized(r, header.at);
        bodies[k] = file[@sizeOf(hs.Header)..];
        cuts[k] = header.at;
    }
    for (0..r.slots) |k| try r.reset(@intCast(k)); // a warm-up of both kinds of pass, so neither timed one pays it
    try feedSlot(r, 0, prompts[0][0..@min(prompts[0].len, 128)], true);
    var warm = [_]u32{try (q.session.Session{ .runner = r }).greedy()};
    const warm_out = try a.alloc([]u32, 1);
    warm_out[0] = try a.alloc(u32, 2);
    try rounds(a, r, &warm, 2, warm_out);
    const alone_toks = try a.alloc(u32, n);
    const alone = try a.alloc([]u32, n);
    const back_toks = try a.alloc(u32, n);
    const back = try a.alloc([]u32, n);
    for (alone, back) |*x, *y| {
        x.* = try a.alloc(u32, gen);
        y.* = try a.alloc(u32, gen);
    }
    for (0..r.slots) |k| try r.reset(@intCast(k));
    const t0 = clock.now(io).toNanoseconds();
    if (alone_only) std.debug.print("PHASE kind=prefill edge=begin awake_ns={d}\n", .{t0});
    for (prompts, 0..) |p, k| {
        try feedSlot(r, @intCast(k), p, true);
        alone_toks[k] = try (q.session.Session{ .runner = r, .slot = @intCast(k) }).greedy();
    }
    const t1 = clock.now(io).toNanoseconds();
    if (alone_only) std.debug.print("PHASE kind=prefill edge=end awake_ns={d}\nPHASE kind=decode edge=begin awake_ns={d}\n", .{ t1, t1 });
    const firsts = try a.dupe(u32, alone_toks);
    try rounds(a, r, alone_toks, gen, alone);
    const t2 = clock.now(io).toNanoseconds();
    if (alone_only) std.debug.print("PHASE kind=decode edge=end awake_ns={d}\n", .{t2});
    if (alone_only) {
        for (firsts, alone, 0..) |f, toks, k| { // each stream's first token, then its `gen` greedy tokens
            std.debug.print("TOKENS stream={d} ids={d}", .{ k, f });
            for (toks) |t| std.debug.print(",{d}", .{t});
            std.debug.print("\n", .{});
        }
        var prompt_tokens: usize = 0;
        for (prompts) |p| prompt_tokens += p.len;
        const ps = @as(f64, @floatFromInt(t1 - t0)) / 1e9;
        const ds = @as(f64, @floatFromInt(t2 - t1)) / 1e9;
        std.debug.print("RESULT alone: {d} streams, {d} prompt tokens, {d} generated each. Mac alone: prefill {d:.3} s, decode {d:.3} s = {d:.2} tok/s\n", .{ n, prompt_tokens, gen, ps, ds, @as(f64, @floatFromInt(n * gen)) / ds });
        return;
    }
    for (0..r.slots) |k| try r.reset(@intCast(k));
    const t3 = clock.now(io).toNanoseconds();
    for (prompts, 0..) |p, k| {
        const st = try q.snapshot.blank(a, r, null, cuts[k]);
        defer q.snapshot.drop(a, st);
        @memcpy(st.buf.contents()[0..bodies[k].len], bodies[k]);
        try q.snapshot.restoreSlot(a, r, @intCast(k), null, st);
        try feedSlot(r, @intCast(k), p[cuts[k]..], true);
        back_toks[k] = try (q.session.Session{ .runner = r, .slot = @intCast(k) }).greedy();
    }
    const t4 = clock.now(io).toNanoseconds();
    var same_first: usize = 0;
    for (firsts, back_toks) |x, y| same_first += @intFromBool(x == y);
    try rounds(a, r, back_toks, gen, back);
    const t5 = clock.now(io).toNanoseconds();
    var shared: usize = 0; // tokens equal to the Mac-alone reply before the first difference, over all streams
    for (alone, back) |x, y| shared += std.mem.indexOfDiff(u32, x, y) orelse gen;
    var tokens: usize = 0;
    for (prompts) |p| tokens += p.len;
    const s_of = struct {
        fn of(ns: anytype) f64 {
            return @as(f64, @floatFromInt(ns)) / 1e9;
        }
    }.of;
    const total: f64 = @floatFromInt(n * gen);
    std.debug.print("RESULT multi: {d} streams, {d} prompt tokens, {d} generated each. Mac alone: prefill {d:.3} s, decode {d:.3} s = {d:.2} tok/s. Restored: restore and tails {d:.3} s, decode {d:.3} s = {d:.2} tok/s. First tokens equal {d} of {d}; greedy equal {d} of {d} tokens\n", .{ n, tokens, gen, s_of(t1 - t0), s_of(t2 - t1), total / s_of(t2 - t1), s_of(t4 - t3), s_of(t5 - t4), total / s_of(t5 - t4), same_first, n, shared, n * gen });
}

pub fn main(init: std.process.Init) !void {
    const a = init.arena.allocator();
    const args = try init.minimal.args.toSlice(a);
    if (args.len != 6 and !(args.len == 7 and std.mem.eql(u8, args[3], "agree"))) {
        std.debug.print("{s}", .{help});
        return error.Usage;
    }
    const text = try std.Io.Dir.cwd().readFileAlloc(init.io, args[2], a, .limited(64 << 20));
    if (std.mem.eql(u8, args[3], "multi")) { // PROMPTS.json is a list of prompts, one stream each
        const prompts = (try std.json.parseFromSlice([][]u32, a, text, .{})).value;
        if (prompts.len == 0 or prompts.len > q.attention_plan.max_streams) return error.BadStreams;
        const gen = try std.fmt.parseInt(usize, args[5], 10);
        var longest: usize = 0;
        for (prompts) |p| longest = @max(longest, p.len);
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        var why: []const u8 = "";
        const h = try host.open(init.gpa, init.io, args[1], .{ .context = @intCast(longest + gen + 64), .streams = @intCast(prompts.len), .cache_gib = 0 }, a, &why);
        defer host.close(h);
        if (h.runner.slots < prompts.len) return error.SlotsDidNotFit;
        for (0..prompts.len) |k| { // each slot's K/V rows, taken as the server takes them on first use, made resident
            const added = try h.runner.ensure(@intCast(k));
            if (h.back.grow) |g| g.added(g.ctx, added);
        }
        return multi(a, init.io, &h.runner, prompts, args[4], gen);
    }
    const ids = (try std.json.parseFromSlice([]u32, a, text, .{})).value;
    if (ids.len < 2) return error.PromptTooShort;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    var why: []const u8 = "";
    const reply: usize = if (args.len == 7) try std.fmt.parseInt(usize, args[5], 10) else 64;
    const h = try host.open(init.gpa, init.io, args[1], .{ .context = @intCast(ids.len + reply + 64), .streams = 1, .cache_gib = 0 }, a, &why);
    defer host.close(h);
    for (ids) |id| if (id >= h.model.config.vocab) return error.BadToken;
    if (std.mem.eql(u8, args[3], "export")) {
        const at = try std.fmt.parseInt(u32, args[4], 10);
        if (at == 0 or at >= ids.len) return error.MarkOutOfRange;
        return exportState(a, init.io, &h.runner, ids, at, args[5]);
    }
    if (std.mem.eql(u8, args[3], "agree")) return @import("import_agree.zig").agree(a, init.io, &h.runner, ids, args[4], args[1], reply, args[6]);
    if (std.mem.eql(u8, args[3], "greedy")) return greedyTo(a, init.io, &h.runner, ids, try std.fmt.parseInt(usize, args[4], 10), args[5]);
    if (std.mem.eql(u8, args[3], "time")) return timeRun(a, init.io, &h.runner, ids, args[4]);
    if (std.mem.eql(u8, args[3], "floor")) return floor(a, init.io, &h.runner, ids, args[4], try std.fmt.parseInt(usize, args[5], 10));
    if (std.mem.eql(u8, args[3], "import")) return importState(a, init.io, &h.runner, ids, args[4], std.mem.eql(u8, args[5], "exact"), false);
    if (std.mem.eql(u8, args[3], "link")) return importState(a, init.io, &h.runner, ids, args[4], std.mem.eql(u8, args[5], "exact"), true);
    if (std.mem.eql(u8, args[3], "remote")) return remoteState(a, init.io, &h.runner, ids, args[4], args[1], std.mem.eql(u8, args[5], "exact"));
    std.debug.print("{s}", .{help});
    return error.Usage;
}
