//! `tensorfold segments`: prompts prefilled at 1-4 segments a call, interleaved, every result hashed against one segment's.

const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const nemotron = @import("nemotron");

pub const Options = struct {
    files: []const []const u8,
    counts: []const usize,
    repeat: usize,
    max_tokens: usize,
    stop_eos: bool,
    profile: bool,
    report: ?[]const u8,
};

const max_repeat = 16;
const checks = 2; // hashed prefills an arm
const R = nemotron.state.prefill_rows;

/// What a prefill leaves: hashes of the caches and states, the head's caches, the last logits and hidden states; the token.
const Digest = struct {
    state: [64]u8,
    head: [64]u8,
    logits: [64]u8,
    hidden: [64]u8,
    token: u32,

    fn eql(a: Digest, b: Digest) bool {
        return std.mem.eql(u8, &a.state, &b.state) and std.mem.eql(u8, &a.head, &b.head) and
            std.mem.eql(u8, &a.logits, &b.logits) and std.mem.eql(u8, &a.hidden, &b.hidden) and a.token == b.token;
    }
};

const Arm = struct {
    segments: usize,
    prefill_s: []f64,
    median_s: f64 = 0,
    prompt_tok_s: f64 = 0,
    digest: ?Digest = null,
    steady: bool = true, // every repeat left the same digest
    same: bool = false, // equal to one segment's
    reply_sha: [64]u8 = @splat('0'),
    reply_tokens: usize = 0,
    reply_same: bool = false,
    cold_prefill_s: f64 = 0,
    decode_s: f64 = 0,
    decode_tok_s: f64 = 0,
    rounds: usize = 0,
    accepted: usize = 0,
};

pub fn run(gpa: std.mem.Allocator, io: std.Io, e: *nemotron.Engine, drafter: ?*nemotron.Drafter, o: Options) !u8 {
    if (o.files.len == 0 or o.counts.len == 0 or o.counts[0] != 1 or o.repeat < 1 or o.repeat > max_repeat) return error.BadSegmentsBench;
    var most: usize = 1;
    for (o.counts) |n| most = @max(most, n);
    try e.setSegments(most); // the streams and scratch every arm needs, made before any clock runs
    const head = if (drafter) |d| d.head else null;
    var all_exact = true;
    var reports: std.ArrayList(u8) = .empty;
    defer reports.deinit(gpa);
    try reports.appendSlice(gpa, "[");
    for (o.files, 0..) |path, fi| {
        const ids = try readIds(gpa, io, path);
        defer gpa.free(ids);
        if (ids.len + nemotron.state.max_rows > e.max_len) return error.ContextFull;
        std.debug.print("PROMPT {s}: {d} tokens, {d} chunks of {d}\n", .{ path, ids.len, (ids.len + R - 1) / R, R });
        const arms = try gpa.alloc(Arm, o.counts.len);
        for (arms, o.counts) |*a, n| a.* = .{ .segments = n, .prefill_s = &.{} };
        defer {
            for (arms) |a| gpa.free(a.prefill_s);
            gpa.free(arms);
        }
        for (arms) |*a| a.prefill_s = try gpa.alloc(f64, o.repeat);
        for (arms) |a| { // warm: every kernel loaded and every arm's scratch touched once
            try e.setSegments(a.segments);
            _ = try e.prefill(ids, null, head);
        }
        for (0..o.repeat) |r| for (arms) |*a| { // timed back to back, the arms interleaved
            try e.setSegments(a.segments);
            const t0 = std.Io.Clock.awake.now(io);
            _ = try e.prefill(ids, null, head);
            a.prefill_s[r] = nemotron.engine.seconds(io, t0);
        };
        const rows = (ids.len - 1) % R + 1;
        for (0..checks) |_| for (arms) |*a| { // hashed apart from the clock, twice an arm to catch a race
            try e.setSegments(a.segments);
            const d = try digest(gpa, e, head, try e.prefill(ids, null, head), rows);
            if (a.digest) |first| a.steady = a.steady and first.eql(d) else a.digest = d;
        };
        if (o.profile) try profile(gpa, e, ids);
        for (arms) |*a| {
            try e.setSegments(a.segments);
            const res = try nemotron.decode.generate(gpa, io, e, drafter, ids, o.max_tokens, .{ .stop_eos = o.stop_eos });
            defer gpa.free(res.tokens);
            const text = try core.ids_json.write(gpa, res.tokens);
            defer gpa.free(text);
            var h: [32]u8 = undefined;
            std.crypto.hash.sha2.Sha256.hash(text, &h, .{});
            a.reply_sha = std.fmt.bytesToHex(h, .lower);
            a.reply_tokens = res.tokens.len;
            a.cold_prefill_s = res.prefill_seconds;
            a.decode_s = res.decode_seconds;
            a.decode_tok_s = @as(f64, @floatFromInt(@max(1, res.tokens.len) - 1)) / @max(res.decode_seconds, 1e-9);
            a.rounds = res.rounds;
            a.accepted = res.accepted;
        }
        for (arms) |*a| {
            a.median_s = median(a.prefill_s);
            a.prompt_tok_s = @as(f64, @floatFromInt(ids.len)) / a.median_s;
        }
        const base = arms[0];
        for (arms) |*a| {
            a.same = a.steady and base.steady and a.digest.?.eql(base.digest.?);
            a.reply_same = std.mem.eql(u8, &a.reply_sha, &base.reply_sha) and a.rounds == base.rounds and a.accepted == base.accepted;
            all_exact = all_exact and a.same and a.reply_same;
            const d = a.digest.?;
            std.debug.print("  segments {d}: prefill median {d:.4} s ({d:.0} tok/s, {d:.3}x one segment) runs", .{ a.segments, a.median_s, a.prompt_tok_s, base.median_s / a.median_s });
            for (a.prefill_s) |s| std.debug.print(" {d:.4}", .{s});
            std.debug.print("\n    state {s} head {s} logits {s} hidden {s} first {d}: {s}\n", .{ d.state[0..12], d.head[0..12], d.logits[0..12], d.hidden[0..12], d.token, if (a.same) "SAME" else "DIFFERENT" });
            std.debug.print("    decode {d} tokens in {d:.4} s ({d:.1} tok/s) after a {d:.4} s prefill, rounds {d} accepted {d}, reply {s}: {s}\n", .{ a.reply_tokens, a.decode_s, a.decode_tok_s, a.cold_prefill_s, a.rounds, a.accepted, a.reply_sha[0..12], if (a.reply_same) "SAME" else "DIFFERENT" });
        }
        const json = try std.json.Stringify.valueAlloc(gpa, .{ .prompt = path, .tokens = ids.len, .arms = arms }, .{});
        defer gpa.free(json);
        if (fi > 0) try reports.appendSlice(gpa, ",");
        try reports.appendSlice(gpa, json);
    }
    try reports.appendSlice(gpa, "]");
    if (o.report) |p| try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = p, .data = reports.items });
    std.debug.print("RESULT segments: {s}\n", .{if (all_exact) "every arm's state, logits, first token and reply equal one segment's" else "MISMATCH"});
    return if (all_exact) 0 else 1;
}

fn median(xs: []const f64) f64 {
    var buf: [max_repeat]f64 = undefined;
    const s = buf[0..xs.len];
    @memcpy(s, xs);
    std.mem.sort(f64, s, {}, std.sort.asc(f64));
    return if (s.len % 2 == 1) s[s.len / 2] else (s[s.len / 2 - 1] + s[s.len / 2]) / 2;
}

/// Token ids from a file: integers separated by commas, spaces or newlines, optionally in brackets.
pub fn readIds(gpa: std.mem.Allocator, io: std.Io, path: []const u8) ![]u32 {
    const text = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 26));
    defer gpa.free(text);
    var out: std.ArrayList(u32) = .empty;
    errdefer out.deinit(gpa);
    var it = std.mem.tokenizeAny(u8, text, ", \t\r\n[]");
    while (it.next()) |t| try out.append(gpa, try std.fmt.parseInt(u32, t, 10));
    if (out.items.len == 0) return error.NoPromptTokens;
    return out.toOwnedSlice(gpa);
}

/// sha256 of device bytes, read back in pieces.
fn hashDevice(h: *std.crypto.hash.sha2.Sha256, gpa: std.mem.Allocator, e: *nemotron.Engine, ptr: u64, len: usize) !void {
    const piece = 64 << 20;
    const host = try gpa.alloc(u8, @min(len, piece));
    defer gpa.free(host);
    var at: usize = 0;
    while (at < len) : (at += piece) {
        const n = @min(piece, len - at);
        try e.ops().download(host[0..n], ptr + at);
        try e.stream.synchronize();
        h.update(host[0..n]);
    }
}

fn hex(h: *std.crypto.hash.sha2.Sha256) [64]u8 {
    var d: [32]u8 = undefined;
    h.final(&d);
    return std.fmt.bytesToHex(d, .lower);
}

fn digest(gpa: std.mem.Allocator, e: *nemotron.Engine, head: ?*nemotron.Head, token: u32, rows: usize) !Digest {
    const Sha = std.crypto.hash.sha2.Sha256;
    var st = Sha.init(.{});
    for (e.b.planes(), e.b.state_bytes) |p, n| try hashDevice(&st, gpa, e, p, n);
    var hd = Sha.init(.{});
    if (head) |h| {
        const sizes = h.seqSizes();
        try hashDevice(&hd, gpa, e, h.k_cache, sizes[0]);
        try hashDevice(&hd, gpa, e, h.v_cache, sizes[1]);
    }
    var lg = Sha.init(.{});
    try hashDevice(&lg, gpa, e, e.b.p_logits, e.c.vocab * 2);
    var hi = Sha.init(.{});
    try hashDevice(&hi, gpa, e, e.b.p_hidden, rows * e.c.hidden * 2);
    return .{ .state = hex(&st), .head = hex(&hd), .logits = hex(&lg), .hidden = hex(&hi), .token = token };
}

/// GPU ms of a serial chunk by block kind and part: the first chunk, then the last whole one after its predecessors.
fn profile(gpa: std.mem.Allocator, e: *nemotron.Engine, ids: []const u32) !void {
    const whole = ids.len / R;
    if (whole == 0) return;
    try e.setSegments(1);
    const L = e.w.blocks.len;
    const evs = try gpa.alloc(cuda.Event, 3 * L + 3);
    defer gpa.free(evs);
    var made: usize = 0;
    defer for (evs[0..made]) |*v| v.deinit();
    while (made < evs.len) : (made += 1) evs[made] = try cuda.Event.init(e.ctx.d, true);
    for ([_]usize{ 0, if (whole > 1) whole - 1 else 0 }, 0..) |m, pass| {
        if (pass == 1 and m == 0) break;
        if (m > 0) _ = try e.prefill(ids[0 .. m * R], null, null) else try e.reset();
        try e.copied.synchronize();
        const host = e.promptHost(R);
        @memcpy(host, ids[m * R ..][0..R]);
        try e.ops().upload(e.b.p_ids, std.mem.sliceAsBytes(host));
        try e.copied.record(e.stream);
        const f = e.forward(null);
        var w: nemotron.Walk = .{ .rows = R, .pos = m * R };
        var j: usize = 0;
        try evs[j].record(e.stream);
        try f.chunkBegin(&w);
        j += 1;
        try evs[j].record(e.stream);
        for (0..L) |i| {
            try f.chunkPre(&w, i);
            j += 1;
            try evs[j].record(e.stream);
            try f.chunkMixer(&w, i);
            j += 1;
            try evs[j].record(e.stream);
            try f.chunkPost(&w, i);
            j += 1;
            try evs[j].record(e.stream);
        }
        try f.chunkFinish(&w);
        j += 1;
        try evs[j].record(e.stream);
        try evs[j].synchronize();
        var ms: [3][3]f64 = @splat(@splat(0)); // kind (mamba, moe, attention) by part (pre, mixer, post)
        for (0..L) |i| for (0..3) |p| {
            const at = 1 + 3 * i + p;
            ms[@backingInt(e.w.blocks[i].kind)][p] += try cuda.Event.elapsedMs(evs[at], evs[at + 1]);
        };
        const total = try cuda.Event.elapsedMs(evs[0], evs[j]);
        const begin = try cuda.Event.elapsedMs(evs[0], evs[1]);
        const fin = try cuda.Event.elapsedMs(evs[j - 1], evs[j]);
        std.debug.print("  PROFILE chunk at {d} ({d} rows, one stream): {d:.2} ms; embed {d:.2}, finish {d:.2}\n", .{ m * R, R, total, begin, fin });
        const names = [_][]const u8{ "mamba", "moe", "attention" };
        const parts = [_][]const u8{ "pre (norm, in projection)", "mixer (conv+scan / cache+attention)", "post (gate norm, out projection / experts)" };
        for (names, 0..) |name, k| for (parts, 0..) |part, p| {
            if (ms[k][p] == 0) continue;
            std.debug.print("    {s:9} {s:46} {d:8.2} ms {d:5.1}%\n", .{ name, part, ms[k][p], 100 * ms[k][p] / total });
        };
    }
}
