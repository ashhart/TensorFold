//! Speed-up mode's rank 1: rank 0's requests run here in step, and this Mac keeps its own states under rank 0's names.
const std = @import("std");
const platform = @import("sys");
const engine = @import("engine.zig");
const snapshot = @import("snapshot.zig");
const fz = @import("replay.zig");
const think = @import("think.zig");
const Engine = engine.Engine;

/// Rank 0's requests as they come, until its empty request; between them a free buffer is readied inside the budget.
pub fn run(e: *Engine) !void {
    while (try one(e, null)) |_| {
        const t0 = platform.c.mach_absolute_time();
        const got = e.snap_pool.ready(e.r.device, engine.NEXT_TURN, .{ .ctx = e.r.tp.?, .check = requestWaiting }, e.peer_budget -| e.peer_held -| e.snap_pool.spare());
        if (got.cap > 0) std.log.info("speed-up rank 1: readied {d} of {d} MiB for the next save in {d:.1} ms", .{ got.touched >> 20, got.cap >> 20, msSince(t0) });
    }
}

fn requestWaiting(ctx: *anyopaque) bool {
    const tp: *fz.Tp2 = @ptrCast(@alignCast(ctx));
    return tp.requestWaiting();
}

fn msSince(t0: u64) f64 {
    return @as(f64, @floatFromInt(platform.c.mach_absolute_time() - t0)) * 125 / 3 / 1e6; // 125/3 ns ticks
}

/// Rank 0's request for `one`: a 16-word head, then the prompt, marks, dropped states' keys and the budget.
pub fn send(e: *Engine, tp: *fz.Tp2, prompt: []const u32, from: usize, marks: []const u32, max_tokens: usize, eos: []const u32, depth: ?usize, budget: think.Think) !void {
    var head: [16]u32 = @splat(0);
    head[0] = @intCast(prompt.len);
    head[1] = @intCast(@min(max_tokens, std.math.maxInt(u32)));
    head[2] = @intCast(@min(eos.len, 8));
    head[3] = if (depth) |d| @intCast(d) else std.math.maxInt(u32);
    for (eos[0..head[2]], 0..) |t, i| head[4 + i] = t;
    head[12] = @intCast(from); // rank 1 restores its own state at the same tokens
    head[13] = @intCast(marks.len); // the marks, then the dropped states' keys, follow the prompt
    head[14] = @intCast(e.peer_drops.items.len);
    head[15] = @intCast(e.pair_min); // rank 1 splits its calls the same way
    const at_budget = prompt.len + marks.len + 2 * e.peer_drops.items.len;
    const words = try e.gpa.alloc(u32, at_budget + think.words(budget));
    defer e.gpa.free(words);
    @memcpy(words[0..prompt.len], prompt);
    @memcpy(words[prompt.len..][0..marks.len], marks);
    for (e.peer_drops.items, 0..) |k, i| words[prompt.len + marks.len + 2 * i ..][0..2].* = .{ @truncate(k), @truncate(k >> 32) };
    think.put(budget, words[at_budget..]);
    try tp.sendRequest(std.mem.asBytes(&head), words);
    e.peer_drops.clearRetainingCapacity();
}

/// Run one request in step, preserving rank 0's named states; null when rank 0 closes.
pub fn one(e: *Engine, out: ?engine.Out) !?engine.Result {
    const tp = e.r.tp orelse return error.NotSpeedUpMode;
    const req = (try tp.waitRequest()) orelse return null;
    tp.request_gate.begin();
    defer tp.request_gate.end();
    var head: [16]u32 = undefined;
    @memcpy(std.mem.asBytes(&head), req.head[0..64]);
    if (head[0] == 0) return null;
    var entered_generate = false;
    defer if (!entered_generate) tp.finishStats();
    const n = head[0];
    const at_budget = n + head[13] + 2 * head[14];
    // Copy tokens before rank 0 can publish its next request.
    const words = try e.gpa.dupe(u32, req.tokens[0 .. at_budget + 3 + @min(req.tokens[at_budget + 2], think.MAX_CLOSE)]);
    defer e.gpa.free(words);
    const prompt = words[0..n];
    const marks = words[n..][0..head[13]];
    const budget = try think.take(words[at_budget..]);
    const from = head[12];
    if (head[15] == 0) return error.TpProtocol; // rank 0 sends its positive pair minimum with every request
    e.pair_min = head[15];
    const t0 = platform.c.mach_absolute_time();
    const ok = from == 0 or blk: {
        const st = e.peer_kept.get(engine.keyOf(prompt[0 .. from + 1])) orelse break :blk false;
        snapshot.restore(e, st) catch break :blk false;
        break :blk true;
    };
    const restore_ms = msSince(t0);
    try tp.ackRequest(ok);
    for (0..head[14]) |i| { // Drop rank 0's released states after restoring this request's resume.
        const w = words[n + head[13] + 2 * i ..][0..2];
        if (e.peer_kept.fetchRemove(@as(u64, w[0]) | @as(u64, w[1]) << 32)) |kv| drop(e, kv.value);
    }
    if (!ok) return .{ .reason = .cancelled }; // rank 0 runs the request again from the start
    const eos = try e.gpa.dupe(u32, head[4 .. 4 + head[2]]);
    defer e.gpa.free(eos);
    var reply: Reply = .{ .e = e, .prompt = prompt };
    entered_generate = true;
    const res = try e.generateFrom(prompt, from, marks, head[1], eos, if (head[3] == std.math.maxInt(u32)) null else head[3], budget, out orelse reply.out());
    if (out != null) return res;
    std.log.info("speed-up rank 1: resumed at {d} of {d} (restore {d:.1} ms), kept {d} marks ({d:.1} ms), {d} states {d} MiB and {d} MiB spare; reply {d} tokens, sha {s}", .{ from, n, restore_ms, reply.kept, reply.keep_ns / 1e6, e.peer_kept.count(), e.peer_held >> 20, e.snap_pool.spare() >> 20, reply.tokens, &reply.sha() });
    return res;
}

/// Rank 1's reply: its tokens hashed as the server hashes rank 0's (tokens.zig tokenSha) and its state at each mark.
const Reply = struct {
    e: *Engine,
    prompt: []const u32,
    hash: std.crypto.hash.sha2.Sha256 = .init(.{}),
    tokens: usize = 0,
    kept: usize = 0,
    keep_ns: f64 = 0,

    fn out(r: *Reply) engine.Out {
        return .{ .ctx = r, .prefilled = prefilled, .tokens = tokensFn, .cancelled = cancelled, .marked = marked };
    }
    fn prefilled(_: *anyopaque) void {}
    fn cancelled(_: *anyopaque) bool {
        return false;
    }
    fn tokensFn(ctx: *anyopaque, toks: []const u32) bool {
        const r: *Reply = @ptrCast(@alignCast(ctx));
        r.add(toks);
        return false;
    }
    fn add(r: *Reply, toks: []const u32) void {
        var buf: [16]u8 = undefined;
        for (toks) |t| {
            if (r.tokens > 0) r.hash.update(",");
            r.hash.update(std.fmt.bufPrint(&buf, "{d}", .{t}) catch unreachable);
            r.tokens += 1;
        }
    }
    fn sha(r: *Reply) [12]u8 {
        var d: [32]u8 = undefined;
        r.hash.final(&d);
        return std.fmt.bytesToHex(d[0..6].*, .lower);
    }
    fn marked(ctx: *anyopaque, at: usize) void {
        const r: *Reply = @ptrCast(@alignCast(ctx));
        const t0 = platform.c.mach_absolute_time();
        defer {
            r.kept += 1;
            r.keep_ns += msSince(t0) * 1e6;
        }
        keep(r.e, r.prompt, at);
    }
};

/// Keep this Mac's state under rank 0's name within the budget and slack, then trim free buffers.
fn keep(e: *Engine, prompt: []const u32, at: usize) void {
    const size = e.snap_pool.size(snapshot.bytes(at));
    if (!admits(e.peer_budget, e.peer_held, size)) return std.log.warn("speed-up rank 1: kept nothing at {d}: {d} MiB held and {d} MiB more pass the {d} MiB budget and its slack", .{ at, e.peer_held >> 20, size >> 20, e.peer_budget >> 20 });
    const st = snapshot.save(e, e.gpa, at) catch |err| return std.log.warn("speed-up rank 1: no state kept at {d}: {s}", .{ at, @errorName(err) });
    const old = e.peer_kept.fetchPut(e.gpa, engine.keyOf(prompt[0 .. at + 1]), st) catch return snapshot.drop(e.gpa, st);
    e.peer_held += st.cap;
    if (old) |kv| drop(e, kv.value);
    e.snap_pool.trim(e.peer_budget -| e.peer_held);
}

/// Extra bytes allowed for rank 1's pooled state buffers beyond rank 0's budget.
const SLACK: u64 = 1 << 30;

/// Whether size fits beside held under rank 0's budget and the allowed slack.
fn admits(budget: u64, held: u64, size: u64) bool {
    return budget > 0 and held + size <= budget + SLACK;
}

fn drop(e: *Engine, st: *snapshot.State) void {
    e.peer_held -|= st.cap;
    snapshot.drop(e.gpa, st);
}

test "rank 1's reply hash is the server's token_sha" {
    var r: Reply = .{ .e = undefined, .prompt = &.{} };
    r.add(&.{ 1, 22 });
    r.add(&.{333});
    var want: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash("1,22,333", &want, .{});
    try std.testing.expectEqualStrings(&std.fmt.bytesToHex(want[0..6].*, .lower), &r.sha());
}

test "rank 1 keeps rank 0's states up to the budget and its slack, and nothing without a budget" {
    try std.testing.expect(admits(4 << 30, 3808 << 20, 352 << 20));
    try std.testing.expect(admits(1 << 30, 600 << 20, 1448 << 20));
    try std.testing.expect(!admits(1 << 30, 600 << 20, 1449 << 20));
    try std.testing.expect(!admits(0, 0, 1));
}
