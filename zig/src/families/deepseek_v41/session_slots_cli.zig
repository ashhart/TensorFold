//! `tf-dsv41 session-slots PACK ASSETS OUT REF...` (rank 0): multi-turn sessions over several live slots, checked.

const std = @import("std");
const api = @import("engine_api");
const se = @import("serve_engine.zig");

const Box = struct {
    io: std.Io,
    gpa: std.mem.Allocator,
    mutex: std.Io.Mutex = .init,
    tokens: std.ArrayList(u32) = .empty,
    done: ?api.Reason = null,

    fn event(ctx: *anyopaque, _: api.Id, e: *const api.Event) void {
        const b: *Box = @ptrCast(@alignCast(ctx));
        b.mutex.lockUncancelable(b.io);
        defer b.mutex.unlock(b.io);
        switch (e.*) {
            .tokens => |t| b.tokens.appendSlice(b.gpa, t) catch {},
            .finished => |f| b.done = f.reason,
            else => {},
        }
    }

    fn finished(b: *Box) ?api.Reason {
        b.mutex.lockUncancelable(b.io);
        defer b.mutex.unlock(b.io);
        return b.done;
    }
};

const Ref = struct { prompt_ids: []const u32, tokens: []const u32 };
const Line = struct { chat: u32, turn: u32, prompt: u64, tokens: []const u32 };

fn envInt(name: [*:0]const u8, d: u64) !u64 {
    const v = std.c.getenv(name) orelse return d;
    return std.fmt.parseInt(u64, std.mem.span(v), 10);
}

/// Submits every prompt at once and waits for all replies (eos ignored: every reply `steps` tokens).
fn turn(eng: api.Engine, io: std.Io, a: std.mem.Allocator, prompts: []const []const u32, steps: u32, id0: api.Id, drafts: bool) ![]Box {
    const boxes = try a.alloc(Box, prompts.len);
    const reqs = try a.alloc(api.Request, prompts.len);
    for (boxes, reqs, prompts, 0..) |*b, *r, p, i| {
        b.* = .{ .io = io, .gpa = a };
        r.* = .{ .prompt = p, .max_tokens = steps, .drafts = drafts };
        try eng.submit(id0 + i, r, .{ .ctx = b, .event = Box.event });
    }
    while (true) {
        var all = true;
        for (boxes) |*b| all = all and b.finished() != null;
        if (all) return boxes;
        std.Io.sleep(io, .fromMilliseconds(2), .awake) catch {};
    }
}

pub fn main(gpa: std.mem.Allocator, io: std.Io, pack: []const u8, assets: []const u8, out: []const u8, refs: []const []const u8) !u8 {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const cwd = std.Io.Dir.cwd();
    if (refs.len == 0 or refs.len > 16) return error.BadRefs;
    const chats = try a.alloc(Ref, refs.len);
    for (chats, refs) |*c, dir| {
        const text = try cwd.readFileAlloc(io, try std.fs.path.join(a, &.{ dir, "ref.json" }), a, .limited(1 << 28));
        c.* = try std.json.parseFromSliceLeaky(Ref, a, text, .{ .ignore_unknown_fields = true });
    }
    const turns: u32 = @intCast(try envInt("TF_DSV41_SESSION_SLOTS_TURNS", 3));
    var steps: u32 = @intCast(try envInt("TF_DSV41_SESSION_SLOTS_STEPS", 24));
    for (chats) |c| steps = @min(steps, @as(u32, @intCast(c.tokens.len)));
    const msg: usize = @intCast(try envInt("TF_DSV41_SESSION_SLOTS_MSG", 48));
    var rng = std.Random.DefaultPrng.init(try envInt("TF_DSV41_SESSION_SLOTS_SEED", 4141));
    const drafts = if (std.c.getenv("TF_DSV41_DRAFTS")) |v| !std.mem.eql(u8, std.mem.span(v), "0") else false;
    const s = try se.Served.open(gpa, io, pack, assets, drafts, null);
    defer se.Served.close(s);
    const eng = s.engine();
    const ss = s.m.sessions;
    std.debug.print("session slots: {d} live slots, {d} chats, {d} turns of {d} tokens, sessions {s}, drafts {}\n", .{ s.m.slots, chats.len, turns, steps, if (ss != null) "on" else "off", drafts });

    var lines: std.ArrayList(Line) = .empty;
    const prompts = try a.alloc([]const u32, chats.len);
    for (prompts, chats) |*p, c| p.* = c.prompt_ids;
    var ok_py: usize = 0;
    const t0 = std.Io.Clock.awake.now(io);
    for (1..turns + 1) |t| {
        const boxes = try turn(eng, io, a, prompts, steps, 1000 * t, drafts);
        for (boxes, prompts, chats, 0..) |*b, p, c, i| {
            try lines.append(a, .{ .chat = @intCast(i), .turn = @intCast(t), .prompt = p.len, .tokens = b.tokens.items });
            if (t == 1) ok_py += @intFromBool(std.mem.eql(u32, b.tokens.items, c.tokens[0..steps]));
        }
        // the next turn: this prompt + its reply + a new message
        for (prompts, boxes) |*p, *b| {
            const next = try a.alloc(u32, p.len + b.tokens.items.len + msg);
            @memcpy(next[0..p.len], p.*);
            @memcpy(next[p.len..][0..b.tokens.items.len], b.tokens.items);
            for (next[p.len + b.tokens.items.len ..]) |*x| x.* = rng.random().intRangeLessThan(u32, 1000, 100_000);
            p.* = next;
        }
    }
    const ms = @divTrunc(t0.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds(), 1_000_000);
    var text: std.Io.Writer.Allocating = .init(a);
    for (lines.items) |l| {
        try std.json.Stringify.value(l, .{}, &text.writer);
        try text.writer.writeByte('\n');
    }
    try cwd.createDirPath(io, out);
    try cwd.writeFile(io, .{ .sub_path = try std.fs.path.join(a, &.{ out, "replies.jsonl" }), .data = text.written() });
    var pass = ok_py == chats.len;
    std.debug.print("{s} session slots turn 1 == Python: {d}/{d}\n", .{ if (ok_py == chats.len) "PASS" else "FAIL", ok_py, chats.len });
    if (ss) |x| {
        const st = x.store.stats;
        var stats: std.Io.Writer.Allocating = .init(a);
        try stats.writer.print("{{\"admits\": {d}, \"spills\": {d}, \"waits\": {d}, \"saves\": {d}, \"dups\": {d}, \"hits_ram\": {d}, \"hits_disk\": {d}, \"parks\": {d}, \"drops\": {d}, \"ms\": {d}}}\n", .{ x.stats.admits, x.stats.spills, x.stats.waits, st.saves, st.dups, st.hits_ram, st.hits_disk, st.parks, st.drops, ms });
        try cwd.writeFile(io, .{ .sub_path = try std.fs.path.join(a, &.{ out, "stats.json" }), .data = stats.written() });
        std.debug.print("session slots sessions: {s}", .{stats.written()});
        if (std.c.getenv("TF_DSV41_SESSION_SLOTS_TIERS")) |v| if (std.mem.eql(u8, std.mem.span(v), "1")) {
            const tiers = st.hits_ram > 0 and st.hits_disk > 0;
            std.debug.print("{s} session slots tiers: {d} RAM and {d} NVMe resumes, {d} parks\n", .{ if (tiers) "PASS" else "FAIL", st.hits_ram, st.hits_disk, st.parks });
            pass = pass and tiers;
        };
    }
    if (std.c.getenv("TF_DSV41_SESSION_SLOTS_FRESH")) |path| {
        const ftext = try cwd.readFileAlloc(io, std.mem.span(path), a, .limited(1 << 30));
        var it = std.mem.tokenizeScalar(u8, ftext, '\n');
        var want: std.ArrayList(Line) = .empty;
        while (it.next()) |l| try want.append(a, try std.json.parseFromSliceLeaky(Line, a, l, .{}));
        var ok: usize = 0;
        for (lines.items) |l| {
            const w = for (want.items) |x| {
                if (x.chat == l.chat and x.turn == l.turn) break x;
            } else null;
            const eq = if (w) |x| x.prompt == l.prompt and std.mem.eql(u32, x.tokens, l.tokens) else false;
            ok += @intFromBool(eq);
            if (!eq) std.debug.print("session slots: chat {d} turn {d} ({d}-token prompt) differs from the fresh run{s}\n", .{ l.chat, l.turn, l.prompt, if (w == null) " (missing there)" else "" });
        }
        std.debug.print("{s} session slots resumed == fresh: {d}/{d}\n", .{ if (ok == lines.items.len) "PASS" else "FAIL", ok, lines.items.len });
        pass = pass and ok == lines.items.len;
    }
    std.debug.print("session slots timing: {d} ms, {d} replies\n", .{ ms, lines.items.len });
    if (s.m.rows) |r| std.debug.print("row windows: {d}, rows {d}, padded {d}\n", .{ r.stats.windows, r.stats.rows, r.stats.padded });
    return if (pass) 0 else 1;
}
