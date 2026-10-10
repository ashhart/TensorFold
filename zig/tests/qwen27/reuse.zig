//! Qwen27 prompt reuse through the served host against fresh prompt passes: each prompt's state and reply, bit for bit.
const std = @import("std");
const mtl = @import("metal");
const q = @import("tensorfold").qwen27;
const api = @import("engine_api");
const host = @import("qwen27_host");
const help = "tf-qwen27-reuse --model TARGET --draft DRAFTER --tokens IDS.json [--max-tokens 64]";

const Box = struct {
    a: std.mem.Allocator,
    io: std.Io,
    mutex: std.Io.Mutex = .init,
    tokens: std.ArrayList(u32) = .empty,
    cached: ?u32 = null,
    done: ?api.Reason = null,
    stats: api.Stats = .{},
    fn event(ptr: *anyopaque, _: api.Id, value: *const api.Event) void {
        const b: *Box = @ptrCast(@alignCast(ptr));
        b.mutex.lockUncancelable(b.io);
        defer b.mutex.unlock(b.io);
        switch (value.*) {
            .prefilled => |n| b.cached = n,
            .logprobs => {},
            .tokens => |got| b.tokens.appendSlice(b.a, got) catch {},
            .finished => |f| {
                b.done = f.reason;
                b.stats = f.stats;
                if (f.reason == .failed) std.debug.print("request failed: {s}\n", .{f.message});
            },
        }
    }
    fn wait(b: *Box) !void {
        for (0..60_000) |_| {
            b.mutex.lockUncancelable(b.io);
            const done = b.done != null;
            b.mutex.unlock(b.io);
            if (done) return;
            try std.Io.sleep(b.io, .fromMilliseconds(10), .awake);
        }
        return error.NoCompletion;
    }
};

/// One request's reply, where its prompt resumed and the state it left.
const Outcome = struct { tokens: []const u32, cached: u32, reason: api.Reason, rounds: u64, accepted: u64, state: [32]u8 };

const Run = struct {
    a: std.mem.Allocator,
    io: std.Io,
    h: *host.Host,
    next: api.Id = 1,
    failures: usize = 0,

    fn begin(r: *Run, box: *Box, request: *const api.Request) !void {
        box.* = .{ .a = r.a, .io = r.io };
        try r.h.engine().submit(r.next, request, .{ .ctx = box, .event = Box.event });
        r.next += 1;
    }
    fn end(r: *Run, box: *Box) !Outcome {
        try box.wait();
        return .{ .tokens = box.tokens.items, .cached = box.cached orelse 0, .reason = box.done.?, .rounds = box.stats.rounds, .accepted = box.stats.accepted, .state = try (q.session.Session{ .runner = &r.h.runner }).fingerprint() };
    }
    fn send(r: *Run, request: api.Request) !Outcome {
        var box: Box = undefined;
        try r.begin(&box, &request);
        return r.end(&box);
    }
    /// The same request with the cache off: the fresh prompt pass a resumed one must equal.
    fn fresh(r: *Run, request: api.Request) !Outcome {
        const store = r.h.serial.driver.cache;
        r.h.serial.driver.cache = null;
        defer r.h.serial.driver.cache = store;
        return r.send(request);
    }
    fn check(r: *Run, name: []const u8, got: Outcome, want: Outcome, cached: ?u32) void {
        const same = std.mem.eql(u32, got.tokens, want.tokens) and std.mem.eql(u8, &got.state, &want.state) and got.rounds == want.rounds and got.accepted == want.accepted and got.reason == want.reason;
        const resumed = if (cached) |c| got.cached == c else got.cached > 0;
        if (!same or !resumed) r.failures += 1;
        std.debug.print("{s} {s}: resumed at {d}, {d} tokens, {d} rounds, {d} accepted (fresh {d}/{d}/{d}); state {s}\n", .{ if (same and resumed) "PASS" else "FAIL", name, got.cached, got.tokens.len, got.rounds, got.accepted, want.tokens.len, want.rounds, want.accepted, if (std.mem.eql(u8, &got.state, &want.state)) "equal" else "DIFFERS" });
    }
};

fn ids(a: std.mem.Allocator, io: std.Io, path: []const u8) ![]const u32 {
    const text = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(16 << 20));
    return (try std.json.parseFromSlice([]u32, a, text, .{})).value;
}

pub fn main(init: std.process.Init) !void {
    const a = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
    var target_dir: ?[]const u8 = null;
    var draft_dir: ?[]const u8 = null;
    var tokens_file: ?[]const u8 = null;
    var count: u32 = 64;
    var i: usize = 1;
    while (i + 1 < args.len) : (i += 2) {
        const value = args[i + 1];
        if (std.mem.eql(u8, args[i], "--model")) target_dir = value else if (std.mem.eql(u8, args[i], "--draft")) draft_dir = value else if (std.mem.eql(u8, args[i], "--tokens")) tokens_file = value else if (std.mem.eql(u8, args[i], "--max-tokens")) count = try std.fmt.parseInt(u32, value, 10) else return error.UnknownOption;
    }
    if (target_dir == null or draft_dir == null or tokens_file == null or i != args.len) {
        try std.Io.File.stdout().writeStreamingAll(io, help ++ "\n");
        return error.BadOptions;
    }
    const all = try ids(a, io, tokens_file.?);
    if (all.len < 8192) return error.PromptTooShort;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const h = try host.open(init.gpa, io, target_dir.?, 16384);
    defer host.close(h);
    try host.enableDraft(h, io, draft_dir.?, 4);
    var why: []const u8 = "";
    try host.enableCache(h, 16, false, a, &why);
    std.debug.print("neural engine share: {s}\n", .{if (h.model.ane != null) "on" else "off"});
    var r = Run{ .a = a, .io = io, .h = h };
    // turn 1, the reply's background prefill, then turn 2; each case's prompts start elsewhere
    const Case = struct { skip: usize, turn1: u32, history: u32, warm: u32, turn2: u32 };
    for ([_]Case{ .{ .skip = 0, .turn1 = 1500, .history = 1400, .warm = 2077, .turn2 = 2600 }, .{ .skip = 97, .turn1 = 5000, .history = 4900, .warm = 6001, .turn2 = 7003 } }) |c| for ([_]bool{ true, false }) |drafts| {
        const p = all[c.skip..];
        const label = if (drafts) "drafted" else "plain";
        const want_prompt = try r.fresh(.{ .prompt = p[0..c.turn2], .max_tokens = 0 });
        const want = try r.fresh(.{ .prompt = p[0..c.turn2], .max_tokens = count, .drafts = drafts });
        _ = try r.send(.{ .prompt = p[0..c.turn1], .max_tokens = count, .drafts = drafts, .history_len = c.history });
        const warm = try r.send(.{ .prompt = p[0..c.warm], .max_tokens = 0, .background = true });
        if (warm.reason != .length) r.failures += 1;
        r.check(try std.fmt.allocPrint(a, "{s} {d}: turn 2's prompt state", .{ label, c.turn2 }), try r.send(.{ .prompt = p[0..c.turn2], .max_tokens = 0 }), want_prompt, c.warm);
        r.check(try std.fmt.allocPrint(a, "{s} {d}: turn 2's reply", .{ label, c.turn2 }), try r.send(.{ .prompt = p[0..c.turn2], .max_tokens = count, .drafts = drafts }), want, c.warm);
    };
    // a background pass yields to a foreground request: it keeps what it prefilled, and the request resumes there
    const p = all[211..];
    const want = try r.fresh(.{ .prompt = p[0..7900], .max_tokens = count });
    var warm_box: Box = undefined;
    const warm_request = api.Request{ .prompt = p[0..7700], .max_tokens = 0, .background = true };
    try r.begin(&warm_box, &warm_request);
    try std.Io.sleep(io, .fromMilliseconds(1500), .awake);
    var box: Box = undefined;
    const request = api.Request{ .prompt = p[0..7900], .max_tokens = count };
    try r.begin(&box, &request);
    try warm_box.wait();
    const got = try r.end(&box);
    if (warm_box.done != .cancelled or got.cached == 0 or got.cached >= 7700) r.failures += 1;
    r.check("yielded background pass", got, want, null);
    std.debug.print("{s}: {d} failures\n", .{ if (r.failures == 0) "REUSE PASS" else "REUSE FAIL", r.failures });
    if (r.failures != 0) return error.ReuseDiffers;
}
