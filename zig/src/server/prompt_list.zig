//! A raw completion's list of prompts: each prompt its own engine request, merged into one reply or one stream.
const std = @import("std");
const json = @import("json");
const openai = @import("openai.zig");
const Server = @import("server.zig").Server;
const Value = json.Value;
const Allocator = std.mem.Allocator;
const errors = @import("errors.zig");

/// Prompts one request may list.
pub const max_prompts = 64;

/// A raw completion's prompts: a list of two or more strings; empty for any other ``prompt``, which keeps its meaning.
pub fn texts(cx: *errors.Cx, prompt: ?Value) errors.Refused![]const []const u8 {
    const p = prompt orelse return &.{};
    if (p != .array or p.array.len < 2) return &.{};
    for (p.array) |x| if (x != .string) return &.{};
    if (p.array.len > max_prompts) return cx.refuse(std.fmt.comptimePrint("a prompt list holds at most {d} prompts", .{max_prompts}));
    const out = try cx.a.alloc([]const u8, p.array.len);
    for (p.array, out) |x, *o| o.* = x.string;
    return out;
}

/// Prompts in flight at once: the rest wait for a free worker.
const workers = 8;

/// What one prompt's reply sent its client, held for the merge (its values live in its arena).
const Slot = struct {
    arena: std.heap.ArenaAllocator,
    status: u16 = 0, // a whole reply's status, or a refusal's (0: none: its client left)
    body: ?Value = null,
    final: ?Value = null, // a stream's finish chunk, sent once every choice finished
    usage: ?Value = null, // a stream's usage, taken off its chunks
    after: ?Value = null, // a stream's event after its finish (a loop stop's error)
};

/// The client's one reply or stream, shared by the choices' threads under `mutex`.
const Fan = struct {
    out: openai.Out,
    io: std.Io,
    mutex: std.Io.Mutex = .init,
    opened: bool = false,
    failed: bool = false, // a choice failed or was refused: no prompt starts after it, the rest end at their next write
    error_sent: bool = false,
    slots: []Slot,

    fn stopped(f: *Fan) bool {
        f.mutex.lockUncancelable(f.io);
        defer f.mutex.unlock(f.io);
        return f.failed;
    }
};

/// One prompt's view of the client: its chunks get its index; the head opens once; finishes wait for the rest.
const Choice = struct {
    fan: *Fan,
    i: u32,

    fn out(c: *Choice) openai.Out {
        return .{ .ctx = c, .vt = &.{ .open = open, .event = event, .reply = reply, .keepalive = keepalive } };
    }

    fn locked(ctx: *anyopaque) *Choice {
        const c: *Choice = @ptrCast(@alignCast(ctx));
        c.fan.mutex.lockUncancelable(c.fan.io);
        return c;
    }

    fn unlock(c: *Choice) void {
        c.fan.mutex.unlock(c.fan.io);
    }

    fn open(ctx: *anyopaque, prompt_len: usize) error{Closed}!void {
        const c = locked(ctx);
        defer c.unlock();
        const f = c.fan;
        if (f.failed) return error.Closed;
        if (f.opened) return;
        try f.out.vt.open(f.out.ctx, prompt_len);
        f.opened = true;
    }

    fn event(ctx: *anyopaque, payload: ?Value) error{Closed}!void {
        const c = locked(ctx);
        defer c.unlock();
        const f = c.fan;
        const p = payload orelse return; // one [DONE], after every choice
        const s = &f.slots[c.i];
        const list = if (p.get("choices")) |l| (if (l == .array) l.array else null) else null;
        if (p == .object) if (p.object.fetchOrderedRemove("usage")) |kv| {
            s.usage = kv.value;
        };
        if (list != null and list.?.len == 0) return; // the separate usage chunk: one for the request, at the end
        if (s.final != null) {
            s.after = p;
            return;
        }
        if (p.get("error") != null) { // the first error is the stream's; the other choices end at their next write
            if (f.failed) return;
            f.failed = true;
            f.error_sent = true;
            return f.out.vt.event(f.out.ctx, p);
        }
        if (f.failed) return error.Closed;
        const l = list orelse return f.out.vt.event(f.out.ctx, p);
        if (l[0] != .object) return f.out.vt.event(f.out.ctx, p);
        const a = s.arena.allocator(); // the choice's own thread built `p` in it
        l[0].object.put(a, "index", json.intValue(a, c.i) catch return error.Closed) catch return error.Closed;
        if (l[0].get("finish_reason")) |r| if (r != .null) {
            s.final = p;
            return;
        };
        return f.out.vt.event(f.out.ctx, p);
    }

    fn reply(ctx: *anyopaque, status: u16, payload: Value) void {
        const c = locked(ctx);
        defer c.unlock();
        const s = &c.fan.slots[c.i];
        s.status = status;
        s.body = payload;
        if (status != 200) c.fan.failed = true;
    }

    fn keepalive(ctx: *anyopaque) error{Closed}!void {
        const c = locked(ctx);
        defer c.unlock();
        const f = c.fan;
        if (f.failed) return error.Closed;
        if (!f.opened) return;
        const send = f.out.vt.keepalive orelse return;
        return send(f.out.ctx);
    }
};

/// Each of `prompts` as the route's own whole reply or stream, on up to `workers` threads, then one answer.
pub fn run(srv: *Server, a: Allocator, out: openai.Out, gone: openai.Gone, plan: openai.Plan, prompts: []const []const u8, raw: Value) void {
    const n: usize = prompts.len;
    var slots: [max_prompts]Slot = undefined;
    for (slots[0..n]) |*s| s.* = .{ .arena = .init(srv.gpa) };
    defer for (slots[0..n]) |*s| s.arena.deinit();
    var fan: Fan = .{ .out = out, .io = srv.io, .slots = slots[0..n] };
    var choices: [max_prompts]Choice = undefined;
    var runs: [max_prompts]openai.Run = undefined;
    const created = std.Io.Clock.real.now(srv.io).toSeconds();
    for (0..n) |i| {
        choices[i] = .{ .fan = &fan, .i = @intCast(i) };
        var p = plan;
        p.input.prompt = .{ .text = prompts[i] }; // the request as if it named this prompt alone
        runs[i] = .{ .srv = srv, .a = slots[i].arena.allocator(), .out = choices[i].out(), .is_chat = false, .plan = p, .id = plan.input.id, .created = created, .call_ids = .{ .key = &srv.tool_ids, .body = raw } };
    }
    var pool: Pool = .{ .runs = runs[0..n], .fan = &fan, .gone = gone, .stream = plan.stream };
    var threads: [workers - 1]?std.Thread = @splat(null);
    // the request holds one in-flight place, so under the caps its prompts take turns in it
    const capped = srv.config.admission.max_inflight > 0 or srv.config.admission.max_inflight_per_key > 0;
    const helpers = if (capped) 0 else @min(workers, n) - 1;
    // a helper that cannot start leaves its prompts to the rest
    for (threads[0..helpers]) |*t| t.* = std.Thread.spawn(.{ .stack_size = 8 << 20 }, Pool.loop, .{&pool}) catch null;
    pool.loop();
    for (threads[0..helpers]) |t| if (t) |h| h.join();
    if (plan.stream) streamed(&fan, a, plan.separate_usage) else whole(&fan, a);
}

/// The prompts' runs, taken in order by whichever worker is free, until one is refused or fails.
const Pool = struct {
    runs: []openai.Run,
    fan: *Fan,
    gone: openai.Gone,
    stream: bool,
    next: std.atomic.Value(usize) = .init(0),

    fn loop(p: *Pool) void {
        while (true) {
            const i = p.next.fetchAdd(1, .monotonic);
            if (i >= p.runs.len or p.fan.stopped()) return;
            if (p.stream) p.runs[i].stream(p.gone, "prompt") else p.runs[i].whole(p.gone, "prompt");
        }
    }
};

/// The first refusal (by index) as the reply, else every choice's reply in one body; nothing if a client left.
fn whole(f: *Fan, a: Allocator) void {
    for (f.slots) |s| if (s.status != 0 and s.status != 200) return f.out.vt.reply(f.out.ctx, s.status, s.body.?);
    for (f.slots) |s| if (s.status == 0) return;
    const merged = mergeWhole(a, f.slots) catch return;
    f.out.vt.reply(f.out.ctx, 200, merged);
}

/// Choice 0's body with every choice indexed, each with its own tensorfold block, and the usage of all.
fn mergeWhole(a: Allocator, slots: []const Slot) !Value {
    const first = slots[0].body.?;
    const o = try json.copyObject(a, first.object);
    for ([_][]const u8{ "stop_sequence", "error" }) |k| _ = o.orderedRemove(k); // each choice's own, below
    const list = try a.alloc(Value, slots.len);
    var usages: [max_prompts]Value = undefined;
    for (slots, list, 0..) |s, *slot, i| {
        const b = s.body.?;
        const c = try json.copyObject(a, onlyChoice(b).object);
        try c.put(a, "index", try json.intValue(a, i));
        try c.put(a, "tensorfold", try stamp(a, b.get("tensorfold"), b.get("usage")));
        for ([_][]const u8{ "stop_sequence", "error" }) |k| if (b.get(k)) |v| try c.put(a, k, v);
        slot.* = .{ .object = c };
        usages[i] = b.get("usage") orelse .null;
    }
    try o.put(a, "choices", .{ .array = list });
    try o.put(a, "usage", try usage(a, usages[0..slots.len]));
    return .{ .object = o };
}

fn onlyChoice(body: Value) Value {
    const list = body.get("choices").?.array;
    return list[0];
}

/// The stream's end: held finishes in index order, the usage of all once, then one [DONE].
fn streamed(f: *Fan, a: Allocator, separate_usage: bool) void {
    if (!f.opened) { // no head went out: the first refusal is the reply (a client gone: nothing)
        for (f.slots) |s| if (s.status != 0) return f.out.vt.reply(f.out.ctx, s.status, s.body.?);
        return;
    }
    defer f.out.vt.event(f.out.ctx, null) catch {};
    if (!f.error_sent) for (f.slots) |s| if (s.status != 0) { // refused after another choice's head
        f.out.vt.event(f.out.ctx, s.body.?) catch {};
        return;
    };
    if (f.failed) return;
    for (f.slots) |s| if (s.final == null) return; // a choice ended unfinished: its client left
    var usages: [max_prompts]Value = undefined;
    for (f.slots, 0..) |s, i| usages[i] = s.usage orelse .null;
    const use = usage(a, usages[0..f.slots.len]) catch return;
    for (f.slots, 0..) |*s, i| {
        const last = s.final.?;
        const sa = s.arena.allocator();
        if (last.get("tensorfold")) |rt| last.object.put(sa, "tensorfold", stamp(sa, rt, s.usage) catch return) catch return;
        if (i + 1 == f.slots.len and !separate_usage) last.object.put(sa, "usage", use) catch return;
        f.out.vt.event(f.out.ctx, last) catch return;
    }
    if (separate_usage) {
        const head = f.slots[0].final.?;
        const u = json.newObject(a) catch return;
        for ([_][]const u8{ "id", "object", "created", "model" }) |k| if (head.get(k)) |v| u.put(a, k, v) catch return;
        u.put(a, "choices", .{ .array = &.{} }) catch return;
        u.put(a, "usage", use) catch return;
        f.out.vt.event(f.out.ctx, .{ .object = u }) catch return;
    }
    for (f.slots) |s| if (s.after) |p| f.out.vt.event(f.out.ctx, p) catch return;
}

/// A choice's tensorfold block with its cached prompt tokens (what the prompt cache held of its prompt).
fn stamp(a: Allocator, runtime: ?Value, use: ?Value) !Value {
    const rt = runtime orelse return .null;
    if (rt != .object) return rt;
    const o = try json.copyObject(a, rt.object);
    const details = if (use) |u| u.get("prompt_tokens_details") else null;
    if (details) |d| if (d.get("cached_tokens")) |c| try o.put(a, "cached_tokens", c);
    return .{ .object = o };
}

fn count(v: ?Value, key: []const u8) i64 {
    const x = (v orelse return 0).get(key) orelse return 0;
    return x.int64() orelse 0;
}

/// One request's usage over its prompts' (Run.usage's shape): every prompt's tokens and every reply's, summed.
fn usage(a: Allocator, each: []const Value) !Value {
    var completion: i64 = 0;
    var reasoning: i64 = 0;
    var prompt: i64 = 0;
    var cached_sum: i64 = 0;
    for (each) |u| {
        completion += count(u, "completion_tokens");
        reasoning += count(u.get("completion_tokens_details"), "reasoning_tokens");
        prompt += count(u, "prompt_tokens");
        cached_sum += count(u.get("prompt_tokens_details"), "cached_tokens");
    }
    const o = try json.newObject(a);
    try o.put(a, "prompt_tokens", try json.intValue(a, prompt));
    try o.put(a, "completion_tokens", try json.intValue(a, completion));
    try o.put(a, "total_tokens", try json.intValue(a, prompt + completion));
    const cached = try json.newObject(a);
    try cached.put(a, "cached_tokens", try json.intValue(a, cached_sum));
    try o.put(a, "prompt_tokens_details", .{ .object = cached });
    const thought = try json.newObject(a);
    try thought.put(a, "reasoning_tokens", try json.intValue(a, reasoning));
    try o.put(a, "completion_tokens_details", .{ .object = thought });
    return .{ .object = o };
}
