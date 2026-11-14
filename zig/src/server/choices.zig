//! ``n`` above 1 on chat and completions: n engine requests on one prompt, merged into one reply or one stream.
const std = @import("std");
const json = @import("json");
const lanes = @import("lanes");
const openai = @import("openai.zig");
const fields = @import("fields.zig");
const Server = @import("server.zig").Server;
const Value = json.Value;
const Allocator = std.mem.Allocator;

const most = fields.most_choices;

/// Choice `i`'s seed: choice 0 keeps the request's (so n = 1 is unchanged); the rest are mix(seed ^ i) in 63 bits.
pub fn seedOf(seed: u64, i: u32) u64 {
    return if (i == 0) seed else lanes.sampling.mix(seed ^ i) & ((@as(u64, 1) << 63) - 1);
}

/// What one choice sent its client, held for the merge (its values live in its arena).
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
    failed: bool = false, // a choice failed or was refused: the others' next writes end them
    error_sent: bool = false,
    slots: []Slot,
};

/// One choice's view of the client: its chunks get its index; the head opens once; finishes wait for the rest.
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

/// The request's `plan.choices` replies, each the route's own whole reply or stream on its thread, then one answer.
pub fn run(srv: *Server, a: Allocator, out: openai.Out, gone: openai.Gone, is_chat: bool, given: openai.Plan, raw: Value) void {
    const n: usize = given.choices;
    // every choice is a model request: the in-flight caps take n places, all or none, refused as a cap refuses
    if (gone.conn.lease) |lease| if (lease.widen(srv.config.admission, given.choices - 1)) |refusal|
        return @import("routes.zig").overloaded(gone.conn, false, refusal);
    const field: []const u8 = if (is_chat) "messages" else "prompt";
    var plan = given;
    // one fresh seed for the request, so every choice's is mix(seed ^ i) of choice 0's (server.resolveSampling's draw)
    if (!srv.config.prompt_seed and plan.input.fields.field("seed") == null) {
        var bits: u64 = undefined;
        if (srv.io.randomSecure(std.mem.asBytes(&bits))) |_| {
            const f = json.copyObject(a, plan.input.fields.object) catch return;
            f.put(a, "seed", json.intValue(a, bits & 0x7fff_ffff_ffff_ffff) catch return) catch return;
            plan.input.fields = .{ .object = f };
        } else |_| {} // each choice's own draw then fails the request as a single one would
    }
    var slots: [most]Slot = undefined;
    for (slots[0..n]) |*s| s.* = .{ .arena = .init(srv.gpa) };
    defer for (slots[0..n]) |*s| s.arena.deinit();
    var fan: Fan = .{ .out = out, .io = srv.io, .slots = slots[0..n] };
    var choices: [most]Choice = undefined;
    var runs: [most]openai.Run = undefined;
    const created = std.Io.Clock.real.now(srv.io).toSeconds();
    for (0..n) |i| {
        choices[i] = .{ .fan = &fan, .i = @intCast(i) };
        var p = plan;
        p.input.choice = @intCast(i);
        p.input.choices = plan.choices;
        if (i > 0) p.input.prompt_logprobs = null; // the prompt's rows are choice 0's: the merge copies them
        runs[i] = .{ .srv = srv, .a = slots[i].arena.allocator(), .out = choices[i].out(), .is_chat = is_chat, .plan = p, .id = plan.input.id, .created = created, .call_ids = .{ .key = &srv.tool_ids, .body = raw }, .warm = i == 0 };
    }
    // every choice in flight at once: a later one's prompt waits for the first fill's cut and resumes it
    var threads: [most]?std.Thread = @splat(null);
    for (1..n) |i| threads[i] = std.Thread.spawn(.{ .stack_size = 8 << 20 }, go, .{ &runs[i], gone, plan.stream, field }) catch null; // as a connection thread
    go(&runs[0], gone, plan.stream, field);
    for (1..n) |i| if (threads[i]) |t| t.join() else go(&runs[i], gone, plan.stream, field);
    if (plan.stream) streamed(&fan, a, plan.separate_usage) else whole(&fan, a);
}

fn go(r: *openai.Run, gone: openai.Gone, stream: bool, field: []const u8) void {
    if (stream) r.stream(gone, field) else r.whole(gone, field);
}

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
    var usages: [most]Value = undefined;
    const prompt_rows = onlyChoice(first).get("prompt_logprobs");
    for (slots, list, 0..) |s, *slot, i| {
        const b = s.body.?;
        const c = try json.copyObject(a, onlyChoice(b).object);
        try c.put(a, "index", try json.intValue(a, i));
        try c.put(a, "tensorfold", try stamp(a, b.get("tensorfold"), b.get("usage")));
        for ([_][]const u8{ "stop_sequence", "error" }) |k| if (b.get(k)) |v| try c.put(a, k, v);
        if (prompt_rows) |rows| try c.put(a, "prompt_logprobs", rows);
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
    var usages: [most]Value = undefined;
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

/// A choice's tensorfold block with its cached prompt tokens (what it resumed of the shared prompt).
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

/// One request's usage over its choices' (Run.usage's shape): the prompt once, every choice's reply tokens.
pub fn usage(a: Allocator, each: []const Value) !Value {
    var completion: i64 = 0;
    var reasoning: i64 = 0;
    for (each) |u| {
        completion += count(u, "completion_tokens");
        reasoning += count(u.get("completion_tokens_details"), "reasoning_tokens");
    }
    const prompt = count(each[0], "prompt_tokens");
    const o = try json.newObject(a);
    try o.put(a, "prompt_tokens", try json.intValue(a, prompt));
    try o.put(a, "completion_tokens", try json.intValue(a, completion));
    try o.put(a, "total_tokens", try json.intValue(a, prompt + completion));
    const cached = try json.newObject(a);
    try cached.put(a, "cached_tokens", try json.intValue(a, count(each[0].get("prompt_tokens_details"), "cached_tokens")));
    try o.put(a, "prompt_tokens_details", .{ .object = cached });
    const thought = try json.newObject(a);
    try thought.put(a, "reasoning_tokens", try json.intValue(a, reasoning));
    try o.put(a, "completion_tokens_details", .{ .object = thought });
    return .{ .object = o };
}
