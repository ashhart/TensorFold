//! ``n`` above 1 on chat and completions: n engine requests on one prompt, merged into one reply or one stream.
const std = @import("std");
const json = @import("json");
const lanes = @import("lanes");
const openai = @import("openai.zig");
const fields = @import("fields.zig");
const prompt_list = @import("prompt_list.zig");
const Server = @import("server.zig").Server;
const Value = json.Value;
const Allocator = std.mem.Allocator;

const most = fields.most_choices;

/// Choice `i`'s seed: choice 0 keeps the request's (so n = 1 is unchanged); the rest are mix(seed ^ i) in 63 bits.
pub fn seedOf(seed: u64, i: u32) u64 {
    return if (i == 0) seed else lanes.sampling.mix(seed ^ i) & ((@as(u64, 1) << 63) - 1);
}

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
    var slots: [most]prompt_list.Slot = undefined;
    for (slots[0..n]) |*s| s.* = .{ .arena = .init(srv.gpa) };
    defer for (slots[0..n]) |*s| s.arena.deinit();
    var fan: prompt_list.Fan = .{ .out = out, .io = srv.io, .slots = slots[0..n] };
    var choices: [most]prompt_list.Choice = undefined;
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
    if (plan.stream) prompt_list.streamed(&fan, a, plan.separate_usage, .once) else prompt_list.whole(&fan, a, true, .once);
}

fn go(r: *openai.Run, gone: openai.Gone, stream: bool, field: []const u8) void {
    if (stream) r.stream(gone, field) else r.whole(gone, field);
}
