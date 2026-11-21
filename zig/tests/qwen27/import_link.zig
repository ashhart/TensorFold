//! #1594 step 2b's Mac half over the fake link: a state file served as a protocol 2 export, pulled into a blank state.
const std = @import("std");
const fabric = @import("fabric");
const host = @import("qwen27_served");
const q = @import("tensorfold").qwen27;
const hs = fabric.hybrid_state;
const hx = fabric.hybrid_export;
const producer = fabric.producer;

const half = 16 << 20; // each mailbox half: a 3 MiB recurrence part fits, and 255 page rows of 64 KiB a frame

/// The GB10's side in-process: the request's export recorded under its handoff id, over the file's bytes.
const Trigger = struct {
    table: *producer.Table,
    built: hx.Built,
    exp: producer.Export = undefined,

    fn start(ptr: *anyopaque, handoff: [16]u8, tokens: []const u32) anyerror!void {
        const t: *Trigger = @ptrCast(@alignCast(ptr));
        t.exp = .{ .handoff = handoff, .tokens = tokens[0 .. tokens.len - 1], .first_token = 0, .tokens_per_row = host.remote.block, .layers = t.built.layers, .sources = t.built.sources };
        try t.table.add(handoff, "import-link", .{ .ready = &t.exp });
    }
};

/// `body` (a state after `at` of `ids`, hybrid order) served over the fake link and pulled by the served host's Pull.
pub fn pulled(a: std.mem.Allocator, io: std.Io, r: *const q.decode_round.Runner, ids: []const u32, at: u32, body: []const u8) !*q.snapshot.State {
    if (ids.len - 1 >= 2 * @as(usize, at)) return error.LinkPrefixNotOneChunk; // the policy's one chunk must end at `at`
    var link = try fabric.link_fake.Link.init(a, half, half, true, .immediate);
    defer link.deinit();
    link.up();
    const mac = link.box(0);
    const peer = link.box(1);
    const plan: hx.Plan = .{ .g = hs.qwen27, .s = hx.qwen27, .at = at, .block = host.remote.block };
    var table: producer.Table = .{ .gpa = a, .ttl_ns = 60 * std.time.ns_per_s, .clock = fabric.words.nowNs };
    defer table.deinit();
    var serving: producer.Serving = undefined;
    serving.init(a, &peer, fabric.words.native, &table, "Qwen3.8-27B");
    defer serving.deinit();
    var trigger: Trigger = .{ .table = &table, .built = try hx.build(a, plan, body) };
    var stop: std.atomic.Value(bool) = .init(false);
    const daemons = try std.Thread.spawn(.{}, fabric.link_fake.Link.run, .{ &link, &stop });
    const service = try std.Thread.spawn(.{}, producer.Serving.run, .{&serving});
    defer {
        stop.store(true, .release);
        serving.stop.store(true, .release);
        daemons.join();
        service.join();
    }
    var client = try fabric.mailbox.Client.init(&mac, fabric.words.native, 30 * std.time.ns_per_s);
    var pull: host.remote.Pull = .{ .gpa = a, .io = io, .runner = r, .remote = .{
        .gpa = a,
        .policy = .{ .enabled = true, .min_prefix = 1, .chunk = at },
        .client = &client,
        .trigger = .{ .ptr = &trigger, .start_fn = Trigger.start },
        .model = "Qwen3.8-27B",
        .protocol = fabric.wire.recurrent_version,
    } };
    const got = (try pull.pull(ids, 0)) orelse return error.NoOffer;
    const st: *q.snapshot.State = @ptrCast(@alignCast(got.saved));
    errdefer q.snapshot.drop(a, st);
    if (got.at != at) return error.LinkPrefixMoved;
    const same = std.mem.eql(u8, st.buf.contents()[0..body.len], body);
    std.debug.print("RESULT link: {d} tokens' state, {d} bytes, pulled over the fake link as protocol 2 in {d:.1} ms; bytes {s} the file's\n", .{ at, body.len, @as(f64, @floatFromInt(pull.ns)) / 1e6, if (same) "equal to" else "DIFFERENT from" });
    if (!same) return error.LinkBytesDiffer;
    return st;
}

/// A state pulled over the real link, and the prompt length it covers.
pub const Pulled = struct { st: *q.snapshot.State, at: u32 };

/// The real link: the GB10's `serve --mcdma-produce` exports `ids` at the policy's cut, pulled as the server pulls.
pub fn pulledRemote(a: std.mem.Allocator, io: std.Io, r: *const q.decode_round.Runner, ids: []const u32, url: []const u8, link_name: []const u8, dir: []const u8) !Pulled {
    const id = try host.remote.checkpointId(a, io, dir);
    const l = try host.remote.Link.start(a, io, r, .{ .url = url, .link = link_name }, id);
    defer l.stop(a);
    l.pull.remote.policy.min_prefix = 0; // a check of the hand-off, not of routing: any prompt pulls at its cut
    const got = (try l.pull.pull(ids, 0)) orelse return error.NoRemoteOffer;
    std.debug.print("RESULT remote: {d} tokens' state pulled over {s} in {d:.1} ms, the GB10's prefill included\n", .{ got.at, link_name, @as(f64, @floatFromInt(l.pull.ns)) / 1e6 });
    return .{ .st = @ptrCast(@alignCast(got.saved)), .at = got.at };
}
