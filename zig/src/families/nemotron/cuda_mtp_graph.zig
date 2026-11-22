//! A lone MTP round: first kept-row level, one conditional while for later levels, then one completion readback.
const std = @import("std");
const cuda = @import("cuda");
const mtp = @import("cuda_mtp.zig");
const policy = @import("cuda_mtp_graph_policy.zig");
const Head = mtp.Head;
const Rule = @import("core").draft_depth.DepthRule;
pub fn enabled() !bool {
    return @import("sys").environment.read(bool, "TF_NEMOTRON_MTP_GRAPH", false);
}
pub const Loop = struct {
    api: cuda.conditional.Api,
    device: cuda.DeviceBuffer,
    host: cuda.HostBuffer,
    ready: cuda.Event,
    unsupported: bool = false,
    pub fn init(h: *Head) !?Loop {
        const d = h.e.ctx.d;
        var library = d.lib;
        const api = cuda.conditional.Api.resolve(&library) orelse return null;
        var device = try cuda.DeviceBuffer.alloc(d, @sizeOf(policy.State));
        errdefer device.free();
        var host = try cuda.HostBuffer.alloc(d, @sizeOf(policy.State));
        errdefer host.free();
        return .{ .api = api, .device = device, .host = host, .ready = try cuda.Event.init(d, false) };
    }
    pub fn deinit(l: *Loop) void {
        l.ready.deinit();
        l.host.free();
        l.device.free();
    }
    pub fn field(l: Loop, comptime name: []const u8) u64 {
        return l.device.ptr + @offsetOf(policy.State, name);
    }
};
pub fn available(h: *const Head) bool {
    return h.auto_allowed and h.loop != null and !h.loop.?.unsupported and h.e.k.triton == null and h.e.tp == null;
}

/// Null means no work ran: the caller queues the original level-one graph and follows the host rule.
pub fn depth(h: *Head, rule: *const Rule, v: mtp.Vocab) !?usize {
    if (!available(h)) return null;
    const p = policy.Policy.init(rule.*) orelse return null;
    const l = &h.loop.?;
    @memcpy(l.host.slice(u8)[0..@sizeOf(policy.Policy)], std.mem.asBytes(&p));
    try h.e.ops().upload(l.device.ptr, l.host.slice(u8)[0..@sizeOf(policy.Policy)]);
    const id: u32 = @intCast(7000 + (2 * @as(usize, @intFromBool(h.e.sampling != null)) + @backingInt(v)) * 33 + h.keep);
    h.e.launchCaptured(id, .{ .mtp_chain = .{ .h = h, .rows = h.keep, .vocab = v } }) catch |err| {
        if (err == error.GraphNotPermitted) {
            l.unsupported = true;
            return null;
        }
        return err;
    };
    try l.ready.record(h.e.stream);
    try l.ready.synchronize();
    const state = l.host.slice(policy.State)[0];
    if (state.n > rule.most or state.levels > rule.most or state.n < 1) return error.BadDraftRequest;
    h.levels = state.levels;
    h.acceptGraphDrafts(state.ids[0..state.levels], state.probs[0..state.levels]);
    return state.n;
}

fn launch(h: *Head, f: cuda.Function, args: *cuda.Args) !void {
    try cuda.launch.launch(f, .{ .grid = .{ .x = 1 }, .block = .{ .x = 1 } }, h.e.stream, args);
}
fn control(h: *Head, handle: cuda.conditional.Handle) !void {
    var args: cuda.Args = .{};
    for ([_]u64{ h.loop.?.device.ptr, h.e.b.ids, h.probs, handle }) |x| args.add(x);
    try launch(h, h.e.k.mtp_control, &args);
}
fn abortCapture(h: *Head) !void {
    if (cuda.graph.endCapture(h.e.stream)) |_| {} else |err| {
        if ((cuda.graph.captureStatus(h.e.stream) catch return err) != .none) return err;
    }
}

/// The bound sequence's graph cache owns the executable, so release/epoch change frees all captured pointers.
pub fn record(h: *Head, rows: usize, v: mtp.Vocab) !cuda.graph.Exec {
    const e = h.e;
    const l = &h.loop.?;
    const begin = std.Io.Clock.awake.now(e.io);
    var g = try cuda.graph.Graph.init(e.ctx.d);
    defer g.deinit();
    const condition = try l.api.handle(g, e.ctx.handle);
    try l.api.begin(e.stream, g, &.{});
    first(h, rows, v, condition) catch |err| {
        try abortCapture(h);
        return err;
    };
    _ = try cuda.graph.endCapture(e.stream);
    const n = try g.nodeCount();
    const buf = try e.gpa.alloc(cuda.graph.Node, n);
    defer e.gpa.free(buf);
    const loop = try l.api.loop(g, try g.nodes(buf), e.ctx.handle, condition);
    try l.api.begin(e.stream, loop.body, &.{});
    later(h, v, condition) catch |err| {
        try abortCapture(h);
        return err;
    };
    _ = try cuda.graph.endCapture(e.stream);
    try l.api.begin(e.stream, g, &.{loop.node});
    e.ops().download(l.host.slice(u8), l.device.ptr) catch |err| {
        try abortCapture(h);
        return err;
    };
    _ = try cuda.graph.endCapture(e.stream);
    var exec = try cuda.conditional.Api.instantiate(g);
    errdefer exec.deinit();
    try cuda.conditional.Api.upload(exec, e.stream);
    // Upload and first replay are ordered on this stream; no startup synchronization enters the round.
    e.stats.captures += 1;
    e.stats.capture_ns += @intCast(std.Io.Clock.awake.now(e.io).toNanoseconds() - begin.toNanoseconds());
    return exec;
}
fn first(h: *Head, rows: usize, v: mtp.Vocab, handle: cuda.conditional.Handle) !void {
    var args: cuda.Args = .{};
    args.add(h.loop.?.device.ptr);
    args.add(h.meta);
    try launch(h, h.e.k.mtp_start, &args);
    try level(h, rows, true, v);
    try control(h, handle);
}
fn later(h: *Head, v: mtp.Vocab, handle: cuda.conditional.Handle) !void {
    try level(h, 1, false, v);
    try control(h, handle);
}
fn level(h: *Head, rows: usize, first_level: bool, v: mtp.Vocab) !void {
    const o = h.e.ops();
    const d: u64 = h.e.c.hidden;
    const l = h.loop.?;
    try o.copy(h.hin, if (first_level) h.e.b.hidden else h.out, rows * d * 2);
    try o.copy(h.tok, if (first_level) h.e.b.sampled else l.field("token"), rows * 4);
    try h.forward(rows, true, if (first_level) h.meta else l.field("meta"), v);
    try o.copy(h.out, h.row, d * 2);
    // meta[1]+1 equals the original fixed base plus j: sample.cu keys only that absolute position.
    try h.sample(h.logits, if (h.e.sampling != null) h.e.b.rule else null, l.field("meta") + 4, l.field("token"), l.field("probability"), 1, v);
}
