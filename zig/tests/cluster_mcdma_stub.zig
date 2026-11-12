//! Bring-up on mcdma.Endpoint's stub verbs: a Mac and a GB10 node in one process agree on plan and view, hold, leave.
const std = @import("std");
const sys = @import("sys");
const fabric = @import("fabric");
const cluster = @import("cluster");

const config = cluster.config;
const bringup = cluster.bringup;
const mcdma_link = cluster.mcdma_link;
const words = fabric.words;

const gpa = std.testing.allocator;
const ms = std.time.ns_per_ms;
const quick: bringup.Options = .{ .membership = .{ .beat_ns = 20 * ms, .suspect_ns = 120 * ms, .dead_ns = 400 * ms }, .settle_ns = 10 * std.time.ns_per_s, .linger_ns = 200 * ms };

fn library() ?[]const u8 {
    const path = sys.getenv("TF_MCDMA_FABRIC_STUB") orelse return null;
    return if (std.fs.path.isAbsolute(path)) path else null;
}

var next_port: u16 = 0;
/// A fresh pair of exchange ports for one bring-up (fabric_mcdma_stub takes 30000 and up).
fn ports() u16 {
    if (next_port == 0) next_port = 42000 + @as(u16, @intCast(@mod(std.c.getpid(), 1000))) * 16;
    next_port += 8;
    return next_port;
}

/// The cluster file both nodes read: one ConnectX link on the stub's roce0 and roce1, the GB10 prefilling for the Mac.
fn file(a: std.mem.Allocator, lib: []const u8, port: u16, mtu: u32) ![]const u8 {
    return std.fmt.allocPrint(a,
        \\{{ "schema": "tensorfold-cluster/1", "cluster": "stub",
        \\  "nodes": [ {{ "name": "mac", "address": "192.0.2.1", "backend": "metal", "memory_gb": 128, "library": "{s}" }},
        \\             {{ "name": "gb10", "address": "192.0.2.2", "backend": "cuda", "memory_gb": 128, "library": "{s}" }} ],
        \\  "links": [ {{ "name": "stubcx", "kind": "cx5", "mtu": {d},
        \\              "ends": [ {{ "node": "mac", "device": "roce0", "via": "lo0", "port": {d}, "gid": 0 }},
        \\                        {{ "node": "gb10", "device": "roce1", "via": "lo0", "port": {d}, "gid": 1 }} ] }} ],
        \\  "models": {{ "m": {{ "path": "/models/m", "nodes": ["gb10", "mac"], "mode": "disaggregated", "prefill": ["gb10"], "decode": ["mac"] }} }} }}
    , .{ lib, lib, mtu, port, port + 4 });
}

const Node = struct {
    arena: std.heap.ArenaAllocator,
    spec: mcdma_link.Spec,
    plan: u64,
    inv: cluster.node.Inventory,
    hold: ?u64,
    out: std.Io.Writer.Allocating,
    outcome: ?bringup.Outcome = null,
    err: ?anyerror = null,
    diag: bringup.Diag = .{},

    /// Node `name` from its own reading of `text`.
    fn init(n: *Node, text: []const u8, name: []const u8, hold: ?u64) !void {
        n.* = .{ .arena = .init(gpa), .spec = undefined, .plan = 0, .inv = .{}, .hold = hold, .out = .init(gpa) };
        const a = n.arena.allocator();
        var probs: config.Problems = .{ .a = a };
        const c = config.parse(a, text, &probs) catch {
            for (probs.list.items) |p| std.debug.print("{s}\n", .{p});
            return error.BadFile;
        };
        const m = c.model("m").?;
        var why: []const u8 = "";
        n.spec = mcdma_link.spec(a, c, m, name, .{ .connect_timeout_ns = 20 * std.time.ns_per_s, .timeout_ns = 5 * std.time.ns_per_s }, &why) catch |e| {
            std.debug.print("{s}: {s}\n", .{ @errorName(e), why });
            return e;
        };
        n.plan = mcdma_link.planDigest(c, m, n.spec.names, .{});
        n.inv = .{ .id = cluster.node.idFrom(name), .name = .of(name), .backend = c.nodes[c.find(name).?].backend };
    }

    fn deinit(n: *Node) void {
        n.out.deinit();
        n.arena.deinit();
    }

    fn run(n: *Node) void {
        n.outcome = bringup.onHost(gpa, n.spec, n.inv, n.plan, .{ .live = quick, .hold_ns = n.hold }, &n.out.writer, &n.diag) catch |e| blk: {
            n.err = e;
            break :blk null;
        };
    }
};

fn both(mac: *Node, gb10: *Node) !void {
    const t = try std.Thread.spawn(.{}, Node.run, .{gb10});
    mac.run();
    t.join();
}

test "a Mac and a GB10 node (stub roce0, roce1) open MCDMA from one cluster file, agree on the plan and view, and the leader's goodbye ends the other's hold" {
    const lib = library() orelse return error.SkipZigTest;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const text = try file(arena.allocator(), lib, ports(), 1024);
    var mac: Node = undefined;
    try mac.init(text, "mac", 300 * ms);
    defer mac.deinit();
    var gb10: Node = undefined;
    try gb10.init(text, "gb10", null);
    defer gb10.deinit();
    try std.testing.expectEqual(mac.plan, gb10.plan);
    const t0 = words.nowNs();
    try both(&mac, &gb10);
    for ([_]*Node{ &mac, &gb10 }) |n| if (n.err) |e| {
        std.debug.print("{s}: {s} ({s} {s})\n{s}", .{ n.inv.name.str(), @errorName(e), n.diag.culprit.name.str(), n.diag.reason, n.out.written() });
        return e;
    };
    const x = mac.outcome.?;
    const y = gb10.outcome.?;
    try std.testing.expectEqual(@as(u32, 2), x.report.nodes);
    try std.testing.expectEqual(x.report.plan, y.report.plan);
    try std.testing.expectEqual(x.report.view, y.report.view);
    try std.testing.expectEqual(x.report.epoch, y.report.epoch);
    try std.testing.expectEqualStrings(x.leader_name.str(), y.leader_name.str());
    try std.testing.expectEqual(@as(?bringup.Lost, null), x.ended_by);
    try std.testing.expectEqualStrings("mac", y.ended_by.?.name.str());
    try std.testing.expectEqual(cluster.membership.State.left, y.ended_by.?.state);
    for ([_]*Node{ &mac, &gb10 }) |n| {
        const out = n.out.written();
        try std.testing.expect(std.mem.indexOf(u8, out, "\"stubcx\"") != null and std.mem.indexOf(u8, out, "(cx5, path MTU 1024)") != null);
        try std.testing.expect(std.mem.indexOf(u8, out, "cluster up: 2 nodes agree on plan") != null);
    }
    std.debug.print("stub roce0/roce1: rank 1 (mac) and rank 0 (gb10) agree on plan {x:0>16}, view {x:0>16} after {d} ms; connected in {d} ms; all done in {d} ms\n", .{ x.report.plan, x.report.view, x.report.agreed_ms, x.connect_ms, (words.nowNs() - t0) / ms });
}

test "two ends that read different path MTUs connect, then refuse each other's plan by name instead of running" {
    const lib = library() orelse return error.SkipZigTest;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const port = ports();
    var mac: Node = undefined;
    try mac.init(try file(arena.allocator(), lib, port, 1024), "mac", 300 * ms);
    defer mac.deinit();
    var gb10: Node = undefined;
    try gb10.init(try file(arena.allocator(), lib, port, 4096), "gb10", null);
    defer gb10.deinit();
    try std.testing.expect(mac.plan != gb10.plan);
    try both(&mac, &gb10);
    // each end names the other and the digest it computed; MCDMA itself cannot see the mismatch at connect
    for ([_]*Node{ &mac, &gb10 }, [_]*Node{ &gb10, &mac }) |n, other| {
        try std.testing.expectEqual(@as(?anyerror, error.PlanMismatch), n.err);
        try std.testing.expectEqualStrings(other.inv.name.str(), n.diag.culprit.name.str());
        try std.testing.expectEqual(other.plan, n.diag.culprit.plan);
    }
}

const Stopper = struct {
    node: *Node,
    done: std.atomic.Value(bool) = .init(false),
    err: ?anyerror = null,

    /// Up and agreed, then silent with its endpoint open (a stopped process) until the other end has reported it.
    fn run(s: *Stopper) void {
        s.go() catch |e| {
            s.err = e;
        };
    }

    fn go(s: *Stopper) !void {
        const n = s.node;
        const o = try mcdma_link.Open.init(gpa, n.spec);
        defer o.deinit();
        var live = bringup.Live.init(gpa, n.inv, o.link.transport(), n.spec.names, n.plan, quick);
        defer live.deinit();
        _ = try live.up();
        while (!s.done.load(.acquire)) words.pause(ms);
    }
};

test "a node that stops after agreeing fails the other's hold by name, within the dead time and not a hang" {
    const lib = library() orelse return error.SkipZigTest;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const text = try file(arena.allocator(), lib, ports(), 1024);
    var mac: Node = undefined;
    try mac.init(text, "mac", null);
    defer mac.deinit();
    var gb10: Node = undefined;
    try gb10.init(text, "gb10", null);
    defer gb10.deinit();
    var stopper: Stopper = .{ .node = &gb10 };
    const t = try std.Thread.spawn(.{}, Stopper.run, .{&stopper});
    mac.run();
    stopper.done.store(true, .release);
    t.join();
    if (stopper.err) |e| return e;
    try std.testing.expectEqual(@as(?anyerror, error.MemberLost), mac.err);
    try std.testing.expectEqualStrings("gb10", mac.diag.culprit.name.str());
    try std.testing.expectEqual(@as(?cluster.membership.State, .dead), mac.diag.culprit.state);
    try std.testing.expect(std.mem.indexOf(u8, mac.out.written(), "cluster up: 2 nodes agree") != null);
}
