//! Link qualification over the system's verbs; `run` swaps connection details on stdin and stdout and opens no socket.
const std = @import("std");
const fabric = @import("fabric");
const suite = @import("linktest_suite.zig");
const probe = @import("linktest_probe.zig");
const vqp = fabric.verbs_qp;
const vl = fabric.verbs_link;
const sr = fabric.sendrecv;
const Verbs = fabric.verbs.Verbs;
const out = suite.out;

const mib = 1 << 20;

const Args = struct { dev: []const u8 = "", rank: u32 = 0, kind: vqp.Kind = .uc, gid: u8 = 1, global: bool = true, quick: bool = false, deadline_s: u64 = 300 };

fn parse(args: []const [:0]const u8) !Args {
    var a: Args = .{};
    var i: usize = 2;
    while (i + 1 < args.len) : (i += 2) {
        const k = args[i];
        const val = args[i + 1];
        if (std.mem.eql(u8, k, "--dev")) a.dev = val else if (std.mem.eql(u8, k, "--rank")) a.rank = try std.fmt.parseInt(u32, val, 10) else if (std.mem.eql(u8, k, "--kind")) a.kind = std.meta.stringToEnum(vqp.Kind, val) orelse return error.BadKind else if (std.mem.eql(u8, k, "--gid")) a.gid = try std.fmt.parseInt(u8, val, 10) else if (std.mem.eql(u8, k, "--global")) a.global = val[0] == '1' else if (std.mem.eql(u8, k, "--quick")) a.quick = val[0] == '1' else if (std.mem.eql(u8, k, "--deadline")) a.deadline_s = try std.fmt.parseInt(u64, val, 10) else return error.BadFlag;
    }
    if (a.dev.len == 0 or a.rank > 1) return error.BadFlag;
    return a;
}

/// One line from stdin, without its newline.
fn readLine(buf: []u8) ![]const u8 {
    var n: usize = 0;
    while (n < buf.len) {
        var c: [1]u8 = undefined;
        if (std.c.read(0, &c, 1) != 1) return error.EndOfInput;
        if (c[0] == '\n') return buf[0..n];
        buf[n] = c[0];
        n += 1;
    }
    return error.LineTooLong;
}

/// HELLO lid gid gid_index qpn psn qpn psn: the collective queue pair, then the measurement queue pair.
fn hello(a: vqp.Info, b: vqp.Info) void {
    out("HELLO {d} {s} {d} {d} {d} {d} {d}", .{ a.lid, &std.fmt.bytesToHex(a.gid, .lower), a.gid_index, a.qpn, a.psn, b.qpn, b.psn });
}

fn parseHello(line: []const u8) ![2]vqp.Info {
    var it = std.mem.tokenizeScalar(u8, line, ' ');
    if (!std.mem.eql(u8, it.next() orelse return error.BadHello, "HELLO")) return error.BadHello;
    var info: vqp.Info = .{ .qpn = 0, .psn = 0, .lid = 0, .gid = undefined, .gid_index = 0 };
    info.lid = try std.fmt.parseInt(u16, it.next() orelse return error.BadHello, 10);
    _ = try std.fmt.hexToBytes(&info.gid, it.next() orelse return error.BadHello);
    info.gid_index = try std.fmt.parseInt(u8, it.next() orelse return error.BadHello, 10);
    var both: [2]vqp.Info = .{ info, info };
    for (&both) |*b| {
        b.qpn = try std.fmt.parseInt(u32, it.next() orelse return error.BadHello, 10);
        b.psn = try std.fmt.parseInt(u32, it.next() orelse return error.BadHello, 10);
    }
    return both;
}

/// Exit when stdin closes: the run ends with the session that started it.
fn tether() void {
    var c: [1]u8 = undefined;
    while (std.c.read(0, &c, 1) == 1) {}
    out("R stdin_closed", .{});
    std.process.exit(4);
}

fn watchdog(seconds: u64) void {
    fabric.words.pause(seconds * std.time.ns_per_s);
    out("R timeout seconds={d}", .{seconds});
    std.process.exit(3);
}

fn mapped(len: usize) ![]align(vl.page) u8 {
    return std.posix.mmap(null, len, .{ .READ = true, .WRITE = true }, .{ .TYPE = .PRIVATE, .ANONYMOUS = true }, -1, 0);
}

/// Two queue pairs on one device: collectives on one, raw measurements on the other, so their receives never mix.
fn run(gpa: std.mem.Allocator, v: *const Verbs, a: Args) !void {
    _ = try std.Thread.spawn(.{}, watchdog, .{a.deadline_s});
    out("PID {d}", .{std.c.getpid()});
    var peer: vl.Peer = .{ .ep = try vqp.Endpoint.open(v, a.dev, a.kind, 4095, a.gid) };
    defer peer.ep.close();
    var raw = try peer.ep.sibling(a.kind, 4095);
    defer raw.close();
    const mem = try mapped(suite.ex_ring + suite.ex_stage + suite.ex_tx + suite.raw_ring + suite.raw_stage + suite.source_bytes);
    defer std.posix.munmap(mem);
    var at: usize = 0;
    const parts = .{ suite.ex_ring, suite.ex_stage, suite.ex_tx, suite.raw_ring, suite.raw_stage, suite.source_bytes };
    var slices: [6][]align(vl.page) u8 = undefined;
    inline for (parts, 0..) |len, i| {
        slices[i] = @alignCast(mem[at..][0..len]);
        at += len;
    }
    peer.rx = try vl.Region.register(&peer.ep, slices[0]);
    defer peer.rx.deregister(&peer.ep);
    peer.tx = try vl.Region.register(&peer.ep, slices[2]);
    defer peer.tx.deregister(&peer.ep);
    var raw_rx = try vl.Region.register(&raw, slices[3]);
    defer raw_rx.deregister(&raw);
    var src = try vl.Region.register(&raw, slices[5]);
    defer src.deregister(&raw);
    hello(try peer.ep.info(), try raw.info());
    var line: [512]u8 = undefined;
    const theirs = try parseHello(try readLine(&line));
    try peer.ep.connect(theirs[0], peer.ep.port.active_mtu, .{ .global = a.global });
    try raw.connect(theirs[1], raw.port.active_mtu, .{ .global = a.global });
    peer.ring = try vl.Ring.init(&peer.ep, peer.rx.mem, peer.rx.mr.lkey, slices[1]);
    var ring = try vl.Ring.init(&raw, raw_rx.mem, raw_rx.mr.lkey, slices[4]);
    out("CONNECTED kind={s} mtu={d} in_order={?d} ex_slots={d} raw_slots={d}", .{ @tagName(a.kind), peer.ep.port.active_mtu, peer.ep.inOrder(), peer.ring.slots, ring.slots });
    if (!std.mem.eql(u8, try readLine(&line), "GO")) return error.NoGo;
    out("POSTED", .{});
    if (!std.mem.eql(u8, try readLine(&line), "GO")) return error.NoGo;
    _ = try std.Thread.spawn(.{}, tether, .{});
    var peers = [2]?*vl.Peer{ null, null };
    peers[1 - a.rank] = &peer;
    var f: vl.Fabric = .{ .me = a.rank, .peers = &peers };
    var ctx: suite.Ctx = .{ .gpa = gpa, .rank = a.rank, .raw = &raw, .ring = &ring, .source = &src, .x = sr.Exchange.init(f.link()), .quick = a.quick };
    try suite.all(&ctx);
    out("DONE", .{});
}

pub fn main(init: std.process.Init) !void {
    const arena = init.arena.allocator();
    const args = try init.minimal.args.toSlice(arena);
    if (args.len < 3) {
        std.debug.print("usage: tf-linktest probe DEV... | run --dev DEV --rank 0|1 [--kind uc|rc] [--gid N] [--global 0|1] [--quick 0|1]\n", .{});
        std.process.exit(2);
    }
    var v = try Verbs.open(null);
    defer v.close();
    if (std.mem.eql(u8, args[1], "probe")) {
        for (args[2..]) |dev| probe.device(&v, dev) catch |err| out("R error dev={s} what={s}", .{ dev, @errorName(err) });
        out("DONE", .{});
        return;
    }
    if (!std.mem.eql(u8, args[1], "run")) std.process.exit(2);
    const a = try parse(args);
    run(init.gpa, &v, a) catch |err| {
        out("R error what={s} errno={d}", .{ @errorName(err), std.c._errno().* });
        std.process.exit(1);
    };
}
