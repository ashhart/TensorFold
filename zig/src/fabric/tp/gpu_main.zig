//! `tf-tp-test suite | failfast RANK fatal|kill|transport [N [HOLD_S]] | roce-host | hca | kernel`: one rank.
const std = @import("std");
const cuda = @import("cuda");
const tp = @import("tp");
const tp_kernel = @import("tp_kernel");
const checks = @import("gpu_checks.zig");
const bench = @import("gpu_bench.zig");
const cli = @import("gpu_cli.zig");

const max_bytes = (4 << 20) + 4096;

pub fn main(init: std.process.Init) !u8 {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const command = cli.parse(args[1..]) catch {
        std.debug.print("{s}", .{cli.usage});
        return 2;
    };
    // `kernel`: whether the build embeds the mailbox kernel (-Dtp-nvcc or -Dtp-fatbin), for scripts to check first
    if (command == .kernel) {
        std.debug.print("mailbox kernel {d} B\n", .{tp_kernel.mailbox.len});
        return if (tp_kernel.mailbox.len > 0) 0 else 1;
    }
    const cfg = try tp.Config.fromEnv();
    if (command == .@"roce-host") return @import("roce_host.zig").run(cfg, init.gpa);
    if (command == .hca) return hca(cfg, init.io);
    if (command == .failfast and (!cfg.failfast or command.failfast.hold != null) and !hostOnlyAllowed(cfg)) {
        std.debug.print("REFUSED: failfast with TF_TP_FAILFAST=0 or a hold is host-only: TF_TP_ALLOW_HOST_ONLY=1 and no RDMA wire\n", .{});
        return 2;
    }
    if (cfg.backend != .nccl and tp_kernel.mailbox.len == 0) {
        std.debug.print("FAIL TF_COMM_BACKEND={t} needs the mailbox kernel: rebuild with -Dtp-nvcc or -Dtp-fatbin\n", .{cfg.backend});
        return 2;
    }
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, @intCast(cfg.device));
    defer ctx.deinit();
    var name_buf: [128]u8 = undefined;
    const t0 = tp.sock.nowNs();
    var s: tp.Session = undefined;
    try s.open(&driver, cfg, tp_kernel.mailbox);
    const up_ms = @as(f64, @floatFromInt(tp.sock.nowNs() - t0)) / 1e6;
    std.debug.print("INFO rank {d}/{d} on {s} (sm_{d}), backend {t}, mailbox kernel {d} B, bootstrap + NCCL init {d:.0} ms\n", .{ cfg.rank, cfg.world, try ctx.name(&name_buf), try ctx.capability(), s.comm().kind(), tp_kernel.mailbox.len, up_ms });
    return switch (command) {
        .suite => suite(&s, &driver, init.gpa, cfg),
        .failfast => |f| if (f.mode == .transport)
            @import("gpu_transport.zig").run(&s, &driver, init.gpa, cfg, f.victim, f.n, f.hold orelse 0)
        else
            failfast(&s, &driver, init.gpa, cfg, f.victim, @tagName(f.mode)),
        else => unreachable,
    };
}

/// `hca` (host only, no GPU): both ranks' RoCE pick over the bootstrap; this rank's `NCCL_IB_HCA=...` line on stdout.
fn hca(cfg: tp.Config, io: std.Io) !u8 {
    var b = tp.Bootstrap.open(cfg) catch |e| {
        const s = @as(f64, @floatFromInt(cfg.connect_timeout_ns)) / std.time.ns_per_s;
        switch (e) {
            // a peer that never runs `hca` (it names its ports, or is not up) leaves this rank waiting until here
            error.Timeout => std.debug.print("FAIL hca rank {d}: no peer joined within TF_TP_PLAN_TIMEOUT ({d:.1} s): is its runner up, and leaving TF_TP_ROCE_HCA and NCCL_IB_HCA unset as this one does?\n", .{ cfg.rank, s }),
            error.AddressInUse => std.debug.print("FAIL hca rank {d}: port {d} (TF_TP_PORT + 7) is in use on this host\n", .{ cfg.rank, cfg.link_port }),
            error.Closed => std.debug.print("FAIL hca rank {d}: the peer closed the bootstrap connection\n", .{cfg.rank}),
            else => std.debug.print("FAIL hca rank {d}: bootstrap: {t}\n", .{ cfg.rank, e }),
        }
        return 1;
    };
    defer b.close();
    var specs: [tp.roce_verbs.max_hcas]tp.roce_verbs.HcaSpec = undefined;
    const pick = tp.roce_pick.select(cfg, &b, &specs) catch |e| {
        std.debug.print("FAIL hca rank {d}: {t} while picking the ports with the peer\n", .{ cfg.rank, e });
        return 1;
    };
    var buf: [tp.roce_pick.list_bytes + 16]u8 = undefined;
    const line = tp.roce_pick.hcaLine(pick, &specs, &buf) catch |e| {
        const why = switch (e) {
            error.VerbsUnavailable => "libibverbs.so.1 could not load or lacks required symbols (VerbsUnavailable); check the RDMA userspace installation",
            error.LibraryUnavailable => "could not load libibverbs.so.1 (LibraryUnavailable); check the RDMA userspace installation",
            error.MissingSymbol => "libibverbs.so.1 lacks required symbols (MissingSymbol); check the RDMA userspace installation",
            error.NoRoceDevice => "no active Ethernet RDMA port with an IPv4 RoCE v2 GID on this host: name the peer's ports in TF_TP_ROCE_HCA or NCCL_IB_HCA",
            error.NoSharedSubnet => "no RoCE port here shares an IPv4 /24 with the peer's: name the peer's ports in TF_TP_ROCE_HCA or NCCL_IB_HCA",
            else => @errorName(e),
        };
        std.debug.print("FAIL hca rank {d}: {s}\n", .{ cfg.rank, why });
        return 1;
    };
    var names: [tp.roce_pick.list_bytes]u8 = undefined;
    const how = if (pick.explicit) "TF_TP_ROCE_HCA" else "on a /24 the peer shares";
    std.debug.print("INFO hca rank {d}: {s} ({s})\n", .{ cfg.rank, try tp.roce_pick.joinNames(specs[0..pick.n], true, &names), how });
    try std.Io.File.stdout().writeStreamingAll(io, line);
    return 0;
}

fn suite(s: *tp.Session, d: *const cuda.Driver, gpa: std.mem.Allocator, cfg: tp.Config) !u8 {
    var c = try checks.Ctx.init(d, gpa, cfg.rank, cfg.world, max_bytes);
    defer c.deinit();
    var backends: [2]struct { comm: tp.Collective, label: []const u8 } = undefined;
    var n_backends: usize = 1;
    backends[0] = .{ .comm = s.nccl.iface(), .label = "nccl" };
    if (s.comm().kind() != .nccl) {
        backends[1] = .{ .comm = s.comm(), .label = if (s.roce != null) "roce" else "mailbox" };
        n_backends = 2;
    } else if (cfg.backend != .nccl and cfg.rank == 0) {
        std.debug.print("WARN TF_COMM_BACKEND={t} fell back to NCCL (see the [tensorfold] tp log lines above): only NCCL is measured\n", .{cfg.backend});
    }
    const sizes = [_]usize{ 6, 1000, 4096, 65536, 256 * 1024, 256 * 1024 + 16, 1 << 20, 4 << 20 };
    for (backends[0..n_backends]) |b| {
        for (sizes, 0..) |n, i| try c.collectives(b.comm, b.label, n, 100 * i);
        try c.pointToPoint(b.comm, b.label, 64 * 1024, 7);
        for ([_]usize{ 4096, 65536, 1 << 20 }) |n| try c.graphs(b.comm, b.label, n, 4);
        try b.comm.barrier();
    }
    for (backends[0..n_backends]) |b| {
        try bench.sweep(&c, b.comm, b.label, 4096, 4 << 20);
        try bench.decodeWindow(&c, b.comm, b.label, 86, 8192);
    }
    // the stop: rank 0 marks it expected first, then every rank meets and closes
    if (s.fate) |*f| f.expectStop();
    try s.comm().barrier();
    std.debug.print("SUMMARY rank {d}: {d} passed, {d} failed\n", .{ cfg.rank, c.passes, c.failures });
    s.close();
    return if (c.failures == 0) 0 else 1;
}

/// The failed-transport check's host-only forms (fail-fast off, a hold): TF_TP_ALLOW_HOST_ONLY=1 and no RDMA wire.
fn hostOnlyAllowed(cfg: tp.Config) bool {
    const flag = std.mem.eql(u8, std.mem.span(std.c.getenv("TF_TP_ALLOW_HOST_ONLY") orelse ""), "1");
    return flag and (cfg.backend != .roce or cfg.roce_wire == .shm);
}

pub fn realNow() f64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.REALTIME, &ts);
    return @as(f64, @floatFromInt(ts.sec)) + @as(f64, @floatFromInt(ts.nsec)) / 1e9;
}

/// Exchanges until the victim fails; the survivors are left waiting inside one, which only the fate channel ends.
fn failfast(s: *tp.Session, d: *const cuda.Driver, gpa: std.mem.Allocator, cfg: tp.Config, victim: u32, mode: []const u8) !u8 {
    var c = try checks.Ctx.init(d, gpa, cfg.rank, cfg.world, 1 << 20);
    defer c.deinit();
    const comm = s.comm();
    var k: u32 = 0;
    while (true) : (k += 1) {
        if (k == 50 and cfg.rank == victim) {
            std.debug.print("INJECT {s} rank {d} at {d:.6}\n", .{ mode, cfg.rank, realNow() });
            if (std.mem.eql(u8, mode, "kill")) _ = std.c.raise(.KILL);
            s.fatal("window", error.Injected);
            return 3; // not reached: the process exits 70
        }
        try comm.allReduce(c.in.ptr, c.out.ptr, 2048, .bf16, .sum, c.stream.handle);
        c.stream.synchronize() catch |e| {
            std.debug.print("rank {d}: stream error {t} after the peer failed\n", .{ cfg.rank, e });
        };
        comm.check() catch |e| {
            std.debug.print("rank {d}: transport reports {t} at exchange {d}; waiting for the fate channel\n", .{ cfg.rank, e, k });
            while (true) tp.sock.sleepNs(std.time.ns_per_s);
        };
    }
}
