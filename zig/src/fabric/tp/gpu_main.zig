//! `tf-tp-test suite | failfast RANK fatal|kill|transport [N [HOLD_S]] | roce-host | kernel`: one rank of a TP=2 test.
const std = @import("std");
const cuda = @import("cuda");
const tp = @import("tp");
const tp_kernel = @import("tp_kernel");
const checks = @import("gpu_checks.zig");
const bench = @import("gpu_bench.zig");

const max_bytes = (4 << 20) + 4096;

pub fn main(init: std.process.Init) !u8 {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 2) {
        std.debug.print("usage: tf-tp-test suite | failfast RANK fatal|kill|transport [N [HOLD_S]] | roce-host | kernel\n", .{});
        return 2;
    }
    // `kernel`: whether the build embeds the mailbox kernel (-Dtp-nvcc or -Dtp-fatbin), for scripts to check first
    if (std.mem.eql(u8, args[1], "kernel")) {
        std.debug.print("mailbox kernel {d} B\n", .{tp_kernel.mailbox.len});
        return if (tp_kernel.mailbox.len > 0) 0 else 1;
    }
    const cfg = try tp.Config.fromEnv();
    if (std.mem.eql(u8, args[1], "roce-host")) return @import("roce_host.zig").run(cfg, init.gpa);
    if (std.mem.eql(u8, args[1], "failfast") and (!cfg.failfast or args.len > 5) and !hostOnlyAllowed(cfg)) {
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
    if (std.mem.eql(u8, args[1], "suite")) return suite(&s, &driver, init.gpa, cfg);
    if (std.mem.eql(u8, args[1], "failfast") and args.len >= 4 and std.mem.eql(u8, args[3], "transport")) {
        const n = if (args.len > 4) try std.fmt.parseInt(u32, args[4], 10) else 50;
        const hold = if (args.len > 5) try std.fmt.parseInt(u32, args[5], 10) else 0;
        return @import("gpu_transport.zig").run(&s, &driver, init.gpa, cfg, try std.fmt.parseInt(u32, args[2], 10), n, hold);
    }
    if (std.mem.eql(u8, args[1], "failfast") and args.len == 4) return failfast(&s, &driver, init.gpa, cfg, try std.fmt.parseInt(u32, args[2], 10), args[3]);
    std.debug.print("unknown command {s}\n", .{args[1]});
    return 2;
}

fn suite(s: *tp.Session, d: *const cuda.Driver, gpa: std.mem.Allocator, cfg: tp.Config) !u8 {
    var c = try checks.Ctx.init(d, gpa, cfg.rank, max_bytes);
    defer c.deinit();
    var backends: [2]struct { comm: tp.Collective, label: []const u8 } = undefined;
    var n_backends: usize = 1;
    backends[0] = .{ .comm = s.nccl.iface(), .label = "nccl" };
    if (s.comm().kind() != .nccl) {
        backends[1] = .{ .comm = s.comm(), .label = if (s.roce != null) "roce" else "mailbox" };
        n_backends = 2;
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
    // the stop: rank 0 marks it expected first, then both meet and close
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

/// Exchanges until the victim fails; the survivor is left waiting inside one, which only the fate channel ends.
fn failfast(s: *tp.Session, d: *const cuda.Driver, gpa: std.mem.Allocator, cfg: tp.Config, victim: u32, mode: []const u8) !u8 {
    var c = try checks.Ctx.init(d, gpa, cfg.rank, 1 << 20);
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
