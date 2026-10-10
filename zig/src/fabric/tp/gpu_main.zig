//! `tf-tp-test suite | failfast RANK fatal|kill | roce-host`: one rank of a two-process TP=2 test on real GPUs.
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
        std.debug.print("usage: tf-tp-test suite | failfast RANK fatal|kill | roce-host\n", .{});
        return 2;
    }
    const cfg = try tp.Config.fromEnv();
    if (std.mem.eql(u8, args[1], "roce-host")) return @import("roce_host.zig").run(cfg, init.gpa);
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

fn realNow() f64 {
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
