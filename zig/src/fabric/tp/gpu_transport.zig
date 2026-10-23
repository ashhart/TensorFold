//! `tf-tp-test failfast V transport [N [HOLD_S]]`: the failed-transport path on real HCAs, injected on an idle link.
const std = @import("std");
const cuda = @import("cuda");
const tp = @import("tp");
const checks = @import("gpu_checks.zig");
const realNow = @import("gpu_main.zig").realNow;

/// N clean sums, an NCCL meet, the victim's `Roce.inject` once every link is idle, then its close (fail-fast: exit 70).
pub fn run(s: *tp.Session, d: *const cuda.Driver, gpa: std.mem.Allocator, cfg: tp.Config, victim: u32, n: u32, hold_s: u32) !u8 {
    const r = s.roce orelse {
        std.debug.print("failfast transport needs TF_COMM_BACKEND=roce (the session runs {t})\n", .{s.comm().kind()});
        return 2;
    };
    var c = try checks.Ctx.init(d, gpa, cfg.rank, cfg.world, 4096);
    defer c.deinit();
    const comm = s.comm();
    for (0..n) |_| {
        try comm.allReduce(c.in.ptr, c.out.ptr, 2048, .bf16, .sum, c.stream.handle);
        try c.stream.synchronize();
        try comm.check();
    }
    try s.nccl.iface().barrier();
    std.debug.print("rank {d}: {d} clean exchanges; both ranks met over NCCL\n", .{ cfg.rank, n });
    // the inject only on an idle link: every rank's posts completed, else no rank injects
    const t_idle = tp.sock.nowNs();
    while (!r.linkIdle() and tp.sock.nowNs() - t_idle < 2 * std.time.ns_per_s) tp.sock.sleepNs(std.time.ns_per_ms);
    var line: [384]u8 = undefined;
    std.debug.print("rank {d}: link {s} before the inject: {s}\n", .{ cfg.rank, if (r.linkIdle()) "idle" else "BUSY", r.describeLink(&line) });
    if (try idleRanks(s, &c, r.linkIdle()) != cfg.world) {
        std.debug.print("REFUSED rank {d}: a link was not idle; no inject, a clean stop\n", .{cfg.rank});
        if (s.fate) |*f| f.expectStop();
        try s.nccl.iface().barrier();
        s.close();
        return 2;
    }
    if (cfg.rank == victim) {
        std.debug.print("INJECT transport rank {d} at {d:.6}\n", .{ cfg.rank, realNow() });
        r.inject();
        comm.check() catch |e| std.debug.print("rank {d}: transport reports {t}\n", .{ cfg.rank, e });
    } else if (s.fate != null) {
        while (true) tp.sock.sleepNs(std.time.ns_per_s); // the fate channel ends this rank (exit 70)
    }
    const kept0 = tp.roce.Roce.kept.load(.monotonic);
    s.close(); // fail-fast on: exit 70 from here
    const kept = tp.roce.Roce.kept.load(.monotonic) - kept0;
    const want: u32 = @intFromBool(cfg.rank == victim);
    const live = cfg.rank != victim or mapped(r.region);
    std.debug.print("rank {d}: closed; regions kept {d} (want {d}){s}\n", .{ cfg.rank, kept, want, if (cfg.rank != victim) "" else if (live) ", region still mapped" else ", region NOT mapped" });
    if (hold_s > 0) {
        std.debug.print("HOLD rank {d} pid {d} for {d} s\n", .{ cfg.rank, std.c.getpid(), hold_s });
        tp.sock.sleepNs(@as(u64, hold_s) * std.time.ns_per_s);
    }
    const ok = kept == want and live;
    std.debug.print("{s} rank {d}\n", .{ if (ok) "PASS" else "FAIL", cfg.rank });
    return if (ok) 0 else 1;
}

fn mapped(region: []const u8) bool {
    var vec: [1]u8 = undefined;
    return std.os.linux.errno(std.os.linux.mincore(@constCast(region.ptr), 4096, &vec)) == .SUCCESS;
}

/// The ranks whose link was idle, summed over NCCL (one f32 a rank).
fn idleRanks(s: *tp.Session, c: *checks.Ctx, idle: bool) !u32 {
    var v: [1]f32 = .{@floatFromInt(@intFromBool(idle))};
    try c.in.uploadAsync(0, std.mem.asBytes(&v), c.stream.handle);
    try s.nccl.iface().allReduce(c.in.ptr, c.out.ptr, 1, .f32, .sum, c.stream.handle);
    try c.stream.synchronize();
    try c.out.download(0, std.mem.asBytes(&v));
    return @intFromFloat(v[0]);
}
