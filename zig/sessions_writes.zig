//! tf-sessions-writes: the sessions NVMe tier's bytes written and read back on synthetic workloads, no model or GPU.
const std = @import("std");
const tier = @import("sessions_writes_tier.zig");
const load = @import("sessions_writes_load.zig");

const Variant = struct { name: []const u8, cfg: tier.Config };

fn variants() [5]Variant {
    var covered = tier.dsv41();
    covered.policy.skip_covered = true;
    covered.policy.delta = true;
    var min4k = covered;
    min4k.min_tokens = 4096;
    var min16k = covered;
    min16k.min_tokens = 16_384;
    var turn = tier.dsv41();
    turn.saves = .turn;
    return .{
        .{ .name = "dsv41 defaults", .cfg = tier.dsv41() },
        .{ .name = "+SKIP_COVERED +DELTA", .cfg = covered },
        .{ .name = "+SKIP_COVERED +DELTA, DISK_MIN=4096", .cfg = min4k },
        .{ .name = "+SKIP_COVERED +DELTA, DISK_MIN=16384", .cfg = min16k },
        .{ .name = "turn saves (PROMPT_TAIL=prefill)", .cfg = turn },
    };
}

const gb: f64 = 1e9;

fn f(x: u64) f64 {
    return @floatFromInt(x);
}

fn ratio(a: u64, b: u64) f64 {
    return if (b == 0) std.math.inf(f64) else f(a) / f(b);
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    var scale: u32 = 1;
    if (args.len == 3 and std.mem.eql(u8, args[1], "--divide")) scale = try std.fmt.parseInt(u32, args[2], 10) else if (args.len != 1) {
        std.debug.print("tf-sessions-writes [--divide N]: every workload's request count divided by N\n", .{});
        return error.UnknownOption;
    }
    var buf: [16 << 10]u8 = undefined;
    var out = std.Io.File.stdout().writerStreaming(init.io, &buf);
    const w = &out.interface;
    {
        const b = try tier.Bench.init(gpa, init.io, tier.dsv41());
        defer b.deinit();
        try w.print("snapshot (bounded state) {d} B; whole files:", .{b.cfg.snapshot.bytes()});
        for ([_]u64{ 1024, 4096, 16_384, 65_536, 131_072 }) |pos| try w.print(" {d} tok {d} B;", .{ pos, try b.fileBytes(pos, 0) });
        try w.print(" deltas on a 64k base: +256 tok {d} B, +2048 tok {d} B\n\n", .{ try b.fileBytes(65_536 + 256, 256), try b.fileBytes(65_536 + 2048, 256) });
    }
    try w.writeAll("| workload | policy | requests | spills (waits) | files (deltas) | written GB | read GB | NVMe-restored Mtok | RAM-restored Mtok | B written / NVMe tok | B written / restored tok | WA | MB / request | GB/day 1 rps | GB/day 10 rps | peak tier GB |\n");
    try w.writeAll("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n");
    for ([_]load.Workload{ .agent, .chats, .mix }) |wl| for (variants()) |v| {
        var sh = load.shape(wl);
        sh.requests = @max(2, sh.requests / scale);
        const b = try tier.Bench.init(gpa, init.io, v.cfg);
        defer b.deinit();
        const r = try load.run(gpa, b, wl, sh, 0xEF01);
        const e = r.end;
        const steady = r.end.minus(r.half);
        const per = ratio(steady.written, steady.requests);
        try w.print("| {t} | {s} | {d} | {d} ({d}) | {d} ({d}) | {d:.2} | {d:.2} | {d:.2} | {d:.2} | {d:.0} | {d:.0} | {d:.2} | {d:.1} | {d:.0} | {d:.0} | {d:.2} |\n", .{
            wl,                     v.name,                e.requests,                      e.spills,                                       e.waits,                  e.files,   e.deltas,          f(e.written) / gb,  f(e.read) / gb,
            f(e.disk_tokens) / 1e6, f(e.ram_tokens) / 1e6, ratio(e.written, e.disk_tokens), ratio(e.written, e.disk_tokens + e.ram_tokens), ratio(e.written, e.read), per / 1e6, per * 86_400 / gb, per * 864_000 / gb, f(b.peak_used) / gb,
        });
        try w.flush();
    };
    try w.flush();
}
