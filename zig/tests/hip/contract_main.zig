//! tf-hip-kernel-contract: HIP kernels through gates 1-4, the controls and gate 5; `limits DIR` audits code objects.
const std = @import("std");
const hip = @import("hip");
const core = @import("core");
const controls = @import("hip_controls");
const gate5 = @import("contract_limits.zig");
const rows_k = @import("contract_rows.zig");
const stream_k = @import("contract_stream.zig");
const mamba_k = @import("contract_mamba.zig");
const scan_k = @import("contract_scan.zig");
const moe_k = @import("contract_nemo_moe.zig");

const g = core.contract.gates;

pub fn main(init: std.process.Init) !u8 {
    const a = init.gpa;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len == 1) return if (try run(a, init.io) > 0) 1 else 0;
    if (args.len != 3 or !std.mem.eql(u8, args[1], "limits")) {
        std.debug.print("usage: tf-hip-kernel-contract [limits DIR]\n", .{});
        return 2;
    }
    const n = try gate5.objects(a, init.io, args[2]);
    if (n > 0) {
        std.debug.print("FAIL kernel-contract-limits: {d} problems\n", .{n});
        return 1;
    }
    std.debug.print("PASS kernel-contract-limits: every registered launch fits gfx1030, gfx1100, gfx1151 and " ++
        "gfx1201, every record current, every control red\n", .{});
    return 0;
}

fn ms(io: std.Io) i96 {
    return @divTrunc(std.Io.Clock.awake.now(io).toNanoseconds(), std.time.ns_per_ms);
}

/// Gates 1-4 on every registered HIP kernel and each control at its own gate, then gate 5; returns the problems.
fn run(a: std.mem.Allocator, io: std.Io) !usize {
    var built: [2]usize = .{ hip.kernels.images[0].len, controls.bytes.len };
    _ = &built; // runtime-known, so a stub build still compiles all that follows
    if (built[0] == 0 or built[1] == 0) {
        std.debug.print("FAIL kernel-contract: built without hipcc, so no gate is proven\n", .{});
        return 1;
    }
    var r = try hip.Runtime.open();
    defer r.close();
    var ctx = try hip.Context.init(&r, 0);
    defer ctx.deinit();
    var arch_buffer: [256]u8 = undefined;
    const arch = try hip.Device.archName(&r, 0, &arch_buffer);
    if (!std.mem.eql(u8, arch, hip.kernels.arch)) {
        const say = "FAIL kernel-contract: this GPU is {s}, the kernels are built for {s}\n";
        std.debug.print(say, .{ arch, hip.kernels.arch });
        return 1;
    }
    var s = try hip.Stream.init(&r);
    defer s.deinit();
    var l = try hip.Launcher.load(&r, .{}, hip.kernels.images);
    defer l.unload();
    var ctl = try hip.Module.load(&r, &controls.bytes);
    defer ctl.unload();
    var kernels: std.ArrayList(g.Kernel) = .empty;
    defer kernels.deinit(a);
    const which = comptime std.enums.values(rows_k.Which);
    var row_list: [which.len]rows_k.Row = undefined;
    var nrows: usize = 0;
    defer for (row_list[0..nrows]) |*x| x.deinit();
    for (which) |w| {
        row_list[nrows] = try rows_k.Row.init(a, &r, s, &l, ctl, w);
        nrows += 1;
    }
    for (row_list[0..nrows]) |*x| try kernels.append(a, x.kernel());
    var mambas: [3]mamba_k.Mamba = undefined;
    for (&mambas, std.enums.values(mamba_k.Which)) |*m, w| m.* = .init(a, &r, s, &l, w);
    defer for (&mambas) |*m| m.deinit();
    for (&mambas) |*m| try kernels.append(a, m.kernel());
    var scans: [2]scan_k.Scan = undefined;
    for (&scans, std.enums.values(scan_k.Part)) |*m, p| m.* = .init(a, &r, s, &l, p);
    defer for (&scans) |*m| m.deinit();
    for (&scans) |*m| try kernels.append(a, m.kernel());
    var picks: [3]moe_k.Moe = undefined;
    for (&picks, [_]moe_k.Which{ .pass, .rank, .localize }) |*m, w| m.* = .init(a, &r, s, &l, w);
    defer for (&picks) |*m| m.deinit();
    for (&picks) |*m| try kernels.append(a, m.kernel());
    var moes: [3]moe_k.Moe = undefined;
    for (&moes, [_]moe_k.Which{ .relu2, .combine, .combine_f32 }) |*m, w| m.* = .init(a, &r, s, &l, w);
    defer for (&moes) |*m| m.deinit();
    for (&moes) |*m| try kernels.append(a, m.kernel());
    var streams: [stream_k.cases.len]stream_k.Stream = undefined;
    for (&streams, 0..) |*x, i| x.* = .init(a, &r, s, &l, i);
    defer for (&streams) |*x| x.deinit();
    for (&streams) |*x| try kernels.append(a, x.kernel());
    var t: g.Tally = .{};
    const t0 = ms(io);
    try g.runAll(a, kernels.items, 0x6b63, &t);
    const t1 = ms(io);
    const lim = try gate5.table(a) + try gate5.live(a, &r, ctx, arch);
    const t2 = ms(io);
    const tally = .{ t.kernels, t.passed, t.failed, t.known, t.controls, t.caught, t.missed, lim, t1 - t0, t2 - t1 };
    std.debug.print("SUMMARY {s} {d} kernels: {d} pass gates 1-4, {d} fail, {d} known findings; {d} controls: " ++
        "{d} red as planted, {d} missed; {d} limit problems; gates {d} ms, limits {d} ms\n", .{arch} ++ tally);
    return t.problems() + lim;
}
