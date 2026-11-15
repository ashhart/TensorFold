//! flashnext-exl3-graph: the EXL3 offload pool planned on the device (cuda_exl3_dplan.zig) and the decode step as one
const std = @import("std");
const check = @import("../../check.zig");

const Gpu = check.Gpu;
const harness = @import("flashnext_exl3_graph_engine.zig");
const Eng = harness.Eng;
const plan_tests = @import("flashnext_exl3_graph_plan_tests.zig");
const eq_tests = @import("flashnext_exl3_graph_eq_tests.zig");
const prefill_tests = @import("flashnext_exl3_graph_prefill_tests.zig");
const snapshot_tests = @import("flashnext_exl3_graph_snapshot_tests.zig");

pub fn run(gpu: Gpu, rest: []const [:0]const u8) !void {
    if (rest.len >= 2 and std.mem.eql(u8, rest[0], "ngram")) return prefill_tests.ngramMode(gpu, rest[1]);
    if (rest.len < 1) return error.MissingArgument;
    if (std.mem.eql(u8, rest[0], "plan")) return plan_tests.planMode(gpu);
    if (std.mem.eql(u8, rest[0], "frequency")) return @import("../offload/exl3_frequency_tests.zig").run(gpu, rest[1..]);
    if (rest.len < 6) return error.MissingArgument;
    const rank = try std.fmt.parseInt(usize, rest[0], 10);
    const world = try std.fmt.parseInt(usize, rest[1], 10);
    const port = try std.fmt.parseInt(u16, rest[2], 10);
    const dir = rest[3];
    const mode = rest[4];
    if ((world != 1 and world != 2) or rank >= world) return error.BadRank;
    const steps = if (rest.len > 6 and !std.mem.eql(u8, mode, "prefill-prof")) try std.fmt.parseInt(usize, rest[6], 10) else 48;
    const which: []const u8 = if (rest.len > 7) rest[7] else "dplan,graph";
    if (std.mem.eql(u8, mode, "rebind")) {
        const engine = try Eng.init(gpu, rank, world, port, dir);
        defer engine.deinit();
        return @import("../kv/exl3_state_rebind_tests.zig").run(engine, .{ .reset = Eng.reset, .forward = Eng.forward, .swap = Eng.swap });
    }
    if (std.mem.eql(u8, mode, "snapshot")) return snapshot_tests.snapshotMode(gpu, rank, world, port, dir, steps);
    if (std.mem.eql(u8, mode, "fork")) return eq_tests.forkMode(gpu, rank, world, port, dir, rest[5], steps, which);
    if (std.mem.eql(u8, mode, "prof")) return prefill_tests.profMode(gpu, rank, world, port, dir, try std.fmt.parseInt(usize, rest[5], 10), steps, which);
    if (std.mem.eql(u8, mode, "prefill")) return prefill_tests.prefillMode(gpu, rank, world, port, dir, try std.fmt.parseInt(usize, rest[5], 10), steps);
    if (std.mem.eql(u8, mode, "prefill-prof")) return prefill_tests.prefillProfMode(gpu, rank, world, port, dir, try std.fmt.parseInt(usize, rest[5], 10), if (rest.len > 6) rest[6] else "graph");
    if (std.mem.eql(u8, mode, "eq")) return eq_tests.eqMode(gpu, rank, world, port, dir, rest[5], steps, which);
    return error.UnknownMode;
}
