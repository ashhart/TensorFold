//! flashnext-exl3-tpe: the Flash Next EXL3 pack through the family Engine at tp=1 or tp=2; this hub parses the mode.
const std = @import("std");
const check = @import("../../check.zig");
const decode = @import("flashnext_exl3_tpe_decode_tests.zig");
const trace = @import("flashnext_exl3_tpe_trace_tests.zig");
const hits = @import("flashnext_exl3_tpe_hits_tests.zig");
const graph = @import("flashnext_exl3_tpe_graph_tests.zig");

const Gpu = check.Gpu;

pub fn run(gpu: Gpu, rest: []const [:0]const u8) !void {
    if (rest.len < 5) return error.MissingArgument;
    const rank = try std.fmt.parseInt(usize, rest[0], 10);
    const world = try std.fmt.parseInt(usize, rest[1], 10);
    const port = try std.fmt.parseInt(u16, rest[2], 10);
    const dir = rest[3];
    const mode = rest[4];
    if ((world != 1 and world != 2) or rank >= world) return error.BadRank;
    if (std.mem.eql(u8, mode, "hit")) return hits.hitMode(gpu, rest[5..]);
    if (std.mem.eql(u8, mode, "policy")) return hits.policyMode(gpu, rest[5..]);
    if (std.mem.eql(u8, mode, "graph")) return graph.graphMode(gpu, rank, port);
    if (std.mem.eql(u8, mode, "bench")) return decode.benchMode(gpu, rank, world, port, dir);
    if (std.mem.eql(u8, mode, "trace")) {
        if (rest.len < 7) return error.MissingArgument;
        const rounds = if (rest.len > 7) try std.fmt.parseInt(usize, rest[7], 10) else 1024;
        if (rest.len > 10) return error.UnexpectedArgument;
        const schedule = if (rest.len > 8) rest[8] else "0";
        const repeats = if (rest.len > 9) try std.fmt.parseInt(usize, rest[9], 10) else 1;
        return trace.traceMode(gpu, rank, world, port, dir, rest[5], rest[6], rounds, schedule, repeats);
    }
    if (rest.len < 7) return error.MissingArgument;
    if (std.mem.eql(u8, mode, "tf")) return decode.tfMode(gpu, rank, world, port, dir, rest[5], rest[6], false);
    if (std.mem.eql(u8, mode, "tfc")) return decode.tfMode(gpu, rank, world, port, dir, rest[5], rest[6], true);
    if (std.mem.eql(u8, mode, "gen")) return decode.genMode(gpu, rank, world, port, dir, rest[5], rest[6]);
    return error.UnknownMode;
}
