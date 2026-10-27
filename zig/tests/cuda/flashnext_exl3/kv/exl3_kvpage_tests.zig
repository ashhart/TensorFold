//! exl3-kvpage: dynamic KV paging for long contexts (docs/plans/exl3-kv-paging.md) on the family's Engine with the

const std = @import("std");
const check = @import("../../check.zig");
const Gpu = check.Gpu;
const engs = @import("exl3_kvpage_eng.zig");
const lives = @import("exl3_kvpage_life.zig");
const cmps = @import("exl3_kvpage_cmp.zig");
const kernels = @import("exl3_kvpage_kernel.zig");

const snapshotLifecycle = lives.snapshotLifecycle;
const traceRun = lives.traceRun;
const cmpRun = cmps.cmpRun;
const kernelReplay = kernels.kernelReplay;
const kernelRun = kernels.kernelRun;
const overflowReplay = kernels.overflowReplay;

pub fn run(gpu: Gpu, rest: []const [:0]const u8) !void {
    if (rest.len < 2) return error.MissingArgument;
    const dir = rest[0];
    const mode = rest[1];
    if (std.mem.eql(u8, mode, "trace")) return traceRun(gpu, dir, rest[2..]);
    if (std.mem.eql(u8, mode, "cmp")) return cmpRun(gpu, dir, rest[2..]);
    if (std.mem.eql(u8, mode, "kernel")) return kernelRun(gpu, if (rest.len > 2) try std.fmt.parseInt(usize, rest[2], 10) else 6144);
    if (std.mem.eql(u8, mode, "kernel-graph")) return kernelReplay(gpu, 49152, true, false);
    if (std.mem.eql(u8, mode, "window-graph")) return kernelReplay(gpu, 49152, true, true);
    if (std.mem.eql(u8, mode, "overflow-graph")) return overflowReplay(gpu);
    if (std.mem.eql(u8, mode, "snapshot")) return snapshotLifecycle(gpu);
    if (std.mem.eql(u8, mode, "recycle")) return @import("../offload/exl3_recycle_tests.zig").run(gpu);
    return error.UnknownMode;
}

test {
    _ = @import("exl3_kvpage_snapshot_tests.zig");
}
