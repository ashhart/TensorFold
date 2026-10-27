//! exl3-prefill: the pieces of a Flash Next EXL3 prefill, timed on the pack's real tensors (one idle card).

const std = @import("std");
const cuda = @import("cuda");
const check = @import("../../check.zig");
const exl3e = cuda.exl3_experts;
const Gpu = check.Gpu;
const packs = @import("exl3_prefill_pack.zig");
const experts = @import("exl3_prefill_experts.zig");
const copies = @import("exl3_prefill_copy.zig");
const dense = @import("exl3_prefill_dense.zig");

const LayerData = packs.LayerData;
const Pack = packs.Pack;
const expertsRun = experts.expertsRun;
const copyRun = copies.copyRun;
const denseRun = dense.denseRun;

pub fn run(gpu: Gpu, rest: []const [:0]const u8) !void {
    if (rest.len < 2) return error.MissingArgument;
    const dir = rest[0];
    const mode = rest[1];
    const layer: usize = if (rest.len > 2) try std.fmt.parseInt(usize, rest[2], 10) else 17;
    var pack: Pack = .{};
    defer pack.close(gpu.io);
    try pack.open(gpu, dir);
    var s = try cuda.Stream.init(gpu.d, true);
    defer s.deinit();
    if (std.mem.eql(u8, mode, "dense")) return denseRun(gpu, &pack, s);
    var ex = try exl3e.Experts.load(gpu.d);
    defer ex.unload();
    const ld = try LayerData.load(gpu, &pack, layer);
    defer ld.free(gpu.gpa);
    if (std.mem.eql(u8, mode, "experts")) return expertsRun(gpu, &ex, s, ld);
    if (std.mem.eql(u8, mode, "copy")) return copyRun(gpu, &ex, s, ld);
    return error.UnknownMode;
}
