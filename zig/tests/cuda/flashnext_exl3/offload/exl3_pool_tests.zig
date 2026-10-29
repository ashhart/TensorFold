//! exl3-pool: the Flash Next EXL3 expert offload pool (cuda_exl3_pool.zig) on a real pack; this hub parses the mode.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("../../check.zig");
const data = @import("exl3_pool_data.zig");
const equal = @import("exl3_pool_equal_tests.zig");
const bench = @import("exl3_pool_bench_tests.zig");
const hits = @import("exl3_pool_hits_tests.zig");

const exl3e = cuda.exl3_experts;
const Gpu = check.Gpu;
const Pack = data.Pack;
const LayerData = data.LayerData;

fn parseLayers(spec: []const u8, out: []usize) !usize {
    var n: usize = 0;
    var it = std.mem.splitScalar(u8, spec, ',');
    while (it.next()) |tok| {
        if (n == out.len) return error.TooManyLayers;
        out[n] = try std.fmt.parseInt(usize, tok, 10);
        n += 1;
    }
    return n;
}

pub fn run(gpu: Gpu, rest: []const [:0]const u8) !void {
    if (rest.len < 1) return error.MissingArgument;
    const dir = rest[0];
    var mode: []const u8 = "all";
    var layer_spec: []const u8 = "0,17";
    if (rest.len > 1) {
        if (std.ascii.isDigit(rest[1][0])) layer_spec = rest[1] else mode = rest[1];
    }
    if (std.mem.eql(u8, mode, "hit")) return hits.hitRun(gpu, rest[2..]);
    if (rest.len > 2) layer_spec = rest[2];
    var layers: [8]usize = undefined;
    const n_layers = try parseLayers(layer_spec, &layers);
    var pack: Pack = .{};
    defer pack.close(gpu.io);
    try pack.open(gpu, dir);
    var ex = try exl3e.Experts.load(gpu.d);
    defer ex.unload();
    var s = try cuda.Stream.init(gpu.d, true);
    defer s.deinit();
    for (layers[0..n_layers]) |layer| {
        std.debug.print("== layer {d}\n", .{layer});
        const ld = try LayerData.load(gpu, &pack, layer);
        defer ld.free(gpu.gpa);
        const all = std.mem.eql(u8, mode, "all");
        if (all or std.mem.eql(u8, mode, "equal")) {
            try equal.equalRun(gpu, &ex, s, ld, 10, 0, 0x10 + layer);
            try equal.equalRun(gpu, &ex, s, ld, 24, 4, 0x24 + layer);
            try equal.equalRun(gpu, &ex, s, ld, 64, 0, 0x64 + layer);
        }
        if (all or std.mem.eql(u8, mode, "graph")) try equal.graphRun(gpu, &ex, s, ld);
        if (std.mem.eql(u8, mode, "bench")) try bench.benchRun(gpu, &ex, s, ld);
    }
}
