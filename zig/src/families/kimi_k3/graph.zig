//! A resident K3 graph owns prepared projections and expert tables; checkpoint storage stays with its source.
const std = @import("std");
const mtl = @import("metal");
const Config = @import("config.zig").Config;
const weights = @import("weights.zig");
const store = @import("store.zig");
const prepare = @import("prepare.zig");
const Kernels = @import("kernels.zig").Kernels;
pub const k31_fixture = @embedFile("fixtures/k31-config.json");

pub const Graph = struct {
    a: std.mem.Allocator,
    c: Config,
    layers: []weights.Layer,
    ready: usize = 0,
    head: weights.Head = undefined,
    prepared: prepare.Owned,

    pub fn load(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, k: *const Kernels, c: Config, src: store.Source) !Graph {
        try nativeShape(c);
        var g: Graph = .{ .a = a, .c = c, .layers = try a.alloc(weights.Layer, c.layers), .prepared = .{ .gpa = a } };
        errdefer g.deinit();
        for (g.layers, 0..) |*layer, i| {
            layer.* = try weights.layer(&c, @intCast(i), src, device, a, 0, c.experts);
            g.ready += 1;
            try prepare.layer(k, device, queue, layer, &g.prepared);
        }
        g.head = try weights.head(&c, src, a);
        try prepare.head(k, device, queue, &g.head, &g.prepared);
        return g;
    }

    pub fn buffers(g: *const Graph, out: *std.ArrayList(mtl.Buffer), a: std.mem.Allocator) !void {
        try out.appendSlice(a, g.prepared.bufs.items);
        for (g.layers[0..g.ready]) |layer| if (layer.mlp == .moe) try out.append(a, layer.mlp.moe.table);
    }

    pub fn deinit(g: *Graph) void {
        g.prepared.deinit();
        for (g.layers[0..g.ready]) |layer| if (layer.mlp == .moe) layer.mlp.moe.table.deinit();
        g.a.free(g.layers);
    }
};

/// Cut-down tests may reduce layers, vocabulary and expert count, while retaining the served kernel geometry.
pub fn nativeShape(c: Config) !void {
    var supported: Config = .{};
    supported.layers = c.layers;
    supported.vocab = c.vocab;
    supported.experts = c.experts;
    supported.full_attn = c.full_attn;
    supported.dsa = c.dsa;
    if (!std.meta.eql(c, supported) or c.layers == 0 or c.layers > 93 or c.vocab < 256 or
        c.vocab > 163840 or c.vocab % 32 != 0 or c.experts < 16 or c.experts > 896 or c.experts % 8 != 0)
        return error.UnsupportedKimiNativeGeometry;
    for (c.layers..128) |i| if (c.full_attn.isSet(i)) return error.UnsupportedKimiNativeGeometry;
}

/// Required source bytes, prepared projections and expert tables; counts physical file padding separately.
pub fn weightBytes(a: std.mem.Allocator, c: Config) !struct { source: u64, prepared: u64 } {
    try nativeShape(c);
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const tmp = arena.allocator();
    var list: std.ArrayList(weights.Spec) = .empty;
    var source: u64 = 0;
    var prepared: u64 = 0;
    for (0..c.layers) |i| {
        list.clearRetainingCapacity();
        try weights.layerSpecs(&c, @intCast(i), tmp, &list);
        for (list.items) |s| {
            const n = bytes(s);
            source += n;
            if (s.rank == 2 and s.dtype == .bf16 and s.shape[1] % 64 == 0 and
                std.mem.indexOf(u8, s.name, "kv_b_proj") == null and s.shape[0] != 1) prepared += n;
        }
        if (c.isMoe(@intCast(i))) {
            list.clearRetainingCapacity();
            try weights.expertSpecs(&c, @intCast(i), 0, tmp, &list);
            for (list.items) |s| source += bytes(s) * c.experts;
            prepared += @as(u64, c.experts) * @sizeOf(@import("kernels.zig").ExpertPtrs);
        }
    }
    list.clearRetainingCapacity();
    try weights.headSpecs(&c, tmp, &list);
    for (list.items) |s| source += bytes(s);
    prepared += @as(u64, c.hidden) * c.vocab * 2;
    return .{ .source = source, .prepared = prepared };
}

fn bytes(s: weights.Spec) u64 {
    return @as(u64, s.shape[0]) * s.shape[1] * s.shape[2] * s.dtype.size();
}

test "Kimi resident admission prices every expert before loading the complete model" {
    const c: Config = .{};
    const n = try weightBytes(std.testing.allocator, c);
    const experts: u64 = 1446456066048;
    try std.testing.expect(n.source > experts);
    try std.testing.expect(n.prepared > 0);
    var cut = c;
    cut.layers = 4;
    cut.experts = 16;
    cut.vocab = 512;
    cut.full_attn = .empty;
    cut.full_attn.set(3);
    const small = try weightBytes(std.testing.allocator, cut);
    try std.testing.expect(small.source + small.prepared < 16 << 30);
    cut.hidden -= 1;
    try std.testing.expectError(error.UnsupportedKimiNativeGeometry, nativeShape(cut));
}
