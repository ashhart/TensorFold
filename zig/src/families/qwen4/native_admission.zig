//! Native geometry and a conservative whole-path budget are proved before a device allocation or kernel load.
const std = @import("std");
const cfg = @import("config.zig");
const qsa = @import("cuda_attention.zig");
pub const Backend = enum { metal, cuda };
pub fn check(c: *const cfg.Config, backend: Backend, p: *cfg.Problem) !qsa.Geometry {
    if (c.streams < 2 or c.streams > 8) return p.reject("hc_count");
    if (c.hidden == 0 or c.hidden % 256 != 0 or c.hidden > std.math.maxInt(i32) / (128 * c.streams)) return p.reject("hidden_size");
    if (c.key_dim != 128) return p.reject("linear_key_head_dim");
    if (c.value_dim != 128) return p.reject("linear_value_head_dim");
    if (c.conv != 4) return p.reject("linear_conv_kernel_dim");
    const g = try qsa.Geometry.of(c, p);
    if (backend == .metal and g.top > 512) return p.reject("indexer_budget");
    if (g.top > std.math.maxInt(i32) / (2 * 1024)) return p.reject("indexer_budget");
    if (c.ple_count > 0) {
        const tail = std.math.mul(usize, c.ple_conv - 1, c.ngram) catch return p.reject("ple_conv_kernel_size");
        if (tail > std.math.maxInt(i32) / c.wide()) return p.reject("ple_conv_kernel_size");
    }
    return g;
}
pub const Options = struct { capacity: usize, window: usize = 16, streams: usize = 1, budget: u64, config_digest: ?[32]u8 = null };
pub const Plan = struct {
    rows: usize,
    scratch: usize,
    state: usize,
    last_logits: usize,
    auxiliary: usize,
    weights: u64,
    device: u64,
    pub fn make(c: *const cfg.Config, backend: Backend, weights: u64, o: Options, p: *cfg.Problem) !Plan {
        const g = try check(c, backend, p);
        if (o.capacity == 0 or o.capacity > c.context or o.capacity > std.math.maxInt(i32)) return p.reject("--context");
        if (o.window == 0 or o.window > 16) return p.reject("--window");
        if (o.streams == 0 or o.streams > 128) return p.reject("--parallel");
        const rows = @min(@as(usize, 128), try std.math.mul(usize, o.window, o.streams));
        // Temporary tensors are bounded for one layer; finishLayer reuses them only after its residual write.
        var columns = try std.math.mul(usize, c.wide(), 64);
        for ([_]usize{ c.intermediate, c.lowrank, c.ple_dim, c.valueWidth(), c.convWidth(), g.projectionWidth() }) |n| columns = try std.math.add(usize, columns, try std.math.mul(usize, n, 8));
        columns = try std.math.add(usize, columns, c.vocab);
        var scratch = try std.math.mul(usize, try std.math.mul(usize, rows, columns), 2);
        if (c.ple_count > 0) {
            const tail = try std.math.mul(usize, c.ple_conv - 1, c.ngram);
            const count = try std.math.mul(usize, try std.math.mul(usize, @min(o.streams, rows), tail), c.wide());
            scratch = try std.math.add(usize, scratch, try std.math.mul(usize, count, 2));
        }
        const call: qsa.Call = .{ .rows = @min(o.window, o.capacity), .start = 0, .capacity = o.capacity, .eps = c.eps };
        if (backend == .metal) {
            const z = try @import("metal_attention.zig").sizes(g, call);
            for ([_]usize{ 6, 7, 10, 13, 14, 15, 16, 17 }) |i| scratch = try std.math.add(usize, scratch, try std.math.add(usize, z.bytes[i], 15));
        } else {
            const z = try qsa.sizes(g, call);
            for ([_]usize{ 1, 7, 10, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22 }) |i| scratch = try std.math.add(usize, scratch, try std.math.add(usize, z.bytes[i], 15));
        }
        const layout = try @import("state_layout.zig").Layout.of(c, o.capacity);
        const state = try layout.deviceBytes(o.window);
        const logits = try std.math.mul(usize, c.vocab, 2);
        const slots = try std.math.mul(u64, o.streams, try std.math.add(usize, state, logits));
        const indexed = for (c.kinds[0..c.layers]) |kind| {
            if (kind == .indexed_attention) break true;
        } else false;
        const auxiliary: usize = if (indexed) try std.math.mul(usize, 3 + c.rotary / 2, 4) else 0;
        const total = try std.math.add(u64, auxiliary, try std.math.add(u64, weights, try std.math.add(u64, scratch, slots)));
        if (total > o.budget) return error.Qwen4PathDoesNotFit;
        return .{ .rows = rows, .scratch = scratch, .state = state, .last_logits = logits, .auxiliary = auxiliary, .weights = weights, .device = total };
    }
};

test "Qwen4 native admission: named geometry refusals and the complete device path precede allocation" {
    const t = std.testing;
    const a = t.allocator;
    var p: cfg.Problem = .{};
    var c = try cfg.parse(a, @embedFile("fixtures/tiny-dense.json"), &p);
    defer c.deinit();
    try t.expectError(error.UnsupportedQwen4, check(&c, .cuda, &p));
    try t.expectEqualStrings("hidden_size", p.name());
    c.hidden = 256;
    c.key_dim = 128;
    c.value_dim = 128;
    c.head_dim = 256;
    c.index_dim = 128;
    c.rotary = 64;
    _ = try check(&c, .cuda, &p);
    _ = try check(&c, .metal, &p);
    const options: Options = .{ .capacity = 32, .streams = 3, .budget = std.math.maxInt(u64) };
    const plan = try Plan.make(&c, .metal, 12345, options, &p);
    try t.expectEqual(@as(usize, 48), plan.rows);
    try t.expect(plan.device > plan.weights + plan.scratch + plan.state);
    var short = options;
    short.budget = plan.device - 1;
    try t.expectError(error.Qwen4PathDoesNotFit, Plan.make(&c, .metal, 12345, short, &p));
    short.budget = plan.device;
    try t.expectEqual(plan, try Plan.make(&c, .metal, 12345, short, &p));
    short.window = 17;
    try t.expectError(error.UnsupportedQwen4, Plan.make(&c, .metal, 0, short, &p));
    try t.expectEqualStrings("--window", p.name());
    c.lowrank = 1 << 20;
    const large = try Plan.make(&c, .metal, 0, options, &p);
    try t.expect(large.scratch > large.rows * c.lowrank * 8);
}
