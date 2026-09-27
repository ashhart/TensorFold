//! Isolate scheduler positions using an identity MTP block and independent Python
//! speculate/settle fixtures. Real checkpoint arithmetic uses test-mtp-state.
const std = @import("std");
const mx = @import("mlx.zig");
const Store = @import("checkpoint.zig").Store;
const Identity = struct {
    pub const DraftCache = @import("model.zig").Cache;
    weights: Store,
    kernels: mx.Kernels,
    pub fn draftStepArray(_: *@This(), _: *mx.Scope, hidden: mx.Array, _: mx.Array, _: *DraftCache, _: bool) !mx.Array {
        return hidden;
    }
    pub fn draftHead(_: *@This(), _: *mx.Scope, hidden: mx.Array) !mx.Array {
        return hidden;
    }
    pub fn draftPrefix(_: *mx.Scope, cache: DraftCache, _: usize, _: usize) !DraftCache {
        return cache.clone();
    }
};
pub fn check(io: std.Io, dir: []const u8) !void {
    try mx.init();
    defer mx.shutdown();
    var fixtures = Store.init(64);
    defer fixtures.deinit();
    var path: [4096]u8 = undefined;
    try fixtures.loadFile(io, try std.fmt.bufPrint(&path, "{s}/arrays.safetensors", .{dir}), "", "");
    const bytes = try @import("weights.zig").readFile(io, try std.fmt.bufPrint(&path, "{s}/cases.json", .{dir}));
    defer mx.allocator.free(bytes);
    const Case = struct { key: []const u8, position: i32, seed: u64, temperature: f64, keep: usize, budget: usize, mapped: bool };
    const cases = try std.json.parseFromSlice([]const Case, mx.allocator, bytes, .{});
    defer cases.deinit();
    var m = Identity{ .weights = Store.init(64), .kernels = mx.Kernels.init() };
    defer m.weights.deinit();
    defer m.kernels.deinit();
    const Pipeline = @import("mtp_pipeline.zig").Pipeline(Identity);
    for (cases.value) |case| {
        var s = mx.Scope{};
        defer s.deinit();
        m.weights.deinit();
        m.weights = Store.init(64);
        if (case.mapped) try m.weights.put("draft_ids", try fixtures.field(case.key, "ids"));
        const hidden = try fixtures.field(case.key, "hidden");
        const settings = @import("sampling.zig").Sampling{ .metal = true, .seed = case.seed, .temperature = case.temperature, .top_k = 12, .top_p = 0.8 };
        var pipeline = try Pipeline.prepare(&m, &s, .{}, try s.slice(hidden, 0, 3, 4), 77, case.position, settings);
        defer pipeline.deinit();
        if (case.keep > 0) {
            var spec = try pipeline.speculate(&m, &s, hidden, try s.ints(&.{ 1, 2, 3, 4 }), case.position, settings);
            defer spec.deinit();
            try pipeline.settle(&s, spec, case.keep);
        }
        const out = try pipeline.propose(&m, &s, case.budget, case.position + @as(i32, @intCast(case.keep)), settings, true);
        try @import("sampling_checks.zig").equal(&s, out, try fixtures.field(case.key, "expected"));
    }
    std.debug.print("PASS: {d} MTP position and retained-row fixtures match original Python\n", .{cases.value.len});
}
