//! Fail every Zig allocation at ownership boundaries, then check MLX error recovery.
const std = @import("std");
const mx = @import("mlx.zig");
const cp = @import("checkpoint.zig");
const sources = @import("kernel_sources.zig");
const Kind = enum { scope, store, dense_weights, indexed, unindexed, draft_vocab, mtp_pipeline, serial_pipeline, kernels };

// A tiny real-MLX model isolates scheduler ownership from full checkpoint loading.
// Numerical model equivalence is checked separately by test-mtp-state.
const PipelineFixture = struct {
    pub const DraftCache = @import("model.zig").Cache;
    weights: cp.Store,
    kernels: mx.Kernels,
    pub fn draftStepArray(m: *@This(), s: *mx.Scope, hidden: mx.Array, tokens: mx.Array, cache: *DraftCache, queued: bool) !mx.Array {
        const out = try s.binary(mx.c.mlx_add, hidden, try m.weights.embedArray(s, "projection", tokens));
        const keys = if (cache.a.ctx != null) try s.cat(&.{ cache.a, out }, 0) else out;
        if (!queued) try mx.eval(keys);
        try mx.replace(&cache.a, keys);
        try mx.replace(&cache.b, keys);
        return out;
    }
    pub fn draftHead(m: *@This(), s: *mx.Scope, hidden: mx.Array) !mx.Array {
        return m.weights.linear(&m.kernels, s, "projection", hidden, true);
    }
    pub fn draftPrefix(s: *mx.Scope, cache: DraftCache, rows: usize, keep: usize) !DraftCache {
        const end = mx.dim(cache.a, 0) - @as(i32, @intCast(rows - keep));
        return (DraftCache{ .a = try s.slice(cache.a, 0, 0, end), .b = try s.slice(cache.b, 0, 0, end) }).clone();
    }
};

fn exercise(a: std.mem.Allocator, kind: Kind, io: std.Io, dir: []const u8) !void {
    const previous = mx.allocator;
    mx.allocator = a;
    defer mx.allocator = previous;
    var s = mx.Scope{};
    defer s.deinit();
    var kernels = mx.Kernels.init();
    defer kernels.deinit();
    switch (kind) {
        .serial_pipeline => {
            try @import("serial_pipeline_checks.zig").exercise(a, 4, 17);
            // Continue beyond the output list's first capacity as well as EOS.
            try @import("serial_pipeline_checks.zig").exercise(a, 8, 65);
        },
        .scope => {
            for (0..40) |_| {
                const x = try s.zeros(&.{ 4, 64 }, mx.bf16);
                try mx.eval(try s.cast(try s.reshape(x, &.{256}), mx.f32t));
            }
        },
        .store => {
            var store = cp.Store.init(64);
            defer store.deinit();
            const x = try s.zeros(&.{4}, mx.f32t);
            var name: [32]u8 = undefined;
            for (0..40) |i| try store.put(try std.fmt.bufPrint(&name, "weight{d}", .{i}), x);
            try store.put("weight0", try s.zeros(&.{8}, mx.f32t));
            try std.testing.expectEqual(@as(i32, 8), mx.dim(try store.get("weight0"), 0));
        },
        .dense_weights => try @import("weights.zig").Weights.checkOwnedInsertions(),
        .mtp_pipeline => {
            var m = PipelineFixture{ .weights = cp.Store.init(64), .kernels = mx.Kernels.init() };
            defer m.weights.deinit();
            defer m.kernels.deinit();
            var path: [4096]u8 = undefined;
            try m.weights.load(io, try std.fmt.bufPrint(&path, "{s}/indexed", .{dir}), "");
            const settings = @import("sampling.zig").Sampling{ .metal = true, .seed = 1234, .temperature = 0.7 };
            const P = @import("mtp_pipeline.zig").Pipeline(PipelineFixture);
            var pipeline = try P.prepare(&m, &s, .{}, try s.zeros(&.{ 1, 64 }, mx.bf16), 7, 0, settings);
            defer pipeline.deinit();
            try std.testing.expectError(error.InvalidDraftBudget, pipeline.propose(&m, &s, 0, 0, settings, true));
            try std.testing.expectError(error.InvalidDraftBudget, pipeline.propose(&m, &s, 16, 0, settings, true));
            try mx.eval(try pipeline.propose(&m, &s, 15, 0, settings, true));
            var spec = try pipeline.speculate(&m, &s, try s.zeros(&.{ 3, 64 }, mx.bf16), try s.ints(&.{ 1, 2, 3 }), 0, settings);
            defer spec.deinit();
            try mx.eval(spec.firsts);
            try std.testing.expectError(error.InvalidCommit, pipeline.settle(&s, spec, 0));
            try std.testing.expectError(error.InvalidCommit, pipeline.settle(&s, spec, 4));
            try pipeline.settle(&s, spec, 2);
            try mx.eval(try pipeline.propose(&m, &s, 3, 2, settings, false));
        },
        .draft_vocab => {
            var store = cp.Store.init(64);
            defer store.deinit();
            var path: [4096]u8 = undefined;
            try store.load(io, try std.fmt.bufPrint(&path, "{s}/indexed", .{dir}), "");
            inline for (.{ "weight", "scales", "biases" }) |suffix|
                try store.put("lm_head." ++ suffix, try store.get("projection." ++ suffix));
            try @import("draft_vocab.zig").install(&store, "1 3 7 9 15 17 23 31", 32, 8);
            try mx.eval(try store.linear(&kernels, &s, "draft_lm_head", try s.zeros(&.{ 1, 64 }, mx.bf16), true));
        },
        .indexed, .unindexed => {
            var store = cp.Store.init(64);
            defer store.deinit();
            var path: [4096]u8 = undefined;
            try store.load(io, try std.fmt.bufPrint(&path, "{s}/{s}", .{ dir, @tagName(kind) }), "");
            const x = try s.zeros(&.{ 1, 64 }, mx.bf16);
            try mx.eval(try store.linear(&kernels, &s, "projection", x, true));
            // Cache hit and replacement ownership are distinct from first insertion.
            try mx.eval(try store.linear(&kernels, &s, "projection", x, true));
        },
        .kernels => {
            const x = try s.zeros(&.{ 1, 32 }, mx.bf16);
            for (0..2) |_| {
                const out = try kernels.run(&s, sources.q4_swiglu, &.{ x, x }, &.{
                    mx.ti("N", 32), mx.ti("GSTRIDE", 32), mx.ti("USTRIDE", 32), mx.ti("GO", 0), mx.ti("UO", 0),
                }, .{ 32, 1, 1 }, .{ 32, 1, 1 }, &.{.{ .shape = &.{ 1, 32 } }});
                try mx.eval(out[0]);
            }
        },
    }
}

fn active() !usize {
    try mx.check(mx.c.mlx_synchronize(mx.stream));
    var result: usize = 0;
    try mx.check(mx.c.mlx_get_active_memory(&result));
    return result;
}

fn errors() !void {
    var s = mx.Scope{};
    defer s.deinit();
    try std.testing.expectError(error.MlxFailure, s.own(mx.empty));
    const x = try s.zeros(&.{ 2, 3 }, mx.f32t);
    try std.testing.expectError(error.MlxFailure, s.reshape(x, &.{5}));
    try std.testing.expectEqual(@as(usize, 1), s.arrays.items.len);
    try mx.eval(try s.reshape(x, &.{6}));
    var kernels = mx.Kernels.init();
    defer kernels.deinit();
    try std.testing.expectError(error.InvalidKernelArity, kernels.run(&s, sources.q4_swiglu, &.{x}, &.{}, .{ 1, 1, 1 }, .{ 1, 1, 1 }, &.{.{ .shape = &.{6} }}));
    try std.testing.expectEqual(@as(usize, 0), kernels.items.count());
}

pub fn check(io: std.Io, dir: []const u8) !void {
    try mx.init();
    defer mx.shutdown();
    const tensor = mx.tensor_units;
    defer mx.tensor_units = tensor;
    var failure_points: usize = 0;
    for (0..if (tensor) @as(usize, 2) else 1) |backend| {
        mx.tensor_units = tensor and backend == 0;
        for ([_]bool{ true, false }) |allow_resize| {
            inline for (std.meta.tags(Kind)) |kind| {
                // Force the allocate/copy fallback too: libc can otherwise grow a
                // Scope in place and hide failures after its first allocation.
                var no_resize = std.testing.FailingAllocator.init(std.heap.c_allocator, .{ .resize_fail_index = 0 });
                const backing = if (allow_resize) std.heap.c_allocator else no_resize.allocator();
                var probe = std.testing.FailingAllocator.init(backing, .{});
                try exercise(probe.allocator(), kind, io, dir);
                try std.testing.expectEqual(probe.allocated_bytes, probe.freed_bytes);
                failure_points += probe.alloc_index;
                const baseline = try active();
                try std.testing.checkAllAllocationFailures(backing, exercise, .{ kind, io, dir });
                try std.testing.expectEqual(no_resize.allocated_bytes, no_resize.freed_bytes);
                const after = try active();
                if (after != baseline) {
                    std.debug.print("MLX memory changed after {s}: {d} -> {d}\n", .{ @tagName(kind), baseline, after });
                    return error.MlxMemoryLeak;
                }
                std.debug.print("PASS: all {d} host allocation failures in {s}, {s}, resize={}; active bytes {d}\n", .{ probe.alloc_index, @tagName(kind), if (mx.tensor_units) "tensor" else "SIMD", allow_resize, after });
            }
        }
    }
    try errors();
    const baseline = try active();
    for (0..16) |_| try errors();
    try std.testing.expectEqual(baseline, try active());
    std.debug.print("PASS: {d} host allocation failure points; MLX API errors release outputs and permit subsequent operations\n", .{failure_points});
}
