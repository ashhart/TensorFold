//! Alternate-source budgets, source selection, parameter mismatches and failure ownership without CUDA.
const std = @import("std");
const plan = @import("cuda_nvfp4_source_plan.zig");
const geometry = @import("cuda_geometry.zig");
const weights = @import("cuda_weights.zig");
const source = @import("cuda_nvfp4_mlp.zig");
const checks = @import("cuda_nvfp4_source_check.zig");
const core = @import("core");
const t = std.testing;

fn target() geometry.Geometry {
    var g: geometry.Geometry = .{ .hidden = 5120, .intermediate = 17408, .layers = 64, .vocab = 248320, .k_heads = 16, .v_heads = 48, .q_heads = 24, .kv_heads = 4, .eps = 1e-6, .tied = false, .context = 131072, .linear = @splat(false) };
    for (0..64) |i| g.linear[i] = i % 4 != 3;
    return g;
}

fn alternate(g: geometry.Geometry) geometry.Geometry {
    var other = g;
    other.kinds = .{ .modelopt = true, .ct = true, .head = .bf16 };
    for (0..64) |i| {
        for (weights.slots(g, i)) |slot| other.kinds.proj[i][@backingInt(slot)] = .nvfp4;
        if (g.linear[i]) {
            other.kinds.proj[i][@backingInt(geometry.Slot.a)] = .bf16;
            other.kinds.proj[i][@backingInt(geometry.Slot.b)] = .bf16;
        }
    }
    for ([_]usize{ 3, 7, 11, 15, 19, 23 }) |i| for ([_]geometry.Slot{ .q, .k, .v }) |slot| {
        other.kinds.proj[i][@backingInt(slot)] = .bf16;
    };
    for ([_]usize{ 3, 7 }) |i| other.kinds.proj[i][@backingInt(geometry.Slot.o)] = .bf16;
    other.kinds.proj[0][@backingInt(geometry.Slot.out)] = .bf16;
    return other;
}

test "NVFP4 source: default remains MLP and full layer budget retains both primary endpoints" {
    const g = target();
    const other = alternate(g);
    try t.expectEqual(plan.Scope.mlp, try plan.selected(null));
    try t.expectEqual(plan.Scope.mlp, try plan.selected("mlp"));
    try t.expectEqual(plan.Scope.layers, try plan.selected("layers"));
    for ([_][]const u8{ "", "all", "1", "MLP", "layers " }) |raw|
        try t.expectError(error.InvalidNvfp4SourceScope, plan.selected(raw));
    try t.expect((try source.selection(null, null)) == null);
    try t.expectError(error.Nvfp4ScopeNeedsSource, source.selection(null, "layers"));
    try t.expectError(error.UnsupportedNvfp4MlpSource, source.selection("", "mlp"));
    try t.expectEqual(plan.Scope.mlp, (try source.selection("weights", null)).?.scope);
    const mlp = try plan.create(g, other, .mlp);
    const all = try plan.create(g, other, .layers);
    try t.expectEqual(@as(usize, 192), mlp.projections);
    try t.expectEqual(@as(usize, 15154935808), mlp.weight_bytes);
    try t.expectEqual(weights.total(g), mlp.weight_bytes);
    try t.expectEqual(@as(usize, 496), all.projections);
    try t.expectEqual(@as(usize, 379), all.nvfp4);
    try t.expectEqual(@as(usize, 117), all.bf16);
    try t.expectEqual(@as(usize, 16014112768), all.weight_bytes);
    try t.expectEqual(@as(usize, 859176960), all.weight_bytes - mlp.weight_bytes);
    try t.expectEqual(@as(usize, 715161600), all.stage_bytes);
    try t.expectEqual(weights.stageBytes(g), all.stage_bytes);
    try t.expect(mlp.kind(0, .a) == null);
    try t.expectEqual(geometry.Kind.bf16, all.kind(0, .a).?);
    try t.expect(!std.mem.eql(u8, &mlp.digest(g), &all.digest(g)));
}

test "NVFP4 source: mismatched geometry and unknown selected weight formats fail before any upload" {
    const g = target();
    var other = alternate(g);
    inline for (.{ "hidden", "intermediate", "layers", "vocab", "k_heads", "v_heads", "q_heads", "kv_heads" }) |field| {
        const saved = @field(other, field);
        @field(other, field) = saved + 1;
        try t.expectError(if (std.mem.eql(u8, field, "hidden") or std.mem.eql(u8, field, "layers"))
            error.UnsupportedNvfp4MlpSource
        else
            error.Nvfp4MlpBaseMismatch, plan.create(g, other, .layers));
        @field(other, field) = saved;
    }
    other.linear[0] = false;
    try t.expectError(error.Nvfp4MlpBaseMismatch, plan.create(g, other, .layers));
    other = alternate(g);
    other.context = g.context - 1;
    try t.expectError(error.Nvfp4SourceContext, plan.create(g, other, .layers));
    other = alternate(g);
    other.kinds.proj[0][@backingInt(geometry.Slot.qkv)] = .fp8c;
    try t.expectError(error.UnsupportedNvfp4MlpSource, plan.create(g, other, .layers));
    _ = try plan.create(g, other, .mlp);
    var huge = g;
    huge.intermediate = std.math.maxInt(usize) & ~@as(usize, 63);
    other = alternate(g);
    other.intermediate = huge.intermediate;
    try t.expectError(error.Overflow, plan.create(huge, other, .layers));
}

test "NVFP4 source: all selected schemas and effective vectors are checked without reading source endpoints" {
    const Fixture = @import("cuda_nvfp4_source_fixture.zig").Fixture;
    const g = target();
    const other = alternate(g);
    for ([_]plan.Scope{ .mlp, .layers }) |scope| {
        var primary: Fixture = .{ .g = g, .source = false };
        var alternate_: Fixture = .{ .g = other, .source = true };
        try checks.validate(&primary, &alternate_, g, try plan.create(g, other, scope));
        try t.expectEqual(@as(usize, if (scope == .mlp) 192 else 496), alternate_.matrices);
        try t.expectEqual(@as(usize, if (scope == .mlp) 129 else 353), primary.reads);
        try t.expectEqual(@as(usize, 0), alternate_.endpoints);
        try t.expectEqual(@as(usize, 0), primary.endpoints);
    }
}

test "NVFP4 source: a changed effective norm or recurrent parameter refuses the all-layer graft" {
    const fixtures = @import("cuda_nvfp4_source_fixture.zig");
    const g = target();
    const other = alternate(g);
    const p = try plan.create(g, other, .layers);
    for ([_]fixtures.Bad{ .input_norm, .post_norm, .final_norm, .q_norm, .k_norm, .gdn_norm, .conv, .a_log, .dt_bias }) |bad| {
        var primary: fixtures.Fixture = .{ .g = g, .source = false };
        var alternate_: fixtures.Fixture = .{ .g = other, .source = true };
        if (bad == .gdn_norm or bad == .conv or bad == .a_log or bad == .dt_bias) alternate_.bad = bad else primary.bad = bad;
        try t.expectError(error.Nvfp4MlpBaseMismatch, checks.validate(&primary, &alternate_, g, p));
    }
}

test "NVFP4 source: missing and malformed selected weights or calibration fail the host preflight" {
    const fixtures = @import("cuda_nvfp4_source_fixture.zig");
    const g = target();
    const other = alternate(g);
    const p = try plan.create(g, other, .layers);
    for ([_]fixtures.Bad{ .weight_shape, .weight_type, .missing }) |bad| {
        var primary: fixtures.Fixture = .{ .g = g, .source = false };
        var alternate_: fixtures.Fixture = .{ .g = other, .source = true, .bad = bad };
        try t.expectError(if (bad == .missing) error.MissingTensor else error.UnexpectedTensor, checks.validate(&primary, &alternate_, g, p));
    }
    for ([_]f32{ 0, -1, std.math.inf(f32), std.math.nan(f32), 0x1p-149, @bitCast(@as(u32, 0x7f7fffff)) }) |bad| {
        var primary: fixtures.Fixture = .{ .g = g, .source = false };
        var alternate_: fixtures.Fixture = .{ .g = other, .source = true, .scale = bad };
        try t.expectError(error.BadNativeNvfp4Scale, checks.validate(&primary, &alternate_, g, p));
    }
}

test "NVFP4 source: identity owns its path and separates source kinds from scope" {
    const g = target();
    var opened: source.Opened = .{ .checkpoint = undefined, .plan = try plan.create(g, alternate(g), .layers), .dir = "source-directory" };
    var id = try opened.identity(t.allocator, g);
    defer id.deinit(t.allocator);
    try t.expectEqualStrings(opened.dir, id.dir);
    try t.expect(id.dir.ptr != opened.dir.ptr);
    var failed = t.FailingAllocator.init(t.allocator, .{ .fail_index = 0 });
    try t.expectError(error.OutOfMemory, opened.identity(failed.allocator(), g));
    try t.expectEqual(failed.allocated_bytes, failed.freed_bytes);
    const before = opened.plan.digest(g);
    opened.plan.kinds.proj[0][@backingInt(geometry.Slot.qkv)] = .bf16;
    try t.expect(!std.mem.eql(u8, &before, &opened.plan.digest(g)));
}

test "NVFP4 source: planned bytes equal the real aligned arena allocator watermark" {
    const E = @import("cuda_engine.zig");
    const g = target();
    for ([_]plan.Scope{ .mlp, .layers }) |scope| {
        const p = try plan.create(g, alternate(g), scope);
        var e: E.Engine = undefined;
        e.arena_at = 0;
        e.arena = .{ .d = undefined, .ptr = 0x10000, .len = p.weight_bytes };
        for (0..2) |_| {
            _ = try e.alloc(g.vocab * g.hidden / 2);
            _ = try e.alloc(g.vocab * g.hidden / 64 * 2);
            _ = try e.alloc(g.vocab * g.hidden / 64 * 2);
        }
        _ = try e.alloc(g.hidden * 2);
        for (0..64) |i| {
            for (weights.slots(g, i)) |slot| {
                const nk = weights.shape(g, slot);
                const rows = (nk[0] + 127) / 128 * 128;
                const kind = p.kind(i, slot);
                if (kind != null and kind.? == .bf16) {
                    _ = try e.alloc(rows * nk[1] * 2);
                } else if (kind == null) {
                    _ = try e.alloc(rows * nk[1] / 2);
                    _ = try e.alloc(rows * nk[1] / 32);
                    _ = try e.alloc(rows * nk[1] / 32);
                } else {
                    _ = try e.alloc(rows * nk[1] / 2);
                    _ = try e.alloc(rows * nk[1] / 16);
                }
            }
            _ = try e.alloc(g.hidden * 2);
            _ = try e.alloc(g.hidden * 2);
            if (g.linear[i]) {
                _ = try e.alloc(10240 * 4 * 2);
                _ = try e.alloc(128 * 2);
                _ = try e.alloc(48 * 4);
                _ = try e.alloc(48 * 2);
            } else {
                _ = try e.alloc(256 * 2);
                _ = try e.alloc(256 * 2);
            }
        }
        try t.expectEqual(p.weight_bytes, e.arena_at);
        try t.expectError(error.ArenaExhausted, e.alloc(1));
    }
}

test "NVFP4 source: persistent state identity separates both checkpoints scope and selected kinds" {
    const E = @import("cuda_engine.zig");
    const reuse = @import("cuda_reuse.zig");
    const disk = @import("cuda_disk.zig");
    var e: E.Engine = undefined;
    e.g = target();
    e.kv8 = false;
    e.n_taps = 0;
    e.int8.on = false;
    e.alternate = null;
    const target_: reuse.Target = .{ .e = &e, .caches = undefined, .seen = undefined };
    const plain = disk.facts(&target_);
    try t.expectEqualSlices(u64, &.{ 0, 0, 0, std.hash.Wyhash.hash(0x726f7065, "plain") }, &plain);
    var id: source.Identity = .{ .dir = try t.allocator.dupe(u8, "source"), .scope = .mlp, .kinds = @splat(0) };
    defer id.deinit(t.allocator);
    e.alternate = id;
    const first = disk.facts(&target_);
    try t.expect(!std.mem.eql(u64, &plain, &first));
    inline for (.{ "primary_hash", "source_hash", "kinds" }) |field| {
        @field(e.alternate.?, field)[0] = 1;
        const changed = disk.facts(&target_);
        try t.expect(!std.mem.eql(u64, &first, &changed));
        @field(e.alternate.?, field)[0] = 0;
    }
    e.alternate.?.scope = .layers;
    try t.expect(!std.mem.eql(u64, &first, &disk.facts(&target_)));
    e.alternate.?.scope = .mlp;
    e.g.rope = .{ .yarn = .{ .factor = 2, .trained = 65536, .table = @splat(1) } };
    const with_rope = disk.facts(&target_);
    try t.expectEqualSlices(u64, first[0..2], with_rope[0..2]);
    try t.expect(first[2] != with_rope[2]);
    try t.expect(first[3] != with_rope[3]);
    e.g.rope.yarn.table[1] = 2;
    const changed_rope = disk.facts(&target_);
    try t.expect(with_rope[2] != changed_rope[2]);
    try t.expect(with_rope[3] != changed_rope[3]);
}

test {
    _ = @import("cuda_nvfp4_source_io_test.zig");
}
