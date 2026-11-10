//! Explicit alternate layer weights; the primary MLX vocabulary, normalization and recurrent parameters stay owned.
const std = @import("std");
const core = @import("core");
const geometry = @import("cuda_geometry.zig");
const planning = @import("cuda_nvfp4_source_plan.zig");
const checks = @import("cuda_nvfp4_source_check.zig");
pub const Scope = planning.Scope;
const sameNorm = checks.sameNorm;

pub fn directory() !?[]const u8 {
    return @import("sys").environment.given("TF_QWEN_NVFP4_MLP_DIR");
}

pub fn scope() !Scope {
    return planning.selected(try @import("sys").environment.given("TF_QWEN_NVFP4_SCOPE"));
}

pub const Selection = struct { dir: []const u8, scope: Scope };

pub fn selection(dir: ?[]const u8, raw_scope: ?[]const u8) !?Selection {
    const chosen = try planning.selected(raw_scope);
    const path = dir orelse {
        if (chosen != .mlp) return error.Nvfp4ScopeNeedsSource;
        return null;
    };
    if (path.len == 0) return error.UnsupportedNvfp4MlpSource;
    return .{ .dir = path, .scope = chosen };
}

pub const Identity = struct {
    dir: []u8,
    scope: Scope,
    kinds: [32]u8,
    primary_hash: [32]u8 = @splat(0),
    source_hash: [32]u8 = @splat(0),

    pub fn words(id: Identity) [13]u64 {
        var out: [13]u64 = undefined;
        out[0] = @as(u64, @backingInt(id.scope)) + 1;
        for ([_][32]u8{ id.primary_hash, id.source_hash, id.kinds }, 0..) |hash, part|
            for (0..4) |i| {
                out[1 + part * 4 + i] = std.mem.readInt(u64, hash[i * 8 ..][0..8], .little);
            };
        return out;
    }

    pub fn deinit(id: *Identity, a: std.mem.Allocator) void {
        a.free(id.dir);
        id.* = undefined;
    }
};

pub const Opened = struct {
    checkpoint: core.Checkpoint,
    plan: planning.Plan,
    dir: []const u8,
    primary_hash: [32]u8 = @splat(0),
    source_hash: [32]u8 = @splat(0),

    pub fn deinit(o: *Opened) void {
        core.checkpoint.cache.close(&o.checkpoint);
    }

    pub fn identity(o: *const Opened, a: std.mem.Allocator, g: geometry.Geometry) !Identity {
        return .{ .dir = try a.dupe(u8, o.dir), .scope = o.plan.scope, .kinds = o.plan.digest(g), .primary_hash = o.primary_hash, .source_hash = o.source_hash };
    }
};

pub fn validateTarget(g: geometry.Geometry) !void {
    if (g.kinds.modelopt or g.exl3 or g.moe() or g.pocket > 0 or !geometry.dense27(g) or g.bits.mixed(g.layers))
        return error.UnsupportedNvfp4MlpSource;
}

pub fn open(a: std.mem.Allocator, io: std.Io, g: geometry.Geometry, original: *core.Checkpoint, primary_dir: []const u8) !?Opened {
    const chosen = (try selection(try directory(), try @import("sys").environment.given("TF_QWEN_NVFP4_SCOPE"))) orelse return null;
    return try openFrom(a, io, g, original, chosen.dir, chosen.scope, primary_dir);
}

pub fn openFrom(a: std.mem.Allocator, io: std.Io, g: geometry.Geometry, original: *core.Checkpoint, dir: []const u8, selected: Scope, primary_dir: []const u8) !Opened {
    try validateTarget(g);
    const other = try geometry.Geometry.read(a, io, dir);
    const plan = try planning.create(g, other, selected);
    var ck = try core.Checkpoint.openModel(a, io, dir);
    errdefer core.checkpoint.cache.close(&ck);
    try checks.validate(original, &ck, g, plan);
    const stamps = @import("cuda_cost_identity.zig");
    const primary_hash = try stamps.checkpoint(a, io, primary_dir);
    const source_hash = try stamps.checkpoint(a, io, dir);
    std.log.info("alternate Qwen source: scope={s}, {d} NVFP4 + {d} BF16 projections, {d} weight bytes; primary vocabulary and vectors retained; new checkpoint profile", .{ @tagName(selected), plan.nvfp4, plan.bf16, plan.weight_bytes });
    return .{ .checkpoint = ck, .plan = plan, .dir = dir, .primary_hash = primary_hash, .source_hash = source_hash };
}

test "mixed NVFP4 source recognizes MLX normalization sanitization and rejects another norm" {
    try std.testing.expect(sameNorm(&.{ 0x80, 0x3f }, &.{ 0, 0 }));
    try std.testing.expect(!sameNorm(&.{ 0, 0x40 }, &.{ 0, 0 }));
    try std.testing.expect(!sameNorm(&.{0}, &.{0}));
}

test "mixed NVFP4 source refuses unsupported target formats before loading kernels" {
    const valid: geometry.Geometry = .{
        .hidden = 5120,
        .intermediate = 17408,
        .layers = 64,
        .vocab = 248320,
        .k_heads = 16,
        .v_heads = 48,
        .q_heads = 24,
        .kv_heads = 4,
        .eps = 1e-6,
        .tied = false,
        .context = 262144,
        .linear = @splat(false),
    };
    try validateTarget(valid);
    var bad = valid;
    bad.exl3 = true;
    try std.testing.expectError(error.UnsupportedNvfp4MlpSource, validateTarget(bad));
    bad = valid;
    bad.kinds.modelopt = true;
    try std.testing.expectError(error.UnsupportedNvfp4MlpSource, validateTarget(bad));
    bad = valid;
    bad.bits.head = 8;
    try std.testing.expectError(error.UnsupportedNvfp4MlpSource, validateTarget(bad));
}

test {
    _ = @import("cuda_nvfp4_source_test.zig");
}
