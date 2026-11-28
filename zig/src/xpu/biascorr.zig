//! Fail-closed F32 BiasCorr sidecar loader, tracker, and verifier for XPU W4K models.

const std = @import("std");
const load = @import("load.zig");
const dev = @import("dev.zig");

const weight_suffixes = [_][]const u8{
    ".mlp.gate_proj.weight",
    ".mlp.up_proj.weight",
    ".mlp.down_proj.weight",
    ".self_attn.q_proj.weight",
    ".self_attn.k_proj.weight",
    ".self_attn.v_proj.weight",
    ".self_attn.o_proj.weight",
    ".linear_attn.in_proj_qkv.weight",
    ".linear_attn.in_proj_z.weight",
    ".linear_attn.in_proj_a.weight",
    ".linear_attn.in_proj_b.weight",
    ".linear_attn.out_proj.weight",
};

pub fn envEnabled() bool {
    const v = std.c.getenv("XPU_BIASCORR_FILE") orelse return false;
    return std.mem.span(v).len != 0;
}

pub fn supportsFamily(model_type: []const u8) bool {
    return std.mem.eql(u8, model_type, "qwen3_5");
}

fn canonical(name: []const u8) []const u8 {
    if (std.mem.startsWith(u8, name, "language_model.")) return name["language_model.".len..];
    if (std.mem.startsWith(u8, name, "model.language_model.")) return name["model.language_model.".len..];
    return name;
}

fn projectionWeight(name: []const u8) bool {
    const c = canonical(name);
    if (std.mem.indexOf(u8, c, ".layers.") == null) return false;
    for (weight_suffixes) |suffix| if (std.mem.endsWith(u8, c, suffix)) return true;
    return false;
}

fn expectedCount(c: *const dev.Checkpoint, before: usize) usize {
    var n: usize = 0;
    for (c.tensors.items[0..before]) |t| {
        if (projectionWeight(t.name)) n += 1;
    }
    return n;
}

fn matchingWeightN(c: *const dev.Checkpoint, before: usize, bias_base: []const u8) ?usize {
    for (c.tensors.items[0..before]) |t| {
        if (!projectionWeight(t.name)) continue;
        const cn = canonical(t.name);
        if (cn.len < ".weight".len) continue;
        if (std.mem.eql(u8, cn[0 .. cn.len - ".weight".len], bias_base)) return @intCast(t.shape[0]);
    }
    return null;
}

fn validateVector(t: dev.Tensor, full_n: usize) !void {
    if (!std.mem.eql(u8, t.dtype, "F32") or t.rank != 1 or t.shape[0] != full_n or t.bytes.len != full_n * 4) return error.BadBiasCorrVector;
    for (0..full_n) |i| {
        const v: f32 = @bitCast(std.mem.readInt(u32, t.bytes[4 * i ..][0..4], .little));
        if (!std.math.isFinite(v)) return error.NonFiniteBiasCorr;
    }
}

fn allowPartial() bool {
    const v = std.c.getenv("XPU_BIASCORR_ALLOW_PARTIAL") orelse return false;
    return std.mem.eql(u8, std.mem.span(v), "1");
}

fn validateCoverage(s: *const BiasSidecar, c: *const dev.Checkpoint) !void {
    const expected = expectedCount(c, s.start);
    var recognized: usize = 0;
    for (c.tensors.items[s.start..]) |t| {
        const cn = canonical(t.name);
        if (!std.mem.endsWith(u8, cn, ".bias_corr")) return error.BadBiasCorrSidecar;
        const base = cn[0 .. cn.len - ".bias_corr".len];
        const full_n = matchingWeightN(c, s.start, base) orelse {
            std.debug.print("unused biascorr tensor in sidecar: {s}\n", .{t.name});
            return error.UnusedBiasCorrTensor;
        };
        validateVector(t, full_n) catch |err| {
            std.debug.print("invalid biascorr vector: {s}: {s}\n", .{ t.name, @errorName(err) });
            return err;
        };
        recognized += 1;
    }
    if (recognized < expected and !allowPartial()) {
        std.debug.print("incomplete biascorr coverage: matched sidecar keys {d}, expected decoder projections {d}\n", .{ recognized, expected });
        return error.IncompleteBiasCorrCoverage;
    }
}

pub const BiasSidecar = struct {
    start: usize = 0,
    count: usize = 0,
    sha: [32]u8 = @splat(0),
    path_disp: []const u8 = "",
    used: []bool = &.{},
    tracker_state: load.BiasTracker = .{ .start = 0, .used = &.{} },

    pub fn openFromEnv(a: std.mem.Allocator, ckpt: *dev.Checkpoint) !BiasSidecar {
        var s: BiasSidecar = .{};
        const v = std.c.getenv("XPU_BIASCORR_FILE") orelse return s;
        const bspan = std.mem.span(v);
        if (bspan.len == 0) return s;
        s.path_disp = bspan;
        s.start = ckpt.tensors.items.len;
        const bpath = try std.fmt.allocPrintSentinel(a, "{s}", .{bspan}, 0);
        defer a.free(bpath);
        try ckpt.addShard(bpath);
        s.count = ckpt.tensors.items.len - s.start;
        if (s.count == 0) return error.EmptyBiasCorrSidecar;
        for (ckpt.tensors.items[s.start..]) |t| {
            if (!std.mem.endsWith(u8, t.name, ".bias_corr") or !std.mem.eql(u8, t.dtype, "F32")) {
                std.debug.print("invalid biascorr tensor: {s} ({s})\n", .{ t.name, t.dtype });
                return error.BadBiasCorrSidecar;
            }
        }
        try validateCoverage(&s, ckpt);
        const raw_map = ckpt.maps.items[ckpt.maps.items.len - 1];
        std.crypto.hash.sha2.Sha256.hash(raw_map, &s.sha, .{});
        s.used = try a.alloc(bool, s.count);
        @memset(s.used, false);
        s.tracker_state = .{ .start = s.start, .used = s.used };
        return s;
    }

    pub fn tracker(s: *BiasSidecar) ?*load.BiasTracker {
        return if (s.count > 0) &s.tracker_state else null;
    }

    pub fn finishAndLog(s: *const BiasSidecar, ckpt: *const dev.Checkpoint) !void {
        if (s.count == 0) return;
        var matched: usize = 0;
        for (s.used, 0..) |u, bi| {
            if (u) {
                matched += 1;
            } else {
                std.debug.print("unused biascorr tensor in sidecar: {s}\n", .{ckpt.tensors.items[s.start + bi].name});
                return error.UnusedBiasCorrTensor;
            }
        }
        const expected = expectedCount(ckpt, s.start);
        if (matched < expected and !allowPartial()) return error.IncompleteBiasCorrCoverage;
        std.debug.print("  biascorr sidecar loaded: {s} (matched={d}/{d}, expected={d}, sha256={s})\n", .{ s.path_disp, matched, s.count, expected, &std.fmt.bytesToHex(s.sha, .lower) });
    }

    pub fn deinit(s: *BiasSidecar, a: std.mem.Allocator) void {
        if (s.used.len > 0) a.free(s.used);
        s.used = &.{};
    }
};

test "biascorr projection names are canonical across checkpoint prefixes" {
    try std.testing.expect(projectionWeight("language_model.model.layers.0.mlp.gate_proj.weight"));
    try std.testing.expect(projectionWeight("model.language_model.model.layers.0.self_attn.o_proj.weight"));
    try std.testing.expect(projectionWeight("model.layers.0.linear_attn.in_proj_qkv.weight"));
    try std.testing.expect(!projectionWeight("language_model.lm_head.weight"));
    try std.testing.expect(!projectionWeight("model.layers.0.linear_attn.conv1d.weight"));
    try std.testing.expect(supportsFamily("qwen3_5"));
    try std.testing.expect(!supportsFamily("gemma4"));
}

test "biascorr expected count covers the 496 decoder projections" {
    var ckpt: dev.Checkpoint = .{};
    defer ckpt.tensors.deinit(std.testing.allocator);
    defer ckpt.maps.deinit(std.testing.allocator);
    defer ckpt.headers.deinit(std.testing.allocator);
    const suffixes = [_][]const u8{
        "linear_attn.in_proj_qkv.weight", "linear_attn.in_proj_z.weight", "linear_attn.in_proj_a.weight", "linear_attn.in_proj_b.weight",
        "linear_attn.out_proj.weight",    "mlp.gate_proj.weight",         "mlp.up_proj.weight",           "mlp.down_proj.weight",
    };
    const attn_suffixes = [_][]const u8{
        "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight", "self_attn.o_proj.weight",
        "mlp.gate_proj.weight",    "mlp.up_proj.weight",      "mlp.down_proj.weight",
    };
    for (0..64) |layer| {
        const attn = (layer + 1) % 4 == 0;
        for (if (attn) attn_suffixes[0..] else suffixes[0..]) |suffix| {
            const name = try std.fmt.allocPrint(std.testing.allocator, "language_model.model.layers.{d}.{s}", .{ layer, suffix });
            defer std.testing.allocator.free(name);
            const copy = try std.testing.allocator.dupe(u8, name);
            try ckpt.tensors.append(std.testing.allocator, .{ .name = copy, .dtype = "I32", .shape = @splat(1), .rank = 2, .bytes = &.{} });
        }
    }
    try std.testing.expectEqual(@as(usize, 496), expectedCount(&ckpt, ckpt.tensors.items.len));
    for (ckpt.tensors.items) |t| std.testing.allocator.free(t.name);
}
