//! Exact alternate-source projection selection and arena budget, with primary vocabulary and vectors retained.
const std = @import("std");
const geometry = @import("cuda_geometry.zig");
const weights = @import("cuda_weights.zig");
const modelopt = @import("cuda_modelopt.zig");
pub const Scope = enum { mlp, layers };

pub fn selected(raw: ?[]const u8) !Scope {
    const text = raw orelse return .mlp;
    if (std.mem.eql(u8, text, "mlp")) return .mlp;
    if (std.mem.eql(u8, text, "layers")) return .layers;
    return error.InvalidNvfp4SourceScope;
}

pub fn takes(scope: Scope, slot: geometry.Slot) bool {
    return scope == .layers or slot == .gate or slot == .up or slot == .down;
}

pub const Plan = struct {
    scope: Scope,
    kinds: geometry.Kinds,
    weight_bytes: usize,
    stage_bytes: usize,
    projections: usize = 0,
    nvfp4: usize = 0,
    bf16: usize = 0,

    pub fn kind(p: *const Plan, layer: usize, slot: geometry.Slot) ?geometry.Kind {
        return if (takes(p.scope, slot)) p.kinds.of(layer, slot) else null;
    }

    pub fn digest(p: *const Plan, g: geometry.Geometry) [32]u8 {
        var hash = std.crypto.hash.sha2.Sha256.init(.{});
        hash.update(@tagName(p.scope));
        for (0..g.layers) |i| for (weights.slots(g, i)) |slot| if (p.kind(i, slot)) |kind_| {
            hash.update(&.{ @intCast(i), @backingInt(slot), @backingInt(kind_) });
        };
        return hash.finalResult();
    }
};

pub fn create(g: geometry.Geometry, source: geometry.Geometry, scope: Scope) !Plan {
    if (!source.kinds.modelopt or !geometry.dense27(source)) return error.UnsupportedNvfp4MlpSource;
    inline for (.{ "hidden", "intermediate", "layers", "vocab", "k_heads", "v_heads", "q_heads", "kv_heads", "eps", "tied" }) |field|
        if (@field(g, field) != @field(source, field)) return error.Nvfp4MlpBaseMismatch;
    if (!std.mem.eql(bool, g.linear[0..g.layers], source.linear[0..source.layers]) or !std.meta.eql(g.rope, source.rope))
        return error.Nvfp4MlpBaseMismatch;
    if (g.context > source.context) return error.Nvfp4SourceContext;
    try checkProducts(g);
    var plan: Plan = .{ .scope = scope, .kinds = source.kinds, .weight_bytes = weights.total(g), .stage_bytes = weights.stageBytes(g) };
    for (0..g.layers) |i| for (weights.slots(g, i)) |slot| if (plan.kind(i, slot)) |kind_| {
        if (kind_ != .nvfp4 and !(scope == .layers and kind_ == .bf16)) return error.UnsupportedNvfp4MlpSource;
        const nk = weights.shape(g, slot);
        const old = weights.packedBytes(nk[0], nk[1], g.bits.of(i, slot));
        const next = modelopt.packedBytes(nk[0], nk[1], kind_);
        plan.weight_bytes = try std.math.add(usize, try std.math.sub(usize, plan.weight_bytes, old), next);
        plan.stage_bytes = @max(plan.stage_bytes, modelopt.rawBytes(nk[0], nk[1], kind_));
        plan.projections += 1;
        if (kind_ == .nvfp4) plan.nvfp4 += 1 else plan.bf16 += 1;
    };
    return plan;
}

fn aligned(n: usize) !usize {
    return (try std.math.add(usize, n, 255)) & ~@as(usize, 255);
}

fn matrix(n: usize, k: usize) !usize {
    const rows = (try std.math.add(usize, n, 127)) / 128 * 128;
    return aligned(try std.math.mul(usize, try std.math.mul(usize, rows, k), 2));
}

fn checkProducts(g: geometry.Geometry) !void {
    const conv = try std.math.mul(usize, try std.math.add(usize, try std.math.mul(usize, 2, g.k_heads), g.v_heads), geometry.dk);
    _ = try std.math.mul(usize, g.v_heads, geometry.dk);
    _ = try std.math.mul(usize, try std.math.mul(usize, g.q_heads, geometry.head_dim), 2);
    _ = try std.math.mul(usize, g.kv_heads, geometry.head_dim);
    var upper = try std.math.mul(usize, try matrix(g.vocab, g.hidden), 2);
    upper = try std.math.add(usize, upper, try aligned(try std.math.mul(usize, g.hidden, 2)));
    for (0..g.layers) |i| {
        upper = try std.math.add(usize, upper, try std.math.mul(usize, try aligned(try std.math.mul(usize, g.hidden, 2)), 2));
        for (weights.slots(g, i)) |slot| {
            const nk = weights.shape(g, slot);
            upper = try std.math.add(usize, upper, try matrix(nk[0], nk[1]));
        }
        if (g.linear[i]) {
            upper = try std.math.add(usize, upper, try aligned(try std.math.mul(usize, conv, geometry.taps * 2)));
            upper = try std.math.add(usize, upper, try aligned(geometry.dk * 2));
            upper = try std.math.add(usize, upper, try aligned(try std.math.mul(usize, g.v_heads, 4)));
            upper = try std.math.add(usize, upper, try aligned(try std.math.mul(usize, g.v_heads, 2)));
        } else upper = try std.math.add(usize, upper, 2 * geometry.head_dim * 2);
    }
}
