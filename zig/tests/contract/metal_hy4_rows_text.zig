//! The rows group's library texts as their loaders build them, its planted faults, and one pipeline from a text.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const sources = @import("kernel_sources");
const h = @import("metal_hy4.zig");

/// A planted control: the word its gate-5 record carries, the source edit, and what the edit drops.
pub const Fault = struct { token: []const u8, edit: h.fl.Edit, drops: []const u8 };

pub const tail: Fault = .{ .token = "bad-tail", .drops = "the 2-row tail dropped", .edit = .{ .kernel = "tf_hy4_rows", .from = "if (m + 2 <= rows)", .to = "if (m + 3 <= rows)" } };
pub const tie: Fault = .{ .token = "bad-tie", .drops = "a tie goes to the higher id", .edit = .{ .kernel = "tf_hy4_argmax", .from = "(ov == best && oi < id)", .to = "(ov == best && oi > id)" } };
pub const nan_wins: Fault = .{ .token = "bad-nan", .drops = "a NaN logit wins", .edit = .{ .kernel = "tf_hy4_argmax", .from = "if (v > best)", .to = "if (v > best || isnan(v))" } };
pub const scale: Fault = .{ .token = "bad-scale", .drops = "the fp8 scale's bf16 rounding dropped", .edit = .{ .kernel = "tf_hy4_engram_lookup", .from = "const float scale = bf(precise::divide(", .to = "const float scale = (precise::divide(" } };
pub const gain: Fault = .{ .token = "bad-gain", .drops = "every stream's key gain read from stream 0's", .edit = .{ .kernel = "tf_hy4_engram_apply", .from = "float(key_gain[gain + j])", .to = "float(key_gain[j])" } };
pub const group: Fault = .{ .token = "bad-group", .drops = "every cell's scale and bias read from its row's first group", .edit = .{ .kernel = "tf_hy4_engram_gather", .from = "row * (head_width / 32) + i / 32", .to = "row * (head_width / 32)" } };
pub const round: Fault = .{ .token = "bad-round", .drops = "a bf16 rounding point added before the bias", .edit = .{ .kernel = "tf_dequant_unit", .from = "out[i] = scale * q + bias;", .to = "out[i] = bfloat(float(bfloat(float(scale) * float(q))) + float(bias));" } };
pub const all = [_]Fault{ tail, tie, nan_wins, scale, gain, group, round };

fn editOf(fault: ?Fault) ?h.fl.Edit {
    return if (fault) |f| f.edit else null;
}

/// metal_linears.zig's Rows.load: the wide core row projection source on every chip, then rows.metal.
pub fn rows(a: std.mem.Allocator, fault: ?Fault) ![]u8 {
    const body = try h.fl.edited(a, sources.hy4_flash_rows, editOf(fault));
    defer a.free(body);
    return std.mem.concat(a, u8, &.{ tf.row_projection.source(true), body });
}

/// metal_engram.zig's Kernels.load: engram.metal as is.
pub fn engram(a: std.mem.Allocator, fault: ?Fault) ![]u8 {
    return h.fl.edited(a, sources.hy4_flash_engram, editOf(fault));
}

/// metal_model.zig's Head.upload: core embed_norm.metal as is.
pub fn embedNorm(a: std.mem.Allocator, fault: ?Fault) ![]u8 {
    return h.fl.edited(a, sources.embed_norm, editOf(fault));
}

/// One function of a text, compiled as the loaders compile it (MLX's options).
pub fn pipeline(device: mtl.Device, text: []const u8, fun: []const u8) !mtl.Pipeline {
    const lib = try mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
    defer lib.deinit();
    return mtl.Pipeline.init(device, lib, fun, false);
}

/// The fault a gate-5 record's name carries, if any.
pub fn faultIn(name: []const u8) ?Fault {
    for (all) |f| if (std.mem.indexOf(u8, name, f.token) != null) return f;
    return null;
}

/// Gate 5's texts for `hy4 rows `, `hy4 engram ` and `hy4 embed_norm ` records, as each loader builds them.
pub fn limitsText(a: std.mem.Allocator, _: mtl.Device, name: []const u8) !?h.fl.Texts {
    const fault = faultIn(name);
    if (std.mem.startsWith(u8, name, "hy4 rows ")) {
        const text = try rows(a, fault);
        return .{ .here = text, .pre_m5 = text };
    }
    if (std.mem.startsWith(u8, name, "hy4 engram ")) {
        const text = try engram(a, fault);
        return .{ .here = text, .pre_m5 = text };
    }
    if (std.mem.startsWith(u8, name, "hy4 embed_norm ")) {
        const text = try embedNorm(a, fault);
        return .{ .here = text, .pre_m5 = text };
    }
    return null;
}
