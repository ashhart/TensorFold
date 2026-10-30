//! GLM-5.3-Flash's image tower tensors: names under either checkpoint prefix, shapes and conv layouts (every backend).
const std = @import("std");
const vc = @import("vision_config.zig");

/// The tower's prefixes: the original checkpoint's, then mlx-vlm's (glm5_next's sanitize renames the first to it).
pub const prefixes = [_][]const u8{ "model.visual.", "vision_tower." };

/// A top-level tower tensor's place, named as the Metal tower's fields.
pub const Part = enum { patch_w, patch_b, post_norm, down_w, down_b, m_proj, m_norm_w, m_norm_b, m_gate, m_up, m_down };
/// A block tensor's place, named as the Metal tower's block fields.
pub const BlockPart = enum { norm1, norm2, qkv_w, qkv_b, proj_w, proj_b, q_norm, k_norm, gate_w, gate_b, up_w, up_b, down_w, down_b };
pub const Place = union(enum) { top: Part, block: struct { i: usize, part: BlockPart } };

/// A tower tensor: its name after the prefix, its bf16 shape in the original (PyTorch) layout and its place.
pub const Spec = struct { suffix: []const u8, shape: []const usize, place: Place };

const head = .{
    .{ "patch_embed.proj.weight", Part.patch_w, &[_]usize{ vc.hidden, 3, vc.temporal, vc.patch, vc.patch } },
    .{ "patch_embed.proj.bias", Part.patch_b, &[_]usize{vc.hidden} },
};
const block = .{
    .{ "norm1.weight", BlockPart.norm1, &[_]usize{vc.hidden} },
    .{ "norm2.weight", BlockPart.norm2, &[_]usize{vc.hidden} },
    .{ "attn.qkv.weight", BlockPart.qkv_w, &[_]usize{ 3 * vc.hidden, vc.hidden } },
    .{ "attn.qkv.bias", BlockPart.qkv_b, &[_]usize{3 * vc.hidden} },
    .{ "attn.proj.weight", BlockPart.proj_w, &[_]usize{ vc.hidden, vc.hidden } },
    .{ "attn.proj.bias", BlockPart.proj_b, &[_]usize{vc.hidden} },
    .{ "attn.q_norm.weight", BlockPart.q_norm, &[_]usize{vc.head_dim} },
    .{ "attn.k_norm.weight", BlockPart.k_norm, &[_]usize{vc.head_dim} },
    .{ "mlp.gate_proj.weight", BlockPart.gate_w, &[_]usize{ vc.mlp_inter, vc.hidden } },
    .{ "mlp.gate_proj.bias", BlockPart.gate_b, &[_]usize{vc.mlp_inter} },
    .{ "mlp.up_proj.weight", BlockPart.up_w, &[_]usize{ vc.mlp_inter, vc.hidden } },
    .{ "mlp.up_proj.bias", BlockPart.up_b, &[_]usize{vc.mlp_inter} },
    .{ "mlp.down_proj.weight", BlockPart.down_w, &[_]usize{ vc.hidden, vc.mlp_inter } },
    .{ "mlp.down_proj.bias", BlockPart.down_b, &[_]usize{vc.hidden} },
};
const tail = .{
    .{ "post_layernorm.weight", Part.post_norm, &[_]usize{vc.hidden} },
    .{ "downsample.weight", Part.down_w, &[_]usize{ vc.out_hidden, vc.hidden, vc.merge, vc.merge } },
    .{ "downsample.bias", Part.down_b, &[_]usize{vc.out_hidden} },
    .{ "merger.proj.weight", Part.m_proj, &[_]usize{ vc.out_hidden, vc.out_hidden } },
    .{ "merger.post_projection_norm.weight", Part.m_norm_w, &[_]usize{vc.out_hidden} },
    .{ "merger.post_projection_norm.bias", Part.m_norm_b, &[_]usize{vc.out_hidden} },
    .{ "merger.gate_proj.weight", Part.m_gate, &[_]usize{ vc.merger_inter, vc.out_hidden } },
    .{ "merger.up_proj.weight", Part.m_up, &[_]usize{ vc.merger_inter, vc.out_hidden } },
    .{ "merger.down_proj.weight", Part.m_down, &[_]usize{ vc.out_hidden, vc.merger_inter } },
};

pub const count = head.len + vc.depth * block.len + tail.len;

/// Every tower tensor, in the Metal tower's buffer order.
pub const tensors: [count]Spec = table: {
    @setEvalBranchQuota(200_000);
    var out: [count]Spec = undefined;
    var n: usize = 0;
    for (head) |t| {
        out[n] = .{ .suffix = t[0], .shape = t[2], .place = .{ .top = t[1] } };
        n += 1;
    }
    for (0..vc.depth) |i| for (block) |t| {
        out[n] = .{ .suffix = std.fmt.comptimePrint("blocks.{d}.{s}", .{ i, t[0] }), .shape = t[2], .place = .{ .block = .{ .i = i, .part = t[1] } } };
        n += 1;
    };
    for (tail) |t| {
        out[n] = .{ .suffix = t[0], .shape = t[2], .place = .{ .top = t[1] } };
        n += 1;
    }
    break :table out;
};

fn index(suffix: []const u8) ?usize {
    for (tensors, 0..) |t, i| if (std.mem.eql(u8, t.suffix, suffix)) return i;
    return null;
}

/// A refusal's reason, logged as an error; tests check the error itself and keep the log quiet.
fn say(comptime fmt: []const u8, args: anytype) void {
    if (@import("builtin").is_test) return;
    std.log.err(fmt, args);
}

/// A checkpoint's names counted under each prefix, with the tower tensors each prefix holds.
pub const Tally = struct {
    names: [prefixes.len]usize = @splat(0),
    held: [prefixes.len]std.StaticBitSet(count) = @splat(.empty),

    pub fn add(t: *Tally, name: []const u8) void {
        for (prefixes, 0..) |p, i| if (std.mem.startsWith(u8, name, p)) {
            t.names[i] += 1;
            if (index(name[p.len..])) |at| t.held[i].set(at);
        };
    }

    /// The one prefix the whole tower is stored under; none, both or a partial tower refuses with the reason logged.
    pub fn tower(t: *const Tally) error{ NoVisionTower, MixedVisionTower, PartialVisionTower }![]const u8 {
        if (t.names[0] > 0 and t.names[1] > 0) {
            say("glm vision: the checkpoint stores image tower tensors under both {s} ({d}) and {s} ({d})", .{ prefixes[0], t.names[0], prefixes[1], t.names[1] });
            return error.MixedVisionTower;
        }
        const i = for (t.names, 0..) |n, i| {
            if (n > 0) break i;
        } else {
            say("glm vision: the checkpoint has no image tower tensors (no {s} or {s} names)", .{ prefixes[0], prefixes[1] });
            return error.NoVisionTower;
        };
        const held = t.held[i].count();
        if (held < count) {
            var gap = t.held[i];
            gap.toggleAll();
            say("glm vision: the image tower under {s} is partial ({d} of {d} tensors); {s}{s} is missing", .{ prefixes[i], held, count, prefixes[i], tensors[gap.findFirstSet().?].suffix });
            return error.PartialVisionTower;
        }
        return prefixes[i];
    }
};

/// How a stored tensor's elements lie: as `Spec.shape`, or a conv kernel channels-last as mlx-vlm saves it.
pub const Layout = enum { original, channels_last };

fn conv(spec: Spec) bool {
    return spec.place == .top and (spec.place.top == .patch_w or spec.place.top == .down_w);
}

/// `stored`'s layout of `spec`'s tensor; null for any other shape (only the two conv kernels have a second layout).
pub fn layoutOf(spec: Spec, stored: []const usize) ?Layout {
    const s = spec.shape;
    if (std.mem.eql(usize, stored, s)) return .original;
    if (!conv(spec) or stored.len != s.len or stored[0] != s[0] or stored[s.len - 1] != s[1]) return null;
    return if (std.mem.eql(usize, stored[1 .. s.len - 1], s[2..])) .channels_last else null;
}

/// The row transpose ([a][b] to [b][a]) putting a conv kernel in the layout the towers read; null: read as stored.
pub fn transposeOf(spec: Spec, layout: Layout) ?[2]usize {
    if (!conv(spec)) return null;
    const s = spec.shape;
    var taps: usize = 1;
    for (s[2..]) |d| taps *= d;
    return switch (spec.place.top) {
        .patch_w => if (layout == .channels_last) .{ taps, s[1] } else null, // a patch is channel by frame by row by column
        else => if (layout == .original) .{ s[1], taps } else null, // the merge reads its four taps outermost
    };
}

/// For the conv kernel stored under `suffix` with shape `stored`, its row transpose (null: none or not a kernel).
pub fn arrangement(suffix: []const u8, stored: []const usize) error{BadVisionShape}!?[2]usize {
    const spec = tensors[index(suffix) orelse return null];
    if (!conv(spec)) return null;
    return transposeOf(spec, layoutOf(spec, stored) orelse {
        say("glm vision: {s} is {any}; the tower reads {any} or its channels-last layout", .{ suffix, stored, spec.shape });
        return error.BadVisionShape;
    });
}

/// `src`'s rows of a * b elements into `dst`, each row's [a][b] transposed to [b][a].
pub fn transposeRows(dst: []u16, src: []align(1) const u16, a: usize, b: usize) void {
    std.debug.assert(dst.len == src.len and src.len % (a * b) == 0);
    var r: usize = 0;
    while (r < src.len) : (r += a * b) for (0..a) |i| for (0..b) |j| {
        dst[r + j * a + i] = src[r + i * b + j];
    };
}

/// A tower tensor as stored: its spec, full name, source and layout.
pub fn Planned(comptime Src: type) type {
    return struct { spec: Spec, name: []const u8, src: Src, layout: Layout };
}

/// Every tower tensor under `prefix` in `srcs` (stored names to sources with `dtype` and `shape`), bf16 or refused.
pub fn plan(comptime Src: type, arena: std.mem.Allocator, prefix: []const u8, srcs: *const std.StringHashMapUnmanaged(Src)) ![]Planned(Src) {
    const out = try arena.alloc(Planned(Src), count);
    for (tensors, out) |spec, *p| {
        const name = try std.fmt.allocPrint(arena, "{s}{s}", .{ prefix, spec.suffix });
        const src = srcs.get(name) orelse {
            say("glm vision: the checkpoint has no tensor {s}", .{name});
            return error.MissingTensor;
        };
        const layout = (if (src.dtype == .bf16) layoutOf(spec, src.shape) else null) orelse {
            say("glm vision: {s} is {t} {any}; the tower reads bf16 {any}", .{ name, src.dtype, src.shape, spec.shape });
            return error.UnexpectedTensor;
        };
        p.* = .{ .spec = spec, .name = name, .src = src, .layout = layout };
    }
    return out;
}

test {
    _ = @import("vision_names_test.zig");
}
