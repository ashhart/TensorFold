//! Host-checkable arithmetic and guard mutations for every Mini DFlash 2 Metal kernel.
const std = @import("std");
const g = @import("contract").gates;

pub const Control = enum {
    embed,
    norm,
    norm16,
    conv,
    ctx_kv,
    block_qkv,
    attend,
    attend_merge,
    silu_mul,
    topk16,
    topk16_nonfinite,
    walk,
    walk_nonfinite,

    pub fn gate(ctl: Control) g.Gate {
        return switch (ctl) {
            .topk16_nonfinite, .walk_nonfinite => .nonfinite,
            else => .shapes,
        };
    }
};

const Edit = struct { from: []const u8, to: []const u8 };

fn edit(ctl: Control) Edit {
    return switch (ctl) {
        .embed => .{
            .from = "for (uint i = t; i < a.width; i += 256) dst[i] = float(bfloat(float(src[i]) * inv * float(weight[i])));",
            .to = "for (uint i = t; i < a.width; i += 256) dst[i] = float(bfloat(float(src[i]) * inv));",
        },
        .norm => .{
            .from = "    if (a.add) {\n      v = float(bfloat(v + float(x[i])));\n      r[i] = v;\n    }",
            .to = "    if (a.add) {\n      v = float(bfloat(v));\n      r[i] = v;\n    }",
        },
        .norm16 => .{
            .from = "for (uint i = t; i < a.width; i += 256) y[i] = bfloat(float(w[i]) * float(bfloat(float(x[i]) * inv)));",
            .to = "for (uint i = t; i < a.width; i += 256) y[i] = bfloat(float(bfloat(float(x[i]) * inv)));",
        },
        .conv => .{
            .from = "if (r % a.block >= 1) y = float(bfloat(y + float(bfloat(c1 * float(X[(size_t)(r - 1) * a.width + ch])))));",
            .to = "if (false) y = float(bfloat(y + float(bfloat(c1 * float(X[(size_t)(r - 1) * a.width + ch])))));",
        },
        .ctx_kv => .{
            .from = "const uint pos = seg.pos0 + (row - seg.row0);",
            .to = "const uint pos = seg.pos0;",
        },
        .block_qkv => .{
            .from = "const uint pos = a.pos0[stream] + (row % a.block);",
            .to = "const uint pos = a.pos0[stream];",
        },
        .attend => .{
            .from = "const uint row_lo = max(st.from, reach > a.window ? reach - a.window : 0u);",
            .to = "const uint row_lo = st.from;",
        },
        .attend_merge => .{
            .from = "const float w = precise::exp(p[DF2_D] - maximum);",
            .to = "const float w = 1.0f;",
        },
        .silu_mul => .{
            .from = "out[i] = bfloat(float(bfloat(g * s)) * float(up[i]));",
            .to = "out[i] = bfloat(g * float(up[i]));",
        },
        .topk16 => .{
            .from = "return key > ok || (key == ok && id < oid);",
            .to = "return key < ok || (key == ok && id < oid);",
        },
        .topk16_nonfinite => .{
            .from = "if ((raw & 0x7f80u) == 0x7f80u) {",
            .to = "if (false && (raw & 0x7f80u) == 0x7f80u) {",
        },
        .walk => .{
            .from = "for (uint k = 0; k < a.rank / 32; ++k) s += part[k][c];",
            .to = "for (uint k = 0; k < 0; ++k) s += part[k][c];",
        },
        .walk_nonfinite => .{
            .from = "    if (prev >= a.vocab) {\n      if (t == 0) s_seq[1 + e] = 0xffffffffu;\n      continue;\n    }",
            .to = "    if (prev >= a.vocab) {\n      prev = 0u;\n    }",
        },
    };
}

pub fn name(ctl: Control) []const u8 {
    return switch (ctl) {
        inline else => |k| "control: metal inkling2 dflash2 " ++ @tagName(k),
    };
}

pub fn source(a: std.mem.Allocator, production: []const u8, ctl: Control) ![]u8 {
    const e = edit(ctl);
    if (std.mem.count(u8, production, e.from) != 1) return error.ControlEditMissing;
    return std.mem.replaceOwned(u8, a, production, e.from, e.to);
}

test "every DFlash2 control applies once and only mutates its arithmetic or guard anchor" {
    const production = @import("inkling2_metal").inkling2_dflash2;
    const a = std.testing.allocator;
    for (std.enums.values(Control)) |ctl| {
        const e = edit(ctl);
        try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, production, e.from));
        const changed = try source(a, production, ctl);
        defer a.free(changed);
        const at = std.mem.indexOf(u8, production, e.from).?;
        try std.testing.expectEqualStrings(production[0..at], changed[0..at]);
        try std.testing.expectEqualStrings(e.to, changed[at..][0..e.to.len]);
        try std.testing.expectEqualStrings(production[at + e.from.len ..], changed[at + e.to.len ..]);
        try std.testing.expectError(error.ControlEditMissing, source(a, changed, ctl));
        try std.testing.expectError(error.ControlEditMissing, source(a, "", ctl));
    }
}

test "repeated DFlash2 control anchors are refused" {
    const production = @import("inkling2_metal").inkling2_dflash2;
    const a = std.testing.allocator;
    const twice = try std.mem.concat(a, u8, &.{ production, production });
    defer a.free(twice);
    for (std.enums.values(Control)) |ctl| try std.testing.expectError(error.ControlEditMissing, source(a, twice, ctl));
}
