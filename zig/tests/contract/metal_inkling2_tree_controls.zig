//! Count-checked, discriminating source edits for every Inkling 2 Mini tree shader kernel.
const std = @import("std");
const contract = @import("contract");
const sources = @import("inkling2_metal");

pub const Which = enum {
    prep_tree,
    attn_scalar_tree,
    prompt_m5_tree_prefix,
    prompt_pre_m5_tree_prefix,
    prompt_m5_tree_tail,
    prompt_pre_m5_tree_tail,
    conv_static_tree,
    conv_dynamic_tree,
    keep_kv,
    keep_conv,

    pub fn function(w: Which) []const u8 {
        return switch (w) {
            .prep_tree => "tf_ink2_prep_tree",
            .attn_scalar_tree => "tf_ink2_attn_scalar_tree",
            .prompt_m5_tree_prefix, .prompt_pre_m5_tree_prefix => "tf_ink2_attn_prompt_tree_prefix",
            .prompt_m5_tree_tail, .prompt_pre_m5_tree_tail => "tf_ink2_attn_prompt_tree_tail",
            .conv_static_tree => "tf_ink2_conv_static_tree",
            .conv_dynamic_tree => "tf_ink2_conv_dynamic_tree",
            .keep_kv => "tf_ink2_keep_kv",
            .keep_conv => "tf_ink2_keep_conv",
        };
    }

    pub fn isPrompt(w: Which) bool {
        return w == .prompt_m5_tree_prefix or w == .prompt_pre_m5_tree_prefix or
            w == .prompt_m5_tree_tail or w == .prompt_pre_m5_tree_tail;
    }

    pub fn isAttention(w: Which) bool {
        return w == .attn_scalar_tree or w.isPrompt();
    }

    pub fn baseSource(w: Which) []const u8 {
        return switch (w) {
            .prep_tree, .attn_scalar_tree => sources.attention,
            .prompt_m5_tree_prefix, .prompt_m5_tree_tail => sources.prompt_attention,
            .prompt_pre_m5_tree_prefix, .prompt_pre_m5_tree_tail => sources.prompt_attention_pre_m5,
            .conv_static_tree, .conv_dynamic_tree, .keep_kv, .keep_conv => sources.tree,
        };
    }
};

pub const Mutation = enum {
    none,
    prep_tree_pos,
    attn_scalar_tree_phys,
    prompt_m5_prefix_live,
    prompt_pre_m5_prefix_live,
    prompt_m5_tail_phys,
    prompt_pre_m5_tail_phys,
    conv_static_tree_tap,
    conv_dynamic_tree_lag,
    keep_kv_identity,
    keep_conv_zero,

    pub fn name(m: Mutation) []const u8 {
        return switch (m) {
            .none => "",
            .prep_tree_pos => "control: metal inkling2 attention prep tree writes kv at logical depth pos tf_ink2_prep_tree",
            .attn_scalar_tree_phys => "control: metal inkling2 attention scalar tree ignores ancestor remap tf_ink2_attn_scalar_tree",
            .prompt_m5_prefix_live => "control: metal inkling2 prompt m5 tree prefix drops first key tf_ink2_attn_prompt_tree_prefix",
            .prompt_pre_m5_prefix_live => "control: metal inkling2 prompt pre-m5 tree prefix drops first key tf_ink2_attn_prompt_tree_prefix",
            .prompt_m5_tail_phys => "control: metal inkling2 prompt m5 tree tail ignores ancestor remap tf_ink2_attn_prompt_tree_tail",
            .prompt_pre_m5_tail_phys => "control: metal inkling2 prompt pre-m5 tree tail ignores ancestor remap tf_ink2_attn_prompt_tree_tail",
            .conv_static_tree_tap => "control: metal inkling2 tree conv_static reverses tap order tf_ink2_conv_static_tree",
            .conv_dynamic_tree_lag => "control: metal inkling2 tree conv_dynamic reverses lag order tf_ink2_conv_dynamic_tree",
            .keep_kv_identity => "control: metal inkling2 tree keep_kv ignores path indices tf_ink2_keep_kv",
            .keep_conv_zero => "control: metal inkling2 tree keep_conv zeroes histories tf_ink2_keep_conv",
        };
    }

    pub fn gate(m: Mutation) ?contract.gates.Gate {
        return if (m == .none) null else .shapes;
    }

    pub fn which(m: Mutation) Which {
        return switch (m) {
            .none => unreachable,
            .prep_tree_pos => .prep_tree,
            .attn_scalar_tree_phys => .attn_scalar_tree,
            .prompt_m5_prefix_live => .prompt_m5_tree_prefix,
            .prompt_pre_m5_prefix_live => .prompt_pre_m5_tree_prefix,
            .prompt_m5_tail_phys => .prompt_m5_tree_tail,
            .prompt_pre_m5_tail_phys => .prompt_pre_m5_tree_tail,
            .conv_static_tree_tap => .conv_static_tree,
            .conv_dynamic_tree_lag => .conv_dynamic_tree,
            .keep_kv_identity => .keep_kv,
            .keep_conv_zero => .keep_conv,
        };
    }

    const Edit = struct { from: []const u8, to: []const u8 };

    fn edit(m: Mutation) Edit {
        return switch (m) {
            .none => unreachable,
            .prep_tree_pos => .{
                .from = "const uint kv_pos = tree ? uint(s.p0) + row - uint(s.row0) : pos;",
                .to = "const uint kv_pos = pos;",
            },
            .attn_scalar_tree_phys => .{
                .from = "const uint phys = (tree && kp >= uint(s.p0)) ? uint(s.p0) + tree_paths[row * 16 + (kp - uint(s.p0))] : kp;",
                .to = "const uint phys = kp;",
            },
            .prompt_m5_prefix_live, .prompt_pre_m5_prefix_live => .{
                .from = "const bool live = row < total && kp <= k_hi && kp >= sub_lo && kp <= sub_hi;",
                .to = "const bool live = row < total && kp <= k_hi && kp >= sub_lo && kp <= sub_hi && kp != b_lo;",
            },
            .prompt_m5_tail_phys, .prompt_pre_m5_tail_phys => .{
                .from = "const uint phys = !live_k ? kt : (kp < uint(s.p0) ? kp : uint(s.p0) + tree_paths[tr * 16 + (kp - uint(s.p0))]);",
                .to = "const uint phys = !live_k ? kt : kp;",
            },
            .conv_static_tree_tap => .{
                .from = "const uint idx = src[r * 4 + tap];",
                .to = "const uint idx = src[r * 4 + 3 - tap];",
            },
            .conv_dynamic_tree_lag => .{
                .from = "const uint idx = src[r * 4 + 3 - lag];",
                .to = "const uint idx = src[r * 4 + lag];",
            },
            .keep_kv_identity => .{
                .from = "const ulong src = ulong((a.start + a.path[j]) % ring) * 128 + d;",
                .to = "const ulong src = ulong((a.start + j) % ring) * 128 + d;",
            },
            .keep_conv_zero => .{
                .from = "conv[hist_base + ulong(out_r) * width] = v[out_r];",
                .to = "conv[hist_base + ulong(out_r) * width] = bfloat(0);",
            },
        };
    }
};

pub fn apply(a: std.mem.Allocator, text: []const u8, m: Mutation) ![]u8 {
    if (m == .none) return a.dupe(u8, text);
    const e = m.edit();
    if (std.mem.count(u8, text, e.from) != 1) return error.ControlEditMissing;
    return std.mem.replaceOwned(u8, a, text, e.from, e.to);
}

test "Inkling 2 tree controls each hit one live statement and cannot apply twice" {
    const a = std.testing.allocator;
    for (std.enums.values(Mutation)) |m| {
        if (m == .none) continue;
        const base = m.which().baseSource();
        const e = m.edit();
        try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, base, e.from));
        const edited = try apply(a, base, m);
        defer a.free(edited);
        const at = std.mem.indexOf(u8, base, e.from).?;
        try std.testing.expectEqualStrings(base[0..at], edited[0..at]);
        try std.testing.expectEqualStrings(e.to, edited[at..][0..e.to.len]);
        try std.testing.expectEqualStrings(base[at + e.from.len ..], edited[at + e.to.len ..]);
        try std.testing.expect(std.mem.indexOf(u8, edited, m.which().function()) != null);
        try std.testing.expectError(error.ControlEditMissing, apply(a, edited, m));
        try std.testing.expectError(error.ControlEditMissing, apply(a, "", m));
    }
}

test "repeated Inkling 2 tree control anchors are ambiguous" {
    const a = std.testing.allocator;
    for (std.enums.values(Mutation)) |m| {
        if (m == .none) continue;
        const base = m.which().baseSource();
        const twice = try std.mem.concat(a, u8, &.{ base, base });
        defer a.free(twice);
        try std.testing.expectError(error.ControlEditMissing, apply(a, twice, m));
    }
}
