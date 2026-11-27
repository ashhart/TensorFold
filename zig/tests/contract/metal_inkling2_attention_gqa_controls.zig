//! Count-checked, discriminating source edits for Inkling 2 Mini GQA 3:1 and batched attention kernels.
const std = @import("std");
const contract = @import("contract");
const sources = @import("inkling2_metal");

pub const Which = enum {
    decode_gqa_sliding,
    decode_gqa_global,
    decode_gqa_global_long,
    prep_batched_sliding,
    prep_batched_global,
    decode_gqa_batched_sliding,
    decode_gqa_batched_global,
    decode_gqa_batched_global_long,

    pub fn function(w: Which) []const u8 {
        return switch (w) {
            .decode_gqa_sliding, .decode_gqa_global, .decode_gqa_global_long => "tf_ink2_attn_scalar_gqa",
            .prep_batched_sliding, .prep_batched_global => "tf_ink2_prep_batched",
            .decode_gqa_batched_sliding, .decode_gqa_batched_global, .decode_gqa_batched_global_long => "tf_ink2_attn_scalar_gqa_batched",
        };
    }

    pub fn isBatched(w: Which) bool {
        return switch (w) {
            .prep_batched_sliding, .prep_batched_global, .decode_gqa_batched_sliding, .decode_gqa_batched_global, .decode_gqa_batched_global_long => true,
            else => false,
        };
    }

    pub fn isAttention(w: Which) bool {
        return w != .prep_batched_sliding and w != .prep_batched_global;
    }

    pub fn isGlobal(w: Which) bool {
        return w != .decode_gqa_sliding and w != .prep_batched_sliding and w != .decode_gqa_batched_sliding;
    }

    pub fn isLong(w: Which) bool {
        return w == .decode_gqa_global_long or w == .decode_gqa_batched_global_long;
    }
};

pub const Mutation = enum {
    none,
    gqa_wrong_head,
    gqa_rel_late,
    gqa_late_bias,
    prep_batched_wrong_seg,
    prep_batched_tau_off,
    gqa_batched_wrong_seg,

    pub fn name(m: Mutation) []const u8 {
        return switch (m) {
            .none => "",
            .gqa_wrong_head => "control: metal inkling2 attention scalar gqa reuses head 0 query for all 3 heads tf_ink2_attn_scalar_gqa",
            .gqa_rel_late => "control: metal inkling2 attention scalar gqa relative projection one column late tf_ink2_attn_scalar_gqa",
            .gqa_late_bias => "control: metal inkling2 attention scalar gqa first key score boosted tf_ink2_attn_scalar_gqa",
            .prep_batched_wrong_seg => "control: metal inkling2 attention prep batched writes all segments into seg 0 tf_ink2_prep_batched",
            .prep_batched_tau_off => "control: metal inkling2 attention prep batched global without tau tf_ink2_prep_batched",
            .gqa_batched_wrong_seg => "control: metal inkling2 attention scalar gqa batched reads seg 0 for all segments tf_ink2_attn_scalar_gqa_batched",
        };
    }

    pub fn gate(m: Mutation) ?contract.gates.Gate {
        return if (m == .none) null else .shapes;
    }

    pub fn which(m: Mutation) Which {
        return switch (m) {
            .none => unreachable,
            .gqa_wrong_head, .gqa_rel_late => .decode_gqa_sliding,
            .gqa_late_bias => .decode_gqa_global,
            .prep_batched_wrong_seg => .prep_batched_sliding,
            .prep_batched_tau_off => .prep_batched_global,
            .gqa_batched_wrong_seg => .decode_gqa_batched_sliding,
        };
    }

    const Edit = struct { from: []const u8, to: []const u8 };

    fn edit(m: Mutation) Edit {
        return switch (m) {
            .none => unreachable,
            .gqa_wrong_head => .{
                .from = "const float dot_score = ink_warp_sum(dot(qv[g], kv));",
                .to = "const float dot_score = ink_warp_sum(dot(qv[0], kv));",
            },
            .gqa_rel_late => .{
                .from = "const float pj = need_bias ? float(proj[j * EXTENT + dist_safe]) : 0.0f;",
                .to = "const float pj = need_bias ? float(proj[j * EXTENT + (dist_safe + 1) % EXTENT]) : 0.0f;",
            },
            .gqa_late_bias => .{
                .from = "const float score_g = (dot_score * a.scale + bias) * LOG2E;",
                .to = "const float score_g = (dot_score * a.scale + bias + (kp == from ? 128.0f : 0.0f)) * LOG2E;",
            },
            .prep_batched_wrong_seg => .{
                .from = "const InkBatchedSeg bs = segs[rseg[row]];",
                .to = "const InkBatchedSeg bs = segs[0];",
            },
            .prep_batched_tau_off => .{
                .from = "const float t_scale = a.global ? 1.0f + a.alpha * precise::log(max(float(pos + 1) / a.floor_pos, 1.0f)) : 1.0f;",
                .to = "const float t_scale = 1.0f;",
            },
            .gqa_batched_wrong_seg => .{
                .from = "const InkBatchedSeg bseg = segs[rseg[row]];",
                .to = "const InkBatchedSeg bseg = segs[0];",
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

test "Inkling 2 GQA and batched attention controls each hit one live statement and cannot apply twice" {
    const a = std.testing.allocator;
    for (std.enums.values(Mutation)) |m| {
        if (m == .none) continue;
        const e = m.edit();
        try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, sources.attention, e.from));
        const edited = try apply(a, sources.attention, m);
        defer a.free(edited);
        try std.testing.expect(std.mem.indexOf(u8, edited, e.to) != null);
        try std.testing.expect(std.mem.indexOf(u8, edited, m.which().function()) != null);
        try std.testing.expectError(error.ControlEditMissing, apply(a, edited, m));
        try std.testing.expectError(error.ControlEditMissing, apply(a, "", m));
    }
    const unchanged = try apply(a, sources.attention, .none);
    defer a.free(unchanged);
    try std.testing.expectEqualStrings(sources.attention, unchanged);
}

test "repeated Inkling 2 GQA control anchors are ambiguous" {
    const a = std.testing.allocator;
    const twice = try std.mem.concat(a, u8, &.{ sources.attention, sources.attention });
    defer a.free(twice);
    for (std.enums.values(Mutation)) |m| {
        if (m == .none) continue;
        try std.testing.expectError(error.ControlEditMissing, apply(a, twice, m));
    }
}
