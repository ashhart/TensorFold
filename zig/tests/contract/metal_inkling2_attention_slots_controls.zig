//! Count-checked source edits for the batched Inkling 2 non-tree Metal shaders.
const std = @import("std");
const contract = @import("contract");

pub const Which = enum {
    prep_slots,
    attn_slots_sliding,
    attn_slots_global,
};
pub fn function(w: Which) []const u8 {
    return switch (w) {
        .prep_slots => "tf_ink2_prep_slots",
        .attn_slots_sliding, .attn_slots_global => "tf_ink2_attn_slots",
    };
}
pub fn isAttention(w: Which) bool {
    return w != .prep_slots;
}

pub const Mutation = enum {
    none,
    prep_slots_bad_slot,
    attn_slots_bad_slot,
    pub fn name(m: Mutation) []const u8 {
        return switch (m) {
            .none => "",
            .prep_slots_bad_slot => "control: metal inkling2 attention prep slots ignoring rseg slot tf_ink2_prep_slots",
            .attn_slots_bad_slot => "control: metal inkling2 attention slots ignoring rseg slot tf_ink2_attn_slots",
        };
    }
    pub fn gate(m: Mutation) ?contract.gates.Gate {
        return if (m == .none) null else .shapes;
    }
    pub fn which(m: Mutation) Which {
        return switch (m) {
            .none => unreachable,
            .prep_slots_bad_slot => .prep_slots,
            .attn_slots_bad_slot => .attn_slots_sliding,
        };
    }
    const Edit = struct { from: []const u8, to: []const u8 };
    fn edit(m: Mutation) Edit {
        return switch (m) {
            .none => unreachable,
            .prep_slots_bad_slot => .{
                .from = "device bfloat* dst = (key ? kc_arr[slot] : vc_arr[slot]) + (size_t(h) * ring + pos % ring) * D;",
                .to = "device bfloat* dst = (key ? kc_arr[0] : vc_arr[0]) + (size_t(h) * ring + pos % ring) * D;",
            },
            .attn_slots_bad_slot => .{
                .from = "const device bfloat* kc_s = kc_arr[slot];",
                .to = "const device bfloat* kc_s = kc_arr[0];",
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

test "Inkling 2 slot controls hit one live statement and cannot apply twice" {
    const a = std.testing.allocator;
    const text = @import("inkling2_metal").attention;
    for (std.enums.values(Mutation)) |m| {
        if (m == .none) continue;
        const edited = try apply(a, text, m);
        defer a.free(edited);
        try std.testing.expectError(error.ControlEditMissing, apply(a, edited, m));
    }
}
