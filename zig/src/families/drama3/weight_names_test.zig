//! weight_names.zig over a recording loader: every read in order against a pinned list, each field's handle checked.
const std = @import("std");
const cfg = @import("config.zig");
const names = @import("weight_names.zig");

const pinned = @embedFile("fixtures/weight_names.txt");

/// The loader binding both backends pass, one line a read; handles count from 1, `first` keeps each one's tensor name.
const Rec = struct {
    gpa: std.mem.Allocator,
    out: std.ArrayList(u8) = .empty,
    first: std.ArrayList([]u8) = .empty,
    next: u64 = 1,

    fn deinit(r: *Rec) void {
        for (r.first.items) |n| r.gpa.free(n);
        r.first.deinit(r.gpa);
        r.out.deinit(r.gpa);
    }

    fn handle(r: *Rec, name: []const u8) !u64 {
        const copy = try r.gpa.dupe(u8, name);
        errdefer r.gpa.free(copy);
        try r.first.append(r.gpa, copy);
        r.next += 1;
        return r.next - 1;
    }

    pub fn one(r: *Rec, name: []const u8, shape: []const usize) !u64 {
        try r.out.print(r.gpa, "one {s} {any}\n", .{ name, shape });
        return r.handle(name);
    }

    pub fn stack(r: *Rec, parts: []const []const u8, rows: []const usize, k: usize, total_rows: usize) !u64 {
        try r.out.print(r.gpa, "stack {d}x{d}", .{ total_rows, k });
        for (parts, rows) |name, n| try r.out.print(r.gpa, " {s}:{d}", .{ name, n });
        try r.out.append(r.gpa, '\n');
        return r.handle(parts[0]);
    }

    /// Handle `h` came from the read of `prefix ++ suffix` (a stack by its first part).
    fn expectFrom(r: *const Rec, h: u64, prefix: []const u8, suffix: []const u8) !void {
        const n = r.first.items[h - 1];
        if (!std.mem.startsWith(u8, n, prefix) or !std.mem.endsWith(u8, n, suffix) or n.len != prefix.len + suffix.len) {
            std.debug.print("handle {d} holds '{s}', expected '{s}{s}'\n", .{ h, n, prefix, suffix });
            return error.FieldHoldsOtherRead;
        }
    }

    /// Every field `want` names holds the handle of the read `prefix ++ want.field`.
    fn expectFields(r: *const Rec, v: anytype, prefix: []const u8, comptime want: anytype) !void {
        inline for (comptime std.meta.fieldNames(@TypeOf(want))) |f| try r.expectFrom(@field(v, f), prefix, @field(want, f));
    }
};

const dense_want = .{
    .in_norm = "input_layernorm.weight",
    .post_norm = "post_attention_layernorm.weight",
    .moe_in = "mlp.shared_expert.gate_proj.weight",
    .shared_down = "mlp.shared_expert.down_proj.weight",
};
const attn_want = .{
    .qkv = "self_attn.q_proj.weight",
    .q_norm = "self_attn.q_norm.weight",
    .k_norm = "self_attn.k_norm.weight",
    .o = "self_attn.o_proj.weight",
};
const gdn_want = .{
    .proj = "linear_attn.in_proj_qkv.weight",
    .conv = "linear_attn.conv1d.weight",
    .a_log = "linear_attn.A_log",
    .dt_bias = "linear_attn.dt_bias",
    .norm = "linear_attn.norm.weight",
    .out = "linear_attn.out_proj.weight",
};
const tower_want = .{
    .enorm = "enorm.weight",
    .hnorm = "hnorm.weight",
    .eh = "eh_proj.weight",
    .attn_norm = "transformer_layer.attention_norm.weight",
    .qkv = "transformer_layer.attention.wqkv.weight",
    .q_norm = "transformer_layer.attention.q_norm.weight",
    .k_norm = "transformer_layer.attention.k_norm.weight",
    .wo = "transformer_layer.attention.wo.weight",
    .ffn_norm = "transformer_layer.ffn_norm.weight",
    .w13 = "transformer_layer.feed_forward.w1.weight",
    .w2 = "transformer_layer.feed_forward.w2.weight",
    .final_norm = "final_layernorm.weight",
};

/// Drama 3's layer_types: every fourth layer full attention, the rest linear (config.zig's test pins the same).
fn full(li: usize) bool {
    return li % 4 == 3;
}

fn record(a: std.mem.Allocator) !Rec {
    var r: Rec = .{ .gpa = a };
    errdefer r.deinit();
    const t = try names.top(&r);
    try r.expectFrom(t.embed, names.embed, "");
    try r.expectFrom(t.cb_emb, names.codebook_embeddings, "");
    try r.expectFrom(t.final_norm, names.final_norm, "");
    try r.out.print(a, "per-backend {s}\n", .{names.lm_head});
    var pb: [96]u8 = undefined;
    for (0..cfg.layers) |li| {
        const Layer = struct { in_norm: u64, post_norm: u64, gdn: names.Gdn, attn: names.Attn, moe_in: u64, shared_down: u64 };
        var L: Layer = undefined;
        names.setDense(&L, try names.dense(&r, li, full(li)));
        const pre = try std.fmt.bufPrint(&pb, "model.language_model.layers.{d}.", .{li});
        try r.expectFields(L, pre, dense_want);
        if (full(li)) try r.expectFields(L.attn, pre, attn_want) else try r.expectFields(L.gdn, pre, gdn_want);
    }
    for (0..cfg.tower_layers) |ti| {
        const T = try names.tower(&r, ti);
        const shared = ti < cfg.shared_layers;
        const pre = if (shared)
            try std.fmt.bufPrint(&pb, "audio_decoder_mtp_shared.layers.{d}.", .{ti})
        else
            try std.fmt.bufPrint(&pb, "audio_decoder_mtp.layers.{d}.", .{ti - cfg.shared_layers});
        try r.expectFields(T, pre, tower_want);
        try r.expectFrom(T.table, pre, if (shared) "semantic_embedding.weight" else "codebook_embedding.weight");
    }
    for (0..cfg.codebooks - 1) |hi| {
        const h = try names.head(&r, hi);
        try r.expectFrom(h, try std.fmt.bufPrint(&pb, "audio_decoder_mtp.heads.{d}.", .{hi}), "weight");
    }
    return r;
}

test "drama weight names: every tensor by name, shape and order, against the pinned list" {
    const a = std.testing.allocator;
    var r = try record(a);
    defer r.deinit();
    if (!std.mem.eql(u8, pinned, r.out.items)) {
        var got = std.mem.splitScalar(u8, r.out.items, '\n');
        var want = std.mem.splitScalar(u8, pinned, '\n');
        var line: usize = 1;
        while (true) : (line += 1) {
            const g = got.next();
            const w = want.next();
            if (g == null and w == null) break;
            if (g == null or w == null or !std.mem.eql(u8, g.?, w.?)) {
                std.debug.print("line {d}: got '{s}', pinned '{s}'\n", .{ line, g orelse "<end>", w orelse "<end>" });
                break;
            }
        }
        return error.WeightNamesChanged;
    }
}

test "drama weight names: expert names in every suffix both backends read" {
    var b: [160]u8 = undefined;
    const p = names.projections;
    try std.testing.expectEqualStrings("model.language_model.layers.39.mlp.experts.255.down_proj.weight_scale_2", names.expert(&b, 39, 255, p[2], "_scale_2"));
    try std.testing.expectEqualStrings("model.language_model.layers.0.mlp.experts.7.up_proj.weight_scale_inv", names.expert(&b, 0, 7, p[1], "_scale_inv"));
    try std.testing.expectEqualStrings("model.language_model.layers.3.mlp.experts.1.gate_proj", names.expertModule(&b, 3, 1, p[0]));
}
