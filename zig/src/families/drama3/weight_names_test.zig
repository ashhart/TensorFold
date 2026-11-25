//! weight_names.zig over a recording loader: every tensor both backends read, in order, against a pinned list.
const std = @import("std");
const cfg = @import("config.zig");
const names = @import("weight_names.zig");

const pinned = @embedFile("fixtures/weight_names.txt");

/// The loader binding both backends pass, writing one line a tensor or stack; addresses count up from 1.
const Rec = struct {
    gpa: std.mem.Allocator,
    out: std.ArrayList(u8) = .empty,
    next: u64 = 1,

    pub fn one(r: *Rec, name: []const u8, shape: []const usize) !u64 {
        try r.out.print(r.gpa, "one {s} {any}\n", .{ name, shape });
        r.next += 1;
        return r.next - 1;
    }

    pub fn stack(r: *Rec, parts: []const []const u8, rows: []const usize, k: usize, total_rows: usize) !u64 {
        try r.out.print(r.gpa, "stack {d}x{d}", .{ total_rows, k });
        for (parts, rows) |name, n| try r.out.print(r.gpa, " {s}:{d}", .{ name, n });
        try r.out.append(r.gpa, '\n');
        r.next += 1;
        return r.next - 1;
    }
};

/// Drama 3's layer_types: every fourth layer full attention, the rest linear (config.zig's test pins the same).
fn full(li: usize) bool {
    return li % 4 == 3;
}

fn record(a: std.mem.Allocator) !Rec {
    var r: Rec = .{ .gpa = a };
    errdefer r.out.deinit(a);
    const t = try names.top(&r);
    try std.testing.expectEqual([3]u64{ 1, 2, 3 }, [3]u64{ t.embed, t.cb_emb, t.final_norm });
    try r.out.print(a, "per-backend {s}\n", .{names.lm_head});
    for (0..cfg.layers) |li| {
        const Layer = struct { in_norm: u64, post_norm: u64, gdn: names.Gdn, attn: names.Attn, moe_in: u64, shared_down: u64 };
        var L: Layer = undefined;
        names.setDense(&L, try names.dense(&r, li, full(li)));
        try std.testing.expectEqual(r.next - 1, L.shared_down);
        var nb: [160]u8 = undefined;
        try r.out.print(a, "per-backend {s}\n", .{names.expert(&nb, li, 0, names.projections[0], "")});
    }
    for (0..cfg.tower_layers) |ti| {
        const T = try names.tower(&r, ti);
        try std.testing.expectEqual(r.next - 1, T.final_norm);
    }
    for (0..cfg.codebooks - 1) |hi| _ = try names.head(&r, hi);
    return r;
}

test "drama weight names: every tensor by name, shape and order, against the pinned list" {
    const a = std.testing.allocator;
    var r = try record(a);
    defer r.out.deinit(a);
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
