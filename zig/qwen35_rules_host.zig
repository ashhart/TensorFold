//! Qwen3.5's computed-part rows through every backend's config entry point: main's admissions, one word per refusal.
const std = @import("std");
const metal27 = @import("src/families/qwen27/config.zig");
const metal2b = @import("src/families/qwen3_5/config.zig");
const hip = @import("src/families/qwen3_5/weights/config.zig");
const cuda = @import("src/families/qwen3_5/cuda_geometry.zig");
const rules = @import("qwen35_rules");
const layers_host = @import("qwen35_layers_host.zig");
const t = std.testing;

/// Each backend's config entry point on config.json's bytes: admitted, or its refusal.
fn entry(b: rules.Backend, json: []const u8) anyerror!void {
    switch (b) {
        .metal27 => _ = try metal27.parse(t.allocator, json, null),
        .metal2b => _ = try metal2b.parse(t.allocator, json),
        .hip => {
            var c = try hip.Config.parse(t.allocator, json, null);
            c.deinit();
        },
        .cuda => _ = try cuda.parse(t.allocator, json),
    }
}

/// The rules on main before the table, each port verbatim from its backend, on a text tower.
const old = struct {
    fn str(v: std.json.Value, want: []const u8) bool {
        return v == .string and std.mem.eql(u8, v.string, want);
    }

    fn run(b: rules.Backend, tw: std.json.ObjectMap) anyerror!void {
        return switch (b) {
            .cuda => cudaRule(tw),
            .metal27 => metal27Rule(tw),
            .metal2b => metal2bRule(tw),
            .hip => hipRule(tw),
        };
    }

    /// qwen3_5/cuda_geometry.zig:274-276: `is`, three keys required as JSON values.
    fn cudaRule(tw: std.json.ObjectMap) !void {
        const bias = tw.get("attention_bias") orelse return error.UnsupportedQwenConfig;
        if (!(bias == .bool and !bias.bool)) return error.UnsupportedQwenConfig;
        const gate = tw.get("attn_output_gate") orelse return error.UnsupportedQwenConfig;
        if (!(gate == .bool and gate.bool)) return error.UnsupportedQwenConfig;
        const act = tw.get("hidden_act") orelse return error.UnsupportedQwenConfig;
        if (!str(act, "silu")) return error.UnsupportedQwenConfig;
    }

    /// qwen27/config.zig:292-295: the gate a boolean when present; three strings when present.
    fn metal27Rule(tw: std.json.ObjectMap) !void {
        if (tw.get("attn_output_gate")) |v| if (v != .bool or !v.bool) return error.UnsupportedAttentionGate;
        const strings = .{ .{ "hidden_act", "silu" }, .{ "output_gate_type", "swish" }, .{ "linear_attn_state_dtype", "float32" } };
        inline for (strings) |e| if (tw.get(e[0])) |v| if (!str(v, e[1])) return error.UnsupportedActivation;
    }

    /// qwen3_5/config.zig:104-107: four keys required (absent BadQwenConfig, other UnsupportedQwenConfig).
    fn metal2bRule(tw: std.json.ObjectMap) !void {
        const bias = tw.get("attention_bias") orelse return error.BadQwenConfig;
        if (!(bias == .bool and !bias.bool)) return error.UnsupportedQwenConfig;
        const gate = tw.get("attn_output_gate") orelse return error.BadQwenConfig;
        if (!(gate == .bool and gate.bool)) return error.UnsupportedQwenConfig;
        const act = tw.get("hidden_act") orelse return error.BadQwenConfig;
        if (!str(act, "silu")) return error.UnsupportedQwenConfig;
        const ssm = tw.get("mamba_ssm_dtype") orelse return error.BadQwenConfig;
        if (!str(ssm, "float32")) return error.UnsupportedQwenConfig;
    }

    /// qwen3_5/config_checks.zig:55-64 (`computes`): six keys when present, 0 and 1 as flags.
    fn hipRule(tw: std.json.ObjectMap) !void {
        const flag = struct {
            fn f(v: std.json.Value, want: bool) bool {
                return (v == .bool and v.bool == want) or (v == .integer and v.integer == @intFromBool(want));
            }
        }.f;
        if (tw.get("hidden_act")) |v| if (!str(v, "silu")) return error.UnsupportedActivation;
        if (tw.get("output_gate_type")) |v| if (!str(v, "swish")) return error.UnsupportedAttentionGate;
        if (tw.get("attn_output_gate")) |v| if (!flag(v, true)) return error.UnsupportedAttentionGate;
        if (tw.get("attention_bias")) |v| if (!flag(v, false)) return error.UnsupportedAttentionGate;
        inline for (.{ "linear_attn_state_dtype", "mamba_ssm_dtype" }) |key| {
            if (tw.get(key)) |v| if (!str(v, "float32")) return error.UnsupportedStateType;
        }
    }
};

fn towerOf(doc: *std.json.Value) *std.json.ObjectMap {
    return if (doc.object.getPtr("text_config")) |v| &v.object else &doc.object;
}

/// `base` with `key` set to `value` (JSON), or removed when `value` is null.
fn variant(a: std.mem.Allocator, base: []const u8, key: []const u8, value: ?[]const u8) ![]const u8 {
    var doc = try std.json.parseFromSliceLeaky(std.json.Value, a, base, .{});
    const tw = towerOf(&doc);
    if (value) |v| try tw.put(a, key, try std.json.parseFromSliceLeaky(std.json.Value, a, v, .{})) else _ = tw.orderedRemove(key);
    return std.json.Stringify.valueAlloc(a, doc, .{});
}

/// The values each row is tried with: wanted, another, null, absent, and 0 / 1 / a string for the booleans.
const values = [_]?[]const u8{ "\"silu\"", "\"swish\"", "\"float32\"", "\"other\"", "true", "false", "0", "1", "null", null };

test "qwen35 rules: real configs, each entry point admits what main admitted, and by the table" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    var admitted: usize = 0;
    for (try layers_host.reals(arena.allocator())) |r| for (std.enums.values(rules.Backend)) |b| {
        const was = if (old.run(b, r.tower)) true else |_| false;
        const now = if (rules.check(r.tower, b)) true else |_| false;
        errdefer std.debug.print("{s} on {s}: old {}, now {}\n", .{ @tagName(b), r.file, was, now });
        try t.expectEqual(was, now);
        admitted += @intFromBool(now);
    };
    try t.expectEqual(@as(usize, 44), admitted); // (config, backend) pairs whose tower passes the rows
}

test "qwen35 rules: every row and value through every entry point admits as main did, refused by the row's word" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var cases: usize = 0;
    var words: usize = 0;
    for (try layers_host.bases(a)) |base| for (std.enums.values(rules.Backend)) |b| {
        entry(b, base) catch continue; // this base is another backend's model
        for (rules.rows) |row| for (values) |value| {
            const json = try variant(a, base, row.key, value);
            var doc = try std.json.parseFromSliceLeaky(std.json.Value, a, json, .{});
            const was = old.run(b, towerOf(&doc).*);
            const now = entry(b, json);
            errdefer std.debug.print("{s}, {s} = {?s}: old {any}, now {any}\n", .{ @tagName(b), row.key, value, was, now });
            try t.expectEqual(std.meta.isError(was), std.meta.isError(now));
            if (now) |_| {} else |e| {
                try t.expectEqual(@as(anyerror, row.refusal), e);
                words += @intFromBool(if (was) |_| false else |w| w != e);
            }
            cases += 1;
        };
    };
    // 2B: Metal's 2B, HIP, CUDA; 27B widths: Metal's 27B, HIP, CUDA; six rows of ten values each
    try t.expectEqual(@as(usize, 6 * 6 * values.len), cases);
    try t.expect(words > 0); // the refusals whose word the table changed (CUDA's, the 2B's, two of Metal's 27B)
}

/// `base` with `tie_word_embeddings` set (JSON) at the root and in `text_config`.
fn ties(a: std.mem.Allocator, base: []const u8, root: []const u8, tower: []const u8) ![]const u8 {
    var doc = try std.json.parseFromSliceLeaky(std.json.Value, a, base, .{});
    const key = "tie_word_embeddings";
    try doc.object.put(a, key, try std.json.parseFromSliceLeaky(std.json.Value, a, root, .{}));
    try doc.object.getPtr("text_config").?.object.put(a, key, try std.json.parseFromSliceLeaky(std.json.Value, a, tower, .{}));
    return std.json.Stringify.valueAlloc(a, doc, .{});
}

test "qwen35 rules: CUDA's parse refuses a root and text_config tie that disagree, and a non-flag root" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const base = (try layers_host.bases(a))[1];
    try t.expect(!(try cuda.parse(t.allocator, try ties(a, base, "false", "false"))).tied);
    try t.expectError(error.BadTiedHead, cuda.parse(t.allocator, try ties(a, base, "true", "false")));
    try t.expectError(error.BadTiedHead, cuda.parse(t.allocator, try ties(a, base, "false", "true")));
    try t.expectError(error.BadTiedHead, cuda.parse(t.allocator, try ties(a, base, "\"yes\"", "false")));
}

test "qwen35 rules: Metal 27B's parse refuses a nested tie disagreement and a tie of 1 as BadTiedHead" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const base = (try layers_host.bases(a))[1];
    _ = try metal27.parse(t.allocator, try ties(a, base, "false", "false"), null);
    try t.expectError(error.BadTiedHead, metal27.parse(t.allocator, try ties(a, base, "true", "false"), null));
    try t.expectError(error.BadTiedHead, metal27.parse(t.allocator, try ties(a, base, "false", "true"), null));
    try t.expectError(error.BadTiedHead, metal27.parse(t.allocator, try ties(a, base, "false", "1"), null));
}
