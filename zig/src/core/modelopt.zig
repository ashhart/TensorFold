//! ModelOpt NVFP4 config check (groups of 16); tensors say which linears are NVFP4, not ignore (423acf37's BF16 MTP).

const std = @import("std");

/// The NVFP4 group size ModelOpt writes for this format.
pub const nvfp4_group = 16;

/// Why the check refused the checkpoint, kept for the caller's log line (tests read it instead).
pub const Why = struct {
    buf: [320]u8 = undefined,
    len: usize = 0,

    /// Keep the reason, cut short if it is long.
    pub fn set(self: *Why, comptime fmt: []const u8, args: anytype) void {
        var w: std.Io.Writer = .fixed(&self.buf);
        w.print(fmt, args) catch {};
        self.len = w.end;
    }

    pub fn text(self: *const Why) []const u8 {
        return self.buf[0..self.len];
    }
};

/// config.json's quantization_config (top level, else text_config's) checked as ModelOpt NVFP4.
pub fn admit(gpa: std.mem.Allocator, config_json: []const u8, why: *Why) !void {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const root = try std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), config_json, .{});
    const q = block(root) orelse {
        why.set("config.json has no ModelOpt quantization_config", .{});
        return error.NotModelOpt;
    };
    const algo = text(q.get("quant_algo")) orelse "";
    if (!std.ascii.eqlIgnoreCase(algo, "NVFP4")) {
        const named = if (algo.len > 0) algo else "without quant_algo";
        why.set("ModelOpt {s}: this engine reads ModelOpt NVFP4 (nvidia/GLM-5.3-Flash-NVFP4's layout)", .{named});
        return error.UnsupportedQuant;
    }
    const groups = object(q.get("config_groups")) orelse {
        why.set("ModelOpt NVFP4 without config_groups", .{});
        return error.UnsupportedQuant;
    };
    var it = groups.iterator();
    while (it.next()) |g| {
        const w = if (object(g.value_ptr.*)) |o| object(o.get("weights")) else null;
        const bits = if (w) |o| int(o.get("num_bits")) else null;
        const size = if (w) |o| int(o.get("group_size")) else null;
        const kind = if (w) |o| text(o.get("type")) orelse "" else "";
        if (bits != 4 or size != nvfp4_group or !std.mem.eql(u8, kind, "float")) {
            why.set(
                "ModelOpt NVFP4 group {s}: weights {?d}-bit {s} in groups of {?d}, " ++
                    "expected 4-bit float in groups of 16",
                .{ g.key_ptr.*, bits, kind, size },
            );
            return error.UnsupportedQuant;
        }
    }
}

fn block(root: std.json.Value) ?std.json.ObjectMap {
    const top = object(root) orelse return null;
    for ([_]?std.json.ObjectMap{ top, object(top.get("text_config")) }) |source| {
        const q = object((source orelse continue).get("quantization_config")) orelse continue;
        if (std.ascii.eqlIgnoreCase(text(q.get("quant_method")) orelse "", "modelopt")) return q;
    }
    return null;
}

fn object(v: ?std.json.Value) ?std.json.ObjectMap {
    const x = v orelse return null;
    return if (x == .object) x.object else null;
}

fn text(v: ?std.json.Value) ?[]const u8 {
    const x = v orelse return null;
    return if (x == .string) x.string else null;
}

fn int(v: ?std.json.Value) ?i64 {
    const x = v orelse return null;
    return if (x == .integer) x.integer else null;
}

test "nvidia/GLM-5.3-Flash-NVFP4's config is admitted" {
    var why: Why = .{};
    try admit(std.testing.allocator, @embedFile("fixtures/glm53_flash_nvfp4/config.json"), &why);
}

/// admit() must refuse `json` with `err`, its reason naming `word`.
fn expectRefused(json: []const u8, err: anyerror, word: []const u8) !void {
    var why: Why = .{};
    try std.testing.expectError(err, admit(std.testing.allocator, json, &why));
    if (std.mem.indexOf(u8, why.text(), word) == null) {
        std.debug.print("refusal \"{s}\" does not name \"{s}\"\n", .{ why.text(), word });
        return error.TestUnexpectedResult;
    }
}

test "other ModelOpt formats, and configs without ModelOpt, are refused by name" {
    try expectRefused("{\"text_config\": {}}", error.NotModelOpt, "no ModelOpt");
    try expectRefused("{\"quantization_config\": {\"quant_method\": \"exl3\"}}", error.NotModelOpt, "no ModelOpt");
    try expectRefused(
        "{\"quantization_config\": {\"quant_method\": \"modelopt\", \"quant_algo\": \"FP8\"}}",
        error.UnsupportedQuant,
        "ModelOpt FP8",
    );
    // MIXED_PRECISION (the AXL re-pack) is refused for now
    try expectRefused(
        "{\"quantization_config\": {\"quant_method\": \"modelopt\", \"quant_algo\": \"MIXED_PRECISION\"}}",
        error.UnsupportedQuant,
        "MIXED_PRECISION",
    );
    try expectRefused(
        \\{"quantization_config": {"quant_method": "modelopt", "quant_algo": "NVFP4", "config_groups":
        \\ {"g": {"weights": {"num_bits": 4, "type": "float", "group_size": 32}}}}}
    , error.UnsupportedQuant, "groups of 32");
    var why: Why = .{};
    try admit(std.testing.allocator,
        \\{"text_config": {"quantization_config": {"quant_method": "modelopt", "quant_algo": "nvfp4", "config_groups":
        \\ {"g": {"weights": {"num_bits": 4, "type": "float", "group_size": 16}}}}}}
    , &why);
}
