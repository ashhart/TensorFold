//! ModelOpt quantization_config checks on config.json alone (std only): NVFP4 in groups of 16, read by modelopt.zig.
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
    try nvfp4(q, why);
}

/// `admit` where tensor dtypes name each layer's format (Inkling 2 Mini): quant_algo NVFP4, config_groups as `admit`.
pub fn admitByTensors(gpa: std.mem.Allocator, config_json: []const u8, why: *Why) !void {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const root = try std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), config_json, .{});
    const q = block(root) orelse {
        why.set("config.json has no ModelOpt quantization_config", .{});
        return error.NotModelOpt;
    };
    if (q.get("config_groups") == null and std.ascii.eqlIgnoreCase(text(q.get("quant_algo")) orelse "", "NVFP4")) return;
    try nvfp4(q, why);
}

/// The quantization_config's NVFP4 check: quant_algo NVFP4, every config group 4-bit floats in groups of 16.
pub fn nvfp4(q: std.json.ObjectMap, why: *Why) !void {
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

pub fn block(root: std.json.Value) ?std.json.ObjectMap {
    const top = object(root) orelse return null;
    for ([_]?std.json.ObjectMap{ top, object(top.get("text_config")) }) |source| {
        const q = object((source orelse continue).get("quantization_config")) orelse continue;
        if (std.ascii.eqlIgnoreCase(text(q.get("quant_method")) orelse "", "modelopt")) return q;
    }
    return null;
}

pub fn object(v: ?std.json.Value) ?std.json.ObjectMap {
    const x = v orelse return null;
    return if (x == .object) x.object else null;
}

pub fn text(v: ?std.json.Value) ?[]const u8 {
    const x = v orelse return null;
    return if (x == .string) x.string else null;
}

pub fn int(v: ?std.json.Value) ?i64 {
    const x = v orelse return null;
    return if (x == .integer) x.integer else null;
}
