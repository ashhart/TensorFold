//! `joint_head_config.json`: the JointSchemaHead constructor's arguments, every field read and checked.
const std = @import("std");

pub const Config = struct {
    hidden_size: u32,
    width: u32,
    routing_layers: u32,
    layers: u32,
    heads: u32,
    feedforward: u32,

    /// Cloudflare/clef-flash's head: the only shape the device head serves.
    pub const clef_flash: Config = .{ .hidden_size = 4096, .width = 1024, .routing_layers = 2, .layers = 4, .heads = 16, .feedforward = 4096 };

    pub fn headDim(c: Config) u32 {
        return c.width / c.heads;
    }

    pub fn eql(c: Config, o: Config) bool {
        return std.meta.eql(c, o);
    }
};

/// A refusal's reason, naming the field or tensor.
pub const Problem = struct {
    buf: [160]u8 = undefined,
    text: []const u8 = "",

    pub fn set(p: *Problem, comptime fmt: []const u8, args: anytype) void {
        p.text = std.fmt.bufPrint(&p.buf, fmt, args) catch fmt;
    }
};

const fields = [_][]const u8{ "hidden_size", "width", "routing_layers", "layers", "heads", "feedforward" };

/// The config's fields; anything unknown, missing, not a positive integer, or heads not dividing width, refused.
pub fn parse(gpa: std.mem.Allocator, text: []const u8, problem: *Problem) !Config {
    const doc = std.json.parseFromSlice(std.json.Value, gpa, text, .{}) catch {
        problem.set("joint_head_config.json is not JSON", .{});
        return error.BadConfig;
    };
    defer doc.deinit();
    if (doc.value != .object) {
        problem.set("joint_head_config.json is not an object", .{});
        return error.BadConfig;
    }
    var it = doc.value.object.iterator();
    while (it.next()) |kv| {
        const name = kv.key_ptr.*;
        const known = for (fields) |f| {
            if (std.mem.eql(u8, f, name)) break true;
        } else false;
        if (known) continue;
        // JointSchemaHead also takes dropout, which eval() leaves unused: a probability in [0, 1]
        if (std.mem.eql(u8, name, "dropout") and probability(kv.value_ptr.*)) continue;
        problem.set("joint_head_config.json: unknown or bad field \"{s}\"", .{name});
        return error.BadConfig;
    }
    var c: Config = undefined;
    inline for (fields) |f| {
        const v = doc.value.object.get(f) orelse {
            problem.set("joint_head_config.json: missing \"{s}\"", .{f});
            return error.BadConfig;
        };
        if (v != .integer or v.integer <= 0 or v.integer > 1 << 20) {
            problem.set("joint_head_config.json: \"{s}\" must be an integer from 1 to 2^20", .{f});
            return error.BadConfig;
        }
        @field(c, f) = @intCast(v.integer);
    }
    if (c.width % c.heads != 0) {
        problem.set("joint_head_config.json: width {d} is not divisible by heads {d}", .{ c.width, c.heads });
        return error.BadConfig;
    }
    return c;
}

fn probability(v: std.json.Value) bool {
    const x: f64 = switch (v) {
        .integer => |i| @floatFromInt(i),
        .float => |f| f,
        else => return false,
    };
    return x >= 0 and x <= 1;
}

test "joint schema config: Clef-Flash's fields, and each refusal names its field" {
    const gpa = std.testing.allocator;
    var p: Problem = .{};
    const release = "{\"hidden_size\": 4096, \"width\": 1024, \"routing_layers\": 2, \"layers\": 4, \"heads\": 16, \"feedforward\": 4096}";
    const c = try parse(gpa, release, &p);
    try std.testing.expect(c.eql(Config.clef_flash));
    try std.testing.expectEqual(@as(u32, 64), c.headDim());
    const tiny = "{\"hidden_size\":128,\"width\":32,\"routing_layers\":2,\"layers\":4,\"heads\":4,\"feedforward\":128,\"dropout\":0.1}";
    try std.testing.expect(!(try parse(gpa, tiny, &p)).eql(Config.clef_flash));
    const bad = [_][2][]const u8{
        .{ "{\"hidden_size\":4096,\"width\":1024,\"routing_layers\":2,\"layers\":4,\"heads\":16}", "missing \"feedforward\"" },
        .{ "{\"hidden_size\":4096,\"width\":1024,\"routing_layers\":2,\"layers\":4,\"heads\":16,\"feedforward\":4096,\"bias\":1}", "\"bias\"" },
        .{ "{\"hidden_size\":4096,\"width\":1000,\"routing_layers\":2,\"layers\":4,\"heads\":16,\"feedforward\":4096}", "not divisible" },
        .{ "{\"hidden_size\":4096.0,\"width\":1024,\"routing_layers\":2,\"layers\":4,\"heads\":16,\"feedforward\":4096}", "\"hidden_size\" must" },
        .{ "{\"hidden_size\":4096,\"width\":1024,\"routing_layers\":0,\"layers\":4,\"heads\":16,\"feedforward\":4096}", "\"routing_layers\" must" },
        .{ "{\"hidden_size\":4096,\"width\":1024,\"routing_layers\":2,\"layers\":true,\"heads\":16,\"feedforward\":4096}", "\"layers\" must" },
        .{ "{\"hidden_size\":4096,\"width\":1024,\"routing_layers\":2,\"layers\":4,\"heads\":16,\"feedforward\":4096,\"dropout\":2}", "\"dropout\"" },
        .{ "[1]", "not an object" },
        .{ "{", "not JSON" },
    };
    for (bad) |b| {
        try std.testing.expectError(error.BadConfig, parse(gpa, b[0], &p));
        if (std.mem.indexOf(u8, p.text, b[1]) == null) {
            std.debug.print("refusal \"{s}\" lacks \"{s}\"\n", .{ p.text, b[1] });
            return error.TestUnexpectedResult;
        }
    }
}
