//! A decision checkpoint's `modeling_overrides` (config.json) and `readout.json`: known values served, others refused.
const std = @import("std");

pub const Attention = enum { causal, block_bidirectional };

/// A `slot_lm` decision model: answers are the LM head's logits of the answer letters at the answer row.
pub const Readout = struct { attention: Attention, slots: u32 };

pub const Error = error{ Refused, OutOfMemory };

/// The checkpoint folder's readout; null: neither file names one (a plain chat checkpoint, served as today).
pub fn read(a: std.mem.Allocator, io: std.Io, dir: []const u8, problem: *[]const u8) Error!?Readout {
    const config = try file(a, io, dir, "config.json");
    defer if (config) |t| a.free(t);
    const readout = try file(a, io, dir, "readout.json");
    defer if (readout) |t| a.free(t);
    return parse(a, config, readout, problem);
}

fn file(a: std.mem.Allocator, io: std.Io, dir: []const u8, name: []const u8) Error!?[]u8 {
    const path = try std.fs.path.join(a, &.{ dir, name });
    defer a.free(path);
    return std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(16 << 20)) catch |err| switch (err) {
        error.FileNotFound => null,
        error.OutOfMemory => error.OutOfMemory,
        else => null,
    };
}

/// `config` and `readout` are the two files' texts (null: absent); `problem` says why a refusal refused.
pub fn parse(a: std.mem.Allocator, config: ?[]const u8, readout: ?[]const u8, problem: *[]const u8) Error!?Readout {
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    const x = arena.allocator();
    const w: Words = .{ .a = a, .problem = problem };
    const overrides: ?std.json.ObjectMap = if (config) |t| blk: {
        const root = try object(x, t, "config.json", w);
        const v = root.get("modeling_overrides") orelse break :blk null;
        if (v == .null) break :blk null;
        if (v != .object) return w.say("config.json: modeling_overrides must be an object", .{});
        break :blk v.object;
    } else null;
    const fields: ?std.json.ObjectMap = if (readout) |t| try object(x, t, "readout.json", w) else null;
    if (overrides == null and fields == null) return null;
    var attention: ?Attention = null;
    var name: ?[]const u8 = null;
    var slots: u32 = 256;
    if (overrides) |o| {
        for (o.keys(), o.values()) |key, v| {
            if (std.mem.eql(u8, key, "attention")) {
                attention = try masks(v, w);
            } else if (std.mem.eql(u8, key, "readout")) {
                name = try string(v, "modeling_overrides.readout", w);
            } else if (std.mem.eql(u8, key, "n_slots")) {
                slots = try count(v, w);
            } else if (!std.mem.eql(u8, key, "head_dim")) return w.key("config.json modeling_overrides", key);
        }
    }
    if (fields) |f| {
        for (f.keys(), f.values()) |key, v| {
            if (std.mem.eql(u8, key, "readout")) {
                const said = try string(v, "readout.json readout", w);
                if (name) |n| if (!std.mem.eql(u8, n, said)) return w.say("readout.json and config.json modeling_overrides name different readouts", .{});
                name = said;
            } else if (std.mem.eql(u8, key, "attention")) {
                // qev's names: causal, or noncausal for block_bidirectional
                const said = try string(v, "readout.json attention", w);
                const mask: Attention = if (std.mem.eql(u8, said, "causal")) .causal else if (std.mem.eql(u8, said, "noncausal")) .block_bidirectional else return w.value("readout.json attention", said);
                if (attention) |have| if (have != mask) return w.say("readout.json attention disagrees with config.json modeling_overrides.attention.full_attention", .{});
                attention = mask;
            } else if (std.mem.eql(u8, key, "n_slots")) {
                slots = try count(v, w);
            } else if (!std.mem.eql(u8, key, "head_dim")) return w.key("readout.json", key);
        }
    }
    const readout_name = name orelse return w.say("config.json has modeling_overrides but names no readout (readout.json or modeling_overrides.readout): refused rather than served causally", .{});
    if (!std.mem.eql(u8, readout_name, "slot_lm")) return w.value("readout", readout_name);
    return .{ .attention = attention orelse .causal, .slots = slots };
}

fn object(x: std.mem.Allocator, text: []const u8, what: []const u8, w: Words) Error!std.json.ObjectMap {
    const parsed = std.json.parseFromSliceLeaky(std.json.Value, x, text, .{}) catch return w.say("{s} is not valid JSON", .{what});
    if (parsed != .object) return w.say("{s} must be a JSON object", .{what});
    return parsed.object;
}

fn masks(v: std.json.Value, w: Words) Error!Attention {
    if (v != .object) return w.say("modeling_overrides.attention must be an object", .{});
    var full: Attention = .causal;
    for (v.object.keys(), v.object.values()) |key, mask| {
        const said = try string(mask, "modeling_overrides.attention value", w);
        if (std.mem.eql(u8, key, "full_attention")) {
            full = std.meta.stringToEnum(Attention, said) orelse return w.value("modeling_overrides.attention.full_attention", said);
        } else if (std.mem.eql(u8, key, "linear_attention")) {
            if (!std.mem.eql(u8, said, "causal")) return w.value("modeling_overrides.attention.linear_attention", said);
        } else return w.key("modeling_overrides.attention", key);
    }
    return full;
}

fn string(v: std.json.Value, what: []const u8, w: Words) Error![]const u8 {
    if (v != .string) return w.say("{s} must be a string", .{what});
    return v.string;
}

fn count(v: std.json.Value, w: Words) Error!u32 {
    if (v != .integer or v.integer < 1 or v.integer > 256) return w.say("n_slots must be an integer from 1 to 256 (prompt qev-labels-v2 has 256 answer slots)", .{});
    return @intCast(v.integer);
}

/// Refusal words in the caller's allocator, so they outlive the parse.
const Words = struct {
    a: std.mem.Allocator,
    problem: *[]const u8,
    fn say(w: Words, comptime fmt: []const u8, args: anytype) Error {
        w.problem.* = std.fmt.allocPrint(w.a, fmt, args) catch return error.OutOfMemory;
        return error.Refused;
    }
    fn key(w: Words, where: []const u8, name: []const u8) Error {
        return w.say("{s}: unknown setting {s} (refused rather than ignored)", .{ where, name });
    }
    fn value(w: Words, where: []const u8, said: []const u8) Error {
        return w.say("{s}: unsupported value {s} (this engine serves causal or block_bidirectional full attention, causal linear attention and the slot_lm readout)", .{ where, said });
    }
};

const elephant_config =
    \\{"model_type":"qwen3_5","text_config":{"hidden_size":4096},"modeling_overrides":{"attention":{"full_attention":"block_bidirectional","linear_attention":"causal"},"readout":"slot_lm","head_dim":256,"n_slots":256}}
;
const elephant_readout =
    \\{"readout":"slot_lm","head_dim":256,"n_slots":256}
;

fn refusal(config: ?[]const u8, readout: ?[]const u8, words: []const u8) !void {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    var problem: []const u8 = "";
    try std.testing.expectError(error.Refused, parse(arena.allocator(), config, readout, &problem));
    if (std.mem.indexOf(u8, problem, words) == null) {
        std.debug.print("refusal said: {s}\n", .{problem});
        return error.WrongRefusal;
    }
}

test "readout: a checkpoint without overrides or readout.json is a plain chat model" {
    var problem: []const u8 = "";
    try std.testing.expectEqual(@as(?Readout, null), try parse(std.testing.allocator, "{\"model_type\":\"qwen3_5\"}", null, &problem));
    try std.testing.expectEqual(@as(?Readout, null), try parse(std.testing.allocator, null, null, &problem));
}

test "readout: elephant-1's overrides and readout.json give a block-bidirectional slot_lm decision model" {
    var problem: []const u8 = "";
    const got = (try parse(std.testing.allocator, elephant_config, elephant_readout, &problem)) orelse return error.NoReadout;
    try std.testing.expectEqual(Attention.block_bidirectional, got.attention);
    try std.testing.expectEqual(@as(u32, 256), got.slots);
}

test "readout: an unknown full_attention mask is refused by name" {
    try refusal("{\"modeling_overrides\":{\"attention\":{\"full_attention\":\"sliding\"},\"readout\":\"slot_lm\"}}", elephant_readout, "sliding");
}

test "readout: linear attention other than causal is refused" {
    try refusal("{\"modeling_overrides\":{\"attention\":{\"full_attention\":\"causal\",\"linear_attention\":\"bidirectional\"}}}", elephant_readout, "linear_attention");
}

test "readout: a readout other than slot_lm is refused" {
    try refusal(null, "{\"readout\":\"joint_head\"}", "joint_head");
    try refusal("{\"modeling_overrides\":{\"readout\":\"hidden\"}}", null, "hidden");
}

test "readout: an unknown override key is refused rather than ignored" {
    try refusal("{\"modeling_overrides\":{\"attention\":{\"full_attention\":\"causal\"},\"readout\":\"slot_lm\",\"pooling\":\"mean\"}}", null, "pooling");
}

test "readout: readout.json and the overrides disagreeing on attention is refused" {
    try refusal(elephant_config, "{\"readout\":\"slot_lm\",\"attention\":\"causal\"}", "attention");
}

test "readout: overrides that name no readout are refused, so the model is never served causally by accident" {
    try refusal("{\"modeling_overrides\":{\"attention\":{\"full_attention\":\"block_bidirectional\"}}}", null, "readout");
}

test "readout: slot counts outside 1 to 256 are refused" {
    try refusal(null, "{\"readout\":\"slot_lm\",\"n_slots\":300}", "n_slots");
}
