//! Qwen3.5's layer pattern through every backend's config entry point, on the repo's real configs and #1863's table.
const std = @import("std");
const metal27 = @import("src/families/qwen27/config.zig");
const metal2b = @import("src/families/qwen3_5/config.zig");
const hip = @import("src/families/qwen3_5/weights/config.zig");
const cuda = @import("src/families/qwen3_5/cuda_geometry.zig");
const t = std.testing;

const max_layers = 128;

/// A layer pattern (true: full attention) or the refusal that ended the reading.
const Outcome = union(enum) {
    pattern: struct { full: [max_layers]bool = @splat(false), len: usize },
    refused: anyerror,

    fn of(full: []const bool) Outcome {
        var o: Outcome = .{ .pattern = .{ .len = full.len } };
        @memcpy(o.pattern.full[0..full.len], full);
        return o;
    }

    fn same(x: Outcome, y: Outcome) bool {
        return x == .pattern and y == .pattern and std.mem.eql(bool, x.pattern.full[0..x.pattern.len], y.pattern.full[0..y.pattern.len]);
    }
};

const Backend = enum { metal27, metal2b, hip, cuda };

/// Each backend's config entry point on config.json's bytes, reduced to the pattern it admits.
fn entry(b: Backend, json: []const u8) Outcome {
    var full: [max_layers]bool = undefined;
    switch (b) {
        .metal27 => {
            const c = metal27.parse(t.allocator, json, null) catch |e| return .{ .refused = e };
            for (c.kinds[0..c.layers], full[0..c.layers]) |k, *f| f.* = k == .attention;
            return .of(full[0..c.layers]);
        },
        .metal2b => {
            _ = metal2b.parse(t.allocator, json) catch |e| return .{ .refused = e };
            for (full[0..metal2b.layers], 0..) |*f, i| f.* = !metal2b.linear(i);
            return .of(full[0..metal2b.layers]);
        },
        .hip => {
            var c = hip.Config.parse(t.allocator, json, null) catch |e| return .{ .refused = e };
            defer c.deinit();
            return .of(c.spec.full_layers[0..c.spec.n_layers]);
        },
        .cuda => {
            const g = cuda.parse(t.allocator, json) catch |e| return .{ .refused = e };
            for (g.linear[0..g.layers], full[0..g.layers]) |l, *f| f.* = !l;
            return .of(full[0..g.layers]);
        },
    }
}

/// The rules on main before #1863, each port verbatim from its backend: `layers` layers of `tower`.
const old = struct {
    fn eql(a: []const u8, b: []const u8) bool {
        return std.mem.eql(u8, a, b);
    }

    fn run(b: Backend, tower: std.json.ObjectMap, layers: usize) Outcome {
        var full: [max_layers]bool = undefined;
        const got = switch (b) {
            .metal27 => metal27Rule(full[0..layers], tower),
            .metal2b => metal2bRule(full[0..layers], tower),
            .hip => hipRule(full[0..layers], tower),
            .cuda => cudaRule(full[0..layers], tower),
        };
        got catch |e| return .{ .refused = e };
        return .of(full[0..layers]);
    }

    /// qwen27/config.zig:319: `layer_types` as written, else every interval-th layer (4 when absent).
    fn metal27Rule(full: []bool, tower: std.json.ObjectMap) !void {
        if (tower.get("layer_types")) |v| {
            if (v != .array or v.array.items.len != full.len) return error.BadConfig;
            for (v.array.items, full) |x, *f| {
                if (x != .string) return error.BadConfig;
                f.* = if (eql(x.string, "linear_attention")) false else if (eql(x.string, "full_attention")) true else return error.UnsupportedLayer;
            }
        } else {
            const interval: usize = if (tower.get("full_attention_interval")) |v| blk: {
                if (v != .integer or v.integer <= 0 or v.integer > std.math.maxInt(u32)) return error.BadConfig;
                break :blk @intCast(v.integer);
            } else 4;
            for (full, 0..) |*f, i| f.* = (i + 1) % interval == 0;
        }
    }

    /// qwen3_5/config.zig:107-121, the 2B: 24 layers, an interval of exactly 4 and the every-fourth list.
    fn metal2bRule(full: []bool, tower: std.json.ObjectMap) !void {
        if (full.len != metal2b.layers) return error.UnsupportedQwenGeometry;
        const n = tower.get("full_attention_interval") orelse return error.BadQwenConfig;
        if (n != .integer or n.integer < 1) return error.BadQwenConfig;
        if (n.integer != 4) return error.UnsupportedQwenGeometry;
        const types = tower.get("layer_types") orelse return error.BadQwenConfig;
        if (types != .array or types.array.items.len != full.len) return error.UnsupportedQwenGeometry;
        for (types.array.items, full, 0..) |kind, *f, i| {
            if (kind != .string or !eql(kind.string, if (metal2b.linear(i)) "linear_attention" else "full_attention")) return error.UnsupportedQwenGeometry;
            f.* = !metal2b.linear(i);
        }
    }

    /// qwen3_5/config.zig:379,389 and config_checks.zig:14: Python's `int(get(k, 4) or 0)`, 0 refused, then the list.
    fn hipRule(full: []bool, tower: std.json.ObjectMap) !void {
        const every: usize = if (tower.get("full_attention_interval")) |v| switch (v) {
            .null => 0,
            .bool => |x| if (x) return error.InvalidConfig else 0,
            .integer => |i| std.math.cast(usize, i) orelse return error.InvalidConfig,
            .float => |x| if (x >= 0 and x < 1e15) @intFromFloat(x) else return error.InvalidConfig,
            .string => |s| if (s.len == 0) 0 else return error.InvalidConfig,
            .array => |x| if (x.items.len == 0) 0 else return error.InvalidConfig,
            .object => |x| if (x.count() == 0) 0 else return error.InvalidConfig,
            else => return error.InvalidConfig,
        } else 4;
        if (every == 0 or full.len == 0 or full.len > max_layers) return error.InvalidConfig;
        const list = tower.get("layer_types") orelse .null;
        if (list == .null) {
            for (full, 0..) |*f, i| f.* = (i + 1) % every == 0;
            return;
        }
        if (list != .array or list.array.items.len != full.len) return error.InvalidLayerTypes;
        for (list.array.items, full) |item, *f| {
            const s = if (item == .string) item.string else "";
            f.* = if (eql(s, "full_attention")) true else if (eql(s, "linear_attention")) false else return error.InvalidLayerTypes;
        }
    }

    /// qwen3_5/cuda_geometry.zig:329: `layer_types` required, as written.
    fn cudaRule(full: []bool, tower: std.json.ObjectMap) !void {
        const types = tower.get("layer_types") orelse return error.UnsupportedQwenGeometry;
        if (types != .array or types.array.items.len != full.len) return error.UnsupportedQwenGeometry;
        for (types.array.items, full) |kind, *f| {
            if (kind != .string) return error.UnsupportedQwenGeometry;
            f.* = if (eql(kind.string, "linear_attention")) false else if (eql(kind.string, "full_attention")) true else return error.UnsupportedQwenGeometry;
        }
    }
};

/// A Qwen3.5 target config.json found in a source file: its bytes, text tower and layer count.
const Real = struct { file: []const u8, json: []const u8, tower: std.json.ObjectMap, layers: usize };

/// Every source holding real Qwen3.5 target configs, and how many each holds (whole literals, `++` joined).
const sources = .{
    .{ "src/families/qwen3_5/config.zig", 3 },
    .{ "src/families/qwen3_5/weights/config.zig", 1 },
    .{ "src/families/qwen3_5/weights/checkpoint_test.zig", 3 },
    .{ "src/families/qwen3_5/cuda_geometry.zig", 3 },
    .{ "src/families/qwen3_5/cuda_modelopt.zig", 1 },
    .{ "src/families/qwen3_5/cuda_quant_admission_test.zig", 1 },
    .{ "src/families/qwen3_5/pocket_shards_test.zig", 1 },
    .{ "src/families/qwen27/config_test.zig", 1 },
    .{ "src/families/qwen27/weights_test.zig", 1 },
    .{ "qwen35_formats_host.zig", 1 },
};

/// The configs of every source, each checked against its count.
fn reals(a: std.mem.Allocator) ![]const Real {
    var out: std.ArrayList(Real) = .empty;
    inline for (sources) |s| {
        const before = out.items.len;
        try scan(a, s[0], @embedFile(s[0]), &out);
        t.expectEqual(@as(usize, s[1]), out.items.len - before) catch |e| {
            std.debug.print("{s}: {d} configs\n", .{ s[0], out.items.len - before });
            return e;
        };
    }
    return out.items;
}

/// Every string literal of a Zig source (lines of a multiline one joined, `++` pieces joined) that is a config.
fn scan(a: std.mem.Allocator, file: []const u8, source: [:0]const u8, out: *std.ArrayList(Real)) !void {
    var tokens = std.zig.Tokenizer.init(source);
    var text: std.ArrayList(u8) = .empty;
    var open = false;
    var lines = false;
    while (true) {
        const tok = tokens.next();
        const raw = source[tok.loc.start..tok.loc.end];
        switch (tok.tag) {
            .multiline_string_literal_line => {
                if (lines) try text.append(a, '\n');
                try text.appendSlice(a, std.mem.trimEnd(u8, raw[2..], "\r\n"));
                open = true;
                lines = true;
                continue;
            },
            .string_literal => {
                try text.appendSlice(a, try std.zig.string_literal.parseAlloc(a, raw));
                open = true;
                lines = false;
                continue;
            },
            .plus_plus => if (open) {
                lines = false;
                continue;
            },
            else => {},
        }
        if (open) if (try config(a, file, text.items)) |r| try out.append(a, r);
        text = .empty;
        open = false;
        lines = false;
        if (tok.tag == .eof) return;
    }
}

/// `text` as a Qwen3.5 target config (model_type qwen3_5 or qwen3_5_moe with a layer count), else null.
fn config(a: std.mem.Allocator, file: []const u8, text: []const u8) !?Real {
    const doc = std.json.parseFromSliceLeaky(std.json.Value, a, text, .{}) catch return null;
    if (doc != .object) return null;
    const kind = doc.object.get("model_type") orelse return null;
    if (kind != .string or !(std.mem.eql(u8, kind.string, "qwen3_5") or std.mem.eql(u8, kind.string, "qwen3_5_moe"))) return null;
    const nested = doc.object.get("text_config");
    const tower = if (nested != null and nested.? == .object) nested.?.object else doc.object;
    const n = tower.get("num_hidden_layers") orelse return null;
    if (n != .integer or n.integer < 1 or n.integer > max_layers) return null;
    return .{ .file = file, .json = try a.dupe(u8, text), .tower = tower, .layers = @intCast(n.integer) };
}

test "qwen35 layers: real configs, every backend's entry point admits exactly the pattern its old rule reads" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    for (try reals(arena.allocator())) |r| for (std.enums.values(Backend)) |b| {
        const was = old.run(b, r.tower, r.layers);
        const now = entry(b, r.json);
        errdefer std.debug.print("{s} on {s}: old {any}, now {any}\n", .{ @tagName(b), r.file, was, now });
        if (now == .pattern) try t.expect(Outcome.same(was, now));
        if (was == .refused) try t.expect(now == .refused);
    };
}
