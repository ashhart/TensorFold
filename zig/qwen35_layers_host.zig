//! Qwen3.5's layer pattern through every backend's config entry point, on the repo's real configs and #1863's table.
const std = @import("std");
const metal27 = @import("src/families/qwen27/config.zig");
const metal2b = @import("src/families/qwen3_5/config.zig");
const hip = @import("src/families/qwen3_5/weights/config.zig");
const cuda = @import("src/families/qwen3_5/cuda_geometry.zig");
const xpu = @import("src/xpu/qwen27_config.zig");
const layer_pattern = @import("qwen35_layers");
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

    /// Every `n`-th of `layers` layers full attention.
    fn every(n: usize, layers: usize) Outcome {
        var o: Outcome = .{ .pattern = .{ .len = layers } };
        for (o.pattern.full[0..layers], 0..) |*f, i| f.* = layer_pattern.every(n, i);
        return o;
    }

    /// The shared reader's pattern for `layers` layers of `tower`.
    fn shared(tower: std.json.ObjectMap, layers: usize) Outcome {
        var full: [max_layers]bool = undefined;
        layer_pattern.read(full[0..layers], tower) catch |e| return .{ .refused = e };
        return .of(full[0..layers]);
    }

    /// Whether this is a refusal of the layer keys themselves, by the shared reader.
    fn byReader(o: Outcome) bool {
        if (o != .refused) return false;
        inline for (@typeInfo(layer_pattern.Error).error_set.error_names.?) |name| if (o.refused == @field(anyerror, name)) return true;
        return false;
    }
};

const Backend = enum { metal27, metal2b, hip, cuda, xpu };

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
        .xpu => {
            xpu.check(t.allocator, json) catch |e| return .{ .refused = e };
            return .every(4, xpu.layers);
        },
    }
}

/// The rules on main before #1863, each port verbatim from its backend: `layers` layers of `tower`.
const old = struct {
    fn eql(a: []const u8, b: []const u8) bool {
        return std.mem.eql(u8, a, b);
    }

    fn run(b: Backend, tower: std.json.ObjectMap, layers: usize) Outcome {
        if (b == .xpu) return .every(4, 64); // xpu/qwen27.zig:74: `(i + 1) % 4` over 64 layers, config.json unread
        var full: [max_layers]bool = undefined;
        const got = switch (b) {
            .metal27 => metal27Rule(full[0..layers], tower),
            .metal2b => metal2bRule(full[0..layers], tower),
            .hip => hipRule(full[0..layers], tower),
            .cuda => cudaRule(full[0..layers], tower),
            .xpu => unreachable,
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
    .{ "src/families/qwen3_5/weights/checkpoint_test.zig", 4 }, // with LightOnOCR-3's HF BF16 layout
    .{ "src/families/qwen3_5/cuda_geometry.zig", 3 },
    .{ "src/families/qwen3_5/cuda_modelopt.zig", 1 },
    .{ "src/families/qwen3_5/cuda_quant_admission_test.zig", 1 },
    .{ "src/families/qwen3_5/pocket_shards_test.zig", 1 },
    .{ "src/families/qwen27/config_test.zig", 2 }, // with #1125's Ornith-1.5-122B (qwen3_5_moe)
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

test "qwen35 layers: real configs, each entry point admits the pattern its old rule read, by the shared reader" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    var admitted: usize = 0;
    var relaxed: usize = 0;
    for (try reals(arena.allocator())) |r| {
        const read = Outcome.shared(r.tower, r.layers);
        for (std.enums.values(Backend)) |b| {
            const was = old.run(b, r.tower, r.layers);
            const now = entry(b, r.json);
            errdefer std.debug.print("{s} on {s}: old {any}, now {any}, shared {any}\n", .{ @tagName(b), r.file, was, now, read });
            // what an entry point admits is the shared reader's pattern, and its old rule's when that rule read one
            if (now == .pattern) try t.expect(Outcome.same(read, now) and (was == .refused or Outcome.same(was, now)));
            // a config an old rule read reads the same, and no entry point now refuses it for its layer keys
            if (was == .pattern and b != .xpu) try t.expect(Outcome.same(was, read) and !now.byReader());
            admitted += @intFromBool(now == .pattern);
            relaxed += @intFromBool(now == .pattern and was == .refused);
        }
    }
    try t.expectEqual(@as(usize, 21), admitted); // (config, backend) pairs, with #2046's and LightOnOCR-3's HIP BF16 and #1125's Ornith
    try t.expectEqual(@as(usize, 0), relaxed); // none of the repo's configs is one the table newly admits
}

/// `base` with `layers` layers (when given), an end id, and the layer keys of one row of the table.
fn variant(a: std.mem.Allocator, base: []const u8, layers: ?usize, types: ?usize, null_types: bool, interval: ?[]const u8) ![]const u8 {
    var doc = try std.json.parseFromSliceLeaky(std.json.Value, a, base, .{});
    const tower = if (doc.object.getPtr("text_config")) |v| &v.object else &doc.object;
    if (layers) |n| try tower.put(a, "num_hidden_layers", .{ .integer = @intCast(n) });
    if (tower.get("eos_token_id") == null and doc.object.get("eos_token_id") == null) try tower.put(a, "eos_token_id", .{ .integer = 248044 });
    const n: usize = @intCast(tower.get("num_hidden_layers").?.integer);
    _ = tower.orderedRemove("layer_types");
    _ = tower.orderedRemove("full_attention_interval");
    if (null_types) try tower.put(a, "layer_types", .null);
    if (types) |every| {
        var list: std.json.Array = .init(a);
        for (0..n) |i| try list.append(.{ .string = if (layer_pattern.every(every, i)) "full_attention" else "linear_attention" });
        try tower.put(a, "layer_types", .{ .array = list });
    }
    if (interval) |v| try tower.put(a, "full_attention_interval", try std.json.parseFromSliceLeaky(std.json.Value, a, v, .{}));
    return std.json.Stringify.valueAlloc(a, doc, .{});
}

/// The 2B's config.json, and the 27B widths' config at the 27B's 64 layers: between them every entry point admits one.
fn bases(a: std.mem.Allocator) ![2][]const u8 {
    var out: [2][]const u8 = undefined;
    for (try reals(a)) |r| {
        if (std.mem.endsWith(u8, r.file, "qwen3_5/config.zig") and r.layers == metal2b.layers) out[0] = r.json;
        if (std.mem.endsWith(u8, r.file, "cuda_geometry.zig") and r.tower.get("hidden_size").?.integer == xpu.hidden) out[1] = try variant(a, r.json, xpu.layers, 4, false, null);
    }
    return out;
}

const Row = struct { name: []const u8, types: ?usize = null, null_types: bool = false, interval: ?[]const u8 = null, refused: ?anyerror = null };

/// #1863's table, a case a row (three for its third): every backend reads each the same.
const table = [_]Row{
    .{ .name = "no layer_types, no interval: every fourth layer" },
    .{ .name = "no layer_types, interval 4: every fourth layer", .interval = "4" },
    .{ .name = "no layer_types, interval 0: the interval rule has none", .interval = "0", .refused = error.BadFullAttentionInterval },
    .{ .name = "layer_types null: as absent", .null_types = true, .interval = "4" },
    .{ .name = "layer_types, interval 0: layer_types", .types = 4, .interval = "0" },
    .{ .name = "layer_types, interval null: layer_types", .types = 4, .interval = "null" },
    .{ .name = "layer_types, interval false: layer_types", .types = 4, .interval = "false" },
    .{ .name = "layer_types contradicting the interval: refused", .types = 4, .interval = "2", .refused = error.LayerTypesContradictFullAttentionInterval },
};

test "qwen35 layers: #1863's table, the same answer or refusal from every backend's entry point" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var took: std.EnumArray(Backend, usize) = .initFill(0);
    for (try bases(a)) |base| for (std.enums.values(Backend)) |b| {
        // the agreed case, layer_types every fourth and an interval of 4: a backend refusing it takes another model
        const probe = entry(b, try variant(a, base, null, 4, false, "4"));
        if (probe == .refused) {
            try t.expect(!probe.byReader());
            continue;
        }
        took.getPtr(b).* += 1;
        for (table) |row| {
            const got = entry(b, try variant(a, base, null, row.types, row.null_types, row.interval));
            errdefer std.debug.print("{s}, {s}: {any}\n", .{ @tagName(b), row.name, got });
            if (row.refused) |e| try t.expect(got == .refused and got.refused == e) else try t.expect(Outcome.same(got, .every(4, probe.pattern.len)));
        }
    };
    // the 2B: Metal's 2B engine, HIP and CUDA; the 27B: Metal's 27B engine, HIP, CUDA and XPU
    for (std.enums.values(Backend)) |b| try t.expect(took.get(b) >= 1);
    try t.expectEqual(@as(usize, 7), took.get(.metal27) + took.get(.metal2b) + took.get(.hip) + took.get(.cuda) + took.get(.xpu));
}

test "qwen35 layers: beside an unset interval, layer_types are read as written; fixed schedules refuse other ones" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const two = try bases(a);
    for ([_][]const u8{ "0", "null", "false" }) |interval| for ([_]Backend{ .metal27, .hip, .cuda }) |b| {
        try t.expect(Outcome.same(entry(b, try variant(a, two[1], null, 2, false, interval)), .every(2, xpu.layers)));
    };
    try t.expectEqual(Outcome{ .refused = error.UnsupportedLayerPattern }, entry(.xpu, try variant(a, two[1], null, 2, false, "null")));
    try t.expectEqual(Outcome{ .refused = error.UnsupportedQwenGeometry }, entry(.metal2b, try variant(a, two[0], null, 2, false, "null")));
}
