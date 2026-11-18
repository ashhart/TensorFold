//! config.json's dfloat11_config: the coding's two sizes, and which module names are groups of which members.

const std = @import("std");

/// A module-name pattern and the members whose weights its group holds, in order (none: the module's own weight).
pub const Pattern = struct {
    text: []const u8,
    members: []const []const u8,
};

pub const Config = struct {
    bytes_per_thread: u32,
    threads_per_block: u32,
    patterns: []const Pattern,

    /// The pattern that describes the whole of `module`; null when none does, an error when two do.
    pub fn group(self: Config, module: []const u8) error{AmbiguousDfloat11Pattern}!?Pattern {
        var found: ?Pattern = null;
        for (self.patterns) |p| {
            if (!matches(p.text, module)) continue;
            if (found) |first| {
                if (!@import("builtin").is_test) std.log.err("dfloat11_config: {s} matches both {s} and {s}", .{ module, first.text, p.text });
                return error.AmbiguousDfloat11Pattern;
            }
            found = p;
        }
        return found;
    }
};

/// `text`'s dfloat11_config with its strings in `arena`; null when config.json has none.
pub fn parse(arena: std.mem.Allocator, text: []const u8) !?Config {
    const root = std.json.parseFromSliceLeaky(std.json.Value, arena, text, .{}) catch |err| switch (err) {
        error.OutOfMemory => return err,
        else => return error.BadDfloat11Config,
    };
    if (root != .object) return error.BadDfloat11Config;
    const value = root.object.get("dfloat11_config") orelse return null;
    if (value != .object) return error.BadDfloat11Config;
    const o = value.object;
    const threads = o.get("threads_per_block") orelse return error.BadDfloat11Config;
    // the writers seen store the number in a list of one
    if (threads != .array or threads.array.items.len != 1) return error.BadDfloat11Config;
    const dict = o.get("pattern_dict") orelse return error.BadDfloat11Config;
    if (dict != .object) return error.BadDfloat11Config;
    const patterns = try arena.alloc(Pattern, dict.object.count());
    var it = dict.object.iterator();
    for (patterns) |*p| {
        const kv = it.next().?;
        if (!readable(kv.key_ptr.*)) {
            if (!@import("builtin").is_test) std.log.err("dfloat11_config: pattern {s} uses more than letters, digits, _ - . \\. \\d and \\d+", .{kv.key_ptr.*});
            return error.UnsupportedDfloat11Pattern;
        }
        if (kv.value_ptr.* != .array) return error.BadDfloat11Config;
        const members = try arena.alloc([]const u8, kv.value_ptr.array.items.len);
        for (members, kv.value_ptr.array.items) |*m, v| {
            if (v != .string or v.string.len == 0) return error.BadDfloat11Config;
            m.* = v.string;
        }
        p.* = .{ .text = kv.key_ptr.*, .members = members };
    }
    return .{
        .bytes_per_thread = try number(o.get("bytes_per_thread")),
        .threads_per_block = try number(threads.array.items[0]),
        .patterns = patterns,
    };
}

fn number(v: ?std.json.Value) error{BadDfloat11Config}!u32 {
    const x = v orelse return error.BadDfloat11Config;
    if (x != .integer) return error.BadDfloat11Config;
    return std.math.cast(u32, x.integer) orelse error.BadDfloat11Config;
}

/// One step of a pattern: a byte as itself, `.` for any byte, `\.` for a dot, `\d` for a digit, `\d+` for digits.
const Atom = struct {
    kind: enum { byte, any, digit },
    byte: u8 = 0,
    many: bool = false,
    len: usize,

    fn read(p: []const u8) ?Atom {
        switch (p[0]) {
            '\\' => {
                if (p.len < 2) return null;
                if (p[1] == '.') return .{ .kind = .byte, .byte = '.', .len = 2 };
                if (p[1] != 'd') return null;
                const many = p.len > 2 and p[2] == '+';
                return .{ .kind = .digit, .many = many, .len = if (many) 3 else 2 };
            },
            '.' => return .{ .kind = .any, .len = 1 },
            'a'...'z', 'A'...'Z', '0'...'9', '_', '-' => return .{ .kind = .byte, .byte = p[0], .len = 1 },
            else => return null,
        }
    }

    fn takes(a: Atom, c: u8) bool {
        return switch (a.kind) {
            .byte => c == a.byte,
            .any => true,
            .digit => std.ascii.isDigit(c),
        };
    }
};

/// Whether every step of `pattern` is one this reader takes (the published configs use no other).
fn readable(pattern: []const u8) bool {
    var p = pattern;
    while (p.len > 0) p = p[(Atom.read(p) orelse return false).len..];
    return true;
}

/// Whether `name` is the whole of what `pattern` (a readable one) describes.
fn matches(pattern: []const u8, name: []const u8) bool {
    if (pattern.len == 0) return name.len == 0;
    const a = Atom.read(pattern).?;
    var n: usize = 0;
    while (n < name.len and a.takes(name[n])) {
        n += 1;
        if (matches(pattern[a.len..], name[n..])) return true;
        if (!a.many) break;
    }
    return false;
}

const testing = std.testing;

// four published repositories' dfloat11_config objects: keys and patterns as stored, the longer member lists cut short
const published = [_][]const u8{
    \\{"dfloat11_config":{"bytes_per_thread":8,"pattern_dict":{"lm_head":[],"model.embed_tokens":[],"model.layers.\\d+":
    \\["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","self_attn.o_proj","mlp.gate_proj","mlp.up_proj",
    \\"mlp.down_proj"]},"threads_per_block":[512],"version":"0.2.0"}}
    ,
    \\{"dfloat11_config":{"bytes_per_thread":8,"pattern_dict":{"transformer_blocks.\\d+":["norm1.linear","attn.to_q"],
    \\"single_transformer_blocks.\\d+":["norm.linear","proj_mlp"]},"threads_per_block":[512],"version":"0.2.0"}}
    ,
    \\{"dfloat11_config":{"bytes_per_thread":8,"pattern_dict":{"language_model.model.layers.\\d+":["self_attn.q_proj",
    \\"mlp.down_proj"]},"threads_per_block":[512],"version":"0.2.0"}}
    ,
    \\{"dfloat11_config":{"bytes_per_thread":8,"pattern_dict":{"transformer_blocks\\.\\d+":["img_mod.1","attn.to_q"]},
    \\"threads_per_block":[512],"version":"0.5.0"}}
};

test "the published configs read: sizes, patterns with and without an escaped dot, members in the list's order" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const llama = (try parse(arena.allocator(), published[0])).?;
    try testing.expectEqual(@as(u32, 8), llama.bytes_per_thread);
    try testing.expectEqual(@as(u32, 512), llama.threads_per_block);
    try testing.expectEqual(@as(usize, 0), (try llama.group("lm_head")).?.members.len);
    try testing.expectEqual(@as(usize, 0), (try llama.group("model.embed_tokens")).?.members.len);
    const layer = (try llama.group("model.layers.31")).?;
    try testing.expectEqual(@as(usize, 7), layer.members.len);
    try testing.expectEqualStrings("self_attn.q_proj", layer.members[0]);
    try testing.expectEqualStrings("mlp.down_proj", layer.members[6]);
    for ([_][]const u8{ "model.layers", "model.layers.", "model.layers.3.mlp", "model.layers.x", "xmodel.layers.3", "lm_head2", "model.layerss.3" }) |name|
        try testing.expectEqual(@as(?Pattern, null), try llama.group(name));

    // the whole name: a block of the single kind is not also a block of the other kind
    const flux = (try parse(arena.allocator(), published[1])).?;
    try testing.expectEqualStrings("norm.linear", (try flux.group("single_transformer_blocks.7")).?.members[0]);
    try testing.expectEqualStrings("norm1.linear", (try flux.group("transformer_blocks.7")).?.members[0]);

    const gemma = (try parse(arena.allocator(), published[2])).?;
    try testing.expect((try gemma.group("language_model.model.layers.0")) != null);
    const edit = (try parse(arena.allocator(), published[3])).?;
    try testing.expect((try edit.group("transformer_blocks.59")) != null);
    // an escaped dot is a dot; a bare one is any byte, as the expression says
    try testing.expectEqual(@as(?Pattern, null), try edit.group("transformer_blocks_59"));
    try testing.expect((try flux.group("transformer_blocks_59")) != null);
}

test "a config without the object is null; a pattern past the readable steps, or an odd size, is refused" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try testing.expectEqual(@as(?Config, null), try parse(a, "{\"model_type\":\"x\"}"));
    const head = "{\"dfloat11_config\":{\"bytes_per_thread\":8,\"threads_per_block\":[512],\"pattern_dict\":{\"";
    for ([_][]const u8{ "layers.(\\\\d+)", "a|b", "x*", "x+", "layers.\\\\w+", "^lm_head$", "layers.[0-9]", "tail\\\\" }) |pattern| {
        const text = try std.fmt.allocPrint(a, "{s}{s}\":[]}}}}}}", .{ head, pattern });
        try testing.expectError(error.UnsupportedDfloat11Pattern, parse(a, text));
    }
    for ([_][]const u8{
        "[]",
        "{\"dfloat11_config\":7}",
        "{\"dfloat11_config\":{\"bytes_per_thread\":8,\"threads_per_block\":512,\"pattern_dict\":{}}}",
        "{\"dfloat11_config\":{\"bytes_per_thread\":8,\"threads_per_block\":[512,256],\"pattern_dict\":{}}}",
        "{\"dfloat11_config\":{\"bytes_per_thread\":8,\"threads_per_block\":[512]}}",
        "{\"dfloat11_config\":{\"threads_per_block\":[512],\"pattern_dict\":{}}}",
        "{\"dfloat11_config\":{\"bytes_per_thread\":-8,\"threads_per_block\":[512],\"pattern_dict\":{}}}",
        "{\"dfloat11_config\":{\"bytes_per_thread\":8,\"threads_per_block\":[512],\"pattern_dict\":{\"m\":[7]}}}",
        "{\"dfloat11_config\":{\"bytes_per_thread\":8,\"threads_per_block\":[512],\"pattern_dict\":{\"m\":\"w\"}}}",
        "{\"dfloat11_config\":",
    }) |text| try testing.expectError(error.BadDfloat11Config, parse(a, text));
}

test "a module two patterns describe is an error, not the first one's" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const text = "{\"dfloat11_config\":{\"bytes_per_thread\":8,\"threads_per_block\":[512],\"pattern_dict\":" ++
        "{\"model.layers.\\\\d+\":[\"a\"],\"model.layers.\\\\d\\\\d\":[\"b\"]}}}";
    const c = (try parse(arena.allocator(), text)).?;
    try testing.expectEqualStrings("a", (try c.group("model.layers.3")).?.members[0]);
    try testing.expectError(error.AmbiguousDfloat11Pattern, c.group("model.layers.12"));
}
