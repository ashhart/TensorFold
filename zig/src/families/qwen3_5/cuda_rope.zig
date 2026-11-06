//! A Qwen checkpoint's rope: plain, or static YaRN from its config or --rope-yarn, and the window that allows.
const std = @import("std");
const core = @import("core");

/// The rope base every admitted Qwen3.5/3.6 checkpoint declares (cuda_geometry.zig checks it).
pub const theta: f64 = 10000000;

pub const Yarn = struct {
    factor: f64,
    trained: usize, // the window the checkpoint was trained on: original_max_position_embeddings
    table: [33]f32, // the attention factor, then 32 frequencies: what the rope kernels read
};

pub const Rope = union(enum) { plain, yarn: Yarn };

pub const Resolved = struct { rope: Rope, ceiling: usize };

pub const Error = error{ UnsupportedRopeScaling, RopeYarnFactor, RopeYarnConflict, BadQwenConfig };

/// The tower's rope and ceiling: its YaRN block (rope_parameters/rope_scaling) or `flag`; both must name one factor.
pub fn resolve(text: std.json.ObjectMap, flag: ?f64) Error!Resolved {
    const max = try positive(text, "max_position_embeddings");
    const rope = text.get("rope_parameters") orelse return error.BadQwenConfig;
    if (rope != .object) return error.BadQwenConfig;
    if (flag) |f| if (!std.math.isFinite(f) or f < 1) return error.RopeYarnFactor;
    if (try scaling(text, rope.object)) |y| {
        const factor = try number(y, "factor");
        const trained = try positive(y, "original_max_position_embeddings");
        if (!(factor > 1) or @as(f64, @floatFromInt(trained)) * factor != @as(f64, @floatFromInt(max))) return error.UnsupportedRopeScaling;
        if (flag) |f| if (f != factor) return error.RopeYarnConflict;
        return .{ .rope = .{ .yarn = yarn(factor, trained) }, .ceiling = max };
    }
    const f = flag orelse return .{ .rope = .plain, .ceiling = max };
    if (f == 1) return .{ .rope = .plain, .ceiling = max };
    const ceiling = @as(f64, @floatFromInt(max)) * f;
    if (ceiling != @floor(ceiling) or ceiling > std.math.maxInt(u32)) return error.RopeYarnFactor;
    return .{ .rope = .{ .yarn = yarn(f, max) }, .ceiling = @intFromFloat(ceiling) };
}

/// `resolve` over `dir`'s config.json (its text_config, else the root), read with `a` (an arena: nothing is freed).
pub fn resolveDir(a: std.mem.Allocator, io: std.Io, dir: []const u8, flag: ?f64) !Resolved {
    const path = try std.fs.path.join(a, &.{ dir, "config.json" });
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(16 << 20));
    const doc = try std.json.parseFromSliceLeaky(std.json.Value, a, bytes, .{});
    if (doc != .object) return error.BadQwenConfig;
    const text = if (doc.object.get("text_config")) |t| t else doc;
    if (text != .object) return error.BadQwenConfig;
    return resolve(text.object, flag);
}

/// The YaRN block, null for plain rope; anything else, or a parameter outside transformers' default formula, refused.
fn scaling(text: std.json.ObjectMap, rope: std.json.ObjectMap) Error!?std.json.ObjectMap {
    var older = false; // the block came from the tower's rope_scaling, which only a scaled rope has
    const block = if (text.get("rope_scaling")) |v| switch (v) {
        .null => rope,
        .object => |o| blk: {
            older = true;
            break :blk o;
        },
        else => return error.UnsupportedRopeScaling,
    } else rope;
    const kind = block.get("rope_type") orelse block.get("type") orelse return if (older) error.UnsupportedRopeScaling else null;
    if (kind != .string) return error.UnsupportedRopeScaling;
    if (std.mem.eql(u8, kind.string, "default") and !older) return null;
    if (!std.mem.eql(u8, kind.string, "yarn")) return error.UnsupportedRopeScaling;
    for ([_][]const u8{ "mscale", "mscale_all_dim", "attention_factor" }) |key| if (block.get(key)) |v| if (v != .null) return error.UnsupportedRopeScaling;
    for ([_]struct { []const u8, f64 }{ .{ "beta_fast", 32 }, .{ "beta_slow", 1 } }) |b| if (block.get(b[0])) |v| {
        if (v != .null and (try value(v)) != b[1]) return error.UnsupportedRopeScaling;
    };
    if (block.get("truncate")) |v| if (v != .null and !(v == .bool and v.bool)) return error.UnsupportedRopeScaling;
    return block;
}

fn yarn(factor: f64, trained: usize) Yarn {
    const y = core.rope_yarn.yarn(theta, factor, trained);
    var out: Yarn = .{ .factor = factor, .trained = trained, .table = undefined };
    out.table[0] = y.scale;
    @memcpy(out.table[1..], &y.freq);
    return out;
}

fn positive(o: std.json.ObjectMap, key: []const u8) Error!usize {
    const v = o.get(key) orelse return error.BadQwenConfig;
    if (v != .integer or v.integer < 1 or v.integer > std.math.maxInt(u32)) return error.BadQwenConfig;
    return @intCast(v.integer);
}

fn number(o: std.json.ObjectMap, key: []const u8) Error!f64 {
    return value(o.get(key) orelse return error.UnsupportedRopeScaling);
}

fn value(v: std.json.Value) Error!f64 {
    return switch (v) {
        .integer => |i| @floatFromInt(i),
        .float => |x| x,
        else => error.UnsupportedRopeScaling,
    };
}

fn tower(a: std.mem.Allocator, text: []const u8) !std.json.Parsed(std.json.Value) {
    return std.json.parseFromSlice(std.json.Value, a, text, .{});
}

const plain_text =
    \\{"max_position_embeddings":262144,"rope_parameters":{"mrope_interleaved":true,"mrope_section":[11,11,10],
    \\ "partial_rotary_factor":0.25,"rope_theta":10000000,"rope_type":"default"}}
;
/// The text_config SGLang's extend-qwen3.6-35b-a3b-nvfp4-context-1m mod stages.
const sglang_text =
    \\{"max_position_embeddings":1048576,"rope_parameters":{"mrope_interleaved":true,"mrope_section":[11,11,10],
    \\ "partial_rotary_factor":0.25,"rope_theta":10000000,"rope_type":"yarn","factor":4.0,
    \\ "original_max_position_embeddings":262144}}
;

test "the downloaded config with --rope-yarn 4 and SGLang's staged 1M config resolve to one rope and ceiling" {
    const a = std.testing.allocator;
    const plain = try tower(a, plain_text);
    defer plain.deinit();
    const staged = try tower(a, sglang_text);
    defer staged.deinit();
    const r0 = try resolve(plain.value.object, null);
    try std.testing.expect(r0.rope == .plain and r0.ceiling == 262144);
    try std.testing.expect((try resolve(plain.value.object, 1)).rope == .plain);
    const r1 = try resolve(plain.value.object, 4);
    const r2 = try resolve(staged.value.object, null);
    const r3 = try resolve(staged.value.object, 4);
    for ([_]Resolved{ r1, r2, r3 }) |r| {
        try std.testing.expectEqual(@as(usize, 1048576), r.ceiling);
        try std.testing.expectEqual(@as(usize, 262144), r.rope.yarn.trained);
        try std.testing.expectEqual(@as(f64, 4), r.rope.yarn.factor);
        const y = core.rope_yarn.yarn(theta, 4, 262144);
        try std.testing.expectEqual(y.scale, r.rope.yarn.table[0]);
        try std.testing.expectEqualSlices(f32, &y.freq, r.rope.yarn.table[1..]);
    }
    try std.testing.expectError(error.RopeYarnConflict, resolve(staged.value.object, 2));
    for ([_]f64{ 0.5, 0, -4, std.math.inf(f64), std.math.nan(f64), 1.0000001 }) |bad| try std.testing.expectError(error.RopeYarnFactor, resolve(plain.value.object, bad));
}

test "a YaRN block outside transformers' default formula, or naming another scaling, is refused by name" {
    const a = std.testing.allocator;
    const cases = [_]struct { []const u8, []const u8 }{
        .{ "\"factor\":4.0", "\"factor\":2.0" }, // the two lengths disagree with the factor
        .{ "\"factor\":4.0", "\"factor\":1.0" },
        .{ "\"factor\":4.0,", "" },
        .{ "\"factor\":4.0", "\"factor\":4.0,\"mscale\":1.0" },
        .{ "\"factor\":4.0", "\"factor\":4.0,\"mscale_all_dim\":1.0" },
        .{ "\"factor\":4.0", "\"factor\":4.0,\"attention_factor\":1.2" },
        .{ "\"factor\":4.0", "\"factor\":4.0,\"beta_fast\":16" },
        .{ "\"factor\":4.0", "\"factor\":4.0,\"beta_slow\":2" },
        .{ "\"factor\":4.0", "\"factor\":4.0,\"truncate\":false" },
        .{ "\"rope_type\":\"yarn\"", "\"rope_type\":\"dynamic\"" },
        .{ "\"rope_type\":\"yarn\"", "\"rope_type\":\"longrope\"" },
    };
    for (cases) |c| {
        const text = try std.mem.replaceOwned(u8, a, sglang_text, c[0], c[1]);
        defer a.free(text);
        const t = try tower(a, text);
        defer t.deinit();
        try std.testing.expectError(error.UnsupportedRopeScaling, resolve(t.value.object, null));
    }
    // the defaults spelled out are the formula; the older rope_scaling block and "type" key read the same
    const spelled = try std.mem.replaceOwned(u8, a, sglang_text, "\"factor\":4.0", "\"factor\":4.0,\"beta_fast\":32,\"beta_slow\":1,\"truncate\":true,\"mscale\":null");
    defer a.free(spelled);
    const t = try tower(a, spelled);
    defer t.deinit();
    try std.testing.expectEqual(@as(usize, 1048576), (try resolve(t.value.object, null)).ceiling);
    const old =
        \\{"max_position_embeddings":1048576,"rope_parameters":{"rope_theta":10000000,"partial_rotary_factor":0.25},
        \\ "rope_scaling":{"type":"yarn","factor":4,"original_max_position_embeddings":262144}}
    ;
    const o = try tower(a, old);
    defer o.deinit();
    try std.testing.expectEqual(@as(f64, 4), (try resolve(o.value.object, null)).rope.yarn.factor);
    const odd = try tower(a, "{\"max_position_embeddings\":262144,\"rope_parameters\":{\"rope_theta\":10000000},\"rope_scaling\":{\"type\":\"default\"}}");
    defer odd.deinit();
    try std.testing.expectError(error.UnsupportedRopeScaling, resolve(odd.value.object, null));
}
