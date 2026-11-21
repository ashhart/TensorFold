//! gptq: the shared AutoGPTQ v1 reader (core/quant/gptq.zig) on a synthetic checkpoint, against binary64.
const std = @import("std");
const gptq = @import("gptq.zig");
const ckpt = @import("../checkpoint.zig");
const fixture = @import("../checkpoint_fixture.zig");
const Why = @import("../modelopt.zig").Why;

const testing = std.testing;
const a = testing.allocator;
const io = testing.io;

const config_json = @embedFile("../fixtures/gptq_synth/config.json");

/// The two modules the synthetic checkpoint holds, and the [n, k] each one stores.
const expert = "model.language_model.layers.0.mlp.experts.0.gate_proj";
const head = "language_model.lm_head";

/// The fp16 bit patterns the fixture stores: awkward mantissas, a subnormal, a negative and a negative zero.
const scale_bits = [2][8]u16{
    .{ 0x3C00, 0x3555, 0xB800, 0x0011, 0x4900, 0x7BFF, 0x8000, 0x3E00 },
    .{ 0x4200, 0xBC00, 0x5000, 0x3400, 0x0001, 0x3800, 0xC000, 0x3900 },
};

/// The code of input `k` in column `c` the fixture writes: a full 0..15 cycle per column, a third of a turn apart.
fn codeOf(c: usize, k: usize) u8 {
    return @intCast((k + 3 * c) % 16);
}

/// What those codes and scale bits mean by the format's definition, in binary64.
fn want(c: usize, k: usize) f64 {
    const half: f16 = @bitCast(scale_bits[k / gptq.group][c]);
    return @as(f64, @floatFromInt(@as(i32, codeOf(c, k)) - 8)) * @as(f64, @floatCast(half));
}

/// AutoGPTQ v1's qweight packing, written from the format: word [k / 8][n], nibble j is input 8w + j, low first.
fn pack(k: usize, n: usize) ![]u8 {
    const out = try a.alloc(u8, (k / 8) * n * 4);
    @memset(out, 0);
    for (0..k / 8) |w| for (0..n) |c| for (0..8) |j| {
        const q = codeOf(c, w * 8 + j);
        out[(w * n + c) * 4 + j / 2] |= if (j % 2 == 0) q else q << 4;
    };
    return out;
}

/// The scales [k / 128][n] as fp16 bit patterns, little-endian.
fn scales(k: usize, n: usize) ![]u8 {
    const out = try a.alloc(u8, (k / gptq.group) * n * 2);
    for (0..k / gptq.group) |g| for (0..n) |c| {
        std.mem.writeInt(u16, out[(g * n + c) * 2 ..][0..2], scale_bits[g][c], .little);
    };
    return out;
}

/// qzeros [k / 128][n / 8] int32 whose every nibble is 7, so every zero point the format reads is 8.
fn zeros(k: usize, n: usize) ![]u8 {
    const out = try a.alloc(u8, (k / gptq.group) * (n / 8) * 4);
    @memset(out, 0x77);
    return out;
}

/// The synthetic checkpoint's six tensors, packed to the format; `entries` is what the safetensors image holds.
const Synth = struct {
    const modules = [2][]const u8{ expert, head };
    const ns = [2]usize{ 8, 8 };
    const ks = [2]usize{ 256, 128 };
    const tails = [3][]const u8{ "qweight", "scales", "qzeros" };
    const dtypes = [3][]const u8{ "I32", "F16", "I32" };

    bytes: [6][]u8 = undefined,
    shapes: [6][2]usize = undefined,
    names: [6][160]u8 = undefined,
    entries: [6]fixture.Entry = undefined,

    fn init(s: *Synth) !void {
        for (0..2) |m| for (0..3) |t| {
            const i = m * 3 + t;
            const n = ns[m];
            const k = ks[m];
            s.bytes[i] = switch (t) {
                0 => try pack(k, n),
                1 => try scales(k, n),
                else => try zeros(k, n),
            };
            s.shapes[i] = switch (t) {
                0 => .{ k / 8, n },
                1 => .{ k / gptq.group, n },
                else => .{ k / gptq.group, n / 8 },
            };
            const name = try std.fmt.bufPrint(&s.names[i], "{s}.{s}", .{ modules[m], tails[t] });
            s.entries[i] = .{ .name = name, .dtype = dtypes[t], .shape = &s.shapes[i], .bytes = s.bytes[i] };
        };
    }

    fn deinit(s: *Synth) void {
        for (s.bytes) |b| a.free(b);
    }
};

/// A checkpoint folder: `config.json` and one `model.safetensors`.
const Folder = struct {
    tmp: testing.TmpDir,
    path: [160]u8 = undefined,
    len: usize = 0,
    ck: ckpt.Checkpoint = undefined,

    fn open(f: *Folder, entries: []const fixture.Entry) !void {
        f.tmp = testing.tmpDir(.{});
        errdefer f.tmp.cleanup();
        try f.tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = config_json });
        try fixture.write(a, io, f.tmp.dir, "model.safetensors", entries);
        const sub = try std.fmt.bufPrint(&f.path, ".zig-cache/tmp/{s}", .{f.tmp.sub_path});
        f.len = sub.len;
        f.ck = try ckpt.Checkpoint.openModel(a, io, sub);
    }

    /// The folder as the checkpoint opened it.
    fn dir(f: *const Folder) []const u8 {
        return f.path[0..f.len];
    }

    fn close(f: *Folder) void {
        f.ck.close();
        f.tmp.cleanup();
    }

    /// The folder's own config.json, as the checkpoint's file.
    fn configText(f: *Folder) ![]u8 {
        var buf: [192]u8 = undefined;
        const path = try std.fmt.bufPrint(&buf, "{s}/config.json", .{f.dir()});
        return std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 16));
    }
};

test "gptq: the synthetic checkpoint's config is admitted and names the modules it stores quantized" {
    var s: Synth = .{};
    try s.init();
    defer s.deinit();
    var f: Folder = .{ .tmp = undefined };
    try f.open(&s.entries);
    defer f.close();
    const json = try f.configText();
    defer a.free(json);
    var why: Why = .{};
    var cfg = (try gptq.admit(a, json, &why)).?;
    defer cfg.deinit();
    try testing.expectEqual(@as(usize, 11), cfg.rules.len);
    try testing.expectEqual(@as(usize, 0), why.len);
    for ([_][]const u8{
        expert,
        "model.language_model.layers.47.mlp.experts.511.down_proj",
        "model.language_model.layers.3.mlp.experts.0.up_proj",
        "language_model.lm_head",
    }) |m| try testing.expect(cfg.quantized(m));
    for ([_][]const u8{
        "model.language_model.layers.0.linear_attn.in_proj_qkv",
        "model.language_model.layers.3.self_attn.q_proj",
        "model.language_model.layers.0.attn_hyper_connection.input_mix_weight_up",
        "model.visual.blocks.0.attn.qkv",
        "model.language_model.layers.0.mlp.shared_expert.gate_proj",
        "model.language_model.layers.0.ple.embedding",
        "model.language_model.embed_tokens",
        "model.language_model.layers.0.mlp.fc_hidden",
        "model.language_model.layers.48.mlp.down_proj",
        "model.language_model.layers.0.mlp.gate",
    }) |m| try testing.expect(!cfg.quantized(m));
}

test "gptq: the stored tensors dequantize to the binary64 fixture" {
    var s: Synth = .{};
    try s.init();
    defer s.deinit();
    var f: Folder = .{ .tmp = undefined };
    try f.open(&s.entries);
    defer f.close();
    const h = try gptq.read(&f.ck, expert, 8, 256);
    try testing.expectEqual(@as(usize, 8 * 256), gptq.elements(h));
    var got: [8 * 256]f64 = undefined;
    try gptq.reference(h, &got);
    for (0..8) |c| for (0..256) |k| {
        const bits: u64 = @bitCast(got[c * 256 + k]);
        if (bits != @as(u64, @bitCast(want(c, k)))) {
            std.debug.print("column {d} input {d}: 0x{x:0>16}, expected 0x{x:0>16}\n", .{ c, k, bits, @as(u64, @bitCast(want(c, k))) });
            return error.TestUnexpectedResult;
        }
    };
    // the binary64 fixture: every code, both groups, a negative, a subnormal and a negative zero
    const spot = [_]struct { c: usize, k: usize, want: f64 }{
        .{ .c = 0, .k = 0, .want = -8.0 }, // q -8 x 1
        .{ .c = 0, .k = 1, .want = -7.0 }, // q -7 x 1
        .{ .c = 0, .k = 2, .want = -6.0 },
        .{ .c = 0, .k = 3, .want = -5.0 },
        .{ .c = 0, .k = 4, .want = -4.0 },
        .{ .c = 0, .k = 5, .want = -3.0 },
        .{ .c = 0, .k = 6, .want = -2.0 },
        .{ .c = 0, .k = 7, .want = -1.0 },
        .{ .c = 0, .k = 8, .want = 0.0 }, // q 0, the zero point itself
        .{ .c = 0, .k = 9, .want = 1.0 },
        .{ .c = 0, .k = 10, .want = 2.0 },
        .{ .c = 0, .k = 11, .want = 3.0 },
        .{ .c = 0, .k = 12, .want = 4.0 },
        .{ .c = 0, .k = 13, .want = 5.0 },
        .{ .c = 0, .k = 14, .want = 6.0 },
        .{ .c = 0, .k = 15, .want = 7.0 },
        .{ .c = 1, .k = 0, .want = -1.666259765625 }, // q -5 x 0.333251953125
        .{ .c = 1, .k = 1, .want = -1.3330078125 },
        .{ .c = 1, .k = 2, .want = -0.999755859375 },
        .{ .c = 1, .k = 3, .want = -0.66650390625 },
        .{ .c = 1, .k = 4, .want = -0.333251953125 },
        .{ .c = 1, .k = 5, .want = 0.0 },
        .{ .c = 1, .k = 6, .want = 0.333251953125 },
        .{ .c = 1, .k = 7, .want = 0.66650390625 },
        .{ .c = 2, .k = 0, .want = 1.0 }, // q -2 x -0.5
        .{ .c = 2, .k = 1, .want = 0.5 },
        .{ .c = 2, .k = 2, .want = -0.0 }, // a zero code under a negative scale keeps the product's sign
        .{ .c = 2, .k = 3, .want = -0.5 },
        .{ .c = 3, .k = 0, .want = 1.0132789611816406e-06 }, // a subnormal scale, q 1
        .{ .c = 3, .k = 1, .want = 2.0265579223632812e-06 },
        .{ .c = 0, .k = 128, .want = -24.0 }, // the second group's own scale, q -8 x 3
        .{ .c = 0, .k = 129, .want = -21.0 },
        .{ .c = 0, .k = 130, .want = -18.0 },
        .{ .c = 0, .k = 131, .want = -15.0 },
        .{ .c = 0, .k = 132, .want = -12.0 },
        .{ .c = 0, .k = 133, .want = -9.0 },
        .{ .c = 0, .k = 134, .want = -6.0 },
        .{ .c = 0, .k = 135, .want = -3.0 },
        .{ .c = 7, .k = 128, .want = -1.875 }, // q -3 x 0.625
        .{ .c = 7, .k = 129, .want = -1.25 },
        .{ .c = 7, .k = 130, .want = -0.625 },
        .{ .c = 7, .k = 131, .want = 0.0 },
        .{ .c = 4, .k = 128, .want = 2.384185791015625e-07 }, // the smallest subnormal, q 4
        .{ .c = 4, .k = 129, .want = 2.980232238769531e-07 },
        .{ .c = 6, .k = 0, .want = 0.0 }, // a negative-zero scale turns a negative code positive
        .{ .c = 6, .k = 1, .want = 0.0 },
    };
    for (spot) |p| try testing.expectEqual(@as(u64, @bitCast(p.want)), @as(u64, @bitCast(got[p.c * 256 + p.k])));
    try testing.expectEqual(@as(u8, 15), gptq.code(h, 0, 15));
    try testing.expectEqual(@as(u8, 8), gptq.code(h, 2, 2)); // the zero point itself
    try testing.expectEqual(@as(f64, 0.333251953125), gptq.scale(h, 1, 100));
    try testing.expectEqual(@as(f64, 3), gptq.scale(h, 0, 128));
    try testing.expectEqual(@as(u64, @bitCast(@as(f64, 2.0))), @as(u64, @bitCast(gptq.weight(h, 0, 10))));
    const hd = try gptq.read(&f.ck, head, 8, 128);
    var one: [8 * 128]f64 = undefined;
    try gptq.reference(hd, &one);
    try testing.expectEqual(@as(u64, @bitCast(@as(f64, -8.0))), @as(u64, @bitCast(one[0])));
    try testing.expectEqual(@as(u64, @bitCast(@as(f64, -24.0))), @as(u64, @bitCast(got[128])));
}

test "gptq: the matcher reads the rule subset and anchors as Python's re.match does" {
    const cases = [_]struct { pattern: []const u8, hit: []const u8, miss: []const u8 }{
        .{ .pattern = ".*lm_head$", .hit = "language_model.lm_head", .miss = "language_model.lm_head.weight" },
        .{ .pattern = ".*\\.gate$", .hit = "model.language_model.layers.0.mlp.gate", .miss = "model.language_model.layers.0.mlp.gate_proj" },
        .{ .pattern = ".*linear_attn.*", .hit = "model.language_model.layers.0.linear_attn.in_proj_qkv", .miss = "model.language_model.layers.0.self_attn.q_proj" },
        .{ .pattern = ".*self_attn.*", .hit = "model.language_model.layers.3.self_attn.q_proj", .miss = "model.language_model.layers.3.mlp.experts.0.gate_proj" },
        .{ .pattern = ".*hyper_connection.*", .hit = "model.language_model.layers.0.attn_hyper_connection.input_mix_weight_up", .miss = "model.language_model.layers.0.mlp.experts.0.gate_proj" },
        .{ .pattern = ".*visual.*", .hit = "model.visual.blocks.0.attn.qkv", .miss = "model.language_model.embed_tokens" },
        .{ .pattern = ".*shared_expert.*", .hit = "model.language_model.layers.0.mlp.shared_expert.gate_proj", .miss = "model.language_model.layers.0.mlp.experts.0.gate_proj" },
        .{ .pattern = ".*\\.ple\\..*", .hit = "model.language_model.layers.0.ple.embedding", .miss = "model.language_model.layers.0.pleb" },
        .{ .pattern = ".*embed.*", .hit = "model.language_model.embed_tokens", .miss = "model.language_model.layers.0.mlp.down_proj" },
        .{ .pattern = ".*fc_hidden.*", .hit = "model.language_model.layers.0.mlp.fc_hidden.weight", .miss = "model.language_model.layers.0.mlp.experts.0.gate_proj" },
        .{ .pattern = ".*layers\\.48\\..*", .hit = "model.language_model.layers.48.mlp.down_proj", .miss = "model.language_model.layers.4.mlp.down_proj" },
    };
    for (cases) |c| {
        try testing.expect(gptq.regexMatch(c.pattern, c.hit));
        try testing.expect(!gptq.regexMatch(c.pattern, c.miss));
    }
    try testing.expect(gptq.regexMatch("abc", "abcdef")); // re.match anchors at the start, not the end
    try testing.expect(!gptq.regexMatch("abc", "xabc"));
    try testing.expect(gptq.regexMatch("a.c", "abc"));
    try testing.expect(!gptq.regexMatch("a.c", "ac"));
    try testing.expect(gptq.regexMatch("a\\.c", "a.c"));
    try testing.expect(!gptq.regexMatch("a\\.c", "abc"));
    try testing.expect(gptq.regexMatch("a*b", "aaab")); // '*' and '+' backtrack
    try testing.expect(gptq.regexMatch("a*b", "b"));
    try testing.expect(!gptq.regexMatch("a*b", "aaac"));
    try testing.expect(gptq.regexMatch("a+b", "aaab"));
    try testing.expect(!gptq.regexMatch("a+b", "b"));
    try testing.expect(gptq.regexMatch("$", ""));
    try testing.expect(!gptq.regexMatch("$", "x"));
    try testing.expect(gptq.regexMatch(".*\\.48\\..*", "layers.48.x"));
    try testing.expect(!gptq.regexMatch(".*\\.48\\..*", "layers.480.x"));
}

test "gptq: an escape of an alphanumeric character is refused, as Python's re reads it as a class or an anchor" {
    // Python's re matches model.layers.12.mlp.x here; read as a literal d it matches layers.d.mlp instead
    try refuses(
        \\"quant_method": "gptq", "bits": 4, "group_size": 128, "sym": true, "desc_act": false, "lm_head": true, "dynamic": {"+:.*lm_head$": {"bits": 4}, "-:.*layers\\.\\d+\\.mlp.*": {"bits": 4}}
    , "quantization_config.dynamic -:.*layers\\.\\d+\\.mlp.*: outside this reader's matcher");
    try refuses(
        \\"quant_method": "gptq", "bits": 4, "group_size": 128, "sym": true, "desc_act": false, "lm_head": true, "dynamic": {"+:.*lm_head$": {"bits": 4}, "-:.*\\w+.*": {"bits": 4}}
    , "quantization_config.dynamic -:.*\\w+.*: outside this reader's matcher");
    try refuses(
        \\"quant_method": "gptq", "bits": 4, "group_size": 128, "sym": true, "desc_act": false, "lm_head": true, "dynamic": {"+:.*lm_head$": {"bits": 4}, "-:.*\\s+\\b.*": {"bits": 4}}
    , "quantization_config.dynamic -:.*\\s+\\b.*: outside this reader's matcher");
    // an escape of a non-alphanumeric character is the character itself, in Python's re and in this matcher
    try testing.expect(gptq.regexMatch("a\\+b", "a+b"));
    try testing.expect(!gptq.regexMatch("a\\+b", "aab"));
    try testing.expect(gptq.regexMatch("a\\$", "a$"));
    try testing.expect(gptq.regexMatch("a\\\\b", "a\\b"));
    var why: Why = .{};
    var plus = (try gptq.admit(a, jsonOf(
        \\"quant_method": "gptq", "bits": 4, "group_size": 128, "sym": true, "desc_act": false, "lm_head": true, "dynamic": {"+:.*lm_head$": {"bits": 4}, "-:a\\+b.*": {"bits": 4}}
    ), &why)).?;
    defer plus.deinit();
    try testing.expect(!plus.quantized("a+bzzz"));
    try testing.expect(plus.quantized("aab"));
}

/// `fields` as a config.json's quantization_config, so one field can be changed per case.
fn jsonOf(comptime fields: []const u8) []const u8 {
    return "{\"quantization_config\": {" ++ fields ++ "}}";
}

/// `jsonOf(fields)` is refused as unsupported, its reason exactly `text`.
fn refuses(comptime fields: []const u8, text: []const u8) !void {
    var why: Why = .{};
    try testing.expectError(error.UnsupportedQuantization, gptq.admit(a, jsonOf(fields), &why));
    try testing.expectEqualStrings(text, why.text());
}

/// The admitted fields: the fixture's shape, with the head rule alone in `dynamic`.
const ok =
    \\"quant_method": "gptq", "bits": 4, "group_size": 128, "sym": true, "desc_act": false, "checkpoint_format": "gptq",
    \\"lm_head": true, "dynamic": {"+:.*lm_head$": {"bits": 4}}
;

test "gptq: a config outside the dialect is refused by its field or rule name" {
    var why: Why = .{};
    var cfg = (try gptq.admit(a, jsonOf(ok), &why)).?;
    cfg.deinit();
    try refuses(
        \\"quant_method": "gptq", "bits": 8, "group_size": 128, "sym": true, "lm_head": true, "dynamic": {"+:.*lm_head$": {"bits": 4}}
    , "quantization_config.bits: 8, expected 4");
    try refuses(
        \\"quant_method": "gptq", "group_size": 128, "sym": true, "lm_head": true, "dynamic": {"+:.*lm_head$": {"bits": 4}}
    , "quantization_config.bits: null, expected 4");
    try refuses(
        \\"quant_method": "gptq", "bits": 4, "group_size": 64, "sym": true, "lm_head": true, "dynamic": {"+:.*lm_head$": {"bits": 4}}
    , "quantization_config.group_size: 64, expected 128");
    try refuses(
        \\"quant_method": "gptq", "bits": 4, "sym": true, "lm_head": true, "dynamic": {"+:.*lm_head$": {"bits": 4}}
    , "quantization_config.group_size: null, expected 128");
    try refuses(
        \\"quant_method": "gptq", "bits": 4, "group_size": 128, "sym": false, "lm_head": true, "dynamic": {"+:.*lm_head$": {"bits": 4}}
    , "quantization_config.sym: false, expected true");
    try refuses(
        \\"quant_method": "gptq", "bits": 4, "group_size": 128, "lm_head": true, "dynamic": {"+:.*lm_head$": {"bits": 4}}
    , "quantization_config.sym: null, expected true");
    try refuses(
        \\"quant_method": "gptq", "bits": 4, "group_size": 128, "sym": true, "desc_act": true, "lm_head": true, "dynamic": {"+:.*lm_head$": {"bits": 4}}
    , "quantization_config.desc_act: true, this reader reads no act order");
    try refuses(
        \\"quant_method": "gptq", "bits": 4, "group_size": 128, "sym": true, "desc_act": "false", "lm_head": true, "dynamic": {"+:.*lm_head$": {"bits": 4}}
    , "quantization_config.desc_act: a string, expected a bool");
    try refuses(
        \\"quant_method": "gptq", "bits": 4, "group_size": 128, "sym": true, "desc_act": null, "lm_head": true, "dynamic": {"+:.*lm_head$": {"bits": 4}}
    , "quantization_config.desc_act: null, expected a bool");
    try refuses(
        \\"quant_method": "gptq", "bits": 4, "group_size": 128, "sym": true, "checkpoint_format": "marlin", "lm_head": true, "dynamic": {"+:.*lm_head$": {"bits": 4}}
    , "quantization_config.checkpoint_format: marlin, expected gptq");
    try refuses(
        \\"quant_method": "gptq", "bits": 4, "group_size": 128, "sym": true, "checkpoint_format": "GPTQ", "lm_head": true, "dynamic": {"+:.*lm_head$": {"bits": 4}}
    , "quantization_config.checkpoint_format: GPTQ, expected gptq");
    try refuses(
        \\"quant_method": "gptq", "bits": 4, "group_size": 128, "sym": true, "checkpoint_format": 1, "lm_head": true, "dynamic": {"+:.*lm_head$": {"bits": 4}}
    , "quantization_config.checkpoint_format: a number, expected gptq");
    try refuses(
        \\"quant_method": "gptq", "bits": 4, "group_size": 128, "sym": true, "desc_act": false, "lm_head": false, "dynamic": {"-:.*linear_attn.*": {"bits": 4}}
    , "quantization_config.lm_head: false, the rules do not store the head int4");
    try refuses(
        \\"quant_method": "gptq", "bits": 4, "group_size": 128, "sym": true, "lm_head": true, "dynamic": {"-:.*lm_head$": {"bits": 4}}
    , "quantization_config.lm_head: true, the rules do not store the head int4");
    try refuses(
        \\"quant_method": "gptq", "bits": 4, "group_size": 128, "sym": true, "dynamic": {"-:.*linear_attn.*": {"bits": 4}}
    , "quantization_config.lm_head: null, the rules do not store the head int4");
    try refuses(
        \\"quant_method": "gptq", "bits": 4, "group_size": 128, "sym": true, "lm_head": true
    , "quantization_config.dynamic: absent, this reader reads its rules");
    try refuses(
        \\"quant_method": "gptq", "bits": 4, "group_size": 128, "sym": true, "lm_head": true, "dynamic": ["+:.*lm_head$"]
    , "quantization_config.dynamic: not an object, this reader reads its rules");
    try refuses(
        \\"quant_method": "gptq", "bits": 4, "group_size": 128, "sym": true, "lm_head": true, "dynamic": {"lm_head": {"bits": 4}}
    , "quantization_config.dynamic lm_head: no +: or -: prefix");
    try refuses(
        \\"quant_method": "gptq", "bits": 4, "group_size": 128, "sym": true, "lm_head": true, "dynamic": {"+:.*lm_head$": {"bits": 4}, "-:.*linear_attn(.*": {"bits": 4}}
    , "quantization_config.dynamic -:.*linear_attn(.*: outside this reader's matcher");
    try refuses(
        \\"quant_method": "gptq", "bits": 4, "group_size": 128, "sym": true, "lm_head": true, "dynamic": {"+:.*lm_head$": {"bits": 4}, "-:^model.*": {"bits": 4}}
    , "quantization_config.dynamic -:^model.*: outside this reader's matcher");
    try refuses(
        \\"quant_method": "gptq", "bits": 4, "group_size": 128, "sym": true, "lm_head": true, "dynamic": {"+:.*lm_head$": {"bits": 4}, "-:a**": {"bits": 4}}
    , "quantization_config.dynamic -:a**: outside this reader's matcher");
    try refuses(
        \\"quant_method": "gptq", "bits": 4, "group_size": 128, "sym": true, "lm_head": true, "dynamic": {"+:.*lm_head$": {"bits": 4}, "-:a$b": {"bits": 4}}
    , "quantization_config.dynamic -:a$b: outside this reader's matcher");
    try refuses(
        \\"quant_method": "gptq", "bits": 4, "group_size": 128, "sym": true, "lm_head": true, "dynamic": {"+:.*lm_head$": {"bits": 4}, "-:": {"bits": 4}}
    , "quantization_config.dynamic -:: outside this reader's matcher");
    try refuses(
        \\"quant_method": "gptq", "bits": 4, "group_size": 128, "sym": true, "lm_head": true, "dynamic": {"+:.*lm_head$": {"bits": 8}}
    , "quantization_config.dynamic +:.*lm_head$: bits 8, expected 4");
    try refuses(
        \\"quant_method": "gptq", "bits": 4, "group_size": 128, "sym": true, "lm_head": true, "dynamic": {"+:.*lm_head$": {"bits": 4, "group_size": 64}}
    , "quantization_config.dynamic +:.*lm_head$: group_size 64, expected 128");
    try refuses(
        \\"quant_method": "gptq", "bits": 4, "group_size": 128, "sym": true, "lm_head": true, "dynamic": {"+:.*lm_head$": 4}
    , "quantization_config.dynamic +:.*lm_head$: not an object of bits");
}

test "gptq: a config that declares no gptq block is not this reader's, and the head rule may stand alone" {
    var why: Why = .{};
    for ([_][]const u8{
        \\{}
        ,
        \\{"quantization": {"group_size": 32, "bits": 4}}
        ,
        \\{"quantization_config": {"quant_method": "modelopt", "quant_algo": "NVFP4"}}
        ,
        \\{"quantization_config": {"quant_method": "exl3"}}
        ,
        \\[1]
    }) |json| {
        const declared = try gptq.admit(a, json, &why);
        try testing.expect(declared == null);
    }
    // the head is quantized by its rule alone, as the export may leave lm_head out of the config
    var forced = (try gptq.admit(a,
        \\{"quantization_config": {"quant_method": "gptq", "bits": 4, "group_size": 128, "sym": true, "lm_head": false,
        \\ "dynamic": {"+:.*lm_head$": null}}}
    , &why)).?;
    defer forced.deinit();
    try testing.expect(forced.quantized("language_model.lm_head"));
    // a rule set with no exclusions quantizes whatever it is asked about
    var all = (try gptq.admit(a,
        \\{"text_config": {"quantization_config": {"quant_method": "gptq", "bits": 4, "group_size": 128, "sym": true,
        \\ "lm_head": true, "dynamic": {}}}}
    , &why)).?;
    defer all.deinit();
    try testing.expectEqual(@as(usize, 0), all.rules.len);
    try testing.expect(all.quantized("model.language_model.layers.0.linear_attn.in_proj_qkv"));
    // GPTQ spelled in another case is the same method
    var upper = (try gptq.admit(a, jsonOf(
        \\"quant_method": "GPTQ", "bits": 4, "group_size": 128, "sym": true, "lm_head": true, "dynamic": {"+:.*lm_head$": {"bits": 4}}
    ), &why)).?;
    upper.deinit();
}

test "gptq: the last rule matching a module decides, and no rule leaves it quantized" {
    var why: Why = .{};
    var cfg = (try gptq.admit(a,
        \\{"quantization_config": {"quant_method": "gptq", "bits": 4, "group_size": 128, "sym": true, "lm_head": true,
        \\ "dynamic": {"-:.*experts.*": {"bits": 4}, "+:.*gate_proj$": {"bits": 4}}}}
    , &why)).?;
    defer cfg.deinit();
    try testing.expect(!cfg.quantized("m.experts.0.down_proj"));
    try testing.expect(cfg.quantized("m.experts.0.gate_proj"));
    var other = (try gptq.admit(a,
        \\{"quantization_config": {"quant_method": "gptq", "bits": 4, "group_size": 128, "sym": true, "lm_head": true,
        \\ "dynamic": {"+:.*gate_proj$": {"bits": 4}, "-:.*experts.*": {"bits": 4}}}}
    , &why)).?;
    defer other.deinit();
    try testing.expect(!other.quantized("m.experts.0.gate_proj"));
    try testing.expect(!other.quantized("m.experts.0.up_proj"));
    try testing.expect(other.quantized("m.layers.0.mlp.gate_proj"));
}

/// `entries` read as `module` [n, k] fail with `err`, the refusal saying exactly `text`.
fn refusedTensor(entries: []const fixture.Entry, module: []const u8, n: usize, k: usize, err: anyerror, text: []const u8) !void {
    var f: Folder = .{ .tmp = undefined };
    try f.open(entries);
    defer f.close();
    var why: Why = .{};
    try testing.expectError(err, gptq.readWhy(&f.ck, module, n, k, &why));
    try testing.expectEqualStrings(text, why.text());
}

const one_shape = [_]usize{1};
const one_byte = [_]u8{0};

/// `entries` in `buf` with tensor `i` replaced by `e`, or by an unrelated tensor when `e` is null.
fn edited(buf: *[7]fixture.Entry, entries: [6]fixture.Entry, i: usize, e: ?fixture.Entry) []const fixture.Entry {
    @memcpy(buf[0..6], &entries);
    buf[6] = .{ .name = "x.unrelated", .dtype = "U8", .shape = &one_shape, .bytes = &one_byte };
    buf[i] = e orelse buf[6];
    return buf[0..6];
}

test "gptq: every tensor it refuses is named, the zero point and act order included" {
    var s: Synth = .{};
    try s.init();
    defer s.deinit();
    var f: Folder = .{ .tmp = undefined };
    try f.open(&s.entries);
    defer f.close();
    _ = try gptq.read(&f.ck, expert, 8, 256);
    _ = try gptq.read(&f.ck, head, 8, 128);
    var buf: [7]fixture.Entry = undefined;
    const E = error.UnexpectedTensor;
    const Q = error.UnsupportedQuantization;
    try refusedTensor(&s.entries, expert, 5, 256, E, expert ++ ": [n, k] of [5, 256] is not 4-bit groups of 128 in words of 8 codes");
    try refusedTensor(&s.entries, expert, 8, 192, E, expert ++ ": [n, k] of [8, 192] is not 4-bit groups of 128 in words of 8 codes");
    try refusedTensor(&s.entries, expert, 8, 128, E, expert ++ ".qweight: i32 { 32, 8 }, expected i32 { 16, 8 }");
    var wide = [_]usize{ 16, 16 };
    const qw_wide: fixture.Entry = .{ .name = s.entries[0].name, .dtype = "I32", .shape = &wide, .bytes = s.bytes[0] };
    try refusedTensor(edited(&buf, s.entries, 0, qw_wide), expert, 8, 256, E, expert ++ ".qweight: i32 { 16, 16 }, expected i32 { 32, 8 }");
    const qw_u8: fixture.Entry = .{ .name = s.entries[0].name, .dtype = "U8", .shape = &s.shapes[0], .bytes = s.bytes[0][0..256] };
    try refusedTensor(edited(&buf, s.entries, 0, qw_u8), expert, 8, 256, E, expert ++ ".qweight: u8 { 32, 8 }, expected i32 { 32, 8 }");
    var f32_scales: [64]u8 = @splat(0);
    const sc_f32: fixture.Entry = .{ .name = s.entries[1].name, .dtype = "F32", .shape = &s.shapes[1], .bytes = &f32_scales };
    try refusedTensor(edited(&buf, s.entries, 1, sc_f32), expert, 8, 256, E, expert ++ ".scales: f32 { 2, 8 }, expected f16 { 2, 8 }");
    const sc_bf16: fixture.Entry = .{ .name = s.entries[1].name, .dtype = "BF16", .shape = &s.shapes[1], .bytes = s.bytes[1] };
    try refusedTensor(edited(&buf, s.entries, 1, sc_bf16), expert, 8, 256, E, expert ++ ".scales: bf16 { 2, 8 }, expected f16 { 2, 8 }");
    var flipped = [_]usize{ 8, 2 };
    const sc_flipped: fixture.Entry = .{ .name = s.entries[1].name, .dtype = "F16", .shape = &flipped, .bytes = s.bytes[1] };
    try refusedTensor(edited(&buf, s.entries, 1, sc_flipped), expert, 8, 256, E, expert ++ ".scales: f16 { 8, 2 }, expected f16 { 2, 8 }");
    var half = [_]usize{ 2, 2 };
    var wide_zeroes: [16]u8 = @splat(0);
    const qz_wide: fixture.Entry = .{ .name = s.entries[2].name, .dtype = "I32", .shape = &half, .bytes = &wide_zeroes };
    try refusedTensor(edited(&buf, s.entries, 2, qz_wide), expert, 8, 256, E, expert ++ ".qzeros: i32 { 2, 2 }, expected i32 { 2, 1 }");
    const bad = try a.dupe(u8, s.bytes[5]);
    defer a.free(bad);
    bad[3] = 0x87;
    const qz_bad: fixture.Entry = .{ .name = s.entries[5].name, .dtype = "I32", .shape = &s.shapes[5], .bytes = bad };
    try refusedTensor(edited(&buf, s.entries, 5, qz_bad), head, 8, 128, Q, head ++ ".qzeros: byte 3 is 0x87, a symmetric weight's every nibble is 7");
    bad[3] = 0x78;
    try refusedTensor(edited(&buf, s.entries, 5, qz_bad), head, 8, 128, Q, head ++ ".qzeros: byte 3 is 0x78, a symmetric weight's every nibble is 7");
    bad[3] = 0x77;
    bad[0] = 0x00;
    try refusedTensor(edited(&buf, s.entries, 5, qz_bad), head, 8, 128, Q, head ++ ".qzeros: byte 0 is 0x00, a symmetric weight's every nibble is 7");
    // a store that hands back fewer bytes than its header declares (checkpoint_direct's copies) is refused by name
    var copies: std.StringHashMapUnmanaged([]const u8) = .empty;
    defer copies.deinit(a);
    try copies.put(a, s.entries[0].name, s.bytes[0][0..1020]);
    var short: Folder = .{ .tmp = undefined };
    try short.open(&s.entries);
    defer short.close();
    short.ck.copies = &copies;
    var why: Why = .{};
    try testing.expectError(E, gptq.readWhy(&short.ck, expert, 8, 256, &why));
    try testing.expectEqualStrings(expert ++ ".qweight: 1020 bytes, i32 { 32, 8 } needs 1024", why.text());
    try refusedTensor(edited(&buf, s.entries, 2, null), expert, 8, 256, error.MissingTensor, expert ++ ": no qzeros, a GPTQ weight stores it");
    try refusedTensor(edited(&buf, s.entries, 3, null), head, 8, 128, error.MissingTensor, head ++ ": no qweight, a GPTQ weight stores it");
    var order = [_]usize{128};
    var gbytes: [128 * 4]u8 = @splat(0);
    var with_g: [7]fixture.Entry = undefined;
    @memcpy(with_g[0..6], &s.entries);
    with_g[6] = .{ .name = head ++ ".g_idx", .dtype = "I32", .shape = &order, .bytes = &gbytes };
    try refusedTensor(&with_g, head, 8, 128, E, head ++ ".g_idx: i32 { 128 }, act order is not read");
    // an empty g_idx is what a desc_act false export leaves behind: it is read and kept out of the way
    var none = [_]usize{0};
    with_g[6] = .{ .name = head ++ ".g_idx", .dtype = "I32", .shape = &none, .bytes = &.{} };
    var g: Folder = .{ .tmp = undefined };
    try g.open(&with_g);
    defer g.close();
    const hd = try gptq.read(&g.ck, head, 8, 128);
    _ = try gptq.read(&g.ck, expert, 8, 256);
    try testing.expectEqual(@as(u64, @bitCast(@as(f64, -8.0))), @as(u64, @bitCast(gptq.weight(hd, 0, 0))));
    try testing.expectEqual(@as(usize, 0), g.ck.unused());
}

test "gptq: the zero point check reads both nibbles of every word" {
    try testing.expectEqual(@as(?usize, null), gptq.zerosOff(&.{ 0x77, 0x77, 0x77, 0x77 }));
    try testing.expectEqual(@as(?usize, null), gptq.zerosOff(&.{}));
    try testing.expectEqual(@as(?usize, null), gptq.zerosOff(&.{ 0x77, 0x77 }));
    try testing.expectEqual(@as(?usize, 0), gptq.zerosOff(&.{ 0x78, 0x77 }));
    try testing.expectEqual(@as(?usize, 1), gptq.zerosOff(&.{ 0x77, 0x87 }));
    try testing.expectEqual(@as(?usize, 2), gptq.zerosOff(&.{ 0x77, 0x77, 0x00 }));
    try testing.expectEqual(@as(?usize, 1), gptq.zerosOff(&.{ 0x77, 0x7F }));
}
