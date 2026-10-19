//! Which stored format a GLM CUDA projection is read in (its recipe, checked against its tensors); csf scales decoded.

const std = @import("std");
const core = @import("core");

const Meta = core.nvfp4.Meta;
const Recipe = core.modelopt.Recipe;

/// A projection's stored format, as the loader reads it.
pub const Format = enum { bf16, nvfp4, nvfp4_csf, mxfp8, fp8_pb };

/// A projection's tensors as stored: its weight, its scale when it has one, and whether csf_32x2_v1 parts replace it.
pub const Stored = struct {
    weight: Meta,
    scale: ?Meta = null,
    csf: bool = false,
};

/// How a checkpoint names its formats: one NVFP4 for every quantized linear, or MIXED_PRECISION's recipe per module.
pub const Declared = union(enum) {
    nvfp4,
    mixed: ?Recipe,
};

/// The format of `stored`, or error.FormatMismatch when its tensors are not what `declared` says they are.
pub fn of(declared: Declared, stored: Stored) error{FormatMismatch}!Format {
    const f = try fromTensors(stored);
    const ok = switch (declared) {
        .nvfp4 => f == .bf16 or f == .nvfp4,
        .mixed => |r| if (r) |recipe| switch (recipe) {
            .nvfp4, .w4a16_nvfp4 => f == .nvfp4,
            .nvfp4_csf => f == .nvfp4_csf,
            .mxfp8 => f == .mxfp8,
            .fp8_pb_wo => f == .fp8_pb,
        } else f == .bf16,
    };
    return if (ok) f else error.FormatMismatch;
}

fn fromTensors(s: Stored) error{FormatMismatch}!Format {
    const w = s.weight;
    if (w.shape.len != 2) return error.FormatMismatch;
    const n = w.shape[0];
    switch (w.dtype) {
        .bf16 => return if (s.scale == null and !s.csf) .bf16 else error.FormatMismatch,
        .u8 => {
            if (s.csf) return if (s.scale == null) .nvfp4_csf else error.FormatMismatch;
            const sc = s.scale orelse return error.FormatMismatch;
            const k = 2 * w.shape[1];
            const fits = sc.dtype == .f8_e4m3 and sc.shape.len == 2 and sc.shape[0] == n and sc.shape[1] * core.nvfp4.group == k;
            return if (fits) .nvfp4 else error.FormatMismatch;
        },
        .f8_e4m3 => {
            const sc = s.scale orelse return error.FormatMismatch;
            const k = w.shape[1];
            if (sc.dtype == .u8) return if (k % 32 == 0 and sc.shape.len == 2 and sc.shape[0] == n and sc.shape[1] == k / 32) .mxfp8 else error.FormatMismatch;
            if (sc.dtype != .f32) return error.FormatMismatch;
            const rb = (n + 127) / 128;
            const cb = (k + 127) / 128;
            const blocks = switch (sc.shape.len) {
                2 => sc.shape[0] == rb and sc.shape[1] == cb,
                4 => sc.shape[0] == rb and sc.shape[1] == 1 and sc.shape[2] == cb and sc.shape[3] == 1,
                else => false,
            };
            return if (blocks) .fp8_pb else error.FormatMismatch;
        },
        else => return error.FormatMismatch,
    }
}

/// `module`'s NVFP4 linear whose block scales are csf_32x2_v1 parts, decoded into `scales` ([n, k/16] e4m3).
pub fn csfLinear(ck: *core.Checkpoint, module: []const u8, scales: []u8) !core.nvfp4.Linear {
    var buf: [256]u8 = undefined;
    const w = try ck.get(try std.fmt.bufPrint(&buf, "{s}.weight", .{module}));
    if (w.dtype != .u8 or w.rank != 2) return error.BadNvfp4;
    const n = w.shape[0];
    const k = 2 * w.shape[1];
    if (k % core.nvfp4.group != 0 or scales.len != n * k / core.nvfp4.group) return error.BadNvfp4;
    var p: core.nvfp4_csf32.Parts = undefined;
    inline for (.{ "bases", "codes", "page_bases", "offsets", "exceptions" }, .{ .u8, .u8, .u32, .u16, .u16 }) |part, dtype| {
        const t = try ck.get(try std.fmt.bufPrint(&buf, "{s}.weight_scale.csf32_" ++ part, .{module}));
        if (t.dtype != dtype or t.rank != 1) return error.BadNvfp4;
        @field(p, part) = t.bytes;
    }
    try core.nvfp4_csf32.decode(p, n, k / core.nvfp4.group, scales);
    const g = try ck.get(try std.fmt.bufPrint(&buf, "{s}.weight_scale_2", .{module}));
    if (g.dtype != .f32 or g.numel() != 1) return error.BadNvfp4;
    _ = try ck.get(try std.fmt.bufPrint(&buf, "{s}.input_scale", .{module})); // the W4A4 math's, unread by W4A16
    return .{ .n = n, .k = k, .codes = w.bytes, .scales = scales, .global = std.mem.bytesToValue(f32, g.bytes[0..4]) };
}

/// `module`'s NVFP4 linear, scales stored plain; only W4A16_NVFP4 may lack the input_scale (FP4-Local's MTP experts).
pub fn plainLinear(ck: *core.Checkpoint, module: []const u8, declared: Declared) !core.nvfp4.Linear {
    var buf: [256]u8 = undefined;
    if (ck.has(try std.fmt.bufPrint(&buf, "{s}.input_scale", .{module}))) return core.nvfp4.Linear.read(ck, module);
    const w4a16 = switch (declared) {
        .nvfp4 => false,
        .mixed => |r| if (r) |recipe| recipe == .w4a16_nvfp4 else false,
    };
    if (!w4a16) return error.NoInputScale;
    const w = try ck.get(try std.fmt.bufPrint(&buf, "{s}.weight", .{module}));
    const s = try ck.get(try std.fmt.bufPrint(&buf, "{s}.weight_scale", .{module}));
    const g = try ck.get(try std.fmt.bufPrint(&buf, "{s}.weight_scale_2", .{module}));
    const d = try core.nvfp4.dims(metaOf(&w), metaOf(&s), metaOf(&g), null);
    return .{ .n = d.n, .k = d.k, .codes = w.bytes, .scales = s.bytes, .global = std.mem.bytesToValue(f32, g.bytes[0..4]) };
}

fn metaOf(t: *const core.checkpoint.Tensor) Meta {
    return .{ .dtype = t.dtype, .shape = t.shape[0..t.rank] };
}

/// The refusal's text: the module, what the checkpoint declares and what its weight holds.
pub fn describe(buf: []u8, module: []const u8, declared: Declared, stored: Stored) []const u8 {
    const said = switch (declared) {
        .nvfp4 => "NVFP4",
        .mixed => |r| if (r) |recipe| @tagName(recipe) else "no recipe (bf16)",
    };
    return std.fmt.bufPrint(buf, "{s}: declared {s}, stored {s} {any}", .{ module, said, @tagName(stored.weight.dtype), stored.weight.shape }) catch buf;
}

/// The entry's check (modelopt.admitRecipes), its table dropped: weights.load reads it again beside the tensors.
pub fn admit(gpa: std.mem.Allocator, config_json: []const u8, why: *core.modelopt.Why) !void {
    var recipes = try core.modelopt.admitRecipes(gpa, config_json, why);
    recipes.deinit();
}

/// A routed expert's NVFP4 linear: scales as stored, or csf_32x2_v1's decoded into `scales`; else refused by name.
pub fn expert(L: anytype, module: []const u8, scales: []u8) !core.nvfp4.Linear {
    var buf: [224]u8 = undefined;
    const csf = L.ck.has(try std.fmt.bufPrint(&buf, "{s}.weight_scale.csf32_bases", .{module}));
    try L.formatIn(L.ck, module, if (csf) .nvfp4_csf else .nvfp4);
    if (!csf) return plainLinear(L.ck, module, L.declaredAs(module)) catch |e| switch (e) {
        error.NoInputScale => {
            std.log.err("{s}: declared NVFP4 (W4A4) but stores no input_scale", .{module});
            return error.UnreadableFormat;
        },
        else => return e,
    };
    return csfLinear(L.ck, module, scales) catch |e| {
        std.log.err("{s}: its csf_32x2_v1 scales do not read as docs/csf-scale-blocks.md has them ({t})", .{ module, e });
        return e;
    };
}

/// config.json's MIXED_PRECISION recipes, or null when it declares none this reader takes (the entry refuses those).
pub fn readRecipes(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !?core.modelopt.Recipes {
    const path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
    defer gpa.free(path);
    const text = std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 24)) catch return null;
    defer gpa.free(text);
    var why: core.modelopt.Why = .{};
    return core.modelopt.admitRecipes(gpa, text, &why) catch null;
}

const testing = std.testing;

fn meta(dtype: core.checkpoint.DType, shape: []const usize) Meta {
    return .{ .dtype = dtype, .shape = shape };
}

test "a recipe-less module of a MIXED_PRECISION checkpoint is bf16, and a plain NVFP4 one may be either" {
    const w = meta(.bf16, &.{ 8192, 4096 });
    try testing.expectEqual(Format.bf16, try of(.{ .mixed = null }, .{ .weight = w }));
    try testing.expectEqual(Format.bf16, try of(.nvfp4, .{ .weight = w }));
    const q = Stored{ .weight = meta(.u8, &.{ 2048, 2048 }), .scale = meta(.f8_e4m3, &.{ 2048, 256 }) };
    try testing.expectEqual(Format.nvfp4, try of(.nvfp4, q));
    try testing.expectError(error.FormatMismatch, of(.{ .mixed = null }, q));
    try testing.expectError(error.FormatMismatch, of(.{ .mixed = null }, .{ .weight = meta(.f8_e4m3, &.{ 64, 64 }), .scale = meta(.u8, &.{ 64, 2 }) }));
}

test "nvidia/GLM-5.3-Flash-FP4-Local's projections, by their recipes and their shapes" {
    const kda_q = Stored{ .weight = meta(.f8_e4m3, &.{ 8192, 4096 }), .scale = meta(.u8, &.{ 8192, 128 }) };
    try testing.expectEqual(Format.mxfp8, try of(.{ .mixed = .mxfp8 }, kda_q));
    const kv_a = Stored{ .weight = meta(.f8_e4m3, &.{ 512, 4096 }), .scale = meta(.f32, &.{ 4, 1, 32, 1 }) };
    try testing.expectEqual(Format.fp8_pb, try of(.{ .mixed = .fp8_pb_wo }, kv_a));
    const mla_o = Stored{ .weight = meta(.f8_e4m3, &.{ 4096, 16384 }), .scale = meta(.f32, &.{ 32, 1, 128, 1 }) };
    try testing.expectEqual(Format.fp8_pb, try of(.{ .mixed = .fp8_pb_wo }, mla_o));
    const routed = Stored{ .weight = meta(.u8, &.{ 2048, 2048 }), .csf = true };
    try testing.expectEqual(Format.nvfp4_csf, try of(.{ .mixed = .nvfp4_csf }, routed));
    const mtp_expert = Stored{ .weight = meta(.u8, &.{ 4096, 1024 }), .scale = meta(.f8_e4m3, &.{ 4096, 128 }) };
    try testing.expectEqual(Format.nvfp4, try of(.{ .mixed = .w4a16_nvfp4 }, mtp_expert));
    const kv_b = Stored{ .weight = meta(.bf16, &.{ 32768, 512 }) };
    try testing.expectEqual(Format.bf16, try of(.{ .mixed = null }, kv_b));
}

test "a recipe the tensors do not hold is refused, and the refusal names the module" {
    const bf = Stored{ .weight = meta(.bf16, &.{ 8192, 4096 }) };
    try testing.expectError(error.FormatMismatch, of(.{ .mixed = .mxfp8 }, bf));
    const pb = Stored{ .weight = meta(.f8_e4m3, &.{ 512, 4096 }), .scale = meta(.f32, &.{ 4, 1, 32, 1 }) };
    try testing.expectError(error.FormatMismatch, of(.{ .mixed = .mxfp8 }, pb));
    try testing.expectError(error.FormatMismatch, of(.nvfp4, pb));
    var buf: [160]u8 = undefined;
    const why = describe(&buf, "model.language_model.layers.0.self_attn.q_proj", .{ .mixed = .mxfp8 }, bf);
    try testing.expect(std.mem.startsWith(u8, why, "model.language_model.layers.0.self_attn.q_proj: declared mxfp8, stored bf16"));
}

test "scales of the wrong shape or kind are refused" {
    const wide = meta(.f8_e4m3, &.{ 64, 96 });
    try testing.expectError(error.FormatMismatch, of(.{ .mixed = .mxfp8 }, .{ .weight = wide, .scale = meta(.u8, &.{ 64, 2 }) }));
    try testing.expectEqual(Format.mxfp8, try of(.{ .mixed = .mxfp8 }, .{ .weight = wide, .scale = meta(.u8, &.{ 64, 3 }) }));
    try testing.expectError(error.FormatMismatch, of(.{ .mixed = .mxfp8 }, .{ .weight = meta(.f8_e4m3, &.{ 64, 80 }), .scale = meta(.u8, &.{ 64, 2 }) }));
    const partial = meta(.f8_e4m3, &.{ 130, 260 });
    try testing.expectEqual(Format.fp8_pb, try of(.{ .mixed = .fp8_pb_wo }, .{ .weight = partial, .scale = meta(.f32, &.{ 2, 3 }) }));
    try testing.expectError(error.FormatMismatch, of(.{ .mixed = .fp8_pb_wo }, .{ .weight = partial, .scale = meta(.f32, &.{ 1, 1, 3, 1 }) }));
    try testing.expectError(error.FormatMismatch, of(.{ .mixed = .fp8_pb_wo }, .{ .weight = partial, .scale = meta(.bf16, &.{ 2, 3 }) }));
    try testing.expectError(error.FormatMismatch, of(.nvfp4, .{ .weight = meta(.u8, &.{ 64, 32 }), .scale = meta(.f8_e4m3, &.{ 64, 2 }) }));
    try testing.expectError(error.FormatMismatch, of(.{ .mixed = .nvfp4_csf }, .{ .weight = meta(.u8, &.{ 64, 32 }), .scale = meta(.f8_e4m3, &.{ 64, 4 }), .csf = true }));
}

test "csf_32x2_v1 scales read as the same NVFP4 linear as the scales stored plain, every part used" {
    const gpa = testing.allocator;
    const n = 64;
    const k = 96; // 6 scale columns: three blocks wide
    var prng: std.Random.DefaultPrng = .init(5);
    const rng = prng.random();
    var codes: [n * k / 2]u8 = undefined;
    rng.bytes(&codes);
    var scale: [n * k / 16]u8 = undefined;
    for (&scale, 0..) |*s, j| s.* = if (j % 37 == 0) 0x7e else 0x30 + rng.uintLessThan(u8, 16); // a band and exceptions
    const enc = try core.nvfp4_csf32.encode(gpa, &scale, n, k / 16);
    defer enc.deinit(gpa);
    const one = std.mem.toBytes(@as(f32, 0.25));
    const lens = [_]usize{ enc.bases.len, enc.codes.len, enc.page_bases.len / 4, enc.offsets.len / 2, enc.exceptions.len / 2 };
    const tensors = [_]core.shard_edit.New{
        .{ .name = "a.weight", .dtype = .u8, .shape = &.{ n, k / 2 }, .bytes = &codes },
        .{ .name = "a.weight_scale", .dtype = .f8_e4m3, .shape = &.{ n, k / 16 }, .bytes = &scale },
        .{ .name = "a.weight_scale_2", .dtype = .f32, .shape = &.{}, .bytes = &one },
        .{ .name = "a.input_scale", .dtype = .f32, .shape = &.{}, .bytes = &one },
        .{ .name = "b.weight", .dtype = .u8, .shape = &.{ n, k / 2 }, .bytes = &codes },
        .{ .name = "b.weight_scale.csf32_bases", .dtype = .u8, .shape = lens[0..1], .bytes = enc.bases },
        .{ .name = "b.weight_scale.csf32_codes", .dtype = .u8, .shape = lens[1..2], .bytes = enc.codes },
        .{ .name = "b.weight_scale.csf32_page_bases", .dtype = .u32, .shape = lens[2..3], .bytes = enc.page_bases },
        .{ .name = "b.weight_scale.csf32_offsets", .dtype = .u16, .shape = lens[3..4], .bytes = enc.offsets },
        .{ .name = "b.weight_scale.csf32_exceptions", .dtype = .u16, .shape = lens[4..5], .bytes = enc.exceptions },
        .{ .name = "b.weight_scale_2", .dtype = .f32, .shape = &.{}, .bytes = &one },
        .{ .name = "b.input_scale", .dtype = .f32, .shape = &.{}, .bytes = &one },
    };
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var dir: [64]u8 = undefined;
    const at = try std.fmt.bufPrint(&dir, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    var path: [96]u8 = undefined;
    try core.shard_edit.create(gpa, try std.fmt.bufPrint(&path, "{s}/model.safetensors", .{at}), &.{}, &tensors);
    var ck = try core.Checkpoint.openModel(gpa, testing.io, at);
    defer ck.close();
    const want = try core.nvfp4.Linear.read(&ck, "a");
    var decoded: [n * k / 16]u8 = undefined;
    const got = try csfLinear(&ck, "b", &decoded);
    try testing.expectEqual(want.n, got.n);
    try testing.expectEqual(want.k, got.k);
    try testing.expectEqual(want.global, got.global);
    try testing.expectEqualSlices(u8, want.codes, got.codes);
    try testing.expectEqualSlices(u8, want.scales, got.scales);
    try testing.expect(enc.exceptions.len > 0);
    try testing.expectEqual(@as(usize, 0), ck.unused());
    try testing.expectError(error.BadNvfp4, csfLinear(&ck, "b", decoded[1..]));
}

test "a W4A16_NVFP4 linear reads without an input_scale, which NVFP4's W4A4 recipe requires" {
    const gpa = testing.allocator;
    var codes: [4 * 16]u8 = undefined; // n = 4, k = 32
    for (&codes, 0..) |*c, j| c.* = @truncate(j * 7);
    const scale = [_]u8{ 0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37 };
    const half = std.mem.toBytes(@as(f32, 0.5));
    const tensors = [_]core.shard_edit.New{
        .{ .name = "m.weight", .dtype = .u8, .shape = &.{ 4, 16 }, .bytes = &codes },
        .{ .name = "m.weight_scale", .dtype = .f8_e4m3, .shape = &.{ 4, 2 }, .bytes = &scale },
        .{ .name = "m.weight_scale_2", .dtype = .f32, .shape = &.{}, .bytes = &half },
        .{ .name = "a.weight", .dtype = .u8, .shape = &.{ 4, 16 }, .bytes = &codes },
        .{ .name = "a.weight_scale", .dtype = .f8_e4m3, .shape = &.{ 4, 2 }, .bytes = &scale },
        .{ .name = "a.weight_scale_2", .dtype = .f32, .shape = &.{}, .bytes = &half },
        .{ .name = "a.input_scale", .dtype = .f32, .shape = &.{}, .bytes = &half },
    };
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var dir: [64]u8 = undefined;
    const at = try std.fmt.bufPrint(&dir, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    var path: [96]u8 = undefined;
    try core.shard_edit.create(gpa, try std.fmt.bufPrint(&path, "{s}/model.safetensors", .{at}), &.{}, &tensors);
    var ck = try core.Checkpoint.openModel(gpa, testing.io, at);
    defer ck.close();
    try testing.expectError(error.NoInputScale, plainLinear(&ck, "m", .nvfp4));
    try testing.expectError(error.NoInputScale, plainLinear(&ck, "m", .{ .mixed = .nvfp4 }));
    const got = try plainLinear(&ck, "m", .{ .mixed = .w4a16_nvfp4 });
    try testing.expectEqual(@as(usize, 4), got.n);
    try testing.expectEqual(@as(usize, 32), got.k);
    try testing.expectEqual(@as(f32, 0.5), got.global);
    try testing.expectEqualSlices(u8, &codes, got.codes);
    try testing.expectEqualSlices(u8, &scale, got.scales);
    const stored = try plainLinear(&ck, "a", .{ .mixed = .w4a16_nvfp4 }); // an input_scale it does store is read too
    try testing.expectEqualSlices(u8, got.scales, stored.scales);
    _ = try plainLinear(&ck, "a", .nvfp4);
    try testing.expectEqual(@as(usize, 0), ck.unused());
}
