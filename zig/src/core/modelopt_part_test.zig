//! The shared ModelOpt part reader: shape rules without storage, and file-backed parts of every kind and refusal.
const std = @import("std");
const part = @import("modelopt_part.zig");
const ckpt = @import("checkpoint.zig");
const fixture = @import("checkpoint_fixture.zig");
const Why = @import("modelopt.zig").Why;
const testing = std.testing;
const a = testing.allocator;
const io = testing.io;

/// A tensor's dtype and shape as a header lists it, no storage.
const Meta = struct { dtype: ckpt.DType, rank: usize, shape: [4]usize = @splat(1) };

fn meta(dtype: ckpt.DType, dims: []const usize) Meta {
    var t: Meta = .{ .dtype = dtype, .rank = dims.len };
    for (dims, 0..) |n, i| t.shape[i] = n;
    return t;
}

test "part: every kind's weight shape gives [n, k], NVFP4 two inputs a byte" {
    try testing.expectEqual([2]usize{ 8, 64 }, try part.weightDims(.nvfp4, meta(.u8, &.{ 8, 32 }), .{}));
    try testing.expectEqual([2]usize{ 8, 64 }, try part.weightDims(.fp8, meta(.f8_e4m3, &.{ 8, 64 }), .{}));
    try testing.expectEqual([2]usize{ 8, 64 }, try part.weightDims(.bf16, meta(.bf16, &.{ 8, 64 }), .{ .n = 8, .k = 64 }));
    try testing.expectEqual(part.Kind.nvfp4, part.kindOf(.u8).?);
    try testing.expectEqual(part.Kind.fp8, part.kindOf(.f8_e4m3).?);
    try testing.expectEqual(part.Kind.bf16, part.kindOf(.bf16).?);
    for ([_]ckpt.DType{ .f32, .f16, .f8_e5m2, .f8_e8m0, .u32, .i8 }) |d| try testing.expect(part.kindOf(d) == null);
}

test "part: a weight of another dtype, rank or width is refused for every kind" {
    const E = error.UnexpectedTensor;
    for ([_]part.Kind{ .nvfp4, .fp8, .bf16 }) |kind| {
        for ([_]ckpt.DType{ .u8, .f8_e4m3, .bf16, .f32, .f16 }) |d| {
            const ok = switch (kind) {
                .nvfp4 => d == .u8,
                .fp8 => d == .f8_e4m3,
                .bf16 => d == .bf16,
            };
            if (!ok) try testing.expectError(E, part.weightDims(kind, meta(d, &.{ 8, 32 }), .{}));
        }
        const d: ckpt.DType = switch (kind) {
            .nvfp4 => .u8,
            .fp8 => .f8_e4m3,
            .bf16 => .bf16,
        };
        try testing.expectError(E, part.weightDims(kind, meta(d, &.{256}), .{}));
        try testing.expectError(E, part.weightDims(kind, meta(d, &.{ 2, 8, 32 }), .{}));
        try testing.expectError(E, part.weightDims(kind, meta(d, &.{ 0, 32 }), .{}));
        try testing.expectError(E, part.weightDims(kind, meta(d, &.{ 8, 0 }), .{}));
        const k: usize = if (kind == .nvfp4) 64 else 32;
        try testing.expectError(E, part.weightDims(kind, meta(d, &.{ 8, 32 }), .{ .n = 9 }));
        try testing.expectError(E, part.weightDims(kind, meta(d, &.{ 8, 32 }), .{ .k = k + 16 }));
        _ = try part.weightDims(kind, meta(d, &.{ 8, 32 }), .{ .n = 8, .k = k });
    }
}

test "part: NVFP4 blocks are e4m3 [n, k / 16], nothing else" {
    const E = error.UnexpectedTensor;
    try part.checkBlocks(meta(.f8_e4m3, &.{ 8, 4 }), 8, 64);
    try testing.expectError(E, part.checkBlocks(meta(.u8, &.{ 8, 4 }), 8, 64));
    try testing.expectError(E, part.checkBlocks(meta(.f32, &.{ 8, 4 }), 8, 64));
    try testing.expectError(E, part.checkBlocks(meta(.f8_e4m3, &.{ 8, 5 }), 8, 64));
    try testing.expectError(E, part.checkBlocks(meta(.f8_e4m3, &.{ 8, 2 }), 8, 64)); // a group of 32
    try testing.expectError(E, part.checkBlocks(meta(.f8_e4m3, &.{ 9, 4 }), 8, 64));
    try testing.expectError(E, part.checkBlocks(meta(.f8_e4m3, &.{32}), 8, 64));
    try testing.expectError(E, part.checkBlocks(meta(.f8_e4m3, &.{ 1, 8, 4 }), 8, 64));
    try testing.expectError(E, part.checkBlocks(meta(.f8_e4m3, &.{ 8, 2 }), 8, 40)); // k not whole groups
}

const one = std.mem.toBytes(@as(f32, 0.5));
const two = std.mem.toBytes(@as(f32, 2.0));
const zero = std.mem.toBytes(@as(f32, 0));
const nan = std.mem.toBytes(std.math.nan(f32));
const codes: [8 * 32]u8 = @splat(0x21);
const blocks: [8 * 4]u8 = @splat(0x38);
const fp8: [8 * 64]u8 = @splat(0x40);
const bf16: [8 * 64 * 2]u8 = @splat(0x3f);

/// m: NVFP4 with input_scale; f: FP8 with input_scale; b: bf16; s: NVFP4 without input_scale.
fn good() [12]fixture.Entry {
    return .{
        .{ .name = "m.weight", .shape = &.{ 8, 32 }, .bytes = &codes },
        .{ .name = "m.weight_scale", .dtype = "F8_E4M3", .shape = &.{ 8, 4 }, .bytes = &blocks },
        .{ .name = "m.weight_scale_2", .dtype = "F32", .shape = &.{}, .bytes = &one },
        .{ .name = "m.input_scale", .dtype = "F32", .shape = &.{}, .bytes = &two },
        .{ .name = "f.weight", .dtype = "F8_E4M3", .shape = &.{ 8, 64 }, .bytes = &fp8 },
        .{ .name = "f.weight_scale", .dtype = "F32", .shape = &.{1}, .bytes = &two },
        .{ .name = "f.input_scale", .dtype = "F32", .shape = &.{}, .bytes = &one },
        .{ .name = "b.weight", .dtype = "BF16", .shape = &.{ 8, 64 }, .bytes = &bf16 },
        .{ .name = "s.weight", .shape = &.{ 8, 32 }, .bytes = &codes },
        .{ .name = "s.weight_scale", .dtype = "F8_E4M3", .shape = &.{ 8, 4 }, .bytes = &blocks },
        .{ .name = "s.weight_scale_2", .dtype = "F32", .shape = &.{}, .bytes = &one },
        .{ .name = "z.weight", .dtype = "F32", .shape = &.{ 8, 32 }, .bytes = &bf16 },
    };
}

/// A checkpoint folder holding `entries` as one file.
const Folder = struct {
    tmp: testing.TmpDir,
    path: [96]u8 = undefined,
    ck: ckpt.Checkpoint = undefined,

    fn open(f: *Folder, entries: []const fixture.Entry) !void {
        f.tmp = testing.tmpDir(.{});
        errdefer f.tmp.cleanup();
        try fixture.write(a, io, f.tmp.dir, "model.safetensors", entries);
        f.ck = try ckpt.Checkpoint.openModel(a, io, try std.fmt.bufPrint(&f.path, ".zig-cache/tmp/{s}", .{f.tmp.sub_path}));
    }

    fn close(f: *Folder) void {
        f.ck.close();
        f.tmp.cleanup();
    }
};

test "part: each kind read from a file, views into the map, its scales and the input scale" {
    var f: Folder = .{ .tmp = undefined };
    try f.open(&good());
    defer f.close();
    const m = try part.read(&f.ck, "m", .{});
    try testing.expectEqual(part.Kind.nvfp4, m.kind);
    try testing.expectEqual([2]usize{ 8, 64 }, [2]usize{ m.n, m.k });
    try testing.expectEqual(@as(f32, 0.5), m.scale);
    try testing.expectEqual(@as(?f32, 2.0), m.act);
    try testing.expect(m.weight.bytes.ptr == (try f.ck.get("m.weight")).bytes.ptr); // a view, no copy
    try testing.expectEqualSlices(u8, &blocks, m.blocks.?.bytes);
    const fp = try part.read(&f.ck, "f", .{ .kind = .fp8, .n = 8, .k = 64, .act = .required });
    try testing.expectEqual(@as(f32, 2.0), fp.scale);
    try testing.expectEqual(@as(?f32, 0.5), fp.act);
    try testing.expect(fp.blocks == null);
    const b = try part.read(&f.ck, "b", .{});
    try testing.expectEqual(part.Kind.bf16, b.kind);
    try testing.expectEqual(@as(f32, 1), b.scale);
    try testing.expect(b.act == null and b.blocks == null);
    const s = try part.read(&f.ck, "s", .{ .kind = .nvfp4, .positive = true });
    try testing.expect(s.act == null);
    _ = try f.ck.get("z.weight");
    try testing.expectEqual(@as(usize, 0), f.ck.unused());
}

test "part: act .skip leaves input_scale unread and unmarked; .required refuses a missing one by name" {
    var f: Folder = .{ .tmp = undefined };
    try f.open(&good());
    defer f.close();
    const m = try part.read(&f.ck, "m", .{ .act = .skip });
    try testing.expect(m.act == null);
    for ([_][]const u8{ "f", "b", "s" }) |x| _ = try part.read(&f.ck, x, .{ .act = .skip });
    for ([_][]const u8{ "m.input_scale", "f.input_scale" }) |x| try testing.expect(!f.ck.used.contains(x));
    try testing.expect(f.ck.used.contains("s.weight_scale_2"));
    var why: Why = .{};
    try testing.expectError(error.MissingTensor, part.readWhy(&f.ck, "s", .{ .act = .required }, &why));
    try testing.expectEqualStrings("s: no input_scale, a ModelOpt part needs it", why.text());
}

/// `entries` with `name` replaced by `with` (null: dropped).
fn edited(name: []const u8, with: ?fixture.Entry) ![12]fixture.Entry {
    var out = good();
    for (&out) |*e| if (std.mem.eql(u8, e.name, name)) {
        e.* = with orelse .{ .name = "x.unrelated", .shape = &.{1}, .bytes = codes[0..1] };
        return out;
    };
    return error.NoSuchEntry;
}

/// `module` read from `entries` with `want` fails with `err`, the refusal naming `text`.
fn refused(entries: [12]fixture.Entry, module: []const u8, want: part.Want, err: anyerror, text: []const u8) !void {
    var f: Folder = .{ .tmp = undefined };
    try f.open(&entries);
    defer f.close();
    var why: Why = .{};
    try testing.expectError(err, part.readWhy(&f.ck, module, want, &why));
    try testing.expectEqualStrings(text, why.text());
}

test "part: a missing scale or weight is refused by name" {
    try refused(try edited("m.weight_scale", null), "m", .{}, error.MissingTensor, "m: no weight_scale, a ModelOpt part needs it");
    try refused(try edited("m.weight_scale_2", null), "m", .{}, error.MissingTensor, "m: no weight_scale_2, a ModelOpt part needs it");
    try refused(try edited("f.weight_scale", null), "f", .{}, error.MissingTensor, "f: no weight_scale, a ModelOpt part needs it");
    try refused(good(), "q", .{}, error.MissingTensor, "q: no weight, a ModelOpt part needs it");
}

test "part: a wrong dtype is refused by name" {
    const E = error.UnexpectedTensor;
    try refused(good(), "z", .{}, E, "z.weight: f32 { 8, 32 }, not a ModelOpt NVFP4, FP8 or bf16 part of [null, null]");
    try refused(good(), "m", .{ .kind = .fp8 }, E, "m.weight: u8 { 8, 32 }, not a ModelOpt fp8 part of [null, null]");
    const bad_blocks: fixture.Entry = .{ .name = "m.weight_scale", .shape = &.{ 8, 4 }, .bytes = &blocks };
    try refused(try edited("m.weight_scale", bad_blocks), "m", .{}, E, "m.weight_scale: u8 { 8, 4 }, not a ModelOpt NVFP4, FP8 or bf16 part of [null, null]");
    const bf_scale: fixture.Entry = .{ .name = "m.weight_scale_2", .dtype = "BF16", .shape = &.{2}, .bytes = &one };
    try refused(try edited("m.weight_scale_2", bf_scale), "m", .{}, E, "m.weight_scale_2: bf16 { 2 }, expected an F32 scalar");
    const int_scale: fixture.Entry = .{ .name = "m.weight_scale_2", .dtype = "I32", .shape = &.{}, .bytes = &one };
    var f: Folder = .{ .tmp = undefined };
    try f.open(&try edited("m.weight_scale_2", int_scale));
    defer f.close();
    var why: Why = .{};
    try testing.expectError(E, part.readWhy(&f.ck, "m", .{}, &why)); // four bytes, one element, not f32
    try testing.expect(std.mem.startsWith(u8, why.text(), "m.weight_scale_2: i32 "));
    const bf_act: fixture.Entry = .{ .name = "f.input_scale", .dtype = "BF16", .shape = &.{2}, .bytes = &one };
    try refused(try edited("f.input_scale", bf_act), "f", .{}, E, "f.input_scale: bf16 { 2 }, expected an F32 scalar");
}

test "part: a wrong shape is refused by name" {
    const E = error.UnexpectedTensor;
    try refused(good(), "m", .{ .n = 16 }, E, "m.weight: u8 { 8, 32 }, not a ModelOpt NVFP4, FP8 or bf16 part of [16, null]");
    try refused(good(), "b", .{ .kind = .bf16, .k = 128 }, E, "b.weight: bf16 { 8, 64 }, not a ModelOpt bf16 part of [null, 128]");
    const wide: fixture.Entry = .{ .name = "m.weight_scale", .dtype = "F8_E4M3", .shape = &.{ 4, 8 }, .bytes = &blocks };
    try refused(try edited("m.weight_scale", wide), "m", .{}, E, "m.weight_scale: f8_e4m3 { 4, 8 }, not a ModelOpt NVFP4, FP8 or bf16 part of [null, null]");
    const pair: fixture.Entry = .{ .name = "f.weight_scale", .dtype = "F32", .shape = &.{2}, .bytes = &(one ++ two) };
    try refused(try edited("f.weight_scale", pair), "f", .{}, E, "f.weight_scale: f32 { 2 }, expected an F32 scalar");
}

test "part: positive refuses a zero or NaN global scale by name; without it they load" {
    for ([_]*const [4]u8{ &zero, &nan }) |v| {
        const g: fixture.Entry = .{ .name = "s.weight_scale_2", .dtype = "F32", .shape = &.{}, .bytes = v };
        const entries = try edited("s.weight_scale_2", g);
        var f: Folder = .{ .tmp = undefined };
        try f.open(&entries);
        defer f.close();
        var why: Why = .{};
        try testing.expectError(error.UnexpectedTensor, part.readWhy(&f.ck, "s", .{ .positive = true }, &why));
        try testing.expect(std.mem.startsWith(u8, why.text(), "s.weight_scale_2: "));
        try testing.expect(std.mem.endsWith(u8, why.text(), " is not a positive finite scale"));
        _ = try part.readWhy(&f.ck, "s", .{}, &why);
    }
}

test "part: rank0 refuses a scale of shape [1] or [1, 1] by name; the default takes it" {
    for ([_][]const usize{ &.{1}, &.{ 1, 1 } }) |shape| {
        for ([_][]const u8{ "m.weight_scale_2", "m.input_scale" }) |name| {
            const g: fixture.Entry = .{ .name = name, .dtype = "F32", .shape = shape, .bytes = &one };
            var f: Folder = .{ .tmp = undefined };
            try f.open(&try edited(name, g));
            defer f.close();
            var why: Why = .{};
            try testing.expectError(error.UnexpectedTensor, part.readWhy(&f.ck, "m", .{ .scalars = .rank0 }, &why));
            try testing.expect(std.mem.startsWith(u8, why.text(), name));
            try testing.expect(std.mem.endsWith(u8, why.text(), "expected an F32 scalar of shape []"));
            const p = try part.readWhy(&f.ck, "m", .{}, &why);
            try testing.expectEqual(@as(f32, 0.5), if (name[2] == 'w') p.scale else p.act.?);
        }
    }
    var f: Folder = .{ .tmp = undefined };
    try f.open(&good());
    defer f.close();
    _ = try part.read(&f.ck, "m", .{ .scalars = .rank0 }); // shape [] passes
}
