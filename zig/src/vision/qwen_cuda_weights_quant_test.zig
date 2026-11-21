//! NVIDIA's ModelOpt vision tower (MXFP8 attention, NVFP4 MLP and merger) through the real reader, no CUDA libraries.
const std = @import("std");
const core = @import("core");
const weights = @import("qwen_cuda_weights.zig");
const fixture = @import("qwen_cuda_weights_fixture.zig");
const diagnostics = @import("qwen_weights_diagnostics");
const convert = core.quant.convert;
const fp8 = core.quant.fp8;
const nvfp4 = core.quant.nvfp4;

const config: fixture.Config = .{ .hidden = 32, .intermediate = 16, .depth = 1, .out_hidden = 4 };
const Form = enum { dense, mxfp8, nvfp4 };
const Spec = struct { suffix: []const u8, shape: []const usize, form: Form = .dense };
const tower = [_]Spec{
    .{ .suffix = "patch_embed.proj.weight", .shape = &.{ 32, 3, 2, 16, 16 } },
    .{ .suffix = "patch_embed.proj.bias", .shape = &.{32} },
    .{ .suffix = "pos_embed.weight", .shape = &.{ 2304, 32 } },
    .{ .suffix = "blocks.0.norm1.weight", .shape = &.{32} },
    .{ .suffix = "blocks.0.norm1.bias", .shape = &.{32} },
    .{ .suffix = "blocks.0.attn.qkv.weight", .shape = &.{ 96, 32 }, .form = .mxfp8 },
    .{ .suffix = "blocks.0.attn.qkv.bias", .shape = &.{96} },
    .{ .suffix = "blocks.0.attn.proj.weight", .shape = &.{ 32, 32 }, .form = .mxfp8 },
    .{ .suffix = "blocks.0.attn.proj.bias", .shape = &.{32} },
    .{ .suffix = "blocks.0.norm2.weight", .shape = &.{32} },
    .{ .suffix = "blocks.0.norm2.bias", .shape = &.{32} },
    .{ .suffix = "blocks.0.mlp.linear_fc1.weight", .shape = &.{ 16, 32 }, .form = .nvfp4 },
    .{ .suffix = "blocks.0.mlp.linear_fc1.bias", .shape = &.{16} },
    .{ .suffix = "blocks.0.mlp.linear_fc2.weight", .shape = &.{ 32, 16 }, .form = .nvfp4 },
    .{ .suffix = "blocks.0.mlp.linear_fc2.bias", .shape = &.{32} },
    .{ .suffix = "merger.norm.weight", .shape = &.{32} },
    .{ .suffix = "merger.norm.bias", .shape = &.{32} },
    .{ .suffix = "merger.linear_fc1.weight", .shape = &.{ 128, 128 }, .form = .nvfp4 },
    .{ .suffix = "merger.linear_fc1.bias", .shape = &.{128} },
    .{ .suffix = "merger.linear_fc2.weight", .shape = &.{ 4, 128 }, .form = .nvfp4 },
    .{ .suffix = "merger.linear_fc2.bias", .shape = &.{4} },
};

const Options = struct {
    e8m0_dtype: bool = false, // the MXFP8 scales as F8_E8M0 instead of U8
    input_scale: bool = false, // NVFP4 and MXFP8 linears also carry the W4A4 activation scale
    no_global: ?usize = null, // the tower index of an NVFP4 linear written without weight_scale_2
    swapped_scale: ?usize = null, // the tower index of a linear whose scale shape is [groups, rows]
    short_codes: ?usize = null, // the tower index of a linear whose stored plane is a column narrower
    scale_two_on_mxfp8: bool = false,
    global_dtype: []const u8 = "F32",
    global_one: bool = false, // weight_scale_2 as shape [1] instead of a scalar
    stored_bf16: ?usize = null, // the tower index of a quantized linear stored as BF16 of the right shape instead
    dense_all: bool = false, // every linear stored as BF16, as a floating-point tower's are
};

/// Seeded bytes; NaN codes and scales (0x7f, 0xff) are left out, e8m0 bytes stay within 2^-27 to 2^13.
fn fill(bytes: []u8, seed: u64, form: Form, plane: enum { codes, scales }) void {
    var rng = std.Random.DefaultPrng.init(seed);
    for (bytes) |*b| {
        b.* = rng.random().int(u8);
        if (plane == .scales and form == .mxfp8) b.* = 100 + b.* % 40;
        if ((plane == .scales and form == .nvfp4) or (plane == .codes and form == .mxfp8)) {
            if (b.* & 0x7f == 0x7f) b.* = 0x38;
        }
    }
}

const Built = struct { header: std.ArrayList(u8) = .empty, data: std.ArrayList(u8) = .empty, expected: std.ArrayList([]u16) = .empty };

fn add(a: std.mem.Allocator, built: *Built, name: []const u8, dtype: []const u8, shape: []const usize, bytes: []const u8) !void {
    var text: std.ArrayList(u8) = .empty;
    defer text.deinit(a);
    for (shape, 0..) |d, i| try text.print(a, "{s}{d}", .{ if (i == 0) "" else ",", d });
    try built.header.print(a, "{s}\"{s}\":{{\"dtype\":\"{s}\",\"shape\":[{s}],\"data_offsets\":[{d},{d}]}}", .{ if (built.header.items.len == 0) "" else ",", name, dtype, text.items, built.data.items.len, built.data.items.len + bytes.len });
    try built.data.appendSlice(a, bytes);
}

fn deinit(a: std.mem.Allocator, built: *Built) void {
    built.header.deinit(a);
    built.data.deinit(a);
    for (built.expected.items) |e| a.free(e);
    built.expected.deinit(a);
}

fn count(shape: []const usize) usize {
    var n: usize = 1;
    for (shape) |d| n *= d;
    return n;
}

/// Writes the tower; `built.expected` gets what the loader must hand the sink, tensor by tensor, from fp64 references.
fn build(a: std.mem.Allocator, built: *Built, o: Options) !void {
    for (tower, 0..) |t, id| {
        var buf: [96]u8 = undefined;
        const name = try std.fmt.bufPrint(&buf, "model.visual.{s}", .{t.suffix});
        const n = count(t.shape);
        const expected = try a.alloc(u16, n);
        errdefer a.free(expected);
        if (t.form == .dense or o.dense_all or o.stored_bf16 == id) {
            const raw = try a.alloc(u16, n);
            defer a.free(raw);
            for (raw, 0..) |*v, i| v.* = fixture.value(id, i);
            @memcpy(expected, raw);
            try add(a, built, name, "BF16", t.shape, std.mem.sliceAsBytes(raw));
            try built.expected.append(a, expected);
            continue;
        }
        const rows = t.shape[0];
        const k = t.shape[1];
        const groups = k / (if (t.form == .nvfp4) @as(usize, 16) else 32);
        const codes = try a.alloc(u8, if (t.form == .nvfp4) rows * k / 2 else rows * k);
        defer a.free(codes);
        const scales = try a.alloc(u8, rows * groups);
        defer a.free(scales);
        fill(codes, id * 31 + 1, t.form, .codes);
        fill(scales, id * 31 + 2, t.form, .scales);
        const global: f32 = 0.0123 + @as(f32, @floatFromInt(id)) / 1000;
        const wide = try a.alloc(f64, rows * k);
        defer a.free(wide);
        if (t.form == .nvfp4) {
            try nvfp4.reference(.{
                .weight = .{ .dtype = .u8, .rank = 2, .shape = .{ rows, k / 2, 1, 1, 1 }, .bytes = codes },
                .scale = .{ .dtype = .f8_e4m3, .rank = 2, .shape = .{ rows, groups, 1, 1, 1 }, .bytes = scales },
                .global = &.{global},
            }, wide);
        } else for (wide, 0..) |*v, i| v.* = fp8.e4m3(codes[i]) * std.math.pow(f64, 2, @as(f64, @floatFromInt(scales[i / 32])) - 127);
        for (expected, wide) |*e, v| e.* = convert.bf16(@floatCast(v));
        const scale_shape: [2]usize = if (o.swapped_scale == id) .{ groups, rows } else .{ rows, groups };
        const stored_k = if (t.form == .nvfp4) k / 2 else k;
        const code_shape: [2]usize = if (o.short_codes == id) .{ rows, stored_k - 1 } else .{ rows, stored_k };
        const code_bytes = if (o.short_codes == id) codes[0 .. rows * (stored_k - 1)] else codes;
        try add(a, built, name, if (t.form == .nvfp4) "U8" else "F8_E4M3", &code_shape, code_bytes);
        var sn: [112]u8 = undefined;
        const scale_dtype = if (t.form == .nvfp4) "F8_E4M3" else if (o.e8m0_dtype) "F8_E8M0" else "U8";
        try add(a, built, try std.fmt.bufPrint(&sn, "{s}_scale", .{name}), scale_dtype, &scale_shape, scales);
        const four = std.mem.asBytes(&global);
        const scalar: []const usize = if (o.global_one) &.{1} else &.{};
        if (t.form == .nvfp4 and o.no_global != id) try add(a, built, try std.fmt.bufPrint(&sn, "{s}_scale_2", .{name}), o.global_dtype, scalar, if (std.mem.eql(u8, o.global_dtype, "F32")) four else four[0..2]);
        if (t.form == .mxfp8 and o.scale_two_on_mxfp8) try add(a, built, try std.fmt.bufPrint(&sn, "{s}_scale_2", .{name}), "F32", &.{}, four);
        if (o.input_scale) try add(a, built, try std.fmt.bufPrint(&sn, "{s}.input_scale", .{name[0 .. name.len - ".weight".len]}), "F32", &.{}, four);
        try built.expected.append(a, expected);
    }
}

fn write(a: std.mem.Allocator, tmp: std.testing.TmpDir, o: Options) !Built {
    var built: Built = .{};
    errdefer deinit(a, &built);
    try build(a, &built, o);
    try built.header.insert(a, 0, '{');
    try built.header.append(a, '}');
    var image: std.ArrayList(u8) = .empty;
    defer image.deinit(a);
    var size: [8]u8 = undefined;
    std.mem.writeInt(u64, &size, built.header.items.len, .little);
    try image.appendSlice(a, &size);
    try image.appendSlice(a, built.header.items);
    try image.appendSlice(a, built.data.items);
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "model.safetensors", .data = image.items });
    return built;
}

fn run(a: std.mem.Allocator, dir: []const u8, fail_at: ?usize) !fixture.Sink {
    var sink: fixture.Sink = .{ .a = a, .fail_at = fail_at };
    errdefer sink.deinit();
    try weights.load(a, std.testing.io, dir, config, &sink);
    return sink;
}

fn loaded(o: Options) !void {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var built = try write(std.testing.allocator, tmp, o);
    defer deinit(std.testing.allocator, &built);
    var buf: [96]u8 = undefined;
    var sink = try run(std.testing.allocator, try fixture.directory(tmp, &buf), null);
    defer sink.deinit();
    try std.testing.expectEqual(tower.len, sink.copies.items.len);
    for (sink.copies.items, built.expected.items, tower) |got, want, t| {
        try std.testing.expectEqual(want.len * 2, got.len);
        for (want, 0..) |bits, i| std.testing.expectEqual(bits, std.mem.readInt(u16, got[i * 2 ..][0..2], .little)) catch |err| {
            std.debug.print("{s}[{d}]\n", .{ t.suffix, i });
            return err;
        };
    }
}

test "Qwen tower weights: a ModelOpt tower's MXFP8 and NVFP4 linears reach the sink as the fp64 references' bf16" {
    try loaded(.{});
}

test "Qwen tower weights: a tower of floating-point linears reaches the sink as stored" {
    try loaded(.{ .dense_all = true });
}

test "Qwen tower weights: e8m0 scale dtype and the activation scales are accepted" {
    try loaded(.{ .e8m0_dtype = true, .input_scale = true, .global_one = true });
}

test "Qwen tower weights: a quantized linear missing or mismatching its planes fails closed" {
    const cases = [_]struct { o: Options, err: anyerror, logs: usize }{
        .{ .o = .{ .no_global = 11 }, .err = error.MissingTensor, .logs = 1 },
        .{ .o = .{ .swapped_scale = 5 }, .err = error.UnexpectedTensor, .logs = 1 },
        .{ .o = .{ .swapped_scale = 17 }, .err = error.UnexpectedTensor, .logs = 1 },
        .{ .o = .{ .short_codes = 5 }, .err = error.UnexpectedTensor, .logs = 1 },
        .{ .o = .{ .global_dtype = "F16" }, .err = error.UnexpectedTensor, .logs = 1 },
        .{ .o = .{ .scale_two_on_mxfp8 = true }, .err = error.UnusedVisionTensors, .logs = 2 },
    };
    for (cases) |case| {
        var tmp = std.testing.tmpDir(.{});
        defer tmp.cleanup();
        var built = try write(std.testing.allocator, tmp, case.o);
        defer deinit(std.testing.allocator, &built);
        var buf: [96]u8 = undefined;
        try std.testing.expectError(case.err, run(std.testing.allocator, try fixture.directory(tmp, &buf), null));
        try diagnostics.expect(case.logs);
    }
}

test "Qwen tower weights: a linear stored as bf16 beside quantized ones is read as stored" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var built = try write(std.testing.allocator, tmp, .{ .stored_bf16 = 5 });
    defer deinit(std.testing.allocator, &built);
    var buf: [96]u8 = undefined;
    var sink = try run(std.testing.allocator, try fixture.directory(tmp, &buf), null);
    defer sink.deinit();
    for (built.expected.items, sink.copies.items) |want, got| try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(want), got);
}

test "Qwen tower weights: copy refusals and allocation failures release the decoded rows" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var built = try write(std.testing.allocator, tmp, .{});
    defer deinit(std.testing.allocator, &built);
    var buf: [96]u8 = undefined;
    const dir = try fixture.directory(tmp, &buf);
    for ([_]usize{ 5, 7, 11, 17, 19 }) |i| try std.testing.expectError(error.InjectedCopyFailure, run(std.testing.allocator, dir, i));
    var n: usize = 0;
    while (true) : (n += 1) {
        var failing = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = n, .resize_fail_index = 0 });
        if (run(failing.allocator(), dir, null)) |ok| {
            var sink = ok;
            sink.deinit();
        } else |err| if (err != error.OutOfMemory or !failing.has_induced_failure) return err;
        try std.testing.expectEqual(failing.allocated_bytes, failing.freed_bytes);
        if (!failing.has_induced_failure) break;
    }
}
