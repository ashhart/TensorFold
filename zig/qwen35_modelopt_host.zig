//! Synthetic compressed-tensors index admission and stored BF16/FP8/NVFP4 layouts.
const std = @import("std");
const core = @import("core");
const modelopt = @import("src/families/qwen3_5/cuda_modelopt.zig");
const geometry = @import("src/families/qwen3_5/cuda_geometry.zig");
const t = std.testing;
const g: geometry.Geometry = .{ .hidden = 16, .intermediate = 16, .layers = 1, .vocab = 2, .k_heads = 1, .v_heads = 1, .q_heads = 1, .kv_heads = 1, .eps = 1e-6, .tied = false, .context = 128, .linear = @splat(false) };
const config = "{\"quantization_config\":{\"quant_method\":\"compressed-tensors\"}}";
const index = "{\"weight_map\":{\"lm_head.weight_packed\":\"model.safetensors\",\"lm_head.weight_scale\":\"model.safetensors\",\"model.language_model.embed_tokens.weight\":\"model.safetensors\",\"model.language_model.layers.0.self_attn.q_proj.weight\":\"model.safetensors\",\"model.language_model.layers.0.self_attn.q_proj.weight_scale\":\"model.safetensors\"}}";
const header = "{\"lm_head.weight_packed\":{\"dtype\":\"U8\",\"shape\":[2,8],\"data_offsets\":[0,16]},\"lm_head.weight_scale\":{\"dtype\":\"F8_E4M3\",\"shape\":[2,1],\"data_offsets\":[16,18]},\"model.language_model.embed_tokens.weight\":{\"dtype\":\"BF16\",\"shape\":[2,16],\"data_offsets\":[18,82]},\"model.language_model.layers.0.self_attn.q_proj.weight\":{\"dtype\":\"F8_E4M3\",\"shape\":[2,16],\"data_offsets\":[82,114]},\"model.language_model.layers.0.self_attn.q_proj.weight_scale\":{\"dtype\":\"BF16\",\"shape\":[2,1],\"data_offsets\":[114,118]}}";
fn pathOf(tmp: anytype) ![]u8 {
    return std.fmt.allocPrint(t.allocator, ".zig-cache/tmp/{s}", .{tmp.sub_path});
}
fn read(gpa: std.mem.Allocator, bytes: []const u8) !void {
    _ = try modelopt.ctKindsBytes(gpa, bytes, g);
}

test "CT admission: malformed index root and weight map tags refuse without a panic" {
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const path = try pathOf(tmp);
    defer t.allocator.free(path);
    for ([_][]const u8{ "[]", "null", "{}", "{\"weight_map\":[]}", "{\"weight_map\":null}" }) |bad| {
        try tmp.dir.writeFile(t.io, .{ .sub_path = "model.safetensors.index.json", .data = bad });
        try t.expectError(error.UnsupportedQwenQuantization, modelopt.ctKinds(t.allocator, t.io, path, g));
    }
}

test "CT admission: small real index config and safetensors preserve mixed stored formats and OOM ownership" {
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const path = try pathOf(tmp);
    defer t.allocator.free(path);
    try tmp.dir.writeFile(t.io, .{ .sub_path = "config.json", .data = config });
    try tmp.dir.writeFile(t.io, .{ .sub_path = "model.safetensors.index.json", .data = index });
    var bytes: [8 + header.len + 118]u8 = @splat(0);
    std.mem.writeInt(u64, bytes[0..8], header.len, .little);
    @memcpy(bytes[8..][0..header.len], header);
    try tmp.dir.writeFile(t.io, .{ .sub_path = "model.safetensors", .data = &bytes });
    const initial = try modelopt.kinds(t.allocator, config, g);
    try t.expect(initial.ct);
    const kinds = try modelopt.ctKinds(t.allocator, t.io, path, g);
    try t.expectEqual(geometry.Kind.nvfp4, kinds.head);
    try t.expectEqual(geometry.Kind.fp8c, kinds.of(0, .q));
    try t.expectEqual(geometry.Kind.bf16, kinds.of(0, .k));
    var ck = try core.Checkpoint.openModel(t.allocator, t.io, path);
    defer ck.close();
    _ = try ck.expect("lm_head.weight_packed", .u8, &.{ 2, 8 });
    _ = try ck.expect("lm_head.weight_scale", .f8_e4m3, &.{ 2, 1 });
    _ = try ck.expect("model.language_model.embed_tokens.weight", .bf16, &.{ 2, 16 });
    _ = try ck.expect("model.language_model.layers.0.self_attn.q_proj.weight", .f8_e4m3, &.{ 2, 16 });
    _ = try ck.expect("model.language_model.layers.0.self_attn.q_proj.weight_scale", .bf16, &.{ 2, 1 });
    var reached = false;
    for (0..256) |i| {
        var failing = t.FailingAllocator.init(t.allocator, .{ .fail_index = i, .resize_fail_index = 0 });
        if (read(failing.allocator(), index)) |_| {
            try t.expect(!failing.has_induced_failure);
        } else |err| if (err != error.OutOfMemory) return err;
        try t.expectEqual(failing.allocated_bytes, failing.freed_bytes);
        if (!failing.has_induced_failure) {
            reached = true;
            break;
        }
    }
    try t.expect(reached);
}
const cuda = @import("cuda");
const Engine = @import("src/families/qwen3_5/cuda_engine.zig").Engine;
const Recorded = struct {
    var count: usize = 0;
    var id: usize = 0;
    var input: u64 = 0;
    fn launch(f: cuda.abi.Function, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: cuda.abi.Stream, args: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
        count += 1;
        id = @intFromPtr(f);
        input = @as(*const u64, @ptrCast(@alignCast(args.?[0].?))).*;
        return 0;
    }
    fn copy(_: u64, _: ?*const anyopaque, _: usize) callconv(.c) cuda.abi.Result {
        return 0;
    }
    fn sync(_: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        return 0;
    }
};
fn function(d: *const cuda.Driver, id: usize) cuda.Function {
    return .{ .d = d, .handle = @ptrFromInt(id) };
}
fn fullKernels(d: *cuda.Driver) modelopt.Kernels {
    d.api.cuLaunchKernel = Recorded.launch;
    d.api.cuMemcpyHtoD_v2 = Recorded.copy;
    d.api.cuStreamSynchronize = Recorded.sync;
    var out: modelopt.Kernels = .{ .lane = undefined, .prompt = undefined, .pack = undefined, .embed = undefined };
    out.lane.clusters = false;
    for (&out.lane.fns) |*set| for (set) |*f| {
        f.* = function(d, 2);
    };
    out.prompt.fp4 = function(d, 3);
    out.prompt.fp8 = function(d, 3);
    out.pack.fp8_lane = function(d, 1);
    return out;
}

test "full mode: stored FP8 packing ignores even malformed activation scales" {
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const path = try pathOf(tmp);
    defer t.allocator.free(path);
    const h = "{\"probe.weight\":{\"dtype\":\"F8_E4M3\",\"shape\":[2,64],\"data_offsets\":[0,128]},\"probe.weight_scale\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[128,132]},\"probe.input_scale\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[132,140]}}";
    var bytes: [8 + h.len + 140]u8 = @splat(0);
    std.mem.writeInt(u64, bytes[0..8], h.len, .little);
    @memcpy(bytes[8..][0..h.len], h);
    std.mem.writeInt(u32, bytes[8 + h.len + 128 ..][0..4], @bitCast(@as(f32, 1)), .little);
    try tmp.dir.writeFile(t.io, .{ .sub_path = "config.json", .data = "{}" });
    try tmp.dir.writeFile(t.io, .{ .sub_path = "model.safetensors", .data = &bytes });
    var ck = try core.Checkpoint.openModel(t.allocator, t.io, path);
    defer ck.close();
    var d: cuda.Driver = undefined;
    var e: Engine = undefined;
    e.gpa = t.allocator;
    e.d = &d;
    e.stream = .{ .d = &d, .handle = null };
    e.mo = fullKernels(&d);
    e.arena = .{ .d = &d, .ptr = 0x2000, .len = 16384 };
    e.arena_at = 0;
    var stage: cuda.DeviceBuffer = .{ .d = &d, .ptr = 0x1000, .len = 128 };
    Recorded.count = 0;
    const result = try modelopt.linear(&e, &ck, &stage, "", "probe", 2, 64, .fp8);
    try t.expectEqual(@as(usize, 1), Recorded.count);
    try t.expectEqual(@as(usize, 1), Recorded.id);
    try t.expectEqual(@as(f32, 0), result.ql.fp8.act);
    try t.expect(!ck.used.contains("probe.input_scale"));
    try t.expectEqual(@as(u64, 0x2000), result.ql.fp8.codes);
}

test "full mode: every stored format launches directly on bf16 rows at the prompt seam" {
    var d: cuda.Driver = undefined;
    var e: Engine = undefined;
    e.stream = .{ .d = &d, .handle = null };
    e.mo = fullKernels(&d);
    const Kind = geometry.Kind;
    for ([_]Kind{ .nvfp4, .fp8, .fp8c, .bf16 }) |kind| {
        const mode: cuda.qmmf.Mode = switch (kind) {
            .nvfp4 => .fp4,
            .fp8 => .fp8,
            .fp8c => .fp8g,
            .bf16 => .bf16,
            else => unreachable,
        };
        const q: cuda.qmmf.Weight = .{ .mode = mode, .codes = 0x3000, .scales = 0x4000, .n = 64, .k = 64, .npad = 128, .act = 1 };
        const weight: cuda.qlinear.Weight = switch (kind) {
            .nvfp4 => .{ .nvfp4 = q },
            .fp8 => .{ .fp8 = q },
            .fp8c => .{ .fp8g = q },
            .bf16 => .{ .bf16 = q },
            else => unreachable,
        };
        for (1..18) |rows| for ([_]bool{ false, true }) |prompt| {
            Recorded.count = 0;
            try modelopt.project(&e, weight, 0x5000, 0x6000, rows, prompt);
            try t.expectEqual(@as(usize, 1), Recorded.count);
            try t.expectEqual(@as(u64, 0x5000), Recorded.input);
            const gemm = prompt and rows > 16 and (kind == .fp8 or kind == .nvfp4);
            try t.expectEqual(if (gemm) @as(usize, 3) else 2, Recorded.id);
        };
    }
}

test "full mode: the real Qwen options expose only full precision" {
    const P = @typeInfo(@FieldType(@import("src/families/qwen3_5/cuda_native.zig").Options, "precision")).optional.child;
    const values = std.enums.values(P);
    try t.expectEqual(@as(usize, 1), values.len);
    try t.expectEqualStrings("full", @tagName(values[0]));
}
