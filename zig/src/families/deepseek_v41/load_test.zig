const std = @import("std");
const t = std.testing;
const cuda = @import("cuda");
const mock = @import("load_mock.zig");
const named = @import("named.zig");
const Weights = @import("cuda_weights.zig").Weights;
fn builder(a: std.mem.Allocator) !named.Builder {
    var b: named.Builder = .{ .gpa = a, .io = std.testing.io, .pack = undefined };
    errdefer b.deinit();
    for ([_][]const u8{ "one", "two" }) |name| {
        const key = try a.dupe(u8, name);
        errdefer a.free(key);
        const bytes = try a.alignedAlloc(u8, .@"16", 16);
        errdefer a.free(bytes);
        @memset(bytes, 7);
        try b.out.append(a, .{ .name = key, .kind = .bf16, .shape = .{ 8, 1, 1, 1 }, .rank = 1, .bytes = bytes });
    }
    return b;
}
test "device upload allocator failures preserve one owner and no duplicate frees" {
    var b = try builder(t.allocator);
    defer b.deinit();
    const control = struct {
        fn run(a: std.mem.Allocator, builder_ptr: *const named.Builder) !void {
            var driver = mock.init();
            var w: Weights = .{ .gpa = a };
            defer {
                w.close() catch unreachable;
                t.expectEqual(mock.allocated, mock.freed) catch @panic("device owner lost");
                t.expectEqual(@as(usize, 0), mock.double_free) catch @panic("duplicate device free");
            }
            try w.upload(&driver, builder_ptr);
        }
    };
    try t.checkAllAllocationFailures(t.allocator, control.run, .{&b});
}
test "copy and drain failures retain device owners until a successful close" {
    var b = try builder(t.allocator);
    defer b.deinit();
    var driver = mock.init();
    var w: Weights = .{ .gpa = t.allocator };
    mock.fail_copy = true;
    mock.fail_sync = true;
    try t.expectError(error.NotReady, w.upload(&driver, &b));
    const result = w.close();
    if (result) |_| return error.UnsafeClose else |e| try t.expectEqual(error.NotReady, e);
    try t.expectEqual(@as(usize, 0), mock.freed);
    try t.expectEqual(@as(usize, 0), mock.early_free);
    mock.fail_sync = false;
    try w.close();
    try t.expectEqual(mock.allocated, mock.freed);
}
test "event record failure keeps staging pending and falls back to a stream drain" {
    var driver = mock.init();
    var s: @import("host_stage.zig").Stage = .{};
    _ = try s.take(&driver, 16);
    const stream: cuda.Stream = .{ .d = &driver, .handle = null };
    mock.pending = true;
    mock.fail_record = true;
    mock.fail_sync = true;
    try t.expectError(error.NotReady, s.issued(&driver, stream));
    const result = s.close();
    if (result) |_| return error.UnsafeClose else |e| try t.expectEqual(error.NotReady, e);
    try t.expect(mock.pinned != null);
    try t.expectError(error.NotReady, s.take(&driver, 32));
    mock.fail_sync = false;
    try s.close();
    try t.expect(mock.pinned == null);
    try t.expectEqual(@as(usize, 0), mock.early_free);
}
test "capture parser returns missing fields without an optional unwrap" {
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(t.io, .{ .sub_path = "aot.json", .data = "{}" });
    const dir = try tmp.dir.realPathFileAlloc(t.io, ".", t.allocator);
    defer t.allocator.free(dir);
    var driver = mock.init();
    try t.expectError(error.MissingField, @import("triton_capture.zig").Set.load(t.allocator, t.io, &driver, 0, dir));
}
test "runtime forms release every intermediate allocation on failure" {
    var b = try builder(t.allocator);
    defer b.deinit();
    t.allocator.free(b.out.items[0].name);
    b.out.items[0].name = try t.allocator.dupe(u8, "L0.hc_attn.fn");
    var driver = mock.init();
    var w: Weights = .{ .gpa = t.allocator };
    defer w.close() catch unreachable;
    try w.upload(&driver, &b);
    const source: @import("forms.zig").Source = .{ .d = &driver, .weights = &w, .plan = undefined };
    const control = struct {
        fn run(a: std.mem.Allocator, s: *const @import("forms.zig").Source) !void {
            const out = (try @import("forms.zig").build(a, s, "s.L0.hc_attn.fn16")).?;
            defer a.free(out);
            try t.expectEqual(@as(usize, 8), out.len);
        }
    };
    try t.checkAllAllocationFailures(t.allocator, control.run, .{&source});
}
test "CUDA tiled capability refuses unsupported codebooks and widths" {
    const admit = @import("cuda_weights.zig").admitWidth;
    try t.expectError(error.UnsupportedCudaCodebook, admit(.mcg, 8));
    try t.expectError(error.UnsupportedCudaWidth, admit(.mul1, 1));
    try t.expectError(error.UnsupportedCudaWidth, admit(.mul1, 15));
    try admit(.mul1, 8);
}
test "paired expert upload publishes two names under one device owner" {
    var b = try builder(t.allocator);
    defer b.deinit();
    for (b.out.items, [_][]const u8{ "L0.moe.w1.suh", "L0.moe.shared.0.w1.suh" }) |*item, name| {
        t.allocator.free(item.name);
        item.name = try t.allocator.dupe(u8, name);
    }
    @memset(b.out.items[0].bytes, 0x12);
    @memset(b.out.items[1].bytes, 0x87);
    const control = struct {
        fn run(a: std.mem.Allocator, input: *const named.Builder) !void {
            var driver = mock.init();
            var weights: Weights = .{ .gpa = a, .pair_stacks = true };
            defer weights.close() catch unreachable;
            try weights.upload(&driver, input);
            try t.expectEqual(@as(usize, 1), weights.bufs.items.len);
            try t.expectEqual(weights.get(input.out.items[0].name).?.ptr + 16, weights.get(input.out.items[1].name).?.ptr);
            try t.expectEqualSlices(u8, input.out.items[0].bytes, mock.memory[1][0..16]);
            try t.expectEqualSlices(u8, input.out.items[1].bytes, mock.memory[1][16..32]);
            try t.expectEqual(@as(usize, 0), mock.double_free);
        }
    };
    try t.checkAllAllocationFailures(t.allocator, control.run, .{&b});
}
test "capture owner publication survives allocation failure and registers supplied modules" {
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.createDir(t.io, "cubins", .default_dir);
    try tmp.dir.writeFile(t.io, .{ .sub_path = "cubins/ab.cubin", .data = "synthetic kernel image" });
    const json = "{\"kernels\":[{\"hash\":\"ab\",\"name\":\"test_kernel\",\"num_warps\":4,\"num_ctas\":1,\"shared\":0,\"global_scratch\":0,\"global_align\":1,\"profile_scratch\":0,\"pdl\":false,\"params\":[]}]}";
    try tmp.dir.writeFile(t.io, .{ .sub_path = "aot.json", .data = json });
    const dir = try tmp.dir.realPathFileAlloc(t.io, ".", t.allocator);
    defer t.allocator.free(dir);
    const control = struct {
        fn run(a: std.mem.Allocator, path: []const u8) !void {
            var driver = mock.init();
            defer t.expectEqual(mock.modules, mock.unloaded) catch @panic("capture module owner lost");
            var set = try @import("triton_capture.zig").Set.load(a, t.io, &driver, 0, path);
            defer set.deinit();
            const declarations = try set.declarations(a);
            defer a.free(declarations);
            try t.expectEqual(@as(usize, 1), declarations.len);
            try t.expectEqual(@as(u32, 128), declarations[0].launch.threads);
        }
    };
    try t.checkAllAllocationFailures(t.allocator, control.run, .{dir});
}
test "staging copy failure retains pinned source before event recording" {
    var driver = mock.init();
    var stage: @import("host_stage.zig").Stage = .{};
    _ = try stage.take(&driver, 16);
    var device = try cuda.DeviceBuffer.alloc(&driver, 16);
    defer device.free();
    const stream: cuda.Stream = .{ .d = &driver, .handle = null };
    mock.fail_copy = true;
    mock.fail_sync = true;
    try t.expectError(error.NotReady, stage.copyTo(device, 0, stream));
    try t.expectError(error.NotReady, stage.close());
    try t.expect(mock.pinned != null);
    mock.fail_sync = false;
    try stage.close();
    try t.expectEqual(@as(usize, 0), mock.early_free);
}

test "capture parameter types and identities are refused before any module load" {
    const a = t.allocator;
    const cases = [_][]const u8{
        "[{\"name\":\"x\",\"type\":\"\"}]",                                      "[{\"name\":\"\",\"type\":\"i32\"}]",
        "[{\"name\":\"x\",\"type\":\"garbage\"}]",                               "[{\"name\":\"x\",\"type\":\"*garbage\"}]",
        "[{\"name\":\"x\",\"type\":\"i32\"},{\"name\":\"x\",\"type\":\"i64\"}]",
    };
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmp.dir.realPathFileAlloc(t.io, ".", a);
    defer a.free(dir);
    for (cases) |params| {
        const json = try std.fmt.allocPrint(a, "{{\"kernels\":[{{\"hash\":\"ab\",\"name\":\"k\",\"num_warps\":4,\"params\":{s}}}]}}", .{params});
        defer a.free(json);
        try tmp.dir.writeFile(t.io, .{ .sub_path = "aot.json", .data = json });
        var d = mock.init();
        try t.expectError(error.BadManifest, @import("triton_capture.zig").Set.load(a, t.io, &d, 0, dir));
        try t.expectEqual(@as(usize, 0), mock.modules);
    }
}

test "expert route admission refuses wide and mixed groups before constructing images" {
    const admit = @import("cuda_weights.zig").admitExpertWidths;
    for ([_]u32{ 13, 14, 16 }) |hi| {
        try t.expectError(error.UnsupportedCudaExpertWidth, admit(hi, hi));
        try t.expectError(error.UnsupportedCudaExpertWidth, admit(2, hi));
    }
    try t.expectError(error.UnsupportedCudaExpertWidth, admit(10, 8));
    try admit(8, 8);
    try admit(2, 10);
    try admit(2, 12);
    // The common operator retains its independent broader capability.
    try @import("cuda_weights.zig").admitWidth(.mul1, 14);
}

test "production loader refuses unsupported expert route before allocating an image or device" {
    const pk = @import("pack.zig");
    const plans = @import("plan.zig");
    const ex = @import("core").exl3_layout;
    const a = t.allocator;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const data = try pk.image(a, &.{
        .{ .name = "p.trellis", .dtype = "I16", .shape = &.{ 8, 8, 112 } },
        .{ .name = "p.suh", .dtype = "F16", .shape = &.{128} },
        .{ .name = "p.svh", .dtype = "F16", .shape = &.{128} },
        .{ .name = "p.mul1", .dtype = "I32", .shape = &.{} },
    });
    defer a.free(data);
    std.mem.writeInt(u32, data[data.len - 4 ..][0..4], ex.marker_mul1, .little);
    try tmp.dir.writeFile(t.io, .{ .sub_path = "model.safetensors", .data = data });
    const dir = try tmp.dir.realPathFileAlloc(t.io, ".", a);
    defer a.free(dir);
    var p = try pk.Pack.open(a, t.io, dir);
    defer p.deinit();
    var parts = [_]ex.Part{try (try ex.Group.fromShape(&.{ 8, 8, 112 })).part(.whole, 0, 1)};
    var prefixes = [_][]const u8{"p"};
    var words = [_]u16{112};
    var offsets = [_]u64{0};
    const expert: plans.ExpertsPlan = .{ .proj = .w1, .parts = &parts, .prefixes = &prefixes, .layout = .{ .kind = .stack, .k_tiles = 8, .n_tiles = 8, .words = &words, .offsets = &offsets, .elements = 7168 } };
    var layers = [_]plans.LayerPlan{.{ .index = 0, .prefix = "layers.0", .mode = .full, .groups = &.{}, .natives = &.{}, .experts = @splat(expert), .gate_up_mismatch = 0 }};
    const plan: plans.Plan = .{ .arena = undefined, .rank = 0, .world = 1, .layers = &layers, .top = &.{}, .head = null, .dspark = null, .leftovers = &.{} };
    var fail = std.testing.FailingAllocator.init(a, .{ .fail_index = 0 });
    var weights: Weights = .{ .gpa = fail.allocator() };
    defer weights.close() catch unreachable;
    var driver = mock.init();
    const cfg: @import("config.zig").Config = .{};
    try t.expectError(error.UnsupportedCudaExpertWidth, @import("cuda_weights.zig").load(&weights, &driver, t.io, &cfg, &p, &plan));
    try t.expectEqual(@as(usize, 0), mock.allocated);
    try t.expectEqual(@as(usize, 0), fail.alloc_index);
}

test "two-plan form construction reads the original uploaded image for both roles" {
    const a = t.allocator;
    const form = @import("forms.zig");
    const buffers = @import("buffers.zig");
    const R = struct {
        a: std.mem.Allocator,
        map: std.StringHashMapUnmanaged(u64) = .empty,
        pub fn addressOf(self: *const @This(), name: []const u8) ?u64 {
            return self.map.get(name);
        }
        pub fn external(self: *@This(), name: []const u8, ptr: u64) !void {
            try self.map.put(self.a, name, ptr);
        }
    };
    var input: named.Builder = .{ .gpa = a, .io = t.io, .pack = undefined };
    defer input.deinit();
    const bytes = try a.alignedAlloc(u8, .@"16", 1024);
    for (bytes, 0..) |*v, i| v.* = @truncate(i *% 37 +% 11);
    try input.out.append(a, .{ .name = try a.dupe(u8, "p.trellis"), .kind = .i16, .rank = 3, .shape = .{ 2, 8, 32, 1 }, .bytes = bytes });
    var driver = mock.init();
    var weights: Weights = .{ .gpa = a, .pair_stacks = false };
    defer weights.close() catch unreachable;
    try weights.upload(&driver, &input);
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    var first: buffers.Plan = .{ .a = arena.allocator() };
    var second: buffers.Plan = .{ .a = arena.allocator() };
    try first.sizes.put(first.a, "s.p.lanes", 1024);
    try second.sizes.put(second.a, "s.p.T", 1024);
    var roles: R = .{ .a = a };
    defer roles.map.deinit(a);
    const result = try @import("forms_inplace.zig").bind(a, &roles, &weights, &.{ &first, &second });
    try t.expectEqual(@as(u32, 0), result.roles);
    const source: form.Source = .{ .d = &driver, .weights = &weights, .plan = undefined };
    const lanes = (try form.build(a, &source, "s.p.lanes")).?;
    defer a.free(lanes);
    if (roles.addressOf("s.p.lanes")) |ptr| @memcpy(mock.memory[ptr / 256][0..1024], lanes);
    const strips = (try form.build(a, &source, "s.p.T")).?;
    defer a.free(strips);
    const expected = try @import("prepare.zig").strips(a, bytes, 2, 8, 32);
    defer a.free(expected);
    try t.expectEqualSlices(u8, expected, strips);
    try t.expectEqualSlices(u8, bytes, &mock.memory[1]);
}
