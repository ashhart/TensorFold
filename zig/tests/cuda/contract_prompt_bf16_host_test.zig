//! The device contract stays reachable and covers every K-slice plan without a GPU.
const std = @import("std");
const cuda = @import("cuda");
const Tile = @import("contract_prompt_bf16.zig").Tile;
const Set = @import("contract_projection_set.zig").Set;

test "BF16 prompt contract: both bounds, byte replacements and controls retain their hooks and slice plans" {
    for ([_]cuda.qmmf.Out{ .bf16, .f32 }) |out| for ([_]bool{ false, true }) |replacement| {
        var tile: Tile = .{
            .product = .{ .rig = undefined, .lane = undefined, .mode = .bf16, .clusters = false, .out_kind = out },
            .prompt = undefined,
            .replacement = replacement,
        };
        const kernel = tile.kernel();
        std.mem.doNotOptimizeAway(kernel);
        try std.testing.expectEqual(replacement, kernel.spec.replaces != null);
        try std.testing.expect(kernel.vt.replaced != null);
        var slices: u16 = 0;
        for (kernel.spec.served) |sh| {
            slices |= @as(u16, 1) << @intCast(cuda.qmmf.splitK(sh.n, sh.k));
            try std.testing.expect(sh.k % 128 == 0);
        }
        try std.testing.expectEqual(@as(u16, (1 << 1) | (1 << 2) | (1 << 4) | (1 << 8)), slices);
        tile.control = true;
        try std.testing.expectEqual(@import("core").contract.gates.Gate.shapes, tile.kernel().control.?);
    };
}

test "BF16 prompt contract: moved registration set binds stable contexts and keeps the original prompt unchanged" {
    const Mock = struct {
        fn function(out: *cuda.abi.Function, _: cuda.abi.Module, name: [*:0]const u8) callconv(.c) cuda.abi.Result {
            const n = std.mem.span(name);
            if (!std.mem.eql(u8, n, "tf_prompt_bf16") and !std.mem.eql(u8, n, "tf_prompt_bf16_f32")) return 500;
            out.* = @ptrFromInt(3);
            return cuda.abi.success;
        }
        fn attribute(_: cuda.abi.Function, _: cuda.abi.FunctionAttribute, bytes: c_int) callconv(.c) cuda.abi.Result {
            return if (bytes == 98304) cuda.abi.success else 1;
        }
        fn sync(_: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
            return cuda.abi.success;
        }
    };
    var d: cuda.Driver = undefined;
    d.api.cuModuleGetFunction = Mock.function;
    d.api.cuFuncSetAttribute = Mock.attribute;
    d.api.cuStreamSynchronize = Mock.sync;
    var prompt: cuda.nvfp4.Prompt = undefined;
    prompt.mod = .{ .d = &d, .handle = null };
    prompt.bf16 = .{ .d = &d, .handle = @ptrFromInt(3) };
    prompt.bf16_f32 = .{ .d = &d, .handle = @ptrFromInt(4) };
    prompt.bf16_serving = false;
    var lane: cuda.qmmf.Lane = undefined;
    var set = try Set.init(.{ .d = &d, .ctx = undefined, .gpa = std.testing.allocator, .io = std.testing.io }, .{ .d = &d, .handle = null }, &lane, &lane, &prompt);
    defer set.deinit();
    var kernels: std.ArrayList(@import("core").contract.gates.Kernel) = .empty;
    defer kernels.deinit(std.testing.allocator);
    try set.append(std.testing.allocator, &kernels);
    try std.testing.expectEqual(@as(usize, 21), kernels.items.len);
    try std.testing.expectEqual(@as(usize, 3), @intFromPtr(prompt.bf16.handle.?));
    try std.testing.expectEqual(@as(usize, 4), @intFromPtr(prompt.bf16_f32.handle.?));
    try std.testing.expect(!prompt.bf16_serving);
    for (&set.tiles, kernels.items[kernels.items.len - set.tiles.len ..]) |*tile, kernel| {
        try std.testing.expectEqual(@intFromPtr(tile), @intFromPtr(kernel.ctx));
        try std.testing.expect(tile.rig == &tile.product.rig);
        try std.testing.expect(tile.prompt == &set.prompt);
    }
}
