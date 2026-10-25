//! Fault controls share one typed constant buffer, outside every production binding through slot 16.
const std = @import("std");
const edit = @import("../flashnext/geometry_source.zig");
pub const slot: usize = 17;
pub const Params = extern struct { repeat: u32 };
pub const declaration = "struct Qwen4Control { uint repeat; };\n";
pub fn bind(encoder: anytype, repeat: u32) void {
    const params: Params = .{ .repeat = repeat };
    encoder.setBytes(std.mem.asBytes(&params), slot);
}
pub fn audit(source: []const u8) !void {
    const marker = "[[buffer(";
    var cursor: usize = 0;
    while (std.mem.indexOfPos(u8, source, cursor, marker)) |start| {
        const end = std.mem.indexOfScalarPos(u8, source, start + marker.len, ')') orelse return error.Qwen4MetalBinding;
        const value = std.fmt.parseInt(usize, source[start + marker.len .. end], 10) catch return error.Qwen4MetalBinding;
        if (value > 30) return error.Qwen4MetalBinding;
        cursor = end + 1;
    }
}
pub fn pack(a: std.mem.Allocator, source: []const u8) ![]u8 {
    if (std.mem.indexOf(u8, source, "[[buffer(17)]]") != null) return error.Qwen4ControlBindingCollision;
    const old = "constant uint& TF_REPEAT [[buffer(31)]]";
    if (std.mem.count(u8, source, old) == 0) return error.Qwen4ControlArgumentMissing;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const mem = arena.allocator();
    const args = try std.mem.replaceOwned(u8, mem, source, old, "constant Qwen4Control& TF_CONTROL [[buffer(17)]]");
    const text = try std.mem.replaceOwned(u8, mem, args, "TF_REPEAT", "TF_CONTROL.repeat");
    return edit.once(a, text, "using namespace metal;\n", "using namespace metal;\n" ++ declaration);
}

test "Qwen4 Metal control: the constant struct and encoder share slot, size and bytes" {
    const Mock = struct {
        var index: usize = 0;
        var value: u32 = 0;
        pub fn setBytes(_: @This(), bytes: []const u8, i: usize) void {
            std.debug.assert(bytes.len == 4);
            index = i;
            value = std.mem.bytesToValue(Params, bytes).repeat;
        }
    };
    bind(Mock{}, 0x12345678);
    try std.testing.expectEqual(@as(usize, 17), Mock.index);
    try std.testing.expectEqual(@as(u32, 0x12345678), Mock.value);
    try std.testing.expectEqual(@as(usize, 4), @sizeOf(Params));
    try std.testing.expectEqual(@as(usize, 4), @alignOf(Params));
    const input = "using namespace metal;\nkernel void f(constant uint& TF_REPEAT [[buffer(31)]]) { x = TF_REPEAT; }";
    const output = try pack(std.testing.allocator, input);
    defer std.testing.allocator.free(output);
    try audit(output);
    try std.testing.expectError(error.Qwen4MetalBinding, audit(input));
    try std.testing.expectError(error.Qwen4MetalBinding, audit("[[buffer(32)]]"));
    try std.testing.expectError(error.Qwen4MetalBinding, audit("[[buffer(x)]]"));
    try audit("[[buffer(0)]] [[buffer(30)]]");
    try std.testing.expect(std.mem.indexOf(u8, output, declaration) != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "x = TF_CONTROL.repeat;") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "buffer(31)") == null);
    try std.testing.expectError(error.Qwen4ControlBindingCollision, pack(std.testing.allocator, "[[buffer(17)]]"));
    try std.testing.expectError(error.Qwen4ControlArgumentMissing, pack(std.testing.allocator, "changed"));
}
