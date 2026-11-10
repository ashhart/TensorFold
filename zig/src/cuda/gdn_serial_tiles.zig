//! Exact serial GDN geometry variants; two threads per value and the original reduction order in every variant.
const std = @import("std");
const Driver = @import("driver.zig").Driver;
const Module = @import("module.zig").Module;
const Function = @import("module.zig").Function;
const Stream = @import("stream.zig").Stream;
const launch = @import("launch.zig");
const base = @import("gdn_chunked.zig");

pub const Choice = struct { rows: usize, steps: usize, symbol: [:0]const u8 };
fn choice(comptime rows: usize, comptime steps: usize) Choice {
    return .{ .rows = rows, .steps = steps, .symbol = std.fmt.comptimePrint("_ZN14tf_gdn_prefill12chain_kernelI13__nv_bfloat16S1_Li{d}ELi{d}EEEvPKT_S4_PKS1_PKfPKT0_S8_PfPS1_iii", .{ rows, steps }) };
}
pub const choices = [_]Choice{ choice(64, 32), choice(16, 16), choice(16, 32), choice(32, 16), choice(32, 32), choice(64, 16) };
pub fn select(rows: usize, steps: usize) !Choice {
    for (choices) |c| if (c.rows == rows and c.steps == steps) return c;
    return error.UnsupportedGdnSerialTile;
}
pub const Serial = struct {
    module: Module,
    function: Function,
    config: Choice,
    pub fn load(d: *const Driver, image: []const u8, rows: usize, steps: usize) !Serial {
        const config = try select(rows, steps);
        var module = try Module.load(d, image);
        errdefer module.unload();
        return .{ .module = module, .function = try module.function(config.symbol), .config = config };
    }
    pub fn unload(self: *Serial) void {
        self.module.unload();
    }
    pub fn run(self: *const Serial, stream: Stream, shape: base.Shape, b: base.Buffers) !void {
        if (!shape.valid()) return error.UnsupportedGdnChunkShape;
        var args: launch.Args = .{};
        for ([_]u64{ b.q, b.k, b.v, b.decay, b.beta, b.initial, b.state, b.output }) |p| args.add(p);
        for ([_]usize{ shape.rows, shape.hk, shape.hv }) |n| args.add(@as(i32, @intCast(n)));
        try launch.launch(self.function, .{
            .grid = .{ .x = @intCast(shape.hv), .y = @intCast(128 / self.config.rows) },
            .block = .{ .x = @intCast(2 * self.config.rows) },
        }, stream, &args);
    }
};
test "GDN serial geometry retains two threads per value and the original template symbol" {
    for (choices) |c| {
        try std.testing.expect(128 % c.rows == 0);
        try std.testing.expect(c.rows == 16 or c.rows == 32 or c.rows == 64);
        try std.testing.expect(c.steps == 16 or c.steps == 32);
        try std.testing.expectEqualStrings(c.symbol, (try select(c.rows, c.steps)).symbol);
        try std.testing.expectEqual(@as(usize, 256), (128 / c.rows) * (2 * c.rows));
    }
    try std.testing.expectError(error.UnsupportedGdnSerialTile, select(8, 32));
    try std.testing.expectError(error.UnsupportedGdnSerialTile, select(32, 64));
}
