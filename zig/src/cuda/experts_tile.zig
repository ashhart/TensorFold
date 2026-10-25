//! K32 shared expert staging preserves decode dots and scale FMAs; selected only by the Flash Next opt-in.
const std = @import("std");
const Module = @import("module.zig").Module;
const Function = @import("module.zig").Function;
const Driver = @import("driver.zig").Driver;
const Stream = @import("stream.zig").Stream;
const launch = @import("launch.zig");
const experts = @import("experts.zig");
const grouped = @import("grouped.zig");
pub const Mode = enum { off, up, both };
pub fn mode(value: ?[]const u8) !Mode {
    const v = value orelse return .off;
    if (std.mem.eql(u8, v, "0")) return .off;
    if (std.mem.eql(u8, v, "1")) return .up;
    if (std.mem.eql(u8, v, "2")) return .both;
    return error.BadPromptExpertTile;
}
pub const Tile = struct {
    mod: Module,
    fns: [2]Function,
    resident: [2]usize,
    selection: Mode,
    pub const symbols = [2][:0]const u8{ "tf_nvfp4_tile_swiglu", "tf_nvfp4_tile_down_f32" };
    pub const shared = [2]u32{ 4 * (4096 + 4 * 2 * 576), 3 * (4096 + 4 * 576) };
    pub fn load(d: *const Driver, sms: usize, selection: Mode) !Tile {
        if (selection == .off) return error.PromptExpertTileDisabled;
        const kernels = @import("kernels.zig");
        if (!kernels.available) return error.BuiltWithoutKernels;
        var m = try Module.load(d, kernels.nvfp4_experts_tile);
        errdefer m.unload();
        var tile: Tile = .{ .mod = m, .fns = undefined, .resident = undefined, .selection = selection };
        for (symbols, shared, &tile.fns, &tile.resident) |name, bytes, *f, *resident| {
            f.* = try m.function(name);
            resident.* = @max(1, try f.occupancy(128, bytes)) * sms;
        }
        return tile;
    }
    pub fn unload(t: *Tile) void {
        t.mod.unload();
    }
    pub fn takes(t: *const Tile, epi: experts.Epi) bool {
        return t.selection != .off and (epi == .up or (epi == .down_f32 and t.selection == .both));
    }
    pub fn run(t: *const Tile, s: Stream, epi: experts.Epi, in: experts.Rows, l: experts.Layer, p: grouped.Plan, out: u64, items: usize, skip: i32) !void {
        if (!t.takes(epi) or (epi == .up and l.relu2)) return error.EpilogueNotBuilt;
        const gate = epi == .up;
        const index: usize = if (gate) 0 else 1;
        const n = if (gate) l.width else l.dims;
        const k = if (gate) l.dims else l.width;
        if (n == 0 or k == 0 or n % 32 != 0 or k % 32 != 0 or in.stride < k) return error.BadExpertShape;
        const grid = @min(items * ((n / 32 + 3) / 4), t.resident[index]);
        if (grid == 0) return;
        var a: launch.Args = .{};
        a.add(in.x);
        a.add(@as(i32, @intCast(in.stride)));
        a.add(@as(i32, @intCast(if (gate) in.slots else 0)));
        a.add(if (gate) l.up else l.down);
        a.add(if (gate) l.up_scale else l.down_scale);
        a.add(@as(i32, @intCast(k / 32)));
        a.add(@as(i32, @intCast(n / 32)));
        for ([_]u64{ p.items, p.counts, p.members, out }) |v| a.add(v);
        a.add(@as(i32, @intCast(n)));
        a.add(if (gate) l.limit else @as(f32, 0));
        a.add(skip);
        try launch.launch(t.fns[index], .{ .grid = .{ .x = @intCast(grid) }, .block = .{ .x = 128 }, .shared = shared[index] }, s, &a);
    }
};
test {
    _ = @import("experts_tile_test.zig");
}
