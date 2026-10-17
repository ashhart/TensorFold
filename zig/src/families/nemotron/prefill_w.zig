//! Prompt chunks' routed experts at a width but 4/64: qmmw_nax's sorted-expert gathers compiled per (bits, group).
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");
const frags = @import("../../core/frags.zig");
const row = @import("../../core/row_projection.zig");

/// qmmw_nax at one width: the sorted-expert gathers of 32, 64 and 128 rows a tile.
pub const Gathers = struct { bits: u8, group: u16, tiles: [3]mtl.Pipeline };

/// qmmw_nax (Flash Next's width-general prompt kernels, fp32 sums) at `bits` and `group`, nax.h inlined for this GPU.
pub fn compile(gpa: std.mem.Allocator, device: mtl.Device, bits: u8, group: u16) !Gathers {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const head = try std.fmt.allocPrint(gpa, "#define WB {d}\n#define WG {d}\n", .{ bits, group });
    defer gpa.free(head);
    const body = try std.mem.concat(gpa, u8, &.{ head, sources.flashnext_qmmw });
    defer gpa.free(body);
    const text = try frags.source(device, gpa, body);
    defer gpa.free(text);
    const lib = try mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
    defer lib.deinit();
    var out: Gathers = .{ .bits = bits, .group = group, .tiles = undefined };
    var made: usize = 0;
    errdefer for (out.tiles[0..made]) |p| p.deinit();
    for (&out.tiles, [_][:0]const u8{ "tf_gather_qmmw_nax_32", "tf_gather_qmmw_nax_64", "tf_gather_qmmw_nax_128" }) |*p, name| {
        p.* = try mtl.Pipeline.init(device, lib, name, false);
        made += 1;
    }
    return out;
}

/// Every width the routed experts take but 4/64, each compiled once (`have` lists those already built).
pub fn missing(have: []const Gathers, x: row.Experts) bool {
    for (have) |g| if (g.bits == x.one.bits and g.group == x.one.group) return false;
    return true;
}

/// The gathers at the experts' width, or null when none was compiled for it.
pub fn of(all: []const Gathers, x: row.Experts) ?Gathers {
    for (all) |g| if (g.bits == x.one.bits and g.group == x.one.group) return g;
    return null;
}

pub fn deinit(gpa: std.mem.Allocator, all: []const Gathers) void {
    for (all) |g| for (g.tiles) |p| p.deinit();
    gpa.free(all);
}

test "qmmw_nax compiles at a routed width Nemotron's oQ checkpoints use" {
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const g = try compile(std.testing.allocator, device, 6, 64);
    for (g.tiles) |p| p.deinit();
}
