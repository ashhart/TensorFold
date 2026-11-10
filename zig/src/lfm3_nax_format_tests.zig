//! LFM3 tensor-unit formats need launch records covering their dispatch within every Apple family's limits.
const std = @import("std");
const nax = @import("families/lfm3/nax_formats.zig");
const limits = @import("core/contract/limits.zig");
const records = @import("core/contract/lfm3_records.zig");

/// The record the limits gate holds for `f`'s launch, checked against each chip; missing or too small fails.
fn gated(a: std.mem.Allocator, list: []const limits.MetalLaunch, f: nax.Format) !void {
    var buf: [64]u8 = undefined;
    const want = try std.fmt.bufPrint(&buf, "LFM3 NAX prompt q{d}g{d} tf_qmmw_t_nax_win", .{ f.bits, f.group });
    const d = for (list) |r| {
        if (std.mem.eql(u8, r.name, want)) break r;
    } else {
        std.debug.print("no launch record for q{d}g{d}\n", .{ f.bits, f.group });
        return error.MissingNaxLaunchRecord;
    };
    // the route dispatches nax.threads threads and nax.threadgroup_bytes bytes: the record must not understate either
    if (d.threads < nax.threads or d.tg_memory < nax.threadgroup_bytes) return error.NaxRecordUnderstatesLaunch;
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(a);
    for (limits.apple) |chip| {
        if ((try limits.metal(d, chip, &out, a)) != 0) {
            std.debug.print("{s}", .{out.items});
            return error.NaxLaunchOutsideChipLimits;
        }
    }
}

test "LFM3 NAX prompt: every format the route builds has a launch record that fits every chip" {
    for (nax.formats) |f| try gated(std.testing.allocator, &records.metal, f);
}

test "LFM3 NAX prompt: q6g32 and q8g32 are formats the route builds, and each fails without its record" {
    const a = std.testing.allocator;
    for ([_]nax.Format{ .{ .bits = 6, .group = 32 }, .{ .bits = 8, .group = 32 } }) |f| {
        try std.testing.expect(nax.supported(f.bits, f.group));
        var buf: [64]u8 = undefined;
        const drop = try std.fmt.bufPrint(&buf, "LFM3 NAX prompt q{d}g{d} tf_qmmw_t_nax_win", .{ f.bits, f.group });
        var kept: [records.metal.len]limits.MetalLaunch = undefined;
        var n: usize = 0;
        for (records.metal) |d| {
            if (std.mem.eql(u8, d.name, drop)) continue;
            kept[n] = d;
            n += 1;
        }
        try std.testing.expectError(error.MissingNaxLaunchRecord, gated(a, kept[0..n], f));
    }
}
