//! CUDA GLPv3 additive launches: compiled usage stays unrecorded (gate 5) until nvcc evidence replaces `registry.none`.
const std = @import("std");
// Format and paired site: Matt Suiche/weightless; add-only, no projection.
const registry = @import("registry.zig");
const limits = @import("limits.zig");

/// The served width the final's dynamic shared bytes (one bf16 row) are sized for.
const served_width = 5120;
/// The one SM the GLP records name: DeepSeek's served target, like the family's other launches.
const sm_121 = 4;

fn record(comptime symbol: []const u8, dynamic: u32) registry.CudaRecord {
    return .{
        .launch = .{ .name = "dsv41_glp_add " ++ symbol, .symbol = symbol, .image = "dsv41_glp_add", .threads = 320, .dynamic = dynamic, .cluster = 1, .opt_in = false, .min_sm = 121 },
        .usage = registry.none,
        .later_pending = true,
    };
}

pub const records = [_]registry.CudaRecord{
    record("dsv41_glp_add_stream", 0),
    record("dsv41_glp_add_writer", 0),
    record("dsv41_glp_add_final", 2 * served_width),
};

test "glp records: unmeasured launches refuse qualification until compiled usage is recorded" {
    comptime std.debug.assert(limits.sms[sm_121].sm == 121);
    for (records) |r| {
        try std.testing.expect(std.meta.eql(registry.none, r.usage));
        try std.testing.expectEqual(@as(u32, 320), r.launch.threads);
        try std.testing.expectEqual(registry.TargetDecision.applicable, registry.targetDecision(r, limits.sms[sm_121], r.usage[sm_121]));
        try std.testing.expect(try registry.checkCuda(std.testing.allocator, &.{r}, true) > 0);
        // the declared geometry fits the target once any plausible usage is recorded (a fixture, not a measurement)
        var fixture = r;
        fixture.usage[sm_121] = .{ .regs = 64, .static = 0, .bound = 320 };
        try std.testing.expectEqual(@as(usize, 0), try registry.checkCuda(std.testing.allocator, &.{fixture}, true));
    }
}

test "glp records: the symbols are the entry points glp.cu exports" {
    const symbols = [_][]const u8{ "dsv41_glp_add_stream", "dsv41_glp_add_writer", "dsv41_glp_add_final" };
    for (records, symbols) |r, symbol| {
        try std.testing.expectEqualStrings(symbol, r.launch.symbol);
        try std.testing.expectEqualStrings("dsv41_glp_add", r.launch.image);
    }
    try std.testing.expectEqual(@as(u32, 10240), records[2].launch.dynamic);
}
