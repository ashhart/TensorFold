//! Add-only GLPv3 CUDA launches; nvcc 13.0 sm_121 receipts: measurements/main_130.zig.

// Format and paired site: Matt Suiche/weightless; add-only, no projection.
const registry = @import("registry.zig");

/// The served width the final's dynamic shared bytes (one bf16 row) are sized for.
const served_width = 5120;

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
