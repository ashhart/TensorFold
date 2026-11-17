//! Gate 5's records of the router over the stored bf16 rows (decode.hip: tf_router_bf16) and the route (ops.hip).
const std = @import("std");
const limits = @import("limits.zig");
const Record = @import("hip_records.zig").Record;

/// The stored-bf16 router, `suffix` naming its activation kind's instantiation; 4 experts a block, 128 threads.
fn routeTile(comptime name: []const u8, comptime symbol: []const u8) Record {
    return .{
        .launch = .{ .name = name, .symbol = symbol, .image = "decode", .threads = 128 },
        .usage = .{
            .{ .vgprs = 35, .sgprs = 64, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 35, .sgprs = 64, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 35, .sgprs = 64, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 35, .sgprs = 65, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    };
}

/// The route plan's one block: the launcher's most LDS, 1,024 experts' counts, starts and ends, 16 segments, the scan.
const plan: Record = .{
    .launch = .{ .name = "moe route tf_moe_route", .symbol = "tf_moe_route", .image = "ops", .threads = 256, .dynamic = 3 * 1024 * 4 + 16 * 1024 * 2 + 2 * 256 * 4 },
    .usage = .{
        .{ .vgprs = 63, .sgprs = 28, .lds = 0, .private = 0, .bound = 1024, .wave = 32 },
        .{ .vgprs = 63, .sgprs = 28, .lds = 0, .private = 0, .bound = 1024, .wave = 32 },
        .{ .vgprs = 63, .sgprs = 28, .lds = 0, .private = 0, .bound = 1024, .wave = 32 },
        .{ .vgprs = 64, .sgprs = 27, .lds = 0, .private = 0, .bound = 1024, .wave = 32 },
    },
};

/// The same launch in bf16, whose weights are the wider activation's rows; then the route plan the picks feed.
pub const records = [3]Record{
    routeTile("router tf_router_bf16_f16", "tf_router_bf16_f16"),
    .{
        .launch = .{ .name = "router tf_router_bf16", .symbol = "tf_router_bf16", .image = "decode", .threads = 128 },
        .usage = .{
            .{ .vgprs = 36, .sgprs = 64, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 36, .sgprs = 64, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 36, .sgprs = 64, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 36, .sgprs = 65, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    plan,
};
