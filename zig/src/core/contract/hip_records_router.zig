//! Gate 5's records of the router over the stored bf16 rows (decode.hip: tf_router_bf16).
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

/// The same launch in bf16, whose weights are the wider activation's rows.
pub const records = [_]Record{
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
};
