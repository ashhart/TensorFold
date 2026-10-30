//! Gate 5's records of the DeltaNet launches (recurrence/gated_delta.hip, ops/linear.hpp).
const Record = @import("hip_records.zig").Record;

pub const records = [_]Record{
    .{
        .launch = .{ .name = "DeltaNet tf_gdn_prompt_128", .symbol = "tf_gdn_prompt_128", .image = "gated_delta", .threads = 128 },
        .usage = .{
            .{ .vgprs = 175, .sgprs = 40, .lds = 20608, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 185, .sgprs = 39, .lds = 20608, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 185, .sgprs = 38, .lds = 20608, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 185, .sgprs = 38, .lds = 20608, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "DeltaNet tf_gdn_prompt_prep", .symbol = "tf_gdn_prompt_prep", .image = "ops", .threads = 256 },
        .usage = .{
            .{ .vgprs = 92, .sgprs = 59, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 93, .sgprs = 59, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 93, .sgprs = 59, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 93, .sgprs = 60, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        },
    },
};
