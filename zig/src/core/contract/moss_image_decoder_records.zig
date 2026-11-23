//! MOSS-Image's image decoder launches (moss_image_decoder.cu): geometry recorded; thirteen usages measured.
const std = @import("std");
const limits = @import("limits.zig");
const CudaRecord = @import("registry.zig").CudaRecord;

/// The whole-tensor launches, in the order the registry prints them.
const whole = [_]CudaRecord{
    .{ .launch = .{ .name = "moss image decoder tf_md_unpack", .symbol = "tf_md_unpack", .image = "moss_image_decoder", .threads = 256 }, .usage = .{ .{ .regs = 14, .static = 0, .bound = 256 }, .{ .regs = 14, .static = 0, .bound = 256 }, .{ .regs = 14, .static = 0, .bound = 256 }, .{ .regs = 16, .static = 0, .bound = 256 }, .{ .regs = 16, .static = 0, .bound = 256 } } },
    .{ .launch = .{ .name = "moss image decoder tf_md_gemm", .symbol = "tf_md_gemm", .image = "moss_image_decoder", .threads = 256 }, .usage = .{ .{ .regs = 62, .static = 8512, .bound = 256 }, .{ .regs = 58, .static = 8512, .bound = 256 }, .{ .regs = 58, .static = 8512, .bound = 256 }, .{ .regs = 59, .static = 8512, .bound = 256 }, .{ .regs = 59, .static = 8512, .bound = 256 } } },
    .{ .launch = .{ .name = "moss image decoder tf_md_conv3x3", .symbol = "tf_md_conv3x3", .image = "moss_image_decoder", .threads = 256 }, .usage = .{ .{ .regs = 57, .static = 29952, .bound = 256 }, .{ .regs = 70, .static = 29952, .bound = 256 }, .{ .regs = 70, .static = 29952, .bound = 256 }, .{ .regs = 55, .static = 29952, .bound = 256 }, .{ .regs = 55, .static = 29952, .bound = 256 } } },
};

/// The GroupNorm statistics' and the finish's launches: the first one's usage, then the finish's.
const part_usage: [limits.sms.len]limits.Usage = .{ .{ .regs = 25, .static = 1536, .bound = 0 }, .{ .regs = 28, .static = 1536, .bound = 0 }, .{ .regs = 28, .static = 1536, .bound = 0 }, .{ .regs = 30, .static = 1536, .bound = 0 }, .{ .regs = 30, .static = 1536, .bound = 0 } };
const norm_usage: [limits.sms.len]limits.Usage = .{ .{ .regs = 16, .static = 0, .bound = 1024 }, .{ .regs = 16, .static = 0, .bound = 1024 }, .{ .regs = 16, .static = 0, .bound = 1024 }, .{ .regs = 18, .static = 0, .bound = 1024 }, .{ .regs = 18, .static = 0, .bound = 1024 } };

/// The two launches a tensor-wide pass makes between the statistics and the norm, and the attention's row.
const finish: CudaRecord = .{ .launch = .{ .name = "moss image decoder tf_md_group_finish", .symbol = "tf_md_group_finish", .image = "moss_image_decoder", .threads = 1024 }, .usage = .{ .{ .regs = 31, .static = 0, .bound = 0 }, .{ .regs = 38, .static = 0, .bound = 0 }, .{ .regs = 38, .static = 0, .bound = 0 }, .{ .regs = 38, .static = 0, .bound = 0 }, .{ .regs = 38, .static = 0, .bound = 0 } } };
const softmax: CudaRecord = .{ .launch = .{ .name = "moss image decoder tf_md_softmax", .symbol = "tf_md_softmax", .image = "moss_image_decoder", .threads = 256 }, .usage = .{ .{ .regs = 27, .static = 32, .bound = 256 }, .{ .regs = 27, .static = 32, .bound = 256 }, .{ .regs = 27, .static = 32, .bound = 256 }, .{ .regs = 27, .static = 32, .bound = 256 }, .{ .regs = 27, .static = 32, .bound = 256 } } };

/// The channels the statistics and the norm launch by, one block a channel count.
const channels = [_]usize{ 32, 96, 192, 384 };

/// The records of a kernel launched once per channel count: its threads are that count, its usage the same.
fn byChannels(comptime symbol: []const u8, comptime usage: [limits.sms.len]limits.Usage) [channels.len]CudaRecord {
    var out: [channels.len]CudaRecord = undefined;
    for (&out, channels) |*r, n| r.* = .{
        .launch = .{ .name = std.fmt.comptimePrint("moss image decoder {s} {d} channels", .{ symbol, n }), .symbol = symbol, .image = "moss_image_decoder", .threads = @intCast(n) },
        .usage = usage,
    };
    return out;
}

pub const records = whole ++ byChannels("tf_md_group_part", part_usage) ++ [_]CudaRecord{finish} ++ byChannels("tf_md_norm", norm_usage) ++ [_]CudaRecord{softmax};
