//! Explicit pending-launch measurements; absent names and earlier compilers stay pending.
pub const records = @import("measurements/qwen_decode_132.zig").records ++ @import("measurements/deepseek_a_132.zig").records ++ @import("measurements/deepseek_b_132.zig").records ++ @import("measurements/glm_sampling_132.zig").records ++ @import("measurements/qwen_lanes_132.zig").records;

const std = @import("std");
const limits = @import("limits.zig");
pub fn lookup(name: []const u8, release: limits.Nvcc) ?[limits.sms.len]limits.Usage {
    if (release.major == 13 and release.minor == 3) {
        for (@import("measurements/qwen_lanes_133.zig").records ++ @import("measurements/pending_133.zig").records) |r|
            if (std.mem.eql(u8, r.name, name)) return r.usage;
    }
    if (release.major < 13 or (release.major == 13 and release.minor < 2)) return null;
    for (records) |r| if (std.mem.eql(u8, r.name, name)) return r.usage;
    return null;
}
