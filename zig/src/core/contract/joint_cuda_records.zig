//! Clef's joint schema head on CUDA (joint_schema.cu): nvcc 13.0's resources, later sets pending.
const R = @import("registry.zig").CudaRecord;
const limits = @import("limits.zig");

/// Each kernel's registers and static shared bytes on sm_80, 86, 89, 120 and 121, and its launch bound (0: none).
const Entry = struct { name: []const u8, regs: [5]u32, static: u32, bound: u32 };

const entries = [_]Entry{
    .{ .name = "tf_js_layer_norm", .regs = .{ 29, 30, 30, 34, 34 }, .static = 32, .bound = 256 },
    .{ .name = "tf_js_gelu", .regs = .{ 13, 13, 13, 13, 13 }, .static = 0, .bound = 0 },
    .{ .name = "tf_js_softmax_rows", .regs = .{ 32, 39, 39, 39, 39 }, .static = 68, .bound = 256 },
    .{ .name = "tf_js_widen", .regs = .{ 10, 10, 10, 10, 10 }, .static = 0, .bound = 0 },
    .{ .name = "tf_js_round", .regs = .{ 12, 12, 12, 11, 11 }, .static = 0, .bound = 0 },
    .{ .name = "tf_js_span_sums", .regs = .{ 32, 36, 36, 36, 36 }, .static = 0, .bound = 0 },
    .{ .name = "tf_js_lex_sums", .regs = .{ 30, 32, 32, 34, 34 }, .static = 0, .bound = 0 },
    .{ .name = "tf_js_span_means", .regs = .{ 18, 18, 18, 18, 18 }, .static = 0, .bound = 0 },
    .{ .name = "tf_js_option_summary", .regs = .{ 40, 40, 40, 40, 40 }, .static = 68, .bound = 256 },
    .{ .name = "tf_js_logits", .regs = .{ 38, 38, 38, 40, 40 }, .static = 32, .bound = 256 },
    .{ .name = "tf_js_combine", .regs = .{ 13, 13, 13, 12, 12 }, .static = 0, .bound = 0 },
    .{ .name = "tf_js_add", .regs = .{ 13, 13, 13, 13, 13 }, .static = 0, .bound = 0 },
    .{ .name = "tf_js_broadcast", .regs = .{ 16, 16, 16, 16, 16 }, .static = 0, .bound = 0 },
};

pub const records = blk: {
    var out: [entries.len]R = undefined;
    for (entries, 0..) |e, i| {
        var usage: [limits.sms.len]limits.Usage = undefined;
        for (&usage, e.regs) |*u, regs| u.* = .{ .regs = regs, .static = e.static, .bound = e.bound };
        out[i] = .{ .launch = .{ .name = e.name, .symbol = e.name, .image = "joint_schema", .threads = 256, .min_sm = 80 }, .usage = usage, .later_pending = true };
    }
    break :blk out;
};
