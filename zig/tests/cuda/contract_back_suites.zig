//! The gradient component suites the kernel contract runs after gates 1-4: FP64 and finite-difference checks.
const check = @import("check.zig");
const std = @import("std");

pub fn run(gpu: check.Gpu) !void {
    try @import("gdn_back_tests.zig").run(gpu);
    try @import("q4_back_tests.zig").run(gpu);
    try @import("moe_back_tests.zig").run(gpu);
    try @import("attn_back_tests.zig").run(gpu);
    try @import("fn_hc_back_tests.zig").run(gpu);
    try @import("fn_moe_back_tests.zig").run(gpu);
    try @import("fn_pocket_tests.zig").run(gpu, 0);
    std.debug.print("PASS gradient component suites (FP64 and finite-difference checks); resource receipts still required\n", .{});
}
