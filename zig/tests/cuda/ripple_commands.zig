//! Ripple's tf-cuda-test commands, which main.zig reaches through one entry.
const std = @import("std");
const check = @import("check.zig");

/// Runs `cmd` when it is one of Ripple's commands; false for any other.
pub fn run(gpu: check.Gpu, cmd: []const u8) !bool {
    if (std.mem.eql(u8, cmd, "ricursive-contract")) {
        try @import("contract_ricursive.zig").run(gpu);
    } else if (std.mem.eql(u8, cmd, "ricursive-dflash2-contract")) {
        try @import("contract_ricursive_dflash2.zig").run(gpu);
    } else if (std.mem.eql(u8, cmd, "ricursive-dflash2-attn-contract")) {
        try @import("contract_ricursive_dflash2_attn.zig").run(gpu);
    } else if (std.mem.eql(u8, cmd, "ripple-forward")) {
        try @import("ripple_forward_tests.zig").run(gpu);
    } else if (std.mem.eql(u8, cmd, "ripple-serve")) {
        try @import("ripple_serve_tests.zig").run(gpu);
    } else return false;
    return true;
}
