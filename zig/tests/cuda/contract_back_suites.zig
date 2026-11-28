//! The gradient component suites the kernel contract runs after gates 1-4: FP64 and finite-difference checks.
const check = @import("check.zig");

/// A standalone suite of the full run: its name and its entry, which owns the device while it runs.
pub const Suite = struct { name: []const u8, run: *const fn (check.Gpu) anyerror!void };

/// A suite of any error set as a `Suite` entry.
pub fn entry(comptime name: []const u8, comptime f: anytype) Suite {
    return .{ .name = name, .run = &struct {
        fn call(gpu: check.Gpu) anyerror!void {
            return f(gpu);
        }
    }.call };
}

fn pocket(gpu: check.Gpu) !void {
    return @import("fn_pocket_tests.zig").run(gpu, 0);
}

pub const list = [_]Suite{
    entry("gdn-back", @import("gdn_back_tests.zig").run),
    entry("q4-back", @import("q4_back_tests.zig").run),
    entry("moe-back", @import("moe_back_tests.zig").run),
    entry("attn-back", @import("attn_back_tests.zig").run),
    entry("fn-hc-back", @import("fn_hc_back_tests.zig").run),
    entry("fn-moe-back", @import("fn_moe_back_tests.zig").run),
    entry("fn-pocket", pocket),
};

/// The line after all of `list` has passed.
pub const passed = "PASS gradient component suites (FP64 and finite-difference checks); resource receipts still required\n";
