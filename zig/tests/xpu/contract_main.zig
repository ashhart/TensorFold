//! tf-xpu-kernel-contract: gate 5 for XPU launches from the records and compiled images, and `limits DIR`.
const std = @import("std");
const core = @import("core");
const gate5 = @import("contract_limits.zig");

pub fn main(init: std.process.Init) !u8 {
    const a = init.gpa;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len == 1) {
        const n = try gate5.table(a);
        if (n > 0) {
            std.debug.print("FAIL xpu-kernel-contract: {d} problems\n", .{n});
            return 1;
        }
        std.debug.print("PASS xpu-kernel-contract: every registered XPU launch fits {s}, every control red\n", .{core.xpu_limits.xe[0].name});
        return 0;
    }
    if (args.len != 3 or !std.mem.eql(u8, args[1], "limits")) {
        std.debug.print("usage: tf-xpu-kernel-contract [limits DIR]\n", .{});
        return 2;
    }
    const n = try gate5.objects(a, init.io, args[2]);
    if (n > 0) {
        std.debug.print("FAIL xpu-kernel-contract-limits: {d} problems\n", .{n});
        return 1;
    }
    std.debug.print("PASS xpu-kernel-contract-limits: every registered launch fits bmg-g31, every record current, " ++
        "every control red\n", .{});
    return 0;
}
