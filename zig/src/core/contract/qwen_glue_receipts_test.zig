//! The prompt-glue launch receipts must reach the real registry in the five-SM order.
const std = @import("std");
const reg = @import("registry.zig");
const limits = @import("limits.zig");
const t = std.testing;

// A zero register count is an SM the receipt has not measured yet (the ModelOpt images: GB10 only).
const receipts = [_]struct { name: []const u8, regs: [5]u32, shared: u32, bound: u32 }{
    .{ .name = "qwen_router_bf16 tf_moe_router_expert_first", .regs = .{ 0, 0, 0, 0, 63 }, .shared = 16512, .bound = 256 },
    .{ .name = "qwen_gdn_bf16 tf_gdn_chain_norm", .regs = .{ 0, 0, 0, 0, 183 }, .shared = 24704, .bound = 256 },
    .{ .name = "qwen_gdn_bf16 chain_kernel", .regs = .{ 0, 0, 0, 0, 193 }, .shared = 37120, .bound = 128 },
    .{ .name = "gdn_prefill chain_kernel", .regs = .{ 229, 231, 231, 193, 193 }, .shared = 37120, .bound = 128 },
    .{ .name = "qwen35 tf_q35_gdn_post", .regs = .{ 28, 29, 29, 31, 31 }, .shared = 256, .bound = 32 },
};

fn record(name: []const u8) !reg.CudaRecord {
    var found: ?reg.CudaRecord = null;
    for (reg.all_cuda) |r| if (std.mem.eql(u8, r.launch.name, name)) {
        try t.expect(found == null);
        found = r;
    };
    try t.expect(found != null);
    return found.?;
}

test "Qwen glue receipts: measured resources fit the actual served launch on every measured SM" {
    for (receipts) |receipt| {
        const r = try record(receipt.name);
        try t.expectEqual(std.mem.countScalar(u32, &receipt.regs, 0), try reg.checkCuda(t.allocator, &.{r}, true));
    }
}

test "Qwen glue receipts: measured compiles reach every launch and SM through the registry" {
    for (limits.sms, [_]u32{ 80, 86, 89, 120, 121 }) |sm, want| try t.expectEqual(want, sm.sm);
    const measured = [_]?limits.Nvcc{ null, .{ .major = 13, .minor = 0 } };
    for (measured) |release| {
        for (receipts) |receipt| {
            const r = try record(receipt.name);
            for (reg.usageFor(r, release), receipt.regs) |got, regs| {
                const want: limits.Usage = if (regs == 0) .{ .regs = 0, .static = 0 } else .{ .regs = regs, .static = receipt.shared, .bound = receipt.bound };
                try t.expectEqualDeep(want, got);
            }
        }
    }
    for (receipts) |receipt| {
        const r = try record(receipt.name);
        for ([_]limits.Nvcc{ .{ .major = 13, .minor = 2 }, .{ .major = 13, .minor = 3 } }) |release| {
            try t.expectEqualDeep(reg.none, reg.usageFor(r, release));
        }
    }
}
