//! The prompt-glue launch receipts must reach the real registry in the five-SM order.
const std = @import("std");
const reg = @import("registry.zig");
const limits = @import("limits.zig");
const t = std.testing;

const receipts = [_]struct { name: []const u8, regs: [5]u32 = @splat(0), shared: u32 = 0, bound: u32 = 0, pending: bool = false }{
    .{ .name = "qwen_router_bf16 tf_moe_router_expert_first", .pending = true },
    .{ .name = "qwen_gdn_bf16 tf_gdn_chain_norm", .pending = true },
    .{ .name = "qwen_gdn_bf16 chain_kernel", .pending = true },
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

test "Qwen glue receipts: measured resources fit the actual served launch on every SM" {
    for (receipts) |receipt| {
        const r = try record(receipt.name);
        const problems = try reg.checkCuda(t.allocator, &.{r}, true);
        if (receipt.pending) try t.expect(problems > 0) else try t.expectEqual(@as(usize, 0), problems);
    }
}

test "Qwen glue receipts: measured compiles reach every launch and SM through the registry" {
    for (limits.sms, [_]u32{ 80, 86, 89, 120, 121 }) |sm, want| try t.expectEqual(want, sm.sm);
    const measured = [_]?limits.Nvcc{ null, .{ .major = 13, .minor = 0 } };
    for (measured) |release| {
        for (receipts) |receipt| {
            const r = try record(receipt.name);
            const usage = reg.usageFor(r, release);
            if (receipt.pending) {
                try t.expectEqualDeep(reg.none, usage);
                continue;
            }
            for (usage, receipt.regs) |got, regs| {
                try t.expectEqualDeep(limits.Usage{ .regs = regs, .static = receipt.shared, .bound = receipt.bound }, got);
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
