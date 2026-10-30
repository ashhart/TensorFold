//! Host regressions for GLM's actual contract references, loader controls and frozen CUDA exports.
const std = @import("std");
const core = @import("core");
const impl = @import("tests/cuda/contract_glm_f32w.zig");
const baseline = @embedFile("kernels/cuda/contract_glm_bf16_baseline.cuh");
const controls = @embedFile("kernels/cuda/contract_controls.cu");

fn skipSpace(source: []const u8) []const u8 {
    return std.mem.trimStart(u8, source, " \t\r\n\\");
}

fn globalExport(source: []const u8, symbol: []const u8) bool {
    var rest = source;
    const prefix = "extern \"C\" __global__ void __launch_bounds__(";
    while (std.mem.indexOf(u8, rest, prefix)) |at| {
        rest = rest[at + prefix.len ..];
        const end = std.mem.indexOfScalar(u8, rest, ')') orelse return false;
        rest = skipSpace(rest[end + 1 ..]);
        const name_end = std.mem.indexOfScalar(u8, rest, '(') orelse return false;
        if (std.mem.eql(u8, std.mem.trimEnd(u8, rest[0..name_end], " \t\r\n\\"), symbol)) return true;
    }
    return false;
}

fn exported(symbol: []const u8) bool {
    const prefix = "tf_ctl_glm_old_rmsnorm_";
    if (!std.mem.startsWith(u8, symbol, prefix)) return globalExport(baseline, symbol);
    if (!globalExport(baseline, prefix ++ "##D")) return false;
    var buf: [64]u8 = undefined;
    const invocation = std.fmt.bufPrint(&buf, "\nRMSNORM({s},", .{symbol[prefix.len..]}) catch return false;
    return std.mem.indexOf(u8, baseline, invocation) != null;
}

test "GLM frozen BF16 exports match every replacement and resource registration" {
    try std.testing.expect(std.mem.indexOf(u8, controls, "#include \"contract_glm_bf16_baseline.cuh\"") != null);
    for (impl.bf16_entries) |entry| {
        try std.testing.expect(exported(entry.old));
        var registered = false;
        for (core.contract.registry.landed_cuda) |record| {
            if (std.mem.eql(u8, record.launch.symbol, entry.old)) {
                try std.testing.expectEqualStrings("contract_controls", record.launch.image);
                registered = true;
            }
        }
        try std.testing.expect(registered);
    }
    var count: usize = 0;
    for (core.contract.registry.landed_cuda) |record| {
        if (!std.mem.startsWith(u8, record.launch.symbol, "tf_ctl_glm_old_")) continue;
        count += 1;
        try std.testing.expect(exported(record.launch.symbol));
        var requested = false;
        for (impl.bf16_entries) |entry| requested = requested or std.mem.eql(u8, entry.old, record.launch.symbol);
        try std.testing.expect(requested);
    }
    try std.testing.expectEqual(@as(usize, 4), count);
}

test {
    _ = @import("tests/cuda/contract_glm_f32w_host.zig");
    _ = @import("tests/cuda/glm_loader_control.zig");
}
