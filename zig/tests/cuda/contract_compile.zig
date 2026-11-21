//! Compile both real resource consumers. The synthetic-image build never executes this entry point.
const std = @import("std");
const check = @import("check.zig");
const contracts = @import("contract_tests.zig");
export fn tf_compile_contract_consumers(gpu: *const check.Gpu, dir: [*:0]const u8) c_int {
    contracts.fatbins(gpu.*, std.mem.span(dir)) catch return 1;
    var coverage: @import("contract_shard.zig").Coverage = .{};
    const problems = contracts.live(gpu.*, &.{}, .{}, &coverage) catch return 1;
    return if (problems == 0) 0 else 1;
}
