//! DeepSeek-V4.1's Metal contract behind one hook: suites and their extra gates.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const g = contract.gates;

const dsv41_fp8_rows = @import("metal_dsv41_fp8_rows.zig");
const dsv41_fp8_mma = @import("metal_dsv41_fp8_mma.zig");

/// Every dsv41 suite, created and destroyed in the order main held them.
pub const Suites = struct {
    fp8: *dsv41_fp8_rows.Suite,
    fp8_mma: *dsv41_fp8_mma.Suite,

    pub fn create(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, kernels: *std.ArrayList(g.Kernel), only: ?[]const u8) !Suites {
        const fp8 = try dsv41_fp8_rows.Suite.create(a, device, queue, kernels, only);
        errdefer fp8.destroy();
        const fp8_mma = try dsv41_fp8_mma.Suite.create(a, device, queue, kernels, only);
        errdefer fp8_mma.destroy();
        return .{ .fp8 = fp8, .fp8_mma = fp8_mma };
    }

    pub fn destroy(s: *Suites) void {
        s.fp8_mma.destroy();
        s.fp8.destroy();
    }

    /// The dsv41 gates after runAll: invariance and accuracy probes, and timings when asked for.
    pub fn gates(s: *Suites, io: std.Io, seed: u64, t: *g.Tally) !void {
        t.failed += try s.fp8.invariance(seed);
        if (std.c.getenv("TF_DSV41_TIME") != null) try s.fp8.timing(io);
        t.failed += try s.fp8_mma.invariance(seed);
        t.failed += try s.fp8_mma.accuracy(seed);
        if (std.c.getenv("TF_DSV41_TIME") != null) try s.fp8_mma.timing(io);
    }
};
