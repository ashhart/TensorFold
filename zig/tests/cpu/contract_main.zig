//! The CPU kernels' contract: gates 1 to 4 over the projection and the norm on this host's cores, no device.
const std = @import("std");
const builtin = @import("builtin");
const contract = @import("contract");
const proj = @import("contract_projection.zig");
const norm = @import("contract_norm.zig");

const g = contract.gates;
const seed: u64 = 0x6370_7531;

/// Every registration and control: the projection's three chain classes and the norm, then their eight controls.
const Suite = struct {
    projs: [std.enums.values(proj.Which).len]proj.Proj,
    norms: [std.enums.values(norm.Which).len]norm.Norm,
    kernels: [std.enums.values(proj.Which).len + std.enums.values(norm.Which).len]g.Kernel,

    fn init(a: std.mem.Allocator) Suite {
        var s: Suite = undefined;
        for (&s.projs, std.enums.values(proj.Which)) |*p, w| p.* = proj.Proj.init(a, w);
        for (&s.norms, std.enums.values(norm.Which)) |*n, w| n.* = norm.Norm.init(a, w);
        return s;
    }

    /// The kernels before the controls, so the gates' PASS lines come first.
    fn list(s: *Suite) []g.Kernel {
        var at: usize = 0;
        for (&s.projs) |*p| if (p.kernel().control == null) {
            s.kernels[at] = p.kernel();
            at += 1;
        };
        for (&s.norms) |*n| if (n.kernel().control == null) {
            s.kernels[at] = n.kernel();
            at += 1;
        };
        for (&s.projs) |*p| if (p.kernel().control != null) {
            s.kernels[at] = p.kernel();
            at += 1;
        };
        for (&s.norms) |*n| if (n.kernel().control != null) {
            s.kernels[at] = n.kernel();
            at += 1;
        };
        return s.kernels[0..at];
    }

    fn deinit(s: *Suite) void {
        for (&s.projs) |*p| p.deinit();
        for (&s.norms) |*n| n.deinit();
    }
};

pub fn main(init: std.process.Init) !u8 {
    const a = std.heap.page_allocator;
    const cpus = std.Thread.getCpuCount() catch 0;
    std.debug.print("tf-cpu-contract on {t} ({d} logical CPUs): each run on 1 to {d} threads in turn, seed 0x{x}\n", .{ builtin.cpu.arch, cpus, proj.thread_counts, seed });
    var suite = Suite.init(a);
    defer suite.deinit();
    var t: g.Tally = .{};
    const t0 = std.Io.Clock.awake.now(init.io);
    try g.runAll(a, suite.list(), seed, &t);
    const ms = @divTrunc(t0.durationTo(std.Io.Clock.awake.now(init.io)).toNanoseconds(), std.time.ns_per_ms);
    std.debug.print("SUMMARY {d} kernels: {d} pass gates 1-4, {d} fail, {d} known findings; {d} controls: {d} red as planted, {d} missed; 0 limit problems (no launch to check); gates {d} ms\n", .{ t.kernels, t.passed, t.failed, t.known, t.controls, t.caught, t.missed, ms });
    return if (t.problems() == 0) 0 else 1;
}

test "the CPU projection and norm pass gates 1 to 4, and each planted control goes red at its gate" {
    var suite = Suite.init(std.testing.allocator);
    defer suite.deinit();
    g.quiet = true;
    defer g.quiet = false;
    var t: g.Tally = .{};
    try g.runAll(std.testing.allocator, suite.list(), seed, &t);
    try std.testing.expectEqual(@as(usize, 4), t.kernels);
    try std.testing.expectEqual(@as(usize, 4), t.passed);
    try std.testing.expectEqual(@as(usize, 8), t.controls);
    try std.testing.expectEqual(@as(usize, 8), t.caught);
    try std.testing.expectEqual(@as(usize, 0), t.known);
    try std.testing.expectEqual(@as(usize, 0), t.problems());
}

test {
    _ = @import("contract_rig.zig");
    _ = proj;
    _ = norm;
}
