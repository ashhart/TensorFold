//! The hold's zero-clock contract (A8): HOLD_PROFILE off zeroes hold_clock_reads; a structural scan gates holdMs.
const std = @import("std");
const cuda = @import("cuda");
const lc = @import("cuda_lilicorr.zig");
const lr = @import("cuda_lilicorr_round.zig");

// The same zero-GPU rig the absorb-guard harness uses: a driver recording fake on the one

// seam runP's family `run` helper drives (cuLaunchKernel), plus a bare engine whose stream

// and device-ctx face are the family fake shapes. No GPU, no allocations.
const Fake = struct {
    fn launch(_: cuda.abi.Function, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: cuda.abi.Stream, _: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
        return cuda.abi.success;
    }
};

const Rig = struct {
    drv: cuda.Driver = undefined,
    ctx_dev: cuda.Context = undefined,
    e: lc.E = undefined,

    fn arm(self: *Rig) *lc.E {
        self.drv.api.cuLaunchKernel = Fake.launch;
        self.ctx_dev = .{ .d = &self.drv, .device = 0, .handle = null };
        self.e = undefined;
        self.e.ctx = &self.ctx_dev;
        self.e.stream = .{ .d = &self.drv, .handle = null };
        return &self.e;
    }
};

test "a latched-off hold profile performs zero clock reads on the hold path (owner A8)" {
    const t = std.testing;
    var rig: Rig = .{};
    const e = rig.arm();
    const f: cuda.Function = .{ .d = &rig.drv, .handle = @ptrFromInt(0x9300) };

    lr.hold_prof_env = false; // the TF_HOLD_PROFILE gate, latched OFF (the lazy read skips getenv)
    lr.hold_clock_reads = 0;
    const n0 = lr.hp_n;
    const sum0 = lr.hp_sum;
    const name0 = lr.hp_max_name;
    defer lr.hold_prof_env = null; // restore the env-latch semantics for the other cells

    try lr.runP(e, f, .{ 1, 1, 1 }, .{ 64, 1, 1 }, .{@as(u64, 0x240000)}, "hold-zero-probe");
    try t.expectEqual(@as(u64, 0), lr.hold_clock_reads);
    try t.expectEqual(n0, lr.hp_n); // zero accumulation
    try t.expectEqual(sum0, lr.hp_sum);
    try t.expectEqualStrings(name0, lr.hp_max_name);
}

test "a latched-on hold profile reads its clocks and accumulates (owner A8's positive arm)" {
    const t = std.testing;
    var rig: Rig = .{};
    const e = rig.arm();
    const f: cuda.Function = .{ .d = &rig.drv, .handle = @ptrFromInt(0x9300) };

    lr.hold_prof_env = true; // latched ON
    lr.hold_clock_reads = 0;
    const n0 = lr.hp_n;
    const sum0 = lr.hp_sum;
    defer lr.hold_prof_env = null;

    try lr.runP(e, f, .{ 1, 1, 1 }, .{ 64, 1, 1 }, .{@as(u64, 0x240000)}, "hold-on-probe");
    try t.expect(lr.hold_clock_reads >= 2); // the section's start read and its gated end read
    try t.expectEqual(n0 + 1, lr.hp_n);
    try t.expect(lr.hp_sum >= sum0);
}

test "every hold-path clock read sits under the TF_HOLD_PROFILE gate (owner A8)" {
    const t = std.testing;
    const src = @embedFile("cuda_lilicorr_round.zig");
    // one clock seam: holdMs owns every read, so the counter inside it counts them all
    try t.expectEqual(@as(usize, 1), std.mem.count(u8, src, "clock_gettime("));
    try t.expect(std.mem.indexOf(u8, src, "hold_clock_reads += 1") != null);
    try t.expect(std.mem.indexOf(u8, src, "holdStamp()") != null);

    // A "holdMs()" call is legal only on a line that carries holdProfOn(), inside the

    // holdMs declaration itself, or inside a brace block opened by a holdProfOn() head.
    var depth: usize = 0;
    var gate_at: [64]usize = undefined;
    var gate_n: usize = 0;
    var violations: usize = 0;
    var it = std.mem.splitScalar(u8, src, '\n');
    while (it.next()) |raw| {
        const line = std.mem.trim(u8, raw, " \t");
        const gated_here = std.mem.indexOf(u8, line, "holdProfOn()") != null;
        const in_gate = gate_n > 0 and depth > gate_at[gate_n - 1];
        const is_comment = std.mem.startsWith(u8, line, "//");
        if (!is_comment and std.mem.indexOf(u8, line, "holdMs()") != null and !gated_here and !in_gate) {
            if (!std.mem.startsWith(u8, line, "pub fn holdMs")) violations += 1;
        }
        const opens = std.mem.count(u8, line, "{");
        if (opens > 0 and gated_here) {
            if (gate_n < gate_at.len) {
                gate_at[gate_n] = depth; // the head's own nesting: the body sits deeper
                gate_n += 1;
            }
        }
        depth += opens;
        depth -= std.mem.count(u8, line, "}");
        while (gate_n > 0 and depth <= gate_at[gate_n - 1]) gate_n -= 1;
    }
    try t.expectEqual(@as(usize, 0), violations);
}
