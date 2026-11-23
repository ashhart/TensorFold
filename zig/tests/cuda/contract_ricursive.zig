//! Ripple's CUDA kernels under the contract, with controls (ricursive-contract): glue, attention prep, attention.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rc = @import("ricursive_cuda");
const check = @import("check.zig");
const glue = @import("contract_ricursive_glue.zig");
const prep = @import("contract_ricursive_prep.zig");
const attn = @import("contract_ricursive_attn.zig");

const g = core.contract.gates;

/// Every registration, on the heap: the gates hold pointers to them.
pub const Set = struct {
    gpa: std.mem.Allocator,
    k: rc.kernels.Kernels,
    glue: glue.Set = .{},
    prep: *prep.Set = undefined, // its own image, and contract_controls' tf_ctl_rc_prep
    attn: *attn.Set = undefined, // window_attention's image (WindowAttention owns it), tree_attention's (GlobalDecode)

    /// The set, its kernels appended to `list`.
    pub fn init(gpu: check.Gpu, s: cuda.Stream, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel)) !*Set {
        const set = try gpu.gpa.create(Set);
        errdefer gpu.gpa.destroy(set);
        set.* = .{ .gpa = gpu.gpa, .k = try rc.kernels.Kernels.load(gpu.d) };
        errdefer set.k.unload();
        try set.glue.init(gpu, s, &set.k, a, list);
        errdefer set.glue.deinit();
        set.prep = try prep.Set.init(gpu, s, a, list);
        errdefer set.prep.deinit();
        set.attn = try attn.Set.init(gpu, s, a, list);
        return set;
    }

    pub fn deinit(set: *Set) void {
        set.attn.deinit();
        set.prep.deinit();
        set.glue.deinit();
        set.k.unload();
        set.gpa.destroy(set);
    }
};

/// Ripple's kernels' gates 1-4 and their controls alone.
pub fn run(gpu: check.Gpu) !void {
    const a = gpu.gpa;
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var list: std.ArrayList(g.Kernel) = .empty;
    defer list.deinit(a);
    const set = try Set.init(gpu, s, a, &list);
    defer set.deinit();
    var t: g.Tally = .{};
    try g.runAll(a, list.items, 0x726970, &t);
    std.debug.print("SUMMARY ricursive: {d} kernels, {d} pass gates 1-4, {d} fail; {d} controls: {d} red as planted, {d} missed\n", .{ t.kernels, t.passed, t.failed, t.controls, t.caught, t.missed });
    try check.expect(t.problems() == 0, "ricursive-contract: {d} gate failures, {d} controls missed", .{ t.failed, t.missed });
}
