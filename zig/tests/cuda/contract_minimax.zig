//! MiniMax-M3.1-Flash's indexer, attention and glue under the contract, with controls (minimax-contract).
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const mm3 = @import("minimax_m3_cuda");
const check = @import("check.zig");
const attn = @import("contract_minimax_attn.zig");
const index = @import("contract_minimax_index.zig");
const glue = @import("contract_minimax_glue.zig");

const g = core.contract.gates;

/// Every registration, on the heap: the gates hold pointers to them.
pub const Set = struct {
    gpa: std.mem.Allocator,
    k: mm3.kernels.Kernels,
    attns: [5]attn.Attn = undefined,
    scores: [3]index.Scores = undefined,
    select: index.Select = undefined,
    glue: glue.Set = .{},

    /// The set, its kernels appended to `list`.
    pub fn init(gpu: check.Gpu, s: cuda.Stream, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel)) !*Set {
        const set = try gpu.gpa.create(Set);
        errdefer gpu.gpa.destroy(set);
        set.* = .{ .gpa = gpu.gpa, .k = try mm3.kernels.Kernels.load(gpu.d) };
        const forms = [_]attn.Form{ .kept, .every, .window, .kept, .window };
        for (&set.attns, forms, 0..) |*x, f, i| x.* = .{ .rig = .{ .gpu = gpu, .s = s }, .k = &set.k, .form = f, .drop = i == 3, .late = i == 4 };
        const which = [_]index.Which{ .tc2, .tc4, .tc2 };
        for (&set.scores, which, 0..) |*x, w, i| x.* = .{ .rig = .{ .gpu = gpu, .s = s }, .k = &set.k, .which = w, .shifted = i == 2 };
        set.select = .{ .rig = .{ .gpu = gpu, .s = s }, .k = &set.k };
        for (&set.attns) |*x| try list.append(a, x.kernel());
        for (&set.scores) |*x| try list.append(a, x.kernel());
        try list.append(a, set.select.kernel());
        try set.glue.init(gpu, s, &set.k, a, list);
        return set;
    }

    pub fn deinit(set: *Set) void {
        for (&set.attns) |*x| x.deinit();
        for (&set.scores) |*x| x.deinit();
        set.select.deinit();
        set.glue.deinit();
        set.k.unload();
        set.gpa.destroy(set);
    }
};

/// The MiniMax kernels' gates 1-4 and their controls alone.
pub fn run(gpu: check.Gpu) !void {
    const a = gpu.gpa;
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var list: std.ArrayList(g.Kernel) = .empty;
    defer list.deinit(a);
    const set = try Set.init(gpu, s, a, &list);
    defer set.deinit();
    var t: g.Tally = .{};
    try g.runAll(a, list.items, 0x6d6d33, &t);
    std.debug.print("SUMMARY minimax: {d} kernels, {d} pass gates 1-4, {d} fail; {d} controls: {d} red as planted, {d} missed\n", .{ t.kernels, t.passed, t.failed, t.controls, t.caught, t.missed });
    try check.expect(t.problems() == 0, "minimax-contract: {d} gate failures, {d} controls missed", .{ t.failed, t.missed });
}
