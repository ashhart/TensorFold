//! int4's contract run: every registration through gates 1-4, and the plan's skipped expert's rows.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const fixture = @import("contract_int4.zig");
const grouped = cuda.grouped;
const int4 = cuda.int4;

const contract = core.contract;
const c = contract.spec;

const g = core.contract.gates;

/// Every registration, its modules and its launched control.
pub const Set = struct {
    gpa: std.mem.Allocator,
    kx: int4.Kernels,
    mod: cuda.Module,
    ctl: cuda.Module,
    items: [fixture.entries.len]fixture.Fixture = undefined,

    const controls = struct {
        pub const bytes align(16) = @embedFile("fatbin_contract_controls").*;
    };

    pub fn init(gpu: check.Gpu, s: cuda.Stream, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel)) !*Set {
        const set = try gpu.gpa.create(Set);
        errdefer gpu.gpa.destroy(set);
        var kx = try int4.Kernels.load(gpu.d);
        errdefer kx.deinit();
        var mod = try cuda.Module.load(gpu.d, cuda.kernels.experts);
        errdefer mod.unload();
        var ctl = try cuda.Module.load(gpu.d, &controls.bytes);
        errdefer ctl.unload();
        set.* = .{ .gpa = gpu.gpa, .kx = kx, .mod = mod, .ctl = ctl };
        const router = try grouped.Router.resolve(set.mod);
        const sums = try set.ctl.function("tf_ctl_int4_bf16_sums");
        const gu_kernel = try set.ctl.function("tf_ctl_int4_swiglu");
        for (&set.items, fixture.entries) |*x, e| x.* = .{ .rig = .{ .gpu = gpu, .s = s }, .entry = e, .kx = &set.kx, .router = router, .ctl = if (e.fault == .bf16_group_sums) sums else null, .gu_ctl = if (e.gu != null) gu_kernel else null };
        for (&set.items) |*x| try list.append(a, x.kernel());
        return set;
    }

    pub fn deinit(set: *Set) void {
        for (&set.items) |*x| x.deinit();
        set.ctl.unload();
        set.mod.unload();
        set.kx.deinit();
        set.gpa.destroy(set);
    }
};

/// A plan whose shared expert's slot is skipped: those rows stay 0xff, and every other row equals the all-live run.
fn skipped(set: *Set) !void {
    const a = set.gpa;
    var f = fixture.Fixture{ .rig = .{ .gpu = set.items[0].rig.gpu, .s = set.items[0].rig.s }, .entry = fixture.entries[0], .kx = &set.kx, .router = set.items[0].router };
    defer f.deinit();
    const sh: c.Shape = .{ .rows = 16, .n = fixture.width, .k = fixture.dims };
    if (!try f.prepare(sh, .random, 0x51ce)) return check.expect(false, "int4 skipped slot: refused a served shape", .{});
    const all = f.out;
    try f.launch();
    try f.rig.s.synchronize();
    const want = try a.dupe(u8, try f.rig.bytes(all, f.count() * 2));
    defer a.free(want);
    f.out = try f.rig.output(f.count() * 2);
    f.skip = fixture.shared;
    try f.launch();
    try f.rig.s.synchronize();
    const got = try f.rig.bytes(f.out, f.count() * 2);
    var skipped_rows: usize = 0;
    var differ: usize = 0;
    for (0..f.pairs) |p| {
        const row = got[p * sh.n * 2 ..][0 .. sh.n * 2];
        const live = f.ids[p] != fixture.shared;
        if (live) {
            if (!std.mem.eql(u8, row, want[p * sh.n * 2 ..][0 .. sh.n * 2])) differ += 1;
            continue;
        }
        skipped_rows += 1;
        // the skip run leaves the row at its poison, and the all-live run wrote it: a 0xff byte is a value too
        const base = want[p * sh.n * 2 ..][0 .. sh.n * 2];
        if (std.mem.indexOfNone(u8, row, &.{0xff}) != null or std.mem.indexOfNone(u8, base, &.{0xff}) == null) differ += 1;
    }
    try check.expect(skipped_rows > 0, "int4 skipped slot: the plan has no skipped pair", .{});
    try check.expect(differ == 0, "int4 skipped slot: {d} of {d} skipped rows differ from the all-live run", .{ differ, skipped_rows });
    std.debug.print("PASS int4 skipped slot: {d} of {d} pair rows left at 0xff, the others byte-equal to the all-live run\n", .{ skipped_rows, f.pairs });
}

/// The int4 kernel's gates 1-4, its controls and the skipped expert's rows alone.
pub fn run(gpu: check.Gpu) !void {
    const a = gpu.gpa;
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var list: std.ArrayList(g.Kernel) = .empty;
    defer list.deinit(a);
    const set = try Set.init(gpu, s, a, &list);
    defer set.deinit();
    var t: g.Tally = .{};
    try g.runAll(a, list.items, 0x696e7434, &t);
    try skipped(set);
    std.debug.print("SUMMARY int4: {d} kernels, {d} pass gates 1-4, {d} fail, {d} known findings; {d} controls: {d} red as planted, {d} missed\n", .{ t.kernels, t.passed, t.failed, t.known, t.controls, t.caught, t.missed });
    try check.expect(t.problems() == 0, "int4 contract: {d} gate failures, {d} controls missed", .{ t.failed, t.missed });
}

/// The skipped expert's rows, for a full kernel-contract run: their gates are registrations, this check is not.
pub fn skippedAlone(gpu: check.Gpu) !void {
    const a = gpu.gpa;
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var list: std.ArrayList(g.Kernel) = .empty;
    defer list.deinit(a);
    const set = try Set.init(gpu, s, a, &list);
    defer set.deinit();
    try skipped(set);
}
