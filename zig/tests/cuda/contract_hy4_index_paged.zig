//! Hy4-flash compressed/paged indexer kernels (`hy4_flash/index_paged.cu`) under the contract.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const hy4 = @import("hy4_index_paged");
const options = @import("contract_options");
const check = @import("check.zig");
const pool_k = @import("contract_hy4_index_paged_pool.zig");
const score_k = @import("contract_hy4_index_paged_score.zig");
const select_k = @import("contract_hy4_index_paged_select.zig");
const paging_k = @import("contract_hy4_index_paged_paging.zig");

const g = core.contract.gates;

pub const eps: f32 = 1e-5;

const controls = struct {
    pub const bytes align(16) = @embedFile("fatbin_contract_controls").*;
};

/// Planted faults matching hy4_index_common.cuh's tf_hy4_idx::Fault and the contract gate each turns red.
pub const Fault = enum {
    none,
    bad_nan,
    bad_repeat,
    bad_tail,
    bad_sums,
    bad_ties,

    pub fn gate(f: Fault) ?g.Gate {
        return switch (f) {
            .none => null,
            .bad_nan => .nonfinite,
            .bad_repeat => .repeat,
            .bad_tail, .bad_ties => .shapes,
            .bad_sums => .sums,
        };
    }
};

const all_faults = [_]Fault{ .bad_nan, .bad_repeat, .bad_tail, .bad_sums, .bad_ties };
const all_kernels = [_][]const u8{
    "rope_table",
    "pool_1cta",
    "pool_small",
    "pool_persist",
    "q_mrope",
    "score_decode",
    "score_decode_rope",
    "score_prefill",
    "select_1cta",
    "select_scan",
    "select_merge",
};

fn labelOf(comptime k: []const u8, comptime f: Fault) []const u8 {
    const what = switch (f) {
        .none => "",
        .bad_nan => "drops NaN inputs or scores",
        .bad_repeat => "bumps its first input every run",
        .bad_tail => "skips its tail elements or span",
        .bad_sums => "keeps running sums in fp16",
        .bad_ties => "breaks ties to the higher group",
    };
    return "control: hy4_index_paged tf_hy4_idx_" ++ k ++ " " ++ what;
}

const labels = blk: {
    var out: [all_kernels.len * all_faults.len][]const u8 = undefined;
    for (all_kernels, 0..) |k, i| for (all_faults, 0..) |f, j| {
        out[i * all_faults.len + j] = labelOf(k, f);
    };
    break :blk out;
};

/// Returns the contract control label for `k` planted with fault `f`.
pub fn controlName(k: []const u8, f: Fault) []const u8 {
    for (all_kernels, 0..) |name, i| if (std.mem.eql(u8, name, k)) for (all_faults, 0..) |fault, j| {
        if (fault == f) return labels[i * all_faults.len + j];
    };
    return "";
}

pub fn ctlSymbol(comptime k: []const u8, comptime f: Fault) [:0]const u8 {
    return "tf_ctl_hy4_idx_" ++ k ++ "_" ++ @tagName(f);
}

const f3 = [_]Fault{ .bad_nan, .bad_repeat, .bad_tail };
const f4 = [_]Fault{ .bad_nan, .bad_repeat, .bad_tail, .bad_sums };
const fsel = [_]Fault{ .bad_nan, .bad_repeat, .bad_tail, .bad_ties };

pub const Set = struct {
    gpa: std.mem.Allocator,
    k: hy4.Kernels,
    ctl: cuda.Module,
    cap: u32,
    ropes: [4]pool_k.RopeTable = undefined,
    pools: [15]pool_k.Pool = undefined,
    n_pools: usize = 0,
    paged_pools: [4]paging_k.PoolPaged = undefined,
    n_paged_pools: usize = 0,
    qmropes: [4]score_k.QMrope = undefined,
    scores: [15]score_k.Score = undefined,
    paged_scores: [4]paging_k.ScorePaged = undefined,
    selects1: [5]select_k.Select1Cta = undefined,
    selects_m: [8]select_k.SelectMulti = undefined,

    pub fn init(gpu: check.Gpu, s: cuda.Stream, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel)) !*Set {
        if (!options.controls) return error.BuiltWithoutControls;
        const self = try gpu.gpa.create(Set);
        errdefer gpu.gpa.destroy(self);
        var k = try hy4.Kernels.load(gpu.d);
        errdefer k.unload();
        var ctl = try cuda.Module.load(gpu.d, &controls.bytes);
        errdefer ctl.unload();
        const cap = try gpu.ctx.capability();
        self.* = .{ .gpa = gpu.gpa, .k = k, .ctl = ctl, .cap = cap };
        const rig: @import("contract_rig.zig").Rig = .{ .gpu = gpu, .s = s };

        self.ropes[0] = .{ .rig = rig, .f = self.k.rope_table };
        inline for (f3, 1..) |f, i| self.ropes[i] = .{ .rig = rig, .f = try self.ctl.function(comptime ctlSymbol("rope_table", f)), .fault = f };
        for (&self.ropes) |*x| try list.append(a, x.kernel());

        self.addPool(.cta1, .none, self.k.pool_1cta, rig);
        inline for (f4) |f| {
            const fn_ = try self.ctl.function(comptime ctlSymbol("pool_1cta", f));
            try fn_.allowDynamicShared(hy4.pool_1cta_smem);
            self.addPool(.cta1, f, fn_, rig);
        }
        if (cap >= 120) {
            self.addPool(.small, .none, self.k.pool_small, rig);
            inline for (f4) |f| self.addPool(.small, f, try self.ctl.function(comptime ctlSymbol("pool_small", f)), rig);
            self.addPool(.persist, .none, self.k.pool_persist, rig);
            inline for (f4) |f| {
                const fn_ = try self.ctl.function(comptime ctlSymbol("pool_persist", f));
                try fn_.allowDynamicShared(hy4.pool_persist_smem);
                self.addPool(.persist, f, fn_, rig);
            }
        }
        for (self.pools[0..self.n_pools]) |*x| try list.append(a, x.kernel());

        self.addPagedPool(.cta1, self.k.pool_1cta_paged, false, rig);
        self.addPagedPool(.cta1, self.k.pool_1cta_paged, true, rig);
        if (cap >= 120) {
            self.addPagedPool(.small, self.k.pool_small_paged, false, rig);
            self.addPagedPool(.persist, self.k.pool_persist_paged, false, rig);
        }
        for (self.paged_pools[0..self.n_paged_pools]) |*x| try list.append(a, x.kernel());

        self.qmropes[0] = .{ .rig = rig, .f = self.k.q_mrope };
        inline for (f3, 1..) |f, i| self.qmropes[i] = .{ .rig = rig, .f = try self.ctl.function(comptime ctlSymbol("q_mrope", f)), .fault = f };
        for (&self.qmropes) |*x| try list.append(a, x.kernel());

        const skinds = [_]score_k.Kind{ .decode, .decode_rope, .prefill };
        inline for (skinds, 0..) |sk, ki| {
            const base = ki * 5;
            const prod = switch (sk) {
                .decode => self.k.score_decode,
                .decode_rope => self.k.score_decode_rope,
                .prefill => self.k.score_prefill,
            };
            self.scores[base] = .{ .rig = rig, .kind = sk, .f = prod, .q_mrope = self.k.q_mrope, .base_decode = self.k.score_decode };
            inline for (f4, 1..) |f, fi| {
                const fn_ = try self.ctl.function(comptime ctlSymbol("score_" ++ @tagName(sk), f));
                self.scores[base + fi] = .{ .rig = rig, .kind = sk, .f = fn_, .q_mrope = self.k.q_mrope, .base_decode = self.k.score_decode, .fault = f };
            }
        }
        for (&self.scores) |*x| try list.append(a, x.kernel());

        self.paged_scores[0] = .{ .rig = rig, .kind = .decode, .f = self.k.score_decode_paged, .base_contig = self.k.score_decode };
        self.paged_scores[1] = .{ .rig = rig, .kind = .decode_rope, .f = self.k.score_decode_rope_paged, .base_contig = self.k.score_decode_rope };
        self.paged_scores[2] = .{ .rig = rig, .kind = .prefill, .f = self.k.score_prefill_paged, .base_contig = self.k.score_prefill };
        self.paged_scores[3] = .{ .rig = rig, .kind = .decode, .f = self.k.score_decode_paged, .base_contig = self.k.score_decode, .wrong_page = true };
        for (&self.paged_scores) |*x| try list.append(a, x.kernel());

        self.selects1[0] = .{ .rig = rig, .f = self.k.select_1cta };
        inline for (fsel, 1..) |f, i| self.selects1[i] = .{ .rig = rig, .f = try self.ctl.function(comptime ctlSymbol("select_1cta", f)), .fault = f };
        for (&self.selects1) |*x| try list.append(a, x.kernel());

        self.selects_m[0] = .{ .rig = rig, .scan = self.k.select_scan, .merge = self.k.select_merge, .base_1cta = self.k.select_1cta };
        inline for (f3, 1..) |f, i| {
            self.selects_m[i] = .{ .rig = rig, .scan = try self.ctl.function(comptime ctlSymbol("select_scan", f)), .merge = self.k.select_merge, .base_1cta = self.k.select_1cta, .fault = f, .fault_in_scan = true };
        }
        inline for (fsel, 4..) |f, i| {
            self.selects_m[i] = .{ .rig = rig, .scan = self.k.select_scan, .merge = try self.ctl.function(comptime ctlSymbol("select_merge", f)), .base_1cta = self.k.select_1cta, .fault = f, .fault_in_scan = false };
        }
        for (&self.selects_m) |*x| try list.append(a, x.kernel());

        return self;
    }

    fn addPool(self: *Set, v: pool_k.Variant, fault: Fault, f: cuda.Function, rig: @import("contract_rig.zig").Rig) void {
        self.pools[self.n_pools] = .{ .rig = rig, .variant = v, .f = f, .base_1cta = self.k.pool_1cta, .fault = fault, .persist_clusters = self.k.persist_clusters };
        self.n_pools += 1;
    }

    fn addPagedPool(self: *Set, v: pool_k.Variant, f: cuda.Function, wrong_page: bool, rig: @import("contract_rig.zig").Rig) void {
        self.paged_pools[self.n_paged_pools] = .{ .rig = rig, .variant = v, .f = f, .base_1cta = self.k.pool_1cta, .persist_clusters = self.k.persist_clusters, .wrong_page = wrong_page };
        self.n_paged_pools += 1;
    }

    pub fn deinit(self: *Set) void {
        for (&self.ropes) |*x| x.deinit();
        for (self.pools[0..self.n_pools]) |*x| x.deinit();
        for (self.paged_pools[0..self.n_paged_pools]) |*x| x.deinit();
        for (&self.qmropes) |*x| x.deinit();
        for (&self.scores) |*x| x.deinit();
        for (&self.paged_scores) |*x| x.deinit();
        for (&self.selects1) |*x| x.deinit();
        for (&self.selects_m) |*x| x.deinit();
        self.ctl.unload();
        self.k.unload();
        self.gpa.destroy(self);
    }
};

/// Runs gates 1-4 on all hy4_index kernels and planted controls, then gate 5 live against this GPU.
pub fn run(gpu: check.Gpu) !void {
    const a = gpu.gpa;
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var list: std.ArrayList(g.Kernel) = .empty;
    defer list.deinit(a);
    const self = try Set.init(gpu, s, a, &list);
    defer self.deinit();
    var t: g.Tally = .{};
    try g.runAll(a, list.items, 0x68793449, &t);
    const lim = try @import("contract_limits.zig").live(gpu, &.{"hy4_index"});
    std.debug.print("SUMMARY hy4 index: {d} kernels, {d} pass gates 1-4, {d} fail; {d} controls: {d} red as planted, {d} missed; {d} limit problems\n", .{ t.kernels, t.passed, t.failed, t.controls, t.caught, t.missed, lim });
    try check.expect(t.problems() + lim == 0, "hy4-index-contract: {d} gate failures, {d} controls missed, {d} limit problems", .{ t.failed, t.missed, lim });
}
