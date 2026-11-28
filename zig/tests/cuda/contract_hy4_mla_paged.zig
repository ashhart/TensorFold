//! Hy4-flash paged MLA CUDA contract set (`hy4_mla_paged`): served forms and 20 planted controls.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const hy4 = @import("hy4_mla_paged_cuda");
const options = @import("contract_options");
const check = @import("check.zig");
const attn_k = @import("contract_hy4_mla_paged_attn.zig");
const proj_k = @import("contract_hy4_mla_paged_proj.zig");
const c12_k = @import("contract_hy4_mla_c1_c2.zig");

const g = core.contract.gates;

pub const eps: f32 = 1e-5;

const controls = struct {
    pub const bytes align(16) = @embedFile("fatbin_contract_controls").*;
};

pub const Fault = enum {
    none,
    bad_nan,
    bad_repeat,
    bad_tail,
    bad_sums,

    pub fn gate(f: Fault) ?g.Gate {
        return switch (f) {
            .none => null,
            .bad_nan => .nonfinite,
            .bad_repeat => .repeat,
            .bad_tail => .shapes,
            .bad_sums => .sums,
        };
    }
};

const faults = [_]Fault{ .bad_nan, .bad_repeat, .bad_tail, .bad_sums };
const kernels = [_][]const u8{ "absorb", "attn", "attn_rows", "value", "post" };

fn symbolOf(comptime k: []const u8, comptime f: Fault) [:0]const u8 {
    return "tf_ctl_hy4_mla_" ++ k ++ "_" ++ @tagName(f);
}

fn labelOf(comptime k: []const u8, comptime f: Fault) []const u8 {
    const what = switch (f) {
        .none => "production",
        .bad_nan => "reads a NaN input as 0",
        .bad_repeat => "bumps its first input every run",
        .bad_tail => "leaves its reduction's last term out",
        .bad_sums => "keeps its running sums in fp16",
    };
    return "control: hy4_mla_paged tf_hy4_mla_" ++ k ++ " " ++ what;
}

const labels = blk: {
    var out: [kernels.len * faults.len][]const u8 = undefined;
    for (kernels, 0..) |k, i| for (faults, 0..) |f, j| {
        out[i * faults.len + j] = labelOf(k, f);
    };
    break :blk out;
};

pub fn controlName(k: []const u8, f: Fault) []const u8 {
    for (kernels, 0..) |name, i| if (std.mem.eql(u8, name, k)) for (faults, 0..) |fault, j| {
        if (fault == f) return labels[i * faults.len + j];
    };
    return "control: hy4_mla_paged unknown";
}

pub const Set = struct {
    gpa: std.mem.Allocator,
    k: hy4.Kernels,
    c1: hy4.c1.C1,
    c2: hy4.c2.C2,
    ctl: ?cuda.Module = null,
    attns: [5]attn_k.Attn = undefined,
    projs: [5]proj_k.Proj = undefined,
    c1_cases: [7]c12_k.C1Case = undefined,
    c2_absorb: c12_k.C2Absorb = undefined,
    c2_expand: c12_k.C2Expand = undefined,
    attn_ctl: [2 * faults.len]attn_k.Attn = undefined,
    proj_ctl: [3 * faults.len]proj_k.Proj = undefined,
    has_ctl: bool = false,

    pub fn init(gpu: check.Gpu, s: cuda.Stream, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel)) !*Set {
        const self = try gpu.gpa.create(Set);
        errdefer gpu.gpa.destroy(self);
        var k = try hy4.Kernels.load(gpu.d);
        errdefer k.unload();
        var c1 = try hy4.c1.C1.load(gpu.d);
        errdefer c1.unload();
        var c2 = try hy4.c2.C2.load(gpu.d);
        errdefer c2.unload();
        const ctl = if (options.controls) try cuda.Module.load(gpu.d, &controls.bytes) else null;
        self.* = .{ .gpa = gpu.gpa, .k = k, .c1 = c1, .c2 = c2, .ctl = ctl };
        errdefer if (self.ctl) |*m| m.unload();
        const rig: @import("contract_rig.zig").Rig = .{ .gpu = gpu, .s = s };
        for (&self.attns, std.enums.values(attn_k.Form)) |*x, form| {
            x.* = .{ .rig = rig, .form = form, .f = if (form == .rows) self.k.attn_rows else self.k.attn };
        }
        const kinds = [_]proj_k.Kind{ .absorb, .value, .post, .post, .post };
        const worlds = [_]usize{ 1, 1, 1, 2, 4 };
        for (&self.projs, kinds, worlds) |*x, kind, world| {
            x.* = .{ .rig = rig, .kind = kind, .f = self.fn_(kind), .world = world };
        }
        for (&self.c1_cases, std.enums.values(c12_k.C1Mode)) |*x, mode| {
            x.* = .{ .rig = rig, .c1 = &self.c1, .mode = mode };
        }
        self.c2_absorb = .{ .rig = rig, .c2 = &self.c2 };
        self.c2_expand = .{ .rig = rig, .c2 = &self.c2 };
        for (&self.attns) |*x| try list.append(a, x.kernel());
        for (&self.projs) |*x| try list.append(a, x.kernel());
        for (&self.c1_cases) |*x| try list.append(a, x.kernel());
        try list.append(a, self.c2_absorb.kernel());
        try list.append(a, self.c2_expand.kernel());
        if (self.ctl) |cm| {
            inline for (faults, 0..) |fault, j| {
                const attn_fn = try cm.function(comptime symbolOf("attn", fault));
                try attn_fn.allowDynamicShared(hy4.attn_smem);
                self.attn_ctl[j] = .{ .rig = rig, .form = .window, .f = attn_fn, .fault = fault };
                const rows_fn = try cm.function(comptime symbolOf("attn_rows", fault));
                try rows_fn.allowDynamicShared(hy4.attn_smem);
                self.attn_ctl[faults.len + j] = .{ .rig = rig, .form = .rows, .f = rows_fn, .fault = fault };
                inline for ([_]proj_k.Kind{ .absorb, .value, .post }, 0..) |kind, i| {
                    const f = try cm.function(comptime symbolOf(@tagName(kind), fault));
                    self.proj_ctl[i * faults.len + j] = .{ .rig = rig, .kind = kind, .f = f, .fault = fault, .world = if (kind == .post) 2 else 1 };
                }
            }
            self.has_ctl = true;
            for (&self.attn_ctl) |*x| try list.append(a, x.kernel());
            for (&self.proj_ctl) |*x| try list.append(a, x.kernel());
        }
        return self;
    }

    fn fn_(self: *const Set, kind: proj_k.Kind) cuda.Function {
        return switch (kind) {
            .absorb => self.k.absorb,
            .value => self.k.value,
            .post => self.k.post,
        };
    }

    pub fn deinit(self: *Set) void {
        if (self.has_ctl) {
            for (&self.attn_ctl) |*x| x.deinit();
            for (&self.proj_ctl) |*x| x.deinit();
        }
        for (&self.attns) |*x| x.deinit();
        for (&self.projs) |*x| x.deinit();
        for (&self.c1_cases) |*x| x.deinit();
        self.c2_absorb.deinit();
        self.c2_expand.deinit();
        if (self.ctl) |*m| m.unload();
        self.c2.unload();
        self.c1.unload();
        self.k.unload();
        self.gpa.destroy(self);
    }
};
