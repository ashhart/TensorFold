//! Muse Spark's CUDA kernels under the contract: glue, attention and drafter against float64, EXL3 as replacements.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const rig_ = @import("contract_rig.zig");
const x3 = @import("contract_muse_exl3.zig");
const norms_k = @import("contract_muse_norms.zig");
const route_k = @import("contract_muse_route.zig");
const attn_k = @import("contract_muse_attn.zig");
const draft_k = @import("contract_muse_draft.zig");
const sel_k = @import("contract_muse_select.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const rows = contract.rows;

/// A thread's 16 squares at most (kMaxPer), then 32 warps' sums of 32 lanes.
fn rmsDepth(s: c.Shape) usize {
    return (s.k + 1023) / 1024 + 12;
}

const rms_why = "an fp16 output follows the sum of squares as ss^-1/2: a bf16 sum's error is a few output ulps";

/// muse_rms: the pre-attention norm on the fp32 residual, and the latent norm on fp16 rows.
pub fn rmsSpec(half: bool) c.Spec {
    return .{
        .name = if (half) "cuda muse muse_rms (f16 in, the latent norm)" else "cuda muse muse_rms (f32 in, the residual's norm)",
        .accum = .f32,
        .depth = rmsDepth,
        .ops = 8,
        .out = .f16,
        .tiles = .{ .{}, .{}, .{ .size = 1024 } },
        .sums = .{ .bf16 = 0, .why = rms_why },
        .served = if (half) &.{ .{ .rows = 8, .n = 4096, .k = 4096 }, .{ .rows = 1, .n = 4096, .k = 4096 } } else &.{ .{ .rows = 8, .n = 8192, .k = 8192 }, .{ .rows = 1, .n = 8192, .k = 8192 } },
    };
}

pub const Rms = struct {
    rig: rig_.Rig,
    half: bool,
    f: cuda.Function,
    r: ?rows.Rows = null,
    x: u64 = 0,
    w: u64 = 0,
    out: u64 = 0,

    pub fn deinit(self: *Rms) void {
        if (self.r) |r| r.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *Rms) g.Kernel {
        return .{ .spec = rmsSpec(self.half), .ctx = self, .vt = &vt };
    }

    const vt = g.hooks(Rms);

    pub fn prepare(self: *Rms, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.n != shape.k or shape.k == 0 or shape.k > 16 * 1024) return false;
        if (self.r) |r| r.deinit();
        self.r = null;
        self.rig.reset();
        const a = self.rig.gpu.gpa;
        const r = try rows.Rows.init(a, .rms, shape);
        self.r = r;
        r.fill(fill, seed);
        if (self.half) {
            const h = try a.alloc(f16, r.x.len);
            defer a.free(h);
            for (h, r.x) |*o, b| o.* = @floatCast(c.fromBf16(b));
            self.x = try self.rig.input(std.mem.sliceAsBytes(h));
        } else {
            const f = try a.alloc(f32, r.x.len);
            defer a.free(f);
            for (f, r.x) |*o, b| o.* = c.fromBf16(b);
            self.x = try self.rig.input(std.mem.sliceAsBytes(f));
        }
        const wf = try a.alloc(f32, r.w.len);
        defer a.free(wf);
        for (wf, r.w) |*o, b| o.* = c.fromBf16(b);
        self.w = try self.rig.input(std.mem.sliceAsBytes(wf));
        self.out = try self.rig.output(r.outputs() * 2);
        return true;
    }

    /// ops.norm's launch: a block of 1,024 threads a row, parts 0.
    pub fn run(self: *Rms) !void {
        const r = self.r.?;
        var a: cuda.Args = .{};
        a.add(self.x);
        a.add(@as(c_int, @intFromBool(self.half)));
        a.add(self.w);
        a.add(self.out);
        a.add(@as(c_int, @intCast(r.shape.k)));
        a.add(@as(f32, r.eps));
        a.add(@as(c_int, 0));
        try cuda.launch.launch(self.f, .{ .grid = .{ .x = @intCast(r.shape.rows), .y = 1, .z = 1 }, .block = .{ .x = 1024 } }, self.rig.s, &a);
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Rms) usize {
        return self.r.?.outputs();
    }

    pub fn reference(self: *Rms, want: []f64, mag: []f64) !void {
        return self.r.?.reference(want, mag);
    }

    pub fn simulate(self: *Rms, acc: c.Accum, order: g.Order, out: []f64) !void {
        return self.r.?.simulate(acc, order, out);
    }
};

/// The family's registrations, appended to the runner's list.
pub const Group = struct {
    mods: [6]?cuda.Module = @splat(null), // glue, linear, experts2, attention, dflash, dflash2
    shared: ?cuda.exl3.Linear = null,
    experts: ?cuda.exl3_experts.Experts = null,
    rms: [2]Rms = undefined,
    lins: [x3.lin_entries.len]x3.Lin = undefined,
    grps: [x3.grp_entries.len]x3.Grp = undefined,
    norms: [norm_list.len]norms_k.Norm = undefined,
    routes: [std.enums.values(route_k.Kind).len]route_k.Route = undefined,
    attns: [std.enums.values(attn_k.Kind).len]attn_k.Attn = undefined,
    drafts: [std.enums.values(draft_k.Kind).len]draft_k.Draft = undefined,
    sels: [std.enums.values(sel_k.Kind).len]sel_k.Sel = undefined,
    made: [8]usize = @splat(0),

    const NormEntry = struct { kind: norms_k.Kind, cluster: bool };
    const norm_list = [_]NormEntry{
        .{ .kind = .embed, .cluster = false },   .{ .kind = .gate_res, .cluster = false }, .{ .kind = .gate_x, .cluster = false },
        .{ .kind = .gate_xf, .cluster = false }, .{ .kind = .esum, .cluster = false },     .{ .kind = .gate_res, .cluster = true },
        .{ .kind = .gate_x, .cluster = true },   .{ .kind = .gate_xf, .cluster = true },   .{ .kind = .esum, .cluster = true },
    };

    fn fnName(comptime T: type, kind: T) [:0]const u8 {
        return switch (T) {
            route_k.Kind => switch (kind) {
                .router => "muse_router",
                .topk => "muse_topk",
                .argmax => "muse_argmax",
                .softcap => "muse_softcap",
                .rope => "muse_rope_tables",
            },
            attn_k.Kind => switch (kind) {
                .prep_rope, .prep_nope => "muse_qk_prep",
                .rows_slide, .rows_full => "muse_attn_rows",
                .merge_slide, .merge_full => "muse_attn_merge",
            },
            draft_k.Kind => switch (kind) {
                .embed => "dr_block_embed",
                .tap => "dr_tap",
                .store => "dr_store",
                .take => "dr_take",
                .norm_q, .norm_k => "dr_norm_rope",
                .attn => "dr_attn",
            },
            sel_k.Kind => switch (kind) {
                .dense_f32, .dense_f16 => "dfl_dense",
                .conv_prep => "dfl_conv_prep",
                .conv_finish => "dfl_conv_finish",
                .silu => "dfl_silu_mul",
                .topk => "dfl_topk",
                .walk => "dfl_walk",
            },
            else => unreachable,
        };
    }

    fn many(group: *Group, comptime T: type, comptime K: type, list: []T, mod: cuda.Module, gpu: check.Gpu, s: cuda.Stream, made: *usize) !void {
        inline for (comptime std.enums.values(K), 0..) |kind, i| {
            list[i] = .{ .rig = .{ .gpu = gpu, .s = s }, .kind = kind, .f = try mod.function(fnName(K, kind)), .arena = std.heap.ArenaAllocator.init(gpu.gpa) };
            made.* += 1;
        }
        _ = group;
    }

    pub fn init(group: *Group, gpu: check.Gpu, s: cuda.Stream) !void {
        errdefer group.deinit();
        const kk = cuda.kernels;
        for (&group.mods, [_][]const u8{ kk.muse_glue, kk.muse_linear, kk.muse_experts2, kk.muse_attention, kk.muse_dflash, kk.muse_dflash2 }) |*m, blob| m.* = try cuda.Module.load(gpu.d, blob);
        const glue = group.mods[0].?;
        group.shared = try cuda.exl3.Linear.load(gpu.d);
        group.experts = try cuda.exl3_experts.Experts.load(gpu.d);
        for (&group.rms, 0..) |*x, i| {
            x.* = .{ .rig = .{ .gpu = gpu, .s = s }, .half = i == 1, .f = try glue.function("muse_rms") };
            group.made[0] += 1;
        }
        for (x3.lin_entries, &group.lins) |e, *x| {
            x.* = try x3.Lin.init(.{ .gpu = gpu, .s = s }, e, group.mods[1].?, &group.shared.?);
            group.made[1] += 1;
        }
        for (x3.grp_entries, &group.grps) |e, *x| {
            x.* = .{ .rig = .{ .gpu = gpu, .s = s }, .entry = e, .f = try group.mods[2].?.function("muse_grouped2"), .base = group.experts.?.grouped[2] };
            group.made[2] += 1;
        }
        for (norm_list, &group.norms) |e, *x| {
            const name = switch (e.kind) {
                .embed => "muse_embed_norm",
                .esum => if (e.cluster) "muse_expert_sum_c" else "muse_expert_sum",
                else => if (e.cluster) "muse_gate_norm_c" else "muse_gate_norm",
            };
            const base: ?cuda.Function = if (!e.cluster) null else try glue.function(if (e.kind == .esum) "muse_expert_sum" else "muse_gate_norm");
            x.* = .{ .rig = .{ .gpu = gpu, .s = s }, .kind = e.kind, .cluster = e.cluster, .f = try glue.function(name), .base = base, .arena = std.heap.ArenaAllocator.init(gpu.gpa) };
            group.made[3] += 1;
        }
        try group.many(route_k.Route, route_k.Kind, &group.routes, glue, gpu, s, &group.made[4]);
        try group.many(attn_k.Attn, attn_k.Kind, &group.attns, group.mods[3].?, gpu, s, &group.made[5]);
        try group.many(draft_k.Draft, draft_k.Kind, &group.drafts, group.mods[4].?, gpu, s, &group.made[6]);
        try group.many(sel_k.Sel, sel_k.Kind, &group.sels, group.mods[5].?, gpu, s, &group.made[7]);
    }

    pub fn append(group: *Group, a: std.mem.Allocator, kernels: *std.ArrayList(g.Kernel)) !void {
        inline for (.{ &group.rms, &group.lins, &group.grps, &group.norms, &group.routes, &group.attns, &group.drafts, &group.sels }, 0..) |list, i| {
            for (list[0..group.made[i]]) |*x| try kernels.append(a, x.kernel());
        }
    }

    pub fn deinit(group: *Group) void {
        inline for (.{ &group.rms, &group.lins, &group.grps, &group.norms, &group.routes, &group.attns, &group.drafts, &group.sels }, 0..) |list, i| {
            for (list[0..group.made[i]]) |*x| x.deinit();
        }
        group.made = @splat(0);
        if (group.experts) |*e| e.unload();
        if (group.shared) |*l| l.unload();
        for (&group.mods) |*m| if (m.*) |*x| x.unload();
        group.* = .{};
    }

    /// A group on the heap, its kernels appended now; `destroy` releases both.
    pub fn create(a: std.mem.Allocator, gpu: check.Gpu, s: cuda.Stream, kernels: *std.ArrayList(g.Kernel)) !*Group {
        const group = try a.create(Group);
        group.* = .{};
        errdefer a.destroy(group);
        try group.init(gpu, s);
        errdefer group.deinit();
        try group.append(a, kernels);
        return group;
    }

    pub fn destroy(group: *Group, a: std.mem.Allocator) void {
        group.deinit();
        a.destroy(group);
    }
};
