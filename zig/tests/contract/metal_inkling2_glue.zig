//! Inkling 2 Mini glue under FP64 references, guarded buffers and real Metal dispatch.
const std = @import("std");
const mtl = @import("metal");
const controls = @import("metal_inkling2_glue_controls.zig");
const kv_conv = @import("metal_inkling2_glue_kv_conv.zig");
const split_controls = @import("metal_inkling2_split_controls.zig");
const fused = @import("metal_inkling2_glue_fused.zig");
const norm_fused = @import("metal_inkling2_glue_norm.zig");
const contract = @import("contract");
const c = contract.spec;
const g = contract.gates;
const host = @import("tensorfold").inkling2_metal_host.glue;
const p = host.plan;
const sources = @import("inkling2_metal");

const segments_label = "metal inkling2 glue conv_dynamic_segs tf_inkling2_conv_dynamic_segs";
const split_label = "metal inkling2 glue router split tf_inkling2_route_scores and tf_inkling2_route_select";

pub const Probe = @import("metal_inkling2_glue_probe.zig").Probe;

pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    kernels: *std.ArrayList(g.Kernel),
    probes: std.ArrayList(*Probe) = .empty,
    fused_probes: std.ArrayList(*fused.Probe) = .empty,
    norm_probes: std.ArrayList(*norm_fused.Probe) = .empty,

    const InitCtx = struct {
        s: *Suite,
        kind: p.Kind,
        text: []const u8,
        split: bool = false,

        fn initProbe(ctx: InitCtx) !Probe {
            return if (ctx.split)
                Probe.initSplit(ctx.s.a, ctx.s.device, ctx.s.queue, ctx.text)
            else
                Probe.initSource(ctx.s.a, ctx.s.device, ctx.s.queue, ctx.kind, ctx.text);
        }

        fn initFused(ctx: InitCtx) !fused.Probe {
            return fused.Probe.initSource(ctx.s.a, ctx.s.device, ctx.s.queue, ctx.kind, ctx.text);
        }

        fn initNormFused(ctx: InitCtx) !norm_fused.Probe {
            return norm_fused.Probe.initSource(ctx.s.a, ctx.s.device, ctx.s.queue, ctx.kind, ctx.text);
        }

        fn loadBase(ctx: InitCtx, probe: *Probe) !void {
            probe.base_pipes = try host.Pipelines.load(ctx.s.device);
        }

        fn markSegmented(_: InitCtx, probe: *Probe) !void {
            probe.segmented = true;
        }
    };

    fn add(s: *Suite, kind: p.Kind, control: bool, only: ?[]const u8) !void {
        const name = if (control) controls.name(kind) else kind.label();
        if (only) |filter| if (std.mem.indexOf(u8, name, filter) == null) return;
        const owned = if (control) try controls.source(s.a, sources.glue, kind) else null;
        defer if (owned) |buf| s.a.free(buf);
        const ctx: InitCtx = .{ .s = s, .kind = kind, .text = owned orelse sources.glue };
        const ctrl = if (control) name else null;
        if (kind == .norm_bf16 or kind == .attn_partial_ar_norm or kind == .mlp_partial_ar_norm) {
            try controls.addAcquired(norm_fused.Probe, InitCtx, s.a, &s.norm_probes, s.kernels, ctx, InitCtx.initNormFused, null, ctrl);
            return;
        }
        if (kind == .cached_ar or kind == .block_inv_rms or kind == .attn_partial_ar or kind == .mlp_partial_ar) {
            try controls.addAcquired(fused.Probe, InitCtx, s.a, &s.fused_probes, s.kernels, ctx, InitCtx.initFused, null, ctrl);
            return;
        }
        const load_base: ?*const fn (InitCtx, *Probe) anyerror!void = if (control and kv_conv.isKvConv(kind)) InitCtx.loadBase else null;
        try controls.addAcquired(Probe, InitCtx, s.a, &s.probes, s.kernels, ctx, InitCtx.initProbe, load_base, ctrl);
    }

    fn addNanControl(s: *Suite, control: controls.NanControl, only: ?[]const u8) !void {
        const name = controls.nanName(control);
        if (only) |filter| if (std.mem.indexOf(u8, name, filter) == null) return;
        const text = try controls.nanSource(s.a, sources.glue, control);
        defer s.a.free(text);
        try controls.addAcquired(Probe, InitCtx, s.a, &s.probes, s.kernels, .{ .s = s, .kind = .router, .text = text }, InitCtx.initProbe, null, name);
    }

    fn addSplit(s: *Suite, control: ?split_controls.Control, only: ?[]const u8) !void {
        const name = if (control) |x| split_controls.name(x) else split_label;
        if (only) |filter| if (std.mem.indexOf(u8, name, filter) == null) return;
        const owned = if (control) |x| try split_controls.source(s.a, sources.glue, x) else null;
        defer if (owned) |buf| s.a.free(buf);
        try controls.addAcquired(Probe, InitCtx, s.a, &s.probes, s.kernels, .{ .s = s, .kind = .router, .text = owned orelse sources.glue, .split = true }, InitCtx.initProbe, null, if (control != null) name else null);
    }

    fn addSegments(s: *Suite, control: bool, only: ?[]const u8) !void {
        const name = if (control) "control: " ++ segments_label else segments_label;
        if (only) |text| if (std.mem.indexOf(u8, name, text) == null) return;
        const from = "const uint r = global_r - seg.row0;";
        if (std.mem.count(u8, sources.glue, from) != 1) return error.ControlEditMissing;
        const text = if (control) try std.mem.replaceOwned(u8, s.a, sources.glue, from, "const uint r = 0;") else try s.a.dupe(u8, sources.glue);
        defer s.a.free(text);
        try controls.addAcquired(Probe, InitCtx, s.a, &s.probes, s.kernels, .{ .s = s, .kind = .conv_dynamic, .text = text }, InitCtx.initProbe, InitCtx.markSegmented, if (control) name else null);
    }

    pub fn collect(s: *Suite, only: ?[]const u8) !void {
        for (std.enums.values(p.Kind)) |kind| {
            try s.add(kind, false, only);
            try s.add(kind, true, only);
        }
        try s.addSegments(false, only);
        try s.addSegments(true, only);
        try s.addSplit(null, only);
        for (std.enums.values(controls.NanControl)) |control| try s.addNanControl(control, only);
        try s.addSplit(.route_strided, only);
        try s.addSplit(.route_index, only);
    }

    pub fn deinit(s: *Suite) void {
        for (s.probes.items) |probe| {
            probe.deinit();
            s.a.destroy(probe);
        }
        s.probes.deinit(s.a);
        for (s.fused_probes.items) |fp| {
            fp.deinit();
            s.a.destroy(fp);
        }
        s.fused_probes.deinit(s.a);
        for (s.norm_probes.items) |np| {
            np.deinit();
            s.a.destroy(np);
        }
        s.norm_probes.deinit(s.a);
    }
};
