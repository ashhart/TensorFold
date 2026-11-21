//! Hy4-flash's hyper-connections (ihc.metal via ihc_metal.zig) and DFly2's glue (metal_draft.zig) under the contract.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const sources = @import("kernel_sources");
const h = @import("metal_hy4.zig");
const m = @import("metal_hy4_ihc_ref.zig");
const collapse = @import("metal_hy4_ihc_collapse.zig");
const seam = @import("metal_hy4_ihc_seam.zig");
const draft = @import("metal_hy4_ihc_draft.zig");

const g = contract.gates;
pub const K = h.hy.ihc_metal.Kernels;

/// The pipeline of the loader's set a planted control stands in for (the draft's are pipelines of their own).
pub const Slot = enum { pre8, finish8, tail8, seam8, mix, fuse, residual };

/// A planted control: one edit to ihc.metal, the function it builds, what it drops and the gate that must go red.
pub const Ctl = struct { slug: []const u8, function: []const u8, slot: Slot, gate: g.Gate, drops: []const u8, edit: h.fl.Edit };

pub const controls = [_]Ctl{
    .{ .slug = "flush", .function = "tf_hy4_ihc_collapse_g8", .slot = .pre8, .gate = .nonfinite, .drops = "non-finite hidden values flushed from its sums", .edit = .{ .kernel = "ihc_collapse_fixed", .from = "const float v = float(x[i]);", .to = "const float v = isfinite(float(x[i])) ? float(x[i]) : 0.0f;" } },
    .{ .slug = "seven", .function = "tf_hy4_ihc_collapse_g8", .slot = .pre8, .gate = .shapes, .drops = "the last simdgroup's partial sums dropped", .edit = .{ .kernel = "ihc_collapse_fixed", .from = "for (uint i = 0; i < 8; ++i) {", .to = "for (uint i = 0; i < 7; ++i) {" } },
    .{ .slug = "half", .function = "tf_hy4_ihc_collapse_g8", .slot = .pre8, .gate = .sums, .drops = "the squares' simdgroup totals added in half", .edit = .{ .kernel = "ihc_collapse_fixed", .from = "sum += partial[8][i];", .to = "sum = float(half(sum + partial[8][i]));" } },
    .{ .slug = "slot0", .function = "tf_hy4_ihc_finish_g8", .slot = .finish8, .gate = .shapes, .drops = "every slot read as simdgroup 0's squares", .edit = .{ .kernel = "ihc_finish_fixed", .from = "sum += p[64 + i];", .to = "sum += p[64];" } },
    .{ .slug = "unweighted", .function = "tf_hy4_tail_g8", .slot = .tail8, .gate = .shapes, .drops = "the branch mixed in without its post gate", .edit = .{ .kernel = "ihc_tail", .from = "post[ulong(row) * 4 + stream] * normed", .to = "normed" } },
    .{ .slug = "once", .function = "tf_hy4_seam_g8", .slot = .seam8, .gate = .shapes, .drops = "the head's cross products counted once", .edit = .{ .kernel = "ihc_seam", .from = "(s0 == u ? 1.0f : 2.0f)", .to = "1.0f" } },
    .{ .slug = "onegate", .function = "tf_hy4_ihc_add", .slot = .mix, .gate = .shapes, .drops = "stream 0's post gate on every stream", .edit = .{ .kernel = "tf_hy4_ihc_add", .from = "post[ulong(tid.y) * 4 + tid.x / a.width]", .to = "post[ulong(tid.y) * 4]" } },
    .{ .slug = "gain0", .function = "tf_hy4_draft_fuse", .slot = .fuse, .gate = .shapes, .drops = "column 0's gain on every column", .edit = .{ .kernel = "tf_hy4_draft_fuse", .from = "float(gain[tid.x])", .to = "float(gain[0])" } },
    .{ .slug = "subtract", .function = "tf_hy4_draft_residual", .slot = .residual, .gate = .shapes, .drops = "the branch subtracted", .edit = .{ .kernel = "tf_hy4_draft_residual", .from = "float(base[i]) + float(branch[o])", .to = "float(base[i]) - float(branch[o])" } },
};

/// A case's control by its slug (each case's slug names one of `controls`).
pub fn control(slug: []const u8) Ctl {
    return find(slug).?;
}

fn find(slug: []const u8) ?Ctl {
    for (controls) |k| if (std.mem.eql(u8, k.slug, slug)) return k;
    return null;
}

/// A control's text: ihc.metal with its edit.
fn controlText(a: std.mem.Allocator, k: Ctl) ![]u8 {
    return h.fl.edited(a, sources.hy4_flash_ihc, k.edit);
}

/// A control's pipeline, compiled from its text as the loaders compile ihc.metal (MLX's options).
pub fn pipeline(a: std.mem.Allocator, device: mtl.Device, k: Ctl) !mtl.Pipeline {
    const text = try controlText(a, k);
    defer a.free(text);
    const lib = try mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
    defer lib.deinit();
    return mtl.Pipeline.init(device, lib, k.function, false);
}

/// The loader's set with a control's pipeline in its slot; the copy owns nothing.
pub fn with(k: K, p: ?mtl.Pipeline, slot: Slot) K {
    const x = p orelse return k;
    var out = k;
    switch (slot) {
        .pre8 => out.pre8 = x,
        .finish8 => out.finish[1] = x,
        .tail8 => out.tail[1] = x,
        .seam8 => out.seam[1] = x,
        .mix => out.mix = x,
        .fuse, .residual => {},
    }
    return out;
}

/// The checkpoint's constants as ihc.zig's Weights.read and the seams pass them.
fn params(a: std.mem.Allocator) !m.Params {
    const cfg = try h.text(a);
    defer cfg.deinit();
    const t = cfg.value.text_config;
    if (t.hidden_size != h.hidden or t.hc_mult != m.streams or cfg.value.speculative_config.num_speculative_tokens + 1 != h.window_rows) return error.BadHy4Fixture;
    return .{ .eps = t.rms_norm_eps, .gate_eps = t.hc_eps, .magnitude = t.hc_magnitude };
}

pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    kernels: *std.ArrayList(g.Kernel),
    srv: ?*K = null,
    collapses: h.fl.List(collapse.Collapse) = .{},
    seams: h.fl.List(seam.Seam) = .{},
    elems: h.fl.List(draft.Elem) = .{},

    pub fn collect(s: *Suite, only: ?[]const u8) !void {
        var any = false;
        inline for (.{ collapse.cases, seam.cases, draft.cases }) |list| for (list) |v| {
            any = any or h.fl.wanted(only, v.name);
        };
        if (!any) return;
        const p = try params(s.a);
        const srv = try s.a.create(K);
        srv.* = K.load(s.device) catch |err| {
            s.a.destroy(srv);
            return err;
        };
        s.srv = srv;
        for (collapse.cases) |v| if (h.fl.wanted(only, v.name)) {
            try s.collapses.add(s.a, s.kernels, try collapse.Collapse.init(s.a, s.device, s.queue, srv, v, p));
        };
        for (seam.cases) |v| if (h.fl.wanted(only, v.name)) {
            try s.seams.add(s.a, s.kernels, try seam.Seam.init(s.a, s.device, s.queue, srv, v, p));
        };
        for (draft.cases) |v| if (h.fl.wanted(only, v.name)) {
            try s.elems.add(s.a, s.kernels, try draft.Elem.init(s.a, s.device, s.queue, srv, v, p));
        };
    }
    pub fn deinit(s: *Suite) void {
        s.collapses.deinit(s.a);
        s.seams.deinit(s.a);
        s.elems.deinit(s.a);
        if (s.srv) |k| {
            k.deinit();
            s.a.destroy(k);
        }
    }
};

/// Gate 5's texts for an `hy4 ihc ` launch: ihc.metal as the loaders compile it, or a control's edit of it.
pub fn limitsText(a: std.mem.Allocator, _: mtl.Device, name: []const u8) !?h.fl.Texts {
    const head = "hy4 ihc ";
    if (!std.mem.startsWith(u8, name, head)) return null;
    const rest = name[head.len..];
    if (std.mem.startsWith(u8, rest, "control ")) {
        const slug = rest["control ".len..];
        const k = find(slug[0 .. std.mem.indexOfScalar(u8, slug, ' ') orelse return error.UnknownLaunch]) orelse return error.UnknownLaunch;
        const text = try controlText(a, k);
        return .{ .here = text, .pre_m5 = text };
    }
    return .{ .here = sources.hy4_flash_ihc, .pre_m5 = sources.hy4_flash_ihc };
}
