//! Ripple's registrations as one set: its kernels (NVFP4, DFlash2 too) and controls, llama's, the draw; the roll call.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const sources = @import("kernel_sources");
const rc = @import("metal_ricursive.zig");
const norm = @import("metal_ricursive_norm.zig");
const attention = @import("metal_ricursive_attention.zig");
const sample = @import("metal_ricursive_sample.zig");
const fp4 = @import("metal_ricursive_fp4.zig");
const fp4_gemm = @import("metal_ricursive_fp4_gemm.zig");
const controls = @import("metal_ricursive_controls.zig");
const dflash2 = @import("metal_ricursive_dflash2.zig");
const llama = @import("metal_llama.zig");
const dense = @import("metal_llama_dense.zig");
const llama_attention = @import("metal_llama_attention.zig");
const c = contract.spec;
const g = contract.gates;
const family = rc.family;
const R = rc.R;

/// The label Ripple's registrations of llama's pipelines carry, those at o_proj's reduction, and its DFlash2 drafter's.
const label = "ricursive llama";
const o_label = label ++ " o";
const draft_label = label ++ " dflash2";
const matvec_rows = [_]u32{ 1, 2, 4, 8, 16, 32 };

fn wanted(only: ?[]const u8, name: []const u8) bool {
    const text = only orelse return true;
    return std.mem.indexOf(u8, name, text) != null;
}

/// The row projections at Ripple's reductions: the FFN's down (k 40,960) and those over the hidden width (k 5,120).
fn matvecShapes(per: u32) [2]c.Shape {
    return .{ .{ .rows = per, .n = 1024, .k = R.intermediate }, .{ .rows = per, .n = 1024, .k = R.hidden } };
}

/// A prompt chunk's GEMM at k 5,120 and the down's 40,960, whole or ragged tiles as the form takes them.
fn gemmShapes(am: bool, an: bool) [2]c.Shape {
    const n: usize = if (an) 1024 else 1088;
    return .{ .{ .rows = if (am) 64 else 33, .n = n, .k = R.hidden }, .{ .rows = if (am) 128 else 100, .n = n, .k = R.intermediate } };
}

/// o_proj's reduction over the heads (k 10,240): its 5,120 outputs, and a slice of them.
fn oMatvecShapes(per: u32) [2]c.Shape {
    const k = R.heads * rc.DH;
    return .{ .{ .rows = per, .n = 1024, .k = k }, .{ .rows = per, .n = R.hidden, .k = k } };
}

/// o_proj's GEMM over a prompt chunk, whole or ragged tiles as the form takes them.
fn oGemmShapes(am: bool, an: bool) [2]c.Shape {
    const n: usize = if (an) 1024 else 1088;
    const k = R.heads * rc.DH;
    return .{ .{ .rows = if (am) 64 else 33, .n = n, .k = k }, .{ .rows = if (am) 128 else 100, .n = n, .k = k } };
}

/// A decode window's merge over a sliding layer's handful of blocks, and over a global layer's most (state.max_blocks).
const merge_shapes = [_]c.Shape{ .{ .rows = 16, .n = R.heads, .k = 544 }, .{ .rows = 16, .n = R.heads, .k = 9 } };
const argmax_shapes = [_]c.Shape{.{ .rows = 1, .n = 1, .k = R.vocab }};

/// The drafter's one new reduction, fc's 25,600 (an absorb to 32 rows); its o_proj's 10,240 is the target's.
fn draftMatvecShapes(per: u32) [2]c.Shape {
    const k = R.hidden * dflash2.D.taps;
    return .{ .{ .rows = per, .n = 1024, .k = k }, .{ .rows = per, .n = R.hidden, .k = k } };
}

/// fc's GEMM over a 2,048-row absorb (its 64-row tiles repeat) and a ragged one; its 5,120 outputs are whole tiles.
fn draftGemmShapes(am: bool) [2]c.Shape {
    const k = R.hidden * dflash2.D.taps;
    return if (am) .{ .{ .rows = 64, .n = 1024, .k = k }, .{ .rows = 128, .n = 1024, .k = k } } else .{ .{ .rows = 33, .n = 1024, .k = k }, .{ .rows = 100, .n = 1024, .k = k } };
}

/// The drafter's merge over a sliding layer's most blocks (34) and the global layer's (130).
const draft_merge_shapes = [_]c.Shape{ .{ .rows = 16, .n = R.heads, .k = 34 }, .{ .rows = 16, .n = R.heads, .k = 130 } };

/// The probes made for a run, freed together.
pub const Set = struct {
    point: std.ArrayList(*rc.Pointwise) = .empty,
    add_norm: std.ArrayList(*norm.AddNorm) = .empty,
    prep: std.ArrayList(*norm.Prep) = .empty,
    decode: std.ArrayList(*attention.DecodeVec) = .empty,
    prompt: std.ArrayList(*attention.Prompt) = .empty,
    matvec: std.ArrayList(*dense.Matvec) = .empty,
    gemm: std.ArrayList(*dense.Gemm) = .empty,
    merge: std.ArrayList(*llama_attention.MergeVec) = .empty,
    argmax: std.ArrayList(*llama.Argmax) = .empty,
    sample: std.ArrayList(*sample.Sample) = .empty,
    fp4_matvec: std.ArrayList(*fp4.Matvec) = .empty,
    fp4_dequant: std.ArrayList(*fp4.Dequant) = .empty,
    fp4_gemm: std.ArrayList(*fp4_gemm.Gemm) = .empty,
    dflash2: dflash2.Suite = .{},

    pub fn deinit(s: *Set, a: std.mem.Allocator) void {
        s.dflash2.deinit(a);
        inline for (@typeInfo(Set).@"struct".field_names) |f| {
            if (comptime std.mem.eql(u8, f, "dflash2")) continue;
            const list = &@field(s, f);
            for (list.items) |p| {
                p.deinit();
                a.destroy(p);
            }
            list.deinit(a);
        }
    }

    /// `v` kept in `list` (freed with the set) and registered; a failure frees it.
    fn keep(a: std.mem.Allocator, list: anytype, v: anytype, kernels: *std.ArrayList(g.Kernel)) !void {
        var x = v;
        const p = a.create(@TypeOf(v)) catch |err| {
            x.deinit();
            return err;
        };
        p.* = x;
        list.append(a, p) catch |err| {
            p.deinit();
            a.destroy(p);
            return err;
        };
        try kernels.append(a, p.kernel());
    }

    /// Every registration whose name holds `only`; the roll call checks the registrations cover every pipeline.
    pub fn collect(s: *Set, a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, tensor: bool, kernels: *std.ArrayList(g.Kernel), only: ?[]const u8) !void {
        var functions: std.ArrayList([]const u8) = .empty;
        defer functions.deinit(a);
        for (std.enums.values(rc.Which)) |w| {
            try functions.append(a, rc.function(w));
            if (wanted(only, rc.spec(w).name)) try keep(a, &s.point, try rc.Pointwise.init(a, device, queue, w, .none), kernels);
        }
        try functions.append(a, "rc_add_norm");
        for (std.enums.values(norm.Form)) |f| if (wanted(only, norm.addNormSpec(f).name)) {
            try keep(a, &s.add_norm, try norm.AddNorm.init(a, device, queue, f, .none), kernels);
        };
        try functions.append(a, "rc_prep");
        for (std.enums.values(norm.Layer)) |l| if (wanted(only, norm.prepSpec(l).name)) {
            try keep(a, &s.prep, try norm.Prep.init(a, device, queue, l, .none), kernels);
        };
        try functions.append(a, "rc_decode_vec");
        for (std.enums.values(attention.Layer)) |l| if (wanted(only, attention.decodeSpec(l).name)) {
            try keep(a, &s.decode, try attention.DecodeVec.init(a, device, queue, l, .none), kernels);
        };
        // the prompt path's GEMM and attention run on the tensor units only (model.zig refuses a GPU without them)
        try functions.append(a, family.attention_name);
        for (std.enums.values(attention.Layer)) |l| if (tensor and wanted(only, attention.promptSpec(l).name)) {
            try keep(a, &s.prompt, try attention.Prompt.init(a, device, queue, l, .none), kernels);
        };
        for (matvec_rows, family.llama_names[0..matvec_rows.len]) |per, fun| {
            try functions.append(a, fun);
            var buf: [48]u8 = undefined;
            if (!wanted(only, try std.fmt.bufPrint(&buf, "{s} {s}", .{ label, fun }))) continue;
            var m = try dense.Matvec.init(a, device, queue, per, .none);
            m.retarget(label, matvecShapes(per)) catch |err| {
                m.deinit();
                return err;
            };
            try keep(a, &s.matvec, m, kernels);
        }
        for (family.gemm_names, 0..) |fun, i| {
            try functions.append(a, fun);
            var buf: [48]u8 = undefined;
            if (!tensor or !wanted(only, try std.fmt.bufPrint(&buf, "{s} {s}", .{ label, fun }))) continue;
            const am = i / 2 == 1;
            const an = i % 2 == 1;
            var m = try dense.Gemm.init(a, device, queue, am, an, .none);
            m.retarget(label, gemmShapes(am, an)) catch |err| {
                m.deinit();
                return err;
            };
            try keep(a, &s.gemm, m, kernels);
        }
        try functions.append(a, "ll_merge_vec");
        if (wanted(only, label ++ " ll_merge_vec")) {
            var m = try llama_attention.MergeVec.init(a, device, queue, .none);
            m.name = label ++ " ll_merge_vec";
            m.served = &merge_shapes;
            try keep(a, &s.merge, m, kernels);
        }
        try functions.append(a, "ll_argmax");
        if (wanted(only, label ++ " ll_argmax")) {
            var m = try llama.Argmax.init(a, device, queue, .none);
            m.name = label ++ " ll_argmax";
            m.served = &argmax_shapes;
            try keep(a, &s.argmax, m, kernels);
        }
        try functions.append(a, family.sample_name);
        if (wanted(only, sample.spec().name)) try keep(a, &s.sample, try sample.Sample.init(a, device, queue), kernels);
        try s.collectFp4(a, device, queue, tensor, kernels, only, &functions);
        try s.collectO(a, device, queue, tensor, kernels, only);
        try s.collectDraft(a, device, queue, tensor, kernels, only);
        try s.dflash2.collect(.{ .a = a, .device = device, .queue = queue, .kernels = kernels, .only = only }, &functions);
        try family.requireComplete(functions.items);
        try s.plant(a, device, queue, tensor, kernels, only);
    }

    /// The NVFP4 MLP's row kernels, its dequantization and (tensor units only) its four GEMM forms.
    fn collectFp4(s: *Set, a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, tensor: bool, kernels: *std.ArrayList(g.Kernel), only: ?[]const u8, functions: *std.ArrayList([]const u8)) !void {
        for (matvec_rows, family.fp4_names[0..matvec_rows.len]) |per, fun| {
            try functions.append(a, fun);
            var buf: [48]u8 = undefined;
            if (wanted(only, try std.fmt.bufPrint(&buf, "ricursive fp4 {s}", .{fun}))) try keep(a, &s.fp4_matvec, try fp4.Matvec.init(a, device, queue, per, .none), kernels);
        }
        try functions.append(a, "rc_fp4_dequant");
        if (wanted(only, "ricursive fp4 rc_fp4_dequant")) try keep(a, &s.fp4_dequant, try fp4.Dequant.init(a, device, queue, .none), kernels);
        for (family.fp4_gemm_names, 0..) |fun, i| {
            try functions.append(a, fun);
            var buf: [48]u8 = undefined;
            if (tensor and wanted(only, try std.fmt.bufPrint(&buf, "ricursive fp4 gemm {s}", .{fun}))) {
                try keep(a, &s.fp4_gemm, try fp4_gemm.Gemm.init(a, device, queue, i / 2 == 1, i % 2 == 1, .none), kernels);
            }
        }
    }

    /// llama's row projections and GEMMs at o_proj's reduction (already on the roll call above).
    fn collectO(s: *Set, a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, tensor: bool, kernels: *std.ArrayList(g.Kernel), only: ?[]const u8) !void {
        var buf: [48]u8 = undefined;
        for (matvec_rows, family.llama_names[0..matvec_rows.len]) |per, fun| {
            if (!wanted(only, try std.fmt.bufPrint(&buf, "{s} {s}", .{ o_label, fun }))) continue;
            var m = try dense.Matvec.init(a, device, queue, per, .none);
            m.retarget(o_label, oMatvecShapes(per)) catch |err| {
                m.deinit();
                return err;
            };
            try keep(a, &s.matvec, m, kernels);
        }
        for (family.gemm_names, 0..) |fun, i| {
            if (!tensor or !wanted(only, try std.fmt.bufPrint(&buf, "{s} {s}", .{ o_label, fun }))) continue;
            const am = i / 2 == 1;
            const an = i % 2 == 1;
            var m = try dense.Gemm.init(a, device, queue, am, an, .none);
            m.retarget(o_label, oGemmShapes(am, an)) catch |err| {
                m.deinit();
                return err;
            };
            try keep(a, &s.gemm, m, kernels);
        }
    }

    /// llama's pipelines at the shapes only the DFlash2 drafter runs them at (already on the roll call above).
    fn collectDraft(s: *Set, a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, tensor: bool, kernels: *std.ArrayList(g.Kernel), only: ?[]const u8) !void {
        var buf: [64]u8 = undefined;
        for (matvec_rows, family.llama_names[0..matvec_rows.len]) |per, fun| {
            if (!wanted(only, try std.fmt.bufPrint(&buf, "{s} {s}", .{ draft_label, fun }))) continue;
            var m = try dense.Matvec.init(a, device, queue, per, .none);
            m.retarget(draft_label, draftMatvecShapes(per)) catch |err| {
                m.deinit();
                return err;
            };
            try keep(a, &s.matvec, m, kernels);
        }
        // fc's 5,120 outputs are whole 128-column tiles: only the forms with `an` run
        for ([_]bool{ false, true }) |am| {
            if (!tensor or !wanted(only, try std.fmt.bufPrint(&buf, "{s} ll_gemm_{d}1", .{ draft_label, @intFromBool(am) }))) continue;
            var m = try dense.Gemm.init(a, device, queue, am, true, .none);
            m.retarget(draft_label, draftGemmShapes(am)) catch |err| {
                m.deinit();
                return err;
            };
            try keep(a, &s.gemm, m, kernels);
        }
        if (wanted(only, draft_label ++ " ll_merge_vec")) {
            var m = try llama_attention.MergeVec.init(a, device, queue, .none);
            m.name = draft_label ++ " ll_merge_vec";
            m.served = &draft_merge_shapes;
            try keep(a, &s.merge, m, kernels);
        }
    }

    /// Each control: its kernel built with one planted fault, registered beside the kernels.
    fn plant(s: *Set, a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, tensor: bool, kernels: *std.ArrayList(g.Kernel), only: ?[]const u8) !void {
        for (std.enums.values(controls.Mutation)) |m| {
            if (m == .none or !wanted(only, m.name())) continue;
            if (m.dflash2()) {
                try s.dflash2.plant(.{ .a = a, .device = device, .queue = queue, .kernels = kernels, .only = only }, m);
                continue;
            }
            switch (m) {
                .embed_own_row => try keep(a, &s.point, try rc.Pointwise.init(a, device, queue, .embed, m), kernels),
                .gate_early_round => try keep(a, &s.point, try rc.Pointwise.init(a, device, queue, .gate, m), kernels),
                .clamp_no_lower => try keep(a, &s.point, try rc.Pointwise.init(a, device, queue, .silu_clamp, m), kernels),
                .add_norm_no_residual => try keep(a, &s.add_norm, try norm.AddNorm.init(a, device, queue, .add, m), kernels),
                .lerp_no_blend => try keep(a, &s.add_norm, try norm.AddNorm.init(a, device, queue, .lerp, m), kernels),
                .prep_ropes_global => try keep(a, &s.prep, try norm.Prep.init(a, device, queue, .global, m), kernels),
                .prep_next_slot => try keep(a, &s.prep, try norm.Prep.init(a, device, queue, .sliding, m), kernels),
                .decode_past_window => try keep(a, &s.decode, try attention.DecodeVec.init(a, device, queue, .sliding, m), kernels),
                .prompt_past_window, .prompt_short_ring => if (tensor) {
                    try keep(a, &s.prompt, try attention.Prompt.init(a, device, queue, .sliding, m), kernels);
                },
                .fp4_high_first, .fp4_scale_bias, .fp4_global_late => try keep(a, &s.fp4_matvec, try fp4.Matvec.init(a, device, queue, 4, m), kernels),
                .fp4_dequant_unscaled => try keep(a, &s.fp4_dequant, try fp4.Dequant.init(a, device, queue, m), kernels),
                .fp4_gemm_no_global => if (tensor) {
                    try keep(a, &s.fp4_gemm, try fp4_gemm.Gemm.init(a, device, queue, false, false, m), kernels);
                },
                .none, .df_tap_next_slot, .df_norm_half_mean, .df_norm16_no_weight, .df_embed_own_row, .df_prep_ropes_nope, .df_conv_three_taps, .df_silu_no_negative, .df_vec_no_floor, .df_topk_ascending, .df_walk_from_pending => unreachable,
            }
        }
    }
};

/// Gate 5's texts for `ricursive ` launches: the glue, DFlash2's, the draw, or the attention (nax.h inlined pre-M5).
pub fn limitsText(a: std.mem.Allocator, device: mtl.Device, name: []const u8) !struct { here: []const u8, pre_m5: []const u8 } {
    if (std.mem.startsWith(u8, name, "ricursive dflash2 ")) return .{ .here = sources.ricursive_dflash2, .pre_m5 = sources.ricursive_dflash2 };
    if (std.mem.indexOf(u8, name, " attention ") != null) {
        const head = try family.attentionHeader(a);
        const body = try std.mem.replaceOwned(u8, a, sources.ricursive_attention_nax, "#include \"../nax.h\"", sources.nax);
        const pre = try std.mem.concat(a, u8, &.{ head, "#define TF_SIMD_FRAGS 1\n", body });
        return .{ .here = if (device.tensorUnits()) try family.attentionSource(device, a) else pre, .pre_m5 = pre };
    }
    if (std.mem.indexOf(u8, name, " fp4 gemm ") != null) {
        const pre = try family.fp4GemmSource(device, a, false);
        return .{ .here = if (device.tensorUnits()) try family.fp4GemmSource(device, a, true) else pre, .pre_m5 = pre };
    }
    if (std.mem.indexOf(u8, name, " fp4 ") != null) return .{ .here = sources.ricursive_fp4, .pre_m5 = sources.ricursive_fp4 };
    if (std.mem.indexOf(u8, name, " sample ") != null) return .{ .here = sources.nemotron_sample, .pre_m5 = sources.nemotron_sample };
    const glue = try family.glueSource(a);
    return .{ .here = glue, .pre_m5 = glue };
}
