//! The LiLiCorr drafter's BODY: the DFlash2 stack under the correlator (the checkpoint's other tensors, audited).
const std = @import("std");
const core = @import("core");
const head = @import("cuda_lilicorr_head.zig");
const nvfp4 = @import("cuda_lilicorr_nvfp4.zig");

pub const conv_proj_width = 1024; // the conv kernel projections' output (the dynamic conv weights)
pub const conv_base_rows = 2; // a conv base kernel's [2, 2, hidden] tile

/// The drafter body's geometry, from the L2 config parse (Bolt: 5/4096/12288/4096/256).
pub const Geo = struct {
    layers: usize,
    hidden: usize,
    intermediate: usize,
    q_width: usize,
    kv_width: usize,
};

/// One layer's weights as the block pass reads them: f32, widened bf16 and dequantized nvfp4.
pub const Layer = struct {
    input_norm: []f32, // [hidden]
    post_norm: []f32, // [hidden]
    q_norm: []f32, // [128]
    k_norm: []f32, // [128]
    sink: []f32, // [q heads] the attention's per-head sink bias
    attn_base: []f32, // [2, 2, hidden]
    attn_proj: []f32, // [conv_proj_width, hidden]
    q: []f32, // [q_width, hidden]
    k: []f32, // [kv_width, hidden]
    v: []f32, // [kv_width, hidden]
    o: []f32, // [hidden, q_width]
    mlp_base: []f32, // [2, 2, hidden]
    mlp_proj: []f32, // [conv_proj_width, hidden]
    gate: []f32, // [intermediate, hidden]
    up: []f32, // [intermediate, hidden]
    down: []f32, // [hidden, intermediate]
};

/// The body's weights: the root projections and norms, then the layers.
pub const Weights = struct {
    fc: []f32, // [hidden, 6 * hidden] — the tap ring's 6 raw rows projected (nvfp4)
    hidden_norm: []f32, // [hidden]
    final_norm: []f32, // [hidden]
    layers: []Layer,
};

/// One expected tensor: name (layers.{i}. prefix), dtype tag and shape; coreKind maps the tag only inside load().
const Kind = enum { bf16, u8, f8_e4m3, f32 };
const Expect = struct { suffix: []const u8, dtype: Kind, shape: []const usize, nvfp4: bool = false };

fn coreKind(k: Kind) core.checkpoint.DType {
    return switch (k) {
        .bf16 => .bf16,
        .u8 => .u8,
        .f8_e4m3 => .f8_e4m3,
        .f32 => .f32,
    };
}

/// The inventory of one layer, audited against the exported checkpoint's layout.
fn layerInventory(g: Geo) [22]Expect {
    const h = g.hidden;
    const qi = g.intermediate / 2; // nvfp4: the header stores packed columns
    return .{
        .{ .suffix = "attention_conv.base_kernel", .dtype = .bf16, .shape = &.{ conv_base_rows, conv_base_rows, h } },
        .{ .suffix = "attention_conv.kernel_projection.weight", .dtype = .bf16, .shape = &.{ conv_proj_width, h } },
        .{ .suffix = "input_layernorm.weight", .dtype = .bf16, .shape = &.{h} },
        .{ .suffix = "mlp.down_proj.weight", .dtype = .u8, .shape = &.{ h, qi }, .nvfp4 = true },
        .{ .suffix = "mlp.down_proj.weight_scale", .dtype = .f8_e4m3, .shape = &.{ h, g.intermediate / 16 } },
        .{ .suffix = "mlp.down_proj.weight_scale_2", .dtype = .f32, .shape = &.{} },
        .{ .suffix = "mlp.gate_proj.weight", .dtype = .u8, .shape = &.{ g.intermediate, h / 2 }, .nvfp4 = true },
        .{ .suffix = "mlp.gate_proj.weight_scale", .dtype = .f8_e4m3, .shape = &.{ g.intermediate, h / 16 } },
        .{ .suffix = "mlp.gate_proj.weight_scale_2", .dtype = .f32, .shape = &.{} },
        .{ .suffix = "mlp.up_proj.weight", .dtype = .u8, .shape = &.{ g.intermediate, h / 2 }, .nvfp4 = true },
        .{ .suffix = "mlp.up_proj.weight_scale", .dtype = .f8_e4m3, .shape = &.{ g.intermediate, h / 16 } },
        .{ .suffix = "mlp.up_proj.weight_scale_2", .dtype = .f32, .shape = &.{} },
        .{ .suffix = "mlp_conv.base_kernel", .dtype = .bf16, .shape = &.{ conv_base_rows, conv_base_rows, h } },
        .{ .suffix = "mlp_conv.kernel_projection.weight", .dtype = .bf16, .shape = &.{ conv_proj_width, h } },
        .{ .suffix = "post_attention_layernorm.weight", .dtype = .bf16, .shape = &.{h} },
        .{ .suffix = "self_attn.attention_sink_bias", .dtype = .bf16, .shape = &.{g.q_width / 128} },
        .{ .suffix = "self_attn.k_norm.weight", .dtype = .bf16, .shape = &.{128} },
        .{ .suffix = "self_attn.k_proj.weight", .dtype = .bf16, .shape = &.{ g.kv_width, h } },
        .{ .suffix = "self_attn.o_proj.weight", .dtype = .bf16, .shape = &.{ h, g.q_width } },
        .{ .suffix = "self_attn.q_norm.weight", .dtype = .bf16, .shape = &.{128} },
        .{ .suffix = "self_attn.q_proj.weight", .dtype = .bf16, .shape = &.{ g.q_width, h } },
        .{ .suffix = "self_attn.v_proj.weight", .dtype = .bf16, .shape = &.{ g.kv_width, h } },
    };
}

test "the layer inventory covers exactly 22 tensors with the Bolt shapes" {
    const g: Geo = .{ .layers = 5, .hidden = 4096, .intermediate = 12288, .q_width = 4096, .kv_width = 256 };
    const inv = layerInventory(g);
    try std.testing.expectEqual(@as(usize, 22), inv.len);
    const down = inv[3];
    try std.testing.expectEqual(@as(usize, 6144), down.shape[1]); // packed: 12288 nvfp4 in
    const gate = inv[6];
    try std.testing.expectEqual(Kind.u8, gate.dtype);
    try std.testing.expectEqual(@as(usize, 2048), gate.shape[1]);
    const sink = inv[15];
    try std.testing.expectEqual(@as(usize, 32), sink.shape[0]);
    var quantized: usize = 0;
    for (inv) |e| quantized += @intFromBool(e.nvfp4);
    try std.testing.expectEqual(@as(usize, 3), quantized); // gate, up, down only
}

/// The whole body from an open checkpoint; the head's lilicorr.* stay with head.load (either missing refuses).
pub fn load(gpa: std.mem.Allocator, ck: *core.Checkpoint, g: Geo) !Weights {
    const a = gpa;
    var w: Weights = .{ .fc = undefined, .hidden_norm = undefined, .final_norm = undefined, .layers = undefined };
    w.fc = try head.dequant(a, ck, "fc", g.hidden, 6 * g.hidden);
    w.hidden_norm = try bf16Of(a, ck, "hidden_norm.weight", &.{g.hidden});
    w.final_norm = try bf16Of(a, ck, "norm.weight", &.{g.hidden});
    w.layers = try a.alloc(Layer, g.layers);
    for (w.layers, 0..) |*l, i| {
        var buf: [96]u8 = undefined;
        const pre = try std.fmt.bufPrint(&buf, "layers.{d}.", .{i});
        l.* = .{
            .input_norm = undefined,
            .post_norm = undefined,
            .q_norm = undefined,
            .k_norm = undefined,
            .sink = undefined,
            .attn_base = undefined,
            .attn_proj = undefined,
            .q = undefined,
            .k = undefined,
            .v = undefined,
            .o = undefined,
            .mlp_base = undefined,
            .mlp_proj = undefined,
            .gate = undefined,
            .up = undefined,
            .down = undefined,
        };
        for (layerInventory(g), 0..) |e, t| {
            var nb: [128]u8 = undefined;
            const name = try std.fmt.bufPrint(&nb, "{s}{s}", .{ pre, e.suffix });
            if (e.nvfp4) {
                const proj = name[0 .. name.len - ".weight".len];
                switch (t) {
                    3 => l.down = try head.dequant(a, ck, proj, e.shape[0], e.shape[1] * 2),
                    6 => l.gate = try head.dequant(a, ck, proj, e.shape[0], e.shape[1] * 2),
                    9 => l.up = try head.dequant(a, ck, proj, e.shape[0], e.shape[1] * 2),
                    else => return error.UnexpectedTensor,
                }
                continue;
            }
            const t2 = try ck.expect(name, coreKind(e.dtype), e.shape);
            const v: []f32 = try head.widen(a, t2);
            switch (t) {
                0 => l.attn_base = v,
                1 => l.attn_proj = v,
                2 => l.input_norm = v,
                12 => l.mlp_base = v,
                13 => l.mlp_proj = v,
                14 => l.post_norm = v,
                15 => l.sink = v,
                16 => l.k_norm = v,
                17 => l.k = v,
                18 => l.o = v,
                19 => l.q_norm = v,
                20 => l.q = v,
                21 => l.v = v,
                else => {},
            }
        }
    }
    return w;
}

fn bf16Of(gpa: std.mem.Allocator, ck: *core.Checkpoint, name: []const u8, shape: []const usize) ![]f32 {
    const t = try ck.expect(name, .bf16, shape);
    return head.widen(gpa, t);
}

/// A load sanity check: mangled bf16 widening lands huge magnitudes; real weights stay small.
pub fn sane(w: *const Weights) bool {
    for ([_][]f32{ w.hidden_norm, w.final_norm }) |v| if (!bounded(v, 16.0)) return false;
    for (w.layers) |l| {
        if (!bounded(l.sink, 64.0) or !bounded(l.input_norm, 16.0) or !bounded(l.q, 64.0)) return false;
        if (!bounded(l.post_norm, 16.0) or !bounded(l.mlp_proj, 64.0)) return false;
    }
    return bounded(w.fc, 64.0);
}

fn bounded(v: []const f32, limit: f64) bool {
    for (v) |x| if (!std.math.isFinite(x) or @abs(x) > limit) return false;
    return true;
}

test "sane refuses the f16-widen corruption class on a synthetic body" {
    var w: Weights = .{ .fc = &.{}, .hidden_norm = &.{}, .final_norm = &.{}, .layers = &.{} };
    var norms: [4]f32 = .{ 1.0, 0.9, 1.1, 1.0 };
    w.hidden_norm = &norms;
    try std.testing.expect(sane(&w));
    norms[0] = 1.875e19; // what a mis-widened exponent reads as
    try std.testing.expect(!sane(&w));
}
