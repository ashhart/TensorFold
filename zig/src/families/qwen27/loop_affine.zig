//! MLX-affine SenseNova-Loop-9B weight loading for physical layers, head, and optional quantized embeddings.
const std = @import("std");
const mtl = @import("metal");
const core = @import("core");
const cp = core.checkpoint_metal;
const sn_cfg = core.sensenova_loop.config;
const cfg = @import("config.zig");
const wts = @import("gpu_weights.zig");
const loop = @import("loop.zig");
const projection = @import("projection.zig");
const widths = @import("widths.zig");

const AffinePart = struct { name: []const u8, n: usize, w: widths.Width };

/// Whether a checkpoint tensor is repacked, dtype-rounded, or ignored during SenseNova MLX-affine loading.
pub fn repackedAffine(name: []const u8, rank: usize) bool {
    if (sn_cfg.isSkippedTensor(name)) return true;
    if (wts.repacked(name, rank)) return true;
    const dot = std.mem.lastIndexOfScalar(u8, name, '.') orelse return false;
    const leaf = name[dot + 1 ..];
    return std.mem.eql(u8, leaf, "A_log") or std.mem.eql(u8, leaf, "dt_bias");
}

fn loadAffine(
    a: std.mem.Allocator,
    device: mtl.Device,
    ck: *const cp.Checkpoint,
    stage: *cp.Stage,
    owned: *std.ArrayList(mtl.Buffer),
    runs_list: *std.ArrayList([]projection.Run),
    parts: []const AffinePart,
    k: usize,
) !projection.Linear {
    if (parts.len == 0 or parts.len > 3) return error.BadProjectionShape;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const ta = arena.allocator();
    const names = try ta.alloc([]const u8, parts.len * 3);
    for (parts, 0..) |p, i| {
        if (k * p.w.bits % 32 != 0 or k % p.w.group != 0) return error.BadNativeTensor;
        for ([_][]const u8{ ".weight", ".scales", ".biases" }, 0..) |suf, j| {
            names[i * 3 + j] = try std.mem.concat(ta, u8, &.{ p.name, suf });
        }
    }
    try stage.reserve(ck, names);
    defer stage.reset();

    var triples: [3]projection.Triple = undefined;
    var ws_arr: [3]widths.Width = undefined;
    for (parts, triples[0..parts.len], ws_arr[0..parts.len], 0..) |p, *t, *wv, i| {
        const w_t = try loop.checked(try stage.read(ck, names[i * 3]), .u32, &.{ p.n, k * p.w.bits / 32 });
        const s_t = try loop.checked(try stage.read(ck, names[i * 3 + 1]), .bf16, &.{ p.n, k / p.w.group });
        const b_t = try loop.checked(try stage.read(ck, names[i * 3 + 2]), .bf16, &.{ p.n, k / p.w.group });
        t.* = .{ .weight = w_t, .scales = s_t, .biases = b_t, .bits = p.w.bits, .group = p.w.group, .gn = p.n };
        wv.* = p.w;
    }
    var buf: [3][2]usize = undefined;
    const list = widths.runs(ws_arr[0..parts.len], &buf);
    if (list.len == 1) return projection.prepare(device, a, owned, triples[0..parts.len], 32);

    const runs = try a.alloc(projection.Run, list.len);
    runs_list.append(a, runs) catch |err| {
        a.free(runs);
        return err;
    };
    var first: u32 = 0;
    for (list, runs) |r, *run| {
        run.* = .{ .linear = try projection.prepare(device, a, owned, triples[r[0]..][0..r[1]], 32), .first = first };
        first += run.linear.n;
    }
    var fused = runs[0].linear;
    fused.n = first;
    fused.gn = 0;
    for (runs) |run| fused.gn += run.linear.gn;
    fused.raw = null;
    fused.runs = runs;
    return fused;
}

fn loadPhysicalLayer(
    a: std.mem.Allocator,
    device: mtl.Device,
    ck: *const cp.Checkpoint,
    stage: *cp.Stage,
    owned: *std.ArrayList(mtl.Buffer),
    runs_list: *std.ArrayList([]projection.Run),
    c: cfg.Config,
    w: [17]widths.Width,
    prefix: []const u8,
    i: usize,
    need_mhc: bool,
) !struct { layer: wts.Layer, mhc: loop.MhcLayer } {
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const ta = arena.allocator();
    const lp = try std.fmt.allocPrint(ta, "{s}layers.{d}.", .{ prefix, i });
    const cat = struct {
        fn s(alloc: std.mem.Allocator, p: []const u8, sub: []const u8) ![]const u8 {
            return std.mem.concat(alloc, u8, &.{ p, sub });
        }
    }.s;
    const M = widths.Member;
    const mixer: @FieldType(wts.Layer, "mixer") = switch (c.physKind(i)) {
        .linear => .{ .linear = .{
            .qkv = try loadAffine(a, device, ck, stage, owned, runs_list, &.{.{ .name = try cat(ta, lp, "linear_attn.in_proj_qkv"), .n = c.gdnQkvDim(), .w = w[@backingInt(M.qkv)] }}, c.hidden),
            .zba = try loadAffine(a, device, ck, stage, owned, runs_list, &.{
                .{ .name = try cat(ta, lp, "linear_attn.in_proj_z"), .n = c.gdnValueDim(), .w = w[@backingInt(M.z)] },
                .{ .name = try cat(ta, lp, "linear_attn.in_proj_b"), .n = c.v_heads, .w = w[@backingInt(M.b)] },
                .{ .name = try cat(ta, lp, "linear_attn.in_proj_a"), .n = c.v_heads, .w = w[@backingInt(M.a)] },
            }, c.hidden),
            .out = try loadAffine(a, device, ck, stage, owned, runs_list, &.{.{ .name = try cat(ta, lp, "linear_attn.out_proj"), .n = c.hidden, .w = w[@backingInt(M.out)] }}, c.gdnValueDim()),
            .conv = try loop.checkedConv(try ck.get(try cat(ta, lp, "linear_attn.conv1d.weight")), c.gdnQkvDim(), c.conv_kernel),
            .a_log = try loop.roundScalarBf16(a, device, ck, stage, owned, try cat(ta, lp, "linear_attn.A_log"), c.v_heads),
            .dt = try loop.roundScalarBf16(a, device, ck, stage, owned, try cat(ta, lp, "linear_attn.dt_bias"), c.v_heads),
            .norm = try loop.checked(try ck.get(try cat(ta, lp, "linear_attn.norm.weight")), .bf16, &.{c.dv}),
        } },
        .attention => .{ .attention = .{
            .q = try loadAffine(a, device, ck, stage, owned, runs_list, &.{.{ .name = try cat(ta, lp, "self_attn.q_proj"), .n = c.qDim(), .w = w[@backingInt(M.q)] }}, c.hidden),
            .kv = try loadAffine(a, device, ck, stage, owned, runs_list, &.{
                .{ .name = try cat(ta, lp, "self_attn.k_proj"), .n = c.kvDim(), .w = w[@backingInt(M.k)] },
                .{ .name = try cat(ta, lp, "self_attn.v_proj"), .n = c.kvDim(), .w = w[@backingInt(M.v)] },
            }, c.hidden),
            .out = try loadAffine(a, device, ck, stage, owned, runs_list, &.{.{ .name = try cat(ta, lp, "self_attn.o_proj"), .n = c.hidden, .w = w[@backingInt(M.o)] }}, c.heads * c.head_dim),
            .q_norm = try loop.checked(try ck.get(try cat(ta, lp, "self_attn.q_norm.weight")), .bf16, &.{c.head_dim}),
            .k_norm = try loop.checked(try ck.get(try cat(ta, lp, "self_attn.k_norm.weight")), .bf16, &.{c.head_dim}),
        } },
    };
    const gu = try loadAffine(a, device, ck, stage, owned, runs_list, &.{
        .{ .name = try cat(ta, lp, "mlp.gate_proj"), .n = c.intermediate, .w = w[@backingInt(M.gate)] },
        .{ .name = try cat(ta, lp, "mlp.up_proj"), .n = c.intermediate, .w = w[@backingInt(M.up)] },
    }, c.hidden);
    const down = try loadAffine(a, device, ck, stage, owned, runs_list, &.{.{ .name = try cat(ta, lp, "mlp.down_proj"), .n = c.hidden, .w = w[@backingInt(M.down)] }}, c.intermediate);
    const layer = wts.Layer{
        .norm = try loop.checked(try ck.get(try cat(ta, lp, "input_layernorm.weight")), .bf16, &.{c.hidden}),
        .post_norm = try loop.checked(try ck.get(try cat(ta, lp, "post_attention_layernorm.weight")), .bf16, &.{c.hidden}),
        .gu = gu,
        .down = down,
        .mixer = mixer,
    };
    const ml: loop.MhcLayer = if (need_mhc) .{
        .attn_w = try loop.checked(try ck.get(try cat(ta, lp, "mhc_attn.weight")), .f32, &.{ 8, 2 * c.hidden }),
        .attn_b = try loop.checked(try ck.get(try cat(ta, lp, "mhc_attn.bias")), .f32, &.{8}),
        .ffn_w = try loop.checked(try ck.get(try cat(ta, lp, "mhc_ffn.weight")), .f32, &.{ 8, 2 * c.hidden }),
        .ffn_b = try loop.checked(try ck.get(try cat(ta, lp, "mhc_ffn.bias")), .f32, &.{8}),
    } else undefined;
    return .{ .layer = layer, .mhc = ml };
}

fn loadEmbed(a: std.mem.Allocator, ck: *const cp.Checkpoint, text_ns: []const u8, c: cfg.Config, ew: widths.Width) !struct { bf16: ?cp.Tensor, affine: projection.Triple } {
    if (ew.kind == .bf16) {
        const ename = try std.mem.concat(a, u8, &.{ text_ns, "embed_tokens.weight" });
        defer a.free(ename);
        const emb = try loop.checked(try ck.get(ename), .bf16, &.{ c.vocab, c.hidden });
        return .{ .bf16 = emb, .affine = .{ .weight = emb, .scales = emb, .biases = emb, .bits = 4, .group = 64 } };
    }
    if (c.hidden * ew.bits % 32 != 0 or c.hidden % ew.group != 0) return error.BadNativeTensor;
    const wn = try std.mem.concat(a, u8, &.{ text_ns, "embed_tokens.weight" });
    defer a.free(wn);
    const sn = try std.mem.concat(a, u8, &.{ text_ns, "embed_tokens.scales" });
    defer a.free(sn);
    const bn = try std.mem.concat(a, u8, &.{ text_ns, "embed_tokens.biases" });
    defer a.free(bn);
    return .{ .bf16 = null, .affine = .{
        .weight = try loop.checked(try ck.get(wn), .u32, &.{ c.vocab, c.hidden * ew.bits / 32 }),
        .scales = try loop.checked(try ck.get(sn), .bf16, &.{ c.vocab, c.hidden / ew.group }),
        .biases = try loop.checked(try ck.get(bn), .bf16, &.{ c.vocab, c.hidden / ew.group }),
        .bits = ew.bits,
        .group = ew.group,
    } };
}

/// Load physical MLX-affine SenseNova-Loop-9B weights and map execution-step aliases.
pub fn loadWeights(a: std.mem.Allocator, device: mtl.Device, ck: *const cp.Checkpoint, c: cfg.Config, ws: *const widths.Widths) !wts.Weights {
    const ns = try loop.namespace(ck);
    defer if (!std.mem.eql(u8, ns.head, "lm_head")) a.free(ns.head);
    var owned: std.ArrayList(mtl.Buffer) = .empty;
    errdefer {
        for (owned.items) |b| b.deinit();
        owned.deinit(a);
    }
    var runs: std.ArrayList([]projection.Run) = .empty;
    errdefer {
        for (runs.items) |r| a.free(r);
        runs.deinit(a);
    }
    const phys_n = c.physicalLayers();
    const physical = try a.alloc(wts.Layer, phys_n);
    errdefer a.free(physical);
    const exec_layers = try a.alloc(wts.Layer, c.layers);
    errdefer a.free(exec_layers);
    const mhc = try a.alloc(loop.MhcLayer, phys_n);
    errdefer a.free(mhc);
    const need_mhc = try a.alloc(bool, phys_n);
    defer a.free(need_mhc);
    @memset(need_mhc, false);
    for (0..c.layers) |exec| if (c.mhc_roles[exec] != .none) {
        need_mhc[c.exec_phys[exec]] = true;
    };

    var stage = cp.Stage{ .device = device };
    defer stage.deinit();
    for (0..phys_n) |i| {
        const res = try loadPhysicalLayer(a, device, ck, &stage, &owned, &runs, c, ws.layers[i], ns.text, i, need_mhc[i]);
        physical[i] = res.layer;
        mhc[i] = res.mhc;
    }
    const norm_name = try std.mem.concat(a, u8, &.{ ns.text, "norm.weight" });
    defer a.free(norm_name);
    const norm = try loop.checked(try ck.get(norm_name), .bf16, &.{c.hidden});
    const head = try loadAffine(a, device, ck, &stage, &owned, &runs, &.{.{ .name = ns.head, .n = c.vocab, .w = ws.head }}, c.hidden);
    const emb = try loadEmbed(a, ck, ns.text, c, ws.embed);

    // Apply tensor-unit regOrder strictly to unique physical matrices before aliasing exec steps.
    if (device.tensorUnits()) try wts.laneOrder(a, device, physical, head);
    for (exec_layers, 0..) |*dst, exec| dst.* = physical[c.exec_phys[exec]];
    return .{ .allocator = a, .layers = exec_layers, .physical_layers = physical, .embed = emb.affine, .embed_bf16 = emb.bf16, .norm = norm, .head = head, .owned = owned, .runs = runs, .mhc = mhc };
}

test "repackedAffine matches SenseNova MLX-affine tensor lifecycle" {
    try std.testing.expect(repackedAffine("model.language_model.layers.0.mlp.gate_proj.weight", 2));
    try std.testing.expect(repackedAffine("model.language_model.layers.3.self_attn.v_proj.scales", 2));
    try std.testing.expect(repackedAffine("model.language_model.layers.0.linear_attn.in_proj_qkv.biases", 2));
    try std.testing.expect(repackedAffine("lm_head.weight", 2));
    try std.testing.expect(repackedAffine("model.language_model.layers.0.linear_attn.A_log", 1));
    try std.testing.expect(repackedAffine("model.language_model.layers.0.linear_attn.dt_bias", 1));
    try std.testing.expect(repackedAffine("model.language_model.loop_gate.logits", 1));
    try std.testing.expect(!repackedAffine("model.language_model.embed_tokens.weight", 2));
    try std.testing.expect(!repackedAffine("model.language_model.embed_tokens.scales", 2));
    try std.testing.expect(!repackedAffine("model.language_model.layers.4.mhc_attn.weight", 2));
    try std.testing.expect(!repackedAffine("model.language_model.layers.0.linear_attn.conv1d.weight", 3));
    try std.testing.expect(!repackedAffine("model.language_model.norm.weight", 1));
}

test "admit resolves and admits SenseNova-Loop MLX-affine g64 4/5/6/8-bit configs" {
    const a = std.testing.allocator;
    const lt = "\"linear_attention\",\"linear_attention\",\"linear_attention\",\"full_attention\",";
    const lt_last = "\"linear_attention\",\"linear_attention\",\"linear_attention\",\"full_attention\"";
    for ([_]u8{ 4, 5, 6, 8 }) |bits| {
        const json = try std.fmt.allocPrint(a,
            \\{{"model_type":"sensenova_loop","loop_settings":{{"counts":[2],"end":28,"eval_count":2,"start":4}},
            \\"mhc":{{"end":28,"fixed_identity":false,"iterations":20,"pre_scale":1.0,"start":4,"streams":2}},
            \\"tie_word_embeddings":false,"text_config":{{"model_type":"qwen3_5_text","hidden_size":4096,
            \\"intermediate_size":12288,"num_hidden_layers":32,"num_attention_heads":16,"num_key_value_heads":4,
            \\"head_dim":256,"vocab_size":248320,"linear_num_key_heads":16,"linear_num_value_heads":32,
            \\"linear_key_head_dim":128,"linear_value_head_dim":128,"linear_conv_kernel_dim":4,
            \\"full_attention_interval":4,"max_position_embeddings":524288,"tie_word_embeddings":false,
            \\"eos_token_id":248044,"layer_types":[{s}{s}{s}{s}{s}{s}{s}{s}],"rope_parameters":{{
            \\"mrope_interleaved":true,"mrope_section":[11,11,10],"partial_rotary_factor":0.25,
            \\"rope_theta":10000000,"rope_type":"yarn","factor":2.0,"original_max_position_embeddings":262144}}}},
            \\"quantization":{{"bits":{d},"group_size":64,"mode":"affine"}}}}
        , .{ lt, lt, lt, lt, lt, lt, lt, lt_last, bits });
        defer a.free(json);
        const c = try cfg.parse(a, json, null);
        var ws = try widths.Widths.resolveModel(a, json, c);
        defer ws.deinit();
        try std.testing.expectEqual(widths.Kind.affine, ws.head.kind);
        try std.testing.expectEqual(bits, ws.head.bits);
        var why: widths.Why = .{};
        try widths.admit(&ws, c, .{ .tensor_units = true }, &why);
    }
}
