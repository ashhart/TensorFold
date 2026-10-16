//! One resident copy of every decoder layer. The prompt forward reads these, it does not reload them.

const std = @import("std");
const cuda = @import("cuda");
const embed = @import("cuda_embed.zig");
const gdn = @import("cuda_gdn.zig");
const wt = @import("cuda_prompt_wt.zig");
const projections = @import("cuda_projection_load.zig");

const layers_n = wt.layers_n;
const dims = wt.dims;
const wide = wt.wide;
const experts_n = wt.experts_n;
const moe_w = wt.moe_w;
const head_dim = wt.head_dim;
const index_dim = wt.index_dim;
const proj_n = wt.proj_n;
const out_k = wt.out_k;
const ple_layer = wt.ple_layer;
const block_bytes = wt.block_bytes;
const cint = wt.cint;
const Store = wt.Store;
const Face = wt.Face;
const Q4 = wt.Q4;
const HcW = wt.HcW;
const linearLayer = wt.linearLayer;
const asF32 = wt.asF32;
const join3 = wt.join3;
const loadHc = wt.loadHc;

const OwnedFace = struct {
    face: Face,
    w: []u8,
    s: []u8,
    b: []u8,

    fn free(self: *OwnedFace, gpa: std.mem.Allocator) void {
        gpa.free(self.w);
        gpa.free(self.s);
        gpa.free(self.b);
    }
};

pub fn faceOf(store: *Store, buf: []u8, prefix: []const u8, name: []const u8, routed: bool, n: usize, k: usize) !Face {
    const w = try store.hold(try std.fmt.bufPrint(buf, "{s}{s}.weight", .{ prefix, name }));
    const sc = try store.hold(try std.fmt.bufPrint(buf, "{s}{s}.scales", .{ prefix, name }));
    const b = try store.hold(try std.fmt.bufPrint(buf, "{s}{s}.biases", .{ prefix, name }));
    const k8 = k / 8;
    const kg = k / 32;
    if (routed) {
        if (!w.is(.u32, &.{ experts_n, n, k8 }) or !sc.is(.bf16, &.{ experts_n, n, kg }) or !b.is(.bf16, &.{ experts_n, n, kg })) return error.UnexpectedTensor;
    } else if (!w.is(.u32, &.{ n, k8 }) or !sc.is(.bf16, &.{ n, kg }) or !b.is(.bf16, &.{ n, kg })) return error.UnexpectedTensor;
    return .{ .w = w.bytes, .s = sc.bytes, .b = b.bytes, .n = n, .k = k, .routed = routed };
}

pub const Layer = struct {
    attn_hc: HcW,
    mlp_hc: HcW,
    proj: Q4,
    out: Q4,
    router: cuda.DeviceBuffer,
    linear: bool,
    cw: cuda.DeviceBuffer,
    a_log: cuda.DeviceBuffer,
    dt: cuda.DeviceBuffer,
    norm: cuda.DeviceBuffer,
    q_scale: cuda.DeviceBuffer,
    k_scale: cuda.DeviceBuffer,
    i_scale: cuda.DeviceBuffer,
    /// The indexer's k_layernorm (fp32), the pooled keys' RMSNorm scale.
    ik_scale: cuda.DeviceBuffer,
    key: Q4,
    val: Q4,
    nk: cuda.DeviceBuffer,
    nq: cuda.DeviceBuffer,
    nc: cuda.DeviceBuffer,
    ple_cw: cuda.DeviceBuffer,
    /// Packed gate and up for every routed expert plus the shared expert at index 512.
    experts_up: cuda.DeviceBuffer,
    experts_down: cuda.DeviceBuffer,

    pub fn free(self: *Layer, gpa: std.mem.Allocator) void {
        _ = gpa;
        self.attn_hc.free();
        self.mlp_hc.free();
        self.proj.free();
        self.out.free();
        self.router.free();
        self.cw.free();
        self.a_log.free();
        self.dt.free();
        self.norm.free();
        self.q_scale.free();
        self.k_scale.free();
        self.i_scale.free();
        self.ik_scale.free();
        self.key.free();
        self.val.free();
        self.nk.free();
        self.nq.free();
        self.nc.free();
        self.ple_cw.free();
        self.experts_up.free();
        self.experts_down.free();
    }
};

fn deadBuf(d: *cuda.Driver) cuda.DeviceBuffer {
    return .{ .d = d, .ptr = 0, .len = 0 };
}

fn blankFace() OwnedFace {
    return .{ .face = .{ .w = &.{}, .s = &.{}, .b = &.{}, .n = 0, .k = 0, .routed = false }, .w = &.{}, .s = &.{}, .b = &.{} };
}

/// Copy one expert tensor out of the checkpoint so the shard can be closed.
fn keepFace(gpa: std.mem.Allocator, store: *Store, buf: []u8, prefix: []const u8, name: []const u8, routed: bool, n: usize, k: usize) !OwnedFace {
    const view = try faceOf(store, buf, prefix, name, routed, n, k);
    const w = try gpa.dupe(u8, view.w);
    errdefer gpa.free(w);
    const s = try gpa.dupe(u8, view.s);
    errdefer gpa.free(s);
    const b = try gpa.dupe(u8, view.b);
    return .{ .face = .{ .w = w, .s = s, .b = b, .n = n, .k = k, .routed = routed }, .w = w, .s = s, .b = b };
}

fn packDevice(driver: *cuda.Driver, stream: *cuda.Stream, pack_fn: cuda.Function, words: []const u8, scales: []const u8, biases: []const u8, e: usize, n: usize, k: usize) !cuda.DeviceBuffer {
    const kg = k / 32;
    const nb = n / 32;
    var w_b = try cuda.DeviceBuffer.fromHost(driver, words);
    defer w_b.free();
    var s_b = try cuda.DeviceBuffer.fromHost(driver, scales);
    defer s_b.free();
    var b_b = try cuda.DeviceBuffer.fromHost(driver, biases);
    defer b_b.free();
    var out_b = try cuda.DeviceBuffer.alloc(driver, e * nb * kg * block_bytes);
    errdefer out_b.free();
    var a: cuda.Args = .{};
    a.add(w_b.ptr);
    a.add(s_b.ptr);
    a.add(b_b.ptr);
    a.add(out_b.ptr);
    a.add(cint(n));
    a.add(cint(k / 8));
    a.add(cint(kg));
    a.add(cint(nb));
    try cuda.launch.launch(pack_fn, .{ .grid = .{ .x = @intCast(kg), .y = @intCast(nb), .z = @intCast(e) }, .block = .{ .x = 160 } }, stream.*, &a);
    try stream.synchronize();
    return out_b;
}

/// Routed experts, then the shared expert as index `experts_n`, in the kernel's packed layout.
fn packAll(driver: *cuda.Driver, stream: *cuda.Stream, pack_fn: cuda.Function, routed: OwnedFace, shared: OwnedFace) !cuda.DeviceBuffer {
    const n = routed.face.n;
    const k = routed.face.k;
    var bulk = try packDevice(driver, stream, pack_fn, routed.w, routed.s, routed.b, experts_n, n, k);
    defer bulk.free();
    var one = try packDevice(driver, stream, pack_fn, shared.w, shared.s, shared.b, 1, n, k);
    defer one.free();
    if (bulk.len != experts_n * one.len) return error.UnexpectedTensor;
    var out = try cuda.DeviceBuffer.alloc(driver, bulk.len + one.len);
    errdefer out.free();
    try out.copyFrom(0, bulk.ptr, bulk.len, stream.handle);
    try out.copyFrom(bulk.len, one.ptr, one.len, stream.handle);
    try stream.synchronize();
    return out;
}

fn stackHalves(gpa: std.mem.Allocator, gate: []const u8, up: []const u8) ![]u8 {
    if (gate.len != up.len or gate.len % block_bytes != 0) return error.UnexpectedTensor;
    const nblk = gate.len / block_bytes;
    const out = try gpa.alloc(u8, gate.len + up.len);
    for (0..nblk) |b| {
        @memcpy(out[b * 2 * block_bytes ..][0..block_bytes], gate[b * block_bytes ..][0..block_bytes]);
        @memcpy(out[b * 2 * block_bytes + block_bytes ..][0..block_bytes], up[b * block_bytes ..][0..block_bytes]);
    }
    return out;
}

/// Gate and up blocks side by side, which is the SwiGLU weight the expert kernel reads.
fn stackUp(gpa: std.mem.Allocator, driver: *cuda.Driver, stream: *cuda.Stream, gate: cuda.DeviceBuffer, up: cuda.DeviceBuffer) !cuda.DeviceBuffer {
    const graw = try gpa.alloc(u8, gate.len);
    defer gpa.free(graw);
    const uraw = try gpa.alloc(u8, up.len);
    defer gpa.free(uraw);
    try stream.synchronize();
    try gate.download(0, graw);
    try up.download(0, uraw);
    const stacked = try stackHalves(gpa, graw, uraw);
    defer gpa.free(stacked);
    return cuda.DeviceBuffer.fromHost(driver, stacked);
}

pub fn loadLayer(gpa: std.mem.Allocator, driver: *cuda.Driver, stream: *cuda.Stream, store: *Store, pack_fn: cuda.Function, layer: usize) !Layer {
    var buf: [80]u8 = undefined;
    const base = try std.fmt.bufPrint(&buf, "language_model.model.layers.{d}", .{layer});
    return loadLayerAt(gpa, driver, stream, store, pack_fn, base, linearLayer(layer), layer == ple_layer);
}

/// The decoder layer under `base` (the MTP head's "language_model.mtp.layers.0": full attention, no n-gram branch).
pub fn loadLayerAt(gpa: std.mem.Allocator, driver: *cuda.Driver, stream: *cuda.Stream, store: *Store, pack_fn: cuda.Function, base: []const u8, linear: bool, with_ple: bool) !Layer {
    var buf: [200]u8 = undefined;
    var base_buf: [80]u8 = undefined;
    @memcpy(base_buf[0..base.len], base);
    const root = base_buf[0..base.len];
    var attn_hc = try loadHc(gpa, driver, store, try std.fmt.bufPrint(&buf, "{s}.attn_hyper_connection", .{root}), true);
    errdefer attn_hc.free();
    var mlp_hc = try loadHc(gpa, driver, store, try std.fmt.bufPrint(&buf, "{s}.mlp_hyper_connection", .{root}), true);
    errdefer mlp_hc.free();
    const mlp = try std.fmt.bufPrint(&buf, "{s}.mlp", .{root});
    var mlp_buf: [96]u8 = undefined;
    @memcpy(mlp_buf[0..mlp.len], mlp);
    const mlp_root = mlp_buf[0..mlp.len];
    const gate = try store.tensor(try std.fmt.bufPrint(&buf, "{s}.gate.weight", .{mlp_root}));
    if (!gate.is(.bf16, &.{ experts_n, dims })) return error.UnexpectedTensor;
    const gate_copy = try gpa.alloc(u8, gate.bytes.len);
    defer gpa.free(gate_copy);
    @memcpy(gate_copy, gate.bytes);
    const router = try gpa.alloc(u16, (experts_n + 1) * dims);
    defer gpa.free(router);
    @memcpy(std.mem.sliceAsBytes(router[0 .. experts_n * dims]), gate_copy);
    const sg_tensor = try store.tensor(try std.fmt.bufPrint(&buf, "{s}.shared_expert_gate.weight", .{mlp_root}));
    if (sg_tensor.dtype == .bf16) {
        if (!sg_tensor.is(.bf16, &.{ 1, dims }) or sg_tensor.bytes.len != dims * 2) return error.UnexpectedTensor;
        @memcpy(std.mem.sliceAsBytes(router[experts_n * dims ..]), sg_tensor.bytes);
    } else {
        const sg = try join3(gpa, store, &buf, mlp_root, ".shared_expert_gate");
        defer gpa.free(sg.w);
        defer gpa.free(sg.s);
        defer gpa.free(sg.b);
        try embed.dequant(sg.w, sg.s, sg.b, dims, 0, router[experts_n * dims ..][0..dims]);
    }
    var router_b = try cuda.DeviceBuffer.fromHost(driver, std.mem.sliceAsBytes(router));
    errdefer router_b.free();

    var proj_q = Q4.empty(driver);
    var out_q = Q4.empty(driver);
    errdefer proj_q.free();
    errdefer out_q.free();
    var cw_b = deadBuf(driver);
    var a_b = deadBuf(driver);
    var dt_b = deadBuf(driver);
    var nw_b = deadBuf(driver);
    var qs_b = deadBuf(driver);
    var ks_b = deadBuf(driver);
    var is_b = deadBuf(driver);
    var iks_b = deadBuf(driver);
    var key_q = Q4.empty(driver);
    var val_q = Q4.empty(driver);
    errdefer key_q.free();
    errdefer val_q.free();
    var nk_b = deadBuf(driver);
    var nq_b = deadBuf(driver);
    var nc_b = deadBuf(driver);
    var pcw_b = deadBuf(driver);
    errdefer cw_b.free();
    errdefer a_b.free();
    errdefer dt_b.free();
    errdefer nw_b.free();
    errdefer qs_b.free();
    errdefer ks_b.free();
    errdefer is_b.free();
    errdefer iks_b.free();
    errdefer nk_b.free();
    errdefer nq_b.free();
    errdefer nc_b.free();
    errdefer pcw_b.free();
    if (linear) {
        const pre = try std.fmt.bufPrint(&buf, "{s}.linear_attn", .{root});
        var pre_buf: [120]u8 = undefined;
        @memcpy(pre_buf[0..pre.len], pre);
        const lp = pre_buf[0..pre.len];
        const parts = [_][]const u8{ ".in_proj_qkv", ".in_proj_z", ".in_proj_b", ".in_proj_a" };
        proj_q = try projections.loadStitched(gpa, driver, store, lp, &parts, gdn.proj_width, dims);
        out_q = try projections.load(gpa, driver, store, try std.fmt.bufPrint(&buf, "{s}.out_proj", .{lp}), dims, gdn.value_dim);
        const conv = try store.tensor(try std.fmt.bufPrint(&buf, "{s}.conv1d.weight", .{lp}));
        if (conv.bytes.len != gdn.conv_dim * gdn.taps * 2) return error.UnexpectedTensor;
        cw_b = try cuda.DeviceBuffer.fromHost(driver, conv.bytes);
        const a_log = try asF32(gpa, try store.tensor(try std.fmt.bufPrint(&buf, "{s}.A_log", .{lp})));
        defer gpa.free(a_log);
        const dt = try asF32(gpa, try store.tensor(try std.fmt.bufPrint(&buf, "{s}.dt_bias", .{lp})));
        defer gpa.free(dt);
        if (a_log.len != gdn.nv or dt.len != gdn.nv) return error.UnexpectedTensor;
        a_b = try cuda.DeviceBuffer.fromHost(driver, std.mem.sliceAsBytes(a_log));
        dt_b = try cuda.DeviceBuffer.fromHost(driver, std.mem.sliceAsBytes(dt));
        const norm = try store.copyOf(try std.fmt.bufPrint(&buf, "{s}.norm.weight", .{lp}));
        defer gpa.free(norm);
        nw_b = try cuda.DeviceBuffer.fromHost(driver, norm);
    } else {
        const pre = try std.fmt.bufPrint(&buf, "{s}.self_attn", .{root});
        var pre_buf: [120]u8 = undefined;
        @memcpy(pre_buf[0..pre.len], pre);
        const ap = pre_buf[0..pre.len];
        const parts = [_][]const u8{ ".q_proj", ".k_proj", ".v_proj", ".indexer.index_qk_proj" };
        proj_q = try projections.loadStitched(gpa, driver, store, ap, &parts, proj_n, dims);
        out_q = try projections.load(gpa, driver, store, try std.fmt.bufPrint(&buf, "{s}.o_proj", .{ap}), dims, out_k);
        const qn = try asF32(gpa, try store.tensor(try std.fmt.bufPrint(&buf, "{s}.q_norm.weight", .{ap})));
        defer gpa.free(qn);
        const kn = try asF32(gpa, try store.tensor(try std.fmt.bufPrint(&buf, "{s}.k_norm.weight", .{ap})));
        defer gpa.free(kn);
        const iqn = try asF32(gpa, try store.tensor(try std.fmt.bufPrint(&buf, "{s}.indexer.q_layernorm.weight", .{ap})));
        defer gpa.free(iqn);
        const ikn = try asF32(gpa, try store.tensor(try std.fmt.bufPrint(&buf, "{s}.indexer.k_layernorm.weight", .{ap})));
        defer gpa.free(ikn);
        if (qn.len != head_dim or kn.len != head_dim or iqn.len != index_dim or ikn.len != index_dim) return error.UnexpectedTensor;
        qs_b = try cuda.DeviceBuffer.fromHost(driver, std.mem.sliceAsBytes(qn));
        ks_b = try cuda.DeviceBuffer.fromHost(driver, std.mem.sliceAsBytes(kn));
        is_b = try cuda.DeviceBuffer.fromHost(driver, std.mem.sliceAsBytes(iqn));
        iks_b = try cuda.DeviceBuffer.fromHost(driver, std.mem.sliceAsBytes(ikn));
    }
    if (with_ple) {
        const pre = try std.fmt.bufPrint(&buf, "{s}.ple", .{root});
        var pre_buf: [96]u8 = undefined;
        @memcpy(pre_buf[0..pre.len], pre);
        const pp = pre_buf[0..pre.len];
        key_q = try projections.load(gpa, driver, store, try std.fmt.bufPrint(&buf, "{s}.key_proj", .{pp}), wide, dims);
        val_q = try projections.load(gpa, driver, store, try std.fmt.bufPrint(&buf, "{s}.value_proj", .{pp}), dims, dims);
        const nk = try asF32(gpa, try store.tensor(try std.fmt.bufPrint(&buf, "{s}.norm_key.weight", .{pp})));
        defer gpa.free(nk);
        const nq = try asF32(gpa, try store.tensor(try std.fmt.bufPrint(&buf, "{s}.norm_query.weight", .{pp})));
        defer gpa.free(nq);
        const nc = try asF32(gpa, try store.tensor(try std.fmt.bufPrint(&buf, "{s}.norm_conv.weight", .{pp})));
        defer gpa.free(nc);
        if (nk.len != wide or nq.len != wide or nc.len != wide) return error.UnexpectedTensor;
        nk_b = try cuda.DeviceBuffer.fromHost(driver, std.mem.sliceAsBytes(nk));
        nq_b = try cuda.DeviceBuffer.fromHost(driver, std.mem.sliceAsBytes(nq));
        nc_b = try cuda.DeviceBuffer.fromHost(driver, std.mem.sliceAsBytes(nc));
        const conv = try store.tensor(try std.fmt.bufPrint(&buf, "{s}.conv1d.weight", .{pp}));
        if (conv.bytes.len != wide * 4 * 2) return error.UnexpectedTensor;
        pcw_b = try cuda.DeviceBuffer.fromHost(driver, conv.bytes);
    }

    var shared: [3]OwnedFace = .{ blankFace(), blankFace(), blankFace() };
    errdefer for (&shared) |*f| f.free(gpa);
    // a folder Living Weights taught holds a wider shared expert (its 640 neurons, then the pocket's)
    const width = (try store.hold(try std.fmt.bufPrint(&buf, "{s}.shared_expert.gate_proj.weight", .{mlp_root}))).dim(0);
    if (width != moe_w) return error.LearnedPocket;
    shared[0] = try keepFace(gpa, store, &buf, mlp_root, ".shared_expert.gate_proj", false, moe_w, dims);
    shared[1] = try keepFace(gpa, store, &buf, mlp_root, ".shared_expert.up_proj", false, moe_w, dims);
    shared[2] = try keepFace(gpa, store, &buf, mlp_root, ".shared_expert.down_proj", false, dims, moe_w);
    var routed: [3]OwnedFace = .{ blankFace(), blankFace(), blankFace() };
    errdefer for (&routed) |*f| f.free(gpa);
    routed[0] = try keepFace(gpa, store, &buf, mlp_root, ".switch_mlp.gate_proj", true, moe_w, dims);
    routed[1] = try keepFace(gpa, store, &buf, mlp_root, ".switch_mlp.up_proj", true, moe_w, dims);
    routed[2] = try keepFace(gpa, store, &buf, mlp_root, ".switch_mlp.down_proj", true, dims, moe_w);
    var gate_p = try packAll(driver, stream, pack_fn, routed[0], shared[0]);
    errdefer gate_p.free();
    var up_p = try packAll(driver, stream, pack_fn, routed[1], shared[1]);
    errdefer up_p.free();
    var down_p = try packAll(driver, stream, pack_fn, routed[2], shared[2]);
    errdefer down_p.free();
    for (&shared) |*f| f.free(gpa);
    for (&routed) |*f| f.free(gpa);
    shared = .{ blankFace(), blankFace(), blankFace() };
    routed = .{ blankFace(), blankFace(), blankFace() };
    var experts_up = try stackUp(gpa, driver, stream, gate_p, up_p);
    errdefer experts_up.free();
    gate_p.free();
    up_p.free();
    store.release();
    return .{
        .attn_hc = attn_hc,
        .mlp_hc = mlp_hc,
        .proj = proj_q,
        .out = out_q,
        .router = router_b,
        .linear = linear,
        .cw = cw_b,
        .a_log = a_b,
        .dt = dt_b,
        .norm = nw_b,
        .q_scale = qs_b,
        .k_scale = ks_b,
        .i_scale = is_b,
        .ik_scale = iks_b,
        .key = key_q,
        .val = val_q,
        .nk = nk_b,
        .nq = nq_b,
        .nc = nc_b,
        .ple_cw = pcw_b,
        .experts_up = experts_up,
        .experts_down = down_p,
    };
}
