//! Strict pinned backbone schema and borrowed resident bindings; only expert descriptor tables are owned.
const std = @import("std");
const mtl = @import("metal");
const cp = @import("core").checkpoint_metal;
const st = @import("core").safetensors;
const exl3 = @import("core").exl3_projection;
const load = @import("load.zig");

pub const Tensor = cp.Tensor;
pub const Matrix = exl3.Matrix;

/// A kept-dense repack stores small matrices as a bf16 weight beside the rotation scales.
pub const Dense = struct { weight: Tensor, suh: Tensor, svh: Tensor, k: usize, n: usize };
pub const AnyMatrix = union(enum) { trellis: Matrix, dense: Dense };
pub const Hc = struct { projection: Tensor, scale: Tensor, base: Tensor };
pub const Compressor = struct { wkv: AnyMatrix, wgate: ?AnyMatrix, norm: Tensor };
pub const Indexer = struct { wq_b: AnyMatrix, wk: ?AnyMatrix, weights_proj: Tensor, k_norm: ?Tensor };
pub const Attention = struct {
    wq_a: Matrix,
    wq_b: Matrix,
    wkv: Matrix,
    wo_b: Matrix,
    wo_a: [8]Matrix,
    /// The eight output-group descriptors of wo_a, for the one indexed decode dispatch.
    wo_a_descriptors: ?mtl.Buffer = null,
    /// Wq_a (n 1280) and wkv (n 512) as one indexed call's two-entry table. They read the same input row and the
    /// same k, so one rotate_in, one GEMM and one rotate_out carry both; each entry keeps its own per-column scales
    /// (suh/svh) and its own output width (descriptor.reserved).
    qkv_descriptors: ?mtl.Buffer = null,
    sink: Tensor,
    q_norm: Tensor,
    kv_norm: Tensor,
    compressor: ?Compressor,
    indexer: ?Indexer,
};
/// Projection order is w1 (gate), w3 (up), w2 (down), for shared and routed experts alike.
pub const Moe = struct {
    gate: Tensor,
    bias: Tensor,
    shared: [3]Matrix,
    descriptors: [3]mtl.Buffer,
    /// It holds copies of the descriptors, not of any weight, so it costs 96 bytes an expert.
    fused: ?mtl.Buffer = null,
    /// The shared expert's gate and up as one indexed dispatch: descriptor 0 is `shared[0]` (w1), descriptor 1 is
    /// `shared[1]` (w3), both read the same input row (the gather map repeats it).
    shared_fused: ?mtl.Buffer = null,
    /// The shared down projection's one-descriptor table and a zero pick map, so the shared SwiGLU's prerotate
    /// (which reads the row's su/sv) and the dense down with a prerotated input can skip the rotate_in dispatch.
    /// Descriptor 0 = `shared[2]`; the map is eight zeros.
    shared_down: ?mtl.Buffer = null,
    shared_down_ids: ?mtl.Buffer = null,
    gate_bf16: ?Tensor = null,
};

/// `t` as BF16: a host pass over F16 rows (round to nearest even), for the tensor-unit gate GEMM. Returns null for a
/// BF16 tensor (already one) and for dtypes the gate never uses.
pub fn bf16Of(device: mtl.Device, t: Tensor) !?Tensor {
    if (t.dtype == .bf16) return null;
    if (t.dtype != .f16) return null;
    const n = t.count();
    const out = try device.buffer(n * 2, mtl.ResourceOptions.shared);
    const src = t.host(u16);
    const dst: [*]u16 = @ptrCast(@alignCast(out.contents()));
    for (dst[0..n], src[0..n]) |*d, c| {
        const f: f32 = @floatCast(@as(f16, @bitCast(c)));
        const u: u32 = @bitCast(f);
        d.* = @intCast((u + 0x7fff + ((u >> 16) & 1)) >> 16);
    }
    return Tensor{ .buffer = out, .offset = 0, .bytes = n * 2, .dtype = .bf16, .shape = t.shape, .rank = t.rank };
}

pub const Engram = struct {
    /// Null when the tables were not bound (`--engram-stream-dir`): the row source reads them a window at a time, and
    /// a consumer that needs them must say so instead of reading an unbound view.
    table: ?Tensor,
    scales: ?Tensor,
    q_weight: Tensor,
    k_weight: Tensor,
    wkv: Matrix,
    /// True when `table`/`scales` are absent because the tables are streamed rather than resident.
    streamed: bool = false,
};
pub const Layer = struct {
    attn_norm: Tensor,
    ffn_norm: Tensor,
    hc_attn: Hc,
    hc_ffn: Hc,
    attn: Attention,
    moe: Moe,
    engram: ?Engram,
};
/// The pack's next-token drafter: three blocks that are backbone layers (no compressor, no indexer, 128 experts,
/// three a token), the tap projection mtp.0 maps layers 37/38/39's hidden states back to 5120, and the final norm
/// mtp.2 carries. The head is the main one; `markov_head`/`confidence_head` serve the tree and its pricing only.
pub const Dspark = struct {
    main_proj: Matrix,
    main_norm: Tensor,
    norm: Tensor,
    blocks: [3]Layer,
    /// The three blocks' `wkv` as ONE grouped indexed call's three-entry descriptor table. The three matrices share
    /// the conditioning's input rows and k, so one rotate_in, one GEMM (three 128-row groups, one expert a group)
    /// and one rotate_out carry all three; each entry keeps its own per-column scales.
    cond_wkv_descriptors: ?mtl.Buffer = null,
    /// The last stage's Markov pair (`mtp.2.markov_head.embed/head` [vocab, rank] bf16): the draft chain's
    /// rank-256 bias of the token before each row. Optional so an older pack without the chain still loads.
    markov_embed: ?Tensor = null,
    markov_head: ?Tensor = null,
};
pub const Model = struct {
    embed: Tensor,
    head: Matrix,
    norm: Tensor,
    layers: [40]Layer,
    /// Whether an attached int8 sidecar was packed pre-rotated (its call is one byte body into the caller's bf16
    /// output). False = the unrotated pack, which keeps rotate_in/rotate_out. Off without the sidecar.
    int8_direct: bool = false,
    dspark: ?Dspark = null,

    /// Checkpoint buffers must outlive this borrowed model and every submitted command using it.
    pub fn deinit(self: *Model) void {
        for (&self.layers) |*layer| for (layer.moe.descriptors) |buffer| buffer.deinit();
        for (&self.layers) |*layer| if (layer.moe.fused) |buffer| buffer.deinit();
        for (&self.layers) |*layer| if (layer.moe.shared_fused) |buffer| buffer.deinit();
        for (&self.layers) |*layer| if (layer.moe.shared_down) |buffer| buffer.deinit();
        for (&self.layers) |*layer| if (layer.moe.shared_down_ids) |buffer| buffer.deinit();
        for (&self.layers) |*layer| if (layer.attn.wo_a_descriptors) |buffer| buffer.deinit();
        for (&self.layers) |*layer| if (layer.attn.qkv_descriptors) |buffer| buffer.deinit();
        for (&self.layers) |*layer| {
            if (layer.attn.wq_b.int8) |v| v.deinit();
            if (layer.attn.wo_b.int8) |v| v.deinit();
            for (&layer.attn.wo_a) |*m| if (m.int8) |v| v.deinit();
        }
        if (self.dspark) |*d| for (&d.blocks) |*layer| for (layer.moe.descriptors) |buffer| buffer.deinit();
        if (self.dspark) |*d| for (&d.blocks) |*layer| if (layer.moe.fused) |buffer| buffer.deinit();
        if (self.dspark) |*d| for (&d.blocks) |*layer| if (layer.moe.shared_fused) |buffer| buffer.deinit();
        if (self.dspark) |*d| for (&d.blocks) |*layer| if (layer.moe.shared_down) |buffer| buffer.deinit();
        if (self.dspark) |*d| for (&d.blocks) |*layer| if (layer.moe.shared_down_ids) |buffer| buffer.deinit();
        if (self.dspark) |*d| for (&d.blocks) |*layer| if (layer.attn.wo_a_descriptors) |buffer| buffer.deinit();
        if (self.dspark) |*d| if (d.cond_wkv_descriptors) |buffer| buffer.deinit();
        self.* = undefined;
    }
    pub fn registerResidency(self: *const Model, set: mtl.ResidencySet) void {
        for (&self.layers) |*layer| for (layer.moe.descriptors) |buffer| set.add(buffer);
        for (&self.layers) |*layer| if (layer.moe.fused) |buffer| set.add(buffer);
        for (&self.layers) |*layer| if (layer.moe.shared_fused) |buffer| set.add(buffer);
        for (&self.layers) |*layer| if (layer.moe.shared_down) |buffer| set.add(buffer);
        for (&self.layers) |*layer| if (layer.moe.shared_down_ids) |buffer| set.add(buffer);
        for (&self.layers) |*layer| if (layer.attn.wo_a_descriptors) |buffer| set.add(buffer);
        for (&self.layers) |*layer| if (layer.attn.qkv_descriptors) |buffer| set.add(buffer);
        for (&self.layers) |*layer| {
            if (layer.attn.wq_b.int8) |v| {
                set.add(v.weight.buffer);
                set.add(v.scales.buffer);
            }
            if (layer.attn.wo_b.int8) |v| {
                set.add(v.weight.buffer);
                set.add(v.scales.buffer);
            }
            for (&layer.attn.wo_a) |*m| if (m.int8) |v| {
                set.add(v.weight.buffer);
                set.add(v.scales.buffer);
            };
        }
        if (self.dspark) |*d| for (&d.blocks) |*layer| for (layer.moe.descriptors) |buffer| set.add(buffer);
        if (self.dspark) |*d| for (&d.blocks) |*layer| if (layer.moe.fused) |buffer| set.add(buffer);
        if (self.dspark) |*d| for (&d.blocks) |*layer| if (layer.moe.shared_fused) |buffer| set.add(buffer);
        if (self.dspark) |*d| for (&d.blocks) |*layer| if (layer.moe.shared_down) |buffer| set.add(buffer);
        if (self.dspark) |*d| for (&d.blocks) |*layer| if (layer.moe.shared_down_ids) |buffer| set.add(buffer);
        if (self.dspark) |*d| for (&d.blocks) |*layer| if (layer.attn.wo_a_descriptors) |buffer| set.add(buffer);
        if (self.dspark) |*d| if (d.cond_wkv_descriptors) |buffer| set.add(buffer);
    }
    pub fn residentBytes(_: *const Model) usize {
        return sizeofDescriptors();
    }
};

/// Admission adds these bytes to load.Plan.bytes; no dequantized weights or duplicate projections are allocated.
pub fn sizeofDescriptors() usize {
    return 40 * 3 * 384 * @sizeOf(exl3.ExpertDescriptor) + 3 * 3 * 128 * @sizeOf(exl3.ExpertDescriptor) + 40 * 8 * @sizeOf(exl3.ExpertDescriptor) +
        40 * 2 * 384 * @sizeOf(exl3.ExpertDescriptor) + 3 * 2 * 128 * @sizeOf(exl3.ExpertDescriptor) +
        40 * @sizeOf(exl3.ExpertDescriptor) + 3 * @sizeOf(exl3.ExpertDescriptor) +
        40 * 2 * @sizeOf(exl3.ExpertDescriptor) + 3 * @sizeOf(exl3.ExpertDescriptor);
}

/// One table the fused gate+up dispatch indexes: the gate experts then the up experts, descriptor copies only,
/// so the two projections are one indexed call and each output row still runs the shipped per-row arithmetic.
fn fusedTable(device: mtl.Device, gate: mtl.Buffer, up: mtl.Buffer, count: usize, shared_gate: exl3.Matrix, shared_up: exl3.Matrix) !mtl.Buffer {
    // The table carries the two shared-expert matrices after the 2E routed entries, so the SAME indexed dispatch
    // can carry the shared gate/up rows (the comparator's moe_up form).
    const table = try device.buffer((2 * count + 2) * @sizeOf(exl3.ExpertDescriptor), mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    const dst: [*]exl3.ExpertDescriptor = @ptrCast(@alignCast(table.contents()));
    const gate_src: [*]const exl3.ExpertDescriptor = @ptrCast(@alignCast(gate.contents()));
    const up_src: [*]const exl3.ExpertDescriptor = @ptrCast(@alignCast(up.contents()));
    @memcpy(dst[0..count], gate_src[0..count]);
    @memcpy(dst[count .. 2 * count], up_src[0..count]);
    dst[2 * count] = try exl3.descriptor(shared_gate);
    dst[2 * count + 1] = try exl3.descriptor(shared_up);
    return table;
}

/// The attention's wq_a and wkv as one indexed call's two-entry table. Descriptor copies only; the entries' own n
/// rides descriptor.reserved, so the GEMM and the epilogue bound each row by its own width.
fn qkvTable(device: mtl.Device, a: exl3.Matrix, b: exl3.Matrix) !mtl.Buffer {
    const table = try device.buffer(2 * @sizeOf(exl3.ExpertDescriptor), mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    errdefer table.deinit();
    const dst: [*]exl3.ExpertDescriptor = @ptrCast(@alignCast(table.contents()));
    dst[0] = try exl3.descriptor(a);
    dst[1] = try exl3.descriptor(b);
    return table;
}

/// The drafter's three blocks and the two tensors the tap path needs. Written separately from the backbone's loop
/// because the differences are the point: no compressor, no indexer, 128 experts instead of 384, and a gate with no
/// bias_vl. `binding` false is the header-only validation pass.
fn populateDspark(comptime binding: bool, source: anytype, model: ?*Model) !void {
    if (!binding) {
        if (!hasDspark(source)) return;
    } else {
        model.?.dspark = @as(Dspark, undefined);
    }
    var name: [160]u8 = undefined;
    var prefix: [128]u8 = undefined;
    for (0..3) |block| {
        const out: ?*Layer = if (binding) &model.?.dspark.?.blocks[block] else null;
        try plain(binding, source, try named(&name, "mtp.{d}.attn_norm.weight", .{block}), .bf16, &.{5120}, if (binding) &out.?.attn_norm else null);
        try plain(binding, source, try named(&name, "mtp.{d}.ffn_norm.weight", .{block}), .bf16, &.{5120}, if (binding) &out.?.ffn_norm else null);
        inline for (.{ "attn", "ffn" }) |kind| {
            const hc: ?*Hc = if (binding) (if (std.mem.eql(u8, kind, "attn")) &out.?.hc_attn else &out.?.hc_ffn) else null;
            try plain(binding, source, try named(&name, "mtp.{d}.hc_{s}_fn", .{ block, kind }), .f32, &.{ 24, 20480 }, if (binding) &hc.?.projection else null);
            try plain(binding, source, try named(&name, "mtp.{d}.hc_{s}_scale", .{ block, kind }), .f32, &.{3}, if (binding) &hc.?.scale else null);
            try plain(binding, source, try named(&name, "mtp.{d}.hc_{s}_base", .{ block, kind }), .f32, &.{24}, if (binding) &hc.?.base else null);
        }
        const attn: ?*Attention = if (binding) &out.?.attn else null;
        inline for (.{ "wq_a", "wq_b", "wkv", "wo_b" }, .{ 5120, 1280, 5120, 8192 }, .{ 1280, 32768, 512, 5120 }) |part, k, n| {
            try matrix(binding, source, try named(&prefix, "mtp.{d}.attn.{s}", .{ block, part }), k, n, if (binding) &@field(attn.?, part) else null);
        }
        for (0..8) |slice| try matrix(binding, source, try named(&prefix, "mtp.{d}.attn.wo_a.slice.{d}", .{ block, slice }), 4096, 1024, if (binding) &attn.?.wo_a[slice] else null);
        try plain(binding, source, try named(&name, "mtp.{d}.attn.attn_sink", .{block}), .f32, &.{64}, if (binding) &attn.?.sink else null);
        try plain(binding, source, try named(&name, "mtp.{d}.attn.q_norm.weight", .{block}), .bf16, &.{1280}, if (binding) &attn.?.q_norm else null);
        try plain(binding, source, try named(&name, "mtp.{d}.attn.kv_norm.weight", .{block}), .bf16, &.{512}, if (binding) &attn.?.kv_norm else null);
        if (binding) {
            attn.?.compressor = null;
            attn.?.indexer = null;
            out.?.engram = null;
        }
        const moe: ?*Moe = if (binding) &out.?.moe else null;
        {
            const tensor = try source.get(try named(&name, "mtp.{d}.ffn.gate.weight", .{block}));
            if ((tensor.dtype != .f16 and tensor.dtype != .bf16) or tensor.rank != 2 or
                tensor.shape[0] != 128 or tensor.shape[1] != 5120) return error.UnsupportedTensor;
            if (binding) moe.?.gate = tensor;
        }
        {
            const tensor = try source.get(try named(&name, "mtp.{d}.ffn.gate.bias", .{block}));
            if ((tensor.dtype != .f16 and tensor.dtype != .bf16 and tensor.dtype != .f32) or tensor.rank != 1 or tensor.shape[0] != 128)
                return error.UnsupportedTensor;
            if (binding) moe.?.bias = tensor;
        }
        inline for (.{ "w1", "w3", "w2" }, 0..) |part, projection| {
            const k: usize = if (projection == 2) 2304 else 5120;
            const n: usize = if (projection == 2) 5120 else 2304;
            try matrix(binding, source, try named(&prefix, "mtp.{d}.ffn.shared_experts.{s}", .{ block, part }), k, n, if (binding) &moe.?.shared[projection] else null);
            if (!binding) for (0..128) |expert| try matrix(false, source, try named(&prefix, "mtp.{d}.ffn.experts.{d}.{s}", .{ block, expert, part }), k, n, null);
        }
    }
    try matrix(binding, source, "mtp.0.main_proj", 15360, 5120, if (binding) &model.?.dspark.?.main_proj else null);
    try plain(binding, source, "mtp.0.main_norm.weight", .bf16, &.{5120}, if (binding) &model.?.dspark.?.main_norm else null);
    try plain(binding, source, "mtp.2.norm.weight", .bf16, &.{5120}, if (binding) &model.?.dspark.?.norm else null);
    if (binding) {
        var emb: Tensor = undefined;
        var hd: Tensor = undefined;
        try plain(true, source, "mtp.2.markov_head.embed.weight", .bf16, &.{ 129280, 256 }, &emb);
        try plain(true, source, "mtp.2.markov_head.head.weight", .f16, &.{ 129280, 256 }, &hd);
        model.?.dspark.?.markov_embed = emb;
        model.?.dspark.?.markov_head = hd;
    } else {
        try plain(false, source, "mtp.2.markov_head.embed.weight", .bf16, &.{ 129280, 256 }, null);
        try plain(false, source, "mtp.2.markov_head.head.weight", .f16, &.{ 129280, 256 }, null);
    }
}

/// The drafter is optional: a pack without it must still load.
fn hasDspark(source: anytype) bool {
    for ([_][]const u8{ "mtp.0.main_proj.trellis", "mtp.2.norm.weight" }) |n| {
        _ = source.get(n) catch return false;
    }
    return true;
}

/// Every executed tensor is checked using headers only, before checkpoint materialization or model allocation.
pub fn validate(plan: *const load.Plan, stream: bool) !void {
    try populate(false, plan, null, stream);
    try populateDspark(false, plan, null);
}

/// Bind resident tensors and marker payloads; route descriptors replace redundant host matrix metadata.
pub fn bind(device: mtl.Device, checkpoint: *const cp.Checkpoint, stream: bool) !Model {
    var model: Model = undefined;
    try populate(true, checkpoint, &model, stream);
    var owned: usize = 0;
    var owned_groups: usize = 0;
    var owned_qkv: usize = 0;
    errdefer for (0..owned) |i| model.layers[i / 3].moe.descriptors[i % 3].deinit();
    errdefer for (0..owned_groups) |i| model.layers[i].attn.wo_a_descriptors.?.deinit();
    errdefer for (0..owned_qkv) |i| model.layers[i].attn.qkv_descriptors.?.deinit();
    errdefer for (&model.layers) |*layer| if (layer.moe.fused) |buffer| buffer.deinit();
    errdefer if (model.dspark) |*d| for (&d.blocks) |*layer| for (layer.moe.descriptors) |buffer| buffer.deinit();
    errdefer if (model.dspark) |*d| for (&d.blocks) |*layer| if (layer.moe.fused) |buffer| buffer.deinit();
    errdefer if (model.dspark) |*d| for (&d.blocks) |*layer| if (layer.moe.shared_fused) |buffer| buffer.deinit();
    try populateDspark(true, checkpoint, &model);
    var dspark_prefix: [128]u8 = undefined;
    if (model.dspark) |*d| for (&d.blocks, 0..) |*layer, block| {
        for (0..3) |projection| {
            const bytes = 128 * @sizeOf(exl3.ExpertDescriptor);
            const buffer = try device.buffer(bytes, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
            layer.moe.descriptors[projection] = buffer;
            const out: [*]exl3.ExpertDescriptor = @ptrCast(@alignCast(buffer.contents()));
            for (0..128) |expert| {
                var m: Matrix = undefined;
                const k: usize = if (projection == 2) 2304 else 5120;
                const n: usize = if (projection == 2) 5120 else 2304;
                const parts = [_][]const u8{ "w1", "w3", "w2" };
                const part = parts[projection];
                try matrix(true, checkpoint, try named(&dspark_prefix, "mtp.{d}.ffn.experts.{d}.{s}", .{ block, expert, part }), k, n, &m);
                out[expert] = try exl3.descriptor(m);
            }
        }
        layer.moe.fused = try fusedTable(device, layer.moe.descriptors[0], layer.moe.descriptors[1], 128, layer.moe.shared[0], layer.moe.shared[1]);
        // The drafter's shared expert takes the same two-entry gate/up table (its own matrices).
        {
            const table = try device.buffer(2 * @sizeOf(exl3.ExpertDescriptor), mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
            errdefer table.deinit();
            layer.moe.shared_fused = table;
            const out: [*]exl3.ExpertDescriptor = @ptrCast(@alignCast(table.contents()));
            out[0] = try exl3.descriptor(layer.moe.shared[0]);
            out[1] = try exl3.descriptor(layer.moe.shared[1]);
        }
        // The shared down's descriptor (one entry) and its zero pick map.
        {
            const table = try device.buffer(@sizeOf(exl3.ExpertDescriptor), mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
            errdefer table.deinit();
            layer.moe.shared_down = table;
            const out: [*]exl3.ExpertDescriptor = @ptrCast(@alignCast(table.contents()));
            out[0] = try exl3.descriptor(layer.moe.shared[2]);
            const ids = try device.buffer(8 * 4, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
            errdefer ids.deinit();
            @memset(ids.contents()[0 .. 8 * 4], 0);
            layer.moe.shared_down_ids = ids;
        }
        // The shared down's descriptor (one entry) and its zero pick map.
        {
            const table = try device.buffer(@sizeOf(exl3.ExpertDescriptor), mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
            errdefer table.deinit();
            layer.moe.shared_down = table;
            const out: [*]exl3.ExpertDescriptor = @ptrCast(@alignCast(table.contents()));
            out[0] = try exl3.descriptor(layer.moe.shared[2]);
            const ids = try device.buffer(8 * 4, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
            errdefer ids.deinit();
            @memset(ids.contents()[0 .. 8 * 4], 0);
            layer.moe.shared_down_ids = ids;
        }
        // THE DRAFTER'S EIGHT wo_a OUTPUT GROUPS AS ONE INDEXED DECODE PROJECTION'S DESCRIPTOR TABLE -- the trunk's
        // own fused form. The drafter's eight slices are the same shape as the trunk's (k 4096, n 1024), so the
        // table is the same table.
        {
            const groups = try device.buffer(8 * @sizeOf(exl3.ExpertDescriptor), mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
            errdefer groups.deinit();
            layer.attn.wo_a_descriptors = groups;
            const table: [*]exl3.ExpertDescriptor = @ptrCast(@alignCast(groups.contents()));
            for (0..8) |slice| table[slice] = try exl3.descriptor(layer.attn.wo_a[slice]);
        }
    };
    // The drafter's three per-block `wkv` in one grouped indexed dispatch's three-entry table.
    if (model.dspark) |*d| {
        const table = try device.buffer(3 * @sizeOf(exl3.ExpertDescriptor), mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        errdefer table.deinit();
        d.cond_wkv_descriptors = table;
        const out: [*]exl3.ExpertDescriptor = @ptrCast(@alignCast(table.contents()));
        for (&d.blocks, 0..) |*layer, block| out[block] = try exl3.descriptor(layer.attn.wkv);
    }
    var prefix: [128]u8 = undefined;
    for (&model.layers, 0..) |*layer, layer_id| {
        inline for (.{ "w1", "w3", "w2" }, 0..) |part, projection| {
            const bytes = 384 * @sizeOf(exl3.ExpertDescriptor);
            const buffer = try device.buffer(bytes, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
            layer.moe.descriptors[projection] = buffer;
            owned += 1;
            const out: [*]exl3.ExpertDescriptor = @ptrCast(@alignCast(buffer.contents()));
            for (0..384) |expert| {
                var m: Matrix = undefined;
                const k: usize = if (projection == 2) 2304 else 5120;
                const n: usize = if (projection == 2) 5120 else 2304;
                try matrix(true, checkpoint, try named(&prefix, "layers.{d}.ffn.experts.{d}.{s}", .{ layer_id, expert, part }), k, n, &m);
                out[expert] = try exl3.descriptor(m);
            }
        }
        layer.moe.fused = try fusedTable(device, layer.moe.descriptors[0], layer.moe.descriptors[1], 384, layer.moe.shared[0], layer.moe.shared[1]);
        // The shared expert's gate (w1) and up (w3) as one indexed dispatch's two-entry table.
        {
            const table = try device.buffer(2 * @sizeOf(exl3.ExpertDescriptor), mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
            errdefer table.deinit();
            layer.moe.shared_fused = table;
            const out: [*]exl3.ExpertDescriptor = @ptrCast(@alignCast(table.contents()));
            out[0] = try exl3.descriptor(layer.moe.shared[0]);
            out[1] = try exl3.descriptor(layer.moe.shared[1]);
        }
        // The eight wo_a output groups as one indexed decode projection's descriptor table.
        {
            const groups = try device.buffer(8 * @sizeOf(exl3.ExpertDescriptor), mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
            errdefer groups.deinit();
            layer.attn.wo_a_descriptors = groups;
            owned_groups += 1;
            const table: [*]exl3.ExpertDescriptor = @ptrCast(@alignCast(groups.contents()));
            for (0..8) |slice| {
                var m: Matrix = undefined;
                try matrix(true, checkpoint, try named(&prefix, "layers.{d}.attn.wo_a.slice.{d}", .{ layer_id, slice }), 4096, 1024, &m);
                table[slice] = try exl3.descriptor(m);
            }
        }
        // The attention's wq_a + wkv in one indexed call's two-entry table.
        layer.attn.qkv_descriptors = try qkvTable(device, layer.attn.wq_a, layer.attn.wkv);
        owned_qkv += 1;
    }
    return model;
}

/// A borrowed CPU/GPU matrix view for checkpoint qualification, using the same strict storage contract.
pub fn lookup(checkpoint: *const cp.Checkpoint, prefix: []const u8, k: usize, n: usize) !Matrix {
    var out: Matrix = undefined;
    try matrix(true, checkpoint, prefix, k, n, &out);
    return out;
}

fn named(buffer: []u8, comptime fmt: []const u8, args: anytype) ![]const u8 {
    return std.fmt.bufPrint(buffer, fmt, args);
}

/// A plain tensor the encoders decode as f32, f16 or bf16; the shape must match exactly.
fn plainEither(comptime binding: bool, source: anytype, name: []const u8, shape: []const usize, destination: ?*Tensor) !void {
    const tensor = try source.get(name);
    if ((tensor.dtype != .f32 and tensor.dtype != .f16 and tensor.dtype != .bf16) or tensor.rank != shape.len or
        !std.mem.eql(usize, tensor.shape[0..tensor.rank], shape))
    {
        std.log.err("unsupported tensor {s}: {s} {any}", .{ name, @tagName(tensor.dtype), tensor.shape[0..tensor.rank] });
        return error.UnsupportedTensor;
    }
    if (binding) {
        var count: usize = 1;
        for (shape) |dim| count = try std.math.mul(usize, count, dim);
        if (tensor.bytes != try std.math.mul(usize, count, tensor.dtype.size())) return error.UnsupportedTensor;
        if (tensor.offset % tensor.dtype.size() != 0 or tensor.offset > tensor.buffer.length() or
            tensor.bytes > tensor.buffer.length() - tensor.offset) return error.InvalidTensorView;
        destination.?.* = tensor;
    } else if (tensor.begin % 4 != 0) return error.InvalidTensorView;
}

fn plain(comptime binding: bool, source: anytype, name: []const u8, dtype: st.DType, shape: []const usize, destination: ?*Tensor) !void {
    const tensor = try source.get(name);
    if (tensor.dtype != dtype or tensor.rank != shape.len or
        !std.mem.eql(usize, tensor.shape[0..tensor.rank], shape))
    {
        std.log.err("unsupported backbone tensor {s}: {s} {any}", .{ name, @tagName(tensor.dtype), tensor.shape[0..tensor.rank] });
        return error.UnsupportedTensor;
    }
    const offset = if (binding) tensor.offset else tensor.begin;
    if (offset % dtype.size() != 0) return error.InvalidTensorView;
    if (binding) {
        var count: usize = 1;
        for (shape) |dim| count = try std.math.mul(usize, count, dim);
        if (tensor.bytes != try std.math.mul(usize, count, dtype.size())) return error.UnsupportedTensor;
        if (tensor.offset % dtype.size() != 0 or tensor.offset > tensor.buffer.length() or
            tensor.bytes > tensor.buffer.length() - tensor.offset) return error.InvalidTensorView;
        destination.?.* = tensor;
    }
}

fn has(source: anytype, name: []const u8) bool {
    return source.has(name);
}

const Scale = struct { name: []const u8, format: exl3.ScaleFormat };
fn scale(comptime binding: bool, source: anytype, prefix: []const u8, size: usize, comptime fp: []const u8, comptime signs: []const u8, storage: []u8) !Scale {
    var alternative: [160]u8 = undefined;
    const f = try named(storage, "{s}.{s}", .{ prefix, fp });
    const s = try named(&alternative, "{s}.{s}", .{ prefix, signs });
    const have_fp = has(source, f);
    if (have_fp == has(source, s)) return error.MissingOrAmbiguousExl3Scale;
    const name = if (have_fp) f else try named(storage, "{s}.{s}", .{ prefix, signs });
    const t = try source.get(name);
    if (t.rank != 1 or t.shape[0] != (if (have_fp) size else size / 16) or
        t.dtype != (if (have_fp) st.DType.f16 else st.DType.i16)) return error.UnsupportedExl3Scale;
    const offset = if (binding) t.offset else t.begin;
    if (offset % 2 != 0) return error.InvalidTensorView;
    if (binding) {
        const bytes = (if (have_fp) size else size / 16) * 2;
        if (t.bytes != bytes or t.offset % 2 != 0 or t.offset > t.buffer.length() or
            bytes > t.buffer.length() - t.offset) return error.InvalidTensorView;
    }
    return .{ .name = name, .format = if (have_fp) .f16 else .packed_signs };
}

/// Compressor/indexer matrices ride either storage: packs quantize; rank-split repacks stay dense with rotation scales.
fn matrixAny(comptime binding: bool, source: anytype, prefix: []const u8, k: usize, n: usize, destination: ?*AnyMatrix) !void {
    var name_buf: [160]u8 = undefined;
    if (has(source, try named(&name_buf, "{s}.weight", .{prefix}))) {
        const weight_name = try named(&name_buf, "{s}.weight", .{prefix});
        const w = try source.get(weight_name);
        if (w.dtype != .bf16 or w.rank != 2 or w.shape[0] != n or w.shape[1] != k) {
            std.log.err("unsupported dense tensor {s}: {s} {any}", .{ weight_name, @tagName(w.dtype), w.shape[0..w.rank] });
            return error.UnsupportedTensor;
        }
        const su = try source.get(try named(&name_buf, "{s}.suh", .{prefix}));
        const sv = try source.get(try named(&name_buf, "{s}.svh", .{prefix}));
        if (su.dtype != .f16 or su.rank != 1 or su.shape[0] != k or
            sv.dtype != .f16 or sv.rank != 1 or sv.shape[0] != n) return error.UnsupportedExl3Scale;
        if (binding) {
            if (w.offset % 2 != 0) return error.InvalidTensorView;
            destination.?.* = .{ .dense = .{ .weight = w, .suh = su, .svh = sv, .k = k, .n = n } };
        }
        return;
    }
    var m: Matrix = undefined;
    try matrix(binding, source, prefix, k, n, if (binding) &m else null);
    if (binding) destination.?.* = .{ .trellis = m };
}

fn matrix(comptime binding: bool, source: anytype, prefix: []const u8, k: usize, n: usize, destination: ?*Matrix) !void {
    var name: [160]u8 = undefined;
    const trellis = try source.get(try named(&name, "{s}.trellis", .{prefix}));
    if (trellis.dtype != .i16 or trellis.rank != 3 or trellis.shape[0] != k / 16 or
        trellis.shape[1] != n / 16 or trellis.shape[2] % 8 != 0) return error.UnsupportedExl3Shape;
    if ((if (binding) trellis.offset else trellis.begin) % 2 != 0) return error.InvalidTensorView;
    const halves = trellis.shape[2] / 8;
    if (halves < 2 or halves > 16 or (halves % 2 != 0 and halves > 7)) return error.UnsupportedExl3Width;
    // Pinned checkpoint uses mul1 throughout; missing markers cannot silently select the legacy codebook.
    const marker = try source.get(try named(&name, "{s}.mul1", .{prefix}));
    if (marker.dtype != .i32 or marker.rank != 0) return error.UnsupportedExl3Marker;
    if ((if (binding) marker.offset else marker.begin) % 4 != 0) return error.InvalidTensorView;
    if (has(source, try named(&name, "{s}.mcg", .{prefix})) or
        has(source, try named(&name, "{s}.3inst", .{prefix})) or
        has(source, try named(&name, "{s}.bias", .{prefix})))
    {
        std.log.err("unexpected exl3 parts beside {s}", .{prefix});
        return error.UnsupportedExl3Parts;
    }
    var su_name: [160]u8 = undefined;
    var sv_name: [160]u8 = undefined;
    const su = try scale(binding, source, prefix, k, "suh", "su", &su_name);
    const sv = try scale(binding, source, prefix, n, "svh", "sv", &sv_name);
    if (binding) {
        const expected = k / 16 * (n / 16) * halves * 16;
        if (trellis.bytes != expected or trellis.offset % 2 != 0 or
            trellis.offset > trellis.buffer.length() or expected > trellis.buffer.length() - trellis.offset)
            return error.InvalidTensorView;
        if (marker.bytes != 4 or marker.offset % 4 != 0 or marker.offset > marker.buffer.length() or
            4 > marker.buffer.length() - marker.offset) return error.InvalidTensorView;
        const marker_bytes = marker.buffer.contents()[marker.offset..][0..4];
        if (std.mem.readInt(u32, marker_bytes, .little) != exl3.format.Mul1Mul) return error.UnsupportedExl3Marker;
        const suh = try source.get(su.name);
        const svh = try source.get(sv.name);
        destination.?.* = .{
            .trellis = .{ .buffer = trellis.buffer, .offset = trellis.offset },
            .suh = .{ .buffer = suh.buffer, .offset = suh.offset },
            .svh = .{ .buffer = svh.buffer, .offset = svh.offset },
            .k = k,
            .n = n,
            .width = .{ .halves = @intCast(halves) },
            .codebook = .mul1,
            .suh_format = su.format,
            .svh_format = sv.format,
        };
        try destination.?.validate();
    }
}

/// Bind a prompt-row int8 sidecar. The files are the repacker's own per-projection output (`layers.<L>.<proj>.fp8`,
/// `.scale`); one GPU buffer a tensor, resident, and the trellis matrices are untouched so decode keeps its path.
/// Off unless the caller sets TF_DSV41_I8_PREFILL.
pub fn attachInt8(gpa: std.mem.Allocator, io: std.Io, device: mtl.Device, model: *Model, dir: []const u8) !void {
    const direct = std.c.getenv("TF_DSV41_I8_ROT") != null;
    // An instrument: attach one projection at a time to localise a numerics fault (TF_DSV41_I8_ONLY=wo_a).
    const only: ?[]const u8 = if (std.c.getenv("TF_DSV41_I8_ONLY")) |v| std.mem.span(v) else null;
    for (&model.layers, 0..) |*layer, l| {
        try attachInt8One(gpa, io, device, &layer.attn.wq_b, dir, l, "attn.wq_b", direct, only);
        try attachInt8One(gpa, io, device, &layer.attn.wo_b, dir, l, "attn.wo_b", direct, only);
        for (&layer.attn.wo_a, 0..) |*m, g| {
            var name_buf: [64]u8 = undefined;
            const name = try std.fmt.bufPrint(&name_buf, "attn.wo_a.slice.{d}", .{g});
            try attachInt8One(gpa, io, device, m, dir, l, name, direct, only);
        }
    }
    model.int8_direct = direct;
    std.log.info("exl3: int8 sidecar attached from {s}: {d} layers, {d} x {d} + {d} x {d} + 8 x {d} x {d}, direct={}", .{
        dir,
        model.layers.len,
        model.layers[0].attn.wq_b.k,
        model.layers[0].attn.wq_b.n,
        model.layers[0].attn.wo_b.k,
        model.layers[0].attn.wo_b.n,
        model.layers[0].attn.wo_a[0].k,
        model.layers[0].attn.wo_a[0].n,
        direct,
    });
}

fn attachInt8One(gpa: std.mem.Allocator, io: std.Io, device: mtl.Device, m: *Matrix, dir: []const u8, layer: usize, projection: []const u8, direct: bool, only: ?[]const u8) !void {
    if (m.int8 != null) return;
    if (only) |o| if (std.mem.indexOf(u8, o, projection) == null) return;
    const weight_bytes = try std.math.mul(usize, m.k, m.n);
    var path: [512]u8 = undefined;
    const wpath = try std.fmt.bufPrint(&path, "{s}/layers.{d}.{s}.fp8", .{ dir, layer, projection });
    const wb = try std.Io.Dir.cwd().readFileAlloc(io, wpath, gpa, .limited(weight_bytes + 16));
    defer gpa.free(wb);
    if (wb.len != weight_bytes) return error.BadInt8Sidecar;
    var spath_buf: [512]u8 = undefined;
    const spath = try std.fmt.bufPrint(&spath_buf, "{s}/layers.{d}.{s}.scale", .{ dir, layer, projection });
    const scale_bytes = try std.math.mul(usize, m.n, 4);
    const sb = try std.Io.Dir.cwd().readFileAlloc(io, spath, gpa, .limited(scale_bytes + 16));
    defer gpa.free(sb);
    if (sb.len != scale_bytes) return error.BadInt8Sidecar;
    const weight = try device.buffer(weight_bytes, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    errdefer weight.deinit();
    @memcpy(weight.contents()[0..weight_bytes], wb);
    const scales = try device.buffer(scale_bytes, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    errdefer scales.deinit();
    @memcpy(scales.contents()[0..scale_bytes], sb);
    m.int8 = .{ .weight = .{ .buffer = weight, .offset = 0 }, .scales = .{ .buffer = scales, .offset = 0 }, .k = m.k, .n = m.n };
    m.int8_direct = direct;
}

fn populate(comptime binding: bool, source: anytype, model: ?*Model, stream: bool) !void {
    try plain(binding, source, "embed.weight", .bf16, &.{ 129280, 5120 }, if (binding) &model.?.embed else null);
    try plain(binding, source, "norm.weight", .bf16, &.{5120}, if (binding) &model.?.norm else null);
    try matrix(binding, source, "head", 5120, 129280, if (binding) &model.?.head else null);
    var name: [160]u8 = undefined;
    var prefix: [128]u8 = undefined;
    for (0..40) |layer| {
        const out: ?*Layer = if (binding) &model.?.layers[layer] else null;
        try plain(binding, source, try named(&name, "layers.{d}.attn_norm.weight", .{layer}), .bf16, &.{5120}, if (binding) &out.?.attn_norm else null);
        try plain(binding, source, try named(&name, "layers.{d}.ffn_norm.weight", .{layer}), .bf16, &.{5120}, if (binding) &out.?.ffn_norm else null);
        inline for (.{ "attn", "ffn" }) |kind| {
            const hc: ?*Hc = if (binding) (if (std.mem.eql(u8, kind, "attn")) &out.?.hc_attn else &out.?.hc_ffn) else null;
            try plain(binding, source, try named(&name, "layers.{d}.hc_{s}_fn", .{ layer, kind }), .f32, &.{ 24, 20480 }, if (binding) &hc.?.projection else null);
            try plain(binding, source, try named(&name, "layers.{d}.hc_{s}_scale", .{ layer, kind }), .f32, &.{3}, if (binding) &hc.?.scale else null);
            try plain(binding, source, try named(&name, "layers.{d}.hc_{s}_base", .{ layer, kind }), .f32, &.{24}, if (binding) &hc.?.base else null);
        }
        const attn: ?*Attention = if (binding) &out.?.attn else null;
        inline for (.{ "wq_a", "wq_b", "wkv", "wo_b" }, .{ 5120, 1280, 5120, 8192 }, .{ 1280, 32768, 512, 5120 }) |part, k, n| {
            try matrix(binding, source, try named(&prefix, "layers.{d}.attn.{s}", .{ layer, part }), k, n, if (binding) &@field(attn.?, part) else null);
        }
        for (0..8) |slice| try matrix(binding, source, try named(&prefix, "layers.{d}.attn.wo_a.slice.{d}", .{ layer, slice }), 4096, 1024, if (binding) &attn.?.wo_a[slice] else null);
        try plain(binding, source, try named(&name, "layers.{d}.attn.attn_sink", .{layer}), .f32, &.{64}, if (binding) &attn.?.sink else null);
        try plain(binding, source, try named(&name, "layers.{d}.attn.q_norm.weight", .{layer}), .bf16, &.{1280}, if (binding) &attn.?.q_norm else null);
        try plain(binding, source, try named(&name, "layers.{d}.attn.kv_norm.weight", .{layer}), .bf16, &.{512}, if (binding) &attn.?.kv_norm else null);
        const owner = layer == 2 or layer == 8 or layer == 14 or layer == 20;
        if (binding) {
            attn.?.compressor = if (owner) @as(Compressor, undefined) else null;
            attn.?.indexer = if (owner or layer == 24 or layer == 28 or layer == 32 or layer == 36)
                @as(Indexer, undefined)
            else
                null;
        }
        if (owner) {
            const compressor: ?*Compressor = if (binding) &attn.?.compressor.? else null;
            try matrixAny(binding, source, try named(&prefix, "layers.{d}.attn.compressor.wkv", .{layer}), 5120, 512, if (binding) &compressor.?.wkv else null);
            try plain(binding, source, try named(&name, "layers.{d}.attn.compressor.norm.weight", .{layer}), .bf16, &.{512}, if (binding) &compressor.?.norm else null);
            if (binding) compressor.?.wgate = if (layer < 20) @as(AnyMatrix, undefined) else null;
            if (layer < 20) try matrixAny(binding, source, try named(&prefix, "layers.{d}.attn.compressor.wgate", .{layer}), 5120, 512, if (binding) &compressor.?.wgate.? else null);
        }
        if (owner or layer == 24 or layer == 28 or layer == 32 or layer == 36) {
            const indexer: ?*Indexer = if (binding) &attn.?.indexer.? else null;
            try matrixAny(binding, source, try named(&prefix, "layers.{d}.attn.indexer.wq_b", .{layer}), 1280, 4096, if (binding) &indexer.?.wq_b else null);
            try plainEither(binding, source, try named(&name, "layers.{d}.attn.indexer.weights_proj.weight", .{layer}), &.{ 32, 5120 }, if (binding) &indexer.?.weights_proj else null);
            if (binding) {
                indexer.?.wk = if (owner) @as(AnyMatrix, undefined) else null;
                indexer.?.k_norm = if (owner) @as(Tensor, undefined) else null;
            }
            if (owner) {
                try matrixAny(binding, source, try named(&prefix, "layers.{d}.attn.indexer.wk", .{layer}), 512, 128, if (binding) &indexer.?.wk.? else null);
                try plain(binding, source, try named(&name, "layers.{d}.attn.indexer.k_norm.weight", .{layer}), .bf16, &.{128}, if (binding) &indexer.?.k_norm.? else null);
            }
        }
        const moe: ?*Moe = if (binding) &out.?.moe else null;
        // The released checkpoints store the router gate f16; rank-split repacks store it bf16.
        if (binding) {
            const tensor = try source.get(try named(&name, "layers.{d}.ffn.gate.weight", .{layer}));
            if ((tensor.dtype != .f16 and tensor.dtype != .bf16) or tensor.rank != 2 or
                tensor.shape[0] != 384 or tensor.shape[1] != 5120)
            {
                std.log.err("unsupported router gate {s}: {s} {any}", .{ name, @tagName(tensor.dtype), tensor.shape[0..tensor.rank] });
                return error.UnsupportedTensor;
            }
            moe.?.gate = tensor;
        } else {
            const tensor = try source.get(try named(&name, "layers.{d}.ffn.gate.weight", .{layer}));
            if ((tensor.dtype != .f16 and tensor.dtype != .bf16) or tensor.rank != 2 or
                tensor.shape[0] != 384 or tensor.shape[1] != 5120) return error.UnsupportedTensor;
        }
        // The bias rides the same three dtypes the route kernel decodes (f32, f16, bf16).
        if (binding) {
            const tensor = try source.get(try named(&name, "layers.{d}.ffn.gate.bias", .{layer}));
            if ((tensor.dtype != .f16 and tensor.dtype != .bf16 and tensor.dtype != .f32) or tensor.rank != 1 or tensor.shape[0] != 384)
                return error.UnsupportedTensor;
            moe.?.bias = tensor;
        } else {
            const tensor = try source.get(try named(&name, "layers.{d}.ffn.gate.bias", .{layer}));
            if ((tensor.dtype != .f16 and tensor.dtype != .bf16 and tensor.dtype != .f32) or tensor.rank != 1 or tensor.shape[0] != 384)
                return error.UnsupportedTensor;
        }
        inline for (.{ "w1", "w3", "w2" }, 0..) |part, projection| {
            const k: usize = if (projection == 2) 2304 else 5120;
            const n: usize = if (projection == 2) 5120 else 2304;
            try matrix(binding, source, try named(&prefix, "layers.{d}.ffn.shared_experts.{s}", .{ layer, part }), k, n, if (binding) &moe.?.shared[projection] else null);
            if (!binding) for (0..384) |expert| try matrix(false, source, try named(&prefix, "layers.{d}.ffn.experts.{d}.{s}", .{ layer, expert, part }), k, n, null);
        }
        const engram = layer == 1 or layer == 14;
        if (binding) out.?.engram = if (engram) @as(Engram, undefined) else null;
        if (engram) {
            const e: ?*Engram = if (binding) &out.?.engram.? else null;
            const rows: usize = if (layer == 1) 384006168 else 384016682;
            if (stream) {
                // The row source reads these a window at a time from the release's shards; there is no resident view.
                if (binding) {
                    e.?.table = null;
                    e.?.scales = null;
                    e.?.streamed = true;
                }
            } else {
                var weight_tensor: Tensor = undefined;
                var scale_tensor: Tensor = undefined;
                try plain(binding, source, try named(&name, "layers.{d}.engram.embed.weight", .{layer}), .f8_e4m3, &.{ rows, 256 }, if (binding) &weight_tensor else null);
                try plain(binding, source, try named(&name, "layers.{d}.engram.embed.scale", .{layer}), .f8_e8m0, &.{ rows, 8 }, if (binding) &scale_tensor else null);
                if (binding) {
                    e.?.table = weight_tensor;
                    e.?.scales = scale_tensor;
                    // Engram is constructed as `undefined`, so the field's default never applies: say it explicitly.
                    e.?.streamed = false;
                }
            }
            try plainEither(binding, source, try named(&name, "layers.{d}.engram.q_weight", .{layer}), &.{ 4, 5120 }, if (binding) &e.?.q_weight else null);
            try plainEither(binding, source, try named(&name, "layers.{d}.engram.k_weight", .{layer}), &.{ 4, 5120 }, if (binding) &e.?.k_weight else null);
            try matrix(binding, source, try named(&prefix, "layers.{d}.engram.wkv", .{layer}), 6144, 25600, if (binding) &e.?.wkv else null);
        }
    }
}
