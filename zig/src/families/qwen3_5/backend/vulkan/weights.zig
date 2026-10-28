//! The native Vulkan Qwen3.5-2B weights: every checkpoint tensor of the one
//! supported configuration uploaded as typed device views over stable,
//! heap-owned buffers. The oracle is the unchanged Metal native loader (raw
//! BF16 norm gains passed through untouched — no `1 + w`, no F32 norm
//! conversion); the shared HIP loader's F32-norm policy is deliberately not
//! used. Admission is fail-closed: the pinned Qwen3.5-2B geometry, affine Q4
//! group-64 quantization, dense MLP, and every tensor's exact dtype and shape
//! are checked before any byte is read; anything else is refused, never
//! silently adapted. `Model.load` keeps its errors inferred, so config and
//! checkpoint I/O errors reach the caller as themselves, not translated.
//! Weights are streamed through a small staging buffer with
//! the existing checked uploads; no CPU dequantization and no inference here.
//!
//! Views point into buffers this module allocated on the heap and owns until
//! `deinit`, so no view ever dangles behind a moved or stack-allocated buffer.
//! Every load-time GPU operation (the staged uploads) runs through `Commands`
//! and completes before `load` returns; a `Timeout` or `InFlight` at any
//! point goes through `commands.orExit` inside the loader — exiting the
//! process rather than unwinding into the frees of a load whose submissions
//! may still be in flight.
const std = @import("std");
const vk = @import("vulkan");
const pinned = @import("../../config.zig");
const loader = @import("../../loader.zig");

const Allocator = std.mem.Allocator;
const Buffer = vk.Buffer;
const Commands = vk.Commands;
const Device = vk.Device;
const Table = loader.table.Table;
const Tensor = loader.table.Tensor;
const View = vk.tensor.View;

pub const Error = loader.table.Error || pinned.Error || vk.runtime.Error ||
    error{UnsupportedModel};

/// The text tower's key prefix, as the shared loader names it.
const prefix = loader.checkpoint.prefix;

/// One quantized projection on the device: packed affine words, per-group
/// scales and biases, and the row/column geometry the qmm kernel checks.
pub const Quant = struct {
    w: View,
    scales: View,
    biases: View,
    n: u64,
    k: u64,
};

pub const DenseMlp = struct { gate: Quant, up: Quant, down: Quant };

/// A full-attention layer: Q4 projections and raw BF16 norms. `q` holds the
/// interleaved query and output-gate rows (4096 rows for 8 query heads of
/// 256); the split is the forward's job, exactly as the Metal oracle reads it.
pub const FullLayer = struct {
    input_norm: View,
    post_norm: View,
    q: Quant,
    k: Quant,
    v: Quant,
    o: Quant,
    q_norm: View,
    k_norm: View,
    mlp: DenseMlp,
};

/// A linear (GatedDeltaNet) layer: everything as the checkpoint stores it.
/// The conv taps stay raw BF16 for both decode and prefill — conv_dw1d reads
/// a uint16 storage type and widens inside the kernel, the same raw taps the
/// Metal oracle passes. Norm gains stay raw BF16; A_log stays raw F32;
/// dt_bias stays raw BF16.
pub const LinearLayer = struct {
    input_norm: View,
    post_norm: View,
    qkv: Quant,
    z: Quant,
    a: Quant,
    b: Quant,
    out: Quant,
    gnorm: View,
    conv: View,
    a_log: View,
    dt_bias: View,
    mlp: DenseMlp,
};

pub const Layer = union(enum) { full: FullLayer, linear: LinearLayer };

/// The one supported model, checked field by field against the pinned family
/// constants before anything is trusted.
pub fn expectPinnedModel(s: loader.config.Spec) Error!void {
    const ok =
        s.hidden == pinned.hidden and
        s.vocab == pinned.vocab and
        s.n_layers == pinned.layers and
        s.intermediate == pinned.intermediate and
        s.heads == pinned.query_heads and
        s.kv_heads == pinned.kv_heads and
        s.head_dim == pinned.head_dim and
        s.key_heads == pinned.linear_heads and
        s.value_heads == pinned.linear_heads and
        s.key_dim == pinned.linear_heads * pinned.linear_dim and
        s.value_dim == pinned.linear_heads * pinned.linear_dim and
        s.conv == pinned.conv_dim and
        s.rotary_dim == pinned.rotary_dim and
        s.full_every == 4 and
        s.bits == 4 and
        s.group == pinned.group and
        s.experts == 0;
    if (!ok) return error.UnsupportedModel;
}

comptime {
    if (pinned.layers % 4 != 0) @compileError("the pinned layer count must keep the full-attention interval exact");
    if (pinned.conv_dim != 2 * pinned.linear_heads * pinned.linear_dim + pinned.linear_heads * pinned.linear_dim)
        @compileError("conv channels drifted from 2*NK*DK + NV*DV");
    if (pinned.query_heads * pinned.head_dim * 2 != 4096) @compileError("qraw row count drifted from the interleaved q/gate layout");
}

fn kindOf(dt: loader.table.DType) ?vk.tensor.Kind {
    return switch (dt) {
        .f32 => .f32,
        .bf16 => .bf16,
        .u32 => .u32,
        else => null,
    };
}

/// A checked exact dtype/shape, logged by key when it refuses.
fn expectTensor(key: []const u8, t: Tensor, dt: loader.table.DType, shape: []const usize) Error!void {
    if (!t.is(dt, shape)) {
        std.log.err("{s}: expected {s} {any}, found {s} {any}", .{ key, @tagName(dt), shape, @tagName(t.dtype), t.shape[0..t.rank] });
        return error.UnsupportedModel;
    }
}

/// One whole-model upload in progress. Every device buffer it creates is
/// heap-owned and recorded; `deinit` frees exactly what was created, also on
/// the error path, so a partial load never leaks or double-frees.
const Builder = struct {
    gpa: Allocator,
    dev: *const Device,
    cmds: *Commands,
    staging: Buffer,
    timeout_ns: u64,
    table: *const Table,
    keep: std.ArrayList(*Buffer) = .empty,

    fn deinit(b: *Builder) void {
        for (b.keep.items) |ptr| {
            var buf = ptr.*;
            buf.deinit();
            b.gpa.destroy(ptr);
        }
        b.keep.deinit(b.gpa);
        b.* = undefined;
    }

    fn key(buf: *[192]u8, comptime fmt: []const u8, args: anytype) Error![]const u8 {
        return std.fmt.bufPrint(buf, fmt, args) catch error.UnsupportedModel;
    }

    /// The named tensor, dtype-checked before its mapped bytes are touched.
    fn named(b: *Builder, name: []const u8) Error!Tensor {
        const t = try b.table.get(name);
        switch (t.dtype) {
            .f32, .bf16, .u32 => {},
            else => {
                std.log.err("{s}: unsupported tensor dtype {s}", .{ name, @tagName(t.dtype) });
                return error.UnsupportedModel;
            },
        }
        return t;
    }

    /// A device buffer holding `t`'s bytes exactly, streamed through the
    /// staging buffer; the returned view is typed and word-padded by `View`.
    ///
    /// Ownership transfers exactly once: until the slot is registered in
    /// `keep`, this function alone cleans up (slot plus the Vk allocation) on
    /// every failure; after registration only `Builder.deinit` frees it, so a
    /// failed upload never double-frees or leaves a dangling slot.
    fn tensor(b: *Builder, t: Tensor) !View {
        const kind = kindOf(t.dtype) orelse return error.UnsupportedModel;
        const elements: u64 = @intCast(t.numel());
        const bytes: u64 = @intCast(t.bytes.len);
        const ptr = try b.gpa.create(Buffer);
        ptr.* = Buffer.init(b.dev, .device, bytes) catch |e| {
            b.gpa.destroy(ptr);
            return e;
        };
        b.keep.append(b.gpa, ptr) catch |e| {
            var buf = ptr.*;
            buf.deinit();
            b.gpa.destroy(ptr);
            return e;
        };
        // registered: in-flight failures exit the process here (orExit),
        // never unwinding into the frees below this load's scope
        try vk.orExit(vk.upload(b.cmds, ptr.*, 0, b.staging, t.bytes, b.timeout_ns));
        return View.init(ptr, kind, 0, elements);
    }

    /// A raw tensor of the exact given dtype and shape, uploaded unchanged.
    fn raw(b: *Builder, name: []const u8, dt: loader.table.DType, shape: []const usize) !View {
        const t = try b.named(name);
        try expectTensor(name, t, dt, shape);
        return b.tensor(t);
    }

    /// A packed affine projection of `n` rows over `k` inputs, group 64:
    /// U32 words [n, k/8] plus BF16 scales and biases [n, k/64], all checked.
    fn quant(b: *Builder, name: []const u8, n: usize, k: usize) !Quant {
        var buf: [192]u8 = undefined;
        const w = try b.raw(try key(&buf, "{s}.weight", .{name}), .u32, &.{ n, k / 8 });
        const scales = try b.raw(try key(&buf, "{s}.scales", .{name}), .bf16, &.{ n, k / 64 });
        const biases = try b.raw(try key(&buf, "{s}.biases", .{name}), .bf16, &.{ n, k / 64 });
        return .{ .w = w, .scales = scales, .biases = biases, .n = n, .k = k };
    }

    /// A raw BF16 norm gain, passed through byte for byte (the Metal oracle's
    /// semantics: the kernel multiplies the gain directly, no `1 + w`).
    fn norm(b: *Builder, name: []const u8, elements: usize) !View {
        return b.raw(name, .bf16, &.{elements});
    }

    fn mlp(b: *Builder, base: []const u8) !DenseMlp {
        var buf: [192]u8 = undefined;
        return .{
            .gate = try b.quant(try key(&buf, "{s}mlp.gate_proj", .{base}), pinned.intermediate, pinned.hidden),
            .up = try b.quant(try key(&buf, "{s}mlp.up_proj", .{base}), pinned.intermediate, pinned.hidden),
            .down = try b.quant(try key(&buf, "{s}mlp.down_proj", .{base}), pinned.hidden, pinned.intermediate),
        };
    }

    fn layer(b: *Builder, index: usize) !Layer {
        var buf: [192]u8 = undefined;
        const base = try key(&buf, prefix ++ "layers.{d}.", .{index});
        const input_norm = try b.norm(try key(&buf, "{s}input_layernorm.weight", .{base}), pinned.hidden);
        const post_norm = try b.norm(try key(&buf, "{s}post_attention_layernorm.weight", .{base}), pinned.hidden);
        const m = try b.mlp(base);
        if (pinned.linear(index)) {
            const lin = try key(&buf, "{s}linear_attn.", .{base});
            return .{ .linear = .{
                .input_norm = input_norm,
                .post_norm = post_norm,
                .qkv = try b.quant(try key(&buf, "{s}in_proj_qkv", .{lin}), 3 * pinned.linear_heads * pinned.linear_dim, pinned.hidden),
                .z = try b.quant(try key(&buf, "{s}in_proj_z", .{lin}), pinned.linear_heads * pinned.linear_dim, pinned.hidden),
                .a = try b.quant(try key(&buf, "{s}in_proj_a", .{lin}), pinned.linear_heads, pinned.hidden),
                .b = try b.quant(try key(&buf, "{s}in_proj_b", .{lin}), pinned.linear_heads, pinned.hidden),
                .out = try b.quant(try key(&buf, "{s}out_proj", .{lin}), pinned.hidden, pinned.linear_heads * pinned.linear_dim),
                .gnorm = try b.norm(try key(&buf, "{s}norm.weight", .{lin}), pinned.linear_dim),
                .conv = try b.raw(try key(&buf, "{s}conv1d.weight", .{lin}), .bf16, &.{ pinned.conv_dim, pinned.conv_taps, 1 }),
                .a_log = try b.raw(try key(&buf, "{s}A_log", .{lin}), .f32, &.{pinned.linear_heads}),
                .dt_bias = try b.raw(try key(&buf, "{s}dt_bias", .{lin}), .bf16, &.{pinned.linear_heads}),
                .mlp = m,
            } };
        }
        const attn = try key(&buf, "{s}self_attn.", .{base});
        return .{ .full = .{
            .input_norm = input_norm,
            .post_norm = post_norm,
            .q = try b.quant(try key(&buf, "{s}q_proj", .{attn}), 2 * pinned.query_heads * pinned.head_dim, pinned.hidden),
            .k = try b.quant(try key(&buf, "{s}k_proj", .{attn}), pinned.kv_heads * pinned.head_dim, pinned.hidden),
            .v = try b.quant(try key(&buf, "{s}v_proj", .{attn}), pinned.kv_heads * pinned.head_dim, pinned.hidden),
            .o = try b.quant(try key(&buf, "{s}o_proj", .{attn}), pinned.hidden, pinned.query_heads * pinned.head_dim),
            .q_norm = try b.norm(try key(&buf, "{s}q_norm.weight", .{attn}), pinned.head_dim),
            .k_norm = try b.norm(try key(&buf, "{s}k_norm.weight", .{attn}), pinned.head_dim),
            .mlp = m,
        } };
    }
};

/// The loaded text tower: typed views over this module's own heap buffers,
/// the head tied to the embedding exactly when the checkpoint says so.
pub const Model = struct {
    gpa: Allocator,
    spec: loader.config.Spec,
    /// config.json's max_position_embeddings, as admission pinned it.
    context: usize,
    /// The head is the packed input embedding (`tie_word_embeddings`).
    tied: bool,
    buffers: []*Buffer,
    embed: Quant,
    final_norm: View,
    /// `null` when tied; `outputHead` returns `embed` in that case. A missing
    /// separate head on an untied checkpoint is a refused load, never an
    /// assumed tie.
    head: ?Quant,
    layers: []Layer,

    /// Loads the checkpoint directory `dir` onto `dev` through `cmds`.
    /// `staging` is the caller's staging buffer (any nonempty size; transfers
    /// chunk through it). Admission runs first, so an unsupported checkpoint
    /// is refused before any GPU work. All returned views stay valid for the
    /// model's lifetime: every backing buffer is heap-owned and never moved.
    pub fn load(
        gpa: Allocator,
        dev: *const Device,
        cmds: *Commands,
        io: std.Io,
        dir: []const u8,
        staging: Buffer,
        timeout_ns: u64,
    ) !Model {
        // pinned admission: model type, geometry, quantization and tying,
        // straight from config.json — before anything else is read
        const cfg = try pinned.Config.read(gpa, io, dir);
        var ck = try loader.checkpoint.Checkpoint.open(gpa, io, dir);
        defer ck.close();
        try expectPinnedModel(ck.spec());

        var b: Builder = .{
            .gpa = gpa,
            .dev = dev,
            .cmds = cmds,
            .staging = staging,
            .timeout_ns = timeout_ns,
            .table = &ck.main,
        };
        defer b.deinit();

        var layers: std.ArrayList(Layer) = .empty;
        errdefer layers.deinit(gpa);

        // the output head: tied means the embedding; untied means a real
        // lm_head that must exist — a missing one refuses the load
        const head: ?Quant = if (cfg.tied) null else try b.quant("language_model.lm_head", pinned.vocab, pinned.hidden);

        const embed = try b.quant(prefix ++ "embed_tokens", pinned.vocab, pinned.hidden);
        const final_norm = try b.norm(prefix ++ "norm.weight", pinned.hidden);
        // each part's mapped pages go once it is up: host RSS holds one layer
        // of the checkpoint at a time, not the whole file set
        ck.release(null);
        for (0..pinned.layers) |i| {
            try layers.append(gpa, try b.layer(i));
            ck.release(i);
        }

        const owned_layers = try layers.toOwnedSlice(gpa);
        errdefer gpa.free(owned_layers);
        // ownership moves to the model; the builder's defer is now a no-op
        const buffers = try b.keep.toOwnedSlice(gpa);
        return .{
            .gpa = gpa,
            .spec = ck.spec(),
            .context = cfg.context,
            .tied = cfg.tied,
            .buffers = buffers,
            .embed = embed,
            .final_norm = final_norm,
            .head = head,
            .layers = owned_layers,
        };
    }

    /// The output head: the embedding itself when the checkpoint ties them.
    pub fn outputHead(m: *const Model) Quant {
        return m.head orelse m.embed;
    }

    /// Frees every device buffer. Call only when no submission touching any
    /// view can still be pending; in-flight errors already exited via orExit.
    pub fn deinit(m: *Model) void {
        for (m.buffers) |ptr| {
            var buf = ptr.*;
            buf.deinit();
            m.gpa.destroy(ptr);
        }
        m.gpa.free(m.buffers);
        m.gpa.free(m.layers);
        m.* = undefined;
    }
};

const testing = std.testing;

test "kindOf maps exactly the dtypes the views carry" {
    try testing.expectEqual(@as(?vk.tensor.Kind, .f32), kindOf(.f32));
    try testing.expectEqual(@as(?vk.tensor.Kind, .bf16), kindOf(.bf16));
    try testing.expectEqual(@as(?vk.tensor.Kind, .u32), kindOf(.u32));
    try testing.expectEqual(@as(?vk.tensor.Kind, null), kindOf(.f16));
}

test "the pinned-model check refuses drift in any field" {
    var s: loader.config.Spec = .{
        .hidden = pinned.hidden,
        .intermediate = pinned.intermediate,
        .n_layers = pinned.layers,
        .heads = pinned.query_heads,
        .kv_heads = pinned.kv_heads,
        .head_dim = pinned.head_dim,
        .key_heads = pinned.linear_heads,
        .value_heads = pinned.linear_heads,
        .key_dim = pinned.linear_heads * pinned.linear_dim,
        .value_dim = pinned.linear_heads * pinned.linear_dim,
        .conv = pinned.conv_dim,
        .vocab = pinned.vocab,
        .eps = pinned.eps,
        .rope_theta = pinned.theta,
        .rotary_dim = pinned.rotary_dim,
        .full_every = 4,
        .bits = 4,
        .group = pinned.group,
        .experts = 0,
        .top_k = 0,
        .moe_width = 0,
        .shared_width = 0,
        .mtp_layers = 0,
    };
    try expectPinnedModel(s);
    s.bits = 8;
    try testing.expectError(error.UnsupportedModel, expectPinnedModel(s));
    s.bits = 4;
    s.experts = 64;
    try testing.expectError(error.UnsupportedModel, expectPinnedModel(s));
    s.experts = 0;
    s.full_every = 6;
    try testing.expectError(error.UnsupportedModel, expectPinnedModel(s));
    s.full_every = 4;
    s.head_dim = 128;
    try testing.expectError(error.UnsupportedModel, expectPinnedModel(s));
}

test "tensor shape checks refuse rank and dim drift by name" {
    const t: Tensor = .{ .dtype = .bf16, .rank = 1, .shape = .{ 2048, 0, 0, 0, 0 }, .bytes = &.{} };
    try testing.expect(t.is(.bf16, &.{2048}));
    try testing.expect(!t.is(.f32, &.{2048}));
    try testing.expect(!t.is(.bf16, &.{2049}));
    try testing.expect(!t.is(.bf16, &.{ 2048, 1 }));
}
