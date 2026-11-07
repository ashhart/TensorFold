// Qwen3.5-2B Vulkan: fail-closed Q4 admission before reads; raw BF16 norms; heap views; staged uploads use orExit.
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

// A `Quant` stores packed affine words, per-group scales and biases, plus row/column geometry checked by qmm.
pub const Quant = struct {
    w: View,
    scales: View,
    biases: View,
    n: u64,
    k: u64,
};

pub const DenseMlp = struct { gate: Quant, up: Quant, down: Quant };

// Full attention uses Q4/raw BF16; `q` has 4096 interleaved query/gate rows (8×256), split by forward to match Metal.
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

// GatedDeltaNet keeps raw BF16 taps/norms/`dt_bias`; `conv_dw1d` widens uint16, and `A_log` stays raw F32.
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

// The one supported model, checked field by field against the pinned family constants before anything is trusted.
pub fn expectPinnedModel(s: loader.config.Spec) Error!void {
    const expected = .{
        .hidden = pinned.hidden,
        .vocab = pinned.vocab,
        .n_layers = pinned.layers,
        .intermediate = pinned.intermediate,
        .heads = pinned.query_heads,
        .kv_heads = pinned.kv_heads,
        .head_dim = pinned.head_dim,
        .key_heads = pinned.linear_heads,
        .value_heads = pinned.linear_heads,
        // Spec stores per-head dimensions and conv taps, not channel widths.
        .key_dim = pinned.linear_dim,
        .value_dim = pinned.linear_dim,
        .conv = pinned.conv_taps,
        .rotary_dim = pinned.rotary_dim,
        .full_every = 4,
        .bits = 4,
        .group = pinned.group,
        .experts = 0,
    };
    inline for (@typeInfo(@TypeOf(expected)).@"struct".field_names) |field| {
        if (@field(s, field) != @field(expected, field)) {
            std.debug.print("Vulkan weights admission: Spec.{s}: expected {d}, found {d}\n", .{ field, @field(expected, field), @field(s, field) });
            return error.UnsupportedModel;
        }
    }
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

// `Builder` records heap-owned buffers; `deinit` frees created buffers on errors, preventing leaks and double-frees.
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
        // Layer prefixes can point into buf: format elsewhere before copying.
        var scratch: [192]u8 = undefined;
        const name = std.fmt.bufPrint(&scratch, fmt, args) catch return error.UnsupportedModel;
        @memcpy(buf[0..name.len], name);
        return buf[0..name.len];
    }

    /// The named tensor, dtype-checked before its mapped bytes are touched.
    fn named(b: *Builder, name: []const u8) Error!Tensor {
        const t = b.table.get(name) catch |err| {
            std.debug.print("Vulkan weights tensor {s}: {s}\n", .{ name, @errorName(err) });
            return err;
        };
        switch (t.dtype) {
            .f32, .bf16, .u32 => {},
            else => {
                std.log.err("{s}: unsupported tensor dtype {s}", .{ name, @tagName(t.dtype) });
                return error.UnsupportedModel;
            },
        }
        return t;
    }

    /// Uploads go through staging; `View` pads words; `Builder.deinit` owns slots after `keep`.
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
        // Registered slots belong to `Builder.deinit`; `orExit` prevents in-flight failures from unwinding into frees.
        try vk.orExit(vk.upload(b.cmds, ptr.*, 0, b.staging, t.bytes, b.timeout_ns));
        return View.init(ptr, kind, 0, elements);
    }

    /// A raw tensor of the exact given dtype and shape, uploaded unchanged.
    fn raw(b: *Builder, name: []const u8, dt: loader.table.DType, shape: []const usize) !View {
        const t = try b.named(name);
        try expectTensor(name, t, dt, shape);
        return b.tensor(t);
    }

    // For `n` rows, `k` inputs, group 64: checked U32 `[n,k/8]` words and BF16 scales/biases `[n,k/64]`.
    fn quant(b: *Builder, name: []const u8, n: usize, k: usize) !Quant {
        var buf: [192]u8 = undefined;
        const w = try b.raw(try key(&buf, "{s}.weight", .{name}), .u32, &.{ n, k / 8 });
        const scales = try b.raw(try key(&buf, "{s}.scales", .{name}), .bf16, &.{ n, k / 64 });
        const biases = try b.raw(try key(&buf, "{s}.biases", .{name}), .bf16, &.{ n, k / 64 });
        return .{ .w = w, .scales = scales, .biases = biases, .n = n, .k = k };
    }

    // Raw BF16 norm gains pass byte-for-byte; the Metal kernel multiplies gain directly, never `1 + w`.
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

// Typed text-tower views use owned heap buffers; checkpoint config decides whether the head ties to the embedding.
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
    // When tied, `outputHead` returns `embed`; untied checkpoints require a separate head or loading is refused.
    head: ?Quant,
    layers: []Layer,

    // Loads `dir` via `cmds` with nonempty `staging`; it checks admission first and keeps heap-backed views valid.
    pub fn load(
        gpa: Allocator,
        dev: *const Device,
        cmds: *Commands,
        io: std.Io,
        dir: []const u8,
        staging: Buffer,
        timeout_ns: u64,
    ) !Model {
        // Admission pins model type, geometry, quantization, and tying from `config.json` before other reads.
        const cfg = pinned.Config.read(gpa, io, dir) catch |err| {
            std.debug.print("Vulkan weights config.json admission: {s}\n", .{@errorName(err)});
            return err;
        };
        var ck = loader.checkpoint.Checkpoint.open(gpa, io, dir) catch |err| {
            std.debug.print("Vulkan weights Checkpoint.open: {s}\n", .{@errorName(err)});
            return err;
        };
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

        // Tied output uses the embedding; untied checkpoints require an existing `lm_head` or loading is refused.
        const head: ?Quant = if (cfg.tied) null else try b.quant("language_model.lm_head", pinned.vocab, pinned.hidden);

        const embed = try b.quant(prefix ++ "embed_tokens", pinned.vocab, pinned.hidden);
        const final_norm = try b.norm(prefix ++ "norm.weight", pinned.hidden);
        // Mapped pages are released per part, so host RSS holds one checkpoint layer, not the full file set.
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

    // Free device buffers only when no submission can touch a view; in-flight errors exit through `orExit`.
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

test "tensor keys can extend a prefix in the destination buffer" {
    var buf: [192]u8 = undefined;
    const base = try Builder.key(&buf, prefix ++ "layers.{d}.", .{0});
    const name = try Builder.key(&buf, "{s}input_layernorm.weight", .{base});
    try testing.expectEqualStrings(prefix ++ "layers.0.input_layernorm.weight", name);
    const oversized: [193]u8 = @splat('x');
    try testing.expectError(error.UnsupportedModel, Builder.key(&buf, "{s}", .{oversized[0..]}));
}

test "the real pinned config geometry is admitted and field drift is refused" {
    const bytes = @embedFile("qwen35_2b_config.json");
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &digest, .{});
    try testing.expectEqualStrings("beb7fc5a6e0405fe332821cf1a8ef7b69bb390a8c8933171647de5579debf949", &std.fmt.bytesToHex(digest, .lower));
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const doc = try pinned.declared(arena.allocator(), bytes);
    const s = try pinned.geometry(try pinned.textTower(doc), 4, 64);
    try testing.expectEqual(@as(usize, 128), s.key_dim);
    try testing.expectEqual(@as(usize, 128), s.value_dim);
    try testing.expectEqual(@as(usize, 4), s.conv);
    try expectPinnedModel(s);
    inline for (.{ "hidden", "vocab", "n_layers", "intermediate", "heads", "kv_heads", "head_dim", "key_heads", "value_heads", "key_dim", "value_dim", "conv", "rotary_dim", "full_every", "bits", "group", "experts" }) |field| {
        var drift = s;
        @field(drift, field) += 1;
        try testing.expectError(error.UnsupportedModel, expectPinnedModel(drift));
    }
}

test "tensor shape checks refuse rank and dim drift by name" {
    const t: Tensor = .{ .dtype = .bf16, .rank = 1, .shape = .{ 2048, 0, 0, 0, 0 }, .bytes = &.{} };
    try testing.expect(t.is(.bf16, &.{2048}));
    try testing.expect(!t.is(.f32, &.{2048}));
    try testing.expect(!t.is(.bf16, &.{2049}));
    try testing.expect(!t.is(.bf16, &.{ 2048, 1 }));
}
