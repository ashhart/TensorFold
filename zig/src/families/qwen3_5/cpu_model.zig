//! A native Qwen3.5-2B model in host memory: the pinned config, each tensor read under weights.zig's dtype gate.
const std = @import("std");
const cfg = @import("config.zig");
const weights = @import("loader.zig");
const kernels = @import("cpu_kernels.zig");
const Pool = @import("cpu_pool.zig").Pool;
const Table = weights.table.Table;
const Tensor = weights.table.Tensor;
const DType = weights.table.DType;

pub const Linear = kernels.Linear;
/// A delta layer's weights: its projections, and its conv taps, norm, dt bias (bf16 bits) and A_log (fp32).
pub const Delta = struct { qkv: Linear, z: Linear, a: Linear, b: Linear, out: Linear, conv: []const u16, norm: []const u16, a_log: []const f32, dt_bias: []const u16 };
pub const Attention = struct { q: Linear, k: Linear, v: Linear, out: Linear, q_norm: []const u16, k_norm: []const u16 };
pub const Layer = struct {
    input_norm: []const u16,
    post_norm: []const u16,
    gate: Linear,
    up: Linear,
    down: Linear,
    mixer: union(enum) { delta: Delta, attention: Attention },
};

/// The accumulation chains of a projection class, the contract's three registrations: a and b, the head, the rest.
pub fn chainsFor(outputs: usize) usize {
    return if (outputs == cfg.linear_heads) 32 else if (outputs == cfg.vocab) 8 else 16;
}

pub const Model = struct {
    gpa: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,
    config: cfg.Config,
    pool: *Pool,
    embedding: Linear,
    /// The checkpoint's own output projection when it ships one, else null (the head is the packed embedding).
    separate_head: ?Linear,
    norm: []const u16,
    blocks: [cfg.layers]Layer,
    /// Bytes of words and tables the model holds.
    bytes: u64 = 0,

    pub fn head(m: *const Model) Linear {
        return m.separate_head orelse m.embedding;
    }

    /// Reads the checkpoint at `dir` into owned buffers; `threads` workers beside the caller (null: the CPUs less one).
    pub fn load(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, threads: ?usize) !*Model {
        const config = try cfg.Config.read(gpa, io, dir);
        const m = try create(gpa, io, config, threads);
        errdefer m.deinit();
        var ckpt = try weights.Checkpoint.open(gpa, io, dir);
        defer ckpt.close();
        const t = &ckpt.main;
        const ships_head = t.has("language_model.lm_head.weight");
        if (ships_head != !config.tied) {
            std.log.err("config.json says tie_word_embeddings={} but the checkpoint {s} a language_model.lm_head projection", .{ config.tied, if (ships_head) "ships" else "lacks" });
            return error.QwenHeadTiednessMismatch;
        }
        m.embedding = try m.projection(t, "language_model.model.", "embed_tokens", cfg.vocab, cfg.hidden);
        if (ships_head) m.separate_head = try m.projection(t, "language_model.", "lm_head", cfg.vocab, cfg.hidden);
        m.norm = try m.bits(t, "language_model.model.", "norm.weight", &.{cfg.hidden});
        for (&m.blocks, 0..) |*block, i| {
            var buf: [128]u8 = undefined;
            const p = try std.fmt.bufPrint(&buf, "language_model.model.layers.{d}.", .{i});
            block.* = .{
                .input_norm = try m.bits(t, p, "input_layernorm.weight", &.{cfg.hidden}),
                .post_norm = try m.bits(t, p, "post_attention_layernorm.weight", &.{cfg.hidden}),
                .gate = try m.projection(t, p, "mlp.gate_proj", cfg.intermediate, cfg.hidden),
                .up = try m.projection(t, p, "mlp.up_proj", cfg.intermediate, cfg.hidden),
                .down = try m.projection(t, p, "mlp.down_proj", cfg.hidden, cfg.intermediate),
                .mixer = undefined,
            };
            if (cfg.linear(i)) {
                block.mixer = .{ .delta = .{
                    .qkv = try m.projection(t, p, "linear_attn.in_proj_qkv", cfg.conv_dim, cfg.hidden),
                    .z = try m.projection(t, p, "linear_attn.in_proj_z", cfg.hidden, cfg.hidden),
                    .a = try m.projection(t, p, "linear_attn.in_proj_a", cfg.linear_heads, cfg.hidden),
                    .b = try m.projection(t, p, "linear_attn.in_proj_b", cfg.linear_heads, cfg.hidden),
                    .out = try m.projection(t, p, "linear_attn.out_proj", cfg.hidden, cfg.hidden),
                    .conv = try m.bits(t, p, "linear_attn.conv1d.weight", &.{ cfg.conv_dim, cfg.conv_taps, 1 }),
                    .norm = try m.bits(t, p, "linear_attn.norm.weight", &.{cfg.linear_dim}),
                    .a_log = try m.floats(t, p, "linear_attn.A_log", &.{cfg.linear_heads}),
                    .dt_bias = try m.bits(t, p, "linear_attn.dt_bias", &.{cfg.linear_heads}),
                } };
            } else {
                block.mixer = .{ .attention = .{
                    .q = try m.projection(t, p, "self_attn.q_proj", 2 * cfg.hidden, cfg.hidden),
                    .k = try m.projection(t, p, "self_attn.k_proj", cfg.kv_heads * cfg.head_dim, cfg.hidden),
                    .v = try m.projection(t, p, "self_attn.v_proj", cfg.kv_heads * cfg.head_dim, cfg.hidden),
                    .out = try m.projection(t, p, "self_attn.o_proj", cfg.hidden, cfg.hidden),
                    .q_norm = try m.bits(t, p, "self_attn.q_norm.weight", &.{cfg.head_dim}),
                    .k_norm = try m.bits(t, p, "self_attn.k_norm.weight", &.{cfg.head_dim}),
                } };
            }
            ckpt.release(i);
        }
        return m;
    }

    fn create(gpa: std.mem.Allocator, io: std.Io, config: cfg.Config, threads: ?usize) !*Model {
        const m = try gpa.create(Model);
        errdefer gpa.destroy(m);
        m.* = .{ .gpa = gpa, .arena = .init(gpa), .config = config, .pool = undefined, .embedding = undefined, .separate_head = null, .norm = &.{}, .blocks = undefined };
        errdefer m.arena.deinit();
        m.pool = try Pool.init(gpa, io, threads);
        return m;
    }

    pub fn deinit(m: *Model) void {
        m.pool.deinit();
        m.arena.deinit();
        m.gpa.destroy(m);
    }

    /// The tensor `prefix ++ suffix` of the expected dtype and shape, as weights.zig's `expect` admits it.
    fn tensor(t: *const Table, prefix: []const u8, suffix: []const u8, dtype: DType, shape: []const usize) !Tensor {
        var buf: [256]u8 = undefined;
        const name = try std.fmt.bufPrint(&buf, "{s}{s}", .{ prefix, suffix });
        const x = t.get(name) catch return error.UnexpectedQwenTensor;
        if (x.dtype != dtype or x.rank != shape.len or !std.mem.eql(usize, x.shape[0..x.rank], shape)) {
            std.log.err("{s}: unexpected Qwen tensor dtype or shape", .{name});
            return error.UnexpectedQwenTensor;
        }
        return x;
    }

    /// A bf16 tensor's bits copied as stored (little-endian pairs on every host built here).
    fn bits(m: *Model, t: *const Table, prefix: []const u8, suffix: []const u8, shape: []const usize) ![]const u16 {
        const x = try tensor(t, prefix, suffix, .bf16, shape);
        const out = try m.arena.allocator().alloc(u16, x.numel());
        for (out, 0..) |*o, i| o.* = std.mem.readInt(u16, x.bytes[2 * i ..][0..2], .little);
        return out;
    }

    /// An fp32 tensor copied as stored.
    fn floats(m: *Model, t: *const Table, prefix: []const u8, suffix: []const u8, shape: []const usize) ![]const f32 {
        const x = try tensor(t, prefix, suffix, .f32, shape);
        const out = try m.arena.allocator().alloc(f32, x.numel());
        for (out, 0..) |*o, i| o.* = @bitCast(std.mem.readInt(u32, x.bytes[4 * i ..][0..4], .little));
        return out;
    }

    /// An affine 4-bit projection: u32 words [N, K/8], bf16 scales and biases [N, K/64], in the kernel's tiles.
    fn projection(m: *Model, t: *const Table, prefix: []const u8, suffix: []const u8, outputs: usize, inputs: usize) !Linear {
        var buf: [256]u8 = undefined;
        const name = try std.fmt.bufPrint(&buf, "{s}{s}", .{ prefix, suffix });
        const w = try tensor(t, name, ".weight", .u32, &.{ outputs, inputs / 8 });
        const sc = try tensor(t, name, ".scales", .bf16, &.{ outputs, inputs / cfg.group });
        const bi = try tensor(t, name, ".biases", .bf16, &.{ outputs, inputs / cfg.group });
        m.bytes += w.bytes.len + sc.bytes.len + bi.bytes.len;
        const a = m.arena.allocator();
        const words = try a.alloc(u32, outputs * inputs / 8);
        defer a.free(words);
        for (words, 0..) |*o, i| o.* = std.mem.readInt(u32, w.bytes[4 * i ..][0..4], .little);
        const scales = try a.alloc(u16, outputs * inputs / cfg.group);
        defer a.free(scales);
        for (scales, 0..) |*o, i| o.* = std.mem.readInt(u16, sc.bytes[2 * i ..][0..2], .little);
        const biases = try a.alloc(u16, outputs * inputs / cfg.group);
        defer a.free(biases);
        for (biases, 0..) |*o, i| o.* = std.mem.readInt(u16, bi.bytes[2 * i ..][0..2], .little);
        return kernels.pack(a, words, scales, biases, outputs, inputs, chainsFor(outputs));
    }

    /// Random weights at the 2B geometry, for the exactness tests (no checkpoint).
    pub fn synthetic(gpa: std.mem.Allocator, io: std.Io, seed: u64, threads: ?usize) !*Model {
        const m = try create(gpa, io, .{ .context = 4096, .tied = true }, threads);
        errdefer m.deinit();
        var prng = std.Random.DefaultPrng.init(seed);
        const rnd = prng.random();
        m.embedding = try m.randomLinear(rnd, cfg.vocab, cfg.hidden);
        m.norm = try m.randomBits(rnd, cfg.hidden, 0.8, 1.2);
        for (&m.blocks, 0..) |*block, i| {
            block.* = .{
                .input_norm = try m.randomBits(rnd, cfg.hidden, 0.8, 1.2),
                .post_norm = try m.randomBits(rnd, cfg.hidden, 0.8, 1.2),
                .gate = try m.randomLinear(rnd, cfg.intermediate, cfg.hidden),
                .up = try m.randomLinear(rnd, cfg.intermediate, cfg.hidden),
                .down = try m.randomLinear(rnd, cfg.hidden, cfg.intermediate),
                .mixer = undefined,
            };
            if (cfg.linear(i)) {
                const a_log = try m.arena.allocator().alloc(f32, cfg.linear_heads);
                for (a_log) |*o| o.* = rnd.float(f32) * 2 - 1;
                block.mixer = .{ .delta = .{
                    .qkv = try m.randomLinear(rnd, cfg.conv_dim, cfg.hidden),
                    .z = try m.randomLinear(rnd, cfg.hidden, cfg.hidden),
                    .a = try m.randomLinear(rnd, cfg.linear_heads, cfg.hidden),
                    .b = try m.randomLinear(rnd, cfg.linear_heads, cfg.hidden),
                    .out = try m.randomLinear(rnd, cfg.hidden, cfg.hidden),
                    .conv = try m.randomBits(rnd, cfg.conv_dim * cfg.conv_taps, -0.5, 0.5),
                    .norm = try m.randomBits(rnd, cfg.linear_dim, 0.8, 1.2),
                    .a_log = a_log,
                    .dt_bias = try m.randomBits(rnd, cfg.linear_heads, -1, 1),
                } };
            } else {
                block.mixer = .{ .attention = .{
                    .q = try m.randomLinear(rnd, 2 * cfg.hidden, cfg.hidden),
                    .k = try m.randomLinear(rnd, cfg.kv_heads * cfg.head_dim, cfg.hidden),
                    .v = try m.randomLinear(rnd, cfg.kv_heads * cfg.head_dim, cfg.hidden),
                    .out = try m.randomLinear(rnd, cfg.hidden, cfg.hidden),
                    .q_norm = try m.randomBits(rnd, cfg.head_dim, 0.8, 1.2),
                    .k_norm = try m.randomBits(rnd, cfg.head_dim, 0.8, 1.2),
                } };
            }
        }
        return m;
    }

    fn randomBits(m: *Model, rnd: std.Random, n: usize, lo: f32, hi: f32) ![]const u16 {
        const out = try m.arena.allocator().alloc(u16, n);
        for (out) |*o| o.* = kernels.bf16of(lo + rnd.float(f32) * (hi - lo));
        return out;
    }

    fn randomLinear(m: *Model, rnd: std.Random, outputs: usize, inputs: usize) !Linear {
        const a = m.arena.allocator();
        const words = try a.alloc(u32, outputs * inputs / 8);
        defer a.free(words);
        for (words) |*w| w.* = rnd.int(u32);
        const groups = inputs / cfg.group;
        const scales = try a.alloc(u16, outputs * groups);
        defer a.free(scales);
        const biases = try a.alloc(u16, outputs * groups);
        defer a.free(biases);
        const spread = 1.0 / @sqrt(@as(f32, @floatFromInt(inputs)));
        for (scales, biases) |*s, *b| {
            const width = rnd.float(f32) * spread * 0.5 + spread * 0.05;
            s.* = kernels.bf16of(width / 15.0 * 2.0);
            b.* = kernels.bf16of(-width);
        }
        return kernels.pack(a, words, scales, biases, outputs, inputs, chainsFor(outputs));
    }
};

test "the chain classes are the contract's three registrations" {
    try std.testing.expectEqual(@as(usize, 32), chainsFor(cfg.linear_heads));
    try std.testing.expectEqual(@as(usize, 8), chainsFor(cfg.vocab));
    for ([_]usize{ cfg.hidden, cfg.intermediate, cfg.conv_dim, 2 * cfg.hidden, cfg.kv_heads * cfg.head_dim }) |n| try std.testing.expectEqual(@as(usize, 16), chainsFor(n));
}
