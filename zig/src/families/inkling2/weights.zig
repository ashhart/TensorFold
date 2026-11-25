//! Inkling 2 Mini's text checkpoint on one rank of any device. Every visit shares these physical layer weights.
const std = @import("std");
const core = @import("core");
const config = @import("config.zig");
const plan_ = @import("plan.zig");
const modelopt = @import("modelopt.zig");

pub const Controller = struct {
    down: u64,
    forget: u64,
    gain: u64,
};

pub const Attention = struct {
    wq: u64,
    wk: u64,
    wv: u64,
    wr: u64,
    wo: u64,
    gate: u64,
    sink: u64,
    q_scale: u64,
    rel_scale: u64,
    rel_proj: u64,
    k_conv: u64,
    v_conv: u64,
    k_dynamic: u64 = 0,
    v_dynamic: u64 = 0,
    /// Sliding layers use entry zero for q, k and v. Global layers use the three separate norms.
    norms: [3]u64,
};

/// Controller order is q, k, v and MLP, matching loop_math_cuda.messages' message layout.
pub const Layer = struct {
    attn: Attention,
    attn_conv: u64,
    mlp_norm: u64,
    mlp_conv: u64,
    router: u64,
    router_bias: u64,
    experts: modelopt.Layer,
    controllers: [4]Controller,
};

/// The device addresses of every text tensor; `Buffer` is the device's buffer: `alloc`, `ptr`, `len`, `upload`, `free`.
pub fn Weights(comptime Buffer: type) type {
    return struct {
        gpa: std.mem.Allocator,
        c: config.Config,
        p: plan_.Plan,
        embed: u64 = 0,
        embed_norm: u64 = 0,
        norm: u64 = 0,
        unembed: u64 = 0,
        output_query: u64 = 0,
        attn_queries: []u64 = &.{},
        mlp_queries: []u64 = &.{},
        cross_qm: u64 = 0,
        cross_kv: u64 = 0,
        layers: []Layer = &.{},
        buffers: std.ArrayList(Buffer) = .empty,
        bytes: usize = 0,

        pub fn deinit(w: *@This()) void {
            for (w.buffers.items) |*b| b.free();
            w.buffers.deinit(w.gpa);
            w.gpa.free(w.layers);
            w.gpa.free(w.attn_queries);
            w.gpa.free(w.mlp_queries);
            w.* = undefined;
        }
    };
}

/// All text tensors occupy one aligned device arena; expert uploads use this same bump allocator.
pub fn Loader(comptime Buffer: type) type {
    return struct {
        w: *Weights(Buffer),
        arena: Buffer,
        used: usize = 0,

        pub fn alloc(l: *@This(), len: usize) !u64 {
            const at = std.mem.alignForward(usize, l.used, 256);
            const end = try std.math.add(usize, at, len);
            if (end > l.arena.len) return error.LoopWeightsOverflow;
            l.used = end;
            l.w.bytes += len;
            return l.arena.ptr + at;
        }

        fn put(l: *@This(), bytes: []const u8) !u64 {
            const p = try l.alloc(bytes.len);
            try l.arena.upload(p - l.arena.ptr, bytes);
            return p;
        }

        fn tensor(l: *@This(), ck: *core.Checkpoint, comptime fmt: []const u8, args: anytype, dtype: core.safetensors.DType, dims: []const usize) !u64 {
            var buf: [256]u8 = undefined;
            const name = try std.fmt.bufPrint(&buf, fmt, args);
            return l.put((try ck.expect(name, dtype, dims)).bytes);
        }
    };
}

/// The arena's size: every text tensor's bytes, each rounded up to the loader's 256-byte alignment.
pub fn checkpointBytes(ck: *const core.Checkpoint) !usize {
    var total: usize = 0;
    for (ck.files.items) |file| for (file.names.keys(), file.names.values()) |name, entry| {
        if (!std.mem.startsWith(u8, name, "model.llm.")) continue;
        const padded = std.mem.alignForward(usize, entry.end - entry.begin, 256);
        total = try std.math.add(usize, total, padded);
    };
    return total;
}

fn controllers(l: anytype, ck: *core.Checkpoint, c: *const config.Text, i: u32, rank: u32, out: *[4]Controller) !void {
    const h: usize = c.hidden;
    const names = [4][]const u8{ "attn", "k", "v", "mlp" };
    for (names, out) |role, *g| {
        var prefix: [160]u8 = undefined;
        const p = try std.fmt.bufPrint(&prefix, "model.llm.layers.{d}.controllers.{s}", .{ i, role });
        var down: [224]u8 = undefined;
        var forget: [224]u8 = undefined;
        var gain: [224]u8 = undefined;
        g.* = .{
            .down = try l.put((try ck.expect(try std.fmt.bufPrint(&down, "{s}.down_weight", .{p}), .bf16, &.{ rank, 2 * h })).bytes),
            .forget = try l.put((try ck.expect(try std.fmt.bufPrint(&forget, "{s}.forget_weight", .{p}), .bf16, &.{ h, rank })).bytes),
            .gain = try l.put((try ck.expect(try std.fmt.bufPrint(&gain, "{s}.gain_weight", .{p}), .bf16, &.{ h, rank })).bytes),
        };
    }
}

/// Load every text tensor, including all four controller triplets and both cross-pass query matrices.
pub fn load(comptime Buffer: type, gpa: std.mem.Allocator, d: anytype, ck: *core.Checkpoint, c: *const config.Config, p: plan_.Plan) !Weights(Buffer) {
    if (c.text.loop == null) return error.Inkling2LoopRequired;
    var formats: [config.max_layers]modelopt.Format = undefined;
    for (p.layers[0..c.text.layers], 0..) |layer, k| {
        formats[k] = switch (layer.precision) {
            .nvfp4 => .nvfp4,
            .mxfp8 => .mxfp8,
        };
    }
    const total = try checkpointBytes(ck);
    var arena = try Buffer.alloc(d, @max(total, 256));
    var w: Weights(Buffer) = .{ .gpa = gpa, .c = c.*, .p = p };
    errdefer w.deinit();
    w.buffers.append(gpa, arena) catch |err| {
        arena.free();
        return err;
    };
    w.layers = try gpa.alloc(Layer, c.text.layers);
    w.attn_queries = try gpa.alloc(u64, c.text.layers);
    w.mlp_queries = try gpa.alloc(u64, c.text.layers);
    var l: Loader(Buffer) = .{ .w = &w, .arena = arena };
    const h: usize = c.text.hidden;
    w.embed = try l.tensor(ck, "model.llm.embed.weight", .{}, .bf16, &.{ c.text.vocab, h });
    w.embed_norm = try l.tensor(ck, "model.llm.embed_norm.weight", .{}, .bf16, &.{h});
    w.norm = try l.tensor(ck, "model.llm.norm.weight", .{}, .bf16, &.{h});
    w.unembed = try l.tensor(ck, "model.llm.unembed.weight", .{}, .bf16, &.{ c.text.vocab, h });
    w.output_query = try l.tensor(ck, "model.llm.attn_res.output_query", .{}, .f32, &.{h});
    w.cross_qm = try l.tensor(ck, "model.llm.middle_cycle_cross_pass_queries", .{}, .f32, &.{ 2 * (c.text.loop.?.core_end - c.text.loop.?.core_start), h });
    w.cross_kv = try l.tensor(ck, "model.llm.middle_cycle_cross_pass_kv_queries", .{}, .f32, &.{ 2 * (c.text.loop.?.core_end - c.text.loop.?.core_start), h });

    for (w.layers, 0..) |*layer, k| {
        const i: u32 = @intCast(k);
        const L = .{i};
        const hd = c.text.heads(i);
        const q: usize = @as(usize, hd.q) * hd.dim;
        const kv: usize = @as(usize, hd.kv) * hd.dim;
        w.mlp_queries[k] = try l.tensor(ck, "model.llm.attn_res.mlp_queries.{d}", L, .f32, &.{h});
        if (i == 0) {
            w.attn_queries[k] = 0;
        } else if (config.isGlobal(i)) {
            w.attn_queries[k] = try l.tensor(ck, "model.llm.attn_res.attn_queries.{d}", .{i - 1}, .f32, &.{ 3, h });
        } else {
            w.attn_queries[k] = try l.tensor(ck, "model.llm.attn_res.attn_queries.{d}", .{i - 1}, .f32, &.{h});
        }
        const A = struct {
            fn put(loader: *Loader(Buffer), checkpoint: *core.Checkpoint, comptime fmt: []const u8, args: anytype, dims: []const usize) !u64 {
                return loader.tensor(checkpoint, fmt, args, .bf16, dims);
            }
        };
        layer.attn = .{
            .wq = try A.put(&l, ck, "model.llm.layers.{d}.attn.wq_du.weight", L, &.{ q, h }),
            .wk = try A.put(&l, ck, "model.llm.layers.{d}.attn.wk_dv.weight", L, &.{ kv, h }),
            .wv = try A.put(&l, ck, "model.llm.layers.{d}.attn.wv_dv.weight", L, &.{ kv, h }),
            .wr = try A.put(&l, ck, "model.llm.layers.{d}.attn.wr_du.weight", L, &.{ @as(usize, hd.q) * c.text.d_rel, h }),
            .wo = try A.put(&l, ck, "model.llm.layers.{d}.attn.wo_ud.weight", L, &.{ h, q }),
            .gate = try A.put(&l, ck, "model.llm.layers.{d}.attn.gate_attn.weight", L, &.{ q, h }),
            .sink = try A.put(&l, ck, "model.llm.layers.{d}.attn.attn_sink_proj.weight", L, &.{ hd.q, h }),
            .q_scale = try A.put(&l, ck, "model.llm.layers.{d}.attn.q_l2_clip.scale", L, &.{ hd.q, hd.dim }),
            .rel_scale = try A.put(&l, ck, "model.llm.layers.{d}.attn.rel_l2_clip.scale", L, &.{ hd.q, c.text.d_rel }),
            .rel_proj = try A.put(&l, ck, "model.llm.layers.{d}.attn.rel_logits_proj.proj", L, &.{ c.text.d_rel, c.text.extent(i) }),
            .k_conv = undefined,
            .v_conv = undefined,
            .norms = undefined,
        };
        inline for ([2][]const u8{ "k", "v" }, 0..) |kind, j| {
            const conv = if (config.isGlobal(i))
                try A.put(&l, ck, "model.llm.layers.{d}.attn." ++ kind ++ "_sconv.static_w", L, &.{ config.conv_taps, kv })
            else
                try A.put(&l, ck, "model.llm.layers.{d}.attn." ++ kind ++ "_sconv.weight", L, &.{ kv, 1, config.conv_taps });
            if (j == 0) layer.attn.k_conv = conv else layer.attn.v_conv = conv;
            if (config.isGlobal(i)) {
                const dyn = try A.put(&l, ck, "model.llm.layers.{d}.attn." ++ kind ++ "_sconv.weight_proj", L, &.{ @as(usize, hd.kv) * config.conv_taps, h });
                if (j == 0) layer.attn.k_dynamic = dyn else layer.attn.v_dynamic = dyn;
            }
            const norm = if (config.isGlobal(i))
                try A.put(&l, ck, "model.llm.layers.{d}." ++ kind ++ "_input_norm.weight", L, &.{h})
            else if (j == 0)
                try A.put(&l, ck, "model.llm.layers.{d}.attn_norm.weight", L, &.{h})
            else
                layer.attn.norms[1]; // a sliding layer's k and v share one attn_norm upload
            layer.attn.norms[j + 1] = norm;
        }
        layer.attn.norms[0] = if (config.isGlobal(i))
            try A.put(&l, ck, "model.llm.layers.{d}.q_input_norm.weight", L, &.{h})
        else
            layer.attn.norms[1];
        layer.attn_conv = try A.put(&l, ck, "model.llm.layers.{d}.attn_sconv.weight", L, &.{ h, 1, config.conv_taps });
        layer.mlp_norm = try A.put(&l, ck, "model.llm.layers.{d}.mlp_norm.weight", L, &.{h});
        layer.mlp_conv = try A.put(&l, ck, "model.llm.layers.{d}.mlp_sconv.weight", L, &.{ h, 1, config.conv_taps });
        layer.router = try A.put(&l, ck, "model.llm.layers.{d}.mlp.gate.weight", L, &.{ c.text.experts, h });
        layer.router_bias = try l.tensor(ck, "model.llm.layers.{d}.mlp.gate.bias", L, .f32, &.{c.text.experts});
        var prefix: [96]u8 = undefined;
        layer.experts = try modelopt.load(&l, ck, try std.fmt.bufPrint(&prefix, "model.llm.layers.{d}.mlp.experts.", L), formats[k], c.text.experts, h, c.text.expert_inter);
        @memset(&layer.controllers, .{ .down = 0, .forget = 0, .gain = 0 });
        const lp = c.text.loop.?;
        if (i >= lp.core_start and i < lp.core_end) try controllers(&l, ck, &c.text, i, lp.gate_rank, &layer.controllers);
    }
    if (ck.unused() != 0) return error.UnusedCheckpointTensor;
    return w;
}
