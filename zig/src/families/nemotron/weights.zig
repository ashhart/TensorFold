//! Nemotron-H weights as the kernels read them: lane-tiled projections with packed scales, fp32 Mamba constants.
const std = @import("std");
const mtl = @import("metal");
const ckpt = @import("../../core/checkpoint_metal.zig");
const row = @import("../../core/row_projection.zig");
const cfg = @import("config.zig");

const Tensor = ckpt.Tensor;

/// The width the Metal kernels read every matrix at: 4-bit codes with a scale and a bias for each 64.
const kernel_bits = 4;
const kernel_group = 64;

/// A 4-bit projection in lane_qmm's 64-wide tiles: weight [N/64][K/64][64 x 8 words], SBt [K/64][N][scale, bias].
pub const Linear = struct {
    w: mtl.Buffer,
    sbt: mtl.Buffer,
    n: usize,
    k: usize,
    rows: ?row.Weights = null, // the checkpoint's row layout, read by the core's row kernels without tensor units
};

pub const Mamba = struct {
    in_proj: Linear,
    out_proj: Linear,
    conv_w: mtl.Buffer, // f32 [KC][CD]
    conv_b: mtl.Buffer, // f32 [CD]
    a_log: mtl.Buffer, // f32 [H]
    d_skip: mtl.Buffer, // f32 [H]
    dt_bias: mtl.Buffer, // f32 [H]
    norm: Tensor, // gated group norm weight [XD]
};

pub const Moe = struct {
    gate: Tensor, // bf16 [E, D]
    gate_bias: Tensor, // f32 [E]
    fc1: [3]Tensor, // weight, scales, biases: [E, W, D/8], [E, W, D/64] x2
    fc2: [3]Tensor,
    shared_up: Linear,
    shared_down: Linear,
};

pub const Attention = struct {
    qkv: Linear,
    o_proj: Linear,
};

/// The MTP head: hidden row i and token i + 1's embedding through attention and an MoE block, then the draft head.
pub const Mtp = struct {
    enorm: Tensor,
    hnorm: Tensor,
    eh: Linear, // [D, 2D]
    norm: Tensor,
    attention: Attention,
    norm2: Tensor,
    moe: Moe,
    final: Tensor,
    draft: Linear, // the LM head's rows for the draft vocabulary
    ids: mtl.Buffer, // u32: the draft vocabulary's token ids, ascending
    vocab: usize,
};

pub const Layer = union(cfg.Kind) {
    mamba: Mamba,
    moe: Moe,
    attention: Attention,
};

pub const Weights = struct {
    embed: [3]Tensor,
    norms: [cfg.max_layers]Tensor = undefined, // each layer's input norm weight
    layers: [cfg.max_layers]Layer = undefined,
    norm_f: Tensor,
    head: Linear,
    mtp: ?Mtp = null,
    owned: std.ArrayList(mtl.Buffer) = .empty,
    linears: std.StringHashMapUnmanaged(Linear) = .empty, // by the Python module path ("fused.qkv.5" for q/k/v)
    allocator: std.mem.Allocator,

    pub fn deinit(self: *Weights) void {
        for (self.owned.items) |b| b.deinit();
        self.owned.deinit(self.allocator);
        var it = self.linears.keyIterator();
        while (it.next()) |k| self.allocator.free(k.*);
        self.linears.deinit(self.allocator);
    }
};

const Builder = struct {
    allocator: std.mem.Allocator,
    device: mtl.Device,
    ck: *const ckpt.Checkpoint,
    c: *const cfg.Config,
    w: *Weights,
    jobs: std.ArrayList(Tile) = .empty,
    why: cfg.Why = .{},
    rows: bool = false, // also keep each projection's row layout (chips without tensor units)

    fn buffer(self: *Builder, bytes: usize) !mtl.Buffer {
        const b = try self.device.buffer(@max(bytes, 16), mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        try self.w.owned.append(self.allocator, b);
        return b;
    }

    fn get(self: *Builder, comptime fmt: []const u8, args: anytype) !Tensor {
        var name: [160]u8 = undefined;
        return self.ck.get(try std.fmt.bufPrint(&name, fmt, args));
    }

    /// A tensor the kernels read as stored (norms, the router), refused unless it has this dtype and shape.
    fn typed(self: *Builder, comptime fmt: []const u8, args: anytype, dtype: ckpt.DType, shape: []const usize) !Tensor {
        var name: [160]u8 = undefined;
        const full = try std.fmt.bufPrint(&name, fmt, args);
        const t = try self.ck.get(full);
        if (t.dtype == dtype and t.rank == shape.len and std.mem.eql(usize, t.shape[0..t.rank], shape)) return t;
        self.why.set("{s} is {t} {any}; the Metal kernels read it as {t} {any}", .{ full, t.dtype, t.shape[0..t.rank], dtype, shape });
        return error.UnexpectedTensor;
    }

    /// fp32 copy of a bf16 vector (or an fp32 one as is).
    fn f32vec(self: *Builder, t: Tensor) !mtl.Buffer {
        const out = try self.buffer(t.count() * 4);
        const dst = out.slice(f32, t.count());
        switch (t.dtype) {
            .bf16 => for (t.host(u16), dst) |v, *d| {
                d.* = @bitCast(@as(u32, v) << 16);
            },
            .f32 => @memcpy(dst, t.host(f32)),
            else => return error.BadDType,
        }
        return out;
    }

    /// A projection from row blocks of stacked checkpoint linears (q, k, v), tiled and packed on worker threads.
    fn linear(self: *Builder, comptime fmt: []const u8, args: anytype, parts: []const [3]Tensor) !Linear {
        var n: usize = 0;
        for (parts) |p| n += p[0].shape[0];
        const words = parts[0][0].shape[1];
        const k = words * 8;
        var lin = Linear{ .w = try self.buffer(n * words * 4), .sbt = try self.buffer(k / 64 * n * 4), .n = n, .k = k };
        var row0: usize = 0;
        for (parts) |p| {
            if (p[0].dtype != .u32 or p[0].shape[1] != words or p[1].shape[1] != k / 64) return error.BadLinear;
            try self.jobs.append(self.allocator, .{ .src = p, .dst = lin, .row0 = row0 });
            row0 += p[0].shape[0];
        }
        if (n % 64 != 0) return error.BadLinear;
        if (self.rows) lin.rows = if (parts.len == 1) rowRefs(parts[0], n, k) else try self.packRows(parts, null, n, k);
        try self.w.linears.put(self.allocator, try std.fmt.allocPrint(self.allocator, fmt, args), lin);
        return lin;
    }

    /// A projection from the rows `pick` of a checkpoint linear (the draft head's vocabulary of the LM head).
    fn picked(self: *Builder, src: [3]Tensor, pick: []const u32) !Linear {
        const words = src[0].shape[1];
        const n = pick.len;
        var lin = Linear{ .w = try self.buffer(n * words * 4), .sbt = try self.buffer(words / 8 * n * 4), .n = n, .k = words * 8 };
        if (n % 64 != 0) return error.BadLinear;
        for (pick) |r| if (r >= src[0].shape[0]) return error.BadDraftIds;
        try self.jobs.append(self.allocator, .{ .src = src, .dst = lin, .row0 = 0, .pick = pick });
        if (self.rows) lin.rows = try self.packRows(&.{src}, pick, n, words * 8);
        return lin;
    }

    /// The checkpoint's own rows: a single linear read in place.
    fn rowRefs(p: [3]Tensor, n: usize, k: usize) row.Weights {
        return .{ .w = p[0].buffer, .w_off = p[0].offset, .scales = p[1].buffer, .s_off = p[1].offset, .biases = p[2].buffer, .b_off = p[2].offset, .n = n, .k = k };
    }

    /// Stacked linears' rows (or the rows `pick` of one) copied into packed row-layout buffers.
    fn packRows(self: *Builder, parts: []const [3]Tensor, pick: ?[]const u32, n: usize, k: usize) !row.Weights {
        const words = k / 8;
        const groups = k / 64;
        const out = row.Weights{ .w = try self.buffer(n * words * 4), .scales = try self.buffer(n * groups * 2), .biases = try self.buffer(n * groups * 2), .n = n, .k = k };
        const w = out.w.slice(u32, n * words);
        const sc = out.scales.slice(u16, n * groups);
        const bi = out.biases.slice(u16, n * groups);
        var at: usize = 0;
        for (parts) |p| {
            const rows = if (pick) |x| x.len else p[0].shape[0];
            for (0..rows) |i| {
                const r = if (pick) |x| x[i] else i;
                @memcpy(w[at * words ..][0..words], p[0].host(u32)[r * words ..][0..words]);
                @memcpy(sc[at * groups ..][0..groups], p[1].host(u16)[r * groups ..][0..groups]);
                @memcpy(bi[at * groups ..][0..groups], p[2].host(u16)[r * groups ..][0..groups]);
                at += 1;
            }
        }
        return out;
    }

    fn moe(self: *Builder, comptime prefix: []const u8, args: anytype) !Moe {
        const c = self.c;
        return .{
            .gate = try self.typed(prefix ++ ".gate.weight", args, .bf16, &.{ c.experts, c.hidden }),
            .gate_bias = try self.typed(prefix ++ ".gate.e_score_correction_bias", args, .f32, &.{c.experts}),
            .fc1 = try self.quantized(prefix ++ ".switch_mlp.fc1", args, &.{ c.experts, c.expert_width, c.hidden }),
            .fc2 = try self.quantized(prefix ++ ".switch_mlp.fc2", args, &.{ c.experts, c.hidden, c.expert_width }),
            .shared_up = try self.linear(prefix ++ ".shared_experts.up_proj", args, &.{try self.quantized(prefix ++ ".shared_experts.up_proj", args, &.{ c.shared_width, c.hidden })}),
            .shared_down = try self.linear(prefix ++ ".shared_experts.down_proj", args, &.{try self.quantized(prefix ++ ".shared_experts.down_proj", args, &.{ c.hidden, c.shared_width })}),
        };
    }

    fn attention(self: *Builder, comptime prefix: []const u8, comptime fused: []const u8, args: anytype) !Attention {
        const c = self.c;
        const q = c.heads * c.head_dim;
        const kv = c.kv_heads * c.head_dim;
        return .{
            .qkv = try self.linear(fused, args, &.{
                try self.quantized(prefix ++ ".q_proj", args, &.{ q, c.hidden }),
                try self.quantized(prefix ++ ".k_proj", args, &.{ kv, c.hidden }),
                try self.quantized(prefix ++ ".v_proj", args, &.{ kv, c.hidden }),
            }),
            .o_proj = try self.linear(prefix ++ ".o_proj", args, &.{try self.quantized(prefix ++ ".o_proj", args, &.{ c.hidden, q })}),
        };
    }

    /// The MTP head (mtp.* tensors) and its draft head over `draft_ids`.
    fn mtpHead(self: *Builder, draft_ids: []const u32) !Mtp {
        const ids = try self.buffer(draft_ids.len * 4);
        @memcpy(ids.slice(u32, draft_ids.len), draft_ids);
        const d: []const usize = &.{self.c.hidden};
        return .{
            .enorm = try self.typed("mtp.layers.0.enorm.weight", .{}, .bf16, d),
            .hnorm = try self.typed("mtp.layers.0.hnorm.weight", .{}, .bf16, d),
            .eh = try self.linear("mtp.layers.0.eh_proj", .{}, &.{try self.quantized("mtp.layers.0.eh_proj", .{}, &.{ self.c.hidden, 2 * self.c.hidden })}),
            .norm = try self.typed("mtp.layers.0.norm.weight", .{}, .bf16, d),
            .attention = try self.attention("mtp.layers.0.mixer", "mtp.fused.qkv", .{}),
            .norm2 = try self.typed("mtp.layers.1.norm.weight", .{}, .bf16, d),
            .moe = try self.moe("mtp.layers.1.mixer", .{}),
            .final = try self.typed("mtp.layers.1.final_layernorm.weight", .{}, .bf16, d),
            .draft = try self.picked(try self.quantized("lm_head", .{}, &.{ self.c.vocab, self.c.hidden }), draft_ids),
            .ids = ids,
            .vocab = draft_ids.len,
        };
    }

    /// A matrix [.., n, k] as `dims` give it, refused unless u32 words and bf16 scales and biases at the kernels' width.
    fn quantized(self: *Builder, comptime fmt: []const u8, args: anytype, dims: []const usize) ![3]Tensor {
        var name: [160]u8 = undefined;
        const base = try std.fmt.bufPrint(&name, fmt, args);
        var out: [3]Tensor = undefined;
        inline for (.{ "weight", "scales", "biases" }, 0..) |part, i| {
            var full: [192]u8 = undefined;
            out[i] = try self.ck.get(try std.fmt.bufPrint(&full, "{s}.{s}", .{ base, part }));
        }
        if (out[0].dtype != .u32 or out[1].dtype != .bf16 or out[2].dtype != .bf16) {
            self.why.set("{s} has {t} weight, {t} scales and {t} biases; the Metal kernels read u32 words with bf16 scales and biases", .{ base, out[0].dtype, out[1].dtype, out[2].dtype });
            return error.UnsupportedQuantization;
        }
        try self.width(base, out, dims);
        return out;
    }

    /// Refuse a matrix whose words or scales do not hold `dims` at the kernels' width (another bits or group size).
    fn width(self: *Builder, base: []const u8, t: [3]Tensor, dims: []const usize) !void {
        const last = dims.len - 1;
        const k = dims[last];
        var rows = true;
        for (t) |x| rows = rows and x.rank == dims.len and std.mem.eql(usize, x.shape[0..last], dims[0..last]);
        const words = t[0].shape[last];
        const groups = t[1].shape[last];
        if (rows and t[2].shape[last] == groups and words * 32 == k * kernel_bits and groups * kernel_group == k) return;
        if (rows and t[2].shape[last] == groups and words * 32 % k == 0 and groups > 0 and k % groups == 0) {
            self.why.set("{s} is {d}-bit in groups of {d}; the Metal kernels read {d}-bit in groups of {d}", .{ base, words * 32 / k, k / groups, kernel_bits, kernel_group });
        } else {
            self.why.set("{s} has {any} words, {any} scales and {any} biases; the Metal kernels read {any} at {d}-bit in groups of {d}", .{ base, t[0].shape[0..t[0].rank], t[1].shape[0..t[1].rank], t[2].shape[0..t[2].rank], dims, kernel_bits, kernel_group });
        }
        return error.MixedQuantization;
    }

    /// Every tensor the kernels read, checked, with the projections queued for tiling.
    fn fill(b: *Builder, draft_ids: ?[]const u32) !void {
        const c = b.c;
        const w = b.w;
        w.norm_f = try b.typed("backbone.norm_f.weight", .{}, .bf16, &.{c.hidden});
        w.embed = try b.quantized("backbone.embeddings", .{}, &.{ c.vocab, c.hidden });
        for (0..c.layers) |i| {
            w.norms[i] = try b.typed("backbone.layers.{d}.norm.weight", .{i}, .bf16, &.{c.hidden});
            w.layers[i] = switch (c.kinds[i]) {
                .mamba => .{ .mamba = .{
                    .in_proj = try b.linear("backbone.layers.{d}.mixer.in_proj", .{i}, &.{try b.quantized("backbone.layers.{d}.mixer.in_proj", .{i}, &.{ c.projDim(), c.hidden })}),
                    .out_proj = try b.linear("backbone.layers.{d}.mixer.out_proj", .{i}, &.{try b.quantized("backbone.layers.{d}.mixer.out_proj", .{i}, &.{ c.hidden, c.inner() })}),
                    .conv_w = try convWeight(b, try b.get("backbone.layers.{d}.mixer.conv1d.weight", .{i}), c.*),
                    .conv_b = try b.f32vec(try b.get("backbone.layers.{d}.mixer.conv1d.bias", .{i})),
                    .a_log = try b.f32vec(try b.get("backbone.layers.{d}.mixer.A_log", .{i})),
                    .d_skip = try b.f32vec(try b.get("backbone.layers.{d}.mixer.D", .{i})),
                    .dt_bias = try b.f32vec(try b.get("backbone.layers.{d}.mixer.dt_bias", .{i})),
                    .norm = try b.typed("backbone.layers.{d}.mixer.norm.weight", .{i}, .bf16, &.{c.inner()}),
                } },
                .moe => .{ .moe = try b.moe("backbone.layers.{d}.mixer", .{i}) },
                .attention => .{ .attention = try b.attention("backbone.layers.{d}.mixer", "fused.qkv.{d}", .{i}) },
            };
        }
        w.head = try b.linear("lm_head", .{}, &.{try b.quantized("lm_head", .{}, &.{ c.vocab, c.hidden })});
        if (draft_ids) |ids| {
            if (b.ck.has("mtp.layers.0.eh_proj.weight")) w.mtp = try b.mtpHead(ids);
        }
    }
};

/// One source linear's rows written into the tiled destination: words [n][g][8] -> [n/64][g][n%64][8].
const Tile = struct {
    src: [3]Tensor,
    dst: Linear,
    row0: usize,
    pick: ?[]const u32 = null, // source rows of destination rows row0.. (else the source's rows in order)

    fn run(self: Tile) void {
        const words = self.dst.k / 8;
        const groups = self.dst.k / 64;
        const w = self.src[0].host(u32);
        const s = self.src[1].host(u16);
        const b = self.src[2].host(u16);
        const tiled = self.dst.w.slice(u32, self.dst.n * words);
        const sbt = self.dst.sbt.slice(u16, groups * self.dst.n * 2);
        const rows = if (self.pick) |p| p.len else self.src[0].shape[0];
        for (0..rows) |i| {
            const r = if (self.pick) |p| p[i] else i;
            const n = self.row0 + i;
            const t = n / 64;
            const c = n % 64;
            for (0..groups) |g| {
                const at = ((t * groups + g) * 64 + c) * 8;
                @memcpy(tiled[at .. at + 8], w[r * words + g * 8 .. r * words + g * 8 + 8]);
                sbt[(g * self.dst.n + n) * 2] = s[r * groups + g];
                sbt[(g * self.dst.n + n) * 2 + 1] = b[r * groups + g];
            }
        }
    }
};

fn runTiles(jobs: []const Tile) void {
    const workers = 8;
    var next = std.atomic.Value(usize).init(0);
    const W = struct {
        fn go(all: []const Tile, counter: *std.atomic.Value(usize)) void {
            while (true) {
                const i = counter.fetchAdd(1, .monotonic);
                if (i >= all.len) return;
                all[i].run();
            }
        }
    };
    var threads: [workers]?std.Thread = @splat(null);
    for (&threads) |*t| t.* = std.Thread.spawn(.{}, W.go, .{ jobs, &next }) catch null;
    W.go(jobs, &next);
    for (threads) |t| if (t) |th| th.join();
}

/// The model's weights; with `draft_ids` and the checkpoint's mtp.* tensors, the MTP head too.
pub fn load(allocator: std.mem.Allocator, device: mtl.Device, ck: *const ckpt.Checkpoint, c: cfg.Config, draft_ids: ?[]const u32) !Weights {
    var w = Weights{ .allocator = allocator, .embed = undefined, .norm_f = undefined, .head = undefined };
    errdefer w.deinit();
    var b = Builder{ .allocator = allocator, .device = device, .ck = ck, .c = &c, .w = &w, .rows = !device.tensorUnits() };
    defer b.jobs.deinit(allocator);
    b.fill(draft_ids) catch |e| {
        if (b.why.len > 0) std.log.err("{s}", .{b.why.text()});
        return e;
    };
    runTiles(b.jobs.items);
    return w;
}

/// conv1d.weight [CD, KC, 1] (bf16) as fp32 [KC][CD], the layout the conv kernel reads.
fn convWeight(b: *Builder, t: Tensor, c: cfg.Config) !mtl.Buffer {
    const cd = c.convDim();
    const kc = c.conv_kernel;
    if (t.shape[0] != cd or t.shape[1] != kc or t.dtype != .bf16) return error.BadConvWeight;
    const out = try b.buffer(kc * cd * 4);
    const dst = out.slice(f32, kc * cd);
    const src = t.host(u16);
    for (0..cd) |ch| for (0..kc) |k| {
        dst[k * cd + ch] = @bitCast(@as(u32, src[ch * kc + k]) << 16);
    };
    return out;
}

/// One tensor of a fake safetensors header.
const Fake = struct { name: []const u8, dtype: []const u8 = "BF16", shape: []const usize };

/// A checkpoint indexed from a fake header naming `tensors` back to back (the checks read headers, never data).
fn fakeCheckpoint(a: std.mem.Allocator, tensors: []const Fake) !ckpt.Checkpoint {
    var json: std.ArrayList(u8) = .empty;
    defer json.deinit(a);
    try json.appendSlice(a, "{\"__metadata__\": {\"format\": \"mlx\"}");
    var at: usize = 0;
    for (tensors) |t| {
        var bytes = (ckpt.DType.parse(t.dtype) orelse return error.UnsupportedDType).size();
        try json.print(a, ", \"{s}\": {{\"dtype\": \"{s}\", \"shape\": [", .{ t.name, t.dtype });
        for (t.shape, 0..) |d, i| {
            bytes *= d;
            try json.print(a, "{s}{d}", .{ if (i > 0) ", " else "", d });
        }
        try json.print(a, "], \"data_offsets\": [{d}, {d}]}}", .{ at, at + bytes });
        at += bytes;
    }
    try json.append(a, '}');
    var ck = ckpt.Checkpoint.init(a);
    errdefer ck.deinit();
    try ck.index(undefined, json.items, at, "");
    return ck;
}

/// A one-layer MoE model: hidden 64, vocab 128, 4 experts of width 64.
fn tinyMoe() cfg.Config {
    var c = cfg.Config{ .hidden = 64, .vocab = 128, .layers = 1, .mamba_heads = 2, .mamba_head_dim = 32, .groups = 1, .state = 16, .conv_kernel = 4, .heads = 2, .kv_heads = 1, .head_dim = 32, .experts = 4, .top_k = 2, .expert_width = 64, .shared_width = 128, .routed_scaling = 1.0, .eps = 1e-5 };
    c.kinds[0] = .moe;
    return c;
}

/// The tensors `fill` reads for tinyMoe before its first buffer, each of `swaps` in place of its namesake or added.
fn tinyTensors(swaps: []const Fake, out: *[24]Fake) []const Fake {
    const base = [_]Fake{
        .{ .name = "backbone.norm_f.weight", .shape = &.{64} },
        .{ .name = "backbone.embeddings.weight", .dtype = "U32", .shape = &.{ 128, 8 } },
        .{ .name = "backbone.embeddings.scales", .shape = &.{ 128, 1 } },
        .{ .name = "backbone.embeddings.biases", .shape = &.{ 128, 1 } },
        .{ .name = "backbone.layers.0.norm.weight", .shape = &.{64} },
        .{ .name = "backbone.layers.0.mixer.gate.weight", .shape = &.{ 4, 64 } },
        .{ .name = "backbone.layers.0.mixer.gate.e_score_correction_bias", .dtype = "F32", .shape = &.{4} },
        .{ .name = "backbone.layers.0.mixer.switch_mlp.fc1.weight", .dtype = "U32", .shape = &.{ 4, 64, 8 } },
        .{ .name = "backbone.layers.0.mixer.switch_mlp.fc1.scales", .shape = &.{ 4, 64, 1 } },
        .{ .name = "backbone.layers.0.mixer.switch_mlp.fc1.biases", .shape = &.{ 4, 64, 1 } },
        .{ .name = "backbone.layers.0.mixer.switch_mlp.fc2.weight", .dtype = "U32", .shape = &.{ 4, 64, 8 } },
        .{ .name = "backbone.layers.0.mixer.switch_mlp.fc2.scales", .shape = &.{ 4, 64, 1 } },
        .{ .name = "backbone.layers.0.mixer.switch_mlp.fc2.biases", .shape = &.{ 4, 64, 1 } },
    };
    @memcpy(out[0..base.len], &base);
    var n: usize = base.len;
    for (swaps) |swap| {
        const at = for (out[0..n], 0..) |t, i| {
            if (std.mem.eql(u8, t.name, swap.name)) break i;
        } else blk: {
            n += 1;
            break :blk n - 1;
        };
        out[at] = swap;
    }
    return out[0..n];
}

/// fill() on tinyMoe with `swaps` must fail with `err`, its reason naming every one of `words`.
fn expectRefused(swaps: []const Fake, err: anyerror, words: []const []const u8) !void {
    const a = std.testing.allocator;
    var buf: [24]Fake = undefined;
    var ck = try fakeCheckpoint(a, tinyTensors(swaps, &buf));
    defer ck.deinit();
    const c = tinyMoe();
    var w = Weights{ .allocator = a, .embed = undefined, .norm_f = undefined, .head = undefined };
    defer w.deinit();
    var b = Builder{ .allocator = a, .device = undefined, .ck = &ck, .c = &c, .w = &w };
    defer b.jobs.deinit(a);
    try std.testing.expectError(err, b.fill(null));
    for (words) |word| if (std.mem.indexOf(u8, b.why.text(), word) == null) {
        std.debug.print("refusal \"{s}\" does not name \"{s}\"\n", .{ b.why.text(), word });
        return error.TestUnexpectedResult;
    };
}

test "the Metal loader refuses scale, bias and norm dtypes its kernels do not read" {
    const fc2 = "backbone.layers.0.mixer.switch_mlp.fc2";
    try expectRefused(&.{.{ .name = "backbone.embeddings.scales", .dtype = "F16", .shape = &.{ 128, 1 } }}, error.UnsupportedQuantization, &.{ "backbone.embeddings", "f16 scales" });
    try expectRefused(&.{.{ .name = "backbone.layers.0.mixer.switch_mlp.fc1.biases", .dtype = "F32", .shape = &.{ 4, 64, 1 } }}, error.UnsupportedQuantization, &.{ "switch_mlp.fc1", "f32 biases" });
    try expectRefused(&.{.{ .name = fc2 ++ ".scales", .dtype = "F16", .shape = &.{ 4, 64, 1 } }}, error.UnsupportedQuantization, &.{ fc2, "f16 scales" });
    try expectRefused(&.{.{ .name = fc2 ++ ".weight", .dtype = "I32", .shape = &.{ 4, 64, 8 } }}, error.UnsupportedQuantization, &.{ fc2, "i32 weight" });
    try expectRefused(&.{.{ .name = "backbone.norm_f.weight", .dtype = "F16", .shape = &.{64} }}, error.UnexpectedTensor, &.{ "backbone.norm_f.weight", "f16" });
    try expectRefused(&.{.{ .name = "backbone.layers.0.norm.weight", .dtype = "F32", .shape = &.{64} }}, error.UnexpectedTensor, &.{ "layers.0.norm.weight", "f32" });
    try expectRefused(&.{.{ .name = "backbone.layers.0.mixer.gate.weight", .dtype = "F16", .shape = &.{ 4, 64 } }}, error.UnexpectedTensor, &.{ "gate.weight", "f16" });
    try expectRefused(&.{.{ .name = "backbone.layers.0.mixer.gate.e_score_correction_bias", .shape = &.{4} }}, error.UnexpectedTensor, &.{ "e_score_correction_bias", "bf16" });
}

test "the Metal loader refuses matrices at another width than its kernels read" {
    const fc1 = "backbone.layers.0.mixer.switch_mlp.fc1";
    const up = "backbone.layers.0.mixer.shared_experts.up_proj";
    try expectRefused(&.{.{ .name = "backbone.embeddings.weight", .dtype = "U32", .shape = &.{ 128, 12 } }}, error.MixedQuantization, &.{ "backbone.embeddings", "6-bit in groups of 64" });
    try expectRefused(&.{
        .{ .name = fc1 ++ ".weight", .dtype = "U32", .shape = &.{ 4, 64, 16 } },
        .{ .name = fc1 ++ ".scales", .shape = &.{ 4, 64, 2 } },
        .{ .name = fc1 ++ ".biases", .shape = &.{ 4, 64, 2 } },
    }, error.MixedQuantization, &.{ fc1, "8-bit in groups of 32" });
    try expectRefused(&.{.{ .name = fc1 ++ ".biases", .shape = &.{ 4, 64, 2 } }}, error.MixedQuantization, &.{ fc1, "words" });
    try expectRefused(&.{.{ .name = "backbone.layers.0.mixer.switch_mlp.fc2.weight", .dtype = "U32", .shape = &.{ 3, 64, 8 } }}, error.MixedQuantization, &.{ "switch_mlp.fc2", "{ 4, 64, 64 }" });
    try expectRefused(&.{
        .{ .name = up ++ ".weight", .dtype = "U32", .shape = &.{ 128, 16 } },
        .{ .name = up ++ ".scales", .shape = &.{ 128, 2 } },
        .{ .name = up ++ ".biases", .shape = &.{ 128, 2 } },
    }, error.MixedQuantization, &.{ up, "8-bit in groups of 32" });
}
