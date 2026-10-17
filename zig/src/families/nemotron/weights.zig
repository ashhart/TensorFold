//! Nemotron-H weights as the kernels read them: 4/64 lane-tiled, other widths in rows, fp32 Mamba constants.
const std = @import("std");
const mtl = @import("metal");
const ckpt = @import("../../core/checkpoint_metal.zig");
const row = @import("../../core/row_projection.zig");
const host4 = @import("../../core/affine4_host.zig");
const cfg = @import("config.zig");

const Tensor = ckpt.Tensor;
const Width = cfg.Width;

/// The group the lane tiles and the pocket read: 4-bit codes with a scale and a bias for each 64.
const kernel_group = 64;

/// A projection: lane_qmm's tiles at 4/64, the checkpoint's rows for the row kernels, or a mixed stack's members.
pub const Linear = struct {
    tiles: ?Tiles = null, // 4/64 only: weight [N/64][K/64][64 x 8 words], SBt [K/64][N][scale, bias]
    n: usize,
    k: usize,
    rows: ?row.Weights = null, // the checkpoint's row layout: chips without tensor units, and every width but 4/64
    stack: [3]Member = undefined, // a fused stack not all 4/64: each member's rows into its own output columns
    stacked: usize = 0, // members in `stack` (0: one matrix)
};

/// lane_qmm's 64-wide tiles of a 4/64 projection.
pub const Tiles = struct { w: mtl.Buffer, sbt: mtl.Buffer };

/// One matrix of a fused stack, written from column `first` of the stack's output rows.
pub const Member = struct { first: usize, rows: row.Weights };

/// A checkpoint matrix: its words, scales and biases, and the width they hold.
pub const Matrix = struct { t: [3]Tensor, w: Width };

/// A learned change after a projection, y += scale (x a^T) b over blocks open on the row: a [rank, in], b [rank, out].
pub const Adapter = struct { a: mtl.Buffer, b: mtl.Buffer, tau: mtl.Buffer, xa: mtl.Buffer, xn: mtl.Buffer, gates: mtl.Buffer, need: mtl.Buffer, latch: mtl.Buffer, score: mtl.Buffer, whole: bool, in: usize, out: usize, rank: usize, scale: f32, unit: f32 };

pub const Mamba = struct {
    in_proj: Linear,
    out_proj: Linear,
    adapter: ?Adapter = null, // after out_proj
    slide: ?mtl.Buffer = null, // bf16 [D, inner]: out_proj's weight read from the shards minus its codes, added after it
    conv_w: mtl.Buffer, // f32 [KC][CD]
    conv_b: mtl.Buffer, // f32 [CD]
    a_log: mtl.Buffer, // f32 [H]
    d_skip: mtl.Buffer, // f32 [H]
    dt_bias: mtl.Buffer, // f32 [H]
    norm: Tensor, // gated group norm weight [XD]
};

/// Learned neurons beside a shared expert's own, as a wider shared expert computes them: y += relu(x up^T)^2 down.
pub const Pocket = struct {
    up: mtl.Buffer, // f32 [neurons, D]: each neuron's input weights
    down: mtl.Buffer, // f32 [neurons, D]: each neuron's output
    z: mtl.Buffer, // f32 [rows, neurons]: the forward's scratch
    zn: mtl.Buffer, // f32 [rows]: the input kernel's row lengths, unread
    used: usize, // neurons in use, the first ones
};

pub const Moe = struct {
    gate: Tensor, // bf16 [E, D]
    gate_bias: Tensor, // f32 [E]
    fc1: [3]Tensor, // weight, scales, biases: [E, W, D * bits / 32], [E, W, D / group] x2
    fc2: [3]Tensor,
    routed: ?[2]row.Experts = null, // fc1 and fc2 on the core's indexed row kernels, when either is not 4/64
    shared_up: Linear,
    shared_down: Linear,
    shared_raw: [2][3]Tensor, // up and down as the kernels read them: the shared expert's own neurons, learned ones left out
    adapter: ?Adapter = null, // after shared_down
    slide: ?mtl.Buffer = null, // bf16 [D, W]: shared_down's weight read from the shards minus its codes, added after it
    pocket: ?Pocket = null, // learned neurons, after shared_down
};

pub const Attention = struct {
    qkv: Linear,
    o_proj: Linear,
    adapter: ?Adapter = null, // after o_proj
    slide: ?mtl.Buffer = null, // bf16 [D, H*Dh]: o_proj's weight read from the shards minus its codes, added after it
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
    embed_width: Width = .{},
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

/// Every tensor the kernels read, checked and laid out for them.
pub const Builder = struct {
    allocator: std.mem.Allocator,
    device: mtl.Device,
    ck: *const ckpt.Checkpoint,
    c: *const cfg.Config,
    w: *Weights,
    jobs: std.ArrayList(Tile) = .empty,
    why: cfg.Why = .{},
    rows: bool = false, // also keep each 4/64 projection's row layout (chips without tensor units)

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

    /// A projection from row blocks of stacked checkpoint matrices (q, k, v): 4/64 lane-tiled, any other width in rows.
    fn linear(self: *Builder, comptime fmt: []const u8, args: anytype, parts: []const Matrix) !Linear {
        for (parts) |p| if (!p.w.four()) return self.rowLinear(fmt, args, parts);
        var n: usize = 0;
        for (parts) |p| n += p.t[0].shape[0];
        const words = parts[0].t[0].shape[1];
        const k = words * 8;
        var lin = Linear{ .tiles = .{ .w = try self.buffer(n * words * 4), .sbt = try self.buffer(k / 64 * n * 4) }, .n = n, .k = k };
        var row0: usize = 0;
        for (parts) |p| {
            if (p.t[0].dtype != .u32 or p.t[0].shape[1] != words or p.t[1].shape[1] != k / 64) return error.BadLinear;
            try self.jobs.append(self.allocator, .{ .src = p.t, .dst = lin, .row0 = row0 });
            row0 += p.t[0].shape[0];
        }
        if (n % 64 != 0) return error.BadLinear;
        if (self.rows) lin.rows = if (parts.len == 1) rowRefs(parts[0], n, k) else try self.packRows(parts, null, n, k);
        try self.w.linears.put(self.allocator, try std.fmt.allocPrint(self.allocator, fmt, args), lin);
        return lin;
    }

    /// A projection the row kernels read as the checkpoint stores it: one matrix in place, or each member of a stack.
    fn rowLinear(self: *Builder, comptime fmt: []const u8, args: anytype, parts: []const Matrix) !Linear {
        var lin = Linear{ .n = 0, .k = parts[0].t[1].shape[1] * parts[0].w.group };
        for (parts, 0..) |p, i| {
            const r = rowRefs(p, p.t[0].shape[0], lin.k);
            r.validate() catch |e| {
                var name: [160]u8 = undefined;
                self.why.set("{s}: the row kernels read no {d} x {d} matrix at {d}-bit in groups of {d}", .{ std.fmt.bufPrint(&name, fmt, args) catch fmt, r.n, r.k, r.bits, r.group });
                return e;
            };
            if (parts.len > 1) lin.stack[i] = .{ .first = lin.n, .rows = r } else lin.rows = r;
            lin.n += r.n;
        }
        lin.stacked = if (parts.len > 1) parts.len else 0;
        try self.w.linears.put(self.allocator, try std.fmt.allocPrint(self.allocator, fmt, args), lin);
        return lin;
    }

    /// A stack of routed experts [E, N, K] read in place by the core's indexed row kernels (one x row a slot).
    fn experts(self: *Builder, comptime fmt: []const u8, args: anytype, m: Matrix) !row.Experts {
        const x = row.Experts{ .one = rowRefs(m, m.t[0].shape[1], m.t[1].shape[2] * m.w.group), .experts = m.t[0].shape[0] };
        x.validate() catch |e| {
            var name: [160]u8 = undefined;
            self.why.set("{s}: the row kernels read no {d} experts of {d} x {d} at {d}-bit in groups of {d}", .{ std.fmt.bufPrint(&name, fmt, args) catch fmt, x.experts, x.one.n, x.one.k, x.one.bits, x.one.group });
            return e;
        };
        return x;
    }

    /// A projection from the rows `pick` of a checkpoint matrix (the draft head's vocabulary of the LM head).
    fn picked(self: *Builder, src: Matrix, pick: []const u32) !Linear {
        const n = pick.len;
        const k = src.t[1].shape[1] * src.w.group;
        if (n % 64 != 0) return error.BadLinear;
        for (pick) |r| if (r >= src.t[0].shape[0]) return error.BadDraftIds;
        if (!src.w.four()) return .{ .n = n, .k = k, .rows = try self.packRows(&.{src}, pick, n, k) };
        const words = k / 8;
        var lin = Linear{ .tiles = .{ .w = try self.buffer(n * words * 4), .sbt = try self.buffer(words / 8 * n * 4) }, .n = n, .k = k };
        try self.jobs.append(self.allocator, .{ .src = src.t, .dst = lin, .row0 = 0, .pick = pick });
        if (self.rows) lin.rows = try self.packRows(&.{src}, pick, n, k);
        return lin;
    }

    /// The checkpoint's own rows of a matrix [n, k] at its width, read in place.
    fn rowRefs(p: Matrix, n: usize, k: usize) row.Weights {
        const t = p.t;
        return .{ .w = t[0].buffer, .w_off = t[0].offset, .scales = t[1].buffer, .s_off = t[1].offset, .biases = t[2].buffer, .b_off = t[2].offset, .n = n, .k = k, .group = p.w.group, .bits = p.w.bits };
    }

    /// Stacked matrices' rows of one width (or the rows `pick` of one) copied into packed row-layout buffers.
    fn packRows(self: *Builder, parts: []const Matrix, pick: ?[]const u32, n: usize, k: usize) !row.Weights {
        const pw = parts[0].w;
        const words = k * pw.bits / 32;
        const groups = k / pw.group;
        const out = row.Weights{ .w = try self.buffer(n * words * 4), .scales = try self.buffer(n * groups * 2), .biases = try self.buffer(n * groups * 2), .n = n, .k = k, .group = pw.group, .bits = pw.bits };
        const w = out.w.slice(u32, n * words);
        const sc = out.scales.slice(u16, n * groups);
        const bi = out.biases.slice(u16, n * groups);
        var at: usize = 0;
        for (parts) |p| {
            const rows = if (pick) |x| x.len else p.t[0].shape[0];
            for (0..rows) |i| {
                const r = if (pick) |x| x[i] else i;
                @memcpy(w[at * words ..][0..words], p.t[0].host(u32)[r * words ..][0..words]);
                @memcpy(sc[at * groups ..][0..groups], p.t[1].host(u16)[r * groups ..][0..groups]);
                @memcpy(bi[at * groups ..][0..groups], p.t[2].host(u16)[r * groups ..][0..groups]);
                at += 1;
            }
        }
        return out;
    }

    /// An MoE block at `ws` (fc1, fc2, shared up, shared down; members(.moe) order).
    fn moe(self: *Builder, comptime prefix: []const u8, args: anytype, ws: [4]Width) !Moe {
        const c = self.c;
        const gate = try self.typed(prefix ++ ".gate.weight", args, .bf16, &.{ c.experts, c.hidden });
        const gate_bias = try self.typed(prefix ++ ".gate.e_score_correction_bias", args, .f32, &.{c.experts});
        const fc1 = try self.quantized(prefix ++ ".switch_mlp.fc1", args, &.{ c.experts, c.expert_width, c.hidden }, ws[0]);
        const fc2 = try self.quantized(prefix ++ ".switch_mlp.fc2", args, &.{ c.experts, c.hidden, c.expert_width }, ws[1]);
        const full = c.shared_width + c.pocket;
        const up = try self.quantized(prefix ++ ".shared_experts.up_proj", args, &.{ full, c.hidden }, ws[2]);
        const down = try self.quantized(prefix ++ ".shared_experts.down_proj", args, &.{ c.hidden, full }, ws[3]);
        const four = up.w.four() and down.w.four();
        if (c.pocket > 0 and !four) {
            var name: [160]u8 = undefined;
            self.why.set("{s} holds learned neurons beside a shared expert not at 4-bit in groups of 64, which the pocket reads", .{std.fmt.bufPrint(&name, prefix, args) catch prefix});
            return error.UnsupportedQuantization;
        }
        const own_up = Matrix{ .t = firstRows(up.t, c.shared_width), .w = up.w };
        const own_down = Matrix{ .t = if (c.pocket > 0) try self.firstColumns(down.t, c.shared_width) else down.t, .w = down.w };
        // the lane up writes the input sums the lane down reads: both on lane tiles, or both in rows
        const routed = fc1.w.four() and fc2.w.four();
        return .{
            .gate = gate,
            .gate_bias = gate_bias,
            .fc1 = fc1.t,
            .fc2 = fc2.t,
            .routed = if (routed) null else .{ try self.experts(prefix ++ ".switch_mlp.fc1", args, fc1), try self.experts(prefix ++ ".switch_mlp.fc2", args, fc2) },
            .shared_up = if (four) try self.linear(prefix ++ ".shared_experts.up_proj", args, &.{own_up}) else try self.rowLinear(prefix ++ ".shared_experts.up_proj", args, &.{own_up}),
            .shared_down = if (four) try self.linear(prefix ++ ".shared_experts.down_proj", args, &.{own_down}) else try self.rowLinear(prefix ++ ".shared_experts.down_proj", args, &.{own_down}),
            .shared_raw = .{ own_up.t, own_down.t },
            .pocket = if (c.pocket > 0) try self.pocketOf(up.t, down.t) else null,
        };
    }

    /// A shared expert's learned neurons (its last up_proj rows, down_proj columns) as f32, in use to the last nonzero.
    fn pocketOf(self: *Builder, up: [3]Tensor, down: [3]Tensor) !Pocket {
        const c = self.c;
        const d = c.hidden;
        const n = c.pocket;
        const p: Pocket = .{ .up = try self.buffer(n * d * 4), .down = try self.buffer(n * d * 4), .z = try self.buffer(pocket_rows * n * 4), .zn = try self.buffer(pocket_rows * 4), .used = 0 };
        const ups = p.up.slice(f32, n * d);
        const ug = d / kernel_group;
        const uw = up[0].host(u32);
        for (0..n) |k| {
            const r = c.shared_width + k;
            host4.dequantize(uw[r * d / 8 ..][0 .. d / 8], up[1].host(u16)[r * ug ..][0..ug], up[2].host(u16)[r * ug ..][0..ug], ups[k * d ..][0..d]);
        }
        const downs = p.down.slice(f32, n * d);
        const full = c.shared_width + n;
        const dg = full / kernel_group;
        const outs = try self.allocator.alloc(f32, n);
        defer self.allocator.free(outs);
        for (0..d) |o| {
            const w = down[0].host(u32)[o * full / 8 ..][c.shared_width / 8 ..][0 .. n / 8];
            host4.dequantize(w, down[1].host(u16)[o * dg ..][c.shared_width / kernel_group ..][0 .. n / kernel_group], down[2].host(u16)[o * dg ..][c.shared_width / kernel_group ..][0 .. n / kernel_group], outs);
            for (outs, 0..) |v, k| downs[k * d + o] = v;
        }
        var used: usize = 0;
        for (0..n) |k| if (!std.mem.allEqual(f32, ups[k * d ..][0..d], 0)) {
            used = k + 1;
        };
        var out = p;
        out.used = used;
        return out;
    }

    /// The first `k` columns of a quantized matrix [rows, wider] in buffers of their own (itself when `k` wide).
    fn firstColumns(self: *Builder, t: [3]Tensor, k: usize) ![3]Tensor {
        const wide = t[0].shape[1] * 8;
        if (wide == k) return t;
        const rows = t[0].shape[0];
        var out = t;
        const per = [3]usize{ k / 8 * 4, k / kernel_group * 2, k / kernel_group * 2 };
        const was = [3]usize{ wide / 8 * 4, wide / kernel_group * 2, wide / kernel_group * 2 };
        for (&out, t, per, was) |*o, x, p, w| {
            const b = try self.buffer(rows * p);
            const src: [*]const u8 = x.buffer.contents() + x.offset;
            for (0..rows) |r| @memcpy(b.contents()[r * p ..][0..p], src[r * w ..][0..p]);
            o.* = .{ .buffer = b, .offset = 0, .bytes = rows * p, .dtype = x.dtype, .shape = .{ rows, p / x.dtype.size(), 1, 1, 1 }, .rank = 2 };
        }
        return out;
    }

    /// An attention block at `ws` (q, k, v, o; members(.attention) order).
    fn attention(self: *Builder, comptime prefix: []const u8, comptime fused: []const u8, args: anytype, ws: [4]Width) !Attention {
        const c = self.c;
        const q = c.heads * c.head_dim;
        const kv = c.kv_heads * c.head_dim;
        return .{
            .qkv = try self.linear(fused, args, &.{
                try self.quantized(prefix ++ ".q_proj", args, &.{ q, c.hidden }, ws[0]),
                try self.quantized(prefix ++ ".k_proj", args, &.{ kv, c.hidden }, ws[1]),
                try self.quantized(prefix ++ ".v_proj", args, &.{ kv, c.hidden }, ws[2]),
            }),
            .o_proj = try self.linear(prefix ++ ".o_proj", args, &.{try self.quantized(prefix ++ ".o_proj", args, &.{ c.hidden, q }, ws[3])}),
        };
    }

    /// The MTP head (mtp.* tensors, our own file at 4/64) and its draft head over `draft_ids` (the LM head's width).
    fn mtpHead(self: *Builder, draft_ids: []const u32) !Mtp {
        const ids = try self.buffer(draft_ids.len * 4);
        @memcpy(ids.slice(u32, draft_ids.len), draft_ids);
        const d: []const usize = &.{self.c.hidden};
        const four: [4]Width = @splat(.{});
        return .{
            .enorm = try self.typed("mtp.layers.0.enorm.weight", .{}, .bf16, d),
            .hnorm = try self.typed("mtp.layers.0.hnorm.weight", .{}, .bf16, d),
            .eh = try self.linear("mtp.layers.0.eh_proj", .{}, &.{try self.quantized("mtp.layers.0.eh_proj", .{}, &.{ self.c.hidden, 2 * self.c.hidden }, .{})}),
            .norm = try self.typed("mtp.layers.0.norm.weight", .{}, .bf16, d),
            .attention = try self.attention("mtp.layers.0.mixer", "mtp.fused.qkv", .{}, four),
            .norm2 = try self.typed("mtp.layers.1.norm.weight", .{}, .bf16, d),
            .moe = try self.moe("mtp.layers.1.mixer", .{}, four),
            .final = try self.typed("mtp.layers.1.final_layernorm.weight", .{}, .bf16, d),
            .draft = try self.picked(try self.quantized("lm_head", .{}, &.{ self.c.vocab, self.c.hidden }, self.c.head_width), draft_ids),
            .ids = ids,
            .vocab = draft_ids.len,
        };
    }

    /// A matrix [.., n, k] as `dims` give it, refused unless u32 words, bf16 scales and biases at its declared width.
    fn quantized(self: *Builder, comptime fmt: []const u8, args: anytype, dims: []const usize, w: Width) !Matrix {
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
        try self.width(base, out, dims, w);
        return .{ .t = out, .w = w };
    }

    /// Refuse a matrix whose words or scales do not hold `dims` at width `w`: stored at another width than declared.
    fn width(self: *Builder, base: []const u8, t: [3]Tensor, dims: []const usize, w: Width) !void {
        const last = dims.len - 1;
        const k = dims[last];
        var rows = true;
        for (t) |x| rows = rows and x.rank == dims.len and std.mem.eql(usize, x.shape[0..last], dims[0..last]);
        const words = t[0].shape[last];
        const groups = t[1].shape[last];
        if (rows and t[2].shape[last] == groups and words * 32 == k * w.bits and groups * w.group == k) return;
        if (rows and t[2].shape[last] == groups and words * 32 % k == 0 and groups > 0 and k % groups == 0) {
            self.why.set("{s} is stored at {d}-bit in groups of {d}; config.json declares {d}-bit in groups of {d}", .{ base, words * 32 / k, k / groups, w.bits, w.group });
        } else {
            self.why.set("{s} has {any} words, {any} scales and {any} biases; config.json declares {any} at {d}-bit in groups of {d}", .{ base, t[0].shape[0..t[0].rank], t[1].shape[0..t[1].rank], t[2].shape[0..t[2].rank], dims, w.bits, w.group });
        }
        return error.UnexpectedTensor;
    }

    /// Every tensor the kernels read, checked at its declared width, with the 4/64 projections queued for tiling.
    pub fn fill(b: *Builder, draft_ids: ?[]const u32) !void {
        const c = b.c;
        const w = b.w;
        w.norm_f = try b.typed("backbone.norm_f.weight", .{}, .bf16, &.{c.hidden});
        const embed = try b.quantized("backbone.embeddings", .{}, &.{ c.vocab, c.hidden }, c.embed_width);
        w.embed = embed.t;
        w.embed_width = embed.w;
        for (0..c.layers) |i| {
            w.norms[i] = try b.typed("backbone.layers.{d}.norm.weight", .{i}, .bf16, &.{c.hidden});
            const ws = c.widths[i];
            w.layers[i] = switch (c.kinds[i]) {
                .mamba => .{ .mamba = .{
                    .in_proj = try b.linear("backbone.layers.{d}.mixer.in_proj", .{i}, &.{try b.quantized("backbone.layers.{d}.mixer.in_proj", .{i}, &.{ c.projDim(), c.hidden }, ws[0])}),
                    .out_proj = try b.linear("backbone.layers.{d}.mixer.out_proj", .{i}, &.{try b.quantized("backbone.layers.{d}.mixer.out_proj", .{i}, &.{ c.hidden, c.inner() }, ws[1])}),
                    .conv_w = try convWeight(b, try b.get("backbone.layers.{d}.mixer.conv1d.weight", .{i}), c.*),
                    .conv_b = try b.f32vec(try b.get("backbone.layers.{d}.mixer.conv1d.bias", .{i})),
                    .a_log = try b.f32vec(try b.get("backbone.layers.{d}.mixer.A_log", .{i})),
                    .d_skip = try b.f32vec(try b.get("backbone.layers.{d}.mixer.D", .{i})),
                    .dt_bias = try b.f32vec(try b.get("backbone.layers.{d}.mixer.dt_bias", .{i})),
                    .norm = try b.typed("backbone.layers.{d}.mixer.norm.weight", .{i}, .bf16, &.{c.inner()}),
                } },
                .moe => .{ .moe = try b.moe("backbone.layers.{d}.mixer", .{i}, ws) },
                .attention => .{ .attention = try b.attention("backbone.layers.{d}.mixer", "fused.qkv.{d}", .{i}, ws) },
            };
        }
        w.head = try b.linear("lm_head", .{}, &.{try b.quantized("lm_head", .{}, &.{ c.vocab, c.hidden }, c.head_width)});
        if (draft_ids) |ids| {
            if (b.ck.has("mtp.layers.0.eh_proj.weight")) w.mtp = try b.mtpHead(ids);
        }
    }
};

/// The first `n` rows of a quantized matrix, in place.
fn firstRows(t: [3]Tensor, n: usize) [3]Tensor {
    var out = t;
    for (&out) |*x| {
        x.bytes = x.bytes / x.shape[0] * n;
        x.shape[0] = n;
    }
    return out;
}

/// Rows the pocket's forward scratch holds: the largest prefill step and then some.
pub const pocket_rows = 8192 + 256;

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
        const tiled = self.dst.tiles.?.w.slice(u32, self.dst.n * words);
        const sbt = self.dst.tiles.?.sbt.slice(u16, groups * self.dst.n * 2);
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

/// A projection's codes ([n, k/8], row layout) rewritten into its tiles and any row copy; scales and biases kept.
pub fn retile(lin: Linear, words: []const u32) void {
    const per = lin.k / 8;
    const groups = lin.k / 64;
    const tiled = lin.tiles.?.w.slice(u32, lin.n * per);
    for (0..lin.n) |n| for (0..groups) |g| {
        const at = ((n / 64 * groups + g) * 64 + n % 64) * 8;
        @memcpy(tiled[at..][0..8], words[n * per + g * 8 ..][0..8]);
    };
    if (lin.rows) |r| {
        const own: [*]u32 = @ptrCast(@alignCast(r.w.contents() + r.w_off));
        if (own != words.ptr) @memcpy(own[0 .. lin.n * per], words[0 .. lin.n * per]);
    }
}

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

test {
    _ = @import("weights_test.zig");
}
