//! Nemotron-H weights as the kernels read them: lane-tiled projections with packed scales, fp32 Mamba constants.
const std = @import("std");
const mtl = @import("metal");
const ckpt = @import("../../core/checkpoint_metal.zig");
const cfg = @import("config.zig");

const Tensor = ckpt.Tensor;

/// A 4-bit projection in lane_qmm's 64-wide tiles: weight [N/64][K/64][64 x 8 words], SBt [K/64][N][scale, bias].
pub const Linear = struct {
    w: mtl.Buffer,
    sbt: mtl.Buffer,
    n: usize,
    k: usize,
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
    w: *Weights,
    jobs: std.ArrayList(Tile) = .empty,

    fn buffer(self: *Builder, bytes: usize) !mtl.Buffer {
        const b = try self.device.buffer(@max(bytes, 16), mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        try self.w.owned.append(self.allocator, b);
        return b;
    }

    fn get(self: *Builder, comptime fmt: []const u8, args: anytype) !Tensor {
        var name: [160]u8 = undefined;
        return self.ck.get(try std.fmt.bufPrint(&name, fmt, args));
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
        const lin = Linear{ .w = try self.buffer(n * words * 4), .sbt = try self.buffer(k / 64 * n * 4), .n = n, .k = k };
        var row: usize = 0;
        for (parts) |p| {
            if (p[0].dtype != .u32 or p[0].shape[1] != words or p[1].shape[1] != k / 64) return error.BadLinear;
            try self.jobs.append(self.allocator, .{ .src = p, .dst = lin, .row0 = row });
            row += p[0].shape[0];
        }
        if (n % 64 != 0) return error.BadLinear;
        try self.w.linears.put(self.allocator, try std.fmt.allocPrint(self.allocator, fmt, args), lin);
        return lin;
    }

    /// A projection from the rows `pick` of a checkpoint linear (the draft head's vocabulary of the LM head).
    fn picked(self: *Builder, src: [3]Tensor, pick: []const u32) !Linear {
        const words = src[0].shape[1];
        const n = pick.len;
        const lin = Linear{ .w = try self.buffer(n * words * 4), .sbt = try self.buffer(words / 8 * n * 4), .n = n, .k = words * 8 };
        if (n % 64 != 0) return error.BadLinear;
        for (pick) |r| if (r >= src[0].shape[0]) return error.BadDraftIds;
        try self.jobs.append(self.allocator, .{ .src = src, .dst = lin, .row0 = 0, .pick = pick });
        return lin;
    }

    fn moe(self: *Builder, comptime prefix: []const u8, args: anytype) !Moe {
        return .{
            .gate = try self.get(prefix ++ ".gate.weight", args),
            .gate_bias = try self.get(prefix ++ ".gate.e_score_correction_bias", args),
            .fc1 = try self.quantized(prefix ++ ".switch_mlp.fc1", args),
            .fc2 = try self.quantized(prefix ++ ".switch_mlp.fc2", args),
            .shared_up = try self.linear(prefix ++ ".shared_experts.up_proj", args, &.{try self.quantized(prefix ++ ".shared_experts.up_proj", args)}),
            .shared_down = try self.linear(prefix ++ ".shared_experts.down_proj", args, &.{try self.quantized(prefix ++ ".shared_experts.down_proj", args)}),
        };
    }

    fn attention(self: *Builder, comptime prefix: []const u8, comptime fused: []const u8, args: anytype) !Attention {
        return .{
            .qkv = try self.linear(fused, args, &.{
                try self.quantized(prefix ++ ".q_proj", args),
                try self.quantized(prefix ++ ".k_proj", args),
                try self.quantized(prefix ++ ".v_proj", args),
            }),
            .o_proj = try self.linear(prefix ++ ".o_proj", args, &.{try self.quantized(prefix ++ ".o_proj", args)}),
        };
    }

    /// The MTP head (mtp.* tensors) and its draft head over `draft_ids`.
    fn mtpHead(self: *Builder, draft_ids: []const u32) !Mtp {
        const ids = try self.buffer(draft_ids.len * 4);
        @memcpy(ids.slice(u32, draft_ids.len), draft_ids);
        return .{
            .enorm = try self.get("mtp.layers.0.enorm.weight", .{}),
            .hnorm = try self.get("mtp.layers.0.hnorm.weight", .{}),
            .eh = try self.linear("mtp.layers.0.eh_proj", .{}, &.{try self.quantized("mtp.layers.0.eh_proj", .{})}),
            .norm = try self.get("mtp.layers.0.norm.weight", .{}),
            .attention = try self.attention("mtp.layers.0.mixer", "mtp.fused.qkv", .{}),
            .norm2 = try self.get("mtp.layers.1.norm.weight", .{}),
            .moe = try self.moe("mtp.layers.1.mixer", .{}),
            .final = try self.get("mtp.layers.1.final_layernorm.weight", .{}),
            .draft = try self.picked(try self.quantized("lm_head", .{}), draft_ids),
            .ids = ids,
            .vocab = draft_ids.len,
        };
    }

    fn quantized(self: *Builder, comptime fmt: []const u8, args: anytype) ![3]Tensor {
        var name: [160]u8 = undefined;
        const base = try std.fmt.bufPrint(&name, fmt, args);
        var out: [3]Tensor = undefined;
        inline for (.{ "weight", "scales", "biases" }, 0..) |part, i| {
            var full: [192]u8 = undefined;
            out[i] = try self.ck.get(try std.fmt.bufPrint(&full, "{s}.{s}", .{ base, part }));
        }
        return out;
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
    var w = Weights{
        .allocator = allocator,
        .embed = undefined,
        .norm_f = try ck.get("backbone.norm_f.weight"),
        .head = undefined,
    };
    errdefer w.deinit();
    var b = Builder{ .allocator = allocator, .device = device, .ck = ck, .w = &w };
    defer b.jobs.deinit(allocator);
    w.embed = try b.quantized("backbone.embeddings", .{});
    for (0..c.layers) |i| {
        w.norms[i] = try b.get("backbone.layers.{d}.norm.weight", .{i});
        w.layers[i] = switch (c.kinds[i]) {
            .mamba => .{ .mamba = .{
                .in_proj = try b.linear("backbone.layers.{d}.mixer.in_proj", .{i}, &.{try b.quantized("backbone.layers.{d}.mixer.in_proj", .{i})}),
                .out_proj = try b.linear("backbone.layers.{d}.mixer.out_proj", .{i}, &.{try b.quantized("backbone.layers.{d}.mixer.out_proj", .{i})}),
                .conv_w = try convWeight(&b, try b.get("backbone.layers.{d}.mixer.conv1d.weight", .{i}), c),
                .conv_b = try b.f32vec(try b.get("backbone.layers.{d}.mixer.conv1d.bias", .{i})),
                .a_log = try b.f32vec(try b.get("backbone.layers.{d}.mixer.A_log", .{i})),
                .d_skip = try b.f32vec(try b.get("backbone.layers.{d}.mixer.D", .{i})),
                .dt_bias = try b.f32vec(try b.get("backbone.layers.{d}.mixer.dt_bias", .{i})),
                .norm = try b.get("backbone.layers.{d}.mixer.norm.weight", .{i}),
            } },
            .moe => .{ .moe = try b.moe("backbone.layers.{d}.mixer", .{i}) },
            .attention => .{ .attention = try b.attention("backbone.layers.{d}.mixer", "fused.qkv.{d}", .{i}) },
        };
        if (c.kinds[i] == .moe and w.layers[i].moe.gate_bias.dtype != .f32) return error.BadDType;
    }
    w.head = try b.linear("lm_head", .{}, &.{try b.quantized("lm_head", .{})});
    if (draft_ids) |ids| {
        if (ck.has("mtp.layers.0.eh_proj.weight")) w.mtp = try b.mtpHead(ids);
    }
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
