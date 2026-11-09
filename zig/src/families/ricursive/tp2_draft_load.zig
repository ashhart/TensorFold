//! Ripple's DFlash2 drafter: its config, its BF16 weights cut over both ranks, ricursive_tp2_draft.cu's launches.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");

/// The full (NoPE) layer's reach: the publisher's drafts train on 16K rows, so it reads at most this far back.
pub const full_window = 16384;
/// Candidates a row keeps at most (the selector's top-k), and the logits a top-k slice block takes.
pub const most_topk = 64;
pub const topk_slice = 2048;
pub const slices_most = 64;

pub const Config = struct {
    hidden: usize,
    intermediate: usize,
    layers: usize,
    vocab: usize,
    heads: usize,
    kv_heads: usize,
    block: usize,
    kernel: usize,
    group: usize,
    rank: usize,
    topk: usize,
    mask: u32,
    eps: f32,
    theta: f32,
    taps: [8]u32 = @splat(0),
    n_taps: usize = 0,
    window: [8]usize = @splat(0), // each layer's keys behind a row
    rope: [8]bool = @splat(true),

    pub fn qWidth(c: Config) usize {
        return c.heads * 128;
    }
    pub fn kvWidth(c: Config) usize {
        return c.kv_heads * 128;
    }
    pub fn dynamic(c: Config) usize {
        return 2 * c.kernel * (c.hidden / c.group);
    }
    /// A context ring's rows (every layer the same): the widest window and a block more.
    pub fn ringRows(c: Config) usize {
        var w: usize = 0;
        for (c.window[0..c.layers]) |x| w = @max(w, x);
        return w + c.block;
    }
    pub fn ringBytes(c: Config) usize {
        return c.layers * 2 * c.ringRows() * c.kvWidth() * 2;
    }
    /// The rank's widths: its query and key heads and FFN columns (the rest stays whole).
    pub fn share(c: Config, world: usize) !Config {
        if (c.heads % world != 0 or c.kv_heads % world != 0 or c.intermediate % world != 0 or (c.n_taps * c.hidden) % world != 0) return error.UnsupportedDraftConfig;
        var r = c;
        r.heads /= world;
        r.kv_heads /= world;
        r.intermediate /= world;
        return r;
    }
    pub fn chunks(c: Config) usize {
        return (c.ringRows() + 255) / 256;
    }

    pub fn parse(gpa: std.mem.Allocator, bytes: []const u8, hidden: usize, vocab: usize, target_layers: usize) !Config {
        const doc = try std.json.parseFromSlice(std.json.Value, gpa, bytes, .{});
        defer doc.deinit();
        if (doc.value != .object) return error.BadDraftConfig;
        const o = doc.value.object;
        const d = o.get("dflash_config") orelse return error.BadDraftConfig;
        if (d != .object) return error.BadDraftConfig;
        var c: Config = .{ .hidden = try int(o, "hidden_size"), .intermediate = try int(o, "intermediate_size"), .layers = try int(o, "num_hidden_layers"), .vocab = try int(o, "vocab_size"), .heads = try int(o, "num_attention_heads"), .kv_heads = try int(o, "num_key_value_heads"), .block = try int(d.object, "block_size"), .kernel = try int(d.object, "conv_kernel_size"), .group = try int(d.object, "conv_group_size"), .rank = try int(d.object, "selector_rank"), .topk = try int(d.object, "selector_top_k"), .mask = @intCast(try int(d.object, "mask_token_id")), .eps = @floatCast(try num(o, "rms_norm_eps")), .theta = @floatCast(try num(o, "rope_theta")) };
        if (try int(o, "head_dim") != 128 or c.hidden != hidden or c.vocab != vocab or c.layers > 8 or c.block < 2 or c.block > 16 or c.topk > most_topk or slices(c.vocab) > slices_most) return error.UnsupportedDraftConfig;
        if (c.hidden % c.group != 0 or c.heads % c.kv_heads != 0 or c.hidden % 256 != 0 or c.mask >= c.vocab) return error.UnsupportedDraftConfig;
        const taps = d.object.get("target_layer_ids") orelse return error.BadDraftConfig;
        if (taps != .array or taps.array.items.len == 0 or taps.array.items.len > 8) return error.BadDraftConfig;
        for (taps.array.items, 0..) |t, i| {
            if (t != .integer or t.integer < 0 or t.integer >= target_layers) return error.BadDraftConfig;
            c.taps[i] = @intCast(t.integer);
        }
        c.n_taps = taps.array.items.len;
        const types = o.get("layer_types") orelse return error.BadDraftConfig;
        if (types != .array or types.array.items.len != c.layers) return error.BadDraftConfig;
        const sliding = try int(o, "sliding_window");
        for (types.array.items, 0..) |t, i| {
            if (t != .string) return error.BadDraftConfig;
            c.window[i] = if (std.mem.eql(u8, t.string, "sliding_attention")) sliding else if (std.mem.eql(u8, t.string, "full_attention")) full_window else return error.UnsupportedDraftConfig;
        }
        if (o.get("no_rope_layers")) |nr| if (nr == .array) for (nr.array.items) |x| {
            if (x != .integer or x.integer < 0 or x.integer >= c.layers) return error.BadDraftConfig;
            c.rope[@intCast(x.integer)] = false;
        };
        return c;
    }
};

fn int(o: std.json.ObjectMap, key: []const u8) !usize {
    const v = o.get(key) orelse return error.BadDraftConfig;
    if (v != .integer or v.integer <= 0) return error.BadDraftConfig;
    return @intCast(v.integer);
}

fn num(o: std.json.ObjectMap, key: []const u8) !f64 {
    return switch (o.get(key) orelse return error.BadDraftConfig) {
        .integer => |i| @floatFromInt(i),
        .float => |f| f,
        else => error.BadDraftConfig,
    };
}

pub const Layer = struct { input_norm: u64, post_norm: u64, q: u64, k: u64, v: u64, o: u64, q_norm: u64, k_norm: u64, gate: u64, up: u64, down: u64, attn_conv: u64, attn_base: u64, mlp_conv: u64, mlp_base: u64 };

/// The drafter's tensors as stored (bf16, [out, in]); the selector's codebooks stay on the host.
pub const Weights = struct {
    gpa: std.mem.Allocator,
    bufs: std.ArrayList(cuda.DeviceBuffer) = .empty,
    fc: u64 = 0,
    hidden_norm: u64 = 0,
    final_norm: u64 = 0,
    selector: u64 = 0,
    predecessor: []u16 = &.{},
    successor: []u16 = &.{},
    layers: [8]Layer = undefined,
    bytes: usize = 0,

    pub fn deinit(w: *Weights) void {
        for (w.bufs.items) |*b| b.free();
        w.bufs.deinit(w.gpa);
        w.gpa.free(w.predecessor);
        w.gpa.free(w.successor);
    }

    fn up(w: *Weights, d: *const cuda.Driver, bytes: []const u8) !u64 {
        try w.bufs.ensureUnusedCapacity(w.gpa, 1);
        const b = try cuda.DeviceBuffer.fromHost(d, bytes);
        w.bufs.appendAssumeCapacity(b);
        w.bytes += bytes.len;
        return b.ptr;
    }

    fn dev(w: *Weights, d: *const cuda.Driver, ck: *core.Checkpoint, key: []const u8, shape: []const usize) !u64 {
        return w.up(d, (try ck.expect(key, .bf16, shape)).bytes);
    }

    /// A [n, k] matrix's share at `cut`: its output rows (.rows) or its input columns (.columns) [lo, hi) for the rank.
    fn part(w: *Weights, d: *const cuda.Driver, ck: *core.Checkpoint, key: []const u8, n: usize, k: usize, side: enum { rows, columns }, cut: Cut) !u64 {
        const t = try ck.expect(key, .bf16, &.{ n, k });
        if (side == .rows) {
            const per = n / cut.world;
            return w.up(d, t.bytes[cut.rank * per * k * 2 ..][0 .. per * k * 2]);
        }
        const per = k / cut.world;
        const g = try w.gpa.alloc(u8, n * per * 2);
        defer w.gpa.free(g);
        for (0..n) |r| @memcpy(g[r * per * 2 ..][0 .. per * 2], t.bytes[(r * k + cut.rank * per) * 2 ..][0 .. per * 2]);
        return w.up(d, g);
    }

    fn host(w: *Weights, ck: *core.Checkpoint, key: []const u8, shape: []const usize) ![]u16 {
        const t = try ck.expect(key, .bf16, shape);
        const out = try w.gpa.alloc(u16, t.bytes.len / 2);
        @memcpy(std.mem.sliceAsBytes(out), t.bytes);
        return out;
    }

    /// The whole drafter `c` (full widths), each rank's share as the target's cuts (heads, FFN, fc and o/down inputs).
    pub fn load(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, dir: []const u8, c: Config, cut: Cut) !Weights {
        var ck = try core.Checkpoint.openModel(gpa, io, dir);
        defer ck.close();
        var w: Weights = .{ .gpa = gpa };
        errdefer w.deinit();
        const h = c.hidden;
        w.fc = try w.part(d, &ck, "fc.weight", h, c.n_taps * h, .columns, cut);
        w.hidden_norm = try w.dev(d, &ck, "hidden_norm.weight", &.{h});
        w.final_norm = try w.dev(d, &ck, "norm.weight", &.{h});
        if (cut.rank == 0) {
            w.selector = try w.dev(d, &ck, "candidate_selector.hidden_projection.weight", &.{ c.rank, h });
            w.predecessor = try w.host(&ck, "candidate_selector.predecessor_codebook", &.{ c.vocab, c.rank });
            w.successor = try w.host(&ck, "candidate_selector.successor_codebook", &.{ c.vocab, c.rank });
        }
        var buf: [96]u8 = undefined;
        for (w.layers[0..c.layers], 0..) |*l, i| {
            const N = struct {
                b: *[96]u8,
                i: usize,
                fn of(s: @This(), suffix: []const u8) ![]const u8 {
                    return std.fmt.bufPrint(s.b, "layers.{d}.{s}", .{ s.i, suffix });
                }
            };
            const n: N = .{ .b = &buf, .i = i };
            l.* = .{
                .input_norm = try w.dev(d, &ck, try n.of("input_layernorm.weight"), &.{h}),
                .post_norm = try w.dev(d, &ck, try n.of("post_attention_layernorm.weight"), &.{h}),
                .q = try w.part(d, &ck, try n.of("self_attn.q_proj.weight"), c.qWidth(), h, .rows, cut),
                .k = try w.part(d, &ck, try n.of("self_attn.k_proj.weight"), c.kvWidth(), h, .rows, cut),
                .v = try w.part(d, &ck, try n.of("self_attn.v_proj.weight"), c.kvWidth(), h, .rows, cut),
                .o = try w.part(d, &ck, try n.of("self_attn.o_proj.weight"), h, c.qWidth(), .columns, cut),
                .q_norm = try w.dev(d, &ck, try n.of("self_attn.q_norm.weight"), &.{128}),
                .k_norm = try w.dev(d, &ck, try n.of("self_attn.k_norm.weight"), &.{128}),
                .gate = try w.part(d, &ck, try n.of("mlp.gate_proj.weight"), c.intermediate, h, .rows, cut),
                .up = try w.part(d, &ck, try n.of("mlp.up_proj.weight"), c.intermediate, h, .rows, cut),
                .down = try w.part(d, &ck, try n.of("mlp.down_proj.weight"), h, c.intermediate, .columns, cut),
                .attn_conv = try w.dev(d, &ck, try n.of("attention_conv.kernel_projection.weight"), &.{ c.dynamic(), h }),
                .attn_base = try w.dev(d, &ck, try n.of("attention_conv.base_kernel"), &.{ 2, c.kernel, h }),
                .mlp_conv = try w.dev(d, &ck, try n.of("mlp_conv.kernel_projection.weight"), &.{ c.dynamic(), h }),
                .mlp_base = try w.dev(d, &ck, try n.of("mlp_conv.base_kernel"), &.{ 2, c.kernel, h }),
            };
        }
        return w;
    }
};

/// A rank's share of the drafter: its rank of the target's two, cut as the target's.
pub const Cut = struct { rank: usize, world: usize };

/// A hold's stream as the attention and head kernels read it (ricursive_tp2_draft.cu's Seg).
pub const Seg = extern struct { ring: u64, begin: i64, position: i64 };

fn blocks(n: usize) u32 {
    return @intCast((n + 255) / 256);
}

/// ricursive_tp2_draft.cu's functions and their launches on one stream.
pub const Kernels = struct {
    mod: cuda.Module,
    f: struct { gather: cuda.Function, rms: cuda.Function, conv_in: cuda.Function, conv_out: cuda.Function, heads: cuda.Function, kv_put: cuda.Function, f2b: cuda.Function, silu: cuda.Function, embed: cuda.Function, attn: cuda.Function, merge: cuda.Function, topk: cuda.Function },

    pub fn load(d: *const cuda.Driver) !Kernels {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var k: Kernels = .{ .mod = try cuda.Module.load(d, cuda.kernels.ricursive_tp2_draft), .f = undefined };
        errdefer k.mod.unload();
        inline for (@typeInfo(@TypeOf(k.f)).@"struct".field_names) |name| @field(k.f, name) = try k.mod.function("tf_rtp2d_" ++ name);
        return k;
    }

    pub fn unload(k: *Kernels) void {
        k.mod.unload();
    }

    fn go(f: cuda.Function, s: cuda.Stream, grid: [3]usize, block: u32, shared: u32, args: anytype) !void {
        var a: cuda.Args = .{};
        inline for (args) |v| a.add(v);
        try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(grid[0]), .y = @intCast(grid[1]), .z = @intCast(grid[2]) }, .block = .{ .x = block }, .shared = shared }, s, &a);
    }

    /// Columns [col, col + width) of tap rows start.. (rows of `stride` bf16) into OUT rows of `width`.
    pub fn gather(k: *const Kernels, s: cuda.Stream, ring: u64, tr: usize, start: u64, stride: usize, col: usize, width: usize, out: u64, rows: usize) !void {
        try go(k.f.gather, s, .{ rows, 1, 1 }, 256, 0, .{ ring, @as(i32, @intCast(tr)), @as(i64, @intCast(start)), @as(i32, @intCast(stride)), @as(i32, @intCast(col)), @as(i32, @intCast(width)), out });
    }
    /// out row j = rms(x) * w; kind 0 bf16, 1 fp16; block > 0 skips each block's first row.
    pub fn rms(k: *const Kernels, s: cuda.Stream, x: u64, w: u64, out: u64, h: usize, eps: f32, kind: i32, block: usize, rows: usize) !void {
        try go(k.f.rms, s, .{ rows, 1, 1 }, 256, 0, .{ x, w, out, @as(i32, @intCast(h)), eps, kind, @as(i32, @intCast(block)) });
    }
    pub fn convIn(k: *const Kernels, s: cuda.Stream, x: u64, dyn: u64, base: u64, out: u64, c: anytype, rows: usize) !void {
        try go(k.f.conv_in, s, .{ rows, blocks(c.hidden), 1 }, 256, 0, .{ x, dyn, base, out, @as(i32, @intCast(c.hidden)), @as(i32, @intCast(c.group)), @as(i32, @intCast(c.kernel)), @as(i32, @intCast(c.block)) });
    }
    pub fn convOut(k: *const Kernels, s: cuda.Stream, x: u64, dyn: u64, base: u64, res: u64, out: u64, c: anytype, rows: usize) !void {
        try go(k.f.conv_out, s, .{ rows, blocks(c.hidden), 1 }, 256, 0, .{ x, dyn, base, res, out, @as(i32, @intCast(c.hidden)), @as(i32, @intCast(c.group)), @as(i32, @intCast(c.kernel)), @as(i32, @intCast(c.block)) });
    }
    pub fn heads(k: *const Kernels, s: cuda.Stream, x: u64, stride: usize, w: u64, out: u64, nh: usize, segs: u64, c: anytype, rope: bool, rows: usize) !void {
        try go(k.f.heads, s, .{ rows * nh, 1, 1 }, 32, 0, .{ x, @as(i32, @intCast(stride)), w, out, @as(i32, @intCast(nh)), segs, @as(i32, @intCast(c.block)), c.eps, c.theta, @as(i32, @intFromBool(rope)) });
    }
    pub fn kvPut(k: *const Kernels, s: cuda.Stream, kx: u64, vx: u64, w: u64, rk: u64, rv: u64, c: anytype, pos0: u64, rope: bool, rows: usize) !void {
        try go(k.f.kv_put, s, .{ rows * c.kv_heads, 1, 1 }, 32, 0, .{ kx, vx, w, rk, rv, @as(i32, @intCast(c.kv_heads)), @as(i64, @intCast(pos0)), @as(i32, @intCast(c.ringRows())), c.eps, c.theta, @as(i32, @intFromBool(rope)) });
    }
    pub fn f2b(k: *const Kernels, s: cuda.Stream, x: u64, out: u64, n: usize) !void {
        try go(k.f.f2b, s, .{ blocks(n), 1, 1 }, 256, 0, .{ x, out, @as(i64, @intCast(n)) });
    }
    pub fn silu(k: *const Kernels, s: cuda.Stream, g: u64, u: u64, out: u64, n: usize) !void {
        try go(k.f.silu, s, .{ blocks(n), 1, 1 }, 256, 0, .{ g, u, out, @as(i64, @intCast(n)) });
    }
    pub fn embed(k: *const Kernels, s: cuda.Stream, ids: u64, table: u64, out: u64, h: usize, rows: usize) !void {
        try go(k.f.embed, s, .{ rows, 1, 1 }, 256, 0, .{ ids, table, out, @as(i32, @intCast(h)) });
    }
    /// Block attention for `rows` rows of layer `layer` (partials pm, pl, po), merged into out.
    pub fn attention(k: *const Kernels, s: cuda.Stream, q: u64, kb: u64, vb: u64, segs: u64, c: anytype, layer: usize, p: [3]u64, out: u64, rows: usize) !void {
        const ch = c.chunks();
        try go(k.f.attn, s, .{ rows, c.heads, ch }, 128, 0, .{ q, kb, vb, segs, @as(i32, @intCast(c.block)), @as(i32, @intCast(c.heads)), @as(i32, @intCast(c.kv_heads)), @as(i32, @intCast(c.ringRows())), @as(i32, @intCast(c.window[layer])), @as(i32, @intCast(layer)), p[0], p[1], p[2] });
        try go(k.f.merge, s, .{ rows, c.heads, 1 }, 32, 0, .{ p[0], p[1], p[2], @as(i32, @intCast(c.heads)), @as(i32, @intCast(ch)), out });
    }
    pub fn topk(k: *const Kernels, s: cuda.Stream, logits: u64, vocab: usize, kk: usize, pv: u64, pi: u64, rows: usize) !void {
        try go(k.f.topk, s, .{ slices(vocab), rows, 1 }, 256, topk_slice * 4, .{ logits, @as(i32, @intCast(vocab)), @as(i32, topk_slice), @as(i32, @intCast(kk)), pv, pi });
    }
};

pub fn slices(vocab: usize) usize {
    return (vocab + topk_slice - 1) / topk_slice;
}

test "Ripple's drafter config: four sliding layers and a NoPE full layer, block 16, four-tap convolutions" {
    const text =
        \\{"dflash_config":{"block_size":16,"conv_group_size":16,"conv_kernel_size":4,"mask_token_id":100278,"selector_rank":256,
        \\ "selector_top_k":32,"target_layer_ids":[1,20,39,58,77]},"head_dim":128,"hidden_size":5120,"intermediate_size":40960,
        \\ "layer_types":["sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention"],
        \\ "no_rope_layers":[4],"num_attention_heads":80,"num_hidden_layers":5,"num_key_value_heads":8,"rms_norm_eps":1e-06,
        \\ "rope_theta":10000.0,"sliding_window":4096,"vocab_size":100352}
    ;
    const c = try Config.parse(std.testing.allocator, text, 5120, 100352, 80);
    try std.testing.expectEqual(@as(usize, 2560), c.dynamic());
    try std.testing.expectEqual(@as(usize, 16384 + 16), c.ringRows());
    try std.testing.expectEqual(@as(usize, 65), c.chunks());
    try std.testing.expect(c.rope[3] and !c.rope[4] and c.window[0] == 4096 and c.window[4] == full_window);
    try std.testing.expectEqual(@as(u32, 77), c.taps[4]);
    try std.testing.expectError(error.UnsupportedDraftConfig, Config.parse(std.testing.allocator, text, 4096, 100352, 80));
    try std.testing.expectEqual(@as(usize, 49), slices(100352));
}
