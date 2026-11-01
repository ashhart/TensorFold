//! A tiny V4.1 with the release's structure and random release-format weights, on this GPU, with a random past.
const std = @import("std");
const mtl = @import("metal");
const pkg = @import("dsv41_metal");
const host = @import("dsv41");
const shards = @import("shards.zig");
const ref = @import("ref.zig");

const Config = host.config.Config;
const Ids = host.config.Ids;
const Spec = host.pack.Spec;
const Ref = pkg.ctx.Ref;

/// Window; ratio-2 KV and index source; ratio-2 reader; ratio-1 source of candidates; ratio-1 indexer in them.
pub fn config() Config {
    return .{
        .format = .release_fp8,
        .vocab = 256,
        .hidden = 256,
        .layers = 5,
        .heads = 4,
        .q_lora = 128,
        .o_lora = 128,
        .o_groups = 2,
        .experts = 16,
        .expert_width = 256,
        .compress_ratios = Ids.of(&.{ 0, 2, 2, 1, 1 }),
        .kv_sources = Ids.of(&.{ 1, 3 }),
        .index_sources = Ids.of(&.{ 1, 3, 4 }),
        .candidate_source = 3,
        .candidate_blocks = 96,
        .engram_layers = Ids.of(&.{}),
        .engram_rows = host.config.List(u64, 8).of(&.{}),
        .mtp_layers = 0,
        .dspark_targets = Ids.of(&.{}),
    };
}

/// The tiny forward: every attention kind, Engram at a window layer and a ratio-1 KV source, taps at the last three.
pub fn forwardConfig() Config {
    var c = config();
    c.layers = 7;
    c.compress_ratios = Ids.of(&.{ 0, 0, 2, 2, 1, 1, 0 });
    c.kv_sources = Ids.of(&.{ 2, 4 });
    c.index_sources = Ids.of(&.{ 2, 4, 5 });
    c.candidate_source = 4;
    c.engram_layers = Ids.of(&.{ 1, 4 });
    c.engram_rows = host.config.List(u64, 8).of(&.{ 0, 0 });
    c.engram_max_ngram = 4;
    c.engram_heads = 2;
    c.engram_vocab = 61;
    c.dspark_targets = Ids.of(&.{ 4, 5, 6 });
    return c;
}

/// The tiny prompt model: ced's split at layer 4 (a ratio-1 KV and candidate source), Engram only before it.
pub fn promptConfig() Config {
    var c = forwardConfig();
    c.engram_layers = Ids.of(&.{ 1, 3 });
    return c;
}

/// The tiny forward with a DSpark drafter: 3 stages, blocks of 5, 16 experts with 3 a token, a Markov rank of 16.
pub fn dsparkConfig() Config {
    var c = forwardConfig();
    c.mtp_layers = 3;
    c.dspark_block = 5;
    c.dspark_noise_token = 200;
    c.dspark_markov_rank = 16;
    c.dspark_experts = 16;
    c.dspark_topk = 3;
    return c;
}

/// How a tensor's bytes are drawn.
const Gen = enum { norm, hc_fn, hc_scale, small, unit_bf16, fp8, e8m0_dense, fp4, e8m0_fp4 };
const Item = struct { spec: Spec, gen: Gen, k: usize };

const Builder = struct {
    a: std.mem.Allocator,
    items: std.ArrayList(Item) = .empty,

    fn add(b: *Builder, name: []const u8, dtype: []const u8, shape: []const usize, gen: Gen, k: usize) !void {
        const n = try b.a.dupe(u8, name);
        const s = try b.a.dupe(usize, shape);
        try b.items.append(b.a, .{ .spec = .{ .name = n, .dtype = dtype, .shape = s }, .gen = gen, .k = k });
    }

    fn fp8(b: *Builder, prefix: []const u8, n: usize, k: usize) !void {
        var buf: [128]u8 = undefined;
        try b.add(try std.fmt.bufPrint(&buf, "{s}.weight", .{prefix}), "F8_E4M3", &.{ n, k }, .fp8, k);
        try b.add(try std.fmt.bufPrint(&buf, "{s}.scale", .{prefix}), "F8_E8M0", &.{ (n + 31) / 32, (k + 31) / 32 }, .e8m0_dense, k);
    }

    fn fp4(b: *Builder, prefix: []const u8, n: usize, k: usize) !void {
        var buf: [128]u8 = undefined;
        try b.add(try std.fmt.bufPrint(&buf, "{s}.weight", .{prefix}), "I8", &.{ n, k / 2 }, .fp4, k);
        try b.add(try std.fmt.bufPrint(&buf, "{s}.scale", .{prefix}), "F8_E8M0", &.{ n, k / 32 }, .e8m0_fp4, k);
    }
};

fn log2Sqrt(k: usize) u8 {
    return @intCast(std.math.log2_int(usize, k) / 2);
}

fn draw(item: Item, out: []u8, rng: std.Random) void {
    const k = item.k;
    switch (item.gen) {
        .norm => for (0..out.len / 2) |i| std.mem.writeInt(u16, out[2 * i ..][0..2], ref.toBf(1 + 0.2 * (rng.float(f32) - 0.5)), .little),
        .unit_bf16 => for (0..out.len / 2) |i| std.mem.writeInt(u16, out[2 * i ..][0..2], ref.toBf((rng.float(f32) * 2 - 1) * @sqrt(3.0 / @as(f32, @floatFromInt(k)))), .little),
        .hc_fn => for (0..out.len / 4) |i| std.mem.writeInt(u32, out[4 * i ..][0..4], @bitCast((rng.float(f32) * 2 - 1) * @sqrt(3.0 / @as(f32, @floatFromInt(k)))), .little),
        .hc_scale => for (0..out.len / 4) |i| std.mem.writeInt(u32, out[4 * i ..][0..4], @bitCast(0.5 + rng.float(f32)), .little),
        .small => for (0..out.len / 4) |i| std.mem.writeInt(u32, out[4 * i ..][0..4], @bitCast(rng.float(f32) * 2 - 1), .little),
        .fp8 => for (out) |*b| {
            b.* = (rng.int(u8) & 0x87) | (@as(u8, 5 + rng.uintLessThan(u8, 4)) << 3);
        },
        .e8m0_dense => for (out) |*b| {
            b.* = 127 - log2Sqrt(k) - 2 + rng.uintLessThan(u8, 2);
        },
        .fp4 => rng.bytes(out),
        .e8m0_fp4 => for (out) |*b| {
            b.* = 127 - log2Sqrt(k) - 3 + rng.uintLessThan(u8, 2);
        },
    }
}

/// The layers' tensors under the release's names, drawn from `seed`, as one safetensors image.
pub fn image(a: std.mem.Allocator, c: *const Config, seed: u64) ![]u8 {
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    var b: Builder = .{ .a = arena.allocator() };
    var name: [128]u8 = undefined;
    for (0..c.layers + c.mtp_layers) |li| {
        const l: u32 = @intCast(li);
        var bb: [32]u8 = undefined;
        const block = try host.names.blockPrefix(c, l, &bb);
        const nat = try host.names.blockNatives(c, l);
        for (nat.items()) |s| {
            const full = try std.fmt.bufPrint(&name, "{s}.{s}", .{ block, s.suffix });
            const dims = [2]usize{ s.shape[0], s.shape[1] };
            const shape = dims[0..s.rank];
            const gen: Gen = switch (s.store) {
                .norm, .param => .norm,
                .half => .unit_bf16,
                .f32 => if (std.mem.endsWith(u8, s.suffix, "_fn")) .hc_fn else if (std.mem.endsWith(u8, s.suffix, "_scale")) .hc_scale else .small,
                else => .small,
            };
            const dtype: []const u8 = if (s.store == .norm or s.store == .half or s.store == .param) "BF16" else "F32";
            try b.add(full, dtype, shape, gen, if (s.rank == 2) dims[1] else 1);
        }
        const kg = c.heads / c.o_groups * c.head_dim;
        const projs = [_]struct { []const u8, usize, usize, bool }{
            .{ "attn.wq_a", c.q_lora, c.hidden, true },                   .{ "attn.wkv", c.head_dim, c.hidden, true },
            .{ "attn.wq_b", c.heads * c.head_dim, c.q_lora, true },       .{ "attn.wo_a", c.o_groups * c.o_lora, kg, true },
            .{ "attn.wo_b", c.hidden, c.o_groups * c.o_lora, true },      .{ "attn.indexer.wq_b", c.index_heads * c.index_dim, c.q_lora, c.isIndexSource(l) },
            .{ "ffn.shared_experts.w1", c.expert_width, c.hidden, true }, .{ "ffn.shared_experts.w2", c.hidden, c.expert_width, true },
            .{ "ffn.shared_experts.w3", c.expert_width, c.hidden, true }, .{ "engram.wkv", (c.hc_mult + 1) * c.hidden, c.hashCols() * 256, c.isEngram(l) },
        };
        for (projs) |p| if (p[3]) try b.fp8(try std.fmt.bufPrint(&name, "{s}.{s}", .{ block, p[0] }), p[1], p[2]);
        // the release keeps the compressor and the index key projection in bf16
        const halves = [_]struct { []const u8, usize, usize, bool }{
            .{ "attn.compressor.wkv", c.head_dim, c.hidden, c.isKvSource(l) },
            .{ "attn.compressor.wgate", c.head_dim, c.hidden, c.isKvSource(l) and c.compressRatio(l) > 1 },
            .{ "attn.indexer.wk", c.index_dim, c.head_dim, c.isKvSource(l) },
        };
        for (halves) |p| if (p[3]) try b.add(try std.fmt.bufPrint(&name, "{s}.{s}.weight", .{ block, p[0] }), "BF16", &.{ p[1], p[2] }, .unit_bf16, p[2]);
        for (0..c.expertsOf(l).count) |e| for ([_][]const u8{ "w1", "w2", "w3" }) |w| {
            const two = std.mem.eql(u8, w, "w2");
            try b.fp4(try std.fmt.bufPrint(&name, "{s}.ffn.experts.{d}.{s}", .{ block, e, w }), if (two) c.hidden else c.expert_width, if (two) c.expert_width else c.hidden);
        };
    }
    if (c.mtp_layers > 0) {
        const T: usize = c.dspark_targets.len;
        const last = c.mtp_layers - 1;
        const rank: usize = c.dspark_markov_rank;
        try b.fp8("mtp.0.main_proj", c.hidden, T * c.hidden);
        try b.add("mtp.0.main_norm.weight", "BF16", &.{c.hidden}, .norm, 1);
        try b.add(try std.fmt.bufPrint(&name, "mtp.{d}.norm.weight", .{last}), "BF16", &.{c.hidden}, .norm, 1);
        try b.add(try std.fmt.bufPrint(&name, "mtp.{d}.markov_head.embed.weight", .{last}), "BF16", &.{ c.vocab, rank }, .unit_bf16, 2 * rank);
        try b.add(try std.fmt.bufPrint(&name, "mtp.{d}.markov_head.head.weight", .{last}), "BF16", &.{ c.vocab, rank }, .unit_bf16, 2 * rank);
        try b.add(try std.fmt.bufPrint(&name, "mtp.{d}.confidence_head.proj.weight", .{last}), "BF16", &.{ 1, c.hidden + rank }, .unit_bf16, c.hidden + rank);
    }
    try b.add("embed.weight", "BF16", &.{ c.vocab, c.hidden }, .unit_bf16, 3);
    try b.add("head.weight", "BF16", &.{ c.vocab, c.hidden }, .unit_bf16, c.hidden);
    try b.add("norm.weight", "BF16", &.{c.hidden}, .norm, 1);
    const specs = try arena.allocator().alloc(Spec, b.items.items.len);
    for (specs, b.items.items) |*s, it| s.* = it.spec;
    const bytes = try host.pack.image(a, specs);
    const head: usize = @intCast(std.mem.readInt(u64, bytes[0..8], .little));
    var at = 8 + head;
    var prng = std.Random.DefaultPrng.init(seed);
    for (b.items.items) |it| {
        var n = host.pack.DType.parse(it.spec.dtype).?.size();
        for (it.spec.shape) |d| n *= d;
        draw(it, bytes[at..][0..n], prng.random());
        at += n;
    }
    return bytes;
}

/// The device, kernels, weights, state and buffers of one test.
pub const Rig = struct {
    a: std.mem.Allocator,
    c: Config,
    device: mtl.Device,
    queue: mtl.Queue,
    p: pkg.pipelines.Pipelines,
    layers: []pkg.weights.Layer,
    /// null for a rig of chosen layers only (the real-weight check)
    top: ?pkg.weights.Top,
    tables: pkg.rope.Tables,
    s: pkg.scratch.Scratch,
    slot: pkg.kv.Slot,
    io: [4]mtl.Buffer,
    /// the mapped shards of a zero-copy rig
    map: ?pkg.mapped.Map = null,

    /// How the weights reach the device: copied, wrapped over 3 mapped shards, or the same with every offset skewed.
    pub const Mode = enum { copy, mapped, skewed };

    /// Weights from a written pack folder, read back through the release loader.
    pub fn init(a: std.mem.Allocator, io: std.Io, limit: u32, seed: u64) !*Rig {
        return initWith(a, io, config(), limit, seed);
    }

    pub fn initWith(a: std.mem.Allocator, io: std.Io, c: Config, limit: u32, seed: u64) !*Rig {
        return initMode(a, io, c, limit, seed, .copy);
    }

    pub fn initMode(a: std.mem.Allocator, io: std.Io, c: Config, limit: u32, seed: u64, mode: Mode) !*Rig {
        const r = try a.create(Rig);
        errdefer a.destroy(r);
        r.a = a;
        r.c = c;
        try r.c.validate();
        r.device = try mtl.Device.init();
        r.queue = try r.device.queue();
        r.p = try pkg.pipelines.Pipelines.load(a, r.device, &r.c);
        var tmp = std.testing.tmpDir(.{});
        defer tmp.cleanup();
        const img = try image(a, &r.c, seed);
        defer a.free(img);
        switch (mode) {
            .copy => try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors", .data = img }),
            .mapped => try shards.write(a, io, tmp.dir, img, 3, 0),
            .skewed => try shards.write(a, io, tmp.dir, img, 3, 4),
        }
        const dir = try tmp.dir.realPathFileAlloc(io, ".", a);
        defer a.free(dir);
        var pack = try host.pack.Pack.open(a, io, dir);
        defer pack.deinit();
        r.map = if (mode == .copy) null else try pkg.mapped.Map.open(a, r.device, &pack);
        const map: ?*pkg.mapped.Map = if (r.map) |*m| m else null;
        r.layers = try a.alloc(pkg.weights.Layer, r.c.layers);
        for (r.layers, 0..) |*l, i| l.* = try pkg.weights.loadFrom(a, io, r.device, &pack, &r.c, @intCast(i), map);
        r.top = try pkg.weights.loadTopFrom(a, io, r.device, &pack, &r.c, map);
        r.tables = try pkg.rope.Tables.init(r.device, &r.c, limit);
        r.s = try pkg.scratch.Scratch.init(r.device, &r.c, limit);
        r.slot = try pkg.kv.Slot.init(a, r.device, &r.c, limit);
        const sizes = [4]usize{ 16 * 4 * r.c.hidden * 2, 16 * 4 * r.c.hidden * 2, 16 * 16, 16 * 16 };
        for (&r.io, sizes) |*b, n| b.* = try r.device.buffer(n, mtl.ResourceOptions.shared);
        return r;
    }

    pub fn deinit(r: *Rig) void {
        for (r.layers) |*l| l.deinit();
        r.a.free(r.layers);
        if (r.top) |*t| t.deinit();
        if (r.map) |*m| m.deinit();
        r.tables.deinit();
        r.s.deinit();
        r.slot.deinit();
        for (r.io) |b| b.deinit();
        r.p.deinit();
        r.queue.deinit();
        r.device.deinit();
        r.a.destroy(r);
    }

    pub fn ctx(r: *const Rig, slot: *const pkg.kv.Slot, p0: u32, rows: u32) pkg.ctx.Ctx {
        return .{ .c = &r.c, .p = &r.p, .s = &r.s, .slot = slot, .tables = &r.tables, .p0 = p0, .rows = rows };
    }

    pub fn streams(r: *const Rig) [2]Ref {
        return .{ .{ .buf = r.io[0] }, .{ .buf = r.io[1] } };
    }

    pub fn pres(r: *const Rig) [2]Ref {
        return .{ .{ .buf = r.io[2] }, .{ .buf = r.io[3] } };
    }
};

/// A command buffer with one serial compute pass, committed and waited for.
pub const Pass = struct {
    pool: mtl.objc.Pool,
    cb: mtl.CommandBuffer,
    e: mtl.ComputeEncoder,

    pub fn begin(q: mtl.Queue) Pass {
        const pool = mtl.objc.Pool.push();
        const cb = q.commandBuffer();
        return .{ .pool = pool, .cb = cb, .e = cb.compute(.serial) };
    }

    pub fn finish(p: Pass) !void {
        defer p.pool.pop();
        p.e.end();
        p.cb.commit();
        p.cb.wait();
        if (p.cb.failure()) |msg| {
            std.debug.print("command buffer failed: {s}\n", .{msg});
            return error.CommandBufferFailed;
        }
    }
};

fn bfRow(out: []u16, rng: std.Random, scale: f32) void {
    for (out) |*v| v.* = ref.toBf((rng.float(f32) * 2 - 1) * scale);
}

/// A random past before position p0 in every layer: window keys, compressed rows, index keys and carries.
pub fn fillPast(c: *const Config, slot: *const pkg.kv.Slot, p0: u32, seed: u64) void {
    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();
    const D = c.head_dim;
    for (slot.regions, 0..) |reg, li| {
        const l: u32 = @intCast(li);
        const win = reg.window.slice(u16, 2 * @as(usize, pkg.kv.windowSlots(c)) * D);
        const first = p0 -| pkg.kv.windowSlots(c);
        for (first..p0) |p| {
            const rows = pkg.kv.windowRows(c, @intCast(p));
            bfRow(win[rows[0] * D ..][0..D], rng, 1.5);
            @memcpy(win[rows[1] * D ..][0..D], win[rows[0] * D ..][0..D]);
        }
        if (!c.isKvSource(l)) continue;
        const n = p0 / c.compressRatio(l);
        bfRow(reg.pool.?.slice(u16, n * D), rng, 1.5);
        const keys = reg.keys.?.contents()[0 .. n * pkg.kv.index_key_bytes];
        rng.bytes(keys);
        for (0..n) |k| for (64..68) |j| {
            keys[k * 68 + j] = 124 + rng.uintLessThan(u8, 5);
        };
        if (reg.carry) |cb| for (cb.slice(f32, 4 * D)) |*v| {
            v.* = rng.float(f32) * 2 - 1;
        };
    }
}

/// Random input streams [rows][4][hidden] (bf16) and pre mixes [rows][4] in the first ping-pong buffers.
pub fn fillInput(r: *const Rig, rows: usize, seed: u64) void {
    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();
    bfRow(r.io[0].slice(u16, rows * 4 * r.c.hidden), rng, 2);
    for (r.io[2].slice(f32, rows * 4)) |*v| v.* = 0.2 + 0.8 * rng.float(f32);
}
