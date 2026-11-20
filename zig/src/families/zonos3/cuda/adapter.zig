//! The speech-edit adapter (zonos-accent-lora/3): bf16 factors on the device, the rows it adds in fp32 on the host.
const std = @import("std");
const cuda = @import("cuda");
const pth = @import("core").torch_pth;
const cfg = @import("../config.zig");

const D = cfg.dim;
const C = cfg.codebooks;

/// The widest factor a pass has scratch for.
pub const max_rank = 64;

/// One weight's factors: `down` bf16 [rank, in], `up` bf16 [out, rank] (experts: a block an expert). rank 0: none.
pub const Pair = struct { down: u64 = 0, up: u64 = 0, rank: usize = 0 };

pub const Layer = struct {
    in_proj: Pair = .{},
    wq: Pair = .{},
    wkv: Pair = .{},
    gater: Pair = .{},
    wo: Pair = .{},
    w_in: Pair = .{},
    w_out: Pair = .{},
    router: Pair = .{},
    latent_down: Pair = .{},
    latent_up: Pair = .{},
    gate_up: Pair = .{},
    down: Pair = .{},
    /// fp32 [experts]: the router's balancing biases plus the adapter's shift; 0: the model's own.
    bias: u64 = 0,
};

pub const Adapter = struct {
    gpa: std.mem.Allocator,
    bufs: std.ArrayList(cuda.DeviceBuffer) = .empty,
    bytes: usize = 0,
    layers: [cfg.layers]Layer = @splat(.{}),
    /// The output head's factors, `up` over the model's own rows.
    head: Pair = .{},
    /// `up`'s rows of the span-end id, one a codebook: [C][head.rank].
    end_up: []f32 = &.{},
    /// The span-end id's head rows and embedding rows, one a codebook.
    end_head: [][D]f32 = &.{},
    end_rows: [][D]f32 = &.{},
    /// The edit-marker text rows [n][D]: a prompt's n-th masked span.
    markers: []f32 = &.{},

    pub fn deinit(a: *Adapter) void {
        for (a.bufs.items) |*b| b.free();
        a.bufs.deinit(a.gpa);
        a.gpa.free(a.end_up);
        a.gpa.free(a.end_head);
        a.gpa.free(a.end_rows);
        a.gpa.free(a.markers);
    }

    /// The span-end logits of a row from its normed output and the head's down product: fp32, softcapped as the head's.
    pub fn endLogits(a: *const Adapter, xn: *const [D]f32, lo: []const f32, out: *[C]f32) void {
        for (out, a.end_head, 0..) |*o, *row, cb| {
            var acc: f32 = 0;
            for (xn, row) |x, w| acc = @mulAdd(f32, w, x, acc);
            var shift: f32 = 0;
            if (a.head.rank != 0) for (lo[0..a.head.rank], a.end_up[cb * a.head.rank ..][0..a.head.rank]) |x, w| {
                shift = @mulAdd(f32, w, x, shift);
            };
            o.* = cfg.softcap * std.math.tanh((acc + shift) / cfg.softcap);
        }
    }
};

const Loader = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    a: *const pth.Archive,
    out: *Adapter,

    fn device(l: *Loader, bytes: []const u8) !u64 {
        var b = try cuda.DeviceBuffer.alloc(l.d, @max(bytes.len, 256));
        errdefer b.free();
        try b.upload(0, bytes);
        try l.out.bufs.append(l.gpa, b);
        l.out.bytes += bytes.len;
        return b.ptr;
    }

    /// fp32 values that are bf16 values, as bf16: the kernels' weight type. Any other value is refused, not rounded.
    fn bf16(l: *Loader, x: []const f32) ![]u8 {
        const raw = try l.gpa.alloc(u8, x.len * 2);
        errdefer l.gpa.free(raw);
        for (x, 0..) |v, i| {
            const bits: u32 = @bitCast(v);
            if (bits & 0xffff != 0) return error.AdapterNotBf16;
            std.mem.writeInt(u16, raw[i * 2 ..][0..2], @intCast(bits >> 16), .little);
        }
        return raw;
    }

    fn name(buf: []u8, comptime f: []const u8, args: anytype) []const u8 {
        return std.fmt.bufPrint(buf, f, args) catch unreachable;
    }

    /// `factors/<module>`'s pair for a weight [n_out, n_in] (experts: [blocks, n_out, n_in]); absent: none.
    fn pair(l: *Loader, module: []const u8, blocks: u64, n_out: u64, n_in: u64, interleaved: bool) !Pair {
        var nb: [200]u8 = undefined;
        const up_t = l.a.tensors.get(name(&nb, "factors/{s}/up", .{module})) orelse return .{};
        const lead: usize = @intFromBool(blocks != 0);
        if (up_t.shape.len != 2 + lead) return error.ShapeMismatch;
        const rank = up_t.shape[1 + lead];
        if (rank == 0 or rank > max_rank or rank % 8 != 0) return error.UnsupportedAdapterRank;
        const up_shape: []const u64 = if (blocks != 0) &.{ blocks, n_out, rank } else &.{ n_out, rank };
        const down_shape: []const u64 = if (blocks != 0) &.{ blocks, rank, n_in } else &.{ rank, n_in };
        const up = try l.a.readF32(l.gpa, name(&nb, "factors/{s}/up", .{module}), up_shape);
        defer l.gpa.free(up);
        const down = try l.a.readF32(l.gpa, name(&nb, "factors/{s}/down", .{module}), down_shape);
        defer l.gpa.free(down);
        if (interleaved) {
            // lingua's experts interleave gate and up rows; the served weight holds the gate rows, then the up rows
            const tmp = try l.gpa.dupe(f32, up);
            defer l.gpa.free(tmp);
            const half = n_out / 2;
            for (0..@max(blocks, 1)) |b| for (0..half) |j| for (0..2) |part| {
                const from = (b * n_out + 2 * j + part) * rank;
                const to = (b * n_out + part * half + j) * rank;
                @memcpy(up[to..][0..rank], tmp[from..][0..rank]);
            };
        }
        const up_raw = try l.bf16(up);
        defer l.gpa.free(up_raw);
        const down_raw = try l.bf16(down);
        defer l.gpa.free(down_raw);
        return .{ .down = try l.device(down_raw), .up = try l.device(up_raw), .rank = @intCast(rank) };
    }
};

/// The adapter at `path` on the device; `model` is the release's model.pth, whose router biases the shifts add to.
pub fn load(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, path: []const u8, model: []const u8) !*Adapter {
    var archive = try pth.Archive.openNested(gpa, io, path);
    defer archive.close();
    var base = try pth.Archive.open(gpa, io, model);
    defer base.close();
    const out = try gpa.create(Adapter);
    errdefer gpa.destroy(out);
    out.* = .{ .gpa = gpa };
    errdefer out.deinit();
    var l: Loader = .{ .gpa = gpa, .d = d, .a = &archive, .out = out };
    var nb: [160]u8 = undefined;
    var known: usize = 0;
    for (&out.layers, 0..) |*L, li| {
        const at = "layers.{d}.attention.";
        const ff = "layers.{d}.feed_forward.";
        const inner: u64 = if (cfg.usesGdn(li)) cfg.gdn_inner else cfg.attn_inner;
        L.in_proj = try l.pair(Loader.name(&nb, at ++ "in_proj.weight", .{li}), 0, cfg.gdn_proj, D, false);
        L.wq = try l.pair(Loader.name(&nb, at ++ "wq.weight", .{li}), 0, cfg.attn_inner, D, false);
        L.wkv = try l.pair(Loader.name(&nb, at ++ "wkv.weight", .{li}), 0, 2 * cfg.kv_width, D, false);
        L.gater = try l.pair(Loader.name(&nb, at ++ "gater.weight", .{li}), 0, cfg.heads, D, false);
        L.wo = try l.pair(Loader.name(&nb, at ++ "wo.weight", .{li}), 0, D, inner, false);
        L.w_in = try l.pair(Loader.name(&nb, ff ++ "w_in.weight", .{li}), 0, 2 * cfg.ffn_hidden, D, false);
        L.w_out = try l.pair(Loader.name(&nb, ff ++ "w_out.weight", .{li}), 0, D, cfg.ffn_hidden, false);
        L.router = try l.pair(Loader.name(&nb, ff ++ "router.router_linear.weight", .{li}), 0, cfg.experts, D, false);
        L.latent_down = try l.pair(Loader.name(&nb, ff ++ "latent_down.weight", .{li}), 0, cfg.moe_latent, D, false);
        L.latent_up = try l.pair(Loader.name(&nb, ff ++ "latent_up.weight", .{li}), 0, D, cfg.moe_latent, false);
        L.gate_up = try l.pair(Loader.name(&nb, ff ++ "experts.w13", .{li}), cfg.experts, 2 * cfg.moe_hidden, cfg.moe_latent, true);
        L.down = try l.pair(Loader.name(&nb, ff ++ "experts.w2", .{li}), cfg.experts, cfg.moe_latent, cfg.moe_hidden, false);
        for ([_]Pair{ L.in_proj, L.wq, L.wkv, L.gater, L.wo, L.w_in, L.w_out, L.router, L.latent_down, L.latent_up, L.gate_up, L.down }) |p| known += @intFromBool(p.rank != 0);
        const key = Loader.name(&nb, "exact/" ++ ff ++ "router.balancing_biases", .{li});
        if (archive.tensors.get(key) == null) continue;
        const shift = try archive.readF32(gpa, key, &.{cfg.experts});
        defer gpa.free(shift);
        const bias = try base.readF32(gpa, Loader.name(&nb, ff ++ "router.balancing_biases", .{li}), &.{cfg.experts});
        defer gpa.free(bias);
        for (bias, shift) |*b, s| b.* += s;
        L.bias = try l.device(std.mem.sliceAsBytes(bias));
        known += 1;
    }
    // the head was trained with the span-end id's row in every codebook's block: that row's factors stay on the host
    const grown = cfg.codebooks * (cfg.audio_vocab + 1);
    if (archive.tensors.get("factors/multi_output.weight/up")) |t| {
        if (t.shape.len != 2 or t.shape[0] != grown or t.shape[1] == 0 or t.shape[1] > max_rank or t.shape[1] % 8 != 0) return error.ShapeMismatch;
        const rank: usize = @intCast(t.shape[1]);
        const up = try archive.readF32(gpa, "factors/multi_output.weight/up", &.{ grown, rank });
        defer gpa.free(up);
        const down = try archive.readF32(gpa, "factors/multi_output.weight/down", &.{ rank, D });
        defer gpa.free(down);
        const own = try gpa.alloc(f32, cfg.head_rows * rank);
        defer gpa.free(own);
        out.end_up = try gpa.alloc(f32, C * rank);
        for (0..C) |cb| {
            const block = up[cb * (cfg.audio_vocab + 1) * rank ..];
            @memcpy(own[cb * cfg.audio_vocab * rank ..][0 .. cfg.audio_vocab * rank], block[0 .. cfg.audio_vocab * rank]);
            @memcpy(out.end_up[cb * rank ..][0..rank], block[cfg.audio_vocab * rank ..][0..rank]);
        }
        const up_raw = try l.bf16(own);
        defer gpa.free(up_raw);
        const down_raw = try l.bf16(down);
        defer gpa.free(down_raw);
        out.head = .{ .down = try l.device(down_raw), .up = try l.device(up_raw), .rank = rank };
        known += 1;
    }
    // every factor and shift the file holds must be one this port applies
    var it = archive.tensors.keyIterator();
    var held: usize = 0;
    while (it.next()) |k| held += @intFromBool(std.mem.startsWith(u8, k.*, "exact/") or (std.mem.startsWith(u8, k.*, "factors/") and std.mem.endsWith(u8, k.*, "/up")));
    if (held != known) return error.UnsupportedAdapterTarget;

    out.end_head = try rows(gpa, &archive, "grown/rows/multi_output.weight", C);
    out.end_rows = try gpa.alloc([D]f32, C);
    for (out.end_rows, 0..) |*row, cb| {
        const one = try rows(gpa, &archive, Loader.name(&nb, "grown/rows/multi_embedder.embedders.{d}.weight", .{cb}), 1);
        defer gpa.free(one);
        row.* = one[0];
    }
    const text = try archive.tensor(Loader.name(&nb, "grown/rows/multi_embedder.embedders.{d}.weight", .{C}));
    if (text.shape.len != 2 or text.shape[0] == 0 or text.shape[1] != D) return error.ShapeMismatch;
    out.markers = try archive.readF32(gpa, Loader.name(&nb, "grown/rows/multi_embedder.embedders.{d}.weight", .{C}), text.shape);
    for (out.markers) |x| if (!std.math.isFinite(x)) return error.BadAdapterRows;
    return out;
}

fn rows(gpa: std.mem.Allocator, a: *const pth.Archive, key: []const u8, n: usize) ![][D]f32 {
    const flat = try a.readF32(gpa, key, &.{ n, D });
    defer gpa.free(flat);
    const out = try gpa.alloc([D]f32, n);
    for (out, 0..) |*row, i| {
        row.* = flat[i * D ..][0..D].*;
        for (row) |x| if (!std.math.isFinite(x)) return error.BadAdapterRows;
    }
    return out;
}

/// Device bytes of the adapter at `path`: its factors as bf16 and its bias shifts, read from the archive's index.
pub fn deviceBytes(gpa: std.mem.Allocator, io: std.Io, path: []const u8) !usize {
    var archive = try pth.Archive.openNested(gpa, io, path);
    defer archive.close();
    var total: usize = 0;
    var it = archive.tensors.iterator();
    while (it.next()) |entry| {
        var n: usize = 1;
        for (entry.value_ptr.shape) |s| n *= @intCast(s);
        const k = entry.key_ptr.*;
        if (std.mem.startsWith(u8, k, "exact/")) total += @max(n * 4, 256);
        if (!std.mem.startsWith(u8, k, "factors/")) continue;
        // the head's span-end rows stay on the host
        if (std.mem.eql(u8, k, "factors/multi_output.weight/up")) n -= C * @as(usize, @intCast(entry.value_ptr.shape[1]));
        total += @max(n * 2, 256);
    }
    return total;
}
