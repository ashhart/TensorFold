//! GLM-5.3-Flash's prepared weights for one rank of two, read from the converted checkpoint (glm-affine-experts-v1:
//! MLX affine 4-bit routed experts in groups of 64 with fp16 scales and biases, everything else BF16 / FP32), as the
//! Python engine prepares them (families/glm5_next/cuda/weights.py, its affine-experts path, split.py's rank rules):
//! every tensor under the name, shape and bytes of the Python engine's prepared weights (its weight tree, the names
//! tools/capture_ref.py hashes), uploaded to this rank's GPU.
//!
//! The derivations are exact: rank rows and columns cut on the host, BF16 rows stacked, the kv_b blocks split per head
//! (the latent path's absorb weights), fp16 expert scales rounded to bf16 (round to nearest even, as torch), and the
//! experts' blocks in experts_pack.cu's index map. The draft head is the verification head itself (no 4-bit copy).

const std = @import("std");
const cuda = @import("cuda");
pub const api = @import("api.zig");
pub const kern = @import("kern.zig");
/// The forward and its engine API (tf-cuda-test glm-ref reaches it through this module).
pub const forward = @import("forward.zig");
pub const decode = @import("decode.zig");
pub const vision = @import("vision.zig");

const BASE = "model.language_model.";
const WORLD = 2;

// ---------------------------------------------------------------------------------------------------------------
// The checkpoint: its shards' headers, files opened once, tensors read whole or as row ranges.

const Entry = struct { dtype: []const u8, shape: []const i64, begin: u64, end: u64, file: usize };

const Ckpt = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    arena: std.heap.ArenaAllocator,
    files: std.ArrayList(std.Io.File) = .empty,
    paths: std.ArrayList([]const u8) = .empty,
    entries: std.StringHashMap(Entry),

    /// ``dir``: the served folder. A shard (or the index) that is a symlink to a container path under /cache/tf/
    /// resolves against the folder's parent directory, so the same folder serves inside the container and on the host.
    fn open(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !Ckpt {
        var ck: Ckpt = .{ .gpa = gpa, .io = io, .arena = .init(gpa), .entries = .init(gpa) };
        errdefer ck.deinit();
        const a = ck.arena.allocator();
        const index_path = try resolve(a, io, dir, "model.safetensors.index.json");
        const text = try std.Io.Dir.cwd().readFileAlloc(io, index_path, a, .limited(1 << 26));
        const idx = try std.json.parseFromSliceLeaky(std.json.Value, a, text, .{});
        var shard_of = std.StringHashMap(usize).init(a);
        var it = idx.object.get("weight_map").?.object.iterator();
        while (it.next()) |e| {
            const shard = e.value_ptr.string;
            if (shard_of.contains(shard)) continue;
            const path = try resolve(a, io, dir, shard);
            try shard_of.put(shard, ck.files.items.len);
            try ck.files.append(gpa, try std.Io.Dir.cwd().openFile(io, path, .{}));
            try ck.paths.append(gpa, path);
            try ck.readHeader(ck.files.items.len - 1);
        }
        return ck;
    }

    fn readHeader(ck: *Ckpt, f: usize) !void {
        const a = ck.arena.allocator();
        var head: [8]u8 = undefined;
        _ = try ck.files.items[f].readPositionalAll(ck.io, &head, 0);
        const n = std.mem.readInt(u64, &head, .little);
        const text = try a.alloc(u8, n);
        _ = try ck.files.items[f].readPositionalAll(ck.io, text, 8);
        const v = try std.json.parseFromSliceLeaky(std.json.Value, a, text, .{});
        var it = v.object.iterator();
        while (it.next()) |e| {
            if (std.mem.eql(u8, e.key_ptr.*, "__metadata__")) continue;
            const o = e.value_ptr.object;
            const shp = o.get("shape").?.array.items;
            const shape = try a.alloc(i64, shp.len);
            for (shp, 0..) |x, i| shape[i] = x.integer;
            const offs = o.get("data_offsets").?.array.items;
            try ck.entries.put(e.key_ptr.*, .{ .dtype = o.get("dtype").?.string, .shape = shape, .begin = 8 + n + @as(u64, @intCast(offs[0].integer)), .end = 8 + n + @as(u64, @intCast(offs[1].integer)), .file = f });
        }
    }

    fn deinit(ck: *Ckpt) void {
        // the shards are read once: their page cache would starve NCCL's first registration (and the n-gram tables)
        for (ck.files.items) |f| {
            _ = std.os.linux.fadvise(f.handle, 0, 0, std.os.linux.POSIX_FADV.DONTNEED);
            f.close(ck.io);
        }
        ck.files.deinit(ck.gpa);
        ck.paths.deinit(ck.gpa);
        ck.entries.deinit();
        ck.arena.deinit();
    }

    fn has(ck: *const Ckpt, name: []const u8) bool {
        return ck.entries.contains(name);
    }

    fn get(ck: *const Ckpt, name: []const u8) !Entry {
        return ck.entries.get(name) orelse {
            std.log.err("glm weights: the checkpoint has no tensor {s}", .{name});
            return error.MissingTensor;
        };
    }

    /// ``len`` bytes at ``off`` into the tensor's data.
    fn readAt(ck: *const Ckpt, e: Entry, off: u64, out: []u8) !void {
        if (off + out.len > e.end - e.begin) return error.ShortRead;
        const got = try ck.files.items[e.file].readPositionalAll(ck.io, out, e.begin + off);
        if (got != out.len) return error.ShortRead;
    }

    fn readAll(ck: *const Ckpt, gpa: std.mem.Allocator, name: []const u8) ![]align(16) u8 {
        const e = try ck.get(name);
        const buf = try gpa.alignedAlloc(u8, .@"16", e.end - e.begin);
        errdefer gpa.free(buf);
        try ck.readAt(e, 0, buf);
        return buf;
    }
};

fn resolve(a: std.mem.Allocator, io: std.Io, dir: []const u8, name: []const u8) ![]const u8 {
    const path = try std.fs.path.join(a, &.{ dir, name });
    if (std.Io.Dir.cwd().openFile(io, path, .{})) |f| {
        f.close(io);
        return path;
    } else |_| {}
    var buf: [4096]u8 = undefined;
    const n = std.Io.Dir.cwd().readLink(io, path, &buf) catch return path;
    const target = buf[0..n];
    const container = "/cache/tf/";
    if (std.mem.startsWith(u8, target, container))
        return std.fs.path.join(a, &.{ std.fs.path.dirname(dir) orelse ".", target[container.len..] });
    if (std.fs.path.isAbsolute(target)) return a.dupe(u8, target);
    return std.fs.path.join(a, &.{ dir, target });
}

fn bf16ToF32(b: u16) f32 {
    return @bitCast(@as(u32, b) << 16);
}

fn f32ToBf16(x: f32) u16 {
    const u: u32 = @bitCast(x);
    if (std.math.isNan(x)) return 0x7FC0;
    return @intCast((u + 0x7FFF + ((u >> 16) & 1)) >> 16);
}

fn f16ToBf16(h: u16) u16 {
    return f32ToBf16(@floatCast(@as(f16, @bitCast(h))));
}

fn numelOf(shape: []const i64) usize {
    var n: usize = 1;
    for (shape) |d| n *= @intCast(d);
    return n;
}

fn dtypeOf(s: []const u8) !api.DType {
    if (std.mem.eql(u8, s, "BF16")) return .bf16;
    if (std.mem.eql(u8, s, "F32")) return .f32;
    if (std.mem.eql(u8, s, "F16")) return .f16;
    return error.UnsupportedDType;
}

// ---------------------------------------------------------------------------------------------------------------
// split.py's rules: a checkpoint tensor's rank part (rows, the last axis' columns, or the whole tensor).

const Split = enum { rep, row, col };

fn splitOf(name: []const u8) Split {
    const ends = std.mem.endsWith;
    const has = struct {
        fn f(n: []const u8, frag: []const u8) bool {
            return std.mem.indexOf(u8, n, frag) != null;
        }
    }.f;
    if (has(name, ".mlp.experts.") or has(name, ".mlp.shared_experts.") or has(name, ".mlp.gate_proj.") or has(name, ".mlp.up_proj.") or has(name, ".mlp.down_proj."))
        return if (has(name, "down_proj.")) .col else .row;
    if (has(name, ".self_attn.indexer.")) return .rep;
    for ([_][]const u8{ ".self_attn.q_proj.", ".self_attn.k_proj.", ".self_attn.v_proj.", "_conv1d.", ".self_attn.f_b_proj.", ".self_attn.g_b_proj.", ".self_attn.b_proj.", ".self_attn.q_b_proj.", ".self_attn.kv_b_proj." }) |f| if (has(name, f)) return .row;
    if (ends(u8, name, ".self_attn.A_log") or ends(u8, name, ".self_attn.dt_bias")) return .row;
    if (has(name, ".self_attn.o_proj.")) return .col;
    return .rep;
}

const Builder = struct {
    gpa: std.mem.Allocator,
    ck: *const Ckpt,
    dry: bool,
    rank: usize,
    store: ?*api.Store = null,
    d: ?*const cuda.Driver = null,
    bytes: u64 = 0,
    tensors: usize = 0,
    /// Dense linears (all but the indexer's) and the head packed to these bits (0: bf16 as stored): see ``quantize``.
    dense_bits: u8 = 0,
    head_bits: u8 = 0,
    /// TF_GLM_KDA_PROJ_BITS: the KDA layers' stacked in-projection (the largest dense linear, 1.3 GB a rank at 6 bits)
    /// at its own width (0: dense_bits).
    kda_proj_bits: u8 = 0,
    /// A packed copy of the head the MTP drafts read (0: they read the head).
    draft_head_bits: u8 = 0,
    /// The routed experts' groups as scales alone (their biases -8 x the scales, checked): TF_GLM_EXPERTS_SYM=0 keeps
    /// the affine blocks.
    experts_sym: bool = true,
    /// Development (TF_GLM_EXPERT_SIM_BITS=2 or 3): the experts re-quantized to fewer levels in their 4-bit words, to
    /// measure a narrower expert format's quality before its kernel exists.
    expert_sim_bits: u8 = 0,
    /// TF_GLM_EXPERT_BITS=3: the experts re-quantized to 3-bit codes and packed as glm/experts_sym3's planes.
    experts3: bool = false,
    /// TF_GLM_EXPERT3_PROJ=down|up: 3-bit codes for that projection alone (the other stays 4-bit in the planes: a
    /// sensitivity probe; the kernel reads 3-bit planes either way, so "4-bit" here means 8 levels of the 4-bit
    /// value re-rounded). TF_GLM_EXPERT_SIM_BITS probes quality on the 4-bit kernels instead: with
    /// TF_GLM_EXPERT_SIM_PROJ=down|up it re-quantizes that projection alone.
    sim_proj: u8 = 0, // 0: all, 1: down only, 2: gate/up only
    /// The checkpoint's routed experts are EXL3 (4-bit "mcg" trellises, every other weight BF16): read as stored.
    exl3: bool = false,
    exl3_shx: bool = true, // every layer's experts share one input rotation for gate and up (cleared by a layer that does not)
    suh_first: [2 * 4096]u8 = undefined, // the first expert's gate suh of the layer being read

    fn put(b: *Builder, name: []const u8, dtype: api.DType, shape: []const i64, data: ?[]const u8) !void {
        var t: api.Tensor = .{ .ptr = 0, .dtype = dtype, .nd = @intCast(shape.len) };
        for (shape, 0..) |x, i| t.shape[i] = x;
        const n = t.bytes();
        b.bytes += n;
        b.tensors += 1;
        if (b.dry) return;
        const store = b.store.?;
        var buf = try cuda.DeviceBuffer.alloc(b.d.?, @max(n, 1));
        errdefer buf.free();
        if (n > 0) try buf.upload(0, data.?[0..n]);
        t.ptr = buf.ptr;
        try store.owned.append(store.gpa, buf);
        try store.put(name, t);
    }

    fn putf(b: *Builder, comptime fmt: []const u8, args: anytype, dtype: api.DType, shape: []const i64, data: ?[]const u8) !void {
        var nb: [256]u8 = undefined;
        try b.put(try std.fmt.bufPrint(&nb, fmt, args), dtype, shape, data);
    }

    /// A checkpoint tensor's rank part as stored: its dtype, rank shape and bytes (null in a dry run).
    const Part = struct { dtype: api.DType, shape: [4]i64, nd: usize, data: ?[]align(16) u8 };

    fn part(b: *Builder, full: []const u8, how: Split) !Part {
        const e = try b.ck.get(full);
        var p: Part = .{ .dtype = try dtypeOf(e.dtype), .shape = .{ 1, 1, 1, 1 }, .nd = e.shape.len, .data = null };
        for (e.shape, 0..) |x, i| p.shape[i] = x;
        const elem = api.dtypeBytes(p.dtype);
        const rows: usize = @intCast(p.shape[0]);
        const inner: usize = numelOf(e.shape) / rows;
        switch (how) {
            .rep => {
                if (!b.dry) p.data = try b.ck.readAll(b.gpa, full);
            },
            .row => {
                const per = rows / WORLD;
                p.shape[0] = @intCast(per);
                if (!b.dry) {
                    p.data = try b.gpa.alignedAlloc(u8, .@"16", per * inner * elem);
                    try b.ck.readAt(e, b.rank * per * inner * elem, p.data.?);
                }
            },
            .col => {
                if (e.shape.len != 2) return error.ColumnSplitNeeds2D;
                const k: usize = @intCast(e.shape[1]);
                const ck = k / WORLD;
                p.shape[1] = @intCast(ck);
                if (!b.dry) {
                    const all = try b.ck.readAll(b.gpa, full);
                    defer b.gpa.free(all);
                    p.data = try b.gpa.alignedAlloc(u8, .@"16", rows * ck * elem);
                    for (0..rows) |r| @memcpy(p.data.?[r * ck * elem ..][0 .. ck * elem], all[(r * k + b.rank * ck) * elem ..][0 .. ck * elem]);
                }
            },
        }
        return p;
    }

    fn free(b: *Builder, p: Part) void {
        if (p.data) |x| b.gpa.free(x);
    }

    /// One checkpoint tensor (its rank part) under ``out``, cast to fp32 when ``f32`` (bf16 -> fp32 is exact).
    fn tensor(b: *Builder, out: []const u8, full: []const u8, f32_out: bool) !void {
        const p = try b.part(full, splitOf(full));
        defer b.free(p);
        if (f32_out and p.dtype == .bf16) {
            const n = numelOf(p.shape[0..p.nd]);
            if (b.dry) return b.put(out, .f32, p.shape[0..p.nd], null);
            const f = try b.gpa.alloc(f32, n);
            defer b.gpa.free(f);
            const h = std.mem.bytesAsSlice(u16, p.data.?);
            for (f, 0..) |*o, i| o.* = bf16ToF32(h[i]);
            return b.put(out, .f32, p.shape[0..p.nd], std.mem.sliceAsBytes(f));
        }
        try b.put(out, p.dtype, p.shape[0..p.nd], p.data);
    }

    /// BF16 linears of the same K, their rank rows stacked in order -> ``out``.weight [sum n, k] (stack_b16).
    fn stack(b: *Builder, out: []const u8, fulls: []const []const u8) !void {
        var parts: [8]Part = undefined;
        var n: i64 = 0;
        var kk: i64 = 0;
        for (fulls, 0..) |f, i| {
            parts[i] = try b.part(f, splitOf(f));
            if (parts[i].dtype != .bf16 or parts[i].nd != 2) return error.NotBf16Matrix;
            n += parts[i].shape[0];
            kk = parts[i].shape[1];
        }
        defer for (parts[0..fulls.len]) |p| b.free(p);
        var nb: [256]u8 = undefined;
        var bits: u8 = if (std.mem.indexOf(u8, out, "index.") != null or @mod(kk, 64) != 0) 0 else b.dense_bits;
        if (b.kda_proj_bits != 0 and fulls.len == 6) bits = b.kda_proj_bits; // the KDA in-projection: 6 stacked linears
        const name = try std.fmt.bufPrint(&nb, "{s}.weight", .{out});
        if (b.dry and bits == 0) return b.put(name, .bf16, &.{ n, kk }, null);
        if (b.dry) return b.putPacked(out, bits, n, kk, null);
        const all = try b.gpa.alignedAlloc(u8, .@"16", @intCast(n * kk * 2));
        defer b.gpa.free(all);
        var at: usize = 0;
        for (parts[0..fulls.len]) |p| {
            @memcpy(all[at..][0..p.data.?.len], p.data.?);
            at += p.data.?.len;
        }
        if (bits != 0) return b.putPacked(out, bits, n, kk, all);
        try b.put(name, .bf16, &.{ n, kk }, all);
    }

    /// bf16 rows [n, kk] -> ``out``.qw (u8 [n, kk * bits / 8]) and ``out``.qs (fp16 [n, kk / 64]).
    fn putPacked(b: *Builder, out: []const u8, bits: u8, n: i64, kk: i64, rows: ?[]align(16) const u8) !void {
        const nu: usize = @intCast(n);
        const ku: usize = @intCast(kk);
        var nb: [256]u8 = undefined;
        var nb2: [256]u8 = undefined;
        const qn = try std.fmt.bufPrint(&nb, "{s}.qw", .{out});
        const sn = try std.fmt.bufPrint(&nb2, "{s}.qs", .{out});
        const wshape = [_]i64{ n, @divExact(kk * bits, 8) };
        const sshape = [_]i64{ n, @divExact(kk, 64) };
        if (rows == null) {
            try b.put(qn, .u8, &wshape, null);
            return b.put(sn, .f16, &sshape, null);
        }
        const words = try b.gpa.alloc(u32, nu * ku * bits / 32);
        defer b.gpa.free(words);
        const scales = try b.gpa.alloc(u16, nu * ku / 64);
        defer b.gpa.free(scales);
        quantize(bits, std.mem.bytesAsSlice(u16, rows.?), nu, ku, words, scales);
        try b.put(qn, .u8, &wshape, std.mem.sliceAsBytes(words));
        try b.put(sn, .f16, &sshape, std.mem.sliceAsBytes(scales));
    }

    fn linear(b: *Builder, out: []const u8, full: []const u8) !void {
        try b.stack(out, &.{full});
    }
};

fn fullName(buf: []u8, comptime fmt: []const u8, args: anytype) ![]const u8 {
    return std.fmt.bufPrint(buf, BASE ++ fmt, args);
}

// ---------------------------------------------------------------------------------------------------------------
// The routed experts: per-expert MLX triples, the rank's rows (gate, up) or columns (down), packed as experts_pack.cu.

const Triple = struct {
    e: usize,
    n: usize,
    k8: usize,
    kg: usize,
    words: []u32,
    scales: []u16,
    biases: []u16,

    fn free(t: Triple, gpa: std.mem.Allocator) void {
        gpa.free(t.words);
        gpa.free(t.scales);
        gpa.free(t.biases);
    }
};

/// Every expert's ``proj`` of layer ``layer``: rows r0..r1 (gate, up) or the rank's columns (down), scales and biases
/// rounded from fp16 to bf16.
fn readExperts(b: *Builder, li: usize, proj: []const u8, count: usize) !Triple {
    var nb: [256]u8 = undefined;
    const w0 = try b.ck.get(try fullName(&nb, "layers.{d}.mlp.experts.0.{s}.weight", .{ li, proj }));
    const n_full: usize = @intCast(w0.shape[0]);
    const k8_full: usize = @intCast(w0.shape[1]);
    var nb2: [256]u8 = undefined;
    const s0 = try b.ck.get(try fullName(&nb2, "layers.{d}.mlp.experts.0.{s}.scales", .{ li, proj }));
    const kg_full: usize = @intCast(s0.shape[1]);
    const down = std.mem.eql(u8, proj, "down_proj");
    const nr = if (down) n_full else n_full / WORLD;
    const r0 = if (down) 0 else b.rank * nr;
    const ok8 = if (down) k8_full / WORLD else k8_full;
    const okg = if (down) kg_full / WORLD else kg_full;
    const c8 = if (down) b.rank * ok8 else 0;
    const cg = if (down) b.rank * okg else 0;
    var t: Triple = .{ .e = count, .n = nr, .k8 = ok8, .kg = okg, .words = try b.gpa.alloc(u32, count * nr * ok8), .scales = try b.gpa.alloc(u16, count * nr * okg), .biases = try b.gpa.alloc(u16, count * nr * okg) };
    errdefer t.free(b.gpa);
    const rowbuf = try b.gpa.alloc(u8, nr * k8_full * 4);
    defer b.gpa.free(rowbuf);
    const sbuf = try b.gpa.alloc(u8, nr * kg_full * 2);
    defer b.gpa.free(sbuf);
    for (0..count) |x| {
        var n1: [256]u8 = undefined;
        const we = try b.ck.get(try fullName(&n1, "layers.{d}.mlp.experts.{d}.{s}.weight", .{ li, x, proj }));
        try b.ck.readAt(we, r0 * k8_full * 4, rowbuf);
        const w32 = std.mem.bytesAsSlice(u32, rowbuf);
        for (0..nr) |r| @memcpy(t.words[(x * nr + r) * ok8 ..][0..ok8], w32[r * k8_full + c8 ..][0..ok8]);
        for ([_][]const u8{ "scales", "biases" }, 0..) |what, which| {
            var n2: [256]u8 = undefined;
            const ent = try b.ck.get(try fullName(&n2, "layers.{d}.mlp.experts.{d}.{s}.{s}", .{ li, x, proj, what }));
            try b.ck.readAt(ent, r0 * kg_full * 2, sbuf);
            const s16 = std.mem.bytesAsSlice(u16, sbuf);
            const dst = if (which == 0) t.scales else t.biases;
            for (0..nr) |r| for (0..okg) |g| {
                dst[(x * nr + r) * okg + g] = f16ToBf16(s16[r * kg_full + cg + g]);
            };
        }
    }
    return t;
}

/// experts_pack.cu's index map: each (group, column block, expert) block holds 32 * 4 * H weight words (nibbles
/// reordered i0 i2 i4 i6 i1 i3 i5 i7) then 32 scale/bias words; written as slot ``m`` of ``mcount`` per block.
/// Every group's bias is -8 x its scale (a symmetric int4 checkpoint as affine groups): the symmetric blocks hold
/// the scales alone.
fn checkSymmetric(t: Triple) !void {
    for (t.scales, t.biases) |sc, bi| {
        if (bf16Value(bi) != -8 * bf16Value(sc)) return error.ExpertsNotSymmetric;
    }
}

fn packExperts(t: Triple, gs: usize, out: []u32, mcount: usize, m: usize, sym: bool, three: bool) void {
    const h = gs / 32;
    const Work = struct {
        fn run(tt: Triple, hh: usize, o: []u32, mc: usize, mm: usize, e0: usize, e1: usize, sy: bool, th: bool) void {
            const wp = 32 * 4 * hh;
            const wsec = if (th) wp * 3 / 4 else wp;
            const blk = wsec + @as(usize, if (sy) 16 else 32);
            var tmp: [512]u32 = undefined; // a 3-bit block's 4-bit words before their planes
            const nbb = tt.n / 32;
            for (e0..e1) |e| for (0..nbb) |bb| for (0..tt.kg) |g| {
                const dst = o[(((e * nbb + bb) * tt.kg + g) * mc + mm) * blk ..][0..blk];
                const wdst = if (th) tmp[0..wp] else dst[0..wp];
                for (0..wp) |i| {
                    const row = i / (4 * hh);
                    const kk = i % (4 * hh);
                    const tq = row / 8;
                    const r = row % 8;
                    const q = kk / hh;
                    const j = kk % hh;
                    const w = tt.words[(e * tt.n + bb * 32 + row) * tt.k8 + g * 4 * hh + kk];
                    const f = ((tq * hh + j) * 8 + r) * 4 + q;
                    wdst[(f / 128) * 128 + (f % 32) * 4 + (f / 32) % 4] = shuffle(w);
                }
                if (th) for (0..hh) |c| for (0..32) |lane| { // the lane's uint4 c as three planes (experts_sym.cuh)
                    var lo: [4]u32 = undefined;
                    var hi: [4]u32 = undefined;
                    for (0..4) |k| {
                        const word = tmp[(c * 32 + lane) * 4 + k];
                        lo[k] = 0;
                        hi[k] = 0;
                        for (0..8) |i| {
                            const code = ((word >> @intCast(4 * i)) & 15) -% 4; // 3-bit codes are stored + 4
                            lo[k] |= (code & 3) << @intCast(2 * i);
                            hi[k] |= ((code >> 2) & 1) << @intCast(i);
                        }
                    }
                    dst[c * 96 + lane] = lo[0] | (lo[1] << 16);
                    dst[c * 96 + 32 + lane] = lo[2] | (lo[3] << 16);
                    dst[c * 96 + 64 + lane] = hi[0] | (hi[1] << 8) | (hi[2] << 16) | (hi[3] << 24);
                };
                if (sy) { // quad p: rows j * 8 + 2p and + 1's scales, j = 0..3 (experts_sym.cuh)
                    for (0..16) |s| {
                        const p = s / 4;
                        const j = s % 4;
                        const at = (e * tt.n + bb * 32 + j * 8 + p * 2) * tt.kg + g;
                        dst[wsec + s] = @as(u32, tt.scales[at]) | (@as(u32, tt.scales[at + tt.kg]) << 16);
                    }
                } else for (0..32) |s| {
                    const p = s / 8;
                    const tt4 = s % 4;
                    const src = if ((s / 4) % 2 == 1) tt.biases else tt.scales;
                    const at = (e * tt.n + bb * 32 + tt4 * 8 + p * 2) * tt.kg + g;
                    dst[wp + s] = @as(u32, src[at]) | (@as(u32, src[at + tt.kg]) << 16);
                }
            };
        }
    };
    const threads = 16;
    var pool: [threads]?std.Thread = @splat(null);
    const per = (t.e + threads - 1) / threads;
    for (0..threads) |i| {
        const e0 = i * per;
        const e1 = @min(t.e, e0 + per);
        if (e0 >= e1) break;
        pool[i] = std.Thread.spawn(.{}, Work.run, .{ t, h, out, mcount, m, e0, e1, sym, three }) catch blk: {
            Work.run(t, h, out, mcount, m, e0, e1, sym, three);
            break :blk null;
        };
    }
    for (pool) |th| if (th) |x| x.join();
}

fn shuffle(w: u32) u32 {
    var even = w & 0x0F0F0F0F;
    var odd = (w >> 4) & 0x0F0F0F0F;
    even = (even | (even >> 4)) & 0x00FF00FF;
    odd = (odd | (odd >> 4)) & 0x00FF00FF;
    even = (even | (even >> 8)) & 0x0000FFFF;
    odd = (odd | (odd >> 8)) & 0x0000FFFF;
    return even | (odd << 16);
}

/// grouped.make([gate, up], down, 64): up [E, N/32, K/64, 2, block], down [E, D/32, NI/64, 1, block] int32.
fn experts(b: *Builder, out: []const u8, li: usize) !void {
    const E: usize = @intCast(kern.EXPERTS);
    const gs: usize = 64;
    if (b.experts3 and !b.experts_sym) return error.Experts3NeedsSymmetric;
    const wsec: usize = 32 * 4 * (gs / 32) * @as(usize, if (b.experts3) 3 else 4) / 4;
    const blk: usize = wsec + @as(usize, if (b.experts_sym) 16 else 32);
    const width: usize = @intCast(@divExact(2048, WORLD)); // moe_intermediate_size / ranks
    const dims: usize = @intCast(kern.D);
    const up_shape = [_]i64{ @intCast(E), @intCast(width / 32), @intCast(dims / gs), 2, @intCast(blk) };
    const down_shape = [_]i64{ @intCast(E), @intCast(dims / 32), @intCast(width / gs), 1, @intCast(blk) };
    var nb: [256]u8 = undefined;
    var nb2: [256]u8 = undefined;
    const up_name = try std.fmt.bufPrint(&nb, "{s}.up", .{out});
    const down_name = try std.fmt.bufPrint(&nb2, "{s}.down", .{out});
    if (b.dry) {
        try b.put(up_name, .i32, &up_shape, null);
        try b.put(down_name, .i32, &down_shape, null);
        return;
    }
    {
        const up = try b.gpa.alloc(u32, numelOf(&up_shape));
        defer b.gpa.free(up);
        for ([_][]const u8{ "gate_proj", "up_proj" }, 0..) |proj, m| {
            const t = try readExperts(b, li, proj, E);
            defer t.free(b.gpa);
            if (t.n != width or t.k8 * 8 != dims) return error.ExpertShape;
            if (b.experts_sym) try checkSymmetric(t);
            if (b.sim_proj != 1) {
                if (b.experts3) simulateBits(t, 3) else if (b.expert_sim_bits != 0) simulateBits(t, b.expert_sim_bits);
            }
            packExperts(t, gs, up, 2, m, b.experts_sym, b.experts3);
        }
        try b.put(up_name, .i32, &up_shape, std.mem.sliceAsBytes(up));
    }
    const dn = try b.gpa.alloc(u32, numelOf(&down_shape));
    defer b.gpa.free(dn);
    const t = try readExperts(b, li, "down_proj", E);
    defer t.free(b.gpa);
    if (t.n != dims or t.k8 * 8 != width) return error.ExpertShape;
    if (b.experts_sym) try checkSymmetric(t);
    if (b.sim_proj != 2) {
        if (b.experts3) simulateBits(t, 3) else if (b.expert_sim_bits != 0) simulateBits(t, b.expert_sim_bits);
    }
    packExperts(t, gs, dn, 1, 0, b.experts_sym, b.experts3);
    try b.put(down_name, .i32, &down_shape, std.mem.sliceAsBytes(dn));
}

/// The EXL3 routed experts of layer ``li``, this rank's whole tiles (glm/exl3.cu's layout): gate and up by output tile
/// columns, down by input tile rows, each matrix's experts stacked as trellis words [E, K/16, N/16, 32] (the stored
/// bytes) and fp16 scales [E, K] (suh) and [E, N] (svh), the rank's parts.
fn expertsExl3(b: *Builder, out: []const u8, li: usize) !void {
    const E: usize = @intCast(kern.EXPERTS);
    const D: usize = @intCast(kern.D);
    const width: usize = @intCast(@divExact(2048, WORLD));
    const tile: usize = 128; // a 16x16 tile's bytes at 4 bits
    var nb: [256]u8 = undefined;
    const Mat = struct { proj: []const u8, kt: usize, nt: usize, by_cols: bool };
    const mats = [_]Mat{ .{ .proj = "gate_proj", .kt = D / 16, .nt = width / 16, .by_cols = true }, .{ .proj = "up_proj", .kt = D / 16, .nt = width / 16, .by_cols = true }, .{ .proj = "down_proj", .kt = width / 16, .nt = D / 16, .by_cols = false } };
    const short = [_][]const u8{ "g", "u", "d" };
    for (mats, short) |m, sh| {
        const tshape = [_]i64{ @intCast(E), @intCast(m.kt), @intCast(m.nt), 32 };
        const k_all: usize = if (m.by_cols) D else 2048;
        const n_all: usize = if (m.by_cols) 2048 else D;
        const sus = [_]i64{ @intCast(E), @intCast(m.kt * 16) };
        const svs = [_]i64{ @intCast(E), @intCast(m.nt * 16) };
        const tn = try std.fmt.bufPrint(&nb, "{s}.{s}t", .{ out, sh });
        if (b.dry) {
            try b.put(tn, .i32, &tshape, null);
            try b.putf("{s}.suh_{s}", .{ out, sh }, .f16, &sus, null);
            try b.putf("{s}.svh_{s}", .{ out, sh }, .f16, &svs, null);
            continue;
        }
        const per = m.kt * m.nt * tile;
        const words = try b.gpa.alignedAlloc(u8, .@"16", E * per);
        defer b.gpa.free(words);
        const suh = try b.gpa.alloc(u16, E * m.kt * 16);
        defer b.gpa.free(suh);
        const svh = try b.gpa.alloc(u16, E * m.nt * 16);
        defer b.gpa.free(svh);
        var f: [256]u8 = undefined;
        for (0..E) |x| {
            const tr = try b.ck.get(try fullName(&f, "layers.{d}.mlp.experts.{d}.{s}.trellis", .{ li, x, m.proj }));
            if (!std.mem.eql(u8, tr.dtype, "I16") or tr.shape.len != 3 or tr.shape[2] != 64 or tr.shape[0] != k_all / 16 or tr.shape[1] != n_all / 16) return error.NotExl3Mcg4;
            if (!b.ck.has(try fullName(&f, "layers.{d}.mlp.experts.{d}.{s}.mcg", .{ li, x, m.proj }))) return error.NotExl3Mcg4;
            const dst = words[x * per ..][0..per];
            if (m.by_cols) { // every tile row, this rank's tile columns
                const all = try b.ck.readAll(b.gpa, try fullName(&f, "layers.{d}.mlp.experts.{d}.{s}.trellis", .{ li, x, m.proj }));
                defer b.gpa.free(all);
                const row = (n_all / 16) * tile;
                for (0..m.kt) |r| @memcpy(dst[r * m.nt * tile ..][0 .. m.nt * tile], all[r * row + b.rank * m.nt * tile ..][0 .. m.nt * tile]);
            } else try b.ck.readAt(tr, b.rank * per, dst); // this rank's tile rows: one contiguous part
            const se = try b.ck.get(try fullName(&f, "layers.{d}.mlp.experts.{d}.{s}.suh", .{ li, x, m.proj }));
            const ve = try b.ck.get(try fullName(&f, "layers.{d}.mlp.experts.{d}.{s}.svh", .{ li, x, m.proj }));
            if (!std.mem.eql(u8, se.dtype, "F16") or !std.mem.eql(u8, ve.dtype, "F16")) return error.NotExl3Mcg4;
            const su = std.mem.sliceAsBytes(suh[x * m.kt * 16 ..][0 .. m.kt * 16]);
            const sv = std.mem.sliceAsBytes(svh[x * m.nt * 16 ..][0 .. m.nt * 16]);
            try b.ck.readAt(se, if (m.by_cols) 0 else b.rank * su.len, su);
            try b.ck.readAt(ve, if (m.by_cols) b.rank * sv.len else 0, sv);
            if (m.by_cols) { // gate and up: one input rotation shared by every expert of the layer? (exl3_shx)
                if (std.mem.eql(u8, sh, "g") and x == 0) @memcpy(b.suh_first[0..su.len], su) else if (!std.mem.eql(u8, su, b.suh_first[0..su.len])) b.exl3_shx = false;
            }
        }
        try b.put(tn, .i32, &tshape, words);
        try b.putf("{s}.suh_{s}", .{ out, sh }, .f16, &sus, std.mem.sliceAsBytes(suh));
        try b.putf("{s}.svh_{s}", .{ out, sh }, .f16, &svs, std.mem.sliceAsBytes(svh));
    }
}

// ---------------------------------------------------------------------------------------------------------------
// The layers, as weights.py's load() builds them (names: its weight tree).

fn hc(b: *Builder, out: []const u8, li: usize, site: []const u8) !void {
    var o: [128]u8 = undefined;
    var f: [256]u8 = undefined;
    try b.tensor(try std.fmt.bufPrint(&o, "{s}.fn", .{out}), try fullName(&f, "layers.{d}.hc_{s}_fn", .{ li, site }), false);
    try b.tensor(try std.fmt.bufPrint(&o, "{s}.base", .{out}), try fullName(&f, "layers.{d}.hc_{s}_base", .{ li, site }), true);
    try b.tensor(try std.fmt.bufPrint(&o, "{s}.scale", .{out}), try fullName(&f, "layers.{d}.hc_{s}_scale", .{ li, site }), true);
}

fn kda(b: *Builder, out: []const u8, i: usize) !void {
    var o: [160]u8 = undefined;
    var n: [6][256]u8 = undefined;
    var names: [6][]const u8 = undefined;
    for ([_][]const u8{ "q_proj", "k_proj", "v_proj", "f_a_proj", "g_a_proj", "b_proj" }, 0..) |p, j|
        names[j] = try fullName(&n[j], "layers.{d}.self_attn.{s}.weight", .{ i, p });
    try b.stack(try std.fmt.bufPrint(&o, "{s}.proj", .{out}), &names);
    var f: [256]u8 = undefined;
    try b.linear(try std.fmt.bufPrint(&o, "{s}.fb", .{out}), try fullName(&f, "layers.{d}.self_attn.f_b_proj.weight", .{i}));
    try b.linear(try std.fmt.bufPrint(&o, "{s}.gb", .{out}), try fullName(&f, "layers.{d}.self_attn.g_b_proj.weight", .{i}));
    // conv: q | k | v taps, each [rank channels, 1, 4], as [3 HL 128, 4]
    {
        var parts: [3]Builder.Part = undefined;
        for ("qkv", 0..) |x, j| {
            var cn: [256]u8 = undefined;
            parts[j] = try b.part(try fullName(&cn, "layers.{d}.self_attn.{c}_conv1d.weight", .{ i, x }), .row);
        }
        defer for (parts) |p| b.free(p);
        const ch = parts[0].shape[0] * 3;
        const shape = [_]i64{ ch, 4 };
        if (b.dry) {
            try b.put(try std.fmt.bufPrint(&o, "{s}.conv", .{out}), .bf16, &shape, null);
        } else {
            const all = try b.gpa.alloc(u8, @intCast(ch * 4 * 2));
            defer b.gpa.free(all);
            var at: usize = 0;
            for (parts) |p| {
                @memcpy(all[at..][0..p.data.?.len], p.data.?);
                at += p.data.?.len;
            }
            try b.put(try std.fmt.bufPrint(&o, "{s}.conv", .{out}), .bf16, &shape, all);
        }
    }
    try b.tensor(try std.fmt.bufPrint(&o, "{s}.a_log", .{out}), try fullName(&f, "layers.{d}.self_attn.A_log", .{i}), true);
    try b.tensor(try std.fmt.bufPrint(&o, "{s}.dt_bias", .{out}), try fullName(&f, "layers.{d}.self_attn.dt_bias", .{i}), true);
    try b.tensor(try std.fmt.bufPrint(&o, "{s}.norm", .{out}), try fullName(&f, "layers.{d}.self_attn.o_norm.weight", .{i}), false);
    try b.linear(try std.fmt.bufPrint(&o, "{s}.o", .{out}), try fullName(&f, "layers.{d}.self_attn.o_proj.weight", .{i}));
}

fn dsa(b: *Builder, out: []const u8, i: usize) !void {
    var o: [160]u8 = undefined;
    var f: [256]u8 = undefined;
    var f2: [256]u8 = undefined;
    try b.stack(try std.fmt.bufPrint(&o, "{s}.proj", .{out}), &.{ try fullName(&f, "layers.{d}.self_attn.q_a_proj.weight", .{i}), try fullName(&f2, "layers.{d}.self_attn.kv_a_proj_with_mqa.weight", .{i}) });
    try b.tensor(try std.fmt.bufPrint(&o, "{s}.q_norm", .{out}), try fullName(&f, "layers.{d}.self_attn.q_a_layernorm.weight", .{i}), false);
    try b.tensor(try std.fmt.bufPrint(&o, "{s}.kv_norm", .{out}), try fullName(&f, "layers.{d}.self_attn.kv_a_layernorm.weight", .{i}), false);
    try b.linear(try std.fmt.bufPrint(&o, "{s}.q_b", .{out}), try fullName(&f, "layers.{d}.self_attn.q_b_proj.weight", .{i}));
    // kv_b per head: 256 key rows then 256 value rows -> absorb.wk, absorb.wv [HL, 256, 512] (the latent path's only use)
    {
        const p = try b.part(try fullName(&f, "layers.{d}.self_attn.kv_b_proj.weight", .{i}), .row);
        defer b.free(p);
        const HL: usize = @intCast(kern.HL);
        const lw: usize = @intCast(p.shape[1]);
        const shape = [_]i64{ kern.HL, kern.QK, @intCast(lw) };
        if (b.dry) {
            try b.put(try std.fmt.bufPrint(&o, "{s}.absorb.wk", .{out}), .bf16, &shape, null);
            try b.put(try std.fmt.bufPrint(&o, "{s}.absorb.wv", .{out}), .bf16, &shape, null);
        } else {
            const half: usize = @intCast(kern.QK);
            const wk = try b.gpa.alloc(u8, HL * half * lw * 2);
            defer b.gpa.free(wk);
            const wv = try b.gpa.alloc(u8, HL * half * lw * 2);
            defer b.gpa.free(wv);
            const row = lw * 2;
            for (0..HL) |h| {
                @memcpy(wk[h * half * row ..][0 .. half * row], p.data.?[(h * 2 * half) * row ..][0 .. half * row]);
                @memcpy(wv[h * half * row ..][0 .. half * row], p.data.?[(h * 2 * half + half) * row ..][0 .. half * row]);
            }
            try b.put(try std.fmt.bufPrint(&o, "{s}.absorb.wk", .{out}), .bf16, &shape, wk);
            try b.put(try std.fmt.bufPrint(&o, "{s}.absorb.wv", .{out}), .bf16, &shape, wv);
        }
    }
    try b.linear(try std.fmt.bufPrint(&o, "{s}.o", .{out}), try fullName(&f, "layers.{d}.self_attn.o_proj.weight", .{i}));
    try b.stack(try std.fmt.bufPrint(&o, "{s}.index.kw", .{out}), &.{ try fullName(&f, "layers.{d}.self_attn.indexer.wk.weight", .{i}), try fullName(&f2, "layers.{d}.self_attn.indexer.weights_proj.weight", .{i}) });
    try b.linear(try std.fmt.bufPrint(&o, "{s}.index.qb", .{out}), try fullName(&f, "layers.{d}.self_attn.indexer.wq_b.weight", .{i}));
    try b.tensor(try std.fmt.bufPrint(&o, "{s}.index.ln_w", .{out}), try fullName(&f, "layers.{d}.self_attn.indexer.k_norm.weight", .{i}), false);
    try b.tensor(try std.fmt.bufPrint(&o, "{s}.index.ln_b", .{out}), try fullName(&f, "layers.{d}.self_attn.indexer.k_norm.bias", .{i}), false);
    try b.tensor(try std.fmt.bufPrint(&o, "{s}.index.gate", .{out}), try fullName(&f, "layers.{d}.self_attn.indexer.index_kpool_compress_gate", .{i}), false);
    try b.tensor(try std.fmt.bufPrint(&o, "{s}.index.ape", .{out}), try fullName(&f, "layers.{d}.self_attn.indexer.index_kpool_compress_ape", .{i}), false);
}

fn mlp(b: *Builder, out: []const u8, prefix: []const u8) !void {
    var o: [160]u8 = undefined;
    var f: [256]u8 = undefined;
    var f2: [256]u8 = undefined;
    try b.stack(try std.fmt.bufPrint(&o, "{s}.gu", .{out}), &.{ try std.fmt.bufPrint(&f, "{s}gate_proj.weight", .{prefix}), try std.fmt.bufPrint(&f2, "{s}up_proj.weight", .{prefix}) });
    try b.linear(try std.fmt.bufPrint(&o, "{s}.down", .{out}), try std.fmt.bufPrint(&f, "{s}down_proj.weight", .{prefix}));
}

fn moe(b: *Builder, out: []const u8, i: usize) !void {
    var o: [160]u8 = undefined;
    var f: [256]u8 = undefined;
    try b.tensor(try std.fmt.bufPrint(&o, "{s}.router", .{out}), try fullName(&f, "layers.{d}.mlp.gate.weight", .{i}), false);
    try b.tensor(try std.fmt.bufPrint(&o, "{s}.bias", .{out}), try fullName(&f, "layers.{d}.mlp.gate.e_score_correction_bias", .{i}), true);
    if (b.exl3) try expertsExl3(b, try std.fmt.bufPrint(&o, "{s}.experts", .{out}), i) else try experts(b, try std.fmt.bufPrint(&o, "{s}.experts", .{out}), i);
    try mlp(b, try std.fmt.bufPrint(&o, "{s}.shared", .{out}), try fullName(&f, "layers.{d}.mlp.shared_experts.", .{i}));
}

/// One decoder layer (``plain``: the MTP head's, without hyper-connections) under ``out``.
fn layer(b: *Builder, out: []const u8, i: usize, kda_layer: bool, dense: bool, plain: bool) !void {
    var o: [160]u8 = undefined;
    var f: [256]u8 = undefined;
    if (!plain) {
        try hc(b, try std.fmt.bufPrint(&o, "{s}.attn_hc", .{out}), i, "attn");
        try hc(b, try std.fmt.bufPrint(&o, "{s}.ffn_hc", .{out}), i, "ffn");
    }
    try b.tensor(try std.fmt.bufPrint(&o, "{s}.in_norm", .{out}), try fullName(&f, "layers.{d}.input_layernorm.weight", .{i}), false);
    try b.tensor(try std.fmt.bufPrint(&o, "{s}.post_norm", .{out}), try fullName(&f, "layers.{d}.post_attention_layernorm.weight", .{i}), false);
    if (kda_layer) try kda(b, try std.fmt.bufPrint(&o, "{s}.kda", .{out}), i) else try dsa(b, try std.fmt.bufPrint(&o, "{s}.dsa", .{out}), i);
    if (dense) try mlp(b, try std.fmt.bufPrint(&o, "{s}.mlp", .{out}), try fullName(&f, "layers.{d}.mlp.", .{i})) else try moe(b, try std.fmt.bufPrint(&o, "{s}.moe", .{out}), i);
}

/// The text config fields the preparation reads, checked against the dimensions the kernels take.
pub const Cfg = struct {
    exl3: bool = false, // EXL3 routed experts (else the converted affine ones)
    layers: usize,
    kinds: [64]bool = @splat(false), // true: a KDA layer
    dense: usize, // first_k_dense_replace
    mtp: bool,
    vocab: usize,

    pub fn read(a: std.mem.Allocator, io: std.Io, dir: []const u8) !Cfg {
        const path = try std.fs.path.join(a, &.{ dir, "config.json" });
        const text = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 22));
        const v = try std.json.parseFromSliceLeaky(std.json.Value, a, text, .{});
        const q = v.object.get("quantization_config") orelse return error.NotAffineExperts;
        var exl3 = false;
        if (q.object.get("quant_method")) |m| if (std.mem.eql(u8, m.string, "exl3")) {
            const cb = q.object.get("codebook") orelse return error.NotExl3Mcg4;
            const bits = q.object.get("bits") orelse return error.NotExl3Mcg4;
            if (!std.mem.eql(u8, cb.string, "mcg") or bits.integer != 4) return error.NotExl3Mcg4;
            exl3 = true;
        };
        if (!exl3) {
            const fmt = q.object.get("format") orelse return error.NotAffineExperts;
            if (!std.mem.eql(u8, fmt.string, "glm-affine-experts-v1")) return error.NotAffineExperts;
        }
        const t = v.object.get("text_config").?.object;
        const n: usize = @intCast(t.get("num_hidden_layers").?.integer);
        var c: Cfg = .{ .exl3 = exl3, .layers = n, .dense = @intCast(t.get("first_k_dense_replace").?.integer), .mtp = (if (t.get("num_nextn_predict_layers")) |x| x.integer else 0) > 0, .vocab = @intCast(t.get("vocab_size").?.integer) };
        if (n > 64) return error.TooManyLayers;
        for (t.get("layer_types").?.array.items, 0..) |x, i| c.kinds[i] = std.mem.eql(u8, x.string, "linear_attention");
        if (t.get("hidden_size").?.integer != kern.D or t.get("n_routed_experts").?.integer != kern.EXPERTS or t.get("num_experts_per_tok").?.integer != kern.TOPK or
            t.get("num_attention_heads").?.integer != kern.HL * WORLD or t.get("kv_lora_rank").?.integer != kern.KV_LORA) return error.UnexpectedDimensions;
        return c;
    }
};

fn build(b: *Builder, c: *const Cfg) !void {
    var o: [64]u8 = undefined;
    for (0..c.layers) |i| try layer(b, try std.fmt.bufPrint(&o, "layers.{d}", .{i}), i, c.kinds[i], i < c.dense, false);
    var f: [256]u8 = undefined;
    try b.tensor("norm", try fullName(&f, "norm.weight", .{}), false);
    try b.tensor("embed", BASE ++ "embed_tokens.weight", false);
    // the head: this rank's vocabulary rows (packed to head_bits when set)
    {
        const e = try b.ck.get("lm_head.weight");
        const rows: usize = @intCast(e.shape[0]);
        const kk: usize = @intCast(e.shape[1]);
        const vl = rows / WORLD;
        const shape = [_]i64{ @intCast(vl), @intCast(kk) };
        if (b.dry) {
            if (b.head_bits != 0) try b.putPacked("head", b.head_bits, shape[0], shape[1], null) else try b.put("head.weight", .bf16, &shape, null);
            if (b.draft_head_bits != 0) try b.putPacked("draft_head", b.draft_head_bits, shape[0], shape[1], null);
        } else {
            const buf = try b.gpa.alignedAlloc(u8, .@"16", vl * kk * 2);
            defer b.gpa.free(buf);
            try b.ck.readAt(e, b.rank * vl * kk * 2, buf);
            if (b.head_bits != 0) try b.putPacked("head", b.head_bits, shape[0], shape[1], buf) else try b.put("head.weight", .bf16, &shape, buf);
            if (b.draft_head_bits != 0) try b.putPacked("draft_head", b.draft_head_bits, shape[0], shape[1], buf);
        }
    }
    if (c.mtp) {
        const m = c.layers;
        try b.tensor("mtp.enorm", try fullName(&f, "layers.{d}.enorm.weight", .{m}), false);
        try b.tensor("mtp.hnorm", try fullName(&f, "layers.{d}.hnorm.weight", .{m}), false);
        try b.linear("mtp.eh", try fullName(&f, "layers.{d}.eh_proj.weight", .{m}));
        try b.tensor("mtp.norm", try fullName(&f, "layers.{d}.shared_head.norm.weight", .{m}), false);
        try layer(b, "mtp.layer", m, false, false, true);
    }
}

pub fn load(gpa: std.mem.Allocator, io: std.Io, ctx: *const api.Ctx, dir: []const u8, rank: u8) !api.Store {
    var ck = try Ckpt.open(gpa, io, dir);
    defer ck.deinit();
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const c = try Cfg.read(arena.allocator(), io, dir);
    var store = api.Store.init(gpa);
    errdefer store.deinit();
    var b: Builder = .{ .gpa = gpa, .ck = &ck, .dry = false, .rank = rank, .store = &store, .d = ctx.d, .dense_bits = try envBits("TF_GLM_DENSE_BITS"), .head_bits = try envBits("TF_GLM_HEAD_BITS"), .draft_head_bits = try envBits("TF_GLM_DRAFT_HEAD_BITS"), .kda_proj_bits = try envBits("TF_GLM_KDA_PROJ_BITS"), .experts_sym = envSym(), .expert_sim_bits = envSimBits(), .experts3 = envBits3(), .sim_proj = envSimProj() };
    b.exl3 = c.exl3;
    if (c.exl3) b.experts_sym = false;
    try build(&b, &c);
    const vl = c.vocab / WORLD;
    const facts = [_]struct { []const u8, i64 }{
        .{ "rank", rank },                                               .{ "world", WORLD },                             .{ "vocab", @intCast(c.vocab) },           .{ "vocab_shard", @intCast(vl) },
        .{ "vocab_offset", @intCast(rank * vl) },                        .{ "layers", @intCast(c.layers) },               .{ "dense", @intCast(c.dense) },           .{ "mtp", @intFromBool(c.mtp) },
        .{ "device_bytes", @intCast(b.bytes) },                          .{ "experts_sym", @intFromBool(b.experts_sym) }, .{ "experts3", @intFromBool(b.experts3) }, .{ "exl3", @intFromBool(b.exl3) },
        .{ "exl3_shx", @intFromBool(b.exl3 and b.exl3_shx and !b.dry) },
    };
    for (facts) |fct| try store.ints.put(try gpa.dupe(u8, fct[0]), fct[1]);
    var kinds: [64]u8 = undefined;
    for (0..c.layers) |i| kinds[i] = @intFromBool(c.kinds[i]);
    try store.host.put(try gpa.dupe(u8, "kinds"), try gpa.dupe(u8, kinds[0..c.layers]));
    return store;
}

/// Device bytes this rank's prepared weights take: the same preparation's shapes, nothing read but headers.
pub fn deviceBytes(io: std.Io, dir: []const u8) !u64 {
    const gpa = std.heap.page_allocator;
    var ck = try Ckpt.open(gpa, io, dir);
    defer ck.deinit();
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const c = try Cfg.read(arena.allocator(), io, dir);
    var b: Builder = .{ .gpa = gpa, .ck = &ck, .dry = true, .rank = 0, .dense_bits = try envBits("TF_GLM_DENSE_BITS"), .head_bits = try envBits("TF_GLM_HEAD_BITS"), .draft_head_bits = try envBits("TF_GLM_DRAFT_HEAD_BITS"), .kda_proj_bits = try envBits("TF_GLM_KDA_PROJ_BITS"), .experts_sym = envSym(), .expert_sim_bits = envSimBits(), .experts3 = envBits3(), .sim_proj = envSimProj() };
    b.exl3 = c.exl3;
    try build(&b, &c);
    return b.bytes;
}

// ---------------------------------------------------------------------------------------------------------------
// Packed dense linears (ops.cu's glm_qgemv layout).

fn envBits3() bool {
    const v = std.c.getenv("TF_GLM_EXPERT_BITS") orelse return false;
    return std.mem.eql(u8, std.mem.span(v), "3");
}

fn envSimProj() u8 {
    const v = std.c.getenv("TF_GLM_EXPERT_SIM_PROJ") orelse return 0;
    const t = std.mem.span(v);
    return if (std.mem.eql(u8, t, "down")) 1 else if (std.mem.eql(u8, t, "up")) 2 else 0;
}

fn envSimBits() u8 {
    const v = std.c.getenv("TF_GLM_EXPERT_SIM_BITS") orelse return 0;
    return std.fmt.parseInt(u8, std.mem.span(v), 10) catch 0;
}

/// Each row's group of 64 re-quantized to ``bits`` symmetric levels (-2^(bits-1) .. 2^(bits-1) - 1, the scale from a
/// search over clip ratios for the least squared error), written back as 4-bit codes + 8 with that scale (bias -8 s).
fn simulateBits(t: Triple, bits: u8) void {
    const Work = struct {
        fn run(tt: Triple, nb: u8, e0: usize, e1: usize) void {
            const lo: f32 = -@as(f32, @floatFromInt(@as(u32, 1) << @intCast(nb - 1)));
            const hi: f32 = -lo - 1;
            const wpg = 8; // 4-bit words a group of 64
            for (e0..e1) |e| for (0..tt.n) |r| for (0..tt.kg) |g| {
                const at = (e * tt.n + r) * tt.kg + g;
                const words = tt.words[(e * tt.n + r) * tt.k8 + g * wpg ..][0..wpg];
                const s0 = bf16Value(tt.scales[at]);
                var v: [64]f32 = undefined;
                var amax: f32 = 0;
                for (0..64) |i| {
                    const q: i32 = @intCast((words[i / 8] >> @intCast(4 * (i % 8))) & 15);
                    v[i] = @as(f32, @floatFromInt(q - 8)) * s0;
                    amax = @max(amax, @abs(v[i]));
                }
                if (amax == 0) continue;
                var best_s: f32 = amax / -lo;
                var best_e: f32 = std.math.inf(f32);
                for ([_]f32{ 1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6 }) |c| {
                    const sc: f32 = bf16Value(toBf16(c * amax / -lo));
                    if (sc == 0) continue;
                    var err: f32 = 0;
                    for (v) |x| {
                        const q = @max(lo, @min(hi, @round(x / sc)));
                        err += (q * sc - x) * (q * sc - x);
                    }
                    if (err < best_e) {
                        best_e = err;
                        best_s = sc;
                    }
                }
                for (words) |*w| w.* = 0;
                for (v, 0..) |x, i| {
                    const q: i32 = @intFromFloat(@max(lo, @min(hi, @round(x / best_s))));
                    words[i / 8] |= @as(u32, @intCast(q + 8)) << @intCast(4 * (i % 8));
                }
                tt.scales[at] = toBf16(best_s);
                tt.biases[at] = toBf16(-8 * best_s);
            };
        }
    };
    const threads = 16;
    var pool: [threads]?std.Thread = @splat(null);
    const per = (t.e + threads - 1) / threads;
    for (0..threads) |i| {
        const e0 = i * per;
        const e1 = @min(t.e, e0 + per);
        if (e0 >= e1) break;
        pool[i] = std.Thread.spawn(.{}, Work.run, .{ t, bits, e0, e1 }) catch blk: {
            Work.run(t, bits, e0, e1);
            break :blk null;
        };
    }
    for (pool) |th| if (th) |x| x.join();
}

fn toBf16(x: f32) u16 {
    const u: u32 = @bitCast(x);
    return @truncate((u + 0x7fff + ((u >> 16) & 1)) >> 16); // round to nearest even
}

fn envSym() bool {
    const v = std.c.getenv("TF_GLM_EXPERTS_SYM") orelse return true;
    return !std.mem.eql(u8, std.mem.span(v), "0");
}

/// TF_GLM_DENSE_BITS / TF_GLM_HEAD_BITS: 4, 6 or 8 (unset or 0: bf16 as stored).
fn envBits(name: [*:0]const u8) !u8 {
    const v = std.c.getenv(name) orelse return 0;
    const bits = std.fmt.parseInt(u8, std.mem.span(v), 10) catch return error.BadPackedBits;
    if (bits != 0 and bits != 4 and bits != 6 and bits != 8) return error.BadPackedBits;
    return bits;
}

fn bf16Value(h: u16) f32 {
    return @bitCast(@as(u32, h) << 16);
}

/// x rounded to the nearest integer, halves to even.
fn roundEven(x: f32) f32 {
    const f = @floor(x);
    const d = x - f;
    if (d > 0.5) return f + 1;
    if (d < 0.5) return f;
    return if (@mod(f, 2) == 0) f else f + 1;
}

/// bf16 rows [n, k] -> symmetric ``bits``-bit values: each group of 64 inputs a scale absmax / (2^(bits-1) - 1)
/// rounded to fp16, values round(w / scale) (halves to even) clamped to +-(2^(bits-1) - 1), stored + 2^(bits-1) in
/// 16-value chunks of bits / 2 words (``words`` [n * k * bits / 32]); ``scales`` [n * k / 64] fp16 bits.
pub fn quantize(bits: u8, src: []const u16, n: usize, k: usize, words: []u32, scales: []u16) void {
    const half: i32 = @as(i32, 1) << @intCast(bits - 1);
    const qmax: f32 = @floatFromInt(half - 1);
    const wpc: usize = bits / 2;
    const groups = k / 64;
    for (0..n) |r| {
        const row = src[r * k ..][0..k];
        for (0..groups) |g| {
            var amax: f32 = 0;
            for (row[g * 64 ..][0..64]) |h| amax = @max(amax, @abs(bf16Value(h)));
            const s16: f16 = @floatCast(amax / qmax);
            scales[r * groups + g] = @bitCast(s16);
            const sc: f32 = s16;
            for (0..4) |c| {
                var q: [16]u32 = undefined;
                for (0..16) |i| {
                    const v = bf16Value(row[g * 64 + c * 16 + i]);
                    const t: i32 = if (sc == 0) 0 else @intFromFloat(@max(-qmax, @min(qmax, roundEven(v / sc))));
                    q[i] = @intCast(t + half);
                }
                pack(bits, &q, words[((r * groups + g) * 4 + c) * wpc ..][0..wpc]);
            }
        }
    }
}

fn pack(bits: u8, q: *const [16]u32, out: []u32) void {
    @memset(out, 0);
    switch (bits) {
        4 => for (0..16) |i| {
            out[i / 8] |= q[i] << @intCast(4 * (i % 8));
        },
        8 => for (0..16) |i| {
            out[i / 4] |= q[i] << @intCast(8 * (i % 4));
        },
        else => {
            for (0..15) |i| out[i / 5] |= q[i] << @intCast(6 * (i % 5));
            for (0..3) |j| out[j] |= ((q[15] >> @intCast(2 * j)) & 3) << 30;
        },
    }
}

test "packed rows: values and scales round trip (4, 6, 8 bits)" {
    const gpa = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(7);
    const n = 3;
    const k = 128;
    var src: [n * k]u16 = undefined;
    for (&src) |*h| h.* = @truncate(@as(u32, @bitCast(prng.random().floatNorm(f32) * 0.02)) >> 16);
    src[5] = 0; // a zero inside a group
    for ([_]u8{ 4, 6, 8 }) |bits| {
        const words = try gpa.alloc(u32, n * k * bits / 32);
        defer gpa.free(words);
        var scales: [n * k / 64]u16 = undefined;
        quantize(bits, &src, n, k, words, &scales);
        const half: i32 = @as(i32, 1) << @intCast(bits - 1);
        const qmax: f32 = @floatFromInt(half - 1);
        const wpc: usize = bits / 2;
        for (0..n * k / 16) |c| {
            const p = words[c * wpc ..][0..wpc];
            for (0..16) |i| {
                const raw: u32 = switch (bits) {
                    4 => (p[i / 8] >> @intCast(4 * (i % 8))) & 15,
                    8 => (p[i / 4] >> @intCast(8 * (i % 4))) & 255,
                    else => if (i < 15) (p[i / 5] >> @intCast(6 * (i % 5))) & 63 else (p[0] >> 30) | ((p[1] >> 30) << 2) | ((p[2] >> 30) << 4),
                };
                const sc: f32 = @as(f16, @bitCast(scales[c / 4]));
                const got = @as(f32, @floatFromInt(@as(i32, @intCast(raw)) - half)) * sc;
                const want = bf16Value(src[c * 16 + i]);
                try std.testing.expect(@abs(got - want) <= sc * 0.5 + 1e-7);
                try std.testing.expect(@abs(@as(f32, @floatFromInt(@as(i32, @intCast(raw)) - half))) <= qmax);
            }
        }
    }
}
