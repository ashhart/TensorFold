//! Qwen3.8 from a GGUF (arch qwen35): header parser and loaders; V heads tiled (Ops.tiled), norms 1 + w, A_log as -exp.

const std = @import("std");
const ld = @import("xpu").loader;
const cq = @import("xpu_config.zig");
const qb = @import("xpu_blocks.zig");
const gg = @import("xpu").ggml;

extern "c" fn open(path: [*:0]const u8, flags: c_int, ...) c_int;
extern "c" fn pread(fd: c_int, buf: [*]u8, n: usize, off: i64) isize;

const Buf = qb.Buf;
const Table = qb.Linear;
const hidden = qb.hidden;
const inter = qb.inter;

/// ggml tensor type ids of the file -> the dtype names Loader.Info carries (gguf-py spelling).
fn typeName(id: u32) ?[]const u8 {
    return switch (id) {
        0 => "F32",
        1 => "F16",
        8 => "Q8_0",
        10 => "Q2_K",
        12 => "Q4_K",
        13 => "Q5_K",
        14 => "Q6_K",
        16 => "IQ2_XXS",
        17 => "IQ2_XS",
        18 => "IQ3_XXS",
        20 => "IQ4_NL",
        21 => "IQ3_S",
        22 => "IQ2_S",
        23 => "IQ4_XS",
        29 => "IQ1_M",
        30 => "BF16",
        else => null,
    };
}

/// Bytes of n elements of a type by name (null for an unsupported type).
fn byteSize(name: []const u8, n: u64) ?u64 {
    if (std.mem.eql(u8, name, "F32")) return n * 4;
    if (std.mem.eql(u8, name, "F16") or std.mem.eql(u8, name, "BF16")) return n * 2;
    if (gg.fromGguf(name)) |t| return n / 256 * gg.blockBytes(t);
    return null;
}

/// Buffered sequential reader over pread with skipping (the header holds megabytes of tokenizer arrays we do not need).
const Rd = struct {
    fd: c_int,
    buf: []u8,
    pos: usize = 0,
    len: usize = 0,
    next: u64 = 0, // file offset of buf[len]

    fn logical(r: *Rd) u64 {
        return r.next - (r.len - r.pos);
    }

    fn take(r: *Rd, n: usize) ![]const u8 {
        if (r.len - r.pos < n) {
            std.mem.copyForwards(u8, r.buf[0 .. r.len - r.pos], r.buf[r.pos..r.len]);
            r.len -= r.pos;
            r.pos = 0;
            while (r.len < n) {
                const got = pread(r.fd, r.buf.ptr + r.len, r.buf.len - r.len, @intCast(r.next));
                if (got <= 0) return error.ReadFailed;
                r.len += @intCast(got);
                r.next += @intCast(got);
            }
        }
        defer r.pos += n;
        return r.buf[r.pos..][0..n];
    }

    fn skip(r: *Rd, n: u64) void {
        if (n <= r.len - r.pos) {
            r.pos += @intCast(n);
        } else {
            r.next = r.logical() + n;
            r.pos = 0;
            r.len = 0;
        }
    }

    fn int(r: *Rd, comptime T: type) !T {
        return std.mem.readInt(T, (try r.take(@sizeOf(T)))[0..@sizeOf(T)], .little);
    }

    fn str(r: *Rd) ![]const u8 {
        const n = try r.int(u64);
        if (n > r.buf.len) return error.BadGguf;
        return r.take(@intCast(n));
    }
};

pub const Val = union(enum) { u: u64, f: f64, s: []const u8 };

pub const Meta = struct {
    kv: std.StringHashMapUnmanaged(Val) = .empty,

    pub fn u(m: Meta, key: []const u8) !u64 {
        const v = m.kv.get(key) orelse {
            std.log.err("gguf metadata missing: {s}", .{key});
            return error.BadGguf;
        };
        return switch (v) {
            .u => |x| x,
            else => error.BadGguf,
        };
    }

    pub fn f(m: Meta, key: []const u8) !f64 {
        const v = m.kv.get(key) orelse return error.BadGguf;
        return switch (v) {
            .f => |x| x,
            .u => |x| @floatFromInt(x),
            else => error.BadGguf,
        };
    }
};

/// Parses the header into `l.map` (tensor name -> file offset, bytes, type name) and returns the scalar metadata.
pub fn readHeader(gpa: std.mem.Allocator, l: *ld.Loader, path: []const u8) !Meta {
    const pz = try std.fmt.allocPrintSentinel(gpa, "{s}", .{path}, 0);
    defer gpa.free(pz);
    const fd = open(pz.ptr, 0);
    if (fd < 0) return error.OpenFailed;
    var r: Rd = .{ .fd = fd, .buf = try gpa.alloc(u8, 1 << 20) };
    defer gpa.free(r.buf);
    if (!std.mem.eql(u8, try r.take(4), "GGUF")) return error.BadGguf;
    if (try r.int(u32) != 3) return error.BadGguf;
    const n_tensors = try r.int(u64);
    const n_kv = try r.int(u64);
    var m: Meta = .{};
    var align_: u64 = 32;
    for (0..n_kv) |_| {
        const key = try gpa.dupe(u8, try r.str());
        const ty = try r.int(u32);
        switch (ty) {
            0, 1, 7 => try m.kv.put(gpa, key, .{ .u = try r.int(u8) }),
            2, 3 => try m.kv.put(gpa, key, .{ .u = try r.int(u16) }),
            4, 5 => try m.kv.put(gpa, key, .{ .u = try r.int(u32) }),
            10, 11 => try m.kv.put(gpa, key, .{ .u = try r.int(u64) }),
            6 => try m.kv.put(gpa, key, .{ .f = @as(f32, @bitCast(try r.int(u32))) }),
            12 => try m.kv.put(gpa, key, .{ .f = @bitCast(try r.int(u64)) }),
            8 => {
                const s = try r.str();
                if (std.mem.eql(u8, key, "general.architecture")) try m.kv.put(gpa, key, .{ .s = try gpa.dupe(u8, s) });
            },
            9 => {
                const et = try r.int(u32);
                const n = try r.int(u64);
                switch (et) {
                    0, 1, 7 => r.skip(n),
                    2, 3 => r.skip(n * 2),
                    4, 5, 6 => r.skip(n * 4),
                    10, 11, 12 => r.skip(n * 8),
                    8 => for (0..n) |_| r.skip(try r.int(u64)),
                    else => return error.BadGguf,
                }
            },
            else => return error.BadGguf,
        }
        if (std.mem.eql(u8, key, "general.alignment")) align_ = try m.u(key);
    }
    const arch = m.kv.get("general.architecture") orelse return error.BadGguf;
    if (!std.mem.eql(u8, arch.s, "qwen35")) {
        std.log.err("gguf architecture {s}, only qwen35 is wired", .{arch.s});
        return error.UnsupportedFormat;
    }
    const Pending = struct { name: []const u8, off: u64, len: u64, dtype: []const u8 };
    var pend: std.ArrayList(Pending) = .empty;
    defer pend.deinit(gpa);
    for (0..n_tensors) |_| {
        const name = try gpa.dupe(u8, try r.str());
        const nd = try r.int(u32);
        var n: u64 = 1;
        for (0..nd) |_| n *= try r.int(u64);
        const ty = try r.int(u32);
        const off = try r.int(u64);
        const tn = typeName(ty) orelse {
            std.log.err("{s}: ggml type {d} not supported", .{ name, ty });
            return error.UnsupportedFormat;
        };
        try pend.append(gpa, .{ .name = name, .off = off, .len = byteSize(tn, n) orelse return error.UnsupportedFormat, .dtype = tn });
    }
    const data = std.mem.alignForward(u64, r.logical(), align_);
    for (pend.items) |p| try l.map.put(gpa, p.name, .{ .fd = fd, .off = data + p.off, .len = p.len, .dtype = p.dtype });
    l.gguf = true;
    return m;
}

/// The engine config of the file: 64 layers (the trailing MTP block is ignored), vocab from the embedding.
pub fn config(gpa: std.mem.Allocator, l: *ld.Loader, m: Meta) !cq.Config {
    const pre = "qwen35.";
    const n_layer: u32 = @intCast(try m.u(pre ++ "block_count") - (m.u(pre ++ "nextn_predict_layers") catch 0));
    const interval: u32 = @intCast(try m.u(pre ++ "full_attention_interval"));
    const types = try gpa.alloc([]const u8, n_layer);
    for (types, 0..) |*t, i| t.* = if ((i + 1) % interval == 0) "full_attention" else "linear_attention";
    const emb = try l.info("token_embd.weight");
    const vocab: u32 = @intCast(emb.len / gg.tensorBytes(gg.fromGguf(emb.dtype) orelse return error.UnsupportedFormat, 1, hidden));
    const head_dim: u32 = @intCast(try m.u(pre ++ "attention.key_length"));
    const eos: u32 = @intCast(try m.u("tokenizer.ggml.eos_token_id"));
    const eos_ids = try gpa.dupe(u32, &.{ eos, 248044 });
    return .{
        .text_config = .{
            .hidden_size = @intCast(try m.u(pre ++ "embedding_length")),
            .intermediate_size = @intCast(try m.u(pre ++ "feed_forward_length")),
            .vocab_size = vocab,
            .num_hidden_layers = n_layer,
            .num_attention_heads = @intCast(try m.u(pre ++ "attention.head_count")),
            .num_key_value_heads = @intCast(try m.u(pre ++ "attention.head_count_kv")),
            .head_dim = head_dim,
            .linear_conv_kernel_dim = @intCast(try m.u(pre ++ "ssm.conv_kernel")),
            .linear_key_head_dim = @intCast(try m.u(pre ++ "ssm.state_size")),
            .linear_value_head_dim = @intCast(try m.u(pre ++ "ssm.state_size")),
            .linear_num_key_heads = @intCast(try m.u(pre ++ "ssm.group_count")),
            .linear_num_value_heads = @intCast(try m.u(pre ++ "ssm.time_step_rank")),
            .rms_norm_eps = @floatCast(try m.f(pre ++ "attention.layer_norm_rms_epsilon")),
            .attn_output_gate = true,
            .tie_word_embeddings = l.map.contains("output.weight") == false,
            .layer_types = types,
            .rope_parameters = .{ .rope_theta = try m.f(pre ++ "rope.freq_base"), .partial_rotary_factor = @as(f64, @floatFromInt(try m.u(pre ++ "rope.dimension_count"))) / @as(f64, @floatFromInt(head_dim)) },
        },
        .eos_token_id = eos_ids,
    };
}

fn toBf(f: f32) u16 {
    const u: u32 = @bitCast(f);
    return @intCast((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

fn fmt(buf: []u8, comptime f: []const u8, a: anytype) ![]const u8 {
    return std.fmt.bufPrint(buf, f, a);
}

/// A small F32 tensor of exactly `count` values on the host.
fn readF32(gpa: std.mem.Allocator, l: *ld.Loader, name: []const u8, count: u64) ![]f32 {
    const inf = try l.info(name);
    if (!std.mem.eql(u8, inf.dtype, "F32") or inf.len != count * 4) {
        std.log.err("{s}: {s} {d} bytes, expected F32 {d} values", .{ name, inf.dtype, inf.len, count });
        return error.UnexpectedTensor;
    }
    const out = try gpa.alloc(f32, count);
    const bytes = std.mem.sliceAsBytes(out);
    var done: usize = 0;
    while (done < bytes.len) {
        const n = pread(inf.fd, bytes.ptr + done, bytes.len - done, @intCast(inf.off + done));
        if (n <= 0) return error.ReadFailed;
        done += @intCast(n);
    }
    return out;
}

fn uploadSmall(ops: *qb.Ops, l: *ld.Loader, bytes: []const u8) !Buf {
    const b = try l.empty(bytes.len);
    try ops.r.upload(b, bytes);
    try ops.r.sync();
    return b;
}

/// An F32 tensor as a bf16 device buffer (RMSNorm weights already 1 + w in the file, conv taps, gated-norm weight).
fn bf16Of(gpa: std.mem.Allocator, ops: *qb.Ops, l: *ld.Loader, name: []const u8, count: u64) !Buf {
    const v = try readF32(gpa, l, name, count);
    defer gpa.free(v);
    const h = try gpa.alloc(u16, count);
    defer gpa.free(h);
    for (v, h) |x, *o| o.* = toBf(x);
    return uploadSmall(ops, l, std.mem.sliceAsBytes(h));
}

fn readAt(fd: c_int, buf: []u8, off: u64) !void {
    var done: usize = 0;
    while (done < buf.len) {
        const n = pread(fd, buf.ptr + done, buf.len - done, @intCast(off + done));
        if (n <= 0) return error.ReadFailed;
        done += @intCast(n);
    }
}

/// ARC_EMBED_HOST=1: the token embedding table lives in pinned host memory, saving device memory.
var host_table = false;

/// Streams a ggml tensor [rows][in] to the device in the kernels' row-interleaved layout (ggml.packRows).
fn loadPacked(l: *ld.Loader, name: []const u8, t: gg.Type, rows: u32, in: u32) !Buf {
    const inf = try l.info(name);
    if (rows % 16 != 0) return error.UnexpectedTensor;
    const nb: u64 = in / 256;
    const raw_group: u64 = 16 * nb * gg.blockBytes(t);
    const pk_group: u64 = 16 * nb * gg.packedBytes(t);
    const out = if (host_table) try l.emptyHost(rows / 16 * pk_group) else try l.empty(rows / 16 * pk_group);
    const half = l.stage.len / 2;
    const per: u64 = half / pk_group;
    var done: u64 = 0;
    while (done < rows / 16) {
        const n = @min(per, rows / 16 - done);
        try readAt(inf.fd, l.stage[0 .. n * raw_group], inf.off + done * raw_group);
        gg.packRows(t, l.stage[0 .. n * raw_group], l.stage[half..][0 .. n * pk_group], n * 16, nb);
        if (host_table) {
            const dst: [*]u8 = @ptrCast(out.ptr.?);
            @memcpy((dst + done * pk_group)[0 .. n * pk_group], l.stage[half..][0 .. n * pk_group]);
            done += n;
            continue;
        }
        try l.r.upload(.{ .rt = out.rt, .ptr = @ptrFromInt(@intFromPtr(out.ptr.?) + done * pk_group), .len = n * pk_group }, l.stage[half..][0 .. n * pk_group]);
        try l.r.sync();
        done += n;
    }
    return out;
}

/// An RMSNorm weight: fp32 on the device (Ops.f32_norm) or rounded to bf16.
pub fn normW(gpa: std.mem.Allocator, ops: *qb.Ops, l: *ld.Loader, name: []const u8, count: u64) !Buf {
    if (!ops.f32_norm) return bf16Of(gpa, ops, l, name, count);
    const v = try readF32(gpa, l, name, count);
    defer gpa.free(v);
    return uploadSmall(ops, l, std.mem.sliceAsBytes(v));
}

/// A quantized (or bf16) [rows][in] projection; the byte size and the GGUF type name pin the format.
pub fn proj(l: *ld.Loader, name: []const u8, rows: u32, in: u32) !Table {
    const inf = try l.info(name);
    if (std.mem.eql(u8, inf.dtype, "BF16") and inf.len == @as(u64, rows) * in * 2) {
        return .{ .format = .bf16, .w = try l.load(name), .s = undefined, .b = undefined, .rows = rows, .in = in };
    }
    const t = gg.fromGguf(inf.dtype) orelse {
        std.log.err("{s}: type {s} has no kernel", .{ name, inf.dtype });
        return error.UnsupportedFormat;
    };
    if (in % 256 != 0 or inf.len != gg.tensorBytes(t, rows, in)) {
        std.log.err("{s}: {s} {d} bytes does not fit [{d}, {d}]", .{ name, inf.dtype, inf.len, in, rows });
        return error.UnexpectedTensor;
    }
    return .{ .format = qb.ggFormat(t), .w = try loadPacked(l, name, t, rows, in), .s = undefined, .b = undefined, .rows = rows, .in = in };
}

pub fn loadEmbed(l: *ld.Loader, vocab: u32) !Table {
    // only when the head is a separate tensor (a tied head would run its matvec out of host memory)
    host_table = std.c.getenv("ARC_EMBED_HOST") != null and l.map.contains("output.weight");
    defer host_table = false;
    return proj(l, "token_embd.weight", vocab, hidden);
}

pub fn loadHead(l: *ld.Loader, vocab: u32) !Table {
    return proj(l, "output.weight", vocab, hidden);
}

pub fn loadFinalNorm(gpa: std.mem.Allocator, ops: *qb.Ops, l: *ld.Loader) !Buf {
    return normW(gpa, ops, l, "output_norm.weight", hidden);
}

pub fn loadGdn(gpa: std.mem.Allocator, ops: *qb.Ops, l: *ld.Loader, i: usize) !qb.GdnW {
    var n: [96]u8 = undefined;
    const gdn_heads = qb.gdn_heads;
    const a_name = try fmt(&n, "blk.{d}.ssm_a", .{i});
    const a = try readF32(gpa, l, a_name, gdn_heads);
    defer gpa.free(a);
    for (a) |*v| v.* = @floatCast(@log(-@as(f64, v.*))); // ssm_a = -exp(A_log)
    var w: qb.GdnW = undefined;
    w.a_log = try uploadSmall(ops, l, std.mem.sliceAsBytes(a));
    w.norm = try normW(gpa, ops, l, try fmt(&n, "blk.{d}.attn_norm.weight", .{i}), hidden);
    w.qkv = try proj(l, try fmt(&n, "blk.{d}.attn_qkv.weight", .{i}), qb.gdn_qkv, hidden);
    w.z = try proj(l, try fmt(&n, "blk.{d}.attn_gate.weight", .{i}), qb.gdn_v, hidden);
    w.b = try proj(l, try fmt(&n, "blk.{d}.ssm_beta.weight", .{i}), gdn_heads, hidden);
    w.a = try proj(l, try fmt(&n, "blk.{d}.ssm_alpha.weight", .{i}), gdn_heads, hidden);
    w.out = try proj(l, try fmt(&n, "blk.{d}.ssm_out.weight", .{i}), hidden, qb.gdn_v);
    w.conv = try bf16Of(gpa, ops, l, try fmt(&n, "blk.{d}.ssm_conv1d.weight", .{i}), @as(u64, qb.gdn_qkv) * 4);
    const dt = try fmt(&n, "blk.{d}.ssm_dt.bias", .{i});
    const dtv = try readF32(gpa, l, dt, gdn_heads);
    defer gpa.free(dtv);
    w.dt_bias = try uploadSmall(ops, l, std.mem.sliceAsBytes(dtv));
    w.gnorm = try bf16Of(gpa, ops, l, try fmt(&n, "blk.{d}.ssm_norm.weight", .{i}), 128);
    w.cstate = try l.zeros(3 * qb.gdn_qkv * 2);
    w.sstate = try l.zeros(@as(usize, gdn_heads) * 128 * 128 * 4);
    return w;
}

pub fn loadAttn(gpa: std.mem.Allocator, ops: *qb.Ops, l: *ld.Loader, i: usize, cap: u32) !qb.AttnW {
    var n: [96]u8 = undefined;
    const kv = @import("xpu_attn_long.zig").cacheBytes(cap);
    var w: qb.AttnW = undefined;
    w.norm = try normW(gpa, ops, l, try fmt(&n, "blk.{d}.attn_norm.weight", .{i}), hidden);
    w.q = try proj(l, try fmt(&n, "blk.{d}.attn_q.weight", .{i}), 2 * qb.q_dim, hidden);
    w.k = try proj(l, try fmt(&n, "blk.{d}.attn_k.weight", .{i}), qb.kv_dim, hidden);
    w.v = try proj(l, try fmt(&n, "blk.{d}.attn_v.weight", .{i}), qb.kv_dim, hidden);
    w.o = try proj(l, try fmt(&n, "blk.{d}.attn_output.weight", .{i}), hidden, qb.q_dim);
    w.qn = try normW(gpa, ops, l, try fmt(&n, "blk.{d}.attn_q_norm.weight", .{i}), qb.head_dim);
    w.kn = try normW(gpa, ops, l, try fmt(&n, "blk.{d}.attn_k_norm.weight", .{i}), qb.head_dim);
    w.kc = try l.zeros(kv);
    w.vc = try l.zeros(kv);
    return w;
}

pub fn loadMlp(gpa: std.mem.Allocator, ops: *qb.Ops, l: *ld.Loader, i: usize) !qb.MlpW {
    var n: [96]u8 = undefined;
    var w: qb.MlpW = undefined;
    w.norm = try normW(gpa, ops, l, try fmt(&n, "blk.{d}.post_attention_norm.weight", .{i}), hidden);
    w.gate = try proj(l, try fmt(&n, "blk.{d}.ffn_gate.weight", .{i}), inter, hidden);
    w.up = try proj(l, try fmt(&n, "blk.{d}.ffn_up.weight", .{i}), inter, hidden);
    w.down = try proj(l, try fmt(&n, "blk.{d}.ffn_down.weight", .{i}), hidden, inter);
    return w;
}
