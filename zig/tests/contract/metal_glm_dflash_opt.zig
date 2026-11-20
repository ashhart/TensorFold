//! GLM DFlash2 fused decode kernels under gates 1-4 and planted controls, bit-exact against draft_ops.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const contract = @import("contract");
const sources = @import("kernel_sources");
const rig_ = @import("rig.zig");

const c = contract.spec;
const g = contract.gates;
const opt = tf.glm.metal_dflash_opt;
const StreamPos = opt.StreamPos;
const KvAppendArgs = opt.KvAppendArgs;
const QkvArgs = opt.QkvArgs;
const ConvNormArgs = opt.ConvNormArgs;

pub const Kind = enum { kv_append, qkv_prep, conv_norm };
const Variant = struct { kind: Kind, ctl: u8 = 0 };

const kv_served = [_]c.Shape{
    .{ .rows = 4, .n = 8, .k = 128 },
    .{ .rows = 2, .n = 2, .k = 64 },
    .{ .rows = 12, .n = 8, .k = 128 },
};
const qkv_served = [_]c.Shape{
    .{ .rows = 4, .n = 32, .k = 128 },
    .{ .rows = 16, .n = 32, .k = 128 },
    .{ .rows = 8, .n = 8, .k = 64 },
};
const conv_served = [_]c.Shape{
    .{ .rows = 4, .n = 512, .k = 512 },
    .{ .rows = 16, .n = 4096, .k = 4096 },
    .{ .rows = 48, .n = 4096, .k = 4096 },
};

fn noSums(_: c.Shape) usize {
    return 0;
}

pub fn entryName(v: Variant) []const u8 {
    return switch (v.ctl) {
        0 => switch (v.kind) {
            .kv_append => "glm dflash2 opt glm_dflash_kv_append",
            .qkv_prep => "glm dflash2 opt glm_dflash_qkv_prep",
            .conv_norm => "glm dflash2 opt glm_dflash_conv_norm",
        },
        1 => switch (v.kind) {
            .kv_append => "control: glm dflash2 opt kv_append writes into input",
            .qkv_prep => "control: glm dflash2 opt qkv_prep writes into input",
            .conv_norm => "control: glm dflash2 opt conv_norm writes into input",
        },
        else => switch (v.kind) {
            .kv_append => "control: glm dflash2 opt kv_append drops last RoPE pair",
            .qkv_prep => "control: glm dflash2 opt qkv_prep drops last RoPE pair",
            .conv_norm => "control: glm dflash2 opt conv_norm drops tail cols",
        },
    };
}

fn spec(v: Variant) c.Spec {
    return switch (v.kind) {
        .kv_append => .{
            .name = entryName(v),
            .accum = .none,
            .depth = noSums,
            .out = .bf16,
            .replaces = "draft_ops norm+rope + shared_attention appendKv",
            .nan = .{ .stated = "ring slots match baseline IEEE class; untouched slots remain poisoned" },
            .tiles = .{ .{ .size = 16 }, .{ .size = 8, .step = 8, .min = 8 }, .{ .size = 128, .step = 64, .min = 64 } },
            .served = &kv_served,
        },
        .qkv_prep => .{
            .name = entryName(v),
            .accum = .none,
            .depth = noSums,
            .out = .bf16,
            .replaces = "draft_ops q/k norm+rope + v copy",
            .tiles = .{ .{ .size = 16 }, .{ .size = 8, .step = 8, .min = 8 }, .{ .size = 128, .step = 64, .min = 64 } },
            .served = &qkv_served,
        },
        .conv_norm => .{
            .name = entryName(v),
            .accum = .none,
            .depth = noSums,
            .out = .bf16,
            .replaces = "draft_ops dynamic_conv2 + draft_rms",
            .tiles = .{ .{ .size = 16 }, .{}, .{ .size = 4096, .step = 512, .min = 512 } },
            .served = &conv_served,
        },
    };
}

const ring_len: u32 = 32;
const conv_group: u32 = 16;
const conv_block: u32 = 16;

pub const Probe = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    v: Variant,
    pipe: mtl.Pipeline,
    ops: tf.draft_ops.Ops,
    shape: c.Shape = undefined,
    streams_n: u32 = 0,
    kv_heads: u32 = 0,
    heads: u32 = 0,
    dim: u32 = 0,
    spos: [3]StreamPos = undefined,
    in_refs: [5]rig_.Ref = undefined,
    out_refs: [8]rig_.Ref = undefined,
    out_count: usize = 0,
    ring_bytes: usize = 0,
    base_bytes: []u8 = &.{},
    x_host: []u16 = &.{},
    qkv_host: []u16 = &.{},
    pos_host: []u64 = &.{},
    msg: [120]u8 = undefined,

    fn drop(p: *Probe) void {
        p.a.free(p.base_bytes);
        p.a.free(p.x_host);
        p.a.free(p.qkv_host);
        p.a.free(p.pos_host);
        p.base_bytes = &.{};
        p.x_host = &.{};
        p.qkv_host = &.{};
        p.pos_host = &.{};
        p.out_count = 0;
        p.rig.reset();
    }

    pub fn deinit(p: *Probe) void {
        p.drop();
        p.pipe.deinit();
        p.ops.deinit();
        p.rig.deinit();
    }

    pub fn kernel(p: *Probe) g.Kernel {
        const ctl: ?g.Gate = switch (p.v.ctl) {
            1 => .repeat,
            2 => .shapes,
            else => null,
        };
        return .{ .spec = spec(p.v), .ctx = p, .vt = &vt, .control = ctl };
    }

    const vt = g.hooks(Probe);

    pub fn prepare(p: *Probe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        p.drop();
        if (shape.rows == 0 or shape.rows > 64) return false;
        switch (p.v.kind) {
            .kv_append => {
                if (shape.rows > 16 or shape.n == 0 or (shape.k != 64 and shape.k != 128)) return false;
                try p.prepareKv(shape, fill, seed);
            },
            .qkv_prep => {
                if (shape.n < 4 or shape.n % 4 != 0 or (shape.k != 64 and shape.k != 128)) return false;
                try p.prepareQkv(shape, fill, seed);
            },
            .conv_norm => {
                if (shape.k < 64 or shape.k > 4096 or shape.k % 64 != 0 or shape.n != shape.k) return false;
                try p.prepareConv(shape, fill, seed);
            },
        }
        p.shape = shape;
        return true;
    }

    fn fillBf16(buf: []u16, rng: std.Random, scale: f32) void {
        for (buf) |*x| x.* = c.toBf16((rng.float(f32) * 2 - 1) * scale);
    }

    fn prepareKv(p: *Probe, shape: c.Shape, fill: g.Fill, seed: u64) !void {
        p.kv_heads = @intCast(shape.n);
        p.dim = @intCast(shape.k);
        const total = shape.rows;
        p.streams_n = if (total >= 8) 3 else if (total >= 4) 2 else 1;
        var rem: u32 = @intCast(total);
        var row0: u32 = 0;
        for (0..p.streams_n) |s| {
            const take: u32 = if (s + 1 == p.streams_n) rem else @max(1, @as(u32, @intCast(total)) / p.streams_n);
            p.spos[s] = .{ .first = 5 + s * 11, .row0 = row0, .rows = take };
            row0 += take;
            rem -= take;
        }
        const kw = p.kv_heads * p.dim;
        p.x_host = try p.a.alloc(u16, total * 2 * kw);
        const k_norm = try p.a.alloc(u16, p.dim);
        defer p.a.free(k_norm);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        fillBf16(p.x_host, rng, 2.0);
        fillBf16(k_norm, rng, 1.5);
        if (fill == .nonfinite) {
            p.x_host[0] = 0x7fc0;
            if (total > 1) p.x_host[2 * kw] = 0x7f80;
            if (total > 2) p.x_host[4 * kw + kw] = 0xff80;
        }
        p.in_refs[0] = try p.rig.input(std.mem.sliceAsBytes(p.x_host));
        p.in_refs[1] = try p.rig.input(std.mem.sliceAsBytes(k_norm));
        p.ring_bytes = @as(usize, ring_len) * kw * 2;
        p.out_count = 2 * p.streams_n;
        for (0..p.out_count) |i| p.out_refs[i] = try p.rig.output(p.ring_bytes);
    }

    fn prepareQkv(p: *Probe, shape: c.Shape, fill: g.Fill, seed: u64) !void {
        p.heads = @intCast(shape.n);
        p.kv_heads = p.heads / 4;
        p.dim = @intCast(shape.k);
        const qkv_w = (p.heads + 2 * p.kv_heads) * p.dim;
        p.qkv_host = try p.a.alloc(u16, shape.rows * qkv_w);
        p.pos_host = try p.a.alloc(u64, shape.rows);
        const qw = try p.a.alloc(u16, p.dim);
        defer p.a.free(qw);
        const kw = try p.a.alloc(u16, p.dim);
        defer p.a.free(kw);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        fillBf16(p.qkv_host, rng, 2.0);
        fillBf16(qw, rng, 1.5);
        fillBf16(kw, rng, 1.5);
        for (p.pos_host, 0..) |*pos, r| pos.* = 17 + r * 3;
        if (fill == .nonfinite) {
            p.qkv_host[0] = 0x7fc0;
            if (shape.rows > 1) p.qkv_host[qkv_w + p.heads * p.dim] = 0x7f80;
            if (shape.rows > 2) p.qkv_host[2 * qkv_w + (p.heads + p.kv_heads) * p.dim] = 0xff80;
        }
        p.in_refs[0] = try p.rig.input(std.mem.sliceAsBytes(p.qkv_host));
        p.in_refs[1] = try p.rig.input(std.mem.sliceAsBytes(p.pos_host));
        p.in_refs[2] = try p.rig.input(std.mem.sliceAsBytes(qw));
        p.in_refs[3] = try p.rig.input(std.mem.sliceAsBytes(kw));
        p.out_refs[0] = try p.rig.output(shape.rows * p.heads * p.dim * 2);
        p.out_refs[1] = try p.rig.output(shape.rows * p.kv_heads * p.dim * 2);
        p.out_refs[2] = try p.rig.output(shape.rows * p.kv_heads * p.dim * 2);
        p.out_count = 3;
    }

    fn prepareConv(p: *Probe, shape: c.Shape, fill: g.Fill, seed: u64) !void {
        const w = shape.k;
        const groups = w / conv_group;
        const x = try p.a.alloc(u16, shape.rows * w);
        defer p.a.free(x);
        const d = try p.a.alloc(u16, shape.rows * 4 * groups);
        defer p.a.free(d);
        const b = try p.a.alloc(u16, 4 * w);
        defer p.a.free(b);
        const r = try p.a.alloc(u16, shape.rows * w);
        defer p.a.free(r);
        const nw = try p.a.alloc(u16, w);
        defer p.a.free(nw);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        fillBf16(x, rng, 1.5);
        fillBf16(d, rng, 0.5);
        fillBf16(b, rng, 0.5);
        fillBf16(r, rng, 1.5);
        fillBf16(nw, rng, 1.5);
        if (fill == .nonfinite) {
            x[0] = 0x7fc0;
            if (shape.rows > 1) r[w] = 0x7f80;
            if (shape.rows > 2) x[2 * w + 1] = 0xff80;
        }
        p.in_refs[0] = try p.rig.input(std.mem.sliceAsBytes(x));
        p.in_refs[1] = try p.rig.input(std.mem.sliceAsBytes(d));
        p.in_refs[2] = try p.rig.input(std.mem.sliceAsBytes(b));
        p.in_refs[3] = try p.rig.input(std.mem.sliceAsBytes(r));
        p.in_refs[4] = try p.rig.input(std.mem.sliceAsBytes(nw));
        p.out_refs[0] = try p.rig.output(shape.rows * w * 2);
        p.out_refs[1] = try p.rig.output(shape.rows * w * 2);
        p.out_count = 2;
    }

    pub fn run(p: *Probe) !void {
        const pass = p.rig.begin();
        switch (p.v.kind) {
            .kv_append => {
                var st_buf: [8]StreamPos = @splat(.{ .first = 0, .row0 = @intCast(p.shape.rows), .rows = 0 });
                @memcpy(st_buf[0..p.streams_n], p.spos[0..p.streams_n]);
                pass.e.setPipeline(p.pipe);
                pass.e.setBuffer(p.in_refs[0].buf, p.in_refs[0].off, 0);
                pass.e.setBuffer(p.in_refs[1].buf, p.in_refs[1].off, 1);
                pass.e.setValue(KvAppendArgs{ .total = @intCast(p.shape.rows), .kv_heads = p.kv_heads, .dim = p.dim, .kv_stride = 2 * p.kv_heads * p.dim, .kv_width = p.kv_heads * p.dim, .ring = ring_len, .streams = p.streams_n, .eps = 1e-5, .theta = 10000.0 }, 2);
                pass.e.setBytes(std.mem.sliceAsBytes(&st_buf), 3);
                for (0..8) |s| {
                    const idx = if (s < p.streams_n) s else 0;
                    pass.e.setBuffer(p.out_refs[2 * idx].buf, p.out_refs[2 * idx].off, 4 + s);
                    pass.e.setBuffer(p.out_refs[2 * idx + 1].buf, p.out_refs[2 * idx + 1].off, 12 + s);
                }
                pass.e.dispatchGroups(mtl.Size.of(p.kv_heads, p.shape.rows, 1), mtl.Size.of(32, 1, 1));
            },
            .qkv_prep => {
                pass.e.setPipeline(p.pipe);
                for (0..4) |i| pass.e.setBuffer(p.in_refs[i].buf, p.in_refs[i].off, i);
                for (0..3) |i| pass.e.setBuffer(p.out_refs[i].buf, p.out_refs[i].off, 4 + i);
                pass.e.setValue(QkvArgs{ .rows = @intCast(p.shape.rows), .heads = p.heads, .kv_heads = p.kv_heads, .dim = p.dim, .qkv_stride = (p.heads + 2 * p.kv_heads) * p.dim, .eps = 1e-5, .theta = 10000.0 }, 7);
                pass.e.dispatchGroups(mtl.Size.of(p.heads + p.kv_heads, p.shape.rows, 1), mtl.Size.of(32, 1, 1));
            },
            .conv_norm => {
                pass.e.setPipeline(p.pipe);
                for (0..5) |i| pass.e.setBuffer(p.in_refs[i].buf, p.in_refs[i].off, i);
                pass.e.setBuffer(p.out_refs[0].buf, p.out_refs[0].off, 5);
                pass.e.setBuffer(p.out_refs[1].buf, p.out_refs[1].off, 6);
                pass.e.setValue(ConvNormArgs{ .rows = @intCast(p.shape.rows), .width = @intCast(p.shape.k), .group = conv_group, .block = conv_block, .branch = 1, .eps = 1e-5 }, 7);
                const tg = ((p.shape.k + 3) / 4 + 31) / 32 * 32;
                pass.e.dispatchGroups(mtl.Size.of(p.shape.rows, 1, 1), mtl.Size.of(tg, 1, 1));
            },
        }
        try p.rig.finish(pass);
    }

    pub fn count(p: *Probe) usize {
        return switch (p.v.kind) {
            .kv_append => p.out_count * p.ring_bytes / 2,
            .qkv_prep => p.shape.rows * (p.heads + 2 * p.kv_heads) * p.dim,
            .conv_norm => 2 * p.shape.rows * p.shape.k,
        };
    }

    pub fn reference(_: *Probe, want: []f64, mag: []f64) !void {
        @memset(want, 0);
        @memset(mag, 0);
    }

    pub fn check(p: *Probe, at: g.Gate) !?[]const u8 {
        const out = try p.rig.outputBytes();
        const want = try p.replaced();
        if (at == .shapes) return if (std.mem.eql(u8, out, want)) null else "bytes differ from draft_ops baseline";
        var finite: usize = 0;
        var nonfin: usize = 0;
        const elems = out.len / 2;
        for (0..elems) |i| {
            const gv = c.read(.bf16, out, i);
            const wv = c.read(.bf16, want, i);
            if (c.classOf(gv) != c.classOf(wv)) {
                return try std.fmt.bufPrint(&p.msg, "nonfinite class mismatch at {d}: got {t}, want {t}", .{ i, c.classOf(gv), c.classOf(wv) });
            }
            if (c.classOf(gv) == .finite) finite += 1 else nonfin += 1;
        }
        if (finite == 0 or nonfin == 0) return "nonfinite check needs both finite and nonfinite elements";
        return null;
    }

    pub fn replaced(p: *Probe) ![]const u8 {
        return switch (p.v.kind) {
            .kv_append => p.replaceKv(),
            .qkv_prep => p.replaceQkv(),
            .conv_norm => p.replaceConv(),
        };
    }

    fn replaceKv(p: *Probe) ![]const u8 {
        const kw = p.kv_heads * p.dim;
        const k_nrm = try p.rig.scratch(p.shape.rows * kw * 2);
        const k_rot = try p.rig.scratch(p.shape.rows * kw * 2);
        const v_tmp = try p.rig.scratch(p.shape.rows * kw * 2);
        const pos = try p.rig.scratch(p.shape.rows * 8);
        const v_host = std.mem.bytesAsSlice(u16, p.rig.bytes(v_tmp, p.shape.rows * kw * 2));
        for (0..p.shape.rows) |r| {
            const src = p.x_host[r * 2 * kw + kw ..][0..kw];
            @memcpy(v_host[r * kw ..][0..kw], src);
        }
        const pos_s = std.mem.bytesAsSlice(u64, p.rig.bytes(pos, p.shape.rows * 8));
        for (p.spos[0..p.streams_n]) |sp| {
            for (0..sp.rows) |j| pos_s[sp.row0 + j] = sp.first + j;
        }
        var rings: [6]rig_.Ref = undefined;
        for (0..p.out_count) |i| {
            rings[i] = try p.rig.scratch(p.ring_bytes);
            @memset(p.rig.bytes(rings[i], p.ring_bytes), 0xff);
        }
        const pass = p.rig.begin();
        try p.ops.norm(pass.e, .{ .buf = p.in_refs[0].buf, .off = p.in_refs[0].off }, .{ .buf = p.in_refs[1].buf, .off = p.in_refs[1].off }, .{ .buf = k_nrm.buf, .off = k_nrm.off }, .{ .rows = @intCast(p.shape.rows), .heads = p.kv_heads, .dim = p.dim, .x_stride = 2 * kw, .y_stride = kw, .eps = 1e-5 });
        pass.e.barrier();
        try p.ops.rope(pass.e, .{ .buf = k_nrm.buf, .off = k_nrm.off }, .{ .buf = pos.buf, .off = pos.off }, .{ .buf = k_rot.buf, .off = k_rot.off }, .{ .rows = @intCast(p.shape.rows), .heads = p.kv_heads, .dim = p.dim, .x_stride = kw, .y_stride = kw, .theta = 10000.0 });
        pass.e.barrier();
        var attn = try tf.shared_attention.Attention.init(p.a, p.rig.device, .{ .query_heads = p.kv_heads * 4, .kv_heads = p.kv_heads, .head_dim = p.dim, .capacity = ring_len, .output = .bf16 });
        defer attn.deinit();
        for (p.spos[0..p.streams_n], 0..) |sp, s| {
            var slots: [16]u32 = undefined;
            for (0..sp.rows) |j| slots[j] = @intCast((sp.first + j) % ring_len);
            const boff = @as(usize, sp.row0) * kw * 2;
            try attn.appendKv(pass.e, slots[0..sp.rows], .{ .buf = k_rot.buf, .off = k_rot.off + boff }, .{ .buf = v_tmp.buf, .off = v_tmp.off + boff }, .{ .buf = rings[2 * s].buf, .off = rings[2 * s].off }, .{ .buf = rings[2 * s + 1].buf, .off = rings[2 * s + 1].off });
            pass.e.barrier();
        }
        try p.rig.finish(pass);
        p.a.free(p.base_bytes);
        p.base_bytes = try p.a.alloc(u8, p.out_count * p.ring_bytes);
        for (0..p.out_count) |i| @memcpy(p.base_bytes[i * p.ring_bytes ..][0..p.ring_bytes], p.rig.bytes(rings[i], p.ring_bytes));
        return p.base_bytes;
    }

    fn replaceQkv(p: *Probe) ![]const u8 {
        const qw = p.heads * p.dim;
        const kw = p.kv_heads * p.dim;
        const qkv_w = qw + 2 * kw;
        const qb_n = p.shape.rows * qw * 2;
        const kb_n = p.shape.rows * kw * 2;
        const q_nrm = try p.rig.scratch(qb_n);
        const k_nrm = try p.rig.scratch(kb_n);
        const q_out = try p.rig.scratch(qb_n);
        const k_out = try p.rig.scratch(kb_n);
        const pass = p.rig.begin();
        try p.ops.norm(pass.e, .{ .buf = p.in_refs[0].buf, .off = p.in_refs[0].off }, .{ .buf = p.in_refs[2].buf, .off = p.in_refs[2].off }, .{ .buf = q_nrm.buf, .off = q_nrm.off }, .{ .rows = @intCast(p.shape.rows), .heads = p.heads, .dim = p.dim, .x_stride = qkv_w, .y_stride = qw, .eps = 1e-5 });
        pass.e.barrier();
        try p.ops.rope(pass.e, .{ .buf = q_nrm.buf, .off = q_nrm.off }, .{ .buf = p.in_refs[1].buf, .off = p.in_refs[1].off }, .{ .buf = q_out.buf, .off = q_out.off }, .{ .rows = @intCast(p.shape.rows), .heads = p.heads, .dim = p.dim, .x_stride = qw, .y_stride = qw, .theta = 10000.0 });
        pass.e.barrier();
        try p.ops.norm(pass.e, .{ .buf = p.in_refs[0].buf, .off = p.in_refs[0].off + qw * 2 }, .{ .buf = p.in_refs[3].buf, .off = p.in_refs[3].off }, .{ .buf = k_nrm.buf, .off = k_nrm.off }, .{ .rows = @intCast(p.shape.rows), .heads = p.kv_heads, .dim = p.dim, .x_stride = qkv_w, .y_stride = kw, .eps = 1e-5 });
        pass.e.barrier();
        try p.ops.rope(pass.e, .{ .buf = k_nrm.buf, .off = k_nrm.off }, .{ .buf = p.in_refs[1].buf, .off = p.in_refs[1].off }, .{ .buf = k_out.buf, .off = k_out.off }, .{ .rows = @intCast(p.shape.rows), .heads = p.kv_heads, .dim = p.dim, .x_stride = kw, .y_stride = kw, .theta = 10000.0 });
        try p.rig.finish(pass);
        p.a.free(p.base_bytes);
        p.base_bytes = try p.a.alloc(u8, qb_n + 2 * kb_n);
        @memcpy(p.base_bytes[0..qb_n], p.rig.bytes(q_out, qb_n));
        @memcpy(p.base_bytes[qb_n..][0..kb_n], p.rig.bytes(k_out, kb_n));
        const vo = std.mem.bytesAsSlice(u16, p.base_bytes[qb_n + kb_n ..][0..kb_n]);
        for (0..p.shape.rows) |r| @memcpy(vo[r * kw ..][0..kw], p.qkv_host[r * qkv_w + qw + kw ..][0..kw]);
        return p.base_bytes;
    }

    fn replaceConv(p: *Probe) ![]const u8 {
        const w: u32 = @intCast(p.shape.k);
        const nb = p.shape.rows * w * 2;
        const h_out = try p.rig.scratch(nb);
        const n_out = try p.rig.scratch(nb);
        const pass = p.rig.begin();
        try p.ops.conv(pass.e, .{ .buf = p.in_refs[0].buf, .off = p.in_refs[0].off }, .{ .buf = p.in_refs[1].buf, .off = p.in_refs[1].off }, .{ .buf = p.in_refs[2].buf, .off = p.in_refs[2].off }, .{ .buf = p.in_refs[3].buf, .off = p.in_refs[3].off }, .{ .buf = h_out.buf, .off = h_out.off }, .{ .rows = @intCast(p.shape.rows), .width = w, .group = conv_group, .block = conv_block, .branch = 1, .residual = 1, .x_stride = w, .dynamic_stride = 4 * (w / conv_group), .y_stride = w });
        pass.e.barrier();
        try p.ops.norm(pass.e, .{ .buf = h_out.buf, .off = h_out.off }, .{ .buf = p.in_refs[4].buf, .off = p.in_refs[4].off }, .{ .buf = n_out.buf, .off = n_out.off }, .{ .rows = @intCast(p.shape.rows), .heads = 1, .dim = w, .x_stride = w, .y_stride = w, .eps = 1e-5 });
        try p.rig.finish(pass);
        p.a.free(p.base_bytes);
        p.base_bytes = try p.a.alloc(u8, 2 * nb);
        @memcpy(p.base_bytes[0..nb], p.rig.bytes(h_out, nb));
        @memcpy(p.base_bytes[nb..][0..nb], p.rig.bytes(n_out, nb));
        return p.base_bytes;
    }
};

pub const Suite = struct {
    a: std.mem.Allocator,
    lib: mtl.Library,
    probes: std.ArrayList(Probe) = .empty,

    pub fn create(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, out: *std.ArrayList(g.Kernel), only: ?[]const u8) !*Suite {
        const s = try a.create(Suite);
        errdefer a.destroy(s);
        s.* = .{ .a = a, .lib = try mtl.Library.fromSource(device, sources.glm_dflash_opt, mtl.CompileOptions.mlx()) };
        const variants = [_]Variant{
            .{ .kind = .kv_append }, .{ .kind = .kv_append, .ctl = 1 }, .{ .kind = .kv_append, .ctl = 2 },
            .{ .kind = .qkv_prep },  .{ .kind = .qkv_prep, .ctl = 1 },  .{ .kind = .qkv_prep, .ctl = 2 },
            .{ .kind = .conv_norm }, .{ .kind = .conv_norm, .ctl = 1 }, .{ .kind = .conv_norm, .ctl = 2 },
        };
        for (variants) |v| {
            if (only) |needle| if (std.mem.indexOf(u8, entryName(v), needle) == null) continue;
            const pipe_lib = if (v.ctl == 0) s.lib else blk: {
                const def = if (v.ctl == 1) "#define TF_CTL_GLM_DFOPT 1\n" else "#define TF_CTL_GLM_DFOPT 2\n";
                const src = try std.mem.concat(a, u8, &.{ def, sources.glm_dflash_opt });
                defer a.free(src);
                break :blk try mtl.Library.fromSource(device, src, mtl.CompileOptions.mlx());
            };
            defer if (v.ctl != 0) pipe_lib.deinit();
            const fn_name = switch (v.kind) {
                .kv_append => "glm_dflash_kv_append",
                .qkv_prep => "glm_dflash_qkv_prep",
                .conv_norm => "glm_dflash_conv_norm",
            };
            try s.probes.append(a, .{
                .a = a,
                .rig = .{ .a = a, .device = device, .queue = queue },
                .v = v,
                .pipe = try mtl.Pipeline.init(device, pipe_lib, fn_name, false),
                .ops = try tf.draft_ops.Ops.init(device),
            });
        }
        for (s.probes.items) |*p| try out.append(a, p.kernel());
        return s;
    }

    pub fn destroy(s: *Suite) void {
        for (s.probes.items) |*p| p.deinit();
        s.probes.deinit(s.a);
        s.lib.deinit();
        s.a.destroy(s);
    }
};
