//! Inkling 2 attention (bias, sink, gate) via ops.attention vs float64 stream formulas crossing 2048-key chunk edges.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");
const u = @import("inkling2_util.zig");
const math = @import("inkling2_math_attention.zig");
const data = @import("metal_inkling2_attention_data.zig");
const rig_ = @import("rig.zig");
const c = u.c;
const g = u.g;
const ops = u.ops;

pub const Kind = enum { whole, split_calls, one_row, offsets, ring_alt };
pub const Case = data.Case;
pub const cases = data.cases;

/// Elements of NaN around the small inputs and the output in the `offsets` kind (96 bytes) and around the caches (32).
const pad_small = 48;
const pad_cache = 16;

const Mode = enum { whole, split, each };

pub fn name(k: Kind) []const u8 {
    return switch (k) {
        .whole => "metal inkling2 attention",
        .split_calls => "metal inkling2 attention, rows split over two calls",
        .one_row => "metal inkling2 attention, one row a call",
        .offsets => "metal inkling2 attention, views at nonzero offsets",
        .ring_alt => "metal inkling2 attention, another ring",
    };
}

fn depth(s: c.Shape) usize {
    return 131 + (s.k + s.rows) / 4;
}

/// Where each argument of a call lives.
const Target = struct { q: u.View, rel: u.View, sink: u.View, gate: u.View, kc: u.View, vc: u.View, proj: u.View, out: u.View, ring: u32 };

fn shift(v: u.View, bytes: usize) u.View {
    return .{ .buffer = v.buffer, .offset = v.offset + bytes };
}

pub const Probe = struct {
    a: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,
    rig: rig_.Rig,
    o: *const ops.Ops,
    kind: Kind,
    label: []const u8,
    shape: c.Shape = .{ .rows = 0, .n = 0, .k = 0 },
    cs: data.Case = data.cases[0],
    b: data.Built = undefined,
    main: Target = undefined,
    base: Target = undefined,
    out: rig_.Ref = undefined,
    alt: rig_.Ref = undefined,
    att: []f64 = &.{},
    err: []f64 = &.{},
    sig: []f64 = &.{},

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, o: *const ops.Ops, kind: Kind, label: []const u8) Probe {
        return .{ .a = a, .arena = .init(a), .rig = .{ .a = a, .device = device, .queue = queue }, .o = o, .kind = kind, .label = label };
    }

    pub fn deinit(p: *Probe) void {
        p.arena.deinit();
        p.rig.deinit();
    }

    const vt = g.hooks(Probe);

    pub fn kernel(p: *Probe) g.Kernel {
        const served: []const c.Shape = &.{
            .{ .rows = 6, .n = 0, .k = 2046 }, // rows 2046, 2047 one chunk; 2048.. two: a mixed call
            .{ .rows = 8, .n = 1, .k = 2500 }, // 48 heads over 8, window 1024 in a ring of 1536 that wraps
            .{ .rows = 17, .n = 2, .k = 1090 }, // 8 over 4, ring 1100 wraps at position 1100
            .{ .rows = 16, .n = 3, .k = 4090 }, // 6 over 6; chunk edges 4095, 4096, 4097 (two to three chunks)
            .{ .rows = 15, .n = 4, .k = 3295 }, // 6 over 2, ring wrap at 3300
            .{ .rows = 33, .n = 5, .k = 2040 }, // 8 over 2, 33 rows crossing the first chunk edge
            .{ .rows = 5, .n = 6, .k = 6000 }, // 16 over 2 (group 8), a long window
            .{ .rows = 40, .n = 8, .k = 1 }, // 5 over 1 (group 5), keys fewer than the window
            .{ .rows = 1, .n = 9, .k = 5000 }, // 7 over 1 (group 7), a decode row over three chunks
            .{ .rows = 32, .n = 7, .k = 100 }, // 48 over 8 global, short context
            .{ .rows = 3, .n = 10, .k = 1500 }, // extent 256: the key at distance extent is hot
            .{ .rows = 20, .n = 11, .k = 90 }, // window 64 in a ring of 100
            .{ .rows = 16, .n = 12, .k = 1144 }, // Mini sliding visit wraps its KV ring
            .{ .rows = 1, .n = 13, .k = 2048 }, // Mini global visit crosses the chunk seam
            .{ .rows = 16, .n = 13, .k = 2040 }, // the rows kernel: two tiles either side of the chunk seam
            .{ .rows = 40, .n = 13, .k = 4090 }, // five tiles, three chunks, edges at 4095 and 4096
            .{ .rows = 24, .n = 13, .k = 1000 }, // shared blocks past 2 per stream: the running max moves
            .{ .rows = 13, .n = 7, .k = 100 }, // 48 over 8, a short context, a tile of 5
            .{ .rows = 19, .n = 9, .k = 2500 }, // group 7 on the rows kernel
            .{ .rows = 9, .n = 3, .k = 300 }, // group 1
            .{ .rows = 12, .n = 0, .k = 2044 }, // the non-finite plants' case on the rows kernel, across the seam
            .{ .rows = 10, .n = 0, .k = 40 }, // the same in the first chunk
        };
        return .{ .ctx = p, .vt = &vt, .spec = .{
            .name = p.label,
            .accum = .none,
            .depth = depth,
            .ops = 4,
            .mids = 1,
            .out = .bf16,
            .replaces = switch (p.kind) {
                .whole => null,
                .split_calls, .one_row => "the same rows in one call",
                .offsets => "the same call with the views at offset zero",
                .ring_alt => "the same window in the case's ring",
            },
            .served = served,
            .tiles = .{ .{ .size = 32, .min = 1 }, .{}, .{ .size = 2048, .min = 0 } },
        } };
    }

    fn padOut(p: *const Probe) usize {
        return if (p.kind == .offsets) pad_small else 0;
    }

    fn elems(p: *const Probe) usize {
        return p.shape.rows * p.cs.heads * data.dim;
    }

    /// `words` as an input; with `pad` > 0 between `pad` NaN elements either side, the view at the first word.
    fn upload(p: *Probe, words: []const u16, pad: usize) !u.View {
        if (pad == 0) return u.view(try p.rig.input(std.mem.sliceAsBytes(words)));
        const padded = try p.arena.allocator().alloc(u16, words.len + 2 * pad);
        @memset(padded, u.nan_bits);
        @memcpy(padded[pad..][0..words.len], words);
        const r = try p.rig.input(std.mem.sliceAsBytes(padded));
        return .{ .buffer = r.buf, .offset = r.off + pad * 2 };
    }

    fn inputs(p: *Probe, small: usize, cache: usize) !Target {
        const b = &p.b;
        return .{
            .q = try p.upload(b.q, small),
            .rel = try p.upload(b.rel, small),
            .sink = try p.upload(b.sink, small),
            .gate = try p.upload(b.gate, small),
            .kc = try p.upload(b.kc, cache),
            .vc = try p.upload(b.vc, cache),
            .proj = try p.upload(b.proj, small),
            .out = undefined,
            .ring = @intCast(b.cs.ring),
        };
    }

    pub fn prepare(p: *Probe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.n >= data.cases.len or shape.rows == 0 or shape.rows > 64) return false;
        const cs = data.cases[shape.n];
        if (fill == .nonfinite and !data.plantable(cs, shape.rows)) return false;
        const a = u.reset(&p.arena, &p.rig);
        p.shape = shape;
        p.cs = cs;
        p.b = try data.build(a, cs, shape.rows, shape.k, fill, seed);
        const pad = p.padOut();
        const out_bytes = (p.elems() + 2 * pad) * 2;
        var base = try p.inputs(0, 0);
        p.out = try p.rig.output(out_bytes);
        p.alt = try p.rig.scratch(out_bytes);
        var main = base;
        switch (p.kind) {
            .whole, .split_calls, .one_row => {},
            .offsets => main = try p.inputs(pad_small, pad_cache),
            .ring_alt => {
                main.kc = try p.upload(p.b.kc2, 0);
                main.vc = try p.upload(p.b.vc2, 0);
                main.ring = @intCast(p.b.ring2);
            },
        }
        base.out = shift(u.view(p.alt), pad * 2);
        main.out = shift(u.view(p.out), pad * 2);
        p.base = base;
        p.main = main;
        return true;
    }

    fn encode(p: *Probe, e: mtl.ComputeEncoder, t: Target, first: usize, n_rows: usize) !void {
        const cs = p.cs;
        const row_bytes = cs.heads * data.dim * 2;
        try p.o.attention(e, .{
            .q = shift(t.q, first * row_bytes),
            .rel = shift(t.rel, first * cs.heads * data.d_rel * 2),
            .sink = shift(t.sink, first * cs.heads * 2),
            .gate = shift(t.gate, first * row_bytes),
            .kcache = t.kc,
            .vcache = t.vc,
            .proj = t.proj,
            .out = shift(t.out, first * row_bytes),
            .rows = @intCast(n_rows),
            .pos0 = @intCast(p.shape.k + first),
            .heads = @intCast(cs.heads),
            .kv_heads = @intCast(cs.kv_heads),
            .extent = @intCast(cs.extent),
            .window = @intCast(cs.window),
            .ring = t.ring,
            .scale = data.scale,
        });
    }

    fn compute(p: *Probe, t: Target, mode: Mode) !void {
        const rows = p.shape.rows;
        const pass = p.rig.begin();
        switch (mode) {
            .whole => try p.encode(pass.e, t, 0, rows),
            .split => {
                const cut = (rows + 1) / 2;
                try p.encode(pass.e, t, 0, cut);
                if (cut < rows) try p.encode(pass.e, t, cut, rows - cut);
            },
            .each => {
                for (0..rows) |r| try p.encode(pass.e, t, r, 1);
            },
        }
        try p.rig.finish(pass);
    }

    pub fn run(p: *Probe) !void {
        try p.compute(p.main, switch (p.kind) {
            .split_calls => Mode.split,
            .one_row => Mode.each,
            else => Mode.whole,
        });
    }

    /// The baseline of the seam kinds: the whole call, plain layout, case ring; NaN (0xff) where the views' pad is.
    pub fn replaced(p: *Probe) ![]const u8 {
        const bytes = (p.elems() + 2 * p.padOut()) * 2;
        @memset(p.rig.bytes(p.alt, bytes), 0xff);
        try p.compute(p.base, .whole);
        return p.rig.bytes(p.alt, bytes);
    }

    pub fn count(p: *Probe) usize {
        return p.elems() + 2 * p.padOut();
    }

    pub fn reference(p: *Probe, want: []f64, mag: []f64) !void {
        const a = p.arena.allocator();
        const cs = p.cs;
        const n = p.elems();
        const pad = p.padOut();
        @memset(want[0..pad], std.math.nan(f64));
        @memset(mag[0..pad], 0);
        @memset(want[pad + n ..], std.math.nan(f64));
        @memset(mag[pad + n ..], 0);
        p.att = try a.alloc(f64, n);
        p.err = try a.alloc(f64, n);
        p.sig = try a.alloc(f64, n);
        const cfg = math.Config{ .rows = p.shape.rows, .pos0 = p.shape.k, .heads = cs.heads, .kv_heads = cs.kv_heads, .window = cs.window, .extent = cs.extent, .scale = data.scale };
        const d = math.Data{ .q = p.b.q, .rel = p.b.rel, .sink = p.b.sink, .gate = p.b.gate, .k = p.b.k, .v = p.b.v, .proj = p.b.proj };
        try math.attention(a, cfg, d, .{ .want = want[pad..][0..n], .att = p.att, .err = p.err, .sig = p.sig });
        for (p.att, 0..) |o, i| mag[pad + i] = if (std.math.isNan(o)) 0 else @abs(o);
    }

    /// The allowance derived in the header; the pad around the views of the `offsets` kind is not compared.
    pub fn bounds(p: *Probe, want: []const f64, _: []const f64, allow: []f64) !void {
        const pad = p.padOut();
        @memset(allow, 0);
        const half = 0x1p-8;
        const unit = 0x1p-24;
        for (p.att, p.err, p.sig, 0..) |o, e, s, i| {
            const att = @abs(o);
            const work = s * (e + half * (att + e)) + 4 * unit * s * att;
            const total = work + half * (@abs(want[pad + i]) + work) + 0x1p-133;
            allow[pad + i] = if (std.math.isFinite(total)) total else 0;
        }
    }
};

/// One planted fault in the attention source (or, for the NaN rule, in what it relies on); `gate` must go red.
pub const Mutation = struct { kind: Kind, gate: c.Gate, label: []const u8, from: []const u8, to: []const u8 };

pub const mutations = [_]Mutation{
    .{ .kind = .whole, .gate = .shapes, .label = "control: inkling2 attention, the sink's weight dropped from the sum", .from = "l[h] = m[h] == -INFINITY ? 0.0f : 1.0f;", .to = "l[h] = 0.0f;" },
    .{ .kind = .whole, .gate = .shapes, .label = "control: inkling2 attention, the bias read one distance off", .from = "const ulong dist = p - key;", .to = "const ulong dist = p - key + 1;" },
    .{ .kind = .whole, .gate = .shapes, .label = "control: inkling2 attention, the bias applied at distance extent too", .from = "if (dist < a.extent) {", .to = "if (dist <= a.extent) {" },
    .{ .kind = .whole, .gate = .shapes, .label = "control: inkling2 attention, the window one key too long", .from = "p + 1 - a.window : 0", .to = "p - a.window : 0" },
    .{ .kind = .whole, .gate = .shapes, .label = "control: inkling2 attention, the merge ignoring the last chunk", .from = "A += recs[c * I2_PART + 4 + d] * e;", .to = "if (c + 1 < nch) A += recs[c * I2_PART + 4 + d] * e;" },
    .{ .kind = .whole, .gate = .shapes, .label = "control: inkling2 attention, the biased scores not scaled", .from = "score[h] = score[h] * a.scale + float(bfloat(b));", .to = "score[h] = score[h] + float(bfloat(b));" },
    .{ .kind = .whole, .gate = .shapes, .label = "control: inkling2 attention, the bias not rounded to bf16", .from = "float(bfloat(b))", .to = "b" },
    .{ .kind = .whole, .gate = .nonfinite, .label = "control: inkling2 attention, a -inf score under a -inf max weighed exp(NaN)", .from = "(score[h] == -INFINITY ? 0.0f : metal::precise::exp(score[h] - m_new))", .to = "metal::precise::exp(score[h] - m_new)" },
    .{ .kind = .whole, .gate = .nonfinite, .label = "control: inkling2 attention, a row of only -inf divided by a floored sum", .from = "const float o = (isnan(M) || M == -INFINITY) ? NAN : A / L;", .to = "const float o = A / max(L, 1e-30f);" },
    .{ .kind = .whole, .gate = .shapes, .label = "control: inkling2 rows attention, the rescale skipped", .from = "rp[0] = rp[0] * al_tg[sg * I2_RB + (lane >> 2)];", .to = "rp[0] = rp[0];" },
    .{ .kind = .whole, .gate = .shapes, .label = "control: inkling2 rows attention, the rescale checked on row 0 only", .from = "simd_all(al_tg[sg * I2_RB + row_of_lane] == 1.0f)", .to = "simd_all(al_tg[sg * I2_RB] == 1.0f)" },
    .{ .kind = .whole, .gate = .shapes, .label = "control: inkling2 rows attention, the first past-the-tile block skipped", .from = "for (ulong block = first; block * 32 < ke - ks; block += I2_ATTN_SGS) {", .to = "for (ulong block = first + I2_ATTN_SGS; block * 32 < ke - ks; block += I2_ATTN_SGS) {" },
    .{ .kind = .whole, .gate = .shapes, .label = "control: inkling2 rows attention, the bias read one distance off", .from = "const ulong back = pf + i - key;", .to = "const ulong back = pf + i - key + 1;" },
    .{ .kind = .whole, .gate = .shapes, .label = "control: inkling2 rows attention, the sink's weight dropped from the sum", .from = "l[i] = m[i] == -INFINITY ? 0.0f : 1.0f;", .to = "l[i] = 0.0f;" },
    .{ .kind = .whole, .gate = .shapes, .label = "control: inkling2 rows attention, the streams' sums not scaled to the max", .from = "L[i] += mrg_l[s * I2_RB + i] * e;", .to = "L[i] += mrg_l[s * I2_RB + i];" },
    .{ .kind = .one_row, .gate = .shapes, .label = "control: inkling2 attention, the direct kernel's sum off by 1e-4", .from = "out[at] = i2_attn_finish(M, L, A, float(gate[at]));\n    } else {", .to = "out[at] = i2_attn_finish(M, L * 1.0001f, A, float(gate[at]));\n    } else {" },
};

pub fn controlSource(a: std.mem.Allocator, m: Mutation) ![]u8 {
    return u.mutate(a, sources.inkling2_ops, m.from, m.to);
}

/// The control's gate is its mutation's (found by label: the suite passes the mutation's label to `Probe.init`).
pub fn controlKernel(p: *Probe) g.Kernel {
    var k = p.kernel();
    k.control = .shapes;
    for (mutations) |m| if (std.mem.eql(u8, m.label, p.label)) {
        k.control = m.gate;
    };
    return k;
}
