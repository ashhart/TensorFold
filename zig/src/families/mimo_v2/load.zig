//! A rank's share of a MiMo-V2 EXL3 pack on the GPU, laid out as the Python family's weights.load builds it.

const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const config = @import("config.zig");
const exl3 = cuda.exl3;
const fmt = core.exl3_format;
const Config = config.Config;

/// Device memory in large blocks, handed out 256-byte aligned (one cuMemAlloc per block, not per tensor).
pub const Arena = struct {
    d: *const cuda.Driver,
    gpa: std.mem.Allocator,
    blocks: std.ArrayList(cuda.DeviceBuffer) = .empty,
    used: usize = 0,
    total: usize = 0,
    pub const block_bytes: usize = 1 << 30;

    pub fn alloc(a: *Arena, bytes: usize) !u64 {
        const n = std.mem.alignForward(usize, @max(bytes, 1), 256);
        if (a.blocks.items.len == 0 or a.used + n > a.blocks.items[a.blocks.items.len - 1].len) {
            try a.blocks.append(a.gpa, try cuda.DeviceBuffer.alloc(a.d, @max(block_bytes, n)));
            a.used = 0;
        }
        const b = a.blocks.items[a.blocks.items.len - 1];
        const at = b.ptr + a.used;
        a.used += n;
        a.total += n;
        return at;
    }
    pub fn upload(a: *Arena, bytes: []const u8) !u64 {
        const at = try a.alloc(bytes.len);
        try check(a.d, a.d.api.cuMemcpyHtoD_v2(at, bytes.ptr, bytes.len));
        return at;
    }
    pub fn zeros(a: *Arena, bytes: usize) !u64 {
        const at = try a.alloc(bytes);
        try check(a.d, a.d.api.cuMemsetD8_v2(at, 0, std.mem.alignForward(usize, @max(bytes, 1), 256)));
        return at;
    }
    pub fn deinit(a: *Arena) void {
        for (a.blocks.items) |*b| b.free();
        a.blocks.deinit(a.gpa);
    }
};

fn check(d: *const cuda.Driver, r: anytype) !void {
    try d.check(r, "mimo_v2 load");
}

/// An EXL3 matrix on the host: trellis int16 [tk, tn, 8 * k2], suh fp16 [K], svh fp16 [N].
pub const Mat = struct {
    trellis: []i16,
    suh: []f16,
    svh: []f16,
    tk: usize,
    tn: usize,
    k2: u32,

    pub fn deinit(m: Mat, a: std.mem.Allocator) void {
        a.free(m.trellis);
        a.free(m.suh);
        a.free(m.svh);
    }
    pub fn k(m: Mat) usize {
        return m.tk * 16;
    }
    pub fn n(m: Mat) usize {
        return m.tn * 16;
    }
};

pub fn readMat(a: std.mem.Allocator, ck: *core.Checkpoint, prefix: []const u8) !Mat {
    var nb: [256]u8 = undefined;
    if ((ck.get(try std.fmt.bufPrint(&nb, "{s}.mul1", .{prefix})) catch null) == null) return error.NotMul1;
    const t = try ck.get(try std.fmt.bufPrint(&nb, "{s}.trellis", .{prefix}));
    if (t.dtype != .i16 or t.rank != 3 or t.shape[2] % 8 != 0) return error.BadTrellis;
    const tr = try a.alloc(i16, t.numel());
    errdefer a.free(tr);
    @memcpy(std.mem.sliceAsBytes(tr), t.bytes);
    const suh = try scales(a, ck, prefix, "suh", t.shape[0] * 16);
    errdefer a.free(suh);
    return .{ .trellis = tr, .suh = suh, .svh = try scales(a, ck, prefix, "svh", t.shape[1] * 16), .tk = t.shape[0], .tn = t.shape[1], .k2 = @intCast(t.shape[2] / 8) };
}

fn scales(a: std.mem.Allocator, ck: *core.Checkpoint, prefix: []const u8, part: []const u8, size: usize) ![]f16 {
    var nb: [256]u8 = undefined;
    const t = try ck.get(try std.fmt.bufPrint(&nb, "{s}.{s}", .{ prefix, part }));
    const out = try a.alloc(f16, size);
    errdefer a.free(out);
    if (t.dtype == .f16 and t.numel() == size) {
        @memcpy(std.mem.sliceAsBytes(out), t.bytes);
    } else if (t.dtype == .i16 and t.numel() * 16 == size) {
        const w = try a.alloc(i16, t.numel());
        defer a.free(w);
        @memcpy(std.mem.sliceAsBytes(w), t.bytes);
        fmt.unpackSigns(w, out);
    } else return error.BadScales;
    return out;
}

/// weights.py cut(): outputs [lo, hi) (trellis columns, svh) or inputs [lo, hi) (trellis rows, suh); `m` is freed.
pub fn cut(a: std.mem.Allocator, m: Mat, c: config.Cut) !Mat {
    if (c.axis == .whole) return m;
    defer m.deinit(a);
    const w = 8 * @as(usize, m.k2);
    if (c.lo % 128 != 0 or c.hi % 128 != 0) return error.CutNotOnBlock;
    if (c.axis == .n) {
        const tn = (c.hi - c.lo) / 16;
        const tr = try a.alloc(i16, m.tk * tn * w);
        for (0..m.tk) |kt| @memcpy(tr[kt * tn * w ..][0 .. tn * w], m.trellis[(kt * m.tn + c.lo / 16) * w ..][0 .. tn * w]);
        return .{ .trellis = tr, .suh = try a.dupe(f16, m.suh), .svh = try a.dupe(f16, m.svh[c.lo..c.hi]), .tk = m.tk, .tn = tn, .k2 = m.k2 };
    }
    const tk = (c.hi - c.lo) / 16;
    return .{ .trellis = try a.dupe(i16, m.trellis[(c.lo / 16) * m.tn * w ..][0 .. tk * m.tn * w]), .suh = try a.dupe(f16, m.suh[c.lo..c.hi]), .svh = try a.dupe(f16, m.svh), .tk = tk, .tn = m.tn, .k2 = m.k2 };
}

/// linear.strips: trellis int32 words [K/16, N/16, W] -> [N/128, K/16, 8, W].
pub fn strips(a: std.mem.Allocator, m: Mat) ![]u32 {
    const w32 = 4 * @as(usize, m.k2);
    const words: []align(2) const u32 = std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes(m.trellis));
    const out = try a.alloc(u32, words.len);
    for (0..m.tk) |kt| for (0..m.tn / 8) |nb8| for (0..8) |j| {
        @memcpy(out[((nb8 * m.tk + kt) * 8 + j) * w32 ..][0..w32], words[((kt * m.tn) + nb8 * 8 + j) * w32 ..][0..w32]);
    };
    return out;
}

/// Exl3Linear.from_tensors (strips) on the device, with its zeroed split counters.
pub fn linear(a: std.mem.Allocator, ar: *Arena, m: Mat) !exl3.Linear {
    if (m.k() % 128 != 0 or m.n() % 128 != 0) return error.NotBlocked;
    const st = try strips(a, m);
    defer a.free(st);
    return .{
        .words = try ar.upload(std.mem.sliceAsBytes(st)),
        .suh = try ar.upload(std.mem.sliceAsBytes(m.suh)),
        .svh = try ar.upload(std.mem.sliceAsBytes(m.svh)),
        .counters = try ar.zeros(8 * (m.n() / 128) * 4),
        .k = m.k(),
        .n = m.n(),
        .k2 = m.k2,
        .split = exl3.plan(m.k(), m.n()),
    };
}

/// torch's float -> bfloat16 (round to nearest even; NaN kept quiet).
pub fn bf16Round(x: f32) f32 {
    const u: u32 = @bitCast(x);
    if (std.math.isNan(x)) return @bitCast((u | 0x0040_0000) & 0xFFFF_0000);
    const r = u + 0x7FFF + ((u >> 16) & 1);
    return @bitCast(r & 0xFFFF_0000);
}

/// A tensor as fp32 (`.float()`): bf16, fp16 or fp32 storage.
pub fn asF32(a: std.mem.Allocator, t: core.safetensors.Tensor) ![]f32 {
    const n = t.numel();
    const out = try a.alloc(f32, n);
    switch (t.dtype) {
        .f32 => @memcpy(std.mem.sliceAsBytes(out), t.bytes),
        .bf16 => for (out, 0..) |*o, i| {
            o.* = @bitCast(@as(u32, std.mem.readInt(u16, t.bytes[2 * i ..][0..2], .little)) << 16);
        },
        .f16 => for (out, 0..) |*o, i| {
            o.* = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, t.bytes[2 * i ..][0..2], .little))));
        },
        else => return error.BadDType,
    }
    return out;
}

pub const Layer = struct {
    input_norm: u64, // fp32 [D]
    post_norm: u64,
    q: exl3.Linear,
    k: exl3.Linear,
    v: exl3.Linear, // 192 a head: V + 64 pad
    o: exl3.Linear, // a rank's input columns: partial outputs under TP
    sinks: u64 = 0, // fp32 [H] of this rank's heads, or 0
    experts: ?exl3.Experts = null,
    router: u64 = 0, // fp16 [E, D]
    bias: u64 = 0, // fp32 [E]
    route_scale: f32 = 1.0,
    gate: ?exl3.Linear = null, // the dense MLP (layer 0)
    up: ?exl3.Linear = null,
    down: ?exl3.Linear = null,
};

pub const Weights = struct {
    full: Config,
    cfg: Config, // this rank's view
    rank: usize,
    world: usize,
    embed: u64, // bf16 [V, D], every rank whole
    layers: []Layer,
    norm: u64,
    head: exl3.Linear, // a rank's V / world rows
    arena: Arena,
    gpa: std.mem.Allocator,

    pub fn deinit(w: *Weights) void {
        for (w.layers) |L| if (L.experts) |e| {
            w.gpa.free(e.host_ptr);
            w.gpa.free(e.host_k2);
        };
        w.gpa.free(w.layers);
        w.arena.deinit();
    }
};

pub fn load(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, model_dir: []const u8, rank: usize, world: usize, log: bool) !Weights {
    const cfg_text = try readFile(gpa, io, model_dir, "config.json");
    defer gpa.free(cfg_text);
    const gen_text = readFile(gpa, io, model_dir, "generation_config.json") catch null;
    defer if (gen_text) |g| gpa.free(g);
    const full = try Config.parse(gpa, cfg_text, gen_text);
    const view = try full.rankView(world);
    var ck = try core.Checkpoint.openModel(gpa, io, model_dir);
    defer ck.close();
    var w: Weights = .{ .full = full, .cfg = view, .rank = rank, .world = world, .embed = 0, .layers = try gpa.alloc(Layer, full.layers), .norm = 0, .head = undefined, .arena = .{ .d = d, .gpa = gpa }, .gpa = gpa };
    errdefer w.deinit();
    const ar = &w.arena;
    const emb = try ck.get("model.embed_tokens.weight");
    if (emb.dtype != .bf16) return error.EmbedNotBf16; // weights.py: .to(bfloat16) of bf16 storage is the identity
    w.embed = try ar.upload(emb.bytes);

    var names: std.ArrayList(u8) = .empty;
    defer names.deinit(gpa);
    try names.ensureTotalCapacity(gpa, 1 << 20);
    var cuts: std.ArrayList(config.Cut) = .empty;
    defer cuts.deinit(gpa);
    var nb: [256]u8 = undefined;
    const t0 = std.Io.Clock.awake.now(io).toNanoseconds();
    for (0..full.layers) |i| {
        names.clearRetainingCapacity();
        cuts.clearRetainingCapacity();
        try config.layerCuts(full, i, rank, world, &names, gpa, &cuts);
        const L = &w.layers[i];
        L.* = .{ .input_norm = 0, .post_norm = 0, .q = undefined, .k = undefined, .v = undefined, .o = undefined };
        L.input_norm = try f32Upload(gpa, ar, try ck.get(try std.fmt.bufPrint(&nb, "model.layers.{d}.input_layernorm.weight", .{i})));
        L.post_norm = try f32Upload(gpa, ar, try ck.get(try std.fmt.bufPrint(&nb, "model.layers.{d}.post_attention_layernorm.weight", .{i})));
        inline for (.{ "q", "k", "v", "o" }, 0..) |f, j| @field(L, f) = try linear(gpa, ar, try cut(gpa, try readMat(gpa, &ck, cuts.items[j].prefix), cuts.items[j]));
        if (if (full.full[i]) full.sinks_full else full.sinks_swa) {
            const s = try asF32(gpa, try ck.get(try std.fmt.bufPrint(&nb, "model.layers.{d}.self_attn.attention_sink_bias", .{i})));
            defer gpa.free(s);
            L.sinks = try ar.upload(std.mem.sliceAsBytes(s[rank * view.heads .. (rank + 1) * view.heads]));
        }
        if (full.moe[i]) {
            L.experts = try experts(gpa, ar, &ck, full, cuts.items[4..]);
            const r = try ck.get(try std.fmt.bufPrint(&nb, "model.layers.{d}.mlp.gate.weight", .{i}));
            const rf = try asF32(gpa, r);
            defer gpa.free(rf);
            const r16 = try gpa.alloc(f16, rf.len); // .to(bfloat16) then .to(fp16 ACT); weights.py checks it is exact
            defer gpa.free(r16);
            for (rf, r16) |x, *o| {
                const b = bf16Round(x);
                o.* = @floatCast(b);
                if (@as(f32, @floatCast(o.*)) != b) return error.RouterNotExact;
            }
            L.router = try ar.upload(std.mem.sliceAsBytes(r16));
            L.bias = try f32Upload(gpa, ar, try ck.get(try std.fmt.bufPrint(&nb, "model.layers.{d}.mlp.gate.e_score_correction_bias", .{i})));
            L.route_scale = if (i == config.interm_div_layer) config.interm_div else 1.0;
        } else {
            L.gate = try linear(gpa, ar, try cut(gpa, try readMat(gpa, &ck, cuts.items[4].prefix), cuts.items[4]));
            L.up = try linear(gpa, ar, try cut(gpa, try readMat(gpa, &ck, cuts.items[5].prefix), cuts.items[5]));
            L.down = try linear(gpa, ar, try cut(gpa, try readMat(gpa, &ck, cuts.items[6].prefix), cuts.items[6]));
        }
        if (log and (i % 8 == 7 or i == full.layers - 1)) {
            const s = @as(f64, @floatFromInt(std.Io.Clock.awake.now(io).toNanoseconds() - t0)) / 1e9;
            std.debug.print("[tensorfold] mimo_v2 (zig): {d}/{d} layers in {d:.0}s, {d:.1} GiB\n", .{ i + 1, full.layers, s, @as(f64, @floatFromInt(ar.total)) / (1 << 30) });
        }
    }
    w.norm = try f32Upload(gpa, ar, try ck.get("model.norm.weight"));
    const hc = config.headCut(full, rank, world);
    w.head = try linear(gpa, ar, try cut(gpa, try readMat(gpa, &ck, hc.prefix), hc));
    return w;
}

fn f32Upload(gpa: std.mem.Allocator, ar: *Arena, t: core.safetensors.Tensor) !u64 {
    const f = try asF32(gpa, t);
    defer gpa.free(f);
    return ar.upload(std.mem.sliceAsBytes(f));
}

/// experts.prepare: per expert (gate, up, down) cut trellises in the stored layout, referenced by pointer tables.
fn experts(gpa: std.mem.Allocator, ar: *Arena, ck: *core.Checkpoint, full: Config, cuts: []const config.Cut) !exl3.Experts {
    const E = full.experts;
    if (cuts.len != 3 * E) return error.ExpertCuts;
    const ptrs = try gpa.alloc(u64, 3 * E); // kept: Experts.host_ptr
    errdefer gpa.free(ptrs);
    const k2s = try gpa.alloc(i32, 3 * E); // kept: Experts.host_k2
    errdefer gpa.free(k2s);
    var st: [6]std.ArrayList(f16) = @splat(.empty); // suh_g, suh_u, svh_g, svh_u, suh_d, svh_d
    defer for (&st) |*l| l.deinit(gpa);
    var lo: [2]u32 = .{ 99, 99 };
    var hi: [2]u32 = .{ 0, 0 };
    var dims: usize = 0;
    var width: usize = 0;
    for (0..E) |e| for (0..3) |j| {
        const m = try cut(gpa, try readMat(gpa, ck, cuts[3 * e + j].prefix), cuts[3 * e + j]);
        defer m.deinit(gpa);
        ptrs[j * E + e] = try ar.upload(std.mem.sliceAsBytes(m.trellis));
        k2s[j * E + e] = @intCast(m.k2);
        const g: usize = if (j == 2) 1 else 0;
        lo[g] = @min(lo[g], m.k2);
        hi[g] = @max(hi[g], m.k2);
        switch (j) {
            0 => {
                dims = m.k();
                width = m.n();
                try st[0].appendSlice(gpa, m.suh);
                try st[2].appendSlice(gpa, m.svh);
            },
            1 => {
                try st[1].appendSlice(gpa, m.suh);
                try st[3].appendSlice(gpa, m.svh);
            },
            else => {
                try st[4].appendSlice(gpa, m.suh);
                try st[5].appendSlice(gpa, m.svh);
            },
        }
    };
    return .{
        .gate_ptr = try ar.upload(std.mem.sliceAsBytes(ptrs[0..E])),
        .up_ptr = try ar.upload(std.mem.sliceAsBytes(ptrs[E .. 2 * E])),
        .down_ptr = try ar.upload(std.mem.sliceAsBytes(ptrs[2 * E ..])),
        .gate_k2 = try ar.upload(std.mem.sliceAsBytes(k2s[0..E])),
        .up_k2 = try ar.upload(std.mem.sliceAsBytes(k2s[E .. 2 * E])),
        .down_k2 = try ar.upload(std.mem.sliceAsBytes(k2s[2 * E ..])),
        .suh_g = try ar.upload(std.mem.sliceAsBytes(st[0].items)),
        .suh_u = try ar.upload(std.mem.sliceAsBytes(st[1].items)),
        .svh_g = try ar.upload(std.mem.sliceAsBytes(st[2].items)),
        .svh_u = try ar.upload(std.mem.sliceAsBytes(st[3].items)),
        .suh_d = try ar.upload(std.mem.sliceAsBytes(st[4].items)),
        .svh_d = try ar.upload(std.mem.sliceAsBytes(st[5].items)),
        .count = E,
        .dims = dims,
        .width = width,
        .k2_gu = .{ lo[0], hi[0] },
        .k2_d = .{ lo[1], hi[1] },
        .host_ptr = ptrs,
        .host_k2 = k2s,
    };
}

fn readFile(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, name: []const u8) ![]u8 {
    const path = try std.fs.path.join(gpa, &.{ dir, name });
    defer gpa.free(path);
    return std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 22));
}

test "bf16 rounding is torch's" {
    try std.testing.expectEqual(@as(u32, 0x3F80_0000), @as(u32, @bitCast(bf16Round(1.0))));
    try std.testing.expectEqual(@as(u32, 0x3F80_0000), @as(u32, @bitCast(bf16Round(@bitCast(@as(u32, 0x3F80_8000)))))); // tie to even
    try std.testing.expectEqual(@as(u32, 0x3F82_0000), @as(u32, @bitCast(bf16Round(@bitCast(@as(u32, 0x3F81_8000)))))); // tie to even (up)
    try std.testing.expectEqual(@as(u32, 0x3F81_0000), @as(u32, @bitCast(bf16Round(@bitCast(@as(u32, 0x3F80_8001))))));
}
