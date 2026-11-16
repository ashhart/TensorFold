//! GPU check of Ling MLA around k3_mla: RoPE, head-gate expand, mixer vs mla.zig.
const std = @import("std");
const mtl = @import("metal");
const mla = @import("mla");
const metallib = @import("kimi_metallib");

const source = @import("ling_metal").mla;
const mla_splits: u32 = 16;

const MlaRow = extern struct { slot: u32, pos: u32 };
const MlaArgs = extern struct {
    cache: u64,
    slot_keys: u32,
    heads: u32,
    head0: u32,
    rows: u32,
    q_stride: u32,
    kv_stride: u32,
    gate_stride: u32,
    out_stride: u32,
    eps: f32,
    scale: f32,
};

comptime {
    if (@sizeOf(MlaArgs) != 48) @compileError("MlaArgs must stay 48 bytes, matching k3");
}

fn toBf16(v: f32) u16 {
    const bits: u32 = @bitCast(v);
    if (std.math.isNan(v)) return @truncate((bits >> 16) | 0x40);
    return @truncate((bits +% 0x7fff +% ((bits >> 16) & 1)) >> 16);
}
fn fromBf16(h: u16) f32 {
    return @bitCast(@as(u32, h) << 16);
}

fn fill(seed: *u64, n: usize, a: std.mem.Allocator) ![]f32 {
    const x = try a.alloc(f32, n);
    for (x) |*v| {
        seed.* = seed.* *% 1103515245 +% 12345;
        v.* = @as(f32, @floatFromInt((seed.* >> 16) & 0x7fff)) / 16384.0 - 1.0;
    }
    return x;
}

fn putBf16(buf: mtl.Buffer, x: []const f32) void {
    const d = buf.slice(u16, x.len);
    for (x, d) |v, *o| o.* = toBf16(v);
}

fn roundSlice(x: []f32) void {
    for (x) |*v| v.* = fromBf16(toBf16(v.*));
}

/// Host f64 cos/sin of each row's pos, rounded to f32. Layout [t][i][comp] = t*64 + i*2 + comp.
fn writeCs(cs: []f32, rows: []const MlaRow, theta: f64) void {
    std.debug.assert(cs.len >= rows.len * mla.nfreq * 2);
    for (rows, 0..) |r, t| {
        for (0..mla.nfreq) |i| {
            const ang = @as(f64, @floatFromInt(r.pos)) * mla.invFreq(theta, i);
            const at = (t * mla.nfreq + i) * 2;
            cs[at] = @floatCast(@cos(ang));
            cs[at + 1] = @floatCast(@sin(ang));
        }
    }
}

const Gpu = struct {
    device: mtl.Device,
    queue: mtl.Queue,
    kimi: mtl.Library,
    ling: mtl.Library,
    cache: mtl.Pipeline,
    qlat: mtl.Pipeline,
    attend: mtl.Pipeline,
    merge: mtl.Pipeline,
    uv: mtl.Pipeline,
    rope_cache: mtl.Pipeline,
    rope_q: mtl.Pipeline,
    gate_expand: mtl.Pipeline,

    fn init() !Gpu {
        const device = try mtl.Device.init();
        errdefer device.deinit();
        const kimi = try mtl.Library.fromBytes(device, metallib.bytes);
        errdefer kimi.deinit();
        const ling = try mtl.Library.fromSource(device, source, mtl.CompileOptions.mlx());
        errdefer ling.deinit();
        const cache = try mtl.Pipeline.init(device, kimi, "k3_mla_cache", false);
        errdefer cache.deinit();
        const qlat = try mtl.Pipeline.init(device, kimi, "k3_mla_qlat", false);
        errdefer qlat.deinit();
        const attend = try mtl.Pipeline.init(device, kimi, "k3_mla_attend", false);
        errdefer attend.deinit();
        const merge = try mtl.Pipeline.init(device, kimi, "k3_mla_merge", false);
        errdefer merge.deinit();
        const uv = try mtl.Pipeline.init(device, kimi, "k3_mla_uv", false);
        errdefer uv.deinit();
        const rope_cache = try mtl.Pipeline.init(device, ling, "tf_ling_mla_rope_cache", false);
        errdefer rope_cache.deinit();
        const rope_q = try mtl.Pipeline.init(device, ling, "tf_ling_mla_rope_q", false);
        errdefer rope_q.deinit();
        const gate_expand = try mtl.Pipeline.init(device, ling, "tf_ling_mla_gate_expand", false);
        errdefer gate_expand.deinit();
        const queue = try device.queue();
        return .{
            .device = device,
            .queue = queue,
            .kimi = kimi,
            .ling = ling,
            .cache = cache,
            .qlat = qlat,
            .attend = attend,
            .merge = merge,
            .uv = uv,
            .rope_cache = rope_cache,
            .rope_q = rope_q,
            .gate_expand = gate_expand,
        };
    }

    fn deinit(g: *Gpu) void {
        g.gate_expand.deinit();
        g.rope_q.deinit();
        g.rope_cache.deinit();
        g.uv.deinit();
        g.merge.deinit();
        g.attend.deinit();
        g.qlat.deinit();
        g.cache.deinit();
        g.ling.deinit();
        g.kimi.deinit();
        g.queue.deinit();
        g.device.deinit();
    }
};

const Launch = struct {
    s: mla.Shape,
    cap: u32,
    b_q: mtl.Buffer,
    b_kv: mtl.Buffer,
    b_w: mtl.Buffer,
    b_kvb: mtl.Buffer,
    b_g: mtl.Buffer,
    b_g2: mtl.Buffer,
    b_y: mtl.Buffer,
    b_rows: mtl.Buffer,
    b_cache: mtl.Buffer,
    b_qlat: mtl.Buffer,
    b_partial: mtl.Buffer,
    b_olat: mtl.Buffer,
    b_cs: mtl.Buffer,

    fn open(g: Gpu, s: mla.Shape, cap: u32, q: []const f32, kv: []const f32, w_rms: []const f32, kvb: []const f32, gate: []const f32) !Launch {
        const H = s.heads;
        const shared = mtl.ResourceOptions.shared;
        const b_q = try g.device.buffer(q.len * 2, shared);
        const b_kv = try g.device.buffer(kv.len * 2, shared);
        const b_w = try g.device.buffer(w_rms.len * 2, shared);
        const b_kvb = try g.device.buffer(kvb.len * 2, shared);
        const b_g = try g.device.buffer(gate.len * 2, shared);
        const b_g2 = try g.device.buffer(cap * H * mla.vd * 2, shared);
        const b_y = try g.device.buffer(cap * H * mla.vd * 2, shared);
        const b_rows = try g.device.buffer(cap * @sizeOf(MlaRow), shared);
        const b_cache = try g.device.buffer(cap * mla.key * 2, shared);
        const b_qlat = try g.device.buffer(cap * H * mla.lat * 4, shared);
        const b_partial = try g.device.buffer(cap * H * mla_splits * (mla.lat + 2) * 4, shared);
        const b_olat = try g.device.buffer(cap * H * mla.lat * 4, shared);
        const b_cs = try g.device.buffer(cap * mla.nfreq * 2 * 4, shared);
        putBf16(b_q, q);
        putBf16(b_kv, kv);
        putBf16(b_w, w_rms);
        putBf16(b_kvb, kvb);
        putBf16(b_g, gate);
        @memset(b_g2.slice(u16, cap * H * mla.vd), 0);
        @memset(b_y.slice(u16, cap * H * mla.vd), 0);
        @memset(b_cache.slice(u16, cap * mla.key), 0);
        return .{
            .s = s,
            .cap = cap,
            .b_q = b_q,
            .b_kv = b_kv,
            .b_w = b_w,
            .b_kvb = b_kvb,
            .b_g = b_g,
            .b_g2 = b_g2,
            .b_y = b_y,
            .b_rows = b_rows,
            .b_cache = b_cache,
            .b_qlat = b_qlat,
            .b_partial = b_partial,
            .b_olat = b_olat,
            .b_cs = b_cs,
        };
    }

    fn deinit(l: *Launch) void {
        l.b_q.deinit();
        l.b_kv.deinit();
        l.b_w.deinit();
        l.b_kvb.deinit();
        l.b_g.deinit();
        l.b_g2.deinit();
        l.b_y.deinit();
        l.b_rows.deinit();
        l.b_cache.deinit();
        l.b_qlat.deinit();
        l.b_partial.deinit();
        l.b_olat.deinit();
        l.b_cs.deinit();
    }

    fn args(l: *Launch, n: u32) MlaArgs {
        const H = l.s.heads;
        const ow = H * mla.vd;
        return .{
            .cache = l.b_cache.gpuAddress(),
            .slot_keys = l.cap,
            .heads = H,
            .head0 = 0,
            .rows = n,
            .q_stride = H * mla.qd,
            .kv_stride = mla.key,
            .gate_stride = ow,
            .out_stride = ow,
            .eps = l.s.eps,
            .scale = @floatCast(l.s.scale()),
        };
    }

    fn fillRows(l: *Launch, first: u32, n: u32) void {
        const rows = l.b_rows.slice(MlaRow, n);
        for (rows, 0..) |*r, i| r.* = .{ .slot = 0, .pos = first + @as(u32, @intCast(i)) };
    }

    fn fillCs(l: *Launch, n: u32) void {
        writeCs(l.b_cs.slice(f32, n * mla.nfreq * 2), l.b_rows.slice(MlaRow, n), l.s.theta);
    }

    fn dispatchMixer(l: *Launch, g: Gpu, first: u32, n: u32) !void {
        const H = l.s.heads;
        const a = l.args(n);
        l.fillRows(first, n);
        l.fillCs(n);
        const q_off = first * a.q_stride * 2;
        const kv_off = first * a.kv_stride * 2;
        const g_off = first * H * 2;
        const y_off = first * a.out_stride * 2;
        const g2_off = first * a.gate_stride * 2;
        const rw = mtl.ResourceUsage.read | mtl.ResourceUsage.write;
        const cb = g.queue.commandBuffer();
        const e = cb.compute(.serial);

        e.setPipeline(g.cache);
        e.setBuffer(l.b_kv, kv_off, 0);
        e.setBuffer(l.b_w, 0, 1);
        e.setBuffer(l.b_rows, 0, 2);
        e.setValue(a, 3);
        e.useResource(l.b_cache, rw);
        e.dispatchGroups(mtl.Size.of(n, 1, 1), mtl.Size.of(256, 1, 1));

        e.setPipeline(g.rope_cache);
        e.setBuffer(l.b_rows, 0, 0);
        e.setValue(a, 1);
        e.setBuffer(l.b_cs, 0, 2);
        e.useResource(l.b_cache, rw);
        e.dispatchGroups(mtl.Size.of(n, 1, 1), mtl.Size.of(32, 1, 1));

        e.setPipeline(g.qlat);
        e.setBuffer(l.b_q, q_off, 0);
        e.setBuffer(l.b_kvb, 0, 1);
        e.setBuffer(l.b_qlat, 0, 2);
        e.setValue(a, 3);
        e.dispatchGroups(mtl.Size.of(2, H, 1), mtl.Size.of(256, 1, 1));

        e.setPipeline(g.rope_q);
        e.setBuffer(l.b_q, q_off, 0);
        e.setValue(a, 1);
        e.setBuffer(l.b_cs, 0, 2);
        e.dispatchGroups(mtl.Size.of(H, n, 1), mtl.Size.of(32, 1, 1));

        e.setPipeline(g.attend);
        e.setBuffer(l.b_q, q_off, 0);
        e.setBuffer(l.b_qlat, 0, 1);
        e.setBuffer(l.b_rows, 0, 2);
        e.setBuffer(l.b_partial, 0, 3);
        e.setValue(a, 4);
        e.useResource(l.b_cache, mtl.ResourceUsage.read);
        const fit: u32 = @intCast(@min(32, g.attend.maxThreads() / 32));
        const groups = (H + fit - 1) / fit;
        const per = (H + groups - 1) / groups;
        e.dispatchGroups(mtl.Size.of(groups, mla_splits, n), mtl.Size.of(32 * per, 1, 1));

        e.setPipeline(g.merge);
        e.setBuffer(l.b_partial, 0, 0);
        e.setBuffer(l.b_rows, 0, 1);
        e.setBuffer(l.b_olat, 0, 2);
        e.setValue(a, 3);
        e.dispatchGroups(mtl.Size.of(H, n, 1), mtl.Size.of(256, 1, 1));

        e.setPipeline(g.gate_expand);
        e.setBuffer(l.b_g, g_off, 0);
        e.setBuffer(l.b_g2, g2_off, 1);
        e.setValue(a, 2);
        e.dispatchThreads(mtl.Size.of(H * mla.vd, n, 1), mtl.Size.of(128, 1, 1));

        e.setPipeline(g.uv);
        e.setBuffer(l.b_olat, 0, 0);
        e.setBuffer(l.b_kvb, 0, 1);
        e.setBuffer(l.b_g2, g2_off, 2);
        e.setBuffer(l.b_y, y_off, 3);
        e.setValue(a, 4);
        e.dispatchGroups(mtl.Size.of(mla.vd / 4, H, 1), mtl.Size.of(128, 1, 1));

        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |why| {
            std.debug.print("GPU mixer failed: {s}\n", .{why});
            return error.GpuFailed;
        }
    }

    fn y(l: *Launch) []u16 {
        return l.b_y.slice(u16, l.cap * l.s.outWidth());
    }
};

const Dist = struct { rel_l2: f64, max_abs: f64 };

fn dist(got: []const f64, ref: []const f64) Dist {
    var num: f64 = 0;
    var den: f64 = 0;
    var max_abs: f64 = 0;
    for (got, ref) |a, b| {
        const d = a - b;
        num += d * d;
        den += b * b;
        max_abs = @max(max_abs, @abs(d));
    }
    const rel = if (den == 0) @sqrt(num) else @sqrt(num) / @sqrt(den);
    return .{ .rel_l2 = rel, .max_abs = max_abs };
}

fn bitsToF64(bits: []const u16, out: []f64) void {
    for (bits, out) |h, *o| o.* = fromBf16(h);
}

fn gateNamed(tag: []const u8, metal: []const u16, f64_y: []const f64, bf16_y: []const f64) !void {
    const a = std.testing.allocator;
    const metal_f = try a.alloc(f64, metal.len);
    defer a.free(metal_f);
    bitsToF64(metal, metal_f);
    try gateNamedF64(tag, metal_f, f64_y, bf16_y);
}

fn gateNamedF64(tag: []const u8, metal: []const f64, f64_y: []const f64, bf16_y: []const f64) !void {
    const m = dist(metal, f64_y);
    const b = dist(bf16_y, f64_y);
    std.debug.print("{s} Metal-to-f64 rel L2 {e} max abs {e}; bf16-ref-to-f64 rel L2 {e} max abs {e}\n", .{ tag, m.rel_l2, m.max_abs, b.rel_l2, b.max_abs });
    if (m.rel_l2 > b.rel_l2 or m.max_abs > b.max_abs) {
        std.debug.print("{s} gate miss: Metal further from f64 than bf16-ref\n", .{tag});
        return error.Gate;
    }
}

fn oProj(out: []f64, y: []const f64, wo: []const f32, rows: u32, hidden: u32, width: u32) void {
    std.debug.assert(out.len == rows * hidden and y.len == rows * width and wo.len == hidden * width);
    for (0..rows) |t| {
        for (0..hidden) |j| {
            var acc: f64 = 0;
            const wr = j * width;
            const yr = t * width;
            for (0..width) |i| acc += y[yr + i] * @as(f64, wo[wr + i]);
            out[t * hidden + j] = acc;
        }
    }
}

fn takeU32(b: []const u8, off: *usize) u32 {
    const v = std.mem.readInt(u32, b[off.*..][0..4], .little);
    off.* += 4;
    return v;
}

fn takeF32s(alloc: std.mem.Allocator, b: []const u8, off: *usize, n: usize) ![]f32 {
    const x = try alloc.alloc(f32, n);
    @memcpy(std.mem.sliceAsBytes(x), b[off.* .. off.* + n * 4]);
    off.* += n * 4;
    return x;
}

fn takeF64s(alloc: std.mem.Allocator, b: []const u8, off: *usize, n: usize) ![]f64 {
    const x = try alloc.alloc(f64, n);
    @memcpy(std.mem.sliceAsBytes(x), b[off.* .. off.* + n * 8]);
    off.* += n * 8;
    return x;
}

const k3_gpu_rows: u32 = 256;
const dump_dir = "/tmp/ling-mla-dumps";
const dump_mixer_path = dump_dir ++ "/mla_tiny_mixer.bin";
const dump_a1_t4_path = dump_dir ++ "/mla_tiny_t4_pos0.bin";
const dump_a1_t2_path = dump_dir ++ "/mla_tiny_t2_pos30000.bin";
const dump_py = "zig/src/families/ling/oracle/mla_dump.py";

const DumpCase = struct {
    T: u32,
    pos0: u32,
    q: []f32,
    kv: []f32,
    g: []f32,
    y_pre: []f64,
    y64: []f64,
    ybf: []f64,
};

const Dump = struct {
    hidden: u32,
    layer: u32,
    heads: u32,
    w: []f32,
    kvb: []f32,
    Wo: []f32,
    cases: []DumpCase,

    fn load(alloc: std.mem.Allocator, bytes: []const u8) !Dump {
        try std.testing.expect(std.mem.eql(u8, bytes[0..8], "TFLINGM1"));
        var off: usize = 8;
        const hidden = takeU32(bytes, &off);
        const layer = takeU32(bytes, &off);
        const heads = takeU32(bytes, &off);
        const n_cases = takeU32(bytes, &off);
        const w = try takeF32s(alloc, bytes, &off, mla.lat);
        errdefer alloc.free(w);
        const kvb = try takeF32s(alloc, bytes, &off, heads * (mla.nope + mla.vd) * mla.lat);
        errdefer alloc.free(kvb);
        const Wo = try takeF32s(alloc, bytes, &off, hidden * heads * mla.vd);
        errdefer alloc.free(Wo);
        const cases = try alloc.alloc(DumpCase, n_cases);
        errdefer alloc.free(cases);
        for (cases) |*c| {
            const T = takeU32(bytes, &off);
            const pos0 = takeU32(bytes, &off);
            const q = try takeF32s(alloc, bytes, &off, T * heads * mla.qd);
            const kv = try takeF32s(alloc, bytes, &off, T * mla.key);
            const g = try takeF32s(alloc, bytes, &off, T * heads);
            const y_pre = try takeF64s(alloc, bytes, &off, T * heads * mla.vd);
            const y64 = try takeF64s(alloc, bytes, &off, T * hidden);
            const ybf = try takeF64s(alloc, bytes, &off, T * hidden);
            c.* = .{ .T = T, .pos0 = pos0, .q = q, .kv = kv, .g = g, .y_pre = y_pre, .y64 = y64, .ybf = ybf };
        }
        try std.testing.expectEqual(bytes.len, off);
        return .{ .hidden = hidden, .layer = layer, .heads = heads, .w = w, .kvb = kvb, .Wo = Wo, .cases = cases };
    }

    fn deinit(d: Dump, alloc: std.mem.Allocator) void {
        for (d.cases) |c| {
            alloc.free(c.q);
            alloc.free(c.kv);
            alloc.free(c.g);
            alloc.free(c.y_pre);
            alloc.free(c.y64);
            alloc.free(c.ybf);
        }
        alloc.free(d.cases);
        alloc.free(d.w);
        alloc.free(d.kvb);
        alloc.free(d.Wo);
    }
};

fn parseA1(alloc: std.mem.Allocator, bytes: []const u8) !struct { T: u32, hidden: u32, layer: u32, pos0: u32, y64: []f64, ybf: []f64 } {
    try std.testing.expect(std.mem.eql(u8, bytes[0..8], "TFLINGA1"));
    var off: usize = 8;
    const T = takeU32(bytes, &off);
    const hidden = takeU32(bytes, &off);
    const layer = takeU32(bytes, &off);
    const pos0 = takeU32(bytes, &off);
    const n = T * hidden;
    const y64 = try takeF64s(alloc, bytes, &off, n);
    errdefer alloc.free(y64);
    const ybf = try takeF64s(alloc, bytes, &off, n);
    errdefer alloc.free(ybf);
    try std.testing.expectEqual(bytes.len, off);
    return .{ .T = T, .hidden = hidden, .layer = layer, .pos0 = pos0, .y64 = y64, .ybf = ybf };
}

fn dupeRound(alloc: std.mem.Allocator, src: []const f32) ![]f32 {
    const d = try alloc.dupe(f32, src);
    roundSlice(d);
    return d;
}

fn expectBitsEqual(tag: []const u8, a: []const u16, b: []const u16) !void {
    if (std.mem.eql(u16, a, b)) return;
    var n: usize = 0;
    var first: usize = 0;
    for (a, b, 0..) |x, yv, i| {
        if (x != yv) {
            if (n == 0) first = i;
            n += 1;
        }
    }
    std.debug.print("{s} bits-equal miss: {d} of {d} first i={d} a={x} b={x}\n", .{ tag, n, a.len, first, a[first], b[first] });
    return error.BitsEqual;
}

const Inputs = struct {
    q: []f32,
    kv: []f32,
    w: []f32,
    kvb: []f32,
    g: []f32,

    fn make(alloc: std.mem.Allocator, s: mla.Shape, rows: u32, seed0: u64) !Inputs {
        var seed = seed0;
        const H = s.heads;
        const q = try fill(&seed, rows * H * mla.qd, alloc);
        const kv = try fill(&seed, rows * mla.key, alloc);
        const w = try fill(&seed, mla.lat, alloc);
        const kvb = try fill(&seed, H * (mla.nope + mla.vd) * mla.lat, alloc);
        const g = try fill(&seed, rows * H, alloc);
        for (w) |*v| v.* = 0.5 + 0.5 * v.*;
        roundSlice(q);
        roundSlice(kv);
        roundSlice(w);
        roundSlice(kvb);
        roundSlice(g);
        return .{ .q = q, .kv = kv, .w = w, .kvb = kvb, .g = g };
    }

    fn free(in: Inputs, alloc: std.mem.Allocator) void {
        alloc.free(in.q);
        alloc.free(in.kv);
        alloc.free(in.w);
        alloc.free(in.kvb);
        alloc.free(in.g);
    }
};

fn checkMixer(rows: u32) !void {
    const a = std.testing.allocator;
    const s = mla.Shape{ .heads = 16, .eps = mla.eps_tiny, .theta = mla.theta_tiny };
    const in = try Inputs.make(a, s, rows, 0x6d6c61);
    defer in.free(a);
    const n = rows * s.outWidth();
    const want64 = try a.alloc(f64, n);
    defer a.free(want64);
    const want_bf = try a.alloc(f64, n);
    defer a.free(want_bf);
    try mla.mixF64(a, want64, in.q, in.kv, in.w, in.kvb, in.g, s, rows, 0);
    try mla.mixBf16(a, want_bf, in.q, in.kv, in.w, in.kvb, in.g, s, rows, 0);

    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    var g = try Gpu.init();
    defer g.deinit();
    var l = try Launch.open(g, s, rows, in.q, in.kv, in.w, in.kvb, in.g);
    defer l.deinit();
    try l.dispatchMixer(g, 0, rows);
    var buf: [64]u8 = undefined;
    const tag = try std.fmt.bufPrint(&buf, "mla {d} rows", .{rows});
    try gateNamed(tag, l.y(), want64, want_bf);
}

fn checkChunks() !void {
    const a = std.testing.allocator;
    const rows: u32 = 4;
    const s = mla.Shape{ .heads = 16, .eps = mla.eps_tiny, .theta = mla.theta_tiny };
    const in = try Inputs.make(a, s, rows, 0x6d6c61);
    defer in.free(a);
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    var g = try Gpu.init();
    defer g.deinit();
    var whole = try Launch.open(g, s, rows, in.q, in.kv, in.w, in.kvb, in.g);
    defer whole.deinit();
    try whole.dispatchMixer(g, 0, rows);
    var chunks = try Launch.open(g, s, rows, in.q, in.kv, in.w, in.kvb, in.g);
    defer chunks.deinit();
    try chunks.dispatchMixer(g, 0, 2);
    try chunks.dispatchMixer(g, 2, 2);
    try expectBitsEqual("whole vs chunks", whole.y(), chunks.y());
}

fn checkResume() !void {
    const a = std.testing.allocator;
    const rows: u32 = 4;
    const s = mla.Shape{ .heads = 16, .eps = mla.eps_tiny, .theta = mla.theta_tiny };
    const in = try Inputs.make(a, s, rows, 0x72657375);
    defer in.free(a);
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    var g = try Gpu.init();
    defer g.deinit();
    var fresh = try Launch.open(g, s, rows, in.q, in.kv, in.w, in.kvb, in.g);
    defer fresh.deinit();
    try fresh.dispatchMixer(g, 0, rows);
    var continued = try Launch.open(g, s, rows, in.q, in.kv, in.w, in.kvb, in.g);
    defer continued.deinit();
    try continued.dispatchMixer(g, 0, 2);
    try continued.dispatchMixer(g, 2, 2);
    const W = s.outWidth();
    try expectBitsEqual("resume vs fresh (all rows)", fresh.y(), continued.y());
    try expectBitsEqual("resume vs fresh (last 2)", fresh.y()[2 * W ..], continued.y()[2 * W ..]);
}

fn checkRopeQ() !void {
    const a = std.testing.allocator;
    const s = mla.Shape{ .heads = 16, .eps = mla.eps_tiny, .theta = mla.theta_tiny };
    const rows: u32 = 4;
    const H = s.heads;
    var seed: u64 = 0x7271;
    const q = try fill(&seed, rows * H * mla.qd, a);
    defer a.free(q);
    roundSlice(q);
    const kv = try a.alloc(f32, rows * mla.key);
    defer a.free(kv);
    @memset(kv, 0);
    const w_rms: [mla.lat]f32 = @splat(1);
    const kvb_w = try a.alloc(f32, H * (mla.nope + mla.vd) * mla.lat);
    defer a.free(kvb_w);
    @memset(kvb_w, 0);
    const g = try a.alloc(f32, rows * H);
    defer a.free(g);
    @memset(g, 0);

    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    var gpu = try Gpu.init();
    defer gpu.deinit();
    var l = try Launch.open(gpu, s, rows, q, kv, &w_rms, kvb_w, g);
    defer l.deinit();
    l.fillRows(0, rows);
    l.fillCs(rows);
    const aa = l.args(rows);
    const cb = gpu.queue.commandBuffer();
    const e = cb.compute(.serial);
    e.setPipeline(gpu.rope_q);
    e.setBuffer(l.b_q, 0, 0);
    e.setValue(aa, 1);
    e.setBuffer(l.b_cs, 0, 2);
    e.dispatchGroups(mtl.Size.of(H, rows, 1), mtl.Size.of(32, 1, 1));
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |why| {
        std.debug.print("rope_q failed: {s}\n", .{why});
        return error.GpuFailed;
    }
    const got = l.b_q.slice(u16, q.len);
    var worst: f32 = 0;
    var bits_miss: usize = 0;
    for (0..rows) |t| {
        for (0..H) |h| {
            var z64: [mla.rope_dim]f64 = undefined;
            var out: [mla.rope_dim]f64 = undefined;
            const base = (t * H + h) * mla.qd + mla.nope;
            for (0..mla.rope_dim) |i| z64[i] = q[base + i];
            mla.rotate(&out, &z64, @intCast(t), s.theta);
            for (0..mla.rope_dim) |i| {
                const want = toBf16(@floatCast(out[i]));
                const gbits = got[base + i];
                if (want != gbits) bits_miss += 1;
                worst = @max(worst, @abs(fromBf16(want) - fromBf16(gbits)));
            }
        }
    }
    std.debug.print("rope_q vs host rotate: bits miss {d} worst abs {e}\n", .{ bits_miss, worst });
    if (worst > 2e-3) return error.RopeQ;
}

fn checkRopeCache() !void {
    const a = std.testing.allocator;
    const s = mla.Shape{ .heads = 16, .eps = mla.eps_tiny, .theta = mla.theta_tiny };
    const rows: u32 = 4;
    var seed: u64 = 0x7263;
    const rope_in = try fill(&seed, rows * mla.rope_dim, a);
    defer a.free(rope_in);
    roundSlice(rope_in);
    const H = s.heads;
    const q = try a.alloc(f32, rows * H * mla.qd);
    defer a.free(q);
    @memset(q, 0);
    const kv = try a.alloc(f32, rows * mla.key);
    defer a.free(kv);
    @memset(kv, 0);
    const w_rms: [mla.lat]f32 = @splat(1);
    const kvb_w = try a.alloc(f32, H * (mla.nope + mla.vd) * mla.lat);
    defer a.free(kvb_w);
    @memset(kvb_w, 0);
    const g = try a.alloc(f32, rows * H);
    defer a.free(g);
    @memset(g, 0);

    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    var gpu = try Gpu.init();
    defer gpu.deinit();
    var l = try Launch.open(gpu, s, rows, q, kv, &w_rms, kvb_w, g);
    defer l.deinit();
    const cache = l.b_cache.slice(u16, rows * mla.key);
    @memset(cache, 0);
    for (0..rows) |t| {
        for (0..mla.rope_dim) |i| cache[t * mla.key + mla.lat + i] = toBf16(rope_in[t * mla.rope_dim + i]);
    }
    l.fillRows(0, rows);
    l.fillCs(rows);
    const aa = l.args(rows);
    const cb = gpu.queue.commandBuffer();
    const e = cb.compute(.serial);
    e.setPipeline(gpu.rope_cache);
    e.setBuffer(l.b_rows, 0, 0);
    e.setValue(aa, 1);
    e.setBuffer(l.b_cs, 0, 2);
    e.useResource(l.b_cache, mtl.ResourceUsage.read | mtl.ResourceUsage.write);
    e.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(32, 1, 1));
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |why| {
        std.debug.print("rope_cache failed: {s}\n", .{why});
        return error.GpuFailed;
    }
    var worst: f32 = 0;
    var bits_miss: usize = 0;
    for (0..rows) |t| {
        var z64: [mla.rope_dim]f64 = undefined;
        var out: [mla.rope_dim]f64 = undefined;
        for (0..mla.rope_dim) |i| z64[i] = rope_in[t * mla.rope_dim + i];
        mla.rotate(&out, &z64, @intCast(t), s.theta);
        for (0..mla.rope_dim) |i| {
            const want = toBf16(@floatCast(out[i]));
            const gbits = cache[t * mla.key + mla.lat + i];
            if (want != gbits) bits_miss += 1;
            worst = @max(worst, @abs(fromBf16(want) - fromBf16(gbits)));
        }
    }
    std.debug.print("rope_cache vs host rotate: bits miss {d} worst abs {e}\n", .{ bits_miss, worst });
    if (worst > 2e-3) return error.RopeCache;
}

fn checkExpand() !void {
    const a = std.testing.allocator;
    const s = mla.Shape{ .heads = 16, .eps = mla.eps_tiny, .theta = mla.theta_tiny };
    const rows: u32 = 4;
    const H = s.heads;
    var seed: u64 = 0x6765;
    const gate = try fill(&seed, rows * H, a);
    defer a.free(gate);
    roundSlice(gate);
    const q = try a.alloc(f32, rows * H * mla.qd);
    defer a.free(q);
    @memset(q, 0);
    const kv = try a.alloc(f32, rows * mla.key);
    defer a.free(kv);
    @memset(kv, 0);
    const w_rms: [mla.lat]f32 = @splat(1);
    const kvb_w = try a.alloc(f32, H * (mla.nope + mla.vd) * mla.lat);
    defer a.free(kvb_w);
    @memset(kvb_w, 0);

    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    var gpu = try Gpu.init();
    defer gpu.deinit();
    var l = try Launch.open(gpu, s, rows, q, kv, &w_rms, kvb_w, gate);
    defer l.deinit();
    const aa = l.args(rows);
    const cb = gpu.queue.commandBuffer();
    const e = cb.compute(.serial);
    e.setPipeline(gpu.gate_expand);
    e.setBuffer(l.b_g, 0, 0);
    e.setBuffer(l.b_g2, 0, 1);
    e.setValue(aa, 2);
    e.dispatchThreads(mtl.Size.of(H * mla.vd, rows, 1), mtl.Size.of(128, 1, 1));
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |why| {
        std.debug.print("gate_expand failed: {s}\n", .{why});
        return error.GpuFailed;
    }
    const g_bits = l.b_g.slice(u16, rows * H);
    const want = try a.alloc(u16, rows * H * mla.vd);
    defer a.free(want);
    mla.expandGate(want, g_bits, rows, H);
    try expectBitsEqual("gate_expand", l.b_g2.slice(u16, want.len), want);
}

test "Ling rope_q GPT-J halves vs host rotate" {
    try checkRopeQ();
}

test "Ling rope_cache GPT-J halves vs host rotate" {
    try checkRopeCache();
}

test "Ling gate_expand (T,H) to (T,H*128) bits-equal" {
    try checkExpand();
}

test "MLA mixer Metal vs f64 and bf16-point on 2 rows from pos 0" {
    try checkMixer(2);
}

test "MLA mixer Metal vs f64 and bf16-point on 4 rows from pos 0" {
    try checkMixer(4);
}

test "MLA mixer 4 rows equals 2+2 bits-equal" {
    try checkChunks();
}

test "MLA mixer resume vs fresh bits-equal" {
    try checkResume();
}

fn fileOk(path: []const u8) bool {
    std.Io.Dir.cwd().access(std.testing.io, path, .{}) catch return false;
    return true;
}

fn hasTorch(py: []const u8) bool {
    const a = std.testing.allocator;
    const r = std.process.run(a, std.testing.io, .{
        .argv = &.{ py, "-c", "import torch" },
        .stdout_limit = .limited(256),
        .stderr_limit = .limited(4096),
    }) catch return false;
    defer a.free(r.stdout);
    defer a.free(r.stderr);
    return r.term.success();
}

fn findPython() ?[]const u8 {
    if (std.c.getenv("TF_LING_PYTHON")) |p| {
        const s = std.mem.span(p);
        if (hasTorch(s)) return s;
    }
    if (hasTorch("python3")) return "python3";
    return null;
}

fn ensureDumps() !void {
    if (fileOk(dump_mixer_path) and fileOk(dump_a1_t4_path) and fileOk(dump_a1_t2_path)) return;
    if (!fileOk(dump_py)) {
        std.debug.print("skip: {s} not in cwd; TFLINGA1 dumps need ling_forward.py at test time\n", .{dump_py});
        return error.SkipZigTest;
    }
    const py = findPython() orelse {
        std.debug.print("skip: no torch (set TF_LING_PYTHON or python3 on PATH); TFLINGA1 dumps come from ling_forward.py at test time\n", .{});
        return error.SkipZigTest;
    };
    const a = std.testing.allocator;
    const r = try std.process.run(a, std.testing.io, .{
        .argv = &.{ py, dump_py, dump_dir },
        .stdout_limit = .limited(4096),
        .stderr_limit = .limited(65536),
    });
    defer a.free(r.stdout);
    defer a.free(r.stderr);
    if (!r.term.success()) {
        std.debug.print("mla_dump.py failed: {s}{s}\n", .{ r.stdout, r.stderr });
        return error.DumpGen;
    }
    if (!fileOk(dump_mixer_path) or !fileOk(dump_a1_t4_path) or !fileOk(dump_a1_t2_path)) return error.DumpGen;
}

fn readDump(path: []const u8) ![]u8 {
    return std.Io.Dir.cwd().readFileAlloc(std.testing.io, path, std.testing.allocator, .limited(32 << 20));
}

fn checkDumpFile(a1_path: []const u8, case_i: usize, tag: []const u8, gpu: bool) !void {
    try ensureDumps();
    const a = std.testing.allocator;
    const mixer_bytes = try readDump(dump_mixer_path);
    defer a.free(mixer_bytes);
    const a1_bytes = try readDump(a1_path);
    defer a.free(a1_bytes);
    try checkDump(mixer_bytes, a1_bytes, case_i, tag, gpu);
}

fn checkDump(mixer_bytes: []const u8, a1_bytes: []const u8, case_i: usize, tag: []const u8, gpu: bool) !void {
    const a = std.testing.allocator;
    const dump = try Dump.load(a, mixer_bytes);
    defer dump.deinit(a);
    const a1 = try parseA1(a, a1_bytes);
    defer a.free(a1.y64);
    defer a.free(a1.ybf);
    const c = dump.cases[case_i];
    try std.testing.expectEqual(a1.T, c.T);
    try std.testing.expectEqual(a1.hidden, dump.hidden);
    try std.testing.expectEqual(a1.layer, dump.layer);
    try std.testing.expectEqual(a1.pos0, c.pos0);
    try std.testing.expect(std.mem.eql(f64, a1.y64, c.y64));

    const s = mla.Shape{ .heads = dump.heads, .eps = mla.eps_tiny, .theta = mla.theta_tiny };
    const width = s.outWidth();
    const n_mix = c.T * width;
    const mix64 = try a.alloc(f64, n_mix);
    defer a.free(mix64);
    try mla.mixF64(a, mix64, c.q, c.kv, dump.w, dump.kvb, c.g, s, c.T, c.pos0);
    const d_pre = dist(mix64, c.y_pre);
    std.debug.print("{s} mixF64-to-python mixer rel L2 {e} max abs {e}\n", .{ tag, d_pre.rel_l2, d_pre.max_abs });
    const layer_mix = try a.alloc(f64, c.T * dump.hidden);
    defer a.free(layer_mix);
    oProj(layer_mix, mix64, dump.Wo, c.T, dump.hidden, width);
    const d_layer = dist(layer_mix, a1.y64);
    std.debug.print("{s} mixF64+o_proj-to-TFLINGA1 rel L2 {e} max abs {e}\n", .{ tag, d_layer.rel_l2, d_layer.max_abs });
    if (!gpu) return;

    const mix_bf = try a.alloc(f64, n_mix);
    defer a.free(mix_bf);
    const q_m = try dupeRound(a, c.q);
    defer a.free(q_m);
    const kv_m = try dupeRound(a, c.kv);
    defer a.free(kv_m);
    const w_m = try dupeRound(a, dump.w);
    defer a.free(w_m);
    const kvb_m = try dupeRound(a, dump.kvb);
    defer a.free(kvb_m);
    const g_m = try dupeRound(a, c.g);
    defer a.free(g_m);
    try mla.mixBf16(a, mix_bf, q_m, kv_m, w_m, kvb_m, g_m, s, c.T, c.pos0);

    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    var gpu_dev = try Gpu.init();
    defer gpu_dev.deinit();
    var l = try Launch.open(gpu_dev, s, c.T, q_m, kv_m, w_m, kvb_m, g_m);
    defer l.deinit();
    try l.dispatchMixer(gpu_dev, 0, c.T);
    const metal_mix = try a.alloc(f64, n_mix);
    defer a.free(metal_mix);
    bitsToF64(l.y(), metal_mix);
    const d_metal_pre = dist(metal_mix, c.y_pre);
    std.debug.print("{s} Metal-to-python mixer rel L2 {e} max abs {e}\n", .{ tag, d_metal_pre.rel_l2, d_metal_pre.max_abs });

    const layer_bf = try a.alloc(f64, c.T * dump.hidden);
    defer a.free(layer_bf);
    const layer_metal = try a.alloc(f64, c.T * dump.hidden);
    defer a.free(layer_metal);
    oProj(layer_bf, mix_bf, dump.Wo, c.T, dump.hidden, width);
    oProj(layer_metal, metal_mix, dump.Wo, c.T, dump.hidden, width);
    var tbuf: [96]u8 = undefined;
    const layer_tag = try std.fmt.bufPrint(&tbuf, "{s} TFLINGA1", .{tag});
    try gateNamedF64(layer_tag, layer_metal, a1.y64, layer_bf);
}

fn checkRopeLargePos() !void {
    const a = std.testing.allocator;
    const s = mla.Shape{ .heads = 16, .eps = mla.eps_tiny, .theta = mla.theta_tiny };
    const rows_n: u32 = 2;
    const first_pos: u32 = 29998;
    const slot_keys: u32 = 30000;
    const H = s.heads;
    var seed: u64 = 0x72333030;
    const q = try fill(&seed, rows_n * H * mla.qd, a);
    defer a.free(q);
    roundSlice(q);
    const rope_in = try fill(&seed, rows_n * mla.rope_dim, a);
    defer a.free(rope_in);
    roundSlice(rope_in);

    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    var gpu = try Gpu.init();
    defer gpu.deinit();
    const shared = mtl.ResourceOptions.shared;
    const b_q = try gpu.device.buffer(q.len * 2, shared);
    defer b_q.deinit();
    putBf16(b_q, q);
    const b_rows = try gpu.device.buffer(rows_n * @sizeOf(MlaRow), shared);
    defer b_rows.deinit();
    const rows = b_rows.slice(MlaRow, rows_n);
    for (rows, 0..) |*r, i| r.* = .{ .slot = 0, .pos = first_pos + @as(u32, @intCast(i)) };
    const b_cs = try gpu.device.buffer(rows_n * mla.nfreq * 2 * 4, shared);
    defer b_cs.deinit();
    writeCs(b_cs.slice(f32, rows_n * mla.nfreq * 2), rows, s.theta);
    const b_cache = try gpu.device.buffer(@as(usize, slot_keys) * mla.key * 2, shared);
    defer b_cache.deinit();
    const cache = b_cache.slice(u16, slot_keys * mla.key);
    for (0..rows_n) |t| {
        const base = (first_pos + t) * mla.key + mla.lat;
        for (0..mla.rope_dim) |i| cache[base + i] = toBf16(rope_in[t * mla.rope_dim + i]);
    }
    const aa = MlaArgs{
        .cache = b_cache.gpuAddress(),
        .slot_keys = slot_keys,
        .heads = H,
        .head0 = 0,
        .rows = rows_n,
        .q_stride = H * mla.qd,
        .kv_stride = mla.key,
        .gate_stride = H * mla.vd,
        .out_stride = H * mla.vd,
        .eps = s.eps,
        .scale = @floatCast(s.scale()),
    };

    {
        const cb = gpu.queue.commandBuffer();
        const e = cb.compute(.serial);
        e.setPipeline(gpu.rope_q);
        e.setBuffer(b_q, 0, 0);
        e.setValue(aa, 1);
        e.setBuffer(b_cs, 0, 2);
        e.dispatchGroups(mtl.Size.of(H, rows_n, 1), mtl.Size.of(32, 1, 1));
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |why| {
            std.debug.print("rope_q large pos failed: {s}\n", .{why});
            return error.GpuFailed;
        }
    }
    const got_q = b_q.slice(u16, q.len);
    const nq = rows_n * H * mla.rope_dim;
    const metal_q = try a.alloc(f64, nq);
    defer a.free(metal_q);
    const f64_q = try a.alloc(f64, nq);
    defer a.free(f64_q);
    const bf_q = try a.alloc(f64, nq);
    defer a.free(bf_q);
    var k: usize = 0;
    for (0..rows_n) |t| {
        for (0..H) |h| {
            var z64: [mla.rope_dim]f64 = undefined;
            var out: [mla.rope_dim]f64 = undefined;
            const base = (t * H + h) * mla.qd + mla.nope;
            for (0..mla.rope_dim) |i| z64[i] = q[base + i];
            mla.rotate(&out, &z64, first_pos + @as(u32, @intCast(t)), s.theta);
            for (0..mla.rope_dim) |i| {
                f64_q[k] = out[i];
                bf_q[k] = fromBf16(toBf16(@floatCast(out[i])));
                metal_q[k] = fromBf16(got_q[base + i]);
                k += 1;
            }
        }
    }
    const q_gate = gateNamedF64("rope_q pos 29998-29999", metal_q, f64_q, bf_q);

    {
        const cb = gpu.queue.commandBuffer();
        const e = cb.compute(.serial);
        e.setPipeline(gpu.rope_cache);
        e.setBuffer(b_rows, 0, 0);
        e.setValue(aa, 1);
        e.setBuffer(b_cs, 0, 2);
        e.useResource(b_cache, mtl.ResourceUsage.read | mtl.ResourceUsage.write);
        e.dispatchGroups(mtl.Size.of(rows_n, 1, 1), mtl.Size.of(32, 1, 1));
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |why| {
            std.debug.print("rope_cache large pos failed: {s}\n", .{why});
            return error.GpuFailed;
        }
    }
    const nc = rows_n * mla.rope_dim;
    const metal_c = try a.alloc(f64, nc);
    defer a.free(metal_c);
    const f64_c = try a.alloc(f64, nc);
    defer a.free(f64_c);
    const bf_c = try a.alloc(f64, nc);
    defer a.free(bf_c);
    k = 0;
    for (0..rows_n) |t| {
        var z64: [mla.rope_dim]f64 = undefined;
        var out: [mla.rope_dim]f64 = undefined;
        for (0..mla.rope_dim) |i| z64[i] = rope_in[t * mla.rope_dim + i];
        mla.rotate(&out, &z64, first_pos + @as(u32, @intCast(t)), s.theta);
        const base = (first_pos + t) * mla.key + mla.lat;
        for (0..mla.rope_dim) |i| {
            f64_c[k] = out[i];
            bf_c[k] = fromBf16(toBf16(@floatCast(out[i])));
            metal_c[k] = fromBf16(cache[base + i]);
            k += 1;
        }
    }
    const c_gate = gateNamedF64("rope_cache pos 29998-29999", metal_c, f64_c, bf_c);
    q_gate catch |e| return e;
    c_gate catch |e| return e;
}

fn checkLarge(rows: u32, left: u32, seed: u64, tag: []const u8) !void {
    const a = std.testing.allocator;
    const s = mla.Shape{ .heads = 16, .eps = mla.eps_tiny, .theta = mla.theta_tiny };
    const in = try Inputs.make(a, s, rows, seed);
    defer in.free(a);
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    var g = try Gpu.init();
    defer g.deinit();
    var whole = try Launch.open(g, s, rows, in.q, in.kv, in.w, in.kvb, in.g);
    defer whole.deinit();
    try whole.dispatchMixer(g, 0, rows);
    var parts = try Launch.open(g, s, rows, in.q, in.kv, in.w, in.kvb, in.g);
    defer parts.deinit();
    try parts.dispatchMixer(g, 0, left);
    try parts.dispatchMixer(g, left, rows - left);
    var buf: [96]u8 = undefined;
    const t1 = try std.fmt.bufPrint(&buf, "{s} whole vs chunks", .{tag});
    try expectBitsEqual(t1, whole.y(), parts.y());
    var buf2: [96]u8 = undefined;
    const t2 = try std.fmt.bufPrint(&buf2, "{s} resume vs fresh", .{tag});
    try expectBitsEqual(t2, whole.y(), parts.y());
}

test "TFLINGA1 tiny layer 3 T=4 pos0=0 Metal and mixF64 vs python f64" {
    try checkDumpFile(dump_a1_t4_path, 0, "dump T=4 pos0=0", true);
}

test "TFLINGA1 tiny layer 3 T=2 pos0=30000 mixF64 vs python host-only" {
    try checkDumpFile(dump_a1_t2_path, 1, "dump T=2 pos0=30000", false);
}

test "Ling rope_q and rope_cache at pos 29998-29999 vs f64" {
    try checkRopeLargePos();
}

test "MLA mixer T=100 as 37+63 bits-equal" {
    try checkLarge(100, 37, 0x74313030, "T=100");
}

test "MLA mixer T=257 as 256+1 bits-equal (k3 gpu_rows)" {
    try checkLarge(257, k3_gpu_rows, 0x74323537, "T=257");
}
