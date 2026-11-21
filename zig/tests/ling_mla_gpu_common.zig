//! Shared Gpu/Launch/helpers for Ling MLA GPU tests around k3_mla.
const std = @import("std");
const mtl = @import("metal");
const mla = @import("mla");
const metallib = @import("kimi_metallib");

const source = @import("ling_metal").mla;
const mla_splits: u32 = 16;

pub const MlaRow = extern struct { slot: u32, pos: u32 };
pub const MlaArgs = extern struct {
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

pub fn toBf16(v: f32) u16 {
    const bits: u32 = @bitCast(v);
    if (std.math.isNan(v)) return @truncate((bits >> 16) | 0x40);
    return @truncate((bits +% 0x7fff +% ((bits >> 16) & 1)) >> 16);
}
pub fn fromBf16(h: u16) f32 {
    return @bitCast(@as(u32, h) << 16);
}

pub fn fill(seed: *u64, n: usize, a: std.mem.Allocator) ![]f32 {
    const x = try a.alloc(f32, n);
    for (x) |*v| {
        seed.* = seed.* *% 1103515245 +% 12345;
        v.* = @as(f32, @floatFromInt((seed.* >> 16) & 0x7fff)) / 16384.0 - 1.0;
    }
    return x;
}

pub fn putBf16(buf: mtl.Buffer, x: []const f32) void {
    const d = buf.slice(u16, x.len);
    for (x, d) |v, *o| o.* = toBf16(v);
}

pub fn roundSlice(x: []f32) void {
    for (x) |*v| v.* = fromBf16(toBf16(v.*));
}

/// Host f64 cos/sin of each row's pos, rounded to f32. Layout [t][i][comp] = t*64 + i*2 + comp.
pub fn writeCs(cs: []f32, rows: []const MlaRow, theta: f64) void {
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

pub const Gpu = struct {
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

    pub fn init() !Gpu {
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

    pub fn deinit(g: *Gpu) void {
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

pub const Launch = struct {
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

    pub fn open(g: Gpu, s: mla.Shape, cap: u32, q: []const f32, kv: []const f32, w_rms: []const f32, kvb: []const f32, gate: []const f32) !Launch {
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

    pub fn deinit(l: *Launch) void {
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

    pub fn args(l: *Launch, n: u32) MlaArgs {
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

    pub fn fillRows(l: *Launch, first: u32, n: u32) void {
        const rows = l.b_rows.slice(MlaRow, n);
        for (rows, 0..) |*r, i| r.* = .{ .slot = 0, .pos = first + @as(u32, @intCast(i)) };
    }

    pub fn fillCs(l: *Launch, n: u32) void {
        writeCs(l.b_cs.slice(f32, n * mla.nfreq * 2), l.b_rows.slice(MlaRow, n), l.s.theta);
    }

    pub fn dispatchMixer(l: *Launch, g: Gpu, first: u32, n: u32) !void {
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

    pub fn y(l: *Launch) []u16 {
        return l.b_y.slice(u16, l.cap * l.s.outWidth());
    }
};

pub const Dist = struct { rel_l2: f64, max_abs: f64 };

pub fn dist(got: []const f64, ref: []const f64) Dist {
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

pub fn bitsToF64(bits: []const u16, out: []f64) void {
    for (bits, out) |h, *o| o.* = fromBf16(h);
}

pub fn gateNamed(tag: []const u8, metal: []const u16, f64_y: []const f64, bf16_y: []const f64) !void {
    const a = std.testing.allocator;
    const metal_f = try a.alloc(f64, metal.len);
    defer a.free(metal_f);
    bitsToF64(metal, metal_f);
    try gateNamedF64(tag, metal_f, f64_y, bf16_y);
}

pub fn gateNamedF64(tag: []const u8, metal: []const f64, f64_y: []const f64, bf16_y: []const f64) !void {
    const m = dist(metal, f64_y);
    const b = dist(bf16_y, f64_y);
    std.debug.print("{s} Metal-to-f64 rel L2 {e} max abs {e}; bf16-ref-to-f64 rel L2 {e} max abs {e}\n", .{ tag, m.rel_l2, m.max_abs, b.rel_l2, b.max_abs });
    if (m.rel_l2 > b.rel_l2 or m.max_abs > b.max_abs) {
        std.debug.print("{s} gate miss: Metal further from f64 than bf16-ref\n", .{tag});
        return error.Gate;
    }
}

pub fn oProj(out: []f64, y: []const f64, wo: []const f32, rows: u32, hidden: u32, width: u32) void {
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

pub fn expectBitsEqual(tag: []const u8, a: []const u16, b: []const u16) !void {
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

pub const Inputs = struct {
    q: []f32,
    kv: []f32,
    w: []f32,
    kvb: []f32,
    g: []f32,

    pub fn make(alloc: std.mem.Allocator, s: mla.Shape, rows: u32, seed0: u64) !Inputs {
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

    pub fn free(in: Inputs, alloc: std.mem.Allocator) void {
        alloc.free(in.q);
        alloc.free(in.kv);
        alloc.free(in.w);
        alloc.free(in.kvb);
        alloc.free(in.g);
    }
};
