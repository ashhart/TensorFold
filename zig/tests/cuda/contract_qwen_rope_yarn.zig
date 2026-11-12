//! YaRN rope (a 33-float table) in tf_q35_rope, tf_q35_rope8 and tf_q36m_norm_rope against float64, to position 1M.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const tc = @import("check.zig");
const rig_ = @import("contract_rig.zig");

const c = core.contract.spec;
const e4m3 = core.quant.fp8.e4m3;

const hd = 256;
const q_heads = 16; // Qwen3.6-35B-A3B's heads
const kv_heads = 2;
const ptok = 256; // positions a page: each row's stream maps every page of its window to one page of its own
const cap = 1 << 20; // the window: positions up to 1,048,575
const texts = [_]i32{ 0, 262143, 262144, 1048575 };
/// Image rows' t, h and w positions (M-RoPE's axes apart).
const images = [_][3]i32{ .{ 1048000, 5, 7 }, .{ 300000, 300001, 2 }, .{ 262144, 0, 1048575 } };

// qwen35.cuh's Geo, Seg and Ancestor (contract_qwen_pages.zig mirrors them too)
const Geo = extern struct { hid: c_int, nk: c_int, nv: c_int, qh: c_int, kvh: c_int, nl: c_int };
const Seg = extern struct { rec: u64 = 0, conv: u64 = 0, kv: u64, tap: u64 = 0, log: u64 = 0, before: u64 = 0, cap: c_int, len: c_int, row0: c_int, rows: c_int, tap_rows: c_int = 0, taps: c_int = 0, delta: c_int = 0, pt: u64 = 0, ps: u64 = 0, ptok: c_int = 0, pn: c_int = 0, plog: u64 = 0 };
const Ancestor = extern struct { bits: u64, depth: u32, chain: u32 };

/// The engine's table for Qwen3.6 (theta 1e7, factor 4 over 262,144): the attention factor, then 32 frequencies.
fn table() [33]f32 {
    const y = core.rope_yarn.yarn(10000000, 4, 262144);
    var t: [33]f32 = undefined;
    t[0] = y.scale;
    @memcpy(t[1..], &y.freq);
    return t;
}

/// qwen35.cuh's mrope_axis: frequency i turns by the t (0), h (1) or w (2) position.
fn axis(i: usize) usize {
    return if (i % 3 == 1 and i < 33) 1 else if (i % 3 == 2 and i < 30) 2 else 0;
}

const Want = struct { want: f64, bound: f64 };

/// Element d < 64 of a rotated head row (a, b its pair's inputs) at the kernel's own fp32 angle, then float64.
fn yarnRope(a: f64, b: f64, d: usize, pos: f32, t: [33]f32) Want {
    const angle: f32 = pos * t[1 + d % 32];
    const f: f64 = t[0];
    const cs = @cos(@as(f64, angle)) * f;
    const sn = @sin(@as(f64, angle)) * f;
    const want = if (d < 32) a * cs - b * sn else b * cs + a * sn;
    // the output's bf16 rounding; the products, the factor and sincosf's 2 ulp, of the pair's magnitude
    const bound = c.unit(.bf16) * @abs(want) + (c.gamma(4, c.unit(.f32)) + 4 * c.unit(.f32)) * (@abs(a) + @abs(b)) * f;
    return .{ .want = want, .bound = bound + 0x1p-133 };
}

fn random(a: std.mem.Allocator, rng: std.Random, n: usize, scale: f32) ![]u16 {
    const out = try a.alloc(u16, n);
    for (out) |*v| v.* = c.toBf16((rng.float(f32) - 0.5) * scale);
    return out;
}

/// tf_q35_rope (bf16 caches) or tf_q35_rope8 (e4m3): text rows at `texts`, or image rows at `images`.
fn target(gpu: tc.Gpu, s: cuda.Stream, f: cuda.Function, fp8: bool, image: bool) !void {
    var arena = std.heap.ArenaAllocator.init(gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var rig = rig_.Rig{ .gpu = gpu, .s = s };
    defer rig.deinit();
    var prng = std.Random.DefaultPrng.init(if (image) 0x1a6e else 0x7e47);
    const rng = prng.random();
    const n = if (image) images.len else texts.len;
    const q = try random(a, rng, n * q_heads * hd, 4);
    const k = try random(a, rng, n * kv_heads * hd, 4);
    const v = try random(a, rng, n * kv_heads * hd, 4);
    const page_bytes: usize = 2 * kv_heads * ptok * @as(usize, if (fp8) hd + 4 else hd * 2);
    const kv = try rig.output(n * page_bytes);
    var segs: [images.len + texts.len]Seg = undefined;
    const rowseg = try a.alloc(i32, n);
    const rowanc = try a.alloc(Ancestor, n);
    const m = try a.alloc(i32, n * 3);
    const slots = try a.alloc(usize, n);
    for (0..n) |r| {
        // a text row's position is its stream's length; an image row's come from M, its slot from the length
        const len: usize = if (image) 1000 + 37 * r else @intCast(texts[r]);
        const pt = try a.alloc(i32, cap / ptok);
        @memset(pt, @intCast(r));
        segs[r] = .{ .kv = kv, .cap = cap, .len = @intCast(len), .row0 = @intCast(r), .rows = 1, .pt = try rig.input(std.mem.sliceAsBytes(pt)), .ps = if (fp8) page_bytes else page_bytes / 2, .ptok = ptok };
        rowseg[r] = @intCast(r);
        rowanc[r] = .{ .bits = 0, .depth = 0, .chain = 1 };
        for (0..3) |x| m[r * 3 + x] = if (image) images[r][x] else texts[r];
        slots[r] = len;
    }
    const t = table();
    const geo: Geo = .{ .hid = 0, .nk = 0, .nv = 0, .qh = q_heads, .kvh = kv_heads, .nl = 0 };
    const segs_d = try rig.input(std.mem.sliceAsBytes(segs[0..n]));
    const rowseg_d = try rig.input(std.mem.sliceAsBytes(rowseg));
    const rowanc_d = try rig.input(std.mem.sliceAsBytes(rowanc));
    const kn = try rig.input(std.mem.sliceAsBytes(k));
    const vr = try rig.input(std.mem.sliceAsBytes(v));
    const m_d: u64 = if (image) try rig.input(std.mem.sliceAsBytes(m)) else 0;
    const y = try rig.input(std.mem.sliceAsBytes(&t));
    const qn = try rig.output(q.len * 2);
    const before = try a.dupe(u8, try rig.inputBytes());
    var first: ?[]const u8 = null;
    for (0..5) |_| {
        try (cuda.DeviceBuffer{ .d = gpu.d, .ptr = qn, .len = q.len * 2 }).upload(0, std.mem.sliceAsBytes(q)); // rotated in place
        var args: cuda.Args = .{};
        inline for (.{ geo, qn, kn, vr, segs_d, rowseg_d, rowanc_d, @as(c_int, 0), @as(u64, 0), m_d, y }) |x| args.add(x);
        try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(n), .y = q_heads + kv_heads }, .block = .{ .x = hd } }, s, &args);
        try s.synchronize();
        const out = try rig.outputBytes();
        if (first) |b| try tc.expect(std.mem.eql(u8, b, out), "YaRN rope: nondeterministic bytes", .{}) else first = try a.dupe(u8, out);
    }
    try tc.expect(try rig.intact() and !rig.overlap() and std.mem.eql(u8, before, try rig.inputBytes()), "YaRN rope changed an input or guard", .{});
    const qo = std.mem.bytesAsSlice(u16, try a.dupe(u8, try rig.bytes(qn, q.len * 2)));
    const cache = try a.dupe(u8, try rig.bytes(kv, n * page_bytes));
    var worst: f64 = 0;
    for (0..n) |r| {
        const pos: [3]i32 = .{ m[r * 3], m[r * 3 + 1], m[r * 3 + 2] };
        for (0..q_heads + kv_heads) |h| {
            const key = h >= q_heads;
            const kh = h -| q_heads;
            const in = if (key) k[(r * kv_heads + kh) * hd ..][0..hd] else q[(r * q_heads + h) * hd ..][0..hd];
            var wants: [hd]Want = undefined;
            var amax: f64 = 0; // the row's largest reference magnitude: an e4m3 row's scale is the kernel's amax / 448
            for (&wants, 0..) |*w, d| {
                w.* = if (d < 64) yarnRope(c.fromBf16(in[d % 32]), c.fromBf16(in[d % 32 + 32]), d, @floatFromInt(pos[axis(d % 32)]), t) else .{ .want = c.fromBf16(in[d]), .bound = 0 };
                amax = @max(amax, @abs(w.want));
            }
            const p = slots[r] % ptok;
            if (key and fp8) {
                const scale: f64 = @as(f32, @bitCast(std.mem.readInt(u32, cache[r * page_bytes + 2 * kv_heads * ptok * hd + (kh * ptok + p) * 4 ..][0..4], .little)));
                try tc.expect(@abs(scale - amax / 448) <= 0x1p-6 * amax / 448, "YaRN rope fp8: key row {d} head {d} scale {d} against the row's {d} / 448", .{ r, kh, scale, amax });
            }
            for (wants, 0..) |w, d| {
                var got: f64 = undefined;
                var bound = w.bound;
                if (!key) {
                    got = c.fromBf16(qo[(r * q_heads + h) * hd + d]);
                } else if (!fp8) {
                    got = c.fromBf16(std.mem.readInt(u16, cache[r * page_bytes + ((kh * ptok + p) * hd + d) * 2 ..][0..2], .little));
                } else {
                    const page = cache[r * page_bytes ..][0..page_bytes];
                    const scale: f64 = @as(f32, @bitCast(std.mem.readInt(u32, page[2 * kv_heads * ptok * hd + (kh * ptok + p) * 4 ..][0..4], .little)));
                    got = e4m3(page[(kh * ptok + p) * hd + d]) * scale;
                    // e4m3's half unit after scaling, of each value; and the scale's own fp32 rounding
                    bound += @max(0x1p-4 * @abs(w.want), 0x1p-10 * scale) + 4 * c.unit(.f32) * @abs(w.want);
                }
                try tc.expect(std.math.isFinite(got) and @abs(got - w.want) <= bound, "YaRN rope {s}{s}: row {d} head {d} dim {d}: {d} vs {d} (bound {e})", .{ if (fp8) "fp8" else "bf16", if (image) " image" else "", r, h, d, got, w.want, bound });
                if (bound > 0) worst = @max(worst, @abs(got - w.want) / bound);
            }
            if (!key) continue;
            // the values: copied (bf16), or quantized within e4m3's half unit (fp8) under a scale of their amax / 448
            var vmax: f64 = 0;
            for (v[(r * kv_heads + kh) * hd ..][0..hd]) |x| vmax = @max(vmax, @abs(c.fromBf16(x)));
            if (fp8) {
                const vs: f64 = @as(f32, @bitCast(std.mem.readInt(u32, cache[r * page_bytes + 2 * kv_heads * ptok * hd + (kv_heads * ptok + kh * ptok + p) * 4 ..][0..4], .little)));
                try tc.expect(@abs(vs - vmax / 448) <= 0x1p-20 * vmax / 448, "YaRN rope fp8: value row {d} head {d} scale {d} against the row's {d} / 448", .{ r, kh, vs, vmax });
            }
            for (0..hd) |d| {
                const want = c.fromBf16(v[(r * kv_heads + kh) * hd + d]);
                if (!fp8) {
                    const got = std.mem.readInt(u16, cache[r * page_bytes + (((kv_heads + kh) * ptok + p) * hd + d) * 2 ..][0..2], .little);
                    try tc.expect(got == v[(r * kv_heads + kh) * hd + d], "YaRN rope bf16: value row {d} head {d} dim {d} not copied", .{ r, kh, d });
                } else {
                    const page = cache[r * page_bytes ..][0..page_bytes];
                    const scale: f64 = @as(f32, @bitCast(std.mem.readInt(u32, page[2 * kv_heads * ptok * hd + (kv_heads * ptok + kh * ptok + p) * 4 ..][0..4], .little)));
                    const got = e4m3(page[kv_heads * ptok * hd + (kh * ptok + p) * hd + d]) * scale;
                    try tc.expect(@abs(got - want) <= @max(0x1p-4 * @abs(want), 0x1p-10 * scale) + 4 * c.unit(.f32) * @abs(want), "YaRN rope fp8: value row {d} head {d} dim {d}: {d} vs {d}", .{ r, kh, d, got, want });
                }
            }
        }
    }
    std.debug.print("PASS {s} YaRN{s}: {d} rows to position 1,048,575 against float64 (worst {d:.3} of the bound), five repeats, guards\n", .{ if (fp8) "tf_q35_rope8" else "tf_q35_rope", if (image) " (image rows, M-RoPE axes)" else "", n, worst });
}

/// tf_q36m_norm_rope with the table: q and k heads normed, rounded to bf16, then roped at pos0 + row * step.
fn mtp(gpu: tc.Gpu, s: cuda.Stream, f: cuda.Function, query: bool, pos0: usize, step: usize) !void {
    var arena = std.heap.ArenaAllocator.init(gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var rig = rig_.Rig{ .gpu = gpu, .s = s };
    defer rig.deinit();
    var prng = std.Random.DefaultPrng.init(0x3a7 + pos0 + step);
    const rng = prng.random();
    const rows = 3;
    const heads: usize = if (query) q_heads else kv_heads;
    const in_head: usize = if (query) 2 * hd else hd;
    const stride = heads * in_head;
    const x = try random(a, rng, rows * stride, 2);
    const w = try random(a, rng, hd, 2);
    const t = table();
    const xp = try rig.input(std.mem.sliceAsBytes(x));
    const wp = try rig.input(std.mem.sliceAsBytes(w));
    const y = try rig.input(std.mem.sliceAsBytes(&t));
    const out = try rig.output(rows * heads * hd * 2);
    var args: cuda.Args = .{};
    inline for (.{ xp, wp, out, @as(c_int, @intCast(heads)), @as(c_int, @intCast(stride)), @as(c_int, @intCast(in_head)), @as(c_longlong, @intCast(pos0)), @as(c_int, @intCast(step)), @as(f32, 1e-6), @as(f32, 10000000), y }) |v| args.add(v);
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(rows * heads) }, .block = .{ .x = 32 } }, s, &args);
    try s.synchronize();
    const got = std.mem.bytesAsSlice(u16, try a.dupe(u8, try rig.outputBytes()));
    try tc.expect(try rig.intact() and !rig.overlap(), "tf_q36m_norm_rope YaRN broke a guard", .{});
    var worst: f64 = 0;
    for (0..rows) |r| for (0..heads) |h| {
        const row = x[r * stride + h * in_head ..][0..hd];
        var ss: f64 = 0;
        for (row) |v| ss += c.fromBf16(v) * c.fromBf16(v);
        const inv = 1 / @sqrt(ss / hd + 1e-6);
        var exact: [hd]f64 = undefined;
        var n: [hd]f64 = undefined; // rounded to bf16 as the kernel rounds its normed values before rope
        for (&exact, &n, row, w) |*e, *b, bits, weight| {
            e.* = c.fromBf16(bits) * inv * c.fromBf16(weight);
            b.* = c.fromBf16(c.toBf16(@floatCast(e.*)));
        }
        const pos: f32 = @floatFromInt(pos0 + r * step);
        for (0..hd) |d| {
            var want: Want = undefined;
            if (d < 64) {
                const pa = n[d % 32];
                const pb = n[d % 32 + 32];
                want = yarnRope(pa, pb, d, pos, t);
                // a normed value whose fp32 and float64 norms round to neighbouring bf16 values: a full spacing each
                want.bound += (@abs(pa) + @abs(pb)) * 2 * c.unit(.bf16) * t[0] + 64 * c.unit(.f32) * (@abs(pa) + @abs(pb)) * t[0];
            } else want = .{ .want = exact[d], .bound = (2 * c.unit(.bf16) + 64 * c.unit(.f32)) * @abs(exact[d]) + 0x1p-126 };
            const g = c.fromBf16(got[(r * heads + h) * hd + d]);
            try tc.expect(std.math.isFinite(g) and @abs(g - want.want) <= want.bound, "tf_q36m_norm_rope YaRN {s} at {d} step {d}: row {d} head {d} dim {d}: {d} vs {d}", .{ if (query) "q" else "k", pos0, step, r, h, d, g, want.want });
            worst = @max(worst, @abs(g - want.want) / want.bound);
        }
    };
    std.debug.print("PASS tf_q36m_norm_rope YaRN {s}: rows from {d}, step {d}, against float64 (worst {d:.3} of the bound)\n", .{ if (query) "q" else "k", pos0, step, worst });
}

/// Every YaRN case: the target's rope on bf16 and e4m3 caches, text and image rows, and the MTP layer's.
pub fn run(gpu: tc.Gpu) !void {
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var mod = try cuda.Module.load(gpu.d, cuda.kernels.qwen35);
    defer mod.unload();
    var kv8 = try cuda.Module.load(gpu.d, cuda.kernels.qwen35_kv8);
    defer kv8.unload();
    var mtp_mod = try cuda.Module.load(gpu.d, cuda.kernels.qwen36_mtp);
    defer mtp_mod.unload();
    for ([_]bool{ false, true }) |image| {
        try target(gpu, s, try mod.function("tf_q35_rope"), false, image);
        try target(gpu, s, try kv8.function("tf_q35_rope8"), true, image);
    }
    const norm_rope = try mtp_mod.function("tf_q36m_norm_rope");
    for ([_]bool{ true, false }) |query| for ([_][2]usize{ .{ 262144, 0 }, .{ 262144, 1 }, .{ 1048573, 1 }, .{ 1048575, 0 } }) |p| try mtp(gpu, s, norm_rope, query, p[0], p[1]);
}
