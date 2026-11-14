//! The grouped experts' tensor-unit rule: safe groups sum exactly in any order; on an M5 so does the tensor unit.
const std = @import("std");
const mtl = @import("metal");
const pkg = @import("dsv41_metal");

const xm = pkg.pipelines.xmma;
const testing = std.testing;
const bound: i64 = (1 << 24) - 1;

/// xu_exact's rule on magnitudes in one unit: 12 * sum < 2^24 * the lowest set bit among them.
fn safe(mags: []const u64) bool {
    return use(mags) < 1;
}

/// 12 * sum over 2^24 * the lowest set bit: below 1 is safe.
fn use(mags: []const u64) f64 {
    var bits: u64 = 0;
    var sum: u64 = 0;
    for (mags) |m| bits |= m;
    for (mags) |m| sum += m;
    if (bits == 0) return 0;
    return @as(f64, @floatFromInt(12 * sum)) / @as(f64, @floatFromInt((@as(u64, 1) << 24) * (bits & (~bits +% 1))));
}

/// fp4 E2M1 magnitudes times 2, by code.
const fp4x2 = [8]i64{ 0, 1, 2, 3, 4, 6, 8, 12 };

/// The largest E4M3 magnitude (value * 512) at most v; down's largest value of at most 8 significant bits at most v.
fn e4Floor(v: u64) u64 {
    return @min(leading(v, 4), 229376);
}
fn b8Floor(v: u64) u64 {
    return leading(v, 8);
}
fn leading(v: u64, bits: u6) u64 {
    if (v < @as(u64, 1) << bits) return v;
    const s: u6 = @intCast(64 - @clz(v) - bits);
    return (v >> s) << s;
}

/// A group of 16 rows near the bound (a quarter just past it): magnitudes, signs and fp4 weights, and each row's scale.
const Group = struct {
    mag: [16][32]u64 = undefined,
    sx: [32]i64 = undefined,
    w: [32][32]u8 = undefined,
    exp: [16]i32 = undefined,

    /// Rows share the roles of their positions (top, mid, low, zero) and signs; columns make sums swing and cancel.
    fn make(rng: std.Random, down: bool) Group {
        var g: Group = .{};
        var pos: [32]usize = undefined;
        var rank: [32]usize = undefined;
        for (&pos, 0..) |*p, i| p.* = i;
        rng.shuffle(usize, &pos);
        for (pos, 0..) |p, i| rank[p] = i;
        var role: [32]u8 = @splat(1);
        const top = 1 + rng.uintLessThan(usize, 12);
        const low = 1 + rng.uintLessThan(usize, 4);
        for (pos[0..top]) |p| role[p] = 0;
        for (pos[top .. top + low]) |p| role[p] = 2;
        for (pos[top + low .. @min(32, top + low + rng.uintLessThan(usize, 6))]) |p| role[p] = 3;
        for (&g.sx) |*s| s.* = if (rng.boolean()) 1 else -1;
        const past = rng.uintLessThan(u8, 4) == 0;
        const shift: i32 = if (down) 0 else rng.uintLessThan(u8, 3);
        for (0..16) |r| {
            const d: i64 = switch (rng.uintLessThan(u8, 6)) {
                0, 1 => 1 + rng.uintLessThan(u32, 4),
                2, 3 => 1 + rng.uintLessThan(u32, 2000),
                else => 1 + rng.uintLessThan(u32, 1_000_000),
            };
            var rem: i64 = if (past) @divTrunc(bound + 12 * d, 12) + 1 else @divTrunc(bound - 12 * d, 12);
            const v = &g.mag[r];
            v.* = @splat(0);
            for (0..32) |k| if (role[k] == 2) {
                v[k] = 1 + 2 * @as(u64, rng.uintLessThan(u8, if (down) 4 else 2));
                rem -= @intCast(v[k]);
            };
            var left: usize = 0;
            for (role) |x| left += @intFromBool(x < 2);
            var first = !down;
            for ([_]u8{ 0, 1 }) |want_role| for (0..32) |k| if (role[k] == want_role) {
                const cap: i64 = if (down) (1 << 20) - 1 else 229376;
                const share = @as(f64, @floatFromInt(@max(rem, 0))) / @as(f64, @floatFromInt(left));
                var want: i64 = if (left == 1) rem else @intFromFloat(share * (if (want_role == 0) 1 + rng.float(f64) else rng.float(f64)));
                if (first) want = 229376; // up: one 448 keeps act_quant's scale a power of two, so its codes are these
                first = false;
                want = std.math.clamp(want, 0, @min(@max(rem, 0), cap));
                v[k] = if (down) b8Floor(@intCast(want)) else e4Floor(@intCast(want));
                rem -= @intCast(v[k]);
                left -= 1;
            };
            const top_v = std.mem.max(u64, v);
            g.exp[r] = if (down) 9 - @as(i32, @intCast(63 - @clz(top_v))) else shift - 9;
        }
        var big: usize = 0;
        for (0..32) |k| if (role[k] == 0 and g.mag[0][k] > g.mag[0][big]) {
            big = k;
        };
        for (0..32) |n| {
            const split = rng.uintLessThan(usize, 32);
            for (0..32) |k| {
                const t: i64 = switch (n % 8) {
                    0 => 1,
                    1 => if (rng.boolean()) 1 else -1,
                    2 => if (k < 16) 1 else -1,
                    3 => if ((k / 8) % 2 == 0) 1 else -1,
                    4 => if (rank[k] < 16) 1 else -1,
                    5 => if (k < split) 1 else -1,
                    6 => if (role[k] == 0) 1 else if (role[k] == 2 or rng.uintLessThan(u8, 3) == 0) -1 else 1,
                    else => if (k == big) -1 else 1,
                };
                const code: u8 = if (role[k] == 2) (if (rng.boolean()) 1 else 3) else (if (rng.uintLessThan(u8, 5) == 0) 6 else 7);
                g.w[n][k] = code | @as(u8, if (g.sx[k] * t < 0) 8 else 0);
            }
        }
        return g;
    }

    fn value(g: *const Group, r: usize, k: usize) f32 {
        return std.math.ldexp(@as(f32, @floatFromInt(g.sx[k] * @as(i64, @intCast(g.mag[r][k])))), g.exp[r]);
    }

    /// The exact sum of row r times column n, in units of 2^(exp - 1) (the fp4 values are halves of fp4x2).
    fn exact(g: *const Group, r: usize, n: usize) i64 {
        var s: i64 = 0;
        for (0..32) |k| {
            const c = g.w[n][k];
            const w = fp4x2[c & 7] * @as(i64, if (c & 8 != 0) -1 else 1);
            s += g.sx[k] * @as(i64, @intCast(g.mag[r][k])) * w;
        }
        return s;
    }
};

test "dsv41 metal units: a safe group's fp32 sum is exact in every order; past the bound some order rounds" {
    var prng = std.Random.DefaultPrng.init(0x5afe);
    const rng = prng.random();
    var safe_rows: usize = 0;
    var past_rows: usize = 0;
    var rounded: usize = 0;
    for (0..200) |gi| {
        const g = Group.make(rng, gi % 2 == 1);
        for (0..16) |r| {
            const ok = safe(&g.mag[r]);
            if (ok) safe_rows += 1 else past_rows += 1;
            for (0..32) |n| {
                const want = std.math.ldexp(@as(f64, @floatFromInt(g.exact(r, n))), g.exp[r] - 1);
                var terms: [32]f32 = undefined;
                for (&terms, 0..) |*t, k| {
                    const c = g.w[n][k];
                    t.* = g.value(r, k) * @as(f32, @floatFromInt(fp4x2[c & 7])) * 0.5 * @as(f32, if (c & 8 != 0) -1 else 1);
                }
                var differs = false;
                for (0..6) |o| {
                    if (o > 0) rng.shuffle(f32, &terms);
                    var s: f32 = 0;
                    for (terms) |t| s += t;
                    if (@as(f64, s) != want) differs = true;
                }
                if (ok) try testing.expect(!differs);
                rounded += @intFromBool(differs);
            }
        }
    }
    std.debug.print("PASS dsv41 metal units rule: {d} safe rows exact in 6 orders of every sum; {d} rows past the bound, {d} sums rounding in some order\n", .{ safe_rows, past_rows, rounded });
    try testing.expect(rounded > 0);
}

/// Group g's sums before the scale fma: the tensor unit as xu_chain runs it, decode's fmas as its fallback does.
const probe =
    \\template <bool FP8>
    \\inline void xu_probe(device const half* xh, device const float* sb, device const bfloat* bb, device const uchar* w,
    \\                     device float* tu, device float* ch, uint lane, uint g) {
    \\  matmul2d<xu_shape, execution_simdgroup> op;
    \\  xu_wt wt((device uchar*)(w + 512 * g), dextents<int32_t, 2>(32, 32));
    \\  const short2 home = xu_home(lane);
    \\  const uint hx = home.x, hy = home.y;
    \\  const uint4 x0 = *(device const uint4*)(xh + 512 * g + 32 * hy + 2 * hx);
    \\  const uint4 x1 = *(device const uint4*)(xh + 512 * g + 32 * (hy + 8) + 2 * hx);
    \\  auto wg = wt.slice<32, 32>(0, 0);
    \\  auto left = op.get_left_input_cooperative_tensor<half, metal_fp4_e2m1_format, float>();
    \\  auto acc = op.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(left)>,
    \\                                                   metal::remove_addrspace_t<decltype(wg)>, float>();
    \\  for (int i = 0; i < 16; ++i) {
    \\    const uint j = (i & 3) + 4 * (i >> 3);
    \\    left[i] = as_type<half2>(((i >> 2) & 1 ? x1 : x0)[j >> 1])[j & 1];
    \\  }
    \\  op.run(left, wg, acc);
    \\  for (int i = 0; i < 16; ++i) {
    \\    const uint row = hy + 8 * ((i >> 2) & 1), col = hx + (i & 3) + 16 * (i >> 3);
    \\    tu[512 * g + 32 * row + col] = acc[i] * (FP8 ? 0x1p-14f : fabs(sb[16 * g + row]) * 0x1p-14f);
    \\  }
    \\  for (uint o = lane; o < 512; o += 32)
    \\    ch[512 * g + o] = xu_exact_sum<FP8>(w + 512 * g + 16 * (o % 32), xh + 512 * g + 32 * (o / 32), bb + 512 * g + 32 * (o / 32));
    \\}
    \\kernel void xu_probe_up(device const half* xh [[buffer(0)]], device const float* sb [[buffer(1)]],
    \\    device const bfloat* bb [[buffer(2)]], device const uchar* w [[buffer(3)]], device float* tu [[buffer(4)]],
    \\    device float* ch [[buffer(5)]], uint lane [[thread_index_in_simdgroup]], uint g [[threadgroup_position_in_grid]]) {
    \\  xu_probe<true>(xh, sb, bb, w, tu, ch, lane, g);
    \\}
    \\kernel void xu_probe_down(device const half* xh [[buffer(0)]], device const float* sb [[buffer(1)]],
    \\    device const bfloat* bb [[buffer(2)]], device const uchar* w [[buffer(3)]], device float* tu [[buffer(4)]],
    \\    device float* ch [[buffer(5)]], uint lane [[thread_index_in_simdgroup]], uint g [[threadgroup_position_in_grid]]) {
    \\  xu_probe<false>(xh, sb, bb, w, tu, ch, lane, g);
    \\}
;

const Tally = struct { sums: usize = 0, tu: usize = 0, chain: usize = 0, apart: usize = 0, near: usize = 0 };

test "dsv41 metal units: near the bound, the tensor unit and decode's fmas give every safe group's exact sum" {
    const device = try mtl.Device.init();
    if (!xm.units(device)) return error.SkipZigTest;
    const queue = try device.queue();
    const a = testing.allocator;
    const units = try xm.source(a, "", &.{}, true);
    defer a.free(units);
    const text = try std.mem.concat(a, u8, &.{ units, probe });
    defer a.free(text);
    const lib = try mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
    defer lib.deinit();
    const names = [_][]const u8{ "tf_dsv41_xu_quant", "tf_dsv41_xu_dprep", "xu_probe_up", "xu_probe_down" };
    var pipes: [4]mtl.Pipeline = undefined;
    for (&pipes, names) |*p, name| p.* = try mtl.Pipeline.init(device, lib, name, false);
    defer for (pipes) |p| p.deinit();
    const G = 8192;
    const sh = mtl.ResourceOptions.shared;
    const bufs = [_]usize{ 2 * 512 * G, 2 * 512 * G, 4 * 16 * G, 512 * G, 4 * 512 * G, 4 * 512 * G };
    var b: [6]mtl.Buffer = undefined; // bf16 rows, halves, block scales, weights, tensor unit, chain
    for (&b, bufs) |*x, len| x.* = try device.buffer(len, sh);
    defer for (b) |x| x.deinit();
    const groups = try a.alloc(Group, G);
    defer a.free(groups);
    const batches = if (std.c.getenv("TF_DSV41_UNITS_BATCHES")) |v| std.fmt.parseInt(usize, std.mem.span(v), 10) catch 2 else 2;
    var prng = std.Random.DefaultPrng.init(0xb0e2);
    const rng = prng.random();
    var flags: usize = 0;
    var tallies: [2][2]Tally = @splat(@splat(.{})); // [layout][safe]
    for (0..2 * batches) |bi| {
        const down = bi % 2 == 1;
        const rows = b[0].slice(u16, 512 * G);
        const wts = b[3].slice(u8, 512 * G);
        for (groups, 0..) |*g, gi| {
            g.* = Group.make(rng, down);
            for (0..16) |r| for (0..32) |k| {
                rows[512 * gi + 32 * r + k] = @truncate(@as(u32, @bitCast(g.value(r, k))) >> 16);
            };
            for (0..32) |n| for (0..16) |j| {
                wts[512 * gi + 16 * n + j] = g.w[n][2 * j] | g.w[n][2 * j + 1] << 4;
            };
        }
        const cb = queue.commandBuffer();
        const e = cb.compute(.serial);
        e.setPipeline(pipes[if (down) 1 else 0]);
        for (0..3) |i| e.setBuffer(b[i], 0, i);
        e.setValue([2]u32{ 32, 16 * G }, 3);
        e.dispatchThreads(mtl.Size.of(if (down) 8 else 1, 16 * G, 1), mtl.Size.of(32, 1, 1));
        e.setPipeline(pipes[if (down) 3 else 2]);
        for ([_]usize{ 1, 2, 0, 3, 4, 5 }, 0..) |buf, i| e.setBuffer(b[buf], 0, i);
        e.dispatchGroups(mtl.Size.of(G, 1, 1), mtl.Size.of(32, 1, 1));
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |msg| {
            std.debug.print("command buffer failed: {s}\n", .{msg});
            return error.CommandBufferFailed;
        }
        const scales = b[2].slice(f32, 16 * G);
        const tu = b[4].slice(f32, 512 * G);
        const chain = b[5].slice(f32, 512 * G);
        for (groups, 0..) |*g, gi| for (0..16) |r| {
            const ok = !std.math.signbit(scales[16 * gi + r]);
            flags += @intFromBool(ok != safe(&g.mag[r]));
            const t = &tallies[@intFromBool(down)][@intFromBool(ok)];
            t.near += @intFromBool(ok and use(&g.mag[r]) >= 0.999);
            for (0..32) |n| {
                const exact = std.math.ldexp(@as(f64, @floatFromInt(g.exact(r, n))), (if (down) g.exp[r] else -9) - 1 - 14);
                const want: f32 = @floatCast(exact);
                const o = 512 * gi + 32 * r + n;
                t.sums += 1;
                t.tu += @intFromBool(@as(u32, @bitCast(tu[o])) != @as(u32, @bitCast(want)));
                t.chain += @intFromBool(@as(u32, @bitCast(chain[o])) != @as(u32, @bitCast(want)));
                t.apart += @intFromBool(@as(u32, @bitCast(tu[o])) != @as(u32, @bitCast(chain[o])));
            }
        };
    }
    for (tallies, [_][]const u8{ "up (act_quant codes)", "down (dprep halves)" }) |l, name| {
        std.debug.print("PASS dsv41 metal units near the bound, {s}: safe {d} sums ({d} rows within 0.1% of it), tensor unit vs exact {d}, chain vs exact {d}; past the bound {d} sums, tensor unit vs chain {d}\n", .{ name, l[1].sums, l[1].near, l[1].tu, l[1].chain, l[0].sums, l[0].apart });
        try testing.expectEqual(@as(usize, 0), l[1].tu + l[1].chain);
        try testing.expect(l[0].apart > 0);
    }
    try testing.expectEqual(@as(usize, 0), flags);
}
