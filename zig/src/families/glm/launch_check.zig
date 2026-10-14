//! GLM's 1,024-thread router, KDA and hc_expand_10 launches: capacity, host values, full coverage and recurrent seams.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");

fn kernel(comptime key: []const u8) sources.glm.Kernel {
    for (sources.glm.all) |k| if (std.mem.eql(u8, key, k.key)) return k;
    unreachable;
}

fn bf(v: f32) u16 {
    const u: u32 = @bitCast(v);
    return @intCast((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

fn value(v: u16) f32 {
    return @bitCast(@as(u32, v) << 16);
}

fn dyadic(i: usize) f32 {
    return @as(f32, @floatFromInt(@as(i32, @intCast(i % 17)) - 8)) / 32;
}

fn routerWeight(e: usize, k: usize) f32 {
    return dyadic(e * 11 + k * 7 + k / 32);
}

test "GLM router: every row count supports 1024 threads and equals FP64" {
    const d = mtl.Device.init() catch return error.SkipZigTest;
    defer d.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const q = try d.queue();
    defer q.deinit();
    const gen = comptime kernel("router");
    const lib = try mtl.Library.fromSource(d, gen.source, mtl.CompileOptions.mlx());
    defer lib.deinit();
    const K = 4096;
    const NE = 288;
    const R = 16;
    const x = try d.buffer(R * K * 4, mtl.ResourceOptions.shared);
    defer x.deinit();
    const w = try d.buffer(NE * K * 2, mtl.ResourceOptions.shared);
    defer w.deinit();
    const y = try d.buffer((R * NE + 32) * 4, mtl.ResourceOptions.shared);
    defer y.deinit();
    const xs = x.slice(f32, R * K);
    for (xs, 0..) |*v, i| v.* = dyadic((i / K) * 7 + (i % K) * 5);
    const weights = w.slice(u16, NE * K);
    for (0..NE / 4) |g| for (0..8) |m| for (0..K / 32) |it| for (0..4) |tm| for (0..4) |tn| {
        const at = ((((g * 8 + m) * (K / 32) + it) * 4 + tm) * 4 + tn);
        weights[at] = bf(routerWeight(g * 4 + tn, it * 32 + m * 4 + tm));
    };
    const refs = try std.testing.allocator.alloc(f64, R * NE);
    defer std.testing.allocator.free(refs);
    for (refs, 0..) |*v, i| {
        v.* = 0;
        for (0..K) |k| v.* += @as(f64, xs[(i / NE) * K + k]) * routerWeight(i % NE, k);
    }
    for (gen.functions, 1..) |name, rows| {
        const p = try mtl.Pipeline.init(d, lib, name, false);
        defer p.deinit();
        try std.testing.expect(p.maxThreads() >= 1024);
        @memset(y.slice(u32, R * NE + 32), 0x7fc00000);
        const cb = q.commandBuffer();
        const e = cb.compute(.serial);
        e.setPipeline(p);
        e.setBuffer(x, 0, 0);
        e.setBuffer(w, 0, 1);
        e.setBuffer(y, 0, 2);
        e.dispatchThreads(mtl.Size.of(1024 * NE / 16, 1, 1), mtl.Size.of(1024, 1, 1));
        e.end();
        cb.commit();
        cb.wait();
        try std.testing.expect(cb.failure() == null);
        for (y.slice(f32, R * NE + 32)[0 .. rows * NE], refs[0 .. rows * NE]) |actual, expected| {
            try std.testing.expect(std.math.isFinite(actual));
            try std.testing.expectEqual(expected, @as(f64, actual));
        }
        for (y.slice(u32, R * NE + 32)[rows * NE ..]) |bits| try std.testing.expectEqual(@as(u32, 0x7fc00000), bits);
    }
}

fn hcValue(i: usize) f32 {
    return value(bf(dyadic(i)));
}

test "GLM hc_expand_10: 1024 threads, every stream of every row equals FP64, rows past the window untouched" {
    const d = mtl.Device.init() catch return error.SkipZigTest;
    defer d.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const q = try d.queue();
    defer q.deinit();
    const gen = comptime kernel("hc_expand_10");
    const lib = try mtl.Library.fromSource(d, gen.source, mtl.CompileOptions.mlx());
    defer lib.deinit();
    const p = try mtl.Pipeline.init(d, lib, gen.functions[0], false);
    defer p.deinit();
    try std.testing.expect(p.maxThreads() >= 1024);
    const D = 4096;
    const S = 4;
    const R = 16;
    const opts = mtl.ResourceOptions.shared;
    const xold = try d.buffer(R * S * D * 2, opts);
    defer xold.deinit();
    const branch = try d.buffer(R * D * 2, opts);
    defer branch.deinit();
    const post = try d.buffer(R * S * 4, opts);
    defer post.deinit();
    const comb = try d.buffer(R * S * S * 4, opts);
    defer comb.deinit();
    const xnew = try d.buffer((R + 1) * S * D * 2, opts);
    defer xnew.deinit();
    const side = try d.buffer((R + R * S * D) * 4, opts);
    defer side.deinit();
    for (xold.slice(u16, R * S * D), 0..) |*v, i| v.* = bf(dyadic(i / D * 7 + i % D * 5));
    for (branch.slice(u16, R * D), 0..) |*v, i| v.* = bf(dyadic(i / D * 3 + i % D * 11));
    for (post.slice(f32, R * S), 0..) |*v, i| v.* = dyadic(i * 5 + 2);
    for (comb.slice(f32, R * S * S), 0..) |*v, i| v.* = dyadic(i * 3 + 1);
    for ([_]usize{ 1, 5, 16 }) |rows| {
        @memset(xnew.slice(u16, (R + 1) * S * D), 0x7fc0);
        @memset(side.slice(u32, R + R * S * D), 0x7fc00000);
        const cb = q.commandBuffer();
        const e = cb.compute(.serial);
        e.setPipeline(p);
        for ([_]mtl.Buffer{ xold, branch, post, comb }, 0..) |b, i| e.setBuffer(b, 0, i);
        e.setValue(@as(f32, 1e-6), 4);
        e.setBuffer(xnew, 0, 5);
        e.setBuffer(side, 0, 6);
        e.setBuffer(side, R * 4, 7);
        e.dispatchThreads(mtl.Size.of(1024 * rows, 1, 1), mtl.Size.of(1024, 1, 1));
        e.end();
        cb.commit();
        cb.wait();
        try std.testing.expect(cb.failure() == null);
        const got = xnew.slice(u16, (R + 1) * S * D);
        // dyadic inputs keep every fp32 product and sum exact, so the kernel's one rounding is the bf16 store
        for (0..rows) |r| for (0..S) |s| for (0..D) |dd| {
            const c = comb.slice(f32, R * S * S)[r * S * S ..];
            var want: f64 = @as(f64, post.slice(f32, R * S)[r * S + s]) * hcValue(r * 3 + dd * 11);
            for (0..S) |t2| want += @as(f64, c[t2 * S + s]) * hcValue((r * S + t2) * 7 + dd * 5);
            try std.testing.expectEqual(bf(@floatCast(want)), got[(r * S + s) * D + dd]);
        };
        for (got[rows * S * D ..]) |v| try std.testing.expectEqual(@as(u16, 0x7fc0), v);
        for (side.slice(u32, R + R * S * D)) |v| try std.testing.expectEqual(@as(u32, 0x7fc00000), v);
    }
}

const Kda = struct {
    d: mtl.Device,
    q: mtl.Queue,
    p: mtl.Pipeline,
    b: [19]mtl.Buffer,
    h: usize,
    w: usize,
    pitch: usize,

    fn init(d: mtl.Device, q: mtl.Queue, p: mtl.Pipeline, h: usize) !Kda {
        const w = h * 128;
        const pitch = 3 * w + 256 + h;
        const sizes = [_]usize{
            16 * pitch * 2, 8,          3 * 3 * w * 2, 4 * 3 * w * 4, w * 16 * 4,
            w * 2 * 2,      w * 2 * 2,  w * 16 * 4,    w * 2 * 2,     w * 2 * 2,
            h * 4,          w * 4,      w * 128 * 4,   128 * 4,       4,
            4,              16 * w * 2, w * 128 * 4,   3 * 3 * w * 2,
        };
        var k = Kda{ .d = d, .q = q, .p = p, .b = undefined, .h = h, .w = w, .pitch = pitch };
        var made: usize = 0;
        errdefer for (k.b[0..made]) |b| b.deinit();
        for (sizes) |n| {
            k.b[made] = try d.buffer(n, mtl.ResourceOptions.shared);
            @memset(k.b[made].contents()[0..n], 0);
            made += 1;
        }
        k.scalarInputs();
        return k;
    }

    fn deinit(k: Kda) void {
        for (k.b) |b| b.deinit();
    }

    fn scalarInputs(k: Kda) void {
        const ps = k.b[0].slice(u16, 16 * k.pitch);
        for (0..16) |r| {
            for (0..3 * k.w) |i| ps[r * k.pitch + i] = bf(activation(@intCast(r), i / k.w, (i % k.w) / 128, i % 128));
            @memset(ps[r * k.pitch + 3 * k.w ..][0..256], bf(1.0 / 16.0));
            for (0..k.h) |h| ps[r * k.pitch + 3 * k.w + 256 + h] = bf(betaInput(r, h));
        }
        const cs = k.b[2].slice(u16, 9 * k.w);
        for (cs, 0..) |*v, i| v.* = bf(activation(
            @as(i32, @intCast(i / (3 * k.w))) - 3,
            (i % (3 * k.w)) / k.w,
            (i % k.w) / 128,
            i % 128,
        ));
        @memset(k.b[3].slice(f32, 12 * k.w), 0.25);
        // These packed projections give constant decay input 0.1875 and output gate 0.4375.
        @memset(k.b[4].slice(u32, k.w * 16), 0x11111111);
        @memset(k.b[5].slice(u16, k.w * 2), bf(1.0 / 64.0));
        @memset(k.b[6].slice(u16, k.w * 2), bf(1.0 / 128.0));
        @memset(k.b[7].slice(u32, k.w * 16), 0x22222222);
        @memset(k.b[8].slice(u16, k.w * 2), bf(1.0 / 32.0));
        @memset(k.b[9].slice(u16, k.w * 2), bf(-1.0 / 128.0));
        @memset(k.b[10].slice(f32, k.h), 1);
        for (k.b[11].slice(f32, k.w), 0..) |*v, i| v.* = dtBias(i / 128);
        k.seed();
        for (k.b[13].slice(f32, 128), 0..) |*v, i| v.* = gain(i);
        k.b[14].slice(f32, 1)[0] = -5;
        k.b[15].slice(f32, 1)[0] = 1e-5;
    }

    fn seed(k: Kda) void {
        for (k.b[12].slice(f32, k.w * 128), 0..) |*v, i| v.* = initial(i / (128 * 128), (i / 128) % 128);
    }

    fn run(k: Kda, p: mtl.Pipeline, first: usize, rows: usize) !void {
        try k.runGroup(p, first, rows, 32);
    }

    fn runGroup(k: Kda, p: mtl.Pipeline, first: usize, rows: usize, ty: usize) !void {
        @memcpy(k.b[1].slice(i32, 2), &[_]i32{ @intCast(rows), @intCast(k.pitch) });
        @memset(k.b[16].slice(u16, 16 * k.w), 0x7fc0);
        @memset(k.b[17].slice(u32, k.w * 128), 0x7fc00000);
        @memset(k.b[18].slice(u16, 9 * k.w), 0x7fc0);
        const cb = k.q.commandBuffer();
        const e = cb.compute(.serial);
        e.setPipeline(p);
        for (k.b, 0..) |b, i| e.setBuffer(b, if (i == 0) first * k.pitch * 2 else 0, i);
        e.dispatchThreads(mtl.Size.of(32, ty, k.h), mtl.Size.of(32, ty, 1));
        e.end();
        cb.commit();
        cb.wait();
        try std.testing.expect(cb.failure() == null);
        for (k.b[16].slice(u16, 16 * k.w)[0 .. rows * k.w]) |v| try std.testing.expect(std.math.isFinite(value(v)));
        for (k.b[16].slice(u16, 16 * k.w)[rows * k.w ..]) |v| try std.testing.expectEqual(@as(u16, 0x7fc0), v);
        for (k.b[17].slice(f32, k.w * 128)) |v| try std.testing.expect(std.math.isFinite(v));
        for (k.b[18].slice(u16, 9 * k.w)) |v| try std.testing.expect(std.math.isFinite(value(v)));
    }

    fn reference(k: Kda, rows: usize) !void {
        var worst_y: f64 = 0;
        var worst_state: f64 = 0;
        for (0..k.h) |h| {
            var state: [128]f64 = undefined;
            for (&state, 0..) |*v, dv| v.* = initial(h, dv);
            const decay = @exp(-5.0 / (1.0 + @exp(-(@as(f64, dtBias(h)) + 0.1875))));
            for (0..rows) |r| {
                var y: [128]f64 = undefined;
                var norm: f64 = 0;
                const qv = sl(@as(f64, activation(0, 0, h, 0)));
                const kv = sl(@as(f64, activation(0, 1, h, 0)));
                const qb: f64 = value(bf(@floatCast(qv / @sqrt(128 * qv * qv + 1e-6) / @sqrt(@as(f64, 128)))));
                const kb: f64 = value(bf(@floatCast(kv / @sqrt(128 * kv * kv + 1e-6))));
                const beta = sigmoidBf(betaInput(r, h));
                for (&state, &y, 0..) |*st, *out, dv| {
                    var conv: f64 = 0;
                    for (0..4) |tap| conv += @as(f64, activation(@as(i32, @intCast(r + tap)) - 3, 2, h, dv)) * 0.25;
                    const vv = sl(value(bf(@floatCast(conv))));
                    st.* *= decay;
                    st.* += kb * (vv - 128 * st.* * kb) * beta;
                    out.* = value(bf(@floatCast(128 * st.* * qb)));
                    norm += out.* * out.*;
                }
                const inv = 1.0 / @sqrt(norm / 128 + @as(f64, k.b[15].slice(f32, 1)[0]));
                for (y, 0..) |out, dv| {
                    const expected = out * inv * gain(dv) / (1 + @exp(-@as(f64, 0.4375)));
                    const actual = value(k.b[16].slice(u16, rows * k.w)[r * k.w + h * 128 + dv]);
                    const err = @abs(@as(f64, actual) - expected);
                    const unit = @as(u32, bf(@floatCast(@abs(expected))) & 0x7f80) << 16;
                    const step: f64 = @as(f32, @bitCast(unit)) / 128;
                    worst_y = @max(worst_y, err / @max(step, 1e-8));
                    try std.testing.expect(err <= 2 * step + 1e-6);
                }
            }
            for (state, 0..) |expected, dv| for (0..128) |dk| {
                const actual = k.b[17].slice(f32, k.w * 128)[(h * 128 + dv) * 128 + dk];
                worst_state = @max(worst_state, @abs(@as(f64, actual) - expected));
                try std.testing.expectApproxEqAbs(expected, @as(f64, actual), 1e-5);
            };
        }
        for (k.b[18].slice(u16, 9 * k.w), 0..) |v, i| {
            const row = @as(i32, @intCast(rows + i / (3 * k.w))) - 3;
            const expected = bf(activation(row, (i % (3 * k.w)) / k.w, (i % k.w) / 128, i % 128));
            try std.testing.expectEqual(expected, v);
        }
        std.debug.print(
            "KDA heads={d} rows={d}: worst {d:.4} bf16 steps, state {d:.9}\n",
            .{ k.h, rows, worst_y, worst_state },
        );
    }

    fn seams(k: Kda) !void {
        try k.run(k.p, 0, 16);
        const a = std.testing.allocator;
        const y = try a.dupe(u16, k.b[16].slice(u16, 16 * k.w));
        defer a.free(y);
        const st = try a.dupe(u8, k.b[17].contents()[0 .. k.w * 128 * 4]);
        defer a.free(st);
        const cs = try a.dupe(u8, k.b[18].contents()[0 .. 9 * k.w * 2]);
        defer a.free(cs);
        var first: usize = 0;
        for ([_]usize{ 1, 2, 5, 8 }) |rows| {
            try k.run(k.p, first, rows);
            try std.testing.expectEqualSlices(u16, y[first * k.w ..][0 .. rows * k.w], k.b[16].slice(u16, rows * k.w));
            @memcpy(k.b[12].contents()[0..st.len], k.b[17].contents()[0..st.len]);
            @memcpy(k.b[2].contents()[0..cs.len], k.b[18].contents()[0..cs.len]);
            first += rows;
        }
        try std.testing.expectEqualSlices(u8, st, k.b[17].contents()[0..st.len]);
        try std.testing.expectEqualSlices(u8, cs, k.b[18].contents()[0..cs.len]);
    }

    fn randomControl(k: Kda, source: []const u8) !void {
        const a = std.testing.allocator;
        const from = try std.fmt.allocPrint(a, "<{d}, 128, 4, 32, 4, 4>", .{k.h});
        defer a.free(from);
        const to = try std.fmt.allocPrint(a, "<{d}, 128, 4, 16, 4, 4>", .{k.h});
        defer a.free(to);
        try std.testing.expectEqual(@as(usize, 2), std.mem.count(u8, source, from));
        const reduced = try std.mem.replaceOwned(u8, a, source, from, to);
        defer a.free(reduced);
        const control = try std.mem.replaceOwned(
            u8,
            a,
            reduced,
            "max_total_threads_per_threadgroup(1024)",
            "max_total_threads_per_threadgroup(512)",
        );
        defer a.free(control);
        const lib = try mtl.Library.fromSource(k.d, control, mtl.CompileOptions.mlx());
        defer lib.deinit();
        const gen = if (k.h == 64) comptime kernel("kda_rows") else comptime kernel("kda_rows_tp");
        const p = try mtl.Pipeline.init(k.d, lib, gen.functions[0], false);
        defer p.deinit();
        try std.testing.expect(p.maxThreads() >= 512);
        var prng = std.Random.DefaultPrng.init(679);
        const rnd = prng.random();
        for ([_]usize{ 0, 2 }) |idx| {
            for (k.b[idx].slice(u16, k.b[idx].length() / 2)) |*v| v.* = bf(dyadic(rnd.int(u16)));
        }
        for ([_]usize{ 4, 7 }) |idx| {
            for (k.b[idx].slice(u32, k.w * 16)) |*v| v.* = rnd.int(u32);
        }
        for ([_]usize{ 5, 8 }) |idx| {
            for (k.b[idx].slice(u16, k.w * 2)) |*v| v.* = bf(@as(f32, @floatFromInt(rnd.int(u8) % 3 + 1)) / 64);
        }
        for ([_]usize{ 6, 9 }) |idx| {
            for (k.b[idx].slice(u16, k.w * 2)) |*v| v.* = bf(dyadic(rnd.int(u16)) / 4);
        }
        for (k.b[3].slice(f32, 12 * k.w)) |*v| v.* = dyadic(rnd.int(u16));
        for (k.b[12].slice(f32, k.w * 128)) |*v| v.* = dyadic(rnd.int(u16)) / 8;
        // TY=16 covers all 128 state rows with the same per-value reductions at a legal 512-thread launch.
        for ([_]usize{ 1, 2, 8, 16 }) |rows| {
            try k.run(k.p, 0, rows);
            var saved: [3][]u8 = undefined;
            var made: usize = 0;
            defer for (saved[0..made]) |bytes| a.free(bytes);
            for ([_]usize{ 16, 17, 18 }, 0..) |idx, i| {
                saved[i] = try a.dupe(u8, k.b[idx].contents()[0..k.b[idx].length()]);
                made += 1;
            }
            try k.runGroup(p, 0, rows, 16);
            for ([_]usize{ 16, 17, 18 }, saved) |idx, bytes| {
                try std.testing.expectEqualSlices(u8, bytes, k.b[idx].contents()[0..bytes.len]);
            }
        }
    }
};

fn activation(r: i32, part: usize, h: usize, dv: usize) f32 {
    return switch (part) {
        0 => @as(f32, @floatFromInt(h % 4 + 1)) / 8,
        1 => @as(f32, @floatFromInt(h % 5 + 1)) / 8,
        else => @as(f32, @floatFromInt(r + 4 + @as(i32, @intCast((h + dv) % 7)))) / 16,
    };
}

fn betaInput(r: usize, h: usize) f32 {
    return @as(f32, @floatFromInt(@as(i32, @intCast((r + h) % 3)) - 1)) / 4;
}

fn dtBias(h: usize) f32 {
    return @as(f32, @floatFromInt(@as(i32, @intCast(h % 5)) - 2)) / 8;
}

fn gain(dv: usize) f32 {
    return 1 + @as(f32, @floatFromInt(dv % 7)) / 16;
}

fn initial(h: usize, dv: usize) f32 {
    return @as(f32, @floatFromInt(@as(i32, @intCast((h + dv) % 9)) - 4)) / 256;
}

fn sigmoidBf(x: f32) f64 {
    const e = value(bf(@floatCast(@exp(@abs(@as(f64, x))))));
    const y = value(bf(1 / value(bf(1 + e))));
    return if (x < 0) y else value(bf(1 - y));
}

fn sl(x: f64) f64 {
    return value(bf(@floatCast(x * sigmoidBf(@floatCast(x)))));
}

fn kdaCheck(comptime heads: usize) !void {
    const d = mtl.Device.init() catch return error.SkipZigTest;
    defer d.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const q = try d.queue();
    defer q.deinit();
    const gen = comptime kernel(if (heads == 64) "kda_rows" else "kda_rows_tp");
    const lib = try mtl.Library.fromSource(d, gen.source, mtl.CompileOptions.mlx());
    defer lib.deinit();
    const p = try mtl.Pipeline.init(d, lib, gen.functions[0], false);
    defer p.deinit();
    try std.testing.expect(p.maxThreads() >= 1024);
    const k = try Kda.init(d, q, p, heads);
    defer k.deinit();
    for ([_]usize{ 1, 2, 8, 16 }) |rows| {
        try k.run(p, 0, rows);
        try k.reference(rows);
    }
    try k.randomControl(gen.source);
    try k.seams();
}

test "GLM KDA 64 heads: 1024 threads, FP64 values and recurrent seams" {
    try kdaCheck(64);
}

test "GLM KDA TP2 32 heads: 1024 threads, FP64 values and recurrent seams" {
    try kdaCheck(32);
}
