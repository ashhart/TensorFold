//! Grouped bf16 experts against float64: the repack, up and downs, a row's bits the same at any row count and form.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const Gpu = check.Gpu;

const grouped = cuda.grouped;
const ex_ = cuda.experts;
const Bf16 = ex_.Bf16Experts;

const E = 6; // routed experts; the shared expert's halves are experts E and E + 1
const D = 256;
const W = 96;
const K = 3; // routed picks a row; a row has K + 2 pairs

fn bf(x: f32) u16 {
    const b: u32 = @bitCast(x);
    return @intCast((b + 0x7FFF + ((b >> 16) & 1)) >> 16);
}

fn f(h: u16) f64 {
    return @as(f32, @bitCast(@as(u32, h) << 16));
}

const Host = struct {
    routed_up: []u16, // [E][D][W]: x @ W, as the stacked checkpoint tensors
    routed_down: []u16, // [E][W][D]
    shared_up: []u16, // [2W][D]: nn.Linear
    shared_down: []u16, // [D][2W]

    /// Expert e's weight from input i to output o, up or down, as the kernels must read it.
    fn at(h: Host, up: bool, e: usize, o: usize, i: usize) f64 {
        if (e < E) return f(if (up) h.routed_up[(e * D + i) * W + o] else h.routed_down[(e * W + i) * D + o]);
        const half = e - E;
        return f(if (up) h.shared_up[(half * W + o) * D + i] else h.shared_down[o * 2 * W + half * W + i]);
    }
};

pub fn experts(gpu: Gpu) !void {
    const a = gpu.gpa;
    var rng = std.Random.DefaultPrng.init(0x7f16);
    const r = rng.random();
    var host: Host = undefined;
    inline for (.{ "routed_up", "routed_down", "shared_up", "shared_down" }, .{ E * D * W, E * W * D, 2 * W * D, D * 2 * W }) |name, n| {
        @field(host, name) = try a.alloc(u16, n);
        for (@field(host, name)) |*v| v.* = bf((r.float(f32) - 0.5) * 0.25);
    }
    defer inline for (.{ "routed_up", "routed_down", "shared_up", "shared_down" }) |name| a.free(@field(host, name));

    const sms: usize = @intCast(try gpu.ctx.attribute(.multiprocessor_count));
    var kx = try Bf16.load(gpu.d, sms);
    defer kx.unload();
    var mod = try cuda.Module.load(gpu.d, cuda.kernels.experts);
    defer mod.unload();
    const router = try grouped.Router.resolve(mod);
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();

    const e = E + 2;
    var up = try cuda.DeviceBuffer.alloc(gpu.d, Bf16.bytes(e, W, D));
    defer up.free();
    var down = try cuda.DeviceBuffer.alloc(gpu.d, Bf16.bytes(e, D, W));
    defer down.free();
    var src: [4]cuda.DeviceBuffer = undefined;
    inline for (.{ "routed_up", "routed_down", "shared_up", "shared_down" }, 0..) |name, i| src[i] = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(@field(host, name)));
    defer for (&src) |*b| b.free();
    const one_up = Bf16.bytes(1, W, D);
    const one_down = Bf16.bytes(1, D, W);
    try kx.pack(s, src[0].ptr, .{ .stride = D * W, .ld = W, .in_out = true }, up.ptr, E, W, D);
    try kx.pack(s, src[1].ptr, .{ .stride = W * D, .ld = D, .in_out = true }, down.ptr, E, D, W);
    for (0..2) |h| {
        try kx.pack(s, src[2].ptr, .{ .ld = D, .n0 = h * W, .in_out = false }, up.ptr + (E + h) * one_up, 1, W, D);
        try kx.pack(s, src[3].ptr, .{ .ld = 2 * W, .k0 = h * W, .in_out = false }, down.ptr + (E + h) * one_down, 1, D, W);
    }
    try s.synchronize();
    const layer: ex_.Layer = .{ .format = .bf16, .up = up.ptr, .down = down.ptr, .width = W, .dims = D, .experts = e, .relu2 = true };
    const g: ex_.Grouped = .{ .bf16 = &kx };

    // 37 rows: three 16-pair decode items for the busiest expert and a partial prompt tile; then each row alone
    const rows = 37;
    const slots = K + 2;
    const x = try a.alloc(u16, rows * D);
    defer a.free(x);
    for (x) |*v| v.* = bf(r.float(f32) * 2 - 1);
    const picks = try a.alloc(i32, rows * slots);
    defer a.free(picks);
    for (0..rows) |row| {
        var chosen: [K]i32 = undefined;
        for (0..K) |k| {
            var p: i32 = undefined;
            while (true) {
                p = @intCast(r.uintLessThan(usize, E));
                if (std.mem.indexOfScalar(i32, chosen[0..k], p) == null) break;
            }
            chosen[k] = p;
            picks[row * slots + k] = p;
        }
        picks[row * slots + K] = E;
        picks[row * slots + K + 1] = E + 1;
    }

    // the float64 reference: act = bf16(relu(bf16(x . up))^2), y = act . down
    const act_ref = try a.alloc(f64, rows * slots * W);
    defer a.free(act_ref);
    const y_ref = try a.alloc(f64, rows * slots * D);
    defer a.free(y_ref);
    const mag = try a.alloc(f64, rows * slots * D);
    defer a.free(mag);
    for (0..rows * slots) |pr| {
        const ex: usize = @intCast(picks[pr]);
        const row = pr / slots;
        for (0..W) |o| {
            var acc: f64 = 0;
            for (0..D) |i| acc += f(x[row * D + i]) * host.at(true, ex, o, i);
            const u = @max(f(bf(@floatCast(acc))), 0);
            act_ref[pr * W + o] = f(bf(@floatCast(u * u)));
        }
        for (0..D) |o| {
            var acc: f64 = 0;
            var m: f64 = 0;
            for (0..W) |i| {
                const t = act_ref[pr * W + i] * host.at(false, ex, o, i);
                acc += t;
                m += @abs(t);
            }
            y_ref[pr * D + o] = acc;
            mag[pr * D + o] = m;
        }
    }

    var dx = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(x));
    defer dx.free();
    var whole: [3][]u8 = undefined; // decode form, all rows: act, y fp32, y bf16
    var got = [3]bool{ false, false, false };
    defer for (whole, got) |w, ok| if (ok) a.free(w);
    for ([_]bool{ false, true }) |prompt| {
        const run = try oneCase(gpu, router, g, s, layer, dx.ptr, picks, rows, slots, prompt);
        defer for (run) |b| a.free(b);
        const act_h = std.mem.bytesAsSlice(u16, run[0]);
        var worst: f64 = 0;
        for (act_h, act_ref) |gv, want| worst = @max(worst, @abs(f(gv) - want) / @max(@abs(want), 1e-3));
        try check.expect(worst <= 1.0 / 64.0, "bf16 experts {s}: up off by {e} (relative)", .{ if (prompt) "prompt" else "decode", worst });
        const y32 = std.mem.bytesAsSlice(f32, run[1]);
        var worst_y: f64 = 0;
        for (y32, y_ref, mag) |gv, want, m| worst_y = @max(worst_y, @abs(@as(f64, gv) - want) / @max(m, 1e-6));
        try check.expect(worst_y <= 1e-5, "bf16 experts {s}: fp32 down off by {e} of its terms' magnitude", .{ if (prompt) "prompt" else "decode", worst_y });
        if (!prompt) {
            for (run, &whole, &got) |b, *w, *ok| {
                w.* = try a.dupe(u8, b);
                ok.* = true;
            }
        } else for (run, whole, [_][]const u8{ "prompt up", "prompt fp32 down", "prompt bf16 down" }) |b, w, what| try check.sameBytes(what, b, w);
        check.pass("bf16 experts: {d} rows x {d} pairs ({s}) up within {e}, fp32 down within {e} of the float64 reference", .{ rows, slots, if (prompt) "prompt, items of 64" else "decode, items of 16", worst, worst_y });
    }
    // each row alone gives the bytes it gave beside the others
    for ([_]usize{ 0, 17, rows - 1 }) |row| {
        const run = try oneCase(gpu, router, g, s, layer, dx.ptr + row * D * 2, picks[row * slots ..][0..slots], 1, slots, false);
        defer for (run) |b| a.free(b);
        for (run, whole, [_]usize{ W * 2, D * 4, D * 2 }, [_][]const u8{ "row up", "row fp32 down", "row bf16 down" }) |b, w, per, what| try check.sameBytes(what, b, w[row * slots * per ..][0 .. slots * per]);
    }
    check.pass("bf16 experts: a row alone, beside 36 others and in the prompt form gives the same bytes", .{});
}

/// Plan, up and both downs over `rows` rows; returns act, fp32 y and bf16 y.
fn oneCase(gpu: Gpu, router: grouped.Router, g: ex_.Grouped, s: cuda.Stream, l: ex_.Layer, x: u64, picks: []const i32, rows: usize, slots: usize, prompt: bool) ![3][]u8 {
    const pairs = rows * slots;
    const tile: usize = if (prompt) ex_.promptTile(.bf16) else 16;
    var dpicks = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(picks));
    defer dpicks.free();
    const bound = grouped.maxItems(pairs, l.experts, tile);
    var bufs: [5]cuda.DeviceBuffer = undefined;
    const big = pairs > grouped.small;
    const sizes = [5]usize{ pairs * 4, bound * 12, 8, (if (big) pairs else 1) * 4, (if (big) (pairs + 1023) / 1024 * l.experts else 1) * 4 };
    var made: usize = 0;
    defer for (bufs[0..made]) |*b| b.free();
    for (sizes, &bufs) |n, *b| {
        b.* = try cuda.DeviceBuffer.alloc(gpu.d, n);
        made += 1;
    }
    const plan: grouped.Plan = .{ .members = bufs[0].ptr, .items = bufs[1].ptr, .counts = bufs[2].ptr, .rank = bufs[3].ptr, .hist = bufs[4].ptr };
    try router.route(s, dpicks.ptr, pairs, l.experts, tile, plan);
    var act = try cuda.DeviceBuffer.alloc(gpu.d, pairs * l.width * 2);
    defer act.free();
    var y32 = try cuda.DeviceBuffer.alloc(gpu.d, pairs * l.dims * 4);
    defer y32.free();
    var y16 = try cuda.DeviceBuffer.alloc(gpu.d, pairs * l.dims * 2);
    defer y16.free();
    for ([_]*cuda.DeviceBuffer{ &act, &y32, &y16 }) |b| try b.fill8(0xff, s.handle); // a call that writes nothing fails
    const call = if (prompt) &ex_.Grouped.prompt else &ex_.Grouped.decode;
    try call(g, s, .up, .{ .x = x, .stride = l.dims, .slots = slots }, l, plan, act.ptr, bound, -1);
    try call(g, s, .down_f32, .{ .x = act.ptr, .stride = l.width }, l, plan, y32.ptr, bound, -1);
    try call(g, s, .down_bf16, .{ .x = act.ptr, .stride = l.width }, l, plan, y16.ptr, bound, -1);
    try s.synchronize();
    var out: [3][]u8 = undefined;
    out[0] = try check.download(gpu, act);
    errdefer gpu.gpa.free(out[0]);
    out[1] = try check.download(gpu, y32);
    errdefer gpu.gpa.free(out[1]);
    out[2] = try check.download(gpu, y16);
    return out;
}
