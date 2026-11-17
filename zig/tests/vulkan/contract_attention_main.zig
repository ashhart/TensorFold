// The attention gate checks rope_bf16 and sdpa_rows_bf16(+_long): decode=1, prefill<=64, keys=1..8192; exits 0/1/2/124.
const std = @import("std");
const vk = @import("vulkan");
const shaders = @import("stage3_shaders");
const ops_shaders = @import("stage1_shaders");
const fx = @import("stage3_fixture.zig");
const ca = @import("contract_attention.zig");
const contract = @import("contract");
const cx = contract.spec;

const usage = "usage: tf-attention-contract [--vendor ID] [--timeout-ms MS] | --help\n" ++
    "  --help          show this text and exit without opening a loader or device\n" ++
    "  --vendor ID     refuse a device whose Vulkan vendorID differs (decimal or 0x hex; no default)\n" ++
    "  --timeout-ms MS bound on every fence wait, 1 to 600000 (default 10000); a timeout ends the process\n";

const Elapsed = struct {
    io: std.Io,
    start: std.Io.Timestamp,

    fn ms(e: Elapsed) i64 {
        return e.start.durationTo(std.Io.Clock.awake.now(e.io)).toMilliseconds();
    }
};

const repeats = 3;
// Large FINITE bf16 planted over non-visible keys: an uncausal read turns the output huge and fails every check.
const finite_poison: u16 = 0x7f7f;
/// 1 MiB: the 8192-key cache buffers (8.4 MB each) upload in few chunks.
const staging_bytes = 1 << 20;

pub fn main(init: std.process.Init) !u8 {
    const clock: Elapsed = .{ .io = init.io, .start = std.Io.Clock.awake.now(init.io) };
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    for (args[1..]) |arg| if (std.mem.eql(u8, arg, "--help")) {
        std.debug.print(usage, .{});
        return 0;
    };
    var vendor: ?u32 = null;
    var timeout_ms: u64 = 10_000;
    var i: usize = 1;
    while (i < args.len) : (i += 2) {
        const value = if (i + 1 < args.len) args[i + 1] else "";
        if (std.mem.eql(u8, args[i], "--vendor")) {
            vendor = std.fmt.parseInt(u32, value, 0) catch null;
            if (vendor == null) break;
        } else if (std.mem.eql(u8, args[i], "--timeout-ms")) {
            timeout_ms = std.fmt.parseInt(u64, value, 10) catch 0;
            if (timeout_ms == 0 or timeout_ms > 600_000) break;
        } else break;
    } else return if (try probe(init.gpa, clock, vendor, timeout_ms * std.time.ns_per_ms)) 0 else 1;
    std.debug.print(usage, .{});
    return 2;
}

fn probe(a: std.mem.Allocator, clock: Elapsed, vendor: ?u32, timeout: u64) !bool {
    var rt = try vk.Runtime.open();
    defer rt.close();
    var dev = try vk.Device.open(&rt, .{ .features = .{ .storage16 = true, .int16 = true } });
    defer dev.close();
    const pr = dev.properties;
    const name = std.mem.sliceTo(&pr.deviceName, 0);
    const v = pr.apiVersion;
    std.debug.print("MILESTONE device-found elapsed_ms={d} name=\"{s}\" vendor=0x{x:0>4} device=0x{x:0>4} " ++
        "api={d}.{d}.{d} storage16={} int16={} shared={d} fence-timeout={d}ms\n", .{
        clock.ms(),         name,                                 pr.vendorID,                  pr.deviceID,
        v >> 22 & 0x7f,     v >> 12 & 0x3ff,                      v & 0xfff,                    dev.features.storage16,
        dev.features.int16, pr.limits.maxComputeSharedMemorySize, timeout / std.time.ns_per_ms,
    });
    if (pr.deviceType == vk.c.VK_PHYSICAL_DEVICE_TYPE_CPU) {
        std.debug.print("FAIL device: a CPU (software) Vulkan device is never a GPU result\n", .{});
        return false;
    }
    if (vendor) |want| if (pr.vendorID != want) {
        std.debug.print("FAIL device: vendor 0x{x:0>4} is not the requested 0x{x:0>4}\n", .{ pr.vendorID, want });
        return false;
    };

    var cmds = try vk.Commands.init(&dev);
    defer cmds.deinit();
    var staging = try vk.Buffer.init(&dev, .staging, staging_bytes);
    defer staging.deinit();
    var ops = try vk.ops.Ops.init(&dev, .{
        .qmm_bf16 = &ops_shaders.qmm_bf16,
        .qmm_q4_vec4 = &ops_shaders.qmm_q4_vec4,
        .qmm_f32 = &ops_shaders.qmm_f32,
        .copy_f32_bf16 = &ops_shaders.copy_f32_bf16,
        .qmm_q4_tile = &ops_shaders.qmm_q4_tile,
        .qmm_q4_tile_rows = &ops_shaders.qmm_q4_tile_rows,
        .qmm_q4_tile_f32 = &ops_shaders.qmm_q4_tile_f32,
        .dequant_bf16 = &ops_shaders.dequant_bf16,
        .gather_u32 = &ops_shaders.gather_u32,
        .gather_u16 = &ops_shaders.gather_u16,
        .cast_bf16_f32 = &ops_shaders.cast_bf16_f32,
        .cast_f32_bf16 = &ops_shaders.cast_f32_bf16,
        .copy_u32 = &ops_shaders.copy_u32,
        .copy_u16 = &ops_shaders.copy_u16,
    });
    defer ops.deinit();
    var attn = try vk.Attention.init(&dev, .{
        .rope_bf16 = &shaders.rope_bf16,
        .sdpa_rows_bf16 = &shaders.sdpa_rows_bf16,
        .sdpa_rows_bf16_long = &shaders.sdpa_rows_bf16_long,
        .gate_bf16 = &shaders.gate_bf16,
        .rope_f32 = &shaders.rope_f32,
        .rope_f32_kcache = &shaders.rope_f32_kcache,
        .sdpa_rows_f32q = &shaders.sdpa_rows_f32q,
        .sdpa_rows_f32q_long = &shaders.sdpa_rows_f32q_long,
        .gate_f32 = &shaders.gate_f32,
        .sdpa_scores_tile_bf16 = &shaders.sdpa_scores_tile_bf16,
        .sdpa_scores_tile_f32q = &shaders.sdpa_scores_tile_f32q,
        .sdpa_softmax_rows = &shaders.sdpa_softmax_rows,
        .sdpa_pv_tile_bf16 = &shaders.sdpa_pv_tile_bf16,
        .sdpa_pv_tile_f32 = &shaders.sdpa_pv_tile_f32,
    }, &ops);
    defer attn.deinit();

    var passed: usize = 0;
    var total: usize = 0;
    for (fx.rope_cases) |case| {
        total += 1;
        passed += @intFromBool(try ropeGate(a, clock, &dev, &cmds, staging, &attn, case, timeout));
    }
    for (ca.served_cases, 0..) |case, i| {
        total += 1;
        passed += @intFromBool(try sdpaGate(a, clock, &dev, &cmds, staging, &attn, case, 0x5dcafe + @as(u64, i), timeout));
    }
    for (ca.tile_cases) |case| {
        total += 1;
        const bf16_ok = try sdpaTileGate(a, clock, &dev, &cmds, staging, &attn, case, 0x71e5, timeout, false);
        const f32_ok = if (bf16_ok) try sdpaTileGate(a, clock, &dev, &cmds, staging, &attn, case, 0x71e5, timeout, true) else false;
        passed += @intFromBool(bf16_ok and f32_ok);
    }
    for (ca.route_keys) |keys| {
        for ([_]bool{ false, true }) |uniform| {
            total += 1;
            const bf16_ok = try routeGate(a, &dev, &cmds, staging, &attn, keys, 0x70e7, timeout, false, uniform);
            const f32_ok = try routeGate(a, &dev, &cmds, staging, &attn, keys, 0x70e7, timeout, true, uniform);
            passed += @intFromBool(bf16_ok and f32_ok);
        }
    }
    for (ca.probe_keys) |keys| {
        for ([_]bool{ false, true }) |uniform| {
            total += 1;
            const bf16_ok = try probsGate(a, &dev, &cmds, staging, &attn, keys, 0x70e7, timeout, false, uniform);
            const f32_ok = try probsGate(a, &dev, &cmds, staging, &attn, keys, 0x70e7, timeout, true, uniform);
            passed += @intFromBool(bf16_ok and f32_ok);
        }
    }
    std.debug.print("RESULT attention-contract elapsed_ms={d} on \"{s}\": passed={d} total={d}\n", .{ clock.ms(), name, passed, total });
    return passed == total;
}

// One row decoded over `keys` keys by the single dispatch and by the tiled path: every output element must match byte for byte.
// uniform data is non-dyadic, so a different f32 summation order shows; dyadic data sums exactly and hides it.
fn routeGate(a: std.mem.Allocator, dev: *const vk.Device, cmds: *vk.Commands, staging: vk.Buffer, attn: *vk.Attention, keys: u64, seed: u64, timeout: u64, comptime wide: bool, uniform: bool) !bool {
    const El = if (wide) u32 else u16;
    const arm: []const u8 = if (wide) "f32" else "bf16";
    const row_elems: usize = @intCast(fx.heads * fx.head_dim);
    const cache_elems: usize = @intCast(fx.kv_heads * keys * fx.head_dim);
    const q_bits = try a.alloc(u16, row_elems);
    defer a.free(q_bits);
    const k_bits = try a.alloc(u16, cache_elems);
    defer a.free(k_bits);
    const v_bits = try a.alloc(u16, cache_elems);
    defer a.free(v_bits);
    var prng = std.Random.DefaultPrng.init(seed ^ keys);
    const rng = prng.random();
    for ([_][]u16{ q_bits, k_bits, v_bits }) |bits| {
        for (bits) |*x| x.* = if (uniform) cx.toBf16(rng.float(f32) * 2 - 1) else fx.sampleDyadic(rng, -5, 3);
    }
    var kbuf = try upload16(dev, cmds, staging, k_bits, timeout);
    defer kbuf.deinit();
    var vbuf = try upload16(dev, cmds, staging, v_bits, timeout);
    defer vbuf.deinit();
    var q = if (wide) wq: {
        const words = try a.alloc(u32, row_elems);
        defer a.free(words);
        for (q_bits, 0..) |bits, i| words[i] = @bitCast(@as(f32, @floatCast(cx.fromBf16(bits))));
        break :wq try upload32(dev, cmds, staging, words, timeout);
    } else try upload16(dev, cmds, staging, q_bits, timeout);
    defer q.deinit();
    const kind: vk.tensor.Kind = if (wide) .f32 else .bf16;
    const q_view = try vk.tensor.View.init(&q, kind, 0, @intCast(row_elems));
    const keys_view = try vk.tensor.View.init(&kbuf, .bf16, 0, @intCast(k_bits.len));
    const values_view = try vk.tensor.View.init(&vbuf, .bf16, 0, @intCast(v_bits.len));
    var got: [2][]El = undefined;
    for ([_]vk.Attention.Path{ .single, .tiled }, 0..) |path, i| {
        got[i] = try a.alloc(El, row_elems);
        const shell = try a.alloc(El, row_elems + fx.tail);
        defer a.free(shell);
        @memset(shell, if (wide) 0x7fc00000 else fx.poison16);
        var out = try uploadWords(El, dev, cmds, staging, shell, timeout);
        defer out.deinit();
        const y_view = try vk.tensor.View.init(&out, kind, 0, @intCast(row_elems));
        try vk.orExit(attn.attendOn(path, cmds, y_view, q_view, keys_view, values_view, null, keys - 1, 1, keys, timeout));
        try fetch(cmds, staging, out, std.mem.sliceAsBytes(got[i]), timeout);
    }
    defer a.free(got[0]);
    defer a.free(got[1]);
    var mismatched: usize = 0;
    var first: usize = 0;
    var max_abs: f64 = 0;
    for (got[0], got[1], 0..) |s, t, i| {
        if (s == t) continue;
        if (mismatched == 0) first = i;
        mismatched += 1;
        const sv: f64 = if (wide) @as(f32, @bitCast(s)) else cx.fromBf16(s);
        const tv: f64 = if (wide) @as(f32, @bitCast(t)) else cx.fromBf16(t);
        max_abs = @max(max_abs, @abs(sv - tv));
    }
    const ok = mismatched == 0;
    std.debug.print("{s} route keys={d} {s} data={s}: single vs tiled mismatched={d} of {d} first={d} max_abs={e}\n", .{
        if (ok) "MILESTONE" else "FAIL", keys, arm, if (uniform) "uniform" else "dyadic", mismatched, row_elems, first, max_abs,
    });
    return ok;
}

// Split a partial-tile divergence: run the single dispatch through the long (stream) image and the tiled path at key counts past the
// shared stream, then byte-compare the probability arrays themselves before the outputs. Tiled probs live in the score scratch; single
// probs live in the f32 stream. A prob mismatch convicts the score/softmax stages; equal probs with differing outputs convict PV.
fn probsGate(a: std.mem.Allocator, dev: *const vk.Device, cmds: *vk.Commands, staging: vk.Buffer, attn: *vk.Attention, keys: u64, seed: u64, timeout: u64, comptime wide: bool, uniform: bool) !bool {
    const El = if (wide) u32 else u16;
    const arm: []const u8 = if (wide) "f32" else "bf16";
    const row_elems: usize = @intCast(fx.heads * fx.head_dim);
    const cache_elems: usize = @intCast(fx.kv_heads * keys * fx.head_dim);
    const stream_words: usize = @intCast(fx.heads * keys);
    const q_bits = try a.alloc(u16, row_elems);
    defer a.free(q_bits);
    const k_bits = try a.alloc(u16, cache_elems);
    defer a.free(k_bits);
    const v_bits = try a.alloc(u16, cache_elems);
    defer a.free(v_bits);
    var prng = std.Random.DefaultPrng.init(seed ^ keys);
    const rng = prng.random();
    for ([_][]u16{ q_bits, k_bits, v_bits }) |bits| {
        for (bits) |*x| x.* = if (uniform) cx.toBf16(rng.float(f32) * 2 - 1) else fx.sampleDyadic(rng, -5, 3);
    }
    var kbuf = try upload16(dev, cmds, staging, k_bits, timeout);
    defer kbuf.deinit();
    var vbuf = try upload16(dev, cmds, staging, v_bits, timeout);
    defer vbuf.deinit();
    var q = if (wide) wq: {
        const words = try a.alloc(u32, row_elems);
        defer a.free(words);
        for (q_bits, 0..) |bits, i| words[i] = @bitCast(@as(f32, @floatCast(cx.fromBf16(bits))));
        break :wq try upload32(dev, cmds, staging, words, timeout);
    } else try upload16(dev, cmds, staging, q_bits, timeout);
    defer q.deinit();
    const zeros = try a.alloc(u32, stream_words);
    defer a.free(zeros);
    @memset(zeros, 0);
    var stream_buf = try upload32(dev, cmds, staging, zeros, timeout);
    defer stream_buf.deinit();
    const kind: vk.tensor.Kind = if (wide) .f32 else .bf16;
    const q_view = try vk.tensor.View.init(&q, kind, 0, @intCast(row_elems));
    const keys_view = try vk.tensor.View.init(&kbuf, .bf16, 0, @intCast(k_bits.len));
    const values_view = try vk.tensor.View.init(&vbuf, .bf16, 0, @intCast(v_bits.len));
    const stream_view = try vk.tensor.View.init(&stream_buf, .f32, 0, @intCast(stream_words));
    const head_stride_words: u64 = vk.Attention.tile_rows * ca.max_visible_keys;
    var got: [2][]El = undefined;
    defer a.free(got[0]);
    defer a.free(got[1]);
    var probs: [2][]u32 = undefined;
    defer a.free(probs[0]);
    defer a.free(probs[1]);
    for ([_]vk.Attention.Path{ .single, .tiled }, 0..) |path, i| {
        got[i] = try a.alloc(El, row_elems);
        const shell = try a.alloc(El, row_elems + fx.tail);
        defer a.free(shell);
        @memset(shell, if (wide) 0x7fc00000 else fx.poison16);
        var out = try uploadWords(El, dev, cmds, staging, shell, timeout);
        defer out.deinit();
        const y_view = try vk.tensor.View.init(&out, kind, 0, @intCast(row_elems));
        try vk.orExit(attn.attendOn(path, cmds, y_view, q_view, keys_view, values_view, if (i == 0) stream_view else null, keys - 1, 1, keys, timeout));
        try fetch(cmds, staging, out, std.mem.sliceAsBytes(got[i]), timeout);
        if (i == 0) {
            probs[i] = try a.alloc(u32, stream_words);
            try fetch(cmds, staging, stream_buf, std.mem.sliceAsBytes(probs[i]), timeout);
        } else {
            // The tiled probs live in per-head strips at h * tile_rows * max_visible_keys; fetch each strip through the 1 MiB
            // staging buffer separately instead of one oversized download that would corrupt the heap past staging.
            probs[i] = try a.alloc(u32, @intCast(fx.heads * ca.max_visible_keys));
            for (0..@intCast(fx.heads)) |h| {
                try vk.orExit(vk.download(cmds, attn.score_scratch, @intCast(h * head_stride_words * 4), staging, std.mem.sliceAsBytes(probs[i][h * @as(usize, @intCast(ca.max_visible_keys)) .. (h + 1) * @as(usize, @intCast(ca.max_visible_keys))]), timeout));
            }
        }
    }
    var pmis: usize = 0;
    var pfirst_h: usize = 0;
    var pfirst_k: usize = 0;
    var pmag: f64 = 0;
    const nk: usize = @intCast(keys);
    const head_stride: usize = @intCast(ca.max_visible_keys);
    for (0..@intCast(fx.heads)) |h| {
        for (0..nk) |k| {
            const s = probs[0][h * nk + k];
            const t = probs[1][h * head_stride + k];
            if (s == t) continue;
            if (pmis == 0) {
                pfirst_h = h;
                pfirst_k = k;
            }
            pmis += 1;
            pmag = @max(pmag, @abs(@as(f64, @as(f32, @bitCast(s))) - @as(f64, @as(f32, @bitCast(t)))));
        }
    }
    var omis: usize = 0;
    var ofirst: usize = 0;
    var omag: f64 = 0;
    for (got[0], got[1], 0..) |s, t, i| {
        if (s == t) continue;
        if (omis == 0) ofirst = i;
        omis += 1;
        const sv: f64 = if (wide) @as(f32, @bitCast(s)) else cx.fromBf16(s);
        const tv: f64 = if (wide) @as(f32, @bitCast(t)) else cx.fromBf16(t);
        omag = @max(omag, @abs(sv - tv));
    }
    const ok = pmis == 0 and omis == 0;
    std.debug.print("{s} probs keys={d} {s} data={s}: prob_mismatched={d} of {d} first=(head {d}, key {d}) prob_max_abs={e} | out_mismatched={d} of {d} first={d} max_abs={e}\n", .{
        if (ok) "MILESTONE" else "FAIL", keys, arm, if (uniform) "uniform" else "dyadic", pmis, @as(usize, @intCast(fx.heads * keys)), pfirst_h, pfirst_k, pmag, omis, row_elems, ofirst, omag,
    });
    return ok;
}

fn upload16(dev: *const vk.Device, cmds: *vk.Commands, staging: vk.Buffer, shell: []const u16, timeout: u64) !vk.Buffer {
    var b = try vk.Buffer.init(dev, .device, std.mem.sliceAsBytes(shell).len);
    errdefer b.deinit();
    try vk.orExit(vk.upload(cmds, b, 0, staging, std.mem.sliceAsBytes(shell), timeout));
    return b;
}

// The long arm's f32 score stream: zero-filled, the kernel writes every word before reading it.
fn upload32(dev: *const vk.Device, cmds: *vk.Commands, staging: vk.Buffer, shell: []const u32, timeout: u64) !vk.Buffer {
    var b = try vk.Buffer.init(dev, .device, std.mem.sliceAsBytes(shell).len);
    errdefer b.deinit();
    try vk.orExit(vk.upload(cmds, b, 0, staging, std.mem.sliceAsBytes(shell), timeout));
    return b;
}

fn uploadWords(comptime El: type, dev: *const vk.Device, cmds: *vk.Commands, staging: vk.Buffer, shell: []const El, timeout: u64) !vk.Buffer {
    return if (El == u16) upload16(dev, cmds, staging, shell, timeout) else upload32(dev, cmds, staging, shell, timeout);
}

fn fetch(cmds: *vk.Commands, staging: vk.Buffer, b: vk.Buffer, out: []u8, timeout: u64) !void {
    try vk.orExit(vk.download(cmds, b, 0, staging, out, timeout));
}

fn unchanged(a: std.mem.Allocator, cmds: *vk.Commands, staging: vk.Buffer, shell: []const u16, buf: vk.Buffer, timeout: u64) !bool {
    const back = try a.alloc(u16, shell.len);
    defer a.free(back);
    try fetch(cmds, staging, buf, std.mem.sliceAsBytes(back), timeout);
    return std.mem.indexOfDiff(u16, shell, back) == null;
}

// Every plant must fail: downloaded GOOD output's first finite-want element gets wrong finite, then nan/+inf/-inf.
fn plantRefuse(f: anytype, good: []const u16, name: []const u8, gate: []const u8) !bool {
    const mutated = try f.a.dupe(u16, good);
    defer f.a.free(mutated);
    var first_finite: ?usize = null;
    for (good, 0..) |bits, i| {
        if (std.math.isFinite(cx.fromBf16(bits))) {
            first_finite = i;
            break;
        }
    }
    const at = first_finite orelse return error.NoFiniteReference;
    mutated[at] = 0x4f00; // 2^31: an absolutely wrong finite the bounds must refuse
    if (f.check(mutated)) |_| {
        std.debug.print("FAIL {s} case=\"{s}\": a planted wrong finite output passed\n", .{ gate, name });
        return false;
    } else |_| {}
    for ([_]u16{ 0x7fc0, 0x7f80, 0xff80 }, [_][]const u8{ "nan", "+inf", "-inf" }) |pb, pname| {
        mutated[at] = pb;
        if (f.check(mutated)) |_| {
            std.debug.print("FAIL {s} case=\"{s}\": a planted {s} output passed\n", .{ gate, name, pname });
            return false;
        } else |_| {}
    }
    return true;
}

// Rope gates dispatch three byte-identical repeats, check stage3_fixture f64, refuse plants, and leave input unchanged.
fn ropeGate(a: std.mem.Allocator, clock: Elapsed, dev: *const vk.Device, cmds: *vk.Commands, staging: vk.Buffer, attn: *vk.Attention, case: fx.RopeCase, timeout: u64) !bool {
    const f = try fx.Rope.init(a, case, 0x57a3);
    defer f.deinit();
    var q = try upload16(dev, cmds, staging, f.q_bits, timeout);
    defer q.deinit();
    // Poison every output word so missing writes cannot inherit a reference.
    const out_shell = try a.alloc(u16, f.want.len + fx.tail);
    defer a.free(out_shell);
    @memset(out_shell, fx.poison16);
    var out = try upload16(dev, cmds, staging, out_shell, timeout);
    defer out.deinit();
    const out_view = try vk.tensor.View.init(&out, .bf16, 0, f.want.len);
    const q_view = try vk.tensor.View.init(&q, .bf16, 0, f.want.len);
    try vk.orExit(attn.setBasePosition(cmds, staging, case.base, timeout));
    const rows: u64 = case.rows;
    var first: ?[]u16 = null;
    defer if (first) |x| a.free(x);
    const full_bits = try a.alloc(u16, f.want.len + fx.tail);
    defer a.free(full_bits);
    const bits = full_bits[0..f.want.len];
    for (0..repeats) |rep| {
        // re-upload the all-poison shell before every dispatch: the kernel must write every output word itself.
        try vk.orExit(vk.upload(cmds, out, 0, staging, std.mem.sliceAsBytes(out_shell), timeout));
        try vk.orExit(attn.ropeQueries(cmds, out_view, q_view, rows, timeout));
        try fetch(cmds, staging, out, std.mem.sliceAsBytes(full_bits), timeout);
        for (full_bits[f.want.len..]) |g_elem| {
            if (g_elem != fx.poison16) {
                std.debug.print("FAIL rope case=\"{s}\": output tail overwritten at element {d}\n", .{ case.name, f.want.len + (std.mem.indexOf(u16, full_bits[f.want.len..], &[_]u16{g_elem}) orelse 0) });
                return false;
            }
        }
        if (first) |x| {
            if (std.mem.indexOfDiff(u16, x, bits)) |at| {
                std.debug.print("FAIL rope case=\"{s}\": repeat {d} differs at element {d}\n", .{ case.name, rep, at });
                return false;
            }
        } else first = try a.dupe(u16, bits);
    }
    const rep = f.check(bits) catch |e| {
        std.debug.print("FAIL rope case=\"{s}\": {s}\n", .{ case.name, @errorName(e) });
        return false;
    };
    if (!try plantRefuse(f, bits, case.name, "rope")) return false;
    if (!try unchanged(a, cmds, staging, f.q_bits, q, timeout)) {
        std.debug.print("FAIL rope case=\"{s}\": the input changed\n", .{case.name});
        return false;
    }
    std.debug.print("MILESTONE rope elapsed_ms={d} case=\"{s}\" rows={d} base={d} worst={e} of bound, repeats byte-identical, " ++
        "passthrough tail bit-exact, input unchanged, planted outputs refused\n", .{ clock.ms(), case.name, rows, case.base, rep.worst });
    return true;
}

// SDPA: poison shell, 3 identical repeats, uploaded-bit f64 bounds, single-key exactness, plants refused, unchanged
fn sdpaGate(a: std.mem.Allocator, clock: Elapsed, dev: *const vk.Device, cmds: *vk.Commands, staging: vk.Buffer, attn: *vk.Attention, case: ca.SdpaCase, seed: u64, timeout: u64) !bool {
    const row_elems: u64 = fx.heads * fx.head_dim;
    const f = try ca.Sdpa.init(a, case, seed);
    defer f.deinit();
    var q = try upload16(dev, cmds, staging, f.q_bits, timeout);
    defer q.deinit();
    var keys = try upload16(dev, cmds, staging, f.k_bits, timeout);
    defer keys.deinit();
    var values = try upload16(dev, cmds, staging, f.v_bits, timeout);
    defer values.deinit();
    // Long cases stage scores in an f32 stream: 8 heads x rows x total keys words, zero-filled once.
    const total_keys = ca.capacityOf(case);
    var stream_buf: ?vk.Buffer = null;
    defer if (stream_buf) |*b| b.deinit();
    const stream_words: u64 = fx.heads * case.rows * total_keys;
    if (total_keys > ca.shared_stream_keys) {
        const zeros = try a.alloc(u32, @intCast(stream_words));
        defer a.free(zeros);
        @memset(zeros, 0);
        stream_buf = try upload32(dev, cmds, staging, zeros, timeout);
    }
    const stream_view: ?vk.tensor.View = if (stream_buf) |*b| try vk.tensor.View.init(b, .f32, 0, stream_words) else null;
    // Poison every output word so missing writes cannot inherit a reference.
    const out_len: usize = @intCast(case.rows * row_elems);
    const out_shell = try a.alloc(u16, out_len + fx.tail);
    defer a.free(out_shell);
    @memset(out_shell, fx.poison16);
    var out = try upload16(dev, cmds, staging, out_shell, timeout);
    defer out.deinit();
    const q_view = try vk.tensor.View.init(&q, .bf16, 0, out_len);
    const keys_view = try vk.tensor.View.init(&keys, .bf16, 0, @intCast(f.k_bits.len));
    const values_view = try vk.tensor.View.init(&values, .bf16, 0, @intCast(f.v_bits.len));
    const y_view = try vk.tensor.View.init(&out, .bf16, 0, out_len);
    var first: ?[]u16 = null;
    defer if (first) |x| a.free(x);
    const full_bits = try a.alloc(u16, out_len + fx.tail);
    defer a.free(full_bits);
    const bits = full_bits[0..out_len];
    for (0..repeats) |rep| {
        try vk.orExit(vk.upload(cmds, out, 0, staging, std.mem.sliceAsBytes(out_shell), timeout));
        try vk.orExit(attn.attend(cmds, y_view, q_view, keys_view, values_view, stream_view, case.prefix, case.rows, f.capacity, timeout));
        try fetch(cmds, staging, out, std.mem.sliceAsBytes(full_bits), timeout);
        for (full_bits[out_len..]) |g_elem| {
            if (g_elem != fx.poison16) {
                std.debug.print("FAIL sdpa case=\"{s}\": output tail overwritten at element {d}\n", .{ case.name, out_len + (std.mem.indexOf(u16, full_bits[out_len..], &[_]u16{g_elem}) orelse 0) });
                return false;
            }
        }
        if (first) |x| {
            if (std.mem.indexOfDiff(u16, x, bits)) |at| {
                std.debug.print("FAIL sdpa case=\"{s}\": repeat {d} differs at element {d}\n", .{ case.name, rep, at });
                return false;
            }
        } else first = try a.dupe(u16, bits);
    }
    const rep = f.check(bits) catch |e| {
        std.debug.print("FAIL sdpa case=\"{s}\": {s}\n", .{ case.name, @errorName(e) });
        return false;
    };
    if (!try plantRefuse(f, bits, case.name, "sdpa")) return false;
    if (!try unchanged(a, cmds, staging, f.q_bits, q, timeout) or
        !try unchanged(a, cmds, staging, f.k_bits, keys, timeout) or
        !try unchanged(a, cmds, staging, f.v_bits, values, timeout))
    {
        std.debug.print("FAIL sdpa case=\"{s}\": an input changed\n", .{case.name});
        return false;
    }
    // Each row decoded alone at q_len 1 and prefix + r must bit-match its multi-row result (decode-rows-1 shape).
    if (case.rows > 1) {
        const one_len: usize = @intCast(row_elems);
        const one_shell = try a.alloc(u16, one_len + fx.tail);
        defer a.free(one_shell);
        @memset(one_shell, fx.poison16);
        var one = try upload16(dev, cmds, staging, one_shell, timeout);
        defer one.deinit();
        const one_view = try vk.tensor.View.init(&one, .bf16, 0, one_len);
        const one_bits = try a.alloc(u16, one_len);
        defer a.free(one_bits);
        for (0..@intCast(case.rows)) |r| {
            const q_row = try q_view.slice(@intCast(r * row_elems), @intCast(row_elems));
            // A row decoded alone gets its own total; long totals slice the front of the same stream buffer.
            const row_total = case.prefix + @as(u64, r) + 1;
            const row_stream: ?vk.tensor.View = if (row_total > ca.shared_stream_keys) try stream_view.?.slice(0, fx.heads * row_total) else null;
            try vk.orExit(vk.upload(cmds, one, 0, staging, std.mem.sliceAsBytes(one_shell), timeout));
            try vk.orExit(attn.attend(cmds, one_view, q_row, keys_view, values_view, row_stream, case.prefix + @as(u64, r), 1, f.capacity, timeout));
            try fetch(cmds, staging, one, std.mem.sliceAsBytes(one_bits), timeout);
            if (std.mem.indexOfDiff(u16, one_bits, bits[@intCast(r * row_elems)..][0..one_len])) |at| {
                std.debug.print("FAIL sdpa case=\"{s}\": row {d} decoded alone differs at element {d}\n", .{ case.name, r, at });
                return false;
            }
        }
    }
    std.debug.print("MILESTONE sdpa elapsed_ms={d} case=\"{s}\" rows={d} prefix={d} visible=[{d}..{d}] worst={e} of bound, " ++
        "repeats byte-identical, {s}planted outputs refused, inputs unchanged\n", .{
        clock.ms(),                                                     case.name, case.rows, case.prefix, f.visible[0], f.visible[f.visible.len - 1], rep.worst,
        if (case.rows > 1) "rows decode alone bit-identical, " else "",
    });
    return true;
}

// Tiled prefill with non-finite V planted past a row's bound: the tile must skip future keys and byte-match decode.
fn sdpaTileGate(a: std.mem.Allocator, clock: Elapsed, dev: *const vk.Device, cmds: *vk.Commands, staging: vk.Buffer, attn: *vk.Attention, case: ca.TileCase, seed: u64, timeout: u64, comptime wide: bool) !bool {
    const El = if (wide) u32 else u16;
    const arm: []const u8 = if (wide) "f32" else "bf16";
    const row_elems: u64 = fx.heads * fx.head_dim;
    const q_len: usize = @intCast(case.rows * row_elems);
    const cache_elems: usize = @intCast(fx.kv_heads * case.capacity * fx.head_dim);
    const q_bits = try a.alloc(u16, q_len);
    defer a.free(q_bits);
    const k_bits = try a.alloc(u16, cache_elems);
    defer a.free(k_bits);
    const v_bits = try a.alloc(u16, cache_elems);
    defer a.free(v_bits);
    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();
    for (q_bits) |*x| x.* = fx.sampleDyadic(rng, -5, 3);
    for (k_bits) |*x| x.* = fx.sampleDyadic(rng, -5, 3);
    for (v_bits) |*x| x.* = fx.sampleDyadic(rng, -5, 3);
    const poison = [_]u16{ 0x7fc0, 0x7f80, 0xff80 }; // NaN, +inf, -inf
    const cap: usize = @intCast(case.capacity);
    const hd: usize = @intCast(fx.head_dim);
    if (case.zero_qk) {
        @memset(q_bits, 0);
        @memset(k_bits, 0);
    }
    for (0..@as(usize, @intCast(fx.kv_heads))) |kh| {
        for (@as(usize, @intCast(case.poison_from))..cap) |j| {
            const at = (kh * cap + j) * hd;
            @memset(v_bits[at..][0..hd], poison[(j - @as(usize, @intCast(case.poison_from))) % poison.len]);
        }
        if (case.zero_qk) @memset(v_bits[(kh * cap) * hd ..][0..hd], 0x3f80); // V0 = 1.0
    }
    var keys = try upload16(dev, cmds, staging, k_bits, timeout);
    defer keys.deinit();
    var values = try upload16(dev, cmds, staging, v_bits, timeout);
    defer values.deinit();
    // Wide q rides the f32 path: bf16 widens exactly, so both arms see identical queries over the same bf16 cache.
    var q = if (wide) wq: {
        const words = try a.alloc(u32, q_len);
        defer a.free(words);
        for (q_bits, 0..) |bits, i| words[i] = @bitCast(@as(f32, @floatCast(cx.fromBf16(bits))));
        break :wq try upload32(dev, cmds, staging, words, timeout);
    } else try upload16(dev, cmds, staging, q_bits, timeout);
    defer q.deinit();
    const q_view = try vk.tensor.View.init(&q, if (wide) .f32 else .bf16, 0, @intCast(q_len));
    const keys_view = try vk.tensor.View.init(&keys, .bf16, 0, @intCast(k_bits.len));
    const values_view = try vk.tensor.View.init(&values, .bf16, 0, @intCast(v_bits.len));
    const tiled = try a.alloc(El, q_len);
    defer a.free(tiled);
    {
        const shell_poison: El = if (wide) 0x7fc00000 else fx.poison16;
        const shell = try a.alloc(El, q_len + fx.tail);
        defer a.free(shell);
        @memset(shell, shell_poison);
        var out = try uploadWords(El, dev, cmds, staging, shell, timeout);
        defer out.deinit();
        const y_view = try vk.tensor.View.init(&out, if (wide) .f32 else .bf16, 0, @intCast(q_len));
        // rows 2..64: attend() dispatches sdpa_scores_tile + sdpa_softmax_rows + sdpa_pv_tile, the PR's tiled path.
        try vk.orExit(attn.attend(cmds, y_view, q_view, keys_view, values_view, null, case.prefix, case.rows, case.capacity, timeout));
        try fetch(cmds, staging, out, std.mem.sliceAsBytes(tiled), timeout);
    }
    const old = try a.alloc(El, q_len);
    defer a.free(old);
    {
        const shell_poison: El = if (wide) 0x7fc00000 else fx.poison16;
        const shell = try a.alloc(El, @intCast(row_elems + fx.tail));
        defer a.free(shell);
        @memset(shell, shell_poison);
        var one = try uploadWords(El, dev, cmds, staging, shell, timeout);
        defer one.deinit();
        const y_view = try vk.tensor.View.init(&one, if (wide) .f32 else .bf16, 0, @intCast(row_elems));
        for (0..@as(usize, @intCast(case.rows))) |r| {
            const q_row = try q_view.slice(@intCast(r * row_elems), @intCast(row_elems));
            // The old path: the row decoded alone at q_len 1 by sdpa_rows_bf16(+f32q), which the PR replaced.
            try vk.orExit(vk.upload(cmds, one, 0, staging, std.mem.sliceAsBytes(shell), timeout));
            try vk.orExit(attn.attend(cmds, y_view, q_row, keys_view, values_view, null, case.prefix + @as(u64, r), 1, case.capacity, timeout));
            try fetch(cmds, staging, one, std.mem.sliceAsBytes(old[@intCast(r * row_elems)..][0..@intCast(row_elems)]), timeout);
        }
    }
    if (std.mem.indexOfDiff(El, tiled, old)) |at| {
        std.debug.print("FAIL sdpa-tile case=\"{s}\" {s}: tiled output differs from per-row decode at element {d}\n", .{ case.name, arm, at });
        return false;
    }
    const finite = ca.finiteRows(case);
    var plant_rows: usize = 0;
    for (0..@as(usize, @intCast(case.rows))) |r| {
        const row = tiled[@intCast(r * row_elems)..][0..@intCast(row_elems)];
        var nonfinite: usize = 0;
        for (row) |bits| {
            const v: f32 = if (wide) @bitCast(bits) else @floatCast(cx.fromBf16(bits));
            if (!std.math.isFinite(v)) nonfinite += 1;
        }
        if (r < finite) {
            if (nonfinite != 0) {
                std.debug.print("FAIL sdpa-tile case=\"{s}\" {s}: row {d} went non-finite past a bound it never sees\n", .{ case.name, arm, r });
                return false;
            }
        } else {
            if (nonfinite == 0) {
                std.debug.print("FAIL sdpa-tile case=\"{s}\" {s}: row {d} stayed finite though its window holds the plants\n", .{ case.name, arm, r });
                return false;
            }
            plant_rows += 1;
        }
    }
    if (case.zero_qk) {
        const one: El = if (wide) 0x3f800000 else 0x3f80;
        for (tiled[0..@intCast(row_elems)]) |bits| {
            if (bits != one) {
                std.debug.print("FAIL sdpa-tile case=\"{s}\" {s}: row 0 is not the exact 1.0 its single visible key forces\n", .{ case.name, arm });
                return false;
            }
        }
    }
    std.debug.print("MILESTONE sdpa-tile elapsed_ms={d} case=\"{s}\" {s} rows={d} prefix={d} poison_from={d} capacity={d}: " ++
        "tiled byte-identical to per-row decode, {d} finite rows finite, {d} plant rows non-finite{s}\n", .{
        clock.ms(), case.name,  arm,                                             case.rows, case.prefix, case.poison_from, case.capacity,
        finite,     plant_rows, if (case.zero_qk) ", row 0 exactly 1.0" else "",
    });
    return true;
}
