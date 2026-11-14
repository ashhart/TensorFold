// Stage 3 GPU bring-up: RoPE/KV/GQA SDPA/gate; exits 0/1/2/124; no model, perf or cross-backend bitexact claim.
const std = @import("std");
const vk = @import("vulkan");
const shaders = @import("stage3_shaders");
const ops_shaders = @import("stage1_shaders");
const fx = @import("stage3_fixture.zig");
const contract = @import("contract");
const cx = contract.spec;

const usage = "usage: tf-stage3-gpu [--vendor ID] [--timeout-ms MS] | --help\n" ++
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
// Large FINITE bf16 (≈ max) poisons non-visible keys; uncausal reads produce ±inf/huge and fail checks.
const finite_poison: u16 = 0x7f7f;

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
    var staging = try vk.Buffer.init(&dev, .staging, 256 << 10);
    defer staging.deinit();
    var ops = try vk.ops.Ops.init(&dev, .{
        .qmm_bf16 = &ops_shaders.qmm_bf16,
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
        .gate_bf16 = &shaders.gate_bf16,
    }, &ops);
    defer attn.deinit();

    var passed: usize = 0;
    var total: usize = 0;
    for (fx.rope_cases) |case| {
        total += 1;
        passed += @intFromBool(try ropeGate(a, clock, &dev, &cmds, staging, &attn, case, timeout));
    }
    total += 1;
    passed += @intFromBool(try attendGate(a, clock, &dev, &cmds, staging, &attn, timeout));
    total += 1;
    passed += @intFromBool(try gateGate(a, clock, &dev, &cmds, staging, &attn, timeout));
    std.debug.print("RESULT stage3 elapsed_ms={d} on \"{s}\": passed={d} total={d}\n", .{ clock.ms(), name, passed, total });
    return passed == total;
}

fn upload16(dev: *const vk.Device, cmds: *vk.Commands, staging: vk.Buffer, shell: []const u16, timeout: u64) !vk.Buffer {
    var b = try vk.Buffer.init(dev, .device, std.mem.sliceAsBytes(shell).len);
    errdefer b.deinit();
    try vk.orExit(vk.upload(cmds, b, 0, staging, std.mem.sliceAsBytes(shell), timeout));
    return b;
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

// Rope checks dispatch, 3 bit-identical repeats, f64 reference, wrong-finite/nonfinite refusals, and unchanged input.
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
    // readback covers data + tail
    const full_bits = try a.alloc(u16, f.want.len + fx.tail);
    defer a.free(full_bits);
    const bits = full_bits[0..f.want.len];
    for (0..repeats) |rep| {
        // Re-upload poison before each dispatch; unwritten output words stay NaN and fail finite-output checks.
        try vk.orExit(vk.upload(cmds, out, 0, staging, std.mem.sliceAsBytes(out_shell), timeout));
        try vk.orExit(attn.ropeQueries(cmds, out_view, q_view, rows, timeout));
        try fetch(cmds, staging, out, std.mem.sliceAsBytes(full_bits), timeout);
        // poison tail must survive untouched
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

// Poison a downloaded GOOD copy with wrong finite then nan/+inf/-inf at the first finite-want element; refuse all.
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
    mutated[at] = 0x4f00; // 2^31: an absolutely wrong finite the budgets must refuse
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

// CacheKV checks f64 rope/V bits; SDPA checks counts, NaN/finite/zero causality, repeats, refusals, unchanged inputs.
fn attendGate(a: std.mem.Allocator, clock: Elapsed, dev: *const vk.Device, cmds: *vk.Commands, staging: vk.Buffer, attn: *vk.Attention, timeout: u64) !bool {
    const ac = fx.attend_case;
    const cc = fx.cache_case;
    const rows = ac.rows;
    const qrow_elems: u64 = fx.heads * fx.head_dim;
    const crow_elems: u64 = fx.kv_heads * fx.head_dim;
    const cache_elems: u64 = fx.kv_heads * cc.capacity * fx.head_dim;

    const f = try fx.Cache.init(a, cc, 0xcacbe);
    defer f.deinit();
    // rotated-query fixture data (rope of these rows is checked by the rope gates)
    var qf = try fx.Rope.init(a, .{ .name = "attend q", .rows = rows, .base = @intCast(cc.base) }, 0x57a5);
    defer qf.deinit();

    var knorm = try upload16(dev, cmds, staging, f.knorm_bits, timeout);
    defer knorm.deinit();
    var vseg = try upload16(dev, cmds, staging, f.v_bits, timeout);
    defer vseg.deinit();
    var keys = try upload16(dev, cmds, staging, f.keys_shell, timeout);
    defer keys.deinit();
    var values = try upload16(dev, cmds, staging, f.values_shell, timeout);
    defer values.deinit();
    var qraw = try upload16(dev, cmds, staging, qf.q_bits, timeout);
    defer qraw.deinit();
    var rotated = try vk.Buffer.init(dev, .device, std.mem.sliceAsBytes(qf.q_bits).len);
    defer rotated.deinit();
    var ybuf = try vk.Buffer.init(dev, .device, @intCast(rows * qrow_elems * 2));
    defer ybuf.deinit();

    const knorm_view = try vk.tensor.View.init(&knorm, .bf16, 0, rows * crow_elems);
    const vseg_view = try vk.tensor.View.init(&vseg, .bf16, 0, rows * crow_elems);
    const keys_view = try vk.tensor.View.init(&keys, .bf16, 0, cache_elems);
    const values_view = try vk.tensor.View.init(&values, .bf16, 0, cache_elems);
    const qraw_view = try vk.tensor.View.init(&qraw, .bf16, 0, rows * qrow_elems);
    const rotated_view = try vk.tensor.View.init(&rotated, .bf16, 0, rows * qrow_elems);
    const y_view = try vk.tensor.View.init(&ybuf, .bf16, 0, rows * qrow_elems);

    // positions of the segment rows are base..base+rows-1
    try vk.orExit(attn.setBasePosition(cmds, staging, @intCast(cc.base), timeout));
    // initialize y with poison; a causal kernel overwrites every visible row
    const y_init = try a.alloc(u16, @intCast(rows * qrow_elems));
    defer a.free(y_init);
    @memset(y_init, fx.poison16);
    var first: ?[]u16 = null;
    defer if (first) |x| a.free(x);
    const ybits = try a.alloc(u16, @intCast(rows * qrow_elems));
    defer a.free(ybits);

    // --- rotate + write the cache ---
    try vk.orExit(attn.ropeQueries(cmds, rotated_view, qraw_view, rows, timeout));
    try vk.orExit(attn.cacheKV(cmds, keys_view, values_view, knorm_view, vseg_view, rows, cc.base, cc.capacity, timeout));
    // --- cache checks ---
    const keys_back = try a.alloc(u16, @intCast(cache_elems));
    defer a.free(keys_back);
    const values_back = try a.alloc(u16, @intCast(cache_elems));
    defer a.free(values_back);
    try fetch(cmds, staging, keys, std.mem.sliceAsBytes(keys_back), timeout);
    try fetch(cmds, staging, values, std.mem.sliceAsBytes(values_back), timeout);
    const kr = fx.Cache.checkKeys(f, keys_back) catch |e| {
        std.debug.print("FAIL cache case=\"{s}\": keys {s}\n", .{ cc.name, @errorName(e) });
        return false;
    };
    fx.Cache.checkValueBits(f, values_back) catch |e| {
        std.debug.print("FAIL cache case=\"{s}\": values {s}\n", .{ cc.name, @errorName(e) });
        return false;
    };
    if (!try unchanged(a, cmds, staging, f.knorm_bits, knorm, timeout) or !try unchanged(a, cmds, staging, f.v_bits, vseg, timeout)) {
        std.debug.print("FAIL cache case=\"{s}\": a segment input changed\n", .{cc.name});
        return false;
    }
    std.debug.print("MILESTONE cache elapsed_ms={d} case=\"{s}\" base={d} rows={d} capacity={d} keys worst={e} of bound, " ++
        "V transport bit-exact, poison rows untouched, inputs unchanged\n", .{ clock.ms(), cc.name, cc.base, rows, cc.capacity, kr.worst });

    // --- SDPA over the ACTUAL bits the GPU reads (rotated q, written caches) ---
    const rotated_back = try a.alloc(u16, @intCast(rows * qrow_elems));
    defer a.free(rotated_back);
    try fetch(cmds, staging, rotated, std.mem.sliceAsBytes(rotated_back), timeout);
    const g = try fx.Attend.init(a, ac, rotated_back, keys_back, values_back, 0);
    defer g.deinit();

    for (0..repeats) |rep| {
        try vk.orExit(vk.upload(cmds, ybuf, 0, staging, std.mem.sliceAsBytes(y_init), timeout));
        try vk.orExit(attn.attend(cmds, y_view, rotated_view, keys_view, values_view, ac.prefix, rows, cc.capacity, timeout));
        try fetch(cmds, staging, ybuf, std.mem.sliceAsBytes(ybits), timeout);
        if (first) |x| {
            if (std.mem.indexOfDiff(u16, x, ybits)) |at| {
                std.debug.print("FAIL sdpa: repeat {d} differs at element {d}\n", .{ rep, at });
                return false;
            }
        } else first = try a.dupe(u16, ybits);
    }
    const rep = g.check(ybits) catch |e| {
        std.debug.print("FAIL sdpa: {s}\n", .{@errorName(e)});
        return false;
    };

    // Fresh cache per row poisons keys >= prefix+r+1; rows=1/prefix=prefix+r must bit-match baseline without budget.
    const alt_keys = try a.alloc(u16, @intCast(cache_elems));
    defer a.free(alt_keys);
    const y_row_bits = try a.alloc(u16, @intCast(qrow_elems));
    defer a.free(y_row_bits);
    const variants = [_]u16{ fx.poison16, finite_poison, 0x0000 };
    const vnames = [_][]const u8{ "nan", "large-finite", "zero" };
    for (variants, vnames) |fill, vname| {
        for (0..@intCast(rows)) |r| {
            @memcpy(alt_keys, keys_back);
            const visible = fx.visibleFrom(ac, @intCast(r));
            for (0..@intCast(fx.kv_heads)) |kh| {
                for (visible..cc.capacity) |j| {
                    const base = (@as(usize, @intCast(kh)) * @as(usize, @intCast(cc.capacity)) + j) * @as(usize, @intCast(fx.head_dim));
                    @memset(alt_keys[base..][0..@intCast(fx.head_dim)], fill);
                }
            }
            var alt_buf = try upload16(dev, cmds, staging, alt_keys, timeout);
            defer alt_buf.deinit();
            const alt_view = try vk.tensor.View.init(&alt_buf, .bf16, 0, cache_elems);
            // row-slice views: q row r of `rotated`, output row r of `ybuf`
            const row_byte_offset = r * qrow_elems * 2;
            const q_row_view = try vk.tensor.View.init(&rotated, .bf16, row_byte_offset, qrow_elems);
            const y_row_view = try vk.tensor.View.init(&ybuf, .bf16, row_byte_offset, qrow_elems);
            try vk.orExit(vk.upload(cmds, ybuf, 0, staging, std.mem.sliceAsBytes(y_init), timeout));
            try vk.orExit(attn.attend(cmds, y_row_view, q_row_view, alt_view, values_view, ac.prefix + @as(u64, @intCast(r)), 1, cc.capacity, timeout));
            try vk.orExit(vk.download(cmds, ybuf, row_byte_offset, staging, std.mem.sliceAsBytes(y_row_bits), timeout));
            const baseline_row = first.?[r * @as(usize, @intCast(qrow_elems)) ..][0..@intCast(qrow_elems)];
            if (std.mem.indexOfDiff(u16, baseline_row, y_row_bits)) |at| {
                std.debug.print("FAIL sdpa causality: {s} future keys changed row {d} output at element {d}\n", .{ vname, r, at });
                return false;
            }
        }
    }
    if (!try plantRefuse(g, first.?, ac.name, "sdpa")) return false;
    if (!try unchanged(a, cmds, staging, qf.q_bits, qraw, timeout)) {
        std.debug.print("FAIL sdpa: the raw q input changed\n", .{});
        return false;
    }
    std.debug.print("MILESTONE sdpa elapsed_ms={d} case=\"{s}\" prefix={d} rows={d} visible-counts={any} worst={e} of bound, " ++
        "repeats byte-identical, nan/large-finite/zero future keys bit-identical, q unchanged, planted outputs refused\n", .{ clock.ms(), ac.name, ac.prefix, rows, g.visible, rep.worst });
    return true;
}

// Interleaved output repeats 3 times, matches f64, refuses plants, and leaves both inputs unchanged.
fn gateGate(a: std.mem.Allocator, clock: Elapsed, dev: *const vk.Device, cmds: *vk.Commands, staging: vk.Buffer, attn: *vk.Attention, timeout: u64) !bool {
    const case = fx.gate_case;
    const f = try fx.Gate.init(a, case, 0x57a6);
    defer f.deinit();
    var qraw = try upload16(dev, cmds, staging, f.qraw_bits, timeout);
    defer qraw.deinit();
    var ctx = try upload16(dev, cmds, staging, f.ctx_bits, timeout);
    defer ctx.deinit();
    // out starts poisoned: every row must be written by the kernel
    var out = try upload16(dev, cmds, staging, f.out_bits, timeout);
    defer out.deinit();
    const elems: u64 = case.rows * 2048;
    const out_view = try vk.tensor.View.init(&out, .bf16, 0, elems);
    const ctx_view = try vk.tensor.View.init(&ctx, .bf16, 0, elems);
    const qraw_view = try vk.tensor.View.init(&qraw, .bf16, 0, case.rows * 4096);
    const bits = try a.alloc(u16, @intCast(elems));
    defer a.free(bits);
    var first: ?[]u16 = null;
    defer if (first) |x| a.free(x);
    for (0..repeats) |rep| {
        try vk.orExit(vk.upload(cmds, out, 0, staging, std.mem.sliceAsBytes(f.out_bits), timeout));
        try vk.orExit(attn.gateOutput(cmds, out_view, ctx_view, qraw_view, case.rows, timeout));
        try fetch(cmds, staging, out, std.mem.sliceAsBytes(bits), timeout);
        if (first) |x| {
            if (std.mem.indexOfDiff(u16, x, bits)) |at| {
                std.debug.print("FAIL gate case=\"{s}\": repeat {d} differs at element {d}\n", .{ case.name, rep, at });
                return false;
            }
        } else first = try a.dupe(u16, bits);
    }
    const rep = f.check(first.?) catch |e| {
        std.debug.print("FAIL gate case=\"{s}\": {s}\n", .{ case.name, @errorName(e) });
        if (f.firstViolation(first.?)) |d| {
            std.debug.print("DIAG gate index={d} row={d} col={d} head={d} ctx=0x{x:0>4} ({e}) raw_gate=0x{x:0>4} ({e}) got_word=0x{x:0>4} got={e} want={e} bound={e} ratio={e} HOST_sigmoid_f64={e} HOST_sigmoid_f32={e} HOST_sigmoid_bf16=0x{x:0>4} ({e})\n", .{
                d.index, d.row, d.d, d.head, d.ctx_bits, d.ctx, d.gate_bits, d.gate, first.?[d.index], d.got, d.want, d.bound, d.ratio, d.sigmoid_f64, d.sigmoid_f32, d.sigmoid_bf16_bits, d.sigmoid_bf16,
            });
        }
        return false;
    };
    if (!try plantRefuse(f, first.?, case.name, "gate")) return false;
    if (!try unchanged(a, cmds, staging, f.qraw_bits, qraw, timeout) or !try unchanged(a, cmds, staging, f.ctx_bits, ctx, timeout)) {
        std.debug.print("FAIL gate case=\"{s}\": an input changed\n", .{case.name});
        return false;
    }
    std.debug.print("MILESTONE gate elapsed_ms={d} case=\"{s}\" rows={d} worst={e} of bound, repeats byte-identical, " ++
        "inputs unchanged, planted outputs refused\n", .{ clock.ms(), case.name, case.rows, rep.worst });
    return true;
}
