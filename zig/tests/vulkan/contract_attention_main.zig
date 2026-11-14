// The attention gate checks rope_bf16 and sdpa_rows_bf16: decode=1, prefill<=32, keys=1..7168; exits 0/1/2/124.
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
/// 1 MiB: the 7168-key cache buffers (7.3 MB each) upload in few chunks.
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
    for (ca.served_cases, 0..) |case, i| {
        total += 1;
        passed += @intFromBool(try sdpaGate(a, clock, &dev, &cmds, staging, &attn, case, 0x5dcafe + @as(u64, i), timeout));
    }
    std.debug.print("RESULT attention-contract elapsed_ms={d} on \"{s}\": passed={d} total={d}\n", .{ clock.ms(), name, passed, total });
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
        try vk.orExit(attn.attend(cmds, y_view, q_view, keys_view, values_view, case.prefix, case.rows, f.capacity, timeout));
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
            try vk.orExit(vk.upload(cmds, one, 0, staging, std.mem.sliceAsBytes(one_shell), timeout));
            try vk.orExit(attn.attend(cmds, one_view, q_row, keys_view, values_view, case.prefix + @as(u64, r), 1, f.capacity, timeout));
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
