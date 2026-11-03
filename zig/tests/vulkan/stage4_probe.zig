// Stage 4: GatedDeltaNet (conv prefill, gate split, recurrence, raw-gates decode, RMSNorm) via Gdn; exits 0/1/2/124.

const std = @import("std");
const vk = @import("vulkan");
const shaders = @import("stage4_shaders");
const fx = @import("stage4_fixture.zig");
const shared = @import("stage4_probe_shared.zig");
const gates = @import("stage4_probe_gates.zig");

const Elapsed = shared.Elapsed;
const repeats = shared.repeats;
const uploadShell = shared.uploadShell;
const fetch = shared.fetch;
const unchanged = shared.unchanged;
const printSplitGateInputs = shared.printSplitGateInputs;

const usage = "usage: tf-stage4-gpu [--vendor ID] [--timeout-ms MS] | --help\n" ++
    "  --help          show this text and exit without opening a loader or device\n" ++
    "  --vendor ID     refuse a device whose Vulkan vendorID differs (decimal or 0x hex; no default)\n" ++
    "  --timeout-ms MS bound on every fence wait, 1 to 600000 (default 10000); a timeout ends the process\n";

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
        "api={d}.{d}.{d} storage16={} int16={} subgroups={d} shuffle-relative={} subgroup-size={d} fence-timeout={d}ms\n", .{
        clock.ms(),                                                                               name,
        pr.vendorID,                                                                              pr.deviceID,
        v >> 22 & 0x7f,                                                                           v >> 12 & 0x3ff,
        v & 0xfff,                                                                                dev.features.storage16,
        dev.features.int16,                                                                       dev.v11.subgroupSupportedStages,
        dev.v11.subgroupSupportedOperations & vk.c.VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT != 0, dev.v11.subgroupSize,
        timeout / std.time.ns_per_ms,
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
    var staging = try vk.Buffer.init(&dev, .staging, 4 << 20);
    defer staging.deinit();
    var m = try vk.Gdn.init(&dev, .{
        .conv_dw1d_bf16 = &shaders.conv_dw1d_bf16,
        .gdn_conv_decode_bf16 = &shaders.gdn_conv_decode_bf16,
        .gdn_split_bf16 = &shaders.gdn_split_bf16,
        .prefill_recur_bf16 = &shaders.prefill_recur_bf16,
        .gated_delta_decode_bf16 = &shaders.gated_delta_decode_bf16,
        .fast_norm_gated_bf16 = &shaders.fast_norm_gated_bf16,
        .copy_u16 = &shaders.copy_u16,
    });
    defer m.deinit();

    var passed: usize = 0;
    var total: usize = 0;
    total += 1;
    passed += @intFromBool(try convSplitGate(a, clock, &dev, &cmds, staging, &m, timeout));
    total += 1;
    passed += @intFromBool(try gates.prefillGate(a, clock, &dev, &cmds, staging, &m, 8, null, timeout));
    total += 1;
    passed += @intFromBool(try gates.prefillGate(a, clock, &dev, &cmds, staging, &m, 4, 1, timeout));
    total += 1;
    passed += @intFromBool(try gates.decodeGate(a, clock, &dev, &cmds, staging, &m, timeout));
    total += 1;
    passed += @intFromBool(try gates.convDecodeGate(a, clock, &dev, &cmds, staging, &m, timeout));
    total += 1;
    passed += @intFromBool(try gates.normGate(a, clock, &dev, &cmds, staging, &m, timeout));
    std.debug.print("RESULT stage4 elapsed_ms={d} on \"{s}\": passed={d} total={d}\n", .{ clock.ms(), name, passed, total });
    return passed == total;
}
// convPrefill: conv input + state carry-out BIT-EXACT; conv output and split soa/g/beta vs f64; repeats and refusals.
fn convSplitGate(a: std.mem.Allocator, clock: Elapsed, dev: *const vk.Device, cmds: *vk.Commands, staging: vk.Buffer, m: *vk.Gdn, timeout: u64) !bool {
    const T = fx.tokens;
    const C: u64 = fx.Model.conv_channels;
    const soa_elems: u64 = 3 * T * fx.Model.linear_dim;
    const gate_elems: u64 = T * fx.Model.heads;
    const f = try fx.Split.init(a, 0x57a4);
    defer f.deinit();
    const cf = try fx.Conv.init(a, 0x57a4 + 1);
    defer cf.deinit();

    var state = try uploadShell(dev, cmds, staging, cf.state_bits, timeout);
    defer state.deinit();
    var qkv = try uploadShell(dev, cmds, staging, cf.qkv_bits, timeout);
    defer qkv.deinit();
    var weight = try uploadShell(dev, cmds, staging, cf.weight_bits, timeout);
    defer weight.deinit();
    var conv_input = try vk.Buffer.init(dev, .device, @intCast((T + 3) * C * 2));
    defer conv_input.deinit();
    var conv_out = try vk.Buffer.init(dev, .device, @intCast(T * C * 2));
    defer conv_out.deinit();
    var soa = try vk.Buffer.init(dev, .device, @intCast(soa_elems * 2));
    defer soa.deinit();
    var g_out = try vk.Buffer.init(dev, .device, @intCast(gate_elems * 4));
    defer g_out.deinit();
    var beta_out = try vk.Buffer.init(dev, .device, @intCast(gate_elems * 2));
    defer beta_out.deinit();
    var state_out = try vk.Buffer.init(dev, .device, @intCast(3 * C * 2));
    defer state_out.deinit();
    var a_log = try uploadShell(dev, cmds, staging, f.a_log_bits, timeout);
    defer a_log.deinit();

    const conv_input_view = try vk.tensor.View.init(&conv_input, .bf16, 0, (T + 3) * C);
    const state_view = try vk.tensor.View.init(&state, .bf16, 0, 3 * C);
    const qkv_view = try vk.tensor.View.init(&qkv, .bf16, 0, T * C);
    const weight_view = try vk.tensor.View.init(&weight, .bf16, 0, 4 * C);
    const conv_out_view = try vk.tensor.View.init(&conv_out, .bf16, 0, T * C);
    const soa_view = try vk.tensor.View.init(&soa, .bf16, 0, soa_elems);
    const g_view = try vk.tensor.View.init(&g_out, .f32, 0, gate_elems);
    const beta_view = try vk.tensor.View.init(&beta_out, .bf16, 0, gate_elems);
    var a_buf = try uploadShell(dev, cmds, staging, f.a_bits, timeout);
    defer a_buf.deinit();
    var b_buf = try uploadShell(dev, cmds, staging, f.b_bits, timeout);
    defer b_buf.deinit();
    var dt_buf = try uploadShell(dev, cmds, staging, f.dt_bits, timeout);
    defer dt_buf.deinit();
    const a_log_view = try vk.tensor.View.init(&a_log, .f32, 0, fx.Model.heads);
    const a_view2 = try vk.tensor.View.init(&a_buf, .bf16, 0, gate_elems);
    const b_view = try vk.tensor.View.init(&b_buf, .bf16, 0, gate_elems);
    const dt_view = try vk.tensor.View.init(&dt_buf, .bf16, 0, fx.Model.heads);
    const state_out_view = try vk.tensor.View.init(&state_out, .bf16, 0, 3 * C);

    var first: ?[]u16 = null;
    defer if (first) |x| a.free(x);
    var g_first: ?[]u16 = null;
    defer if (g_first) |x| a.free(x);
    var beta_first: ?[]u16 = null;
    defer if (beta_first) |x| a.free(x);
    var conv_first: ?[]u16 = null;
    defer if (conv_first) |x| a.free(x);
    var state_first: ?[]u16 = null;
    defer if (state_first) |x| a.free(x);
    // poison shells for EVERY output: soa, g, beta, conv_out, state_out
    const init16 = try a.alloc(u16, @intCast(soa_elems));
    defer a.free(init16);
    @memset(init16, fx.poison16);
    const init32 = try a.alloc(u16, @intCast(2 * gate_elems));
    defer a.free(init32);
    @memset(init32, 0xffff);
    const init_beta = try a.alloc(u16, @intCast(gate_elems));
    defer a.free(init_beta);
    @memset(init_beta, fx.poison16);
    const init_conv = try a.alloc(u16, @intCast(T * C));
    defer a.free(init_conv);
    @memset(init_conv, fx.poison16);
    const init_state = try a.alloc(u16, @intCast(3 * C));
    defer a.free(init_state);
    @memset(init_state, fx.poison16);
    // per-repeat readbacks of every output, compared byte-for-byte
    const soa_repeat = try a.alloc(u16, @intCast(soa_elems));
    defer a.free(soa_repeat);
    const g_repeat = try a.alloc(u16, @intCast(2 * gate_elems));
    defer a.free(g_repeat);
    const beta_repeat = try a.alloc(u16, @intCast(gate_elems));
    defer a.free(beta_repeat);
    const conv_repeat = try a.alloc(u16, @intCast(T * C));
    defer a.free(conv_repeat);
    const state_repeat = try a.alloc(u16, @intCast(3 * C));
    defer a.free(state_repeat);
    for (0..repeats) |rep| {
        try vk.orExit(vk.upload(cmds, soa, 0, staging, std.mem.sliceAsBytes(init16[0..@intCast(soa_elems)]), timeout));
        try vk.orExit(vk.upload(cmds, g_out, 0, staging, std.mem.sliceAsBytes(init32), timeout));
        try vk.orExit(vk.upload(cmds, beta_out, 0, staging, std.mem.sliceAsBytes(init_beta), timeout));
        try vk.orExit(vk.upload(cmds, conv_out, 0, staging, std.mem.sliceAsBytes(init_conv), timeout));
        try vk.orExit(vk.upload(cmds, state_out, 0, staging, std.mem.sliceAsBytes(init_state), timeout));
        try vk.orExit(m.convPrefill(cmds, soa_view, g_view, beta_view, conv_out_view, conv_input_view, state_view, qkv_view, weight_view, a_view2, b_view, a_log_view, dt_view, state_out_view, T, timeout));
        try fetch(cmds, staging, soa, std.mem.sliceAsBytes(soa_repeat), timeout);
        try fetch(cmds, staging, g_out, std.mem.sliceAsBytes(g_repeat), timeout);
        try fetch(cmds, staging, beta_out, std.mem.sliceAsBytes(beta_repeat), timeout);
        try fetch(cmds, staging, conv_out, std.mem.sliceAsBytes(conv_repeat), timeout);
        try fetch(cmds, staging, state_out, std.mem.sliceAsBytes(state_repeat), timeout);
        if (first) |x| {
            if (std.mem.indexOfDiff(u16, x, soa_repeat)) |at| {
                std.debug.print("FAIL conv-split: repeat {d} soa differs at element {d}\n", .{ rep, at });
                return false;
            }
            // compare the remaining outputs against the FIRST repeat's copies
            if (g_first) |gf| {
                if (std.mem.indexOfDiff(u16, gf, g_repeat)) |at| {
                    std.debug.print("FAIL conv-split: repeat {d} g differs at element {d}\n", .{ rep, at });
                    return false;
                }
            } else g_first = try a.dupe(u16, g_repeat);
            if (beta_first) |bf| {
                if (std.mem.indexOfDiff(u16, bf, beta_repeat)) |at| {
                    std.debug.print("FAIL conv-split: repeat {d} beta differs at element {d}\n", .{ rep, at });
                    return false;
                }
            } else beta_first = try a.dupe(u16, beta_repeat);
            if (conv_first) |cf2| {
                if (std.mem.indexOfDiff(u16, cf2, conv_repeat)) |at| {
                    std.debug.print("FAIL conv-split: repeat {d} conv_out differs at element {d}\n", .{ rep, at });
                    return false;
                }
            } else conv_first = try a.dupe(u16, conv_repeat);
            if (state_first) |sf2| {
                if (std.mem.indexOfDiff(u16, sf2, state_repeat)) |at| {
                    std.debug.print("FAIL conv-split: repeat {d} state_out differs at element {d}\n", .{ rep, at });
                    return false;
                }
            } else state_first = try a.dupe(u16, state_repeat);
        } else {
            first = try a.dupe(u16, soa_repeat);
            g_first = try a.dupe(u16, g_repeat);
            beta_first = try a.dupe(u16, beta_repeat);
            conv_first = try a.dupe(u16, conv_repeat);
            state_first = try a.dupe(u16, state_repeat);
        }
    }
    // download and check everything
    const soa_bits = try a.alloc(u16, @intCast(soa_elems));
    defer a.free(soa_bits);
    const g_pairs = try a.alloc(u16, @intCast(2 * gate_elems));
    defer a.free(g_pairs);
    const beta_bits = try a.alloc(u16, @intCast(gate_elems));
    defer a.free(beta_bits);
    const conv_input_bits = try a.alloc(u16, @intCast((T + 3) * C));
    defer a.free(conv_input_bits);
    const conv_out_bits = try a.alloc(u16, @intCast(T * C));
    defer a.free(conv_out_bits);
    const state_out_bits = try a.alloc(u16, @intCast(3 * C));
    defer a.free(state_out_bits);
    try fetch(cmds, staging, soa, std.mem.sliceAsBytes(soa_bits), timeout);
    try fetch(cmds, staging, g_out, std.mem.sliceAsBytes(g_pairs), timeout);
    try fetch(cmds, staging, beta_out, std.mem.sliceAsBytes(beta_bits), timeout);
    try fetch(cmds, staging, conv_input, std.mem.sliceAsBytes(conv_input_bits), timeout);
    try fetch(cmds, staging, conv_out, std.mem.sliceAsBytes(conv_out_bits), timeout);
    try fetch(cmds, staging, state_out, std.mem.sliceAsBytes(state_out_bits), timeout);

    // The split's reference reads the ACTUAL conv output the kernel produced, not the synthetic conv_out_bits.
    @memcpy(f.conv_out_bits[0..conv_out_bits.len], conv_out_bits);
    var sr_diag: fx.Violation = undefined;
    const sr = fx.Split.check(f, soa_bits, g_pairs, beta_bits, &sr_diag) catch |e| {
        std.debug.print("FAIL conv-split: split {s} first at index {d}: want={e} got={e} bound={e} ratio={e} want_f32_bits={x}\n", .{ @errorName(e), sr_diag.index, sr_diag.want, sr_diag.got, sr_diag.bound, sr_diag.ratio, sr_diag.want_f32_bits });
        printSplitGateInputs(f, sr_diag);
        return false;
    };
    fx.Conv.checkInput(cf, conv_input_bits) catch |e| {
        std.debug.print("FAIL conv-split: conv input {s}\n", .{@errorName(e)});
        return false;
    };
    const cr = fx.Conv.checkOutput(cf, conv_out_bits) catch |e| {
        std.debug.print("FAIL conv-split: conv {s}\n", .{@errorName(e)});
        return false;
    };
    // the state carry-out is conv input rows tokens..tokens+3, bit-exact
    if (std.mem.indexOfDiff(u16, cf.conv_input_want[@intCast(T * C)..], state_out_bits) != null) {
        std.debug.print("FAIL conv-split: the conv state carry-out is not the trailing window\n", .{});
        return false;
    }
    if (!try unchanged(a, cmds, staging, cf.state_bits, state, timeout) or
        !try unchanged(a, cmds, staging, cf.qkv_bits, qkv, timeout) or
        !try unchanged(a, cmds, staging, cf.weight_bits, weight, timeout) or
        !try unchanged(a, cmds, staging, f.a_bits, a_buf, timeout) or
        !try unchanged(a, cmds, staging, f.b_bits, b_buf, timeout) or
        !try unchanged(a, cmds, staging, f.dt_bits, dt_buf, timeout) or
        !try unchanged(a, cmds, staging, f.a_log_bits, a_log, timeout))
    {
        std.debug.print("FAIL conv-split: an input changed\n", .{});
        return false;
    }
    {
        const mutated = try a.dupe(u16, soa_bits);
        defer a.free(mutated);
        const gmut = try a.dupe(u16, g_pairs);
        defer a.free(gmut);
        var first_finite: ?usize = null;
        for (soa_bits, 0..) |bits, i| {
            if (std.math.isFinite(fx.c.fromBf16(bits))) {
                first_finite = i;
                break;
            }
        }
        const at = first_finite orelse return error.NoFiniteReference;
        mutated[at] = 0x4f00; // 2^31: an absolutely wrong finite the budgets must refuse
        if (fx.Split.check(f, mutated, g_pairs, beta_bits, null)) |_| {
            std.debug.print("FAIL conv-split: a planted wrong finite split output passed\n", .{});
            return false;
        } else |_| {}
        for ([_]u16{ 0x7fc0, 0x7f80, 0xff80 }) |pb| {
            mutated[at] = pb;
            if (fx.Split.check(f, mutated, g_pairs, beta_bits, null)) |_| {
                std.debug.print("FAIL conv-split: a planted nonfinite split output passed\n", .{});
                return false;
            } else |_| {}
        }
    }
    std.debug.print("MILESTONE conv-split elapsed_ms={d} tokens={d} split worst={e} conv worst={e} of bounds, " ++
        "conv input and state carry-out bit-exact, repeats byte-identical, inputs unchanged, planted outputs refused\n", .{ clock.ms(), T, sr.worst, cr.worst });
    return true;
}
