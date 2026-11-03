// Stage 4 probe: the conv-split, prefill, decode, conv-decode and gated-norm GPU gates.

const std = @import("std");
const vk = @import("vulkan");
const fx = @import("stage4_fixture.zig");
const shared = @import("stage4_probe_shared.zig");

const Elapsed = shared.Elapsed;
const repeats = shared.repeats;
const state_elems = shared.state_elems;
const uploadShell = shared.uploadShell;
const fetch = shared.fetch;
const unchanged = shared.unchanged;
const plantRefuse = shared.plantRefuse;
const plantRefuse2 = shared.plantRefuse2;
const printGateInputs = shared.printGateInputs;
const printGateInputsOne = shared.printGateInputsOne;
const printSplitGateInputs = shared.printSplitGateInputs;
const printRecurInputs = shared.printRecurInputs;

const SharedSplit = struct {
    a: std.mem.Allocator,
    f: fx.Split,
    soa_buf: vk.Buffer,
    g_buf: vk.Buffer,
    beta_buf: vk.Buffer,
    soa_bits: []u16,
    g_pairs: []u16,
    beta_bits: []u16,
    soa_view: vk.tensor.View,
    g_view: vk.tensor.View,
    beta_view: vk.tensor.View,
    a_log_buf: vk.Buffer,
    a_buf: vk.Buffer,
    b_buf: vk.Buffer,
    dt_buf: vk.Buffer,

    fn deinit(s: *SharedSplit) void {
        s.f.deinit();
        s.soa_buf.deinit();
        s.g_buf.deinit();
        s.beta_buf.deinit();
        s.a_log_buf.deinit();
        s.a_buf.deinit();
        s.b_buf.deinit();
        s.dt_buf.deinit();
        s.a.free(s.soa_bits);
        s.a.free(s.g_pairs);
        s.a.free(s.beta_bits);
    }
};

// The split runs standalone into a caller-owned struct (its views must outlive it); the gates read real GPU soa/g/beta.
pub fn runSplit(s: *SharedSplit, a: std.mem.Allocator, dev: *const vk.Device, cmds: *vk.Commands, staging: vk.Buffer, m: *vk.Gdn, timeout: u64) !void {
    const T = fx.tokens;
    const soa_elems: u64 = 3 * T * fx.Model.linear_dim;
    const gate_elems: u64 = T * fx.Model.heads;
    // staged construction: each field owns its errdefer, so a failure never runs deinit on undefined handles
    s.a = a;
    s.f = try fx.Split.init(a, 0x57a4);
    errdefer s.f.deinit();
    s.soa_bits = try a.alloc(u16, @intCast(soa_elems));
    errdefer a.free(s.soa_bits);
    s.g_pairs = try a.alloc(u16, @intCast(2 * gate_elems));
    errdefer a.free(s.g_pairs);
    s.beta_bits = try a.alloc(u16, @intCast(gate_elems));
    errdefer a.free(s.beta_bits);
    s.soa_buf = try vk.Buffer.init(dev, .device, @intCast(soa_elems * 2));
    errdefer s.soa_buf.deinit();
    s.g_buf = try vk.Buffer.init(dev, .device, @intCast(gate_elems * 4));
    errdefer s.g_buf.deinit();
    s.beta_buf = try vk.Buffer.init(dev, .device, @intCast(gate_elems * 2));
    errdefer s.beta_buf.deinit();
    s.soa_view = undefined;
    s.g_view = undefined;
    s.beta_view = undefined;
    s.a_log_buf = try uploadShell(dev, cmds, staging, s.f.a_log_bits, timeout);
    errdefer s.a_log_buf.deinit();
    s.a_buf = try uploadShell(dev, cmds, staging, s.f.a_bits, timeout);
    errdefer s.a_buf.deinit();
    s.b_buf = try uploadShell(dev, cmds, staging, s.f.b_bits, timeout);
    errdefer s.b_buf.deinit();
    s.dt_buf = try uploadShell(dev, cmds, staging, s.f.dt_bits, timeout);
    errdefer s.dt_buf.deinit();
    s.soa_view = try vk.tensor.View.init(&s.soa_buf, .bf16, 0, soa_elems);
    s.g_view = try vk.tensor.View.init(&s.g_buf, .f32, 0, gate_elems);
    s.beta_view = try vk.tensor.View.init(&s.beta_buf, .bf16, 0, gate_elems);
    var conv_out_buf = try uploadShell(dev, cmds, staging, s.f.conv_out_bits, timeout);
    defer conv_out_buf.deinit();
    // poison ALL THREE split outputs before the dispatch: a dead or partial g/beta write fails the check.
    const poison = try a.alloc(u16, @intCast(soa_elems));
    defer a.free(poison);
    @memset(poison, fx.poison16);
    const poison_g = try a.alloc(u16, @intCast(2 * gate_elems));
    defer a.free(poison_g);
    @memset(poison_g, 0xffff);
    const poison_beta = try a.alloc(u16, @intCast(gate_elems));
    defer a.free(poison_beta);
    @memset(poison_beta, fx.poison16);
    try vk.orExit(vk.upload(cmds, s.soa_buf, 0, staging, std.mem.sliceAsBytes(poison), timeout));
    try vk.orExit(vk.upload(cmds, s.g_buf, 0, staging, std.mem.sliceAsBytes(poison_g), timeout));
    try vk.orExit(vk.upload(cmds, s.beta_buf, 0, staging, std.mem.sliceAsBytes(poison_beta), timeout));
    try vk.orExit(m.split(cmds, s.soa_view, s.g_view, s.beta_view, try vk.tensor.View.init(&conv_out_buf, .bf16, 0, T * fx.Model.conv_channels), try vk.tensor.View.init(&s.a_buf, .bf16, 0, gate_elems), try vk.tensor.View.init(&s.b_buf, .bf16, 0, gate_elems), try vk.tensor.View.init(&s.a_log_buf, .f32, 0, fx.Model.heads), try vk.tensor.View.init(&s.dt_buf, .bf16, 0, fx.Model.heads), T, timeout));
    try fetch(cmds, staging, s.soa_buf, std.mem.sliceAsBytes(s.soa_bits), timeout);
    try fetch(cmds, staging, s.g_buf, std.mem.sliceAsBytes(s.g_pairs), timeout);
    try fetch(cmds, staging, s.beta_buf, std.mem.sliceAsBytes(s.beta_bits), timeout);
    var s_diag: fx.Violation = undefined;
    const srep = s.f.check(s.soa_bits, s.g_pairs, s.beta_bits, &s_diag) catch |e| {
        std.debug.print("FAIL split standalone: {s} first at index {d}: want={e} got={e} bound={e} ratio={e} want_f32_bits={x}\n", .{ @errorName(e), s_diag.index, s_diag.want, s_diag.got, s_diag.bound, s_diag.ratio, s_diag.want_f32_bits });
        printSplitGateInputs(s.f, s_diag);
        return error.SplitFailed;
    };
    std.debug.print("MILESTONE split-standalone worst={e} of bound\n", .{srep.worst});
}

// One prefill recurrence gate at chunk size `chunk` (snap slots when >1) vs the f64 reference over GPU soa/g/beta.
pub fn prefillGate(a: std.mem.Allocator, clock: Elapsed, dev: *const vk.Device, cmds: *vk.Commands, staging: vk.Buffer, m: *vk.Gdn, chunk: u64, snap_chunks: ?u64, timeout: u64) !bool {
    const T = fx.tokens;
    const section: u64 = T * fx.Model.linear_dim;
    var split: SharedSplit = undefined;
    runSplit(&split, a, dev, cmds, staging, m, timeout) catch return false;
    defer split.deinit();
    const rf = try fx.Recur.init(a, split.soa_bits, split.g_pairs, split.beta_bits, 0x57a7, chunk);
    defer rf.deinit();

    var h0 = try uploadShell(dev, cmds, staging, rf.h0_bits, timeout);
    defer h0.deinit();
    var y = try vk.Buffer.init(dev, .device, @intCast(section * 2));
    defer y.deinit();
    var hf = try vk.Buffer.init(dev, .device, @intCast(state_elems * 4));
    defer hf.deinit();
    var snap: ?vk.Buffer = null;
    defer if (snap) |*s| s.deinit();
    if (snap_chunks) |n| {
        snap = try vk.Buffer.init(dev, .device, @intCast((n * state_elems) * 4));
    }
    const h0_view = try vk.tensor.View.init(&h0, .f32, 0, state_elems);
    const y_view = try vk.tensor.View.init(&y, .bf16, 0, section);
    const hf_view = try vk.tensor.View.init(&hf, .f32, 0, state_elems);
    const snap_view: ?vk.tensor.View = if (snap) |*s| try vk.tensor.View.init(s, .f32, 0, snap_chunks.? * state_elems) else null;

    var r_diag: fx.Violation = undefined;
    // poison y and hf/snap before EVERY repeat: a dead or partial store leaves poison and fails; readback vs f64.
    const poison = try a.alloc(u16, @intCast(section));
    defer a.free(poison);
    @memset(poison, fx.poison16);
    const poison_state = try a.alloc(u16, @intCast(2 * state_elems));
    defer a.free(poison_state);
    @memset(poison_state, 0xffff);
    var first: ?[]u16 = null;
    defer if (first) |x| a.free(x);
    var hf_first: ?[]u16 = null;
    defer if (hf_first) |x| a.free(x);
    var snap_first: ?[]u16 = null;
    defer if (snap_first) |x| a.free(x);
    const ybits = try a.alloc(u16, @intCast(section));
    defer a.free(ybits);
    const hfpairs = try a.alloc(u16, @intCast(2 * state_elems));
    defer a.free(hfpairs);
    const snappairs = try a.alloc(u16, @intCast(2 * state_elems));
    defer a.free(snappairs);
    for (0..repeats) |rep| {
        try vk.orExit(vk.upload(cmds, y, 0, staging, std.mem.sliceAsBytes(poison), timeout));
        try vk.orExit(vk.upload(cmds, hf, 0, staging, std.mem.sliceAsBytes(poison_state), timeout));
        if (snap) |*s| try vk.orExit(vk.upload(cmds, s.*, 0, staging, std.mem.sliceAsBytes(poison_state), timeout));
        try vk.orExit(m.prefillRecur(cmds, split.soa_view, split.g_view, split.beta_view, h0_view, y_view, hf_view, snap_view, T, chunk, timeout));
        try fetch(cmds, staging, y, std.mem.sliceAsBytes(ybits), timeout);
        try fetch(cmds, staging, hf, std.mem.sliceAsBytes(hfpairs), timeout);
        // EVERY repeat: y byte-equal to the first AND hf re-checked against the f64 reference.
        if (first) |x| {
            if (std.mem.indexOfDiff(u16, x, ybits)) |at| {
                std.debug.print("FAIL prefill chunk={d}: repeat {d} y differs at element {d}\n", .{ chunk, rep, at });
                return false;
            }
            if (hf_first) |hf0| {
                if (std.mem.indexOfDiff(u16, hf0, hfpairs)) |at| {
                    std.debug.print("FAIL prefill chunk={d}: repeat {d} hf differs at element {d}\n", .{ chunk, rep, at });
                    return false;
                }
            } else hf_first = try a.dupe(u16, hfpairs);
        } else {
            first = try a.dupe(u16, ybits);
            hf_first = try a.dupe(u16, hfpairs);
        }
        _ = rf.check(ybits, hfpairs, &r_diag) catch |e| {
            std.debug.print("FAIL prefill chunk={d}: repeat {d} {s} first at index {d}: want={e} got={e} bound={e} ratio={e} want_f32_bits={x}\n", .{ chunk, rep, @errorName(e), r_diag.index, r_diag.want, r_diag.got, r_diag.bound, r_diag.ratio, r_diag.want_f32_bits });
            printRecurInputs(split.f, r_diag, T - 1);
            return false;
        };
        // the snapshot slot: downloaded, checked against the walker's boundary state, and byte-compared across repeats
        if (snap) |*s| {
            try fetch(cmds, staging, s.*, std.mem.sliceAsBytes(snappairs), timeout);
            if (snap_first) |sp0| {
                if (std.mem.indexOfDiff(u16, sp0, snappairs)) |at| {
                    std.debug.print("FAIL prefill chunk={d}: repeat {d} snap differs at element {d}\n", .{ chunk, rep, at });
                    return false;
                }
            } else snap_first = try a.dupe(u16, snappairs);
            _ = rf.checkSnap(snappairs, &r_diag) catch |e| {
                std.debug.print("FAIL prefill chunk={d}: repeat {d} snap {s} first at index {d}: want={e} got={e} bound={e} ratio={e}\n", .{ chunk, rep, @errorName(e), r_diag.index, r_diag.want, r_diag.got, r_diag.bound, r_diag.ratio });
                printRecurInputs(split.f, r_diag, chunk - 1);
                return false;
            };
        }
    }
    const rrep = rf.check(ybits, hfpairs, &r_diag) catch |e| {
        std.debug.print("FAIL prefill chunk={d}: {s} first at index {d}: want={e} got={e} bound={e} ratio={e} want_f32_bits={x}\n", .{ chunk, @errorName(e), r_diag.index, r_diag.want, r_diag.got, r_diag.bound, r_diag.ratio, r_diag.want_f32_bits });
        printRecurInputs(split.f, r_diag, T - 1);
        return false;
    };
    if (!try unchanged(a, cmds, staging, rf.h0_bits, h0, timeout) or
        !try unchanged(a, cmds, staging, split.soa_bits, split.soa_buf, timeout))
    {
        std.debug.print("FAIL prefill chunk={d}: an input changed\n", .{chunk});
        return false;
    }
    if (!try plantRefuse2(rf, first.?, hfpairs, "y", "prefill")) return false;
    std.debug.print("MILESTONE prefill elapsed_ms={d} tokens={d} chunk={d} chunks={?d} worst={e} of bound, " ++
        "hf matches independent f64 reference from poison, repeats byte-identical, h0 and soa unchanged, planted outputs refused\n", .{ clock.ms(), T, chunk, snap_chunks, rrep.worst });
    return true;
}

// The raw-gates decode step: one token, state in/out and y vs the serial reference from the kernel's raw inputs.
pub fn decodeGate(a: std.mem.Allocator, clock: Elapsed, dev: *const vk.Device, cmds: *vk.Commands, staging: vk.Buffer, m: *vk.Gdn, timeout: u64) !bool {
    const linear: u64 = fx.Model.linear_dim;
    const f = try fx.Decode.init(a, 0x57a8);
    defer f.deinit();
    var soa = try uploadShell(dev, cmds, staging, f.soa_bits, timeout);
    defer soa.deinit();
    var a_buf = try uploadShell(dev, cmds, staging, f.a_bits, timeout);
    defer a_buf.deinit();
    var b_buf = try uploadShell(dev, cmds, staging, f.b_bits, timeout);
    defer b_buf.deinit();
    var a_log = try uploadShell(dev, cmds, staging, f.a_log_bits, timeout);
    defer a_log.deinit();
    var dt = try uploadShell(dev, cmds, staging, f.dt_bits, timeout);
    defer dt.deinit();
    var state_in = try uploadShell(dev, cmds, staging, f.state_in_bits, timeout);
    defer state_in.deinit();
    var state_out = try vk.Buffer.init(dev, .device, @intCast(state_elems * 4));
    defer state_out.deinit();
    var y = try vk.Buffer.init(dev, .device, @intCast(linear * 2));
    defer y.deinit();

    const soa_view = try vk.tensor.View.init(&soa, .bf16, 0, 3 * linear);
    const a_view = try vk.tensor.View.init(&a_buf, .bf16, 0, fx.Model.heads);
    const b_view = try vk.tensor.View.init(&b_buf, .bf16, 0, fx.Model.heads);
    const a_log_view = try vk.tensor.View.init(&a_log, .u16, 0, 2 * fx.Model.heads);
    const dt_view = try vk.tensor.View.init(&dt, .bf16, 0, fx.Model.heads);
    const state_in_view = try vk.tensor.View.init(&state_in, .f32, 0, state_elems);
    const state_out_view = try vk.tensor.View.init(&state_out, .f32, 0, state_elems);
    const y_view = try vk.tensor.View.init(&y, .bf16, 0, linear);

    const poison = try a.alloc(u16, @intCast(linear));
    defer a.free(poison);
    @memset(poison, fx.poison16);
    // poison state_out before EVERY repeat: a dead store leaves poison, which the f64 comparison refuses.
    const poison_state = try a.alloc(u16, @intCast(2 * state_elems));
    defer a.free(poison_state);
    @memset(poison_state, 0xffff);
    var first: ?[]u16 = null;
    defer if (first) |x| a.free(x);
    const ybits = try a.alloc(u16, @intCast(linear));
    defer a.free(ybits);
    const souts = try a.alloc(u16, @intCast(2 * state_elems));
    defer a.free(souts);
    for (0..repeats) |rep| {
        try vk.orExit(vk.upload(cmds, y, 0, staging, std.mem.sliceAsBytes(poison), timeout));
        try vk.orExit(vk.upload(cmds, state_out, 0, staging, std.mem.sliceAsBytes(poison_state), timeout));
        try vk.orExit(m.decodeRecur(cmds, soa_view, a_view, b_view, a_log_view, dt_view, state_in_view, state_out_view, y_view, timeout));
        try fetch(cmds, staging, y, std.mem.sliceAsBytes(ybits), timeout);
        try fetch(cmds, staging, state_out, std.mem.sliceAsBytes(souts), timeout);
        if (first) |x| {
            if (std.mem.indexOfDiff(u16, x, ybits)) |at| {
                std.debug.print("FAIL decode: repeat {d} differs at element {d}\n", .{ rep, at });
                return false;
            }
        } else first = try a.dupe(u16, ybits);
    }
    var d_diag: fx.Violation = undefined;
    const drep = f.check(ybits, souts, &d_diag) catch |e| {
        std.debug.print("FAIL decode: {s} first at index {d}: want={e} got={e} bound={e} ratio={e} want_f32_bits={x}\n", .{ @errorName(e), d_diag.index, d_diag.want, d_diag.got, d_diag.bound, d_diag.ratio, d_diag.want_f32_bits });
        std.debug.print("  DIAG output={s}\n", .{@tagName(d_diag.output)});
        const hv = if (d_diag.output == .state) d_diag.index / (128 * 128) else d_diag.index / 128;
        printGateInputs(f.a_bits, f.b_bits, f.a_log_bits, f.dt_bits, hv);
        return false;
    };
    if (!try unchanged(a, cmds, staging, f.soa_bits, soa, timeout) or
        !try unchanged(a, cmds, staging, f.state_in_bits, state_in, timeout) or
        !try unchanged(a, cmds, staging, f.a_bits, a_buf, timeout) or
        !try unchanged(a, cmds, staging, f.b_bits, b_buf, timeout) or
        !try unchanged(a, cmds, staging, f.a_log_bits, a_log, timeout) or
        !try unchanged(a, cmds, staging, f.dt_bits, dt, timeout))
    {
        std.debug.print("FAIL decode: an input changed\n", .{});
        return false;
    }
    if (!try plantRefuse2(f, first.?, souts, "y", "decode")) return false;
    std.debug.print("MILESTONE decode elapsed_ms={d} worst={e} of bound, repeats byte-identical, inputs unchanged, planted outputs refused\n", .{ clock.ms(), drep.worst });
    return true;
}

// convDecodeStep: 4-tap conv vs f64 (raw store; split owns SiLU); carry-out bit-exact; split ACTUAL bits vs SplitOne.
pub fn convDecodeGate(a: std.mem.Allocator, clock: Elapsed, dev: *const vk.Device, cmds: *vk.Commands, staging: vk.Buffer, m: *vk.Gdn, timeout: u64) !bool {
    const C: u64 = fx.Model.conv_channels;
    const linear: u64 = fx.Model.linear_dim;
    const cf = try fx.Conv.init(a, 0x57ab);
    defer cf.deinit();
    const sf = try fx.SplitOne.init(a, 0x57ac);
    defer sf.deinit();
    // x is the first conv channel-row of the qkv fixture
    const x_bits = cf.qkv_bits[0..@intCast(C)];

    var state = try uploadShell(dev, cmds, staging, cf.state_bits, timeout);
    defer state.deinit();
    var xbuf = try uploadShell(dev, cmds, staging, x_bits, timeout);
    defer xbuf.deinit();
    var weight = try uploadShell(dev, cmds, staging, cf.weight_bits, timeout);
    defer weight.deinit();
    var a_log = try uploadShell(dev, cmds, staging, sf.a_log_bits, timeout);
    defer a_log.deinit();
    var a_buf = try uploadShell(dev, cmds, staging, sf.a_bits, timeout);
    defer a_buf.deinit();
    var b_buf = try uploadShell(dev, cmds, staging, sf.b_bits, timeout);
    defer b_buf.deinit();
    var dt_buf = try uploadShell(dev, cmds, staging, sf.dt_bits, timeout);
    defer dt_buf.deinit();
    var activated = try vk.Buffer.init(dev, .device, @intCast(C * 2));
    defer activated.deinit();
    var state_out = try vk.Buffer.init(dev, .device, @intCast(3 * C * 2));
    defer state_out.deinit();
    var soa = try vk.Buffer.init(dev, .device, @intCast(3 * linear * 2));
    defer soa.deinit();
    var g_out = try vk.Buffer.init(dev, .device, @intCast(fx.Model.heads * 4));
    defer g_out.deinit();
    var beta_out = try vk.Buffer.init(dev, .device, @intCast(fx.Model.heads * 2));
    defer beta_out.deinit();

    const state_view = try vk.tensor.View.init(&state, .bf16, 0, 3 * C);
    const x_view = try vk.tensor.View.init(&xbuf, .bf16, 0, C);
    const weight_view = try vk.tensor.View.init(&weight, .bf16, 0, 4 * C);
    const activated_view = try vk.tensor.View.init(&activated, .bf16, 0, C);
    const state_out_view = try vk.tensor.View.init(&state_out, .bf16, 0, 3 * C);
    const soa_view = try vk.tensor.View.init(&soa, .bf16, 0, 3 * linear);
    const g_view = try vk.tensor.View.init(&g_out, .f32, 0, fx.Model.heads);
    const beta_view = try vk.tensor.View.init(&beta_out, .bf16, 0, fx.Model.heads);
    const a_log_view = try vk.tensor.View.init(&a_log, .f32, 0, fx.Model.heads);
    const a_view = try vk.tensor.View.init(&a_buf, .bf16, 0, fx.Model.heads);
    const b_view = try vk.tensor.View.init(&b_buf, .bf16, 0, fx.Model.heads);
    const dt_view = try vk.tensor.View.init(&dt_buf, .bf16, 0, fx.Model.heads);

    const poison = try a.alloc(u16, @intCast(C));
    defer a.free(poison);
    @memset(poison, fx.poison16);
    const init32 = try a.alloc(u16, @intCast(2 * fx.Model.heads));
    defer a.free(init32);
    @memset(init32, 0xffff);
    var first: ?[]u16 = null;
    defer if (first) |xx| a.free(xx);
    const act_bits = try a.alloc(u16, @intCast(C));
    defer a.free(act_bits);
    for (0..repeats) |rep| {
        try vk.orExit(vk.upload(cmds, activated, 0, staging, std.mem.sliceAsBytes(poison), timeout));
        try vk.orExit(vk.upload(cmds, g_out, 0, staging, std.mem.sliceAsBytes(init32), timeout));
        try vk.orExit(m.convDecodeStep(cmds, soa_view, g_view, beta_view, activated_view, state_view, x_view, weight_view, a_view, b_view, a_log_view, dt_view, state_out_view, timeout));
        try fetch(cmds, staging, activated, std.mem.sliceAsBytes(act_bits), timeout);
        if (first) |xx| {
            if (std.mem.indexOfDiff(u16, xx, act_bits)) |at| {
                std.debug.print("FAIL conv-decode: repeat {d} differs at element {d}\n", .{ rep, at });
                return false;
            }
        } else first = try a.dupe(u16, act_bits);
    }
    // raw conv reference: ascending 4 taps on exact dyadic operands
    var cr: fx.Report = .{};
    for (0..@intCast(C)) |ch| {
        var acc: f64 = 0;
        for (0..3) |k| acc += fx.c.fromBf16(cf.state_bits[k * @as(usize, @intCast(C)) + ch]) * fx.c.fromBf16(cf.weight_bits[ch * 4 + k]);
        acc += fx.c.fromBf16(x_bits[ch]) * fx.c.fromBf16(cf.weight_bits[ch * 4 + 3]);
        const got = fx.c.fromBf16(first.?[ch]);
        if (!std.math.isFinite(got)) {
            std.debug.print("FAIL conv-decode: nonfinite conv output at {d}\n", .{ch});
            return false;
        }
        const d = @abs(got - acc);
        const bound = fx.budget(4, 0, 0, acc, @abs(acc));
        if (d > bound) {
            std.debug.print("FAIL conv-decode: conv output over bound at {d}\n", .{ch});
            return false;
        }
        const ratio = d / bound;
        if (ratio > cr.worst) cr = .{ .worst = ratio, .where = ch };
    }
    // negative controls on the conv stage
    {
        const mutated = try a.dupe(u16, first.?);
        defer a.free(mutated);
        mutated[0] = 0x4f00;
        var refused = false;
        for (mutated, 0..) |bits, ch| {
            var acc: f64 = 0;
            for (0..3) |k| acc += fx.c.fromBf16(cf.state_bits[k * @as(usize, @intCast(C)) + ch]) * fx.c.fromBf16(cf.weight_bits[ch * 4 + k]);
            acc += fx.c.fromBf16(x_bits[ch]) * fx.c.fromBf16(cf.weight_bits[ch * 4 + 3]);
            if (!std.math.isFinite(fx.c.fromBf16(bits)) or @abs(fx.c.fromBf16(bits) - acc) > fx.budget(4, 0, 0, acc, @abs(acc))) {
                refused = true;
                break;
            }
        }
        if (!refused) {
            std.debug.print("FAIL conv-decode: a planted wrong conv output passed\n", .{});
            return false;
        }
    }
    // carry-out state: concat(state[1..3], x), bit-exact
    const souts = try a.alloc(u16, @intCast(3 * C));
    defer a.free(souts);
    try fetch(cmds, staging, state_out, std.mem.sliceAsBytes(souts), timeout);
    var want_state = try a.alloc(u16, @intCast(3 * C));
    defer a.free(want_state);
    @memcpy(want_state[0 .. 2 * @as(usize, @intCast(C))], cf.state_bits[@intCast(C)..][0 .. 2 * @as(usize, @intCast(C))]);
    @memcpy(want_state[2 * @as(usize, @intCast(C)) ..], x_bits);
    if (std.mem.indexOfDiff(u16, want_state, souts) != null) {
        std.debug.print("FAIL conv-decode: the carry-out state is not concat(state[1..3], x)\n", .{});
        return false;
    }
    // split on the ACTUAL activated bits
    @memcpy(sf.conv_out_bits[0..@intCast(C)], first.?);
    const soa_bits = try a.alloc(u16, @intCast(3 * linear));
    defer a.free(soa_bits);
    const g_pairs = try a.alloc(u16, @intCast(2 * fx.Model.heads));
    defer a.free(g_pairs);
    const beta_bits = try a.alloc(u16, @intCast(fx.Model.heads));
    defer a.free(beta_bits);
    try fetch(cmds, staging, soa, std.mem.sliceAsBytes(soa_bits), timeout);
    try fetch(cmds, staging, g_out, std.mem.sliceAsBytes(g_pairs), timeout);
    try fetch(cmds, staging, beta_out, std.mem.sliceAsBytes(beta_bits), timeout);
    var s2_diag: fx.Violation = undefined;
    const srep = sf.check(soa_bits, g_pairs, beta_bits, &s2_diag) catch |e| {
        std.debug.print("FAIL conv-decode: split {s} first at index {d}: want={e} got={e} bound={e} ratio={e} want_f32_bits={x}\n", .{ @errorName(e), s2_diag.index, s2_diag.want, s2_diag.got, s2_diag.bound, s2_diag.ratio, s2_diag.want_f32_bits });
        printGateInputsOne(sf, s2_diag);
        return false;
    };
    {
        const mutated = try a.dupe(u16, soa_bits);
        defer a.free(mutated);
        mutated[0] = 0x4f00;
        if (sf.check(mutated, g_pairs, beta_bits, null)) |_| {
            std.debug.print("FAIL conv-decode: a planted wrong split output passed\n", .{});
            return false;
        } else |_| {}
        mutated[0] = 0x7fc0;
        if (sf.check(mutated, g_pairs, beta_bits, null)) |_| {
            std.debug.print("FAIL conv-decode: a planted nonfinite split output passed\n", .{});
            return false;
        } else |_| {}
    }
    if (!try unchanged(a, cmds, staging, cf.state_bits, state, timeout) or
        !try unchanged(a, cmds, staging, x_bits, xbuf, timeout) or
        !try unchanged(a, cmds, staging, cf.weight_bits, weight, timeout) or
        !try unchanged(a, cmds, staging, sf.a_bits, a_buf, timeout) or
        !try unchanged(a, cmds, staging, sf.b_bits, b_buf, timeout) or
        !try unchanged(a, cmds, staging, sf.dt_bits, dt_buf, timeout) or
        !try unchanged(a, cmds, staging, sf.a_log_bits, a_log, timeout))
    {
        std.debug.print("FAIL conv-decode: an input changed\n", .{});
        return false;
    }
    std.debug.print("MILESTONE conv-decode elapsed_ms={d} conv worst={e} split worst={e} of bounds, " ++
        "carry-out state bit-exact, split ran on the ACTUAL activated bits, repeats byte-identical, " ++
        "inputs unchanged, planted outputs refused\n", .{ clock.ms(), cr.worst, srep.worst });
    return true;
}

/// The gated RMSNorm epilogue at rows=4.
pub fn normGate(a: std.mem.Allocator, clock: Elapsed, dev: *const vk.Device, cmds: *vk.Commands, staging: vk.Buffer, m: *vk.Gdn, timeout: u64) !bool {
    const rows: u64 = 4;
    const total: u64 = rows * fx.Model.linear_dim;
    const f = try fx.Norm.init(a, 0x57a9);
    defer f.deinit();
    var y = try uploadShell(dev, cmds, staging, f.y_bits, timeout);
    defer y.deinit();
    var z = try uploadShell(dev, cmds, staging, f.z_bits, timeout);
    defer z.deinit();
    var w = try uploadShell(dev, cmds, staging, f.w_bits, timeout);
    defer w.deinit();
    var out = try vk.Buffer.init(dev, .device, @intCast(total * 2));
    defer out.deinit();
    const y_view = try vk.tensor.View.init(&y, .bf16, 0, total);
    const z_view = try vk.tensor.View.init(&z, .bf16, 0, total);
    const w_view = try vk.tensor.View.init(&w, .bf16, 0, fx.Model.head_dim);
    const out_view = try vk.tensor.View.init(&out, .bf16, 0, total);
    const poison = try a.alloc(u16, @intCast(total));
    defer a.free(poison);
    @memset(poison, fx.poison16);
    var first: ?[]u16 = null;
    defer if (first) |x| a.free(x);
    const bits = try a.alloc(u16, @intCast(total));
    defer a.free(bits);
    for (0..repeats) |rep| {
        try vk.orExit(vk.upload(cmds, out, 0, staging, std.mem.sliceAsBytes(poison), timeout));
        try vk.orExit(m.gatedNorm(cmds, out_view, y_view, z_view, w_view, rows, 1e-6, timeout));
        try fetch(cmds, staging, out, std.mem.sliceAsBytes(bits), timeout);
        if (first) |x| {
            if (std.mem.indexOfDiff(u16, x, bits)) |at| {
                std.debug.print("FAIL norm: repeat {d} differs at element {d}\n", .{ rep, at });
                return false;
            }
        } else first = try a.dupe(u16, bits);
    }
    const nrep = f.check(first.?) catch |e| {
        std.debug.print("FAIL norm: {s}\n", .{@errorName(e)});
        return false;
    };
    if (!try unchanged(a, cmds, staging, f.y_bits, y, timeout) or
        !try unchanged(a, cmds, staging, f.z_bits, z, timeout) or
        !try unchanged(a, cmds, staging, f.w_bits, w, timeout))
    {
        std.debug.print("FAIL norm: an input changed\n", .{});
        return false;
    }
    if (!try plantRefuse(f, first.?, "norm", "norm")) return false;
    std.debug.print("MILESTONE norm elapsed_ms={d} rows={d} worst={e} of bound, repeats byte-identical, inputs unchanged, planted outputs refused\n", .{ clock.ms(), rows, nrep.worst });
    return true;
}
