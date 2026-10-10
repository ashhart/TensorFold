//! MoE block on real layer-1 weights: router logits, top-6 routing, routed/shared expert MLPs, combine, full chain.

const std = @import("std");
const xpu = @import("xpu");
const rt = @import("rt.zig");
const fixture = @import("fixture.zig");

const spv = xpu.kernels.moe;
var gate_w: []const u8 = &.{};
var bias: []const u8 = &.{};
var fx: []const u8 = &.{};
var f_logits: []const u8 = &.{};
var f_ids: []const u8 = &.{};
var f_wts: []const u8 = &.{};
var tie_logits: []const u8 = &.{};
var tie_ids: []const u8 = &.{};
var tie_wts: []const u8 = &.{};
var slot_idx: []const u8 = &.{};
var fc1 = [3][]const u8{ &.{}, &.{}, &.{} };
var fc2 = [3][]const u8{ &.{}, &.{}, &.{} };
var sh_up = [3][]const u8{ &.{}, &.{}, &.{} };
var sh_dn = [3][]const u8{ &.{}, &.{}, &.{} };
var f_act: []const u8 = &.{};
var f_y: []const u8 = &.{};
var f_sh_act: []const u8 = &.{};
var f_sh_y: []const u8 = &.{};
var f_delta: []const u8 = &.{};
var comb_y: []const u8 = &.{};
var comb_s: []const u8 = &.{};
var comb_delta: []const u8 = &.{};

const hidden: u32 = 2688;
const width: u32 = 1856;
const shared_width: u32 = 3712;
const n_experts: u32 = 128;
const top_k: u32 = 6;
const keep: u32 = 2; // routed experts present in the fixture
const scaling: f32 = 2.5;

/// Embedded bytes carry no alignment guarantee; copies them into typed storage.
fn copyAs(comptime T: type, bytes: []const u8) ![]T {
    const out = try std.heap.page_allocator.alloc(T, bytes.len / @sizeOf(T));
    @memcpy(std.mem.sliceAsBytes(out), bytes[0 .. out.len * @sizeOf(T)]);
    return out;
}

/// Distance in bf16 steps, with the sign folded into a monotonic key.
fn ulps(a: u16, b: u16) u32 {
    const ka: i32 = if (a & 0x8000 != 0) -@as(i32, a & 0x7fff) else a;
    const kb: i32 = if (b & 0x8000 != 0) -@as(i32, b & 0x7fff) else b;
    return @abs(ka - kb);
}

fn checkBf16(name: []const u8, got: []const u16, want: []const u16) !void {
    var worst: u32 = 0;
    var differ: usize = 0;
    for (got, want) |g, w| {
        const d = ulps(g, w);
        if (d > 0) differ += 1;
        worst = @max(worst, d);
    }
    std.debug.print("{s}: {d} values, {d} differ, worst {d} bf16 ulp\n", .{ name, got.len, differ, worst });
    if (worst > 1) return error.TooInaccurate;
}

/// Max |got - want| / max(|want|, floor) over fp32 values.
fn checkF32(name: []const u8, got: []const f32, want: []const f32, floor: f32, tol: f32) !void {
    var worst: f32 = 0;
    for (got, want) |g, w| worst = @max(worst, @abs(g - w) / @max(@abs(w), floor));
    std.debug.print("{s}: {d} values, max relative error {e}\n", .{ name, got.len, worst });
    if (worst > tol) return error.TooInaccurate;
}

fn checkIds(name: []const u8, got: []const u32, want: []const u32) !void {
    const ok = std.mem.eql(u32, got, want);
    std.debug.print("{s}: ids {any} {s}\n", .{ name, got, if (ok) "match" else "MISMATCH" });
    if (!ok) return error.WrongExperts;
}

const Gpu = struct {
    r: *rt.Runtime,
    m: rt.Module,
    bufs: [64]rt.Buffer = undefined,
    n: usize = 0,

    fn dev(g: *Gpu, bytes: []const u8) !rt.Buffer {
        const b = try g.empty(bytes.len);
        try g.r.upload(b, bytes);
        return b;
    }

    fn empty(g: *Gpu, bytes: usize) !rt.Buffer {
        const b = try g.r.alloc(bytes);
        g.bufs[g.n] = b;
        g.n += 1;
        return b;
    }

    fn read(g: *Gpu, comptime T: type, b: rt.Buffer, count: usize) ![]T {
        const out = try std.heap.page_allocator.alloc(T, count);
        try g.r.download(std.mem.sliceAsBytes(out), b);
        try g.r.sync();
        return out;
    }
};

/// A 4-bit table (weight, scales, biases) on the device.
const Table = struct { w: rt.Buffer, s: rt.Buffer, b: rt.Buffer };

fn table(g: *Gpu, t: [3][]const u8) !Table {
    return .{ .w = try g.dev(t[0]), .s = try g.dev(t[1]), .b = try g.dev(t[2]) };
}

fn launchLogits(g: *Gpu, x: rt.Buffer, gate: rt.Buffer, out: rt.Buffer) !void {
    var k = try g.m.kernel("router_logits", .{ 16, 1, 1 });
    defer k.deinit();
    try k.setBuffer(0, x);
    try k.setBuffer(1, gate);
    try k.setBuffer(2, out);
    try k.setU32(3, hidden);
    try k.setU32(4, n_experts);
    try k.launch(.{ n_experts, 1, 1 });
}

fn launchRoute(g: *Gpu, logits: rt.Buffer, b: rt.Buffer, ids: rt.Buffer, wts: rt.Buffer) !void {
    var k = try g.m.kernel("moe_route", .{ 16, 1, 1 });
    defer k.deinit();
    try k.setBuffer(0, logits);
    try k.setBuffer(1, b);
    try k.setBuffer(2, ids);
    try k.setBuffer(3, wts);
    try k.setU32(4, n_experts);
    try k.setU32(5, top_k);
    try k.setU32(6, @bitCast(scaling));
    try k.launch(.{ 1, 1, 1 });
}

/// Name is expert_up_relu2 or expert_down_f32; grid is (n_rows, slots).
fn launchExpert(g: *Gpu, name: [*:0]const u8, t: Table, x: rt.Buffer, ids: rt.Buffer, out: rt.Buffer, in_dim: u32, n_rows: u32, x_stride: u32, slots: u32) !void {
    var k = try g.m.kernel(name, .{ 16, 1, 1 });
    defer k.deinit();
    try k.setBuffer(0, t.w);
    try k.setBuffer(1, t.s);
    try k.setBuffer(2, t.b);
    try k.setBuffer(3, x);
    try k.setBuffer(4, ids);
    try k.setBuffer(5, out);
    try k.setU32(6, in_dim);
    try k.setU32(7, n_rows);
    try k.setU32(8, x_stride);
    try k.launch(.{ n_rows, slots, 1 });
}

fn launchCombine(g: *Gpu, y: rt.Buffer, wts: rt.Buffer, shared: rt.Buffer, out: rt.Buffer, slots: u32) !void {
    var k = try g.m.kernel("moe_combine", .{ 64, 1, 1 });
    defer k.deinit();
    try k.setBuffer(0, y);
    try k.setBuffer(1, wts);
    try k.setBuffer(2, shared);
    try k.setBuffer(3, out);
    try k.setU32(4, hidden);
    try k.setU32(5, slots);
    try k.launch(.{ (hidden + 63) / 64, 1, 1 });
}

const Chain = struct { x: rt.Buffer, gate: rt.Buffer, bias: rt.Buffer, logits: rt.Buffer, ids: rt.Buffer, wts: rt.Buffer, slots: rt.Buffer, zero: rt.Buffer, act: rt.Buffer, y: rt.Buffer, sact: rt.Buffer, sy: rt.Buffer, out: rt.Buffer };

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

pub fn run() !void {
    try loadFixtures();
    var g: Gpu = .{ .r = try rt.Runtime.init(), .m = undefined };
    defer g.r.deinit();
    g.m = try g.r.module(spv);
    defer g.m.deinit();
    defer for (g.bufs[0..g.n]) |*b| b.free();

    const x = try g.dev(fx);
    const gate = try g.dev(gate_w);
    const bias_d = try g.dev(bias);
    const want_ids = try copyAs(u32, f_ids);
    const want_wts = try copyAs(f32, f_wts);
    const want_logits = try copyAs(u16, f_logits);
    const slots = try g.dev(slot_idx);

    // router logits (fp32 sum order differs from the reference's fp64 dot: 1 bf16 ulp allowed)
    const logits = try g.empty(n_experts * 2);
    try launchLogits(&g, x, gate, logits);
    try checkBf16("router_logits", try g.read(u16, logits, n_experts), want_logits);

    // route on the reference logits, then on the GPU's own logits
    const ids = try g.empty(top_k * 4);
    const wts = try g.empty(top_k * 4);
    const ref_logits = try g.dev(f_logits);
    try launchRoute(&g, ref_logits, bias_d, ids, wts);
    try checkIds("route (real)", try g.read(u32, ids, top_k), want_ids);
    try checkF32("route weights (real)", try g.read(f32, wts, top_k), want_wts, 1e-3, 2e-6);
    try launchRoute(&g, logits, bias_d, ids, wts);
    try checkIds("route (gpu logits)", try g.read(u32, ids, top_k), want_ids);
    const zeros = try g.empty(n_experts * 4);
    const zero_bytes = std.mem.zeroes([n_experts * 4]u8);
    try g.r.upload(zeros, &zero_bytes);
    try launchRoute(&g, try g.dev(tie_logits), zeros, ids, wts);
    try checkIds("route (ties)", try g.read(u32, ids, top_k), try copyAs(u32, tie_ids));
    try checkF32("route weights (ties)", try g.read(f32, wts, top_k), try copyAs(f32, tie_wts), 1e-3, 2e-6);

    // routed experts on the fixture's reference activations, one stage at a time
    const t1 = try table(&g, fc1);
    const t2 = try table(&g, fc2);
    const act = try g.empty(keep * width * 2);
    try launchExpert(&g, "expert_up_relu2", t1, x, slots, act, hidden, width, 0, keep);
    try checkBf16("fc1+relu2 (2 experts)", try g.read(u16, act, keep * width), try copyAs(u16, f_act));
    const y = try g.empty(keep * hidden * 4);
    try launchExpert(&g, "expert_down_f32", t2, try g.dev(f_act), slots, y, width, hidden, width, keep);
    try checkF32("fc2 (2 experts)", try g.read(f32, y, keep * hidden), try copyAs(f32, f_y), 1e-3, 1e-4);

    // shared expert
    const zero_id = try g.empty(4);
    try g.r.upload(zero_id, &[_]u8{ 0, 0, 0, 0 });
    const su = try table(&g, sh_up);
    const sd = try table(&g, sh_dn);
    const sh_act = try g.empty(shared_width * 2);
    try launchExpert(&g, "expert_up_relu2", su, x, zero_id, sh_act, hidden, shared_width, 0, 1);
    try checkBf16("shared up+relu2", try g.read(u16, sh_act, shared_width), try copyAs(u16, f_sh_act));
    const sh_y = try g.empty(hidden * 4);
    try launchExpert(&g, "expert_down_f32", sd, try g.dev(f_sh_act), zero_id, sh_y, shared_width, hidden, 0, 1);
    try checkF32("shared down", try g.read(f32, sh_y, hidden), try copyAs(f32, f_sh_y), 1e-3, 1e-4);

    // combine with 6 slots of random data and the real weights
    const out = try g.empty(hidden * 2);
    try launchCombine(&g, try g.dev(comb_y), try g.dev(f_wts), try g.dev(comb_s), out, top_k);
    try checkBf16("combine (6 slots)", try g.read(u16, out, hidden), try copyAs(u16, comb_delta));

    // full chain on the GPU's own intermediates: x -> logits -> route -> experts (+ shared) -> combine over 2 slots
    const want_delta = try copyAs(u16, f_delta);
    const chain = struct {
        fn go(gg: *Gpu, b: Chain, tabs: [4]Table) !void {
            try launchLogits(gg, b.x, b.gate, b.logits);
            try launchRoute(gg, b.logits, b.bias, b.ids, b.wts);
            try launchExpert(gg, "expert_up_relu2", tabs[0], b.x, b.slots, b.act, hidden, width, 0, keep);
            try launchExpert(gg, "expert_down_f32", tabs[1], b.act, b.slots, b.y, width, hidden, width, keep);
            try launchExpert(gg, "expert_up_relu2", tabs[2], b.x, b.zero, b.sact, hidden, shared_width, 0, 1);
            try launchExpert(gg, "expert_down_f32", tabs[3], b.sact, b.zero, b.sy, shared_width, hidden, 0, 1);
            try launchCombine(gg, b.y, b.wts, b.sy, b.out, keep);
        }
    };
    const bufs: Chain = .{ .x = x, .gate = gate, .bias = bias_d, .logits = logits, .ids = ids, .wts = wts, .slots = slots, .zero = zero_id, .act = act, .y = y, .sact = sh_act, .sy = sh_y, .out = out };
    const tabs = [4]Table{ t1, t2, su, sd };
    try chain.go(&g, bufs, tabs);
    try checkIds("chain ids", try g.read(u32, ids, top_k), want_ids);
    try checkBf16("chain delta (2 routed + shared)", try g.read(u16, out, hidden), want_delta);

    // timing: 20 chains back to back (decode would run 6 slots; this one runs 2 plus shared)
    const t0 = nowNs();
    for (0..20) |_| try chain.go(&g, bufs, tabs);
    try g.r.sync();
    std.debug.print("chain: {d:.2} ms per run (20 runs, 2 routed slots + shared)\n", .{@as(f64, @floatFromInt(nowNs() - t0)) / 20e6});
}

fn loadFixtures() !void {
    gate_w = try fixture.load("moe_gate_w");
    bias = try fixture.load("moe_bias");
    fx = try fixture.load("moe_x");
    f_logits = try fixture.load("moe_logits");
    f_ids = try fixture.load("moe_ids");
    f_wts = try fixture.load("moe_wts");
    tie_logits = try fixture.load("moe_tie_logits");
    tie_ids = try fixture.load("moe_tie_ids");
    tie_wts = try fixture.load("moe_tie_wts");
    slot_idx = try fixture.load("moe_slot_idx");
    f_act = try fixture.load("moe_act");
    f_y = try fixture.load("moe_y");
    f_sh_act = try fixture.load("moe_sh_act");
    f_sh_y = try fixture.load("moe_sh_y");
    f_delta = try fixture.load("moe_delta");
    comb_y = try fixture.load("moe_comb_y");
    comb_s = try fixture.load("moe_comb_s");
    comb_delta = try fixture.load("moe_comb_delta");
    fc1[0] = try fixture.load("moe_fc1_w");
    fc1[1] = try fixture.load("moe_fc1_s");
    fc1[2] = try fixture.load("moe_fc1_b");
    fc2[0] = try fixture.load("moe_fc2_w");
    fc2[1] = try fixture.load("moe_fc2_s");
    fc2[2] = try fixture.load("moe_fc2_b");
    sh_up[0] = try fixture.load("moe_shup_w");
    sh_up[1] = try fixture.load("moe_shup_s");
    sh_up[2] = try fixture.load("moe_shup_b");
    sh_dn[0] = try fixture.load("moe_shdn_w");
    sh_dn[1] = try fixture.load("moe_shdn_s");
    sh_dn[2] = try fixture.load("moe_shdn_b");
}
