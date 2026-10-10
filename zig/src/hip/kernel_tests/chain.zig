//! The merged linear-attention and MoE-pick launches against the chains they replace: the same bytes.
const std = @import("std");
const rig = @import("rig.zig");
const Rig = rig.Rig;
const gpa = rig.gpa;
const at = rig.at;
const DeviceBuffer = @import("../memory.zig").DeviceBuffer;

fn same(a: DeviceBuffer, b: DeviceBuffer) !void {
    const ha = try rig.download(u8, a);
    defer gpa.free(ha);
    const hb = try rig.download(u8, b);
    defer gpa.free(hb);
    try std.testing.expectEqualSlices(u8, ha, hb);
}

fn ptr(v: u64) ?*anyopaque {
    return @ptrFromInt(v);
}

/// The pick rule over `rows` rows of (experts + 1) logits: picks, weights, items and members alike.
fn select(t: *Rig, rng: *rig.Rng, rows: usize, experts: usize, top_k: usize) !void {
    const slots = top_k + 1;
    const capacity = slots + 7;
    const logits = try gpa.alloc(f32, rows * (experts + 1));
    defer gpa.free(logits);
    for (logits) |*v| v.* = rng.unit() * 4.0;
    // equal logits: the lower index wins
    for (0..rows) |r| logits[r * (experts + 1) + 5] = logits[r * (experts + 1) + 9];
    var dev_logits = try t.upload(logits);
    defer dev_logits.free();
    var picks: [2]DeviceBuffer = undefined;
    var wts: [2]DeviceBuffer = undefined;
    var items: [2]DeviceBuffer = undefined;
    var members: [2]DeviceBuffer = undefined;
    for (0..2) |v| {
        picks[v] = try t.alloc(rows * slots * 4);
        wts[v] = try t.alloc(rows * slots * 4);
        items[v] = try t.alloc(capacity * 12);
        members[v] = try t.alloc(capacity * 4);
        try members[v].fill8(0);
    }
    defer for (0..2) |v| {
        picks[v].free();
        wts[v].free();
        items[v].free();
        members[v].free();
    };
    const plan = rows == 1;
    for ([2]*const @import("../launches.zig").Launcher{ &t.off, &t.on }, 0..) |l, v| {
        try l.tf_moe_select(@ptrFromInt(at(dev_logits)), @ptrFromInt(at(picks[v])), @ptrFromInt(at(wts[v])), if (plan) @ptrFromInt(at(items[v])) else null, if (plan) @ptrFromInt(at(members[v])) else null, @intCast(capacity), @intCast(rows), @intCast(experts), @intCast(top_k), t.stream.handle);
    }
    try t.stream.synchronize();
    try same(picks[0], picks[1]);
    try same(wts[0], wts[1]);
    if (plan) {
        try same(items[0], items[1]);
        try same(members[0], members[1]);
    }
}

/// A window's conv chain (cast, conv, three column copies, q and k norms, gate) against one launch.
fn conv(t: *Rig, rng: *rig.Rng, rows: usize) !void {
    const kw: usize = 2048;
    const vw: usize = 4096;
    const ch = 2 * kw + vw;
    const kernel: usize = 4;
    const eps: f32 = 1e-6;
    const vheads: usize = 32;
    const hx = try gpa.alloc(u16, rows * ch);
    defer gpa.free(hx);
    for (hx) |*v| v.* = t.bits(rng.unit() * 2.0);
    const hw = try gpa.alloc(f32, ch * kernel);
    defer gpa.free(hw);
    for (hw) |*v| v.* = rng.unit() / 2.0;
    const hs = try gpa.alloc(f32, (kernel - 1) * ch);
    defer gpa.free(hs);
    for (hs) |*v| v.* = rng.unit();
    const hn = try gpa.alloc(f32, 128 * 2);
    defer gpa.free(hn);
    for (hn) |*v| v.* = 1.0 + rng.unit() / 4.0;
    const ha = try gpa.alloc(u16, rows * vheads);
    defer gpa.free(ha);
    for (ha) |*v| v.* = t.bits(rng.unit() * 3.0);
    const hb = try gpa.alloc(u16, rows * vheads);
    defer gpa.free(hb);
    for (hb) |*v| v.* = t.bits(rng.unit() * 3.0);
    const hlog = try gpa.alloc(f32, vheads * 2);
    defer gpa.free(hlog);
    for (hlog) |*v| v.* = rng.unit();
    var x = try t.upload(hx);
    defer x.free();
    var weight = try t.upload(hw);
    defer weight.free();
    var norm = try t.upload(hn);
    defer norm.free();
    var dev_a = try t.upload(ha);
    defer dev_a.free();
    var dev_b = try t.upload(hb);
    defer dev_b.free();
    var dev_log = try t.upload(hlog);
    defer dev_log.free();
    var xr = try t.alloc(rows * ch * 4);
    defer xr.free();
    var mixed = try t.alloc(rows * ch * 4);
    defer mixed.free();
    var bufs: [2][9]DeviceBuffer = undefined; // gate, beta, state, snaps, qc, kc, v, qn, kn
    for (0..2) |v| {
        bufs[v] = .{
            try t.alloc(rows * vheads * 4),            try t.alloc(rows * vheads * 4), try t.upload(hs),
            try t.alloc(rows * (kernel - 1) * ch * 4), try t.alloc(rows * kw * 4),     try t.alloc(rows * kw * 4),
            try t.alloc(rows * vw * 4),                try t.alloc(rows * kw * 4),     try t.alloc(rows * kw * 4),
        };
    }
    defer for (&bufs) |*set| for (set) |*b| b.free();
    const s = t.stream.handle;
    const l = &t.on;
    const heads = rows * (kw / 128);
    const kind = t.kind();
    {
        const b = bufs[0];
        try l.tf_cast(ptr(at(x)), kind, ptr(at(xr)), 0, @intCast(rows * ch), s);
        if (rows == 1) {
            try l.tf_conv_decode(@ptrFromInt(at(xr)), @ptrFromInt(at(weight)), @ptrFromInt(at(b[2])), @ptrFromInt(at(mixed)), 1, @intCast(ch), @intCast(kernel), s);
        } else {
            try l.tf_conv_rows(@ptrFromInt(at(xr)), @ptrFromInt(at(weight)), @ptrFromInt(at(b[2])), @ptrFromInt(at(mixed)), @ptrFromInt(at(b[3])), @intCast(rows), @intCast(ch), @intCast(kernel), s);
        }
        try l.tf_copy_cols(ptr(at(mixed)), @intCast(ch), 0, ptr(at(b[4])), 0, @intCast(rows), @intCast(kw), s);
        try l.tf_copy_cols(ptr(at(mixed)), @intCast(ch), @intCast(kw), ptr(at(b[5])), 0, @intCast(rows), @intCast(kw), s);
        try l.tf_copy_cols(ptr(at(mixed)), @intCast(ch), @intCast(2 * kw), ptr(at(b[6])), 0, @intCast(rows), @intCast(vw), s);
        try l.tf_rms(ptr(at(b[4])), @ptrFromInt(at(norm)), ptr(at(b[7])), 0, @intCast(heads), 128, eps, s);
        try l.tf_rms(ptr(at(b[5])), @ptrFromInt(at(norm) + 512), ptr(at(b[8])), 0, @intCast(heads), 128, eps, s);
        try l.tf_gdn_gate(ptr(at(dev_a)), ptr(at(dev_b)), kind, @ptrFromInt(at(dev_log)), @ptrFromInt(at(dev_log) + 128), @ptrFromInt(at(b[0])), @ptrFromInt(at(b[1])), @intCast(rows * 32), 32, s);
    }
    {
        const b = bufs[1];
        try l.tf_conv_split(.{
            .x = at(x),
            .kind = kind,
            .weight = at(weight),
            .state = at(b[2]),
            .states = if (rows > 1) at(b[3]) else 0,
            .qn = at(b[7]),
            .kn = at(b[8]),
            .v = at(b[6]),
            .channels = @intCast(ch),
            .kernel = @intCast(kernel),
            .rows = @intCast(rows),
            .kw = @intCast(kw),
            .vw = @intCast(vw),
            .qw = at(norm),
            .kw_w = at(norm) + 512,
            .eps = eps,
            .norm = 128,
            .ga = at(dev_a),
            .gb = at(dev_b),
            .a_log = at(dev_log),
            .dt_bias = at(dev_log) + 128,
            .gate = at(b[0]),
            .beta = at(b[1]),
            .gcount = @intCast(rows * 32),
            .heads = 32,
        }, s);
    }
    try t.stream.synchronize();
    // gate, beta, state, snapshots (several rows only), v, q norm, k norm
    for ([_]usize{ 0, 1, 2, 6, 7, 8 }) |i| try same(bufs[0][i], bufs[1][i]);
    if (rows > 1) try same(bufs[0][3], bufs[1][3]);
}

/// The linear attention's gated norm: rms then gnorm_silu against one launch.
fn gnorm(t: *Rig, rng: *rig.Rng, rows: usize) !void {
    const n = rows * 32 * 128;
    const eps: f32 = 1e-6;
    const hy = try gpa.alloc(f32, n);
    defer gpa.free(hy);
    for (hy) |*v| v.* = rng.unit() * 3.0;
    const hz = try gpa.alloc(u16, n);
    defer gpa.free(hz);
    for (hz) |*v| v.* = t.bits(rng.unit() * 4.0);
    const hn = try gpa.alloc(f32, 128);
    defer gpa.free(hn);
    for (hn) |*v| v.* = 1.0 + rng.unit() / 4.0;
    var y = try t.upload(hy);
    defer y.free();
    var z = try t.upload(hz);
    defer z.free();
    var weight = try t.upload(hn);
    defer weight.free();
    var yn = try t.alloc(n * 4);
    defer yn.free();
    var outs: [2]DeviceBuffer = .{ try t.alloc(n * 2), try t.alloc(n * 2) };
    defer for (&outs) |*o| o.free();
    const s = t.stream.handle;
    try t.on.tf_rms(ptr(at(y)), @ptrFromInt(at(weight)), ptr(at(yn)), 0, @intCast(rows * 32), 128, eps, s);
    try t.on.tf_gnorm_silu(@ptrFromInt(at(yn)), ptr(at(z)), ptr(at(outs[0])), t.kind(), @intCast(n), s);
    try t.on.tf_gnorm_out(@ptrFromInt(at(y)), @ptrFromInt(at(weight)), ptr(at(z)), ptr(at(outs[1])), t.kind(), @intCast(rows * 32), 128, eps, s);
    try t.stream.synchronize();
    try same(outs[0], outs[1]);
}

test "the merged MoE pick writes the scanning kernel's picks, weights, items and members" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    var rng: rig.Rng = .{ .state = 0x9E3779B97F4A7C15 };
    for ([_]usize{ 1, 4, 16 }) |rows| try select(t, &rng, rows, 256, 8);
    try select(t, &rng, 1, 128, 6);
}

test "the merged conv split writes the chain's state, snapshots, q, k, v, gate and beta" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    var rng: rig.Rng = .{ .state = 0x9E3779B97F4A7C15 };
    for ([_]usize{ 1, 2, 4, 16 }) |rows| try conv(t, &rng, rows);
}

test "the merged gated norm writes rms then gnorm_silu's bytes" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    var rng: rig.Rng = .{ .state = 0x9E3779B97F4A7C15 };
    for ([_]usize{ 1, 4, 16, 128 }) |rows| try gnorm(t, &rng, rows);
}
