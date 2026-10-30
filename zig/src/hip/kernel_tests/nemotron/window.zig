//! Nemotron's Mamba window launches against the mixer a row at a time, each state copied out but the last.

const std = @import("std");
const rig = @import("../rig.zig");
const raw = @import("../../raw.zig");
const DeviceBuffer = @import("../../memory.zig").DeviceBuffer;
const mamba = @import("../../ops/mamba.zig");
const Tensor = @import("../../ops/types.zig").Tensor;

const Rig = rig.Rig;
const gpa = rig.gpa;
const at = rig.at;

const lim: mamba.Limits = .{ .dt_min = 0.001, .dt_max = 0.1, .eps = 1e-5 };

/// One layer's weights on the device, drawn as the checkpoint's are spread.
const Weights = struct {
    bufs: [6]DeviceBuffer,

    fn make(t: *Rig, rng: *rig.Rng, sh: mamba.Shape) !Weights {
        const sizes = [6]usize{ 4 * sh.cd(), sh.cd(), sh.heads, sh.heads, sh.heads, sh.xd() };
        const spread = [6]f64{ 0.5, 0.1, 0.1, 2.5, 0.1, 0.1 };
        var w: Weights = undefined;
        for (sizes, spread, 0..) |n, s, i| {
            const host = try gpa.alloc(f32, n);
            defer gpa.free(host);
            for (host) |*v| v.* = @floatCast(switch (i) {
                3 => s * rng.uniform(),
                4, 5 => 1 + s * rng.normal(),
                else => s * rng.normal(),
            });
            w.bufs[i] = try t.upload(host);
        }
        return w;
    }

    fn free(w: *Weights) void {
        for (&w.bufs) |*b| b.free();
    }

    fn of(w: Weights) mamba.Weights {
        const b = w.bufs;
        return .{
            .conv_w = at(b[0]),
            .conv_b = at(b[1]),
            .dt_bias = at(b[2]),
            .a_log = at(b[3]),
            .d = at(b[4]),
            .gnorm = at(b[5]),
        };
    }
};

/// A stream's layer state ([conv tail | SSM state], as a snapshot slot lays it out), its slots and the outputs.
const Run = struct {
    state: DeviceBuffer,
    snaps: DeviceBuffer,
    out: DeviceBuffer,
    xc: DeviceBuffer,
    dt: DeviceBuffer,
    y: DeviceBuffer,

    fn make(t: *Rig, host_state: []const u8, sh: mamba.Shape, rows: usize, stride: usize) !Run {
        var r: Run = .{
            .state = try t.upload(host_state),
            .snaps = try t.alloc((rows + 1) * stride),
            .out = try t.alloc(rows * sh.xd() * 2),
            .xc = try t.alloc(rows * sh.cd() * 2),
            .dt = try t.alloc(rows * sh.heads * 4),
            .y = try t.alloc(rows * sh.xd() * 2),
        };
        try r.snaps.fill8(0);
        try r.out.fill8(0);
        return r;
    }

    fn free(r: *Run) void {
        for ([_]*DeviceBuffer{ &r.state, &r.snaps, &r.out, &r.xc, &r.dt, &r.y }) |b| b.free();
    }

    fn st(r: Run, conv_bytes: usize) mamba.State {
        return .{ .conv = at(r.state), .ssm = at(r.state) + conv_bytes };
    }
};

fn same(a: DeviceBuffer, b: DeviceBuffer) !bool {
    const x = try rig.download(u8, a);
    defer gpa.free(x);
    const y = try rig.download(u8, b);
    defer gpa.free(y);
    return std.mem.eql(u8, x, y);
}

/// Window `rows` of shape `sh` both ways, with snapshots or without: outputs, states and slots byte for byte.
fn window(t: *Rig, rng: *rig.Rng, sh: mamba.Shape, rows: usize, snaps: bool) !bool {
    const kind: @import("../../ops/types.zig").Kind = if (t.bf16) .bf16 else .f16;
    const conv_bytes = 3 * sh.cd() * 2;
    const each = conv_bytes + sh.heads * sh.dh * 128 * 4;
    const stride = each + 256;
    const p = try gpa.alloc(u16, rows * sh.proj());
    defer gpa.free(p);
    for (p, 0..) |*v, i| {
        const col = i % sh.proj();
        const zx = col < 2 * sh.xd();
        const f: f32 = if (zx) rng.unit() * 2 else if (col < sh.xd() + sh.cd()) rng.unit() else rng.unit() * 4 - 3;
        v.* = t.bits(f);
    }
    const host_state = try gpa.alignedAlloc(u8, .fromByteUnits(16), each);
    defer gpa.free(host_state);
    const conv: []u16 = @alignCast(std.mem.bytesAsSlice(u16, host_state[0..conv_bytes]));
    for (conv) |*v| v.* = t.bits(rng.unit() * 2);
    const ssm: []f32 = @alignCast(std.mem.bytesAsSlice(f32, host_state[conv_bytes..]));
    for (ssm) |*v| v.* = @floatCast(0.1 * rng.normal());
    var w = try Weights.make(t, rng, sh);
    defer w.free();
    var src = try t.upload(p);
    defer src.free();
    const x: Tensor = .{ .ptr = at(src), .kind = kind };
    var old = try Run.make(t, host_state, sh, rows, stride);
    defer old.free();
    var new = try Run.make(t, host_state, sh, rows, stride);
    defer new.free();
    const o = t.ops(&t.on);
    const s = t.stream.handle;
    if (snaps) {
        for (0..rows) |r| {
            const row: Tensor = .{ .ptr = x.ptr + r * sh.proj() * 2, .kind = kind };
            const sc: mamba.Scratch = .{
                .xc = at(old.xc) + r * sh.cd() * 2,
                .dt = at(old.dt) + r * sh.heads * 4,
                .y = at(old.y) + r * sh.xd() * 2,
            };
            try mamba.mambaMixer(o, row, old.st(conv_bytes), sc, at(old.out) + r * sh.xd() * 2, w.of(), 1, sh, lim);
            if (r + 1 < rows) try raw.copy(old.snaps, (r + 1) * stride, at(old.state), each, s);
        }
    } else {
        const sc: mamba.Scratch = .{ .xc = at(old.xc), .dt = at(old.dt), .y = at(old.y) };
        try mamba.mambaMixer(o, x, old.st(conv_bytes), sc, at(old.out), w.of(), rows, sh, lim);
    }
    const sn: ?mamba.Snap = if (snaps) .{ .conv = at(
        new.snaps,
    ), .ssm = at(new.snaps) + conv_bytes, .stride = stride } else null;
    const sc: mamba.Scratch = .{ .xc = at(new.xc), .dt = at(new.dt), .y = at(new.y) };
    try mamba.mambaWindow(o, x, new.st(conv_bytes), sc, at(new.out), w.of(), rows, sh, lim, sn);
    try t.stream.synchronize();
    return try same(old.out, new.out) and try same(old.state, new.state) and try same(old.snaps, new.snaps);
}

test "the Mamba window's two launches write the bytes of the mixer a row at a time, its states and snapshots" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    var rng: rig.Rng = .{ .state = 0x5CA9 };
    const shapes = [_]mamba.Shape{
        .{ .heads = 64, .dh = 64, .groups = 8 },
        .{ .heads = 32, .dh = 64, .groups = 4 },
        .{ .heads = 16, .dh = 64, .groups = 2 },
        .{ .heads = 4, .dh = 64, .groups = 2 },
    };
    var cases: usize = 0;
    for (shapes) |sh| for ([_]usize{ 1, 2, 3, 4, 7, 17 }) |rows| for ([_]bool{ true, false }) |snaps| {
        try rig.expect(
            try window(t, &rng, sh, rows, snaps),
            "mamba window {d} heads, {d} groups, {d} rows, snapshots {}: the bytes differ",
            .{ sh.heads, sh.groups, rows, snaps },
        );
        cases += 1;
    };
    std.debug.print("  nemotron mamba window: {d} cases, the bytes of the mixer a row at a time\n", .{cases});
}

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

test "the Mamba window's time against the mixer's four launches, Nemotron's layer at tp 1" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    var rng: rig.Rng = .{ .state = 0x7171 };
    const sh: mamba.Shape = .{ .heads = 64, .dh = 64, .groups = 8 };
    const kind: @import("../../ops/types.zig").Kind = if (t.bf16) .bf16 else .f16;
    const conv_bytes = 3 * sh.cd() * 2;
    const each = conv_bytes + sh.heads * sh.dh * 128 * 4;
    for ([_]usize{ 1, 4 }) |rows| {
        var w = try Weights.make(t, &rng, sh);
        defer w.free();
        var src = try t.alloc(rows * sh.proj() * 2);
        defer src.free();
        try src.fill8(0x10);
        const host = try gpa.alignedAlloc(u8, .fromByteUnits(16), each);
        defer gpa.free(host);
        @memset(host, 0);
        var r = try Run.make(t, host, sh, rows, each);
        defer r.free();
        const x: Tensor = .{ .ptr = at(src), .kind = kind };
        const o = t.ops(&t.on);
        const sc: mamba.Scratch = .{ .xc = at(r.xc), .dt = at(r.dt), .y = at(r.y) };
        var us: [2]f64 = undefined;
        for (0..2) |which| {
            const n = 500;
            var t0: u64 = 0;
            for (0..n + 10) |i| {
                if (i == 10) {
                    try t.stream.synchronize();
                    t0 = nowNs();
                }
                if (which == 0) {
                    try mamba.mambaMixer(o, x, r.st(conv_bytes), sc, at(r.out), w.of(), rows, sh, lim);
                } else {
                    try mamba.mambaWindow(o, x, r.st(conv_bytes), sc, at(r.out), w.of(), rows, sh, lim, null);
                }
            }
            try t.stream.synchronize();
            us[which] = @as(f64, @floatFromInt(nowNs() - t0)) / n / 1e3;
        }
        std.debug.print(
            "  nemotron mamba window, {d} rows: {d:.2} us the four launches, {d:.2} us the window (back to back)\n",
            .{ rows, us[0], us[1] },
        );
    }
}
