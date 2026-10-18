//! Nemotron-H's relu^2 as the stream tile's epilogue against the product and tf_nemo_relu2: the same bytes.

const std = @import("std");
const rig = @import("../rig.zig");
const fx = @import("fixture.zig");
const nm = @import("../../moe_nemotron.zig");
const ops = @import("../../ops/ops.zig");

const Rig = rig.Rig;
const gpa = rig.gpa;

const lightning: fx.Shape = .{ .experts = 128, .top_k = 6, .hidden = 2688, .width = 1856 };
const small: fx.Shape = .{ .experts = 64, .top_k = 6, .hidden = 512, .width = 256 };

/// The shared up of `l` over `rows` rows both ways: whether the epilogue ran, and whether the bytes are the same.
fn sharedUp(t: *Rig, o: ops.Ops, l: *const fx.Layer, x: fx.Input, rows: usize) !struct { ran: bool, same: bool } {
    const up = l.view.shared_up.?;
    const kind = fx.kind(t);
    const n = rows * up.n;
    o.arena.reset();
    const want: ops.Tensor = .{ .ptr = try o.arena.take(n * 2), .kind = kind };
    const u = try o.project(x.tensor(t, 0, up.k), up, rows, true);
    try nm.relu2(o, u.ptr, want, n);
    const got: ops.Tensor = .{ .ptr = try o.arena.take(n * 2), .kind = kind };
    const ran = try o.projectRelu2(x.tensor(t, 0, up.k), up, rows, got);
    if (!ran) return .{ .ran = false, .same = true };
    const a = try fx.bytesAt(t, want.ptr, n * 2);
    defer gpa.free(a);
    const b = try fx.bytesAt(t, got.ptr, n * 2);
    defer gpa.free(b);
    return .{ .ran = true, .same = std.mem.eql(u8, a, b) };
}

/// The whole layer with the epilogues against the layer with relu^2 as its own launches, over `rows` rows of x.
fn layer(t: *Rig, o: ops.Ops, view: nm.Layer, x: fx.Input, rows: usize) !bool {
    const d: usize = view.fc2.n;
    const size: usize = if (view.share) 4 else 2;
    var plain = view;
    plain.epilogue = false;
    o.arena.reset();
    const y0 = try nm.run(o, plain, x.tensor(t, 0, d), rows);
    const a = try fx.bytesAt(t, y0.ptr, rows * d * size);
    defer gpa.free(a);
    o.arena.reset();
    const y1 = try nm.run(o, view, x.tensor(t, 0, d), rows);
    const b = try fx.bytesAt(t, y1.ptr, rows * d * size);
    defer gpa.free(b);
    return std.mem.eql(u8, a, b);
}

test "the relu^2 epilogue writes the bytes of the product and tf_nemo_relu2: the shared up and whole layers" {
    const t = try Rig.open(256 << 20);
    defer t.close();
    var rng: rig.Rng = .{ .state = 0xE91 };
    var o = t.ops(&t.on);
    o.window = true;
    var cases: usize = 0;
    var ran: usize = 0;
    for ([_]fx.Shape{ lightning, small }) |s| {
        const l = try fx.Layer.init(t, &rng, s);
        defer l.deinit();
        for ([_]usize{ 1, 2, 4, 8, 17, 32 }) |rows| {
            var x = try fx.Input.init(t, &rng, rows, s.hidden);
            defer x.deinit();
            const r = try sharedUp(t, o, l, x, rows);
            try rig.expect(r.same, "relu2 shared up, hidden {d}, {d} rows: the bytes differ", .{ s.hidden, rows });
            ran += @intFromBool(r.ran);
            const half = s.experts / 2;
            for ([_]nm.Layer{ l.view, l.rank(0, half, true, true), l.rank(half, half, false, true) }) |view| {
                try rig.expect(
                    try layer(t, o, view, x, rows),
                    "relu2 layer, hidden {d}, {d} rows, experts from {d}: the bytes differ",
                    .{ s.hidden, rows, view.first },
                );
                cases += 1;
            }
        }
    }
    try std.testing.expect(ran > 0);
    std.debug.print(
        "  nemotron relu2 epilogue: {d} layers and {d} shared ups by the epilogue, the same bytes\n",
        .{ cases, ran },
    );
}

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

test "the relu^2 epilogue's time against the layer with relu^2 as its own launches" {
    const t = try Rig.open(256 << 20);
    defer t.close();
    var rng: rig.Rng = .{ .state = 0x71E };
    var o = t.ops(&t.on);
    o.window = true;
    const l = try fx.Layer.init(t, &rng, lightning);
    defer l.deinit();
    for ([_]usize{ 1, 2, 3, 4, 8 }) |rows| {
        var x = try fx.Input.init(t, &rng, rows, lightning.hidden);
        defer x.deinit();
        var us: [2]f64 = undefined;
        for (0..2) |which| {
            var view = l.view;
            view.epilogue = which == 1;
            const n = 200;
            var t0: u64 = 0;
            for (0..n + 10) |i| {
                if (i == 10) {
                    try t.stream.synchronize();
                    t0 = nowNs();
                }
                o.arena.reset();
                _ = try nm.run(o, view, x.tensor(t, 0, lightning.hidden), rows);
            }
            try t.stream.synchronize();
            us[which] = @as(f64, @floatFromInt(nowNs() - t0)) / n / 1e3;
        }
        std.debug.print(
            "  nemotron moe layer, {d} rows: {d:.1} us with the relu^2 epilogue, {d:.1} us without (back to back)\n",
            .{ rows, us[1], us[0] },
        );
    }
}
