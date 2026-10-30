//! Nemotron-H's merged decode launches against the launches they replace: the same bytes, every shape they take.

const std = @import("std");
const rig = @import("../rig.zig");
const DeviceBuffer = @import("../../memory.zig").DeviceBuffer;
const moe_launch = @import("../../launches/nemotron_moe.zig");

const Rig = rig.Rig;
const gpa = rig.gpa;
const at = rig.at;

fn ptr(b: DeviceBuffer) ?*anyopaque {
    return @ptrFromInt(at(b));
}

/// A 16-bit activation pattern of `kind` (1 fp16, 2 bf16) from a float, rounded to nearest even.
fn bitsOf(kind: c_int, v: f32) u16 {
    if (kind == 1) return rig.f16Bits(v);
    const u: u32 = @bitCast(v);
    return @intCast((u + (((u >> 16) & 1) + 0x7fff)) >> 16);
}

/// Residual rows: mostly normal values, some large enough to round to infinity, a few NaN and infinite ones.
fn residualRows(kind: c_int, rng: *rig.Rng, n: usize, scale: f32, specials: bool) ![]u16 {
    const out = try gpa.alloc(u16, n);
    for (out, 0..) |*v, i| {
        const big: f32 = if (kind == 1) 40000 else 1e38;
        var f: f32 = @floatCast(rng.normal() * scale);
        if (specials and i % 997 == 13) f = big * @as(f32, if (i % 2 == 0) 1 else -1);
        if (specials and i % 4093 == 77) f = std.math.nan(f32);
        if (specials and i % 5011 == 91) f = std.math.inf(f32);
        v.* = bitsOf(kind, f);
    }
    return out;
}

/// tf_add (or tf_add_wide) then tf_rms, against tf_add_rms: x and the normed rows, byte for byte.
fn addRms(t: *Rig, rng: *rig.Rng, kind: c_int, rows: usize, width: usize, wide: bool, specials: bool) !bool {
    const n = rows * width;
    const eps: f32 = 1e-5;
    const hx = try residualRows(kind, rng, n, 3, specials);
    defer gpa.free(hx);
    const hy16 = try residualRows(kind, rng, if (wide) 0 else n, 0.5, specials);
    defer gpa.free(hy16);
    const hy32 = try gpa.alloc(f32, if (wide) n else 0);
    defer gpa.free(hy32);
    for (hy32) |*v| v.* = @floatCast(rng.normal() * 0.5);
    const hw = try gpa.alloc(f32, width);
    defer gpa.free(hw);
    for (hw) |*v| v.* = 1.0 + rng.unit() / 4.0;
    var y = if (wide) try t.upload(hy32) else try t.upload(hy16);
    defer y.free();
    var weight = try t.upload(hw);
    defer weight.free();
    var xs: [2]DeviceBuffer = undefined;
    var normed: [2]DeviceBuffer = undefined;
    for (0..2) |v| {
        xs[v] = try t.upload(hx);
        normed[v] = try t.alloc(n * 2);
    }
    defer for (0..2) |v| {
        xs[v].free();
        normed[v].free();
    };
    const s = t.stream.handle;
    const w: ?[*]const f32 = @ptrFromInt(at(weight));
    if (wide) {
        try t.off.tf_add_wide(ptr(xs[0]), @ptrFromInt(at(y)), ptr(xs[0]), kind, @intCast(n), s);
    } else {
        try t.off.tf_add(ptr(xs[0]), ptr(y), ptr(xs[0]), kind, @intCast(n), s);
    }
    try t.off.tf_rms(ptr(xs[0]), w, ptr(normed[0]), kind, @intCast(rows), @intCast(width), eps, s);
    try t.on.tf_add_rms(
        ptr(xs[1]),
        .{ .y = at(y), .mode = @intFromBool(wide) },
        w,
        ptr(normed[1]),
        kind,
        @intCast(rows),
        @intCast(width),
        eps,
        s,
    );
    try t.stream.synchronize();
    var got: [4][]u16 = undefined;
    for (0..2) |v| {
        got[2 * v] = try rig.download(u16, xs[v]);
        got[2 * v + 1] = try rig.download(u16, normed[v]);
    }
    defer for (got) |g| gpa.free(g);
    return std.mem.eql(u16, got[0], got[2]) and std.mem.eql(u16, got[1], got[3]);
}

/// tf_nemo_combine, tf_add and tf_rms against tf_add_rms over the experts' rows: x and the normed rows, byte for byte.
fn moeRms(t: *Rig, rng: *rig.Rng, kind: c_int, rows: usize, width: usize, slots: usize, shared: bool) !bool {
    const n = rows * width;
    const eps: f32 = 1e-5;
    const hx = try residualRows(kind, rng, n, 3, false);
    defer gpa.free(hx);
    const hy = try gpa.alloc(f32, rows * slots * width);
    defer gpa.free(hy);
    for (hy) |*v| v.* = @floatCast(rng.normal() * 0.5);
    const hs = try gpa.alloc(f32, if (shared) n else 1);
    defer gpa.free(hs);
    for (hs) |*v| v.* = @floatCast(rng.normal() * 0.3);
    const hwts = try gpa.alloc(f32, rows * slots);
    defer gpa.free(hwts);
    for (hwts) |*v| v.* = @floatCast(rng.uniform() * 2.5);
    const hw = try gpa.alloc(f32, width);
    defer gpa.free(hw);
    for (hw) |*v| v.* = 1.0 + rng.unit() / 4.0;
    var y = try t.upload(hy);
    defer y.free();
    var sh = try t.upload(hs);
    defer sh.free();
    var wts = try t.upload(hwts);
    defer wts.free();
    var weight = try t.upload(hw);
    defer weight.free();
    var sum = try t.alloc(n * 2);
    defer sum.free();
    var xs: [2]DeviceBuffer = undefined;
    var normed: [2]DeviceBuffer = undefined;
    for (0..2) |v| {
        xs[v] = try t.upload(hx);
        normed[v] = try t.alloc(n * 2);
    }
    defer for (0..2) |v| {
        xs[v].free();
        normed[v].free();
    };
    const s = t.stream.handle;
    const w: ?[*]const f32 = @ptrFromInt(at(weight));
    const sp: u64 = if (shared) at(sh) else 0;
    try moe_launch.combine(
        &t.off,
        @ptrFromInt(at(y)),
        @ptrFromInt(at(wts)),
        @ptrFromInt(sp),
        ptr(sum),
        kind,
        @intCast(rows),
        @intCast(slots),
        @intCast(width),
        s,
    );
    try t.off.tf_add(ptr(xs[0]), ptr(sum), ptr(xs[0]), kind, @intCast(n), s);
    try t.off.tf_rms(ptr(xs[0]), w, ptr(normed[0]), kind, @intCast(rows), @intCast(width), eps, s);
    const in: @import(
        "../../launches/norms.zig",
    ).AddRmsIn = .{ .y = at(y), .mode = 2, .wts = at(wts), .shared = sp, .slots = @intCast(slots) };
    try t.on.tf_add_rms(ptr(xs[1]), in, w, ptr(normed[1]), kind, @intCast(rows), @intCast(width), eps, s);
    try t.stream.synchronize();
    var got: [4][]u16 = undefined;
    for (0..2) |v| {
        got[2 * v] = try rig.download(u16, xs[v]);
        got[2 * v + 1] = try rig.download(u16, normed[v]);
    }
    defer for (got) |g| gpa.free(g);
    return std.mem.eql(u16, got[0], got[2]) and std.mem.eql(u16, got[1], got[3]);
}

test "the MoE sum, the residual and the norm in one launch write the bytes of tf_nemo_combine, tf_add and tf_rms" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    var rng: rig.Rng = .{ .state = 0xC0B1 };
    var cases: usize = 0;
    for ([_]c_int{ 1, 2 }) |kind| for ([_]usize{ 1, 3, 17 }) |rows| for ([_]usize{ 2688, 4096 }) |width| {
        for ([_]usize{ 1, 6, 8 }) |slots| for ([_]bool{ true, false }) |shared| {
            try rig.expect(
                try moeRms(t, &rng, kind, rows, width, slots, shared),
                "moe add rms kind {d}, {d} rows of {d}, {d} slots, shared {}: the bytes differ",
                .{ kind, rows, width, slots, shared },
            );
            cases += 1;
        };
    };
    std.debug.print(
        "  nemotron moe sum + add + rms: {d} cases, the same bytes as tf_nemo_combine, tf_add and tf_rms\n",
        .{cases},
    );
}

test "the residual and the norm in one launch write the bytes of tf_add (or tf_add_wide) and tf_rms" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    var rng: rig.Rng = .{ .state = 0xADD5 };
    var cases: usize = 0;
    for ([_]c_int{ 1, 2 }) |kind| for ([_]usize{ 1, 3, 17, 64 }) |rows| for (
        [_]usize{ 1032, 2688, 4096, 8192 },
    ) |width| {
        for ([_]bool{ false, true }) |wide| for ([_]bool{ false, true }) |specials| {
            try rig.expect(
                try addRms(t, &rng, kind, rows, width, wide, specials),
                "add rms kind {d}, {d} rows of {d}, wide {}, specials {}: the bytes differ",
                .{ kind, rows, width, wide, specials },
            );
            cases += 1;
        };
    };
    std.debug.print("  nemotron add + rms: {d} cases, the same bytes as tf_add and tf_rms\n", .{cases});
}

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

test "the residual and norm's time against tf_add and tf_rms, Nemotron's 2688-wide rows" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    const width = 2688;
    const kind = t.kind();
    for ([_]usize{ 1, 4 }) |rows| {
        var x = try t.alloc(rows * width * 2);
        defer x.free();
        var y = try t.alloc(rows * width * 2);
        defer y.free();
        var normed = try t.alloc(rows * width * 2);
        defer normed.free();
        for ([_]*DeviceBuffer{ &x, &y, &normed }) |b| try b.fill8(0x11);
        const s = t.stream.handle;
        var us: [4]f64 = undefined;
        for (0..4) |which| {
            const n = 2000;
            for (0..n + 20) |i| {
                if (i == 20) {
                    try t.stream.synchronize();
                    us[which] = @floatFromInt(nowNs());
                }
                if (which == 0) {
                    try t.off.tf_add(ptr(x), ptr(y), ptr(x), kind, @intCast(rows * width), s);
                    try t.off.tf_rms(ptr(x), null, ptr(normed), kind, @intCast(rows), width, 1e-5, s);
                } else if (which == 2) {
                    try t.off.tf_add(ptr(x), ptr(y), ptr(x), kind, @intCast(rows * width), s);
                } else if (which == 3) {
                    try t.off.tf_rms(ptr(x), null, ptr(normed), kind, @intCast(rows), width, 1e-5, s);
                } else {
                    try t.on.tf_add_rms(
                        ptr(x),
                        .{ .y = at(y), .mode = 0 },
                        null,
                        ptr(normed),
                        kind,
                        @intCast(rows),
                        width,
                        1e-5,
                        s,
                    );
                }
            }
            try t.stream.synchronize();
            us[which] = (@as(f64, @floatFromInt(nowNs())) - us[which]) / n / 1e3;
        }
        std.debug.print(
            "  nemotron add + rms, {d} rows: {d:.2} us merged, {d:.2} us as tf_add and tf_rms, {d:.2} and {d:.2} " ++
                "alone (back to back)\n",
            .{ rows, us[1], us[0], us[2], us[3] },
        );
    }
}
