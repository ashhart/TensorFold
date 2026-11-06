//! An independent binary64 recurrence freezes explicit cast offsets and differentiates its last-row loss.
const std = @import("std");
const core = @import("core");
const spec = core.living_gdn;
const C = @import("learning_fixture.zig").Case(spec);
const a = std.testing.allocator;
const nr = 3;
const nk = 1;
const nv = 2;
const ch = (2 * nk + nv) * 128;
const pw = ch + nv * 130;
fn f(x: u16) f64 {
    return @as(f32, @bitCast(@as(u32, x) << 16));
}
fn sig(x: f64) f64 {
    return 1 / (1 + @exp(-x));
}
const Cuts = struct {
    data: []f64,
    at: usize = 0,
    base: bool = true,
    fn apply(c: *Cuts, x: f64, bf16: bool) f64 {
        const at = c.at;
        c.at += 1;
        if (c.base) {
            const rounded = if (bf16) f(core.quant.convert.bf16(@floatCast(x))) else @as(f64, @as(f32, @floatCast(x)));
            c.data[at] = rounded - x;
        }
        return x + c.data[at];
    }
};
fn forward(raw: []const f64, conv: []const f32, alog: []const f32, dt: []const f32, norm: []const f32, upstream: []const f32, cuts: *Cuts, states: []f32, gate_sigmoid: bool, conv_bf16: bool, mid: bool) f64 {
    cuts.at = 0;
    var state: [nv * 128 * 128]f64 = @splat(0);
    var out: f64 = 0;
    for (0..nr) |row| {
        var front: [ch]f64 = undefined;
        for (&front, 0..) |*v, col| {
            var sum: f64 = 0;
            for (0..4) |tap| if (row + tap >= 3) {
                sum += raw[(row + tap - 3) * pw + col] * conv[col * 4 + tap];
            };
            if (conv_bf16) sum = cuts.apply(sum, true);
            v.* = cuts.apply(sum * sig(sum), true);
        }
        var qk: [2 * nk * 128]f64 = undefined;
        for (0..2 * nk) |head| {
            var square: f64 = 1e-6;
            for (front[head * 128 ..][0..128]) |v| square += v * v;
            const inv = (if (head < nk) 1 / @sqrt(@as(f64, 128)) else @as(f64, 1)) / @sqrt(square);
            for (0..128) |i| qk[head * 128 + i] = cuts.apply(front[head * 128 + i] * inv, false);
        }
        for (0..nv) |head| {
            const rate = @exp(@as(f64, alog[head]));
            const ar = raw[row * pw + ch + nv * 129 + head] + dt[head];
            const decay = @exp(-rate * @log(1 + @exp(ar)));
            const beta = cuts.apply(sig(raw[row * pw + ch + nv * 128 + head]), true);
            const kh = head / (nv / nk);
            const query = qk[kh * 128 ..][0..128];
            const key = qk[(nk + kh) * 128 ..][0..128];
            var y: [128]f64 = undefined;
            for (&y, 0..) |*value, d| {
                const s = state[(head * 128 + d) * 128 ..][0..128];
                var predicted: f64 = 0;
                for (s, key) |v, k| predicted += decay * v * k;
                const correction = beta * (front[2 * nk * 128 + head * 128 + d] - predicted);
                var z: f64 = 0;
                for (s, key, query, 0..) |*v, k, q, i| {
                    v.* = cuts.apply(decay * v.* + k * correction, false);
                    if (cuts.base) states[((row * nv + head) * 128 + d) * 128 + i] = @floatCast(v.*);
                    z += v.* * q;
                }
                value.* = cuts.apply(z, true);
            }
            var ss: f64 = 0;
            for (y) |v| ss += v * v;
            const inv = 1 / @sqrt(ss / 128 + 1e-6);
            for (y, 0..) |v, i| {
                var weighted = cuts.apply(v * inv, true) * norm[i];
                if (mid) weighted = cuts.apply(weighted, true);
                const raw_gate = raw[row * pw + ch + head * 128 + i];
                const gate = sig(raw_gate) * (if (gate_sigmoid) @as(f64, 1) else raw_gate);
                out += weighted * gate * upstream[(row * nv + head) * 128 + i];
            }
        }
    }
    return out;
}
fn check(gate: bool, conv_bf: bool, mid: bool) !void {
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    const mem = arena.allocator();
    var p: spec.Params = .{};
    p.u[0] = nr;
    p.u[1] = nk;
    p.u[2] = nv;
    p.u[3] = pw;
    p.u[4] = @intFromBool(conv_bf);
    p.u[5] = @intFromBool(gate);
    p.u[6] = @intFromBool(mid);
    p.f[0] = 1e-6;
    var front = try C.init(.front, p);
    defer front.deinit();
    var normed = try C.init(.norm, p);
    defer normed.deinit();
    var readout = try C.init(.readout, p);
    defer readout.deinit();
    var scan = try C.init(.scan, p);
    defer scan.deinit();
    var qk = try C.init(.qk, p);
    defer qk.deinit();
    var gates = try C.init(.gates, p);
    defer gates.deinit();
    var conv = try C.init(.conv, p);
    defer conv.deinit();
    const raw = try mem.alloc(f64, nr * pw);
    const bits = std.mem.bytesAsSlice(u16, front.data[0]);
    for (raw, 0..) |*v, i| {
        const x = @as(f32, @floatFromInt(@as(i32, @intCast((i * 11 + i / 17) % 83)) - 41)) * 0.015625;
        bits[i] = core.quant.convert.bf16(x);
        v.* = f(bits[i]);
    }
    for (front.floats(1), 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast((i * 7) % 19)) - 9)) * 0.03125;
    const alog = try mem.alloc(f32, nv);
    const dt = try mem.alloc(f32, nv);
    for (alog, dt, 0..) |*al, *d, i| {
        al.* = -1 + @as(f32, @floatFromInt(i)) * 0.125;
        d.* = 0.25;
    }
    const weights = readout.floats(3)[0..128];
    for (weights, 0..) |*v, i| v.* = 0.5 + @as(f32, @floatFromInt(i % 7)) * 0.0625;
    const up = readout.floats(4)[0 .. nr * nv * 128];
    @memset(up, 0);
    for (up[(nr - 1) * nv * 128 ..], 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 13)) - 6)) * 0.0078125;
    const states = std.mem.bytesAsSlice(f32, readout.data[2])[0 .. nr * nv * 128 * 128];
    var cuts: Cuts = .{ .data = try mem.alloc(f64, 300000) };
    _ = forward(raw, front.floats(1), alog, dt, weights, up, &cuts, states, gate, conv_bf, mid);
    cuts.base = false;
    try front.run();
    try normed.put(0, front.data[8]);
    try normed.run();
    try readout.put(0, front.data[0]);
    try readout.put(1, normed.data[8]);
    try readout.run();
    try scan.put(0, front.data[8]);
    try scan.put(1, normed.data[8]);
    try scan.put(2, readout.data[2]);
    try scan.put(3, readout.data[8]);
    try scan.put(4, front.data[0]);
    try scan.put(5, std.mem.sliceAsBytes(alog));
    try scan.put(6, std.mem.sliceAsBytes(dt));
    try scan.run();
    try qk.put(0, front.data[8]);
    try qk.put(2, scan.data[8]);
    try qk.run();
    try gates.put(0, scan.data[8]);
    try gates.put(1, front.data[0]);
    try gates.put(2, std.mem.sliceAsBytes(alog));
    try gates.put(3, std.mem.sliceAsBytes(dt));
    try gates.run();
    try conv.put(0, front.data[0]);
    try conv.put(1, front.data[1]);
    try conv.put(2, qk.data[8]);
    try conv.run();
    const coords = [_]usize{ 0, 83, 129, 2 * 128 + 9, ch + 7, ch + nv * 128, ch + nv * 129, pw + 7, pw + 2 * 128 + 11, pw + ch + nv * 128 + 1, pw + ch + nv * 129 + 1, 2 * pw + ch + 17, 2 * pw + ch + nv * 128 + 1, 2 * pw + ch + nv * 129 };
    for (coords) |coord| {
        const col = coord % pw;
        const grad = if (col < ch) conv.floats(8)[coord] else if (col < ch + nv * 128) readout.floats(9)[coord] else gates.floats(8)[coord];
        const orig = raw[coord];
        for ([_]f64{ 0.00001, 0.000005 }) |step| {
            raw[coord] = orig + step;
            const hi = forward(raw, front.floats(1), alog, dt, weights, up, &cuts, states, gate, conv_bf, mid);
            raw[coord] = orig - step;
            const lo = forward(raw, front.floats(1), alog, dt, weights, up, &cuts, states, gate, conv_bf, mid);
            raw[coord] = orig;
            const slope = (hi - lo) / (2 * step);
            @import("learning_fixture.zig").near(grad, slope) catch |err| {
                std.debug.print("GDN_NATIVE_FD gate={} conv={} mid={} coord={d} grad={d} slope={d}\n", .{ gate, conv_bf, mid, coord, grad, slope });
                return err;
            };
        }
    }
}
test "Qwen4 learning: native GDN reverse matches independent recurrent early-row slopes in both gate modes" {
    try check(false, false, false);
    try check(true, true, true);
}
