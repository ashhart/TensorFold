//! The actual four-kernel trainer chain is checked against step one's independent backward at served QSA geometry.
const std = @import("std");
const core = @import("core");
const n = @import("../../../tests/qwen4/numeric_gpu.zig");
const Ref = @import("train_tensor.zig").Ref(n.B);
const Slot = @import("train_buffers.zig").Slot;
const spec = core.living_qsa;
const data = @import("learning_contract_data.zig");
const ref = @import("qsa_backward.zig");
const a = std.testing.allocator;
extern fn tf_living_qsa_cpu(c_uint, spec.Params, [*][*]u8) void;
const T = struct {
    values: std.EnumArray(Slot, Ref),
    dummy: Ref,
    pub fn buf(t: *T, slot: Slot) Ref {
        return t.values.get(slot);
    }
    pub fn ops(t: *T) O {
        return .{ .t = t };
    }
};
const O = struct {
    const Binding = struct { usize, Ref };
    t: *T,
    pub fn qsa(o: O, op: spec.Op, p: spec.Params, bindings: []const Binding) !void {
        var ptrs: [11][*]u8 = @splat(o.t.dummy.buffer.ptr);
        var ranges: [11]core.weight_only.Range = @splat(.{ .address = @intFromPtr(o.t.dummy.buffer.ptr), .len = o.t.dummy.len });
        var seen: u16 = 0;
        for (bindings) |v| {
            if (v[0] >= 11 or seen & (@as(u16, 1) << @intCast(v[0])) != 0) return error.Binding;
            seen |= @as(u16, 1) << @intCast(v[0]);
            ptrs[v[0]] = v[1].buffer.ptr + v[1].offset;
            ranges[v[0]] = .{ .address = @intFromPtr(ptrs[v[0]]), .len = v[1].len };
        }
        const finished = try spec.finish(op, p);
        try spec.validate(op, finished, ranges);
        tf_living_qsa_cpu(@intCast(@backingInt(op)), finished, &ptrs);
    }
};
fn r(bytes: []align(16) u8) Ref {
    return .{ .buffer = bytes, .len = bytes.len };
}
fn widened(mem: std.mem.Allocator, f: anytype, index: usize) ![]f64 {
    const out = try mem.alloc(f64, f.counts[index]);
    for (out, 0..) |*v, i| v.* = if (index < 3)
        @as(f32, @bitCast(@as(u32, std.mem.readInt(u16, f.raw.data[index][i * 2 ..][0..2], .little)) << 16))
    else
        f.raw.floats(index)[i];
    return out;
}
test "Qwen4 learning: actual QSA chain agrees with step one on 24 over 4 heads at width 256" {
    const rows = 17;
    const heads = 24;
    const kv = 4;
    const dim = 256;
    var p = data.params(spec, .scores, rows, dim);
    p.u[1] = heads;
    p.u[2] = kv;
    p.u[5] = 0;
    var source = try data.Fixture(spec).init(a, .scores, p, 0x19033, false);
    defer source.deinit();
    @memset(source.raw.floats(3)[0 .. (rows - 2) * heads * dim], 0);
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    const mem = arena.allocator();
    const dummy = r(source.raw.data[7]);
    var t: T = .{ .values = .initFill(dummy), .dummy = dummy };
    inline for (std.enums.values(Slot)) |slot| {
        const bytes = switch (slot) {
            .qsa_context, .qsa_q => rows * heads * dim * 4,
            .qsa_scores, .qsa_dp, .qsa_p, .qsa_ds => rows * heads * rows * 4,
            .qsa_k, .qsa_v => rows * kv * dim * 4,
            else => 0,
        };
        if (bytes > 0) {
            const buffer = try mem.alignedAlloc(u8, .@"16", bytes);
            @memset(buffer, 0);
            t.values.set(slot, r(buffer));
        }
    }
    @memcpy(t.buf(.qsa_context).buffer, source.raw.data[3]);
    try @import("train_mixers.zig").selected(&t, p, r(source.raw.data[0]), r(source.raw.data[1]), r(source.raw.data[2]), r(source.raw.data[4]), r(source.raw.data[5]), r(source.raw.data[6]));
    const q = try widened(mem, &source, 0);
    const k = try widened(mem, &source, 1);
    const v = try widened(mem, &source, 2);
    const dy = try widened(mem, &source, 3);
    const dq = try mem.alloc(f64, q.len);
    const dk = try mem.alloc(f64, k.len);
    const dv = try mem.alloc(f64, v.len);
    const ids = std.mem.bytesAsSlice(u32, source.raw.data[4]);
    const counts = std.mem.bytesAsSlice(u32, source.raw.data[5]);
    const flags = std.mem.bytesAsSlice(u32, source.raw.data[6]);
    var masks: [rows]ref.Row = undefined;
    for (&masks, 0..) |*mask, row| mask.* = .{ .end = row + 1, .ids = if (flags[row] != 0) ids[row * rows ..][0..counts[row]] else null };
    try ref.backward(f64, mem, .{ .rows = rows, .heads = heads, .kv = kv, .width = dim, .capacity = rows, .scale = 0.0625 }, &masks, q, k, v, dy, dq, dk, dv);
    inline for (.{ .qsa_q, .qsa_k, .qsa_v }, .{ dq, dk, dv }) |slot, expected| {
        const actual = std.mem.bytesAsSlice(f32, t.buf(slot).buffer);
        for (actual, expected) |x, want| try @import("learning_fixture.zig").near(x, want);
    }
    var prefix_magnitude: f64 = 0;
    for (dk[0 .. kv * dim], dv[0 .. kv * dim]) |x, y| prefix_magnitude += @abs(x) + @abs(y);
    try std.testing.expect(prefix_magnitude > 0);
    for (dq[0 .. (rows - 2) * heads * dim]) |x| try std.testing.expectEqual(@as(f64, 0), x);
}
