//! A stream product writes the same bytes with a round's row blocks interleaved on x as with them laid on y.

const std = @import("std");
const base = @import("../rig.zig");
const affine = @import("../../launches/affine.zig");
const Buffer = base.Buffer;
const gpa = base.gpa;
const data = @import("data.zig");
const Rig = @import("product.zig").Rig;
const Kernels = affine.Kernels;

/// Round widths past one row block: a ragged last block, whole blocks and the widest rounds.
const rounds = [_]usize{ 5, 8, 9, 16, 31, 32 };
const most_rows = 32;
const n = 1024;
const k_dim = 2048;
const group = 64;

const Kind = enum { plain, relu2, pair, routed, group };

/// The tables of `experts` stacked products of `rows` columns each, on the device.
const Tables = struct {
    words: Buffer,
    scale: Buffer,
    bias: Buffer,

    fn init(t: *Rig, rng: *data.Rng, rows: usize, bits: c_int) !Tables {
        const words_row = k_dim * @as(usize, @intCast(bits)) / 32;
        const groups = k_dim / group;
        const hw = try data.fill(gpa, u32, rows * words_row, rng, data.makeWord);
        defer gpa.free(hw);
        const hs = try data.fill(gpa, u16, rows * groups, rng, data.makeScale);
        defer gpa.free(hs);
        const hb = try data.fill(gpa, u16, rows * groups, rng, data.makeBias);
        defer gpa.free(hb);
        var words = try Buffer.fromHost(t.base, hw);
        errdefer words.free();
        var scale = try Buffer.fromHost(t.base, hs);
        errdefer scale.free();
        return .{ .words = words, .scale = scale, .bias = try Buffer.fromHost(t.base, hb) };
    }

    fn deinit(self: *Tables) void {
        self.words.free();
        self.scale.free();
        self.bias.free();
    }
};

/// One launch of `kind` through `k`: its output goes to `arg.out` (fp32) or `arg.out16` (activation type).
fn launch(t: *Rig, k: *const Kernels, kind: Kind, arg: affine.Arg, sides: []const affine.Side, items: c_int) !void {
    const r = &t.base.r;
    const s = t.base.stream.handle;
    switch (kind) {
        .plain => try k.streamGo(r, arg, .{}, arg.n, 1, s),
        .relu2 => try k.relu2Go(r, arg, 1, s),
        .pair => try k.streamGo(r, arg, .{ .pair_cols = @divExact(arg.n, 2) }, @divExact(arg.n, 2), 1, s),
        .routed => try k.routed(r, arg, items, s),
        .group => try k.groupGo(r, arg, sides, false, s),
    }
    try t.base.stream.synchronize();
}

/// `kind` at every round width, interleaved against the reference layout: the 32-bit words that differ.
fn compare(t: *Rig, rng: *data.Rng, kind: Kind, bits: c_int) !usize {
    var reference = t.kernels.*;
    reference.stream_interleave = false;
    const experts: usize = if (kind == .routed) 2 else 1;
    var tables = try Tables.init(t, rng, experts * n, bits);
    defer tables.deinit();
    const hx = try data.fill(gpa, u16, most_rows * k_dim, rng, if (t.fp16) data.makeX16 else data.makeXB);
    defer gpa.free(hx);
    var x = try Buffer.fromHost(t.base, hx);
    defer x.free();
    const bytes = most_rows * n * 4;
    var outs: [2]Buffer = .{ try Buffer.alloc(t.base, bytes), undefined };
    defer outs[0].free();
    outs[1] = try Buffer.alloc(t.base, bytes);
    defer outs[1].free();
    var members: [most_rows]c_int = undefined;
    for (&members, 0..) |*mb, i| mb.* = @intCast(i);
    var mem = try Buffer.fromHost(t.base, @as([]const c_int, &members));
    defer mem.free();
    const words_bytes = n * k_dim * @as(usize, @intCast(bits)) / 32 * 4;
    const table_bytes = n * (k_dim / group) * 2;

    var diff: usize = 0;
    for (rounds) |m| {
        // two experts split the round's rows, the first holding the most
        const first: c_int = @intCast((m + 1) / 2);
        const plan = [_]c_int{ 0, 0, first, 1, first, @as(c_int, @intCast(m)) - first };
        var items = try Buffer.fromHost(t.base, @as([]const c_int, &plan));
        defer items.free();
        for (outs, [_]*const Kernels{ t.kernels, &reference }) |out, kern| {
            try out.fill8(0xA5, null);
            var arg: affine.Arg = .{
                .x = x.ptr,
                .words = tables.words.ptr,
                .scale = .{ .p = tables.scale.ptr, .kind = 1 },
                .bias = .{ .p = tables.bias.ptr, .kind = 1 },
                .out = if (kind == .relu2 or kind == .pair) 0 else out.ptr,
                .out16 = if (kind == .relu2 or kind == .pair) out.ptr else 0,
                .m = @intCast(m),
                .n = n,
                .k = k_dim,
                .bits = bits,
                .group = group,
                .fp16 = @intFromBool(t.fp16),
            };
            if (kind == .routed) {
                arg.m = first;
                arg.route = .{ .items = items.ptr, .members = mem.ptr, .x_div = 1 };
            }
            // three sides of unequal widths over one x, each writing its own stretch of the output
            const sides = [_]affine.Side{
                .{ .words = tables.words.ptr, .scale = tables.scale.ptr, .bias = tables.bias.ptr, .n = 512, .out = out.ptr },
                .{ .words = tables.words.ptr + words_bytes / 2, .scale = tables.scale.ptr + table_bytes / 2, .bias = tables.bias.ptr + table_bytes / 2, .n = 256, .out = out.ptr + most_rows * 512 * 4 },
                .{ .words = tables.words.ptr + words_bytes * 3 / 4, .scale = tables.scale.ptr + table_bytes * 3 / 4, .bias = tables.bias.ptr + table_bytes * 3 / 4, .n = 192, .out = out.ptr + most_rows * 768 * 4 },
            };
            try launch(t, kern, kind, arg, &sides, 2);
        }
        const d = try data.sameOnDevice(outs[0], outs[1], bytes);
        if (d != 0) std.debug.print("  stream layout {t} {d}-bit, {d} rows: {d} words differ\n", .{ kind, bits, m, d });
        diff += d;
    }
    return diff;
}

test "a stream product writes the bytes of the reference layout with its row blocks interleaved on x" {
    const b = try base.Rig.open(1 << 20);
    defer b.close();
    var rig = Rig.open(b);
    var rng: data.Rng = .{ .state = 0x2545F4914F6CDD1D };
    var diff: usize = 0;
    for ([_]Kind{ .plain, .relu2, .pair, .routed, .group }) |kind| diff += try compare(&rig, &rng, kind, 4);
    diff += try compare(&rig, &rng, .plain, 3);
    try base.expect(diff == 0, "stream layout: {d} words differ from the reference layout", .{diff});
}
