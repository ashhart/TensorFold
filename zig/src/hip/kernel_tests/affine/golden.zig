//! The 4-bit BF16 stream tile writes the bytes its code pairs made one at a time wrote: a digest per architecture.

const std = @import("std");
const base = @import("../rig.zig");
const affine = @import("../../launches/affine.zig");
const kernels = @import("../../kernels.zig");
const Buffer = base.Buffer;
const gpa = base.gpa;
const data = @import("data.zig");
const prod = @import("product.zig");
const Rig = prod.Rig;

/// One row (the 1-row tile), each row block's ragged and whole sizes, and interleaved rounds.
const rounds = [_]usize{ 1, 2, 3, 4, 5, 8, 16 };
const most_rows = 16;
const n = 512;
const k_dim = 2048;
const group = 64;
const bits = 4;

/// The Wyhash of the bytes the tile wrote with stream_pair's code pairs, on each architecture that recorded it.
const pins = [_]struct { arch: []const u8, digest: u64 }{
    .{ .arch = "gfx1151", .digest = 0x24742f9ea12d63cd },
};

const Kind = enum { plain, pair, routed };

fn digestOf(t: *Rig, rng: *data.Rng, kind: Kind) !u64 {
    const experts: usize = if (kind == .routed) 2 else 1;
    const words_row = k_dim * bits / 32;
    const groups = k_dim / group;
    const hw = try data.fill(gpa, u32, experts * n * words_row, rng, data.makeWord);
    defer gpa.free(hw);
    const hs = try data.fill(gpa, u16, experts * n * groups, rng, data.makeScale);
    defer gpa.free(hs);
    const hb = try data.fill(gpa, u16, experts * n * groups, rng, data.makeBias);
    defer gpa.free(hb);
    const hx = try data.fill(gpa, u16, most_rows * k_dim, rng, data.makeXB);
    defer gpa.free(hx);
    var words = try Buffer.fromHost(t.base, hw);
    defer words.free();
    var scale = try Buffer.fromHost(t.base, hs);
    defer scale.free();
    var bias = try Buffer.fromHost(t.base, hb);
    defer bias.free();
    var x = try Buffer.fromHost(t.base, hx);
    defer x.free();
    const bytes = 2 * most_rows * n * 4;
    var out = try Buffer.alloc(t.base, bytes);
    defer out.free();

    const r = &t.base.r;
    const s = t.base.stream.handle;
    var h: u64 = 0;
    for (rounds) |m| {
        // top-2 routing: every row goes to both experts, pair 2r to the first and 2r + 1 to the second
        const rows: c_int = @intCast(m);
        const plan = [_]c_int{ 0, 0, rows, 1, rows, rows };
        var members: [2 * most_rows]c_int = undefined;
        for (0..m) |i| {
            members[i] = @intCast(2 * i);
            members[m + i] = @intCast(2 * i + 1);
        }
        var items = try Buffer.fromHost(t.base, @as([]const c_int, &plan));
        defer items.free();
        var mem = try Buffer.fromHost(t.base, @as([]const c_int, members[0 .. 2 * m]));
        defer mem.free();
        try out.fill8(0xA5, null);
        var arg: affine.Arg = .{
            .x = x.ptr,
            .words = words.ptr,
            .scale = .{ .p = scale.ptr, .kind = 1 },
            .bias = .{ .p = bias.ptr, .kind = 1 },
            .out = if (kind == .pair) 0 else out.ptr,
            .out16 = if (kind == .pair) out.ptr else 0,
            .m = rows,
            .n = n,
            .k = k_dim,
            .bits = bits,
            .group = group,
            .fp16 = 0,
        };
        switch (kind) {
            .plain => try t.kernels.streamGo(r, arg, .{}, n, 1, s),
            .pair => try t.kernels.streamGo(r, arg, .{ .pair_cols = n / 2 }, n / 2, 1, s),
            .routed => {
                arg.route = .{ .items = items.ptr, .members = mem.ptr, .x_div = 2 };
                try t.kernels.routed(r, arg, 2, s);
            },
        }
        try t.base.stream.synchronize();
        const written: usize = switch (kind) {
            .plain => m * n * 4,
            .pair => m * (n / 2) * 2,
            .routed => 2 * m * n * 4,
        };
        h = std.hash.Wyhash.hash(h, std.mem.asBytes(&try prod.digestOf(out, written)));
    }
    return h;
}

test "the 4-bit BF16 stream tile writes the pinned bytes of its plain, pair and routed launches" {
    const b = try base.Rig.open(1 << 20);
    defer b.close();
    var rig = Rig.open(b);
    if (rig.fp16) return error.SkipZigTest; // the FP16 build keeps stream_pair's magic-number pairs
    var rng: data.Rng = .{ .state = 0x9E3779B97F4A7C15 };
    var h: u64 = 0;
    for ([_]Kind{ .plain, .pair, .routed }) |kind| {
        h = std.hash.Wyhash.hash(h, std.mem.asBytes(&try digestOf(&rig, &rng, kind)));
    }
    const want = for (pins) |p| {
        if (std.mem.eql(u8, p.arch, kernels.arch)) break p.digest;
    } else {
        std.debug.print("stream golden: no digest pinned for {s}; this run wrote 0x{x:0>16}\n", .{ kernels.arch, h });
        return error.SkipZigTest;
    };
    try base.expect(h == want, "stream golden: digest 0x{x:0>16}, pinned 0x{x:0>16}", .{ h, want });
}
