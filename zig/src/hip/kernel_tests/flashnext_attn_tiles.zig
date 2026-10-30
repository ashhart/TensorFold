//! Flash Next's QSA decode prep against the launches it replaces.
const std = @import("std");
const rig = @import("rig.zig");
const fx = @import("flashnext.zig");
const Rig = rig.Rig;
const gpa = rig.gpa;
const at = rig.at;
const ptr = fx.ptr;
const rnd = fx.rnd;

/// tf_fn_qsa_prep against the launches it replaces: the same q, caches and pooled keys; `n` rows at `first`.
fn qsaPrep(t: *Rig, rng: *rig.Rng, n: usize, first: i32, with_q: bool) !void {
    const heads = 24;
    const kv_heads = 2;
    const hd = 256;
    const ih = 4;
    const di = 128;
    const width = (ih + 1) * di;
    const ratio = 4;
    const rotary = 64;
    const theta: f32 = 10_000_000;
    const eps: f32 = 1e-6;
    const cap = 64;
    const s = t.stream.handle;
    const hq = try fx.acts(t, rng, n * heads * 2 * hd, 2.0);
    defer gpa.free(hq);
    const hk = try fx.acts(t, rng, n * kv_heads * hd, 2.0);
    defer gpa.free(hk);
    const hv = try fx.acts(t, rng, n * kv_heads * hd, 2.0);
    defer gpa.free(hv);
    const hi = try fx.acts(t, rng, n * width, 2.0);
    defer gpa.free(hi);
    // the earlier rows of each block already in the raw cache
    const hraw0 = try fx.acts(t, rng, cap * di, 2.0);
    defer gpa.free(hraw0);
    var norms: [4][hd]f32 = undefined;
    for (&norms) |*w| for (w) |*x| {
        x.* = 1.0 + rng.unit() * 0.5;
    };
    var q = try t.upload(hq);
    defer q.free();
    var k = try t.upload(hk);
    defer k.free();
    var v = try t.upload(hv);
    defer v.free();
    var idx = try t.upload(hi);
    defer idx.free();
    var w = [4]@import("../memory.zig").DeviceBuffer{ try t.upload(&norms[0]), try t.upload(&norms[1]), try t.upload(&norms[2]), try t.upload(&norms[3]) };
    defer for (&w) |*b| b.free();
    var hpos: [4]i32 = undefined;
    for (0..n) |r| hpos[r] = first + @as(i32, @intCast(r));
    var pos = try t.upload(hpos[0..n]);
    defer pos.free();
    // [0]: the separate launches, [1]: the one launch; each has q, index q, K, V, raw and pooled buffers
    var outs: [2][6]@import("../memory.zig").DeviceBuffer = undefined;
    const sizes = [6]usize{ n * heads * hd * 4, n * ih * di * 4, kv_heads * cap * hd * 2, kv_heads * cap * hd * 2, cap * di * 2, (cap / ratio) * di * 2 };
    for (&outs) |*set| for (set, sizes) |*b, size| {
        b.* = try t.alloc(size);
        try b.fill8(0xA5);
    };
    defer for (&outs) |*set| for (set) |*b| b.free();
    for (&outs) |*set| try set[4].upload(0, std.mem.sliceAsBytes(hraw0));
    const kind = t.kind();
    {
        const o = &outs[0];
        if (with_q) {
            try t.on.tf_qk_rope(ptr(?*const anyopaque, q), kind, 2 * heads * hd, 2 * hd, ptr(?[*]const f32, w[0]), eps, @intCast(n), heads, hd, rotary, theta, ptr(?[*]const i32, pos), ptr(?[*]f32, o[0]), null, 0, s);
            try t.on.tf_qk_rope(ptr(?*const anyopaque, idx), kind, width, di, ptr(?[*]const f32, w[1]), eps, @intCast(n), ih, di, rotary, theta, ptr(?[*]const i32, pos), ptr(?[*]f32, o[1]), null, 0, s);
        }
        try t.on.tf_qk_rope(ptr(?*const anyopaque, k), kind, kv_heads * hd, hd, ptr(?[*]const f32, w[2]), eps, @intCast(n), kv_heads, hd, rotary, theta, ptr(?[*]const i32, pos), null, ptr(?*anyopaque, o[2]), cap, s);
        try t.on.tf_kv_write_at(ptr(?*const anyopaque, v), ptr(?*anyopaque, o[3]), kind, @intCast(n), kv_heads, hd, cap, ptr(?[*]const i32, pos), s);
        var raw = try t.alloc(n * di * 2);
        defer raw.free();
        try t.on.tf_copy_cols(ptr(?*const anyopaque, idx), width, ih * di, ptr(?*anyopaque, raw), kind, @intCast(n), di, s);
        try t.on.tf_kv_write_at(ptr(?*const anyopaque, raw), ptr(?*anyopaque, o[4]), kind, @intCast(n), 1, di, cap, ptr(?[*]const i32, pos), s);
        try t.on.fn_idx_pool_at(ptr(?*const anyopaque, o[4]), ptr(?*anyopaque, o[5]), ptr(?[*]const f32, w[3]), kind, ptr(?[*]const i32, pos), @intCast(n), di, ratio, rotary, theta, eps, s);
        try t.stream.synchronize();
    }
    {
        const o = &outs[1];
        try t.on.fn_qsa_prep(.{
            .q = if (with_q) .{ .src = at(q), .s_row = 2 * heads * hd, .s_head = 2 * hd, .heads = heads, .width = hd, .weight = at(w[0]), .wide = at(o[0]) } else .{},
            .iq = if (with_q) .{ .src = at(idx), .s_row = width, .s_head = di, .heads = ih, .width = di, .weight = at(w[1]), .wide = at(o[1]) } else .{},
            .k = .{ .src = at(k), .s_row = kv_heads * hd, .s_head = hd, .heads = kv_heads, .width = hd, .weight = at(w[2]), .cache = at(o[2]) },
            .v = at(v),
            .v_cache = at(o[3]),
            .v_width = hd,
            .raw = at(idx),
            .raw_stride = width,
            .raw_offset = ih * di,
            .raw_cache = at(o[4]),
            .raw_width = di,
            .pooled = at(o[5]),
            .pool_scale = at(w[3]),
            .ratio = ratio,
            .pos = at(pos),
            .seg_pos = at(pos),
            .rows = @intCast(n),
            .n = @intCast(n),
            .total = cap,
            .kind = kind,
            .eps = eps,
            .rotary = rotary,
            .theta = theta,
        }, s);
        try t.stream.synchronize();
    }
    const names = [6][]const u8{ "q", "index q", "K cache", "V cache", "raw cache", "pooled" };
    for (0..6) |b| {
        const want = try rig.download(u8, outs[0][b]);
        defer gpa.free(want);
        const got = try rig.download(u8, outs[1][b]);
        defer gpa.free(got);
        if (!std.mem.eql(u8, want, got)) {
            std.debug.print("qsa prep n={d} first={d} q={}: {s} differs\n", .{ n, first, with_q, names[b] });
            return error.TestUnexpectedResult;
        }
    }
}

test "the QSA decode prep in one launch writes the bytes of the seven launches it replaces (Flash Next, 1 to 4 rows)" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    var rng: rig.Rng = .{ .state = 0x51A9 };
    // a row completing a block at its end (15), inside the window (10 .. 13 completes 11), none (12, 13 .. 14)
    try qsaPrep(t, &rng, 1, 15, true);
    try qsaPrep(t, &rng, 1, 13, true);
    try qsaPrep(t, &rng, 4, 10, true);
    try qsaPrep(t, &rng, 2, 12, true);
    try qsaPrep(t, &rng, 3, 20, false);
}
