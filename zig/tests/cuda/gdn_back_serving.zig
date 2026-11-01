//! Compare the real served and training GDN fronts on the same contiguous sequence and convolution history.
const std = @import("std");
const cuda = @import("cuda");
const qwen = @import("qwen35");
const ref = @import("gdn_back_ref.zig");
const check = @import("check.zig");
const Geo = qwen.engine.Geo;
const Seg = @typeInfo(@TypeOf(qwen.engine.Cache.seg)).@"fn".return_type.?;

pub fn compare(r: anytype, shape: cuda.gdn_train.Shape, p: []const u16, hist: ?[]const u16, cw: u64, al: u64, dt: []const f32, q: []const f32, k: []const f32) !void {
    const rows: usize = @intCast(shape.rows);
    const nk: usize = @intCast(shape.nk);
    const nv: usize = @intCast(shape.nv);
    const cols: usize = @intCast(shape.conv);
    const width: usize = @intCast(shape.proj);
    const x = try r.a.alloc(u16, rows * cols);
    const a = try r.a.alloc(u16, rows * nv);
    const b = try r.a.alloc(u16, rows * nv);
    for (0..rows) |row| {
        @memcpy(x[row * cols ..][0..cols], p[row * width ..][0..cols]);
        @memcpy(b[row * nv ..][0..nv], p[row * width + cols + nv * 128 ..][0..nv]);
        @memcpy(a[row * nv ..][0..nv], p[row * width + cols + nv * 129 ..][0..nv]);
    }
    const db = try r.a.alloc(u16, nv);
    for (db, dt) |*v, f| v.* = ref.bf16Bits(f);
    const history = try r.a.alloc(u16, 3 * cols);
    if (hist) |h| @memcpy(history, h) else @memset(history, 0);
    const parents = try r.a.alloc(c_int, rows);
    for (parents, 0..) |*v, row| v.* = @as(c_int, @intCast(row)) - 1;
    const rowseg = try r.a.alloc(c_int, rows);
    @memset(rowseg, 0);
    const sg: Seg = .{ .rec = 0, .conv = try r.dev(std.mem.sliceAsBytes(history)), .kv = 0, .tap = 0, .log = 0, .before = 0, .cap = 0, .len = 0, .row0 = 0, .rows = shape.rows, .tap_rows = 0, .taps = 0, .delta = 0 };
    const geo: Geo = .{ .hid = 5120, .nk = shape.nk, .nv = shape.nv, .qh = 24, .kvh = 4, .nl = 1 };
    const qd = try r.alloc(rows * nk * 128 * 2);
    const kd = try r.alloc(rows * nk * 128 * 2);
    const vd = try r.alloc(rows * nv * 128 * 2);
    const gd = try r.alloc(rows * nv * 4);
    const bd = try r.alloc(rows * nv * 2);
    var image = try cuda.Module.load(r.gpu.d, cuda.kernels.qwen35);
    defer image.unload();
    const f = try image.function("tf_q35_gdn_pre");
    var args: cuda.launch.Args = .{};
    const values = .{ geo, try r.dev(std.mem.sliceAsBytes(x)), try r.dev(std.mem.asBytes(&sg)), try r.dev(std.mem.sliceAsBytes(rowseg)), try r.dev(std.mem.sliceAsBytes(parents)), @as(c_int, 0), cw, try r.dev(std.mem.sliceAsBytes(a)), try r.dev(std.mem.sliceAsBytes(b)), al, try r.dev(std.mem.sliceAsBytes(db)), qd, kd, vd, gd, bd, @as(u64, 0) };
    inline for (values) |v| args.add(v);
    try cuda.launch.launch(f, .{ .grid = .{ .x = 1, .y = @intCast(2 * nk + nv), .z = @intCast(rows) }, .block = .{ .x = 32 } }, r.s, &args);
    const sq = try r.back(u16, qd, q.len);
    const sk = try r.back(u16, kd, k.len);
    // Compare every FP32 bit, not merely a second BF16 rounding of a possibly wrong trainer value.
    const wantq = try r.a.alloc(f32, q.len);
    const wantk = try r.a.alloc(f32, k.len);
    for (wantq, wantk, sq, sk) |*v, *w, qv, kv| {
        v.* = ref.bf16Value(qv);
        w.* = ref.bf16Value(kv);
    }
    try check.sameBytes("served/trainer GDN Q", std.mem.sliceAsBytes(q), std.mem.sliceAsBytes(wantq));
    try check.sameBytes("served/trainer GDN K", std.mem.sliceAsBytes(k), std.mem.sliceAsBytes(wantk));
    check.pass("served/trainer GDN Q/K: all {d} rows byte-identical, prefix {any}", .{ rows, hist != null });
}
