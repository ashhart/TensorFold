//! Flash Next's two sparse attention kernels on the same rows: the lanes kernel's parts equal the scalar kernel's bits.
const std = @import("std");
const rig = @import("rig.zig");
const fx = @import("flashnext.zig");
const Rig = rig.Rig;
const gpa = rig.gpa;
const ptr = fx.ptr;
const DeviceBuffer = @import("../memory.zig").DeviceBuffer;
const Launcher = @import("../launches.zig").Launcher;

const Shape = struct { heads: c_int, kv_heads: c_int, dims: c_int, parts: c_int };

test "sparse attention: the lanes kernel gives the scalar kernel's bits at every count, part and cache kind" {
    const t = try Rig.open(1 << 22);
    defer t.close();
    const shapes = [_]Shape{
        .{ .heads = 24, .kv_heads = 2, .dims = 256, .parts = 16 },
        .{ .heads = 24, .kv_heads = 2, .dims = 256, .parts = 1 },
        .{ .heads = 6, .kv_heads = 1, .dims = 256, .parts = 3 },
        .{ .heads = 6, .kv_heads = 2, .dims = 128, .parts = 16 },
        .{ .heads = 12, .kv_heads = 1, .dims = 128, .parts = 5 },
    };
    for ([_]c_int{ 2, 1 }) |kind| for (shapes, 0..) |s, i| try same(t, s, kind, 0x5A11 + i);
}

fn same(t: *Rig, s: Shape, kind: c_int, seed: u64) !void {
    var rng: rig.Rng = .{ .state = seed };
    const heads: usize = @intCast(s.heads);
    const kvh: usize = @intCast(s.kv_heads);
    const dims: usize = @intCast(s.dims);
    const parts: usize = @intCast(s.parts);
    const cap = 2100;
    const stride = 2048;
    // counts across tile seams, a part shorter than a tile, empty rows and the indexer's full budget
    const count = [_]i32{ 0, 1, 7, 31, 32, 33, 63, 64, 65, 97, 300, 2000, 2048, 650 };
    const sparse = [_]i32{ 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0 };
    const rows = count.len;
    const keys = try gpa.alloc(u16, kvh * cap * dims * 2);
    defer gpa.free(keys);
    for (keys) |*x| {
        const v = rng.unit() * 2.0;
        x.* = if (kind == 2) rig.bf16Bits(v) else rig.f16Bits(v);
    }
    const hq = try gpa.alloc(f32, rows * heads * dims);
    defer gpa.free(hq);
    for (hq) |*x| x.* = rng.unit() * 4.0;
    const ids = try gpa.alloc(i32, rows * stride);
    defer gpa.free(ids);
    for (0..rows) |r| for (0..stride) |j| {
        ids[r * stride + j] = @intCast((j * 37 + r * 11 + (j * j) % 13) % cap);
    };
    var dk = try t.upload(keys[0 .. kvh * cap * dims]);
    defer dk.free();
    var dv = try t.upload(keys[kvh * cap * dims ..]);
    defer dv.free();
    var dq = try t.upload(hq);
    defer dq.free();
    var di = try t.upload(ids);
    defer di.free();
    var dc = try t.upload(&count);
    defer dc.free();
    var ds = try t.upload(&sparse);
    defer ds.free();
    var outs: [2][3]DeviceBuffer = undefined;
    for (&outs) |*o| {
        o[0] = try t.alloc(rows * heads * parts * dims * 4);
        o[1] = try t.alloc(rows * heads * parts * 2 * 4);
        o[2] = try t.alloc(rows * heads * dims * 4);
    }
    defer for (&outs) |*o| for (o) |*b| b.free();
    const scale: f32 = 1.0 / 16.0;
    const st = t.stream.handle;
    const Fn = *const @TypeOf(Launcher.fn_sparse_attention);
    for ([_]Fn{ &Launcher.fn_sparse_attention_scalar, &Launcher.fn_sparse_attention }, 0..) |call, w| {
        try call(
            &t.on,
            ptr(?[*]const f32, dq),
            ptr(?*const anyopaque, dk),
            ptr(?*const anyopaque, dv),
            kind,
            ptr(?[*]const i32, di),
            stride,
            ptr(?[*]const i32, dc),
            ptr(?[*]const i32, ds),
            ptr(?[*]f32, outs[w][0]),
            ptr(?[*]f32, outs[w][1]),
            ptr(?[*]f32, outs[w][2]),
            @intCast(rows),
            s.heads,
            s.kv_heads,
            s.dims,
            cap,
            scale,
            s.parts,
            st,
        );
    }
    try t.stream.synchronize();
    for (0..3) |b| {
        const want = try rig.download(u32, outs[0][b]);
        defer gpa.free(want);
        const got = try rig.download(u32, outs[1][b]);
        defer gpa.free(got);
        for (want, got, 0..) |x, y, i| try rig.expect(
            x == y,
            "kind {d} heads {d}/{d} dims {d} parts {d}: buffer {d} word {d}: scalar {x} lanes {x}",
            .{ kind, s.heads, s.kv_heads, s.dims, s.parts, b, i, x, y },
        );
    }
}
