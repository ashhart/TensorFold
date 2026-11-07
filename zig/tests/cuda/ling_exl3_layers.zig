//! `ling-exl3-layers DIR REF OUT`: real EXL3 layers, each on the reference's own input rows (teacher-forced).

// REF (tools/g2/ling_layers_ref.py, exllamav3): "TFLX3L01", u32 T, D, n, i32 ids[T], u32 layer[n], f32 in, out a layer

// OUT: "TFLX3O01", u32 T, D, n, top, per layer f32 prompt[T][D], decode[T][D], i32 picks[T][top + 1] prompt, decode
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const rig = @import("ling_forward_rig.zig");
const ling = @import("ling_cuda");

const F = rig.F;
const Rig = rig.Rig;

fn toBf16(x: f32) u16 {
    const u: u32 = @bitCast(x);
    if (std.math.isNan(x)) return 0x7fc0;
    return @intCast((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

fn upload(r: *Rig, ptr: u64, rows: []const f32) !void {
    const h = try r.a.alloc(u16, rows.len);
    for (h, rows) |*o, x| o.* = toBf16(x);
    try r.s.synchronize();
    const b: cuda.DeviceBuffer = .{ .d = r.gpu.d, .ptr = ptr, .len = h.len * 2 };
    try b.upload(0, std.mem.sliceAsBytes(h));
}

fn rowsBack(r: *Rig, ptr: u64, out: []f32) !void {
    const raw = std.mem.bytesAsSlice(u16, try r.back(ptr, out.len * 2));
    for (out, raw) |*o, u| o.* = @floatCast(rig.bfOf(u));
}

/// rel L2 of a against b, and of a - b against the layer's own change (b - in).
fn rel(a: []const f32, b: []const f32, in: []const f32) [2]f64 {
    var e: f64 = 0;
    var n: f64 = 0;
    var dd: f64 = 0;
    for (a, b, in) |x, y, z| {
        e += (@as(f64, x) - y) * (@as(f64, x) - y);
        n += @as(f64, y) * y;
        dd += (@as(f64, y) - z) * (@as(f64, y) - z);
    }
    return .{ @sqrt(e / n), @sqrt(e / dd) };
}

pub fn run(gpu: check.Gpu, rest: []const [:0]const u8) !void {
    if (rest.len < 3) return error.MissingArgument;
    var arena = std.heap.ArenaAllocator.init(gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const bytes = try std.Io.Dir.cwd().readFileAlloc(gpu.io, rest[1], a, .limited(8 << 30));
    if (bytes.len < 20 or !std.mem.eql(u8, bytes[0..8], "TFLX3L01")) return error.BadFixture;
    const T: usize = std.mem.readInt(u32, bytes[8..12], .little);
    const D: usize = std.mem.readInt(u32, bytes[12..16], .little);
    const n: usize = std.mem.readInt(u32, bytes[16..20], .little);
    var at: usize = 20 + T * 4;
    const layers = try a.alloc(u32, n);
    for (layers) |*l| {
        l.* = std.mem.readInt(u32, bytes[at..][0..4], .little);
        at += 4;
    }
    if (bytes.len != at + n * 2 * T * D * 4) return error.BadFixture;
    const vals = try a.alloc(f32, n * 2 * T * D); // the file's offsets need not be 4-aligned
    @memcpy(std.mem.sliceAsBytes(vals), bytes[at..]);

    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var why: @import("ling_cuda").config.Why = .{};
    var o = rig.W.open(gpu.gpa, gpu.io, rest[0], &why) catch |e| {
        std.debug.print("{s}: {s}\n", .{ rest[0], why.text() });
        return e;
    };
    defer o.ck.close();
    var shift: u32 = 0; // --rope-shift N: the rope reads positions N.. (keys from 0), as the reference's `position`
    var k: usize = 3;
    while (k < rest.len) : (k += 1) {
        if (std.mem.eql(u8, rest[k], "--rope-shift") and k + 1 < rest.len) {
            k += 1;
            shift = try std.fmt.parseInt(u32, rest[k], 10);
        } else return error.BadArgument;
    }
    try check.expect(o.c.hidden == D, "hidden {d}, the reference's {d}", .{ o.c.hidden, D });
    var m = try F.Modules.load(gpu.d, gpu.ctx);
    defer m.deinit();
    var pack = try cuda.modelopt.Packer.load(gpu.d);
    defer pack.unload();
    // TP=2 by TF_LING_RANK, TF_LING_WORLD, NCCL_COMM_ID: both ranks run this loop on their shares; rank 0 writes
    const p = try ling.tp.place(ling.tp.env);
    var group: ?cuda.tp_group.Group = null;
    if (p.world > 1) {
        try cuda.tp_watch.start(gpu.io, p.rank, p.world);
        group = try ling.tp.group(gpu.io, p);
    }
    defer if (group) |*gr| gr.close();
    var w = try rig.W.load(gpu.gpa, gpu.d, s, &pack, &o.ck, o.c, layers, .{ .globals = false, .rank = p.rank, .world = p.world });
    defer w.deinit();
    std.debug.print("RESULT ling-exl3-layers: {t} layers {any} loaded\n", .{ o.c.quant, layers });
    var g = try F.Graph.init(gpu.gpa, gpu.d, &m, s, &w);
    defer g.deinit();
    if (group) |*gr| {
        g.tp = gr;
        var dg: [32]u8 = @splat(0);
        std.crypto.hash.sha2.Sha256.hash(rest[1], &dg, .{});
        try cuda.tp_watch.ready(gpu.io, dg);
    }
    var r: Rig = .{ .gpu = gpu, .a = a, .s = s, .m = &m, .g = &g };
    defer r.deinit();
    var pb = try F.Buffers.init(gpu.d, &m, &w, T, true, T + 8, null);
    defer pb.deinit();
    var db = try F.Buffers.init(gpu.d, &m, &w, F.window_rows, false, T + 8, null);
    defer db.deinit();
    const top: usize = w.config.topk;
    const slots = top + 1;

    var out: std.ArrayList(u8) = .empty;
    try out.appendSlice(a, "TFLX3O01");
    for ([_]usize{ T, D, n, top }) |v| try out.appendSlice(a, std.mem.asBytes(&@as(u32, @intCast(v))));
    const got = try a.alloc(f32, T * D);
    const dec = try a.alloc(f32, T * D);
    const pp = try r.zeros(T * slots * 4);
    const dp = try r.zeros(slots * 4);
    for (layers, 0..) |li, j| {
        const in = vals[j * 2 * T * D ..][0 .. T * D];
        const want = vals[(j * 2 + 1) * T * D ..][0 .. T * D];
        const moe = w.layers[j].ffn == .moe;
        // prompt: all T rows in one forward of a fresh sequence
        const sp = try r.state(T + 8);
        sp.rope_base = shift;
        try upload(&r, pb.x, in);
        try r.fwd(&pb, if (moe) pp else null).one(&.{.{ .st = sp, .row0 = 0, .rows = T }}, j, if (moe) pp else null);
        try rowsBack(&r, pb.x, got);
        const ppicks = if (moe) try r.back(pp, T * slots * 4) else try a.alloc(u8, T * slots * 4);
        if (!moe) @memset(ppicks, 0);
        // decode: one row a forward, each committed, on another fresh sequence
        const sd = try r.state(T + 8);
        sd.rope_base = shift;
        const dpicks = try a.alloc(u8, T * slots * 4);
        @memset(dpicks, 0);
        for (0..T) |t| {
            try upload(&r, db.x, in[t * D ..][0..D]);
            try r.fwd(&db, if (moe) dp else null).one(&.{.{ .st = sd, .row0 = 0, .rows = 1 }}, j, if (moe) dp else null);
            try rowsBack(&r, db.x, dec[t * D ..][0..D]);
            if (moe) @memcpy(dpicks[t * slots * 4 ..][0 .. slots * 4], try r.back(dp, slots * 4));
            try r.fwd(&db, null).commit(sd, 1);
        }
        const pr = rel(got, want, in);
        const dr = rel(dec, want, in);
        const pd = rel(got, dec, in);
        std.debug.print("RESULT   layer {d} ({t}, {t}): prompt vs ref rel {e:.3} (of the layer's change {e:.3}); decode vs ref {e:.3} ({e:.3}); prompt vs decode {e:.3}\n", .{ li, w.layers[j].mixer, w.layers[j].ffn, pr[0], pr[1], dr[0], dr[1], pd[1] });
        try out.appendSlice(a, std.mem.sliceAsBytes(got));
        try out.appendSlice(a, std.mem.sliceAsBytes(dec));
        try out.appendSlice(a, ppicks);
        try out.appendSlice(a, dpicks);
    }
    if (group) |*gr| { // both ranks done before either closes (a rank closing first stops the other)
        try gr.connect(s, try r.zeros(128), 1 - gr.rank);
        cuda.tp_watch.ended();
    }
    if (p.rank != 0) return check.pass("ling-exl3-layers rank {d}: {d} layers ran on its share", .{ p.rank, n });
    try std.Io.Dir.cwd().writeFile(gpu.io, .{ .sub_path = rest[2], .data = out.items });
    check.pass("ling-exl3-layers: {d} layers ran on {d} rows, outputs in {s} (the parity numbers are the script's)", .{ n, T, rest[2] });
}
