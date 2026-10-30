//! A real GLM-5.3-Flash EXL3 layer's rank-0 TP-2 share through the prompt kernels: runSlots, float64 and time.
const std = @import("std");
const core = @import("core");
const check = @import("check.zig");
const t = @import("exl3_prompt_k6_tests.zig");
const st = core.safetensors;
const D = t.D;
const N = t.N;
const E = t.E;

const layer_index = 4; // a layer with 3-bit gate and up and 4-bit down in the 3.5 bpw GLMQ checkpoint

/// The checkpoint's shard files, opened once each.
const Shards = struct {
    dir: []const u8,
    map: std.json.ObjectMap,
    open: std.StringHashMapUnmanaged(st.File) = .empty,
    fn tensor(s: *Shards, c: *t.Ctx, name: []const u8) !st.Tensor {
        const shard = (s.map.get(name) orelse return error.TensorNotInIndex).string;
        const got = try s.open.getOrPut(c.a, shard);
        if (!got.found_existing) got.value_ptr.* = try st.File.open(c.gpu.gpa, c.gpu.io, try std.fmt.allocPrint(c.a, "{s}/{s}", .{ s.dir, shard }));
        return got.value_ptr.get(name) orelse error.TensorNotInShard;
    }
    fn close(s: *Shards, c: *t.Ctx) void {
        var it = s.open.valueIterator();
        while (it.next()) |f| f.close(c.gpu.io);
    }
};

/// Every expert's words of one projection: gate and up keep tile columns 0-63, down tile rows 0-63 (rank 0 of 2).
fn trellis(c: *t.Ctx, sh: *Shards, which: []const u8, k2: *u32) ![]const []const i16 {
    const out = try c.a.alloc([]const i16, E);
    var all: []i16 = &.{};
    for (0..E) |e| {
        const tt = try sh.tensor(c, try std.fmt.allocPrint(c.a, "model.language_model.layers.{d}.mlp.experts.{d}.{s}_proj.trellis", .{ layer_index, e, which }));
        const down = std.mem.eql(u8, which, "down");
        const tk: usize = if (down) 2 * N / 16 else D / 16;
        const tn: usize = if (down) D / 16 else 2 * N / 16;
        if (tt.dtype != .i16 or tt.rank != 3 or tt.dim(0) != tk or tt.dim(1) != tn or tt.dim(2) % 16 != 0) return error.UnexpectedTrellis;
        const w: u32 = @intCast(tt.dim(2) / 8);
        if (e == 0) {
            k2.* = w;
            all = try c.a.alloc(i16, E * t.words(w));
        } else if (w != k2.*) return error.MixedWidthsInProjection;
        const src: []align(1) const i16 = std.mem.bytesAsSlice(i16, tt.bytes);
        const dst = all[e * t.words(w) ..][0..t.words(w)];
        const tw = tt.dim(2);
        if (down) {
            @memcpy(dst, src[0..dst.len]);
        } else for (0..D / 16) |r| @memcpy(dst[r * (N / 16) * tw ..][0 .. (N / 16) * tw], src[r * tn * tw ..][0 .. (N / 16) * tw]);
        out[e] = dst;
    }
    return out;
}

/// Each expert's fp16 scale vector `part` of a projection, its first `n` values (rank 0's share of a split one).
fn scale(c: *t.Ctx, sh: *Shards, which: []const u8, part: []const u8, n: usize) ![]f16 {
    const out = try c.a.alloc(f16, E * n);
    for (0..E) |e| {
        const tt = try sh.tensor(c, try std.fmt.allocPrint(c.a, "model.language_model.layers.{d}.mlp.experts.{d}.{s}_proj.{s}", .{ layer_index, e, which, part }));
        if (tt.dtype != .f16 or tt.numel() < n) return error.UnexpectedScale;
        const src: []align(1) const f16 = std.mem.bytesAsSlice(f16, tt.bytes);
        @memcpy(out[e * n ..][0..n], src[0..n]);
    }
    return out;
}

fn upload(b: *t.Bufs, c: *t.Ctx, each: []const []const i16) !u64 {
    const n = each[0].len;
    const base = try b.alloc(E * n * 2);
    const table = try c.a.alloc(u64, E);
    for (each, 0..) |w, e| {
        const buf: @import("cuda").DeviceBuffer = .{ .d = c.gpu.d, .ptr = base, .len = E * n * 2 };
        try buf.upload(e * n * 2, std.mem.sliceAsBytes(w));
        table[e] = base + e * n * 2;
    }
    return b.put(std.mem.sliceAsBytes(table));
}

pub fn run(c: *t.Ctx, dir: []const u8, rng: std.Random) !void {
    const index_bytes = try std.Io.Dir.cwd().readFileAlloc(c.gpu.io, try std.fmt.allocPrint(c.a, "{s}/model.safetensors.index.json", .{dir}), c.a, .limited(64 << 20));
    const index = try std.json.parseFromSlice(std.json.Value, c.a, index_bytes, .{});
    var sh: Shards = .{ .dir = dir, .map = index.value.object.get("weight_map").?.object };
    defer sh.close(c);
    var kg: u32 = 0;
    var ku: u32 = 0;
    var kd: u32 = 0;
    const hg = try trellis(c, &sh, "gate", &kg);
    const hu = try trellis(c, &sh, "up", &ku);
    const hd = try trellis(c, &sh, "down", &kd);
    try check.expect(kg == ku and (kg == 6 or kg == 8) and (kd == 6 or kd == 8), "real layer {d}: gate {d}, up {d}, down {d} half-bits", .{ layer_index, kg, ku, kd });
    const sg = try scale(c, &sh, "gate", "suh", D);
    const su = try scale(c, &sh, "up", "suh", D);
    const vg = try scale(c, &sh, "gate", "svh", N);
    const vu = try scale(c, &sh, "up", "svh", N);
    const sd = try scale(c, &sh, "down", "suh", N);
    const vd = try scale(c, &sh, "down", "svh", D);
    var shared = true; // the loader's check: one input scale vector for every gate and up
    for (0..E) |e| shared = shared and std.mem.eql(u8, std.mem.sliceAsBytes(sg[e * D ..][0..D]), std.mem.sliceAsBytes(sg[0..D])) and std.mem.eql(u8, std.mem.sliceAsBytes(su[e * D ..][0..D]), std.mem.sliceAsBytes(sg[0..D]));
    std.debug.print("INFO real layer {d} (rank 0 of 2): gate/up {d}-bit, down {d}-bit, input scales {s}\n", .{ layer_index, kg / 2, kd / 2, if (shared) "shared by every gate and up" else "per projection" });
    var b: t.Bufs = .{ .d = c.gpu.d, .gpa = c.gpu.gpa };
    defer b.deinit();
    const s: t.Set = .{ .gp = try upload(&b, c, hg), .up = try upload(&b, c, hu), .dp = try upload(&b, c, hd), .k2 = try t.widthTables(&b), .suh_g = try b.put(std.mem.sliceAsBytes(sg)), .suh_u = try b.put(std.mem.sliceAsBytes(su)), .svh_g = try b.put(std.mem.sliceAsBytes(vg)), .svh_u = try b.put(std.mem.sliceAsBytes(vu)), .suh_d = try b.put(std.mem.sliceAsBytes(sd)), .svh_d = try b.put(std.mem.sliceAsBytes(vd)), .shared = shared, .hg = hg, .hu = hu, .hd = hd, .sg = sg, .su = su, .vg = vg, .vu = vu, .sd = sd, .vd = vd };
    const in = try t.inputs(c, rng, 4096);
    const dx = try b.put(std.mem.sliceAsBytes(in.x));
    const dpk = try b.put(std.mem.sliceAsBytes(in.pk));
    var m: t.Mats = .{};
    defer m.deinit(c.gpu.gpa);
    for ([_]bool{ true, false }) |one| {
        if (one and !shared) continue;
        try t.widthsCase(c, &m, &s, kg, kd, one, if (one) &.{ 1, 64, 1000, 4096 } else &.{ 1, 64, 1000 }, in.x, in.pk, dx, dpk);
    }
    try t.timing(c, &s, dx, dpk, &.{.{ kg, kd }}, shared);
    check.pass("exl3 prompt real layer {d}: gate/up {d}-bit, down {d}-bit against runSlots and float64", .{ layer_index, kg / 2, kd / 2 });
}
