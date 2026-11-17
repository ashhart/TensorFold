//! zonos3-generate's checks of the codec encoder against the reference's dumps: the acoustic branch, the SSL features.
const std = @import("std");
const z3 = @import("zonos3");
const cuda = @import("cuda");
const gen = @import("generate.zig");

const Check = gen.Check;
const nowNs = gen.nowNs;
const readNpy = gen.readNpy;

pub fn acoustic(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, generator: []const u8) !u8 {
    const path = try std.fs.path.join(gpa, &.{ dir, "audio.npy" });
    defer gpa.free(path);
    const f = try readNpy(gpa, io, path);
    defer gpa.free(f.bytes);
    const audio = try f.arr.floats(gpa);
    defer gpa.free(audio);
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    var stream = try cuda.Stream.init(&driver, true);
    defer stream.deinit();
    const t_load = nowNs(io);
    const enc = try z3.encoder.Encoder.create(gpa, io, &driver, stream, generator);
    defer enc.destroy();
    const Enc = struct {
        fn get(c: *anyopaque, ptr: u64, out: []u8) anyerror!void {
            try @as(*z3.encoder.Encoder, @ptrCast(@alignCast(c))).download(ptr, out);
        }
    };
    // the reference ran in bf16: a row's cosine against it, not fp32 rounding
    var c: Check = .{ .gpa = gpa, .io = io, .ctx = enc, .get = Enc.get, .dir = dir, .prefix = true, .floor = 0.99 };
    std.debug.print("zonos3 encoder check: {d} samples, {d} MiB of weights, loaded in {d:.1} s\n", .{ audio.len, enc.w.bytes >> 20, @as(f64, @floatFromInt(nowNs(io) - t_load)) / 1e9 });
    const lane = audio[0..@min(audio.len, z3.encoder.lane_frames * z3.encoder.frame_samples)];
    const probe: z3.encoder.Probe = .{ .ctx = &c, .f = Check.probe };
    try enc.acoustic(0, try enc.load(lane, probe), probe);
    const t0 = nowNs(io);
    const frames = try enc.load(lane, null);
    try enc.acoustic(0, frames, null);
    try stream.synchronize();
    const seconds = @as(f64, @floatFromInt(nowNs(io) - t0)) / 1e9;
    const pass = c.fails == 0 and c.n > 0 and c.missing == 0;
    std.debug.print("RESULT zonos3 encoder {s} probes={d} low={d} missing={d} worst_row_cos={d:.6} frames={d} encode_s={d:.3}\n", .{ if (pass) "PASS" else "FAIL", c.n, c.fails, c.missing, c.worst, frames, seconds });
    return if (pass) 0 else 1;
}

/// The SSL teacher's features of a lane's 44.1 kHz audio, computed on the host, against the reference's.
pub fn features(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !u8 {
    const audio_path = try std.fs.path.join(gpa, &.{ dir, "lane.0.teacher_audio.npy" });
    defer gpa.free(audio_path);
    const a = try readNpy(gpa, io, audio_path);
    defer gpa.free(a.bytes);
    const audio = try a.arr.floats(gpa);
    defer gpa.free(audio);
    const want_path = try std.fs.path.join(gpa, &.{ dir, "lane.0.ssl.features.npy" });
    defer gpa.free(want_path);
    const w = try readNpy(gpa, io, want_path);
    defer gpa.free(w.bytes);
    const want = try w.arr.floats(gpa);
    defer gpa.free(want);
    const t0 = nowNs(io);
    const low = try z3.features.resample(gpa, audio);
    defer gpa.free(low);
    const got = try z3.features.rows(gpa, low);
    defer gpa.free(got);
    const seconds = @as(f64, @floatFromInt(nowNs(io) - t0)) / 1e9;
    var max_d: f64 = 0;
    var worst: f64 = 1;
    const width = z3.features.width;
    const same_shape = got.len == want.len;
    if (same_shape) for (0..got.len / width) |r| {
        var d: f64 = 0;
        var a2: f64 = 0;
        var b2: f64 = 0;
        for (got[r * width ..][0..width], want[r * width ..][0..width]) |x, y| {
            d += @as(f64, x) * y;
            a2 += @as(f64, x) * x;
            b2 += @as(f64, y) * y;
            max_d = @max(max_d, @abs(@as(f64, x) - y));
        }
        worst = @min(worst, d / @sqrt(a2 * b2));
    };
    const pass = same_shape and worst >= 0.9999;
    std.debug.print("RESULT zonos3 features {s} rows={d} reference_rows={d} worst_row_cos={d:.7} max_abs_diff={e:.3} seconds={d:.3}\n", .{ if (pass) "PASS" else "FAIL", got.len / width, want.len / width, worst, max_d, seconds });
    return if (pass) 0 else 1;
}

/// The SSL teacher over the reference's feature rows, against its fp32 run's layers.
pub fn teacher(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, teacher_dir: []const u8) !u8 {
    const path = try std.fs.path.join(gpa, &.{ dir, "lane.0.ssl.features.npy" });
    defer gpa.free(path);
    const f = try readNpy(gpa, io, path);
    defer gpa.free(f.bytes);
    const feats = try f.arr.floats(gpa);
    defer gpa.free(feats);
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    var stream = try cuda.Stream.init(&driver, true);
    defer stream.deinit();
    const t = try z3.teacher.Teacher.create(gpa, io, &driver, stream, teacher_dir);
    defer t.destroy();
    const Get = struct {
        fn get(c: *anyopaque, ptr: u64, out: []u8) anyerror!void {
            try @as(*z3.teacher.Teacher, @ptrCast(@alignCast(c))).download(ptr, out);
        }
    };
    var c: Check = .{ .gpa = gpa, .io = io, .ctx = t, .get = Get.get, .dir = dir };
    std.debug.print("zonos3 teacher check: {d} rows, {d} MiB of weights\n", .{ feats.len / z3.teacher.feature_width, t.bytes >> 20 });
    _ = try t.run(feats, .{ .ctx = &c, .f = Check.probe });
    const t0 = nowNs(io);
    _ = try t.run(feats, null);
    try stream.synchronize();
    const seconds = @as(f64, @floatFromInt(nowNs(io) - t0)) / 1e9;
    const pass = c.fails == 0 and c.n > 0 and c.missing == 0;
    std.debug.print("RESULT zonos3 teacher {s} probes={d} low={d} missing={d} worst_row_cos={d:.7} seconds={d:.3}\n", .{ if (pass) "PASS" else "FAIL", c.n, c.fails, c.missing, c.worst, seconds });
    return if (pass) 0 else 1;
}

/// A whole clip encoded natively (lanes of 20 s with 10 s of left context) against the reference's fp32 codes.
pub fn encode(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, root: []const u8) !u8 {
    const audio_path = try std.fs.path.join(gpa, &.{ dir, "audio.npy" });
    defer gpa.free(audio_path);
    const f = try readNpy(gpa, io, audio_path);
    defer gpa.free(f.bytes);
    const audio = try f.arr.floats(gpa);
    defer gpa.free(audio);
    const want_path = try std.fs.path.join(gpa, &.{ dir, "codes.npy" });
    defer gpa.free(want_path);
    const w = try readNpy(gpa, io, want_path);
    defer gpa.free(w.bytes);
    const want = try w.arr.ints(gpa);
    defer gpa.free(want);

    var rel = try z3.release.open(gpa, io, root, null);
    defer rel.deinit();
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    var stream = try cuda.Stream.init(&driver, true);
    defer stream.deinit();
    const dec = try z3.codec_weights.load(gpa, io, &driver, rel.generator, rel.head);
    defer {
        dec.deinit();
        gpa.destroy(dec);
    }
    const t_load = nowNs(io);
    const clip = try z3.clip.Clip.create(gpa, io, &driver, stream, root, rel.generator, dec);
    defer clip.destroy();
    const load_s = @as(f64, @floatFromInt(nowNs(io) - t_load)) / 1e9;
    _ = try clip.encode(gpa, audio);
    const t0 = nowNs(io);
    const codes = try clip.encode(gpa, audio);
    defer gpa.free(codes);
    const seconds = @as(f64, @floatFromInt(nowNs(io) - t0)) / 1e9;

    const cb = z3.quantizer.codebooks;
    const n = @min(codes.len, w.arr.shape[0]);
    var same: [cb]usize = @splat(0);
    for (codes[0..n], 0..) |row, r| for (row, 0..) |code, j| {
        same[j] += @intFromBool(code == want[r * cb + j]);
    };
    var total: usize = 0;
    for (same) |v| total += v;
    const pass = codes.len == w.arr.shape[0] and total == n * cb;
    std.debug.print("codes equal to the fp32 reference's, per codebook of {d} frames: {any}\n", .{ n, same });
    std.debug.print("RESULT zonos3 encode {s} frames={d} reference_frames={d} codes_equal={d}/{d} semantic_equal={d}/{d} audio_s={d:.2} encode_s={d:.3} load_s={d:.1} weights_mib={d}\n", .{ if (pass) "PASS" else "FAIL", codes.len, w.arr.shape[0], total, n * cb, same[0], n, @as(f64, @floatFromInt(audio.len)) / 44100.0, seconds, load_s, clip.bytes() >> 20 });
    return if (pass) 0 else 1;
}
