//! zonos3-generate --aligner-check: the aligner on a dumped clip against fusealign_hub on the same weights.
const std = @import("std");
const z3 = @import("zonos3");
const gen = @import("generate.zig");

const on_metal = @hasDecl(z3, "gpu");
const cuda = if (on_metal) void else @import("cuda");

/// The aligner on a device of its own: Metal's, or a CUDA context with the cuBLAS handle the decoder would give it.
const Dev = if (on_metal) struct {
    g: *z3.gpu.Gpu,

    fn open(d: *Dev, gpa: std.mem.Allocator) !void {
        d.g = try z3.gpu.Gpu.openFamily(gpa);
    }
    fn close(d: *Dev) void {
        d.g.close();
    }
    fn aligner(d: *Dev, gpa: std.mem.Allocator, io: std.Io, weights: []const u8) !*z3.aligner.Aligner {
        return z3.aligner.Aligner.create(gpa, io, d.g, weights);
    }
} else struct {
    driver: cuda.Driver,
    ctx: cuda.Context,
    stream: cuda.Stream,
    ws: cuda.DeviceBuffer,
    blas: z3.blas.Blas,

    fn open(d: *Dev, _: std.mem.Allocator) !void {
        d.driver = try cuda.Driver.open();
        errdefer d.driver.close();
        d.ctx = try cuda.Context.init(&d.driver, 0);
        errdefer d.ctx.deinit();
        d.stream = try cuda.Stream.init(&d.driver, true);
        errdefer d.stream.deinit();
        d.ws = try cuda.DeviceBuffer.alloc(&d.driver, 32 << 20);
        errdefer d.ws.free();
        d.blas = try z3.blas.Blas.open(d.ws.ptr, 32 << 20);
    }
    fn close(d: *Dev) void {
        d.blas.close();
        d.ws.free();
        d.stream.deinit();
        d.ctx.deinit();
        d.driver.close();
    }
    fn aligner(d: *Dev, gpa: std.mem.Allocator, io: std.Io, weights: []const u8) !*z3.aligner.Aligner {
        return z3.aligner.Aligner.create(gpa, io, &d.driver, d.stream, weights, &d.blas);
    }
};

const Check = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    dir: []const u8,
    worst: f64 = 1,

    fn tap(ctx: *anyopaque, name: []const u8, values: []const f32, rows: usize, width: usize) anyerror!void {
        const c: *Check = @ptrCast(@alignCast(ctx));
        const path = try std.fmt.allocPrint(c.gpa, "{s}/{s}.npy", .{ c.dir, name });
        defer c.gpa.free(path);
        const bytes = try std.Io.Dir.cwd().readFileAlloc(c.io, path, c.gpa, .limited(1 << 30));
        defer c.gpa.free(bytes);
        const want = try (try z3.npy.parse(bytes)).floats(c.gpa);
        defer c.gpa.free(want);
        if (want.len != rows * width) return error.AlignerDumpShape;
        var worst: f64 = 1;
        for (0..rows) |r| {
            var dot: f64 = 0;
            var mine: f64 = 0;
            var theirs: f64 = 0;
            for (values[r * width ..][0..width], want[r * width ..][0..width]) |x, y| {
                dot += @as(f64, x) * y;
                mine += @as(f64, x) * x;
                theirs += @as(f64, y) * y;
            }
            worst = @min(worst, dot / @sqrt(@max(mine * theirs, 1e-300)));
        }
        c.worst = @min(c.worst, worst);
        std.debug.print("  {s:<10} rows {d:>5} worst-row cos {d:.6}\n", .{ name, rows, worst });
    }
};

fn ints(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, comptime name: []const u8) ![]i32 {
    const path = try std.fs.path.join(gpa, &.{ dir, name });
    defer gpa.free(path);
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 24));
    defer gpa.free(bytes);
    return (try z3.npy.parse(bytes)).ints(gpa);
}

/// DIR's audio.npy and words.json through the aligner in `weights`: each word's times beside the reference's.
pub fn run(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, weights: []const u8) !u8 {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const cwd = std.Io.Dir.cwd();
    const audio = try (try z3.npy.parse(try cwd.readFileAlloc(io, try std.fs.path.join(a, &.{ dir, "audio.npy" }), a, .limited(1 << 30)))).floats(a);
    const doc = (try std.json.parseFromSliceLeaky(std.json.Value, a, try cwd.readFileAlloc(io, try std.fs.path.join(a, &.{ dir, "words.json" }), a, .limited(1 << 24)), .{})).object;
    var words: std.ArrayList([]const u8) = .empty;
    for (doc.get("tokens").?.array.items) |v| if (z3.align_words.isWordy(v.string)) try words.append(a, v.string);
    var dev: Dev = undefined;
    try dev.open(gpa);
    defer dev.close();
    const t0 = gen.nowNs(io);
    const al = try dev.aligner(gpa, io, weights);
    defer al.destroy();
    const loaded = @as(f64, @floatFromInt(gen.nowNs(io) - t0)) / 1e9;
    const rate = try @import("check_speaker.zig").dumpRate(a, io, dir);
    const wav = try z3.features.resampleFrom(a, audio, rate);
    var c: Check = .{ .gpa = gpa, .io = io, .dir = dir };
    const first = try al.forward(a, wav, words.items, .{ .ctx = &c, .f = Check.tap });
    const t1 = gen.nowNs(io);
    const found = try al.forward(a, wav, words.items, null);
    const seconds = @as(f64, @floatFromInt(gen.nowNs(io) - t1)) / 1e9;
    const want_start = try ints(a, io, dir, "start_ms.npy");
    const want_end = try ints(a, io, dir, "end_ms.npy");
    var exact: usize = 0;
    var total: u64 = 0;
    var widest: u64 = 0;
    for (found.start_ms, found.end_ms, want_start, want_end, first.start_ms, first.end_ms) |s, e, ws_, we, s1, e1| {
        if (s != s1 or e != e1) return error.AlignerNotRepeatable;
        const ds = @abs(@as(i64, s) - ws_);
        const de = @abs(@as(i64, e) - we);
        exact += @intFromBool(ds == 0 and de == 0);
        total += ds + de;
        widest = @max(widest, @max(ds, de));
    }
    const n = found.start_ms.len;
    const mean = @as(f64, @floatFromInt(total)) / @as(f64, @floatFromInt(2 * n));
    const pass = n == want_start.len and widest <= 20 and c.worst > 0.99;
    std.debug.print("RESULT zonos3 aligner {s} words={d} identical={d} mean_abs_ms={d:.2} widest_ms={d} worst_row_cos={d:.6} audio_s={d:.2} seconds={d:.3} load_seconds={d:.2} weight_bytes={d}\n", .{ if (pass) "PASS" else "FAIL", n, exact, mean, widest, c.worst, @as(f64, @floatFromInt(audio.len)) / @as(f64, @floatFromInt(rate)), seconds, loaded, al.bytes() });
    return if (pass) 0 else 1;
}
