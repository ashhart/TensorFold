//! `tf-bolt-vision <model dir> <dump dir> [reps]`: Nemotron 3.5 Bolt's image path end to end against the fp32 reference.
//!
//! The dump directory is tools/zig/bolt_vision_reference.py's (`--vision-only --dump DIR`): image_rgb.npy (the decoded
//! RGB the reference resized), meta.json, pixel_values.npy, tower_embed.npy, tower_out.npy, projector_out.npy. Host
//! preparation, the embedded rows, the last block's rows and the projector's rows are each compared (max |d|, the
//! relative Frobenius error, cosine over the whole tensor and the worst row's), then `reps` timed calls of `encode`.
//! Exits 1 when the projector rows' cosine is below 0.999 or the host pixels are off by more than 1e-4.
const std = @import("std");
const cuda = @import("cuda");
const nemotron = @import("nemotron");
const qi = @import("qwen_image");

const V = nemotron.vision;
const NI = qi.nemotron;

pub fn main(init: std.process.Init) !u8 {
    const a = init.arena.allocator();
    f32_arena = a;
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
    if (args.len < 3 or args.len > 4) {
        std.debug.print("usage: tf-bolt-vision <model dir> <dump dir> [reps]\n", .{});
        return 2;
    }
    const model = args[1];
    const dir = args[2];
    const reps: usize = if (args.len == 4) try std.fmt.parseInt(usize, args[3], 10) else 3;
    const cwd = std.Io.Dir.cwd();

    // host: decode is the shared decoder's; the reference's RGB is the input here
    const rgb = try readNpy(a, io, dir, "image_rgb.npy");
    if (rgb.shape.len != 3 or rgb.shape[2] != 3 or !std.mem.eql(u8, rgb.descr, "|u1")) return error.BadFixture;
    const im: NI.Image = .{ .w = @intCast(rgb.shape[1]), .h = @intCast(rgb.shape[0]), .rgb = @constCast(rgb.data) };
    const meta = try std.json.parseFromSliceLeaky(std.json.Value, a, try cwd.readFileAlloc(io, try std.fs.path.join(a, &.{ dir, "meta.json" }), a, .limited(1 << 20)), .{});
    const ghw = meta.object.get("grid_hw").?.array.items;
    var t0 = now(io);
    const g = try NI.grid(im.w, im.h);
    const px = try NI.pixelValues(a, im, g[1] * NI.PATCH, g[0] * NI.PATCH);
    const patches = try NI.patchify(a, px, g[0], g[1]);
    const host_ms = ms(io, t0);
    std.debug.print("image {d}x{d} -> grid {d}x{d} (reference {d}x{d}), {d} patches, {d} image tokens; host prep {d:.1} ms\n", .{ im.w, im.h, g[0], g[1], ghw[0].integer, ghw[1].integer, @as(usize, g[0]) * g[1], @as(usize, g[0]) * g[1] / 4, host_ms });
    if (g[0] != ghw[0].integer or g[1] != ghw[1].integer) return 1;
    var fail = false;
    {
        const want = try readNpy(a, io, dir, "pixel_values.npy");
        const st = compare(f32Slice(want.data), px, 3 * @as(usize, g[1]) * NI.PATCH);
        report("pixel_values", st);
        if (st.max_abs > 1e-4) fail = true;
    }

    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    var stream = try cuda.Stream.init(ctx.d, true);
    defer stream.deinit();
    const before = try ctx.memInfo();
    t0 = now(io);
    const tower = try V.Tower.load(init.gpa, io, ctx.d, model);
    defer tower.deinit();
    const after = try ctx.memInfo();
    std.debug.print("loaded in {d:.0} ms: arena {d:.3} GB (weights {d:.3} GB, scratch for {d} patches {d:.3} GB); device use +{d:.3} GB\n", .{ ms(io, t0), gb(tower.arena.len), gb(V.Tower.weightBytes()), V.max_patches, gb(V.Tower.scratchBytes(V.max_patches)), gb(before.free -| after.free) });

    const n = @as(usize, g[0]) * g[1];
    const rows = V.prefix_tokens + n;
    const tokens = n / 4;
    var out = try cuda.DeviceBuffer.alloc(ctx.d, tokens * V.out_hidden * 2);
    defer out.free();
    const xs = try a.alloc(f32, rows * V.hidden);

    // the embedded rows: no blocks run
    tower.run_depth = 0;
    var stages = try tower.encode(stream, patches, g[0], g[1], out.ptr);
    try dl(ctx.d, stages.x, std.mem.sliceAsBytes(xs));
    report("tower_embed", compare(f32Slice((try readNpy(a, io, dir, "tower_embed.npy")).data), xs, V.hidden));
    tower.run_depth = V.depth;

    var times: std.ArrayList(f64) = .empty;
    for (0..@max(reps, 1)) |_| {
        t0 = now(io);
        stages = try tower.encode(stream, patches, g[0], g[1], out.ptr);
        try times.append(a, ms(io, t0));
    }
    try dl(ctx.d, stages.x, std.mem.sliceAsBytes(xs));
    const tower_want = f32Slice((try readNpy(a, io, dir, "tower_out.npy")).data);
    report("tower_out (all rows)", compare(tower_want, xs, V.hidden));
    report("tower_features (patch rows)", compare(tower_want[V.prefix_tokens * V.hidden ..], xs[V.prefix_tokens * V.hidden ..], V.hidden));
    const got16 = try a.alloc(u16, tokens * V.out_hidden);
    try out.download(0, std.mem.sliceAsBytes(got16));
    const got = try a.alloc(f32, got16.len);
    for (got, got16) |*o, h| o.* = @bitCast(@as(u32, h) << 16);
    const proj = compare(f32Slice((try readNpy(a, io, dir, "projector_out.npy")).data), got, V.out_hidden);
    report("projector_out", proj);
    if (proj.cosine < 0.999) fail = true;
    std.debug.print("encode ms ({d} patches, {d} tokens; first call makes the GEMM plans):", .{ n, tokens });
    for (times.items) |t| std.debug.print(" {d:.1}", .{t});
    std.debug.print("\n{s}\n", .{if (fail) "FAIL" else "PASS"});
    return if (fail) 1 else 0;
}

const Stats = struct { max_abs: f64, max_ref: f64, rel: f64, cosine: f64, row_min: f64, rows: usize };

/// `got` against `want`, both row-major with rows of `width`.
fn compare(want: []const f32, got: []const f32, width: usize) Stats {
    std.debug.assert(want.len == got.len);
    var s: Stats = .{ .max_abs = 0, .max_ref = 0, .rel = 0, .cosine = 0, .row_min = 1, .rows = got.len / width };
    var dot: f64 = 0;
    var nw: f64 = 0;
    var ng: f64 = 0;
    var dd: f64 = 0;
    for (0..s.rows) |r| {
        var rd: f64 = 0;
        var rw: f64 = 0;
        var rg: f64 = 0;
        for (want[r * width ..][0..width], got[r * width ..][0..width]) |w0, g0| {
            const w: f64 = w0;
            const gg: f64 = g0;
            s.max_abs = @max(s.max_abs, @abs(w - gg));
            s.max_ref = @max(s.max_ref, @abs(w));
            rd += w * gg;
            rw += w * w;
            rg += gg * gg;
            dd += (w - gg) * (w - gg);
        }
        dot += rd;
        nw += rw;
        ng += rg;
        if (rw > 0 and rg > 0) s.row_min = @min(s.row_min, rd / @sqrt(rw * rg));
    }
    s.cosine = dot / @sqrt(nw * ng);
    s.rel = @sqrt(dd / nw);
    return s;
}

fn report(name: []const u8, s: Stats) void {
    std.debug.print("{s}: {d} rows, max|d| {e:.3} (max|ref| {d:.3}), rel {e:.3}, cosine {d:.6}, worst row {d:.6}\n", .{ name, s.rows, s.max_abs, s.max_ref, s.rel, s.cosine, s.row_min });
}

const Npy = struct { descr: []const u8, shape: []usize, data: []const u8 };

/// A little-endian C-order .npy (version 1 or 2 header).
fn readNpy(a: std.mem.Allocator, io: std.Io, dir: []const u8, name: []const u8) !Npy {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(a, &.{ dir, name }), a, .limited(1 << 31));
    if (bytes.len < 10 or !std.mem.eql(u8, bytes[0..6], "\x93NUMPY")) return error.BadNpy;
    const v2 = bytes[6] >= 2;
    const hl: usize = if (v2) std.mem.readInt(u32, bytes[8..12], .little) else std.mem.readInt(u16, bytes[8..10], .little);
    const start: usize = if (v2) 12 else 10;
    const header = bytes[start..][0..hl];
    if (std.mem.indexOf(u8, header, "'fortran_order': True") != null) return error.BadNpy;
    const d0 = (std.mem.indexOf(u8, header, "'descr': '") orelse return error.BadNpy) + 10;
    const d1 = std.mem.indexOfScalarPos(u8, header, d0, '\'') orelse return error.BadNpy;
    const s0 = (std.mem.indexOf(u8, header, "'shape': (") orelse return error.BadNpy) + 10;
    const s1 = std.mem.indexOfScalarPos(u8, header, s0, ')') orelse return error.BadNpy;
    var shape: std.ArrayList(usize) = .empty;
    var it = std.mem.tokenizeAny(u8, header[s0..s1], ", ");
    while (it.next()) |tok| try shape.append(a, try std.fmt.parseInt(usize, tok, 10));
    return .{ .descr = header[d0..d1], .shape = shape.items, .data = bytes[start + hl ..] };
}

var f32_arena: std.mem.Allocator = undefined;

/// The npy payload as f32 (copied: the file's bytes need not be 4-aligned in memory).
fn f32Slice(b: []const u8) []const f32 {
    const out = f32_arena.alloc(f32, b.len / 4) catch @panic("out of memory");
    @memcpy(std.mem.sliceAsBytes(out), b[0 .. out.len * 4]);
    return out;
}

fn dl(d: *const cuda.Driver, src: u64, out: []u8) !void {
    try d.check(d.api.cuMemcpyDtoH_v2(out.ptr, src, out.len), "cuMemcpyDtoH");
}

fn now(io: std.Io) i96 {
    return std.Io.Clock.awake.now(io).toNanoseconds();
}

fn ms(io: std.Io, t0: i96) f64 {
    return @as(f64, @floatFromInt(now(io) - t0)) / 1e6;
}

fn gb(b: usize) f64 {
    return @as(f64, @floatFromInt(b)) / 1e9;
}
