//! tf-moss-image-decoder: MOSS-Image's image decoder on a latents file, compared with a float64 decode of it.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const mi = tf.moss_image;

const usage =
    \\tf-moss-image-decoder MODEL_DIR LATENTS WIDTH HEIGHT REFERENCE TORCH OUT
    \\MODEL_DIR is the OpenMOSS-Team/MOSS-Image-Base folder; LATENTS the transformer's output (height/16 x width/16 rows of
    \\128 float32, as tf-moss-image writes it). REFERENCE and TORCH hold the publisher's decoder's output before the
    \\clamp, [height, width, 3], in float64 and in float32 as the pipeline runs it (moss_image_case.py vae). The decode
    \\must be as close to float64 as twice torch float32's cosine error and worst pixel row, and a second decode must
    \\repeat the first byte for byte; else it fails. It is written to OUT as a PNG and timed.
;

const check = mi.reference;

pub fn main(init: std.process.Init) !u8 {
    const a = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
    if (args.len != 8) {
        std.debug.print("{s}\n", .{usage});
        return 2;
    }
    const width = try std.fmt.parseInt(usize, args[3], 10);
    const height = try std.fmt.parseInt(usize, args[4], 10);
    const cwd = std.Io.Dir.cwd();
    const lat_bytes = try cwd.readFileAlloc(io, args[2], a, .limited(1 << 30));
    const latents = try a.alloc(f32, lat_bytes.len / 4);
    @memcpy(std.mem.sliceAsBytes(latents), lat_bytes[0 .. latents.len * 4]);
    const ref_bytes = try cwd.readFileAlloc(io, args[5], a, .limited(1 << 32));
    const ref = try a.alloc(f64, ref_bytes.len / 8);
    @memcpy(std.mem.sliceAsBytes(ref), ref_bytes[0 .. ref.len * 8]);
    const torch_bytes = try cwd.readFileAlloc(io, args[6], a, .limited(1 << 31));
    const torch = try a.alloc(f32, torch_bytes.len / 4);
    @memcpy(std.mem.sliceAsBytes(torch), torch_bytes[0 .. torch.len * 4]);
    for ([_]struct { []const u8, usize }{ .{ args[5], ref.len }, .{ args[6], torch.len } }) |f| if (f[1] != width * height * 3) {
        std.debug.print("{s} holds {d} values; {d}x{d} needs {d}\n", .{ f[0], f[1], width, height, width * height * 3 });
        return 2;
    };

    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    const loading = mtl.clock.seconds();
    const vae = try std.fs.path.join(a, &.{ args[1], "vae" });
    const d = try mi.Decoder.load(std.heap.page_allocator, io, device, vae, width, height);
    defer d.deinit();
    const load_s = mtl.clock.seconds() - loading;
    const rgb = try a.alloc(u8, width * height * 3);
    const first = try d.decode(latents, rgb);
    const out = d.output();
    std.debug.print("{s}: {d}x{d} decoded; loaded in {d:.2} s, first decode {d:.3} s\n", .{ device.name(), width, height, load_s, first.wall });
    const ours = check.match(f32, out, ref, width * 3);
    const theirs = check.match(f32, torch, ref, width * 3);
    const v = check.within(ours, theirs);
    const torch_rgb = try a.alloc(u8, rgb.len);
    for (torch_rgb, torch) |*p, x| p.* = pixel(x);
    for ([_]struct { []const u8, []const u8, check.Match }{ .{ "ours", rgb, ours }, .{ "torch float32", torch_rgb, theirs } }) |c| {
        const px = pixels(c[1], ref);
        std.debug.print("{s} against float64: max |diff| {e:.3}, 1-cosine {e:.2}, worst row {e:.3} (row {d}); 8-bit values differing {d} of {d}, by at most {d}\n", .{ c[0], c[2].max_abs, 1 - c[2].cosine, c[2].worst_row, c[2].worst_at, px.differ, rgb.len, px.off });
    }
    std.debug.print("gate: 1-cosine {e:.2} of {e:.2}, worst row {e:.3} of {e:.3}: {s}\n", .{ v.cosine_err, v.cosine_limit, v.row, v.row_limit, if (v.pass) "pass" else "FAIL" });
    try cwd.writeFile(io, .{ .sub_path = args[7], .data = try mi.png.encode(a, rgb, width, height) });
    const again = try a.alloc(u8, rgb.len);
    const t = try d.decode(latents, again);
    const same = std.mem.eql(u8, rgb, again);
    std.debug.print("decoded again in {d:.3} s ({d:.3} s on the GPU), the same bytes: {}\n", .{ t.wall, t.gpu, same });
    const pass = check.accept(&.{v}, same);
    std.debug.print("{s}\n", .{if (pass) "PASS within the gate, repeated byte for byte" else "FAIL"});
    return if (pass) 0 else 1;
}

/// The 8-bit values in `got` that differ from float64's, and by how much at most.
fn pixels(got_all: []const u8, ref: []const f64) struct { differ: usize, off: u8 } {
    var differ: usize = 0;
    var off: u8 = 0;
    for (got_all, ref) |got, r| {
        const want = pixel(r);
        if (want != got) {
            differ += 1;
            off = @max(off, if (want > got) want - got else got - want);
        }
    }
    return .{ .differ = differ, .off = off };
}

/// The 8-bit value of a decoder output in float64: clamped, (x * 0.5 + 0.5) * 255 rounded half to even.
fn pixel(x: f64) u8 {
    const v = std.math.clamp(std.math.clamp(x, -1, 1) * 0.5 + 0.5, 0, 1) * 255;
    return @intFromFloat(@round(v) - @as(f64, if (@abs(v - @trunc(v)) == 0.5 and @mod(@round(v), 2) == 1) 1 else 0));
}
