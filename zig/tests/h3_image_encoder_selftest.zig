//! tf-h3-image-encoder selftest: each kernel on small synthetic weights against a float64 host reference, gated.
const std = @import("std");
const mtl = @import("metal");
const he = @import("h3_image_encoder");

const ref = he.reference;
const vae = he.vae;
const Item = ref.Item;

/// Bounds fixed before any run: the worst |GPU - host| over the case's largest |host|.
pub const op_bound: f64 = 1e-5;
pub const encode_bound: f64 = 1e-4;
/// The fp16 posterior samples: one fp16 step at the largest sample's magnitude (a step near zero is far finer).
pub const sample_bound: f64 = 1.0 / 1024.0;

/// Widths that change where the release's do (levels 1, 3, 5), with 1 to 4 channels a group.
const small: ref.Arch = .{ .widths = .{ 32, 64, 64, 96, 96, 128 } };

/// Synthetic tensors by name, as `packLayers` reads a checkpoint.
const Weights = struct {
    map: std.StringHashMapUnmanaged([]f32) = .empty,

    pub fn get(w: Weights, name: []const u8) ?[]align(1) const f32 {
        return w.map.get(name);
    }

    fn of(w: Weights, it: Item, part: []const u8) []const f32 {
        var buf: [96]u8 = undefined;
        return w.map.get(std.fmt.bufPrint(&buf, "{s}.{s}", .{ it.name, part }) catch unreachable).?;
    }
};

fn uniform(a: std.mem.Allocator, rng: std.Random, n: usize, centre: f32, scale: f32) ![]f32 {
    const out = try a.alloc(f32, n);
    for (out) |*v| v.* = centre + scale * (rng.float(f32) * 2 - 1);
    return out;
}

/// Kernels at about 1.7 / sqrt(fan in) in all three time slices, small biases, norms near 1 and 0.
fn weights(a: std.mem.Allocator, rng: std.Random, list: []const Item) !Weights {
    var w: Weights = .{};
    for (list) |it| {
        const fan: f32 = @floatFromInt(it.cin * ref.taps(it));
        const n = if (it.kind == .norm) it.cin else it.cout * it.cin * (if (it.kind == .conv3) @as(usize, 27) else 1);
        const kernel = if (it.kind == .norm) try uniform(a, rng, n, 1, 0.2) else try uniform(a, rng, n, 0, 1.7 / @sqrt(fan));
        try w.map.put(a, try std.fmt.allocPrint(a, "{s}.weight", .{it.name}), kernel);
        try w.map.put(a, try std.fmt.allocPrint(a, "{s}.bias", .{it.name}), try uniform(a, rng, it.cout, 0, if (it.kind == .norm) 0.2 else 0.1));
    }
    return w;
}

/// fp64 convolution of x [h * w, cin] with a kernel's last time slice, reflect-padded by one (or the downsample's).
fn hostConv(a: std.mem.Allocator, it: Item, ws: Weights, x: []const f64, h: usize, w: usize, down: bool, r: ?[]const f64) ![]f64 {
    const k: usize = if (it.kind == .conv3) 3 else 1;
    const kw = ws.of(it, "weight");
    const b = ws.of(it, "bias");
    const oh = if (down) h / 2 else h;
    const ow = if (down) w / 2 else w;
    const stride: isize = if (down) 2 else 1;
    const pad: isize = if (k == 3 and !down) 1 else 0;
    const wt = try a.alloc(f64, k * k * it.cin * it.cout);
    for (0..it.cout) |co| for (0..it.cin) |ci| for (0..k * k) |tap| {
        wt[(tap * it.cin + ci) * it.cout + co] = if (k == 3) kw[((co * it.cin + ci) * 3 + 2) * 9 + tap] else kw[co * it.cin + ci];
    };
    const y = try a.alloc(f64, oh * ow * it.cout);
    for (0..oh) |oy| for (0..ow) |ox| {
        const acc = y[(oy * ow + ox) * it.cout ..][0..it.cout];
        for (acc, b) |*v, bias| v.* = bias;
        for (0..k * k) |tap| {
            const iy = ref.reflect(@as(isize, @intCast(oy)) * stride + @as(isize, @intCast(tap / k)) - pad, h);
            const ix = ref.reflect(@as(isize, @intCast(ox)) * stride + @as(isize, @intCast(tap % k)) - pad, w);
            for (0..it.cin) |ci| {
                const xv = x[(iy * w + ix) * it.cin + ci];
                for (acc, wt[(tap * it.cin + ci) * it.cout ..][0..it.cout]) |*v, wv| v.* += xv * wv;
            }
        }
        if (r) |res| for (acc, res[(oy * ow + ox) * it.cout ..][0..it.cout]) |*v, rv| {
            v.* += rv;
        };
    };
    return y;
}

/// fp64 SiLU(GroupNorm(x)) over x [pixels, channels]: 32 groups, biased variance, eps 1e-6.
fn hostNorm(a: std.mem.Allocator, it: Item, ws: Weights, x: []const f64, pixels: usize) ![]f64 {
    const c = it.cin;
    const per = c / ref.groups;
    const g = ws.of(it, "weight");
    const b = ws.of(it, "bias");
    const y = try a.alloc(f64, x.len);
    for (0..ref.groups) |gr| {
        var sum: f64 = 0;
        for (0..pixels) |p| for (0..per) |j| {
            sum += x[p * c + gr * per + j];
        };
        const n: f64 = @floatFromInt(pixels * per);
        const mean = sum / n;
        var sq: f64 = 0;
        for (0..pixels) |p| for (0..per) |j| {
            const d = x[p * c + gr * per + j] - mean;
            sq += d * d;
        };
        const rstd = 1 / @sqrt(sq / n + vae.group_eps);
        for (0..pixels) |p| for (0..per) |j| {
            const at = p * c + gr * per + j;
            const v = (x[at] - mean) * rstd * @as(f64, g[gr * per + j]) + @as(f64, b[gr * per + j]);
            y[at] = v / (1 + @exp(-v));
        };
    }
    return y;
}

/// fp64 block: items norm1, conv1, norm2, conv2 and any shortcut.
fn hostBlock(a: std.mem.Allocator, ls: []const Item, ws: Weights, x: []const f64, h: usize, w: usize) ![]f64 {
    const n1 = try hostNorm(a, ls[0], ws, x, h * w);
    const c1 = try hostConv(a, ls[1], ws, n1, h, w, false, null);
    const n2 = try hostNorm(a, ls[2], ws, c1, h * w);
    const skip = if (ls.len == 5) try hostConv(a, ls[4], ws, x, h, w, false, null) else x;
    return hostConv(a, ls[3], ws, n2, h, w, false, skip);
}

/// fp64 tile: pixels [h, w, 3] to moments [48, h / 16, w / 16], rounded once to fp32.
fn hostTile(a: std.mem.Allocator, list: []const Item, ws: Weights, pixels: []const f32, h: usize, w: usize, out: []f32) !void {
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const x = arena.allocator();
    const in = try x.alloc(f64, pixels.len);
    for (in, pixels) |*d, v| d.* = v;
    var t = try hostConv(x, list[0], ws, in, h, w, false, null);
    var at: usize = 1;
    var rh = h;
    var rw = w;
    for (0..ref.levels) |l| {
        for (0..2) |_| {
            const n: usize = if (list[at].cin != list[at + 1].cout) 5 else 4;
            t = try hostBlock(x, list[at..][0..n], ws, t, rh, rw);
            at += n;
        }
        if (l < ref.downsampled) {
            t = try hostConv(x, list[at], ws, t, rh, rw, true, null);
            at += 1;
            rh /= 2;
            rw /= 2;
        }
    }
    t = try hostConv(x, list[at + 1], ws, try hostNorm(x, list[at], ws, t, rh * rw), rh, rw, false, null);
    t = try hostConv(x, list[at + 2], ws, t, rh, rw, false, null);
    for (0..rh * rw) |p| for (0..ref.moments) |c| {
        out[c * rh * rw + p] = @floatCast(t[p * ref.moments + c]);
    };
}

var failed = false;

/// One check's line: the worst |GPU - host| over the largest |host|, against its bound.
fn gate(name: []const u8, gpu: []const f32, host: []const f64, bound: f64) void {
    var worst: f64 = 0;
    var top: f64 = 1e-30;
    for (gpu, host) |g, h| {
        worst = @max(worst, @abs(@as(f64, g) - h));
        top = @max(top, @abs(h));
    }
    const ok = std.math.isFinite(worst) and worst / top <= bound;
    failed = failed or !ok;
    std.debug.print("{s}: worst |gpu - host| {e:.3} over max |host| {d:.3} = {e:.3} (bound {e:.0}): {s}\n", .{ name, worst, top, worst / top, bound, if (ok) "pass" else "FAIL" });
}

const Gpu = struct {
    device: mtl.Device,
    queue: mtl.Queue,
    k: he.Kernels,

    /// A shared buffer holding `data`, with room for `room` floats.
    fn upload(g: *const Gpu, data: []const f32, room: usize) !he.Ref {
        const r: he.Ref = .{ .buf = try g.device.buffer(@max(room, data.len, 64) * 4, mtl.ResourceOptions.shared) };
        @memcpy(r.floats(data.len), data);
        return r;
    }

    fn submit(_: *const Gpu, cb: mtl.CommandBuffer, e: mtl.ComputeEncoder) !void {
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |msg| {
            std.debug.print("GPU failure: {s}\n", .{msg});
            return error.GpuFailure;
        }
    }
};

fn wide(a: std.mem.Allocator, v: []const f32) ![]f64 {
    const out = try a.alloc(f64, v.len);
    for (out, v) |*o, x| o.* = x;
    return out;
}

/// `list` (a block, or one layer) on random h x w input against the host; `down` makes a conv the downsample.
fn layerCase(a: std.mem.Allocator, g: *const Gpu, rng: std.Random, name: []const u8, list: []const Item, h: usize, w: usize, down: bool, residual: bool) !void {
    const ws = try weights(a, rng, list);
    const buf, const layers = try he.packLayers(a, g.device, list, ws);
    defer buf.deinit();
    const cin = list[0].cin;
    const cout = list[list.len - 1].cout;
    const most = h * w * @max(cin, list[1 % list.len].cout);
    const x = try uniform(a, rng, h * w * cin, 0, 1);
    const r = try uniform(a, rng, h * w * cout, 0, 1);
    const t = [3]he.Ref{ try g.upload(x, most), try g.upload(r, most), try g.upload(&.{}, most) };
    const stats = try g.upload(&.{}, 64);
    const part = try g.upload(&.{}, he.chunks(h * w) * @max(cin, cout));
    defer for (t ++ [2]he.Ref{ stats, part }) |b| b.buf.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const cb = g.queue.commandBuffer();
    const e = cb.compute(.serial);
    var out = t[1];
    var host: []f64 = undefined;
    const in = try wide(a, x);
    if (list.len >= 4) {
        out = t[he.block(e, &g.k, layers, t, 0, stats, part, h, w)];
        host = try hostBlock(a, list, ws, in, h, w);
    } else if (list[0].kind == .norm) {
        he.normSilu(e, &g.k, layers[0], t[0], t[2], stats, part, h * w);
        out = t[2];
        host = try hostNorm(a, list[0], ws, in, h * w);
    } else {
        he.conv(e, &g.k, layers[0], t[0], t[1], if (residual) t[1] else null, h, w, down);
        host = try hostConv(a, list[0], ws, in, h, w, down, if (residual) try wide(a, r) else null);
    }
    try g.submit(cb, e);
    gate(name, out.floats(host.len), host, op_bound);
}

/// The small encoder on a picture in overlapping tiles, against the host's tiles and the same blend.
fn encodeCase(gpa: std.mem.Allocator, a: std.mem.Allocator, g: *const Gpu, rng: std.Random) !void {
    const list = try ref.items(a, small);
    const ws = try weights(a, rng, list);
    const c: vae.Config = .{ .mean = @splat(0), .std = @splat(1), .tile = 64, .overlap = 32, .rope_theta = 100 };
    const s: ref.Size = .{ .width = 176, .height = 96 };
    const full = ref.fullGrid(s);
    const image = try uniform(a, rng, s.width * s.height * 3, 0, 2);
    const enc = try he.Encoder.init(gpa, g.device, small, c.tile, ws);
    defer enc.deinit();
    const n = ref.moments * full.rows * full.columns;
    const gpu = try a.alloc(f32, n);
    const took = try enc.encode(a, &c, image, s, gpu);
    const again = try a.alloc(f32, n);
    _ = try enc.encode(a, &c, image, s, again);
    const rows = try vae.split(a, s.height, c.tile, c.overlap);
    const cols = try vae.split(a, s.width, c.tile, c.overlap);
    const host = try a.alloc(f32, n);
    var blend = try vae.Assembler.init(a, try spans(a, rows), try spans(a, cols), ref.moments, host);
    const pixels = try a.alloc(f32, c.tile * c.tile * 3);
    const tile = try a.alloc(f32, ref.moments * (c.tile / 16) * (c.tile / 16));
    for (rows, 0..) |r, ri| for (cols, 0..) |col, ci| {
        for (0..r.len) |y| @memcpy(pixels[y * col.len * 3 ..][0 .. col.len * 3], image[((r.start + y) * s.width + col.start) * 3 ..][0 .. col.len * 3]);
        try hostTile(a, list, ws, pixels[0 .. r.len * col.len * 3], r.len, col.len, tile);
        try blend.place(ri, ci, tile);
    };
    std.debug.print("tiled encode: {d}x{d} in {d} x {d} tiles of {d} (overlaps {d} and {d}), {d:.3} s GPU\n", .{ s.width, s.height, rows.len, cols.len, c.tile, cols[0].overlap, cols[cols.len - 2].overlap, took.gpu });
    gate("tiled encode, blended moments", gpu, try wide(a, host), encode_bound);
    const same = std.mem.eql(u8, std.mem.sliceAsBytes(gpu), std.mem.sliceAsBytes(again));
    failed = failed or !same;
    std.debug.print("tiled encode, a second run: {s}\n", .{if (same) "bit-identical: pass" else "differs: FAIL"});
    const z = try uniform(a, rng, n / 2, 0, 2);
    const ours = try a.alloc(f32, n / 2);
    const theirs = try a.alloc(f32, n / 2);
    try vae.posterior(gpu[0 .. n / 2], gpu[n / 2 ..], z, ours);
    try vae.posterior(host[0 .. n / 2], host[n / 2 ..], z, theirs);
    var most: u32 = 0;
    var moved: usize = 0;
    for (ours, theirs) |o, t| {
        const d = he.picture.ulps(@bitCast(@as(f16, @floatCast(o))), @bitCast(@as(f16, @floatCast(t))));
        most = @max(most, d);
        moved += @intFromBool(d != 0);
    }
    std.debug.print("tiled encode, fp16 posterior samples: {d} of {d} differ, by at most {d} fp16 steps of their own\n", .{ moved, ours.len, most });
    gate("tiled encode, fp16 posterior samples", ours, try wide(a, theirs), sample_bound);
}

fn spans(a: std.mem.Allocator, px: []const vae.Span) ![]vae.Span {
    const out = try a.alloc(vae.Span, px.len);
    for (out, px) |*o, s| o.* = .{ .start = s.start / 16, .len = s.len / 16, .overlap = s.overlap / 16 };
    return out;
}

pub fn run(gpa: std.mem.Allocator, a: std.mem.Allocator) !u8 {
    const device = try mtl.Device.init();
    defer device.deinit();
    var g: Gpu = .{ .device = device, .queue = try device.queue(), .k = try he.Kernels.init(device) };
    defer g.queue.deinit();
    defer g.k.deinit();
    std.debug.print("{s}: bounds {e:.0} a layer, {e:.0} a tiled encode, {e:.3} a sample\n", .{ device.name(), op_bound, encode_bound, sample_bound });
    var prng = std.Random.DefaultPrng.init(0x4833);
    const rng = prng.random();
    try layerCase(a, &g, rng, "conv 3x3, reflect padding, 40 -> 72 at 13x17", &.{.{ .name = "c", .kind = .conv3, .cin = 40, .cout = 72 }}, 13, 17, false, false);
    try layerCase(a, &g, rng, "downsample 3x3 stride 2, right and bottom reflected, 64 at 14x18", &.{.{ .name = "d", .kind = .conv3, .cin = 64, .cout = 64 }}, 14, 18, true, false);
    try layerCase(a, &g, rng, "conv 1x1 plus a residual in place, 96 -> 48 at 10x6", &.{.{ .name = "q", .kind = .conv1, .cin = 96, .cout = 48 }}, 10, 6, false, true);
    try layerCase(a, &g, rng, "GroupNorm(32) + SiLU, 96 channels over 576 pixels", &.{.{ .name = "n", .kind = .norm, .cin = 96, .cout = 96 }}, 24, 24, false, false);
    const blocks = try ref.items(a, small);
    try layerCase(a, &g, rng, "ResNet block with nin_shortcut, 32 -> 64 at 16x16", blocks[10..15], 16, 16, false, false);
    try layerCase(a, &g, rng, "ResNet block, identity shortcut, 32 at 12x20", blocks[1..5], 12, 20, false, false);
    try encodeCase(gpa, a, &g, rng);
    std.debug.print("{s}\n", .{if (failed) "selftest FAILED" else "selftest passed"});
    return if (failed) 1 else 0;
}
