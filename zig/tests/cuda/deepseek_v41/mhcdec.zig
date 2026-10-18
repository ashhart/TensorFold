//! TF_DSV41_MHC_PFDEC kernels against the reference decode mHC site (oracle.py `mhcdec_case`), bit for bit, 1-64 rows.

const std = @import("std");
const cuda = @import("cuda");
const dsv41 = @import("dsv41_kernels");
const check = @import("../check.zig");
const Fixture = @import("../fixture.zig").Fixture;

const Gpu = check.Gpu;
const ops = dsv41.ops;

const Bufs = struct {
    gpu: Gpu,
    fx: *const Fixture,
    list: std.ArrayList(cuda.DeviceBuffer) = .empty,

    fn deinit(b: *Bufs) void {
        for (b.list.items) |*x| x.free();
        b.list.deinit(b.gpu.gpa);
    }

    fn up(b: *Bufs, name: []const u8) !cuda.DeviceBuffer {
        const host = try b.fx.bytes(name);
        defer b.gpu.gpa.free(host);
        const buf = try cuda.DeviceBuffer.fromHost(b.gpu.d, host);
        try b.list.append(b.gpu.gpa, buf);
        return buf;
    }

    fn zeros(b: *Bufs, len: usize) !cuda.DeviceBuffer {
        const buf = try cuda.DeviceBuffer.alloc(b.gpu.d, @max(len, 4));
        try buf.fill8(0, null);
        try b.list.append(b.gpu.gpa, buf);
        return buf;
    }
};

fn same(gpu: Gpu, fx: Fixture, buf: cuda.DeviceBuffer, name: []const u8, what: []const u8) !void {
    const want = try fx.bytes(name);
    defer gpu.gpa.free(want);
    const got = try gpu.gpa.alloc(u8, want.len);
    defer gpu.gpa.free(got);
    try buf.download(0, got);
    try check.sameBytes(what, got, want);
}

/// cvt.rn.bf16.f32 of a finite value.
fn bf16Rne(f: f32) u16 {
    const u: u32 = @bitCast(f);
    return @intCast((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

/// `_finish_k`'s normed input: rn = div_rn(1, sqrt_rn(div_rn(ssc, D) + eps)), then bf16((c * rn) * w) per column.
fn normed(part: []const f32, c: []const u16, nw: []const f32, R: usize, D: usize, eps: f32, out: []u16) void {
    for (0..R) |r| {
        var ssc: f32 = 0.0;
        for (0..40) |b| ssc += part[(r * 4 * 40 + b) * 32 + 25];
        const rn: f32 = 1.0 / @sqrt(ssc / @as(f32, @floatFromInt(D)) + eps);
        for (0..D) |d| {
            const cv: f32 = @bitCast(@as(u32, c[r * D + d]) << 16);
            out[r * D + d] = bf16Rne((cv * rn) * nw[d]);
        }
    }
}

pub fn cases(gpu: Gpu, k: *const dsv41.Kernels, dir: []const u8) !void {
    var fx = try Fixture.open(gpu.gpa, gpu.io, dir);
    defer fx.deinit();
    var b: Bufs = .{ .gpu = gpu, .fx = &fx };
    defer b.deinit();
    var stream = try cuda.Stream.init(gpu.d, false); // blocking: ordered after the uploads on the default stream
    defer stream.deinit();
    const o = k.others(stream);
    const R: usize = @intCast(try fx.int("R"));
    const D: usize = @intCast(try fx.int("D"));
    const W: usize = @intCast(try fx.int("W"));
    const mode: u32 = @intCast(try fx.int("mode"));
    const tap = try fx.int("tap") != 0;
    const in_place = try fx.int("in_place") != 0;

    const x = try b.up("x");
    const xout = if (in_place) x else try b.up("xout0");
    const g = try b.up("g");
    const ppre = try b.up("ppre");
    const ppost = try b.up("ppost");
    const pcomb = try b.up("pcomb");
    const f = try b.up("fn");
    const base = try b.up("base");
    const scale = try b.up("scale");
    const taps = try b.up("taps0");
    const part = try b.zeros(R * 4 * 40 * 32 * 4);
    const c = try b.zeros(R * D * 2);
    const npre = try b.zeros(R * 4 * 4);
    const npost = try b.zeros(R * 4 * 4);
    const ncomb = try b.zeros(R * 16 * 4);

    // the forward's PFDEC triple: block_wide.zig's mhc_pf call, then the coefficients (norm checked on the host below)
    const post = mode == 0;
    const a: ops.mhc_pf.Args = .{
        .x = x.ptr,
        .xs = @intCast(4 * D),
        .xout = if (post) xout.ptr else 0,
        .g = if (post) g.ptr else 0,
        .gr = if (post) @intCast(R * D) else 0,
        .world = if (post) @intCast(W) else 1,
        .gbf16 = @intFromBool(post),
        .post = if (post) ppost.ptr else 0,
        .comb = if (post) pcomb.ptr else 0,
        .pre = if (mode != 1) ppre.ptr else 0,
        .@"fn" = f.ptr,
        .part = part.ptr,
        .c = c.ptr,
        .tap = if (tap) taps.ptr + D * 2 else 0, // column block 1 of [R, 3 D]
        .ts = if (tap) @intCast(3 * D) else 0,
        .R = @intCast(R),
    };
    try o.mhcSite(a, mode, false, tap);
    try o.mhcCoef(.{
        .part = part.ptr,
        .base = base.ptr,
        .scale = scale.ptr,
        .opre = npre.ptr,
        .opost = npost.ptr,
        .ocomb = ncomb.ptr,
        .R = @intCast(R),
        .eps = @floatCast(try fx.float("eps")),
        .hc_eps = @floatCast(try fx.float("hc_eps")),
        .post_alpha = @floatCast(try fx.float("post_alpha")),
        .iters = @intCast(try fx.int("iters")),
    });
    try stream.synchronize();

    const what = std.fs.path.basename(dir);
    var name_buf: [160]u8 = undefined;
    const n = struct {
        fn of(buf: []u8, case: []const u8, part_name: []const u8) []const u8 {
            return std.fmt.bufPrint(buf, "{s}: {s}", .{ case, part_name }) catch part_name;
        }
    };
    if (post) try same(gpu, fx, xout, "xout", n.of(&name_buf, what, "new streams (mhc_pf vs _site)"));
    try same(gpu, fx, x, "xs", n.of(&name_buf, what, "streams after the site"));
    try same(gpu, fx, c, "c", n.of(&name_buf, what, "collapsed row"));
    try same(gpu, fx, taps, "taps", n.of(&name_buf, what, "DSpark tap buffer"));
    try same(gpu, fx, part, "part", n.of(&name_buf, what, "partials [R, 4, 40, 32]"));
    try same(gpu, fx, npre, "npre", n.of(&name_buf, what, "next pre (coef_kernel vs _finish_k)"));
    try same(gpu, fx, npost, "npost", n.of(&name_buf, what, "next post"));
    try same(gpu, fx, ncomb, "ncomb", n.of(&name_buf, what, "next comb"));

    // the normed input from these partials and this collapsed row, against the reference `_finish_k`
    const ph = try check.download(gpu, part);
    defer gpu.gpa.free(ph);
    const ch = try check.download(gpu, c);
    defer gpu.gpa.free(ch);
    const nwb = try fx.bytes("nw");
    defer gpu.gpa.free(nwb);
    const got = try gpu.gpa.alloc(u16, R * D);
    defer gpu.gpa.free(got);
    const pf: []const f32 = @alignCast(std.mem.bytesAsSlice(f32, ph));
    const cf: []const u16 = @alignCast(std.mem.bytesAsSlice(u16, ch[0 .. R * D * 2]));
    const wf: []const f32 = @alignCast(std.mem.bytesAsSlice(f32, nwb));
    normed(pf, cf, wf, R, D, @floatCast(try fx.float("eps")), got);
    const want = try fx.bytes("out");
    defer gpu.gpa.free(want);
    try check.sameBytes(n.of(&name_buf, what, "normed input (_finish_k without COEF's arithmetic)"), std.mem.sliceAsBytes(got), want);

    check.pass("BITEXACT mhcdec {s}: R {d}, mode {d}{s}{s}: mhc_pf site + coef_kernel + normed input == Triton _site + _finish_k", .{ what, R, mode, if (tap) ", tap" else "", if (in_place) ", in place" else "" });
}
