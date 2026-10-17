//! Gemma's fused lane extension; packing and ordinary projections use the unchanged shared lane module.
const std = @import("std");
const mtl = @import("metal");
const ks = @import("kernel_sources");
const lane = @import("../../core/lane_projection.zig");
const frags = @import("../../core/frags.zig");
pub const Layout = struct {
    projection: lane.Layout,
    source: []const u8,
    defines: []const u8 = "",
    x_rows: usize = 1,
    epilogue: bool = false,
    threads: usize = 256,
    pub fn tilesPerGroup(l: @This()) usize {
        return l.threads / (32 * l.projection.sk);
    }
    pub fn threadgroupBytes(l: @This()) usize {
        const p = l.projection;
        return l.tilesPerGroup() * @max(p.sk -| 1, 1) * 16 * 32 * 4 + l.x_rows * p.k * 2 + (p.k / p.format.group + 16) * 4 + 16 * 4;
    }
    pub fn validate(l: @This()) !void {
        try l.projection.validate();
        const p = l.projection;
        if (!p.reg or l.threads == 0 or l.threads > 1024 or l.threads % (32 * p.sk) != 0 or p.tileCount() % l.tilesPerGroup() != 0 or l.x_rows > 16 or p.ranges.len != 0) return error.InvalidFusedShape;
        if (l.threadgroupBytes() > 32768) return error.ThreadgroupMemoryLimit;
    }
};

pub fn source(a: std.mem.Allocator, l: Layout) ![]u8 {
    try l.validate();
    const base = try lane.source(a, l.projection);
    defer a.free(base);
    std.debug.assert(std.mem.endsWith(u8, base, ks.core_lane_projection));
    const head = base[0 .. base.len - ks.core_lane_projection.len];
    return std.fmt.allocPrint(a, "{s}#define TF_FUSED 1\n#define TF_TPT {d}\n#define TF_XROWS {d}\n#define TF_THREADS {d}\n{s}{s}struct tf_fused {{ const device bfloat* a0; const device bfloat* a1; const device bfloat* a2; const device bfloat* a3; const device bfloat* a4; const device bfloat* a5; device bfloat* o0; device bfloat* o1; float4 f; int4 i; }};\n{s}{s}", .{ head, l.tilesPerGroup(), l.x_rows, l.threads, if (l.epilogue) "#define TF_EPILOGUE 1\n" else "", l.defines, l.source, ks.gemma4_lane_projection });
}

pub const Args = struct { a: [6]lane.Ref, o: [2]lane.Ref, f: [4]f32 = .{ 0, 0, 0, 0 }, i: [4]i32 = .{ 0, 0, 0, 0 } };
pub const Projection = struct {
    pipe: mtl.Pipeline,
    layout: Layout,
    pub fn init(a: std.mem.Allocator, device: mtl.Device, l: Layout) !Projection {
        try l.validate();
        if (!device.tensorUnits()) return error.TensorUnitsRequired;
        const text = try source(a, l);
        defer a.free(text);
        const adapted = try frags.source(device, a, text);
        defer a.free(adapted);
        const lib = try mtl.Library.fromSource(device, adapted, mtl.CompileOptions.mlx());
        defer lib.deinit();
        const pipe = try mtl.Pipeline.init(device, lib, "tf_lane_reg", false);
        errdefer pipe.deinit();
        if (pipe.simdWidth() != 32 or pipe.maxThreads() < l.threads) return error.UnsupportedPipelineShape;
        return .{ .pipe = pipe, .layout = l };
    }
    pub fn deinit(p: Projection) void {
        p.pipe.deinit();
    }
    pub fn encodeFused(p: Projection, e: mtl.ComputeEncoder, x: lane.Ref, w: lane.Ref, sb: lane.Ref, y: lane.Ref, args: Args, rows: u32) !void {
        const l = p.layout;
        const b = l.projection;
        if (rows == 0 or rows > (if (l.x_rows > 0) l.x_rows else 16)) return error.InvalidRows;
        try check(x, rows * b.k * 2, 2);
        try check(w, b.weightWords() * 4, 16);
        try check(sb, b.metadataElements() * 2, 16);
        try check(y, ((rows - 1) * (b.y_stride orelse b.n) + b.n) * (if (b.output == .f32) @as(usize, 4) else 2), if (b.output == .f32) @as(usize, 4) else 2);
        e.setPipeline(p.pipe);
        e.setBuffer(x.buf, x.off, 0);
        e.setBuffer(w.buf, w.off, 1);
        e.setBuffer(sb.buf, sb.off, 2);
        e.setValue([2]i32{ @intCast(rows), 1 }, 3);
        e.setBuffer(y.buf, y.off, 4);
        e.setBuffer(x.buf, x.off, 5);
        for (args.a, 6..) |v, i| e.setBuffer(v.buf, v.off, i);
        for (args.o, 12..) |v, i| e.setBuffer(v.buf, v.off, i);
        e.setValue(args.f, 14);
        e.setValue(args.i, 15);
        e.dispatchGroups(mtl.Size.of(b.tiles() / l.tilesPerGroup(), 1, 1), mtl.Size.of(l.threads, 1, 1));
    }
};
fn check(r: lane.Ref, need: usize, alignment: usize) !void {
    if (r.off % alignment != 0) return error.MisalignedBuffer;
    if (r.off > r.buf.length() or need > r.buf.length() - r.off) return error.BufferTooSmall;
}
