//! Fused six-projection attention-input BF16 matvec for Inkling 2 Mini decode windows (1 to 32 rows).
const std = @import("std");
const mtl = @import("metal");
const sources = @import("inkling2_metal");

pub const Ref = @import("inkling2_metal_ref").Ref;
pub const row_tiers = [6]u32{ 1, 2, 4, 8, 16, 32 };
pub const threads: u32 = 256;

pub const InkMv6Args = extern struct {
    rows: u32,
    k: u32,
    nq: u32,
    nk: u32,
    nv: u32,
    nr: u32,
    ns: u32,
    ng: u32,
};

pub const Proj = struct { w: Ref, y: Ref, n: u32 };

pub const Pipelines = struct {
    dense: [6]mtl.Pipeline,

    pub fn load(device: mtl.Device) !Pipelines {
        return loadSource(device, sources.dense_fuse);
    }

    pub fn loadSource(device: mtl.Device, text: []const u8) !Pipelines {
        const lib = try mtl.Library.fromSource(device, text, .{
            .language = mtl.CompileOptions.mlx().language,
            .math_functions = .precise,
        });
        defer lib.deinit();
        var out: Pipelines = undefined;
        var made: usize = 0;
        errdefer for (out.dense[0..made]) |p| p.deinit();
        for (row_tiers, 0..) |rows, i| {
            var buf: [32]u8 = undefined;
            const name = try std.fmt.bufPrintSentinel(&buf, "tf_ink2_mv6_{d}", .{rows}, 0);
            out.dense[i] = try mtl.Pipeline.init(device, lib, name, false);
            made += 1;
        }
        return out;
    }

    pub fn deinit(p: *Pipelines) void {
        for (p.dense) |pipe| pipe.deinit();
    }
};

pub fn variantFor(rows: usize) usize {
    std.debug.assert(rows > 0 and rows <= 32);
    if (rows > 16) return 5;
    if (rows > 8) return 4;
    if (rows > 4) return 3;
    if (rows > 2) return 2;
    if (rows > 1) return 1;
    return 0;
}

pub fn totalSgs(nq: u32, nk: u32, nv: u32, nr: u32, ns: u32, ng: u32) usize {
    var sgs: usize = 0;
    for ([_]u32{ nq, nk, nv, nr, ns, ng }) |n| sgs += ((@as(usize, n) + 7) / 8) * 8;
    return sgs;
}

pub fn encode6(
    p: *const Pipelines,
    e: mtl.ComputeEncoder,
    xq: Ref,
    xk: Ref,
    xv: Ref,
    q: Proj,
    k_proj: Proj,
    v: Proj,
    r: Proj,
    s: Proj,
    g: Proj,
    rows: u32,
    k: u32,
) !void {
    if (rows == 0 or rows > 32 or k == 0 or k % 8 != 0) return error.InvalidInkling2DenseFuseShape;
    for ([_]u32{ q.n, k_proj.n, v.n, r.n, s.n, g.n }) |n| {
        if (n == 0) return error.InvalidInkling2DenseFuseShape;
    }
    e.setPipeline(p.dense[variantFor(rows)]);
    const inputs = [_]Ref{ xq, xk, xv, q.w, k_proj.w, v.w, r.w, s.w, g.w, q.y, k_proj.y, v.y, r.y, s.y, g.y };
    for (inputs, 0..) |buf, idx| e.setBuffer(buf.buf, buf.off, idx);
    e.setValue(InkMv6Args{
        .rows = rows,
        .k = k,
        .nq = q.n,
        .nk = k_proj.n,
        .nv = v.n,
        .nr = r.n,
        .ns = s.n,
        .ng = g.n,
    }, 15);
    const total_sgs = totalSgs(q.n, k_proj.n, v.n, r.n, s.n, g.n);
    e.dispatchGroups(mtl.Size.of((total_sgs + 7) / 8, 1, 1), mtl.Size.of(threads, 1, 1));
}
