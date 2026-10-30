//! The grouped prompt emitter forwards an admitted page table through each actual attention dispatch.
const std = @import("std");
const cuda = @import("cuda");
const flash = @import("cuda_launch.zig");
const B = flash.prompt.batch;
const C = struct {
    var expected_pg: flash.Pg = .{};
    var owned: flash.Glue = undefined;
    var captured: cuda.aot.Set = undefined;
    var mode: u2 = 0;
    var bits: u32 = 0;
    var chunks: usize = 0;
    var merge: usize = 0;
    var fused: usize = 0;
    fn tri() flash.Tri {
        owned.mixed = bits;
        return .{ .set = if (mode == 0) null else &captured, .own = if (mode == 1) null else &owned, .s = undefined };
    }
    var launches: usize = 0;
    var kept: usize = 0;
    var plain: usize = 0;
    var rinv: usize = 0;
    var mix: usize = 0;
    var up: u64 = 0;
    fn reset() void {
        chunks = 0;
        merge = 0;
        fused = 0;
        launches = 0;
        kept = 0;
        plain = 0;
        rinv = 0;
        mix = 0;
        up = 0;
    }
    fn launch(_: cuda.abi.Function, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: cuda.abi.Stream, _: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
        launches += 1;
        return cuda.abi.success;
    }
};
const Fake = struct {
    pub fn readoutTilesOwned(_: @This()) bool {
        return C.tri().readoutTilesOwned();
    }
    pub fn fusedAttentionOwned(_: @This()) bool {
        return C.tri().fusedAttentionOwned();
    }
    pub fn attnPrep(_: @This(), _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, pg: flash.Pg) !void {
        try std.testing.expectEqual(C.expected_pg, pg);
    }
    pub fn qsaPool(_: @This(), _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, pg: flash.Pg) !void {
        try std.testing.expectEqual(C.expected_pg, pg);
    }
    pub fn qsaScores(_: @This(), _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, pg: flash.Pg) !void {
        try std.testing.expectEqual(C.expected_pg, pg);
    }
    pub fn attnGate(_: @This(), _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype) !void {}
    pub fn attnChunks(_: @This(), _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, pg: flash.Pg) !void {
        try std.testing.expectEqual(C.expected_pg, pg);
        C.chunks += 1;
    }
    pub fn attnMerge(_: @This(), _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype) !void {
        C.merge += 1;
    }
    pub fn attnFused(_: @This(), _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, pg: flash.Pg) !void {
        try std.testing.expectEqual(C.expected_pg, pg);
        C.fused += 1;
    }
    pub fn hcNormalize(_: @This(), _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype) !void {}
    pub fn hcDenseMix(_: @This(), _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype) !void {}
    pub fn hcDown(_: @This(), _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype) !void {}
    pub fn hcReduceAct(_: @This(), _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype) !void {}
    pub fn hcUpmixUp(_: @This(), _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, up: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype) !void {
        C.kept += 1;
        C.up = up;
    }
    pub fn hcUpmix(_: @This(), _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype) !void {
        C.plain += 1;
    }
    pub fn hcRinv(_: @This(), _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype) !void {
        C.rinv += 1;
    }
    pub fn hcMixH(_: @This(), _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype) !void {
        C.mix += 1;
    }
};
fn fun(d: *const cuda.Driver, id: usize) cuda.Function {
    return .{ .d = d, .handle = @ptrFromInt(id) };
}
test "paged grouped prompts: every current attention route receives the sequence page table" {
    const P = B.Pass(Fake);
    const LayerPtr = @typeInfo(@TypeOf(P.attnBlock)).@"fn".param_types[1].?;
    const Layer = @typeInfo(LayerPtr).pointer.child;
    var driver: cuda.Driver = undefined;
    driver.api.cuLaunchKernel = C.launch;
    var k: B.Kernels = undefined;
    k.tiles.fact = fun(&driver, 1);
    k.tiles.fact_fold = fun(&driver, 2);
    k.lane = fun(&driver, 3);
    k.select = fun(&driver, 4);
    var s: B.Scratch = undefined;
    s.mixed = 0x100000000;
    s.xsm = 0x200000000;
    s.proj = 0x300000000;
    s.dummy = 0x400000000;
    s.native_part = 0;
    s.cap = 4096;
    s.pos = 0x500000000;
    s.q = 0x600000000;
    s.iq = 0x700000000;
    s.ao = 0x800000000;
    s.inv = 0x900000000;
    s.qids = 0xa00000000;
    s.qnk = 0xb00000000;
    s.qspr = 0xc00000000;
    s.po = 0xd00000000;
    s.pm = 0xe00000000;
    s.pl = 0xf00000000;
    s.gated = 0x1100000000;
    s.gxs_attn = 0x1200000000;
    s.part = 0x1300000000;
    var lw: Layer = undefined;
    lw.proj.native = null;
    lw.proj.w.ptr = 0x1000;
    lw.proj.s.ptr = 0x2000;
    lw.proj.b.ptr = 0x3000;
    lw.proj.k = 7680;
    lw.proj.n = 13952;
    lw.out.native = null;
    lw.out.w.ptr = 0x4000;
    lw.out.s.ptr = 0x5000;
    lw.out.b.ptr = 0x6000;
    lw.out.k = 6144;
    lw.out.n = 7680;
    lw.q_scale.ptr = 0x7000;
    lw.k_scale.ptr = 0x8000;
    lw.i_scale.ptr = 0x9000;
    lw.ik_scale.ptr = 0xa000;
    var buffers: [1]cuda.DeviceBuffer = undefined;
    buffers[0].ptr = 0xb000;
    var p: P = undefined;
    p.tri = .{};
    p.s = &s;
    p.k = &k;
    p.stream = .{ .d = &driver, .handle = null };
    p.lanes = &.{};
    p.pos0 = 0;
    p.seq.cap = 4096;
    p.seq.media = null;
    p.seq.pg = .{};
    p.seq.kc = &buffers;
    p.seq.vc = &buffers;
    p.seq.ikc = &buffers;
    p.seq.pooled = &buffers;
    p.seq.scores = 0xc000;
    const modes = [_]flash.Pg{ .{}, .{ .pt = 0x50000, .ps = 12 * 48 * 1184, .ptok = 48 } };
    for (modes) |pg| for ([_]usize{ 1, 16, 17, 256, 257, 1024, 1025, 4096 }) |rows| {
        C.mode = 0;
        C.bits = 0;
        C.expected_pg = pg;
        C.reset();
        p.rows = rows;
        p.seq.pg = pg;
        try p.attnBlock(&lw, 0);
        const fused = rows > 256;
        try std.testing.expectEqual(if (fused) (rows + 1023) / 1024 else @as(usize, 0), C.fused);
        const old = if (fused) @as(usize, 0) else (rows + 255) / 256;
        try std.testing.expectEqual(old, C.chunks);
        try std.testing.expectEqual(old, C.merge);
    };
}
