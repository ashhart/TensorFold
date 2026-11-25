//! Prompt-only MTP pieces retain identical cache bytes while unread proposal outputs stay poisoned.
const std = @import("std");
const cuda = @import("cuda");
const flash = @import("cuda_launch.zig");
const B = flash.prompt.batch;
const mtp = @import("cuda_mtp.zig");
const pieces = @import("cuda_mtp_pieces.zig");

fn fill(ptr: u64, n: usize, value: u8) void {
    const bytes: [*]u8 = @ptrFromInt(ptr);
    @memset(bytes[0..n], value);
}

const CacheTri = struct {
    var launches: usize = 0;

    fn launch(_: cuda.abi.Function, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: cuda.abi.Stream, _: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
        launches += 1;
        return cuda.abi.success;
    }

    pub fn fusedAttentionOwned(_: @This()) bool {
        return false;
    }

    pub fn attnPrep(_: @This(), _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, q: u64, kc: u64, vc: u64, iq: u64, ikc: u64, _: anytype, _: anytype, _: anytype, n: usize, _: anytype, _: anytype, _: anytype) !void {
        fill(q, n * 4, 0x11);
        fill(iq, n * 4, 0x22);
        fill(kc, n * 8, 0x33);
        fill(vc, n * 8, 0x44);
        fill(ikc, n * 8, 0x55);
    }

    pub fn qsaPool(_: @This(), _: anytype, pooled: u64, _: anytype, _: anytype, _: anytype, _: anytype, n: usize, _: anytype) !void {
        fill(pooled, n * 8, 0x66);
    }

    pub fn qsaScores(_: @This(), _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype) !void {}
    pub fn attnChunks(_: @This(), _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype) !void {}

    pub fn attnMerge(_: @This(), _: anytype, _: anytype, _: anytype, _: anytype, ao: u64, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype) !void {
        fill(ao, 32, 0x77);
    }

    pub fn attnFused(_: @This(), _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype) !void {}
};

fn cacheBuffer(driver: *cuda.Driver, bytes: []u8) cuda.DeviceBuffer {
    return .{ .d = driver, .ptr = @intFromPtr(bytes.ptr), .len = bytes.len };
}

test "Flash Next MTP lean prefill: cache bytes match while attention output poison survives" {
    const P = B.Pass(CacheTri);
    const LayerPtr = @typeInfo(@TypeOf(P.attnBlock)).@"fn".param_types[1].?;
    const Layer = @typeInfo(LayerPtr).pointer.child;
    var driver: cuda.Driver = undefined;
    driver.api.cuLaunchKernel = CacheTri.launch;
    var k: B.Kernels = undefined;
    k.select = .{ .d = &driver, .handle = @ptrFromInt(1) };
    var s: B.Scratch = undefined;
    s.inv = 0;
    s.dummy = 0;
    s.qids = 0;
    s.qnk = 0;
    s.qspr = 0;
    s.po = 0;
    s.pm = 0;
    s.pl = 0;
    var lw: Layer = undefined;
    lw.heads = .full;
    lw.q_scale.ptr = 0;
    lw.k_scale.ptr = 0;
    lw.i_scale.ptr = 0;
    lw.ik_scale.ptr = 0;

    var full_kc: [32]u8 = @splat(0xa5);
    var full_vc: [32]u8 = @splat(0xa5);
    var full_ikc: [32]u8 = @splat(0xa5);
    var full_pooled: [32]u8 = @splat(0xa5);
    var lean_kc: [32]u8 = @splat(0xa5);
    var lean_vc: [32]u8 = @splat(0xa5);
    var lean_ikc: [32]u8 = @splat(0xa5);
    var lean_pooled: [32]u8 = @splat(0xa5);
    var full_kcb = [_]cuda.DeviceBuffer{cacheBuffer(&driver, &full_kc)};
    var full_vcb = [_]cuda.DeviceBuffer{cacheBuffer(&driver, &full_vc)};
    var full_ikcb = [_]cuda.DeviceBuffer{cacheBuffer(&driver, &full_ikc)};
    var full_pooledb = [_]cuda.DeviceBuffer{cacheBuffer(&driver, &full_pooled)};
    var lean_kcb = [_]cuda.DeviceBuffer{cacheBuffer(&driver, &lean_kc)};
    var lean_vcb = [_]cuda.DeviceBuffer{cacheBuffer(&driver, &lean_vc)};
    var lean_ikcb = [_]cuda.DeviceBuffer{cacheBuffer(&driver, &lean_ikc)};
    var lean_pooledb = [_]cuda.DeviceBuffer{cacheBuffer(&driver, &lean_pooled)};
    var full_q: [32]u8 = @splat(0xa5);
    var full_iq: [32]u8 = @splat(0xa5);
    var full_ao: [32]u8 = @splat(0xa5);
    var lean_q: [32]u8 = @splat(0xa5);
    var lean_iq: [32]u8 = @splat(0xa5);
    var lean_ao: [32]u8 = @splat(0xa5);
    const poison: [32]u8 = @splat(0xa5);

    var p: P = undefined;
    p.tri = .{};
    p.s = &s;
    p.k = &k;
    p.stream = .{ .d = &driver, .handle = null };
    p.pos0 = 0;
    p.seq.cap = 4;
    p.seq.media = null;
    p.seq.sel = null;
    p.seq.pg = .{};
    p.seq.scores = 0;
    p.seq.kc = &full_kcb;
    p.seq.vc = &full_vcb;
    p.seq.ikc = &full_ikcb;
    p.seq.pooled = &full_pooledb;
    CacheTri.launches = 0;
    try p.attnRows(&lw, 0, 0, 4, .{ .pos = 0, .q = @intFromPtr(&full_q), .iq = @intFromPtr(&full_iq), .ao = @intFromPtr(&full_ao) });
    try std.testing.expectEqual(@as(usize, 1), CacheTri.launches);
    try std.testing.expectEqualSlices(u8, &.{0x77}, full_ao[0..1]);

    p.seq.kc = &lean_kcb;
    p.seq.vc = &lean_vcb;
    p.seq.ikc = &lean_ikcb;
    p.seq.pooled = &lean_pooledb;
    CacheTri.launches = 0;
    try p.attnRowsCache(&lw, 0, 0, 4, .{ .pos = 0, .q = @intFromPtr(&lean_q), .iq = @intFromPtr(&lean_iq), .ao = @intFromPtr(&lean_ao) });
    try std.testing.expectEqual(@as(usize, 0), CacheTri.launches);
    try std.testing.expectEqualSlices(u8, &full_kc, &lean_kc);
    try std.testing.expectEqualSlices(u8, &full_vc, &lean_vc);
    try std.testing.expectEqualSlices(u8, &full_ikc, &lean_ikc);
    try std.testing.expectEqualSlices(u8, &full_pooled, &lean_pooled);
    try std.testing.expectEqualSlices(u8, &full_q, &lean_q);
    try std.testing.expectEqualSlices(u8, &full_iq, &lean_iq);
    try std.testing.expectEqualSlices(u8, &poison, &lean_ao);
}

var retained: [64]u8 = undefined;
var proposal_attentions: usize = 0;

fn retainCache() void {
    for (&retained, 0..) |*v, i| v.* = @truncate(i * 37 + 11);
}

const PieceScratch = struct {
    ids: u64 = 0,
    ple_keys: u64 = 0,
    mixed: u64 = 0,
    xsm: u64 = 0,
    dummy: u64 = 0,
    part: u64 = 0,
    branch: u64 = 0,
    norm: u64 = 0,
    gq: u64 = 0,
    ple_gated: u64 = 0,
    h: u64 = 0,
    pss: u64 = 0,
    inj: u64 = 0,
    native_part: u64 = 0,
};

const PieceKernels = struct { native_lane: cuda.qmmf.Lane };

const PiecePass = struct {
    s: *PieceScratch,
    k: *const PieceKernels,
    out: u64,
    dlogits: u64,
    stream: cuda.Stream,
    pending: ?usize = null,
    rows: usize = 0,
    out_slices: usize = 0,

    pub fn lane(self: *@This(), _: anytype, _: anytype, _: anytype, out: u64, _: anytype, _: anytype, sk: usize) !usize {
        if (out == self.dlogits) fill(out, 32, 0x88);
        return sk;
    }
    pub fn sumSlices(_: *@This(), _: anytype, _: anytype, _: anytype, _: anytype, _: anytype) !void {}
    pub fn writeback(_: *@This(), _: anytype, _: anytype) !void {}
    pub fn readout(_: *@This(), _: anytype, _: anytype, _: anytype) !void {}
    pub fn attnBlock(_: *@This(), _: anytype, _: anytype) !void {
        retainCache();
        proposal_attentions += 1;
    }
    pub fn attnBlockCache(_: *@This(), _: anytype, _: anytype) !void {
        retainCache();
    }
    pub fn copy(self: *@This(), dst: u64, _: anytype, _: anytype) !void {
        if (dst == self.out) fill(dst, 32, 0x99);
    }
    pub fn moeBlock(_: *@This(), _: anytype) !void {}
};

const PieceTri = struct {
    pub fn embedDense(_: @This(), _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype) !void {}
    pub fn embed(_: @This(), _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype) !void {}
    pub fn rmsnorm(_: @This(), _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype) !void {}
    pub fn addStreams(_: @This(), _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype) !void {}
    pub fn hcWriteback(_: @This(), _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype) !void {}
};

const PieceEngine = struct {
    table: struct { dense: bool = false, w: struct { ptr: u64 = 0 } = .{}, s: struct { ptr: u64 = 0 } = .{}, b: struct { ptr: u64 = 0 } = .{} },
    stream: *const cuda.Stream,
};

const PieceOwner = struct {
    e: *PieceEngine,
    tri: PieceTri,
    h: *mtp.Head,
};

test "Flash Next MTP lean prefill: d.out and dlogits poison survives with identical retained bytes" {
    var driver: cuda.Driver = undefined;
    const stream: cuda.Stream = .{ .d = &driver, .handle = null };
    var engine: PieceEngine = .{ .table = .{}, .stream = &stream };
    var head: mtp.Head = undefined;
    head.draft.native = null;
    const owner: PieceOwner = .{ .e = &engine, .tri = .{}, .h = &head };
    var scratch: PieceScratch = .{};
    var kernels: PieceKernels = undefined;
    var out: [32]u8 = @splat(0xa5);
    var dlogits: [32]u8 = @splat(0xa5);
    var draft: mtp.Draft = undefined;
    draft.media = null;
    draft.len = 0;
    draft.out = cacheBuffer(&driver, &out);
    draft.dlogits = cacheBuffer(&driver, &dlogits);
    var pass: PiecePass = .{ .s = &scratch, .k = &kernels, .stream = stream, .out = @intFromPtr(&out), .dlogits = @intFromPtr(&dlogits) };
    const Poison = [32]u8;
    const poison: Poison = @splat(0xa5);

    const full: pieces.Piece(PieceOwner, PiecePass) = .{ .o = owner, .p = &pass, .d = &draft, .input = 0, .m = 1, .last = true, .score = true };
    proposal_attentions = 0;
    try full.run();
    try std.testing.expectEqual(@as(usize, 1), proposal_attentions);
    try std.testing.expectEqualSlices(u8, &.{0x99}, out[0..1]);
    try std.testing.expectEqualSlices(u8, &.{0x88}, dlogits[0..1]);
    const expected = retained;

    @memset(&out, 0xa5);
    @memset(&dlogits, 0xa5);
    @memset(&retained, 0xa5);
    proposal_attentions = 0;
    pass.pending = null;
    const lean: pieces.Piece(PieceOwner, PiecePass) = .{ .o = owner, .p = &pass, .d = &draft, .input = 0, .m = 1, .last = true, .score = false, .kv_only = true };
    try lean.run();
    try std.testing.expectEqual(@as(usize, 0), proposal_attentions);
    try std.testing.expectEqualSlices(u8, &expected, &retained);
    try std.testing.expectEqualSlices(u8, &poison, &out);
    try std.testing.expectEqualSlices(u8, &poison, &dlogits);
}
