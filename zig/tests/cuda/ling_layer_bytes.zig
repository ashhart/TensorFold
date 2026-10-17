//! `ling-layer-bytes DIR FIXTURE OUT`: one real Ling-3.1 layer's outputs and states on every path, written for cmp.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const ling = @import("ling_cuda");
const rig = @import("ling_forward_rig.zig");

const F = rig.F;
const Rig = rig.Rig;
const Opened = @typeInfo(@typeInfo(@TypeOf(rig.W.open)).@"fn".return_type.?).error_union.payload;

/// The loaded layer, its graph, a prompt and a decode Buffers, the fixture's rows tiled to `rows`.
pub const Kit = struct {
    o: Opened,
    m: F.Modules,
    w: rig.W.Weights,
    g: F.Graph,
    r: Rig,
    pb: F.Buffers,
    db: F.Buffers,
    x: []u16,
    D: usize,
    L: u32,
    cap: usize,

    /// Loads FIXTURE's layer from DIR; prompt buffers for `rows` rows, states of `cap` positions.
    pub fn init(k: *Kit, gpu: check.Gpu, a: std.mem.Allocator, s: cuda.Stream, dir: []const u8, path: []const u8, rows: usize, cap: usize) !void {
        const bytes = try std.Io.Dir.cwd().readFileAlloc(gpu.io, path, a, .limited(1 << 30));
        if (bytes.len < 24 or !std.mem.eql(u8, bytes[0..8], "TFLINGL1")) return error.BadFixture;
        const dims = std.mem.bytesAsSlice(u32, bytes[8..24]);
        k.L = dims[0];
        const T: usize = dims[1];
        k.D = dims[2];
        k.x = try a.alloc(u16, rows * k.D);
        for (0..rows) |t| @memcpy(std.mem.sliceAsBytes(k.x[t * k.D ..][0..k.D]), bytes[24 + (t % T) * k.D * 2 ..][0 .. k.D * 2]);
        var why: ling.config.Why = .{};
        k.o = rig.W.open(gpu.gpa, gpu.io, dir, &why) catch |e| {
            std.debug.print("{s}: {s}\n", .{ dir, why.text() });
            return e;
        };
        k.m = try F.Modules.load(gpu.d, gpu.ctx);
        var pack = try cuda.modelopt.Packer.load(gpu.d);
        defer pack.unload();
        k.w = try rig.W.load(gpu.gpa, gpu.d, s, &pack, &k.o.ck, k.o.c, &.{k.L}, .{ .globals = false });
        k.g = try F.Graph.init(gpu.gpa, gpu.d, &k.m, s, &k.w);
        k.r = .{ .gpu = gpu, .a = a, .s = s, .m = &k.m, .g = &k.g };
        k.cap = cap;
        k.pb = try F.Buffers.init(gpu.d, &k.m, &k.w, rows, true, cap);
        k.db = try F.Buffers.init(gpu.d, &k.m, &k.w, F.window_rows, false, cap);
    }

    pub fn deinit(k: *Kit) void {
        k.pb.deinit();
        k.db.deinit();
        k.r.deinit();
        k.g.deinit();
        k.w.deinit();
        k.m.deinit();
        k.o.ck.close();
    }

    /// Rows lo .. hi through the layer on `st`, committing `keep` of them (0: all); the output rows' bytes.
    pub fn pass(k: *Kit, b: *const F.Buffers, st: *F.State, lo: usize, hi: usize, keep: usize) ![]u8 {
        const d = k.r.gpu.d;
        const n = hi - lo;
        try d.check(d.api.cuMemcpyHtoDAsync_v2(b.x, k.x[lo * k.D ..].ptr, n * k.D * 2, k.r.s.handle), "cuMemcpyHtoDAsync");
        const f = k.r.fwd(b, null);
        try f.stack(&.{.{ .st = st, .row0 = 0, .rows = n }});
        const out = try k.r.back(b.x, n * k.D * 2);
        try f.commit(st, if (keep == 0) n else keep);
        return out;
    }
};

fn put(list: *std.ArrayList(u8), a: std.mem.Allocator, bytes: []const u8) !void {
    try list.appendSlice(a, bytes);
}

/// Every path's bytes: prompts of 512 and 2048 rows, 16 decode rows, windows of 2..16 with partial commits, states.
pub fn run(gpu: check.Gpu, dir: []const u8, path: []const u8, out: []const u8) !void {
    var arena = std.heap.ArenaAllocator.init(gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var k: Kit = undefined;
    try k.init(gpu, a, s, dir, path, 2048, 2600);
    defer k.deinit();
    var all: std.ArrayList(u8) = .empty;
    const P = 400;
    for ([_]usize{ 512, 2048 }) |T| {
        const st = try k.r.state(k.cap);
        defer st.deinit();
        try put(&all, a, try k.pass(&k.pb, st, 0, T, 0));
        try put(&all, a, try k.r.back(st.arena.buf.ptr, st.arena.buf.len));
    }
    for ([_]bool{ true, false }) |wide| {
        k.r.kda_wide = wide;
        const st = try k.r.state(k.cap);
        defer st.deinit();
        _ = try k.pass(&k.pb, st, 0, P, 0);
        for (0..F.window_rows) |i| try put(&all, a, try k.pass(&k.db, st, P + i, P + i + 1, 0));
        try put(&all, a, try k.r.back(st.arena.buf.ptr, st.arena.buf.len));
        for ([_]usize{ 2, 3, 4, 8, 16 }) |w| {
            const sw = try k.r.state(k.cap);
            defer sw.deinit();
            _ = try k.pass(&k.pb, sw, 0, P, 0);
            const keep = (w + 1) / 2;
            try put(&all, a, try k.pass(&k.db, sw, P, P + w, keep));
            try put(&all, a, try k.pass(&k.db, sw, P + keep, P + keep + w, 0));
            try put(&all, a, try k.r.back(sw.arena.buf.ptr, sw.arena.buf.len));
        }
    }
    k.r.kda_wide = true;
    var h = std.hash.Wyhash.init(0);
    h.update(all.items);
    try std.Io.Dir.cwd().writeFile(gpu.io, .{ .sub_path = out, .data = all.items });
    check.pass("ling-layer-bytes layer {d}: {d} bytes into {s}, wyhash {x:0>16}", .{ k.L, all.items.len, out, h.final() });
}
