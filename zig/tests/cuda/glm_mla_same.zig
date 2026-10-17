//! GLM's MLA launches on this build's glm_glue against a 6380c8da glm_glue fatbin: the same bytes for GLM's shapes.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const qr = @import("quant_ref_tests.zig");
const glm = @import("glm_cuda");
const Gpu = check.Gpu;
const Rig = qr.RigT;
const latent = glm.latent;
const Set = glm.glue.Set;

const heads = 32;
const lw = 512;
const d = 256;

/// The kernels Ling added, absent from the old image: taken from the new one (GLM's launches never reach them).
const added = [_][]const u8{ "absorb_q_128", "expand_v_128", "rope", "head_gate", "chunks_r16" };

fn oldKernels(old: cuda.Module, new: glm.glue.Kernels) !glm.glue.Kernels {
    var k = new;
    inline for (@typeInfo(glm.glue.Kernels).@"struct".field_names) |name| {
        const fresh = for (added) |a| {
            if (std.mem.eql(u8, a, name)) break true;
        } else false;
        if (!fresh) @field(k, name) = try old.function("tf_glm_" ++ name);
    }
    try k.chunks_16.allowDynamicShared(glm.glue.chunksSmem(16));
    try k.chunks_32.allowDynamicShared(glm.glue.chunksSmem(32));
    return k;
}

fn partials(r: *Rig, nch: usize, rows: usize) !latent.Partials {
    return .{ .po = try r.zeros(nch * rows * heads * lw * 4), .pm = try r.zeros(nch * rows * heads * 4), .pl = try r.zeros(nch * rows * heads * 4) };
}

const Inputs = struct { q: u64, wk: u64, ol: u64, wv: u64, cache: u64, lat: u64, tok: u64, cnt: u64 };

/// Every GLM MLA launch (absorb, expand, cache write, dense 16/32-head and sparse attention) on `set`: output bytes.
fn outputs(r: *Rig, set: Set, in: Inputs) ![]const []const u8 {
    var list: std.ArrayList([]const u8) = .empty;
    for ([_]usize{ 1, 5, 70 }) |rows| {
        const qa = try r.zeros(rows * heads * lw * 2);
        try latent.absorbQ(set, r.s, in.q, in.wk, qa, rows, heads, d, lw);
        try list.append(r.a, try r.back(qa, rows * heads * lw * 2));
        const vn = try r.zeros(rows * heads * d * 2);
        try latent.expandV(set, r.s, in.ol, in.wv, vn, rows, heads, d, lw);
        try list.append(r.a, try r.back(vn, rows * heads * d * 2));
    }
    const c = try r.zeros(1500 * lw * 2);
    const p0: i32 = 1400;
    try latent.latentWrite(set, r.s, in.lat, lw + 64, c, try r.dev(std.mem.asBytes(&p0)), 100, lw);
    try list.append(r.a, try r.back(c, 1500 * lw * 2));
    for ([_][3]usize{ .{ 1430, 70, 32 }, .{ 1430, 70, 16 }, .{ 1499, 1, 16 }, .{ 0, 64, 32 } }) |case| {
        const pos, const rows, const hb = case;
        const nch = latent.chunksFor(pos + rows);
        const out = try r.zeros(rows * heads * lw * 2);
        const at: i32 = @intCast(pos);
        try latent.attention(set, r.s, in.q, in.cache, try r.dev(std.mem.asBytes(&at)), try partials(r, nch, rows), out, try r.zeros(4), rows, heads, lw, nch, 0.125, hb);
        try list.append(r.a, try r.back(out, rows * heads * lw * 2));
    }
    for ([_]usize{ 70, 1 }) |rows| {
        const out = try r.zeros(rows * heads * lw * 2);
        try latent.sparseAttention(set, r.s, in.q, in.cache, in.tok, in.cnt, 1100, try partials(r, latent.chunksFor(1100), rows), out, rows, heads, lw, 0.125);
        try list.append(r.a, try r.back(out, rows * heads * lw * 2));
    }
    return list.items;
}

/// `glm-mla-same <old glm_glue.fatbin>`: the old and new images' GLM MLA outputs, byte for byte, on seeded inputs.
pub fn run(gpu: Gpu, old_path: []const u8) !void {
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var prng = std.Random.DefaultPrng.init(804);
    var r: Rig = .{ .gpu = gpu, .a = a, .s = s, .rng = prng.random() };
    defer {
        for (r.bufs.items) |*b| b.free();
        r.bufs.deinit(gpu.gpa);
    }
    const raw = try std.Io.Dir.cwd().readFileAlloc(gpu.io, old_path, a, .limited(64 << 20));
    const image = try a.alignedAlloc(u8, .@"16", raw.len);
    @memcpy(image, raw);
    var new_mod = try cuda.Module.load(gpu.d, cuda.kernels.glm_glue);
    defer new_mod.unload();
    var old_mod = try cuda.Module.load(gpu.d, image);
    defer old_mod.unload();
    const new = try glm.glue.Kernels.load(new_mod);
    const old = try oldKernels(old_mod, new);
    const tok = try a.alloc(i32, 70 * 1100);
    const cnt = try a.alloc(i32, 70);
    for (cnt, 0..) |*n, row| {
        n.* = if (row == 5) 0 else @intCast(1 + r.rng.uintLessThan(usize, 1100));
        for (tok[row * 1100 ..][0..1100], 0..) |*t, i| t.* = if (i < n.*) @intCast(r.rng.uintLessThan(u32, 6000)) else -1;
    }
    const in: Inputs = .{
        .q = try r.dev(std.mem.sliceAsBytes(try r.bfs(70 * heads * lw, 1.0))),
        .wk = try r.dev(std.mem.sliceAsBytes(try r.bfs(heads * d * lw, 0.05))),
        .ol = try r.dev(std.mem.sliceAsBytes(try r.bfs(70 * heads * lw, 1.0))),
        .wv = try r.dev(std.mem.sliceAsBytes(try r.bfs(heads * d * lw, 0.05))),
        .cache = try r.dev(std.mem.sliceAsBytes(try r.bfs(6000 * lw, 1.0))),
        .lat = try r.dev(std.mem.sliceAsBytes(try r.bfs(100 * (lw + 64), 1.0))),
        .tok = try r.dev(std.mem.sliceAsBytes(tok)),
        .cnt = try r.dev(std.mem.sliceAsBytes(cnt)),
    };
    const want = try outputs(&r, .{ .glue = &old }, in);
    const got = try outputs(&r, .{ .glue = &new }, in);
    const names = [_][]const u8{ "absorb_q 1 row", "expand_v 1 row", "absorb_q 5 rows", "expand_v 5 rows", "absorb_q 70 rows", "expand_v 70 rows", "lat_write 100 rows", "dense 70 rows at 1430, 32-head", "dense 70 rows at 1430, 16-head", "dense 1 row at 1499", "dense 64 rows at 0, 32-head", "sparse 70 rows", "sparse 1 row" };
    var bytes: usize = 0;
    for (names, want, got) |name, w, g| {
        try check.sameBytes(name, g, w);
        bytes += g.len;
    }
    check.pass("glm-mla-same: {d} GLM MLA launches, {d} output bytes equal 6380c8da's", .{ names.len, bytes });
}
