//! TF_Q35_GROUP_PROJ's grouped lane launch: each projection's bytes against its own launch, rows 1-16, then times.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const rig = @import("flashnext_rig.zig");

const Mode = cuda.qmmf.Mode;
const group = cuda.qmmf_group;

const Set = struct { name: []const u8, mode: Mode, k: usize, ns: []const usize };

/// The served Qwen3.8-27B ModelOpt groups (NVFP4 MLP, FP8 mixers, BF16 a/b heads), then padding, edges and four.
const sets = [_]Set{
    .{ .name = "MLP gate+up", .mode = .fp4, .k = 5120, .ns = &.{ 17408, 17408 } },
    .{ .name = "GDN qkv+z", .mode = .fp8, .k = 5120, .ns = &.{ 10240, 6144 } },
    .{ .name = "GDN a+b", .mode = .bf16, .k = 5120, .ns = &.{ 48, 48 } },
    .{ .name = "attention q+k+v", .mode = .fp8, .k = 5120, .ns = &.{ 12288, 1024, 1024 } },
    .{ .name = "fp4 one slice padded to the cluster", .mode = .fp4, .k = 4096, .ns = &.{ 12352, 200, 72 } },
    .{ .name = "bf16 64-column tiles", .mode = .bf16, .k = 1024, .ns = &.{ 12288, 6400 } },
    .{ .name = "fp8 four, an odd width", .mode = .fp8, .k = 2048, .ns = &.{ 64, 2048, 131, 4096 } },
};

/// Tensor scales: one, small, subnormal and infinite.
const scales = [_]f32{ 1.0, 0x1p-12, 1e-40, std.math.inf(f32) };

/// bf16 specials an adversarial row mixes in: zeros, subnormals, the largest finite and infinities.
const specials = [_]u16{ 0x0000, 0x8000, 0x0001, 0x8001, 0x7F7F, 0xFF7F, 0x7F80, 0xFF80 };

/// One projection packed for the lane: random bytes, or bounded values without NaN codes.
fn build(r: *rig.Rig, a: std.mem.Allocator, rng: std.Random, mode: Mode, n: usize, k: usize, adversarial: bool) !cuda.qmmf.Weight {
    const np = cuda.modelopt.padded(n);
    switch (mode) {
        .fp4 => {
            const codes = try a.alloc(u8, n * k / 2);
            rng.bytes(codes);
            const blocks = try a.alloc(u8, n * k / 16);
            for (blocks) |*s| s.* = if (adversarial) rng.int(u8) else 0x30 + rng.uintLessThan(u8, 16);
            const words = try a.alloc(u32, np / 64 * (k / 64) * 512);
            cuda.nvfp4.packWords(words, codes, n, k);
            const tiles = try a.alloc(u8, np * (k / 16));
            cuda.nvfp4.packScales(tiles, blocks, n, k);
            return cuda.nvfp4.weight(try r.dev(std.mem.sliceAsBytes(words)), try r.dev(tiles), 1.0, n, k);
        },
        .fp8 => {
            const codes = try a.alloc(u8, n * k);
            for (codes) |*c| c.* = if (adversarial) rng.int(u8) else rng.int(u8) & 0xF7; // no e4m3 NaN
            const tiled = try a.alloc(u8, np * k);
            cuda.fp8.packCodes(tiled, codes, n, k);
            return cuda.modelopt.fp8Weight(try r.dev(tiled), 1.0, n, k);
        },
        .bf16 => {
            const w: []const u16 = if (adversarial) std.mem.bytesAsSlice(u16, std.mem.sliceAsBytes(r.words(n * k / 2))) else r.bfs(n * k, 0.05);
            const tiled = try a.alloc(u16, np * k);
            cuda.qmmf.packBf16(tiled, w, n, k);
            return cuda.modelopt.bf16Weight(try r.dev(std.mem.sliceAsBytes(tiled)), n, k);
        },
        else => unreachable,
    }
}

fn refill(r: *rig.Rig, ptr: u64, bytes: usize) !void {
    const b: cuda.DeviceBuffer = .{ .d = r.gpu.d, .ptr = ptr, .len = bytes };
    try b.fill8(0xff, r.s.handle);
}

pub fn run(gpu: check.Gpu) !void {
    var arena = std.heap.ArenaAllocator.init(gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var prng = std.Random.DefaultPrng.init(0x6a1d);
    const rng = prng.random();
    var r = try rig.Rig.init(gpu, a, rng);
    defer r.deinit();
    var lane = try cuda.qmmf.Lane.load(gpu.d, @intCast(try gpu.ctx.capability() / 10));
    defer lane.unload();
    if (!lane.clusters) {
        std.debug.print("SKIP group-proj: no clusters before sm_90, where the engine never groups\n", .{});
        return;
    }
    const g = try group.Group.load(&lane);
    var cells: usize = 0;
    for (sets) |set| for ([_]bool{ false, true }) |adversarial| {
        var ws: [group.most]cuda.qmmf.Weight = undefined;
        var sep: [group.most]u64 = undefined;
        var grp: [group.most]u64 = undefined;
        var bytes: [group.most]usize = undefined;
        var part_bytes: usize = 16;
        const count = set.ns.len;
        for (set.ns, 0..) |n, i| {
            ws[i] = try build(&r, a, rng, set.mode, n, set.k, adversarial);
            bytes[i] = (group.max_rows + 1) * n * 2; // the 16 rows and a guard row nothing may write
            sep[i] = try r.out(bytes[i]);
            grp[i] = try r.out(bytes[i]);
            part_bytes = @max(part_bytes, lane.partBytes(group.max_rows, ws[i]));
        }
        const part = try r.out(part_bytes);
        const xs = if (adversarial) r.bfs(group.max_rows * set.k, 4.0) else r.bfs(group.max_rows * set.k, 1.0);
        if (adversarial) for (xs) |*v| {
            if (rng.uintLessThan(u8, 8) == 0) v.* = specials[rng.uintLessThan(usize, specials.len)];
        };
        const x = try r.dev(std.mem.sliceAsBytes(xs));
        const tensor_scales: []const f32 = if (set.mode == .bf16) scales[0..1] else &scales;
        for (tensor_scales) |scale| for (1..group.max_rows + 1) |m| {
            for (ws[0..count]) |*w| w.scale = scale;
            for (0..count) |i| {
                try refill(&r, sep[i], bytes[i]);
                try refill(&r, grp[i], bytes[i]);
            }
            const sh = group.plan(lane, ws[0..count], grp[0..count], m) orelse
                return check.expect(false, "group-proj {s}: rows {d} not planned", .{ set.name, m });
            for (ws[0..count], sep[0..count]) |w, y| try lane.matmul(r.s, x, set.k, m, w, y, part);
            try g.launch(r.s, x, m, set.k, sh, false);
            try r.s.synchronize();
            for (0..count) |i| {
                const want = try r.back(u8, sep[i], bytes[i]);
                const got = try r.back(u8, grp[i], bytes[i]);
                try check.sameBytes(set.name, got, want);
                cells += 1;
            }
        };
        if (!adversarial) try time(&r, gpu, lane, g, set, ws[0..count], grp[0..count], x, part);
    };
    check.pass("group-proj: {d} projections at rows 1-16, tensor scales and adversarial values: bytes equal", .{cells});
}

/// The grouped launch against the separate ones, warm, at 1 and 16 rows (this engine's lanes only).
fn time(r: *rig.Rig, gpu: check.Gpu, lane: cuda.qmmf.Lane, g: group.Group, set: Set, ws: []const cuda.qmmf.Weight, outs: []const u64, x: u64, part: u64) !void {
    for ([_]usize{ 1, 16 }) |m| {
        const sh = group.plan(lane, ws, outs, m).?;
        var us: [2]f64 = undefined;
        const reps = 200;
        for (0..2) |grouped| {
            const t0 = check.now(gpu.io);
            for (0..reps) |_| {
                if (grouped == 1) try g.launch(r.s, x, m, set.k, sh, false) else for (ws, outs) |w, y| try lane.matmul(r.s, x, set.k, m, w, y, part);
            }
            try r.s.synchronize();
            us[grouped] = @as(f64, @floatFromInt(check.now(gpu.io) - t0)) / 1e3 / reps;
        }
        std.debug.print("group-proj {s} rows {d}: separate {d:.1} us, grouped {d:.1} us ({d} blocks, cluster {d})\n", .{ set.name, m, us[0], us[1], sh.blocks, sh.cluster });
    }
}
