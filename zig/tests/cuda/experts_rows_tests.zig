//! experts_rows.cu against experts.cu's decode kernel: every pair row of a window or chunk, byte for byte.

const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const Gpu = check.Gpu;

const grouped = cuda.grouped;
const Rows = cuda.experts.RowsExperts;

/// experts_pack.cu's groups of 32 (one block: 160 uint32 words of codes, scales and biases).
const pack_symbol = "_ZN15tf_experts_pack11pack_kernelILi1EEEvPKjPKtS4_Pjiiiiii";
const block_bytes = 160 * 4;
const dims = 256;
const width = 128;
const routed = 16;
const experts = routed + 1;
const slots = 4;

const Rig = struct {
    gpu: Gpu,
    s: cuda.Stream,
    rng: std.Random,
    bufs: std.ArrayList(cuda.DeviceBuffer) = .empty,

    fn dev(r: *Rig, bytes: []const u8) !u64 {
        const b = try cuda.DeviceBuffer.fromHost(r.gpu.d, bytes);
        try r.bufs.append(r.gpu.gpa, b);
        return b.ptr;
    }

    fn zeros(r: *Rig, n: usize) !u64 {
        var b = try cuda.DeviceBuffer.alloc(r.gpu.d, @max(n, 16));
        try b.fill8(0, null);
        try r.bufs.append(r.gpu.gpa, b);
        return b.ptr;
    }

    fn get(r: *Rig, ptr: u64, n: usize) ![]u8 {
        try r.s.synchronize();
        const out = try r.gpu.gpa.alloc(u8, n);
        try r.gpu.d.check(r.gpu.d.api.cuMemcpyDtoH_v2(out.ptr, ptr, n), "cuMemcpyDtoH");
        return out;
    }

    fn bf16(v: f32) u16 {
        return @truncate(@as(u32, @bitCast(v)) >> 16);
    }

    /// Random MLX 4-bit words with small scales and biases, packed for `count` experts of n outputs from k inputs.
    fn table(r: *Rig, pack: cuda.Function, count: usize, n: usize, k: usize) !u64 {
        const a = r.gpu.gpa;
        const words = try a.alloc(u32, count * n * k / 8);
        defer a.free(words);
        for (words) |*w| w.* = r.rng.int(u32);
        const groups = count * n * k / 32;
        const scales = try a.alloc(u16, groups);
        defer a.free(scales);
        const biases = try a.alloc(u16, groups);
        defer a.free(biases);
        for (scales, biases) |*s, *b| {
            s.* = bf16(0.002 + 0.004 * r.rng.float(f32));
            b.* = bf16(-0.015 + 0.01 * r.rng.float(f32));
        }
        const out = try r.zeros(count * (n / 32) * (k / 32) * block_bytes);
        var args: cuda.Args = .{};
        for ([_]u64{ try r.dev(std.mem.sliceAsBytes(words)), try r.dev(std.mem.sliceAsBytes(scales)), try r.dev(std.mem.sliceAsBytes(biases)), out }) |v| args.add(v);
        for ([_]usize{ n, k / 8, k / 32, n / 32, 1, 0 }) |v| args.add(@as(c_int, @intCast(v))); // one table, the 0th
        try cuda.launch.launch(pack, .{ .grid = .{ .x = @intCast(k / 32), .y = @intCast(n / 32), .z = @intCast(count) }, .block = .{ .x = 160 } }, r.s, &args);
        return out;
    }

    /// Gate and up blocks side by side, the SwiGLU table the kernels read.
    fn stack(r: *Rig, gate: u64, up: u64, blocks: usize) !u64 {
        const g = try r.get(gate, blocks * block_bytes);
        defer r.gpu.gpa.free(g);
        const u = try r.get(up, blocks * block_bytes);
        defer r.gpu.gpa.free(u);
        const both = try r.gpu.gpa.alloc(u8, 2 * blocks * block_bytes);
        defer r.gpu.gpa.free(both);
        for (0..blocks) |b| {
            @memcpy(both[2 * b * block_bytes ..][0..block_bytes], g[b * block_bytes ..][0..block_bytes]);
            @memcpy(both[(2 * b + 1) * block_bytes ..][0..block_bytes], u[b * block_bytes ..][0..block_bytes]);
        }
        return r.dev(both);
    }
};

/// The plan scratch for `pairs` pairs at `tile` pairs an item.
fn planFor(r: *Rig, pairs: usize, tile: usize) !grouped.Plan {
    const wide = pairs > grouped.small;
    return .{
        .members = try r.zeros(pairs * 4),
        .items = try r.zeros(grouped.maxItems(pairs, experts, tile) * 12),
        .counts = try r.zeros(8),
        .rank = try r.zeros((if (wide) pairs else 1) * 4),
        .hist = try r.zeros((if (wide) (pairs + 1023) / 1024 * experts else 1) * 4),
    };
}

fn launch(r: *Rig, f: cuda.Function, x: u64, x_stride: usize, slot_n: usize, w: u64, kg: usize, nb: usize, p: grouped.Plan, out: u64, n: usize, units: usize) !void {
    var a = cuda.experts.Affine4Experts.argsOf(x, x_stride, slot_n, w, kg, nb, p, out, n);
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast((units + 3) / 4) }, .block = .{ .x = 128 } }, r.s, &a);
}

pub fn run(gpu: Gpu) !void {
    const a = gpu.gpa;
    var experts_mod = try cuda.Module.load(gpu.d, cuda.kernels.experts);
    defer experts_mod.unload();
    var rows_mod = try cuda.Module.load(gpu.d, cuda.kernels.experts_rows);
    defer rows_mod.unload();
    var pack_mod = try cuda.Module.load(gpu.d, cuda.kernels.experts_pack);
    defer pack_mod.unload();
    const router = try grouped.Router.resolve(experts_mod);
    const kernels = try Rows.resolve(rows_mod);
    const one_up = try experts_mod.function(Rows.symbols.decode_up);
    const one_down = try experts_mod.function(Rows.symbols.decode_down);
    var prng = std.Random.DefaultPrng.init(7);
    var r: Rig = .{ .gpu = gpu, .s = try cuda.Stream.init(gpu.d, true), .rng = prng.random() };
    defer r.s.deinit();
    defer {
        for (r.bufs.items) |*b| b.free();
        r.bufs.deinit(a);
    }
    const pack = try pack_mod.function(pack_symbol);
    const up_w = try r.stack(try r.table(pack, experts, width, dims), try r.table(pack, experts, width, dims), experts * (width / 32) * (dims / 32));
    const down_w = try r.table(pack, experts, dims, width);
    for ([_]usize{ 1, 2, 5, 16, 33, 100, 300 }) |n_rows| {
        const pairs = n_rows * slots;
        const x = try a.alloc(u16, n_rows * dims);
        defer a.free(x);
        for (x) |*v| v.* = Rig.bf16(r.rng.float(f32) * 2 - 1);
        // each row's routed picks distinct, then the shared expert in the last slot, as Flash Next routes
        const picks = try a.alloc(i32, pairs);
        defer a.free(picks);
        for (0..n_rows) |row| {
            var used: [routed]bool = @splat(false);
            for (0..slots - 1) |j| {
                var e = r.rng.uintLessThan(usize, routed);
                while (used[e]) e = (e + 1) % routed;
                used[e] = true;
                picks[row * slots + j] = @intCast(e);
            }
            picks[row * slots + slots - 1] = routed;
        }
        const dx = try r.dev(std.mem.sliceAsBytes(x));
        const dpicks = try r.dev(std.mem.sliceAsBytes(picks));
        // the rows kernel: every row in one plan of 32-pair items
        const p = try planFor(&r, pairs, Rows.tile);
        try router.route(r.s, dpicks, pairs, experts, Rows.tile, p);
        const items = pairs / Rows.tile + experts;
        const act = try r.zeros(pairs * width * 2);
        const y = try r.zeros(pairs * dims * 4);
        try kernels.run(r.s, true, dx, dims, slots, up_w, dims / 32, width / 32, p, act, width, items);
        try kernels.run(r.s, false, act, width, 0, down_w, width / 32, dims / 32, p, y, dims, items);
        // the decode kernel: each row alone, its pairs written where the rows kernel writes them
        const act1 = try r.zeros(pairs * width * 2);
        const y1 = try r.zeros(pairs * dims * 4);
        const p1 = try planFor(&r, slots, 16);
        for (0..n_rows) |row| {
            try router.route(r.s, dpicks + row * slots * 4, slots, experts, 16, p1);
            try launch(&r, one_up, dx + row * dims * 2, dims, slots, up_w, dims / 32, width / 32, p1, act1 + row * slots * width * 2, width, slots * (width / 32));
            try launch(&r, one_down, act1 + row * slots * width * 2, width, 0, down_w, width / 32, dims / 32, p1, y1 + row * slots * dims * 4, dims, slots * (dims / 32));
        }
        const got_act = try r.get(act, pairs * width * 2);
        defer a.free(got_act);
        const want_act = try r.get(act1, pairs * width * 2);
        defer a.free(want_act);
        try check.sameBytes("experts_rows SwiGLU rows", got_act, want_act);
        const got_y = try r.get(y, pairs * dims * 4);
        defer a.free(got_y);
        const want_y = try r.get(y1, pairs * dims * 4);
        defer a.free(want_y);
        try check.sameBytes("experts_rows down rows", got_y, want_y);
        check.pass("experts-rows: {d} rows ({d} pairs) equal the decode kernel's one-row launches byte for byte", .{ n_rows, pairs });
    }
}
