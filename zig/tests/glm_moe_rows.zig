//! tf-glm-moe-rows: GLM-5.3-Flash's decode experts at two and four outputs a simdgroup, 1-16 rows: bits, then GPU time.
const std = @import("std");
const platform = @import("sys");
const mtl = @import("metal");
const tf = @import("tensorfold");
const kernels = tf.glm.kernels;
const affine = tf.glm.affine;
const Ref = affine.Ref;

const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;
const a = std.heap.page_allocator;
const reps = 32;
const runs = 7;
const E = 64; // experts in the pool (906 MB: past the GPU's caches)
const D = 4096;
const N = 2048;
const TOPK = 8;
const MAXR = 16;

fn buf(device: mtl.Device, bytes: usize) !mtl.Buffer {
    return device.buffer(@max(bytes, 16), opts);
}

fn toBf(v: f32) u16 {
    const u: u32 = @bitCast(v);
    return @truncate((u +% 0x7fff +% ((u >> 16) & 1)) >> 16);
}

fn mat(device: mtl.Device, r: std.Random, n: usize, k: usize, list: *std.ArrayList(mtl.Buffer)) !affine.Affine {
    const w: affine.Width = .{ .bits = 4, .group = 64 };
    const wb = affine.wBytes(n, k, w);
    const sb = affine.sBytes(n, k, w);
    const bs = [3]mtl.Buffer{ try buf(device, wb), try buf(device, sb), try buf(device, sb) };
    for (bs[0].slice(u32, wb / 4)) |*v| v.* = r.int(u32);
    for (bs[1].slice(u16, sb / 2)) |*v| v.* = toBf(r.float(f32) * 0.02 + 0.001);
    for (bs[2].slice(u16, sb / 2)) |*v| v.* = toBf((r.float(f32) - 0.5) * 0.1);
    try list.appendSlice(a, &bs);
    return .{ .w = .{ .buf = bs[0] }, .s = .{ .buf = bs[1] }, .b = .{ .buf = bs[2] }, .n = @intCast(n), .k = @intCast(k), .bits = 4, .group = 64 };
}

/// Launch i's route: picks from a window of the pool (a drafted window's overlap), grouped as route_select groups them.
fn route(ids: []i32, members: []i32, count: *i32, i: usize, rows: u32, r: std.Random) void {
    const span: usize = @min(E, 8 + 5 * (rows - 1));
    var picks: [MAXR * TOPK]i32 = undefined;
    for (0..rows) |row| {
        var k: usize = 0;
        while (k < TOPK) {
            const e: i32 = @intCast((i * 13 + r.uintLessThan(usize, span)) % E);
            if (std.mem.indexOfScalar(i32, picks[row * TOPK .. row * TOPK + k], e) == null) {
                picks[row * TOPK + k] = e;
                k += 1;
            }
        }
    }
    var n: i32 = 0;
    @memset(members, -1);
    for (0..E) |e| {
        var c: usize = 0;
        for (picks[0 .. rows * TOPK], 0..) |p, pi| if (p == @as(i32, @intCast(e))) {
            members[@as(usize, @intCast(n)) * MAXR + c] = @intCast(pi);
            c += 1;
        };
        if (c > 0) {
            ids[@intCast(n)] = @intCast(e);
            n += 1;
        }
    }
    count.* = n;
}

const Route = struct { ids: Ref, members: Ref, count: Ref };
const Experts = struct { gate: affine.Affine, up: affine.Affine, down: affine.Affine, sh_gate_up: affine.Affine, sh_down: affine.Affine };
const Set = struct { w: Experts, rt: [reps]Route, x: Ref, act: Ref, y: Ref, act2: Ref, y2: Ref };

/// moe.zig `both` with gate/up at `rps[0]` and down at `rps[1]` outputs a simdgroup (2: the _one pipelines, 4: not).
fn launch(e: mtl.ComputeEncoder, i: usize, k: *const kernels.Kernels, s: *const Set, rows: u32, rps: [2]u32, out: [2]Ref) void {
    const w = s.w;
    const rt = s.rt[i % reps];
    const zs = rows * TOPK + 1;
    e.setPipeline(if (rps[0] == 2) k.moe_gateup_0_one else k.moe_gateup_0);
    e.setBuffer(s.x.buf, s.x.off, 0);
    const xs = [2]i32{ @intCast(rows), D };
    e.setBytes(std.mem.asBytes(&xs), 1);
    inline for (.{ w.gate.w, w.gate.s, w.gate.b, w.up.w, w.up.s, w.up.b, w.sh_gate_up.w, w.sh_gate_up.s, w.sh_gate_up.b, rt.ids, rt.members, rt.count }, 2..) |r, j| e.setBuffer(r.buf, r.off, j);
    e.setValue(@as(f32, 10.0), 14);
    e.setBuffer(out[0].buf, out[0].off, 15);
    e.dispatchThreads(mtl.Size.of(32 * rows, N / rps[0], zs), mtl.Size.of(32 * rows, 1, 1));
    e.setPipeline(if (rps[1] == 2) k.moe_down_0_one else k.moe_down_0);
    e.setBuffer(out[0].buf, out[0].off, 0);
    const ds = [3]i32{ @intCast(rows), TOPK + 1, N };
    e.setBytes(std.mem.asBytes(&ds), 1);
    inline for (.{ w.down.w, w.down.s, w.down.b, w.sh_down.w, w.sh_down.s, w.sh_down.b, rt.ids, rt.members, rt.count, out[1] }, 2..) |r, j| e.setBuffer(r.buf, r.off, j);
    e.dispatchThreads(mtl.Size.of(32 * rows, D / rps[1], zs), mtl.Size.of(32 * rows, 1, 1));
}

fn run(q: mtl.Queue, comptime f: anytype, args: anytype, n: usize) !f64 {
    const cb = q.commandBuffer();
    const e = cb.compute(.serial);
    for (0..n) |i| @call(.auto, f, .{ e, i } ++ args);
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |msg| {
        std.debug.print("GPU failed: {s}\n", .{msg});
        return error.GpuFailed;
    }
    return cb.gpuSeconds() * 1e6 / @as(f64, @floatFromInt(n));
}

fn time(q: mtl.Queue, comptime f: anytype, args: anytype) !f64 {
    var t: [runs + 1]f64 = undefined;
    for (&t) |*v| v.* = try run(q, f, args, reps);
    std.mem.sort(f64, t[1..], {}, std.sort.asc(f64));
    return t[1 + runs / 2];
}

pub fn main() !void {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    const k = try kernels.load(a, device);
    defer k.deinit();
    var prng = std.Random.DefaultPrng.init(0x77);
    const r = prng.random();
    var bufs: std.ArrayList(mtl.Buffer) = .empty;
    defer for (bufs.items) |b| b.deinit();
    var s: Set = undefined;
    s.w = .{ .gate = try mat(device, r, E * N, D, &bufs), .up = try mat(device, r, E * N, D, &bufs), .down = try mat(device, r, E * D, N, &bufs), .sh_gate_up = try mat(device, r, 2 * N, D, &bufs), .sh_down = try mat(device, r, D, N, &bufs) };
    const io = [_]mtl.Buffer{ try buf(device, MAXR * D * 2), try buf(device, MAXR * 9 * N * 2), try buf(device, MAXR * 9 * D * 2), try buf(device, MAXR * 9 * N * 2), try buf(device, MAXR * 9 * D * 2) };
    try bufs.appendSlice(a, &io);
    for (io[0].slice(u16, MAXR * D)) |*v| v.* = toBf((r.float(f32) - 0.5) * 2);
    s.x = .{ .buf = io[0] };
    s.act = .{ .buf = io[1] };
    s.y = .{ .buf = io[2] };
    s.act2 = .{ .buf = io[3] };
    s.y2 = .{ .buf = io[4] };
    const route_buf = try buf(device, reps * (128 + 128 * MAXR + 1) * 4);
    try bufs.append(a, route_buf);
    const one = (128 + 128 * MAXR + 1) * 4;
    const per_row = 3 * affine.bytes(N, D, .{ .bits = 4, .group = 64 });
    var checked: usize = 0;
    std.debug.print("GPU {s}: decode experts, {d} launches a command buffer, median of {d}\n", .{ device.name(), reps, runs });
    var it = std.mem.tokenizeScalar(u8, if (platform.c.getenv("GLM_MOE_ROWS")) |v| std.mem.span(v) else "1,2,4,8", ',');
    while (it.next()) |rv| {
        const rows = try std.fmt.parseInt(u32, rv, 10);
        var uniq: usize = 0;
        for (0..reps) |i| {
            const base = Ref{ .buf = route_buf, .off = i * one };
            const ids = route_buf.slice(i32, reps * one / 4)[i * one / 4 ..][0..128];
            const members = route_buf.slice(i32, reps * one / 4)[i * one / 4 + 128 ..][0 .. 128 * MAXR];
            const count = &route_buf.slice(i32, reps * one / 4)[i * one / 4 + 128 + 128 * MAXR];
            route(ids, members, count, i, rows, r);
            uniq += @intCast(count.*);
            s.rt[i] = .{ .ids = base, .members = base.at(128 * 4), .count = base.at((128 + 128 * MAXR) * 4) };
        }
        const bytes = @as(f64, @floatFromInt(uniq)) / reps * @as(f64, @floatFromInt(per_row)) + @as(f64, @floatFromInt(per_row));
        const planes = [2]usize{ rows * 9 * N, rows * 9 * D };
        @memset(io[1].slice(u16, planes[0]), 0);
        @memset(io[2].slice(u16, planes[1]), 0);
        _ = try run(queue, launch, .{ k, &s, rows, [2]u32{ 4, 4 }, [2]Ref{ s.act, s.y } }, 1);
        for (0..24) |_| _ = try run(queue, launch, .{ k, &s, rows, [2]u32{ 4, 4 }, [2]Ref{ s.act2, s.y2 } }, reps); // clocks up first
        std.debug.print("  {d:2} rows ({d:.1} experts + shared), gate/up + down outputs a simdgroup:", .{ rows, @as(f64, @floatFromInt(uniq)) / reps });
        for ([_][2]u32{ .{ 4, 4 }, .{ 2, 2 }, .{ 2, 4 }, .{ 4, 2 }, .{ 4, 4 } }) |rps| {
            @memset(io[3].slice(u16, planes[0]), 0xffff);
            @memset(io[4].slice(u16, planes[1]), 0xffff);
            _ = try run(queue, launch, .{ k, &s, rows, rps, [2]Ref{ s.act2, s.y2 } }, 1);
            if (!std.mem.eql(u16, io[3].slice(u16, planes[0]), io[1].slice(u16, planes[0])) or !std.mem.eql(u16, io[4].slice(u16, planes[1]), io[2].slice(u16, planes[1]))) {
                std.debug.print("\n{d} rows, {d} + {d}: BITS DIFFER\n", .{ rows, rps[0], rps[1] });
                return error.BitsDiffer;
            }
            checked += planes[0] + planes[1];
            const us = try time(queue, launch, .{ k, &s, rows, rps, [2]Ref{ s.act2, s.y2 } });
            std.debug.print(" {d}+{d} {d:.1} us ({d:.0} GB/s)", .{ rps[0], rps[1], us, bytes / us / 1e3 });
        }
        std.debug.print("\n", .{});
    }
    std.debug.print("{d} outputs equal four outputs a simdgroup's\n", .{checked});
}
