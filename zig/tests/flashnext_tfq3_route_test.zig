//! Flash Next TFQ3 B=2..16 routing: fwht_x_fwd's parallel rows and dedup_par against route_dedup, byte for byte.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const tfq3 = tf.flashnext_replay.tfq3;

const D = tfq3.hidden_size;
const NE = tfq3.num_experts;
const TOPK = tfq3.topk;

/// Logits, rows and both paths' outputs (pick, wts, uids, umem, ucount), each output buffer prefilled with 0xA5.
const RouteRig = struct {
    lg: mtl.Buffer,
    x: mtl.Buffer,
    rows: mtl.Buffer,
    out: [2][5]mtl.Buffer,
    const sizes = [5]usize{ 16 * TOPK * 4, 16 * TOPK * 4, 160 * 4, 160 * 16 * 4, 16 };
    const opt = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;

    fn init(dev: mtl.Device) !RouteRig {
        var rig: RouteRig = undefined;
        rig.lg = try dev.buffer(16 * 513 * 4, opt);
        errdefer rig.lg.deinit();
        rig.x = try dev.buffer(16 * D * 2, opt);
        errdefer rig.x.deinit();
        @memset(rig.x.contents()[0 .. 16 * D * 2], 0);
        rig.rows = try dev.buffer(16, opt);
        errdefer rig.rows.deinit();
        var made: usize = 0;
        errdefer for (0..made) |k| rig.out[k / 5][k % 5].deinit();
        for (&rig.out) |*side| for (side, sizes) |*b, n| {
            b.* = try dev.buffer(n, opt);
            made += 1;
        };
        return rig;
    }

    fn deinit(rig: *RouteRig) void {
        rig.lg.deinit();
        rig.x.deinit();
        rig.rows.deinit();
        for (rig.out) |side| for (side) |b| b.deinit();
    }

    /// Every case's outputs from route_dedup and from fx (fwht_x_fwd, flags 4) + dd (dedup_par) byte-equal, 5 runs.
    fn agrees(rig: *RouteRig, queue: anytype, p: *const tfq3.Pipes, fx: mtl.Pipeline, dd: mtl.Pipeline) !bool {
        var rng = std.Random.DefaultPrng.init(0x7f3a);
        const rand = rng.random();
        for (2..17) |window| for ([_]usize{ window, window - 1, 1, 0, window + 3 }) |live| for (0..5) |variant| {
            const lg = rig.lg.slice(f32, 16 * 513);
            for (lg) |*v| v.* = (rand.float(f32) - 0.5) * 2.0;
            for (0..window) |r| switch (variant) {
                0 => {}, // independent rows: few shared experts
                1 => for (0..12) |k| { // every row's top experts the same set, in a different order a row
                    lg[r * 513 + (k * 37 + 5) % NE] = 9.0 - @as(f32, @floatFromInt((k + r) % 12)) * 0.25;
                },
                2 => for (0..TOPK) |k| { // ties: equal logits across experts, broken by expert id
                    lg[r * 513 + (k * 51 + r * 3) % NE] = 7.5;
                },
                3 => for (0..TOPK) |k| { // rows in pairs: row r shares its picks with row r ^ 1
                    lg[r * 513 + ((r >> 1) * 29 + k * 7) % NE] = 8.0 + @as(f32, @floatFromInt(k)) * 0.125;
                },
                else => { // three finite logits: the rest of the top 10 repeats the lowest id among -inf, in one row
                    @memset(lg[r * 513 ..][0..513], -std.math.inf(f32));
                    for (0..3) |k| lg[r * 513 + (k * 97 + r * 11) % NE] = 1.0 + @as(f32, @floatFromInt(k));
                },
            };
            rig.rows.slice(i32, 1)[0] = @intCast(live);
            const d: tfq3.Dims = .{ .rows = @intCast(window), .slots = @intCast(window * TOPK), .topk = TOPK, .experts = NE, .flags = 4 };
            var first: [5][]u8 = undefined;
            var have_first = false;
            defer if (have_first) for (first) |f| std.testing.allocator.free(f);
            for (0..5) |run| {
                for (rig.out) |side| for (side, sizes) |b, n| @memset(b.contents()[0..n], 0xA5);
                const cb = queue.commandBuffer();
                const e = cb.compute(.serial);
                e.setPipeline(p.get(.route_dedup));
                e.setBuffer(rig.lg, 0, 0);
                e.setBuffer(rig.rows, 0, 1);
                for (0..5) |j| e.setBuffer(rig.out[0][j], 0, 2 + j);
                e.setValue(d, 7);
                e.dispatchGroups(mtl.Size.of(1, 1, 1), mtl.Size.of(32, 1, 1));
                e.setPipeline(fx);
                e.setBuffer(rig.x, 0, 0);
                e.setBuffer(p.xrot_buf, 0, 1);
                e.setBuffer(rig.rows, 0, 2);
                e.setValue(d, 3);
                e.setBuffer(rig.lg, 0, 4);
                e.setBuffer(rig.out[1][0], 0, 5);
                e.setBuffer(rig.out[1][1], 0, 6);
                e.dispatchGroups(mtl.Size.of(21, window, 1), mtl.Size.of(32, 1, 1));
                e.setPipeline(dd);
                e.setBuffer(rig.out[1][0], 0, 0);
                e.setBuffer(rig.rows, 0, 1);
                for (2..5) |j| e.setBuffer(rig.out[1][j], 0, j);
                e.setValue(d, 5);
                e.dispatchGroups(mtl.Size.of(1, 1, 1), mtl.Size.of(tfq3.max_decode_slots, 1, 1));
                e.end();
                cb.commit();
                cb.wait();
                if (cb.failure()) |msg| {
                    std.debug.print("route rig: {s}\n", .{msg});
                    return error.GpuFailed;
                }
                for (0..5) |j| {
                    const got = rig.out[1][j].contents()[0..sizes[j]];
                    if (!std.mem.eql(u8, rig.out[0][j].contents()[0..sizes[j]], got)) return false;
                    if (have_first and !std.mem.eql(u8, first[j], got)) return false;
                }
                if (run == 0) {
                    var made: usize = 0;
                    errdefer for (first[0..made]) |f| std.testing.allocator.free(f);
                    for (0..5) |j| {
                        first[j] = try std.testing.allocator.dupe(u8, rig.out[1][j].contents()[0..sizes[j]]);
                        made += 1;
                    }
                    have_first = true;
                }
            }
        };
        return true;
    }
};

test "Flash Next TFQ3: parallel routing and dedup give route_dedup's bytes at B=2..16, live rows short too; planted controls fail" {
    const a = std.testing.allocator;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const dev = try mtl.Device.init();
    defer dev.deinit();
    const queue = try dev.queue();
    defer queue.deinit();
    var pipes = try tfq3.Pipes.init(a, dev, null);
    defer pipes.deinit();
    var rig = try RouteRig.init(dev);
    defer rig.deinit();
    try std.testing.expect(try rig.agrees(queue, &pipes, pipes.get(.fwht_x_fwd), pipes.get(.dedup_par)));

    const src = try tfq3.buildSource(a, dev.tensorUnits());
    defer a.free(src);
    const Control = struct { name: []const u8, from: []const u8, to: []const u8 };
    const controls = [_]Control{
        .{ .name = "lists in expert order", .from = "t < i; ++t) u += int(opens[t]);", .to = "t < i; ++t) u += int(opens[t] != 0 && pk[t] < pk[i]);" },
        .{ .name = "no -1 tail", .from = "for (; m < 16; ++m) umem[u * 16 + m] = -1;", .to = "" },
        .{ .name = "window rows, not live rows", .from = "const int S = metal::min(metal::max(R, 0) * 10, 160);", .to = "const int S = int(d.rows) * 10;" },
        .{ .name = "members past the first dropped", .from = "if (pk[t] == pk[i]) umem[u * 16 + m++] = t;", .to = "if (pk[t] == pk[i] && t == i) umem[u * 16 + m++] = t;" },
        .{ .name = "row routed at the wrong place", .from = "pick + ulong(gid.y) * 10ul", .to = "pick + ulong(gid.y) * 9ul" },
    };
    for (controls) |c| {
        try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, src, c.from));
        const text = try std.mem.replaceOwned(u8, a, src, c.from, c.to);
        defer a.free(text);
        const lib = try mtl.Library.fromSource(dev, text, mtl.CompileOptions.mlx());
        defer lib.deinit();
        const fx = try mtl.Pipeline.init(dev, lib, "fn_pw_fwht_x_fwd", false);
        defer fx.deinit();
        const dd = try mtl.Pipeline.init(dev, lib, "fn_pw_dedup_par", false);
        defer dd.deinit();
        if (try rig.agrees(queue, &pipes, fx, dd)) {
            std.debug.print("planted control passed: {s}\n", .{c.name});
            return error.PlantedControlPassed;
        }
    }
}
