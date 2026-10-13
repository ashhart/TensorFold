//! Two ranks of one GLM-5.3 layer on one Mac, partials summed in rank order, against the fp32 reference and each other.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const g = tf.glm_dsa;
const ex = tf.exlr8;
const a = std.heap.page_allocator;
const R = g.gpu.max_rows;

fn readRef(io: std.Io, dir: []const u8, name: []const u8, kind: []const u8) ![]u8 {
    return std.Io.Dir.cwd().readFileAlloc(io, try std.fmt.allocPrint(a, "{s}/{s}.{s}", .{ dir, name, kind }), a, .limited(16 << 30));
}

fn finish(cb: mtl.CommandBuffer) !void {
    cb.commit();
    cb.wait();
    if (cb.failure()) |m| {
        std.debug.print("GPU command failed: {s}\n", .{m});
        return error.GpuFailure;
    }
}

const Rank = struct {
    c: g.config.Config,
    pk: ex.pack.Pack,
    layer: g.layer.Layer,
    s: g.gpu.Scratch,
};

/// rms(d) / rms(ref) and max |d| / rms(ref) over rows [p0, p0 + n), skipping flagged rows.
const Err = struct {
    sq: f64 = 0,
    ref: f64 = 0,
    max: f64 = 0,
    n: usize = 0,

    fn add(e: *Err, ours: []const f32, want: []const f32, skip: bool) void {
        if (skip) return;
        for (ours, want) |o, w| {
            const d = @abs(@as(f64, o) - w);
            e.sq += d * d;
            e.ref += @as(f64, w) * w;
            e.max = @max(e.max, d);
            e.n += 1;
        }
    }

    fn print(e: Err, name: []const u8) void {
        const rms = @sqrt(e.ref / @as(f64, @floatFromInt(@max(e.n, 1))));
        std.debug.print("  {s:<10} rms(d)/rms(ref) {e:.3}  max/rms(ref) {e:.3}\n", .{ name, @sqrt(e.sq / @as(f64, @floatFromInt(@max(e.n, 1)))) / rms, e.max / rms });
    }
};

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const argv = try init.minimal.args.toSlice(init.arena.allocator());
    if (argv.len < 3) return error.Usage;
    const model = argv[1];
    const rdir = argv[2];
    const meta_text = try std.Io.Dir.cwd().readFileAlloc(io, try std.fmt.allocPrint(a, "{s}/meta.json", .{rdir}), a, .limited(1 << 20));
    const meta = try std.json.parseFromSliceLeaky(std.json.Value, a, meta_text, .{ .allocate = .alloc_always });
    const L: u32 = @intCast(meta.object.get("layer").?.integer);
    const T: u32 = @intCast(meta.object.get("T").?.integer);
    const cfg_text = try std.Io.Dir.cwd().readFileAlloc(io, try std.fmt.allocPrint(a, "{s}/config.json", .{model}), a, .limited(1 << 20));
    var manifest = try ex.manifest.Manifest.load(a, io, try std.fmt.allocPrint(a, "{s}/manifests/exlr8-manifest-home.json", .{model}));
    defer manifest.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    var xl = try ex.experts.Layer.open(a, io, model, @intCast(@max(L, 3)));
    defer xl.close(io);
    var ranks: [2]Rank = undefined;
    for (&ranks, 0..) |*rk, r| {
        rk.c = try g.config.parse(a, cfg_text);
        try g.config.split(&rk.c, @intCast(r), 2);
        rk.pk = ex.pack.Pack.init(a);
        const w = try g.weights.load(a, io, model, &rk.c, &manifest, L, &rk.pk, &xl);
        rk.layer = try g.layer.Layer.init(a, device, w, &rk.pk, 1, T, mtl.ResourceOptions.shared);
        rk.s = try g.gpu.Scratch.init(device, &rk.c, T);
    }
    var K = try g.layer.Kernels.load(a, device, &ranks[0].c);
    defer K.deinit();
    const c = &ranks[0].c;
    const hid = c.hidden;
    const h_in: []const f32 = @alignCast(std.mem.bytesAsSlice(f32, try readRef(io, rdir, "h_in", "f32")));
    const attn_out: []const f32 = @alignCast(std.mem.bytesAsSlice(f32, try readRef(io, rdir, "attn_out", "f32")));
    const mlp_out: []const f32 = @alignCast(std.mem.bytesAsSlice(f32, try readRef(io, rdir, "mlp_out", "f32")));
    const h_out: []const f32 = @alignCast(std.mem.bytesAsSlice(f32, try readRef(io, rdir, "h_out", "f32")));
    const topk: []const i32 = @alignCast(std.mem.bytesAsSlice(i32, try readRef(io, rdir, "topk", "i32")));
    const pick_ref: []const i32 = if (c.isMoe(L)) @alignCast(std.mem.bytesAsSlice(i32, try readRef(io, rdir, "pick", "i32"))) else &.{};
    const kcols = topk.len / T;
    var e_attn: Err = .{};
    var e_mlp: Err = .{};
    var e_h: Err = .{};
    var disagree: usize = 0;
    var flips: usize = 0;
    var key_rows: usize = 0;
    var p0: u32 = 0;
    while (p0 < T) : (p0 += R) {
        const n: u32 = @min(R, T - p0);
        var want_keys: [R * 2048]u32 = undefined;
        var want_n: [R]u32 = undefined;
        for (&ranks) |*rk| {
            @memcpy(rk.s.h.slice(f32, n * hid), h_in[p0 * hid ..][0 .. n * hid]);
            const rows = rk.s.rows.slice(g.kernels.Row, n);
            const keys = rk.s.keys.slice(u32, R * c.i_topk);
            for (rows, 0..) |*row, i| {
                const pos = p0 + @as(u32, @intCast(i));
                row.* = .{ .slot = 0, .pos = pos, .list = @intCast(i * c.i_topk), .nkeys = @min(pos + 1, c.i_topk) };
                const list = keys[i * c.i_topk ..][0..c.i_topk];
                if (pos + 1 <= c.i_topk) {
                    for (0..pos + 1) |k| list[k] = @intCast(k);
                } else {
                    var m: usize = 0;
                    for (topk[pos * kcols ..][0..kcols]) |k| if (k >= 0 and k <= pos) {
                        list[m] = @intCast(k);
                        m += 1;
                    };
                    std.mem.sort(u32, list[0..m], {}, std.sort.asc(u32));
                    row.nkeys = @intCast(m);
                }
            }
            @memcpy(want_keys[0 .. n * c.i_topk], keys[0 .. n * c.i_topk]);
            for (rows, 0..) |row, i| want_n[i] = row.nkeys;
            const cb = queue.commandBuffer();
            const e = cb.compute(.serial);
            try g.layer.attention(&K, &rk.layer, &rk.s, e, &rk.c, n, p0 + n - 1, 0, .{ .buf = rk.s.x }, .{ .buf = rk.s.y });
            e.end();
            try finish(cb);
        }
        // both ranks' indexers are the same computation: their key lists must agree exactly
        if (!std.mem.eql(u32, ranks[0].s.keys.slice(u32, n * c.i_topk), ranks[1].s.keys.slice(u32, n * c.i_topk))) disagree += 1;
        // a row whose key list differs from the reference's (a near-tie at the 2,048th score) is left out of the errors
        var other: [R]bool = @splat(false);
        const got_rows = ranks[0].s.rows.slice(g.kernels.Row, n);
        for (0..n) |i| {
            const got = ranks[0].s.keys.slice(u32, n * c.i_topk)[i * c.i_topk ..][0..got_rows[i].nkeys];
            other[i] = got_rows[i].nkeys != want_n[i] or !std.mem.eql(u32, got, want_keys[i * c.i_topk ..][0..want_n[i]]);
            key_rows += @intFromBool(other[i]);
        }
        var y: [R * 6144]f32 = undefined;
        for (0..n * hid) |i| y[i] = ranks[0].s.y.slice(f32, n * hid)[i] + ranks[1].s.y.slice(f32, n * hid)[i];
        for (0..n) |i| e_attn.add(y[i * hid ..][0..hid], attn_out[(p0 + i) * hid ..][0..hid], other[i]);
        for (&ranks) |*rk| for (rk.s.h.slice(f32, n * hid), y[0 .. n * hid]) |*h, v| {
            h.* += v;
        };
        for (&ranks) |*rk| {
            var cb = queue.commandBuffer();
            var e = cb.compute(.serial);
            g.layer.postNorm(&K, &rk.layer, &rk.s, e, &rk.c, n, 0);
            e.barrier();
            if (rk.c.isMoe(L)) try g.layer.route(&K, &rk.layer, &rk.s, e, &rk.c, n, .{ .buf = rk.s.x }, null, null) else try g.layer.denseMlp(&K, &rk.layer, &rk.s, e, &rk.c, n, .{ .buf = rk.s.x }, .{ .buf = rk.s.y });
            e.end();
            try finish(cb);
            if (!rk.c.isMoe(L)) continue;
            var plan = try g.moe.Plan.build(a, &rk.c, &rk.layer.w.mlp.moe, n, rk.s.route[0].slice(i32, n * 8), rk.s.route[1].slice(f32, n * 8));
            defer plan.deinit(a);
            @memcpy(rk.s.moe_jobs.slice(ex.job.Job, plan.jobs.items.len), plan.jobs.items);
            @memcpy(rk.s.members.slice(u32, plan.members.items.len), plan.members.items);
            @memcpy(rk.s.wts.slice(f32, n * g.gpu.slots), plan.wts[0 .. n * g.gpu.slots]);
            @memcpy(rk.s.sv.slice(u32, n * g.gpu.slots), plan.sv[0 .. n * g.gpu.slots]);
            @memcpy(rk.s.suh.slice(u32, n * g.gpu.slots), plan.suh[0 .. n * g.gpu.slots]);
            cb = queue.commandBuffer();
            e = cb.compute(.serial);
            try g.layer.experts(&K, &rk.layer, &rk.s, e, &rk.c, n, plan.gu_jobs, plan.down_jobs, .{ .buf = rk.s.y });
            e.end();
            try finish(cb);
        }
        if (!std.mem.eql(f32, ranks[0].s.x.slice(f32, n * hid), ranks[1].s.x.slice(f32, n * hid))) disagree += 1;
        for (0..n * hid) |i| y[i] = ranks[0].s.y.slice(f32, n * hid)[i] + ranks[1].s.y.slice(f32, n * hid)[i];
        for (0..n) |i| {
            var flip: bool = false;
            if (c.isMoe(L)) {
                var mine: [8]i32 = undefined;
                var want: [8]i32 = undefined;
                @memcpy(&mine, ranks[0].s.route[0].slice(i32, n * 8)[i * 8 ..][0..8]);
                @memcpy(&want, pick_ref[(p0 + i) * 8 ..][0..8]);
                std.mem.sort(i32, &mine, {}, std.sort.asc(i32));
                std.mem.sort(i32, &want, {}, std.sort.asc(i32));
                flip = !std.mem.eql(i32, &mine, &want);
            }
            flips += @intFromBool(flip);
            flip = flip or other[i];
            e_mlp.add(y[i * hid ..][0..hid], mlp_out[(p0 + i) * hid ..][0..hid], flip);
            var ho: [6144]f32 = undefined;
            for (&ho, ranks[0].s.h.slice(f32, n * hid)[i * hid ..][0..hid], y[i * hid ..][0..hid]) |*o, h, v| o.* = h + v;
            e_h.add(&ho, h_out[(p0 + i) * hid ..][0..hid], flip);
        }
    }
    std.debug.print("layer {d}, {d} tokens over two ranks: {d} windows where the ranks disagree; {d} rows attend other keys and {d} pick other experts than the reference (left out)\n", .{ L, T, disagree, key_rows, flips });
    e_attn.print("attn_out");
    e_mlp.print("mlp_out");
    e_h.print("h_out");
}
