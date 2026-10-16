//! One GLM-5.3 layer on Metal against the fp32 reference forward of that layer (ref_layer.py), window by window.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const g = tf.glm_dsa;
const ex = tf.exlr8;
const a = std.heap.page_allocator;
const R = g.gpu.max_rows;

/// A reference tensor read from REF_DIR/<name>.<f32|i32>.
const Ref = struct {
    f: []f32 = &.{},
    i: []i32 = &.{},
    cols: usize = 0,

    fn load(io: std.Io, dir: []const u8, meta: std.json.Value, name: []const u8) !Ref {
        const info = (meta.object.get("saved") orelse return error.BadRef).object.get(name) orelse return .{};
        const shape = info.object.get("shape").?.array.items;
        var cols: usize = 1;
        for (shape[1..]) |d| cols *= @intCast(d.integer);
        const kind = info.object.get("dtype").?.string;
        const path = try std.fmt.allocPrint(a, "{s}/{s}.{s}", .{ dir, name, kind });
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(16 << 30));
        if (std.mem.eql(u8, kind, "i32")) return .{ .i = @alignCast(std.mem.bytesAsSlice(i32, bytes)), .cols = cols };
        return .{ .f = @alignCast(std.mem.bytesAsSlice(f32, bytes)), .cols = cols };
    }
};

/// Our error against the reference over a run of rows: the largest |d| and rms(d), beside rms(ref).
const Stat = struct {
    name: []const u8,
    max: f64 = 0,
    sq: f64 = 0,
    ref_sq: f64 = 0,
    n: usize = 0,
    worst_row: usize = 0,

    fn add(s: *Stat, ours: []const f32, ref: []const f32, row0: usize, cols: usize, stride: usize, rows: usize, skip: []const bool) void {
        for (0..rows) |r| {
            if (skip.len > 0 and skip[row0 + r]) continue;
            for (0..cols) |c| {
                const o: f64 = ours[r * stride + c];
                const w: f64 = ref[(row0 + r) * cols + c];
                const d = @abs(o - w);
                if (!(d <= s.max)) {
                    s.max = d;
                    s.worst_row = row0 + r;
                }
                s.sq += d * d;
                s.ref_sq += w * w;
                s.n += 1;
            }
        }
    }

    fn print(s: Stat) void {
        const rms = @sqrt(s.ref_sq / @as(f64, @floatFromInt(@max(s.n, 1))));
        std.debug.print("  {s:<12} rms(ref) {e:.3}  max |d| {e:.3} (row {d})  rms(d)/rms(ref) {e:.3}  max/rms(ref) {e:.3}\n", .{ s.name, rms, s.max, s.worst_row, @sqrt(s.sq / @as(f64, @floatFromInt(@max(s.n, 1)))) / rms, s.max / rms });
    }
};

fn finish(cb: mtl.CommandBuffer) !void {
    cb.commit();
    cb.wait();
    if (cb.failure()) |m| {
        std.debug.print("GPU command failed: {s}\n", .{m});
        return error.GpuFailure;
    }
}

fn bf16s(b: mtl.Buffer, at: usize, n: usize, out: []f32) void {
    const v = b.slice(u16, at + n)[at..][0..n];
    for (out[0..n], v) |*o, x| o.* = @bitCast(@as(u32, x) << 16);
}

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const argv = try init.minimal.args.toSlice(init.arena.allocator());
    if (argv.len < 3) {
        std.debug.print("usage: tf-glm-dsa-layer-check MODEL_DIR REF_DIR [--ref-topk]\n", .{});
        return error.Usage;
    }
    const model = argv[1];
    const rdir = argv[2];
    const ref_topk = argv.len > 3 and std.mem.eql(u8, argv[3], "--ref-topk");
    const meta_text = try std.Io.Dir.cwd().readFileAlloc(io, try std.fmt.allocPrint(a, "{s}/meta.json", .{rdir}), a, .limited(1 << 20));
    const meta = try std.json.parseFromSliceLeaky(std.json.Value, a, meta_text, .{ .allocate = .alloc_always });
    const L: u32 = @intCast(meta.object.get("layer").?.integer);
    const T: u32 = @intCast(meta.object.get("T").?.integer);
    const cfg_text = try std.Io.Dir.cwd().readFileAlloc(io, try std.fmt.allocPrint(a, "{s}/config.json", .{model}), a, .limited(1 << 20));
    var c = try g.config.parse(a, cfg_text);
    var manifest = try ex.manifest.Manifest.load(a, io, try std.fmt.allocPrint(a, "{s}/manifests/exlr8-manifest-home.json", .{model}));
    defer manifest.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    var pk = ex.pack.Pack.init(a);
    defer pk.deinit();
    var xl = if (c.isMoe(L)) try ex.experts.Layer.open(a, io, model, @intCast(L)) else undefined;
    defer if (c.isMoe(L)) xl.close(io);
    const t0 = std.Io.Clock.awake.now(io).toNanoseconds();
    const w = try g.weights.load(a, io, model, &c, &manifest, L, &pk, &xl);
    var layer = try g.layer.Layer.init(a, device, w, &pk, 1, T, mtl.ResourceOptions.shared);
    defer layer.deinit(a);
    var K = try g.layer.Kernels.load(a, device, &c);
    defer K.deinit();
    var s = try g.gpu.Scratch.init(device, &c, T);
    defer s.deinit();
    std.debug.print("layer {d}{s}, {d} tokens: packed {d:.1} MB in {d:.1} s on {s}\n", .{ L, if (w.full) " (own indexer)" else " (shared top-k)", T, @as(f64, @floatFromInt(pk.bytes.items.len)) / 1e6, @as(f64, @floatFromInt(std.Io.Clock.awake.now(io).toNanoseconds() - t0)) / 1e9, device.name() });
    var refs: std.StringHashMapUnmanaged(Ref) = .empty;
    for ([_][]const u8{ "h_in", "x", "q_resid", "latent", "k_rot", "q_rot", "attn", "attn_out", "x2", "router_logits", "pick", "pick_w", "mlp_out", "h_out", "topk", "index_key" }) |n| try refs.put(a, n, try Ref.load(io, rdir, meta, n));
    var stats = [_]Stat{ .{ .name = "x" }, .{ .name = "q_resid" }, .{ .name = "latent" }, .{ .name = "k_rot" }, .{ .name = "index_key" }, .{ .name = "attn" }, .{ .name = "attn_out" }, .{ .name = "x2" }, .{ .name = "logits" }, .{ .name = "mlp_out" }, .{ .name = "h_out" } };
    const hid = c.hidden;
    const nh = c.localHeads();
    const topk = refs.get("topk").?;
    var pick_diff = try a.alloc(bool, T);
    @memset(pick_diff, false);
    var topk_diff = try a.alloc(bool, T);
    @memset(topk_diff, false);
    var topk_rows: usize = 0;
    var gpu_jobs_differ: usize = 0;
    var missing_keys: usize = 0;
    var p0: u32 = 0;
    while (p0 < T) : (p0 += R) {
        const n: u32 = @min(R, T - p0);
        @memcpy(s.h.slice(f32, n * hid), refs.get("h_in").?.f[p0 * hid ..][0 .. n * hid]);
        const rows = s.rows.slice(g.kernels.Row, n);
        const keys = s.keys.slice(u32, R * c.i_topk);
        for (rows, 0..) |*r, i| {
            const pos = p0 + @as(u32, @intCast(i));
            r.* = .{ .slot = 0, .pos = pos, .list = @intCast(i * c.i_topk), .nkeys = @min(pos + 1, c.i_topk) };
            // the reference's key set for the row, sorted: what a shared layer (or --ref-topk) attends to
            const list = keys[i * c.i_topk ..][0..c.i_topk];
            if (pos + 1 <= c.i_topk) {
                for (0..pos + 1) |k| list[k] = @intCast(k);
            } else {
                var m: usize = 0;
                for (topk.i[pos * topk.cols ..][0..topk.cols]) |k| if (k >= 0 and k <= pos) {
                    list[m] = @intCast(k);
                    m += 1;
                };
                std.mem.sort(u32, list[0..m], {}, std.sort.asc(u32));
                r.nkeys = @intCast(m);
            }
        }
        var want_keys: [R][2048]u32 = undefined;
        for (0..n) |i| @memcpy(want_keys[i][0..rows[i].nkeys], keys[i * c.i_topk ..][0..rows[i].nkeys]);
        const want_n = blk: {
            var v: [R]u32 = undefined;
            for (0..n) |i| v[i] = rows[i].nkeys;
            break :blk v;
        };
        var cb = queue.commandBuffer();
        var e = cb.compute(.serial);
        var lw = layer;
        if (ref_topk) lw.w.full = false;
        try g.layer.attention(&K, &lw, &s, e, &c, n, p0 + n - 1, 0, .{ .buf = s.x }, .{ .buf = s.y });
        e.end();
        try finish(cb);
        if (w.full and !ref_topk) for (0..n) |i| {
            const got = keys[i * c.i_topk ..][0..rows[i].nkeys];
            if (rows[i].nkeys != want_n[i] or !std.mem.eql(u32, got, want_keys[i][0..want_n[i]])) {
                topk_diff[p0 + i] = true;
                topk_rows += 1;
                var miss: usize = 0;
                for (want_keys[i][0..want_n[i]]) |k| {
                    if (std.mem.indexOfScalar(u32, got, k) == null) miss += 1;
                }
                missing_keys += miss;
            }
        };
        if (p0 == 0 and init.environ_map.get("TF_GLM_DEBUG") != null) {
            var c0: [8]f32 = undefined;
            bf16s(layer.cache, 0, 8, &c0);
            const uvj = pk.dense_jobs.items[w.attn.uv_jobs.lo];
            std.debug.print("debug uv job 0: {any}\n  groups uv {any} x {any}\n", .{ uvj, layer.groups[1], layer.groups[0] });
            const gl = layer.k.group_list.items;
            std.debug.print("  first uv groups {any}\n", .{gl[layer.groups[1].off / 8 ..][0..3]});
            std.debug.print("debug row 0: latent {any}\n  olat h0 {any}\n  o {any}\n  ref attn {any}\n  qrope h0 {any}\n  ref q_rot h0 {any}\n  nkeys {d} keys {any}\n", .{ c0, s.olat.slice(f32, 8)[0..8], s.o.slice(f32, 8)[0..8], refs.get("attn").?.f[0..8], s.qrope.slice(f32, 8)[0..8], refs.get("q_rot").?.f[0..8], rows[0].nkeys, keys[0..4] });
        }
        stats[0].add(s.x.slice(f32, n * hid), refs.get("x").?.f, p0, hid, hid, n, &.{});
        stats[1].add(s.q_resid.slice(f32, n * c.q_lora), refs.get("q_resid").?.f, p0, c.q_lora, c.q_lora, n, &.{});
        var lat: [R * 576]f32 = undefined;
        bf16s(layer.cache, p0 * 576, n * 576, &lat);
        var only_lat: [R * 512]f32 = undefined;
        var only_rope: [R * 64]f32 = undefined;
        for (0..n) |i| {
            @memcpy(only_lat[i * 512 ..][0..512], lat[i * 576 ..][0..512]);
            @memcpy(only_rope[i * 64 ..][0..64], lat[i * 576 + 512 ..][0..64]);
        }
        stats[2].add(&only_lat, refs.get("latent").?.f, p0, 512, 512, n, &.{});
        stats[3].add(&only_rope, refs.get("k_rot").?.f, p0, 64, 64, n, &.{});
        if (layer.index_cache) |ic| if (!ref_topk) {
            var ik: [R * 128]f32 = undefined;
            bf16s(ic, p0 * 128, n * 128, &ik);
            stats[4].add(&ik, refs.get("index_key").?.f, p0, 128, 128, n, &.{});
        };
        stats[5].add(s.o.slice(f32, n * nh * c.v_dim), refs.get("attn").?.f, p0, nh * c.v_dim, nh * c.v_dim, n, topk_diff);
        stats[6].add(s.y.slice(f32, n * hid), refs.get("attn_out").?.f, p0, hid, hid, n, topk_diff);
        cb = queue.commandBuffer();
        e = cb.compute(.serial);
        g.layer.postNorm(&K, &layer, &s, e, &c, n, 1);
        e.barrier();
        if (c.isMoe(L)) try g.layer.route(&K, &layer, &s, e, &c, n, .{ .buf = s.x }, null, null) else try g.layer.denseMlp(&K, &layer, &s, e, &c, n, .{ .buf = s.x }, .{ .buf = s.y });
        e.end();
        try finish(cb);
        stats[7].add(s.x.slice(f32, n * hid), refs.get("x2").?.f, p0, hid, hid, n, topk_diff);
        if (c.isMoe(L)) {
            stats[8].add(s.logits.slice(f32, n * c.experts), refs.get("router_logits").?.f, p0, c.experts, c.experts, n, topk_diff);
            const pick = s.route[0].slice(i32, n * c.topk);
            const want = refs.get("pick").?.i;
            for (0..n) |i| {
                var a_set: [8]i32 = undefined;
                var b_set: [8]i32 = undefined;
                @memcpy(&a_set, pick[i * 8 ..][0..8]);
                @memcpy(&b_set, want[(p0 + i) * 8 ..][0..8]);
                std.mem.sort(i32, &a_set, {}, std.sort.asc(i32));
                std.mem.sort(i32, &b_set, {}, std.sort.asc(i32));
                if (!std.mem.eql(i32, &a_set, &b_set)) pick_diff[p0 + i] = true;
            }
            var plan = try g.moe.Plan.build(a, &c, &w.mlp.moe, n, pick, s.route[1].slice(f32, n * c.topk));
            defer plan.deinit(a);
            @memcpy(s.moe_jobs.slice(ex.job.Job, plan.jobs.items.len), plan.jobs.items);
            @memcpy(s.members.slice(u32, plan.members.items.len), plan.members.items);
            @memcpy(s.wts.slice(f32, n * g.gpu.slots), plan.wts[0 .. n * g.gpu.slots]);
            @memcpy(s.sv.slice(u32, n * g.gpu.slots), plan.sv[0 .. n * g.gpu.slots]);
            @memcpy(s.suh.slice(u32, n * g.gpu.slots), plan.suh[0 .. n * g.gpu.slots]);
            cb = queue.commandBuffer();
            e = cb.compute(.serial);
            try g.layer.experts(&K, &layer, &s, e, &c, n, plan.gu_jobs, plan.down_jobs, .{ .buf = s.y });
            e.end();
            try finish(cb);
            // the jobs the GPU writes from the route must give the host-built jobs' bytes
            const host_y = try a.dupe(f32, s.y.slice(f32, n * hid));
            defer a.free(host_y);
            cb = queue.commandBuffer();
            e = cb.compute(.serial);
            try g.layer.expertsGpu(&K, &layer, &s, e, &c, n, .{ .buf = s.y });
            e.end();
            try finish(cb);
            if (!std.mem.eql(u8, std.mem.sliceAsBytes(host_y), std.mem.sliceAsBytes(s.y.slice(f32, n * hid)))) gpu_jobs_differ += 1;
        }
        const skip = try a.alloc(bool, T);
        for (skip, pick_diff, topk_diff) |*k, p, t| k.* = p or t;
        stats[9].add(s.y.slice(f32, n * hid), refs.get("mlp_out").?.f, p0, hid, hid, n, skip);
        const h = s.h.slice(f32, n * hid);
        for (h, s.y.slice(f32, n * hid)) |*x, y| x.* += y;
        stats[10].add(h, refs.get("h_out").?.f, p0, hid, hid, n, skip);
    }
    var picks_off: usize = 0;
    for (pick_diff) |d| picks_off += @intFromBool(d);
    std.debug.print("layer {d}: {d} rows; {d} rows' top-k sets differ from the reference ({d} keys missing in all){s}; {d} rows pick other experts; {d} windows where GPU-written expert jobs differ from host-built ones\n", .{ L, T, topk_rows, missing_keys, if (ref_topk) " (reference top-k used)" else "", picks_off, gpu_jobs_differ });
    for (stats) |st| if (st.n > 0) st.print();
}
