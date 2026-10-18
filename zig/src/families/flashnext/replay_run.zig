//! Flash Next replay internals split along ownership and dispatch boundaries.
const fz = @import("replay.zig");
const std = @import("std");
const platform = @import("sys");
const mtl = @import("metal");
const split = fz.split;
const dense = fz.dense;
const lane_w = fz.lane_w;
const variants = fz.variants;
const gdn_step = fz.gdn_step;
const Tp2 = fz.Tp2;
const opts = fz.opts;
const WindowSettings = fz.WindowSettings;
const Buf = fz.Buf;
const Entry = fz.Entry;
const Variant = fz.Variant;
const Site = fz.Site;
const readAll = fz.readAll;
const Select = fz.Select;
const GSelect = fz.GSelect;
const Lane = fz.Lane;

pub const Run = struct {
    window: WindowSettings = .{},
    arena: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    index: std.StringHashMapUnmanaged(Entry) = .empty,
    variants: std.StringHashMapUnmanaged(*Variant) = .empty,
    roles: std.StringHashMapUnmanaged(Site) = .empty,
    shapes: std.StringHashMapUnmanaged(mtl.Buffer) = .empty,
    held: std.ArrayList(mtl.Buffer) = .empty, // what buffer, load and group made: a host's residency set holds them
    follow: ?mtl.ResidencySet.Follow = null, // set once the host's set is built: later buffers join it
    loaded: usize = 0,
    rows: usize = 1,
    fused_xsum: bool = false,
    serial: bool = false,
    enc: mtl.ComputeEncoder = undefined,
    rotary_pipe: ?mtl.Pipeline = null,
    rotary_enabled: bool = false,
    rotary_buffer: ?mtl.Buffer = null,
    kv_pipe: mtl.Pipeline = undefined,
    argmax_pipe: mtl.Pipeline = undefined,
    add_pipe: mtl.Pipeline = undefined,
    argids_pipe: mtl.Pipeline = undefined,
    pleids_pipe: mtl.Pipeline = undefined,
    copy_pipe: mtl.Pipeline = undefined,
    accept_pipe: mtl.Pipeline = undefined,
    lookup_pipe: mtl.Pipeline = undefined,
    force_pipe: mtl.Pipeline = undefined,
    span_pipe: mtl.Pipeline = undefined,
    positions_pipe: mtl.Pipeline = undefined,
    draw_pipe: mtl.Pipeline = undefined, // tf_sample_full: a sampled reply's draws (sample.zig)
    take_pipe: mtl.Pipeline = undefined,
    gpu_round: bool = false,
    ar: Buf = undefined,
    probe: ?Buf = null,
    lg_probe: ?Buf = null, // FZ_DBENCH: every layer's router logits, MAXR * 513 floats a layer
    xnew: bool = false,
    xgu_pipe: mtl.Pipeline = undefined,
    xdown_pipe: mtl.Pipeline = undefined,
    dense: bool = false,
    dense_target: bool = false,
    skip: u32 = 0,
    hcskip: u32 = 0, // FZ_HCSKIP: hyper-connection knock-outs by kernel (1 norms, 2 downs, 4 ups)
    hc_up: bool = false, // fz_hc_up for the hyper-connection's up projection at 3-8 rows (the recorded kernels' bits)
    hc_up_pipe: mtl.Pipeline = undefined,
    split: bool = false,
    gdn_step: bool = false,
    hc_mma: bool = false,
    event: mtl.SharedEvent = undefined,
    event_value: u64 = 0,
    dense8_pipe: mtl.Pipeline = undefined,
    grouped: bool = false,
    gskip: u32 = 0,
    xpack: bool = false,
    xfused: bool = false,
    xsx: bool = false,
    copy: bool = false,
    xgu_sx_pipe: mtl.Pipeline = undefined,
    xdown_sx_pipe: mtl.Pipeline = undefined,
    prefetch: bool = false,
    touch_pipe: mtl.Pipeline = undefined,
    sink: Buf = undefined,
    xfused_pipe: mtl.Pipeline = undefined,
    xpart: Buf = undefined,
    xdone: Buf = undefined,
    repack_w: mtl.Pipeline = undefined,
    repack_s: mtl.Pipeline = undefined,
    route_pipe: mtl.Pipeline = undefined,
    router_pipe: mtl.Pipeline = undefined,
    ggu_pipe: mtl.Pipeline = undefined,
    gdown_pipe: mtl.Pipeline = undefined,
    ul: Buf = undefined,
    dense16_pipe: mtl.Pipeline = undefined,
    tp_gdn: std.AutoHashMapUnmanaged(*Variant, mtl.Pipeline) = .empty, // TP: DeltaNet steps from value head 24
    tp_lane: std.AutoHashMapUnmanaged(u64, mtl.Pipeline) = .empty, // TP: recorded lane kernels over tile maps
    lane_new: bool = false, // the target's lane projections on core lane (dense.zig): the recorded bits, weights read ahead
    lane_pf: usize = 1, // groups a simdgroup reads ahead in core lane
    lane_pipes: std.AutoHashMapUnmanaged(u64, mtl.Pipeline) = .empty,
    wide_pipes: lane_w.Cache = .empty, // lane products at other widths than 6/32 (lane_w.zig)
    lane_shape: std.AutoHashMapUnmanaged(*Variant, [3]usize) = .empty, // recorded lane kernels' N, K, SK
    gdn_pipe: ?mtl.Pipeline = null, // the DeltaNet window step on fz_gdn (gdn.zig) when set
    gdn_kept: ?mtl.Pipeline = null, // GPU-side rounds: fz_gdn keeping one state a layer, kept rows replayed (gdn.zig)
    xnew_header: []const u8 = "",
    tp: ?*Tp2 = null, // TP=2 across two Macs (tp.zig): the target layers' experts split by id, outputs exchanged
    tp_layer: bool = false, // the experts being encoded are a target layer's (the MTP head keeps all of its own)
    sel: ?Select = null, // long contexts: the indexer's pool, scores and selection
    gsel: ?GSelect = null, // the same in GPU-side rounds
    sel_meta_pipe: mtl.Pipeline = undefined,
    pool_abs_pipe: mtl.Pipeline = undefined,
    ngram: []const u8 = "shard_", // how the checkpoint spells its n-gram table shards (index.ngramSpelling)

    load_limit: u64 = 0,
    objects: @import("owned.zig").Owner(mtl) = .{},
    files: std.ArrayList(c_int) = .empty,
    maps: std.ArrayList(mtl.MappedFile) = .empty,
    rows_cleanup: ?struct { ctx: *anyopaque, call: *const fn (*anyopaque) void } = null,
    closed: bool = false,

    pub const deinit = @import("run_cleanup.zig").drop;

    pub fn ownResource(r: *Run, value: anytype) !@TypeOf(value) {
        return r.objects.keep(r.arena, value);
    }

    pub fn mapFile(r: *Run, path: [:0]const u8) !mtl.MappedFile {
        const file = try mtl.MappedFile.open(path);
        errdefer file.deinit();
        try r.maps.append(r.arena, file);
        return file;
    }

    pub fn allocate(r: *Run, n: usize) !mtl.Buffer {
        return r.allocateOptions(n, opts);
    }

    pub fn checkGrowth(r: *const Run, n: usize) !void {
        if (r.load_limit != 0 and !@import("flashnext_memory").growingFits(r.device.allocated(), n, r.load_limit)) {
            std.log.warn("flash next: {d} more buffer bytes would cross the load ceiling and runtime reserve", .{n});
            return error.FlashNextMemoryShortfall;
        }
    }

    pub fn allocateOptions(r: *Run, n: usize, options: usize) !mtl.Buffer {
        try r.checkGrowth(n);
        return r.device.buffer(n, options);
    }

    pub fn buffer(r: *Run, len: usize) !mtl.Buffer {
        const n = @max(len, 64);
        const b = try r.allocate(n);
        errdefer b.deinit();
        @memset(b.contents()[0..n], 0);
        try r.keep(b);
        return b;
    }

    /// `b` on the held list, and in the host's residency set when one was built before it.
    pub fn keep(r: *Run, b: mtl.Buffer) !void {
        try r.held.append(r.arena, b);
        if (r.follow) |f| f.join(f.ctx, b);
    }

    /// A buffer `buffer`, `load` or `group` made, freed by its caller: off the held list and out of the set first.
    pub fn release(r: *Run, b: mtl.Buffer) void {
        for (r.held.items, 0..) |x, i| if (x.id == b.id) {
            _ = r.held.swapRemove(i);
            break;
        };
        if (r.follow) |f| f.leave(f.ctx, b);
        b.deinit();
    }

    /// Every tensor of a safetensors file in the index, at its absolute offset.
    pub fn indexFile(r: *Run, path: [:0]const u8) !void {
        const fd = platform.c.open(path, .{ .ACCMODE = .RDONLY });
        if (fd < 0) return error.OpenFailed;
        r.files.append(r.arena, fd) catch |err| {
            _ = platform.c.close(fd);
            return err;
        };
        var head: [8]u8 = undefined;
        try readAll(fd, &head, 0);
        const n = std.mem.readInt(u64, &head, .little);
        const text = try r.arena.alloc(u8, n);
        try readAll(fd, text, 8);
        const parsed = try std.json.parseFromSliceLeaky(std.json.Value, r.arena, text, .{});
        var it = parsed.object.iterator();
        while (it.next()) |kv| {
            if (std.mem.eql(u8, kv.key_ptr.*, "__metadata__")) continue;
            const offs = kv.value_ptr.object.get("data_offsets").?.array.items;
            const lo: usize = @intCast(offs[0].integer);
            const hi: usize = @intCast(offs[1].integer);
            try r.index.put(r.arena, kv.key_ptr.*, .{ .fd = fd, .at = 8 + n + lo, .len = hi - lo });
        }
    }

    pub fn entry(r: *Run, name: []const u8) !Entry {
        return r.index.get(name) orelse {
            std.log.err("no tensor {s}", .{name});
            return error.MissingTensor;
        };
    }

    pub fn load(r: *Run, name: []const u8) !Buf {
        const e = try r.entry(name);
        const b = try r.allocate(@max(e.len, 64));
        errdefer b.deinit();
        try readAll(e.fd, b.contents()[0..e.len], e.at);
        r.loaded += e.len;
        try r.keep(b);
        return .{ .b = b };
    }

    pub fn loadf(r: *Run, comptime fmt: []const u8, args: anytype) !Buf {
        var name: [160]u8 = undefined;
        return r.load(try std.fmt.bufPrint(&name, fmt, args));
    }

    /// Shards `first..first+count` of `suffix` concatenated into one buffer (PleTables' group).
    pub fn group(r: *Run, first: usize, count: usize, suffix: []const u8) !Buf {
        var total: usize = 0;
        var name: [160]u8 = undefined;
        const fmt = "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.{s}{d}.{s}";
        for (first..first + count) |s| total += (try r.entry(try std.fmt.bufPrint(&name, fmt, .{ r.ngram, s, suffix }))).len;
        const b = try r.allocate(total);
        errdefer b.deinit();
        var at: usize = 0;
        for (first..first + count) |s| {
            const e = try r.entry(try std.fmt.bufPrint(&name, fmt, .{ r.ngram, s, suffix }));
            try readAll(e.fd, b.contents()[at .. at + e.len], e.at);
            at += e.len;
        }
        r.loaded += total;
        try r.keep(b);
        return .{ .b = b };
    }

    /// Checked-in kernels and the role table serve with no dump, each variant compiled from its embedded source.
    pub const compileChecked = @import("replay_compile.zig").compileChecked;

    /// The dump path: every variant and role from the recorded plan, then the shared pipelines.
    pub const compile = @import("replay_compile.zig").compile;

    /// The glue and select pipelines both compile paths share.
    pub const compileGlue = @import("replay_compile.zig").compileGlue;

    /// fz_xgu and friends after the gate/up kernel's header (simd_topk, bsilu): the recorded or checked-in text.
    pub const compileXnew = @import("replay_compile.zig").compileXnew;

    /// fz_hc_up after the recorded qa_hc_down_row header, refused unless every recorded qa_hc_up variant has the...
    pub const compileHc = @import("replay_compile.zig").compileHc;

    /// A variant's source: the embedded text when the run is checked-in, else its recorded file.
    pub const variantText = @import("replay_compile.zig").variantText;

    /// A hyper-connection's down then up projection at `as_rows` from normed streams `hn` (sums of squares in...
    pub const hcProject = @import("replay_compile.zig").hcProject;

    pub const strings = @import("replay_compile.zig").strings;

    pub const size3 = @import("replay_compile.zig").size3;

    /// FZ_XPACK: an expert set's rows (gate, up, shared gate, shared up, down, shared down) to the packed layout.
    pub const repack = @import("replay_compile.zig").repack;

    /// FZ_PREFETCH: stream `b` into the cache beside the next launch (no barrier between them).
    pub fn touch(r: *Run, b: Buf) void {
        if (!r.prefetch) return;
        const n: u32 = @intCast((b.b.length() - b.off) / 16);
        r.enc.setPipeline(r.touch_pipe);
        r.enc.setBuffer(b.b, b.off, 0);
        r.enc.setBuffer(r.sink.b, 0, 1);
        r.enc.setBytes(std.mem.asBytes(&n), 2);
        r.enc.dispatchThreads(mtl.Size.of(256 * 64, 1, 1), mtl.Size.of(256, 1, 1));
    }

    /// fz_copy_kept: `layers` copies of `n` words from row (keep + base) of src (row and layer strides in words).
    pub fn copyKept(r: *Run, src: Buf, dst: Buf, n: usize, row: usize, src_stride: usize, dst_stride: usize, layers: usize, base: i32) void {
        r.enc.setPipeline(r.copy_pipe);
        r.enc.setBuffer(src.b, src.off, 0);
        r.enc.setBuffer(dst.b, dst.off, 1);
        r.enc.setBuffer(r.ar.b, r.ar.off, 2);
        const p = [4]u32{ @intCast(n), @intCast(row), @intCast(src_stride), @intCast(dst_stride) };
        r.enc.setBytes(std.mem.asBytes(&p), 3);
        r.enc.setBytes(std.mem.asBytes(&base), 4);
        r.enc.dispatchThreads(mtl.Size.of(n, layers, 1), mtl.Size.of(256, 1, 1));
        if (!r.serial) r.enc.barrier();
    }
    /// y = x W for `rows` rows of K inputs through fz_dense (blocks of 8 rows; 16 K-slices for narrow outputs).
    pub fn denseTiles(r: *Run, x: Buf, k: usize, l: Lane.Six, y: Buf, t0: usize, tn: usize) void {
        if (tn == 0) return;
        const n = l.wq.b.length() * 4 / (3 * k);
        const narrow = n <= 4096;
        const dims = [4]u32{ 1, @intCast(n), @intCast(k), @intCast(t0) };
        r.enc.setPipeline(if (narrow) r.dense16_pipe else r.dense8_pipe);
        r.enc.setBuffer(x.b, x.off, 0);
        r.enc.setBuffer(l.wq.b, l.wq.off, 1);
        r.enc.setBuffer(l.sbt.b, l.sbt.off, 2);
        r.enc.setBuffer(y.b, y.off, 3);
        r.enc.setBytes(std.mem.asBytes(&dims), 4);
        const sk: usize = if (narrow) 16 else 8;
        r.enc.dispatchThreads(mtl.Size.of(32 * sk * tn, 1, 1), mtl.Size.of(32 * sk, 1, 1));
        if (!r.serial) r.enc.barrier();
    }

    pub fn denseRows(r: *Run, x: Buf, k: usize, l: Lane.Six, rows: usize, y: Buf) void {
        const n = l.wq.b.length() * 4 / (3 * k);
        const narrow = n <= 4096;
        var at: usize = 0;
        while (at < rows) : (at += 8) {
            const rr = @min(8, rows - at);
            const dims = [4]u32{ @intCast(rr), @intCast(n), @intCast(k), 0 };
            r.enc.setPipeline(if (narrow) r.dense16_pipe else r.dense8_pipe);
            r.enc.setBuffer(x.b, x.off + at * k * 2, 0);
            r.enc.setBuffer(l.wq.b, l.wq.off, 1);
            r.enc.setBuffer(l.sbt.b, l.sbt.off, 2);
            r.enc.setBuffer(y.b, y.off + at * n * 2, 3);
            r.enc.setBytes(std.mem.asBytes(&dims), 4);
            const sk: usize = if (narrow) 16 else 8;
            r.enc.dispatchThreads(mtl.Size.of(32 * sk * (n / 32), 1, 1), mtl.Size.of(32 * sk, 1, 1));
            if (!r.serial) r.enc.barrier();
        }
    }
    /// The router's logits for up to `rows` rows (the count itself read from `rows_buf`): fz_router, the recorded...
    pub fn router(r: *Run, role: []const u8, x: Buf, w: Buf, rows_buf: Buf, out: Buf, rows: usize) !void {
        if (!r.xnew) return r.call(role, &.{ x, w, rows_buf }, &.{out});
        if (r.skip & class(role) != 0) return;
        r.enc.setPipeline(r.router_pipe);
        for ([_]Buf{ x, w, rows_buf, out }, 0..) |b, j| r.enc.setBuffer(b.b, b.off, j);
        r.enc.dispatchGroups(mtl.Size.of((rows * 513 + 7) / 8, 1, 1), mtl.Size.of(256, 1, 1));
        if (!r.serial) r.enc.barrier();
    }

    /// The MoE's routed and shared experts for `rows` rows: gate/up (with routing) then down.
    pub fn experts(r: *Run, gu_role: []const u8, down_role: []const u8, x: Buf, lg: Buf, e: [18]Buf, act: Buf, pick: Buf, wts: Buf, rows_buf: Buf, y: Buf) !void {
        if (r.skip & (1 << 2) != 0) return;
        if (!r.xnew) {
            try r.call(gu_role, &.{ x, lg, e[0], e[1], e[2], e[3], e[4], e[5], e[6], e[7], e[8], e[9], e[10], e[11] }, &.{ act, pick, wts });
            try r.call(down_role, &.{ act, pick, e[12], e[13], e[14], e[15], e[16], e[17], rows_buf }, &.{y});
            return;
        }
        if (r.xfused) { // gate/up and down in one launch
            r.enc.setPipeline(r.xfused_pipe);
            const fb = [_]Buf{ x, lg, e[0], e[1], e[2], e[3], e[4], e[5], e[6], e[7], e[8], e[9], e[10], e[11], e[12], e[13], e[14], e[15], e[16], e[17], pick, wts, y, r.xpart, r.xdone };
            for (fb, 0..) |b, j| r.enc.setBuffer(b.b, b.off, j);
            r.enc.dispatchThreads(mtl.Size.of(512 * 4, r.rows * 11, 1), mtl.Size.of(512, 1, 1));
            if (!r.serial) r.enc.barrier();
            return;
        }
        if (r.grouped and r.rows > 1) { // each distinct expert read once for every row that picked it
            if (r.gskip & 1 == 0) {
                r.enc.setPipeline(r.route_pipe);
                for ([_]Buf{ lg, rows_buf, pick, wts, r.ul }, 0..) |b, j| r.enc.setBuffer(b.b, b.off, j);
                r.enc.dispatchThreads(mtl.Size.of(32 * r.rows, 1, 1), mtl.Size.of(32 * r.rows, 1, 1));
                if (!r.serial) r.enc.barrier();
            }
            if (r.gskip & 2 != 0) return;
            r.enc.setPipeline(r.ggu_pipe);
            const ggu = [_]Buf{ x, r.ul, e[0], e[1], e[2], e[3], e[4], e[5], e[6], e[7], e[8], e[9], e[10], e[11], act, rows_buf };
            for (ggu, 0..) |b, j| r.enc.setBuffer(b.b, b.off, j);
            r.enc.dispatchThreads(mtl.Size.of(128, 80, r.rows * 10 + 1), mtl.Size.of(128, 1, 1));
            if (!r.serial) r.enc.barrier();
            if (r.gskip & 4 != 0) return;
            r.enc.setPipeline(r.gdown_pipe);
            const gdn = [_]Buf{ act, r.ul, e[12], e[13], e[14], e[15], e[16], e[17], rows_buf, y };
            for (gdn, 0..) |b, j| r.enc.setBuffer(b.b, b.off, j);
            r.enc.dispatchThreads(mtl.Size.of(128, 80, r.rows * 10 + 1), mtl.Size.of(128, 1, 1));
            if (!r.serial) r.enc.barrier();
            return;
        }
        const rows_own = if (r.tp != null and r.tp_layer) r.tp.?.ownRows(r.rows) else [2]usize{ 0, r.rows };
        if (rows_own[1] == rows_own[0]) return; // TP: a one-row window's experts are rank 0's
        const own = [4]u32{ @intCast(rows_own[0]), @intCast(rows_own[1]), 0, 0 };
        const pairs = (rows_own[1] - rows_own[0]) * 11;
        if (r.tp != null and r.tp_layer and r.xsx) return error.TpNeedsPlainExperts;
        r.enc.setPipeline(if (r.xsx) r.xgu_sx_pipe else r.xgu_pipe);
        const gu = [_]Buf{ x, lg, e[0], e[1], e[2], e[3], e[4], e[5], e[6], e[7], e[8], e[9], e[10], e[11], act, pick, wts };
        for (gu, 0..) |b, j| r.enc.setBuffer(b.b, b.off, j);
        r.enc.setBytes(std.mem.asBytes(&own), 17);
        r.enc.dispatchThreads(mtl.Size.of(128, 80, pairs), mtl.Size.of(128, 1, 1));
        if (!r.serial) r.enc.barrier();
        r.enc.setPipeline(if (r.xsx) r.xdown_sx_pipe else r.xdown_pipe);
        const dn = [_]Buf{ act, pick, e[12], e[13], e[14], e[15], e[16], e[17], rows_buf, y };
        for (dn, 0..) |b, j| r.enc.setBuffer(b.b, b.off, j);
        r.enc.setBytes(std.mem.asBytes(&own), 10);
        r.enc.dispatchThreads(mtl.Size.of(128, 80, pairs), mtl.Size.of(128, 1, 1));
        if (!r.serial) r.enc.barrier();
    }

    /// One launch of `role` at the current row count, inputs and outputs in the variant's order.
    pub fn class(role: []const u8) u32 {
        const names = [_][]const u8{ "hc_", "lane_qmm", "expert", "router", "gdn", "attn", "ple", "head" };
        if (std.mem.indexOf(u8, role, "@head") != null) return 1 << 7;
        for (names, 0..) |n, i| if (std.mem.indexOf(u8, role, n) != null) return @as(u32, 1) << @intCast(i);
        return 0;
    }

    /// `role` at `rows` rows: the recorded one-row launch with its row dimension (the one that grows from one row to...
    pub fn callRows(r: *Run, role: []const u8, rows: usize, ins: []const Buf, outs: []const Buf, v: ?*Variant) !void {
        var k1: [96]u8 = undefined;
        var k2: [96]u8 = undefined;
        const s1 = r.roles.get(try std.fmt.bufPrint(&k1, "{s}|1", .{role})) orelse return error.NoSite;
        const s2 = r.roles.get(try std.fmt.bufPrint(&k2, "{s}|2", .{role})) orelse return error.NoSite;
        var grid = s1.grid;
        if (s2.grid.height != s1.grid.height) grid.height = rows * s1.grid.height else if (s2.grid.depth != s1.grid.depth) grid.depth = rows * s1.grid.depth;
        try r.bindV(v orelse s1.v, ins, outs);
        r.enc.dispatchThreads(grid, s1.tg);
        if (!r.serial) r.enc.barrier();
    }

    pub fn bindV(r: *Run, v: *Variant, ins: []const Buf, outs: []const Buf) !void {
        if (ins.len != v.inputs.len or outs.len != v.outputs.len) return error.Arity;
        r.enc.setPipeline(v.pipe);
        var at: usize = 0;
        for (v.inputs, ins) |input, b| {
            r.enc.setBuffer(b.b, b.off, at);
            at += 1;
            for ([_][]const u8{ "_shape", "_strides", "_ndim" }) |suffix| {
                for (v.meta) |mm| {
                    if (mm.len == input.len + suffix.len and std.mem.startsWith(u8, mm, input) and std.mem.endsWith(u8, mm, suffix)) {
                        r.enc.setBuffer(r.shapes.get(mm) orelse return error.NoShape, 0, at);
                        at += 1;
                    }
                }
            }
        }
        for (outs) |b| {
            r.enc.setBuffer(b.b, b.off, at);
            at += 1;
        }
    }

    pub fn call(r: *Run, role: []const u8, ins: []const Buf, outs: []const Buf) !void {
        return r.callAs(role, r.rows, ins, outs);
    }

    /// `role` launched as recorded at `as_rows` rows (its kernel reads the row count at run time).
    pub fn callAs(r: *Run, role: []const u8, as_rows: usize, ins: []const Buf, outs: []const Buf) !void {
        if (r.skip & class(role) != 0) return;
        if (r.hcskip != 0) for ([_][]const u8{ "hc_norm", "hc_down", "hc_up" }, 0..) |n, i| {
            if (r.hcskip & (@as(u32, 1) << @intCast(i)) != 0 and std.mem.indexOf(u8, role, n) != null) return;
        };
        var key: [96]u8 = undefined;
        const name = try std.fmt.bufPrint(&key, "{s}|{d}", .{ role, as_rows });
        const s = r.roles.get(name) orelse {
            std.log.err("no launch site {s}", .{name});
            return error.NoSite;
        };
        const v = s.v;
        if (ins.len != v.inputs.len or outs.len != v.outputs.len) {
            std.log.err("{s}: {d} inputs and {d} outputs given, the kernel takes {d} and {d}", .{ name, ins.len, outs.len, v.inputs.len, v.outputs.len });
            return error.Arity;
        }
        r.enc.setPipeline(v.pipe);
        var at: usize = 0;
        for (v.inputs, ins) |input, b| {
            r.enc.setBuffer(b.b, b.off, at);
            at += 1;
            for ([_][]const u8{ "_shape", "_strides", "_ndim" }) |suffix| {
                for (v.meta) |m| {
                    if (m.len == input.len + suffix.len and std.mem.startsWith(u8, m, input) and std.mem.endsWith(u8, m, suffix)) {
                        r.enc.setBuffer(r.shapes.get(m) orelse return error.NoShape, 0, at);
                        at += 1;
                    }
                }
            }
        }
        for (outs) |b| {
            r.enc.setBuffer(b.b, b.off, at);
            at += 1;
        }
        r.enc.dispatchThreads(s.grid, s.tg);
        if (!r.serial) r.enc.barrier();
    }
};
