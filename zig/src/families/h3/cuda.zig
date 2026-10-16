//! MiniMax H3 / FastH3's transformer blocks in int8 over the CUDA runtime; the host owns the rest of the pipeline.
const std = @import("std");
const cuda = @import("cuda");
const weights = @import("cuda_weights.zig");

pub const head_dim = 128;
pub const eps: f32 = 1e-5;
const slot_tile = 64;
/// Rows are padded to whole product tiles of this many.
const tile = 256;
const wide_group = 1024;
/// The six modulation tables of a block, in the order the checkpoint's projection splits them.
const shift_a = 0;
const scale_a = 1;
const gate_a = 2;
const shift_m = 3;
const scale_m = 4;
const gate_m = 5;

const image align(16) = @embedFile("fatbin_h3").*;
const kernel_names = [_][:0]const u8{ "h3_gemm", "h3_gemm_n256", "h3_gemm_swiglu", "h3_gemm_wide", "h3_quant_weight", "h3_norm_q8", "h3_gate_add", "h3_heads", "h3_pool", "h3_pool_quant", "h3_topk", "h3_mix_quant", "h3_quant_wide", "h3_attention" };

pub const stages = [_][]const u8{ "norms", "q k v gate", "heads", "routing", "attention", "gate mix", "attention out", "mlp in", "quantize", "mlp out" };

extern "c" fn clock_gettime(id: c_int, ts: *Timespec) c_int;
const Timespec = extern struct { sec: i64, nsec: i64 };

pub fn seconds() f64 {
    var ts: Timespec = undefined;
    _ = clock_gettime(1, &ts);
    return @as(f64, @floatFromInt(ts.sec)) + @as(f64, @floatFromInt(ts.nsec)) * 1e-9;
}

/// An int8 projection (N, K) and its scale per output channel.
pub const Quant = struct { w: cuda.DeviceBuffer, s: cuda.DeviceBuffer };

/// `norms` holds norm1, norm2 (hidden each), then the heads' q and k norms (128 each), bf16.
pub const Block = struct { qkvc: Quant, out: Quant, fc1: Quant, fc2: Quant, norms: cuda.DeviceBuffer };

/// What depends on the packed sequence: allocated when its rows or tiles change.
const Geometry = struct {
    rows: usize = 0,
    tiles: usize = 0,
    prefix: usize = 0,
    keep: usize = 0,
    buffers: [40]cuda.DeviceBuffer = undefined,
    used: usize = 0,
    y: u64 = 0,
    q8: u64 = 0,
    xs: u64 = 0,
    qkvc: u64 = 0,
    tq: u64 = 0,
    tqs: u64 = 0,
    tk: u64 = 0,
    tvt: u64 = 0,
    ks: u64 = 0,
    vs: u64 = 0,
    qp: u64 = 0,
    kp: u64 = 0,
    vp: u64 = 0,
    scores: u64 = 0,
    chosen: u64 = 0,
    coarse: u64 = 0,
    // the tiles' means as a sequence of their own, in groups of 64 tiles
    pq: u64 = 0,
    pqs: u64 = 0,
    pk: u64 = 0,
    pks: u64 = 0,
    pvt: u64 = 0,
    pvs: u64 = 0,
    group_sizes: u64 = 0,
    group_rows: u64 = 0,
    ya: u64 = 0,
    a8: u64 = 0,
    as: u64 = 0,
    wide: u64 = 0,
    w8: u64 = 0,
    wxs: u64 = 0,
    row_of: u64 = 0,
    all: u64 = 0,
};

pub const Model = struct {
    gpa: std.mem.Allocator,
    driver: cuda.Driver,
    ctx: cuda.Context,
    module: cuda.Module,
    functions: [kernel_names.len]cuda.Function,
    stream: cuda.Stream,
    hidden: usize,
    inner: usize,
    heads: usize,
    mlp: usize,
    blocks: []Block,
    geometry: Geometry = .{},
    profile: bool = false,
    spent: [stages.len]f64 = @splat(0),
    loaded_seconds: f64 = 0,
    /// The int8 projections came from the copy saved by an earlier load.
    from_cache: bool = false,

    fn function(self: *const Model, comptime name: []const u8) cuda.Function {
        inline for (kernel_names, 0..) |n, i| if (comptime std.mem.eql(u8, n, name)) return self.functions[i];
        @compileError("no kernel " ++ name);
    }

    pub fn launch(self: *const Model, comptime name: []const u8, grid: cuda.Dim3, threads: u32, shared: u32, args: *cuda.Args) !void {
        try cuda.launch.launch(self.function(name), .{ .grid = grid, .block = .{ .x = threads }, .shared = shared }, self.stream, args);
    }

    /// `path`: FastH3's transformer in one safetensors file (ComfyUI's layout: blocks.N.attn.qkv_proj.weight ...).
    pub fn load(gpa: std.mem.Allocator, path: [*:0]const u8) !*Model {
        const started = seconds();
        const self = try gpa.create(Model);
        errdefer gpa.destroy(self);
        self.gpa = gpa;
        self.geometry = .{};
        self.profile = false;
        self.spent = @splat(0);
        self.driver = try cuda.Driver.open();
        self.ctx = try cuda.Context.init(&self.driver, 0);
        const d = &self.driver;
        self.module = try cuda.Module.load(d, &image);
        inline for (kernel_names, 0..) |name, i| self.functions[i] = try self.module.function(name);
        // the products stage two K slices of their row and column tiles
        try self.function("h3_gemm_wide").allowDynamicShared(2 * (256 + 128) * 128);
        try self.function("h3_gemm_n256").allowDynamicShared(2 * (128 + 256) * 128);
        try self.function("h3_attention").allowDynamicShared(attention_shared);
        // the legacy default stream: work queued by the caller's framework comes first
        self.stream = .{ .d = d, .handle = null };

        self.from_cache = try weights.read(self, path);
        try self.ctx.synchronize();
        self.loaded_seconds = seconds() - started;
        return self;
    }

    const attention_shared = slot_tile * head_dim + slot_tile * slot_tile + 2 * 2 * slot_tile * head_dim;

    pub fn deinit(self: *Model) void {
        self.release();
        weights.free(self);
        self.module.unload();
        self.ctx.deinit();
        self.driver.close();
        self.gpa.destroy(self);
    }

    fn release(self: *Model) void {
        const g = &self.geometry;
        for (g.buffers[0..g.used]) |*b| b.free();
        g.* = .{};
    }

    fn buffer(self: *Model, bytes: usize, zero: bool) !u64 {
        const g = &self.geometry;
        g.buffers[g.used] = try cuda.DeviceBuffer.alloc(&self.driver, bytes);
        g.used += 1;
        if (zero) try g.buffers[g.used - 1].fill8(0, null);
        return g.buffers[g.used - 1].ptr;
    }

    fn padded(rows: usize) usize {
        return (rows + tile - 1) / tile * tile;
    }

    /// Buffers for `rows` packed rows in `tiles` routing tiles; `slot` is each row's padded slot on the device.
    fn prepare(self: *Model, rows: usize, tiles: usize, prefix: usize, keep: usize, slot: u64) !void {
        const g = &self.geometry;
        if (g.rows == rows and g.tiles == tiles and g.prefix == prefix and g.keep == keep) return;
        self.release();
        const hidden = self.hidden;
        const inner = self.inner;
        const heads = self.heads;
        const slots = tiles * slot_tile;
        const mp = padded(rows);
        g.y = try self.buffer(rows * hidden * 2, false);
        g.q8 = try self.buffer(mp * hidden, false);
        g.xs = try self.buffer(mp * 4, false);
        g.qkvc = try self.buffer(rows * 4 * inner * 2, false);
        // slots past a tile's real rows are never written: they stay zero
        g.tq = try self.buffer(heads * slots * head_dim, true);
        g.tqs = try self.buffer(heads * slots * 4, true);
        g.tk = try self.buffer(heads * slots * head_dim, true);
        g.tvt = try self.buffer(heads * slots * head_dim, true);
        g.ks = try self.buffer(heads * tiles * 4, false);
        g.vs = try self.buffer(heads * tiles * 4, false);
        g.qp = try self.buffer(heads * tiles * head_dim * 4, false);
        g.kp = try self.buffer(heads * tiles * head_dim * 4, false);
        g.vp = try self.buffer(heads * tiles * head_dim * 4, false);
        const groups = (tiles + slot_tile - 1) / slot_tile;
        const span = groups * slot_tile;
        g.scores = try self.buffer(heads * span * span * 4, false);
        g.chosen = try self.buffer(heads * @max(tiles - prefix, 1) * (prefix + keep) * 4, false);
        g.coarse = try self.buffer(tiles * inner * 2, false);
        g.pq = try self.buffer(heads * span * head_dim, false);
        g.pqs = try self.buffer(heads * span * 4, false);
        g.pk = try self.buffer(heads * span * head_dim, false);
        g.pks = try self.buffer(heads * groups * 4, false);
        g.pvt = try self.buffer(heads * span * head_dim, false);
        g.pvs = try self.buffer(heads * groups * 4, false);
        {
            // each group's real tiles, and the tile at each of its places (or -1)
            const sizes = try self.gpa.alloc(i32, groups);
            defer self.gpa.free(sizes);
            for (sizes, 0..) |*n, i| n.* = @intCast(@min(slot_tile, tiles - i * slot_tile));
            const places = try self.gpa.alloc(i32, span);
            defer self.gpa.free(places);
            for (places, 0..) |*t, i| t.* = if (i < tiles) @intCast(i) else -1;
            g.group_sizes = try self.buffer(groups * 4, false);
            try g.buffers[g.used - 1].upload(0, std.mem.sliceAsBytes(sizes));
            g.group_rows = try self.buffer(span * 4, false);
            try g.buffers[g.used - 1].upload(0, std.mem.sliceAsBytes(places));
        }
        g.ya = try self.buffer(rows * inner * 2, false);
        g.a8 = try self.buffer(mp * inner, false);
        g.as = try self.buffer(mp * 4, false);
        g.wide = try self.buffer(rows * self.mlp * 2, false);
        g.w8 = try self.buffer(mp * self.mlp, false);
        g.wxs = try self.buffer(mp * (self.mlp / wide_group) * 4, false);
        // the row at each slot (or -1), and the list of every tile
        const row_of = try self.gpa.alloc(i32, slots);
        defer self.gpa.free(row_of);
        @memset(row_of, -1);
        const slot_host = try self.gpa.alloc(i32, rows);
        defer self.gpa.free(slot_host);
        try self.driver.check(self.driver.api.cuMemcpyDtoH_v2(@ptrCast(slot_host.ptr), slot, rows * 4), "cuMemcpyDtoH");
        for (slot_host, 0..) |at, row| {
            if (at < 0 or @as(usize, @intCast(at)) >= slots) return error.BadGeometry;
            row_of[@intCast(at)] = @intCast(row);
        }
        const all = try self.gpa.alloc(i32, tiles);
        defer self.gpa.free(all);
        for (all, 0..) |*t, i| t.* = @intCast(i);
        g.row_of = try self.buffer(slots * 4, false);
        try g.buffers[g.used - 1].upload(0, std.mem.sliceAsBytes(row_of));
        g.all = try self.buffer(tiles * 4, false);
        try g.buffers[g.used - 1].upload(0, std.mem.sliceAsBytes(all));
        g.rows = rows;
        g.tiles = tiles;
        g.prefix = prefix;
        g.keep = keep;
    }

    fn lap(self: *Model, comptime stage: []const u8, mark: *f64) !void {
        if (!self.profile) return;
        try self.stream.synchronize();
        const now = seconds();
        inline for (stages, 0..) |s, i| if (comptime std.mem.eql(u8, s, stage)) {
            self.spent[i] += now - mark.*;
        };
        mark.* = now;
    }

    fn norm(self: *Model, x: u64, weight: u64, gates: u64, tab: u64, line: u64, gate: c_int, scale: c_int, shift: c_int) !void {
        const g = &self.geometry;
        var args: cuda.Args = .{};
        args.add(x);
        args.add(g.y);
        args.add(weight);
        args.add(gates);
        args.add(tab);
        args.add(line);
        args.add(g.q8);
        args.add(g.xs);
        args.add(@as(c_int, @intCast(g.rows)));
        args.add(@as(c_int, @intCast(self.hidden)));
        args.add(gate);
        args.add(scale);
        args.add(shift);
        args.add(eps);
        try self.launch("h3_norm_q8", .{ .x = @intCast(padded(g.rows)) }, 128, 0, &args);
    }

    /// d = x w in int8 to `n` outputs from `k` inputs; a CTA owns `bm` by `bn` and stages K slices of `kc` bytes.
    fn product(self: *Model, comptime kernel: []const u8, bm: usize, bn: usize, kc: usize, threads: u32, x: u64, scales: u64, w: Quant, d: u64, n: usize, k: usize, group: c_int) !void {
        var args: cuda.Args = .{};
        self.productArgs(&args, x, scales, w, d, n, k, group);
        try self.launch(kernel, .{ .x = @intCast((self.geometry.rows + bm - 1) / bm * (n / bn)) }, threads, @intCast(2 * (bm + bn) * kc), &args);
    }

    fn productArgs(self: *Model, args: *cuda.Args, x: u64, scales: u64, w: Quant, d: u64, n: usize, k: usize, group: c_int) void {
        args.add(x);
        args.add(scales);
        args.add(w.w.ptr);
        args.add(w.s.ptr);
        args.add(d);
        args.add(@as(c_int, @intCast(self.geometry.rows)));
        args.add(@as(c_int, @intCast(n)));
        args.add(@as(c_int, @intCast(k)));
        args.add(group);
    }

    /// softmax(q k) v over tiles of the rows' sequence: the prefix tiles and each video tile's chosen ones.
    fn attend(self: *Model, list: u64, sizes: u64, queries: usize, keys: usize, first: usize, per_query: bool) !void {
        const g = &self.geometry;
        var args: cuda.Args = .{};
        args.add(g.tq);
        args.add(g.tqs);
        args.add(g.tk);
        args.add(g.ks);
        args.add(g.tvt);
        args.add(g.vs);
        args.add(list);
        args.add(sizes);
        args.add(g.row_of);
        args.add(g.ya);
        args.add(@as(c_int, @intCast(g.tiles * slot_tile)));
        args.add(@as(c_int, @intCast(g.tiles)));
        args.add(@as(c_int, @intCast(self.heads)));
        args.add(@as(c_int, @intCast(queries)));
        args.add(@as(c_int, @intCast(keys)));
        args.add(@as(c_int, @intCast(first)));
        args.add(@as(c_int, @intFromBool(per_query)));
        args.add(@as(f32, 1.0 / @sqrt(@as(f32, head_dim))));
        args.add(@as(u64, 0));
        try self.launch("h3_attention", .{ .x = @intCast(queries), .y = @intCast(self.heads) }, 128, attention_shared, &args);
    }

    /// The attention kernel over the tiles' means: its output is the pooled branch, its scores pick the key tiles.
    fn attendPooled(self: *Model) !void {
        const g = &self.geometry;
        const groups = (g.tiles + slot_tile - 1) / slot_tile;
        var args: cuda.Args = .{};
        args.add(g.pq);
        args.add(g.pqs);
        args.add(g.pk);
        args.add(g.pks);
        args.add(g.pvt);
        args.add(g.pvs);
        args.add(g.all);
        args.add(g.group_sizes);
        args.add(g.group_rows);
        args.add(g.coarse);
        args.add(@as(c_int, @intCast(groups * slot_tile)));
        args.add(@as(c_int, @intCast(groups)));
        args.add(@as(c_int, @intCast(self.heads)));
        args.add(@as(c_int, @intCast(groups)));
        args.add(@as(c_int, @intCast(groups)));
        args.add(@as(c_int, 0));
        args.add(@as(c_int, 0));
        args.add(@as(f32, 1.0 / @sqrt(@as(f32, head_dim))));
        args.add(g.scores);
        try self.launch("h3_attention", .{ .x = @intCast(groups), .y = @intCast(self.heads) }, 128, attention_shared, &args);
    }

    pub const Inputs = extern struct {
        /// The packed stream, (rows, hidden) bf16, updated in place.
        x: u64,
        /// Every block's modulation tables: (blocks, lines, 6, hidden) float.
        tables: u64,
        /// Each row's line in the tables: (rows) int32.
        line: u64,
        /// Rotary cosines and sines: (rows, rot) float each.
        cos: u64,
        sin: u64,
        /// Each row's padded slot in tile order: (rows) int32. Each tile's real rows: (tiles) int32.
        slot: u64,
        sizes: u64,
        rows: u32,
        lines: u32,
        rot: u32,
        tiles: u32,
        /// Tiles ahead of the video's; they attend to and are seen by every tile.
        prefix_tiles: u32,
        /// Video tiles a video tile keeps; 0 for dense attention (every tile sees every tile, no pooled branch).
        keep: u32,
        /// Blocks to run, from the first; 0 for all.
        blocks: u32 = 0,
        /// When not 0: (rows, inner) bf16 that receives the last run block's attention output before its gate mix.
        attention_out: u64 = 0,
    };

    /// Every block over the stream, in place; returns once the GPU has finished.
    pub fn forward(self: *Model, in: *const Inputs) !f64 {
        const started = seconds();
        try self.ctx.makeCurrent();
        const rows: usize = in.rows;
        const tiles: usize = in.tiles;
        const dense = in.keep == 0 or in.prefix_tiles >= in.tiles;
        const prefix: usize = if (dense) tiles else in.prefix_tiles;
        const keep: usize = if (dense) 0 else in.keep;
        try self.prepare(rows, tiles, prefix, keep, in.slot);
        const g = &self.geometry;
        const hidden = self.hidden;
        const inner = self.inner;
        const heads = self.heads;
        const table_bytes = @as(usize, in.lines) * 6 * hidden * 4;
        const video_tiles = tiles - prefix;
        const count = if (in.blocks == 0) self.blocks.len else @min(in.blocks, self.blocks.len);
        var mark = seconds();
        for (self.blocks[0..count], 0..) |*b, index| {
            const tab = in.tables + index * table_bytes;
            // the block before left its MLP's rows in y: its gated add runs with this block's first norm
            if (index == 0) try self.norm(in.x, b.norms.ptr, tab, tab, in.line, -1, scale_a, shift_a) else try self.norm(in.x, b.norms.ptr, tab - table_bytes, tab, in.line, gate_m, scale_a, shift_a);
            try self.lap("norms", &mark);
            try self.product("h3_gemm_n256", 128, 256, 128, 256, g.q8, g.xs, b.qkvc, g.qkvc, 4 * inner, hidden, 16);
            try self.lap("q k v gate", &mark);
            {
                // q, k and v to int8 in tile order, the tile's k and v scales found on the way
                var args: cuda.Args = .{};
                args.add(g.qkvc);
                args.add(b.norms.ptr + hidden * 4);
                args.add(b.norms.ptr + hidden * 4 + head_dim * 2);
                args.add(in.cos);
                args.add(in.sin);
                args.add(g.row_of);
                args.add(g.tq);
                args.add(g.tqs);
                args.add(g.tk);
                args.add(g.tvt);
                args.add(g.ks);
                args.add(g.vs);
                args.add(@as(c_int, @intCast(heads)));
                args.add(@as(c_int, @intCast(4 * inner)));
                args.add(@as(c_int, @intCast(in.rot)));
                args.add(@as(c_int, @intCast(tiles * slot_tile)));
                args.add(@as(c_int, @intCast(tiles)));
                args.add(eps);
                try self.launch("h3_heads", .{ .x = @intCast(tiles) }, 256, 0, &args);
            }
            try self.lap("heads", &mark);
            if (dense) {
                try self.attend(g.all, in.sizes, tiles, tiles, 0, false);
                try self.lap("attention", &mark);
            } else {
                var args: cuda.Args = .{};
                args.add(g.tq);
                args.add(g.tqs);
                args.add(g.tk);
                args.add(g.ks);
                args.add(g.tvt);
                args.add(g.vs);
                args.add(in.sizes);
                args.add(g.qp);
                args.add(g.kp);
                args.add(g.vp);
                args.add(@as(c_int, @intCast(tiles * slot_tile)));
                args.add(@as(c_int, @intCast(tiles)));
                try self.launch("h3_pool", .{ .x = @intCast(tiles), .y = @intCast(heads) }, 32, 0, &args);
                const groups = (tiles + slot_tile - 1) / slot_tile;
                args = .{};
                args.add(g.qp);
                args.add(g.kp);
                args.add(g.vp);
                args.add(g.pq);
                args.add(g.pqs);
                args.add(g.pk);
                args.add(g.pks);
                args.add(g.pvt);
                args.add(g.pvs);
                args.add(@as(c_int, @intCast(tiles)));
                args.add(@as(c_int, @intCast(groups)));
                try self.launch("h3_pool_quant", .{ .x = @intCast(groups), .y = @intCast(heads) }, 64, 0, &args);
                try self.attendPooled();
                args = .{};
                args.add(g.scores);
                args.add(g.chosen);
                args.add(@as(c_int, @intCast(tiles)));
                args.add(@as(c_int, @intCast(groups * slot_tile)));
                args.add(@as(c_int, @intCast(prefix)));
                args.add(@as(c_int, @intCast(keep)));
                try self.launch("h3_topk", .{ .x = @intCast(video_tiles), .y = @intCast(heads) }, 32, 0, &args);
                try self.lap("routing", &mark);
                // the prefix's query tiles see every tile; a video tile sees the prefix and its chosen tiles
                try self.attend(g.all, in.sizes, prefix, tiles, 0, false);
                try self.attend(g.chosen, in.sizes, video_tiles, prefix + keep, prefix, true);
                try self.lap("attention", &mark);
            }
            if (in.attention_out != 0 and index == count - 1) try self.driver.check(self.driver.api.cuMemcpyDtoD_v2(in.attention_out, g.ya, rows * inner * 2), "cuMemcpyDtoD");
            {
                // the pooled branch gated in and the rows rounded to int8, in one pass
                var args: cuda.Args = .{};
                args.add(g.ya);
                args.add(g.qkvc + 3 * inner * 2);
                args.add(if (dense) @as(u64, 0) else g.coarse);
                args.add(in.slot);
                args.add(g.a8);
                args.add(g.as);
                args.add(@as(c_int, @intCast(rows)));
                args.add(@as(c_int, @intCast(inner)));
                args.add(@as(c_int, @intCast(4 * inner)));
                try self.launch("h3_mix_quant", .{ .x = @intCast(padded(rows)) }, 128, 0, &args);
            }
            try self.lap("gate mix", &mark);
            try self.product("h3_gemm", 128, 128, 64, 128, g.a8, g.as, b.out, g.y, hidden, inner, 12);
            try self.lap("attention out", &mark);
            try self.norm(in.x, b.norms.ptr + hidden * 2, tab, tab, in.line, gate_a, scale_m, shift_m);
            try self.lap("norms", &mark);
            try self.product("h3_gemm_swiglu", 128, 128, 64, 128, g.q8, g.xs, b.fc1, g.wide, 2 * self.mlp, hidden, 16);
            try self.lap("mlp in", &mark);
            {
                var args: cuda.Args = .{};
                args.add(g.wide);
                args.add(g.w8);
                args.add(g.wxs);
                args.add(@as(c_int, @intCast(rows)));
                args.add(@as(c_int, @intCast(self.mlp)));
                try self.launch("h3_quant_wide", .{ .x = @intCast(padded(rows)) }, 128, 0, &args);
            }
            try self.lap("quantize", &mark);
            try self.product("h3_gemm_wide", 256, 128, 128, 256, g.w8, g.wxs, b.fc2, g.y, hidden, self.mlp, 4);
            try self.lap("mlp out", &mark);
        }
        {
            var args: cuda.Args = .{};
            args.add(in.x);
            args.add(g.y);
            args.add(in.tables + (count - 1) * table_bytes);
            args.add(in.line);
            args.add(@as(c_int, @intCast(hidden)));
            args.add(@as(c_int, gate_m));
            try self.launch("h3_gate_add", .{ .x = @intCast(rows) }, 128, 0, &args);
        }
        try self.stream.synchronize();
        return seconds() - started;
    }
};
