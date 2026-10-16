//! MiniMax H3 / FastH3's transformer blocks over the CUDA runtime in the checkpoint's bf16; the host owns the rest.
const std = @import("std");
const cuda = @import("cuda");
const weights = @import("cuda_weights.zig");
pub const geometry = @import("cuda_geometry.zig");
pub const tp = @import("cuda_tp.zig");
const Gemms = @import("cuda_gemm.zig").Gemms;

pub const head_dim = 128;
pub const eps: f32 = 1e-5;
pub const slot_tile = 64;
/// The six modulation tables of a block, in the order the checkpoint's projection splits them.
const shift_a = 0;
const scale_a = 1;
const gate_a = 2;
const shift_m = 3;
const scale_m = 4;
const gate_m = 5;

const image align(16) = @embedFile("fatbin_h3").*;
const kernel_names = [_][:0]const u8{ "h3_norm", "h3_gate_add", "h3_heads", "h3_pool", "h3_tile_scores", "h3_topk", "h3_coarse", "h3_gate_mix", "h3_swiglu", "h3_attention", "h3_spans" };

pub const stages = [_][]const u8{ "norms", "q k v gate", "heads", "routing", "attention", "gate mix", "attention out", "mlp in", "mlp out", "exchange" };

/// h3_attention's dynamic shared bytes: a query, a key and a value tile in bf16.
const attention_shared = 3 * slot_tile * head_dim * 2;

extern "c" fn clock_gettime(id: c_int, ts: *Timespec) c_int;
const Timespec = extern struct { sec: i64, nsec: i64 };

pub fn seconds() f64 {
    var ts: Timespec = undefined;
    _ = clock_gettime(1, &ts);
    return @as(f64, @floatFromInt(ts.sec)) + @as(f64, @floatFromInt(ts.nsec)) * 1e-9;
}

/// A block's bf16 projections (N, K) as stored, `qkvc` qkv_proj then the gate; `norms`: norm1, norm2, q and k norms.
pub const Block = struct { qkvc: cuda.DeviceBuffer, out: cuda.DeviceBuffer, fc1: cuda.DeviceBuffer, fc2: cuda.DeviceBuffer, norms: cuda.DeviceBuffer };

pub const Model = struct {
    gpa: std.mem.Allocator,
    driver: cuda.Driver,
    ctx: cuda.Context,
    module: cuda.Module,
    functions: [kernel_names.len]cuda.Function,
    stream: cuda.Stream,
    gemms: Gemms,
    hidden: usize,
    inner: usize,
    heads: usize,
    mlp: usize,
    blocks: []Block,
    geometry: geometry.Geometry = .{},
    profile: bool = false,
    spent: [stages.len]f64 = @splat(0),
    loaded_seconds: f64 = 0,
    /// The sequence-parallel world this rank is in, if any.
    tp: ?tp.Tp = null,
    /// A host's inputs copied to the device (tp.forwardHost).
    inputs: [7]?cuda.DeviceBuffer = @splat(null),

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
        self.tp = null;
        self.inputs = @splat(null);
        self.driver = try cuda.Driver.open();
        errdefer self.driver.close();
        self.ctx = try cuda.Context.init(&self.driver, 0);
        errdefer self.ctx.deinit();
        const d = &self.driver;
        self.module = try cuda.Module.load(d, &image);
        errdefer self.module.unload();
        inline for (kernel_names, 0..) |name, i| self.functions[i] = try self.module.function(name);
        try self.function("h3_attention").allowDynamicShared(attention_shared);
        // the legacy default stream: work queued by the caller's framework comes first
        self.stream = .{ .d = d, .handle = null };
        self.gemms = try Gemms.init(gpa, d);
        errdefer self.gemms.deinit();
        try weights.read(self, path);
        try self.ctx.synchronize();
        self.loaded_seconds = seconds() - started;
        return self;
    }

    pub fn deinit(self: *Model) void {
        geometry.release(self);
        tp.free(self);
        weights.free(self);
        self.gemms.deinit();
        self.module.unload();
        self.ctx.deinit();
        self.driver.close();
        self.gpa.destroy(self);
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

    /// The modulated norm of the stream into xn, after the gated add of y when `gate` names a part of `gates`.
    fn norm(self: *Model, x: u64, weight: u64, gates: u64, tab: u64, line: u64, gate: c_int, scale: c_int, shift: c_int) !void {
        const g = &self.geometry;
        var args: cuda.Args = .{};
        for ([_]u64{ x, g.y, weight, gates, tab, line, g.xn }) |p| args.add(p);
        args.add(@as(c_int, @intCast(self.hidden)));
        args.add(gate);
        args.add(scale);
        args.add(shift);
        args.add(eps);
        try self.launch("h3_norm", .{ .x = @intCast(g.rows) }, 256, 0, &args);
    }

    /// softmax(q k) v for `count` query tiles from `first`; `queries` is how many a head's lists are laid out for.
    fn attend(self: *Model, par: usize, list: u64, sizes: u64, count: usize, queries: usize, keys: usize, first: usize, per_query: bool) !void {
        const g = &self.geometry;
        if (count == 0) return;
        var args: cuda.Args = .{};
        for ([_]u64{ g.tq, g.tk[par], g.tv[par], list, sizes, g.row_of, g.ya }) |p| args.add(p);
        for ([_]usize{ g.tiles * slot_tile, self.heads, queries, keys, first, @intFromBool(per_query) }) |n| args.add(@as(c_int, @intCast(n)));
        args.add(@as(f32, 1.0 / @sqrt(@as(f32, head_dim))));
        try self.launch("h3_attention", .{ .x = @intCast(count), .y = @intCast(self.heads) }, 128, attention_shared, &args);
    }

    /// FastH3's routing for this rank's tiles: tile scores, each video tile's key tiles, and the pooled branch.
    fn route(self: *Model, par: usize, own0: usize, own1: usize) !void {
        const g = &self.geometry;
        const tiles = g.tiles;
        const owned = own1 - own0;
        var args: cuda.Args = .{};
        args.add(g.qp + own0 * head_dim * 4);
        args.add(g.kp[par]);
        args.add(g.scores + own0 * tiles * 4);
        args.add(@as(c_int, @intCast(tiles)));
        try self.launch("h3_tile_scores", .{ .x = @intCast((tiles + 7) / 8), .y = @intCast(owned), .z = @intCast(self.heads) }, 256, 0, &args);
        const video0 = @max(own0, g.prefix) - g.prefix;
        const video1 = @max(own1, g.prefix) - g.prefix;
        if (video1 > video0) {
            args = .{};
            args.add(g.scores + video0 * tiles * 4);
            args.add(g.chosen + video0 * (g.prefix + g.keep) * 4);
            for ([_]usize{ tiles, g.prefix, g.keep }) |n| args.add(@as(c_int, @intCast(n)));
            try self.launch("h3_topk", .{ .x = @intCast(video1 - video0), .y = @intCast(self.heads) }, 32, 0, &args);
        }
        args = .{};
        args.add(g.scores + own0 * tiles * 4);
        args.add(g.vp[par]);
        args.add(g.coarse + own0 * head_dim * 4);
        args.add(@as(c_int, @intCast(tiles)));
        try self.launch("h3_coarse", .{ .x = @intCast(owned), .y = @intCast(self.heads) }, 32, 0, &args);
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
        /// This rank's tiles [tile0, tile1), both 0 for all; with a range the row inputs hold only those tiles' rows.
        tile0: u32 = 0,
        tile1: u32 = 0,
        /// With a range: every rank's share, rank r owning tiles [cuts[r], cuts[r + 1]).
        cuts: [9]u32 = @splat(0),
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
        const own0: usize = if (in.tile1 == 0) 0 else in.tile0;
        const own1: usize = if (in.tile1 == 0) tiles else in.tile1;
        if (own0 >= own1 or own1 > tiles) return error.BadGeometry;
        // TF_H3_TP_SOLO: a rank alone, nothing posted; TF_H3_TP_FORCE: the region even when this rank owns every tile
        const solo = std.c.getenv("TF_H3_TP_SOLO") != null;
        const exchange = own1 - own0 < tiles or (self.tp != null and std.c.getenv("TF_H3_TP_FORCE") != null);
        if (exchange) {
            const t = if (self.tp) |*held| held else return error.BadWorld;
            if (t.connected + 1 != t.world and !solo) return error.BadWorld;
            t.waited = 0;
            t.swaps = 0;
            t.packed_seconds = 0;
        }
        const owned = own1 - own0;
        try geometry.prepare(self, rows, tiles, prefix, keep, in.slot, own0, own1, exchange);
        const g = &self.geometry;
        const hidden = self.hidden;
        const inner = self.inner;
        const heads = self.heads;
        // the dense forward needs no routing gate: its product stops after q, k and v
        const width = if (dense) 3 * inner else 4 * inner;
        const table_bytes = @as(usize, in.lines) * 6 * hidden * 4;
        const video_tiles = tiles - prefix;
        const count = if (in.blocks == 0) self.blocks.len else @min(in.blocks, self.blocks.len);
        var mark = seconds();
        for (self.blocks[0..count], 0..) |*b, index| {
            const tab = in.tables + index * table_bytes;
            const par = index & 1;
            // the block before left its MLP's rows in y: its gated add runs with this block's first norm
            if (index == 0) try self.norm(in.x, b.norms.ptr, tab, tab, in.line, -1, scale_a, shift_a) else try self.norm(in.x, b.norms.ptr, tab - table_bytes, tab, in.line, gate_m, scale_a, shift_a);
            try self.lap("norms", &mark);
            try self.gemms.run(self.stream, g.xn, b.qkvc.ptr, g.qkvc, rows, width, hidden);
            try self.lap("q k v gate", &mark);
            {
                // q, k and v in tile order; the kernel's tile 0 is this rank's first, where its buffers start
                const at = own0 * slot_tile * head_dim * 2;
                var args: cuda.Args = .{};
                for ([_]u64{ g.qkvc, b.norms.ptr + hidden * 4, b.norms.ptr + hidden * 4 + head_dim * 2, in.cos, in.sin, g.row_of + own0 * slot_tile * 4, g.tq + at, g.tk[par] + at, g.tv[par] + at }) |p| args.add(p);
                for ([_]usize{ heads, width, in.rot, tiles * slot_tile }) |n| args.add(@as(c_int, @intCast(n)));
                args.add(eps);
                try self.launch("h3_heads", .{ .x = @intCast(owned) }, 256, 0, &args);
            }
            try self.lap("heads", &mark);
            if (dense) {
                if (exchange) {
                    try tp.swap(self, par, false, &in.cuts);
                    mark = seconds();
                }
                try self.attend(par, g.all, in.sizes, owned, owned, tiles, own0, false);
                try self.lap("attention", &mark);
            } else {
                var args: cuda.Args = .{};
                const at = own0 * slot_tile * head_dim * 2;
                const mean = own0 * head_dim * 4;
                for ([_]u64{ g.tq + at, g.tk[par] + at, g.tv[par] + at, in.sizes + own0 * 4, g.qp + mean, g.kp[par] + mean, g.vp[par] + mean }) |p| args.add(p);
                args.add(@as(c_int, @intCast(tiles * slot_tile)));
                args.add(@as(c_int, @intCast(tiles)));
                try self.launch("h3_pool", .{ .x = @intCast(owned), .y = @intCast(heads) }, 32, 0, &args);
                if (exchange) {
                    try self.lap("heads", &mark);
                    try tp.swap(self, par, true, &in.cuts);
                    mark = seconds();
                }
                try self.route(par, own0, own1);
                try self.lap("routing", &mark);
                // the prefix's query tiles see every tile; a video tile sees the prefix and its chosen tiles
                const prefix0 = @min(own0, prefix);
                const prefix1 = @min(own1, prefix);
                const video0 = @max(own0, prefix) - prefix;
                const video1 = @max(own1, prefix) - prefix;
                try self.attend(par, g.all, in.sizes, prefix1 - prefix0, prefix1 - prefix0, tiles, prefix0, false);
                try self.attend(par, g.chosen + video0 * (prefix + keep) * 4, in.sizes, video1 - video0, video_tiles, prefix + keep, prefix + video0, true);
                try self.lap("attention", &mark);
            }
            if (in.attention_out != 0 and index == count - 1) try self.driver.check(self.driver.api.cuMemcpyDtoD_v2(in.attention_out, g.ya, rows * inner * 2), "cuMemcpyDtoD");
            if (!dense) {
                var args: cuda.Args = .{};
                for ([_]u64{ g.ya, g.qkvc + 3 * inner * 2, g.coarse, in.slot }) |p| args.add(p);
                for ([_]usize{ inner, width, tiles }) |n| args.add(@as(c_int, @intCast(n)));
                try self.launch("h3_gate_mix", .{ .x = @intCast(rows) }, 256, 0, &args);
                try self.lap("gate mix", &mark);
            }
            try self.gemms.run(self.stream, g.ya, b.out.ptr, g.y, rows, hidden, inner);
            try self.lap("attention out", &mark);
            try self.norm(in.x, b.norms.ptr + hidden * 2, tab, tab, in.line, gate_a, scale_m, shift_m);
            try self.lap("norms", &mark);
            try self.gemms.run(self.stream, g.xn, b.fc1.ptr, g.wide, rows, 2 * self.mlp, hidden);
            {
                var args: cuda.Args = .{};
                args.add(g.wide);
                args.add(g.act);
                args.add(@as(c_int, @intCast(rows)));
                args.add(@as(c_int, @intCast(self.mlp)));
                try self.launch("h3_swiglu", .{ .x = 1024 }, 256, 0, &args);
            }
            try self.lap("mlp in", &mark);
            try self.gemms.run(self.stream, g.act, b.fc2.ptr, g.y, rows, hidden, self.mlp);
            try self.lap("mlp out", &mark);
        }
        {
            var args: cuda.Args = .{};
            for ([_]u64{ in.x, g.y, in.tables + (count - 1) * table_bytes, in.line }) |p| args.add(p);
            args.add(@as(c_int, @intCast(hidden)));
            args.add(@as(c_int, gate_m));
            try self.launch("h3_gate_add", .{ .x = @intCast(rows) }, 256, 0, &args);
        }
        try self.stream.synchronize();
        return seconds() - started;
    }
};
