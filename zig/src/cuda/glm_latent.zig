//! GLM-5.3-Flash latent attention on CUDA: cache write, kv_b absorb/expand, 512-key chunks and their merge.

const launch_ = @import("launch.zig");
const stream_ = @import("stream.zig");
const module = @import("module.zig");
const driver = @import("driver.zig");
const kernels = @import("kernels.zig");
const Driver = driver.Driver;
const Module = module.Module;
const Function = module.Function;
const Stream = stream_.Stream;

pub const heads = 32; // index heads
pub const d = 256; // qk dims
pub const lw = 512; // latent width
pub const v = 256; // v dims

pub const Latent = struct {
    d: *const Driver,
    m: Module,
    ma: Module,
    lat_fn: Function,
    absorb_fn: Function,
    expand_fn: Function,
    merge_fn: Function,
    chunks_fns: [6]Function, // d16_bf16, d16_fp8, d32_bf16, d32_fp8, s16_fp8, s32_fp8
    loaded: bool = false,

    pub fn load(dr: *const Driver) !Latent {
        var k: Latent = .{
            .d = dr,
            .m = undefined,
            .ma = undefined,
            .lat_fn = undefined,
            .absorb_fn = undefined,
            .expand_fn = undefined,
            .merge_fn = undefined,
            .chunks_fns = undefined,
        };
        errdefer k.unload();
        k.m = try Module.load(dr, kernels.glm_latent);
        k.ma = try Module.load(dr, kernels.glm_latent_att);
        k.lat_fn = try k.m.function("tf_glm_lat_write");
        k.absorb_fn = try k.m.function("tf_glm_absorb_q");
        k.expand_fn = try k.m.function("tf_glm_expand_v");
        k.chunks_fns[0] = try k.ma.function("tf_glm_attn_d16_bf16");
        k.chunks_fns[1] = try k.ma.function("tf_glm_attn_d16_fp8");
        k.chunks_fns[2] = try k.ma.function("tf_glm_attn_d32_bf16");
        k.chunks_fns[3] = try k.ma.function("tf_glm_attn_d32_fp8");
        k.chunks_fns[4] = try k.ma.function("tf_glm_attn_s16_fp8");
        k.chunks_fns[5] = try k.ma.function("tf_glm_attn_s32_fp8");
        k.merge_fn = try k.ma.function("tf_glm_merge");
        k.loaded = true;
        return k;
    }

    pub fn unload(self: *Latent) void {
        if (!self.loaded) return;
        self.m.unload();
        self.ma.unload();
        self.loaded = false;
    }

    /// Write rows into the cache slot the device-side pos names; fp8 rows get an amax/448 scale.
    pub fn latWrite(
        self: *Latent,
        s: *Stream,
        lat: u64,
        lat_stride: i64,
        lc: u64,
        pos: u64,
        sc: u64,
        rows: u32,
        fp8: bool,
    ) !void {
        var args = launch_.Args{};
        args.add(lat);
        args.add(lat_stride);
        args.add(lc);
        args.add(pos);
        args.add(sc);
        args.add(@as(i32, if (fp8) 1 else 0));
        try launch_.launch(self.lat_fn, .{ .grid = .{ .x = rows, .y = 1, .z = 1 }, .block = .{ .x = 128, .y = 1, .z = 1 } }, s.*, &args);
    }

    /// qa = q @ wk (the absorbed kv_b qk side): a head a block-x, a 32-latent col group a block-y; rows looped.
    pub fn absorbQ(self: *Latent, s: *Stream, q: u64, wk: u64, qa: u64, rows: u32) !void {
        var args = launch_.Args{};
        args.add(q);
        args.add(wk);
        args.add(qa);
        args.add(@as(i32, @intCast(rows)));
        try launch_.launch(self.absorb_fn, .{ .grid = .{ .x = heads, .y = 512 / 32, .z = 1 }, .block = .{ .x = 128, .y = 1, .z = 1 } }, s.*, &args);
    }

    /// out = ol @ wv (the expanded kv_b v side): a head a block-x, a 16-v col group a block-y; rows looped.
    pub fn expandV(self: *Latent, s: *Stream, ol: u64, wv: u64, out: u64, rows: u32) !void {
        var args = launch_.Args{};
        args.add(ol);
        args.add(wv);
        args.add(out);
        args.add(@as(i32, @intCast(rows)));
        try launch_.launch(self.expand_fn, .{ .grid = .{ .x = heads, .y = 256 / 16, .z = 1 }, .block = .{ .x = 128, .y = 1, .z = 1 } }, s.*, &args);
    }

    /// One 512-key chunk a block (z), a 16/32-row qk group a block (y; 32 qk rows when rows >= 64).
    pub fn attnChunks(
        self: *Latent,
        s: *Stream,
        qa: u64,
        cache: u64,
        sc: u64,
        pos: u64,
        tok: u64,
        cnt: u64,
        po: u64,
        pm: u64,
        pl: u64,
        rows: u32,
        tokw: i32,
        nch: u32,
        fp8: bool,
        sparse: bool,
        scale: f32,
    ) !void {
        const hbt: u32 = if (rows >= 64) 32 else 16;
        if (sparse and !fp8) return error.SparseNeedsFp8;
        const wide: u32 = if (hbt == 32) 1 else 0;
        const idx: u32 = if (sparse) 4 + wide else 2 * wide + (if (fp8) @as(u32, 1) else 0);
        var args = launch_.Args{};
        args.add(qa);
        args.add(cache);
        args.add(sc);
        args.add(pos);
        args.add(tok);
        args.add(cnt);
        args.add(po);
        args.add(pm);
        args.add(pl);
        args.add(@as(i32, @intCast(rows)));
        args.add(tokw);
        args.add(scale);
        try launch_.launch(self.chunks_fns[idx], .{ .grid = .{ .x = rows, .y = 32 / hbt, .z = nch }, .block = .{ .x = 256, .y = 1, .z = 1 } }, s.*, &args);
    }

    /// Combine the per-chunk partials into the bf16 output rows; sparse rows with no tokens keep their input.
    pub fn merge(
        self: *Latent,
        s: *Stream,
        po: u64,
        pm: u64,
        pl: u64,
        out: u64,
        cnt: u64,
        rows: u32,
        nch: u32,
        sparse: bool,
    ) !void {
        var args = launch_.Args{};
        args.add(po);
        args.add(pm);
        args.add(pl);
        args.add(out);
        args.add(cnt);
        args.add(@as(i32, @intCast(rows)));
        args.add(@as(i32, @intCast(nch)));
        args.add(@as(i32, if (sparse) 1 else 0));
        try launch_.launch(self.merge_fn, .{ .grid = .{ .x = rows, .y = heads, .z = 1 }, .block = .{ .x = 128, .y = 1, .z = 1 } }, s.*, &args);
    }
};
