// CUDA wrappers for GLM-5.3-Flash's KDA decode chain (glm5_next/cuda/kda.cu ported to kernels/cuda/glm_kda.cu).

// One 1024-thread block per head runs a window's rows in order; a replay runs saved rows into a state; a long

// window splits the state-independent work (all rows) from the row-by-row delta rule; conv windows shift in place.

// Every function takes device pointers, shapes and a stream.

const std = @import("std");
const driver = @import("driver.zig");
const module = @import("module.zig");
const launch_ = @import("launch.zig");
const stream_ = @import("stream.zig");
const kernels = @import("kernels.zig");
const Driver = driver.Driver;
const Module = module.Module;
const Function = module.Function;
const Stream = stream_.Stream;

pub const d_k = 128; // keys and values per head
pub const taps = 4; // the conv's window length (3 rows of state + this row's projection)

/// The saved row inputs a replay needs (fp32 k and decays, bf16 v, fp32 beta; the wide path also q and y).
pub const Scratch = struct {
    k: u64 = 0,
    v: u64 = 0,
    g: u64 = 0,
    b: u64 = 0,
    q: u64 = 0,
    y: u64 = 0,

    /// Byte counts for H heads and R rows (6 views, laid out back to back by the caller).
    pub fn bytes(h: u32, rows: u32) u64 {
        const per: u64 = @as(u64, h) * d_k;
        const r: u64 = rows;
        return r * per * (4 + 2 + 4 + 4 + 2) + r * @as(u64, h) * 4;
    }
};

pub const Kda = struct {
    d: *const Driver,
    m: Module,
    chain_fn: Function,
    prep_fn: Function,
    step_fn: Function,
    out_fn: Function,
    replay_fn: Function,
    replay_layers_fn: Function,
    loaded: bool = false,
    shift_fn: Function,

    pub fn load(dr: *const Driver) !Kda {
        var k: Kda = .{
            .d = dr,
            .m = undefined,
            .chain_fn = undefined,
            .prep_fn = undefined,
            .step_fn = undefined,
            .out_fn = undefined,
            .replay_fn = undefined,
            .replay_layers_fn = undefined,
            .shift_fn = undefined,
        };
        errdefer k.unload();
        k.m = try Module.load(dr, kernels.glm_kda);
        k.chain_fn = try k.m.function("tf_glm_kda_chain");
        k.prep_fn = try k.m.function("tf_glm_kda_prep");
        k.step_fn = try k.m.function("tf_glm_kda_step");
        k.out_fn = try k.m.function("tf_glm_kda_out");
        k.replay_fn = try k.m.function("tf_glm_kda_replay");
        k.replay_layers_fn = try k.m.function("tf_glm_kda_replay_layers");
        k.shift_fn = try k.m.function("tf_glm_kda_conv_shift");
        k.loaded = true;
        return k;
    }

    pub fn unload(self: *Kda) void {
        if (!self.loaded) return;
        self.m.unload();
        self.loaded = false;
    }

    // One head per block, R rows in order: P's rows hold q|k|v (3 * H * 128 channels) then the beta logit at b_off;

    // A/G's rows the decays' pre-activations and gates. Zeroed Scratch views skip the saves; zero state_out skips

    // the state write (state_in stays the input).
    pub fn chain(
        self: *Kda,
        s: *Stream,
        h: u32,
        p: u64,
        p_stride: i64,
        b_off: i32,
        a: u64,
        a_stride: i64,
        g: u64,
        g_stride: i64,
        conv_state: u64,
        conv_w: u64,
        state_in: u64,
        a_log: u64,
        dt_bias: u64,
        norm_w: u64,
        eps: f32,
        lower: f32,
        rows: u32,
        out: u64,
        state_out: u64,
        scratch: Scratch,
    ) !void {
        var args = launch_.Args{};
        args.add(@as(i32, @intCast(h)));
        args.add(p);
        args.add(@as(i32, @intCast(p_stride)));
        args.add(b_off);
        args.add(a);
        args.add(@as(i32, @intCast(a_stride)));
        args.add(g);
        args.add(@as(i32, @intCast(g_stride)));
        args.add(conv_state);
        args.add(conv_w);
        args.add(state_in);
        args.add(a_log);
        args.add(dt_bias);
        args.add(norm_w);
        args.add(eps);
        args.add(lower);
        args.add(@as(i32, @intCast(rows)));
        args.add(out);
        args.add(state_out);
        args.add(scratch.k);
        args.add(scratch.v);
        args.add(scratch.g);
        args.add(scratch.b);
        try launch_.launch(self.chain_fn, .{ .grid = .{ .x = h, .y = 1, .z = 1 }, .block = .{ .x = 1024, .y = 1, .z = 1 } }, s.*, &args);
    }

    // The wide path: prep and out kernels over all rows, the delta rule row by row; q/y live in scratch.q/.y.
    pub fn chainWide(
        self: *Kda,
        s: *Stream,
        h: u32,
        p: u64,
        p_stride: i64,
        b_off: i32,
        a: u64,
        a_stride: i64,
        g: u64,
        g_stride: i64,
        conv_state: u64,
        conv_w: u64,
        state_in: u64,
        a_log: u64,
        dt_bias: u64,
        norm_w: u64,
        eps: f32,
        lower: f32,
        rows: u32,
        out: u64,
        state_out: u64,
        scratch: Scratch,
    ) !void {
        // The engine's host sequence: prep writes all rows' q, k, v, decay, beta; step runs the delta rule.
        var args = launch_.Args{};
        args.add(@as(i32, @intCast(h)));
        args.add(p);
        args.add(@as(i32, @intCast(p_stride)));
        args.add(b_off);
        args.add(a);
        args.add(@as(i32, @intCast(a_stride)));
        args.add(conv_state);
        args.add(conv_w);
        args.add(a_log);
        args.add(dt_bias);
        args.add(lower);
        args.add(scratch.q);
        args.add(scratch.k);
        args.add(scratch.v);
        args.add(scratch.g);
        args.add(scratch.b);
        try launch_.launch(self.prep_fn, .{ .grid = .{ .x = rows, .y = h, .z = 1 }, .block = .{ .x = 512, .y = 1, .z = 1 } }, s.*, &args);
        var sargs = launch_.Args{};
        sargs.add(@as(i32, @intCast(h)));
        sargs.add(state_in);
        sargs.add(scratch.q);
        sargs.add(scratch.k);
        sargs.add(scratch.v);
        sargs.add(scratch.g);
        sargs.add(scratch.b);
        sargs.add(@as(i32, @intCast(rows)));
        sargs.add(scratch.y);
        sargs.add(state_out);
        try launch_.launch(self.step_fn, .{ .grid = .{ .x = h, .y = 8, .z = 1 }, .block = .{ .x = 128, .y = 1, .z = 1 } }, s.*, &sargs);
        var oargs = launch_.Args{};
        oargs.add(@as(i32, @intCast(h)));
        oargs.add(scratch.y);
        oargs.add(g);
        oargs.add(@as(i32, @intCast(g_stride)));
        oargs.add(norm_w);
        oargs.add(eps);
        oargs.add(out);
        try launch_.launch(self.out_fn, .{ .grid = .{ .x = rows, .y = h, .z = 1 }, .block = .{ .x = 128, .y = 1, .z = 1 } }, s.*, &oargs);
    }

    /// Rows saved by a chain (or chain_wide) into a fresh state, skipping their conv and norms.
    pub fn replay(
        self: *Kda,
        s: *Stream,
        h: u32,
        state_in: u64,
        scratch: Scratch,
        rows: u32,
        state_out: u64,
    ) !void {
        var args = launch_.Args{};
        args.add(@as(i32, @intCast(h)));
        args.add(state_in);
        args.add(scratch.k);
        args.add(scratch.v);
        args.add(scratch.g);
        args.add(scratch.b);
        args.add(@as(i32, @intCast(rows)));
        args.add(state_out);
        try launch_.launch(self.replay_fn, .{ .grid = .{ .x = h, .y = 1, .z = 1 }, .block = .{ .x = 1024, .y = 1, .z = 1 } }, s.*, &args);
    }

    /// All layers at once: layers * H blocks, per-layer state strides and per-layer saved-row strides.
    pub fn replayLayers(
        self: *Kda,
        s: *Stream,
        h: u32,
        layers: u32,
        state_in: u64,
        state_stride: u64,
        scratch_k: u64,
        scratch_v: u64,
        scratch_g: u64,
        scratch_b: u64,
        kv_stride: u64,
        b_stride: u64,
        rows: u32,
        state_out: u64,
    ) !void {
        var args = launch_.Args{};
        args.add(@as(i32, @intCast(h)));
        args.add(state_in);
        args.add(state_stride);
        args.add(scratch_k);
        args.add(scratch_v);
        args.add(scratch_g);
        args.add(scratch_b);
        args.add(kv_stride);
        args.add(b_stride);
        args.add(@as(i32, @intCast(rows)));
        args.add(state_out);
        try launch_.launch(self.replay_layers_fn, .{ .grid = .{ .x = layers * h, .y = 1, .z = 1 }, .block = .{ .x = 1024, .y = 1, .z = 1 } }, s.*, &args);
    }

    /// conv [L, 3, C] in place takes rows keep .. keep + 2 of [old window; proj rows] (proj [R, W >= C]).
    pub fn convShift(
        self: *Kda,
        s: *Stream,
        conv: u64,
        proj: u64,
        layers: u32,
        keep: u32,
        conv_stride: i64,
        proj_stride: i64,
        proj_row: i64,
        c: u32,
    ) !void {
        var args = launch_.Args{};
        args.add(conv);
        args.add(proj);
        args.add(@as(i32, @intCast(keep)));
        args.add(conv_stride);
        args.add(proj_stride);
        args.add(proj_row);
        args.add(@as(i32, @intCast(c)));
        try launch_.launch(self.shift_fn, .{ .grid = .{ .x = layers, .y = (c + 1023) / 1024, .z = 1 }, .block = .{ .x = 1024, .y = 1, .z = 1 } }, s.*, &args);
    }
};
