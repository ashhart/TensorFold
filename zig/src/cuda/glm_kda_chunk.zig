// CUDA wrappers for GLM-5.3-Flash's KDA prompt chunks in chunked (WY) form (glm5_next/cuda/kda_chunk.cu ported to

// kernels/cuda/glm_kda_chunk.cu): one 256-thread block per head runs a prompt chunk's 32-row sub-chunks in order.

// The 128 x 128 state lives in registers; every function takes device pointers, shapes and a stream.

const std = @import("std");
const driver = @import("driver.zig");
const module = @import("module.zig");
const launch_ = @import("launch.zig");
const kernels = @import("kernels.zig");
const stream_ = @import("stream.zig");
const memory = @import("memory.zig");
const Driver = driver.Driver;
const Module = module.Module;
const Function = module.Function;
const Stream = stream_.Stream;

pub const sub_chunk = 32; // rows of a sub-chunk

pub const Chunk = struct {
    d: *const Driver,
    m: Module,
    fn_: Function,
    smem: u32,
    loaded: bool = false,

    fn smemBytes(dr: *const Driver, m: Module) !u32 {
        const g = try m.global("tf_glm_kda_chunk_smem_v");
        var buf = try memory.DeviceBuffer.alloc(dr, 8);
        defer buf.free();
        try buf.copyFrom(0, g.ptr, 8, null);
        var out: [8]u8 = undefined;
        try buf.download(0, &out);
        return @intCast(std.mem.readInt(i64, &out, .little));
    }

    pub fn load(dr: *const Driver) !Chunk {
        var c: Chunk = .{ .d = dr, .m = undefined, .fn_ = undefined, .smem = 0 };
        errdefer c.unload();
        c.m = try Module.load(dr, kernels.glm_kda_chunk);
        c.fn_ = try c.m.function("tf_glm_kda_chunk");
        c.smem = try smemBytes(dr, c.m);
        try c.fn_.allowDynamicShared(c.smem);
        c.loaded = true;
        return c;
    }

    pub fn unload(self: *Chunk) void {
        if (!self.loaded) return;
        self.m.unload();
        self.loaded = false;
    }

    // One head per block over the prompt chunk's rows: off = pos % 32 rows ahead may be zero-padded implicitly

    // (rows = the chunk's real rows, pos = the chunk's first absolute position). Zeroed out or state_out skips

    // that write.
    pub fn chain(
        self: *Chunk,
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
        off: u32,
        out: u64,
        state_out: u64,
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
        args.add(@as(i32, @intCast(off)));
        args.add(out);
        args.add(state_out);
        try launch_.launch(self.fn_, .{ .grid = .{ .x = h, .y = 1, .z = 1 }, .block = .{ .x = 256, .y = 1, .z = 1 }, .shared = self.smem }, s.*, &args);
    }
};
