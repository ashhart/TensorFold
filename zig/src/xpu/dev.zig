//! Low-level XPU device plumbing for the Qwen3.8-27B runner: safetensors shards, the Level Zero device, profiling.
const std = @import("std");
const ze = @import("ze.zig");
const q = @import("qwen27.zig");
const fwd = @import("forward.zig");
const a8 = fwd.a8;
const alloc = q.alloc;

// ---------------------------------------------------------------- safetensors (mmap through libc)

extern "c" fn open(path: [*:0]const u8, flags: c_int, ...) c_int;
extern "c" fn close(fd: c_int) c_int;
extern "c" fn lseek(fd: c_int, off: i64, whence: c_int) i64;
extern "c" fn mmap(addr: ?*anyopaque, len: usize, prot: c_int, flags: c_int, fd: c_int, off: i64) ?*anyopaque;

pub const Tensor = struct { name: []const u8, dtype: []const u8, shape: [8]u64, rank: usize, bytes: []const u8 };

pub const Checkpoint = struct {
    tensors: std.ArrayList(Tensor) = .empty,

    pub fn addShard(c: *Checkpoint, path: [:0]const u8) !void {
        const fd = open(path.ptr, 0);
        if (fd < 0) return error.FileNotFound;
        defer _ = close(fd);
        const size: usize = @intCast(lseek(fd, 0, 2));
        const base = mmap(null, size, 1, 2, fd, 0) orelse return error.MmapFailed; // PROT_READ, MAP_PRIVATE
        const bytes: [*]const u8 = @ptrCast(base);
        const n = std.mem.readInt(u64, bytes[0..8], .little);
        const header = bytes[8 .. 8 + n];
        const data = bytes[8 + n .. size];
        const parsed = try std.json.parseFromSlice(std.json.Value, alloc, header, .{});
        var it = parsed.value.object.iterator();
        while (it.next()) |e| {
            if (std.mem.eql(u8, e.key_ptr.*, "__metadata__")) continue;
            const o = e.value_ptr.object;
            var t: Tensor = .{ .name = e.key_ptr.*, .dtype = o.get("dtype").?.string, .shape = @splat(1), .rank = 0, .bytes = &.{} };
            if (o.get("shape").?.array.items.len > t.shape.len) return error.TensorRank; // vision conv is 5-D
            for (o.get("shape").?.array.items, 0..) |d, i| t.shape[i] = @intCast(d.integer);
            t.rank = o.get("shape").?.array.items.len;
            const span = o.get("data_offsets").?.array.items;
            t.bytes = data[@intCast(span[0].integer)..@intCast(span[1].integer)];
            try c.tensors.append(alloc, t);
        }
    }

    pub fn get(c: *const Checkpoint, name: []const u8) !Tensor {
        for (c.tensors.items) |t| if (std.mem.endsWith(u8, t.name, name) and
            (t.name.len == name.len or std.mem.startsWith(u8, t.name, "language_model.") or std.mem.startsWith(u8, t.name, "model.language_model."))) return t;
        std.debug.print("missing tensor: {s}\n", .{name});
        return error.MissingTensor;
    }
    pub fn has(c: *const Checkpoint, name: []const u8) bool {
        for (c.tensors.items) |t| if (std.mem.endsWith(u8, t.name, name)) return true;
        return false;
    }
};

// ---------------------------------------------------------------- device

pub const Dev = struct {
    g: ze.Gpu,
    list: ze.H,
    staging: []u8 = &.{}, // USM host memory: the driver cannot copy from file-backed mmap pages directly
    allocated: usize = 0, // device bytes allocated (frees not counted: buffers live as long as the runner)
    k: struct {
        embed: ze.H,
        add_rmsnorm_dec: ze.H,
        add_rmsnorm: ze.H,
        gdn_pre: ze.H,
        gdn_pre_rows: ze.H,
        gated_norm: ze.H,
        attn_prep: ze.H,
        attn_gate: ze.H,
        swiglu: ze.H,
        argmax: ze.H,
        argmax_part: ze.H,
        argmax_part_masked: ze.H,
        tap: ze.H,
        conv_commit: ze.H,
        repack: ze.H,
        gdn_step: ze.H,
        attn_decode: ze.H,
        attn_merge: ze.H,
        kv_move: ze.H,
        exl3_gemv: ze.H,
        rot_heads: ze.H,
        attn_prompt: ze.H,
        attn_prompt8: ze.H,
        embed_dense: ze.H,
        dense_rows: ze.H,
        repack_gptq: ze.H,
        quantize_mlx: ze.H,
        dec_wide_g128: ze.H,
        dec_wide64_g128: ze.H,
        dec_narrow_g128: ze.H,
        ngen_wide_g128: ze.H,
        ngen_narrow_g128: ze.H,
        ngen_swiglu: ze.H,
        ngen_swiglu_g128: ze.H,
        rowsums_ff: ze.H,
        fakequant_rows: ze.H,
        a8: ze.H,
        a8_swiglu: ze.H,
        quant_a8: ze.H,
        a4: ze.H,
        quant_a4: ze.H,
        dec_wide: ze.H,
        dec_wide64: ze.H,
        dec_narrow: ze.H,
        gdn_prep: ze.H,
        gdn_state: ze.H,
        gdn_out: ze.H,
        ngen_wide: ze.H,
        ngen_narrow: ze.H,
    },

    pub fn kernel(d: *Dev, m: ze.H, name: [*:0]const u8, local: u32) !ze.H {
        const k = try d.g.kernel(m, name);
        try ze.check(d.g.api.zeKernelSetGroupSize(k, local, 1, 1));
        return k;
    }

    pub fn alloc_(d: *Dev, bytes: usize) !*anyopaque {
        d.allocated += @max(bytes, 64);
        return d.g.deviceAlloc(@max(bytes, 64));
    }

    pub fn upload(d: *Dev, bytes: []const u8) !*anyopaque {
        const p = try d.alloc_(bytes.len);
        try d.copyIn(p, bytes);
        return p;
    }

    /// Host bytes (any memory, mmap included) to the device, through the USM staging buffer.
    pub fn copyIn(d: *Dev, dst: *anyopaque, bytes: []const u8) !void {
        if (d.staging.len == 0) d.staging = @as([*]u8, @ptrCast(try d.g.hostAlloc(256 << 20)))[0 .. 256 << 20];
        var done: usize = 0;
        while (done < bytes.len) {
            const n = @min(d.staging.len, bytes.len - done);
            @memcpy(d.staging[0..n], bytes[done .. done + n]);
            try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, off(dst, done), d.staging.ptr, n, null, 0, null));
            try d.sync();
            done += n;
        }
    }

    pub fn launch(d: *Dev, k: ze.H, groups: ze.GroupCount, args: anytype) !void {
        inline for (args, 0..) |a, i| try d.g.arg(k, i, a);
        var ev: ze.H = null;
        if (q.prof.on) {
            if (q.prof.used == q.prof.events.len) try d.sync();
            ev = q.prof.events[q.prof.used];
            q.prof.kernel[q.prof.used] = k;
            q.prof.used += 1;
        }
        try ze.check(d.g.api.zeCommandListAppendLaunchKernel(d.list, k, &groups, ev, 0, null));
    }

    pub fn sync(d: *Dev) !void {
        try ze.check(d.g.api.zeCommandListHostSynchronize(d.list, std.math.maxInt(u64)));
        if (q.prof.on) {
            for (q.prof.events[0..q.prof.used], q.prof.kernel[0..q.prof.used]) |e, k| {
                const t = try d.g.seconds(e);
                try ze.check(d.g.api.zeEventHostReset(e));
                for (&q.prof.names, &q.prof.total) |n, *tot| {
                    if (n.k == k) {
                        tot.* += t;
                        break;
                    }
                }
            }
            q.prof.used = 0;
        }
    }

    pub fn fill(d: *Dev, p: *anyopaque, pattern: anytype, bytes: usize) !void {
        const T = @TypeOf(pattern);
        const host = try alloc.alloc(T, bytes / @sizeOf(T));
        defer alloc.free(host);
        @memset(host, pattern);
        try d.copyIn(p, std.mem.sliceAsBytes(host));
    }
};

pub fn copy(d: *Dev, dst: *anyopaque, src: *anyopaque, n: usize) !void {
    try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, dst, src, n, null, 0, null));
}

pub fn off(p: *anyopaque, bytes: usize) *anyopaque {
    return @ptrFromInt(@intFromPtr(p) + bytes);
}

/// XPU_PROFILE=1: GPU time per kernel, summed over the run.
pub const Prof = struct {
    on: bool = false,
    events: [4096]ze.H = @splat(null),
    kernel: [4096]ze.H = @splat(null),
    used: usize = 0,
    names: [64]struct { k: ze.H = null, name: []const u8 = "" } = @splat(.{}),
    total: [64]f64 = @splat(0),
};

pub fn profReport(label: []const u8) void {
    if (!q.prof.on) return;
    var sum: f64 = 0;
    for (q.prof.total) |t| sum += t;
    std.debug.print("GPU time, {s}: {d:.1} ms\n", .{ label, sum * 1e3 });
    for (q.prof.names, &q.prof.total) |n, *t| {
        if (t.* > 0) std.debug.print("  {s:<14} {d:8.2} ms {d:5.1} %\n", .{ n.name, t.* * 1e3, t.* / sum * 100 });
        t.* = 0;
    }
}
