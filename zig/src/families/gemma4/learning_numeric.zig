//! Test-only recording storage runs the native pullback bodies; the forward instrument is an independent scalar graph.
const std = @import("std");
const core = @import("core");
const profile = @import("learning_profile.zig");
const Hook = @import("learning_hook.zig");
const a = std.testing.allocator;
extern fn tf_living_rows_cpu(c_uint, core.living_rows.Params, [*][*]u8) void;
extern fn tf_living_qsa_cpu(c_uint, core.living_rows.Params, [*][*]u8) void;
extern fn tf_living_gemma_cpu(c_uint, core.living_rows.Params, [*][*]u8) void;
pub const B = struct {
    pub const Buffer = []align(16) u8;
    pub const Pass = void;
    held: usize = 0,
    allocations: usize = 0,
    fail_drain: bool = false,
    pub fn allocationBytes(n: usize) !u64 {
        return n;
    }
    pub fn allocate(b: *B, n: usize) !Buffer {
        const out = try a.alignedAlloc(u8, .@"16", n);
        b.held += n;
        b.allocations += 1;
        return out;
    }
    pub fn release(b: *B, v: Buffer) void {
        b.held -= v.len;
        a.free(v);
    }
    pub fn upload(b: *B, data: []const u8) !Buffer {
        const out = try b.allocate(data.len);
        @memcpy(out, data);
        return out;
    }
    pub fn write(_: *B, b: Buffer, off: usize, data: []const u8) !void {
        if (off > b.len or data.len > b.len - off) return error.Range;
        @memcpy(b[off..][0..data.len], data);
    }
    pub fn read(b: *B, v: Buffer, off: usize, out: []u8) !void {
        try b.drain();
        @memcpy(out, v[off..][0..out.len]);
    }
    pub fn begin(_: *B) !void {}
    pub fn submit(_: *B, _: void) !void {}
    pub fn abort(_: *B) void {}
    pub fn drain(b: *B) !void {
        if (b.fail_drain) return error.PlantedDrain;
    }
};
pub const K = struct {
    pub const Core = core;
    pub const Ref = @import("learning_ref.zig").Ref(B);
    calls: usize = 0,
    pub fn init(_: std.mem.Allocator) K {
        return .{};
    }
    pub fn ensure(_: *K, _: *B) !void {}
    pub fn deinitDrained(_: *K) void {}
    pub fn length(v: B.Buffer) usize {
        return v.len;
    }
    fn run(k: *K, comptime spec: type, op: spec.Op, p: core.living_rows.Params, refs: [11]Ref) !void {
        var ptrs: [11][*]u8 = undefined;
        var ranges: [11]core.weight_only.Range = undefined;
        for (refs, &ptrs, &ranges) |r, *ptr, *range| {
            if (r.offset > r.buffer.len or r.len > r.buffer.len - r.offset) return error.Range;
            ptr.* = r.buffer.ptr + r.offset;
            range.* = .{ .address = @intFromPtr(ptr.*), .len = r.len };
        }
        try spec.validate(op, p, ranges);
        const run_body = if (spec == core.living_rows) tf_living_rows_cpu else if (spec == core.living_qsa) tf_living_qsa_cpu else tf_living_gemma_cpu;
        run_body(@intCast(@backingInt(op)), p, &ptrs);
        k.calls += 1;
    }
    pub fn row(k: *K, _: *B, _: void, op: core.living_rows.Op, p: core.living_rows.Params, refs: [11]Ref) !void {
        try k.run(core.living_rows, op, p, refs);
    }
    pub fn qsa(k: *K, _: *B, _: void, op: core.living_qsa.Op, p: core.living_rows.Params, refs: [11]Ref) !void {
        try k.run(core.living_qsa, op, p, refs);
    }
    pub fn gemma(k: *K, _: *B, _: void, op: core.living_gemma.Op, p: core.living_rows.Params, refs: [11]Ref) !void {
        try k.run(core.living_gemma, op, p, refs);
    }
};
pub const E = struct {
    pub const Runtime = @import("learning_runtime.zig").Runtime(B, K);
    pub const probability_mode: u32 = 0;
    r: *Runtime,
    dir: []const u8,
    learned: usize = 0,
    unified: bool = false,
    wrapped: bool = true,
    omit: ?Hook.Node = null,
    poison_open: bool = false,
    pub fn trainBytes(_: *E, _: usize) !u64 {
        return 0;
    }
    pub fn trainOpen(e: *E, _: usize) !Train {
        if (e.poison_open) e.r.backend.fail_drain = true;
        return .{ .e = e };
    }
    pub fn frequencies(_: *E, l: usize, out: []f32) void {
        for (out, 0..) |*v, i| v.* = if (l % 6 == 5 and i >= 64) 0 else @floatCast(std.math.pow(f64, if (l % 6 == 5) 1e6 else 10000, -2 * @as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(out.len * 2))));
    }
    pub const Train = struct {
        e: *E,
        pub fn deinit(_: *Train) void {}
        pub fn forward(t: *Train, listener: Runtime.Hook, ids: []const u32) !void {
            try @import("learning_numeric_forward.zig").run(t.e, listener, ids);
        }
    };
};
pub fn bits(x: f32) u16 {
    return core.quant.convert.bf16(x);
}
pub fn f(x: u16) f32 {
    return @bitCast(@as(u32, x) << 16);
}
pub fn rounded(x: f32) f32 {
    return f(bits(x));
}
