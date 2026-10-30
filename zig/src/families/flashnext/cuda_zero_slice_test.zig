//! The zero slice against the old clears: Pass.lane over an emulated part buffer, poisoned, reused across passes.
const std = @import("std");
const cuda = @import("cuda");
const wt = @import("cuda_prompt_wt.zig");
const parts = @import("cuda_batch_state.zig");
const batch = @import("cuda_batch.zig");
const zs = @import("cuda_zero_slice.zig");

const dims = wt.dims;
const cap = 8;
const qmmf_id = 0x100;
const reduce_id = 0x200;
const legacy_id = 0x300;
const base: u64 = 0x1000_0000;
const poison: u32 = 0xffc0_dead;

/// Device memory as f32 bits from `base`: lane tiles write values from a running count, memsets write bytes.
const Emu = struct {
    var mem: []u32 = &.{};
    var writes: u32 = 0;
    var memsets: usize = 0;

    fn arg(comptime T: type, args: ?[*]?*anyopaque, i: usize) T {
        return @as(*const T, @ptrCast(@alignCast(args.?[i].?))).*;
    }

    /// -0, +0, a NaN with a payload and both infinities among finite values of either sign.
    fn value(call: u32, i: usize) u32 {
        return switch (i % 61) {
            0 => 0x8000_0000,
            1 => 0,
            2 => 0x7fc0_1234,
            3 => 0x7f80_0000,
            4 => 0xff80_0000,
            else => blk: {
                const h = std.hash.Wyhash.hash(call, std.mem.asBytes(&i));
                break :blk @as(u32, @truncate(h & 0x807f_ffff)) | (@as(u32, 100 + @as(u32, @truncate(h >> 40)) % 50) << 23);
            },
        };
    }

    fn fill(ptr: u64, count: usize) void {
        writes += 1;
        const at: usize = @intCast((ptr - base) / 4);
        for (mem[at..][0..count], 0..) |*v, i| v.* = value(writes, i);
    }

    fn launch(f: cuda.abi.Function, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: cuda.abi.Stream, args: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) c_int {
        switch (@intFromPtr(f)) {
            reduce_id => fill(arg(u64, args, 1), @intCast(arg(i64, args, 2))),
            legacy_id => {
                const m: usize = @intCast(arg(c_int, args, 7));
                const n: usize = @intCast(arg(c_int, args, 8));
                for (0..@as(usize, @intCast(arg(c_int, args, 10)))) |j| fill(arg(u64, args, 6) + j * m * n * 4, m * n);
            },
            // a qmmf tile writes y [m, n] itself, or its partials elsewhere when a reduce follows
            else => if (arg(u64, args, 5) == 0) fill(arg(u64, args, 4), @as(usize, @intCast(arg(i32, args, 6))) * @as(usize, @intCast(arg(i32, args, 7)))),
        }
        return 0;
    }

    fn memset(ptr: u64, v: u8, count: usize, _: cuda.abi.Stream) callconv(.c) c_int {
        @memset(std.mem.sliceAsBytes(mem)[@intCast(ptr - base)..][0..count], v);
        memsets += 1;
        return 0;
    }
};

const Path = enum { old, new };

const Rig = struct {
    d: cuda.Driver,
    k: parts.Kernels,
    s: parts.Scratch,
    native: wt.Q4,
    legacy: wt.Q4,

    fn init(r: *Rig, mem: []u32) void {
        r.d = undefined;
        r.d.api.cuLaunchKernel = Emu.launch;
        r.d.api.cuMemsetD8Async = Emu.memset;
        const f = struct {
            fn of(d: *const cuda.Driver, id: usize) cuda.Function {
                return .{ .d = d, .handle = @ptrFromInt(id) };
            }
        }.of;
        r.k = undefined;
        r.k.lane = f(&r.d, legacy_id);
        @memset(std.mem.asBytes(&r.k.native_lane), 0); // zeroed: no optional tile path (half tiles), only these handles
        r.k.native_lane.fns = @splat(@splat(f(&r.d, qmmf_id)));
        r.k.native_lane.f32_fns = @splat(@splat(f(&r.d, qmmf_id)));
        r.k.native_lane.reduce = @splat(f(&r.d, reduce_id));
        r.k.native_lane.clusters = false;
        r.s = undefined;
        r.s.cap = cap;
        r.s.part = base;
        r.s.native_part = base + mem.len * 4;
        r.s.part_zero = true;
        const z: cuda.DeviceBuffer = .{ .d = &r.d, .ptr = 0, .len = 0 };
        r.native = .{ .w = z, .s = z, .b = z, .n = dims, .k = 2048, .native = cuda.modelopt.bf16Weight(0x2000, dims, 2048) };
        r.legacy = .{ .w = z, .s = z, .b = z, .n = dims, .k = 2048 };
        Emu.mem = mem;
        Emu.writes = 0;
        Emu.memsets = 0;
    }

    fn pass(r: *Rig) batch.Pass(void) {
        return .{ .gpa = std.testing.allocator, .io = undefined, .stream = .{ .d = &r.d, .handle = null }, .tri = {}, .k = &r.k, .s = &r.s, .m = undefined, .seq = undefined, .hooks = undefined };
    }

    /// One layer's out projection and writeback as cuda_batch.zig's chunk runs them; out: each value's slice sum.
    fn layer(r: *Rig, path: Path, n: usize, kept: bool, collective: bool, out: *std.ArrayList(u32)) !void {
        var p = r.pass();
        p.zero_slices = path == .new and zs.usable(r.s.part_zero, @intFromBool(kept), collective, true);
        p.out_slices = try p.lane(0x10, 0x20, r.native, 0, r.s.part, n, wt.out_sk);
        try r.writeback(zs.slices(p.zero_slices, cap, n, p.out_slices), n, out);
    }

    /// fn_hc_writeback_slices' branch sum over part: slice 0, then each further slice added in fp32.
    fn writeback(r: *Rig, sl: zs.Slices, n: usize, out: *std.ArrayList(u32)) !void {
        _ = r;
        for (0..n * dims) |i| {
            var acc: f32 = @bitCast(Emu.mem[i]);
            for (1..sl.count) |j| acc += @as(f32, @bitCast(Emu.mem[j * sl.stride + i]));
            try out.append(std.testing.allocator, @bitCast(acc));
        }
    }
};

const Step = union(enum) { layer: struct { n: usize, kept: bool = false, collective: bool = false }, mtp: usize, legacy: usize };

/// A sequence of passes over one part buffer; `undo`: put the zero slice back after each spending call (the control).
fn run(path: Path, steps: []const Step, undo: bool, out: *std.ArrayList(u32)) !usize {
    const a = std.testing.allocator;
    const mem = try a.alloc(u32, wt.out_sk * cap * dims);
    defer a.free(mem);
    @memset(mem, poison);
    @memset(mem[cap * dims ..][0 .. cap * dims], 0);
    var r: Rig = undefined;
    r.init(mem);
    for (steps) |st| switch (st) {
        .layer => |l| try r.layer(path, l.n, l.kept, l.collective, out),
        .mtp => |m| {
            var p = r.pass();
            _ = try p.lane(0x10, 0x20, r.native, 0, r.s.part, wt.streams * m, 4);
            if (undo) r.s.part_zero = true;
        },
        .legacy => |m| {
            var p = r.pass();
            _ = try p.lane(0x10, 0x20, r.legacy, 0, r.s.part, m, 4);
            if (undo) r.s.part_zero = true;
        },
    };
    return Emu.memsets;
}

fn same(steps: []const Step) !usize {
    const a = std.testing.allocator;
    var old: std.ArrayList(u32) = .empty;
    defer old.deinit(a);
    var new: std.ArrayList(u32) = .empty;
    defer new.deinit(a);
    const cleared = try run(.old, steps, false, &old);
    const kept = try run(.new, steps, false, &new);
    try std.testing.expectEqualSlices(u32, old.items, new.items);
    return cleared - kept;
}

test "zero slice: native passes at part and full capacity give the old clears' bits and skip their fills" {
    try std.testing.expectEqual(@as(usize, 3), try same(&.{ .{ .layer = .{ .n = 3 } }, .{ .layer = .{ .n = cap } }, .{ .layer = .{ .n = 1 } } }));
}

test "zero slice: an MTP head pass or a legacy lane between native passes spends it, and the old path runs" {
    try std.testing.expectEqual(@as(usize, 1), try same(&.{ .{ .layer = .{ .n = 5 } }, .{ .mtp = 3 }, .{ .layer = .{ .n = cap } }, .{ .layer = .{ .n = 2 } } }));
    try std.testing.expectEqual(@as(usize, 1), try same(&.{ .{ .layer = .{ .n = cap } }, .{ .legacy = cap / 2 }, .{ .layer = .{ .n = 4 } } }));
}

test "zero slice: a kept branch or two ranks keep the old path; only another rank clears the MoE output" {
    try std.testing.expectEqual(@as(usize, 1), try same(&.{ .{ .layer = .{ .n = 4, .kept = true } }, .{ .layer = .{ .n = 4, .collective = true } }, .{ .layer = .{ .n = 4 } } }));
    try std.testing.expect(!zs.clearsY(1, 0));
    try std.testing.expect(zs.clearsY(2, 0) and zs.clearsY(2, 1));
}

test "zero slice control: with the spending calls' invalidation undone, the new path's sums leave the old path's" {
    const a = std.testing.allocator;
    for ([_][3]Step{ .{ .{ .layer = .{ .n = 5 } }, .{ .mtp = 3 }, .{ .layer = .{ .n = cap } } }, .{ .{ .layer = .{ .n = cap } }, .{ .legacy = cap / 2 }, .{ .layer = .{ .n = 4 } } } }) |steps| {
        var old: std.ArrayList(u32) = .empty;
        defer old.deinit(a);
        var new: std.ArrayList(u32) = .empty;
        defer new.deinit(a);
        _ = try run(.old, &steps, false, &old);
        _ = try run(.new, &steps, true, &new);
        try std.testing.expect(!std.mem.eql(u32, old.items, new.items));
    }
}
