//! Exact rank gathers over recorded launches, with the unsplit lane as the control.
const std = @import("std");
const builtin = @import("builtin");
const cuda = @import("cuda");
const wt = @import("cuda_prompt_wt.zig");
const parts = @import("cuda_batch_state.zig");
const batch = @import("cuda_batch.zig");
const split_lane = @import("cuda_split_lane.zig");
const Layer = @import("cuda_prompt_load.zig").Layer;
const Cut = @import("cuda_modelopt_experts.zig").Cut;
const Collective = @import("cuda_collective.zig").Collective;
const nccl = cuda.nccl;

const dims = wt.dims;
const out_k = wt.out_k;
const qmmf_id = 0x100;
const reduce_id = 0x200;
const legacy_id = 0x300;
const copy_id = 0x400;
const strided_id = 0x500;
const x_at: u64 = 0x10_0000;
const part_at: u64 = 0x2000_0000;
const native_part_at: u64 = 0x4000_0000;

/// One recorded call: a launch's function and the arguments that place it, a memset, or a rank sum.
const Call = struct { kind: enum { launch, memset, sum, gather }, f: usize = 0, a: [6]i64 = @splat(0) };

const Rec = struct {
    var calls: [64]Call = undefined;
    var n: usize = 0;

    fn arg(comptime T: type, args: ?[*]?*anyopaque, i: usize) i64 {
        const v = @as(*const T, @ptrCast(@alignCast(args.?[i].?))).*;
        return if (T == u64) @bitCast(v) else @intCast(v);
    }

    fn push(c: Call) void {
        calls[n] = c;
        n += 1;
    }

    fn launch(f: cuda.abi.Function, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: cuda.abi.Stream, args: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) c_int {
        const id = @intFromPtr(f);
        push(.{
            .kind = .launch,
            .f = id,
            .a = switch (id) {
                // x, y, part, m, n and k as n * 10^6 + k, and the row stride a qmmf tile reads
                qmmf_id => .{ arg(u64, args, 0), arg(u64, args, 4), arg(u64, args, 5), arg(i32, args, 6), arg(i32, args, 7) * 1_000_000 + arg(i32, args, 8), arg(i32, args, 11) },
                strided_id => .{ arg(u64, args, 0), arg(u64, args, 3), arg(i32, args, 4), arg(i32, args, 5) * 1_000_000 + arg(i32, args, 6), arg(i32, args, 10), arg(i32, args, 8) },
                copy_id => .{ arg(u64, args, 0), arg(i64, args, 1), arg(u64, args, 2), arg(i64, args, 3), arg(i32, args, 4), 0 },
                reduce_id => .{ arg(u64, args, 0), arg(u64, args, 1), arg(i64, args, 2), arg(i32, args, 3), 0, 0 },
                else => .{ arg(u64, args, 0), arg(u64, args, 6), arg(c_int, args, 7), arg(c_int, args, 8), arg(c_int, args, 9), arg(c_int, args, 10) },
            },
        });
        return 0;
    }

    fn memset(ptr: u64, _: u8, count: usize, _: cuda.abi.Stream) callconv(.c) c_int {
        push(.{ .kind = .memset, .a = .{ @bitCast(ptr), @intCast(count), 0, 0, 0, 0 } });
        return 0;
    }

    fn allReduce(send: cuda.abi.DevicePtr, recv: cuda.abi.DevicePtr, count: usize, dt: nccl.DataType, op: nccl.RedOp, _: nccl.Comm, _: cuda.abi.Stream) callconv(.c) c_int {
        push(.{ .kind = .sum, .a = .{ @bitCast(send), @bitCast(recv), @intCast(count), @backingInt(dt), @backingInt(op), 0 } });
        return 0;
    }

    fn gather(send: u64, recv: u64, count: usize, dt: nccl.DataType, _: nccl.Comm, _: cuda.abi.Stream) callconv(.c) c_int {
        push(.{ .kind = .gather, .a = .{ @bitCast(send), @bitCast(recv), @intCast(count), @backingInt(dt), 0, 0 } });
        return 0;
    }

    fn asyncError(_: nccl.Comm, state: *c_int) callconv(.c) c_int {
        state.* = 0;
        return 0;
    }

    fn abort(_: nccl.Comm) callconv(.c) c_int {
        return 0;
    }
};

const Rig = struct {
    d: cuda.Driver,
    k: parts.Kernels,
    s: parts.Scratch,
    c: Collective,
    pair: [2]c_int,

    fn init(r: *Rig) !void {
        r.d = undefined;
        r.d.api.cuLaunchKernel = Rec.launch;
        r.d.api.cuMemsetD8Async = Rec.memset;
        const f = struct {
            fn of(d: *const cuda.Driver, id: usize) cuda.Function {
                return .{ .d = d, .handle = @ptrFromInt(id) };
            }
        }.of;
        r.k = undefined;
        r.k.copy_cols = f(&r.d, copy_id);
        r.k.lane = f(&r.d, legacy_id);
        r.k.tiles.fact = f(&r.d, legacy_id + 1);
        r.k.tiles.fact_fold = f(&r.d, legacy_id + 2);
        @memset(std.mem.asBytes(&r.k.native_lane), 0);
        r.k.native_lane.fns = @splat(@splat(f(&r.d, qmmf_id)));
        r.k.native_lane.f32_fns = @splat(@splat(f(&r.d, qmmf_id)));
        r.k.native_lane.reduce = @splat(f(&r.d, reduce_id));
        r.k.native_lane.clusters = false;
        r.s = undefined;
        r.s.cap = 256;
        r.s.part = part_at;
        r.s.native_part = native_part_at;
        r.s.dummy = 0x99;
        r.s.part_zero = true;
        r.s.rank_input = 0x5000_0000;
        r.s.rank_gather = 0x6000_0000;
        r.s.rank_output = 0x7000_0000;
        r.s.rank_result = 0x8000_0000;
        // the peer's end of a socket pair stays open, so the sum's peer check passes
        if (std.c.socketpair(std.posix.AF.UNIX, std.posix.SOCK.STREAM, 0, &r.pair) != 0) return error.SocketPair;
        r.c = undefined;
        r.c.rank = 0;
        r.c.failed = false;
        r.c.comm = @ptrFromInt(0x1000);
        r.c.link = .{ .fd = r.pair[0] };
        r.c.timeout_ms = 1000;
        r.c.lib.api.ncclAllReduce = Rec.allReduce;
        r.c.lib.api.ncclAllGather = Rec.gather;
        r.c.api.ncclCommGetAsyncError = Rec.asyncError;
        r.c.api.ncclCommAbort = Rec.abort;
        Rec.n = 0;
    }

    fn deinit(r: *Rig) void {
        for (r.pair) |fd| _ = std.c.close(fd);
    }

    fn pass(r: *Rig, rows: usize, collective: bool) batch.Pass(void) {
        const m: parts.Model = .{ .collective = if (collective) &r.c else null, .table_w = 0, .table_s = 0, .table_b = 0, .layers = &.{}, .mixer = undefined, .head = undefined, .ngram = undefined };
        return .{ .gpa = std.testing.allocator, .io = undefined, .stream = .{ .d = &r.d, .handle = null }, .tri = {}, .k = &r.k, .s = &r.s, .m = m, .seq = undefined, .hooks = undefined, .rows = rows };
    }

    fn tryPin(w: cuda.qmmf.Weight) cuda.qmmf.Weight {
        return cuda.qmmf.pinned(w, cuda.qmmf.splitK(dims, out_k)) catch unreachable;
    }

    fn layer(r: *Rig, k: usize, native: bool, cut: ?Cut) Layer {
        const z: cuda.DeviceBuffer = .{ .d = &r.d, .ptr = 0, .len = 0 };
        var lw: Layer = undefined;
        lw.heads = if (cut != null) .half else .full;
        lw.out = .{ .w = z, .s = z, .b = z, .n = dims, .k = k, .native = if (native) cuda.modelopt.bf16Weight(0x2000, dims, k) else null };
        if (cut != null) {
            lw.out.n = dims / 2;
            if (native) lw.out.native = tryPin(cuda.modelopt.bf16Weight(0x2000, dims / 2, k));
        }
        lw.dense_cut = cut;
        return lw;
    }
};

fn recorded() []const Call {
    return Rec.calls[0..Rec.n];
}

test "split out lane gathers full BF16 inputs and disjoint FP32 outputs without a reduction" {
    if (builtin.os.tag == .windows) return error.SkipZigTest;
    var r: Rig = undefined;
    try r.init();
    defer r.deinit();
    for ([_]usize{ 1, 4, 17, 33, 64, 256 }) |n| for (0..2) |rank| {
        r.c.rank = rank;
        const cut = try Cut.init(dims, 2, rank);
        const lw = r.layer(out_k, true, cut);
        var p = r.pass(n, true);
        Rec.n = 0;
        try std.testing.expectEqual(@as(usize, 1), try split_lane.out(&p, x_at, 0x20, &lw));
        const calls = recorded();
        try std.testing.expect(calls.len == 7 or calls.len == 8);
        try std.testing.expectEqual(Call{ .kind = .gather, .a = .{ x_at, @intCast(r.s.rank_gather), @intCast(n * out_k / 2), @backingInt(nccl.DataType.bf16), 0, 0 } }, calls[0]);
        for (0..2) |peer| {
            const c = calls[1 + peer];
            try std.testing.expectEqual(copy_id, c.f);
            try std.testing.expectEqualSlices(i64, &.{ @intCast(r.s.rank_gather + peer * n * out_k), out_k / 2, @intCast(r.s.rank_input + peer * out_k), out_k, out_k / 2, 0 }, &c.a);
        }
        const q = calls[3];
        try std.testing.expectEqual(qmmf_id, q.f);
        try std.testing.expectEqual(@as(i64, @intCast(r.s.rank_input)), q.a[0]);
        try std.testing.expectEqual(@as(i64, 1280 * 1_000_000 + 6144), q.a[4]);
        try std.testing.expectEqual(@as(i64, out_k), q.a[5]);
        const at = calls.len - 3;
        try std.testing.expectEqual(Call{ .kind = .gather, .a = .{ @intCast(r.s.rank_output), @intCast(r.s.rank_result), @intCast(n * dims / 2), @backingInt(nccl.DataType.f32), 0, 0 } }, calls[at]);
        for (0..2) |peer| {
            const c = calls[at + 1 + peer];
            try std.testing.expectEqual(copy_id, c.f);
            try std.testing.expectEqualSlices(i64, &.{ @intCast(r.s.rank_result + peer * n * dims * 2), dims, @intCast(part_at + peer * dims * 2), dims * 2, dims, 0 }, &c.a);
        }
        for (calls) |c| try std.testing.expect(c.kind != .sum and c.kind != .memset);
    };
}

test "split out lane: an unsplit layer launches exactly what Pass.lane launches, with or without a collective" {
    if (builtin.os.tag == .windows) return error.SkipZigTest;
    var r: Rig = undefined;
    try r.init();
    defer r.deinit();
    var want: [64]Call = undefined;
    for ([_]usize{ 1, 4, 64, 256 }) |n| for ([_]bool{ true, false }) |native| for ([_]bool{ true, false }) |collective| {
        const lw = r.layer(out_k, native, null);
        var p = r.pass(n, collective);
        Rec.n = 0;
        const lane_slices = try p.lane(x_at, 0x20, lw.out, r.s.dummy, r.s.part, n, wt.out_sk);
        const count = Rec.n;
        @memcpy(want[0..count], recorded());
        Rec.n = 0;
        var q = r.pass(n, collective);
        try std.testing.expectEqual(lane_slices, try split_lane.out(&q, x_at, 0x20, &lw));
        try std.testing.expectEqual(count, Rec.n);
        for (want[0..count], recorded()) |a, b| {
            try std.testing.expectEqual(a.kind, b.kind);
            try std.testing.expectEqual(a.f, b.f);
            try std.testing.expectEqualSlices(i64, &a.a, &b.a);
            try std.testing.expect(b.kind != .sum);
        }
    };
}

test "split out lane: a split layer refuses a missing collective or a non-native weight" {
    if (builtin.os.tag == .windows) return error.SkipZigTest;
    var r: Rig = undefined;
    try r.init();
    defer r.deinit();
    const cut = try Cut.init(dims, 2, 0);
    const native = r.layer(out_k, true, cut);
    const legacy = r.layer(out_k, false, cut);
    var p = r.pass(4, false);
    try std.testing.expectError(error.TensorParallelUnavailable, split_lane.out(&p, x_at, 0x20, &native));
    var q = r.pass(4, true);
    try std.testing.expectError(error.UnexpectedTensor, split_lane.out(&q, x_at, 0x20, &legacy));
    try std.testing.expectEqual(@as(usize, 0), Rec.n);
}

const SharedGlue = struct {
    pub fn sharedSwiGLU(_: *SharedGlue, _: cuda.Stream, _: u64, _: u64, _: u64, _: usize, _: usize) !void {}
    pub fn sharedScatter(_: *SharedGlue, _: cuda.Stream, input: u64, output: u64, rows: usize, width: usize, slots: usize) !void {
        Rec.push(.{ .kind = .launch, .f = 0x600, .a = .{ @intCast(input), @intCast(output), @intCast(rows), @intCast(width), @intCast(slots), 0 } });
    }
};

test "shared expert composition keeps whole direct slots and split full-K output copies" {
    if (builtin.os.tag == .windows) return error.SkipZigTest;
    var r: Rig = undefined;
    try r.init();
    defer r.deinit();
    var glue: SharedGlue = .{};
    r.s.mixed = x_at;
    r.s.shared_gate = 0x9000_0000;
    r.s.shared_up = 0x9100_0000;
    r.s.shared_act = 0x9200_0000;
    r.s.shared_out = 0x9300_0000;
    r.s.y = 0x9400_0000;
    for (&r.k.native_lane.shared_down) |*f| f.* = .{ .d = &r.d, .handle = @ptrFromInt(strided_id) };
    const Self = struct {
        k: *parts.Kernels,
        s: *parts.Scratch,
        m: parts.Model,
        stream: cuda.Stream,
        tri: struct { own: ?*SharedGlue },
    };
    for ([_]bool{ false, true }) |split| for ([_]bool{ false, true }) |prompt_loaded| for ([_]usize{ 64, 512 }) |n| for (0..2) |rank| {
        r.c.rank = rank;
        r.k.native_lane.prompt = if (prompt_loaded) @splat(@splat(cuda.Function{ .d = &r.d, .handle = @ptrFromInt(qmmf_id) })) else null;
        var native: @import("cuda_mixed_experts.zig").Owned = undefined;
        native.rank = rank;
        native.split_shared = split;
        const width: usize = if (split) wt.moe_w / 2 else wt.moe_w;
        const columns: usize = if (split) dims / 2 else dims;
        for (0..3) |i| native.shared[i].native = cuda.modelopt.bf16Weight(0x2000, if (i == 2) columns else width, if (i == 2) wt.moe_w else dims);
        var self: Self = .{ .k = &r.k, .s = &r.s, .m = r.pass(n, true).m, .stream = .{ .d = &r.d, .handle = null }, .tri = .{ .own = &glue } };
        Rec.n = 0;
        try @import("cuda_shared_split.zig").run(&self, &native, n);
        if (!split and rank != 0) {
            try std.testing.expectEqual(@as(usize, 0), Rec.n);
            continue;
        }
        const direct = !split and !(prompt_loaded and n >= cuda.qmmf.prompt_rows);
        const last = recorded()[Rec.n - 1];
        if (direct) {
            try std.testing.expectEqual(strided_id, last.f);
            try std.testing.expectEqualSlices(i64, &.{ @intCast(r.s.shared_act), @intCast(r.s.y + (wt.slots - 1) * dims * 4), @intCast(n), dims * 1_000_000 + wt.moe_w, wt.slots * dims, wt.moe_w }, &last.a);
        } else {
            const down = recorded()[Rec.n - 2];
            try std.testing.expectEqual(qmmf_id, down.f);
            try std.testing.expectEqual(@as(i64, @intCast(columns * 1_000_000 + wt.moe_w)), down.a[4]);
            try std.testing.expectEqual(@as(i64, wt.moe_w), down.a[5]);
            if (split) {
                try std.testing.expectEqual(copy_id, last.f);
                try std.testing.expectEqualSlices(i64, &.{ @intCast(r.s.shared_out), dims, @intCast(r.s.y + (wt.slots - 1) * dims * 4 + rank * dims * 2), wt.slots * dims * 2, dims, 0 }, &last.a);
                var gather_at: ?usize = null;
                for (recorded(), 0..) |call, i| if (call.kind == .gather) {
                    try std.testing.expect(gather_at == null);
                    gather_at = i;
                };
                const at = gather_at orelse return error.TestExpectedGather;
                try std.testing.expectEqual(at + 5, Rec.n);
                try std.testing.expectEqual(Call{ .kind = .gather, .a = .{ @intCast(r.s.shared_act), @intCast(r.s.rank_gather), @intCast(n * width), @backingInt(nccl.DataType.bf16), 0, 0 } }, recorded()[at]);
                for (0..2) |peer| {
                    const copy = recorded()[at + 1 + peer];
                    try std.testing.expectEqual(copy_id, copy.f);
                    try std.testing.expectEqualSlices(i64, &.{ @intCast(r.s.rank_gather + peer * n * width * 2), @intCast(width), @intCast(r.s.rank_input + peer * width * 2), wt.moe_w, @intCast(width), 0 }, &copy.a);
                }
                try std.testing.expectEqual(@as(i64, @intCast(r.s.rank_input)), down.a[0]);
                try std.testing.expectEqual(@as(i64, @intCast(r.s.shared_out)), down.a[1]);
                for (recorded()) |c| try std.testing.expect(c.f != strided_id and c.kind != .sum);
            } else try std.testing.expectEqual(@as(usize, 0x600), last.f);
        }
    };
}
