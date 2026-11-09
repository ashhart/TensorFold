//! Failure paths on a recording fake device: no buffer freed before completion, no draw over non-finite logits.
const std = @import("std");
const cuda = @import("cuda");
const mdl = @import("model.zig");
const kern = @import("kernels.zig");
const draws = @import("draws.zig");
const Geometry = @import("recipe.zig").Geometry;
const t = std.testing;

pub const Fake = struct {
    pub const queue: cuda.abi.Stream = @ptrFromInt(0x1800);
    const base: u64 = 0x100000;
    var mem: [4096]u8 = undefined; // device bytes from `base` on, for downloads
    var next: u64 = 0x200000;
    pub var live: usize = 0;
    pub var pending = false;
    var freed_pending: usize = 0;
    pub var proofs: usize = 0;
    var fail_wait = false;
    var argmax: usize = 0;
    var drawn: usize = 0;
    fn reset() void {
        live = 0;
        pending = false;
        freed_pending = 0;
        proofs = 0;
        fail_wait = false;
        argmax = 0;
        drawn = 0;
    }
    fn alloc(out: *u64, len: usize) callconv(.c) c_int {
        out.* = next;
        next += len + 256;
        live += 1;
        return 0;
    }
    fn free(_: u64) callconv(.c) c_int {
        freed_pending += @intFromBool(pending);
        live -= 1;
        return 0;
    }
    fn fill(_: u64, _: u8, _: usize, _: cuda.abi.Stream) callconv(.c) c_int {
        pending = true;
        return 0;
    }
    fn host(out: *?*anyopaque, len: usize, _: c_uint) callconv(.c) c_int {
        out.* = std.c.malloc(len) orelse return cuda.abi.error_not_ready;
        return 0;
    }
    fn hostFree(p: ?*anyopaque) callconv(.c) c_int {
        std.c.free(p);
        return 0;
    }
    fn copy(_: u64, _: ?*const anyopaque, _: usize, _: cuda.abi.Stream) callconv(.c) c_int {
        pending = true;
        return 0;
    }
    fn wait(_: cuda.abi.Stream) callconv(.c) c_int {
        if (fail_wait) return cuda.abi.error_not_ready;
        pending = false;
        return 0;
    }
    fn context() callconv(.c) c_int {
        proofs += 1;
        pending = false;
        return 0;
    }
    fn event(out: *cuda.abi.Event, _: c_uint) callconv(.c) c_int {
        out.* = @ptrFromInt(1);
        return 0;
    }
    fn record(_: cuda.abi.Event, _: cuda.abi.Stream) callconv(.c) c_int {
        return 0;
    }
    fn query(_: cuda.abi.Event) callconv(.c) c_int {
        return if (pending) cuda.abi.error_not_ready else 0;
    }
    fn destroy(_: cuda.abi.Event) callconv(.c) c_int {
        return 0;
    }
    fn launch(f: cuda.abi.Function, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: cuda.abi.Stream, _: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) c_int {
        argmax += @intFromBool(f == @as(cuda.abi.Function, @ptrFromInt(1)));
        drawn += @intFromBool(f == @as(cuda.abi.Function, @ptrFromInt(2)));
        return 0;
    }
    fn down(dst: ?*anyopaque, src: u64, len: usize) callconv(.c) c_int {
        @memcpy(@as([*]u8, @ptrCast(dst.?))[0..len], mem[src - base ..][0..len]);
        return 0;
    }
    fn device(out: *cuda.abi.Device) callconv(.c) c_int {
        out.* = 0;
        return 0;
    }
    fn attribute(out: *c_int, _: cuda.abi.DeviceAttribute, _: cuda.abi.Device) callconv(.c) c_int {
        out.* = 0;
        return 0;
    }
    pub fn driver() cuda.Driver {
        reset();
        var d: cuda.Driver = .{ .lib = undefined, .api = undefined };
        d.api.cuMemAlloc_v2 = alloc;
        d.api.cuMemFree_v2 = free;
        d.api.cuMemsetD8Async = fill;
        d.api.cuMemHostAlloc = host;
        d.api.cuMemFreeHost = hostFree;
        d.api.cuMemcpyHtoDAsync_v2 = copy;
        d.api.cuStreamSynchronize = wait;
        d.api.cuCtxSynchronize = context;
        d.api.cuEventCreate = event;
        d.api.cuEventRecord = record;
        d.api.cuEventQuery = query;
        d.api.cuEventDestroy_v2 = destroy;
        d.api.cuLaunchKernel = launch;
        d.api.cuMemcpyDtoH_v2 = down;
        d.api.cuCtxGetDevice = device;
        d.api.cuDeviceGetAttribute = attribute;
        return d;
    }
};

const tiny: Geometry = .{ .hidden = 256, .layers = 2, .heads = 4, .kv_heads = 2, .head_dim = 64, .vocab = 512, .eps = 1e-6, .context = 4096, .acc = 512, .wide = 1024, .mid = 512, .theta = 1e4 };

pub fn lane() cuda.qmmf.Lane {
    var l: cuda.qmmf.Lane = undefined;
    l.clusters = false; // partBytes reads the tile plan alone
    return l;
}

test "a failed scratch upload wait proves completion by the context before init frees its buffers" {
    var d = Fake.driver();
    const s: cuda.Stream = .{ .d = &d, .handle = Fake.queue };
    const l = lane();
    Fake.fail_wait = true;
    try t.expectError(error.NotReady, mdl.Model.init(t.allocator, &d, s, tiny, 4096, &l));
    try t.expectEqual(@as(usize, 0), Fake.freed_pending);
    try t.expectEqual(@as(usize, 0), Fake.live);
    try t.expect(Fake.proofs >= 1);
    Fake.fail_wait = false;
    try s.synchronize();
    cuda.copy_source.dropSpares(&d, cuda.copy_source.current());
}

test "a failed attach wait proves completion before its arena and tables go, and leaves the model attachable" {
    var d = Fake.driver();
    const s: cuda.Stream = .{ .d = &d, .handle = Fake.queue };
    const l = lane();
    var m = try mdl.Model.init(t.allocator, &d, s, tiny, 4096, &l);
    const held = Fake.live;
    Fake.fail_wait = true;
    try t.expectError(error.NotReady, m.attach(s, 1, 2));
    try t.expectEqual(@as(usize, 0), Fake.freed_pending);
    try t.expectEqual(held, Fake.live);
    try t.expect(m.pages == 0 and m.slots == 0 and m.ids.len == 0 and m.arena == 0);
    Fake.fail_wait = false;
    try m.attach(s, 1, 2);
    try t.expect(m.pages == 2 and m.slots == 2);
    m.deinit();
    try t.expectEqual(@as(usize, 0), Fake.live);
    cuda.copy_source.dropSpares(&d, cuda.copy_source.current());
}

fn buffer(d: *const cuda.Driver, offset: u64, len: usize) cuda.DeviceBuffer {
    return .{ .d = d, .ptr = Fake.base + offset, .len = len, .borrowed = true };
}

fn function(d: *const cuda.Driver, id: usize) cuda.Function {
    return .{ .d = d, .handle = @ptrFromInt(id) };
}

test "a sampled row over NaN or an infinity is refused before any draw reads it, as a greedy one is" {
    var d = Fake.driver();
    const s: cuda.Stream = .{ .d = &d, .handle = Fake.queue };
    var k: kern.Kernels = undefined;
    k.round = .{ .argmax_fn = function(&d, 1), .match_fn = function(&d, 9), .keep_fn = function(&d, 9), .gather_fn = function(&d, 9), .head_fn = function(&d, 9), .ids_fn = function(&d, 9) };
    k.draws = .{ .draw_fn = function(&d, 2), .draw_ids_fn = function(&d, 9), .logprob_rows_fn = function(&d, 9), .topk_slices_fn = function(&d, 9), .topk_merge_fn = function(&d, 9) };
    var greedy: [draws.ring + mdl.round_rows]bool = @splat(true);
    var x: draws.Draws = .{ .d = &d, .k = &k, .s = s, .vocab = 512, .tok = buffer(&d, 0, 64), .picks = buffer(&d, 64, 128), .greedy = &greedy, .meta = buffer(&d, 512, 64) };
    const Pick = cuda.tree_round.Pick;
    var picks = [2]Pick{ .{ .token = 3, .nonfinite = 0 }, .{ .token = 4, .nonfinite = 0 } };
    @memcpy(Fake.mem[64..][0..16], std.mem.sliceAsBytes(&picks)); // what the argmax writes for finite rows
    @memcpy(Fake.mem[0..8], std.mem.sliceAsBytes(&[2]u32{ 7, 8 }));
    try x.rows(true, 0x9000, 0xa000, 0, 2, 5, 0, 0);
    try t.expectEqual(@as(usize, 1), Fake.argmax); // the sampled rows' non-finite flags
    try t.expectEqual(@as(usize, 1), Fake.drawn);
    var out: [2]u32 = undefined;
    try x.tokens(0, &out);
    try t.expectEqualSlices(u32, &.{ 7, 8 }, &out); // drawn, not picked
    for ([_]u32{ 1, 2, 4 }) |mask| { // NaN, +inf, -inf (tree_round.cu's mask)
        picks[1] = .{ .token = 0xFFFFFFFF, .nonfinite = mask };
        @memcpy(Fake.mem[64..][0..16], std.mem.sliceAsBytes(&picks));
        try t.expectError(error.NonFiniteLogits, x.rows(true, 0x9000, 0xa000, 0, 2, 5, 0, 0));
        try t.expectEqual(@as(usize, 1), Fake.drawn); // tf_draw never saw the row
    }
    try x.rows(false, 0, 0xa000, 0, 2, 5, 0, 0);
    try t.expectError(error.NonFiniteLogits, x.tokens(0, &out));
    try t.expectEqual(@as(usize, 1), Fake.drawn);
    picks[1] = .{ .token = 4, .nonfinite = 0 };
    @memcpy(Fake.mem[64..][0..16], std.mem.sliceAsBytes(&picks));
    try x.tokens(0, &out);
    try t.expectEqualSlices(u32, &.{ 3, 4 }, &out);
    cuda.copy_source.dropSpares(&d, cuda.copy_source.current());
}
