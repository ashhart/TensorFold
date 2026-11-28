//! Real prompt-product download through an injected CUDA driver, with poisoned inactive neurons.
const std = @import("std");
const cuda = @import("cuda");
const usage = @import("lanes").pocket_usage;
const Read = @import("cuda_usage.zig").Read;
const kern = @import("cuda_kernels.zig");
const t = std.testing;
const F = struct {
    var reads: usize = 0;
    var waits: usize = 0;
    var fail_read = false;
    fn read(dst: ?*anyopaque, src: u64, n: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        reads += 1;
        if (fail_read) return cuda.abi.error_not_ready;
        @memcpy(@as([*]u8, @ptrCast(dst.?))[0..n], @as([*]const u8, @ptrFromInt(src))[0..n]);
        return 0;
    }
    fn wait(_: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        waits += 1;
        return 0;
    }
    fn launch(_: cuda.abi.Function, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: cuda.abi.Stream, _: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
        @panic("usage must never launch a kernel or forward");
    }
};
test "pocket usage: actual CUDA read only copies existing products, excludes inactive poison and drains a failed copy" {
    var driver: cuda.Driver = undefined;
    driver.api.cuMemcpyDtoHAsync_v2 = F.read;
    driver.api.cuStreamSynchronize = F.wait;
    driver.api.cuLaunchKernel = F.launch;
    var kernels: kern.Kernels = undefined;
    kernels.d = &driver;
    const ops: kern.Ops = .{ .k = &kernels, .s = .{ .d = &driver, .handle = null } };
    var products: [3 * 64]f32 = @splat(std.math.nan(f32));
    products[0] = 0;
    products[64] = usage.activation_floor;
    products[128] = 1;
    const original = products;
    var memory: [3 * 64 * 4]u8 align(16) = undefined;
    var r: Read = .{ .trace = try usage.Capture.init(t.allocator, 4, 3), .stage = .{ .d = &driver, .bytes = &memory } };
    defer r.trace.deinit();
    try r.trace.start(0, 3);
    F.reads = 0;
    F.waits = 0;
    F.fail_read = false;
    try r.read(ops, 2, @intFromPtr(&products), 1);
    try t.expectEqual(@as(usize, 1), F.reads);
    try t.expectEqual(@as(usize, 1), F.waits);
    try t.expect(!(try r.trace.through(2)).has(2, 0));
    const all = try r.trace.through(3);
    try t.expect(all.complete and all.has(2, 0) and !all.has(2, 1));
    try t.expectEqualSlices(u8, std.mem.asBytes(&original), std.mem.asBytes(&products));
    F.fail_read = true;
    defer F.fail_read = false;
    try t.expectError(error.NotReady, r.read(ops, 2, @intFromPtr(&products), 1));
    try t.expectEqual(@as(usize, 2), F.waits);
    F.fail_read = false;
    // The next chunk includes its restored prefix, but never the rows after an interior mark.
    r.trace.restore(3, all);
    try r.trace.start(3, 1);
    try r.read(ops, 1, @intFromPtr(&products), 1);
    const next = try r.trace.through(4);
    try t.expect(next.complete and next.has(2, 0) and !next.has(1, 0));
    r.trace.restore(3, null);
    try t.expect(!(try r.trace.through(3)).complete);
    r.trace.restore(0, .{});
    try r.trace.start(0, 3);
    // Active poison cannot make an unknown usage count eligible for retirement.
    F.fail_read = false;
    try r.read(ops, 2, @intFromPtr(&products), 2);
    try t.expect(!(try r.trace.through(3)).complete);
}

test "pocket usage: warm end snapshots retain their completed prefix without a live read" {
    var e: @import("cuda_engine.zig").Engine = undefined;
    e.usage = null;
    var s: @import("lanes").Stream = undefined;
    s.prompt_len = 7;
    var mask: usage.Masks = .{};
    mask.bits[3] = 2;
    s.pocket_usage = mask;
    const kept = (try @import("cuda_usage.zig").saved(&e, &s, 7)).?;
    try t.expect(kept.complete and kept.has(3, 1));
    try t.expect((try @import("cuda_usage.zig").saved(&e, &s, 6)) == null);
    try t.expect((try @import("cuda_usage.zig").saved(&e, null, 7)) == null);
}

test "pocket usage: actual native CUDA gating keeps slide-only on the ordinary prompt path" {
    const native = @import("cuda_native.zig");
    for ([_]bool{ false, true }) |slide| for ([_]bool{ false, true }) |forget| {
        const o: native.Options = .{ .context = 64, .drafts = false, .slide = slide, .slide_forget = forget };
        try t.expectEqual(slide and forget, native.usageEnabled(o, false, 0));
        try t.expect(!native.usageEnabled(o, true, 0));
        try t.expect(!native.usageEnabled(o, false, 1));
    };
}

test "pocket usage: usage readers retain final MoE blocks for wide no-MTP prompt chunks" {
    const prompt = @import("cuda_prompt.zig");
    const tail = @import("core").prompt_tail;
    var e: @import("cuda_engine.zig").Engine = undefined;
    var r: Read = undefined;
    e.taps = null;
    e.prompt_rows = null;
    e.usage = &r;
    for ([_]usize{ 2049, 4096 }) |rows| {
        const active = prompt.readers(&e, null, null, false);
        try t.expect(active);
        try t.expect(!tail.unread(.{ .logits = false, .readers = active }));
        try t.expect(!tail.unread(.{ .logits = true, .readers = active }));
        try t.expect(rows > 2048);
    }
    e.usage = null;
    try t.expect(tail.unread(.{ .logits = false, .readers = prompt.readers(&e, null, null, false) }));
}

const Stepped = struct {
    next: [2]usize = .{ 0, 0 },
    prefixes: [2][2]?usage.Masks = @splat(@splat(null)),

    fn index(s: *const @import("lanes").Stream) usize {
        return if (s.id[0] == 'a') 0 else 1;
    }
    fn read(_: *anyopaque, r: *Read, layer: usize, ptr: u64, width: usize) !void {
        const products: [*]const f32 = @ptrFromInt(ptr);
        try r.trace.layer(layer, products[0 .. r.trace.rows * 64], width, 64);
    }
    fn run(ptr: *anyopaque, b: *@import("cuda_lanes.zig").Cuda, s: *@import("lanes").Stream) anyerror!bool {
        const self: *Stepped = @ptrCast(@alignCast(ptr));
        const i = index(s);
        const chunk = self.next[i];
        const at = chunk * 2;
        if (b.e.usage) |r| {
            try r.trace.start(at, 2);
            var products: [2 * 64]f32 = @splat(0);
            products[i * 2 + chunk] = 2;
            const ops: @import("cuda_kernels.zig").Ops = undefined;
            try r.read(ops, 3, @intFromPtr(&products), 4);
            self.prefixes[i][chunk] = try @import("cuda_usage.zig").saved(b.e, s, at + 1);
            r.trace.restore(at + 2, try r.trace.through(at + 2));
        }
        self.next[i] += 1;
        return self.next[i] == 2;
    }
};

test "pocket usage: two foreground stepped streams retain independent row unions and cache marks" {
    const lanes = @import("lanes");
    const cuda_lanes = @import("cuda_lanes.zig");
    const cuda_usage = @import("cuda_usage.zig");
    var e: @import("cuda_engine.zig").Engine = undefined;
    e.usage = null;
    var fixture: Stepped = .{};
    var b: cuda_lanes.Cuda = undefined;
    b.gpa = t.allocator;
    b.e = &e;
    b.usage_enabled = true;
    b.usage_reads = .empty;
    defer b.usage_reads.deinit(t.allocator);
    b.test_step = .{ .ctx = &fixture, .run = Stepped.run, .read = Stepped.read };
    const ids_a = [_]u32{ 10, 11, 12, 13 };
    const ids_b = [_]u32{ 20, 21, 22, 23 };
    var a = try lanes.Stream.init(t.allocator, .{ .id = "a", .prompt = &ids_a, .max_new = 1 });
    defer a.deinit(t.allocator);
    var b_stream = try lanes.Stream.init(t.allocator, .{ .id = "b", .prompt = &ids_b, .max_new = 1 });
    defer b_stream.deinit(t.allocator);
    try t.expect(!try cuda_usage.prefillStep(&b, &a, 32));
    try t.expect(!try cuda_usage.prefillStep(&b, &b_stream, 32));
    try t.expect(try cuda_usage.prefillStep(&b, &a, 32));
    try t.expect(try cuda_usage.prefillStep(&b, &b_stream, 32));
    for ([_]*lanes.Stream{ &a, &b_stream }, 0..) |s, i| {
        const all = s.pocket_usage orelse return error.MissingSteppedUsage;
        try t.expect(all.complete);
        try t.expect(all.has(3, i * 2) and all.has(3, i * 2 + 1));
        try t.expect(!all.has(3, (1 - i) * 2) and !all.has(3, (1 - i) * 2 + 1));
        for (fixture.prefixes[i], 0..) |saved, chunk| {
            const prefix = saved orelse return error.MissingSteppedCacheMark;
            try t.expect(prefix.complete);
            try t.expect(prefix.has(3, i * 2 + chunk));
        }
    }
    try t.expectEqualSlices(u32, &ids_a, a.prompt());
    try t.expectEqualSlices(u32, &ids_b, b_stream.prompt());
    try t.expectEqual(@as(usize, 0), b.usage_reads.count());
    try t.expect(e.usage == null);
}
