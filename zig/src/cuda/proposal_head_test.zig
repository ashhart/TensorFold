const std = @import("std");
const q = @import("qlinear.zig");
const h = q.proposal;
const qmmf = @import("qmmf.zig");
const abi = @import("abi.zig");
const Driver = @import("driver.zig").Driver;
const Function = @import("module.zig").Function;
const Stream = @import("stream.zig").Stream;
const sample = @import("sample.zig");
const t = std.testing;
const gpa = t.allocator;
const Fixture = struct {
    storage: enum { bf16, affine } = .bf16,
    recipe: enum { qlinear_decode, qlinear_prompt, unsupported } = .qlinear_decode,
    fingerprint: [32]u8 = @splat(1),
    queue: u64 = 0,
    output_f32: bool = false,
    bits: u8 = 16,
    group: u32 = 0,
    physical: u32 = 128,
    vocab: u32 = 11,
    hidden: u32 = 64,
    ids: []const u32 = &.{ 0, 3, 7 },
    codes: []align(16) const u8,
};
fn source() q.Weight {
    return .{ .bf16 = .{ .mode = .bf16, .codes = 0x2000, .scales = 0, .n = 11, .k = 64, .npad = 128 } };
}
const Trace = struct {
    const Call = struct { id: usize, grid: [3]u32, block: [3]u32, stream: usize, shared: u32, args: [13]u64 = @splat(0) };
    var calls: [80]Call = undefined;
    var n: usize = 0;
    fn read(comptime T: type, a: ?*anyopaque) T {
        return @as(*const T, @ptrCast(@alignCast(a.?))).*;
    }
    fn launch(f: abi.Function, gx: c_uint, gy: c_uint, gz: c_uint, bx: c_uint, by: c_uint, bz: c_uint, sm: c_uint, stream: abi.Stream, args: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) abi.Result {
        const id = @intFromPtr(f);
        const a = args.?;
        var call = Call{ .id = id, .grid = .{ gx, gy, gz }, .block = .{ bx, by, bz }, .stream = @intFromPtr(stream), .shared = sm };
        if (id < 20) {
            inline for (.{ 0, 1, 2, 4, 5 }) |i| call.args[i] = read(u64, a[i]);
            call.args[3] = read(u32, a[3]);
            inline for (6..13) |i| call.args[i] = read(u32, a[i]);
        } else if (id == 82) {
            call.args[0] = read(u64, a[0]);
            call.args[1] = read(u32, a[1]);
            call.args[2] = read(u32, a[2]);
            call.args[3] = read(u64, a[3]);
            call.args[4] = read(u64, a[4]);
            call.args[5] = read(u32, a[5]);
            inline for (6..9) |i| call.args[i] = read(u64, a[i]);
        } else {
            call.args[0] = read(u64, a[0]);
            call.args[1] = read(u32, a[1]);
            call.args[2] = read(u64, a[2]);
            call.args[3] = read(u64, a[3]);
            call.args[4] = read(u32, a[4]);
            call.args[5] = read(u64, a[5]);
            call.args[6] = read(u64, a[6]);
            if (id == 81) call.args[7] = read(u64, a[7]);
        }
        calls[n] = call;
        n += 1;
        return 0;
    }
    fn function(d: *const Driver, id: usize) Function {
        return .{ .d = d, .handle = @ptrFromInt(id) };
    }
};
test "proposal rows: qlinear packs exact selected BF16 bits and full view launches remain byte equal" {
    var words: [3 * 64]u16 align(16) = undefined;
    for (&words, 0..) |*x, i| x.* = @truncate(i *% 113);
    words[0] = 0x8000;
    words[1] = 0x7fc1;
    var tiled = try h.bf16(gpa, Fixture{ .codes = std.mem.sliceAsBytes(&words) }, source());
    defer tiled.deinit();
    var expected: [128 * 64]u16 = undefined;
    qmmf.packBf16(&expected, &words, 3, 64);
    try t.expectEqualSlices(u16, &expected, tiled.words);
    var d: Driver = undefined;
    d.api.cuLaunchKernel = Trace.launch;
    var lane: qmmf.Lane = undefined;
    lane.clusters = false;
    for (&lane.fns) |*set| {
        for (set, 0..) |*f, i| f.* = Trace.function(&d, i + 1);
    }
    lane.half = @splat(@splat(@splat(Trace.function(&d, 120))));
    lane.half_below = 0; // the released launch parameters: the 64-column tile's
    const linear: q.Linear = .{ .lane = &lane };
    const s: Stream = .{ .d = &d, .handle = null };
    for ([_]usize{ 1, 3, 16, 17, 32, 64 }) |rows| {
        Trace.n = 0;
        try linear.decode(s, source(), .{ .x = 0x1000, .part = 0x4000 }, 0x3000, rows);
        const baseline = Trace.calls[0];
        try t.expectEqual(@as(usize, 1), Trace.n);
        Trace.n = 0;
        try h.Head.full(source(), .decode).project(linear, s, .{ .x = 0x1000, .part = 0x4000 }, 0x3000, rows);
        try t.expectEqual(@as(usize, 1), Trace.n);
        try t.expectEqualDeep(baseline, Trace.calls[0]);
        Trace.n = 0;
        try tiled.view(0x9000).project(linear, s, .{ .x = 0x1000, .part = 0x4000 }, 0x3000, rows);
        try t.expectEqual(@as(u64, 128), Trace.calls[0].args[7]);
        try t.expectEqual(@as(u64, 1), Trace.calls[0].args[9]);
        try t.expectEqual(tiled.words.len * 2, @as(usize, 128 * 64 * 2));
    }
    var prompt = try h.bf16(gpa, Fixture{ .codes = std.mem.sliceAsBytes(&words), .recipe = .qlinear_prompt }, source());
    defer prompt.deinit();
    Trace.n = 0;
    try linear.prompt(s, source(), .{ .x = 0x1000, .part = 0x4000 }, 0x3000, 17);
    const prompt_call = Trace.calls[0];
    Trace.n = 0;
    try h.Head.full(source(), .prompt).project(linear, s, .{ .x = 0x1000, .part = 0x4000 }, 0x3000, 17);
    try t.expectEqualDeep(prompt_call, Trace.calls[0]);
    try t.expectEqual(h.Role.prompt, prompt.role);
    var wrong = Fixture{ .codes = std.mem.sliceAsBytes(&words) };
    wrong.recipe = .unsupported;
    try t.expectError(error.ProposalBasisUnsupported, h.bf16(gpa, wrong, source()));
    wrong = Fixture{ .codes = std.mem.sliceAsBytes(&words), .output_f32 = true };
    try t.expectError(error.ProposalBasisUnsupported, h.bf16(gpa, wrong, source()));
    wrong = Fixture{ .codes = &.{}, .vocab = 32768, .hidden = 4096 };
    try t.expectError(error.ProposalBasisUnsupported, h.bf16(gpa, wrong, .{ .bf16 = .{ .mode = .bf16, .codes = 0, .scales = 0, .n = 32768, .k = 4096, .npad = 32768 } }));
}
test "proposal rows: padded keyed draws use logical widths global IDs and each view's actual stride" {
    var d: Driver = undefined;
    d.api.cuLaunchKernel = Trace.launch;
    const draws: sample.Draws = .{ .draw_fn = Trace.function(&d, 80), .draw_ids_fn = Trace.function(&d, 81), .pitched_fn = Trace.function(&d, 82), .logprob_rows_fn = undefined, .topk_slices_fn = undefined, .topk_merge_fn = undefined };
    const s: Stream = .{ .d = &d, .handle = null };
    const view = .{ .vocab = @as(u32, 11), .logical = @as(u32, 3), .stride = @as(u32, 128), .ids = @as(?[]const u32, &.{ 0, 3, 7 }) };
    for ([_]usize{ 1, 17, 64 }) |rows| {
        Trace.n = 0;
        try draws.proposal(s, 0x1000, view, 0x5000, 0x2000, 0x3000, 4, 0x6000, rows, 0x7000);
        try t.expectEqual(@as(usize, 1), Trace.n);
        const c = Trace.calls[0];
        try t.expectEqual(@as(usize, 82), c.id);
        try t.expectEqual([3]u32{ @intCast(rows), 1, 1 }, c.grid);
        try t.expectEqual([3]u32{ 1024, 1, 1 }, c.block);
        try t.expectEqualSlices(u64, &.{ 0x1000, 3, 128, 0x2000, 0x3000, 4, 0x6000, 0x5000, 0x7000 }, c.args[0..9]);
    }
    const full = .{ .vocab = @as(u32, 11), .logical = @as(u32, 11), .stride = @as(u32, 11), .ids = @as(?[]const u32, null) };
    Trace.n = 0;
    try draws.draw(s, 0x1000, 11, 0x2000, 0x3000, 4, 0x6000, 3, null, 0x7000);
    const baseline = Trace.calls[0];
    Trace.n = 0;
    try draws.proposal(s, 0x1000, full, null, 0x2000, 0x3000, 4, 0x6000, 3, 0x7000);
    try t.expectEqual(@as(usize, 1), Trace.n);
    try t.expectEqualDeep(baseline, Trace.calls[0]);
    const compact = .{ .vocab = view.vocab, .logical = view.logical, .stride = @as(u32, 3), .ids = view.ids };
    for ([_]usize{ 1, 17, 64 }) |rows| {
        Trace.n = 0;
        try draws.draw(s, 0x1000, 3, 0x2000, 0x3000, 4, 0x6000, rows, 0x5000, 0x7000);
        const keyed = Trace.calls[0];
        Trace.n = 0;
        try draws.proposal(s, 0x1000, compact, 0x5000, 0x2000, 0x3000, 4, 0x6000, rows, 0x7000);
        try t.expectEqual(@as(usize, 1), Trace.n);
        try t.expectEqualDeep(keyed, Trace.calls[0]);
    }
    try t.expectError(error.ProposalScoreShape, draws.proposal(s, 0x1000, view, null, 0x2000, 0x3000, 4, 0x6000, 3, null));
}
const CopyFake = @import("copy_source_extra_fake.zig").Fake;
const Memory = struct {
    var slots: [2]?[]align(16) u8 = @splat(null);
    var allocs: usize = 0;
    var copies: usize = 0;
    var fail_alloc: usize = 0;
    var fail_copy: usize = 0;
    fn alloc(out: *u64, n: usize) callconv(.c) abi.Result {
        allocs += 1;
        if (allocs == fail_alloc) return abi.error_not_ready;
        for (&slots) |*slot| if (slot.* == null) {
            slot.* = gpa.alignedAlloc(u8, .@"16", n) catch return abi.error_not_ready;
            out.* = @intFromPtr(slot.*.?.ptr);
            return 0;
        };
        unreachable;
    }
    fn free(ptr: u64) callconv(.c) abi.Result {
        for (&slots) |*slot| if (slot.*) |bytes| {
            if (@intFromPtr(bytes.ptr) == ptr) {
                gpa.free(bytes);
                slot.* = null;
                return 0;
            }
        };
        unreachable;
    }
    fn copy(dst: u64, src: ?*const anyopaque, n: usize) callconv(.c) abi.Result {
        copies += 1;
        if (copies == fail_copy) return abi.error_not_ready;
        @memcpy(@as([*]u8, @ptrFromInt(dst))[0..n], @as([*]const u8, @ptrCast(src.?))[0..n]);
        return 0;
    }
    fn init() Driver {
        std.debug.assert(slots[0] == null and slots[1] == null);
        allocs = 0;
        copies = 0;
        fail_alloc = 0;
        fail_copy = 0;
        var d = CopyFake.init();
        CopyFake.fail = false;
        d.api.cuMemAlloc_v2 = alloc;
        d.api.cuMemFree_v2 = free;
        d.api.cuMemcpyHtoD_v2 = copy;
        return d;
    }
};
test "proposal rows: upload failure preserves prepared generation and live captures or undrained queues retain buffers" {
    const words: [3 * 64]u16 align(16) = @splat(0x3f80);
    for ([_]usize{ 1, 2, 3 }) |failure| {
        var d = Memory.init();
        var p: ?h.Packed = try h.bf16(gpa, Fixture{ .codes = std.mem.sliceAsBytes(&words) }, source());
        defer if (p) |*owned| owned.deinit();
        if (failure < 3) Memory.fail_alloc = failure else Memory.fail_copy = 1;
        try t.expectError(error.NotReady, h.Packed.upload(&p, &d, 1 << 20));
        try t.expect(Memory.slots[0] == null and Memory.slots[1] == null);
        try t.expectEqualSlices(u32, &.{ 0, 3, 7 }, p.?.ids);
    }
    var d = Memory.init();
    var p: ?h.Packed = try h.bf16(gpa, Fixture{ .codes = std.mem.sliceAsBytes(&words) }, source());
    defer if (p) |*owned| owned.deinit();
    const price = try p.?.uploadBytes();
    try t.expectError(error.ProposalRowBudget, h.Packed.upload(&p, &d, price - 1));
    var device = try h.Packed.upload(&p, &d, price);
    errdefer {
        p = null;
        device.deinit() catch {};
    }
    try t.expect(p == null);
    try t.expectError(error.ProposalAlreadyUploaded, h.Packed.upload(&p, &d, price));
    try t.expectEqualSlices(i64, &.{ 0, 3, 7 }, @as([*]const i64, @ptrFromInt(device.ids.ptr))[0..3]);
    try device.retainCapture();
    try t.expectError(error.ProposalCaptureLive, device.deinit());
    try device.releaseCapture();
    try t.expectError(error.ProposalCaptureUnderflow, device.releaseCapture());
    CopyFake.fail = true;
    try t.expectError(error.NotReady, device.deinit());
    try t.expect(Memory.slots[0] != null and Memory.slots[1] != null);
    CopyFake.fail = false;
    try device.deinit();
    try t.expect(Memory.slots[0] == null and Memory.slots[1] == null);
}

const PackTrace = struct {
    var calls: u32 = 0;
    fn launch(_: abi.Function, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: abi.Stream, args: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) abi.Result {
        const n = @as(*const c_int, @ptrCast(@alignCast(args.?[1].?))).*;
        std.debug.assert(n == 128);
        calls += 1;
        return 0;
    }
};
test "proposal rows: selected affine4 uses the existing packer and refuses another group" {
    var d: Driver = undefined;
    d.api.cuLaunchKernel = PackTrace.launch;
    var kernels: q.Affine4Kernels = undefined;
    kernels.words_fn = Trace.function(&d, 90);
    kernels.scales_fn = Trace.function(&d, 91);
    const s: Stream = .{ .d = &d, .handle = null };
    var selected = Fixture{ .storage = .affine, .bits = 4, .group = 64, .codes = &.{} };
    const original: q.Weight = .{ .affine4 = .{ .w = 1, .s = 2, .b = 3, .n = 11, .k = 64, .npad = 128 } };
    const tiled: q.Affine4 = .{ .w = 4, .s = 5, .b = 6, .n = 128, .k = 64, .npad = 128 };
    PackTrace.calls = 0;
    const view = try h.affine4(&kernels, s, selected, original, .{ 100, 200, 300 }, tiled);
    try t.expectEqual(@as(u32, 3), PackTrace.calls);
    try t.expectEqual(@as(u32, 3), view.logical);
    try t.expectEqual(@as(u32, 128), view.stride);
    selected.group = 32;
    try t.expectError(error.ProposalBasisUnsupported, h.affine4(&kernels, s, selected, original, .{ 100, 200, 300 }, tiled));
    try t.expectEqual(@as(u32, 3), PackTrace.calls);
}
