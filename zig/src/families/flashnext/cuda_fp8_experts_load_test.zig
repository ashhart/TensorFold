//! Block-FP8 MTP experts on the host: scale widening and packing, world 2 refused, the weight store's load, headFp8.
const std = @import("std");
const core = @import("core");
const cuda = @import("cuda");
const fp8 = @import("cuda_fp8_experts_load.zig");
const experts = @import("cuda_modelopt_experts.zig");
const wt = @import("cuda_prompt_wt.zig");
const lanes_bytes = @import("cuda_lanes_bytes.zig");
const Fake = @import("cuda_modelopt_threads_test.zig").Fake;
const env = @import("sys").env;
const fx = cuda.experts.Fp8Experts;
const ta = std.testing.allocator;
const io = std.testing.io;

const count = 2;
const width = 128; // NI: one scale block
const dims = 256; // D: two scale blocks
const projs = [_][]const u8{ "gate_proj", "up_proj", "down_proj" };

/// Two experts as the checkpoint stores them: E4M3 codes [expert][gate, up, down] and their BF16 scale_inv values.
const Set = struct {
    codes: [count][3][width * dims]u8,
    scales: [count][3][2]u16, // gate / up [1, 2], down [2, 1]
    values: [count][3][2]f32, // the scales' values, built in f32 and exact in BF16
    releases: usize = 0,

    fn init(s: *Set, seed: u64) void {
        var prng = std.Random.DefaultPrng.init(seed);
        const r = prng.random();
        s.releases = 0;
        for (0..count) |e| for (0..3) |p| {
            r.bytes(&s.codes[e][p]);
            for (&s.values[e][p], &s.scales[e][p]) |*v, *bits| {
                // 2^-9 to 2^-6 with all seven mantissa bits used, so a widening that drops any shows
                const m: f32 = @floatFromInt(r.uintLessThan(u8, 128));
                v.* = std.math.ldexp(1 + m / 128, -9 + @as(i32, r.uintLessThan(u8, 4)));
                bits.* = @truncate(@as(u32, @bitCast(v.*)) >> 16);
            }
        };
    }

    fn shape(p: usize, scale: bool) [2]usize {
        const n: usize = if (p == 2) dims else width;
        const k: usize = if (p == 2) width else dims;
        return if (scale) .{ n / 128, k / 128 } else .{ n, k };
    }

    pub fn has(_: *Set, name: []const u8) bool {
        return std.mem.indexOf(u8, name, ".experts.") != null;
    }

    pub fn get(s: *Set, name: []const u8) !core.safetensors.Tensor {
        const marker = std.mem.indexOf(u8, name, ".experts.") orelse return error.MissingTensor;
        const tail = name[marker + ".experts.".len ..];
        const dot = std.mem.indexOfScalar(u8, tail, '.') orelse return error.MissingTensor;
        const e = try std.fmt.parseInt(usize, tail[0..dot], 10);
        if (e >= count) return error.MissingTensor;
        for (projs, 0..) |proj, p| {
            const rest = tail[dot + 1 ..];
            if (!std.mem.startsWith(u8, rest, proj)) continue;
            if (std.mem.eql(u8, rest[proj.len..], ".weight")) return tensor(.f8_e4m3, shape(p, false), &s.codes[e][p]);
            if (std.mem.eql(u8, rest[proj.len..], ".weight_scale_inv")) return tensor(.bf16, shape(p, true), std.mem.sliceAsBytes(&s.scales[e][p]));
        }
        return error.MissingTensor;
    }

    pub fn release(s: *Set) void {
        s.releases += 1;
    }
};

fn tensor(dtype: core.safetensors.DType, shape: [2]usize, bytes: []const u8) core.safetensors.Tensor {
    var t: core.safetensors.Tensor = .{ .dtype = dtype, .rank = 2, .shape = @splat(1), .bytes = bytes };
    t.shape[0] = shape[0];
    t.shape[1] = shape[1];
    return t;
}

const one = fp8.Shape.of(dims, width) catch unreachable;

/// The layer's words and scales as Fp8Experts lays them out, from the set's codes and f32 values.
const Want = struct {
    up: []u32,
    down: []u32,
    up_scale: []f32,
    down_scale: []f32,

    fn init(s: *const Set) !Want {
        const w: Want = .{ .up = try ta.alloc(u32, count * one.up), .down = try ta.alloc(u32, count * one.down), .up_scale = try ta.alloc(f32, count * 4), .down_scale = try ta.alloc(f32, count * 2) };
        for (0..count) |e| {
            try fx.packGateUp(ta, w.up[e * one.up ..][0..one.up], &s.codes[e][0], &s.codes[e][1], width, dims);
            fx.packOne(w.down[e * one.down ..][0..one.down], &s.codes[e][2], dims, width);
            // [E][2][NI/128][D/128]: gate's two blocks, then up's; [E][1][D/128][NI/128]: down's two
            @memcpy(w.up_scale[e * 4 ..][0..2], &s.values[e][0]);
            @memcpy(w.up_scale[e * 4 + 2 ..][0..2], &s.values[e][1]);
            @memcpy(w.down_scale[e * 2 ..][0..2], &s.values[e][2]);
        }
        return w;
    }

    fn deinit(w: Want) void {
        ta.free(w.up);
        ta.free(w.down);
        ta.free(w.up_scale);
        ta.free(w.down_scale);
    }

    fn expectOwned(w: Want, o: experts.Owned) !void {
        try std.testing.expectEqual(cuda.experts.Format.fp8g, o.view().format);
        try std.testing.expectEqual(@as(usize, width), o.view().width);
        try std.testing.expectEqual(@as(usize, dims), o.view().dims);
        try std.testing.expectEqual(@as(usize, count), o.view().experts);
        try std.testing.expectEqual(@as(f32, 0), o.view().limit);
        for ([_]struct { []const u8, cuda.DeviceBuffer }{ .{ std.mem.sliceAsBytes(w.up), o.up }, .{ std.mem.sliceAsBytes(w.down), o.down }, .{ std.mem.sliceAsBytes(w.up_scale), o.up_scale }, .{ std.mem.sliceAsBytes(w.down_scale), o.down_scale } }) |pair| {
            try std.testing.expectEqualSlices(u8, pair[0], @as([*]const u8, @ptrFromInt(pair[1].ptr))[0..pair[1].len]);
        }
    }
};

test "Flash Next expert loader: FP8 MTP experts: a hand-built 2-expert set packs as Fp8Experts lays it out, BF16 block scales widened exactly" {
    const s = try ta.create(Set);
    defer ta.destroy(s);
    s.init(0xf8);
    const want = try Want.init(s);
    defer want.deinit();
    const up = try ta.alloc(u32, one.up);
    defer ta.free(up);
    const down = try ta.alloc(u32, one.down);
    defer ta.free(down);
    var us: [4]f32 = undefined;
    var ds: [2]f32 = undefined;
    for (0..count) |e| {
        try fp8.packExpert(ta, s, "language_model.mtp.layers.0.mlp", e, dims, width, .{ .up = up, .down = down, .up_scale = &us, .down_scale = &ds });
        try std.testing.expectEqualSlices(u32, want.up[e * one.up ..][0..one.up], up);
        try std.testing.expectEqualSlices(u32, want.down[e * one.down ..][0..one.down], down);
        try std.testing.expectEqualSlices(f32, want.up_scale[e * 4 ..][0..4], &us);
        try std.testing.expectEqualSlices(f32, want.down_scale[e * 2 ..][0..2], &ds);
    }
    try std.testing.expectEqual(@as(usize, count), s.releases);
    // the whole layer through loadShape from a plain reader, on the counting fake driver
    Fake.reset();
    var d = Fake.driver();
    var owned = try experts.loadShape(ta, &d, s, "language_model.mtp.layers.0.mlp", 1, 0, dims, width, count);
    try want.expectOwned(owned);
    owned.free();
    try std.testing.expectEqual(@as(usize, 0), Fake.live_device);
    // a NaN scale is refused, a F32 scale (as Drama 3 stores them) is not this layout; nothing stays on the device
    s.scales[1][2][1] = 0x7fc0;
    try std.testing.expectError(error.UnsupportedQuantization, experts.loadShape(ta, &d, s, "language_model.mtp.layers.0.mlp", 1, 0, dims, width, count));
    try std.testing.expectEqual(@as(usize, 0), Fake.live_device);
    try std.testing.expectError(error.UnexpectedTensor, fp8.packExpert(ta, s, "p", 0, dims, width, .{ .up = up, .down = down[1..], .up_scale = &us, .down_scale = &ds }));
}

test "Flash Next expert loader: FP8 MTP experts: world 2 is refused before the driver is touched, the format probe's borrow released" {
    const s = try ta.create(Set);
    defer ta.destroy(s);
    s.init(1);
    Fake.reset();
    var d = Fake.driver();
    // 128 channels cut in two are 32-column whole (Cut accepts them), but each half splits a 128-wide scale block
    try std.testing.expectError(error.UnevenExpertSplit, experts.loadShape(ta, &d, s, "language_model.mtp.layers.0.mlp", 2, 0, dims, width, count));
    try std.testing.expectError(error.UnevenExpertSplit, experts.loadShape(ta, &d, s, "language_model.mtp.layers.0.mlp", 2, 1, dims, width, count));
    try std.testing.expectEqual(@as(usize, 2), s.releases);
    for (Fake.calls) |n| try std.testing.expectEqual(@as(usize, 0), n);
}

/// One tensor of a fixture shard: its checkpoint name, dtype, [rows, cols] and bytes.
const Entry = struct { name: []const u8, dtype: []const u8, shape: [2]usize, data: []const u8 };

/// `entries` in one shard, model.safetensors, with the index that names it.
fn fixture(dir: std.Io.Dir, entries: []const Entry) !void {
    var header: std.Io.Writer.Allocating = .init(ta);
    defer header.deinit();
    var index: std.Io.Writer.Allocating = .init(ta);
    defer index.deinit();
    var data: std.ArrayList(u8) = .empty;
    defer data.deinit(ta);
    try header.writer.writeByte('{');
    try index.writer.writeAll("{\"weight_map\":{");
    for (entries, 0..) |e, i| {
        if (i > 0) {
            try header.writer.writeByte(',');
            try index.writer.writeByte(',');
        }
        const at = data.items.len;
        try data.appendSlice(ta, e.data);
        try header.writer.print("\"{s}\":{{\"dtype\":\"{s}\",\"shape\":[{d},{d}],\"data_offsets\":[{d},{d}]}}", .{ e.name, e.dtype, e.shape[0], e.shape[1], at, data.items.len });
        try index.writer.print("\"{s}\":\"model.safetensors\"", .{e.name});
    }
    try header.writer.writeByte('}');
    try index.writer.writeAll("}}");
    var image: std.Io.Writer.Allocating = .init(ta);
    defer image.deinit();
    try image.writer.writeAll(&std.mem.toBytes(std.mem.nativeToLittle(u64, header.written().len)));
    try image.writer.writeAll(header.written());
    try image.writer.writeAll(data.items);
    try dir.writeFile(io, .{ .sub_path = "model.safetensors", .data = image.written() });
    try dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = index.written() });
}

/// The set under the checkpoint's own names (mtp.layers.0...), which the store reaches as language_model.mtp...
fn setFixture(dir: std.Io.Dir, s: *const Set, names: std.mem.Allocator) !void {
    var entries: [count * 3 * 2]Entry = undefined;
    for (0..count) |e| for (projs, 0..) |proj, p| {
        const at = (e * 3 + p) * 2;
        entries[at] = .{ .name = try std.fmt.allocPrint(names, "mtp.layers.0.mlp.experts.{d}.{s}.weight", .{ e, proj }), .dtype = "F8_E4M3", .shape = Set.shape(p, false), .data = &s.codes[e][p] };
        entries[at + 1] = .{ .name = try std.fmt.allocPrint(names, "mtp.layers.0.mlp.experts.{d}.{s}.weight_scale_inv", .{ e, proj }), .dtype = "BF16", .shape = Set.shape(p, true), .data = std.mem.sliceAsBytes(&s.scales[e][p]) };
    };
    try fixture(dir, &entries);
}

test "Flash Next expert loader: FP8 MTP experts: the weight store with default load threads takes the FP8 loop, not the threaded NVFP4 loader" {
    _ = env.unsetenv("TF_FLASHNEXT_LOAD_THREADS"); // default: 8 threads, which an NVFP4 layer at world 1 takes
    const s = try ta.create(Set);
    defer ta.destroy(s);
    s.init(7);
    var arena = std.heap.ArenaAllocator.init(ta);
    defer arena.deinit();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try setFixture(tmp.dir, s, arena.allocator());
    const path = try tmp.dir.realPathFileAlloc(io, ".", ta);
    defer ta.free(path);
    var store = try wt.Store.open(ta, io, path);
    defer store.deinit();
    const want = try Want.init(s);
    defer want.deinit();
    Fake.reset();
    var d = Fake.driver();
    var owned = try experts.loadShape(ta, &d, &store, "language_model.mtp.layers.0.mlp", 1, 0, dims, width, count);
    try want.expectOwned(owned);
    try std.testing.expectEqual(@as(usize, 0), Fake.calls[@backingInt(Fake.Kind.pinned)]); // no NVFP4 staging
    try std.testing.expect(store.staging == null);
    try std.testing.expectEqual(@as(usize, 0), store.held.items.len); // every borrow released
    owned.free();
    try std.testing.expectEqual(@as(usize, 0), Fake.live_device);
}

test "Flash Next expert loader: FP8 MTP experts: headFp8 reads the head's first gate weight dtype; U8 or absent is false" {
    const s = try ta.create(Set);
    defer ta.destroy(s);
    s.init(3);
    var arena = std.heap.ArenaAllocator.init(ta);
    defer arena.deinit();
    const codes: [8]u8 = @splat(0);
    const cases = [_]struct { entries: []const Entry, fp8: bool }{
        .{ .entries = &.{.{ .name = "mtp.layers.0.mlp.experts.0.gate_proj.weight", .dtype = "U8", .shape = .{ 2, 4 }, .data = &codes }}, .fp8 = false },
        .{ .entries = &.{.{ .name = "language_model.mtp.layers.0.mlp.experts.0.gate_proj.weight", .dtype = "F8_E4M3", .shape = .{ 2, 4 }, .data = &codes }}, .fp8 = true },
        .{ .entries = &.{.{ .name = "model.language_model.embed_tokens.weight", .dtype = "U8", .shape = .{ 2, 4 }, .data = &codes }}, .fp8 = false },
    };
    for (cases) |c| {
        var tmp = std.testing.tmpDir(.{});
        defer tmp.cleanup();
        try fixture(tmp.dir, c.entries);
        const path = try tmp.dir.realPathFileAlloc(io, ".", ta);
        defer ta.free(path);
        try std.testing.expectEqual(c.fp8, try lanes_bytes.headFp8(ta, io, path));
    }
    // the set's own layout, under the checkpoint's mtp.layers.0 names
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try setFixture(tmp.dir, s, arena.allocator());
    const path = try tmp.dir.realPathFileAlloc(io, ".", ta);
    defer ta.free(path);
    try std.testing.expect(try lanes_bytes.headFp8(ta, io, path));
}

/// A wait that fails while `fail` is set; the context's own wait succeeds and is counted, each free sees the count.
const Waits = struct {
    var fail = false;
    var ctx_waits: usize = 0;
    var waits_at_free: usize = std.math.maxInt(usize);
    fn stream(_: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        return if (fail) cuda.abi.error_not_ready else cuda.abi.success; // a wait that has not finished, unlogged
    }
    fn context() callconv(.c) cuda.abi.Result {
        ctx_waits += 1;
        return cuda.abi.success;
    }
    fn free(p: cuda.abi.DevicePtr) callconv(.c) cuda.abi.Result {
        waits_at_free = @min(waits_at_free, ctx_waits);
        return Fake.devFree(p);
    }
};

test "Flash Next expert loader: FP8 MTP experts: a failed upload wait drains the context before the buffers are freed" {
    const s = try ta.create(Set);
    defer ta.destroy(s);
    s.init(3);
    Fake.reset();
    var d = Fake.driver();
    d.api.cuStreamSynchronize = Waits.stream;
    d.api.cuCtxSynchronize = Waits.context;
    d.api.cuMemFree_v2 = Waits.free;
    cuda.copy_source.setCurrent(@ptrFromInt(11));
    Waits.fail = true;
    Waits.ctx_waits = 0;
    Waits.waits_at_free = std.math.maxInt(usize);
    defer {
        Waits.fail = false;
        cuda.copy_source.synchronizeContext(&d) catch unreachable;
    }
    try std.testing.expect(std.meta.isError(experts.loadShape(ta, &d, s, "language_model.mtp.layers.0.mlp", 1, 0, dims, width, count)));
    try std.testing.expect(Waits.waits_at_free != std.math.maxInt(usize) and Waits.waits_at_free >= 1);
    try std.testing.expectEqual(@as(usize, 0), Fake.live_device);
}
