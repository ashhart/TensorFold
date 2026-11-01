//! A layer's routed NVFP4 experts decoded and packed as Experts4 by TF_GLM_LOAD_PACK threads, each its own experts.

const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const formats = @import("cuda_formats.zig");
const X = cuda.experts.Nvfp4Experts;

/// Packers when TF_GLM_LOAD_PACK is unset.
pub const default_packers = 8;
pub const max_packers = 16;

/// The packed layer, `count` experts in a row: gate and up interleaved, down, and each matrix's global scale.
pub const Out = struct { up: []u32, down: []u32, up_scale: []f32, down_scale: []f32 };

/// Layer `index`'s cut: hidden `d`, the rank's `ni` rows from `lo`, `per` scale bytes a matrix, expert `first` on.
pub const Cut = struct { index: u32, d: usize, ni: usize, lo: usize, per: usize, first: usize = 0 };

/// The pinned staging's most, as Flash Next's threaded loader stages its experts.
pub const staging_bytes: usize = 192 << 20;

/// Pinned memory mapped into the device, reused piece to piece: the packers write a piece's experts, the GPU copies it.
pub const Staging = struct {
    buf: ?cuda.HostBuffer = null,
    cap: usize = staging_bytes,

    /// Experts in a piece under `cap`: a multiple of the packers when that many fit, and at least one.
    pub fn per(s: Staging, up_len: usize, down_len: usize, threads: usize) usize {
        const fit = s.cap / ((up_len + down_len) * 4);
        return if (fit >= threads) fit - fit % threads else @max(1, fit);
    }

    /// `m` experts' up and down words in the mapped memory, taken anew when a piece needs more than the last one did.
    pub fn words(s: *Staging, d: *const cuda.Driver, m: usize, up_len: usize, down_len: usize) ![2][]u32 {
        const n = m * (up_len + down_len);
        if (s.buf) |b| if (b.bytes.len < n * 4) s.deinit();
        if (s.buf == null) s.buf = try cuda.HostBuffer.allocMapped(d, n * 4);
        const w = s.buf.?.slice(u32);
        return .{ w[0 .. m * up_len], w[m * up_len ..][0 .. m * down_len] };
    }

    /// The piece's `m` experts copied by the GPU into `up` and `down` from expert `first` on: done when it returns.
    pub fn send(s: *const Staging, stream: cuda.Stream, up: cuda.DeviceBuffer, down: cuda.DeviceBuffer, first: usize, m: usize, up_len: usize, down_len: usize) !void {
        const b = s.buf orelse return error.Invalid;
        const base = try b.device();
        errdefer stream.synchronize() catch {}; // no copy may still write a buffer freed after an error
        try up.copyFrom(first * up_len * 4, base, m * up_len * 4, stream.handle);
        try down.copyFrom(first * down_len * 4, base + m * up_len * 4, m * down_len * 4, stream.handle);
        try stream.synchronize(); // the next piece packs into the same memory
    }

    pub fn deinit(s: *Staging) void {
        if (s.buf) |*b| b.free();
        s.buf = null;
    }
};

/// TF_GLM_LOAD_PACK as a packer count: unset for `default_packers`, 1 for the loader's thread alone.
pub fn packers(value: ?[]const u8) !usize {
    const n = std.fmt.parseInt(usize, value orelse return default_packers, 10) catch return error.BadSetting;
    return if (n >= 1 and n <= max_packers) n else error.BadSetting;
}

/// Each expert's gate, up and down (`pending`, read on the loader's thread) decoded and packed into `out`.
pub fn pack(gpa: std.mem.Allocator, pending: []const [3]formats.Pending, cut: Cut, out: Out, threads: usize) !void {
    if (pending.len == 0) return;
    const n = @max(1, @min(threads, max_packers, pending.len));
    var spawned: [max_packers]?std.Thread = @splat(null);
    var failed: [max_packers]?anyerror = @splat(null);
    for (1..n) |t| spawned[t] = std.Thread.spawn(.{}, run, .{ gpa, pending, cut, out, part(pending.len, n, t), &failed[t] }) catch null;
    run(gpa, pending, cut, out, part(pending.len, n, 0), &failed[0]);
    for (spawned[1..n], 1..) |thread, t| if (thread) |x| x.join() else run(gpa, pending, cut, out, part(pending.len, n, t), &failed[t]);
    for (failed[0..n]) |f| if (f) |err| return err;
}

/// A layer's experts packed into `s` a piece at a time, each piece copied into `up` and `down` before the next packs.
pub fn pieces(gpa: std.mem.Allocator, s: *Staging, d: *const cuda.Driver, stream: cuda.Stream, pending: []const [3]formats.Pending, cut: Cut, up: cuda.DeviceBuffer, down: cuda.DeviceBuffer, up_scale: []f32, down_scale: []f32, threads: usize) !void {
    const up_len = X.bytes(1, cut.ni, cut.d, 2) / 4;
    const down_len = X.bytes(1, cut.d, cut.ni, 1) / 4;
    const m = s.per(up_len, down_len, threads);
    var first: usize = 0;
    while (first < pending.len) : (first += m) {
        const k = @min(m, pending.len - first);
        const w = try s.words(d, k, up_len, down_len);
        var at = cut;
        at.first = first;
        try pack(gpa, pending[first..][0..k], at, .{ .up = w[0], .down = w[1], .up_scale = up_scale[2 * first ..][0 .. 2 * k], .down_scale = down_scale[first..][0..k] }, threads);
        try s.send(stream, up, down, first, k, up_len, down_len);
    }
}

/// Thread `t`'s experts of `count` split `n` ways, as [from, to).
fn part(count: usize, n: usize, t: usize) [2]usize {
    return .{ count * t / n, count * (t + 1) / n };
}

fn run(gpa: std.mem.Allocator, pending: []const [3]formats.Pending, cut: Cut, out: Out, experts: [2]usize, failed: *?anyerror) void {
    share(gpa, pending, cut, out, experts) catch |err| {
        failed.* = err;
    };
}

/// Experts [from, to) as the loader packed each before (cuda_weights' nvfp4Experts), into their own parts of `out`.
fn share(gpa: std.mem.Allocator, pending: []const [3]formats.Pending, cut: Cut, out: Out, experts: [2]usize) !void {
    const d = cut.d;
    const ni = cut.ni;
    const lo = cut.lo;
    const decoded = try gpa.alloc(u8, 3 * cut.per); // csf_32x2_v1's, decoded before the rank's cut
    defer gpa.free(decoded);
    const codes = try gpa.alloc(u8, d * ni / 2);
    defer gpa.free(codes);
    const scales = try gpa.alloc(u8, d * ni / 16);
    defer gpa.free(scales);
    const up_len = out.up.len / pending.len;
    const down_len = out.down.len / pending.len;
    for (experts[0]..experts[1]) |e| {
        var ls: [3]core.nvfp4.Linear = undefined;
        for (&ls, pending[e], [_][]const u8{ "gate_proj", "up_proj", "down_proj" }, 0..) |*l, p, proj, m| {
            l.* = p.linear(decoded[m * cut.per ..][0..cut.per]) catch |err| {
                std.log.err("layers.{d}.mlp.experts.{d}.{s}: its csf_32x2_v1 scales do not decode ({t})", .{ cut.index, cut.first + e, proj, err });
                return err;
            };
        }
        // gate and up: the rank's rows; down: every row, the rank's columns
        const gate = .{ ls[0].codes[lo * d / 2 ..][0 .. ni * d / 2], ls[0].scales[lo * d / 16 ..][0 .. ni * d / 16] };
        const upm = .{ ls[1].codes[lo * d / 2 ..][0 .. ni * d / 2], ls[1].scales[lo * d / 16 ..][0 .. ni * d / 16] };
        X.packGateUp(out.up[e * up_len ..][0..up_len], gate, upm, ni, d);
        const l = ls[2];
        for (0..d) |r| {
            @memcpy(codes[r * ni / 2 ..][0 .. ni / 2], l.codes[r * l.k / 2 + lo / 2 ..][0 .. ni / 2]);
            @memcpy(scales[r * ni / 16 ..][0 .. ni / 16], l.scales[r * l.k / 16 + lo / 16 ..][0 .. ni / 16]);
        }
        X.packOne(out.down[e * down_len ..][0..down_len], codes, scales, d, ni);
        out.up_scale[e * 2] = ls[0].global;
        out.up_scale[e * 2 + 1] = ls[1].global;
        out.down_scale[e] = l.global;
    }
}

test "TF_GLM_LOAD_PACK: 8 when unset, the loader alone at 1, a count up to the limit, a refusal for 0, past it or a word" {
    try std.testing.expectEqual(@as(usize, 8), try packers(null));
    try std.testing.expectEqual(@as(usize, 1), try packers("1"));
    try std.testing.expectEqual(@as(usize, 16), try packers("16"));
    for ([_][]const u8{ "0", "17", "eight", "" }) |bad| try std.testing.expectError(error.BadSetting, packers(bad));
}

test "packers pack the same bytes as the loader's one thread, whole or a piece at a time through the mapped staging" {
    const gpa = std.testing.allocator;
    const experts = 5;
    const inter = 64; // the experts' rows; this rank holds the second half
    const d = 64;
    const ni = 32;
    var prng: std.Random.DefaultPrng = .init(7);
    const rng = prng.random();
    const one = std.mem.toBytes(@as(f32, 0.5));
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var tensors: std.ArrayList(core.shard_edit.New) = .empty;
    for (0..experts) |e| for ([_][]const u8{ "gate_proj", "up_proj", "down_proj" }, 0..) |proj, m| {
        const n: usize = if (m < 2) inter else d;
        const k: usize = if (m < 2) d else inter;
        const codes = try a.alloc(u8, n * k / 2);
        rng.bytes(codes);
        const scale = try a.alloc(u8, n * k / 16);
        for (scale, 0..) |*s, j| s.* = if (j % 29 == 0) 0x7e else 0x30 + rng.uintLessThan(u8, 16); // a band and exceptions
        const enc = try core.nvfp4_csf32.encode(a, scale, n, k / 16);
        const lens = try a.dupe(usize, &.{ enc.bases.len, enc.codes.len, enc.page_bases.len / 4, enc.offsets.len / 2, enc.exceptions.len / 2 });
        const name = try std.fmt.allocPrint(a, "e{d}.{s}", .{ e, proj });
        try tensors.append(a, .{ .name = try std.fmt.allocPrint(a, "{s}.weight", .{name}), .dtype = .u8, .shape = try a.dupe(usize, &.{ n, k / 2 }), .bytes = codes });
        inline for (.{ "bases", "codes", "page_bases", "offsets", "exceptions" }, .{ .u8, .u8, .u32, .u16, .u16 }, 0..) |p, dtype, j| {
            try tensors.append(a, .{ .name = try std.fmt.allocPrint(a, "{s}.weight_scale.csf32_" ++ p, .{name}), .dtype = dtype, .shape = lens[j .. j + 1], .bytes = @field(enc, p) });
        }
        try tensors.append(a, .{ .name = try std.fmt.allocPrint(a, "{s}.weight_scale_2", .{name}), .dtype = .f32, .shape = &.{}, .bytes = &one });
        try tensors.append(a, .{ .name = try std.fmt.allocPrint(a, "{s}.input_scale", .{name}), .dtype = .f32, .shape = &.{}, .bytes = &one });
    };
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var dir: [64]u8 = undefined;
    const at = try std.fmt.bufPrint(&dir, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    var path: [96]u8 = undefined;
    try core.shard_edit.create(gpa, try std.fmt.bufPrint(&path, "{s}/model.safetensors", .{at}), &.{}, tensors.items);
    var ck = try core.Checkpoint.openModel(gpa, std.testing.io, at);
    defer ck.close();
    const cut: Cut = .{ .index = 3, .d = d, .ni = ni, .lo = inter - ni, .per = inter * d / 16 };
    const up_len = X.bytes(1, ni, d, 2) / 4;
    const down_len = X.bytes(1, d, ni, 1) / 4;
    const want = try a.alloc(u32, experts * (up_len + down_len) + experts * 3);
    var pending: [experts][3]formats.Pending = undefined;
    for (&pending, 0..) |*p, e| for (p, [_][]const u8{ "gate_proj", "up_proj", "down_proj" }) |*q, proj| {
        var name: [32]u8 = undefined;
        q.* = .{ .csf = try formats.csfParts(&ck, try std.fmt.bufPrint(&name, "e{d}.{s}", .{ e, proj })) };
    };
    try std.testing.expectEqual(@as(usize, 0), ck.unused());
    { // the loader's one thread as it packed before: csfLinear, then the same packs, expert by expert
        const decoded = try a.alloc(u8, 3 * cut.per);
        const codes = try a.alloc(u8, d * ni / 2);
        const scales = try a.alloc(u8, d * ni / 16);
        const w = layout(want, experts, up_len, down_len);
        for (0..experts) |e| {
            var ls: [3]core.nvfp4.Linear = undefined;
            for (&ls, [_][]const u8{ "gate_proj", "up_proj", "down_proj" }, 0..) |*l, proj, m| {
                var name: [32]u8 = undefined;
                l.* = try formats.csfLinear(&ck, try std.fmt.bufPrint(&name, "e{d}.{s}", .{ e, proj }), decoded[m * cut.per ..][0..cut.per]);
            }
            const lo = cut.lo;
            X.packGateUp(w.up[e * up_len ..][0..up_len], .{ ls[0].codes[lo * d / 2 ..][0 .. ni * d / 2], ls[0].scales[lo * d / 16 ..][0 .. ni * d / 16] }, .{ ls[1].codes[lo * d / 2 ..][0 .. ni * d / 2], ls[1].scales[lo * d / 16 ..][0 .. ni * d / 16] }, ni, d);
            for (0..d) |r| {
                @memcpy(codes[r * ni / 2 ..][0 .. ni / 2], ls[2].codes[r * inter / 2 + lo / 2 ..][0 .. ni / 2]);
                @memcpy(scales[r * ni / 16 ..][0 .. ni / 16], ls[2].scales[r * inter / 16 + lo / 16 ..][0 .. ni / 16]);
            }
            X.packOne(w.down[e * down_len ..][0..down_len], codes, scales, d, ni);
            w.up_scale[e * 2] = ls[0].global;
            w.up_scale[e * 2 + 1] = ls[1].global;
            w.down_scale[e] = ls[2].global;
        }
    }
    for ([_]usize{ 1, 2, 3, 8 }) |threads| {
        const got = try a.alloc(u32, want.len);
        @memset(got, 0xdeadbeef);
        try pack(gpa, &pending, cut, layout(got, experts, up_len, down_len), threads);
        try std.testing.expectEqualSlices(u32, want, got);
    }
    const R = cuda.abi.Result;
    const F = struct { // the mapped staging and the device as host memory: the GPU's copies done as memcpy
        var host: [2048]u32 align(16) = undefined;
        var dev: [8192]u32 = undefined;
        var next: u64 = 0;
        var waits: usize = 0;
        const host_at: u64 = 0x9000_0000;
        const dev_at: u64 = 0x1000_0000;
        fn hostAlloc(p: *?*anyopaque, len: usize, _: c_uint) callconv(.c) R {
            std.debug.assert(len <= @sizeOf(@TypeOf(host)));
            p.* = &host;
            return cuda.abi.success;
        }
        fn hostFree(_: ?*anyopaque) callconv(.c) R {
            return cuda.abi.success;
        }
        fn mapped(p: *u64, _: ?*anyopaque, _: c_uint) callconv(.c) R {
            p.* = host_at;
            return cuda.abi.success;
        }
        fn alloc(p: *u64, len: usize) callconv(.c) R {
            std.debug.assert(next + len <= @sizeOf(@TypeOf(dev)));
            p.* = dev_at + next;
            next += len;
            return cuda.abi.success;
        }
        fn free(_: u64) callconv(.c) R {
            return cuda.abi.success;
        }
        fn copy(dst: u64, src: u64, len: usize, _: cuda.abi.Stream) callconv(.c) R {
            @memcpy(std.mem.asBytes(&dev)[dst - dev_at ..][0..len], std.mem.asBytes(&host)[src - host_at ..][0..len]);
            return cuda.abi.success;
        }
        fn wait(_: cuda.abi.Stream) callconv(.c) R {
            waits += 1;
            return cuda.abi.success;
        }
    };
    var dr: cuda.Driver = undefined;
    dr.api.cuMemHostAlloc = F.hostAlloc;
    dr.api.cuMemFreeHost = F.hostFree;
    dr.api.cuMemHostGetDevicePointer_v2 = F.mapped;
    dr.api.cuMemAlloc_v2 = F.alloc;
    dr.api.cuMemFree_v2 = F.free;
    dr.api.cuMemcpyDtoDAsync_v2 = F.copy;
    dr.api.cuStreamSynchronize = F.wait;
    const w = layout(want, experts, up_len, down_len);
    for ([_]usize{ 1, 2, 3 }) |threads| {
        F.next = 0;
        F.waits = 0;
        @memset(&F.dev, 0);
        var s: Staging = .{ .cap = 2 * (up_len + down_len) * 4 }; // two experts a piece
        defer s.deinit();
        var up = try cuda.DeviceBuffer.alloc(&dr, experts * up_len * 4);
        defer up.free();
        var down = try cuda.DeviceBuffer.alloc(&dr, experts * down_len * 4);
        defer down.free();
        const up_scale = try a.alloc(f32, experts * 2);
        const down_scale = try a.alloc(f32, experts);
        try pieces(gpa, &s, &dr, .{ .d = &dr, .handle = null }, &pending, cut, up, down, up_scale, down_scale, threads);
        try std.testing.expectEqualSlices(u32, w.up, F.dev[0..w.up.len]);
        try std.testing.expectEqualSlices(u32, w.down, F.dev[w.up.len..][0..w.down.len]);
        try std.testing.expectEqualSlices(f32, w.up_scale, up_scale);
        try std.testing.expectEqualSlices(f32, w.down_scale, down_scale);
        try std.testing.expectEqual(@as(usize, 3), F.waits); // five experts in pieces of two, two and one
    }
}

/// `words` as `Out` for `count` experts: their up, their down, then each one's two up scales, then the down scales.
fn layout(words: []u32, count: usize, up_len: usize, down_len: usize) Out {
    const scales: []f32 = @ptrCast(words[count * (up_len + down_len) ..]);
    return .{ .up = words[0 .. count * up_len], .down = words[count * up_len ..][0 .. count * down_len], .up_scale = scales[0 .. count * 2], .down_scale = scales[count * 2 ..][0..count] };
}
