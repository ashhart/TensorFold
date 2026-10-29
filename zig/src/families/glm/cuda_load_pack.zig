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

/// Layer `index`'s rank cut: hidden `d`, the rank's `ni` expert rows from `lo`, and `per` scale bytes a whole matrix.
pub const Cut = struct { index: u32, d: usize, ni: usize, lo: usize, per: usize };

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
                std.log.err("layers.{d}.mlp.experts.{d}.{s}: its csf_32x2_v1 scales do not decode ({t})", .{ cut.index, e, proj, err });
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

test "packers pack the same bytes as the loader's one thread, the csf scales decoded by whichever packs the expert" {
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
}

/// One buffer cut as `Out` for `count` experts (the test's: one slice to compare).
fn layout(words: []u32, count: usize, up_len: usize, down_len: usize) Out {
    const scales: []f32 = @ptrCast(words[count * (up_len + down_len) ..]);
    return .{ .up = words[0 .. count * up_len], .down = words[count * up_len ..][0 .. count * down_len], .up_scale = scales[0 .. count * 2], .down_scale = scales[count * 2 ..][0..count] };
}
