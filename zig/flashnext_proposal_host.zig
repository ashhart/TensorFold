//! Actual CUDA family head routing and launch parity against the preceding default path, without devices.
const std = @import("std");
const core = @import("core");
const cuda = @import("cuda");
const lanes = @import("lanes");
const family = @import("flashnext");
const old = @import("flashnext_proposal_oracle.zig");
const table = @embedFile("src/families/flashnext/draft_vocab_default.txt");
const t = std.testing;
const gpa = t.allocator;
const pv = core.proposal_vocab;

test "Flash Next proposal: original IDs match the common legacy artifact and cannot claim calibration" {
    try t.expect(!@hasDecl(lanes.proposal_head.Service, "choose"));
    const before = try old.rawIds(gpa, table);
    defer gpa.free(before);
    const after = try family.mtp.draftIds(gpa);
    defer gpa.free(after);
    try t.expectEqualSlices(u32, before, after);
    try t.expectEqualStrings("998317c397023e24afeb9b6839ba54d8611bd3a4dddab4f7288344fd191b6ed8", &pv.idsHash(after));
    var artifact = try pv.legacy.artifact(gpa, after, .{ .vocab = 248320, .hidden = 2560, .format = "bf16", .group = 0, .scale_recipe = "existing draft head" });
    defer artifact.deinit();
    var service = try lanes.proposal_head.Service.legacy(gpa, &artifact);
    const plan = lanes.proposal_head.Service.testing.choose(&service, undefined);
    try t.expect(service.mode == .legacy and plan.kind == .subset);
    try t.expectEqualSlices(u32, before, plan.map.?.ids);
    for (before, 0..) |id, col| try t.expectEqual(id, plan.map.?.global(col).?);
    try t.expectEqual(@as(?u32, null), plan.map.?.global(before.len));
    try t.expectError(error.ProposalIdentityMismatch, lanes.proposal_head.Service.calibrated(gpa, &artifact, artifact.identity(), .{ .subset = true, .full = false }));
    try t.expect(artifact.owned.value.coverage.len == 0);
    const encoded = try std.json.Stringify.valueAlloc(gpa, artifact.owned.value, .{});
    defer gpa.free(encoded);
    try t.expectError(error.UnsupportedVersion, pv.load(gpa, encoded, artifact.identity(), &.{}));
}

const Trace = struct {
    const Call = struct { function: usize, grid: [3]u32, block: [3]u32, shared: u32, stream: usize, args: [13]u64 = @splat(0) };
    var calls: [16]Call = undefined;
    var n: usize = 0;
    fn read(comptime T: type, a: ?*anyopaque) T {
        return @as(*const T, @ptrCast(@alignCast(a.?))).*;
    }
    fn launch(f: cuda.abi.Function, gx: c_uint, gy: c_uint, gz: c_uint, bx: c_uint, by: c_uint, bz: c_uint, sm: c_uint, stream: cuda.abi.Stream, args: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
        const a = args.?;
        var c: Call = .{ .function = @intFromPtr(f), .grid = .{ gx, gy, gz }, .block = .{ bx, by, bz }, .shared = sm, .stream = @intFromPtr(stream) };
        inline for (.{ 0, 1, 2, 4, 5 }) |i| c.args[i] = read(u64, a[i]);
        c.args[3] = read(u32, a[3]);
        inline for (6..13) |i| c.args[i] = read(u32, a[i]);
        calls[n] = c;
        n += 1;
        return 0;
    }
};
const Pass = struct {
    k: *struct { native_lane: cuda.qmmf.Lane },
    s: *struct { native_part: u64, dummy: u64 },
    stream: cuda.Stream,
    fallback: usize = 0,
    pub fn lane(p: *@This(), _: u64, _: u64, _: anytype, _: u64, _: u64, _: usize, _: usize) !void {
        p.fallback += 1;
    }
};
test "Flash Next proposal: resident default head keeps exact logical stride and native launch list" {
    var d: cuda.Driver = undefined;
    d.api.cuLaunchKernel = Trace.launch;
    var k: @typeInfo(@FieldType(Pass, "k")).pointer.child = undefined;
    k.native_lane.clusters = false;
    for (&k.native_lane.fns) |*set| for (set, 0..) |*f, i| {
        f.* = .{ .d = &d, .handle = @ptrFromInt(i + 1) };
    };
    k.native_lane.half = @splat(@splat(@splat(.{ .d = &d, .handle = @ptrFromInt(120) })));
    k.native_lane.half_below = 0; // the released launch parameters: the 64-column tile's
    var scratch: @typeInfo(@FieldType(Pass, "s")).pointer.child = .{ .native_part = 0x4000, .dummy = 0x5000 };
    var pass: Pass = .{ .k = &k, .s = &scratch, .stream = .{ .d = &d, .handle = null } };
    var head: family.mtp.Head = undefined;
    const ids = try family.mtp.draftIds(gpa);
    defer gpa.free(ids);
    head.artifact = try pv.legacy.artifact(gpa, ids, .{ .vocab = 248320, .hidden = 2560, .format = "bf16", .group = 0, .scale_recipe = "existing draft head" });
    defer head.artifact.deinit();
    head.ids = head.artifact.ids();
    head.service = try lanes.proposal_head.Service.legacy(gpa, &head.artifact);
    head.draft = family.projection.Projection.empty(&d);
    head.draft.n = ids.len;
    head.draft.k = 2560;
    const weight = cuda.modelopt.bf16Weight(0x2000, ids.len, 2560);
    head.draft.native = weight;
    const view = head.view();
    try t.expectEqual(@as(u32, 79591), view.logical);
    try t.expectEqual(view.logical, view.stride);
    try t.expectEqualSlices(u32, ids, view.ids.?);
    for ([_]usize{ 1, 3, 16, 17, 32, 64 }) |rows| {
        Trace.n = 0;
        try k.native_lane.matmul(pass.stream, 0x1000, 2560, rows, weight, 0x3000, 0x4000);
        const count = Trace.n;
        var baseline: [16]Trace.Call = undefined;
        @memcpy(baseline[0..count], Trace.calls[0..count]);
        Trace.n = 0;
        try head.project(&pass, 0x1000, 0x6000, 0x3000, rows);
        try t.expectEqual(count, Trace.n);
        try t.expectEqualDeep(baseline[0..count], Trace.calls[0..count]);
    }
    head.draft.native = null;
    try head.project(&pass, 0x1000, 0x6000, 0x3000, 1);
    try t.expectEqual(@as(usize, 1), pass.fallback);
}

test "Flash Next proposal: exact BF16 and affine rows pack identically for all default IDs" {
    const before = try old.rawIds(gpa, table);
    defer gpa.free(before);
    const ids = try family.mtp.draftIds(gpa);
    defer gpa.free(ids);
    for ([_]bool{ false, true }) |affine| {
        const row: usize = if (affine) 32 else 128;
        const source = try gpa.alignedAlloc(u8, .@"16", 248320 * row);
        defer gpa.free(source);
        for (source, 0..) |*b, i| b.* = @truncate(i *% 113 +% (i / row));
        const side = try gpa.alloc(u8, if (affine) 248320 * 4 else 0);
        defer gpa.free(side);
        for (side, 0..) |*b, i| b.* = @truncate(i *% 71);
        var artifact = try pv.legacy.artifact(gpa, ids, .{ .vocab = 248320, .hidden = 64, .format = if (affine) "affine4" else "bf16", .group = if (affine) 32 else 0, .scale_recipe = "existing draft head" });
        defer artifact.deinit();
        var selected = try (core.proposal_rows.Source{ .identity = artifact.identity(), .storage = if (affine) .affine else .bf16, .recipe = .qlinear_decode, .codes = .{ .bytes = source, .stride = row }, .scales = if (affine) .{ .bytes = side, .stride = 4 } else .{}, .biases = if (affine) .{ .bytes = side, .stride = 4 } else .{}, .scale_dtype = .bf16, .bias_dtype = .bf16 }).extract(gpa, &artifact, 1, std.math.maxInt(usize));
        defer selected.deinit();
        try t.expectEqual(@as(u32, 79591), selected.physical);
        try selected.scores().check();
        try t.expectEqualSlices(u32, ids, selected.scores().ids.?);
        for (before, 0..) |id, i| {
            try t.expectEqualSlices(u8, source[id * row ..][0..row], selected.codes[i * row ..][0..row]);
            if (affine) {
                try t.expectEqualSlices(u8, side[id * 4 ..][0..4], selected.scales[i * 4 ..][0..4]);
                try t.expectEqualSlices(u8, side[id * 4 ..][0..4], selected.biases[i * 4 ..][0..4]);
            }
        }
        if (!affine) {
            const expected = try gpa.alloc(u16, before.len * 64);
            defer gpa.free(expected);
            for (before, 0..) |id, i| @memcpy(expected[i * 64 ..][0..64], std.mem.bytesAsSlice(u16, source[id * 128 ..][0..128]));
            const n = cuda.nvfp4.padded(before.len);
            const old_pack = try gpa.alloc(u16, n * 64);
            defer gpa.free(old_pack);
            const new_pack = try gpa.alloc(u16, n * 64);
            defer gpa.free(new_pack);
            cuda.qmmf.packBf16(old_pack, expected, before.len, 64);
            cuda.qmmf.packBf16(new_pack, @alignCast(std.mem.bytesAsSlice(u16, selected.codes)), ids.len, 64);
            try t.expectEqualSlices(u16, old_pack, new_pack);
        }
    }
}

fn allocateLegacy(a: std.mem.Allocator) !void {
    var artifact = try pv.legacy.artifact(a, &.{ 0, 3, 7 }, .{ .vocab = 11, .hidden = 64, .format = "bf16", .group = 0, .scale_recipe = "existing draft head" });
    defer artifact.deinit();
    try t.expectEqualSlices(u32, &.{ 0, 3, 7 }, artifact.ids());
}
test "Flash Next proposal: legacy artifact allocation failure releases the complete immutable owner" {
    try t.checkAllAllocationFailures(gpa, allocateLegacy, .{});
}
