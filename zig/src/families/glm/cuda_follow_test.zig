//! Host checks for GLM's follower control headers, folded keep ordering and link allocation cleanup.
const std = @import("std");
const cuda = @import("cuda");
const engine = @import("cuda_engine.zig");
const dflash = @import("dflash.zig");
const images_mod = @import("cuda_images.zig");
const follow = @import("cuda_follow.zig");
const Kind = follow.Kind;
const Header = follow.Header;
const Link = follow.Link;
const keepHeader = follow.keepHeader;
const outgoing = follow.outgoing;
const foldableStep = follow.foldableStep;
const bindStep = follow.bindStep;
const promptRows = follow.promptRows;
const structuredStep = follow.structuredStep;
const indexSplit = follow.indexSplit;
const spanWords = follow.spanWords;

test "a step's header is 64 bytes and keeps its fields" {
    const h: Header = .{ .kind = @backingInt(Kind.draft), .n = 3, .depth = 2, .flags = 1, .a = 7, .seed = 1 << 62, .temperature = 0.6, .top_p = 0.95, .top_k = 20 };
    const bytes = std.mem.asBytes(&h);
    var back: Header = undefined;
    @memcpy(std.mem.asBytes(&back), bytes);
    try std.testing.expectEqual(h, back);
    try std.testing.expectEqual(Kind.draft, @as(Kind, @fromBackingInt(@intCast(back.kind))));
}

test "128 video pairs cross the span serializer and a follower's receiver; one word short is refused" {
    const a = std.testing.allocator;
    var spans: [128]images_mod.Span = undefined;
    for (&spans, 0..) |*sp, i| sp.* = .{ .at = @intCast(i * 2), .n = 1, .row = @intCast(i) };
    const words = try spanWords(a, &spans, 3 * spans.len);
    defer a.free(words);
    var v: images_mod.Images = undefined;
    v.gpa = a;
    v.spans = .empty;
    v.slot = 9;
    defer v.spans.deinit(a);
    v.limits = .{ .image_tokens = 64, .max_images = 8, .video_tokens = 128, .max_frames = 256 };
    v.video_ids = .{ 1, 2, 3 };
    try v.take(9, words);
    try std.testing.expectEqualSlices(images_mod.Span, &spans, v.spans.items);
    try std.testing.expectEqual(@as(u32, 9), v.slot);
    try std.testing.expectEqual(@as(usize, 128), v.rowCount());
    try std.testing.expectError(error.TooManyIds, spanWords(a, &spans, 3 * spans.len - 1));
}

const FoldEvent = union(enum) { bind: u32, keep: usize, step: Kind };

const FoldRecorder = struct {
    gpa: std.mem.Allocator = std.testing.allocator,
    events: [8]FoldEvent = undefined,
    len: usize = 0,

    fn add(r: *FoldRecorder, event: FoldEvent) void {
        std.debug.assert(r.len < r.events.len);
        r.events[r.len] = event;
        r.len += 1;
    }
    pub fn bind(r: *FoldRecorder, slot: u32) !void {
        r.add(.{ .bind = slot });
    }
    pub fn keep(r: *FoldRecorder, n: usize) !void {
        r.add(.{ .keep = n });
    }
    pub fn verify(r: *FoldRecorder, ids: []const u32, out: []u32, bits: u64) !void {
        _ = ids;
        _ = out;
        _ = bits;
        r.add(.{ .step = .verify });
    }
    pub fn draft(r: *FoldRecorder, from_prompt: bool, ids: []u32, depth: u32, drafts: *[engine.max_drafts]u32, chances: ?*[engine.max_drafts]f64) !void {
        _ = from_prompt;
        _ = ids;
        _ = depth;
        _ = drafts;
        _ = chances;
        r.add(.{ .step = .draft });
    }
    pub fn draftBlock(r: *FoldRecorder, reqs: []const dflash.Req) !void {
        _ = reqs;
        r.add(.{ .step = .block });
    }
};

fn recordFoldable(r: *FoldRecorder, h: Header, ids: []const u32) !void {
    var sampled: [engine.shared_rows]u32 = undefined;
    var drafts: [engine.max_drafts]u32 = undefined;
    var chances: [engine.max_drafts]f64 = undefined;
    try std.testing.expect(try foldableStep(r, h, ids, &sampled, &drafts, &chances));
}

fn recordBound(r: *FoldRecorder, h: Header, ids: []const u32) !void {
    try bindStep(r, h);
    try recordFoldable(r, h, ids);
}

test "TF_GLM_KEEP_FOLD: every foldable step runs k0 or kR before it, with drafts off and on" {
    const ids = [_]u32{ 0, 0, 0 };
    for ([_]Kind{ .verify, .draft, .block }) |kind| {
        for (0..engine.max_drafts + 1) |round| {
            for ([_]usize{ 0, round }) |k| {
                var plain: FoldRecorder = .{};
                try recordFoldable(&plain, keepHeader(k), &ids);
                const h: Header = .{ .kind = @backingInt(kind), .depth = if (kind == .block) 1 else @intCast(round), .flags = if (kind == .draft and round > 0) 3 else 0 };
                try recordFoldable(&plain, h, &ids);

                var folded: FoldRecorder = .{};
                var with_keep = h;
                with_keep.a = k + 1;
                try recordFoldable(&folded, with_keep, &ids);

                try std.testing.expectEqual(@as(usize, 2), plain.len);
                try std.testing.expectEqual(FoldEvent{ .keep = k }, plain.events[0]);
                try std.testing.expectEqual(FoldEvent{ .step = kind }, plain.events[1]);
                try std.testing.expectEqualSlices(FoldEvent, plain.events[0..plain.len], folded.events[0..folded.len]);
            }
        }
    }
}

test "TF_GLM_KEEP_FOLD: consecutive keeps on different slots preserve block bind and commit order" {
    const ids = [_]u32{ 0, 0, 0 };
    var plain: FoldRecorder = .{};
    var first = keepHeader(1);
    first.slot = 3;
    try recordBound(&plain, first, &ids);
    var second = keepHeader(2);
    second.slot = 4;
    try recordBound(&plain, second, &ids);
    const block: Header = .{ .kind = @backingInt(Kind.block), .depth = 1, .slot = 4 };
    try recordBound(&plain, block, &ids);

    var folded: FoldRecorder = .{};
    try recordBound(&folded, first, &ids);
    var with_keep = block;
    with_keep.a = 3;
    try recordBound(&folded, with_keep, &ids);

    try std.testing.expectEqualSlices(FoldEvent, plain.events[0..plain.len], folded.events[0..folded.len]);
    try std.testing.expectEqual(@as(usize, 5), folded.len);
    try std.testing.expectEqual(FoldEvent{ .bind = 4 }, folded.events[2]);
    try std.testing.expectEqual(FoldEvent{ .keep = 2 }, folded.events[3]);
}

test "TF_GLM_KEEP_FOLD: same-slot steps fold; release, end and a changed slot flush the keep first" {
    var link: Link = undefined;
    link.commit_keep = 5;
    link.commit_slot = 7;
    const verify_h: Header = .{ .kind = @backingInt(Kind.verify) };
    const same = outgoing(&link, verify_h, 7);
    try std.testing.expect(same.before == null and same.folded and same.h.a == 6);

    for ([_]Kind{ .release, .end, .drop, .idle }) |kind| {
        const stopped = outgoing(&link, .{ .kind = @backingInt(kind) }, 7);
        try std.testing.expect(!stopped.folded and stopped.h.a == 0);
        try std.testing.expectEqual(@as(u32, 7), stopped.before_slot);
        try std.testing.expectEqual(@as(Kind, .keep), @as(Kind, @fromBackingInt(@intCast(stopped.before.?.kind))));
        try std.testing.expectEqual(@as(u64, 5), stopped.before.?.a);
    }
    const moved = outgoing(&link, verify_h, 8);
    try std.testing.expect(!moved.folded and moved.before != null and moved.before_slot == 7);

    link.commit_keep = null;
    const ordinary = outgoing(&link, verify_h, 7);
    try std.testing.expect(ordinary.before == null and !ordinary.folded and ordinary.h.a == 0);
}

test "a link whose host allocation fails frees its device buffer" {
    const Mock = struct {
        var frees: usize = 0;
        fn alloc(ptr: *cuda.abi.DevicePtr, _: usize) callconv(.c) cuda.abi.Result {
            ptr.* = 4096;
            return 0;
        }
        fn free(_: cuda.abi.DevicePtr) callconv(.c) cuda.abi.Result {
            frees += 1;
            return 0;
        }
    };
    Mock.frees = 0;
    var d: cuda.Driver = undefined;
    d.api.cuMemAlloc_v2 = Mock.alloc;
    d.api.cuMemFree_v2 = Mock.free;
    var lib: cuda.nccl.Library = undefined;
    var failing = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = 0 });
    const before = cuda.usage(false).device;
    defer {
        if (Mock.frees == 0) {
            var leaked: cuda.DeviceBuffer = .{ .d = &d, .ptr = 4096, .len = 64 + 128 * 4 };
            leaked.free();
        }
    }
    try std.testing.expectError(error.OutOfMemory, Link.init(failing.allocator(), &d, &lib, null, .{ .d = &d, .handle = null }, 128, false));
    try std.testing.expectEqual(@as(usize, 1), Mock.frees);
    try std.testing.expectEqual(before, cuda.usage(false).device);
}

test "flag 1 asks for prompt rows on prefill and chunk steps only (begin's is sampling, draft's from the prompt)" {
    try std.testing.expect(promptRows(.{ .kind = @backingInt(Kind.chunk), .flags = 1, .depth = 2048, .top_k = 4000 }));
    try std.testing.expect(promptRows(.{ .kind = @backingInt(Kind.prefill), .flags = 1 }));
    try std.testing.expect(!promptRows(.{ .kind = @backingInt(Kind.chunk), .depth = 2048, .top_k = 4000 }));
    try std.testing.expect(!promptRows(.{ .kind = @backingInt(Kind.begin), .flags = 1, .seed = 7 }));
    try std.testing.expect(!promptRows(.{ .kind = @backingInt(Kind.draft), .flags = 1, .depth = 3 }));
}

test "flag 1 on verify and shared steps draws from whole rows; a prompt step's flag 1 does not" {
    try std.testing.expect(structuredStep(.{ .kind = @backingInt(Kind.verify), .flags = 1, .n = 4 }));
    try std.testing.expect(structuredStep(.{ .kind = @backingInt(Kind.shared), .flags = 1, .depth = 2 }));
    try std.testing.expect(!structuredStep(.{ .kind = @backingInt(Kind.verify), .n = 4 }));
    try std.testing.expect(!structuredStep(.{ .kind = @backingInt(Kind.chunk), .depth = 2048 }));
    try std.testing.expect(!structuredStep(.{ .kind = @backingInt(Kind.begin), .flags = 1 }));
}

test "flag 2 on a chunk splits its indexer rows across the ranks, beside flag 1's prompt rows; elsewhere it does not" {
    try std.testing.expect(indexSplit(.{ .kind = @backingInt(Kind.chunk), .flags = 2, .depth = 65536, .top_k = 70000 }));
    try std.testing.expect(indexSplit(.{ .kind = @backingInt(Kind.chunk), .flags = 3, .depth = 65536, .top_k = 70000 }));
    try std.testing.expect(promptRows(.{ .kind = @backingInt(Kind.chunk), .flags = 3, .depth = 65536, .top_k = 70000 }));
    try std.testing.expect(!indexSplit(.{ .kind = @backingInt(Kind.chunk), .flags = 1, .depth = 65536, .top_k = 70000 }));
    try std.testing.expect(!indexSplit(.{ .kind = @backingInt(Kind.prefill), .flags = 2 }));
}
