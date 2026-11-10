//! The core depth sweep through the XPU DFlash2 adapter, against main's private drafting path as the reference.
const std = @import("std");
const lanes = @import("lanes");
const sweep = lanes.drafter_depth_sweep;
const be = lanes.backend;
const Stream = lanes.Stream;
const ad = @import("xpu_dflash2.zig");
const x = @import("xpu_dflash2_test.zig");

const gpa = std.testing.allocator;
const SLOT = 2; // not the first slot: the adapter and the reference both look the stream up

/// The adapter over its own recorded drafter; the reference: main's private path over a second one, fed the round.
const Harness = struct {
    sampling: ?lanes.Sampling,
    fake: x.Fake = .{},
    host: x.Host = .{},
    ref: x.Fake = .{},
    ref_host: x.Host = .{},
    adapter: ad.Adapter(x.M, x.Host) = undefined,
    stream: Stream = undefined,
    prompt: []u32 = &.{},
    chunk: [1]x.Chunk = undefined, // the prompt pass's one chunk, for the private path's prefill
    prefilled: bool = false,

    fn harness(h: *Harness) sweep.Harness {
        return .{ .ctx = h, .begin = begin, .verify = verify, .want = want, .end = end };
    }

    fn self(ctx: *anyopaque) *Harness {
        return @ptrCast(@alignCast(ctx));
    }

    fn begin(ctx: *anyopaque, n: usize) anyerror!sweep.Case {
        const h = self(ctx);
        h.fake = .{};
        h.ref = .{};
        h.host = .{};
        h.ref_host = .{};
        h.prefilled = false;
        h.prompt = try gpa.alloc(u32, n);
        for (h.prompt, 0..) |*t, i| t.* = @intCast((i * 7 + 3) % 97);
        h.stream = try Stream.init(gpa, .{ .id = "s", .prompt = h.prompt, .max_new = 64, .sampling = h.sampling });
        h.host.owners[SLOT] = &h.stream;
        h.ref_host.owners[SLOT] = &h.stream;
        h.adapter = .{ .d = &h.fake, .host = &h.host };
        const next: sweep.Round = .{ .pending = @intCast(11 + n % 5), .position = n + 1 };
        const prompt: sweep.Step = .{ .absorb = h.rows(0, n, true), .next = next };
        return .{ .drafter = h.adapter.drafter(), .stream = &h.stream, .prompt = prompt };
    }

    /// `n` tap rows from `start` as the lane target hands them over: device bf16 rows at the taps buffer.
    fn rows(h: *Harness, start: u64, n: usize, prompt: bool) lanes.drafter.Absorb {
        const f: be.Features = .{ .buffer = x.BASE, .rows = @intCast(n), .row_bytes = x.ROW, .space = .device, .dtype = .bf16 };
        return .{ .stream = &h.stream, .features = f, .start = start, .follow = &.{}, .prompt = prompt };
    }

    /// The target keeps the pending row and a position-keyed count of the drafts, then picks the next pending.
    fn verify(ctx: *anyopaque, r: sweep.Round, drafts: []const u32) anyerror!sweep.Step {
        const h = self(ctx);
        const k = (r.position *% 2654435761) % (drafts.len + 1);
        const next: sweep.Round = .{ .pending = @intCast((r.position * 13 + k) % 97), .position = r.position + 1 + k };
        return .{ .absorb = h.rows(r.position - 1, 1 + k, false), .next = next };
    }

    /// main's private drafting path for one stream at this depth, its context at the round's position.
    fn want(ctx: *anyopaque, r: sweep.Round, out: []u32) anyerror!void {
        const h = self(ctx);
        if (!h.prefilled) try h.samePrompt();
        h.ref.ends[SLOT] = @intCast(r.position - 1);
        var p: x.Private = .{ .d = &h.ref, .h = &h.ref_host };
        const depth: u32 = @intCast(out.len);
        const q = .{ .s = 0, .rows = @as(?[]const u32, null), .depth = depth, .start = r.position - 1, .first = r.pending };
        try p.draft(&.{x.request(&.{&h.stream}, q)});
        if (p.chain_n[SLOT] != out.len) return error.ReferenceShort;
        @memcpy(out, p.chain[SLOT][0..out.len]);
    }

    /// Before the first hold: the adapter's prompt calls (skip past WIN, then taps) are the private prefill's.
    fn samePrompt(h: *Harness) !void {
        h.prefilled = true;
        h.chunk = .{.{ .start = 0, .rows = @intCast(h.prompt.len) }};
        h.ref_host.chunks = &h.chunk;
        var p: x.Private = .{ .d = &h.ref, .h = &h.ref_host };
        try p.prefill(&h.stream, SLOT, 0);
        try std.testing.expectEqualSlices(u64, h.ref.log.items, h.fake.log.items);
        h.ref.log.clearRetainingCapacity();
    }

    fn end(ctx: *anyopaque) void {
        const h = self(ctx);
        h.fake.log.deinit(gpa);
        h.ref.log.deinit(gpa);
        h.stream.deinit(gpa);
        gpa.free(h.prompt);
    }
};

test "draft sources: XPU DFlash2 holds exactly each depth's drafts, main's private path's, greedy and sampled" {
    for ([_]?lanes.Sampling{ null, .{ .seed = 7, .temperature = 0.8 } }) |s| {
        var h: Harness = .{ .sampling = s };
        // 2100: rows past WIN skipped as main's prefill skips them (samePrompt compares the drafter calls)
        const r = try sweep.sweep(gpa, h.harness(), .{ .prompts = &.{ 6, 40, 2100 } });
        // three prompts x (depths 1..7 + one zig-zag run) x 12 rounds; drafts 12 x 28 fixed + 48 zig-zag a prompt
        try std.testing.expectEqual(sweep.Report{ .runs = 24, .rounds = 288, .drafts = 3 * (12 * 28 + 48) }, r);
    }
}
