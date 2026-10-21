//! Real native DFlash frames under a fixed schedule; every stream's returned bytes must survive regrouping.
const std = @import("std");
const q = @import("qwen35");
const lanes = @import("lanes");
const check = @import("check.zig");
const plan = @import("qwen_draft_frames_plan.zig");
const control = @import("qwen_draft_frames_control.zig");
const dr = lanes.drafter;
const be = lanes.backend;

const Proxy = struct {
    d: *q.draft.Draft,
    prompt: bool = false,
    stats: control.Stats = .{},
    fn interface(p: *Proxy) dr.Drafter {
        return .{ .ptr = p, .vtable = &.{ .taps = taps, .facts = facts, .open = open, .absorb = absorb, .hold = hold, .held = held, .release = release } };
    }
    fn self(p: *anyopaque) *Proxy {
        return @ptrCast(@alignCast(p));
    }
    fn taps(p: *anyopaque) []const u32 {
        return self(p).d.drafter().taps();
    }
    fn facts(p: *anyopaque) dr.Facts {
        return self(p).d.drafter().facts();
    }
    fn open(p: *anyopaque, s: *lanes.Stream) !void {
        return self(p).d.drafter().open(s);
    }
    fn absorb(p: *anyopaque, items: []const dr.Absorb) !void {
        return self(p).d.drafter().absorb(items);
    }
    fn hold(p: *anyopaque, items: []const dr.Hold) !void {
        const x = self(p);
        return control.hold(x.d, items, x.prompt, &x.stats);
    }
    fn held(p: *anyopaque, streams: []const *lanes.Stream, out: []const []u32) !void {
        return self(p).d.drafter().held(streams, out);
    }
    fn release(p: *anyopaque, s: *lanes.Stream) void {
        self(p).d.drafter().release(s);
    }
};

const Device = struct {
    a: std.mem.Allocator,
    backend: be.Backend,
    drafter: dr.Drafter,
    d: *q.draft.Draft,
    capture: *plan.Capture,
    streams: [plan.streams]lanes.Stream = undefined,
    initialized: usize = 0,
    processed: [plan.streams]usize = plan.lengths,
    pending: [plan.streams]u32 = undefined,

    fn close(x: *Device) void {
        for (x.streams[0..x.initialized]) |*s| {
            x.backend.release(s);
            s.deinit(x.a);
        }
    }

    fn begin(x: *Device, ids: []const u32) !void {
        const names = [_][]const u8{ "0", "1", "2", "3", "4", "5", "6", "7" };
        for (&x.streams, plan.lengths, names, 0..) |*s, len, name, id| {
            s.* = try lanes.Stream.init(x.a, .{ .id = name, .prompt = ids[0..len], .max_new = plan.rounds * plan.block + 1, .loop_guard = false });
            x.initialized += 1;
            try x.backend.prefill(s);
            const token = try x.backend.read(try x.backend.first(s, len));
            if (try s.commit(x.a, &.{token}, &.{}) != 1) return error.ShortFrameCommit;
            s.pending = token;
            x.pending[id] = token;
            try x.backend.draft(&.{.{ .stream = s, .first = .{ .value = token }, .follow = &.{}, .rows = null, .start = len - 1, .position = len + 1, .depth = 0 }});
        }
    }

    pub fn frame(x: *Device, round: usize, group: []const usize) !void {
        var holds: [plan.streams]dr.Hold = undefined;
        var streams: [plan.streams]*lanes.Stream = undefined;
        var proposals: [plan.streams][plan.depth]u32 = undefined;
        var outputs: [plan.streams][]u32 = undefined;
        for (group, 0..) |id, i| {
            streams[i] = &x.streams[id];
            outputs[i] = &proposals[i];
            holds[i] = .{ .stream = streams[i], .pending = x.pending[id], .position = x.processed[id] + 1, .depth = plan.depth };
        }
        try x.drafter.hold(holds[0..group.len]);
        try x.drafter.held(streams[0..group.len], outputs[0..group.len]);
        const words = try x.a.dupe(u8, x.d.staged.bytes[q.draft.frame.words_at..][0 .. group.len * plan.word_bytes]);
        defer x.a.free(words);
        const selector_bytes = plan.depth * x.d.c.rank * 2;
        const selector = try x.a.dupe(u8, x.d.staged.bytes[q.draft.frame.hp_at..][0 .. group.len * selector_bytes]);
        defer x.a.free(selector);
        var positions: [plan.streams][plan.block]u64 = undefined;
        var sampled: [plan.streams][plan.block]u32 = undefined;
        var verified_drafts: [plan.streams][plan.depth]u32 = undefined;
        var windows: [plan.streams]be.Window = undefined;
        var verified: [plan.streams]be.Verified = undefined;
        for (group, 0..) |id, i| {
            for (&positions[i], 0..) |*p, r| p.* = x.processed[id] + r + 1;
            windows[i] = .{ .stream = streams[i], .pending = x.pending[id], .held = 0, .tokens = &proposals[i], .parents = null, .positions = &positions[i] };
            verified[i] = .{ .sampled = &sampled[i], .drafts = &verified_drafts[i] };
        }
        try x.backend.verify(windows[0..group.len], verified[0..group.len]);
        var paths: [plan.streams][plan.block]u32 = undefined;
        var kept: [plan.streams][]const u32 = undefined;
        var update: [plan.streams]be.DraftRequest = undefined;
        for (group, 0..) |id, i| {
            if (!std.mem.eql(u32, &proposals[i], &verified_drafts[i])) return error.ChangedFrameProposals;
            const accepted = try plan.accepted(&proposals[i], &sampled[i]);
            const count = accepted + 1;
            for (paths[i][0..count], 0..) |*p, row| p.* = @intCast(row);
            kept[i] = paths[i][0..count];
            try x.capture.write(round, id, &proposals[i], words[i * plan.word_bytes ..][0..plan.word_bytes], selector[i * selector_bytes ..][0..selector_bytes], accepted);
            if (try streams[i].commit(x.a, sampled[i][0..count], &.{}) != count) return error.ShortFrameCommit;
            update[i] = .{ .stream = streams[i], .follow = sampled[i][0..count], .rows = kept[i], .start = x.processed[id], .position = x.processed[id] + count + 1, .depth = 0 };
            x.processed[id] += count;
            x.pending[id] = sampled[i][accepted];
            streams[i].pending = x.pending[id];
        }
        try x.backend.keep(windows[0..group.len], kept[0..group.len]);
        try x.backend.draft(update[0..group.len]);
    }
};

fn compare(base: plan.Capture, candidate: plan.Capture, mode: plan.Mode, negative: bool) !void {
    if (try base.difference(candidate)) |diff| std.debug.print("DIFF qwen-draft-frames {t}: stream={d} round={d} byte={d}\n", .{ mode, diff.stream, diff.round, diff.byte });
    _ = try plan.compare(base, candidate, negative);
    std.debug.print("{s} qwen-draft-frames {t}: {d} complete stream-round records\n", .{ if (negative) "CONTROL RED" else "PASS", mode, plan.streams * plan.rounds });
}

fn cases(gpu: check.Gpu, model: []const u8, drafter: []const u8, ids: []const u32, out: *std.Io.Writer) !void {
    if (try gpu.ctx.capability() != 121) return error.FrameGateNeedsGB10;
    const capacity = plan.lengths[plan.streams - 1] + (plan.rounds + 1) * plan.block;
    const loaded = try q.native.open(gpu.gpa, gpu.io, gpu.ctx, model, null, .{ .context = capacity, .drafts = true });
    defer loaded.deinit(loaded.ctx);
    const opened = try q.native.openDrafter(gpu.gpa, gpu.io, loaded.ctx, drafter, 4);
    defer opened.close(opened.ctx);
    const d: *q.draft.Draft = @ptrCast(@alignCast(opened.ctx));
    if (d.e.g.hidden != 5120 or d.e.g.layers != 64 or d.e.g.experts != 0 or d.c.block != plan.block or d.e.head != .a4 or d.e.window_rows < plan.block or d.e.verify_rows < plan.block) return error.FrameGateNeeds27BAffine4;
    for (ids) |id| if (id >= d.e.g.vocab) return error.InvalidQwenToken;
    var proxy: Proxy = .{ .d = d };
    var wrapped = try lanes.drafted.Drafted.init(gpu.gpa, loaded.backend, proxy.interface());
    defer wrapped.deinit();
    try loaded.reserve(loaded.ctx, plan.streams);
    try opened.reserve(opened.ctx, plan.streams);
    try std.json.Stringify.value(.{ .version = 1, .streams = plan.streams, .rounds = plan.rounds, .block = plan.block, .selector_rank = d.c.rank, .words_per_row = 43, .byte_order = .little, .prompt_lengths = plan.lengths }, .{}, out);
    try out.writeByte('\n');
    var base = try plan.Capture.init(gpu.gpa, d.c.rank);
    defer base.deinit();
    for (0..plan.partitions.len + 1) |index| {
        const negative = index == plan.partitions.len;
        const partition = plan.partitions[if (negative) 1 else index];
        const mode: plan.Mode = if (negative) .prompt_control else partition.mode;
        var capture = try plan.Capture.init(gpu.gpa, d.c.rank);
        defer capture.deinit();
        proxy.prompt = negative;
        proxy.stats = .{};
        var device: Device = .{ .a = gpu.gpa, .backend = wrapped.backend(), .drafter = proxy.interface(), .d = d, .capture = &capture };
        defer device.close();
        try device.begin(ids);
        plan.schedule(&device, partition) catch |err| {
            try capture.json(out, mode);
            return err;
        };
        try capture.complete();
        try capture.json(out, mode);
        try check.expect(proxy.stats.frames == partition.groups.len * plan.rounds and proxy.stats.prompt_frames == (if (negative) @as(usize, plan.rounds) else 0), "draft frame control did not reach its hold", .{});
        std.debug.print("FRAME qwen-draft-frames {t}: holds={d} prompt-control={d} max-rows={d}\n", .{ mode, proxy.stats.frames, proxy.stats.prompt_frames, proxy.stats.max_rows });
        if (index == 0) {
            @memcpy(base.bytes, capture.bytes);
            base.seen = capture.seen;
            check.pass("qwen-draft-frames alone: {d} complete stream-round records", .{plan.streams * plan.rounds});
        } else try compare(base, capture, mode, negative);
    }
    var swapped = try plan.Capture.init(gpu.gpa, d.c.rank);
    defer swapped.deinit();
    @memcpy(swapped.bytes, base.bytes);
    swapped.seen = base.seen;
    try swapped.swapStreams(0, 1);
    try swapped.json(out, .swap_control);
    try compare(base, swapped, .swap_control, true);
}

/// IDS is a local list of at least 1,000 token ids; OUT retains raw returned bytes even if comparison fails.
pub fn run(gpu: check.Gpu, model: []const u8, drafter: []const u8, ids_path: []const u8, output: []const u8) !void {
    const text = try std.Io.Dir.cwd().readFileAlloc(gpu.io, ids_path, gpu.gpa, .limited(64 << 20));
    defer gpu.gpa.free(text);
    var ids: std.ArrayList(u32) = .empty;
    defer ids.deinit(gpu.gpa);
    var tokens = std.mem.tokenizeAny(u8, text, ", \t\r\n");
    while (tokens.next()) |token| try ids.append(gpu.gpa, try std.fmt.parseInt(u32, token, 10));
    if (ids.items.len < plan.lengths[plan.streams - 1]) return error.ShortFrameIds;
    var out: std.Io.Writer.Allocating = .init(gpu.gpa);
    defer out.deinit();
    cases(gpu, model, drafter, ids.items, &out.writer) catch |err| {
        try std.Io.Dir.cwd().writeFile(gpu.io, .{ .sub_path = output, .data = out.written() });
        return err;
    };
    try std.Io.Dir.cwd().writeFile(gpu.io, .{ .sub_path = output, .data = out.written() });
    check.pass("qwen-draft-frames: all partitions equal, both controls red; records in {s}", .{output});
}
