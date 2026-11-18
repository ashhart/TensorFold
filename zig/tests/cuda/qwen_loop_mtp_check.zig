//! SenseNova Loop greedy parity and forced rejection on the real checkpoint.
const std = @import("std");
const core = @import("core");
const lanes = @import("lanes");
const q = @import("qwen35");
const check = @import("check.zig");
const dr = lanes.drafter;

const Reject = struct {
    real: dr.Drafter,
    table: dr.Drafter.VTable = undefined,
    oracle: []const u32,
    vocab: u32,
    index: usize,
    injections: usize = 0,
    reached: usize = 0,

    fn interface(p: *Reject) dr.Drafter {
        p.table = p.real.vtable.*;
        p.table.taps = taps;
        p.table.facts = facts;
        p.table.open = open;
        p.table.absorb = absorb;
        p.table.hold = hold;
        p.table.held = held;
        p.table.release = release;
        if (p.table.hold_planned != null) p.table.hold_planned = holdPlanned;
        if (p.table.clear_proposals != null) p.table.clear_proposals = clear;
        if (p.table.tree != null) p.table.tree = tree;
        if (p.table.chances != null) p.table.chances = chances;
        return .{ .kind = p.real.kind, .proposal_head = p.real.proposal_head, .ptr = p, .vtable = &p.table };
    }
    fn self(p: *anyopaque) *Reject {
        return @ptrCast(@alignCast(p));
    }
    fn taps(p: *anyopaque) []const u32 {
        return self(p).real.taps();
    }
    fn facts(p: *anyopaque) dr.Facts {
        return self(p).real.facts();
    }
    fn open(p: *anyopaque, s: *lanes.Stream) !void {
        return self(p).real.open(s);
    }
    fn absorb(p: *anyopaque, items: []const dr.Absorb) !void {
        return self(p).real.absorb(items);
    }
    fn hold(p: *anyopaque, items: []const dr.Hold) !void {
        return self(p).real.hold(items);
    }
    fn holdPlanned(p: *anyopaque, items: []const dr.Hold) !void {
        const r = self(p).real;
        return r.vtable.hold_planned.?(r.ptr, items);
    }
    fn clear(p: *anyopaque, s: *lanes.Stream) void {
        const r = self(p).real;
        r.vtable.clear_proposals.?(r.ptr, s);
    }
    fn tree(p: *anyopaque, s: *lanes.Stream, a: std.mem.Allocator) !?lanes.stream.Held {
        return self(p).real.tree(s, a);
    }
    fn chances(p: *anyopaque, s: *lanes.Stream, out: []f64) !bool {
        return self(p).real.chances(s, out);
    }
    fn held(p: *anyopaque, streams: []const *lanes.Stream, out: []const []u32) !void {
        const x = self(p);
        try x.real.held(streams, out);
        for (streams, out) |s, tokens| {
            if (tokens.len <= x.index) continue;
            const pos = s.emitted().len + x.index;
            if (pos >= x.oracle.len) continue;
            if (!std.mem.eql(u32, s.emitted(), x.oracle[0..s.emitted().len])) return error.LoopMtpTokenMismatch;
            const first = s.emitted().len;
            if (std.mem.eql(u32, tokens[0..x.index], x.oracle[first..pos])) x.reached += 1;
            const wrong = (x.oracle[pos] + 1) % x.vocab;
            tokens[x.index] = if (wrong == tokens[x.index]) (wrong + 1) % x.vocab else wrong;
            x.injections += 1;
        }
    }
    fn release(p: *anyopaque, s: *lanes.Stream) void {
        self(p).real.release(s);
    }
};

const Result = struct { ids: []u32, accepted: u64, proposed: u64, rounds: u64 };

fn decode(gpu: check.Gpu, backend: lanes.backend.Backend, facts: lanes.Model, prompt: []const u32, drafts: bool, count: u32) !Result {
    const a = gpu.gpa;
    var cfg = try lanes.Config.init(a, facts, 16, 15);
    defer cfg.deinit(a);
    var clock: lanes.fake.FixedClock = .{};
    var engine = lanes.Engine.init(a, &cfg, backend, clock.clock());
    defer engine.deinit();
    var stream = try lanes.Stream.init(a, .{ .id = "loop-mtp", .prompt = prompt, .max_new = count, .drafts = drafts });
    defer stream.deinit(a);
    try engine.addStream(&stream);
    while (engine.activeCount() > 0) try engine.step();
    return .{ .ids = try a.dupe(u32, stream.emitted()), .accepted = stream.accepted, .proposed = stream.drafted, .rounds = stream.rounds };
}

fn fresh(gpu: check.Gpu, backend: lanes.backend.Backend, facts: lanes.Model, prompt: []const u32, want: []const u32) !void {
    const got = try decode(gpu, backend, facts, prompt, false, 16);
    defer gpu.gpa.free(got.ids);
    if (!std.mem.eql(u32, got.ids, want)) return error.LoopMtpFreshMismatch;
}

pub fn run(gpu: check.Gpu) !void {
    const model = std.c.getenv("TF_LOOP9B_DIR") orelse return error.MissingLoopModel;
    const dir = std.mem.span(model);
    const draft = if (std.c.getenv("TF_LOOP9B_MTP")) |v| std.mem.span(v) else dir;
    var tokenizer = try core.tokenizer.loadTokenizer(gpu.io, gpu.gpa, dir);
    defer tokenizer.deinit();
    const prompt = try tokenizer.encode(gpu.gpa, "Explain why the night sky appears dark.");
    defer gpu.gpa.free(prompt);
    const second = try tokenizer.encode(gpu.gpa, "Describe how a seed becomes a tree.");
    defer gpu.gpa.free(second);
    const loaded = try q.native_loop.open(gpu.gpa, gpu.io, gpu.ctx, dir, null, .{ .context = 512, .drafts = true, .drafter = draft });
    defer loaded.deinit(loaded.ctx);
    const opened = try q.native_loop.openDrafter(gpu.gpa, gpu.io, loaded.ctx, draft, 4);
    defer opened.close(opened.ctx);
    const mtp: *q.mtp.Mtp = @ptrCast(@alignCast(opened.ctx));
    if (opened.drafter.facts().depth != 3) return error.LoopMtpNeedsDepthThree;
    if (mtp.e.g.vocab < 3) return error.LoopMtpVocabTooSmall;
    var proxy: Reject = .{ .real = opened.drafter, .oracle = &.{}, .vocab = @intCast(mtp.e.g.vocab), .index = 0 };
    var rejecting = try lanes.drafted.Drafted.init(gpu.gpa, loaded.backend, proxy.interface());
    defer rejecting.deinit();
    var wrapped = try lanes.drafted.Drafted.init(gpu.gpa, loaded.backend, opened.drafter);
    defer wrapped.deinit();
    try loaded.reserve(loaded.ctx, 1);
    try opened.reserve(opened.ctx, 1);
    const reference = try decode(gpu, loaded.backend, loaded.facts, second, false, 16);
    defer gpu.gpa.free(reference.ids);
    if (reference.ids.len != 16) return error.LoopMtpFreshMismatch;
    const plain = try decode(gpu, loaded.backend, loaded.facts, prompt, false, 64);
    defer gpu.gpa.free(plain.ids);
    std.debug.print("Loop MTP k=plain accepted/proposed={d}/{d} rounds={d}\n", .{ plain.accepted, plain.proposed, plain.rounds });
    if (plain.ids.len != 64) return error.LoopMtpTokenMismatch;
    try fresh(gpu, loaded.backend, loaded.facts, second, reference.ids);
    proxy.oracle = plain.ids;
    for (0..3) |k| {
        proxy.index = k;
        proxy.injections = 0;
        proxy.reached = 0;
        var facts = rejecting.facts(loaded.facts);
        facts.plain_guard = false;
        facts.mtp_step_ms = 0;
        facts.draft_prior = &.{ 1, 1, 1 };
        facts.window_costs = &.{};
        facts.lane_costs = &.{};
        const got = try decode(gpu, rejecting.backend(), facts, prompt, true, 64);
        defer gpu.gpa.free(got.ids);
        std.debug.print("Loop MTP k={d} accepted/proposed={d}/{d} rounds={d}\n", .{ k, got.accepted, got.proposed, got.rounds });
        if (got.proposed == 0) return error.NoDraftsProposed;
        if (proxy.injections == 0 or proxy.reached == 0) return error.RejectionDepthNotReached;
        if (got.accepted > @as(u64, @intCast(k)) * got.rounds or !std.mem.eql(u32, plain.ids, got.ids)) return error.LoopMtpTokenMismatch;
        try fresh(gpu, loaded.backend, loaded.facts, second, reference.ids);
    }
    var facts = wrapped.facts(loaded.facts);
    facts.plain_guard = false;
    facts.mtp_step_ms = 0;
    const got = try decode(gpu, wrapped.backend(), facts, prompt, true, 64);
    defer gpu.gpa.free(got.ids);
    std.debug.print("Loop MTP k=none accepted/proposed={d}/{d} rounds={d}\n", .{ got.accepted, got.proposed, got.rounds });
    if (got.proposed == 0 or !std.mem.eql(u32, plain.ids, got.ids)) return error.LoopMtpTokenMismatch;
    try fresh(gpu, loaded.backend, loaded.facts, second, reference.ids);
}
