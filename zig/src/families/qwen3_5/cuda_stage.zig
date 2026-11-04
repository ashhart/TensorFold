//! A Qwen CUDA engine as one stage of the ring (RFC #2005 v1): its boot, its streams' slots, rank 0's rounds.
const std = @import("std");
const stage = @import("stage");
const E = @import("cuda_engine.zig");
const sf = @import("cuda_stage_forward.zig");

const plan = stage.plan;
const boot = stage.boot;

/// Every stage of one placement reports this; rank 0 refuses another.
pub const version = "tensorfold qwen3_5-cuda stage-v1";

pub const Error = error{ StreamOpen, NoFreeSlot, UnknownStream, BadAccepted, OtherPosition, ResumeNotInV1, CutsNotInV1, OtherEcho, PromptParts };

/// The ring's boundary for this engine: h and r as bf16 planes, no drafter taps (v1).
pub fn boundary(e: *const E.Engine) boot.Boundary {
    return .{ .hidden = @intCast(e.g.hidden), .row_dtype = .bf16, .planes = 2, .draft_width = 0 };
}

/// The shapes every stage runs: a prompt chunk and a round at most `max_rows` rows, `parallel` slots of `context`.
pub fn shapes(parallel: u32, context: u32) boot.Shapes {
    return .{ .prompt_chunk = E.max_rows, .max_rows = E.max_rows, .parallel = parallel, .context = context };
}

/// The checkpoint's and layout's fingerprint: SHA-256 over config.json, the version and every rank's range.
pub fn fingerprint(config: []const u8, ranges: []const boot.Range) [32]u8 {
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    h.update(config);
    h.update(version);
    for (ranges) |r| {
        var b: [8]u8 = undefined;
        std.mem.writeInt(u32, b[0..4], r.first, .little);
        std.mem.writeInt(u32, b[4..8], r.end, .little);
        h.update(&b);
    }
    return h.finalResult();
}

/// One stream's state on this stage: its cache (its own layers only) and its state generation.
pub const Slot = struct { cache: E.Cache, id: u32 = 0, used: bool = false, generation: u32 = 0 };

/// A stage's streams: the same code on rank 0 and on every follower, so each applies a plan's keeps alike.
pub const Slots = struct {
    e: *E.Engine,
    all: []Slot,

    pub fn init(gpa: std.mem.Allocator, e: *E.Engine, n: usize, context: usize) !Slots {
        const all = try gpa.alloc(Slot, n);
        var made: usize = 0;
        errdefer {
            for (all[0..made]) |*s| s.cache.deinit(gpa);
            gpa.free(all);
        }
        for (all) |*s| {
            s.* = .{ .cache = try E.Cache.init(gpa, e, context) };
            made += 1;
        }
        return .{ .e = e, .all = all };
    }

    pub fn deinit(s: *Slots, gpa: std.mem.Allocator) void {
        for (s.all) |*x| x.cache.deinit(gpa);
        gpa.free(s.all);
    }

    pub fn find(s: *Slots, id: u32) ?*Slot {
        for (s.all) |*x| if (x.used and x.id == id) return x;
        return null;
    }

    /// The stream's previous window kept to its first `accepted` rows (a keep advances the generation).
    fn settle(s: *Slots, x: *Slot, accepted: u32) !void {
        const last = x.cache.last orelse return if (accepted != 0) error.BadAccepted;
        if (accepted > last.rows) return error.BadAccepted;
        var path: [E.most_window]u32 = undefined;
        for (path[0..accepted], 0..) |*r, i| r.* = @intCast(i);
        try x.cache.keep(s.e, path[0..accepted]);
        x.generation +%= 1;
    }

    /// One stream entry before the round's forward: opened, kept and checked, or closed; its slot when it has rows.
    pub fn apply(s: *Slots, st: plan.Stream) !?*Slot {
        switch (st.op) {
            .open => {
                if (s.find(st.id) != null) return error.StreamOpen;
                const x = for (s.all) |*x| {
                    if (!x.used) break x;
                } else return error.NoFreeSlot;
                try x.cache.reset(s.e);
                x.* = .{ .cache = x.cache, .id = st.id, .used = true };
                try plan.sameGeneration(x.generation, st);
                return x;
            },
            .@"continue" => {
                const x = s.find(st.id) orelse return error.UnknownStream;
                try s.settle(x, st.accepted);
                try plan.sameGeneration(x.generation, st);
                if (x.cache.len != st.first) return error.OtherPosition;
                return if (st.rows > 0) x else null;
            },
            .close, .cancel => {
                const x = s.find(st.id) orelse return error.UnknownStream;
                try plan.sameGeneration(x.generation, st);
                x.used = false;
                return null;
            },
            .@"resume" => return error.ResumeNotInV1, // a prompt cache across stages is step 7
        }
    }
};

/// What rank 0's caller asks of a stream in one round.
pub const Entry = struct { id: u32, op: plan.Op, rows: u32 = 0, accepted: u32 = 0 };

/// A plan's forward parts, in stream order.
const Round = struct {
    parts: [E.max_segs]E.Part = undefined,
    n: usize = 0,
};

/// The ring's hop for one round: rows to the next rank, rows from the one before (on rank 0, after the plan's echo).
const Link = struct {
    ring: *stage.Ring,
    round: u64,
    bound: boot.Boundary,
    plan: plan.Plan,
    /// Rank 0: the plan it sent, which the last stage's echo must equal.
    sent: ?[]const u8 = null,
    echo: []u8 = &.{},

    fn hop(l: *Link) sf.Hop {
        return .{ .ctx = l, .give = give, .take = take };
    }

    fn give(ctx: *anyopaque, rows: usize, bytes: []const u8) anyerror!void {
        const l: *Link = @ptrCast(@alignCast(ctx));
        try l.ring.send(.rows, l.round, @intCast(rows), &.{bytes});
    }

    fn take(ctx: *anyopaque, rows: usize, bytes: []u8) anyerror!void {
        const l: *Link = @ptrCast(@alignCast(ctx));
        if (l.sent) |p| {
            try echoed(l.ring, l.round, p, l.echo);
            l.sent = null;
        }
        const h = try l.ring.recv(.rows, l.round, bytes);
        const r = try plan.checkRows(h, l.plan, l.bound);
        if (r.rows != rows) return error.OtherRowCount;
    }
};

/// The last stage's copy of rank 0's plan, which must be byte for byte the one sent.
fn echoed(ring: *stage.Ring, round: u64, sent: []const u8, buf: []u8) !void {
    const h = try ring.recv(.plan, round, buf);
    if (!std.mem.eql(u8, buf[0..@intCast(h.bytes)], sent)) return error.OtherEcho;
}

/// Rank 0: the plan of each round to the ring, its own streams applied alike, then the forward with the hop.
pub const Lead = struct {
    e: *E.Engine,
    /// null: one engine, no ring (the whole-model reference runs the same rounds).
    ring: ?*stage.Ring,
    slots: Slots,
    limits: plan.Limits,
    round: u64,
    mem: []u8,

    pub fn init(gpa: std.mem.Allocator, e: *E.Engine, ring: ?*stage.Ring, parallel: u32, context: u32, first_round: u64) !Lead {
        if (parallel > E.max_segs) return error.TooManyStreams;
        var slots = try Slots.init(gpa, e, parallel, context);
        errdefer slots.deinit(gpa);
        const mem = try gpa.alloc(u8, 2 * plan.planSize(parallel, 0));
        return .{ .e = e, .ring = ring, .slots = slots, .limits = plan.Limits.of(shapes(parallel, context)), .round = first_round, .mem = mem };
    }

    pub fn deinit(l: *Lead, gpa: std.mem.Allocator) void {
        l.slots.deinit(gpa);
        gpa.free(l.mem);
    }

    pub fn cache(l: *Lead, id: u32) ?*E.Cache {
        return if (l.slots.find(id)) |x| &x.cache else null;
    }

    /// One round over `entries` (ascending ids): every stage applies them, then one forward of `ids` and `keep`.
    pub fn step(l: *Lead, entries: []const Entry, ids: []const u32, keep: E.Keep) !void {
        var streams: [E.max_segs]plan.Stream = undefined;
        if (entries.len > streams.len) return error.TooManyStreams;
        for (entries, streams[0..entries.len]) |en, *st| st.* = l.expect(en);
        const mode: plan.Mode = if (keep == .last) .prompt else .window;
        const half = l.mem.len / 2;
        const p = try plan.encodeMode(mode, streams[0..entries.len], &.{}, l.limits, l.mem[0..half]);
        defer l.round += 1;
        var r: Round = .{};
        for (streams[0..entries.len]) |st| {
            const x = try l.slots.apply(st) orelse continue; // rank 0 checks what it planned, as every stage does
            r.parts[r.n] = .{ .cache = &x.cache, .rows = st.rows };
            r.n += 1;
        }
        const ring = l.ring orelse {
            if (r.n > 0) try l.e.forward(r.parts[0..r.n], ids, keep);
            return;
        };
        try ring.send(.plan, l.round, 0, &.{p});
        if (r.n == 0) return echoed(ring, l.round, p, l.mem[half..]);
        var link: Link = .{ .ring = ring, .round = l.round, .bound = boundary(l.e), .plan = try plan.decode(p, l.limits), .sent = p, .echo = l.mem[half..] };
        l.e.hop = link.hop();
        defer l.e.hop = null;
        try l.e.forward(r.parts[0..r.n], ids, keep);
    }

    /// The plan's entry for `en`: where the stream stands once its previous window keeps `accepted` rows.
    fn expect(l: *Lead, en: Entry) plan.Stream {
        var st: plan.Stream = .{ .id = en.id, .op = en.op, .first = 0, .rows = en.rows, .accepted = en.accepted };
        const x = l.slots.find(en.id) orelse return st;
        st.generation = x.generation;
        if (en.op != .@"continue") return st;
        st.first = @intCast(x.cache.len);
        if (x.cache.last) |w| {
            st.first = @intCast(w.start + en.accepted);
            st.generation +%= 1;
        }
        return st;
    }

    /// A keep-alive: a plan with no streams, which every stage forwards and none acts on.
    pub fn beat(l: *Lead) !void {
        try l.step(&.{}, &.{}, .logits);
    }
};

/// A follower's rounds until rank 0 stops the ring: each plan passed on, its streams applied, its rows run.
pub fn serve(gpa: std.mem.Allocator, e: *E.Engine, ring: *stage.Ring, parallel: u32, context: u32, first_round: u64) !void {
    var slots = try Slots.init(gpa, e, parallel, context);
    defer slots.deinit(gpa);
    const limits = plan.Limits.of(shapes(parallel, context));
    const buf = try gpa.alloc(u8, plan.planSize(parallel, 0));
    defer gpa.free(buf);
    const zeros = try gpa.alloc(u32, E.max_rows); // later stages read no token ids
    defer gpa.free(zeros);
    @memset(zeros, 0);
    var round = first_round;
    while (true) : (round += 1) {
        const h = ring.recv(.plan, round, buf) catch |err| return if (err == error.Stopped) {} else err;
        const bytes = buf[0..@intCast(h.bytes)];
        const p = try plan.decode(bytes, limits);
        try ring.send(.plan, round, 0, &.{bytes}); // on round, so the last stage echoes it to rank 0
        if (p.cuts != 0) return error.CutsNotInV1;
        var r: Round = .{};
        for (0..p.streams) |i| {
            const st = p.stream(i);
            const x = try slots.apply(st) orelse continue;
            r.parts[r.n] = .{ .cache = &x.cache, .rows = st.rows };
            r.n += 1;
        }
        if (r.n == 0) continue;
        if (p.mode == .prompt and r.n != 1) return error.PromptParts;
        var link: Link = .{ .ring = ring, .round = round, .bound = boundary(e), .plan = p };
        e.hop = link.hop();
        defer e.hop = null;
        try e.forward(r.parts[0..r.n], zeros[0..p.rows], if (p.mode == .prompt) .last else .logits);
    }
}

/// This stage's boot description.
pub fn describe(e: *const E.Engine, rank: u32, world: u32, range: boot.Range, parallel: u32, context: u32) !boot.Description {
    return .{
        .rank = rank,
        .world = world,
        .version = try boot.Text.init(version),
        .backend = .cuda,
        .kv_dtype = if (e.kv8) .f8e4m3 else .bf16,
        .range = range,
        .boundary = boundary(e),
        .shapes = shapes(parallel, context),
        .taps = .{},
        .kv_rows = context,
        .states = parallel,
    };
}

/// Boot on the ring, rounds 1 and 2: descriptions, then ready reports, each stage's appended; rank 0 checks all.
pub fn bootRing(gpa: std.mem.Allocator, ring: *stage.Ring, own: *const boot.Description, model: *const boot.Model, ranges: []const boot.Range) !u64 {
    try boot.checkLayout(model.layers, ranges);
    const world = ring.world;
    const most = world * (boot.describe_fixed + 4 * boot.most_taps + boot.most_version);
    const buf = try gpa.alloc(u8, most);
    defer gpa.free(buf);
    var mine: [boot.describe_fixed + 4 * boot.most_taps + boot.most_version]u8 = undefined;
    const d = try boot.encodeDescription(own, &mine);
    var ready: [boot.ready_size]u8 = undefined;
    const rd = try boot.encodeReady(.{ .rank = own.rank, .loaded = true, .warm = true }, &ready);
    for ([_][]const u8{ d, rd }, 1..) |msg, round| {
        if (ring.rank == 0) {
            try ring.send(.plan, round, 0, &.{msg});
            const h = try ring.recv(.plan, round, buf);
            try checkAll(buf[0..@intCast(h.bytes)], world, model, ranges, round == 1);
        } else {
            const h = try ring.recv(.plan, round, buf);
            try ring.send(.plan, round, 0, &.{ buf[0..@intCast(h.bytes)], msg });
        }
    }
    return 3;
}

/// Every stage's description (or ready report) in rank order, each against rank 0's plan for it.
fn checkAll(all: []const u8, world: u32, model: *const boot.Model, ranges: []const boot.Range, describes: bool) !void {
    var at: usize = 0;
    for (0..world) |rank| {
        if (describes) {
            if (all.len < at + boot.describe_fixed) return error.BadLength;
            const n = boot.describe_fixed + 4 * @as(usize, all[at + 20]) + all[at + 21];
            if (all.len < at + n) return error.BadLength;
            const desc = try boot.decodeDescription(all[at..][0..n]);
            try boot.check(.{ .model = model, .rank = @intCast(rank), .world = world, .backend = .cuda, .range = ranges[rank] }, &desc);
            at += n;
        } else {
            if (all.len < at + boot.ready_size) return error.BadLength;
            const r = try boot.decodeReady(all[at..][0..boot.ready_size]);
            if (r.rank != rank or !r.ready()) return error.NotReady;
            at += boot.ready_size;
        }
    }
    if (at != all.len) return error.BadLength;
}
