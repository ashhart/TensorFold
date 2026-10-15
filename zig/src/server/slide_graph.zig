//! The Sliding Weights fact graph: what was taught, how far each lesson is, where its neurons went; never the weights.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json.zig");
const slide_model = @import("slide_model.zig");
const Allocator = std.mem.Allocator;
const Io = std.Io;

pub const State = enum { queued, learning, learned, missed, replaced }; // replaced: a newer fact took its place

/// Where a learning fact stands: its lesson written by the model, then the engine's stages in turn.
pub const Phase = enum { writing, reading, fitting, values, checking };

pub const Fact = struct {
    id: u32,
    text: []const u8,
    source: []const u8,
    at: i64,
    state: State,
    neurons: []const api.Neuron = &.{}, // where its neurons sit in the pocket, once placed
    questions: []const []const u8 = &.{}, // the wordings its checks ask
    message: []const u8 = "", // why it stopped, or what its lesson last did
    round: u32 = 0, // its lesson's round now, or the rounds it took
    checks: u32 = 0, // asked checks done, this session
    recalled: ?bool = null, // and the last one's verdict
    phase: ?Phase = null, // set while it learns
    since: i64 = 0, // when the phase began (ms)
    step: u32 = 0, // the phase's unit, of `of` (0: unknown)
    of: u32 = 0,
    loss: ?f32 = null, // its outputs' loss at the last step
};

/// The last save of the pocket into the shards: when, how long, the files it changed and their shared experts' bytes.
const Save = struct {
    state: enum { none, saving, saved, failed } = .none,
    at: i64 = 0,
    seconds: f64 = 0,
    bytes: u64 = 0,
    files: []const []const u8 = &.{},
    message: []const u8 = "",
};

/// Words of five letters or more that say nothing about a fact, so they never link two facts.
const common = [_][]const u8{ "about", "after", "again", "being", "could", "every", "other", "should", "still", "their", "there", "these", "those", "under", "until", "where", "which", "while", "would" };

const trace_cap = 600; // loss steps kept for the learning fact, every round together

/// Facts in teaching order, saved to `path` after every change (null: memory only); ids are never reused.
pub const Graph = struct {
    gpa: Allocator,
    io: Io,
    path: ?[]const u8,
    mutex: Io.Mutex = .init,
    arena: std.heap.ArenaAllocator,
    facts: std.ArrayList(Fact) = .empty,
    next: u32 = 1,
    trace: std.ArrayList(f32) = .empty, // the learning fact's loss at each step of its outputs
    cuts: std.ArrayList(u32) = .empty, // where each of its rounds begins in `trace`
    traced: u32 = 0, // whose trace it is
    model: ?slide_model.Model = null, // the served folder as last scanned
    model_arena: std.heap.ArenaAllocator,
    saved: Save = .{},

    /// A graph that starts from `path`'s file when it reads, else empty.
    pub fn init(gpa: Allocator, io: Io, path: ?[]const u8) Graph {
        var g: Graph = .{ .gpa = gpa, .io = io, .path = path, .arena = .init(gpa), .model_arena = .init(gpa) };
        g.load() catch {};
        return g;
    }

    pub fn deinit(g: *Graph) void {
        g.facts.deinit(g.gpa);
        g.trace.deinit(g.gpa);
        g.cuts.deinit(g.gpa);
        g.arena.deinit();
        g.model_arena.deinit();
    }

    /// A newly taught fact, queued; returns its id (0 when memory ran out).
    pub fn add(g: *Graph, text: []const u8, source: []const u8, at: i64) u32 {
        g.mutex.lockUncancelable(g.io);
        defer g.mutex.unlock(g.io);
        const a = g.arena.allocator();
        const fact: Fact = .{ .id = g.next, .text = a.dupe(u8, text) catch return 0, .source = a.dupe(u8, source) catch return 0, .at = at, .state = .queued };
        g.facts.append(g.gpa, fact) catch return 0;
        g.next += 1;
        g.save();
        return fact.id;
    }

    /// A fact's state; leaving `learning` ends its phase, and a missed fact keeps no neurons.
    pub fn mark(g: *Graph, id: u32, state: State) void {
        g.mutex.lockUncancelable(g.io);
        defer g.mutex.unlock(g.io);
        const f = g.find(id) orelse return;
        f.state = state;
        if (state != .learning) f.phase = null;
        if (state == .missed or state == .replaced) f.neurons = &.{};
        g.save();
    }

    /// Learned chat facts but `id`, the latest first, at most `max`, copied into `a`: what a new chat fact may replace.
    pub fn learnedChat(g: *Graph, a: Allocator, id: u32, max: usize) ![]const Fact {
        g.mutex.lockUncancelable(g.io);
        defer g.mutex.unlock(g.io);
        var out: std.ArrayList(Fact) = .empty;
        var i = g.facts.items.len;
        while (i > 0 and out.items.len < max) : (i -= 1) {
            const f = g.facts.items[i - 1];
            if (f.id == id or f.state != .learned or !std.mem.eql(u8, f.source, "chat") or f.neurons.len == 0) continue;
            const qs = try a.alloc([]const u8, f.questions.len);
            for (f.questions, qs) |q, *c| c.* = try a.dupe(u8, q);
            try out.append(a, .{ .id = f.id, .text = try a.dupe(u8, f.text), .source = "chat", .at = f.at, .state = f.state, .neurons = try a.dupe(api.Neuron, f.neurons), .questions = qs });
        }
        return out.items;
    }

    /// A learning fact's next phase; writing starts its lesson afresh.
    pub fn phase(g: *Graph, id: u32, p: Phase) void {
        g.mutex.lockUncancelable(g.io);
        defer g.mutex.unlock(g.io);
        const f = g.find(id) orelse return;
        g.enter(f, p);
        if (p != .writing) return;
        const old = f.*;
        f.* = .{ .id = old.id, .text = old.text, .source = old.source, .at = old.at, .state = old.state, .phase = .writing, .since = old.since };
        g.trace.clearRetainingCapacity();
        g.cuts.clearRetainingCapacity();
        g.traced = id;
    }

    fn enter(g: *Graph, f: *Fact, p: Phase) void {
        if (f.phase != p) f.since = now(g.io);
        f.phase = p;
        f.step = 0;
        f.of = 0;
    }

    /// The engine's word on a learning fact's lesson: its stage and unit, its outputs' loss, its neurons once placed.
    pub fn progress(g: *Graph, id: u32, p: api.LearnProgress) void {
        g.mutex.lockUncancelable(g.io);
        defer g.mutex.unlock(g.io);
        const f = g.find(id) orelse return;
        g.enter(f, switch (p.phase) {
            .reading => .reading,
            .fitting => .fitting,
            .values => .values,
            .checking => .checking,
        });
        f.step = p.at;
        f.of = p.of;
        if (p.neurons.len > 0) g.place(f, p.neurons);
        if (p.phase != .values or p.at == 0 or !std.math.isFinite(p.loss)) return;
        f.loss = p.loss;
        if (g.traced != id or g.trace.items.len >= trace_cap) return;
        // the steps count from one again in each round
        const begun = if (g.cuts.items.len > 0) g.cuts.items[g.cuts.items.len - 1] else 0;
        if (g.cuts.items.len == 0 or p.at <= g.trace.items.len - begun) g.cuts.append(g.gpa, @intCast(g.trace.items.len)) catch return;
        g.trace.append(g.gpa, p.loss) catch {};
    }

    /// Where a fact's neurons sit now (none: taken out of the pocket).
    pub fn placed(g: *Graph, id: u32, neurons: []const api.Neuron) void {
        g.mutex.lockUncancelable(g.io);
        defer g.mutex.unlock(g.io);
        const f = g.find(id) orelse return;
        g.place(f, neurons);
    }

    fn place(g: *Graph, f: *Fact, neurons: []const api.Neuron) void {
        if (f.neurons.len == neurons.len) for (f.neurons, neurons) |x, y| {
            if (x.layer != y.layer or x.neuron != y.neuron or x.key != y.key) break;
        } else return;
        f.neurons = g.arena.allocator().dupe(api.Neuron, neurons) catch return;
    }

    /// A learning fact's round, from one.
    pub fn round(g: *Graph, id: u32, n: u32) void {
        g.mutex.lockUncancelable(g.io);
        defer g.mutex.unlock(g.io);
        const f = g.find(id) orelse return;
        f.round = n;
    }

    /// An asked check's verdict on a learning fact.
    pub fn checked(g: *Graph, id: u32, recalled: bool) void {
        g.mutex.lockUncancelable(g.io);
        defer g.mutex.unlock(g.io);
        const f = g.find(id) orelse return;
        f.checks += 1;
        f.recalled = recalled;
    }

    /// What a fact's lesson last did, or why it stopped.
    pub fn note(g: *Graph, id: u32, message: []const u8) void {
        g.mutex.lockUncancelable(g.io);
        defer g.mutex.unlock(g.io);
        const f = g.find(id) orelse return;
        f.message = g.arena.allocator().dupe(u8, message) catch return;
        g.save();
    }

    /// The wordings a fact's checks ask.
    pub fn asked(g: *Graph, id: u32, questions: []const []const u8) void {
        g.mutex.lockUncancelable(g.io);
        defer g.mutex.unlock(g.io);
        const f = g.find(id) orelse return;
        const a = g.arena.allocator();
        const out = a.alloc([]const u8, questions.len) catch return;
        for (questions, out) |q, *o| o.* = a.dupe(u8, q) catch return;
        f.questions = out;
        g.save();
    }

    /// A save of the pocket into `dir`'s shards begins; the shards as they are now are what it is measured against.
    pub fn saving(g: *Graph, dir: ?[]const u8) void {
        if (dir) |d| g.scanned(d);
        g.mutex.lockUncancelable(g.io);
        defer g.mutex.unlock(g.io);
        g.saved = .{ .state = .saving, .at = now(g.io) };
    }

    /// The save ended after `seconds` (`failed`: why it did not): the shards scanned again, the ones it changed kept.
    pub fn saveEnded(g: *Graph, dir: ?[]const u8, seconds: f64, failed: ?[]const u8) void {
        var s: Save = .{ .state = if (failed == null) .saved else .failed, .seconds = seconds, .message = failed orelse "" };
        g.mutex.lockUncancelable(g.io);
        s.at = g.saved.at;
        g.mutex.unlock(g.io);
        if (dir) |d| return g.rescan(d, s);
        g.mutex.lockUncancelable(g.io);
        defer g.mutex.unlock(g.io);
        s.message = g.model_arena.allocator().dupe(u8, s.message) catch "";
        g.saved = s;
    }

    /// The folder scanned once, when it has not been.
    fn scanned(g: *Graph, dir: []const u8) void {
        g.mutex.lockUncancelable(g.io);
        const have = g.model != null;
        g.mutex.unlock(g.io);
        if (!have) g.rescan(dir, null);
    }

    /// The folder scanned outside the lock, then swapped in; with a save's end, what it changed against the last scan.
    fn rescan(g: *Graph, dir: []const u8, ended: ?Save) void {
        var fresh = std.heap.ArenaAllocator.init(g.gpa);
        const a = fresh.allocator();
        const m = slide_model.scan(a, g.io, dir) catch {
            fresh.deinit();
            return;
        };
        g.mutex.lockUncancelable(g.io);
        defer g.mutex.unlock(g.io);
        var s = ended orelse g.saved;
        if (ended != null) if (g.model) |old| if (slide_model.written(a, old.shards, m.shards)) |w| {
            s.files = w.files;
            s.bytes = w.bytes;
        } else |_| {};
        // what the old arena held, copied into the new one
        if (ended == null) s.files = dupeAll(a, s.files) catch &.{};
        s.message = a.dupe(u8, s.message) catch "";
        g.model_arena.deinit();
        g.model_arena = fresh;
        g.model = m;
        g.saved = s;
    }

    /// Forget every fact and remove the file; the weights keep what they learned.
    pub fn clear(g: *Graph) void {
        g.mutex.lockUncancelable(g.io);
        defer g.mutex.unlock(g.io);
        g.facts.clearRetainingCapacity();
        if (g.path) |p| Io.Dir.cwd().deleteFile(g.io, p) catch {};
    }

    /// {"next", "facts": [...], "model": {...}}: each fact's links join facts sharing a key word; `dir` is the folder.
    pub fn render(g: *Graph, a: Allocator, dir: ?[]const u8) ![]u8 {
        if (dir) |d| g.scanned(d);
        g.mutex.lockUncancelable(g.io);
        defer g.mutex.unlock(g.io);
        return g.document(a, true);
    }

    fn find(g: *Graph, id: u32) ?*Fact {
        for (g.facts.items) |*f| if (f.id == id) return f;
        return null;
    }

    /// The graph as JSON; `live` adds what holds only while this server runs: phases, the loss trace, the model.
    fn document(g: *Graph, a: Allocator, live: bool) ![]u8 {
        const facts = try a.alloc(json.Value, g.facts.items.len);
        for (g.facts.items, facts) |*f, *out| {
            var links: std.ArrayList(json.Value) = .empty;
            for (g.facts.items) |other| if (other.id != f.id and shareWord(f.text, other.text)) try links.append(a, try json.intValue(a, other.id));
            const o = try json.newObject(a);
            try o.put(a, "id", try json.intValue(a, f.id));
            try o.put(a, "text", .{ .string = f.text });
            try o.put(a, "source", .{ .string = f.source });
            try o.put(a, "at", try json.intValue(a, f.at));
            try o.put(a, "state", .{ .string = @tagName(f.state) });
            try o.put(a, "links", .{ .array = links.items });
            if (f.round > 0) try o.put(a, "round", try json.intValue(a, f.round));
            if (f.message.len > 0) try o.put(a, "message", .{ .string = f.message });
            if (f.questions.len > 0) try o.put(a, "questions", try strings(a, f.questions));
            if (f.neurons.len > 0) try o.put(a, "neurons", try neuronList(a, f.neurons));
            if (live) try g.liveFields(a, o, f);
            out.* = .{ .object = o };
        }
        const root = try json.newObject(a);
        try root.put(a, "next", try json.intValue(a, g.next));
        try root.put(a, "facts", .{ .array = facts });
        if (live) if (g.model) |m| {
            const v = try slide_model.value(a, m);
            if (g.saved.state != .none) try v.object.put(a, "save", try saveValue(a, g.saved));
            try root.put(a, "model", v);
        };
        return json.stringify(a, .{ .object = root }, .{ .compact = true });
    }

    fn liveFields(g: *Graph, a: Allocator, o: *json.Object, f: *const Fact) !void {
        if (f.checks > 0) try o.put(a, "checks", try json.intValue(a, f.checks));
        if (f.recalled) |r| try o.put(a, "recalled", .{ .bool = r });
        if (f.loss) |l| try o.put(a, "loss", round4(l));
        const p = f.phase orelse return;
        try o.put(a, "phase", .{ .string = @tagName(p) });
        try o.put(a, "since", try json.intValue(a, f.since));
        try o.put(a, "step", try json.intValue(a, f.step));
        try o.put(a, "of", try json.intValue(a, f.of));
        if (g.traced != f.id or g.trace.items.len == 0) return;
        const rounds = try a.alloc(json.Value, g.cuts.items.len);
        for (g.cuts.items, rounds, 0..) |begin, *r, i| {
            const end = if (i + 1 < g.cuts.items.len) g.cuts.items[i + 1] else g.trace.items.len;
            const xs = try a.alloc(json.Value, end - begin);
            for (g.trace.items[begin..end], xs) |x, *v| v.* = round4(x);
            r.* = .{ .array = xs };
        }
        try o.put(a, "trace", .{ .array = rounds });
    }

    /// Write the file through a renamed temporary; a failed write keeps the old file (called under the lock).
    fn save(g: *Graph) void {
        const path = g.path orelse return;
        var scratch = std.heap.ArenaAllocator.init(g.gpa);
        defer scratch.deinit();
        const a = scratch.allocator();
        const body = g.document(a, false) catch return;
        const tmp = std.fmt.allocPrint(a, "{s}.tmp", .{path}) catch return;
        if (std.fs.path.dirname(path)) |dir| Io.Dir.cwd().createDirPath(g.io, dir) catch return;
        Io.Dir.cwd().writeFile(g.io, .{ .sub_path = tmp, .data = body }) catch return;
        Io.Dir.cwd().rename(tmp, Io.Dir.cwd(), path, g.io) catch return;
    }

    fn load(g: *Graph) !void {
        const path = g.path orelse return;
        const a = g.arena.allocator();
        const bytes = try Io.Dir.cwd().readFileAlloc(g.io, path, a, .limited(1 << 26));
        const root = switch (try json.parse(a, bytes)) {
            .ok => |v| v,
            .err => return error.BadGraph,
        };
        for ((root.field("facts") orelse return error.BadGraph).items() orelse return error.BadGraph) |f| {
            const state = std.meta.stringToEnum(State, f.strField("state") orelse "") orelse .queued;
            var fact: Fact = .{ .id = try int(u32, f.field("id")), .text = f.strField("text") orelse "", .source = f.strField("source") orelse "", .at = try int(i64, f.field("at")), .state = state };
            fact.message = f.strField("message") orelse "";
            fact.round = int(u32, f.field("round")) catch 0;
            if (f.field("questions")) |qs| if (qs.items()) |xs| {
                const out = try a.alloc([]const u8, xs.len);
                for (xs, out) |x, *o| o.* = x.str() orelse "";
                fact.questions = out;
            };
            if (f.field("neurons")) |ns| if (ns.items()) |xs| {
                var out: std.ArrayList(api.Neuron) = .empty;
                for (xs) |x| try out.append(a, .{ .layer = int(u16, x.field("layer")) catch continue, .neuron = int(u16, x.field("neuron")) catch continue, .key = std.meta.stringToEnum(api.NeuronKey, x.strField("key") orelse "") orelse continue });
                fact.neurons = out.items;
            };
            try g.facts.append(g.gpa, fact);
        }
        g.next = try int(u32, root.field("next"));
    }
};

fn now(io: Io) i64 {
    return @intCast(@divTrunc(std.Io.Clock.real.now(io).toNanoseconds(), std.time.ns_per_ms));
}

fn int(comptime T: type, v: ?json.Value) !T {
    const x = v orelse return error.BadGraph;
    if (x != .int) return error.BadGraph;
    return std.fmt.parseInt(T, x.int, 10);
}

/// A loss as JSON, to four places (never NaN or infinite, which JSON cannot carry).
fn round4(x: f32) json.Value {
    const v: f64 = if (std.math.isFinite(x)) x else 0;
    return .{ .float = @round(v * 1e4) / 1e4 };
}

fn strings(a: Allocator, xs: []const []const u8) !json.Value {
    const out = try a.alloc(json.Value, xs.len);
    for (xs, out) |x, *v| v.* = .{ .string = x };
    return .{ .array = out };
}

fn dupeAll(a: Allocator, xs: []const []const u8) ![]const []const u8 {
    const out = try a.alloc([]const u8, xs.len);
    for (xs, out) |x, *o| o.* = try a.dupe(u8, x);
    return out;
}

/// [{"layer", "neuron", "key"}].
fn neuronList(a: Allocator, ns: []const api.Neuron) !json.Value {
    const out = try a.alloc(json.Value, ns.len);
    for (ns, out) |n, *v| {
        const o = try json.newObject(a);
        try o.put(a, "layer", try json.intValue(a, n.layer));
        try o.put(a, "neuron", try json.intValue(a, n.neuron));
        try o.put(a, "key", .{ .string = @tagName(n.key) });
        v.* = .{ .object = o };
    }
    return .{ .array = out };
}

/// {"state", "at", "seconds", "bytes", "files", "message"}.
fn saveValue(a: Allocator, s: Save) !json.Value {
    const o = try json.newObject(a);
    try o.put(a, "state", .{ .string = @tagName(s.state) });
    try o.put(a, "at", try json.intValue(a, s.at));
    if (s.state == .saving) return .{ .object = o };
    try o.put(a, "seconds", .{ .float = @round(s.seconds * 10) / 10 });
    try o.put(a, "bytes", try json.intValue(a, s.bytes));
    try o.put(a, "files", try strings(a, s.files));
    if (s.message.len > 0) try o.put(a, "message", .{ .string = s.message });
    return .{ .object = o };
}

/// Whether two facts share a word of five or more letters that is not on the common list, ignoring case.
fn shareWord(x: []const u8, y: []const u8) bool {
    var it = std.mem.tokenizeAny(u8, x, " \t\n.,;:!?()[]\"'");
    while (it.next()) |w| {
        if (w.len < 5 or isCommon(w)) continue;
        var other = std.mem.tokenizeAny(u8, y, " \t\n.,;:!?()[]\"'");
        while (other.next()) |v| if (std.ascii.eqlIgnoreCase(w, v)) return true;
    }
    return false;
}

fn isCommon(w: []const u8) bool {
    for (common) |c| if (std.ascii.eqlIgnoreCase(w, c)) return true;
    return false;
}
