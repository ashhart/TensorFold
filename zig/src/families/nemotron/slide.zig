//! Nemotron's Sliding Weights learner: a lesson's rows from the model's forward, GPU steps, then its shards.
const std = @import("std");
const mtl = @import("metal");
const shard_edit = @import("../../core/shard_edit.zig");
const ckpt = @import("../../core/checkpoint_metal.zig");
const affine4 = @import("../../core/affine4.zig");
const st = @import("state.zig");
const fwd = @import("forward.zig");
const pl = @import("prefill_launch.zig");
const learned = @import("learned.zig");
const sg = @import("slide_gpu.zig");
const backend = @import("backend.zig");

const Metal = backend.Metal;
const At = pl.At;
const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;

/// Token ids whose answer starts at `start`: the rows from start - 1 on predict it.
pub const Example = struct { ids: []const u32, start: u32 };

/// One fact's examples; or none, and `save` writes what the weights learned into the shards, `undo` takes it back.
pub const Lesson = struct {
    train: []const Example = &.{},
    held: []const Example = &.{},
    near: []const Example = &.{},
    keep: []const Example = &.{},
    save: bool = false,
    undo: bool = false,
    steps: u32 = max_steps,
    more: bool = false, // more steps on the last lesson's rows
};

pub const Report = union(enum) {
    learned: struct { recalled: bool, steps: u32, loss: f32 },
    saved: u32,
    failed: []const u8,
};

pub const Step = struct { done: bool, changed: bool = false, report: ?Report = null };

const max_steps = 60;
const check_every = 5;
const longest = 1024; // an example's tokens at most
const keep_rows = 640;
const lesson_rows = 512;
const ring_rows = 2048;
const replay_rows = 128; // earlier lessons' answers a lesson keeps steady
const batch_rows = 1280;
const held_rows = 128;

/// Captured rows: keep prompts' at [0, keep_rows), the lesson's next, then a ring of earlier lessons' answers.
const Store = struct {
    h: mtl.Buffer, // bf16 [rows, D]: final residuals as captured
    hd: mtl.Buffer, // bf16 [lesson_rows, D]: keys times the change the forward applied
    base: mtl.Buffer, // f32 [rows, D]: final residuals without the learned change
    keys: mtl.Buffer, // bf16 [rows, W]
    targets: []u32,

    const rows = keep_rows + lesson_rows + ring_rows;
    const lesson0 = keep_rows;
    const ring0 = keep_rows + lesson_rows;

    fn init(gpa: std.mem.Allocator, device: mtl.Device, d: usize, w: usize) !Store {
        const h = try device.buffer(rows * d * 2, opts);
        errdefer h.deinit();
        const hd = try device.buffer(@max(keep_rows, lesson_rows) * d * 2, opts);
        errdefer hd.deinit();
        const base = try device.buffer(rows * d * 4, opts);
        errdefer base.deinit();
        const keys = try device.buffer(rows * w * 2, opts);
        errdefer keys.deinit();
        return .{ .h = h, .hd = hd, .base = base, .keys = keys, .targets = try gpa.alloc(u32, rows) };
    }

    fn deinit(s: *Store, gpa: std.mem.Allocator) void {
        inline for (.{ "h", "hd", "base", "keys" }) |f| @field(s, f).deinit();
        gpa.free(s.targets);
    }
};

const Phase = enum { idle, capture, steps, save, undo };

/// Where each part of a lesson's rows went: train, held and near in turn from Store.lesson0.
const Parts = struct { train: usize = 0, held: usize = 0, near: usize = 0 };

pub const Learner = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    b: *Metal,
    dir: []const u8,
    gpu: ?sg.Gpu = null,
    store: ?Store = null,
    train: ?sg.Batch = null,
    held: ?sg.Batch = null,
    cache: ?st.Cache = null,
    lesson: Lesson = .{},
    phase: Phase = .idle,
    next: usize = 0, // the lesson's next example to capture, over keep, train, held and near in turn
    parts: Parts = .{},
    keep_len: usize = 0, // keep rows captured (once, by the first lesson that brings keep examples)
    keep_new: bool = false, // this lesson captures the keep rows
    ring_at: usize = 0,
    ring_len: usize = 0,
    taken: u32 = 0,
    budget: u32 = max_steps, // steps this lesson may take
    ready: bool = false, // the last lesson's rows are in the batches, for `more`
    first: bool = false, // this lesson's install is its first, whose answers join the ring
    added: bool = false, // the last install added answers to the ring
    loss: f32 = 0,
    dirty: bool = false, // learned since the last save
    before: ?mtl.Buffer = null, // bf16 [D, W]: the change the last lesson replaced, for undo
    had: bool = false, // the forward applied a change before the last lesson
    ring_before: [2]usize = .{ 0, 0 }, // ring_at and ring_len before the last lesson's answers joined
    can_undo: bool = false,
    made: ?mtl.Buffer = null, // the change's buffer this learner made (the weights own it), reused after an undo

    /// A learner for the model `b` serves from `dir`; its GPU buffers come with the first lesson.
    pub fn init(gpa: std.mem.Allocator, io: std.Io, b: *Metal, dir: []const u8) !Learner {
        return .{ .gpa = gpa, .io = io, .b = b, .dir = try gpa.dupe(u8, dir) };
    }

    pub fn deinit(l: *Learner) void {
        if (l.gpu) |*g| g.deinit();
        if (l.store) |*s| s.deinit(l.gpa);
        if (l.train) |*t| t.deinit();
        if (l.held) |*t| t.deinit();
        if (l.cache) |*c| c.deinit(&l.b.pool);
        if (l.before) |b| b.deinit();
        l.gpa.free(l.dir);
    }

    /// Start a lesson; its examples must stay valid until its last step.
    pub fn begin(l: *Learner, lesson: Lesson) !void {
        std.debug.assert(l.phase == .idle);
        l.lesson = lesson;
        l.budget = @max(@min(lesson.steps, max_steps), 1);
        if (lesson.undo) {
            l.phase = .undo;
        } else if (lesson.more) {
            if (!l.ready) return error.NothingToContinue;
            l.taken = 0;
            l.first = false;
            l.phase = .steps;
        } else if (lesson.train.len == 0) {
            l.phase = if (lesson.save) .save else .idle;
        } else {
            try l.ensure();
            l.next = 0;
            l.parts = .{};
            l.keep_new = l.keep_len == 0 and lesson.keep.len > 0;
            l.ready = false;
            l.first = true;
            l.phase = .capture;
        }
    }

    pub fn abort(l: *Learner) void {
        l.phase = .idle;
        if (l.gpu) |*g| g.resetTo(l.installed());
    }

    /// One bounded unit: an example captured, one step taken, or the save.
    pub fn step(l: *Learner) Step {
        return l.advance() catch |e| {
            if (l.gpu) |*g| g.resetTo(l.installed());
            l.phase = .idle;
            return .{ .done = true, .report = .{ .failed = @errorName(e) } };
        };
    }

    fn advance(l: *Learner) !Step {
        switch (l.phase) {
            .idle => return .{ .done = true },
            .capture => {
                if (try l.captureNext()) return .{ .done = false };
                try l.settle();
                l.fill();
                l.ready = true;
                l.taken = 0;
                l.phase = .steps;
                return .{ .done = false };
            },
            .steps => return l.stepOnce(),
            .save => {
                l.phase = .idle;
                if (!l.dirty) return .{ .done = true, .report = .{ .saved = 0 } };
                try l.bake();
                l.dirty = false;
                return .{ .done = true, .report = .{ .saved = 1 } };
            },
            .undo => {
                l.phase = .idle;
                return .{ .done = true, .changed = l.takeBack() };
            },
        }
    }

    /// The model's buffers for learning, made once: the step's, the rows', a capture cache, the head transposed.
    fn ensure(l: *Learner) !void {
        if (l.gpu != null) return;
        const m = l.b.m;
        const c = m.config;
        if (c.kinds[c.layers - 1] != .moe) return error.LastLayerNotMoe;
        var head: [3]At = undefined;
        inline for (.{ "weight", "scales", "biases" }, 0..) |part, i| {
            const t = try m.checkpoint.get("lm_head." ++ part);
            head[i] = .{ .b = t.buffer, .off = t.offset };
        }
        const norm: At = .{ .b = m.weights.norm_f.buffer, .off = m.weights.norm_f.offset };
        var g = try sg.Gpu.init(m.device, c, &m.kernels, &m.prefill, head, norm, l.installed());
        errdefer g.deinit();
        var store = try Store.init(l.gpa, m.device, c.hidden, c.shared_width);
        errdefer store.deinit(l.gpa);
        var train = try sg.Batch.init(m.device, c, batch_rows);
        errdefer train.deinit();
        var held = try sg.Batch.init(m.device, c, held_rows);
        errdefer held.deinit();
        var cache = try st.Cache.init(m.device, c, longest, false);
        errdefer cache.deinit(&l.b.pool);
        const Head = struct {
            g: *const sg.Gpu,
            pub fn encode(j: @This(), _: *Metal, e: *fwd.Enc) !void {
                j.g.transposeHead(e);
            }
        };
        try l.b.drain();
        try l.b.submit(.prefill, l.b.next, Head{ .g = &g });
        try l.b.drain();
        l.gpu = g;
        l.store = store;
        l.train = train;
        l.held = held;
        l.cache = cache;
    }

    /// The change the forward applies now (null: none).
    fn installed(l: *const Learner) ?mtl.Buffer {
        const c = l.b.m.config;
        return l.b.m.weights.layers[c.layers - 1].moe.slide;
    }

    /// Capture the lesson's next example; false once every example is in.
    fn captureNext(l: *Learner) !bool {
        const lesson = l.lesson;
        const keeps = if (l.keep_new) lesson.keep.len else 0;
        var i = l.next;
        if (i >= keeps + lesson.train.len + lesson.held.len + lesson.near.len) return false;
        l.next += 1;
        if (i < keeps) {
            l.keep_len += try l.capture(lesson.keep[i], l.keep_len, keep_rows - l.keep_len);
            return true;
        }
        i -= keeps;
        const at = Store.lesson0 + l.parts.train + l.parts.held + l.parts.near;
        const room = Store.ring0 - at;
        if (i < lesson.train.len) {
            l.parts.train += try l.capture(lesson.train[i], at, room);
        } else if (i < lesson.train.len + lesson.held.len) {
            l.parts.held += try l.capture(lesson.held[i - lesson.train.len], at, room);
        } else l.parts.near += try l.capture(lesson.near[i - lesson.train.len - lesson.held.len], at, room);
        return true;
    }

    /// One example through the forward in windows of the decode kernels' rows, its answer rows stored from `at`.
    fn capture(l: *Learner, ex: Example, at: usize, room: usize) !usize {
        const b = l.b;
        const n = ex.ids.len;
        if (ex.start < 1 or ex.start >= n or n > longest) return error.BadExample;
        if (n - ex.start > room) return error.LessonTooLong;
        const d = b.m.config.hidden;
        const w = b.m.config.shared_width;
        const s = &l.store.?;
        const c = &l.cache.?;
        b.pool.give(c.slot);
        c.slot = 0;
        c.len = 0;
        @memcpy(b.prompt.slice(u32, n - 1), ex.ids[0 .. n - 1]);
        const Window = struct {
            c: *st.Cache,
            at: usize,
            rows: usize,
            pub fn encode(j: @This(), m: *Metal, e: *fwd.Enc) !void {
                const fresh = try m.pool.take();
                const segs = [_]fwd.Seg{.{ .rows = j.rows, .cache = j.c, .store = .full, .slot = @intCast(fresh) }};
                m.forward().body(e, &segs, m.prompt, j.at * 4);
                j.c.advance(&m.pool, j.rows, fresh);
            }
        };
        var row = at;
        var p: usize = 0;
        while (p < n - 1) {
            const rows = @min(backend.fused_rows, n - 1 - p);
            try b.submit(.prefill, b.next, Window{ .c = c, .at = p, .rows = rows });
            try b.drain();
            const h = b.scratch.h.slice(u16, rows * d);
            const k = b.scratch.sh_act.slice(u16, rows * w);
            for (@min(@max(p, ex.start - 1), p + rows)..p + rows) |q| {
                @memcpy(s.h.slice(u16, Store.rows * d)[row * d ..][0..d], h[(q - p) * d ..][0..d]);
                @memcpy(s.keys.slice(u16, Store.rows * w)[row * w ..][0..w], k[(q - p) * w ..][0..w]);
                s.targets[row] = ex.ids[q + 1];
                row += 1;
            }
            p += rows;
        }
        return row - at;
    }

    /// The lesson's rows (and keep rows captured with it) without the change the forward applied when they ran.
    fn settle(l: *Learner) !void {
        const c = l.b.m.config;
        const d = c.hidden;
        const w = c.shared_width;
        if (l.parts.train == 0 or l.parts.held == 0) return error.NoAnswers;
        const s = &l.store.?;
        const Settle = struct {
            g: *const sg.Gpu,
            s: *const Store,
            first: usize,
            rows: usize,
            change: ?mtl.Buffer,
            d: usize,
            w: usize,
            pub fn encode(j: @This(), _: *Metal, e: *fwd.Enc) !void {
                j.g.settle(e, At.of(j.s.h).plus(j.first * j.d * 2), At.of(j.s.keys).plus(j.first * j.w * 2), At.of(j.s.hd), At.of(j.s.base).plus(j.first * j.d * 4), j.rows, j.change);
            }
        };
        const lesson_len = l.parts.train + l.parts.held + l.parts.near;
        for ([_][2]usize{ .{ 0, if (l.keep_new) l.keep_len else 0 }, .{ Store.lesson0, lesson_len } }) |part| {
            if (part[1] == 0) continue;
            try l.b.submit(.prefill, l.b.next, Settle{ .g = &l.gpu.?, .s = s, .first = part[0], .rows = part[1], .change = l.installed(), .d = d, .w = w });
            try l.b.drain();
        }
        l.keep_new = false;
    }

    /// The step's rows: the fact's answers and near misses, earlier answers spread over the ring, the keep rows.
    fn fill(l: *Learner) void {
        const c = l.b.m.config;
        const t = &l.train.?;
        t.rows = 0;
        l.put(t, Store.lesson0, l.parts.train);
        l.put(t, Store.lesson0 + l.parts.train + l.parts.held, l.parts.near);
        const replay = @min(replay_rows, l.ring_len);
        for (0..replay) |i| l.put(t, Store.ring0 + i * l.ring_len / replay, 1);
        l.put(t, 0, l.keep_len);
        t.seal(c.shared_width, c.hidden);
        const h = &l.held.?;
        h.rows = 0;
        l.put(h, Store.lesson0 + l.parts.train, l.parts.held);
        h.seal(c.shared_width, c.hidden);
    }

    /// Rows [first, first + n) of the store appended to a batch.
    fn put(l: *Learner, t: *sg.Batch, first: usize, n: usize) void {
        const c = l.b.m.config;
        const d = c.hidden;
        const w = c.shared_width;
        const s = &l.store.?;
        const n_fit = @min(n, t.cap - t.rows);
        @memcpy(t.base.slice(f32, t.cap * d)[t.rows * d ..][0 .. n_fit * d], s.base.slice(f32, Store.rows * d)[first * d ..][0 .. n_fit * d]);
        @memcpy(t.keys.slice(u16, t.cap * w)[t.rows * w ..][0 .. n_fit * w], s.keys.slice(u16, Store.rows * w)[first * w ..][0 .. n_fit * w]);
        @memcpy(t.targets.slice(u32, t.cap)[t.rows..][0..n_fit], s.targets[first..][0..n_fit]);
        t.rows += n_fit;
    }

    /// One bounded step on the GPU, and every few steps the held-out check; the change goes in when the lesson ends.
    fn stepOnce(l: *Learner) !Step {
        const g = &l.gpu.?;
        const check = (l.taken + 1) % check_every == 0 or l.taken + 1 == l.budget;
        const Job = struct {
            g: *const sg.Gpu,
            train: *const sg.Batch,
            held: ?*const sg.Batch,
            pub fn encode(j: @This(), _: *Metal, e: *fwd.Enc) !void {
                j.g.forward(e, j.train);
                j.g.backward(e, j.train);
                if (j.held) |h| j.g.forward(e, h);
            }
        };
        try l.b.submit(.prefill, l.b.next, Job{ .g = g, .train = &l.train.?, .held = if (check) &l.held.? else null });
        try l.b.drain();
        if (!g.stepped()) return error.NonfiniteStep;
        l.taken += 1;
        l.loss = meanLoss(&l.train.?);
        const recalled = check and allRecalled(&l.held.?);
        if (!recalled and l.taken < l.budget) return .{ .done = false };
        try l.install();
        l.added = l.first;
        if (l.first) l.keepAnswers();
        l.dirty = true;
        l.phase = if (l.lesson.save) .save else .idle;
        return .{ .done = l.phase == .idle, .changed = true, .report = .{ .learned = .{ .recalled = recalled, .steps = l.taken, .loss = l.loss } } };
    }

    /// The stepped change into the forward (made and attached the first time), the one it replaces kept for undo.
    fn install(l: *Learner) !void {
        const m = l.b.m;
        const c = m.config;
        const n = c.hidden * c.shared_width;
        const moe = &m.weights.layers[c.layers - 1].moe;
        if (l.before == null) l.before = try m.device.buffer(n * 2, opts);
        l.had = moe.slide != null;
        if (moe.slide) |now| @memcpy(l.before.?.slice(u16, n), now.slice(u16, n));
        l.can_undo = true;
        if (moe.slide == null) {
            if (l.made == null) {
                const buffer = try m.device.buffer(n * 2, opts);
                m.weights.owned.append(m.allocator, buffer) catch |e| {
                    buffer.deinit();
                    return e;
                };
                l.made = buffer;
            }
            moe.slide = l.made;
        }
        @memcpy(moe.slide.?.slice(u16, n), l.gpu.?.work.slice(u16, n));
    }

    /// The lesson's answers join the ring, the oldest giving way.
    fn keepAnswers(l: *Learner) void {
        l.ring_before = .{ l.ring_at, l.ring_len };
        const c = l.b.m.config;
        const d = c.hidden;
        const w = c.shared_width;
        const s = &l.store.?;
        const base = s.base.slice(f32, Store.rows * d);
        const keys = s.keys.slice(u16, Store.rows * w);
        for (0..l.parts.train) |i| {
            const from = Store.lesson0 + i;
            const to = Store.ring0 + l.ring_at;
            @memcpy(base[to * d ..][0..d], base[from * d ..][0..d]);
            @memcpy(keys[to * w ..][0..w], keys[from * w ..][0..w]);
            s.targets[to] = s.targets[from];
            l.ring_at = (l.ring_at + 1) % ring_rows;
            l.ring_len = @min(l.ring_len + 1, ring_rows);
        }
    }

    /// The last lesson taken back: the forward's change as before it, its answers out of the ring; false if none.
    fn takeBack(l: *Learner) bool {
        if (!l.can_undo) return false;
        l.can_undo = false;
        const m = l.b.m;
        const c = m.config;
        const n = c.hidden * c.shared_width;
        const moe = &m.weights.layers[c.layers - 1].moe;
        if (l.had) @memcpy(moe.slide.?.slice(u16, n), l.before.?.slice(u16, n)) else moe.slide = null;
        l.gpu.?.resetTo(moe.slide);
        if (l.added) {
            l.ring_at = l.ring_before[0];
            l.ring_len = l.ring_before[1];
            l.added = false;
        }
        return true;
    }

    /// The learned projection written into the model's own shards: its codes' values plus the change, in bf16.
    fn bake(l: *Learner) !void {
        const m = l.b.m;
        const c = m.config;
        const n = c.hidden * c.shared_width;
        var name: [160]u8 = undefined;
        const base = try learned.downName(&name, c.layers - 1);
        var full: [3][192]u8 = undefined;
        var names: [3][]const u8 = undefined;
        var codes: [3]ckpt.Tensor = undefined;
        inline for (.{ "weight", "scales", "biases" }, 0..) |part, i| {
            names[i] = try std.fmt.bufPrint(&full[i], "{s}." ++ part, .{base});
            codes[i] = try m.checkpoint.get(names[i]);
        }
        const values = try l.gpa.alloc(f32, n);
        defer l.gpa.free(values);
        affine4.dequantize(codes[0].host(u32), codes[1].host(u16), codes[2].host(u16), values);
        const out = try l.gpa.alloc(u16, n);
        defer l.gpa.free(out);
        for (out, values, l.gpu.?.master.slice(f32, n)) |*o, v, x| o.* = affine4.bf16of(v + x);
        try shard_edit.bake(l.gpa, l.io, l.dir, &.{.{ .name = names[0], .dtype = .bf16, .shape = &.{ c.hidden, c.shared_width }, .bytes = std.mem.sliceAsBytes(out), .drop = &.{ names[1], names[2] } }});
    }
};

fn meanLoss(b: *const sg.Batch) f32 {
    var sum: f32 = 0;
    for (b.stats.slice([2]f32, b.rows)) |s| sum += s[0];
    return sum / @as(f32, @floatFromInt(@max(b.rows, 1)));
}

fn allRecalled(b: *const sg.Batch) bool {
    for (b.stats.slice([2]f32, b.rows)) |s| if (!(s[1] > 0.5)) return false;
    return b.rows > 0;
}
