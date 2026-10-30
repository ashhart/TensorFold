//! Flash Next EXL3 draft chain decisions and the trace checkpoint under gates 1-4.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const u = @import("contract_flashnext_exl3_dplan_util.zig");
const c = core.contract.spec;
const g = core.contract.gates;
const noSums = u.noSums;
const put = u.put;
const fetch = u.fetch;
const same = u.same;
const experts = u.experts;
const flash = @import("flashnext");
const stateWords = u.stateWords;
const recent_rows_words = u.recent_rows_words;
const trace_words = u.trace_words;

pub const Draft = struct {
    rig: rig_.Rig,
    token_fn: cuda.Function,
    next_fn: cuda.Function,
    /// The decision kernel, which sets a graph conditional, runs inside a graph with an IF node.
    decide: bool,
    ids: []u32 = &.{},
    best: [4]f32 = undefined,
    index: u32 = 0,
    alive_in: i32 = 1,
    confidence: f64 = 0,
    d: [3]u64 = @splat(0),
    msg: [160]u8 = undefined,

    pub fn deinit(self: *Draft) void {
        self.rig.gpu.gpa.free(self.ids);
        self.rig.deinit();
    }

    const vt = g.hooks(Draft);

    pub fn kernel(self: *Draft) g.Kernel {
        return .{ .spec = .{
            .name = if (self.decide) "cuda Flash Next EXL3 fn_exl3_draft_next" else "cuda Flash Next EXL3 fn_exl3_draft_token",
            .accum = .none,
            .depth = noSums,
            .out = .raw,
            .nan = .{ .stated = "the best record's column index picks the token whatever its score fields hold; the chain ends only when a probability exp(best - lse) is below a positive confidence, so a NaN score never ends it and a probability of 0 does" },
            .served = &.{ .{ .rows = 1, .n = 1, .k = 0 }, .{ .rows = 1, .n = 4096, .k = 0 }, .{ .rows = 1, .n = 248320, .k = 0 } },
        }, .ctx = self, .vt = &vt };
    }

    fn expectedAlive(self: *Draft) i32 {
        const prob = @exp(@as(f64, self.best[0]) - @as(f64, self.best[1]));
        const low = self.confidence > 0.0 and prob < self.confidence;
        return @intFromBool(self.alive_in != 0 and !low);
    }

    pub fn prepare(self: *Draft, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.n == 0 or shape.n > 1 << 20) return false;
        self.rig.reset();
        const a = self.rig.gpu.gpa;
        a.free(self.ids);
        var prng = std.Random.DefaultPrng.init(seed ^ 0x64726166);
        const rng = prng.random();
        self.ids = try a.alloc(u32, shape.n);
        rng.bytes(std.mem.sliceAsBytes(self.ids));
        self.index = if (seed % 3 == 0) 0 else if (seed % 3 == 1) @intCast(shape.n - 1) else rng.uintLessThan(u32, @intCast(shape.n));
        const top = -rng.float(f32) * 4;
        self.best = .{ top, top + rng.float(f32) * 3, @bitCast(self.index), 0 };
        if (fill != .random) {
            const specials = [_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) };
            self.best[0] = specials[seed % 3];
            self.best[1] = specials[(seed / 3) % 3];
        }
        self.alive_in = if (seed & 4 == 0) 1 else 0;
        const prob = @exp(@as(f64, self.best[0]) - @as(f64, self.best[1]));
        self.confidence = switch (seed % 4) {
            0 => 0,
            1 => if (std.math.isFinite(prob) and prob > 0) prob * 0.9 else 0.5,
            2 => if (std.math.isFinite(prob) and prob > 0) prob * 1.1 else 0.5,
            else => 0.5,
        };
        self.d[0] = try self.rig.input(std.mem.asBytes(&self.best));
        self.d[1] = try self.rig.input(std.mem.sliceAsBytes(self.ids));
        self.d[2] = try self.rig.output(16);
        return true;
    }

    pub fn run(self: *Draft) !void {
        const init = [4]i32{ -1, self.alive_in, -1, 0 };
        try put(&self.rig, self.d[2], std.mem.asBytes(&init));
        const s = self.rig.s;
        if (!self.decide) {
            var a: cuda.Args = .{};
            a.add(self.d[0]);
            a.add(self.d[1]);
            a.add(self.d[2]);
            try cuda.launch.launch(self.token_fn, .{ .grid = .{ .x = 1 }, .block = .{ .x = 32 } }, s, &a);
            return s.synchronize();
        }
        const chain = flash.exl3.chain;
        var cond = try chain.Cond.open();
        defer cond.lib.close();
        var side = try cuda.Stream.init(self.rig.gpu.d, true);
        defer side.deinit();
        try cuda.graph.beginCapture(s, .thread_local);
        const handle = try cond.handle(s);
        var a: cuda.Args = .{};
        a.add(self.d[0]);
        a.add(self.d[1]);
        a.add(self.d[2]);
        a.add(self.d[2] + 4);
        a.add(self.d[2] + 8);
        a.add(handle);
        a.add(self.confidence);
        try cuda.launch.launch(self.next_fn, .{ .grid = .{ .x = 1 }, .block = .{ .x = 32 } }, s, &a);
        const body = try cond.ifNode(s, handle);
        const d = self.rig.gpu.d;
        try chain.Cond.ok(cond.begin_to_graph(@ptrCast(side.handle), body, null, null, 0, 1), "cuStreamBeginCaptureToGraph");
        try d.check(d.api.cuMemsetD32Async(self.d[2] + 12, 1, 1, side.handle), "cuMemsetD32Async");
        var inner: cuda.abi.Graph = null;
        try d.check(d.api.cuStreamEndCapture(side.handle, &inner), "cuStreamEndCapture");
        var graph = try cuda.graph.endCapture(s);
        defer graph.deinit();
        var exec = try graph.instantiate();
        defer exec.deinit();
        try exec.launchOn(s);
        try s.synchronize();
    }

    pub fn count(_: *Draft) usize {
        return 4;
    }

    pub fn reference(self: *Draft, want: []f64, mag: []f64) !void {
        for (want, mag, 0..) |*w, *m, i| {
            w.* = if (i == 0) @floatFromInt(self.ids[self.index]) else @floatFromInt(self.expectedAlive());
            m.* = @abs(w.*);
        }
    }

    pub fn check(self: *Draft, _: g.Gate) !?[]const u8 {
        if (!try self.rig.intact()) return "a payload guard changed";
        const out = try fetch(&self.rig, i32, self.d[2], 4);
        defer self.rig.gpu.gpa.free(out);
        const token: i32 = @bitCast(self.ids[self.index]);
        if (out[0] != token) return try std.fmt.bufPrint(&self.msg, "token {d}, not ids[{d}] = {d}", .{ out[0], self.index, token });
        if (!self.decide) return null;
        const alive = self.expectedAlive();
        if (out[1] != alive or out[2] != alive) return try std.fmt.bufPrint(&self.msg, "alive {d} and {d}, not {d} (best {d} {d}, confidence {d})", .{ out[1], out[2], alive, self.best[0], self.best[1], self.confidence });
        if (out[3] != alive) return try std.fmt.bufPrint(&self.msg, "the conditional body ran {d}, not {d}", .{ out[3], alive });
        return null;
    }
};

// ---- fn_exl3_dcheckpoint ----------------------------------------------------------------------------------

/// The kernel's checkpoint on a host copy of the trace buffer (dp_trace_valid, then the transfer 0 / 1 rules).
fn modelCheckpoint(tr: []u32, st: []const u32, lines: u32, transfer: i32) void {
    const seed_words: u64 = 2 * @as(u64, lines) + 3 * experts + recent_rows_words + 10;
    const record_words: u64 = 15 + 11 * @as(u64, tr[5]);
    const base = 16 + 2 * seed_words;
    const stride: u64 = 4 + 2 * seed_words;
    const capacity = if (record_words == 0) 0 else (trace_words - base) / record_words;
    const valid = lines > 0 and tr[0] == 0x44505431 and tr[1] == 4 and tr[4] == lines and tr[5] != 0 and base < trace_words and
        record_words <= trace_words - base and tr[6] == seed_words and tr[7] == record_words and tr[8] == capacity and tr[9] <= capacity and
        tr[12] <= 1 and tr[13] <= 1 and tr[14] <= (trace_words - base) / stride and
        base + @as(u64, tr[9]) * record_words <= trace_words - @as(u64, tr[14]) * stride;
    if (!valid or (transfer != 0 and transfer != 1)) {
        tr[13] = 1;
        return;
    }
    const sw: usize = @intCast(seed_words);
    const initialized = tr[9] != 0 or tr[14] != 0;
    if (transfer == 0) {
        if (initialized) for (0..sw) |i| if (tr[16 + sw + i] != st[i]) {
            tr[12] = 1;
        };
        return;
    }
    const end: u32 = @intCast(16 + 2 * seed_words + tr[9] * record_words);
    const start: u32 = @intCast(trace_words - tr[14] * stride);
    if (start -% end < stride or tr[10] != 0 or tr[11] != 0 or tr[15] != 0) {
        if (tr[15] != 0xffffffff) tr[15] += 1;
        return;
    }
    const at: usize = start - @as(usize, @intCast(stride));
    tr[at] = tr[9];
    tr[at + 1] = 1;
    tr[at + 2] = @intCast(sw);
    tr[at + 3] = tr[14];
    for (0..sw) |i| {
        if (!initialized) {
            tr[16 + i] = st[i];
            tr[16 + sw + i] = st[i];
        }
        tr[at + 4 + i] = tr[16 + sw + i];
        tr[at + 4 + sw + i] = st[i];
        tr[16 + sw + i] = st[i];
    }
    tr[14] += 1;
}

pub const Checkpoint = struct {
    rig: rig_.Rig,
    f: cuda.Function,
    lines: u32 = 0,
    st: [2][]u32 = .{ &.{}, &.{} },
    header: [16]u32 = undefined,
    d: [3]u64 = @splat(0),
    msg: [160]u8 = undefined,

    pub fn deinit(self: *Checkpoint) void {
        for (self.st) |s| self.rig.gpu.gpa.free(s);
        self.rig.deinit();
    }

    const vt = g.hooks(Checkpoint);

    pub fn kernel(self: *Checkpoint) g.Kernel {
        return .{ .spec = .{
            .name = "cuda Flash Next EXL3 fn_exl3_dcheckpoint",
            .accum = .none,
            .depth = noSums,
            .out = .raw,
            .nan = .{ .stated = "integer state words only: a checkpoint copies the planner's words, flags a discontinuity when they differ from the last record's end, and counts an invalid launch" },
            .served = &.{ .{ .rows = 1, .n = 6, .k = 0 }, .{ .rows = 1, .n = 38, .k = 0 }, .{ .rows = 1, .n = 160, .k = 0 } },
        }, .ctx = self, .vt = &vt };
    }

    pub fn prepare(self: *Checkpoint, shape: c.Shape, _: g.Fill, seed: u64) !bool {
        if (shape.n == 0 or shape.n > 512) return false;
        self.rig.reset();
        const a = self.rig.gpu.gpa;
        self.lines = @intCast(shape.n);
        var prng = std.Random.DefaultPrng.init(seed ^ 0x636b7074);
        const words = stateWords(shape.n);
        for (&self.st) |*s| {
            a.free(s.*);
            s.* = try a.alloc(u32, words);
            prng.random().bytes(std.mem.sliceAsBytes(s.*));
        }
        const max_rows: u32 = 16;
        const record_words = 15 + 11 * max_rows;
        const base = 16 + 2 * @as(u32, @intCast(words));
        self.header = .{ 0x44505431, 4, 0, 3, self.lines, max_rows, @intCast(words), record_words, (trace_words - base) / record_words, 0, 0, 0, 0, 0, 0, 0 };
        self.d[0] = try self.rig.input(std.mem.sliceAsBytes(self.st[0]));
        self.d[1] = try self.rig.input(std.mem.sliceAsBytes(self.st[1]));
        self.d[2] = try self.rig.output(trace_words * 4);
        return true;
    }

    /// A first checkpoint, a repeat, a different state, its checkpoint and a bad transfer.
    const steps = [_][2]i32{ .{ 0, 1 }, .{ 0, 0 }, .{ 1, 0 }, .{ 1, 1 }, .{ 1, 2 } };

    pub fn run(self: *Checkpoint) !void {
        const zeros = try self.rig.gpu.gpa.alloc(u8, trace_words * 4);
        defer self.rig.gpu.gpa.free(zeros);
        @memset(zeros, 0);
        @memcpy(zeros[0..64], std.mem.sliceAsBytes(&self.header));
        try put(&self.rig, self.d[2], zeros);
        for (steps) |step| {
            var a: cuda.Args = .{};
            a.add(self.d[@intCast(step[0])]);
            a.add(@as(c_int, @intCast(self.lines)));
            a.add(self.d[2]);
            a.add(@as(c_int, step[1]));
            try cuda.launch.launch(self.f, .{ .grid = .{ .x = 1 }, .block = .{ .x = 1 } }, self.rig.s, &a);
        }
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Checkpoint) usize {
        return self.st[0].len;
    }

    pub fn reference(self: *Checkpoint, want: []f64, mag: []f64) !void {
        for (self.st[0], want, mag) |x, *w, *m| {
            w.* = @floatFromInt(x);
            m.* = w.*;
        }
    }

    pub fn check(self: *Checkpoint, _: g.Gate) !?[]const u8 {
        if (!try self.rig.intact()) return "a payload guard changed";
        const a = self.rig.gpu.gpa;
        const tr = try a.alloc(u32, trace_words);
        defer a.free(tr);
        @memset(tr, 0);
        @memcpy(tr[0..16], &self.header);
        for (steps) |step| modelCheckpoint(tr, self.st[@intCast(step[0])], self.lines, step[1]);
        return try same(&self.rig, u32, self.d[2], tr, "trace word", &self.msg);
    }
};
