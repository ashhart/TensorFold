//! Flash Next EXL3 device pool plan under gates 1-4: the planner against a host model, and the kernel list.
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
const check = @import("check.zig");
const tp_ = @import("contract_flashnext_exl3_tp.zig");
const small = @import("contract_flashnext_exl3_dplan_copy.zig");
const draft = @import("contract_flashnext_exl3_dplan_draft.zig");
const stateWords = u.stateWords;
const recent_rows_words = u.recent_rows_words;

/// The planner's state, tables and outputs after one round: the kernel's formula on the same words.
const Model = struct {
    st: []u32,
    tab: [3][experts]i64,
    k2t: [3][experts]i32,
    work: []i64,
    hit: []i32,
    miss: []i32,
    ovf: []i32,
    work_n: i32 = 0,
    ovf_n: i32 = 0,

    fn colder(ca: u32, sa: u32, ia: usize, cb: u32, sb: u32, ib: usize) bool {
        if (ca != cb) return ca < cb;
        if (sa != sb) return sa < sb;
        return ia > ib;
    }

    fn round(m: *Model, lines: usize, recent: bool, picks: []const i32, src: []const i64, lens: []const i64, k2: []const i32, line_base: u64, line_bytes: u64, offs: [3]u64) void {
        const line_id: []i32 = @ptrCast(m.st[0..lines]);
        const stamp = m.st[lines .. 2 * lines];
        const line_of: []i32 = @ptrCast(m.st[2 * lines ..][0..experts]);
        const counts = m.st[2 * lines + experts ..][0..experts];
        const recent_counts = m.st[2 * lines + 2 * experts ..][0..experts];
        const recent_rows = m.st[2 * lines + 3 * experts ..][0..recent_rows_words];
        const misc = m.st[2 * lines + 3 * experts + recent_rows_words ..][0..10];
        const score = if (recent) recent_counts else counts;
        const n = picks.len;
        misc[0] += 1;
        const now = misc[0];
        misc[1] += 1;
        if (recent) for (0..n / 11) |row| {
            const expired = recent_rows[misc[8] * (experts / 32) ..][0 .. experts / 32];
            for (expired, 0..) |*word, w| {
                var removed = word.*;
                while (removed != 0) : (removed &= removed - 1) recent_counts[w * 32 + @ctz(removed)] -%= 1;
                word.* = 0;
            }
            for (0..11) |slot| {
                const e = picks[row * 11 + slot];
                if (e < 0 or e >= experts) continue;
                const bit = @as(u32, 1) << @intCast(e & 31);
                if (expired[@intCast(e >> 5)] & bit == 0) {
                    expired[@intCast(e >> 5)] |= bit;
                    recent_counts[@intCast(e)] += 1;
                }
            }
            misc[8] = (misc[8] + 1) % 256;
            if (misc[9] < 256) misc[9] += 1;
        };
        var uniq: [256]u16 = undefined;
        var app: [256]u16 = undefined;
        var miss: [256]u16 = undefined;
        var keep: [experts / 32]u32 = @splat(0);
        var nu: usize = 0;
        var over = false;
        for (picks) |p| {
            if (p < 0 or p >= experts) continue;
            const e: usize = @intCast(p);
            if (counts[e] != 0xffffffff) counts[e] += 1;
            if (line_of[e] >= 0) misc[2] += 1 else misc[3] += 1;
            var j: usize = 0;
            while (j < nu and uniq[j] != e) j += 1;
            if (j == nu) {
                if (nu == 256) {
                    over = true;
                    continue;
                }
                uniq[nu] = @intCast(e);
                app[nu] = 0;
                keep[e >> 5] |= @as(u32, 1) << @intCast(e & 31);
                nu += 1;
            }
            app[j] += 1;
        }
        var nm: usize = 0;
        for (0..nu) |j| {
            const ln = line_of[uniq[j]];
            if (ln >= 0) stamp[@intCast(ln)] = now else {
                miss[nm] = @intCast(j);
                nm += 1;
            }
        }
        for (1..nm) |a| {
            const x = miss[a];
            var b: isize = @as(isize, @intCast(a)) - 1;
            while (b >= 0) : (b -= 1) {
                const y = miss[@intCast(b)];
                const cx = score[uniq[x]];
                const cy = score[uniq[y]];
                const hotter = if (cx != cy) cx > cy else if (app[x] != app[y]) app[x] > app[y] else uniq[x] < uniq[y];
                if (!hotter) break;
                miss[@intCast(b + 1)] = y;
            }
            miss[@intCast(b + 1)] = x;
        }
        if (over) misc[6] += 1;
        const n_miss = if (over) 0 else nm;
        for (picks, 0..) |p, i| {
            const held = p >= experts or (p >= 0 and line_of[@intCast(p)] >= 0);
            m.hit[i] = if (p < 0) -1 else if (held) p else -1;
            m.miss[i] = if (p < 0) -1 else if (held) -1 else p;
        }
        for (miss[0..n_miss]) |mi| {
            var victim: ?usize = null;
            for (line_id, 0..) |id, i| if (id < 0) {
                victim = i;
                break;
            };
            if (victim == null) for (line_id, 0..) |id, i| {
                const e: usize = @intCast(id);
                if (keep[e >> 5] & (@as(u32, 1) << @intCast(e & 31)) != 0) continue;
                if (victim == null or colder(score[e], stamp[i], i, score[@intCast(line_id[victim.?])], stamp[victim.?], victim.?)) victim = i;
            };
            const id: usize = uniq[mi];
            if (victim) |v| {
                const old = line_id[v];
                if (old >= 0) {
                    line_of[@intCast(old)] = -1;
                    for (0..3) |w| {
                        m.tab[w][@intCast(old)] = 0;
                        m.k2t[w][@intCast(old)] = 0;
                    }
                    misc[5] += 1;
                }
                line_id[v] = @intCast(id);
                line_of[id] = @intCast(v);
                stamp[v] = now;
                const base = line_base + @as(u64, v) * line_bytes;
                for (0..3) |w| {
                    const dst: i64 = @bitCast(base + offs[w]);
                    m.tab[w][id] = dst;
                    m.k2t[w][id] = k2[id * 3 + w];
                    m.work[@as(usize, @intCast(m.work_n)) * 9 + w * 3 ..][0..3].* = .{ src[id * 3 + w], dst, lens[id * 3 + w] };
                }
                m.work_n += 1;
                misc[4] += 1;
            } else {
                for (0..3) |w| {
                    m.tab[w][id] = src[id * 3 + w];
                    m.k2t[w][id] = k2[id * 3 + w];
                }
                m.ovf[@intCast(m.ovf_n)] = @intCast(id);
                m.ovf_n += 1;
                misc[7] += 1;
            }
        }
    }
};

pub const Plan = struct {
    rig: rig_.Rig,
    f: cuda.Function,
    recent: bool = false,
    lines: usize = 0,
    picks: []i32 = &.{},
    st0: []u32 = &.{},
    tab0: [3][experts]i64 = undefined,
    k2t0: [3][experts]i32 = undefined,
    src: [experts * 3]i64 = undefined,
    lens: [experts * 3]i64 = undefined,
    k2: [experts * 3]i32 = undefined,
    // device buffers: picks src lens k2 | st tab0-2 k2t0-2 work work_n hit miss ovf ovf_n
    d: [17]u64 = @splat(0),
    msg: [160]u8 = undefined,
    const line_base: u64 = 0x7000_0000_0000;
    const line_bytes: u64 = 1 << 20;
    const offs: [3]u64 = .{ 0, 4096, 8192 };

    pub fn deinit(self: *Plan) void {
        const a = self.rig.gpu.gpa;
        a.free(self.picks);
        a.free(self.st0);
        self.rig.deinit();
    }

    const vt = g.hooks(Plan);

    pub fn kernel(self: *Plan) g.Kernel {
        return .{
            .spec = .{
                .name = "cuda Flash Next EXL3 fn_exl3_dplan",
                .accum = .none,
                .depth = noSums,
                .out = .raw,
                .nan = .{ .stated = "integer picks only: a pick outside [0, 512) is skipped (512 and above counts as held, as the shared expert), and no pick changes a table entry it does not plan" },
                // the victim search strides the lines by the 256 threads of the block
                .tiles = .{ .{}, .{ .size = 256, .step = 1, .min = 1 }, .{} },
                .served = &.{ .{ .rows = 1, .n = 38, .k = 0 }, .{ .rows = 4, .n = 38, .k = 0 }, .{ .rows = 16, .n = 160, .k = 0 }, .{ .rows = 16, .n = 6, .k = 0 } },
            },
            .ctx = self,
            .vt = &vt,
        };
    }

    pub fn prepare(self: *Plan, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.rows > 24 or shape.n == 0 or shape.n > 2048) return false;
        self.rig.reset();
        const a = self.rig.gpu.gpa;
        a.free(self.picks);
        a.free(self.st0);
        const lines = shape.n;
        self.lines = lines;
        self.recent = seed & 1 == 1;
        var prng = std.Random.DefaultPrng.init(seed ^ 0x706c616e);
        const rng = prng.random();
        self.picks = try a.alloc(i32, shape.rows * 11);
        const edge = [_]i32{ -1, 0, 511, 512, 513, std.math.minInt(i32), std.math.maxInt(i32) };
        for (self.picks, 0..) |*p, i| {
            p.* = if (rng.uintLessThan(u8, 10) < 6) rng.intRangeLessThan(i32, 0, 48) else rng.intRangeLessThan(i32, 0, experts);
            if (i % 11 == 10) p.* = 512;
            if (fill == .nonfinite and i % 3 == 0) p.* = edge[(i / 3) % edge.len];
        }
        self.st0 = try a.alloc(u32, stateWords(lines));
        @memset(self.st0, 0);
        const line_id: []i32 = @ptrCast(self.st0[0..lines]);
        const line_of: []i32 = @ptrCast(self.st0[2 * lines ..][0..experts]);
        @memset(line_of, -1);
        // held lines: a stress fill leaves none empty (every miss must evict, or be read in place)
        var order: [experts]u16 = undefined;
        for (&order, 0..) |*o, i| o.* = @intCast(i);
        rng.shuffle(u16, &order);
        for (line_id, 0..) |*id, i| {
            if (i >= experts or (fill != .stress and rng.uintLessThan(u8, 10) < 3)) {
                id.* = -1;
                continue;
            }
            id.* = order[i];
            line_of[order[i]] = @intCast(i);
        }
        for (self.st0[lines .. 2 * lines]) |*s| s.* = rng.uintLessThan(u32, 8);
        for (self.st0[2 * lines + experts ..][0..experts]) |*cnt| cnt.* = rng.uintLessThan(u32, 4);
        for (self.st0[2 * lines + 2 * experts ..][0..experts]) |*cnt| cnt.* = 0;
        self.st0[2 * lines + 3 * experts + recent_rows_words] = rng.uintLessThan(u32, 100); // now
        for (0..3) |w| for (0..experts) |i| {
            self.tab0[w][i] = 0x1111 + @as(i64, @intCast(rng.uintLessThan(u16, 1000)));
        };
        for (0..3) |w| for (0..experts) |i| {
            self.k2t0[w][i] = 7;
        };
        for (&self.src, &self.lens, 0..) |*s, *l, i| {
            s.* = 0x6000_0000_0000 + @as(i64, @intCast(i)) * 4096;
            l.* = 16 * (1 + @as(i64, @intCast(i % 7)));
        }
        for (&self.k2, 0..) |*x, i| x.* = @intCast(2 + i % 5);
        const r = &self.rig;
        self.d[0] = try r.input(std.mem.sliceAsBytes(self.picks));
        self.d[1] = try r.input(std.mem.sliceAsBytes(&self.src));
        self.d[2] = try r.input(std.mem.sliceAsBytes(&self.lens));
        self.d[3] = try r.input(std.mem.sliceAsBytes(&self.k2));
        self.d[4] = try r.output(self.st0.len * 4);
        for (0..3) |w| {
            self.d[5 + w] = try r.output(experts * 8);
            self.d[8 + w] = try r.output(experts * 4);
        }
        self.d[11] = try r.output(self.picks.len * 9 * 8);
        self.d[12] = try r.output(4);
        self.d[13] = try r.output(self.picks.len * 4);
        self.d[14] = try r.output(self.picks.len * 4);
        self.d[15] = try r.output(self.picks.len * 4);
        self.d[16] = try r.output(4);
        return true;
    }

    /// The in-out buffers go back to their prepared contents, so every run plans the same round.
    fn reset(self: *Plan) !void {
        const r = &self.rig;
        try put(r, self.d[4], std.mem.sliceAsBytes(self.st0));
        for (0..3) |w| {
            try put(r, self.d[5 + w], std.mem.asBytes(&self.tab0[w]));
            try put(r, self.d[8 + w], std.mem.asBytes(&self.k2t0[w]));
        }
        const zeros = try r.gpu.gpa.alloc(u8, self.picks.len * 9 * 8);
        defer r.gpu.gpa.free(zeros);
        @memset(zeros, 0);
        try put(r, self.d[11], zeros);
        try put(r, self.d[12], zeros[0..4]);
        try put(r, self.d[13], zeros[0 .. self.picks.len * 4]);
        try put(r, self.d[14], zeros[0 .. self.picks.len * 4]);
        try put(r, self.d[15], zeros[0 .. self.picks.len * 4]);
        try put(r, self.d[16], zeros[0..4]);
    }

    pub fn run(self: *Plan) !void {
        try self.reset();
        const d = self.d;
        var a: cuda.Args = .{};
        a.add(d[0]);
        a.add(@as(c_int, @intCast(self.picks.len)));
        a.add(d[4]);
        a.add(@as(c_int, @intCast(self.lines)));
        a.add(@as(c_int, @intFromBool(self.recent)));
        for (5..11) |i| a.add(d[i]);
        a.add(d[1]);
        a.add(d[2]);
        a.add(d[3]);
        a.add(line_base);
        a.add(line_bytes);
        for (offs) |o| a.add(o);
        a.add(d[11]);
        a.add(d[12]);
        a.add(d[13]);
        a.add(d[14]);
        a.add(d[15]);
        a.add(d[16]);
        a.add(@as(u64, 0)); // no trace
        try cuda.launch.launch(self.f, .{ .grid = .{ .x = 1 }, .block = .{ .x = 256 } }, self.rig.s, &a);
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Plan) usize {
        return self.picks.len;
    }

    pub fn reference(self: *Plan, want: []f64, mag: []f64) !void {
        for (self.picks, want, mag) |p, *w, *m| {
            w.* = @floatFromInt(p);
            m.* = @abs(w.*);
        }
    }

    pub fn check(self: *Plan, _: g.Gate) !?[]const u8 {
        if (!try self.rig.intact()) return "a payload guard changed";
        const a = self.rig.gpu.gpa;
        var m: Model = .{ .st = try a.dupe(u32, self.st0), .tab = self.tab0, .k2t = self.k2t0, .work = try a.alloc(i64, self.picks.len * 9), .hit = try a.alloc(i32, self.picks.len), .miss = try a.alloc(i32, self.picks.len), .ovf = try a.alloc(i32, self.picks.len) };
        defer a.free(m.st);
        defer a.free(m.work);
        defer a.free(m.hit);
        defer a.free(m.miss);
        defer a.free(m.ovf);
        @memset(m.work, 0);
        @memset(m.hit, 0);
        @memset(m.miss, 0);
        @memset(m.ovf, 0);
        m.round(self.lines, self.recent, self.picks, &self.src, &self.lens, &self.k2, line_base, line_bytes, offs);
        const r = &self.rig;
        const msg = &self.msg;
        if (try same(r, u32, self.d[4], m.st, "state", msg)) |why| return why;
        for (0..3) |w| {
            if (try same(r, i64, self.d[5 + w], &m.tab[w], "table", msg)) |why| return why;
            if (try same(r, i32, self.d[8 + w], &m.k2t[w], "k2 table", msg)) |why| return why;
        }
        if (try same(r, i64, self.d[11], m.work, "copy work", msg)) |why| return why;
        if (try same(r, i32, self.d[12], &.{m.work_n}, "copies", msg)) |why| return why;
        if (try same(r, i32, self.d[13], m.hit, "held picks", msg)) |why| return why;
        if (try same(r, i32, self.d[14], m.miss, "missed picks", msg)) |why| return why;
        if (try same(r, i32, self.d[15], m.ovf, "in-place experts", msg)) |why| return why;
        return try same(r, i32, self.d[16], &.{m.ovf_n}, "in-place count", msg);
    }
};

/// Every kernel of the image, plus both ranks' pick masks, for the runner's list.
pub const Set = struct {
    masks: tp_.Masks,
    mod: cuda.Module,
    plan: Plan,
    restore: small.Restore,
    copy: small.Copy,
    drafts: [2]draft.Draft,
    checkpoint: draft.Checkpoint,

    pub fn init(gpu: check.Gpu, s: cuda.Stream) !Set {
        var masks = try tp_.Masks.init(gpu, s);
        errdefer masks.deinit();
        var mod = try cuda.Module.load(gpu.d, cuda.kernels.flashnext_exl3_dplan);
        errdefer mod.unload();
        const rig: rig_.Rig = .{ .gpu = gpu, .s = s };
        const token_fn = try mod.function("fn_exl3_draft_token");
        const next_fn = try mod.function("fn_exl3_draft_next");
        return .{
            .masks = masks,
            .mod = mod,
            .plan = .{ .rig = rig, .f = try mod.function("fn_exl3_dplan") },
            .restore = .{ .rig = rig, .f = try mod.function("fn_exl3_drestore") },
            .copy = .{ .rig = rig, .f = try mod.function("fn_exl3_dcopy") },
            .drafts = .{ .{ .rig = rig, .token_fn = token_fn, .next_fn = next_fn, .decide = false }, .{ .rig = rig, .token_fn = token_fn, .next_fn = next_fn, .decide = true } },
            .checkpoint = .{ .rig = rig, .f = try mod.function("fn_exl3_dcheckpoint") },
        };
    }

    pub fn deinit(self: *Set) void {
        self.plan.deinit();
        self.restore.deinit();
        self.copy.deinit();
        for (&self.drafts) |*d| d.deinit();
        self.checkpoint.deinit();
        self.mod.unload();
        self.masks.deinit();
    }

    pub fn add(self: *Set, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel)) !void {
        try self.masks.add(a, list);
        try list.append(a, self.plan.kernel());
        try list.append(a, self.restore.kernel());
        try list.append(a, self.copy.kernel());
        for (&self.drafts) |*d| try list.append(a, d.kernel());
        try list.append(a, self.checkpoint.kernel());
    }
};
