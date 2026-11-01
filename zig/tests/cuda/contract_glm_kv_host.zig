//! TF_GLM_KV_HOST's gather kernels (glm_ops.cu) under the contract on CUDA: ids and rows moved, checked exactly.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const glm = @import("glm_cuda");
const check = @import("check.zig");
const rig_ = @import("contract_rig.zig");
const pages = @import("contract_glm_pages.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const sparse = glm.sparse;

const row_bytes = pages.lw * 2; // a latent row, as the gather moves it
const rv = row_bytes / 16; // its 16-byte words
const page = pages.rows_a_page;
const cache_rows = 8192; // a layer's row cache, as cuda_kv_host.zig keeps it
const at = 3 * page - 7; // the call's first position: its rows cross a page edge

pub const Which = enum { mark, copy, remap, reset, rows, prev_mark, copy_cached, keep, unmark };

/// What a planted control breaks, and the clean case each one shadows.
const planted = std.enums.EnumArray(Which, []const u8).init(.{
    .mark = "control: GLM kv host mark meets a stale mark",
    .copy = "control: GLM kv host copy drops its last row",
    .remap = "control: GLM kv host remap reads swapped slots",
    .reset = "control: GLM kv host reset drops its last row",
    .rows = "control: GLM kv host dense rows use a swapped table",
    .prev_mark = "control: GLM kv host prev_mark keeps the rows this call rewrites (a stale mark)",
    .copy_cached = "control: GLM kv host copy_cached reads swapped cache slots",
    .keep = "control: GLM kv host keep drops its last row",
    .unmark = "control: GLM kv host unmark drops a previous row (a stale mark)",
});

fn noSums(_: c.Shape) usize {
    return 0;
}

const ids_rule = "integers only: arena rows, slots and counts";
const rows_rule = "row copies: every byte as stored, NaN and infinities included";

/// Rows a call x picks a row: decode and verify calls (16 rows or fewer) use the row cache, prompt chunks do not.
pub fn spec(which: Which) c.Spec {
    const w = sparse.width;
    const both: []const c.Shape = &.{ .{ .rows = 1, .n = w, .k = 1 }, .{ .rows = 8, .n = w, .k = 1 }, .{ .rows = 2048, .n = w, .k = 1 } };
    const chunk: []const c.Shape = &.{ .{ .rows = 64, .n = w, .k = 1 }, .{ .rows = 2048, .n = w, .k = 1 } };
    const cached: []const c.Shape = &.{ .{ .rows = 1, .n = w, .k = 1 }, .{ .rows = 8, .n = w, .k = 1 }, .{ .rows = 16, .n = w, .k = 1 } };
    const s: struct { []const u8, []const c.Shape, []const u8 } = switch (which) {
        .mark => .{ "cuda glm tf_glm_kvh_mark: a slot for each distinct pick", both, "integers only: slots follow atomicAdd's order, so each pick is checked through its slot to its row" },
        .copy => .{ "cuda glm tf_glm_kvh_copy: marked rows into their slots", chunk, rows_rule },
        .remap => .{ "cuda glm tf_glm_kvh_remap: token lists name slots", both, ids_rule },
        .reset => .{ "cuda glm tf_glm_kvh_reset: marked rows unseen again", chunk, ids_rule },
        .rows => .{ "cuda glm tf_glm_kvh_rows: visible rows in order through pages out of order", both, rows_rule },
        .prev_mark => .{ "cuda glm tf_glm_kvh_prev_mark: the layer cache's rows, less this call's", cached, ids_rule },
        .copy_cached => .{ "cuda glm tf_glm_kvh_copy_cached: rows from the layer cache or host pages", cached, rows_rule },
        .keep => .{ "cuda glm tf_glm_kvh_keep: the call's rows become the layer cache", cached, rows_rule },
        .unmark => .{ "cuda glm tf_glm_kvh_unmark: both calls' marks cleared", cached, ids_rule },
    };
    return .{ .name = s[0], .accum = .none, .depth = noSums, .out = .raw, .nan = .{ .stated = s[2] }, .served = s[1] };
}

/// One gather kernel's case: a sequence at `at` with `rows` rows, its picks on pages out of order.
pub const Kvh = struct {
    rig: rig_.Rig,
    f: cuda.Function,
    which: Which,
    plant: bool = false,
    short: bool = false, // mark only: half the slots its distinct picks need
    shape: c.Shape = undefined,
    ids: []i32 = &.{}, // the page table
    tokens: []i32 = &.{}, // [rows][width] arena rows
    counts: []i32 = &.{},
    distinct: []i32 = &.{}, // the picks' distinct rows, in slot order
    slot_of: []i32 = &.{}, // per arena row: its slot, -1 unpicked
    prev: []i32 = &.{}, // the layer cache's rows (prev_mark, unmark) or a cache slot per arena row (copy_cached)
    lc: []u8 = &.{}, // the arena's rows
    cache: []u8 = &.{}, // the layer cache's rows (copy_cached), the gathered slots (keep)
    arena_rows: usize = 0,
    unused: i32 = 0, // a row on a page the table skips: what the lists hold past their counts
    in: [8]u64 = @splat(0),
    state: [4]u64 = @splat(0),
    out: [3]u64 = @splat(0),
    msg: [200]u8 = undefined,

    pub fn deinit(self: *Kvh) void {
        self.free();
        self.rig.deinit();
    }

    fn free(self: *Kvh) void {
        const a = self.rig.gpu.gpa;
        inline for (.{ &self.ids, &self.tokens, &self.counts, &self.distinct, &self.slot_of, &self.prev }) |x| {
            a.free(x.*);
            x.* = &.{};
        }
        a.free(self.lc);
        a.free(self.cache);
        self.lc = &.{};
        self.cache = &.{};
    }

    pub fn kernel(self: *Kvh) g.Kernel {
        var declared = spec(self.which);
        if (self.short) declared.name = "cuda glm tf_glm_kvh_mark past its slots: flags the call, leaves no row marked";
        if (self.plant) declared.name = planted.get(self.which);
        return .{ .spec = declared, .ctx = self, .vt = &vt, .control = if (self.plant) .shapes else null };
    }

    const vt = g.hooks(Kvh);

    fn width(self: *const Kvh) usize {
        return self.shape.n;
    }

    fn total(self: *const Kvh) usize {
        return self.shape.rows * self.width();
    }

    fn visible(self: *const Kvh) usize {
        return at + self.shape.rows;
    }

    /// Position q's row in the arena: row q % 512 of page ids[q / 512].
    fn arena(self: *const Kvh, q: usize) i32 {
        return self.ids[q / page] * page + @as(i32, @intCast(q % page));
    }

    fn picked(self: *const Kvh, i: usize) bool {
        return i % self.width() < self.counts[i / self.width()];
    }

    fn cap(self: *const Kvh) usize {
        return if (self.short) self.distinct.len / 2 else self.distinct.len;
    }

    fn i32s(self: *Kvh, x: []const i32) !u64 {
        return self.rig.input(std.mem.sliceAsBytes(x));
    }

    fn one(self: *Kvh, v: usize) !u64 {
        const x: i32 = @intCast(v);
        return self.rig.input(std.mem.asBytes(&x));
    }

    /// Per arena row: -1, or its index in `rows` where given.
    fn marks(self: *Kvh, rows: []const i32) ![]i32 {
        const m = try self.rig.gpu.gpa.alloc(i32, self.arena_rows);
        @memset(m, -1);
        for (rows, 0..) |t, i| m[@intCast(t)] = @intCast(i);
        return m;
    }

    pub fn prepare(self: *Kvh, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0) return false;
        self.free();
        self.rig.reset();
        self.shape = shape;
        const a = self.rig.gpu.gpa;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const n_pages = (self.visible() + page - 1) / page;
        self.ids = try pages.shuffled(a, rng, n_pages);
        self.arena_rows = (n_pages + 3) * page;
        const skipped = for (0..n_pages + 3) |p| {
            if (std.mem.indexOfScalar(i32, self.ids, @intCast(p)) == null) break p;
        } else unreachable;
        self.unused = @intCast(skipped * page + 7);
        // the picks: each row's first counts[r] are rows it sees (row 1 sees none), the rest a row no call picks
        self.counts = try a.alloc(i32, shape.rows);
        for (self.counts, 0..) |*n, r| n.* = if (r == 1) 0 else @intCast(self.width() - r % 4);
        self.tokens = try a.alloc(i32, self.total());
        for (self.tokens, 0..) |*t, i| t.* = if (self.picked(i)) self.arena(rng.uintAtMost(usize, at + i / self.width())) else self.unused;
        var seen = try a.alloc(bool, self.arena_rows);
        defer a.free(seen);
        @memset(seen, false);
        var list: std.ArrayList(i32) = .empty;
        defer list.deinit(a);
        for (self.tokens, 0..) |t, i| if (self.picked(i) and !seen[@intCast(t)]) {
            seen[@intCast(t)] = true;
            try list.append(a, t);
        };
        rng.shuffle(i32, list.items);
        self.distinct = try list.toOwnedSlice(a);
        self.slot_of = try self.marks(self.distinct);
        self.lc = try a.alloc(u8, self.arena_rows * row_bytes);
        rng.bytes(self.lc);
        if (fill == .nonfinite) for (self.distinct[0..@min(3, self.distinct.len)], [_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) }) |t, v| {
            const at_row = @as(usize, @intCast(t)) * row_bytes;
            std.mem.writeInt(u16, self.lc[at_row..][0..2], c.toBf16(v), .little);
            std.mem.writeInt(u16, self.lc[at_row + row_bytes - 2 ..][0..2], c.toBf16(v), .little);
        };
        try self.bind(rng);
        return true;
    }

    /// The kernel's inputs, state (reset from inputs each run) and outputs.
    fn bind(self: *Kvh, rng: std.Random) !void {
        const a = self.rig.gpu.gpa;
        const d = self.distinct.len;
        const n_in = if (self.plant and (self.which == .copy or self.which == .reset or self.which == .keep)) d - 1 else d; // a dropped row
        switch (self.which) {
            .mark => {
                const clean = try self.marks(&.{});
                defer a.free(clean);
                if (self.plant) clean[@intCast(self.distinct[0])] = -2; // a stale mark from an earlier call
                self.in[0] = try self.i32s(self.tokens);
                self.in[1] = try self.i32s(self.counts);
                self.in[2] = try self.i32s(clean);
                self.state = .{ try self.rig.scratch(self.arena_rows * 4), try self.rig.scratch(self.cap() * 4), try self.rig.scratch(16), try self.rig.scratch(16) };
                self.out[0] = try self.rig.output(self.viewLen() * 4);
            },
            .copy => {
                self.in[0] = try self.rig.input(self.lc);
                self.in[1] = try self.i32s(self.distinct);
                self.in[2] = try self.one(n_in);
                self.out[0] = try self.rig.output(d * row_bytes);
            },
            .remap => {
                const marks_in = try a.dupe(i32, self.slot_of);
                defer a.free(marks_in);
                if (self.plant and d > 1) std.mem.swap(i32, &marks_in[@intCast(self.distinct[0])], &marks_in[@intCast(self.distinct[1])]);
                self.in[0] = try self.i32s(self.tokens);
                self.in[1] = try self.i32s(self.counts);
                self.in[2] = try self.i32s(marks_in);
                self.out[0] = try self.rig.output(self.total() * 4);
            },
            .reset => {
                self.in[0] = try self.i32s(self.distinct);
                self.in[1] = try self.one(n_in);
                self.in[2] = try self.i32s(self.slot_of);
                self.out[0] = try self.rig.output(self.arena_rows * 4);
            },
            .rows => {
                self.in[0] = try self.rig.input(self.lc);
                self.in[1] = try pages.uploadTable(&self.rig, self.ids, self.plant);
                self.in[2] = try self.one(at);
                self.out[0] = try self.rig.output(self.visible() * row_bytes);
            },
            .prev_mark => {
                // the cache: half the picks' rows and every row this call writes (a rejected draft's), shuffled
                var rows: std.ArrayList(i32) = .empty;
                defer rows.deinit(a);
                for (self.distinct[0 .. d / 2]) |t| if (!self.written(t)) try rows.append(a, t);
                for (at..self.visible()) |q| try rows.append(a, self.arena(q));
                rng.shuffle(i32, rows.items);
                self.prev = try a.dupe(i32, rows.items[0..@min(rows.items.len, cache_rows)]);
                const clean = try self.marks(&.{});
                defer a.free(clean);
                self.in[0] = try self.i32s(self.prev);
                self.in[1] = try self.one(self.prev.len);
                self.in[2] = try pages.uploadTable(&self.rig, self.ids, false);
                self.in[3] = try self.one(if (self.plant) at - self.shape.rows else at); // planted: this call's rows unseen
                self.in[4] = try self.i32s(clean);
                self.out[0] = try self.rig.output(self.arena_rows * 4);
            },
            .copy_cached => {
                // half the picks sit in the layer cache, at shuffled slots, with rows unlike their host rows
                const held = d / 2;
                const slots = try a.alloc(i32, held);
                defer a.free(slots);
                for (slots, 0..) |*s, i| s.* = @intCast(i);
                rng.shuffle(i32, slots);
                self.prev = try self.marks(&.{});
                for (self.distinct[0..held], slots) |t, s| self.prev[@intCast(t)] = s;
                const prev_in = try a.dupe(i32, self.prev);
                defer a.free(prev_in);
                if (self.plant and held > 1) std.mem.swap(i32, &prev_in[@intCast(self.distinct[0])], &prev_in[@intCast(self.distinct[1])]);
                self.cache = try a.alloc(u8, @max(held, 1) * row_bytes);
                rng.bytes(self.cache);
                self.in[0] = try self.rig.input(self.lc);
                self.in[1] = try self.i32s(self.distinct);
                self.in[2] = try self.one(d);
                self.in[3] = try self.i32s(prev_in);
                self.in[4] = try self.rig.input(self.cache);
                self.out[0] = try self.rig.output(d * row_bytes);
            },
            .keep => {
                self.cache = try a.alloc(u8, d * row_bytes); // the call's gathered slots
                rng.bytes(self.cache);
                self.in[0] = try self.rig.input(self.cache);
                self.in[1] = try self.i32s(self.distinct);
                self.in[2] = try self.one(n_in);
                const m: usize = @min(d, cache_rows);
                self.out = .{ try self.rig.output(m * row_bytes), try self.rig.output(m * 4), try self.rig.output(4) };
            },
            .unmark => {
                // the previous call's rows: a third of these picks and as many others
                var rows: std.ArrayList(i32) = .empty;
                defer rows.deinit(a);
                for (self.distinct[0 .. d / 3]) |t| try rows.append(a, t);
                for (0..d / 3) |i| try rows.append(a, self.arena(at - 1 - i % at));
                std.mem.sortUnstable(i32, rows.items, {}, std.sort.asc(i32));
                var k: usize = 0;
                for (rows.items) |t| if (k == 0 or t != rows.items[k - 1]) {
                    rows.items[k] = t;
                    k += 1;
                };
                rng.shuffle(i32, rows.items[0..k]);
                self.prev = try a.dupe(i32, rows.items[0..@min(k, cache_rows)]);
                const prev_marks = try self.marks(self.prev);
                defer a.free(prev_marks);
                self.in[0] = try self.i32s(self.prev);
                self.in[1] = try self.one(if (self.plant) self.prev.len - 1 else self.prev.len);
                self.in[2] = try self.i32s(prev_marks);
                self.in[3] = try self.i32s(self.distinct);
                self.in[4] = try self.one(d);
                self.in[5] = try self.i32s(self.slot_of);
                self.out = .{ try self.rig.output(self.arena_rows * 4), try self.rig.output(self.arena_rows * 4), 0 };
            },
        }
    }

    /// Whether arena row t is one this call writes (positions at .. at + rows).
    fn written(self: *const Kvh, t: i32) bool {
        for (at..self.visible()) |q| if (self.arena(q) == t) return true;
        return false;
    }

    /// mark's checked view: each pick's row through its slot, n, the flag, -2 rows; past its slots, counts instead.
    fn viewLen(self: *const Kvh) usize {
        return if (self.short) 5 else self.total() + 3;
    }

    fn launch(self: *Kvh, blocks: usize, args: anytype) !void {
        var a: cuda.Args = .{};
        inline for (args) |v| a.add(v);
        try cuda.launch.launch(self.f, .{ .grid = .{ .x = @intCast(@max(blocks, 1)) }, .block = .{ .x = 256 } }, self.rig.s, &a);
    }

    fn copy(self: *Kvh, dst: u64, src: u64, len: usize) !void {
        try (cuda.DeviceBuffer{ .d = self.rig.gpu.d, .ptr = dst, .len = len }).copyFrom(0, src, len, self.rig.s.handle);
    }

    /// As cuda_kv_host.zig launches each: 256 threads a block, a warp a row for the row copies.
    pub fn run(self: *Kvh) !void {
        const w: c_int = @intCast(self.width());
        const all: i64 = @intCast(self.total());
        const cap_: c_int = @intCast(self.cap());
        const upper = @min(self.total(), self.cap());
        const r: c_int = @intCast(self.shape.rows);
        switch (self.which) {
            .mark => {
                try self.copy(self.state[0], self.in[2], self.arena_rows * 4);
                for (self.state[2..4]) |p| try (cuda.DeviceBuffer{ .d = self.rig.gpu.d, .ptr = p, .len = 16 }).fill8(0, self.rig.s.handle);
                try self.launch(cdiv(self.total(), 256), .{ self.in[0], self.in[1], w, all, self.state[0], self.state[1], self.state[2], cap_, self.state[3] });
                try self.view();
            },
            .copy => try self.launch(cdiv(upper, 8), .{ self.in[0], self.in[1], self.in[2], self.out[0], @as(c_int, rv), cap_ }),
            .remap => {
                try self.copy(self.out[0], self.in[0], self.total() * 4);
                try self.launch(cdiv(self.total(), 256), .{ self.out[0], self.in[1], w, all, self.in[2] });
            },
            .reset => {
                try self.copy(self.out[0], self.in[2], self.arena_rows * 4);
                try self.launch(cdiv(upper, 256), .{ self.in[0], self.in[1], self.out[0], cap_ });
            },
            .rows => try self.launch(cdiv(self.visible(), 8), .{ self.in[0], self.in[1], @as(i64, page), self.in[2], r, self.out[0], @as(c_int, rv), @as(c_int, @intCast(self.visible())) }),
            .prev_mark => {
                try self.copy(self.out[0], self.in[4], self.arena_rows * 4);
                try self.launch(cdiv(cache_rows, 256), .{ self.in[0], self.in[1], self.out[0], self.in[2], @as(i64, page), self.in[3], r });
            },
            .copy_cached => try self.launch(cdiv(upper, 8), .{ self.in[0], self.in[1], self.in[2], self.out[0], @as(c_int, rv), cap_, self.in[3], self.in[4] }),
            .keep => try self.launch(cdiv(cache_rows, 8), .{ self.in[0], self.in[1], self.in[2], self.out[0], self.out[1], self.out[2], @as(c_int, rv), @as(c_int, cache_rows) }),
            .unmark => {
                try self.copy(self.out[0], self.in[2], self.arena_rows * 4);
                try self.copy(self.out[1], self.in[5], self.arena_rows * 4);
                try self.launch(cdiv(@max(upper, cache_rows), 256), .{ self.in[0], self.in[1], self.out[0], self.in[3], self.in[4], self.out[1], cap_ });
            },
        }
        try self.rig.s.synchronize();
    }

    /// mark's state read back and reduced to the view its check compares (slot numbers themselves vary run to run).
    fn view(self: *Kvh) !void {
        const a = self.rig.gpu.gpa;
        const marks_now = try a.dupe(u8, try self.rig.bytes(self.state[0], self.arena_rows * 4));
        defer a.free(marks_now);
        const list = try a.dupe(u8, try self.rig.bytes(self.state[1], self.cap() * 4));
        defer a.free(list);
        const head = try self.rig.bytes(self.state[2], 4);
        const n = std.mem.readInt(i32, head[0..4], .little);
        const flag_b = try self.rig.bytes(self.state[3], 4);
        const flag = std.mem.readInt(i32, flag_b[0..4], .little);
        const slot = std.mem.bytesAsSlice(i32, marks_now);
        const rows = std.mem.bytesAsSlice(i32, list);
        var stale: i32 = 0;
        for (slot) |s| stale += @intFromBool(s == -2);
        const out = try a.alloc(i32, self.viewLen());
        defer a.free(out);
        if (self.short) {
            var resolved: i32 = 0;
            var wrong: i32 = 0;
            for (slot, 0..) |s, t| if (s >= 0) {
                resolved += 1;
                wrong += @intFromBool(s >= rows.len or rows[@intCast(s)] != @as(i32, @intCast(t)));
            };
            const most: i32 = @intCast(self.distinct.len); // a row past the slots is unseen again: its later picks count again
            out[0..5].* = .{ resolved, wrong, @min(n, most), flag, stale };
        } else {
            for (self.tokens, 0..) |t, i| {
                const s = if (self.picked(i)) slot[@intCast(t)] else -1;
                out[i] = if (s >= 0 and s < rows.len) rows[@intCast(s)] else -1;
            }
            out[self.total()..][0..3].* = .{ n, flag, stale };
        }
        try (cuda.DeviceBuffer{ .d = self.rig.gpu.d, .ptr = self.out[0], .len = out.len * 4 }).upload(0, std.mem.sliceAsBytes(out));
    }

    pub fn count(self: *Kvh) usize {
        return switch (self.which) {
            .mark => self.viewLen() * 4,
            .copy, .copy_cached => self.distinct.len * row_bytes,
            .remap => self.total() * 4,
            .reset, .prev_mark => self.arena_rows * 4,
            .rows => self.visible() * row_bytes,
            .keep => @as(usize, @min(self.distinct.len, cache_rows)) * (row_bytes + 4) + 4,
            .unmark => self.arena_rows * 8,
        };
    }

    pub fn reference(_: *Kvh, want: []f64, mag: []f64) !void {
        @memset(want, 0);
        @memset(mag, 0);
    }

    fn got(self: *Kvh, ptr: u64, len: usize) ![]u8 {
        return self.rig.gpu.gpa.dupe(u8, try self.rig.bytes(ptr, len));
    }

    /// Ids `v` against `want`: null when equal, else the first difference named.
    fn ids_(self: *Kvh, what: []const u8, v: []const u8, want: []const i32) !?[]const u8 {
        for (want, 0..) |w, i| {
            const x = std.mem.readInt(i32, v[i * 4 ..][0..4], .little);
            if (x != w) return try std.fmt.bufPrint(&self.msg, "{s} {d}: {d} where {d} is due", .{ what, i, x, w });
        }
        return null;
    }

    /// Row `s` of `v` against `src`'s row: null when the bytes match.
    fn row(self: *Kvh, what: []const u8, v: []const u8, s: usize, src: []const u8, from: usize) !?[]const u8 {
        if (std.mem.eql(u8, v[s * row_bytes ..][0..row_bytes], src[from * row_bytes ..][0..row_bytes])) return null;
        return try std.fmt.bufPrint(&self.msg, "{s} slot {d}: not the bytes of row {d}", .{ what, s, from });
    }

    pub fn check(self: *Kvh, _: g.Gate) !?[]const u8 {
        const a = self.rig.gpu.gpa;
        const d = self.distinct.len;
        switch (self.which) {
            .mark => {
                const v = try self.got(self.out[0], self.viewLen() * 4);
                defer a.free(v);
                const want = try a.alloc(i32, self.viewLen());
                defer a.free(want);
                if (self.short) {
                    want[0..5].* = .{ @intCast(self.cap()), 0, @intCast(d), 1, 0 };
                    return self.ids_("past its slots: (resolved, misresolved, n, flag, rows left at -2)", v, want);
                }
                for (self.tokens, 0..) |t, i| want[i] = if (self.picked(i)) t else -1;
                want[self.total()..][0..3].* = .{ @intCast(d), 0, 0 };
                return self.ids_("pick (then n, flag, rows left at -2)", v, want);
            },
            .copy, .copy_cached => {
                const v = try self.got(self.out[0], d * row_bytes);
                defer a.free(v);
                for (self.distinct, 0..) |t, s| {
                    const p = if (self.which == .copy_cached) self.prev[@intCast(t)] else -1;
                    const why = if (p >= 0) try self.row("from the cache", v, s, self.cache, @intCast(p)) else try self.row("from host pages", v, s, self.lc, @intCast(t));
                    if (why != null) return why;
                }
                return null;
            },
            .remap => {
                const v = try self.got(self.out[0], self.total() * 4);
                defer a.free(v);
                const want = try a.alloc(i32, self.total());
                defer a.free(want);
                for (self.tokens, want, 0..) |t, *w, i| w.* = if (self.picked(i)) self.slot_of[@intCast(t)] else t;
                return self.ids_("token", v, want);
            },
            .reset, .prev_mark => {
                const v = try self.got(self.out[0], self.arena_rows * 4);
                defer a.free(v);
                const want = try self.marks(&.{});
                defer a.free(want);
                if (self.which == .prev_mark) for (self.prev, 0..) |t, i| if (!self.written(t)) {
                    want[@intCast(t)] = @intCast(i);
                };
                return self.ids_("arena row", v, want);
            },
            .rows => {
                const v = try self.got(self.out[0], self.visible() * row_bytes);
                defer a.free(v);
                for (0..self.visible()) |q| if (try self.row("visible", v, q, self.lc, @intCast(self.arena(q)))) |why| return why;
                return null;
            },
            .keep => {
                const m: usize = @min(d, cache_rows);
                const v = try self.got(self.out[0], m * row_bytes);
                defer a.free(v);
                for (0..m) |s| if (try self.row("cache", v, s, self.cache, s)) |why| return why;
                const kept = try self.got(self.out[1], m * 4);
                defer a.free(kept);
                if (try self.ids_("cache row", kept, self.distinct[0..m])) |why| return why;
                const n = try self.got(self.out[2], 4);
                defer a.free(n);
                return self.ids_("cache count", n, &.{@intCast(m)});
            },
            .unmark => {
                const want = try self.marks(&.{});
                defer a.free(want);
                for (self.out[0..2], [_][]const u8{ "cache mark of arena row", "slot mark of arena row" }) |p, what| {
                    const v = try self.got(p, self.arena_rows * 4);
                    defer a.free(v);
                    if (try self.ids_(what, v, want)) |why| return why;
                }
                return null;
            },
        }
    }
};

fn cdiv(x: usize, y: usize) usize {
    return (x + y - 1) / y;
}

/// The nine kernels, mark past its slots, and a planted control each, at stable addresses.
pub const Set = struct {
    cases: [2 * std.enums.values(Which).len + 1]Kvh,

    pub fn create(a: std.mem.Allocator, gpu: check.Gpu, s: cuda.Stream, ops: cuda.Module, kernels: *std.ArrayList(g.Kernel)) !*Set {
        const self = try a.create(Set);
        errdefer a.destroy(self);
        const all = std.enums.values(Which);
        for (all, 0..) |w, i| {
            self.cases[i] = .{ .rig = .{ .gpu = gpu, .s = s }, .f = try ops.function(symbol(w)), .which = w };
            self.cases[all.len + i] = .{ .rig = .{ .gpu = gpu, .s = s }, .f = try ops.function(symbol(w)), .which = w, .plant = true };
        }
        self.cases[2 * all.len] = .{ .rig = .{ .gpu = gpu, .s = s }, .f = try ops.function(symbol(.mark)), .which = .mark, .short = true };
        for (&self.cases) |*k| try kernels.append(a, k.kernel());
        return self;
    }

    pub fn destroy(self: *Set, a: std.mem.Allocator) void {
        for (&self.cases) |*k| k.deinit();
        a.destroy(self);
    }
};

fn symbol(w: Which) [:0]const u8 {
    return switch (w) {
        inline else => |x| "tf_glm_kvh_" ++ @tagName(x),
    };
}
