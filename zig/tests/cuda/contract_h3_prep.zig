//! The H3 contract test's inputs for each kernel and its launch, bound as the family binds it (families/h3/cuda.zig).
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const t = @import("contract_h3.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const H3 = t.H3;
const heads = t.heads;
const slots = t.slots;

/// h3_attention's dynamic shared bytes: a query, a key and a value tile.
pub const attention_shared = 3 * 64 * 128 * 2;

/// Gate 4's dot rows: integer terms whose second half cancels the first, so fp32 sums are exact and the result small.
fn mirror(x: []f64, w: []f64, k: usize, rows: usize, n: usize) void {
    const h = k / 2;
    for (0..rows) |r| for (0..h) |i| {
        x[r * k + h + i] = x[r * k + i];
    };
    for (0..n) |o| for (0..h) |i| {
        w[o * k + h + i] = -w[o * k + i] + @as(f64, if (i < 4) 1 else 0);
    };
}

/// Tile sizes: most whole, some short (the last rows of a segment), never empty.
fn sizes(self: *H3, n: usize, rng: std.Random) ![]i32 {
    const v = try self.arena.allocator().alloc(i32, n);
    for (v) |*s| s.* = if (rng.uintLessThan(u32, 4) == 0) @intCast(1 + rng.uintLessThan(u32, 64)) else 64;
    return v;
}

/// Each slot's row: the inverse of a shuffled row-to-slot map.
fn rowOf(self: *H3, slot: []const i32) ![]i32 {
    const v = try self.arena.allocator().alloc(i32, slot.len);
    for (slot, 0..) |s, r| v[@intCast(s)] = @intCast(r);
    return v;
}

pub fn prepare(self: *H3, s: c.Shape, fill: g.Fill, rng: std.Random) !bool {
    const al = self.arena.allocator();
    const stress = fill == .stress;
    self.in = .{ .rows = s.rows, .n = s.n, .k = s.k, .heads = heads, .slots = slots };
    switch (self.which) {
        .norm, .gate_add => {
            const C = s.n;
            if (self.which == .norm and s.k != C) return false;
            // terms of one sign: no cancellation hides an intermediate rounding from the bound
            const x = try self.values(s.rows * C, 0, 2, fill, rng, true, true);
            const y = try self.values(s.rows * C, 0, 2, .random, rng, true, false);
            const w = try self.values(C, 0.5, 1.5, .random, rng, true, false);
            // the six modulation tables of one line, bf16 values held in fp32 as the engine's are
            const tab = try self.values(6 * C, 0, 0.5, .random, rng, true, false);
            self.in.a = x;
            self.in.b = y;
            self.in.d = w;
            self.in.e = tab[2 * C .. 3 * C];
            self.in.f = tab[C .. 2 * C];
            self.in.g = tab[0..C];
            const line = try al.alloc(i32, s.rows);
            @memset(line, 0);
            try self.output(x.len * 2);
            self.fresh = try self.encode(x, true);
            try self.upload(y, true);
            if (self.which == .norm) try self.upload(w, true);
            try self.upload(tab, false);
            try self.ints(line);
            if (self.which == .norm) try self.output(x.len * 2);
            self.outs = (if (self.which == .norm) @as(usize, 2) else 1) * s.rows * C;
        },
        .gate_mix => {
            const R = s.rows;
            const y = try self.values(R * s.n, 0, 2, fill, rng, true, true);
            const gte = try self.values(R * s.n, 0, 2, .random, rng, true, false);
            const cc = try self.values(heads * (slots / 64) * 128, 0, 1, .random, rng, false, false);
            const slot = try al.alloc(i32, R);
            for (slot, 0..) |*o, i| o.* = @intCast(i);
            rng.shuffle(i32, slot);
            self.in.a = y;
            self.in.b = gte;
            self.in.d = cc;
            self.in.ints = slot;
            self.in.slots = slots / 64;
            try self.output(y.len * 2);
            self.fresh = try self.encode(y, true);
            try self.upload(gte, true);
            try self.upload(cc, false);
            try self.ints(slot);
            self.outs = R * s.n;
        },
        .swiglu => {
            const f = try self.values(s.rows * 2 * s.n, -4, 4, fill, rng, true, true);
            self.in.a = f;
            try self.upload(f, true);
            try self.output(s.rows * s.n * 2);
            self.outs = s.rows * s.n;
        },
        .heads => {
            // every slot holds a row, so the outputs hold no poison
            const R: usize = slots;
            const n = R * heads * 128;
            // a channel and its partner share a table entry: a small sine, a large cosine, so values stay one-signed
            const q = try self.values(n, 1, 2, fill, rng, true, true);
            const k = try self.values(n, 1, 2, .random, rng, true, false);
            const v = try self.values(n, -2, 2, .random, rng, true, false);
            const w = try self.values(256, 0.5, 1.5, .random, rng, true, false);
            const half = s.k / 2;
            const cs = try al.alloc(f64, R * half);
            const sn = try al.alloc(f64, R * half);
            for (cs, sn) |*co, *si| {
                co.* = @as(f32, @floatCast(0.9 + 0.1 * rng.float(f64)));
                si.* = @as(f32, @floatCast(0.002 * rng.float(f64)));
            }
            const slot = try al.alloc(i32, R);
            for (slot, 0..) |*o, i| o.* = @intCast(i);
            rng.shuffle(i32, slot);
            self.in = .{ .a = q, .b = k, .d = v, .e = cs, .f = sn, .g = w, .ints = slot, .rows = R, .n = s.n, .k = s.k, .heads = heads, .slots = slots };
            // one product's rows: every head's q, then k, then v
            const qkv = try al.alloc(f64, 3 * n);
            const width = heads * 128;
            for (0..R) |r| for ([_][]const f64{ q, k, v }, 0..) |x, which| {
                @memcpy(qkv[(r * 3 + which) * width ..][0..width], x[r * width ..][0..width]);
            };
            try self.upload(qkv, true);
            try self.upload(w[0..128], true);
            try self.upload(w[128..], true);
            try self.upload(cs, false);
            try self.upload(sn, false);
            try self.ints(try rowOf(self, slot));
            for (0..3) |_| try self.output(heads * slots * 128 * 2);
            self.outs = 3 * heads * slots * 128;
        },
        .pool => {
            const n = heads * slots * 128;
            const lo: f64 = if (stress) -16 else -2;
            const q = try self.values(n, lo, -lo, fill, rng, true, true);
            const k = try self.values(n, lo, -lo, .random, rng, true, false);
            const v = try self.values(n, lo, -lo, .random, rng, true, false);
            const sz = try sizes(self, s.rows, rng);
            self.in = .{ .a = q, .b = k, .d = v, .ints = sz, .rows = s.rows, .n = s.n, .k = s.k, .heads = heads, .slots = slots };
            for ([_][]const f64{ q, k, v }) |x| try self.upload(x, true);
            try self.ints(sz);
            for (0..3) |_| try self.output(heads * s.rows * 128 * 4);
            self.outs = 3 * heads * s.rows * 128;
        },
        .tile_scores => {
            const lo: f64 = if (stress) -6 else -1;
            const qp = try self.values(heads * s.rows * 128, lo, -lo, fill, rng, false, true);
            const kp = try self.values(heads * s.rows * 128, lo, -lo, fill, rng, false, false);
            if (stress) mirror(qp, kp, 128, heads * s.rows, heads * s.rows);
            self.in.a = qp;
            self.in.b = kp;
            try self.upload(qp, false);
            try self.upload(kp, false);
            try self.output(heads * s.rows * s.rows * 4);
            self.outs = heads * s.rows * s.rows;
        },
        .coarse => {
            const sc = try self.values(heads * s.rows * s.rows, -4, 4, fill, rng, false, true);
            const vp = try self.values(heads * s.rows * 128, -2, 2, .random, rng, false, false);
            self.in.a = sc;
            self.in.b = vp;
            try self.upload(sc, false);
            try self.upload(vp, false);
            try self.output(heads * s.rows * 128 * 4);
            self.outs = heads * s.rows * 128;
        },
        .attention => {
            // `rows` query tiles over all 16 tiles of keys; each query tile attends `k` / 64 of them
            const T = s.rows;
            const ksel = s.k / 64;
            if (s.k % 64 != 0 or ksel > 16 or T > 16) return false;
            const n = heads * slots * 128;
            const q = try self.values(n, -2, 2, fill, rng, true, true);
            const k = try self.values(n, -2, 2, .random, rng, true, false);
            // values of one sign: a weighted mean, so a weight's bf16 rounding is relative to the output
            const v = try self.values(n, 0, 2, .random, rng, true, false);
            const sz = try sizes(self, 16, rng);
            // each (head, query tile)'s key tiles: ksel distinct ones in tile order
            const list = try al.alloc(i32, heads * T * ksel);
            const pick = try al.alloc(i32, 16);
            for (0..heads * T) |row| {
                for (pick, 0..) |*p, i| p.* = @intCast(i);
                rng.shuffle(i32, pick);
                std.mem.sort(i32, pick[0..ksel], {}, std.sort.asc(i32));
                @memcpy(list[row * ksel ..][0..ksel], pick[0..ksel]);
            }
            const row_of = try al.alloc(i32, slots);
            @memset(row_of, -1);
            var rows: usize = 0;
            for (sz[0..T], 0..) |size, tile| for (0..@intCast(size)) |r| {
                row_of[tile * 64 + r] = @intCast(rows);
                rows += 1;
            };
            self.in = .{ .a = q, .b = k, .d = v, .ints = list, .more = sz, .rows = T, .n = s.n, .k = s.k, .heads = heads, .slots = slots, .scale = 1.0 / @sqrt(128.0) };
            for ([_][]const f64{ q, k, v }) |x| try self.upload(x, true);
            try self.ints(list);
            try self.ints(sz);
            try self.ints(row_of);
            try self.output(rows * heads * 128 * 2);
            self.outs = rows * heads * 128;
        },
    }
    return true;
}

fn int(n: usize) c_int {
    return @intCast(n);
}

fn go(self: *H3, grid: cuda.Dim3, threads: u32, shared: u32, args: *cuda.Args) !void {
    try cuda.launch.launch(self.f, .{ .grid = grid, .block = .{ .x = threads }, .shared = shared }, self.rig.s, args);
}

pub fn launch(self: *H3) !void {
    const s = self.shape;
    const b = self.bufs;
    var a: cuda.Args = .{};
    switch (self.which) {
        .norm => {
            // X (in place), Y, W, the gate table, the tables, LINE, the normed rows
            for ([_]u64{ b[0], b[1], b[2], b[3], b[3], b[4], b[5] }) |p| a.add(p);
            for ([_]usize{ s.n, 2, 1, 0 }) |n| a.add(int(n));
            a.add(@as(f32, 1e-5));
            try go(self, .{ .x = @intCast(s.rows) }, 256, 0, &a);
        },
        .gate_add => {
            for ([_]u64{ b[0], b[1], b[2], b[3] }) |p| a.add(p);
            a.add(int(s.n));
            a.add(int(2));
            try go(self, .{ .x = @intCast(s.rows) }, 256, 0, &a);
        },
        .gate_mix => {
            for (b[0..4]) |p| a.add(p);
            for ([_]usize{ s.n, s.n, slots / 64 }) |n| a.add(int(n));
            try go(self, .{ .x = @intCast(s.rows) }, 256, 0, &a);
        },
        .swiglu => {
            a.add(b[0]);
            a.add(b[1]);
            a.add(int(s.rows));
            a.add(int(s.n));
            try go(self, .{ .x = 64 }, 256, 0, &a);
        },
        .heads => {
            for (b[0..9]) |p| a.add(p);
            for ([_]usize{ heads, 3 * heads * 128, s.k / 2, slots }) |n| a.add(int(n));
            a.add(@as(f32, 1e-5));
            try go(self, .{ .x = slots / 64 }, 256, 0, &a);
        },
        .pool => {
            for (b[0..7]) |p| a.add(p);
            a.add(int(slots));
            a.add(int(s.rows));
            try go(self, .{ .x = @intCast(s.rows), .y = heads }, 32, 0, &a);
        },
        .tile_scores => {
            for (b[0..3]) |p| a.add(p);
            a.add(int(s.rows));
            try go(self, .{ .x = @intCast((s.rows + 7) / 8), .y = @intCast(s.rows), .z = heads }, 256, 0, &a);
        },
        .coarse => {
            for (b[0..3]) |p| a.add(p);
            a.add(int(s.rows));
            try go(self, .{ .x = @intCast(s.rows), .y = heads }, 32, 0, &a);
        },
        .attention => {
            for (b[0..7]) |p| a.add(p);
            for ([_]usize{ slots, heads, s.rows, s.k / 64, 0, 1 }) |n| a.add(int(n));
            a.add(@as(f32, @floatCast(self.in.scale)));
            try go(self, .{ .x = @intCast(s.rows), .y = heads }, 128, attention_shared, &a);
        },
    }
}
