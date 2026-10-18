//! The H3 contract test's inputs for each kernel and its launch, bound as the family binds it (families/h3).
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const t = @import("metal_h3.zig");
const ref = @import("metal_h3_ref.zig");

const c = contract.spec;
const g = contract.gates;
const H3 = t.H3;
const heads = 2;
const slots = 1024;

/// Gate 4's dot rows: integer terms whose second half cancels the first, so fp32 sums are exact and the result small.
fn mirror(x: []f64, w: []f64, k: usize, rows: usize, n: usize) void {
    const h = k / 2;
    for (0..rows) |r| for (0..h) |i| {
        // one sign, and 1 where w is nudged: partials pass fp16's spacing and every result is 4, a bf16 bound of 1/64
        x[r * k + i] = if (i < 4) 1 else @abs(x[r * k + i]);
        x[r * k + h + i] = x[r * k + i];
    };
    for (0..n) |o| for (0..h) |i| {
        w[o * k + i] = @abs(w[o * k + i]);
        w[o * k + h + i] = -w[o * k + i] + @as(f64, if (i < 4) 1 else 0);
    };
}

/// Tile sizes: most whole, some short (the last rows of a segment), never empty.
fn sizes(self: *H3, n: usize, rng: std.Random) ![]i32 {
    const v = try self.arena.allocator().alloc(i32, n);
    for (v) |*s| s.* = if (rng.uintLessThan(u32, 4) == 0) @intCast(1 + rng.uintLessThan(u32, 64)) else 64;
    return v;
}

pub fn prepare(self: *H3, s: c.Shape, fill: g.Fill, rng: std.Random) !bool {
    const al = self.arena.allocator();
    const stress = fill == .stress;
    self.in = .{ .rows = s.rows, .n = s.n, .k = s.k, .heads = heads, .slots = slots };
    switch (self.which) {
        .rows_in, .rows_out, .modulation => {
            const lo: f64 = if (stress) -6 else -1;
            const xr = s.rows;
            const half = self.which != .rows_in;
            if (self.which == .modulation and s.n % 18 != 0) return false;
            const x = try self.values(xr * s.k, lo, -lo, fill, rng, half, true);
            const w = try self.values(s.n * s.k, lo, -lo, fill, rng, self.which == .modulation, false);
            if (stress) mirror(x, w, s.k, xr, s.n);
            const b = try self.values(s.n, -0.5, 0.5, if (stress) .random else fill, rng, self.which == .modulation, false);
            if (stress) @memset(b, 0);
            self.in.a = x;
            self.in.b = w;
            self.in.d = b;
            try self.upload(x, half);
            try self.upload(w, self.which == .modulation);
            try self.upload(b, self.which == .modulation);
            try self.output(s.rows * s.n * @as(usize, if (self.which == .rows_out) 4 else 2));
        },
        .norm, .final_norm, .gate_add => {
            const C = s.n;
            if (self.which != .gate_add and s.k != C) return false;
            // terms of one sign: no cancellation hides an intermediate rounding from the bound
            const x = try self.values(s.rows * C, 0, 2, fill, rng, true, true);
            const y = try self.values(s.rows * C, 0, 2, .random, rng, true, false);
            const w = try self.values(C, 0.5, 1.5, .random, rng, true, false);
            // the six modulation tables of one line: gate, scale and shift rows (the kernel reads them by part)
            const tab = try self.values(6 * C, 0, 0.5, .random, rng, true, false);
            self.in.a = x;
            self.in.b = y;
            self.in.d = w;
            self.in.e = tab[2 * C .. 3 * C];
            self.in.f = tab[C .. 2 * C];
            self.in.g = tab[0..C];
            if (self.which == .final_norm) {
                // the final table is (shift, scale) a line
                self.in.g = tab[0..C];
                self.in.f = tab[C .. 2 * C];
            }
            const line = try al.alloc(i32, s.rows);
            @memset(line, 0);
            const fresh = try al.alloc(u16, x.len);
            for (fresh, x) |*o, v| o.* = c.toBf16(@floatCast(v));
            if (self.which == .final_norm) try self.upload(x, true) else {
                try self.output(x.len * 2);
                self.fresh = std.mem.sliceAsBytes(fresh);
            }
            try self.upload(y, true);
            try self.upload(w, true);
            try self.upload(tab, true);
            try self.ints(line);
            if (self.which == .norm) try self.output(x.len * 2);
            if (self.which == .final_norm) try self.output(x.len * 2);
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
            const fresh = try al.alloc(u16, y.len);
            for (fresh, y) |*o, v| o.* = c.toBf16(@floatCast(v));
            try self.output(y.len * 2);
            self.fresh = std.mem.sliceAsBytes(fresh);
            try self.upload(gte, true);
            try self.upload(cc, false);
            try self.ints(slot);
        },
        .swiglu, .swiglu_gate_first => {
            const f = try self.values(s.rows * 2 * s.n, -4, 4, fill, rng, true, true);
            self.in.a = f;
            try self.upload(f, true);
            try self.output(s.rows * s.n * 2);
        },
        .heads => {
            // every slot written, so the outputs hold no poison: all the slots' rows whatever the shape asks
            const R: usize = slots;
            const n = R * heads * 128;
            const q = try self.values(n, 0, 2, fill, rng, true, true);
            const k = try self.values(n, 0, 2, .random, rng, true, false);
            const v = try self.values(n, -2, 2, .random, rng, true, false);
            const w = try self.values(256, 0.5, 1.5, .random, rng, true, false);
            const cs = try self.values(R * s.k, 0, 1, .random, rng, false, false);
            const sn = try self.values(R * s.k, 0, 1, .random, rng, false, false);
            // rotate-half subtracts below the half: a negative sine there keeps every rotated term of one sign
            for (0..R) |r| for (0..s.k / 2) |i| {
                sn[r * s.k + i] = -sn[r * s.k + i];
            };
            const slot = try al.alloc(i32, R);
            for (slot, 0..) |*o, i| o.* = @intCast(i);
            rng.shuffle(i32, slot);
            self.in = .{ .a = q, .b = k, .d = v, .e = cs, .f = sn, .g = w, .ints = slot, .rows = R, .n = s.n, .k = s.k, .heads = heads, .slots = slots };
            for ([_][]const f64{ q, k, v }) |x| try self.upload(x, true);
            try self.upload(w[0..128], true);
            try self.upload(w[128..], true);
            try self.upload(cs, false);
            try self.upload(sn, false);
            try self.ints(slot);
            for (0..3) |_| try self.output(heads * slots * 128 * 2);
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
        },
        .coarse => {
            const sc = try self.values(heads * s.rows * s.rows, -4, 4, fill, rng, false, true);
            const vp = try self.values(heads * s.rows * 128, -2, 2, .random, rng, false, false);
            self.in.a = sc;
            self.in.b = vp;
            try self.upload(sc, false);
            try self.upload(vp, false);
            try self.output(heads * s.rows * 128 * 4);
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
            return true;
        },
    }
    self.outs = switch (self.which) {
        .rows_in, .rows_out, .swiglu, .swiglu_gate_first, .modulation => s.rows * s.n,
        .norm => 2 * s.rows * s.n,
        .final_norm, .gate_add, .gate_mix => s.rows * s.n,
        .heads => 3 * heads * slots * 128,
        .pool => 3 * heads * s.rows * 128,
        .tile_scores => heads * s.rows * s.rows,
        .coarse => heads * s.rows * 128,
        .attention => unreachable,
    };
    return true;
}

fn bind(self: *H3, e: mtl.ComputeEncoder, i: usize, at: usize) void {
    e.setBuffer(self.bufs[i].buf, self.bufs[i].off, at);
}

pub fn launch(self: *H3, e: mtl.ComputeEncoder) !void {
    const s = self.shape;
    const eps = [1]f32{1e-5};
    e.setPipeline(self.pipe);
    switch (self.which) {
        .rows_in, .rows_out, .modulation => {
            for (0..3) |i| bind(self, e, i, i);
            e.setValue([4]i32{ @intCast(s.rows), @intCast(s.k), @intCast(s.n), 0 }, 3);
            bind(self, e, 3, 4);
            if (self.which == .modulation) {
                e.setValue([6]i32{ @intCast(s.rows), @intCast(s.k), @intCast(s.n), 3, 6, @intCast(s.n / 18) }, 3);
                e.dispatchGroups(mtl.Size.of((s.n + 7) / 8, 1, 1), mtl.Size.of(256, 1, 1));
            } else if (self.which == .rows_in) e.dispatchThreads(mtl.Size.of(s.n, s.rows, 1), mtl.Size.of(64, 16, 1)) else e.dispatchGroups(mtl.Size.of(s.n, s.rows, 1), mtl.Size.of(32, 1, 1));
        },
        .norm => {
            // X (in place), Y, W, the gate table, the tables, LINE; then the normed rows
            for (0..3) |i| bind(self, e, i, i);
            bind(self, e, 3, 3);
            bind(self, e, 3, 4);
            bind(self, e, 4, 5);
            e.setValue([6]i32{ @intCast(s.rows), @intCast(s.n), 1, 2, 1, 0 }, 6);
            e.setValue(eps, 7);
            bind(self, e, 5, 8);
            e.dispatchGroups(mtl.Size.of(s.rows, 1, 1), mtl.Size.of(32, 1, 1));
        },
        .gate_add => {
            for (0..2) |i| bind(self, e, i, i);
            bind(self, e, 3, 2);
            bind(self, e, 4, 3);
            e.setValue([4]i32{ @intCast(s.rows), @intCast(s.n), 1, 2 }, 4);
            e.dispatchGroups(mtl.Size.of(s.rows, 1, 1), mtl.Size.of(32, 1, 1));
        },
        .final_norm => {
            bind(self, e, 0, 0);
            bind(self, e, 2, 1);
            bind(self, e, 3, 2);
            bind(self, e, 4, 3);
            e.setValue([2]i32{ @intCast(s.rows), @intCast(s.n) }, 4);
            e.setValue(eps, 5);
            bind(self, e, 5, 6);
            e.dispatchGroups(mtl.Size.of(s.rows, 1, 1), mtl.Size.of(32, 1, 1));
        },
        .gate_mix => {
            for (0..4) |i| bind(self, e, i, i);
            e.setValue([3]i32{ @intCast(s.rows), heads, slots / 64 }, 4);
            e.dispatchThreads(mtl.Size.of(s.rows, heads, 1), mtl.Size.of(64, 2, 1));
        },
        .swiglu, .swiglu_gate_first => {
            bind(self, e, 0, 0);
            e.setValue([3]i32{ @intCast(s.rows), @intCast(s.n), @intFromBool(self.which == .swiglu_gate_first) }, 1);
            bind(self, e, 1, 2);
            e.dispatchThreads(mtl.Size.of(s.n, s.rows, 1), mtl.Size.of(256, 1, 1));
        },
        .heads => {
            for (0..8) |i| bind(self, e, i, i);
            e.setValue([5]i32{ slots, heads, @intCast(s.k), 0, 128 }, 8);
            e.setValue(eps, 9);
            for (0..3) |i| bind(self, e, 8 + i, 10 + i);
            e.dispatchGroups(mtl.Size.of(slots, heads, 1), mtl.Size.of(32, 1, 1));
        },
        .pool => {
            for (0..7) |i| bind(self, e, i, i);
            e.dispatchGroups(mtl.Size.of(s.rows, heads, 1), mtl.Size.of(32, 1, 1));
        },
        .tile_scores => {
            bind(self, e, 0, 0);
            bind(self, e, 1, 1);
            e.setValue([2]i32{ @intCast(s.rows), heads }, 2);
            bind(self, e, 2, 3);
            e.dispatchThreads(mtl.Size.of(s.rows, s.rows, heads), mtl.Size.of(32, 8, 1));
        },
        .coarse => {
            bind(self, e, 0, 0);
            bind(self, e, 1, 1);
            e.setValue([1]i32{@intCast(s.rows)}, 2);
            bind(self, e, 2, 3);
            e.dispatchGroups(mtl.Size.of(s.rows, heads, 1), mtl.Size.of(32, 1, 1));
        },
        .attention => {
            for (0..6) |i| bind(self, e, i, i);
            e.setValue([4]i32{ @intCast(s.rows), @intCast(s.k / 64), 0, 1 }, 6);
            e.setValue([1]f32{@floatCast(self.in.scale)}, 7);
            bind(self, e, 6, 8);
            e.dispatchGroups(mtl.Size.of(s.rows, heads, 1), mtl.Size.of(256, 1, 1));
        },
    }
}
