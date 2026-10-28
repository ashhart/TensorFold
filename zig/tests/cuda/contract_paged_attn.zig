//! Liquid's paged attention (head 64 LFM2.5, 128 LFM3), decode and prompt rows, over shuffled pages against float64.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const rig_ = @import("contract_rig.zig");
const attn = @import("contract_attn.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;

const page = 512; // positions a page holds (paged_attn.cuh's PAGE)
const max_segs = 8; // streams a decode launch takes (paged_attn.cuh's MAXSEG)
const chunk_blocks = 64; // key chunks a decode launch walks in parallel at most (the families' chunk_blocks)
const pad = 64; // elements between a page's values and the next page: a stride that ignores PS reads them

/// Decode: the chunk and merge kernels' symbols; prompt: pattn_kernel's symbol, rows a block, dynamic shared bytes.
pub const Form = union(enum) {
    decode: struct { chunk: [:0]const u8, merge: [:0]const u8 },
    prompt: struct { sym: [:0]const u8, rows: usize, smem: u32 },
};

/// A launch's fixed heads and head dim, its kernels, and whether they come from the controls image.
pub const Cfg = struct { name: []const u8, h: usize, hk: usize, d: usize, form: Form, control: bool = false, served: []const c.Shape };

const pattn64 = "_ZN20tf_prefill_attention12pattn_kernelILi64ELi8ELi4ELi8EEEvPK13__nv_bfloat16S3_S3_PS1_iiiiifPKix";
const pattn128 = "_ZN20tf_prefill_attention12pattn_kernelILi128ELi8ELi4ELi8EEEvPK13__nv_bfloat16S3_S3_PS1_iiiiifPKix";
const prompt128_served = [_]c.Shape{ .{ .rows = 256, .n = 16, .k = 0 }, .{ .rows = 64, .n = 16, .k = 3000 }, .{ .rows = 100, .n = 16, .k = 500 } };
const decode64: Form = .{ .decode = .{ .chunk = "tf_paged_attn_chunk_segs_64", .merge = "tf_paged_attn_merge_segs_64" } };
const prompt64: Form = .{ .prompt = .{ .sym = pattn64, .rows = 32, .smem = 8 * 32 * 8 * 16 } };
const decode_served = [_]c.Shape{ .{ .rows = 1, .n = 32, .k = 3000 }, .{ .rows = 16, .n = 32, .k = 1500 }, .{ .rows = 128, .n = 32, .k = 700 }, .{ .rows = 8, .n = 32, .k = 4000 } };
const prompt_served = [_]c.Shape{ .{ .rows = 256, .n = 32, .k = 0 }, .{ .rows = 64, .n = 32, .k = 3000 }, .{ .rows = 100, .n = 32, .k = 500 } };

pub const cfgs = [_]Cfg{
    // LFM2.5-24B-A2B: 32 query heads over 8 KV heads of 64
    .{ .name = "cuda paged decode tf_paged_attn_chunk_segs_64 + tf_paged_attn_merge_segs_64, 32/8 heads of 64, 512-key pages", .h = 32, .hk = 8, .d = 64, .form = decode64, .served = &decode_served },
    .{ .name = "control: paged decode at head 64 reading its values from the neighbouring lane's dims", .h = 32, .hk = 8, .d = 64, .form = .{ .decode = .{ .chunk = "tf_ctl_paged_attn_chunk_segs_64", .merge = "tf_paged_attn_merge_segs_64" } }, .control = true, .served = &decode_served },
    .{ .name = "cuda paged prompt pattn_kernel<64, 8, 4, 8>, 32/8 heads of 64, 512-key pages", .h = 32, .hk = 8, .d = 64, .form = prompt64, .served = &prompt_served },
    .{ .name = "control: paged prompt rows at head 64 reading their values from the neighbouring 8 dims", .h = 32, .hk = 8, .d = 64, .form = .{ .prompt = .{ .sym = "tf_ctl_pattn_64", .rows = 32, .smem = 8 * 32 * 8 * 16 } }, .control = true, .served = &prompt_served },
    // LFM3's full-attention layers: 16 query heads over 4 KV heads of 128
    .{ .name = "cuda paged prompt pattn_kernel<128, 8, 4, 8>, 16/4 heads of 128, 512-key pages", .h = 16, .hk = 4, .d = 128, .form = .{ .prompt = .{ .sym = pattn128, .rows = 32, .smem = 8 * 32 * 16 * 16 } }, .served = &prompt128_served },
    .{ .name = "control: paged prompt rows at head 128 reading their values from the neighbouring 8 dims", .h = 16, .hk = 4, .d = 128, .form = .{ .prompt = .{ .sym = "tf_ctl_pattn_128", .rows = 32, .smem = 8 * 32 * 16 * 16 } }, .control = true, .served = &prompt128_served },
};

/// Gates 1 and 3 run attention_ref's bound: decode weights are fp32 through fast exp, prompt P enters the mma as bf16.
const decode_why = "a score's absolute rounding error is its weight's relative error: gates 1 and 3 hold outputs to attention_ref's bound";

/// The P x V chain of a row: every key it sees (positions below k, then the rows before it), a tile, then the fold.
fn keyChain(s: c.Shape) usize {
    return s.k + s.rows + 64 + 64;
}

pub fn spec(f: Cfg) c.Spec {
    return switch (f.form) {
        .decode => .{ .name = f.name, .accum = .f32, .depth = keyChain, .ops = 1, .out = .bf16, .tiles = .{ .{}, .{}, .{ .size = 64 } }, .served = f.served, .sums = .{ .why = decode_why } },
        .prompt => |p| .{ .name = f.name, .accum = .f32, .depth = keyChain, .ops = 1, .out = .bf16, .tiles = .{ .{ .size = p.rows }, .{}, .{ .size = 64, .min = 0 } }, .served = f.served, .sums = .{ .why = attn.why } },
    };
}

/// nemotron_attention.cu's AttnSegs (paged_attn.cuh): each stream's caches, position, page table and first row.
const AttnSegs = extern struct { kc: [max_segs]u64, vc: [max_segs]u64, meta: [max_segs]u64, pt: [max_segs]u64, ps: i64, row0: [max_segs]c_int, n: c_int };

/// One stream: its rows from row0 on at positions pos0.., and its pages (the first of them at `first` in `ids`).
const Seg = struct { row0: usize, rows: usize, pos0: usize, first: usize, pages: usize };

pub const Paged = struct {
    rig: rig_.Rig,
    f: Cfg,
    fns: [2]cuda.Function, // decode: chunk, merge; prompt: the kernel twice
    r: ?attn.Ref = null,
    shape: c.Shape = .{ .rows = 0, .n = 0, .k = 0 },
    segs: std.ArrayList(Seg) = .empty,
    ids: std.ArrayList(i32) = .empty, // every stream's page table, one after another
    arena: std.ArrayList(u16) = .empty, // pages of PS elements: page rows of keys, then of values, then pad
    q: std.ArrayList(u16) = .empty,
    dev: struct { q: u64 = 0, arena: u64 = 0, meta: u64 = 0, ids: u64 = 0, out: u64 = 0, xs: u64 = 0, po: u64 = 0, pm: u64 = 0, pl: u64 = 0 } = .{},
    nch: usize = 0,

    pub fn deinit(self: *Paged) void {
        const a = self.rig.gpu.gpa;
        if (self.r) |*r| r.deinit();
        inline for (.{ "segs", "ids", "arena", "q" }) |n| @field(self, n).deinit(a);
        self.rig.deinit();
    }

    pub fn kernel(self: *Paged) g.Kernel {
        return .{ .spec = spec(self.f), .ctx = self, .vt = &vt, .control = if (self.f.control) .shapes else null };
    }

    const vt = g.hooks(Paged);

    fn kvd(self: *const Paged) usize {
        return self.f.hk * self.f.d;
    }

    fn ps(self: *const Paged) usize {
        return 2 * page * self.kvd() + pad;
    }

    fn scale(self: *const Paged) f64 {
        return @floatCast(@as(f32, 1) / @sqrt(@as(f32, @floatFromInt(self.f.d))));
    }

    /// Row r's stream and position.
    fn at(self: *const Paged, r: usize) struct { seg: Seg, p: usize } {
        for (self.segs.items) |s| if (r < s.row0 + s.rows) return .{ .seg = s, .p = s.pos0 + r - s.row0 };
        unreachable;
    }

    /// Element of key `key`'s row (key head kh) in stream s's pages; values sit page * kvd further on.
    fn keyAt(self: *const Paged, s: Seg, key: usize, kh: usize) usize {
        const id: usize = @intCast(self.ids.items[s.first + key / page]);
        return id * self.ps() + (key % page) * self.kvd() + kh * self.f.d;
    }

    fn value(self: *const Paged, s: Seg, key: usize, kh: usize) usize {
        return self.keyAt(s, key, kh) + page * self.kvd();
    }

    pub fn prepare(self: *Paged, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        self.rig.reset();
        const f = self.f;
        const n = shape.rows;
        if (shape.n != f.h or n == 0) return false;
        const a = self.rig.gpu.gpa;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        self.shape = shape;
        // streams: decode rows split over up to 8 (one under stress), the first ending at k; a prompt chunk is one at k
        const nseg: usize = if (f.form == .prompt or fill == .stress) 1 else @min(n, max_segs);
        self.segs.clearRetainingCapacity();
        var row0: usize = 0;
        var pages: usize = 0;
        for (0..nseg) |j| {
            const rows = n / nseg + @intFromBool(j < n % nseg);
            const pos0 = switch (f.form) {
                .prompt => shape.k,
                .decode => blk: {
                    if (shape.k < rows) return false;
                    const room = shape.k - rows + 1;
                    break :blk if (j == 0) room - 1 else (j * 7919 + seed % 97) % room;
                },
            };
            const need = (pos0 + rows + page - 1) / page;
            try self.segs.append(a, .{ .row0 = row0, .rows = rows, .pos0 = pos0, .first = pages, .pages = need });
            row0 += rows;
            pages += need;
        }
        // page ids: every stream's pages shuffled over the arena, two spare pages left random
        const total = pages + 2;
        try self.ids.resize(a, total);
        for (self.ids.items, 0..) |*x, i| x.* = @intCast(i);
        rng.shuffle(i32, self.ids.items);
        try self.arena.resize(a, total * self.ps());
        for (self.arena.items) |*e| e.* = attn.normal(rng);
        try self.q.resize(a, n * f.h * f.d);
        for (self.q.items) |*e| e.* = attn.normal(rng);
        var most: usize = 1;
        var top: usize = 0;
        for (self.segs.items) |s| {
            most = @max(most, s.pos0 + s.rows);
            top = @max(top, s.pos0 + s.rows);
        }
        self.nch = (top + page - 1) / page + 1;
        if (self.r) |*r| r.deinit();
        self.r = null;
        self.r = try attn.Ref.init(a, f.d, most);
        switch (fill) {
            .random => {},
            .nonfinite => self.plant(),
            .stress => try self.stress(rng),
        }
        const metas = try a.alloc(i32, nseg);
        defer a.free(metas);
        for (metas, self.segs.items) |*m, s| m.* = @intCast(s.pos0);
        self.dev.q = try self.rig.input(std.mem.sliceAsBytes(self.q.items));
        self.dev.arena = try self.rig.input(std.mem.sliceAsBytes(self.arena.items));
        self.dev.meta = try self.rig.input(std.mem.sliceAsBytes(metas));
        self.dev.ids = try self.rig.input(std.mem.sliceAsBytes(self.ids.items));
        self.dev.out = try self.rig.output(n * f.h * f.d * 2);
        if (f.form == .decode) {
            self.dev.xs = try self.rig.output(n * f.h * f.d / 64 * 4);
            const parts = n * self.nch * f.h;
            self.dev.po = try self.rig.scratch(parts * f.d * 4);
            self.dev.pm = try self.rig.scratch(parts * 4);
            self.dev.pl = try self.rig.scratch(parts * 4);
        }
        return true;
    }

    /// NaN query (row 0, head 0); +inf and -inf values at row 0's key, NaN where they meet +inf a key earlier; NaN key.
    fn plant(self: *Paged) void {
        const f = self.f;
        const s = self.segs.items[0];
        const p = s.pos0; // row 0's own key: every row of its stream sees it
        self.q.items[3] = c.toBf16(std.math.nan(f32));
        self.arena.items[self.value(s, p, 0) + 7] = c.toBf16(std.math.inf(f32));
        self.arena.items[self.value(s, p, 1 % f.hk) + 9] = c.toBf16(-std.math.inf(f32));
        if (p > 0) self.arena.items[self.value(s, p - 1, 1 % f.hk) + 9] = c.toBf16(std.math.inf(f32));
        self.arena.items[self.keyAt(s, p, 2 % f.hk)] = c.toBf16(std.math.nan(f32));
    }

    /// Gate 4: every key of the stream alike (each p is 1); values mirrored over the keys row 0 sees, 0 after them.
    fn stress(self: *Paged, rng: std.Random) !void {
        const a = self.rig.gpu.gpa;
        const f = self.f;
        const s = self.segs.items[0];
        const key = try a.alloc(u16, self.kvd());
        defer a.free(key);
        for (key) |*e| e.* = attn.normal(rng);
        const last = s.pos0 + s.rows;
        for (0..last) |k| {
            @memcpy(self.arena.items[self.keyAt(s, k, 0)..][0..self.kvd()], key);
            @memset(self.arena.items[self.value(s, k, 0)..][0..self.kvd()], 0);
        }
        const col = try a.alloc(f32, s.pos0 + 1);
        defer a.free(col);
        for (0..f.hk) |kh| for (0..f.d) |e| {
            attn.mirrored(rng, col);
            for (col, 0..) |y, k| self.arena.items[self.value(s, k, kh) + e] = c.toBf16(y);
        };
    }

    pub fn run(self: *Paged) !void {
        const f = self.f;
        const n = self.shape.rows;
        const g_: usize = f.h / f.hk;
        const s = self.rig.s;
        switch (f.form) {
            .decode => {
                var sg = std.mem.zeroes(AttnSegs);
                for (self.segs.items, 0..) |x, j| {
                    sg.kc[j] = self.dev.arena;
                    sg.vc[j] = self.dev.arena + page * self.kvd() * 2;
                    sg.meta[j] = self.dev.meta + j * 4;
                    sg.pt[j] = self.dev.ids + x.first * 4;
                    sg.row0[j] = @intCast(x.row0);
                }
                sg.ps = @intCast(self.ps());
                sg.n = @intCast(self.segs.items.len);
                var a: cuda.Args = .{};
                a.add(sg);
                inline for (.{ "q", "po", "pm", "pl" }) |name| a.add(@field(self.dev, name));
                for ([_]usize{ f.h * f.d, f.h, f.hk, g_, page, self.nch }) |v| a.add(@as(c_int, @intCast(v)));
                a.add(@as(f32, @floatCast(self.scale())));
                try cuda.launch.launch(self.fns[0], .{ .grid = .{ .x = @intCast(n), .y = @intCast(f.hk), .z = @intCast(@min(self.nch, chunk_blocks)) }, .block = .{ .x = 256 } }, s, &a);
                var b: cuda.Args = .{};
                b.add(sg);
                inline for (.{ "po", "pm", "pl", "out", "xs" }) |name| b.add(@field(self.dev, name));
                for ([_]usize{ f.h, f.hk, g_, page, self.nch }) |v| b.add(@as(c_int, @intCast(v)));
                try cuda.launch.launch(self.fns[1], .{ .grid = .{ .x = @intCast(n), .y = @intCast(f.hk), .z = 1 }, .block = .{ .x = 256 } }, s, &b);
            },
            .prompt => |p| {
                const x = self.segs.items[0];
                var a: cuda.Args = .{};
                a.add(self.dev.q);
                a.add(self.dev.arena);
                a.add(self.dev.arena + page * self.kvd() * 2);
                a.add(self.dev.out);
                for ([_]usize{ x.pos0, n, f.h, f.hk, g_ }) |v| a.add(@as(c_int, @intCast(v)));
                a.add(@as(f32, @floatCast(self.scale())));
                a.add(self.dev.ids + x.first * 4);
                a.add(@as(i64, @intCast(self.ps())));
                const hpc = 16 * 8 / p.rows; // a block's query heads: 16 rows a warp, eight warps
                try cuda.launch.launch(self.fns[0], .{ .grid = .{ .x = @intCast((n + p.rows - 1) / p.rows), .y = @intCast(f.hk * (g_ / hpc)) }, .block = .{ .x = 256 }, .shared = p.smem }, s, &a);
            },
        }
        try s.synchronize();
    }

    pub fn count(self: *Paged) usize {
        return self.shape.rows * self.f.h * self.f.d;
    }

    /// Output head i (row i / h, head i % h): its stream's keys [0, p] through the page table.
    pub fn load(self: *Paged, i: usize, hd: *attn.Head) !void {
        const f = self.f;
        const d = f.d;
        const r = i / f.h;
        const head = i % f.h;
        const kh = head / (f.h / f.hk);
        const w = self.at(r);
        for (hd.q, 0..) |*x, e| x.* = c.fromBf16(self.q.items[(r * f.h + head) * d + e]);
        for (0..w.p + 1) |key| {
            const kat = self.keyAt(w.seg, key, kh);
            const vat = self.value(w.seg, key, kh);
            for (0..d) |e| {
                hd.k[key * d + e] = c.fromBf16(self.arena.items[kat + e]);
                hd.v[key * d + e] = c.fromBf16(self.arena.items[vat + e]);
            }
        }
        hd.n = w.p + 1;
    }

    fn heads(self: *Paged) usize {
        return self.shape.rows * self.f.h;
    }

    pub fn check(self: *Paged, gate: g.Gate) !?[]const u8 {
        return self.r.?.check(self, self.heads(), self.scale(), try self.rig.outputBytes(), gate);
    }

    pub fn reference(self: *Paged, want: []f64, mag: []f64) !void {
        return self.r.?.reference(self, self.heads(), self.scale(), want, mag);
    }

    pub fn simulate(self: *Paged, acc: c.Accum, order: g.Order, out: []f64) !void {
        return self.r.?.simulate(self, self.heads(), self.scale(), acc, order, out);
    }
};

/// The kernels and their controls, registered onto `kernels`; the controls from the contract_controls image `ctl`.
pub const Set = struct {
    mods: [2]?cuda.Module = .{ null, null },
    items: [cfgs.len]Paged = undefined,
    n: usize = 0,

    pub fn add(self: *Set, gpu: check.Gpu, s: cuda.Stream, ctl: cuda.Module, kernels: *std.ArrayList(g.Kernel)) !void {
        self.mods[0] = try cuda.Module.load(gpu.d, cuda.kernels.nemotron_attention);
        self.mods[1] = try cuda.Module.load(gpu.d, cuda.kernels.prefill_attention);
        const segs = self.mods[0].?;
        const prompt = self.mods[1].?;
        for (cfgs, &self.items) |f, *x| {
            const fns: [2]cuda.Function = switch (f.form) {
                .decode => |k| .{ try (if (f.control) ctl else segs).function(k.chunk), try segs.function(k.merge) },
                .prompt => |k| blk: {
                    const fn_ = try (if (f.control) ctl else prompt).function(k.sym);
                    try fn_.allowDynamicShared(k.smem);
                    break :blk .{ fn_, fn_ };
                },
            };
            x.* = .{ .rig = .{ .gpu = gpu, .s = s }, .f = f, .fns = fns };
            self.n += 1;
        }
        for (self.items[0..self.n]) |*x| try kernels.append(gpu.gpa, x.kernel());
    }

    pub fn deinit(self: *Set) void {
        for (self.items[0..self.n]) |*x| x.deinit();
        for (&self.mods) |*m| if (m.*) |*x| x.unload();
    }
};

test "the paged forms' geometry: 4 query heads a KV head, AttnSegs as nemotron_attention.cu pads it" {
    try std.testing.expectEqual(@as(usize, 304), @sizeOf(AttnSegs));
    for (cfgs) |f| try std.testing.expectEqual(@as(usize, 4), f.h / f.hk);
    try std.testing.expectEqual(@as(usize, 32), prompt64.prompt.rows);
}
