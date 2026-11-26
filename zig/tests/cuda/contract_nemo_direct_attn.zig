//! Direct Nemotron/Qwen chunk, prefetch and merge kernels, bit-compared with their _segs forms.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const rig_ = @import("contract_rig.zig");
const attn = @import("contract_attn.zig");

const c = core.contract.spec;
const g = core.contract.gates;
const hd = 128;
const chunk = 512;
const chunk_blocks = 64;
const page = 512;
const max_segs = 8;
const pad = 64;

const Kind = enum { chunk, prefetch, control };
const Layout = enum { nemotron, qwen16, qwen32 };
const Form = struct { name: []const u8, layout: Layout, kind: Kind };

const forms = [_]Form{
    .{ .name = "cuda direct tf_nemo_attn_chunk + tf_nemo_attn_merge, Nemotron 32/2 paged QKV rows", .layout = .nemotron, .kind = .chunk },
    .{ .name = "cuda direct tf_nemo_attn_chunk_pf + tf_nemo_attn_merge, Nemotron 32/2 paged QKV rows", .layout = .nemotron, .kind = .prefetch },
    .{ .name = "control: directABI tf_ctl_nemo_attn_chunk_segs values from the neighbouring lane, Nemotron 32/2", .layout = .nemotron, .kind = .control },
    .{ .name = "cuda direct tf_nemo_attn_chunk + tf_nemo_attn_merge, Qwen3 16/8 contiguous query rows", .layout = .qwen16, .kind = .chunk },
    .{ .name = "cuda direct tf_nemo_attn_chunk_pf + tf_nemo_attn_merge, Qwen3 16/8 contiguous query rows", .layout = .qwen16, .kind = .prefetch },
    .{ .name = "control: directABI tf_ctl_nemo_attn_chunk_segs values from the neighbouring lane, Qwen3 16/8", .layout = .qwen16, .kind = .control },
    .{ .name = "cuda direct tf_nemo_attn_chunk + tf_nemo_attn_merge, Qwen3 32/8 contiguous query rows", .layout = .qwen32, .kind = .chunk },
    .{ .name = "cuda direct tf_nemo_attn_chunk_pf + tf_nemo_attn_merge, Qwen3 32/8 contiguous query rows", .layout = .qwen32, .kind = .prefetch },
    .{ .name = "control: directABI tf_ctl_nemo_attn_chunk_segs values from the neighbouring lane, Qwen3 32/8", .layout = .qwen32, .kind = .control },
};

/// Nemotron's direct decode calls, including the tail, exact-chunk and >64-chunk-block classes.
const nemo_served = [_]c.Shape{
    .{ .rows = 1, .n = 32, .k = 0 },
    .{ .rows = 1, .n = 32, .k = 3000 },
    .{ .rows = 16, .n = 32, .k = 1500 },
    .{ .rows = 128, .n = 32, .k = 700 },
    .{ .rows = 8, .n = 32, .k = 4000 },
    .{ .rows = 2, .n = 32, .k = 511 },
    .{ .rows = 2, .n = 32, .k = 32768 },
};

/// Qwen3's one-row and row-block calls, including whole prompt blocks and the grid.z chunk wrap.
fn qwenServed(comptime h: usize) [7]c.Shape {
    return .{
        .{ .rows = 1, .n = h, .k = 0 },
        .{ .rows = 1, .n = h, .k = 3000 },
        .{ .rows = 256, .n = h, .k = 0 },
        .{ .rows = 64, .n = h, .k = 3000 },
        .{ .rows = 100, .n = h, .k = 500 },
        .{ .rows = 512, .n = h, .k = 0 },
        .{ .rows = 2, .n = h, .k = 32768 },
    };
}
const qwen16_served = qwenServed(16);
const qwen32_served = qwenServed(32);

fn noDepth(_: c.Shape) usize {
    return 0;
}

fn served(layout: Layout) []const c.Shape {
    return switch (layout) {
        .nemotron => &nemo_served,
        .qwen16 => &qwen16_served,
        .qwen32 => &qwen32_served,
    };
}

fn heads(layout: Layout) usize {
    return if (layout == .qwen16) 16 else 32;
}

fn kvHeads(layout: Layout) usize {
    return if (layout == .nemotron) 2 else 8;
}

fn spec(f: Form) c.Spec {
    return .{
        .name = f.name,
        .accum = .none,
        .depth = noDepth,
        .out = .raw,
        .replaces = "tf_nemo_attn_chunk_segs + tf_nemo_attn_merge_segs",
        .nan = .{ .stated = "PO, PM, PL, OUT and XS are byte-identical to the one-segment kernels, including non-finite inputs and untouched tail chunks" },
        .tiles = .{ .{}, .{}, .{ .size = 64, .step = 1, .min = 0 } },
        .served = served(f.layout),
    };
}

/// paged_attn.cuh's AttnSegs for the single stream that defines a direct launch.
const AttnSegs = extern struct {
    kc: [max_segs]u64,
    vc: [max_segs]u64,
    meta: [max_segs]u64,
    pt: [max_segs]u64,
    ps: i64,
    row0: [max_segs]c_int,
    n: c_int,
};

const Dev = struct {
    q: u64 = 0,
    cache: u64 = 0,
    meta: u64 = 0,
    ids: u64 = 0,
    po: u64 = 0,
    pm: u64 = 0,
    pl: u64 = 0,
    out: u64 = 0,
    xs: u64 = 0,
    bpo: u64 = 0,
    bpm: u64 = 0,
    bpl: u64 = 0,
    bout: u64 = 0,
    bxs: u64 = 0,
};

pub const Entry = struct {
    rig: rig_.Rig,
    form: Form,
    f: cuda.Function,
    merge: cuda.Function,
    seg_chunk: cuda.Function,
    seg_merge: cuda.Function,
    shape: c.Shape = .{ .rows = 0, .n = 0, .k = 0 },
    nch: usize = 0,
    nqkv: usize = 0,
    vc_offset: usize = 0,
    ps: usize = 0,
    pages: usize = 0,
    q: std.ArrayList(u16) = .empty,
    cache: std.ArrayList(u16) = .empty,
    ids: std.ArrayList(i32) = .empty,
    base: std.ArrayList(u8) = .empty,
    dev: Dev = .{},
    msg: [192]u8 = undefined,

    pub fn kernel(self: *Entry) g.Kernel {
        return .{
            .spec = spec(self.form),
            .ctx = self,
            .vt = &vt,
            .control = if (self.form.kind == .control) .shapes else null,
        };
    }
    const vt = g.hooks(Entry);

    pub fn deinit(self: *Entry) void {
        const a = self.rig.gpu.gpa;
        inline for (.{ "q", "cache", "ids", "base" }) |name| @field(self, name).deinit(a);
        self.rig.deinit();
    }

    fn h(self: *const Entry) usize {
        return heads(self.form.layout);
    }
    fn hk(self: *const Entry) usize {
        return kvHeads(self.form.layout);
    }
    fn kvd(self: *const Entry) usize {
        return self.hk() * hd;
    }
    fn total(self: *const Entry) usize {
        return self.shape.k + self.shape.rows;
    }
    fn paged(self: *const Entry) bool {
        return self.form.layout == .nemotron;
    }
    fn parts(self: *const Entry) usize {
        return self.shape.rows * self.nch * self.h();
    }
    fn poBytes(self: *const Entry) usize {
        return self.parts() * hd * 4;
    }
    fn pmBytes(self: *const Entry) usize {
        return self.parts() * 4;
    }
    fn outBytes(self: *const Entry) usize {
        return self.shape.rows * self.h() * hd * 2;
    }
    fn xsBytes(self: *const Entry) usize {
        return self.shape.rows * self.h() * (hd / 64) * 4;
    }
    pub fn count(self: *const Entry) usize {
        return self.poBytes() + 2 * self.pmBytes() + self.outBytes() + self.xsBytes();
    }
    fn scale(_: *const Entry) f32 {
        return @floatCast(std.math.pow(f64, hd, -0.5));
    }

    /// Element offset of a logical key, exactly as direct chunk_at/key_at and the _segs reference see it.
    fn keyAt(self: *const Entry, key: usize, kh: usize) usize {
        if (!self.paged()) return key * self.kvd() + kh * hd;
        const id: usize = @intCast(self.ids.items[key / page]);
        return id * self.ps + (key % page) * self.kvd() + kh * hd;
    }
    fn valueAt(self: *const Entry, key: usize, kh: usize) usize {
        return self.keyAt(key, kh) + self.vc_offset;
    }
    fn query(self: *Entry, row: usize, head: usize) []u16 {
        return self.q.items[row * self.nqkv + head * hd ..][0..hd];
    }

    pub fn prepare(self: *Entry, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n != self.h()) return false;
        self.rig.reset();
        self.base.clearRetainingCapacity();
        self.shape = shape;
        const a = self.rig.gpu.gpa;
        const nq = self.h() * hd;
        self.nqkv = if (self.paged()) (self.h() + 2 * self.hk()) * hd else nq;
        self.nch = (self.total() + chunk - 1) / chunk + 1;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        try self.q.resize(a, shape.rows * self.nqkv);
        for (self.q.items) |*x| x.* = attn.normal(rng);
        self.vc_offset = if (self.paged()) page * self.kvd() else (self.total() + pad) * self.kvd();
        self.ps = if (self.paged()) 2 * page * self.kvd() + pad else 0;
        self.pages = (self.total() + pad + page - 1) / page + 2;
        const cap = if (self.paged()) self.pages * page else self.total() + pad;
        try self.cache.resize(a, if (self.paged()) self.pages * self.ps else 2 * cap * self.kvd());
        for (self.cache.items) |*x| x.* = attn.normal(rng);
        try self.ids.resize(a, self.pages);
        for (self.ids.items, 0..) |*x, i| x.* = @intCast(i);
        rng.shuffle(i32, self.ids.items);
        // A poisoned, unallocated logical tail proves that causal/tile masks never read it.
        for (self.total()..cap) |key| for (0..self.hk()) |kh| {
            @memset(self.cache.items[self.keyAt(key, kh)..][0..hd], 0xff);
            @memset(self.cache.items[self.valueAt(key, kh)..][0..hd], 0xff);
        };
        if (fill == .random and self.form.kind != .control) self.adversarial();
        switch (fill) {
            .random => {},
            .nonfinite => self.plant(),
            .stress => try self.stress(rng),
        }
        const meta = [1]i32{@intCast(shape.k)};
        self.dev.q = try self.rig.input(std.mem.sliceAsBytes(self.q.items));
        self.dev.cache = try self.rig.input(std.mem.sliceAsBytes(self.cache.items));
        self.dev.meta = try self.rig.input(std.mem.sliceAsBytes(&meta));
        self.dev.ids = if (self.paged()) try self.rig.input(std.mem.sliceAsBytes(self.ids.items)) else 0;
        self.dev.po = try self.rig.output(self.poBytes());
        self.dev.pm = try self.rig.output(self.pmBytes());
        self.dev.pl = try self.rig.output(self.pmBytes());
        self.dev.out = try self.rig.output(self.outBytes());
        self.dev.xs = try self.rig.output(self.xsBytes());
        inline for (.{ "bpo", "bpm", "bpl", "bout", "bxs" }, .{ self.poBytes(), self.pmBytes(), self.pmBytes(), self.outBytes(), self.xsBytes() }) |name, len| {
            const ptr = try self.rig.scratch(len);
            @field(self.dev, name) = ptr;
            try (cuda.DeviceBuffer{ .d = self.rig.gpu.d, .ptr = ptr, .len = len }).fill8(0xff, self.rig.s.handle);
        }
        return true;
    }

    /// NaN in one query and key, both infinities in values at row 0's own (universally visible) key.
    fn plant(self: *Entry) void {
        const at = self.shape.k;
        self.query(0, 0)[3] = c.toBf16(std.math.nan(f32));
        self.cache.items[self.valueAt(at, 0) + 7] = c.toBf16(std.math.inf(f32));
        self.cache.items[self.valueAt(at, 1 % self.hk()) + 9] = c.toBf16(-std.math.inf(f32));
        if (at > 0) self.cache.items[self.valueAt(at - 1, 1 % self.hk()) + 9] = c.toBf16(std.math.inf(f32));
        self.cache.items[self.keyAt(at, 2 % self.hk())] = c.toBf16(std.math.nan(f32));
    }

    fn spot(self: *Entry, key: usize, score: f32, index: usize) void {
        const k = self.keyAt(key, 0);
        self.cache.items[k] = c.toBf16(score);
        self.cache.items[k + 1] = if (index == 0) 0x8000 else 0;
        self.cache.items[k + 2] = if (index == 0) 0x0001 else 0x8001;
        self.cache.items[k + 3] = if (index == 0) 0x807f else 0x007f;
        const v = self.valueAt(key, 0);
        self.cache.items[v] = c.toBf16(1 + 2 * @as(f32, @floatFromInt(index)));
        self.cache.items[v + 1] = if (index == 0) 0x8000 else 0;
        self.cache.items[v + 2] = if (index == 0) 0x0001 else 0x8001;
        self.cache.items[v + 3] = if (index == 0) 0x8001 else 0x807f;
        self.cache.items[v + 4] = if (index == 0) 0x007f else 0x807f;
    }

    /// Tied tile/chunk maxima followed by a changed maximum, plus signed zero and bf16 subnormals.
    fn adversarial(self: *Entry) void {
        const last = self.shape.rows - 1;
        const end = self.total() - 1;
        const q = self.query(last, 0);
        @memset(q, 0);
        q[0] = c.toBf16(1);
        q[1] = 0x8000;
        q[2] = 0x0001;
        q[3] = 0x8001;
        q[4] = 0x007f;
        q[5] = 0x807f;
        for (0..end + 1) |key| @memset(self.cache.items[self.keyAt(key, 0)..][0..hd], 0);
        self.spot(0, 16, 0);
        if (end >= 63) self.spot(63, 16, 1);
        if (end >= 64) self.spot(64, 16, 2);
        if (end >= 511) self.spot(511, 16, 3);
        if (end >= 512) self.spot(512, 32, 4);
        if (end >= 576) self.spot(576, 32, 5);
        if (end > 0 and end < 512) self.spot(end, 32, 6);
    }

    /// Every key scores alike; mirrored values exercise numerator and mass sums in all tiles.
    fn stress(self: *Entry, rng: std.Random) !void {
        const a = self.rig.gpu.gpa;
        const same = try a.alloc(u16, self.kvd());
        defer a.free(same);
        for (same) |*x| x.* = attn.normal(rng);
        const col = try a.alloc(f32, self.total());
        defer a.free(col);
        for (0..self.total()) |key| {
            @memcpy(self.cache.items[self.keyAt(key, 0)..][0..self.kvd()], same);
            @memset(self.cache.items[self.valueAt(key, 0)..][0..self.kvd()], 0);
        }
        for (0..self.hk()) |kh| for (0..hd) |d| {
            attn.mirrored(rng, col);
            for (col, 0..) |x, key| self.cache.items[self.valueAt(key, kh) + d] = c.toBf16(x);
        };
    }

    fn segs(self: *const Entry) AttnSegs {
        var sg = std.mem.zeroes(AttnSegs);
        sg.kc[0] = self.dev.cache;
        sg.vc[0] = self.dev.cache + self.vc_offset * 2;
        sg.meta[0] = self.dev.meta;
        sg.pt[0] = self.dev.ids;
        sg.ps = @intCast(self.ps);
        sg.n = 1;
        return sg;
    }

    fn launchChunk(self: *Entry, f: cuda.Function, segmented: bool, po: u64, pm: u64, pl: u64) !void {
        var args: cuda.Args = .{};
        if (segmented) args.add(self.segs());
        args.add(self.dev.q);
        if (!segmented) {
            args.add(self.dev.cache);
            args.add(self.dev.cache + self.vc_offset * 2);
            args.add(self.dev.meta);
        }
        inline for (.{ po, pm, pl }) |p| args.add(p);
        for ([_]usize{ self.nqkv, self.h(), self.hk(), self.h() / self.hk(), chunk, self.nch }) |x| args.add(@as(c_int, @intCast(x)));
        args.add(self.scale());
        if (!segmented) {
            args.add(self.dev.ids);
            args.add(@as(i64, @intCast(self.ps)));
        }
        try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(self.shape.rows), .y = @intCast(self.hk()), .z = @intCast(@min(self.nch, chunk_blocks)) }, .block = .{ .x = 256 } }, self.rig.s, &args);
    }

    fn launchMerge(self: *Entry, f: cuda.Function, segmented: bool, po: u64, pm: u64, pl: u64, out: u64, xs: u64) !void {
        var args: cuda.Args = .{};
        if (segmented) args.add(self.segs());
        inline for (.{ po, pm, pl }) |p| args.add(p);
        if (!segmented) args.add(self.dev.meta);
        args.add(out);
        args.add(xs);
        for ([_]usize{ self.h(), self.hk(), self.h() / self.hk(), chunk, self.nch }) |x| args.add(@as(c_int, @intCast(x)));
        try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(self.shape.rows), .y = @intCast(self.hk()), .z = 1 }, .block = .{ .x = 256 } }, self.rig.s, &args);
    }

    pub fn run(self: *Entry) !void {
        try self.launchChunk(self.f, self.form.kind == .control, self.dev.po, self.dev.pm, self.dev.pl);
        try self.launchMerge(self.merge, false, self.dev.po, self.dev.pm, self.dev.pl, self.dev.out, self.dev.xs);
        try self.rig.s.synchronize();
    }

    /// The exact one-segment kernels over the same pages, with untouched chunks poisoned identically.
    pub fn replaced(self: *Entry) ![]const u8 {
        try self.launchChunk(self.seg_chunk, true, self.dev.bpo, self.dev.bpm, self.dev.bpl);
        try self.launchMerge(self.seg_merge, true, self.dev.bpo, self.dev.bpm, self.dev.bpl, self.dev.bout, self.dev.bxs);
        try self.rig.s.synchronize();
        const a = self.rig.gpu.gpa;
        self.base.clearRetainingCapacity();
        inline for (.{ "bpo", "bpm", "bpl", "bout", "bxs" }, .{ self.poBytes(), self.pmBytes(), self.pmBytes(), self.outBytes(), self.xsBytes() }) |name, len| {
            const old = self.base.items.len;
            try self.base.resize(a, old + len);
            const bytes = try self.rig.bytes(@field(self.dev, name), len);
            @memcpy(self.base.items[old..], bytes);
        }
        return self.base.items;
    }

    pub fn check(self: *Entry, at: g.Gate) !?[]const u8 {
        const a = self.rig.gpu.gpa;
        const got = try a.dupe(u8, try self.rig.outputBytes());
        defer a.free(got);
        const want = try self.replaced();
        if (std.mem.indexOfDiff(u8, got, want)) |i| return try std.fmt.bufPrint(&self.msg, "byte {d} of {d} differs from the one-segment kernels", .{ i, got.len });
        if (at != .nonfinite) return null;
        const first = self.poBytes() + 2 * self.pmBytes();
        const out = got[first..][0..self.outBytes()];
        var finite: usize = 0;
        var planted: usize = 0;
        for (0..out.len / 2) |i| switch (c.classOf(c.read(.bf16, out, i))) {
            .finite => finite += 1,
            else => planted += 1,
        };
        if (finite == 0 or planted == 0) return try std.fmt.bufPrint(&self.msg, "the plant gives {d} finite and {d} non-finite final outputs: no mixed proof", .{ finite, planted });
        return null;
    }

    pub fn reference(_: *Entry, _: []f64, _: []f64) !void {
        return error.ReplacementUsesOneSegmentBytes;
    }
};

const controls = struct {
    pub const bytes align(16) = @embedFile("fatbin_contract_controls").*;
};

/// Heap-owned registrations and both images, stable for the gate and raw-output runs.
pub const Group = struct {
    a: std.mem.Allocator,
    mod: ?cuda.Module = null,
    ctl: ?cuda.Module = null,
    items: [forms.len]Entry = undefined,
    made: usize = 0,

    pub fn create(a: std.mem.Allocator, gpu: check.Gpu, s: cuda.Stream, kernels: *std.ArrayList(g.Kernel)) !*Group {
        const self = try a.create(Group);
        self.* = .{ .a = a };
        errdefer self.destroy();
        self.mod = try cuda.Module.load(gpu.d, cuda.kernels.nemotron_attention);
        self.ctl = try cuda.Module.load(gpu.d, &controls.bytes);
        const direct = try self.mod.?.function("tf_nemo_attn_chunk");
        const prefetch = try self.mod.?.function("tf_nemo_attn_chunk_pf");
        const merge = try self.mod.?.function("tf_nemo_attn_merge");
        const seg_chunk = try self.mod.?.function("tf_nemo_attn_chunk_segs");
        const seg_merge = try self.mod.?.function("tf_nemo_attn_merge_segs");
        const planted = try self.ctl.?.function("tf_ctl_nemo_attn_chunk_segs");
        for (&self.items, forms) |*entry, form| {
            entry.* = .{
                .rig = .{ .gpu = gpu, .s = s },
                .form = form,
                .f = switch (form.kind) {
                    .chunk => direct,
                    .prefetch => prefetch,
                    .control => planted,
                },
                .merge = merge,
                .seg_chunk = seg_chunk,
                .seg_merge = seg_merge,
            };
            self.made += 1;
            try kernels.append(a, entry.kernel());
        }
        return self;
    }

    pub fn destroy(self: *Group) void {
        for (self.items[0..self.made]) |*entry| entry.deinit();
        if (self.ctl) |mod| mod.unload();
        if (self.mod) |mod| mod.unload();
        self.a.destroy(self);
    }
};
