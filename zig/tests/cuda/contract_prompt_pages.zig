//! hpattn_kernel and hpattn8_kernel against float64 at the 27B's heads, their paged twins bit-equal through pages.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const attn = @import("contract_attn.zig");
const q35c = @import("contract_q35_contiguous.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const e4m3 = core.quant.fp8.e4m3;

/// The head dim, Qwen3.8-27B's 24 query heads over 4 key heads, its softmax scale.
const hd = 256;
const qh = 24;
const kvh = 4;
const group = qh / kvh;
const scale: f32 = 1.0 / 16.0;
/// Attention layers a block or page holds: the kernels read layer 1, so a page's layer offset counts.
const layers = 2;
/// cuda_prompt.zig's launch: 64 rows a block (four warp rows of 16), two query heads a block, 96 KiB of slots.
const block_rows = 64;
const hpc = 2;
const smem = 98304;
/// Gate 4's live keys a head: their mirrored sums stay below 2^24 in fp32; later keys score alike with values 0.
const stress_keys = 2048;

pub const symbols = struct {
    pub const bf16 = "_ZN20tf_prefill_attention13hpattn_kernelILi256ELi8ELi2ELi2EEEvPK13__nv_bfloat16S3_S3_PS1_iiiiifi";
    pub const fp8 = "_ZN20tf_prefill_attention14hpattn8_kernelILi256ELi8ELi2ELi2EEEvPK13__nv_bfloat16PKhS5_PKfS7_PS1_iiiiifi";
    pub const bf16_paged = "_ZN20tf_prefill_attention19hpattn_paged_kernelILi256ELi8ELi2ELi2EEEvPK13__nv_bfloat16S3_S3_PS1_iiiiifiPKix";
    pub const fp8_paged = "_ZN20tf_prefill_attention20hpattn8_paged_kernelILi256ELi8ELi2ELi2EEEvPK13__nv_bfloat16PKhS5_PKfS7_PS1_iiiiifiPKix";
};

pub const Cache = enum { bf16, fp8 };

/// A registration: its cache, its page size (0: one contiguous run of cap rows a key head), and the control's fault.
pub const Form = struct { name: []const u8, cache: Cache, ptok: usize = 0, control: bool = false };
pub const forms = [_]Form{
    .{ .name = "cuda prompt hpattn_kernel<256, 8, 2, 2>, 24/4 heads, bf16 cache", .cache = .bf16 },
    .{ .name = "cuda prompt hpattn8_kernel<256, 8, 2, 2>, 24/4 heads, e4m3 cache", .cache = .fp8 },
    .{ .name = "cuda prompt hpattn_paged_kernel<256, 8, 2, 2>, bf16 pages of 256", .cache = .bf16, .ptok = 256 },
    .{ .name = "cuda prompt hpattn_paged_kernel<256, 8, 2, 2>, bf16 pages of 48", .cache = .bf16, .ptok = 48 },
    .{ .name = "cuda prompt hpattn8_paged_kernel<256, 8, 2, 2>, e4m3 pages of 256", .cache = .fp8, .ptok = 256 },
    .{ .name = "cuda prompt hpattn8_paged_kernel<256, 8, 2, 2>, e4m3 pages of 48", .cache = .fp8, .ptok = 48 },
    .{ .name = "control: Qwen paged prompt tiles whose table's first page id is a later page's", .cache = .bf16, .ptok = 256, .control = true },
};

/// The longest chain: the 64-key tiles' folds of every key a row sees, then the products and the division.
fn keyChain(s: c.Shape) usize {
    return (s.k + s.rows) / 64 + 4;
}

/// The 27B's prompt chunks over short and long prefixes; the paged forms, compared by bytes, add 4,096-row chunks.
const served_ref = [_]c.Shape{ .{ .rows = 64, .n = qh, .k = 256 }, .{ .rows = 64, .n = qh, .k = 4096 }, .{ .rows = 512, .n = qh, .k = 0 }, .{ .rows = 129, .n = qh, .k = 1000 } };
const served_pages = served_ref ++ [_]c.Shape{ .{ .rows = 4096, .n = qh, .k = 0 }, .{ .rows = 2048, .n = qh, .k = 2048 } };

pub fn spec(f: Form) c.Spec {
    const pages = f.ptok > 0;
    const replaces: ?[]const u8 = if (!pages) null else if (f.cache == .bf16) "hpattn_kernel over the same rows in one run" else "hpattn8_kernel over the same rows in one run";
    return .{ .name = f.name, .accum = .f32, .depth = keyChain, .ops = if (f.cache == .bf16) 4 else 6, .out = .bf16, .replaces = replaces, .tiles = .{ .{ .size = block_rows }, .{}, .{ .size = 64, .min = 0 } }, .served = if (pages) &served_pages else &served_ref, .sums = .{ .why = attn.why } };
}

pub const Prompt = struct {
    rig: rig_.Rig,
    form: Form,
    /// The kernel under test, and the contiguous kernel a paged form must equal.
    f: cuda.Function,
    base: cuda.Function,
    r: ?attn.Ref = null,
    shape: c.Shape = .{ .rows = 0, .n = 0, .k = 0 },
    q: std.ArrayList(u16) = .empty,
    /// Each (key or value, key head, position)'s row: bf16 elements, or e4m3 codes and one fp32 scale.
    rows16: std.ArrayList(u16) = .empty,
    codes: std.ArrayList(u8) = .empty,
    scales: std.ArrayList(f32) = .empty,
    /// The same rows as the kernels widen them, for the float64 side.
    vals: std.ArrayList(f32) = .empty,
    host: std.ArrayList(u8) = .empty,
    ids: std.ArrayList(i32) = .empty,
    dq: u64 = 0,
    flat: u64 = 0,
    arena: u64 = 0,
    table: u64 = 0,
    out: u64 = 0,
    base_out: u64 = 0,
    cap: usize = 0,

    pub fn deinit(self: *Prompt) void {
        const a = self.rig.gpu.gpa;
        if (self.r) |*r| r.deinit();
        inline for (.{ "q", "rows16", "codes", "scales", "vals", "host", "ids" }) |x| @field(self, x).deinit(a);
        self.rig.deinit();
    }

    pub fn kernel(self: *Prompt) g.Kernel {
        return .{ .spec = spec(self.form), .ctx = self, .vt = &vt, .control = if (self.form.control) .shapes else null };
    }

    const vt = g.hooks(Prompt);

    fn positions(self: *const Prompt) usize {
        return self.shape.k + self.shape.rows;
    }

    /// Row index of (s: 0 keys, 1 values; key head h; position pos) in the logical rows.
    fn at(self: *const Prompt, s: usize, h: usize, pos: usize) usize {
        return (s * kvh + h) * self.positions() + pos;
    }

    /// One layer's bytes in a block of n rows a key head.
    fn layerBytes(self: *const Prompt, n: usize) usize {
        return switch (self.form.cache) {
            .bf16 => 2 * kvh * n * hd * 2,
            .fp8 => 2 * kvh * n * (hd + 4),
        };
    }

    /// Writes logical row (s, h, pos) at row r of layer 1 of a block of n rows a key head starting at buf.
    fn place(self: *const Prompt, buf: []u8, n: usize, s: usize, h: usize, pos: usize, r: usize) void {
        self.write(buf[self.layerBytes(n)..], n, s * kvh + h, r, self.at(s, h, pos));
    }

    fn write(self: *const Prompt, layer: []u8, n: usize, blk: usize, r: usize, src: usize) void {
        switch (self.form.cache) {
            .bf16 => @memcpy(std.mem.bytesAsSlice(u16, layer)[(blk * n + r) * hd ..][0..hd], self.rows16.items[src * hd ..][0..hd]),
            .fp8 => {
                @memcpy(layer[(blk * n + r) * hd ..][0..hd], self.codes.items[src * hd ..][0..hd]);
                std.mem.bytesAsSlice(f32, layer[2 * kvh * n * hd ..])[blk * n + r] = self.scales.items[src];
            },
        }
    }

    /// Every slot of a block of n rows a key head (both layers) holds some row: a wrong address reads finite junk.
    fn junk(self: *const Prompt, buf: []u8, n: usize, rng: std.Random) void {
        const total = 2 * kvh * self.positions();
        for (0..layers) |l| for (0..2 * kvh) |blk| for (0..n) |r| self.write(buf[l * self.layerBytes(n) ..], n, blk, r, rng.uintLessThan(usize, total));
    }

    pub fn prepare(self: *Prompt, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        self.rig.reset();
        if (shape.n != qh or shape.rows == 0) return false;
        const a = self.rig.gpu.gpa;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        self.shape = shape;
        const npos = self.positions();
        const nrows = 2 * kvh * npos;
        try self.q.resize(a, shape.rows * qh * hd);
        for (self.q.items) |*x| x.* = attn.normal(rng);
        switch (self.form.cache) {
            .bf16 => {
                try self.rows16.resize(a, nrows * hd);
                for (self.rows16.items) |*x| x.* = attn.normal(rng);
            },
            .fp8 => {
                try self.codes.resize(a, nrows * hd);
                try self.scales.resize(a, nrows);
                for (self.codes.items) |*x| {
                    const code = rng.int(u8);
                    x.* = if (code & 0x7f == 0x7f) code ^ 1 else code; // no NaN codes
                }
                for (self.scales.items) |*s| s.* = 3.0 / 448.0 * (0.5 + rng.float(f32));
            },
        }
        if (self.r) |*r| r.deinit();
        self.r = null;
        self.r = try attn.Ref.init(a, hd, npos);
        switch (fill) {
            .random => {},
            .nonfinite => self.plant(),
            .stress => try self.stress(rng),
        }
        try self.widen();
        // the contiguous run: cap rows a key head, a few past the chunk's last so cap is no position count
        self.cap = npos + 3;
        try self.host.resize(a, layers * self.layerBytes(self.cap));
        self.junk(self.host.items, self.cap, rng);
        for (0..2) |s| for (0..kvh) |h| for (0..npos) |pos| self.place(self.host.items, self.cap, s, h, pos, pos);
        self.dq = try self.rig.input(std.mem.sliceAsBytes(self.q.items));
        const pages = self.form.ptok > 0;
        // a paged form reads its pages; the contiguous run is its baseline's, outside the inputs it is held to
        self.flat = if (pages) try self.rig.scratch(self.host.items.len) else try self.rig.input(self.host.items);
        if (pages) try (cuda.DeviceBuffer{ .d = self.rig.gpu.d, .ptr = self.flat, .len = self.host.items.len }).upload(0, self.host.items);
        if (pages) try self.paginate(rng);
        self.out = try self.rig.output(shape.rows * qh * hd * 2);
        self.base_out = try self.rig.scratch(shape.rows * qh * hd * 2);
        return true;
    }

    /// The rows into shuffled pages of a two-spare arena (junk elsewhere), and the stream's table.
    fn paginate(self: *Prompt, rng: std.Random) !void {
        const a = self.rig.gpu.gpa;
        const t = self.form.ptok;
        const npos = self.positions();
        const used = (npos + t - 1) / t;
        const page = layers * self.layerBytes(t);
        try self.ids.resize(a, used + 2);
        for (self.ids.items, 0..) |*x, i| x.* = @intCast(i);
        rng.shuffle(i32, self.ids.items);
        try self.host.resize(a, (used + 2) * page);
        for (0..used + 2) |p| self.junk(self.host.items[p * page ..][0..page], t, rng);
        for (0..2) |s| for (0..kvh) |h| for (0..npos) |pos| {
            const id: usize = @intCast(self.ids.items[pos / t]);
            self.place(self.host.items[id * page ..][0..page], t, s, h, pos, pos % t);
        };
        self.arena = try self.rig.input(self.host.items);
        // the control reads a table the rows were not placed by: its first page names the second's id
        if (self.form.control and used > 1) self.ids.items[0] = self.ids.items[1];
        self.table = try self.rig.input(std.mem.sliceAsBytes(self.ids.items[0..used]));
    }

    /// The rows as the kernels stage them: bf16, or each e4m3 code times its scale in fp32, rounded to bf16.
    fn widen(self: *Prompt) !void {
        const a = self.rig.gpu.gpa;
        const n = 2 * kvh * self.positions() * hd;
        try self.vals.resize(a, n);
        switch (self.form.cache) {
            .bf16 => for (self.vals.items, self.rows16.items) |*v, x| {
                v.* = c.fromBf16(x);
            },
            .fp8 => for (self.vals.items, self.codes.items, 0..) |*v, code, i| {
                const x: f32 = @floatCast(e4m3(code));
                v.* = c.fromBf16(c.toBf16(x * self.scales.items[i / hd]));
            },
        }
    }

    fn setRow(self: *Prompt, s: usize, h: usize, pos: usize, e: usize, v: f32) void {
        const r = self.at(s, h, pos);
        switch (self.form.cache) {
            .bf16 => self.rows16.items[r * hd + e] = c.toBf16(v),
            .fp8 => self.scales.items[r] = v,
        }
    }

    /// NaN in a query; +inf in a value every row sees; +inf and -inf in one column (bf16); a NaN key every row sees.
    fn plant(self: *Prompt) void {
        const p0 = self.shape.k;
        self.q.items[3] = c.toBf16(std.math.nan(f32)); // row 0, head 0
        self.setRow(1, 0, p0, 7, std.math.inf(f32)); // fp8: the scale, so zero codes give NaN
        if (self.form.cache == .bf16 and self.positions() >= 3) {
            self.setRow(1, 1, 1, 11, std.math.inf(f32));
            self.setRow(1, 1, 2, 11, -std.math.inf(f32));
        }
        self.setRow(0, 3, 0, 0, std.math.nan(f32));
    }

    /// Gate 4: every key of a head alike (each p is 1), values mirrored over the keys row 0 sees, 0 past them.
    fn stress(self: *Prompt, rng: std.Random) !void {
        const a = self.rig.gpu.gpa;
        const npos = self.positions();
        const p = @min(self.shape.k + 1, stress_keys);
        const col = try a.alloc(f32, p);
        defer a.free(col);
        for (0..kvh) |h| switch (self.form.cache) {
            .bf16 => {
                const x = self.rows16.items;
                for (1..npos) |pos| @memcpy(x[self.at(0, h, pos) * hd ..][0..hd], x[self.at(0, h, 0) * hd ..][0..hd]);
                for (0..hd) |e| {
                    attn.mirrored(rng, col);
                    for (0..npos) |pos| x[self.at(1, h, pos) * hd + e] = if (pos < p) c.toBf16(col[pos]) else 0;
                }
            },
            .fp8 => {
                const x = self.codes.items;
                for (1..npos) |pos| {
                    @memcpy(x[self.at(0, h, pos) * hd ..][0..hd], x[self.at(0, h, 0) * hd ..][0..hd]);
                    self.scales.items[self.at(0, h, pos)] = self.scales.items[self.at(0, h, 0)];
                }
                const half = @max(1, p / 2);
                for (0..npos) |pos| self.scales.items[self.at(1, h, pos)] = if (pos < 2 * half and pos % half % 2 == 1) 1024 else 1;
                for (0..hd) |e| {
                    q35c.mirroredE4m3(rng, col);
                    for (0..npos) |pos| x[self.at(1, h, pos) * hd + e] = if (pos < p) q35c.toCode(col[pos]) else 0;
                }
            },
        };
    }

    /// One launch of `f` over layer 1 of a block of n rows a key head into `out`, through the table or not.
    fn launch(self: *Prompt, f: cuda.Function, block: u64, n: usize, paged: bool, out: u64) !void {
        const kc = block + self.layerBytes(n);
        var args: cuda.Args = .{};
        args.add(self.dq);
        switch (self.form.cache) {
            .bf16 => for ([_]u64{ kc, kc + kvh * n * hd * 2 }) |x| args.add(x),
            .fp8 => {
                const ks = kc + 2 * kvh * n * hd;
                for ([_]u64{ kc, kc + kvh * n * hd, ks, ks + kvh * n * 4 }) |x| args.add(x);
            },
        }
        args.add(out);
        for ([_]usize{ self.shape.k, self.shape.rows, qh, kvh, group }) |x| args.add(@as(c_int, @intCast(x)));
        args.add(scale);
        args.add(@as(c_int, @intCast(n)));
        if (paged) {
            args.add(self.table);
            const page = layers * self.layerBytes(n);
            args.add(@as(c_longlong, @intCast(if (self.form.cache == .bf16) page / 2 else page)));
        }
        const cfg: cuda.Config = .{ .grid = .{ .x = @intCast((self.shape.rows + block_rows - 1) / block_rows), .y = kvh * (group / hpc) }, .block = .{ .x = 256 }, .shared = smem };
        try cuda.launch.launch(f, cfg, self.rig.s, &args);
    }

    pub fn run(self: *Prompt) !void {
        if (self.form.ptok > 0) try self.launch(self.f, self.arena, self.form.ptok, true, self.out) else try self.launch(self.f, self.flat, self.cap, false, self.out);
        try self.rig.s.synchronize();
    }

    /// The contiguous kernel over the same rows in one run of cap rows a key head.
    pub fn replaced(self: *Prompt) ![]const u8 {
        try self.launch(self.base, self.flat, self.cap, false, self.base_out);
        return self.rig.bytes(self.base_out, self.count() * 2);
    }

    pub fn count(self: *Prompt) usize {
        return self.shape.rows * qh * hd;
    }

    /// Output head i (row i / qh, head i % qh): keys 0 .. p0 + row, causal.
    pub fn load(self: *Prompt, i: usize, h: *attn.Head) !void {
        const m = i / qh;
        const head = i % qh;
        const kh = head / group;
        for (h.q, 0..) |*x, e| x.* = c.fromBf16(self.q.items[(m * qh + head) * hd + e]);
        const n = self.shape.k + m + 1;
        for (0..n) |t| for (0..hd) |e| {
            h.k[t * hd + e] = self.vals.items[self.at(0, kh, t) * hd + e];
            h.v[t * hd + e] = self.vals.items[self.at(1, kh, t) * hd + e];
        };
        h.n = n;
    }

    fn heads(self: *Prompt) usize {
        return self.shape.rows * qh;
    }

    pub fn check(self: *Prompt, at_: g.Gate) !?[]const u8 {
        return self.r.?.check(self, self.heads(), scale, try self.rig.outputBytes(), at_);
    }

    pub fn reference(self: *Prompt, want: []f64, mag: []f64) !void {
        return self.r.?.reference(self, self.heads(), scale, want, mag);
    }

    pub fn simulate(self: *Prompt, acc: c.Accum, order: g.Order, out: []f64) !void {
        return self.r.?.simulate(self, self.heads(), scale, acc, order, out);
    }
};

/// The registrations over one load of prefill_attention.cu, each with its kernel and its cache's contiguous kernel.
pub const Set = struct {
    m: cuda.Module,
    prompts: [forms.len]Prompt,

    pub fn create(gpu: @import("check.zig").Gpu, s: cuda.Stream, kernels: *std.ArrayList(g.Kernel)) !*Set {
        const a = gpu.gpa;
        const self = try a.create(Set);
        errdefer a.destroy(self);
        self.m = try cuda.Module.load(gpu.d, cuda.kernels.prefill_attention);
        errdefer self.m.unload();
        const m = self.m;
        const fns = [_]cuda.Function{ try m.function(symbols.bf16), try m.function(symbols.fp8), try m.function(symbols.bf16_paged), try m.function(symbols.fp8_paged) };
        for (fns) |f| try f.allowDynamicShared(smem);
        for (forms, &self.prompts) |f, *x| {
            const base = fns[@intFromBool(f.cache == .fp8)];
            x.* = .{ .rig = .{ .gpu = gpu, .s = s }, .form = f, .f = if (f.ptok > 0) fns[2 + @as(usize, @intFromBool(f.cache == .fp8))] else base, .base = base };
        }
        for (&self.prompts) |*x| try kernels.append(a, x.kernel());
        return self;
    }

    pub fn destroy(self: *Set) void {
        const a = self.prompts[0].rig.gpu.gpa;
        for (&self.prompts) |*x| x.deinit();
        self.m.unload();
        a.destroy(self);
    }
};

test "every paged form replaces its cache's contiguous kernel; the control is paged and named as one" {
    for (forms) |f| {
        const sp = spec(f);
        try std.testing.expectEqual(f.ptok > 0, sp.replaces != null);
        if (f.control) {
            try std.testing.expect(f.ptok > 0);
            try std.testing.expect(std.mem.startsWith(u8, f.name, "control: "));
        }
        // gate 4 takes the largest prefix: at most 64 rows, so its float64 replay stays short
        var most = sp.served[0];
        for (sp.served) |x| if (x.k > most.k) {
            most = x;
        };
        try std.testing.expect(most.rows <= block_rows);
    }
}
