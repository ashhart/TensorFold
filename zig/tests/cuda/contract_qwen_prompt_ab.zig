//! Direct raw-output fixtures for every 256-wide HPATTN prompt kernel and its serving replacement.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const attn = @import("contract_attn.zig");
const check = @import("check.zig");

const c = core.contract.spec;
const g = core.contract.gates;
const e4m3 = core.quant.fp8.e4m3;

const D = 256;
const qh = 24;
const kvh = 4;
const group = qh / kvh;
const smem: u32 = 98304;
const scale: f32 = 1.0 / 16.0;
const page_ids = 768;

const Kind = enum { paged_bf16, paged_fp8, p256, p256_ids, whole_bf16, whole_fp8 };
const Form = struct { kind: Kind, control: bool = false };
const forms = [_]Form{
    .{ .kind = .paged_bf16 }, .{ .kind = .paged_bf16, .control = true },
    .{ .kind = .paged_fp8 },  .{ .kind = .paged_fp8, .control = true },
    .{ .kind = .p256 },       .{ .kind = .p256, .control = true },
    .{ .kind = .p256_ids },   .{ .kind = .p256_ids, .control = true },
    .{ .kind = .whole_bf16 }, .{ .kind = .whole_bf16, .control = true },
    .{ .kind = .whole_fp8 },  .{ .kind = .whole_fp8, .control = true },
};

/// Cold chunks, causal/page seams, a second 64-row CTA, 256-page crossings, and the served long-prefix class.
const served = [_]c.Shape{
    .{ .rows = 1, .n = qh, .k = 0 },
    .{ .rows = 6, .n = qh, .k = 47 },
    .{ .rows = 65, .n = qh, .k = 63 },
    .{ .rows = 3, .n = qh, .k = 257 },
    .{ .rows = 16, .n = qh, .k = 4095 },
};

fn isFp8(kind: Kind) bool {
    return kind == .paged_fp8 or kind == .whole_fp8;
}
fn isWhole(kind: Kind) bool {
    return kind == .whole_bf16 or kind == .whole_fp8;
}
fn isP256(kind: Kind) bool {
    return kind == .p256 or kind == .p256_ids;
}
fn ptok(kind: Kind) usize {
    return if (isP256(kind)) 256 else 48;
}

fn symbol(kind: Kind) [:0]const u8 {
    return switch (kind) {
        .paged_bf16 => "_ZN20tf_prefill_attention13hpattn_kernelILi256ELi8ELi2ELi2EEEvPK13__nv_bfloat16S3_S3_PS1_iiiiifiPKixi",
        .paged_fp8 => "_ZN20tf_prefill_attention14hpattn8_kernelILi256ELi8ELi2ELi2EEEvPK13__nv_bfloat16PKhS5_PKfS7_PS1_iiiiifiPKixi",
        .p256 => "_ZN20tf_prefill_attention18hpattn_p256_kernelILi256ELi8ELi2ELi2EEEvPK13__nv_bfloat16S3_S3_PS1_iiiiifiPKixi",
        .p256_ids => "_ZN20tf_prefill_attention22hpattn_p256_ids_kernelILi256ELi8ELi2ELi2EEEvPK13__nv_bfloat16S3_S3_PS1_iiiiifiPKixi",
        .whole_bf16 => "_ZN20tf_prefill_attention19hpattn_whole_kernelILi256ELi8ELi2ELi2EEEvPK13__nv_bfloat16S3_S3_PS1_iiiiifi",
        .whole_fp8 => "_ZN20tf_prefill_attention20hpattn8_whole_kernelILi256ELi8ELi2ELi2EEEvPK13__nv_bfloat16PKhS5_PKfS7_PS1_iiiiifi",
    };
}

fn baseSymbol(kind: Kind) [:0]const u8 {
    return switch (kind) {
        .p256_ids => symbol(.p256),
        .paged_fp8, .whole_fp8 => symbol(.paged_fp8),
        else => symbol(.paged_bf16),
    };
}

fn name(f: Form) []const u8 {
    if (f.control) return switch (f.kind) {
        .paged_bf16 => "control: hpattn_kernel<256> doubling its score scale",
        .paged_fp8 => "control: hpattn8_kernel<256> doubling its score scale",
        .p256 => "control: hpattn_p256_kernel<256> doubling its score scale",
        .p256_ids => "control: hpattn_p256_ids_kernel<256> doubling its score scale",
        .whole_bf16 => "control: hpattn_whole_kernel<256> doubling its score scale",
        .whole_fp8 => "control: hpattn8_whole_kernel<256> doubling its score scale",
    };
    return switch (f.kind) {
        .paged_bf16 => "cuda qwen prompt hpattn_kernel<256, 8, 2, 2>, 48-position pages",
        .paged_fp8 => "cuda qwen prompt hpattn8_kernel<256, 8, 2, 2>, 48-position pages",
        .p256 => "cuda qwen prompt hpattn_p256_kernel<256, 8, 2, 2>",
        .p256_ids => "cuda qwen prompt hpattn_p256_ids_kernel<256, 8, 2, 2>",
        .whole_bf16 => "cuda qwen prompt hpattn_whole_kernel<256, 8, 2, 2>",
        .whole_fp8 => "cuda qwen prompt hpattn8_whole_kernel<256, 8, 2, 2>",
    };
}

fn keyChain(s: c.Shape) usize {
    return s.k + s.rows + 128;
}

fn spec(f: Form) c.Spec {
    const replacement: ?[]const u8 = if (f.control) null else switch (f.kind) {
        .p256 => "hpattn_kernel over the same 256-position pages",
        .p256_ids => "hpattn_p256_kernel reading PT for every tile",
        .whole_bf16, .whole_fp8 => "the page-aware entry over the identical contiguous cache",
        else => null,
    };
    return .{
        .name = name(f),
        .accum = .f32,
        .depth = keyChain,
        .ops = if (isFp8(f.kind)) 6 else 4,
        .out = .bf16,
        .replaces = replacement,
        .tiles = .{ .{ .size = 64 }, .{}, .{ .size = 64 } },
        .served = &served,
        .sums = .{ .why = attn.why },
    };
}

const Prompt = struct {
    rig: rig_.Rig,
    form: Form,
    f: cuda.Function,
    base: cuda.Function,
    r: ?attn.Ref = null,
    shape: c.Shape = undefined,
    cap: usize = 0,
    pages: usize = 0,
    q: std.ArrayList(u16) = .empty,
    cache: std.ArrayList(u8) = .empty,
    ids: std.ArrayList(i32) = .empty,
    dev_q: u64 = 0,
    dev_cache: u64 = 0,
    dev_ids: u64 = 0,
    out: u64 = 0,
    alt: u64 = 0,

    fn kernel(self: *Prompt) g.Kernel {
        return .{ .spec = spec(self.form), .ctx = self, .vt = &vt, .control = if (self.form.control) .shapes else null };
    }
    const vt = g.hooks(Prompt);

    fn deinit(self: *Prompt) void {
        const a = self.rig.gpu.gpa;
        if (self.r) |*r| r.deinit();
        self.q.deinit(a);
        self.cache.deinit(a);
        self.ids.deinit(a);
        self.rig.deinit();
    }

    fn width(self: *const Prompt) usize {
        return if (isWhole(self.form.kind)) self.cap else ptok(self.form.kind);
    }

    fn pageStride(self: *const Prompt) usize {
        return 2 * kvh * self.width() * @as(usize, if (isFp8(self.form.kind)) D + 4 else 2 * D);
    }

    fn locate(self: *const Prompt, pos: usize) struct { page: usize, row: usize } {
        if (isWhole(self.form.kind)) return .{ .page = 0, .row = pos };
        return .{ .page = @intCast(self.ids.items[pos / self.width()]), .row = pos % self.width() };
    }

    fn offset(self: *const Prompt, value: bool, head: usize, pos: usize, e: usize) usize {
        const at = self.locate(pos);
        const plane = (@as(usize, @intFromBool(value)) * kvh + head) * self.width() + at.row;
        const unit: usize = if (isFp8(self.form.kind)) 1 else 2;
        return at.page * self.pageStride() + plane * D * unit + e * unit;
    }

    fn scaleOffset(self: *const Prompt, value: bool, head: usize, pos: usize) usize {
        const at = self.locate(pos);
        const plane = (@as(usize, @intFromBool(value)) * kvh + head) * self.width() + at.row;
        return at.page * self.pageStride() + 2 * kvh * self.width() * D + 4 * plane;
    }

    fn writeScale(self: *Prompt, value: bool, head: usize, pos: usize, x: f32) void {
        std.mem.writeInt(u32, self.cache.items[self.scaleOffset(value, head, pos)..][0..4], @bitCast(x), .little);
    }

    fn readScale(self: *const Prompt, value: bool, head: usize, pos: usize) f32 {
        return @bitCast(std.mem.readInt(u32, self.cache.items[self.scaleOffset(value, head, pos)..][0..4], .little));
    }

    /// One exact bf16 element, or one e4m3 element and its row's fp32 scale.
    fn put(self: *Prompt, value: bool, head: usize, pos: usize, e: usize, bf: u16, code: u8, sc: f32) void {
        const at = self.offset(value, head, pos, e);
        if (isFp8(self.form.kind)) {
            self.cache.items[at] = code;
            self.writeScale(value, head, pos, sc);
        } else std.mem.writeInt(u16, self.cache.items[at..][0..2], bf, .little);
    }

    fn get(self: *const Prompt, value: bool, head: usize, pos: usize, e: usize) f64 {
        const at = self.offset(value, head, pos, e);
        if (isFp8(self.form.kind)) return e4m3(self.cache.items[at]) * self.readScale(value, head, pos);
        return c.fromBf16(std.mem.readInt(u16, self.cache.items[at..][0..2], .little));
    }

    fn prepare(self: *Prompt, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        self.rig.reset();
        if (shape.n != qh or shape.rows == 0) return false;
        const a = self.rig.gpu.gpa;
        self.shape = shape;
        const span = shape.k + shape.rows;
        const whole = isWhole(self.form.kind);
        const page_width = ptok(self.form.kind);
        self.cap = if (whole) span else (span + page_width - 1) / page_width * page_width;
        const used = if (whole) 1 else self.cap / page_width;
        self.pages = used + @as(usize, @intFromBool(!whole)) * 2;
        try self.ids.resize(a, if (whole) 0 else used);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        if (!whole) {
            const order = try a.alloc(i32, self.pages);
            defer a.free(order);
            for (order, 0..) |*x, i| x.* = @intCast(i);
            rng.shuffle(i32, order);
            @memcpy(self.ids.items, order[0..used]);
        }
        try self.cache.resize(a, self.pages * self.pageStride());
        try self.q.resize(a, shape.rows * qh * D);
        for (self.q.items) |*x| x.* = attn.normal(rng);
        self.fillRandom(rng);
        if (self.r) |*r| r.deinit();
        self.r = null;
        self.r = try attn.Ref.init(a, D, span);
        if (fill == .random) self.plantAdversarial();
        switch (fill) {
            .random => {},
            .nonfinite => self.plantNonfinite(),
            .stress => try self.stress(rng),
        }
        self.dev_q = try self.rig.input(std.mem.sliceAsBytes(self.q.items));
        self.dev_cache = try self.rig.input(self.cache.items);
        self.dev_ids = if (whole) 0 else try self.rig.input(std.mem.sliceAsBytes(self.ids.items));
        self.out = try self.rig.output(self.count() * 2);
        self.alt = try self.rig.scratch(self.count() * 2);
        return true;
    }

    fn fillRandom(self: *Prompt, rng: std.Random) void {
        for (0..self.pages) |page| for (0..2) |v| for (0..kvh) |h| for (0..self.width()) |row| {
            // Spare pages use the same layout but need no page-table entry; address them directly here.
            const plane = (v * kvh + h) * self.width() + row;
            const unit: usize = if (isFp8(self.form.kind)) 1 else 2;
            const off = page * self.pageStride() + plane * D * unit;
            if (isFp8(self.form.kind)) {
                for (0..D) |e| {
                    const code = rng.int(u8);
                    self.cache.items[off + e] = if (code & 0x7f == 0x7f) code ^ 1 else code;
                }
                const so = page * self.pageStride() + 2 * kvh * self.width() * D + 4 * plane;
                const sc = 3.0 / 448.0 * (0.5 + rng.float(f32));
                std.mem.writeInt(u32, self.cache.items[so..][0..4], @bitCast(sc), .little);
            } else for (0..D) |e| std.mem.writeInt(u16, self.cache.items[off + 2 * e ..][0..2], attn.normal(rng), .little);
        };
    }

    fn score(self: *Prompt, pos: usize, i: usize, changed: bool) void {
        self.put(false, 0, pos, 0, c.toBf16(if (changed) 32 else 16), if (changed) toCode(32) else toCode(16), 1);
        self.put(false, 0, pos, 1, if (i == 0) 0x8000 else 0, if (i == 0) 0x80 else 0, 1);
        self.put(false, 0, pos, 2, if (i == 0) 0x0001 else 0x8001, if (i == 0) 0x01 else 0x81, 1);
        self.put(false, 0, pos, 3, if (i == 0) 0x807f else 0x007f, if (i == 0) 0x87 else 0x07, 1);
        self.put(true, 0, pos, 0, c.toBf16(if (changed) 5 else 1 + 2 * @as(f32, @floatFromInt(i))), if (changed) toCode(5) else toCode(1 + 2 * @as(f32, @floatFromInt(i))), 1);
        self.put(true, 0, pos, 1, if (i == 0) 0x8000 else 0, if (i == 0) 0x80 else 0, 1);
        self.put(true, 0, pos, 2, if (i == 0) 0x0001 else 0x8001, if (i == 0) 0x01 else 0x81, 1);
        self.put(true, 0, pos, 3, if (i == 0) 0x8001 else 0x007f, if (i == 0) 0x81 else 0x07, 1);
        self.put(true, 0, pos, 4, if (i == 0) 0x007f else 0x807f, if (i == 0) 0x07 else 0x87, 1);
    }

    /// Stable maxima cross key-tile and physical-page boundaries before a later maximum changes.
    fn plantAdversarial(self: *Prompt) void {
        const q = self.q.items[0..D];
        @memset(q, 0);
        q[0] = c.toBf16(1);
        q[1] = 0x8000;
        q[2] = 0x0001;
        q[3] = 0x8001;
        q[4] = 0x007f;
        q[5] = 0x807f;
        if (self.shape.rows > 1) {
            const other = self.q.items[qh * D ..][0..D];
            @memset(other, 0);
            other[0] = c.toBf16(-1);
        }
        const visible = self.shape.k + 1;
        for (0..visible) |pos| self.put(false, 0, pos, 0, 0, 0, 1);
        const tile = @min(64, (visible - 1) / 2);
        if (tile == 0) return;
        const changed = @min(2 * tile, visible - 1);
        for ([3]usize{ tile - 1, tile, changed }, 0..) |pos, i| self.score(pos, i, i == 2);
        if (!isWhole(self.form.kind) and self.width() < visible) {
            for ([2]usize{ self.width() - 1, self.width() }, 0..) |pos, i| {
                if (pos != changed) self.score(pos, i, false);
            }
        }
    }

    fn nonfinite(self: *Prompt, value: bool, head: usize, pos: usize, e: usize, x: f32) void {
        self.put(value, head, pos, e, c.toBf16(x), 0x38, x);
    }

    fn plantNonfinite(self: *Prompt) void {
        const span = self.shape.k + self.shape.rows;
        self.q.items[3] = c.toBf16(std.math.nan(f32));
        self.nonfinite(true, 0, 0, 7, std.math.inf(f32));
        self.nonfinite(true, 1, @min(1, span - 1), 11, std.math.inf(f32));
        self.nonfinite(true, 1, 0, 11, -std.math.inf(f32));
        self.nonfinite(false, 2, span - 1, 5, std.math.nan(f32));
    }

    fn stress(self: *Prompt, rng: std.Random) !void {
        const a = self.rig.gpu.gpa;
        const span = self.shape.k + self.shape.rows;
        const key = try a.alloc(u16, D);
        defer a.free(key);
        const codes = try a.alloc(u8, D);
        defer a.free(codes);
        const col = try a.alloc(f32, span);
        defer a.free(col);
        for (0..kvh) |h| {
            for (key, codes) |*b, *code| {
                b.* = attn.normal(rng);
                code.* = if (rng.boolean()) 0x38 else 0;
            }
            for (0..span) |pos| for (0..D) |e| self.put(false, h, pos, e, key[e], codes[e], 1.0 / 16.0);
            for (0..D) |e| {
                if (isFp8(self.form.kind)) mirroredE4m3(rng, col) else attn.mirrored(rng, col);
                for (col, 0..) |x, pos| self.put(true, h, pos, e, c.toBf16(x), if (isFp8(self.form.kind)) toCode(x) else 0, 1);
            }
        }
    }

    fn pn(self: *const Prompt) u32 {
        return @intCast((self.shape.k + self.shape.rows + 255) / 256);
    }

    fn launch(self: *Prompt, replacement: bool, dst: u64) !void {
        const kind = self.form.kind;
        const whole = isWhole(kind) and !replacement;
        const null_table = isWhole(kind) and replacement;
        const fp8 = isFp8(kind);
        const cache_width = self.width();
        var args: cuda.Args = .{};
        args.add(self.dev_q);
        const key_bytes = kvh * cache_width * D * (if (fp8) @as(usize, 1) else 2);
        args.add(self.dev_cache);
        args.add(self.dev_cache + key_bytes);
        if (fp8) {
            const scales = self.dev_cache + 2 * kvh * cache_width * D;
            args.add(scales);
            args.add(scales + kvh * cache_width * 4);
        }
        args.add(dst);
        for ([_]usize{ self.shape.k, self.shape.rows, qh, kvh, group }) |x| args.add(@as(c_int, @intCast(x)));
        const sc = scale * (if (self.form.control and !replacement) @as(f32, 2) else 1);
        args.add(sc);
        args.add(@as(c_int, @intCast(self.cap)));
        const ids = if (kind == .p256 or kind == .p256_ids) self.pn() else 0;
        if (!whole) {
            args.add(if (null_table) @as(u64, 0) else self.dev_ids);
            args.add(@as(i64, if (null_table) 0 else @intCast(if (fp8) self.pageStride() else self.pageStride() / 2)));
            const last: usize = if (isP256(kind) and !replacement) ids else if (null_table) 1 else cache_width;
            args.add(@as(c_int, @intCast(last)));
        }
        const shared: u32 = smem + if (kind == .p256_ids and !replacement) 4 * ids else 0;
        const cfg: cuda.Config = .{ .grid = .{ .x = @intCast((self.shape.rows + 63) / 64), .y = kvh * (group / 2) }, .block = .{ .x = 256 }, .shared = shared };
        try cuda.launch.launch(if (replacement) self.base else self.f, cfg, self.rig.s, &args);
    }

    fn run(self: *Prompt) !void {
        try self.launch(false, self.out);
        try self.rig.s.synchronize();
    }

    fn replaced(self: *Prompt) ![]const u8 {
        try self.launch(true, self.alt);
        return self.rig.bytes(self.alt, self.count() * 2);
    }

    fn count(self: *Prompt) usize {
        return self.shape.rows * qh * D;
    }

    fn load(self: *Prompt, i: usize, hd: *attn.Head) !void {
        const row = i / qh;
        const head = i % qh;
        const kh = head / group;
        for (hd.q, self.q.items[(row * qh + head) * D ..][0..D]) |*y, x| y.* = c.fromBf16(x);
        const n = self.shape.k + row + 1;
        for (0..n) |pos| for (0..D) |e| {
            hd.k[pos * D + e] = self.get(false, kh, pos, e);
            hd.v[pos * D + e] = self.get(true, kh, pos, e);
        };
        hd.n = n;
    }

    fn heads(self: *Prompt) usize {
        return self.shape.rows * qh;
    }

    fn check(self: *Prompt, gate: g.Gate) !?[]const u8 {
        return self.r.?.check(self, self.heads(), scale, try self.rig.outputBytes(), gate);
    }
    fn reference(self: *Prompt, want: []f64, mag: []f64) !void {
        return self.r.?.reference(self, self.heads(), scale, want, mag);
    }
    fn simulate(self: *Prompt, acc: c.Accum, order: g.Order, out: []f64) !void {
        return self.r.?.simulate(self, self.heads(), scale, acc, order, out);
    }
};

fn mirroredE4m3(rng: std.Random, out: []f32) void {
    const h = out.len / 2;
    for (0..h) |i| {
        const v: f32 = @floatFromInt(1 + 2 * rng.uintLessThan(u32, 8));
        out[i] = v;
        out[h + i] = (if (i < 8 and i % 2 == 0) @as(f32, 1) else 0) - v;
    }
    if (out.len % 2 == 1) out[out.len - 1] = 0;
}

fn toCode(v: f32) u8 {
    var code: u8 = 0;
    while (code < 0x7f) : (code += 1) if (e4m3(code) == @abs(v)) break;
    return code | @as(u8, if (v < 0) 0x80 else 0);
}

pub const Group = struct {
    a: std.mem.Allocator,
    mod: cuda.Module,
    items: [forms.len]Prompt = undefined,
    made: usize = 0,

    pub fn create(a: std.mem.Allocator, gpu: check.Gpu, s: cuda.Stream, kernels: *std.ArrayList(g.Kernel)) !*Group {
        const self = try a.create(Group);
        errdefer a.destroy(self);
        self.* = .{ .a = a, .mod = try cuda.Module.load(gpu.d, cuda.kernels.prefill_attention) };
        errdefer self.mod.unload();
        errdefer for (self.items[0..self.made]) |*item| item.deinit();
        for (forms) |form| {
            const f = try self.mod.function(symbol(form.kind));
            const base = try self.mod.function(baseSymbol(form.kind));
            try f.allowDynamicShared(smem + if (form.kind == .p256_ids) 4 * page_ids else 0);
            try base.allowDynamicShared(smem);
            self.items[self.made] = .{ .rig = .{ .gpu = gpu, .s = s }, .form = form, .f = f, .base = base };
            self.made += 1;
            try kernels.append(a, self.items[self.made - 1].kernel());
        }
        return self;
    }

    pub fn destroy(self: *Group) void {
        for (self.items[0..self.made]) |*item| item.deinit();
        self.mod.unload();
        self.a.destroy(self);
    }
};
