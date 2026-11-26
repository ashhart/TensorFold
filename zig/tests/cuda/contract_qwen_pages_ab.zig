//! Qwen3.5's paged decode partials: raw PO/PM/PL, merged rows, and their contiguous replacements.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const rig_ = @import("contract_rig.zig");
const attn = @import("contract_attn.zig");
const fixture = @import("contract_qwen_pages.zig");

const c = core.contract.spec;
const g = core.contract.gates;
const hd = fixture.hd;
const qh = fixture.q_heads;
const kvh = fixture.kv_heads;
const group = qh / kvh;
const chunks = fixture.max_chunks;
const layers = fixture.attentions;
const scale: f32 = 1.0 / 16.0;
const Format = fixture.Format;
const Geo = fixture.Geo;
const Seg = fixture.Seg;
const Ancestor = fixture.Ancestor;
const Case = fixture.Case;

/// Every key/page/split class exercised by contract_qwen_pages, in the same logical layouts.
const cases = [_]Case{
    .{ .lens = .{ 0, 3 }, .rows = 1, .ptok = 16 },
    .{ .lens = .{ 127, 128 }, .rows = 4, .ptok = 16 },
    .{ .lens = .{ 129, 1000 }, .rows = 8, .ptok = 128 },
    .{ .lens = .{ 700, 33 }, .rows = 2, .ptok = 48 },
    .{ .lens = .{ 4095, 5000 }, .rows = 16, .ptok = 256 },
    .{ .lens = .{ 9000, 300 }, .rows = 3, .ptok = 512 },
    .{ .lens = .{ 31, 128 }, .rows = 128, .ptok = 48 },
    .{ .lens = .{ 4096, 4097 }, .rows = 64, .ptok = 48 },
    .{ .lens = .{ 4096, 4097 }, .rows = 65, .ptok = 48 },
    .{ .lens = .{ 32767, 32768 }, .rows = 1, .ptok = 256 },
    .{ .lens = .{ 4096, 8192 }, .rows = 1, .ptok = 48 },
    .{ .lens = .{ 8, 16 }, .rows = 5, .ptok = 16 },
};
const Form = struct { case: usize, format: Format, control: bool = false };
const forms = blk: {
    var out: [2 * cases.len + 2]Form = undefined;
    var n = 0;
    for (cases, 0..) |_, i| for ([_]Format{ .bf16, .fp8 }) |format| {
        out[n] = .{ .case = i, .format = format };
        n += 1;
    };
    for ([_]Format{ .bf16, .fp8 }) |format| {
        out[n] = .{ .case = 1, .format = format, .control = true };
        n += 1;
    }
    break :blk out;
};

fn code(v: f32) u8 {
    var x: u8 = 0;
    while (x < 0x7f) : (x += 1) if (core.quant.fp8.e4m3(x) == @abs(v)) break;
    return x | @as(u8, if (v < 0) 0x80 else 0);
}

const Entry = struct {
    rig: rig_.Rig,
    alt: rig_.Rig,
    form: Form,
    primary: cuda.Function,
    contiguous: cuda.Function,
    merge: cuda.Function,
    label: [240]u8 = undefined,
    served_shape: [1]c.Shape = undefined,
    arena: ?std.heap.ArenaAllocator = null,
    r: ?attn.Ref = null,
    f: fixture.Fixture = undefined,
    pages: []u8 = &.{},
    segs: u64 = 0,
    flat_segs: u64 = 0,
    rowseg: u64 = 0,
    rowanc: u64 = 0,
    q: u64 = 0,
    kv: u64 = 0,
    p: u64 = 0,
    pm: u64 = 0,
    pl: u64 = 0,
    po: u64 = 0,
    out: u64 = 0,
    alt_pm: u64 = 0,
    alt_pl: u64 = 0,
    alt_po: u64 = 0,
    alt_out: u64 = 0,
    m: usize = 0,
    rows_per: [2]usize = .{ 0, 0 },

    fn kernel(self: *Entry) g.Kernel {
        const ca = cases[self.form.case];
        self.served_shape[0] = .{ .rows = 2 * ca.rows, .n = qh, .k = @max(ca.lens[0], ca.lens[1]) + ca.rows };
        const name = std.fmt.bufPrint(&self.label, "{s}cuda qwen35 {s} raw PM/PL/PO + merge, {d}/{d} heads, keys {d}+{d} rows, {d}-position pages", .{
            if (self.form.control) "control: swapped first/last pages on " else "",
            if (self.form.format == .bf16) "tf_q35_attn_partial_paged" else "tf_q35_attn_partial8_paged",
            qh,
            kvh,
            @max(ca.lens[0], ca.lens[1]),
            ca.rows,
            ca.ptok,
        }) catch unreachable;
        return .{
            .spec = .{
                .name = name,
                .accum = .none,
                .depth = depth,
                .out = .raw,
                .replaces = if (self.form.control) null else if (self.form.format == .bf16) "tf_q35_attn_partial_contiguous over identical logical rows" else "tf_q35_attn_partial8_contiguous over identical logical rows",
                .nan = .{ .stated = "every raw partial byte, including unwritten empty-chunk PO poison, and merged BF16 classes" },
                .served = &self.served_shape,
            },
            .ctx = self,
            .vt = &vt,
            .control = if (self.form.control) .shapes else null,
        };
    }
    const vt = g.hooks(Entry);

    fn deinit(self: *Entry) void {
        if (self.r) |*r| r.deinit();
        if (self.arena) |*a| a.deinit();
        self.alt.deinit();
        self.rig.deinit();
    }

    fn partialBytes(self: *const Entry) usize {
        return self.m * qh * chunks * (hd + 2) * 4;
    }
    fn mergedBytes(self: *const Entry) usize {
        return self.m * qh * hd * 2;
    }
    fn count(self: *Entry) usize {
        return self.partialBytes() + self.mergedBytes();
    }

    fn put(self: *Entry, st: usize, part: usize, head: usize, pos: usize, d: usize, bits: u16, fp8: u8, sc: f32) void {
        const sp = self.f.logical(1, part, head, pos);
        const cache = self.f.kv[st];
        if (self.form.format == .bf16) {
            std.mem.writeInt(u16, cache[sp[0].off + 2 * d ..][0..2], bits, .little);
        } else {
            cache[sp[0].off + d] = fp8;
            std.mem.writeInt(u32, cache[sp[1].off..][0..4], @bitCast(sc), .little);
        }
    }

    fn putScore(self: *Entry, pos: usize, which: usize, changed: bool) void {
        const v: f32 = if (changed) 32 else 16;
        self.put(0, 0, 0, pos, 0, c.toBf16(v), code(v), 1);
        self.put(0, 0, 0, pos, 1, if (which == 0) 0x8000 else 0, if (which == 0) 0x80 else 0, 1);
        self.put(0, 0, 0, pos, 2, if (which == 0) 1 else 0x8001, if (which == 0) 1 else 0x81, 1);
        self.put(0, 0, 0, pos, 3, if (which == 0) 0x807f else 0x007f, if (which == 0) 0x87 else 7, 1);
        const value: f32 = if (changed) 9 else 1 + 2 * @as(f32, @floatFromInt(which));
        self.put(0, 1, 0, pos, 0, c.toBf16(value), code(value), 1);
        self.put(0, 1, 0, pos, 1, if (which == 0) 0x8000 else 0, if (which == 0) 0x80 else 0, 1);
        self.put(0, 1, 0, pos, 2, if (which == 0) 1 else 0x8001, if (which == 0) 1 else 0x81, 1);
        self.put(0, 1, 0, pos, 3, if (which == 0) 0x807f else 0x007f, if (which == 0) 0x87 else 7, 1);
        self.put(0, 1, 0, pos, 4, if (which == 0) 0x007f else 0x807f, if (which == 0) 7 else 0x87, 1);
    }

    fn adversarial(self: *Entry) void {
        const ca = cases[self.form.case];
        const row = self.rows_per[0] - 1;
        const visible = ca.lens[0] + row + 1;
        const q = self.f.q[(row * qh) * hd ..][0..hd];
        @memset(q, 0);
        q[0] = c.toBf16(1);
        q[1] = 0x8000;
        q[2] = 1;
        q[3] = 0x8001;
        q[4] = 0x007f;
        q[5] = 0x807f;
        for (0..visible) |pos| self.put(0, 0, 0, pos, 0, 0, 0, 1);
        // One term has no tie or later maximum; two terms can tie but cannot then change it.
        if (visible == 1) {
            self.putScore(0, 0, false);
            return;
        }
        const ck = @max(128, (visible + chunks * 128 - 1) / (chunks * 128) * 128);
        var first: usize = 0;
        var second: usize = 1;
        if (visible > 2 * ck) {
            first = ck - 1;
            second = ck;
        }
        self.putScore(first, 0, false);
        self.putScore(second, 1, false);
        const changed: ?usize = if (visible >= 3) @min(2 * ck, visible - 1) else null;
        if (changed) |pos| self.putScore(pos, 2, true);
        if (ca.ptok < visible) for ([2]usize{ ca.ptok - 1, ca.ptok }, 0..) |pos, i| {
            if (pos != first and pos != second and (changed == null or pos != changed.?)) self.putScore(pos, i, false);
        };
    }

    fn nonfinite(self: *Entry) void {
        const ca = cases[self.form.case];
        const visible = ca.lens[0] + self.rows_per[0];
        self.f.q[3] = c.toBf16(std.math.nan(f32));
        self.put(0, 1, 0, 0, 7, c.toBf16(std.math.inf(f32)), 0x38, std.math.inf(f32));
        self.put(0, 1, 1, 0, 11, c.toBf16(std.math.inf(f32)), 0x38, std.math.inf(f32));
        if (visible > 1) self.put(0, 1, 1, 1, 11, c.toBf16(-std.math.inf(f32)), 0x38, -std.math.inf(f32));
        self.put(0, 0, 2, visible - 1, 5, c.toBf16(std.math.nan(f32)), 0x7f, 1);
    }

    fn stress(self: *Entry) void {
        @memset(self.f.q, 0);
        for (0..2) |st| for (0..kvh) |head| for (0..self.f.cap) |pos| {
            for (0..hd) |d| self.put(st, 0, head, pos, d, 0, 0, 1);
            for (0..hd) |d| {
                const half = self.f.cap / 2;
                const mag: f32 = @floatFromInt(1 + 2 * (d % 8));
                const value = if (pos < half) mag else @as(f32, @intFromBool(pos - half < 8)) - mag;
                self.put(st, 1, head, pos, d, c.toBf16(value), code(value), 1);
            }
        };
    }

    fn prepare(self: *Entry, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const ca = cases[self.form.case];
        if (shape.rows == 0 or shape.n != qh) return false;
        self.rig.reset();
        self.alt.reset();
        if (self.arena) |*old| old.deinit();
        self.arena = .init(self.rig.gpu.gpa);
        const a = self.arena.?.allocator();
        if (self.r) |*r| r.deinit();
        self.r = null;
        self.m = shape.rows;
        self.rows_per = .{ (self.m + 1) / 2, self.m / 2 };
        const most = @max(ca.lens[0] + self.rows_per[0], ca.lens[1] + self.rows_per[1]);
        self.r = try attn.Ref.init(self.rig.gpu.gpa, hd, most);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const cap = (most + ca.ptok - 1) / ca.ptok * ca.ptok;
        const np = cap / ca.ptok;
        const ids = try a.alloc(u32, 2 * np + 2);
        for (ids, 0..) |*id, i| id.* = @intCast(i);
        rng.shuffle(u32, ids);
        self.f = .{ .c = ca, .format = self.form.format, .cap = cap, .kv = .{ &.{}, &.{} }, .q = try fixture.randoms(a, rng, self.m * qh * hd), .k = &.{}, .v = &.{}, .ids = .{ ids[0..np], ids[np .. 2 * np] }, .pages = ids.len };
        for (&self.f.kv) |*cache| {
            cache.* = try a.alloc(u8, fixture.blockBytes(self.form.format, cap, layers));
            for (0..layers) |layer| for (0..2) |part| for (0..kvh) |head| for (0..cap) |pos| {
                fixture.Fixture.fill(self.form.format, rng, cache.*, self.f.logical(layer, part, head, pos));
            };
        }
        if (fill == .random) self.adversarial();
        switch (fill) {
            .random => {},
            .nonfinite => self.nonfinite(),
            .stress => self.stress(),
        }
        const page_bytes = fixture.blockBytes(self.form.format, ca.ptok, layers);
        self.pages = try a.alloc(u8, ids.len * page_bytes);
        rng.bytes(self.pages);
        for (0..2) |st| for (0..layers) |layer| for (0..2) |part| for (0..kvh) |head| for (0..cap) |pos| {
            const from = self.f.logical(layer, part, head, pos);
            const to = fixture.spans(self.form.format, ca.ptok, layer, part, head, pos % ca.ptok);
            for (from, to) |x, y| {
                const off = self.f.ids[st][pos / ca.ptok] * page_bytes + y.off;
                @memcpy(self.pages[off..][0..y.len], self.f.kv[st][x.off..][0..x.len]);
            }
        };
        // Requested gate rows change causal depths and therefore which trailing IDs must stay unread.
        for (0..2) |st| {
            const used = (ca.lens[st] + self.rows_per[st] + ca.ptok - 1) / ca.ptok;
            for (self.f.ids[st][used..]) |*id| id.* = @bitCast(@as(i32, -1));
        }
        if (self.form.control) {
            const used = (ca.lens[0] + self.rows_per[0] + ca.ptok - 1) / ca.ptok;
            if (used > 1) std.mem.swap(u32, &self.f.ids[0][0], &self.f.ids[0][used - 1]);
        }
        var paged: [2]Seg = undefined;
        var flat: [2]Seg = undefined;
        const rs = try a.alloc(i32, self.m);
        const ra = try a.alloc(Ancestor, self.m);
        for (0..self.m) |row| {
            const st: usize = @intFromBool(row >= self.rows_per[0]);
            const local = row - @as(usize, if (st == 0) 0 else self.rows_per[0]);
            rs[row] = @intCast(st);
            ra[row] = .{ .bits = 0, .depth = @intCast(local), .chain = 1 };
        }
        const table_ptrs = try a.alloc(u64, 2);
        defer a.free(table_ptrs);
        for (0..2) |st| {
            table_ptrs[st] = try self.rig.input(std.mem.sliceAsBytes(self.f.ids[st]));
            const row0 = @as(usize, if (st == 0) 0 else self.rows_per[0]);
            paged[st] = .{ .kv = 0, .cap = @intCast(cap), .len = @intCast(ca.lens[st]), .row0 = @intCast(row0), .rows = @intCast(self.rows_per[st]), .pt = table_ptrs[st], .ps = if (self.form.format == .fp8) page_bytes else page_bytes / 2, .ptok = @intCast(ca.ptok) };
            flat[st] = paged[st];
            flat[st].kv = try self.alt.input(self.f.kv[st]);
            flat[st].pt = 0;
            flat[st].ps = 0;
            flat[st].ptok = 0;
        }
        self.kv = try self.rig.input(self.pages);
        for (&paged) |*sg| sg.kv = self.kv;
        self.segs = try self.rig.input(std.mem.sliceAsBytes(&paged));
        self.flat_segs = try self.alt.input(std.mem.sliceAsBytes(&flat));
        self.rowseg = try self.rig.input(std.mem.sliceAsBytes(rs));
        self.rowanc = try self.rig.input(std.mem.sliceAsBytes(ra));
        self.q = try self.rig.input(std.mem.sliceAsBytes(self.f.q));
        const counts: [5]i32 = .{ 0, 0, @intCast(self.m), 2, 0 };
        self.p = try self.rig.input(std.mem.sliceAsBytes(&counts));
        const slots = self.m * qh * chunks;
        self.pm = try self.rig.output(slots * 4);
        self.pl = try self.rig.output(slots * 4);
        self.po = try self.rig.output(slots * hd * 4);
        self.out = try self.rig.output(self.mergedBytes());
        self.alt_pm = try self.alt.output(slots * 4);
        self.alt_pl = try self.alt.output(slots * 4);
        self.alt_po = try self.alt.output(slots * hd * 4);
        self.alt_out = try self.alt.output(self.mergedBytes());
        return true;
    }

    fn launch(self: *Entry, replacement: bool) !void {
        const geo: Geo = .{ .hid = 0, .nk = 0, .nv = 0, .qh = qh, .kvh = kvh, .nl = 0 };
        const layer: c_int = 1;
        const segs = if (replacement) self.flat_segs else self.segs;
        const pm = if (replacement) self.alt_pm else self.pm;
        const pl = if (replacement) self.alt_pl else self.pl;
        const po = if (replacement) self.alt_po else self.po;
        const out = if (replacement) self.alt_out else self.out;
        var args: cuda.Args = .{};
        inline for (.{ geo, self.q, segs, self.rowseg, self.rowanc, layer, self.p, scale, pm, pl, po }) |x| args.add(x);
        try cuda.launch.launch(if (replacement) self.contiguous else self.primary, .{ .grid = .{ .x = chunks, .y = kvh, .z = @intCast(self.m) }, .block = .{ .x = 32 * group } }, self.rig.s, &args);
        args = .{};
        inline for (.{ geo, pm, pl, po, segs, self.rowseg, self.rowanc, self.p, out }) |x| args.add(x);
        try cuda.launch.launch(self.merge, .{ .grid = .{ .x = qh, .y = @intCast(self.m) }, .block = .{ .x = 32 } }, self.rig.s, &args);
    }

    fn run(self: *Entry) !void {
        try self.launch(false);
        try self.rig.s.synchronize();
    }
    fn replaced(self: *Entry) ![]const u8 {
        try self.launch(true);
        return self.alt.outputBytes();
    }

    fn load(self: *Entry, i: usize, head: *attn.Head) !void {
        const ca = cases[self.form.case];
        const row = i / qh;
        const st: usize = @intFromBool(row >= self.rows_per[0]);
        const local = row - @as(usize, if (st == 0) 0 else self.rows_per[0]);
        const kh = (i % qh) / group;
        head.n = ca.lens[st] + local + 1;
        for (head.q, self.f.q[(row * qh + i % qh) * hd ..][0..hd]) |*v, x| v.* = c.fromBf16(x);
        for (0..head.n) |pos| for (0..hd) |d| {
            head.k[pos * hd + d] = self.f.value(self.f.kv[st], 1, 0, kh, pos, d);
            head.v[pos * hd + d] = self.f.value(self.f.kv[st], 1, 1, kh, pos, d);
        };
    }
    fn reference(_: *Entry, want: []f64, mag: []f64) !void {
        @memset(want, 0);
        @memset(mag, 0);
    }
    fn check(self: *Entry, gate: g.Gate) !?[]const u8 {
        const raw = try self.rig.outputBytes();
        return self.r.?.check(self, self.m * qh, scale, raw[self.partialBytes()..], gate);
    }
};

fn depth(shape: c.Shape) usize {
    return shape.k + 64;
}

pub const Group = struct {
    a: std.mem.Allocator,
    bf16: ?cuda.Module = null,
    fp8: ?cuda.Module = null,
    items: [forms.len]Entry = undefined,
    made: usize = 0,

    pub fn create(a: std.mem.Allocator, gpu: check.Gpu, s: cuda.Stream, kernels: *std.ArrayList(g.Kernel)) !*Group {
        const self = try a.create(Group);
        self.* = .{ .a = a };
        errdefer self.destroy();
        self.bf16 = try cuda.Module.load(gpu.d, cuda.kernels.qwen35);
        self.fp8 = try cuda.Module.load(gpu.d, cuda.kernels.qwen35_kv8);
        const merge = try self.bf16.?.function("tf_q35_attn_merge");
        for (&self.items, forms) |*item, form| {
            const mod = if (form.format == .bf16) self.bf16.? else self.fp8.?;
            item.* = .{
                .rig = .{ .gpu = gpu, .s = s },
                .alt = .{ .gpu = gpu, .s = s },
                .form = form,
                .merge = merge,
                .primary = try mod.function(if (form.format == .bf16) "tf_q35_attn_partial_paged" else "tf_q35_attn_partial8_paged"),
                .contiguous = try mod.function(if (form.format == .bf16) "tf_q35_attn_partial_contiguous" else "tf_q35_attn_partial8_contiguous"),
            };
            self.made += 1;
            try kernels.append(a, item.kernel());
        }
        return self;
    }

    pub fn destroy(self: *Group) void {
        for (self.items[0..self.made]) |*item| item.deinit();
        if (self.fp8) |mod| mod.unload();
        if (self.bf16) |mod| mod.unload();
        self.a.destroy(self);
    }
};
