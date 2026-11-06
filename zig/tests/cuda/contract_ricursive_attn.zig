//! Ripple's attention under the contract on CUDA (80/8 heads): prompt rows, sliding decode, global decode over pages.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rc = @import("ricursive_cuda").attn;
const check = @import("check.zig");
const rig_ = @import("contract_rig.zig");
const attn = @import("contract_attn.zig");
const wk = @import("contract_window.zig");

const c = core.contract.spec;
const g = core.contract.gates;
const wa = cuda.window_attention;

/// Ripple-65B's attention: 80 query heads over 8 key heads, head dim 128, a 1,024-key window in a ring of 3,072.
const h = 80;
const hk = 8;
const window = 1024;
const ring = 3072;
/// Global layers' pages: tree_attention's span, as its decode reads them.
const page = 512;

/// Sliding chunks: before the window fills, across its edge, and past the ring's wrap (key 9,216 at slot 0).
const sliding_served = [_]c.Shape{ .{ .rows = 333, .n = h, .k = 0 }, .{ .rows = 1000, .n = h, .k = 700 }, .{ .rows = 2048, .n = h, .k = 8216 } };
/// Global chunks (every key): from the start, past 1,500 keys, and a whole 2,048-row pass.
const global_served = [_]c.Shape{ .{ .rows = 333, .n = h, .k = 0 }, .{ .rows = 1000, .n = h, .k = 1500 }, .{ .rows = 2048, .n = h, .k = 0 } };
/// A control's one shape: past the window and the ring's wrap, where each fault binds.
const control_served = [_]c.Shape{.{ .rows = 333, .n = h, .k = 8216 }};
/// The chunk control's: chunks of 333 after the first.
const chunk_served = [_]c.Shape{.{ .rows = 1000, .n = h, .k = 700 }};

/// A control's fault: score scale doubled, window doubled, ring one slot short, or later chunks' scale one ulp up.
pub const Fault = enum { none, scale, window, ring, late_scale };

/// A prompt registration: the window test's call, cut into `chunk` rows a launch (0: one launch), and its fault.
pub const Cfg = struct { base: wk.Cfg, chunk: usize = 0, fault: Fault = .none };

fn slidingCfg(name: []const u8, served: []const c.Shape) wk.Cfg {
    return .{ .name = name, .h = h, .hk = hk, .d = 128, .win = window, .ring = ring, .prompt = true, .served = served };
}

fn globalCfg(name: []const u8, pages: usize, served: []const c.Shape) wk.Cfg {
    return .{ .name = name, .h = h, .hk = hk, .d = 128, .win = 0, .ring = 0, .prompt = true, .pages = pages, .served = served };
}

pub const prompts = [_]Cfg{
    .{ .base = slidingCfg("cuda ricursive prompt wattn_kernel<128, 8, 2, 8>, 80/8 heads, window 1024, ring 3072", &sliding_served) },
    .{ .base = globalCfg("cuda ricursive prompt wattn_kernel<128, 8, 2, 8>, 80/8 heads, every key (global layers)", 0, &global_served) },
    .{ .base = globalCfg("cuda ricursive prompt tf_window_prompt_h2_pg, 80/8 heads, every key, pages of 512", page, &global_served) },
    // a row's bits do not depend on its chunk: cuts of the same rows bit-equal to one call
    .{ .base = slidingCfg("cuda ricursive prompt in chunks of 1,000, window 1024: one call's bits", &sliding_served), .chunk = 1000 },
    .{ .base = slidingCfg("cuda ricursive prompt in chunks of 333, window 1024: one call's bits", &sliding_served), .chunk = 333 },
    .{ .base = globalCfg("cuda ricursive prompt in chunks of 333, every key: one call's bits", 0, &global_served), .chunk = 333 },
    .{ .base = slidingCfg("control: Ripple's prompt rows with the score scaled twice", &control_served), .fault = .scale },
    .{ .base = slidingCfg("control: Ripple's prompt rows with a window of 2,048 keys", &control_served), .fault = .window },
    .{ .base = slidingCfg("control: Ripple's prompt rows with a ring one slot short (3,071)", &control_served), .fault = .ring },
    .{ .base = .{ .name = "control: Ripple's paged prompt rows whose first page id is a spare page's (all NaN)", .h = h, .hk = hk, .d = 128, .win = 0, .ring = 0, .prompt = true, .pages = page, .stale = true, .served = &control_served } },
    .{ .base = slidingCfg("control: Ripple's prompt in chunks of 333, the later chunks' scale one ulp up", &chunk_served), .chunk = 333, .fault = .late_scale },
};

/// Sliding decode through tf_window_decode_tc as it is: one-row steps past the ring's wrap, and 16-row windows.
pub const decodes = [_]wk.Cfg{
    .{ .name = "cuda ricursive decode tf_window_decode_tc, 80/8 heads, window 1024, ring 3072", .h = h, .hk = hk, .d = 128, .win = window, .ring = ring, .streams = 2, .served = &.{ .{ .rows = 1, .n = h, .k = 9500 }, .{ .rows = 16, .n = h, .k = 3000 } } },
};

/// The P x V chain of a sliding prompt's last row: at most its window's keys, a tile at a time, then the fold.
fn slidingChain(s: c.Shape) usize {
    return @min(s.k + s.rows, window) + 64 + 64;
}

/// A global prompt's last row: every key to its own.
fn globalChain(s: c.Shape) usize {
    return s.k + s.rows + 64 + 64;
}

/// Global decode: two pages' rows across a page's edge, rows 1 and 16 from the first key, a page's first key, deep.
const gdec_served = [_]c.Shape{ .{ .rows = 16, .n = h, .k = 505 }, .{ .rows = 1, .n = h, .k = 0 }, .{ .rows = 16, .n = h, .k = 0 }, .{ .rows = 1, .n = h, .k = 512 }, .{ .rows = 1, .n = h, .k = 30000 }, .{ .rows = 16, .n = h, .k = 30000 } };
/// A control's shape: 16 rows across a page's edge.
const gdec_control = [_]c.Shape{.{ .rows = 16, .n = h, .k = 505 }};
/// The short control's: rows over their first keys, where one key is much of a row's weight.
const gdec_short = [_]c.Shape{.{ .rows = 16, .n = h, .k = 1 }};
/// GlobalDecode's rows and keys in the contract: the served shapes' most.
const global_rows = 16;
const global_context = 30000 + 16;

/// Gate 4 at 30,000 keys: past about 8,000 keys the stress values' sums leave fp32's 24 bits, so no order is exact.
const deep_sums: c.Sums = .{ .bf16 = 0, .exact = false, .why = "past 8,000 keys the stress values' sums leave fp32's 24 bits: gate 4 holds the float64 bound, which fp16 sums overflow" };

/// A global decode control's fault: score scale doubled, each row one key short, or 64 query heads (group 8).
pub const GlobalFault = enum { none, scale, short, heads };

/// A global decode registration: the window test's paged prompt rows (row r at k + r over keys 0..k + r).
pub const GlobalCfg = struct { base: wk.Cfg, fault: GlobalFault = .none };

fn pagedCfg(name: []const u8, served: []const c.Shape, stale: bool) wk.Cfg {
    return .{ .name = name, .h = h, .hk = hk, .d = 128, .win = 0, .ring = 0, .prompt = true, .pages = page, .stale = stale, .served = served };
}

pub const globals = [_]GlobalCfg{
    .{ .base = pagedCfg("cuda ricursive global decode tf_tree_*_pg + tf_tree_merge, 80/8 heads, every key, pages of 512", &gdec_served, false) },
    .{ .base = pagedCfg("control: Ripple's global decode with the score scaled twice", &gdec_control, false), .fault = .scale },
    .{ .base = pagedCfg("control: Ripple's global decode whose first page id is a spare page's (all NaN)", &gdec_control, true) },
    .{ .base = pagedCfg("control: Ripple's global decode with each row one key short", &gdec_short, false), .fault = .short },
    .{ .base = pagedCfg("control: Ripple's global decode with 64 query heads over the 8 key heads (group 8)", &gdec_control, false), .fault = .heads },
};

/// Decode rows: rows, query heads and the first row's position; tree_attention's 16-pair tiles and 512-key groups.
pub fn globalSpec(f: GlobalCfg) c.Spec {
    const b = f.base;
    return .{ .name = b.name, .accum = .f32, .depth = &globalChain, .ops = 1, .out = .bf16, .tiles = .{ .{ .size = 16 }, .{}, .{ .size = page, .min = 0 } }, .served = b.served, .sums = deep_sums };
}

/// Prompt rows: chunk rows, query heads and the chunk's start; 64-row blocks, 64-key tiles.
pub fn spec(f: Cfg) c.Spec {
    const b = f.base;
    return .{ .name = b.name, .accum = .f32, .depth = if (b.win > 0) &slidingChain else &globalChain, .ops = 1, .out = .bf16, .replaces = if (f.chunk > 0) "wattn_kernel<128, 8, 2, 8> in one call" else null, .tiles = .{ .{ .size = rc.rows_a_block }, .{}, .{ .size = 64, .min = 0 } }, .served = b.served, .sums = .{ .why = attn.why } };
}

pub const Prompt = struct {
    rig: *rig_.Rig, // the window test's own
    w: wk.Window,
    k: *const rc.Kernels,
    f: Cfg,
    base: u64 = 0, // chunked: the one call's output (scratch)

    pub fn deinit(self: *Prompt) void {
        self.w.deinit();
    }

    pub fn kernel(self: *Prompt) g.Kernel {
        const control = self.f.fault != .none or self.f.base.stale;
        return .{ .spec = spec(self.f), .ctx = self, .vt = &vt, .control = if (control) .shapes else null };
    }

    const vt = g.hooks(Prompt);

    pub fn prepare(self: *Prompt, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (!try self.w.prepare(shape, fill, seed)) return false;
        if (self.f.chunk > 0) self.base = try self.rig.scratch(self.w.count() * 2);
        return true;
    }

    /// The launch's window: the registration's, with its control's fault (`later`: a chunk after the first).
    fn window(self: *const Prompt, later: bool) wa.Window {
        const f = self.f.base;
        const scale: f32 = @floatCast(1 / @sqrt(@as(f64, @floatFromInt(f.d)))); // the window test's, so its reference's
        const pg: ?wa.Pages = if (f.pages > 0) .{ .table = self.w.table, .page_elems = 2 * f.pages * f.hk * f.d, .tokens = f.pages } else null;
        var w = if (f.win > 0) rc.sliding(f.h, f.hk, f.win, f.ring, scale) else rc.global(f.h, f.hk, scale, pg);
        switch (self.f.fault) {
            .none => {},
            .scale => w.scale *= 2,
            .window => w.win *= 2,
            .ring => w.ring -= 1,
            .late_scale => if (later) {
                w.scale = std.math.nextAfter(f32, w.scale, 1);
            },
        }
        return w;
    }

    /// Rows [r, r + m) of the prepared call into `out`, at positions k + r, ...
    fn launch(self: *Prompt, r: usize, m: usize, out: u64) !void {
        const at = r * self.f.base.h * self.f.base.d * 2; // a row's heads, bf16, in q and out
        const w = &self.w;
        try rc.prompt(self.k, self.rig.s, w.dev[0] + at, w.dev[1], w.dev[2], out + at, w.shape.k + r, m, self.window(r > 0));
    }

    pub fn run(self: *Prompt) !void {
        const n = self.w.shape.rows;
        const step = if (self.f.chunk > 0) self.f.chunk else n;
        var r: usize = 0;
        while (r < n) : (r += step) try self.launch(r, @min(step, n - r), self.w.out);
        try self.rig.s.synchronize();
    }

    /// The chunks' baseline: the prepared rows in one call.
    pub fn replaced(self: *Prompt) ![]const u8 {
        const n = self.w.shape.rows;
        try rc.prompt(self.k, self.rig.s, self.w.dev[0], self.w.dev[1], self.w.dev[2], self.base, self.w.shape.k, n, self.window(false));
        return try self.rig.bytes(self.base, self.w.count() * 2);
    }

    pub fn count(self: *Prompt) usize {
        return self.w.count();
    }

    pub fn check(self: *Prompt, gate: g.Gate) !?[]const u8 {
        return self.w.check(gate);
    }

    pub fn reference(self: *Prompt, want: []f64, mag: []f64) !void {
        return self.w.reference(want, mag);
    }

    pub fn simulate(self: *Prompt, acc: c.Accum, order: g.Order, out: []f64) !void {
        return self.w.simulate(acc, order, out);
    }
};

/// Global decode through rc.decodeGlobal, over the window test's paged prompt inputs and its float64 reference.
pub const Global = struct {
    rig: *rig_.Rig, // the window test's own
    w: wk.Window,
    gd: *const rc.GlobalDecode,
    f: GlobalCfg,

    pub fn deinit(self: *Global) void {
        self.w.deinit();
    }

    pub fn kernel(self: *Global) g.Kernel {
        const control = self.f.fault != .none or self.f.base.stale;
        return .{ .spec = globalSpec(self.f), .ctx = self, .vt = &vt, .control = if (control) .shapes else null };
    }

    const vt = g.hooks(Global);

    pub fn prepare(self: *Global, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows > self.gd.max_rows or shape.k + shape.rows > self.gd.context) return false;
        return self.w.prepare(shape, fill, seed);
    }

    pub fn run(self: *Global) !void {
        const f = self.f.base;
        const w = &self.w;
        var a: rc.DecodeArgs = .{
            .q = w.dev[0],
            .kc = w.dev[1],
            .vc = w.dev[2],
            .out = w.out,
            .pos = w.shape.k,
            .rows = w.shape.rows,
            .pages = .{ .table = w.table, .page_elems = 2 * f.pages * f.hk * f.d, .tokens = f.pages },
            .scale = @floatCast(1 / @sqrt(@as(f64, @floatFromInt(f.d)))), // the window test's, so its reference's
        };
        switch (self.f.fault) {
            .none, .heads => {},
            .scale => a.scale *= 2,
            .short => a.pos -|= 1,
        }
        if (self.f.fault == .heads) try rc.decodeHeads(self.gd, w.rig.s, a, 64) else try rc.decodeGlobal(self.gd, w.rig.s, a);
        try w.rig.s.synchronize();
    }

    pub fn count(self: *Global) usize {
        return self.w.count();
    }

    pub fn check(self: *Global, gate: g.Gate) !?[]const u8 {
        return self.w.check(gate);
    }

    pub fn reference(self: *Global, want: []f64, mag: []f64) !void {
        return self.w.reference(want, mag);
    }

    pub fn simulate(self: *Global, acc: c.Accum, order: g.Order, out: []f64) !void {
        return self.w.simulate(acc, order, out);
    }
};

/// Every registration, on the heap: the gates hold pointers to them, and each prompt's rig pointer into itself.
pub const Set = struct {
    gpa: std.mem.Allocator,
    win: wa.WindowAttention,
    k: rc.Kernels = undefined,
    gd: rc.GlobalDecode = undefined,
    prompts: [prompts.len]Prompt = undefined,
    decodes: [decodes.len]wk.Window = undefined,
    globals: [globals.len]Global = undefined,

    /// The set, its kernels appended to `list`.
    pub fn init(gpu: check.Gpu, s: cuda.Stream, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel)) !*Set {
        const set = try gpu.gpa.create(Set);
        errdefer gpu.gpa.destroy(set);
        set.* = .{ .gpa = gpu.gpa, .win = try wa.WindowAttention.load(gpu.d) };
        errdefer set.win.unload();
        set.k = try rc.Kernels.init(&set.win);
        set.gd = try rc.GlobalDecode.init(gpu.d, gpu.gpa, global_rows, global_context);
        errdefer set.gd.deinit();
        for (&set.prompts, prompts) |*x, f| {
            x.* = .{ .rig = undefined, .w = .{ .rig = .{ .gpu = gpu, .s = s }, .f = f.base, .g = &set.win }, .k = &set.k, .f = f };
            x.rig = &x.w.rig;
        }
        for (&set.decodes, decodes) |*x, f| x.* = .{ .rig = .{ .gpu = gpu, .s = s }, .f = f, .g = &set.win };
        for (&set.globals, globals) |*x, f| {
            x.* = .{ .rig = undefined, .w = .{ .rig = .{ .gpu = gpu, .s = s }, .f = f.base, .g = &set.win }, .gd = &set.gd, .f = f };
            x.rig = &x.w.rig;
        }
        for (&set.prompts) |*x| try list.append(a, x.kernel());
        for (&set.decodes) |*x| try list.append(a, x.kernel());
        for (&set.globals) |*x| try list.append(a, x.kernel());
        return set;
    }

    pub fn deinit(set: *Set) void {
        for (&set.prompts) |*x| x.deinit();
        for (&set.decodes) |*x| x.deinit();
        for (&set.globals) |*x| x.deinit();
        set.gd.deinit();
        set.win.unload();
        set.gpa.destroy(set);
    }
};
