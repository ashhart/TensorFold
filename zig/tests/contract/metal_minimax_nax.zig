//! MiniMax-M3.1-Flash's decode experts (experts.metal) and its tensor-unit suite (attention, index: _nax_attn.zig).
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const contract = @import("contract");
const sources = @import("kernel_sources");
const rig_ = @import("rig.zig");
const attn = @import("metal_minimax_nax_attn.zig");

const c = contract.spec;
const g = contract.gates;
const Affine = contract.affine.Affine;
const row = tf.row_projection;
const Texts = @import("metal_minimax.zig").Texts;

/// The release's shape (config.zig's defaults are MiniMax-M3.1-Flash's config.json).
pub const config: tf.minimax_m3.config.Config = .{};
const st = tf.minimax_m3.state;

pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    kernels: *std.ArrayList(g.Kernel),
    grouped: std.ArrayList(*Grouped) = .empty,
    index: ?*attn.Index = null,
    prompt: ?*attn.Prompt = null,

    pub fn collect(s: *Suite, only: ?[]const u8) !void {
        for (grouped_forms) |f| {
            if (!wanted(only, f.name)) continue;
            const x = try s.a.create(Grouped);
            x.* = try Grouped.init(s.a, s.device, s.queue, f);
            try s.grouped.append(s.a, x);
            try s.kernels.append(s.a, x.kernel());
        }
        if (wanted(only, attn.Index.spec().name)) {
            s.index = try s.a.create(attn.Index);
            s.index.?.* = try attn.Index.init(s.a, s.device, s.queue);
            try s.kernels.append(s.a, s.index.?.kernel());
        }
        if (wanted(only, attn.Prompt.spec().name)) {
            s.prompt = try s.a.create(attn.Prompt);
            s.prompt.?.* = try attn.Prompt.init(s.a, s.device, s.queue);
            try s.kernels.append(s.a, s.prompt.?.kernel());
        }
    }
    pub fn deinit(s: *Suite) void {
        for (s.grouped.items) |x| {
            x.deinit();
            s.a.destroy(x);
        }
        s.grouped.deinit(s.a);
        if (s.index) |x| {
            x.deinit();
            s.a.destroy(x);
        }
        if (s.prompt) |x| {
            x.deinit();
            s.a.destroy(x);
        }
    }
};

fn wanted(only: ?[]const u8, name: []const u8) bool {
    const o = only orelse return true;
    return std.mem.indexOf(u8, name, o) != null;
}

/// Gate 5's texts for `minimax_m3 nax ` launches as model.zig builds them, nax.h inlined (fragments before the M5).
pub fn limitsText(a: std.mem.Allocator, device: mtl.Device, name: []const u8) !?Texts {
    if (!std.mem.startsWith(u8, name, "minimax_m3 nax ")) return null;
    if (std.mem.indexOf(u8, name, " experts ") != null) {
        const text = try expertsText(a);
        return .{ .here = text, .pre_m5 = text };
    }
    const head = if (std.mem.indexOf(u8, name, " attention ") != null) try attn.promptHead(a) else try attn.indexHead(a);
    const body_src = if (std.mem.indexOf(u8, name, " attention ") != null) sources.minimax_m3_attention_nax else sources.minimax_m3_index_nax;
    const body = try std.mem.replaceOwned(u8, a, body_src, "#include \"../nax.h\"", sources.nax);
    const pre = try std.mem.concat(a, u8, &.{ head, "#define TF_SIMD_FRAGS 1\n", body });
    return .{ .here = if (device.tensorUnits()) try std.mem.concat(a, u8, &.{ head, body }) else pre, .pre_m5 = pre };
}

fn expertsText(a: std.mem.Allocator) ![]u8 {
    return std.mem.concat(a, u8, &.{ sources.core_row_projection, "\n", sources.minimax_m3_experts });
}

fn expertsPipeline(a: std.mem.Allocator, device: mtl.Device, name: []const u8) !mtl.Pipeline {
    const text = try expertsText(a);
    defer a.free(text);
    return rig_.pipeline(device, text, name);
}

fn noSums(_: c.Shape) usize {
    return 0;
}

/// Picks of `tokens` tokens, `topk` distinct experts each out of `experts`; a few hot experts take most slots.
fn picks(a: std.mem.Allocator, tokens: usize, topk: usize, experts: u32, seed: u64) ![]i32 {
    const ids = try a.alloc(i32, tokens * topk);
    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();
    for (0..tokens) |t| for (0..topk) |j| {
        var e: u32 = if (rng.boolean()) rng.uintLessThan(u32, @min(experts, 3)) else rng.uintLessThan(u32, experts);
        while (std.mem.indexOfScalar(i32, ids[t * topk ..][0..j], @intCast(e)) != null) e = (e + 1) % experts;
        ids[t * topk + j] = @intCast(e);
    };
    return ids;
}

/// The host's expert groups: G[0] the experts picked, then (id, first member) by id; M the slots by expert, in order.
fn hostGroups(a: std.mem.Allocator, ids: []const i32) !struct { g: []u32, m: []u32 } {
    const gs = try a.alloc(u32, 1 + 2 * ids.len);
    errdefer a.free(gs);
    const ms = try a.alloc(u32, ids.len);
    var n: u32 = 0;
    var at: u32 = 0;
    var last: i64 = -1;
    while (true) {
        var next: i64 = std.math.maxInt(i64);
        for (ids) |e| if (e > last and e < next) {
            next = e;
        };
        if (next == std.math.maxInt(i64)) break;
        gs[1 + 2 * n] = @intCast(next);
        gs[2 + 2 * n] = at;
        for (ids, 0..) |e, s| if (e == next) {
            ms[at] = @intCast(s);
            at += 1;
        };
        last = next;
        n += 1;
    }
    gs[0] = n;
    return .{ .g = gs, .m = ms };
}

// --- mm_experts_grouped: each picked expert's rows read once a window, tf_rp_rows_at's sums ----------------------

/// The checkpoints' expert formats: 3-bit groups of 64, and the 4-bit groups of 128 on their chosen layers.
const Form = struct { bits: u8, group: usize, name: []const u8 };
const grouped_forms = [_]Form{
    .{ .bits = 3, .group = 64, .name = "metal minimax_m3 mm_experts_grouped_q3 g64" },
    .{ .bits = 4, .group = 128, .name = "metal minimax_m3 mm_experts_grouped_q4 g128" },
};

/// As tf_row_projection_indexed (Ornith's q27_rows_extra): 16-input pieces, a lane's pieces in turn, a simd_sum.
fn rowDepth(s: c.Shape) usize {
    return 16 + 3 * ((s.k + 511) / 512) + 8;
}

/// Slot s reads token s / topk and expert ids[s] of `experts` stacked matrices; members 4, 2 then 1 at a time.
pub const Grouped = struct {
    /// Matrices stacked in the test: enough for distinct picks and empty experts, small enough for float64.
    const experts = 6;
    const topk = 4;
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    form: Form,
    f: ?Affine = null,
    ids: []i32 = &.{},
    refs: [7]rig_.Ref = undefined,
    out: rig_.Ref = undefined,
    dims: [6]i32 = undefined,

    /// Rows: a window's tokens; n and k: one expert's matrix.
    pub fn spec(f: Form) c.Spec {
        const gu = config.moe_intermediate * 2;
        return .{
            .name = f.name,
            .accum = .f32,
            .depth = rowDepth,
            .out = .bf16,
            .tiles = .{ .{ .size = 4 }, .{ .size = 8, .step = 8, .min = 8 }, .{ .size = 512, .step = f.group, .min = f.group } },
            // gate|up at one token and a drafted window, down at the widest; one token first (gate 4 replays 4 slots)
            .served = &.{ .{ .rows = 1, .n = gu, .k = config.hidden }, .{ .rows = st.window_rows, .n = gu, .k = config.hidden }, .{ .rows = st.batch_rows, .n = config.hidden, .k = config.moe_intermediate } },
        };
    }

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, f: Form) !Grouped {
        const fun = if (f.bits == 3) "mm_experts_grouped_q3" else "mm_experts_grouped_q4";
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = try expertsPipeline(a, device, fun), .form = f };
    }
    pub fn deinit(t: *Grouped) void {
        t.drop();
        t.rig.deinit();
        t.pipe.deinit();
    }
    fn drop(t: *Grouped) void {
        if (t.f) |f| f.deinit();
        t.f = null;
        t.a.free(t.ids);
        t.ids = &.{};
        t.rig.reset();
    }
    pub fn kernel(t: *Grouped) g.Kernel {
        return .{ .spec = spec(t.form), .ctx = t, .vt = &vt };
    }
    const vt = g.hooks(Grouped);

    pub fn prepare(t: *Grouped, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        t.drop();
        if (shape.rows == 0 or shape.k % t.form.group != 0) return false;
        (row.Weights{ .w = undefined, .scales = undefined, .biases = undefined, .n = shape.n, .k = shape.k, .group = t.form.group, .bits = t.form.bits }).validate() catch return false;
        // every expert in one family: rows [e n, (e + 1) n) are expert e's
        const f = try Affine.init(t.a, .{ .rows = shape.rows, .n = experts * shape.n, .k = shape.k }, t.form.bits, t.form.group);
        t.f = f;
        try f.fill(fill, seed);
        const words = try t.a.alloc(u32, experts * shape.n * shape.k * t.form.bits / 32);
        defer t.a.free(words);
        f.words(words);
        t.ids = try picks(t.a, shape.rows, topk, experts, seed +% 1);
        const grp = try hostGroups(t.a, t.ids);
        defer t.a.free(grp.g);
        defer t.a.free(grp.m);
        t.dims = .{ @intCast(shape.k), @intCast(shape.n), @intCast(t.form.group), @intCast(t.ids.len), experts, topk };
        t.refs[0] = try t.rig.input(std.mem.sliceAsBytes(f.x));
        t.refs[1] = try t.rig.input(std.mem.sliceAsBytes(words));
        t.refs[2] = try t.rig.input(std.mem.sliceAsBytes(f.scales));
        t.refs[3] = try t.rig.input(std.mem.sliceAsBytes(f.biases));
        t.refs[4] = try t.rig.input(std.mem.sliceAsBytes(&t.dims));
        t.refs[5] = try t.rig.input(std.mem.sliceAsBytes(grp.g));
        t.refs[6] = try t.rig.input(std.mem.sliceAsBytes(grp.m));
        t.out = try t.rig.output(t.count() * 2);
        return true;
    }

    pub fn run(t: *Grouped) !void {
        const n: usize = @intCast(t.dims[1]);
        const p = t.rig.begin();
        const e = p.e;
        e.setPipeline(t.pipe);
        for (t.refs[0..5], 0..) |r, i| e.setBuffer(r.buf, r.off, i);
        e.setBuffer(t.out.buf, t.out.off, 5);
        e.setBuffer(t.refs[5].buf, t.refs[5].off, 6);
        e.setBuffer(t.refs[6].buf, t.refs[6].off, 7);
        // as forward.zig's experts(): (columns / (RPS * SG), slots) groups of 64 threads
        e.dispatchGroups(mtl.Size.of(n / (row.columns * row.groups), t.ids.len, 1), mtl.Size.of(32 * row.groups, 1, 1));
        try t.rig.finish(p);
    }

    pub fn count(t: *Grouped) usize {
        return t.ids.len * @as(usize, @intCast(t.dims[1]));
    }

    /// Slot s's family: its token's row of x over its expert's matrix (slices of the whole family, nothing copied).
    fn slot(t: *Grouped, s: usize) Affine {
        var v = t.f.?;
        const n = v.shape.n / experts;
        const k = v.shape.k;
        const m: usize = @intCast(t.ids[s]);
        v.shape = .{ .rows = 1, .n = n, .k = k };
        v.x = v.x[(s / topk) * k ..][0..k];
        v.codes = v.codes[m * n * k ..][0 .. n * k];
        v.scales = v.scales[m * n * (k / v.group) ..][0 .. n * (k / v.group)];
        v.biases = v.biases[m * n * (k / v.group) ..][0 .. n * (k / v.group)];
        return v;
    }

    pub fn reference(t: *Grouped, want: []f64, mag: []f64) !void {
        const n: usize = @intCast(t.dims[1]);
        for (0..t.ids.len) |s| try t.slot(s).reference(want[s * n ..][0..n], mag[s * n ..][0..n]);
    }

    pub fn simulate(t: *Grouped, acc: c.Accum, order: g.Order, out: []f64) !void {
        const n: usize = @intCast(t.dims[1]);
        for (0..t.ids.len) |s| try t.slot(s).simulate(acc, order, out[s * n ..][0..n]);
    }
};

test {
    _ = attn;
}
