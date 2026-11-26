//! tf-glm-dflash-speed: GLM's DFlash2 drafter pass at incoai's real shapes on synthetic weights, ms a stage and pass.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const glm = tf.glm;
const cfg = glm.config;
const wts = glm.weights;
const kernels = glm.kernels;
const affine = glm.affine;
const op = tf.qwen27.dflash.operators;
const ck = tf.qwen27.checkpoint;
const Backend = glm.metal_dflash_backend.Backend;
const Target = glm.metal_dflash_backend.Target;
const Model = glm.metal_dflash_model.Model;
const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;
var io: std.Io = undefined;
pub fn now() i128 {
    return std.Io.Clock.awake.now(io).toNanoseconds();
}
pub fn ms(t0: i128, t1: i128) f64 {
    return @as(f64, @floatFromInt(t1 - t0)) / 1e6;
}

const stock = glm.metal_dflash_config.stock;
const streams_mode = @import("glm_metal_dflash_speed_streams.zig");
const resident = @import("glm_metal_dflash_speed_resident.zig");

/// A tensor's shape at config `c`: the drafter's checkpoint layout (weights.zig's checks).
fn dimensions(c: tf.qwen27.dflash.config.Config, name: []const u8, out: *[3]usize) []const usize {
    const h = c.hidden;
    out.* = .{ h, 1, 1 };
    const pick = struct {
        fn two(o: *[3]usize, a: usize, b: usize) []const usize {
            o.* = .{ a, b, 1 };
            return o[0..2];
        }
    };
    if (std.mem.eql(u8, name, "fc.weight")) return pick.two(out, h, c.tapWidth());
    if (std.mem.eql(u8, name, "candidate_selector.hidden_projection.weight")) return pick.two(out, c.rank, h);
    if (std.mem.startsWith(u8, name, "candidate_selector.")) return pick.two(out, c.vocab, c.rank);
    if (std.mem.endsWith(u8, name, "base_kernel")) {
        out.* = .{ 2, 2, h };
        return out[0..3];
    }
    if (std.mem.endsWith(u8, name, "kernel_projection.weight")) return pick.two(out, c.dynamicWidth(), h);
    if (std.mem.endsWith(u8, name, "k_proj.weight") or std.mem.endsWith(u8, name, "v_proj.weight")) return pick.two(out, c.kvWidth(), h);
    if (std.mem.endsWith(u8, name, "q_proj.weight")) return pick.two(out, c.qWidth(), h);
    if (std.mem.endsWith(u8, name, "o_proj.weight")) return pick.two(out, h, c.qWidth());
    if (std.mem.endsWith(u8, name, "gate_proj.weight") or std.mem.endsWith(u8, name, "up_proj.weight")) return pick.two(out, c.intermediate, h);
    if (std.mem.endsWith(u8, name, "down_proj.weight")) return pick.two(out, h, c.intermediate);
    if (std.mem.endsWith(u8, name, "q_norm.weight") or std.mem.endsWith(u8, name, "k_norm.weight")) {
        out.* = .{ c.head_dim, 1, 1 };
        return out[0..1];
    }
    return out[0..1];
}

/// Generated bf16 values: norms 1, base kernels a fixed pattern, the rest small values scaled to the row width.
fn fill(name: []const u8, out: []u16, k: usize, rng: std.Random) void {
    const norm = std.mem.endsWith(u8, name, "norm.weight") or std.mem.endsWith(u8, name, "layernorm.weight");
    const base = std.mem.endsWith(u8, name, "base_kernel");
    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(@max(k, 1))));
    for (out, 0..) |*item, i| {
        const value: f32 = if (norm) 1 else if (base) (if ((i / (out.len / 4)) % 2 == 0) @as(f32, 1) else 0) else (rng.float(f32) * 2 - 1) * 1.7 * scale;
        item.* = @truncate(@as(u32, @bitCast(value)) >> 16);
    }
}

/// The target stand-in: device, queue, GLM's kernels and a 4-bit embedding and `head_bits` head at the real shape.
pub const Harness = struct {
    device: mtl.Device,
    queue: mtl.Queue,
    resident: ?mtl.ResidencySet = null, // --resident-gib: the target's set stand-in, on `queue`
    dummies: std.ArrayList(mtl.Buffer) = .empty,
    k: *kernels.Kernels,
    w: wts.Weights,
    c: cfg.Config,
    fn init(gpa: std.mem.Allocator, head_bits: u8) !Harness {
        const device = try mtl.Device.init();
        const queue = try device.queue();
        var h = Harness{ .device = device, .queue = queue, .k = try kernels.load(gpa, device), .w = .{ .gpa = gpa }, .c = .{ .hidden = stock.hidden, .vocab = stock.vocab, .layers = stock.target_layers } };
        h.w.embed = try matrix(device, stock.vocab, stock.hidden, .{ .bits = 4, .group = 64 }, 11);
        h.w.head = try matrix(device, stock.vocab, stock.hidden, .{ .bits = head_bits, .group = 64 }, 13);
        return h;
    }
    fn target(h: *Harness) Target {
        return .{ .device = h.device, .queue = h.queue, .k = h.k, .w = &h.w, .c = h.c };
    }
};

/// One [n, k] affine matrix: random codes, scales of 1/(8 sqrt k) and biases that centre the codes.
fn matrix(device: mtl.Device, n: usize, k: usize, width: affine.Width, seed: u64) !affine.Affine {
    const wb = affine.wBytes(n, k, width);
    const sb = affine.sBytes(n, k, width);
    const buffer = try device.buffer(wb + 2 * sb, opts);
    var random: std.Random.DefaultPrng = .init(seed);
    random.random().bytes(buffer.slice(u8, wb));
    const words = buffer.slice(u16, (wb + 2 * sb) / 2);
    const s: f32 = 1.0 / (8 * @sqrt(@as(f32, @floatFromInt(k))));
    const mid: f32 = -s * @as(f32, @floatFromInt((@as(u32, 1) << @intCast(width.bits)) / 2));
    for (words[wb / 2 ..][0 .. sb / 2]) |*v| v.* = @truncate(@as(u32, @bitCast(s)) >> 16);
    for (words[wb / 2 + sb / 2 ..]) |*v| v.* = @truncate(@as(u32, @bitCast(mid)) >> 16);
    return .{ .w = .{ .buf = buffer, .off = 0 }, .s = .{ .buf = buffer, .off = wb }, .b = .{ .buf = buffer, .off = wb + sb }, .n = @intCast(n), .k = @intCast(k), .bits = width.bits, .group = width.group };
}

/// The stock drafter over generated tensors; `q4`: its 47 linears read through 4-bit copies.
pub fn create(a: std.mem.Allocator, h: *Harness, q4: bool) !*Model {
    const c = stock;
    const names = try glm.metal_dflash_weights.names(a, c);
    defer glm.metal_dflash_weights.freeNames(a, names);
    const m = try a.create(Model);
    m.allocator = a;
    m.tree_block = 16;
    m.committed_end = 0;
    m.checkpoint = tf.checkpoint_metal.Checkpoint.init(a);
    var random: std.Random.DefaultPrng = .init(5051);
    for (names) |name| {
        var space: [3]usize = undefined;
        const shape = dimensions(c, name, &space);
        var count: usize = 1;
        for (shape) |d| count *= d;
        const buffer = try h.device.buffer(count * 2, opts);
        try m.checkpoint.shards.append(a, .{ .buffer = buffer, .bytes = count * 2 });
        fill(name, buffer.slice(u16, count), if (shape.len == 2) shape[1] else 1, random.random());
        var tensor = tf.checkpoint_metal.Tensor{ .buffer = buffer, .offset = 0, .bytes = count * 2, .dtype = .bf16, .rank = shape.len };
        @memcpy(tensor.shape[0..shape.len], shape);
        try m.checkpoint.tensors.put(a, try a.dupe(u8, name), tensor);
    }
    const t = try a.create(Target);
    t.* = h.target();
    m.backend = try Backend.init(a, t, &m.checkpoint, c);
    m.graph = try glm.metal_dflash_model.loadGraph(a, &m.backend);
    m.pending = try h.device.buffer(@as(usize, c.window) * c.tapWidth() * 2, opts);
    m.session = try tf.qwen27.dflash.session.Session.init(c, 1, 0);
    m.preparation = .{ .mode = .bf16_reference };
    if (q4) {
        var cp = try glm.metal_dflash_prepared.Copies.prepare(a, h.device, &m.checkpoint, c.linearCount());
        errdefer cp.deinit();
        try cp.prepareOpt(h.device, &m.checkpoint, c);
        m.backend.copies = cp;
        m.preparation = .{ .mode = .prepared_q4_reference, .quantization_verified = true, .source_sha256 = tf.affine4.signature() };
    }
    return m;
}

/// `rows` rows of target taps, small random values, seeded so both arms read the same context.
pub fn taps(device: mtl.Device, rows: usize, seed: u64) !mtl.Buffer {
    const n = rows * stock.tapWidth();
    const buffer = try device.buffer(n * 2, opts);
    var random: std.Random.DefaultPrng = .init(seed);
    const rng = random.random();
    for (buffer.slice(u16, n)) |*v| v.* = @truncate(@as(u32, @bitCast((rng.float(f32) * 2 - 1) * 2)) >> 16);
    return buffer;
}

/// The vtable calls a pass makes, each committed and waited alone: GPU ms by operator and shape.
pub const Split = struct {
    inner: op.Ops,
    b: *Backend,
    labels: [48][40]u8 = undefined,
    lens: [48]usize = @splat(0),
    calls: [48]u64 = @splat(0),
    seconds: [48]f64 = @splat(0),
    count: usize = 0,
    passes: u64 = 0,

    fn slot(s: *Split, label: []const u8) usize {
        for (0..s.count) |i| if (std.mem.eql(u8, s.labels[i][0..s.lens[i]], label)) return i;
        @memcpy(s.labels[s.count][0..label.len], label);
        s.lens[s.count] = label.len;
        s.count += 1;
        return s.count - 1;
    }
    /// The open buffer committed and waited, its GPU time charged to `label`; the next op opens a fresh one.
    fn cut(s: *Split, label: []const u8) void {
        const f = &s.b.frame;
        if (!f.active or f.completed) return;
        f.encoder.end();
        f.cb.commit();
        f.cb.wait();
        const i = s.slot(label);
        s.calls[i] += 1;
        s.seconds[i] += f.cb.gpuSeconds();
        f.cb = f.queue.commandBuffer();
        f.encoder = f.cb.compute(.serial);
    }
    fn charge(s: *Split, label: []const u8, seconds: f64) void {
        const i = s.slot(label);
        s.calls[i] += 1;
        s.seconds[i] += seconds;
    }
    pub fn ops(s: *Split) op.Ops {
        return .{ .ptr = s, .vtable = &split_vtable };
    }
    pub fn report(s: *const Split) void {
        var total: f64 = 0;
        for (s.seconds[0..s.count]) |v| total += v;
        const passes: f64 = @floatFromInt(@max(s.passes, 1));
        std.debug.print("  stage (vtable call, own buffer)          calls/pass   ms/pass   share\n", .{});
        for (0..s.count) |i| {
            const per = s.seconds[i] * 1e3 / passes;
            std.debug.print("  {s:<40} {d:>8.1} {d:>9.3} {d:>6.1}%\n", .{ s.labels[i][0..s.lens[i]], @as(f64, @floatFromInt(s.calls[i])) / passes, per, 100 * s.seconds[i] / total });
        }
        std.debug.print("  {s:<40} {s:>8} {d:>9.3}\n", .{ "sum of stages", "", total * 1e3 / passes });
    }
};
fn sp(ptr: *anyopaque) *Split {
    return @ptrCast(@alignCast(ptr));
}
fn sBegin(ptr: *anyopaque) !u64 {
    return sp(ptr).inner.vtable.begin(sp(ptr).inner.ptr);
}
fn sRelease(ptr: *anyopaque, e: u64) void {
    sp(ptr).inner.vtable.release(sp(ptr).inner.ptr, e);
}
fn sProject(ptr: *anyopaque, e: u64, x: op.View, w: ck.Tensor, m: op.Mode) !op.View {
    const s = sp(ptr);
    const y = try s.inner.vtable.project(s.inner.ptr, e, x, w, m);
    var buf: [40]u8 = undefined;
    s.cut(try std.fmt.bufPrint(&buf, "project {d}x{d} rows {d}", .{ w.shape[0], w.shape[1], x.rows }));
    return y;
}
fn sNorm(ptr: *anyopaque, e: u64, x: op.View, p: op.Norm) !op.View {
    const s = sp(ptr);
    const y = try s.inner.vtable.norm(s.inner.ptr, e, x, p);
    var buf: [40]u8 = undefined;
    s.cut(try std.fmt.bufPrint(&buf, "norm {d}x{d} rows {d}", .{ p.heads, p.head_dim, x.rows }));
    return y;
}
fn sEmbed(ptr: *anyopaque, e: u64, l: ck.Linear, ids: []const u32) !op.View {
    const s = sp(ptr);
    const y = try s.inner.vtable.embed(s.inner.ptr, e, l, ids);
    s.cut("embed");
    return y;
}
fn sConv(ptr: *anyopaque, e: u64, x: op.View, p: op.Conv) !op.View {
    const s = sp(ptr);
    const y = try s.inner.vtable.conv(s.inner.ptr, e, x, p);
    s.cut(if (p.residual != null) "conv + residual" else "conv");
    return y;
}
fn sRope(ptr: *anyopaque, e: u64, x: op.View, p: op.Rotary) !op.View {
    const s = sp(ptr);
    const y = try s.inner.vtable.rope(s.inner.ptr, e, x, p);
    var buf: [40]u8 = undefined;
    s.cut(try std.fmt.bufPrint(&buf, "rope {d} heads rows {d}", .{ p.heads, x.rows }));
    return y;
}
fn sAttention(ptr: *anyopaque, e: u64, p: op.Attention) !op.View {
    const s = sp(ptr);
    const y = try s.inner.vtable.attention(s.inner.ptr, e, p);
    s.cut("attention (copies, mask, attend)");
    return y;
}
fn sSwiglu(ptr: *anyopaque, e: u64, g: op.View, u: op.View) !op.View {
    const s = sp(ptr);
    const y = try s.inner.vtable.swiglu(s.inner.ptr, e, g, u);
    s.cut("swiglu");
    return y;
}
fn sMask(ptr: *anyopaque, e: u64, x: op.View, first: u32, rows: u32) !op.View {
    return sp(ptr).inner.vtable.mask_rows(sp(ptr).inner.ptr, e, x, first, rows);
}
fn sHead(ptr: *anyopaque, e: u64, x: op.View) !op.Head {
    const s = sp(ptr);
    const y = try s.inner.vtable.head(s.inner.ptr, e, x);
    s.cut("head (target's, all block rows)");
    return y;
}
fn sLattice(ptr: *anyopaque, e: u64, h: op.Head, p: op.View, c: tf.qwen27.dflash.config.Config) !op.Lattice {
    const s = sp(ptr);
    const l = try s.inner.vtable.lattice(s.inner.ptr, e, h, p, c);
    s.charge("top-k + wait", s.b.frame.cb.gpuSeconds());
    s.passes += 1;
    return l;
}
fn sLattices(ptr: *anyopaque, e: u64, h: op.Head, p: op.View, c: tf.qwen27.dflash.config.Config, block: u32, out: []op.Lattice) !void {
    const s = sp(ptr);
    try s.inner.vtable.lattices(s.inner.ptr, e, h, p, c, block, out);
    s.charge("top-k + wait", s.b.frame.cb.gpuSeconds());
    s.passes += 1;
}
fn sCBegin(ptr: *anyopaque, c: op.Context) !u64 {
    return sp(ptr).inner.vtable.context_begin(sp(ptr).inner.ptr, c);
}
fn sCAppend(ptr: *anyopaque, t: u64, e: u64, layer: u32, k: op.View, v: op.View, p: []const u64) !void {
    const s = sp(ptr);
    try s.inner.vtable.context_append(s.inner.ptr, t, e, layer, k, v, p);
    s.cut("append K/V");
}
fn sCCommit(ptr: *anyopaque, t: u64, e: u64, c: op.Context) !void {
    return sp(ptr).inner.vtable.context_commit(sp(ptr).inner.ptr, t, e, c);
}
fn sCStage(ptr: *anyopaque, t: u64, e: u64, c: op.Context) !void {
    return sp(ptr).inner.vtable.context_stage(sp(ptr).inner.ptr, t, e, c);
}
fn sCAbort(ptr: *anyopaque, t: u64) void {
    sp(ptr).inner.vtable.context_abort(sp(ptr).inner.ptr, t);
}
fn sStream(ptr: *anyopaque, e: u64, k: u32, l: u32, a: op.View, b: op.View, p: []const u64) !void {
    const s = sp(ptr);
    try s.inner.vtable.stream_append(s.inner.ptr, e, k, l, a, b, p);
    s.cut("append K/V");
}
const split_vtable = op.Ops.VTable{ .begin = sBegin, .release = sRelease, .project = sProject, .norm = sNorm, .embed = sEmbed, .conv = sConv, .rope = sRope, .attention = sAttention, .swiglu = sSwiglu, .mask_rows = sMask, .head = sHead, .lattice = sLattice, .lattices = sLattices, .context_begin = sCBegin, .context_append = sCAppend, .context_commit = sCCommit, .context_stage = sCStage, .context_abort = sCAbort, .stream_append = sStream };

pub const Args = struct {
    bits: enum { both, bf16, q4 } = .both,
    context: u32 = 512,
    accepted: u32 = 4,
    rounds: u32 = 40,
    head_bits: u8 = 8,
    dump: ?[]const u8 = null,
    streams: u32 = 0,
    nodes: u32 = 15,
    resident_gib: u32 = 0, // a served target's residency set, stood in for by dummy wired buffers
    resident_buffers: u32 = 1000,
    resident_mode: resident.Mode = .none,
    resident_unset: bool = false, // the dummy buffers without the set: the GLM_RESIDENCY=0 stand-in
    window_mb: u32 = 0, // a target window stand-in before each pass: MiB copied on the target's queue
    overlap: bool = false, // --overlap: the same-buffer pass's bound (needs --resident-gib and --window-mb)
};

pub fn median(values: []f64) f64 {
    std.mem.sort(f64, values, {}, std.sort.asc(f64));
    return values[values.len / 2];
}

/// One arm: prime `context` rows, then rounds of `accepted` new taps + a 16-row block, timed whole and split.
fn arm(a: std.mem.Allocator, h: *Harness, q4: bool, args: Args, hash: *std.crypto.hash.sha2.Sha256) !void {
    const start = now();
    const m = try create(a, h, q4);
    const loaded = ms(start, now()) / 1e3;
    std.debug.print("\n== {s} drafter (load {d:.1} s), context {d}, {d} new taps a round, head {d}-bit ==\n", .{ if (q4) "4-bit" else "bf16", loaded, args.context, args.accepted, args.head_bits });
    const rows = args.context + args.accepted * (2 * args.rounds + 4);
    const source = try taps(h.device, rows, 99);
    const stride = @as(usize, stock.tapWidth()) * 2;
    var fed: u32 = 0;
    while (fed < args.context) {
        const n = @min(args.context - fed, 128);
        try m.absorb(.{ .buf = source, .off = fed * stride }, n);
        fed += n;
    }
    try m.flush();
    try resident.place(a, h.device, m, h.resident, args.resident_mode);
    var trace = tf.gpu_profile.Trace{ .queue = h.queue, .enabled = true, .report_every = 0 };
    m.backend.frame.profiler = &trace;
    var walls: [256]f64 = undefined;
    var gpus: [256]f64 = undefined;
    var block_walls: [256]f64 = undefined;
    var phases: [5][256]f64 = undefined;
    const rounds = @min(args.rounds, 256);
    var lags: [2][256]f64 = undefined;
    for (0..rounds + 2) |r| {
        try m.absorb(.{ .buf = source, .off = fed * stride }, args.accepted);
        fed += args.accepted;
        resident.window(h.queue, h.dummies.items, args.window_mb, r);
        const before = trace.seconds[@backingInt(tf.gpu_profile.Class.drafter)];
        const t0 = now();
        var tree = try m.proposeOptions(@intCast((r * 7919) % 150000), .{ .nodes = args.nodes, .children = if (args.nodes < 15) 1 else 4 });
        const t1 = now();
        const after = trace.seconds[@backingInt(tf.gpu_profile.Class.drafter)];
        defer tree.deinit();
        hash.update(std.mem.sliceAsBytes(tree.tokens));
        hash.update(std.mem.sliceAsBytes(tree.parents));
        // a block alone, no new taps: the frame that drafts without absorbing
        const st = m.backend.stamps;
        if (r >= 2) lags[0][r - 2] = (st.gpu[0] - @as(f64, @floatFromInt(st.first)) / 24e6) * 1e3;
        if (r >= 2) lags[1][r - 2] = (@as(f64, @floatFromInt(st.waited)) / 24e6 - st.gpu[1]) * 1e3;
        const t2 = now();
        var again = try m.proposeOptions(@intCast((r * 104729) % 150000), .{ .nodes = args.nodes, .children = if (args.nodes < 15) 1 else 4 });
        const t3 = now();
        again.deinit();
        if (r >= 2) {
            const k = r - 2;
            const tick = struct {
                fn ms(a_: u64, b_: u64) f64 {
                    return @as(f64, @floatFromInt(b_ -| a_)) / 24e3;
                }
            }.ms;
            phases[0][k] = tick(st.begin, st.commit);
            phases[1][k] = tick(st.commit, st.waited);
            phases[2][k] = (st.gpu[1] - st.gpu[0]) * 1e3;
            phases[3][k] = tick(st.waited, st.read);
            phases[4][k] = ms(t0, t1) - tick(st.begin, st.read);
        }
        if (r < 2) continue; // warm-up
        walls[r - 2] = ms(t0, t1);
        block_walls[r - 2] = ms(t2, t3);
        gpus[r - 2] = (after - before) * 1e3;
    }
    m.backend.frame.profiler = null;
    std.debug.print("  absorb {d} + block 16, wall ms median {d:.3}  (frame GPU {d:.3} ms)\n", .{ args.accepted, median(walls[0..rounds]), median(gpus[0..rounds]) });
    std.debug.print("  pass start after its first commit {d:.3} ms, its end to the wait's return {d:.3} ms (medians)\n", .{ median(lags[0][0..rounds]), median(lags[1][0..rounds]) });
    std.debug.print("  block 16 alone,        wall ms median {d:.3}\n", .{median(block_walls[0..rounds])});
    std.debug.print("  host: encode to commit {d:.3}, commit to wait's end {d:.3} (frame GPU span {d:.3}), lattice read {d:.3}, rest (swap, tree) {d:.3} ms\n", .{ median(phases[0][0..rounds]), median(phases[1][0..rounds]), median(phases[2][0..rounds]), median(phases[3][0..rounds]), median(phases[4][0..rounds]) });
    // the same rounds through the split vtable: every call its own buffer
    var split = Split{ .inner = m.backend.ops(), .b = &m.backend };
    for (0..@min(rounds, 12)) |r| {
        try m.absorb(.{ .buf = source, .off = fed * stride }, args.accepted);
        fed += args.accepted;
        const behind: u32 = @intCast(m.committed_end - m.session.context.end);
        const end = m.session.context.end;
        const slot: u32 = @intCast(end % stock.window);
        var positions: [128]u64 = undefined;
        for (positions[0..behind], 0..) |*p, i| p.* = end + i;
        const view = try m.backend.frame.borrow(.{ .buf = m.pending, .off = @as(usize, slot) * stride }, behind, stock.tapWidth());
        var tree = try m.session.absorbTreeBlock(a, split.ops(), &m.graph, m.preparation, .{ .values = view, .positions = positions[0..behind], .layers = stock.taps, .target_end = m.committed_end }, @intCast((r * 31) % 150000), 16, .{ .nodes = 15 });
        tree.deinit();
    }
    split.report();
    try attentionAlone(h, m);
    if (args.dump) |path| try dumpLattice(a, m, path, q4);
    m.deinit();
}

/// The drafter attention launch alone at a full window: GPU ms of the attend and merge steps, 50 calls a buffer.
fn attentionAlone(h: *Harness, m: *Model) !void {
    const d = &m.backend.draft_attention;
    const keys: u32 = 2047 + 16;
    const big = try h.device.buffer(64 << 20, opts);
    defer big.deinit();
    @memset(big.contents()[0..big.length()], 0x3c);
    const scale: f32 = 0.088;
    const D = glm.metal_dflash_attention;
    const table = [_]D.Stream{.{ .keys = keys, .context = 2047, .at = 1000 }};
    const args = try D.plan(&table, 16, 2048, scale);
    const R = D.Ref;
    const ring = [_]D.Ring{.{ .k = .{ .buf = big, .off = 1 << 20 }, .v = .{ .buf = big, .off = 8 << 20 } }};
    for (0..3) |_| {
        const cb = h.queue.commandBuffer();
        const e = cb.compute(.serial);
        for (0..50) |_| {
            try d.encode(e, args, &table, &ring, R{ .buf = big }, R{ .buf = big, .off = 16 << 20 }, R{ .buf = big, .off = 17 << 20 }, R{ .buf = big, .off = 18 << 20 }, R{ .buf = big, .off = 20 << 20 }, R{ .buf = big, .off = 40 << 20 });
            e.barrier();
        }
        e.end();
        cb.commit();
        cb.wait();
        std.debug.print("  attention alone, 16 rows x {d} keys: {d:.3} ms a call\n", .{ keys, cb.gpuSeconds() * 1e3 / 50 });
    }
}

/// The lattice of a fixed block (candidates, scores, projections) as raw f32/u32 rows: the A/B record of proposals.
fn dumpLattice(a: std.mem.Allocator, m: *Model, path: []const u8, q4: bool) !void {
    var file_name: [256]u8 = undefined;
    const name = try std.fmt.bufPrint(&file_name, "{s}.{s}", .{ path, if (q4) "q4" else "bf16" });
    var bytes: std.ArrayList(u8) = .empty;
    defer bytes.deinit(a);
    for (0..24) |i| {
        try m.flush();
        var l2 = try tf.qwen27.dflash.execution.forwardBlock(m.backend.ops(), &m.graph, m.preparation, m.session.context, @intCast((i * 6151) % 150000), 16);
        defer l2.deinit();
        try bytes.appendSlice(a, std.mem.sliceAsBytes(l2.candidates));
        try bytes.appendSlice(a, std.mem.sliceAsBytes(l2.unary));
        try bytes.appendSlice(a, std.mem.sliceAsBytes(l2.projected));
    }
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = name, .data = bytes.items });
    std.debug.print("  lattices of 24 fixed blocks written to {s}\n", .{name});
}

pub fn main(init: std.process.Init) !void {
    const a = std.heap.page_allocator;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    var args = Args{};
    io = init.io;
    const list = try init.minimal.args.toSlice(init.arena.allocator());
    var at: usize = 1;
    const next = struct {
        fn f(l: []const [:0]const u8, i: *usize) ![]const u8 {
            i.* += 1;
            return if (i.* < l.len) l[i.*] else error.Usage;
        }
    }.f;
    while (at < list.len) : (at += 1) {
        const arg = list[at];
        if (std.mem.eql(u8, arg, "--bits")) args.bits = std.meta.stringToEnum(@TypeOf(args.bits), try next(list, &at)) orelse return error.Usage;
        if (std.mem.eql(u8, arg, "--context")) args.context = try std.fmt.parseInt(u32, try next(list, &at), 10);
        if (std.mem.eql(u8, arg, "--accepted")) args.accepted = try std.fmt.parseInt(u32, try next(list, &at), 10);
        if (std.mem.eql(u8, arg, "--rounds")) args.rounds = try std.fmt.parseInt(u32, try next(list, &at), 10);
        if (std.mem.eql(u8, arg, "--head-bits")) args.head_bits = try std.fmt.parseInt(u8, try next(list, &at), 10);
        if (std.mem.eql(u8, arg, "--dump")) args.dump = try next(list, &at);
        if (std.mem.eql(u8, arg, "--nodes")) args.nodes = try std.fmt.parseInt(u32, try next(list, &at), 10);
        if (std.mem.eql(u8, arg, "--streams")) args.streams = try std.fmt.parseInt(u32, try next(list, &at), 10);
        if (std.mem.eql(u8, arg, "--resident-gib")) args.resident_gib = try std.fmt.parseInt(u32, try next(list, &at), 10);
        if (std.mem.eql(u8, arg, "--resident-buffers")) args.resident_buffers = try std.fmt.parseInt(u32, try next(list, &at), 10);
        if (std.mem.eql(u8, arg, "--resident-unset")) args.resident_unset = true;
        if (std.mem.eql(u8, arg, "--overlap")) args.overlap = true;
        if (std.mem.eql(u8, arg, "--window-mb")) args.window_mb = try std.fmt.parseInt(u32, try next(list, &at), 10);
        if (std.mem.eql(u8, arg, "--resident-mode")) args.resident_mode = std.meta.stringToEnum(@TypeOf(args.resident_mode), try next(list, &at)) orelse return error.Usage;
    }
    if (args.accepted == 0 or args.accepted > 64 or args.context == 0) return error.Usage;
    var h = try Harness.init(a, args.head_bits);
    h.resident = try resident.target(a, h.device, h.queue, &.{ h.w.embed.w.buf, h.w.head.w.buf }, args.resident_gib, @max(1, args.resident_buffers), args.resident_unset, &h.dummies);
    std.debug.print("tf-glm-dflash-speed: {s}, tensor units {}, synthetic weights at incoai's shapes (hidden {d}, mlp {d}, vocab {d})\n", .{ h.device.name(), h.device.tensorUnits(), stock.hidden, stock.intermediate, stock.vocab });
    if (args.overlap) return @import("glm_metal_dflash_speed_overlap.zig").run(a, &h, args.bits != .bf16, args);
    if (args.streams > 0) {
        if (args.streams > 8) return error.Usage;
        if (args.bits != .q4) try streams_mode.run(a, &h, false, args);
        if (args.bits != .bf16) try streams_mode.run(a, &h, true, args);
        return;
    }
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    if (args.bits != .q4) try arm(a, &h, false, args, &hash);
    var hq = std.crypto.hash.sha2.Sha256.init(.{});
    if (args.bits != .bf16) try arm(a, &h, true, args, &hq);
    std.debug.print("\nproposal sha256 bf16 {x} q4 {x}\n", .{ hash.finalResult()[0..8].*, hq.finalResult()[0..8].* });
}
