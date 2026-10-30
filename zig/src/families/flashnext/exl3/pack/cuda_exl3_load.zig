//! EXL3 weights for the Flash Next CUDA family's own loaders (cuda_projection_load.zig, cuda_prompt_load.zig)

const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const wt = @import("../../cuda_prompt_wt.zig");
const projection = @import("../../cuda_projection.zig");
const pool_ = @import("../offload/cuda_exl3_pool.zig");

const Tensor = core.safetensors.Tensor;
const Store = wt.Store;

/// format.py's mul1 marker value (core.exl3_load.markers).
const mul1_magic: u32 = blk: {
    for (core.exl3_load.markers) |m| if (m.codebook == .mul1) break :blk m.value;
    unreachable;
};

/// The EXL3 kernels and scratch one engine's calls share, on the engine's one compute stream.
pub const Ops = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    refs: usize = 1,
    fork: ?*@import("../tp/cuda_exl3_fork.zig").Fork = null,
    dense: ?cuda.exl3.Dense = null,
    experts: ?cuda.exl3_experts.Experts = null,
    /// A part's rows before its columns land in the stitched row (bf16 [rows, part n]).
    tmp: ?cuda.DeviceBuffer = null,
    /// The routed experts' scratch for `moe_rows` rows of 11 slots.
    moe: ?cuda.DeviceBuffer = null,
    moe_rows: usize = 0,
    /// fp16 rows split into bf16 hi and lo [2, rows, k], and their fp32 products [2, rows, n] (an fp16-input Proj).
    halves: ?cuda.DeviceBuffer = null,
    sums: ?cuda.DeviceBuffer = null,
    glue: ?cuda.Module = null,
    split_fn: cuda.Function = undefined,
    /// Sizing maxima over what was loaded (`reserve` allocates every scratch for them once, before any round): the
    max_k: usize = 0,
    max_z: usize = 0,
    max_counters: usize = 0,
    max_part_n: usize = 0,
    max_f16_k: usize = 0,
    max_f16_n: usize = 0,
    moe_layer: ?cuda.exl3_experts.Layer = null,
    /// Pinned [moe_rows, slots] int32: one block's router picks on the host (the offload pool's A/B cut).
    picks: ?cuda.HostBuffer = null,

    pub fn create(gpa: std.mem.Allocator, d: *const cuda.Driver) !*Ops {
        const o = try gpa.create(Ops);
        o.* = .{ .gpa = gpa, .d = d };
        return o;
    }

    pub fn retain(o: *Ops) *Ops {
        o.refs += 1;
        return o;
    }

    pub fn release(o: *Ops) void {
        o.refs -= 1;
        if (o.refs != 0) return;
        @import("../tp/cuda_exl3_fork.zig").release(o);
        if (o.dense) |*x| x.unload();
        if (o.experts) |*x| x.unload();
        if (o.tmp) |*b| b.free();
        if (o.moe) |*b| b.free();
        if (o.halves) |*b| b.free();
        if (o.sums) |*b| b.free();
        if (o.picks) |*b| b.free();
        if (o.glue) |*m| m.unload();
        o.gpa.destroy(o);
    }

    /// The dense EXL3 linear, loaded on first use (no GPU work at checkpoint load).
    pub fn linear(o: *Ops) !*cuda.exl3.Dense {
        if (o.dense == null) o.dense = try cuda.exl3.Dense.load(o.d);
        return &o.dense.?;
    }

    /// fn_exl3_split_f16 (flashnext_exl3.cu), loaded on first use.
    pub fn split(o: *Ops) !cuda.Function {
        if (o.glue == null) {
            if (!cuda.kernels.available) return error.BuiltWithoutKernels;
            var m = try cuda.Module.load(o.d, cuda.kernels.flashnext_exl3);
            errdefer m.unload();
            o.split_fn = try m.function("fn_exl3_split_f16");
            o.glue = m;
        }
        return o.split_fn;
    }

    pub fn grouped(o: *Ops) !*const cuda.exl3_experts.Experts {
        if (o.experts == null) o.experts = try cuda.exl3_experts.Experts.load(o.d);
        return &o.experts.?;
    }

    pub fn noteWeight(o: *Ops, w: cuda.exl3.Weight, row_n: usize) void {
        o.max_k = @max(o.max_k, w.k);
        o.max_z = @max(o.max_z, cuda.exl3.Linear.zBytes(w, 128));
        o.max_counters = @max(o.max_counters, cuda.exl3.Linear.counterBytes(w));
        if (w.n != row_n) o.max_part_n = @max(o.max_part_n, w.n);
    }

    /// Every scratch the EXL3 calls of a pass of up to `rows` rows read, allocated now (after `s` drains), so no call
    pub fn reserve(o: *Ops, s: cuda.Stream, rows: usize, moe_rows: usize, slots: usize) !void {
        try s.synchronize();
        if (o.max_k > 0) {
            const de = try o.linear();
            try grow(o.d, &de.xh, 128 * o.max_k * 2);
            if (o.max_z > 0) try grow(o.d, &de.z, o.max_z);
            try grow(o.d, &de.counters, o.max_counters);
            try @import("../tp/cuda_exl3_fork.zig").reserve(o); // parallel parts' branches (decode projections)
        }
        if (o.max_part_n > 0) try grow(o.d, &o.tmp, rows * o.max_part_n * 2);
        if (o.max_f16_k > 0) {
            _ = try o.split();
            try grow(o.d, &o.halves, 2 * rows * o.max_f16_k * 2);
            try grow(o.d, &o.sums, 2 * rows * o.max_f16_n * 4);
        }
        if (o.moe_layer) |l| {
            _ = try o.grouped();
            try grow(o.d, &o.moe, try cuda.exl3_experts.scratchBytes(l, moe_rows, slots));
            if (o.picks == null) o.picks = try cuda.HostBuffer.alloc(o.d, moe_rows * slots * 4);
        }
    }

    fn grow(d: *const cuda.Driver, slot: *?cuda.DeviceBuffer, bytes: usize) !void {
        if (bytes == 0) return;
        if (slot.*) |b| if (b.len >= bytes) return;
        if (slot.*) |*b| b.free();
        slot.* = null;
        slot.* = try cuda.DeviceBuffer.alloc(d, bytes);
    }

    /// Every scratch buffer's address and size: unchanged across a pass means nothing grew (nor synchronised) in it.
    pub fn fingerprint(o: *const Ops) [8][2]u64 {
        var f: [8][2]u64 = @splat(.{ 0, 0 });
        const de: ?cuda.exl3.Dense = o.dense;
        const bufs = [8]?cuda.DeviceBuffer{ o.tmp, o.moe, o.halves, o.sums, if (de) |x| x.xh else null, if (de) |x| x.z else null, if (de) |x| x.counters else null, null };
        for (bufs, &f) |b, *e| if (b) |x| {
            e.* = .{ x.ptr, x.len };
        };
        if (o.picks) |h| f[7] = .{ @intFromPtr(h.bytes.ptr), h.bytes.len };
        return f;
    }

    /// Device bytes `reserve` holds (for the pool's budget).
    pub fn reservedBytes(o: *const Ops) usize {
        var n: usize = 0;
        for ([_]?cuda.DeviceBuffer{ o.tmp, o.moe, o.halves, o.sums }) |b| if (b) |x| {
            n += x.len;
        };
        if (o.dense) |de| for ([_]?cuda.DeviceBuffer{ de.xh, de.z, de.counters }) |b| if (b) |x| {
            n += x.len;
        };
        return n;
    }

    /// At least `bytes` of part scratch; a growth waits for `s` first (an older call may still read the old buffer).
    pub fn scratch(o: *Ops, s: cuda.Stream, slot: *?cuda.DeviceBuffer, bytes: usize) !u64 {
        if (slot.*) |b| if (b.len >= bytes) return b.ptr;
        try s.synchronize();
        if (slot.*) |*b| b.free();
        slot.* = null;
        slot.* = try cuda.DeviceBuffer.alloc(o.d, bytes);
        return slot.*.?.ptr;
    }
};

/// The store's shared Ops, made on first use; every holder retains it.
pub fn opsOf(store: *Store, d: *const cuda.Driver) !*Ops {
    if (store.exl3_ops == null) store.exl3_ops = try Ops.create(store.gpa, d);
    return store.exl3_ops.?.retain();
}

/// One EXL3 matrix read from the store: words as stored, its width and shape.
pub const Mat = struct {
    trellis: []const u8,
    suh: []const u8,
    svh: []const u8,
    k: usize,
    n: usize,
    halves: u32,
};

/// Whether `prefix` names an EXL3 matrix (its trellis is indexed).
pub fn has(store: *const Store, prefix: []const u8) bool {
    var buf: [256]u8 = undefined;
    return store.has(std.fmt.bufPrint(&buf, "{s}.trellis", .{prefix}) catch return false);
}

/// `prefix`'s trellis, scales and mul1 marker, checked: int16 [K/16, N/16, 16 * bits], fp16 suh [K] and svh [N].
pub fn mat(store: *Store, prefix: []const u8) !Mat {
    var buf: [256]u8 = undefined;
    const t = try store.hold(try std.fmt.bufPrint(&buf, "{s}.trellis", .{prefix}));
    if (t.dtype != .i16 or t.rank != 3 or t.dim(2) % 8 != 0) return error.UnexpectedTensor;
    const halves = t.dim(2) / 8;
    // the mul1 half steps are odd halves; 2..16 halves (1 to 8 bits) are what the kernels instantiate
    if (halves < 2 or halves > 16) return error.UnsupportedExl3Width;
    const k = t.dim(0) * 16;
    const n = t.dim(1) * 16;
    if (k % 128 != 0 or n % 128 != 0) return error.UnexpectedTensor;
    const marker = store.hold(try std.fmt.bufPrint(&buf, "{s}.mul1", .{prefix})) catch return error.NotExl3Mul1;
    if (marker.dtype != .i32 or marker.bytes.len != 4 or std.mem.readInt(u32, marker.bytes[0..4], .little) != mul1_magic) return error.NotExl3Mul1;
    const suh = try store.hold(try std.fmt.bufPrint(&buf, "{s}.suh", .{prefix}));
    const svh = try store.hold(try std.fmt.bufPrint(&buf, "{s}.svh", .{prefix}));
    if (!suh.is(.f16, &.{k}) or !svh.is(.f16, &.{n})) return error.UnexpectedTensor;
    if (store.has(try std.fmt.bufPrint(&buf, "{s}.bias", .{prefix}))) return error.UnexpectedTensor; // none in the text model
    return .{ .trellis = t.bytes, .suh = suh.bytes, .svh = svh.bytes, .k = k, .n = n, .halves = @intCast(halves) };
}

/// One column range of a projection's output row.
pub const Part = struct {
    col: usize,
    n: usize,
    kind: union(enum) { exl3: cuda.exl3.Weight, native: cuda.qmmf.Weight },
};

/// A projection's parts and the device buffers they own.
pub const Proj = struct {
    gpa: std.mem.Allocator,
    ops: *Ops,
    k: usize,
    n: usize,
    parts: []Part,
    /// The rows the pass hands in: bf16, or fp16 bit patterns (an EXL3 pack's n-gram rows, cuda_exl3_ple.zig).
    x: enum { bf16, f16 } = .bf16,
    bufs: std.ArrayList(cuda.DeviceBuffer) = .empty,
    natives: std.ArrayList(projection.Projection) = .empty,

    pub fn destroy(p: *Proj) void {
        for (p.bufs.items) |*b| b.free();
        p.bufs.deinit(p.gpa);
        for (p.natives.items) |*q| q.free();
        p.natives.deinit(p.gpa);
        p.gpa.free(p.parts);
        p.ops.release();
        p.gpa.destroy(p);
    }

    fn up(p: *Proj, d: *const cuda.Driver, bytes: []const u8) !u64 {
        try p.bufs.ensureUnusedCapacity(p.gpa, 1);
        const b = try cuda.DeviceBuffer.fromHost(d, bytes);
        p.bufs.appendAssumeCapacity(b);
        return b.ptr;
    }

    /// An EXL3 matrix as the stored-layout weight (trellis [K/16][N/16][W]: strides(k, n, halves, false)).
    fn exl3Part(p: *Proj, d: *const cuda.Driver, m: Mat) !cuda.exl3.Weight {
        const st = cuda.exl3.strides(m.k, m.n, m.halves, false);
        return .{ .trellis = try p.up(d, m.trellis), .suh = try p.up(d, m.suh), .svh = try p.up(d, m.svh), .k = @intCast(m.k), .n = @intCast(m.n), .halves = m.halves, .codebook = .mul1, .stride_k = st.k, .stride_nb = st.nb };
    }
};

/// A Projection whose rows come from `parts` (EXL3 matrices or dense BF16 rows), `n` columns of `k` inputs.
fn wrap(gpa: std.mem.Allocator, d: *cuda.Driver, store: *Store, prefix: []const u8, names: []const []const u8, n: usize, k: usize) !projection.Projection {
    const p = try gpa.create(Proj);
    p.* = .{ .gpa = gpa, .ops = try opsOf(store, d), .k = k, .n = n, .parts = &.{} };
    errdefer p.destroy();
    var parts: std.ArrayList(Part) = .empty;
    errdefer parts.deinit(gpa);
    var buf: [256]u8 = undefined;
    var col: usize = 0;
    var i: usize = 0;
    while (i < names.len) {
        const name = try std.fmt.bufPrint(&buf, "{s}{s}", .{ prefix, names[i] });
        if (has(store, name)) {
            const m = try mat(store, name);
            if (m.k != k) return error.UnexpectedTensor;
            const w = try p.exl3Part(d, m);
            p.ops.noteWeight(w, n);
            try parts.append(gpa, .{ .col = col, .n = m.n, .kind = .{ .exl3 = w } });
            col += m.n;
            i += 1;
            continue;
        }
        // a run of dense BF16 parts (after the Store's exact widening) packs as one lane weight
        var rows: usize = 0;
        var j = i;
        var bytes: std.ArrayList(u8) = .empty;
        defer bytes.deinit(gpa);
        while (j < names.len) : (j += 1) {
            var nb: [256]u8 = undefined;
            const dense_name = try std.fmt.bufPrint(&nb, "{s}{s}", .{ prefix, names[j] });
            if (has(store, dense_name)) break;
            const t = try store.tensor(try std.fmt.bufPrint(&nb, "{s}{s}.weight", .{ prefix, names[j] }));
            if (t.dtype != .bf16 or t.rank != 2 or t.dim(1) != k or t.bytes.len != t.dim(0) * k * 2) return error.UnexpectedTensor;
            try bytes.appendSlice(gpa, t.bytes);
            rows += t.dim(0);
        }
        try p.natives.ensureUnusedCapacity(gpa, 1);
        const q = try projection.dense(gpa, d, .{ .dtype = .bf16, .rank = 2, .shape = .{ rows, k, 1, 1, 1 }, .bytes = bytes.items });
        p.natives.appendAssumeCapacity(q);
        try parts.append(gpa, .{ .col = col, .n = rows, .kind = .{ .native = q.native.? } });
        col += rows;
        i = j;
    }
    if (col != n) return error.UnexpectedTensor;
    p.parts = try parts.toOwnedSlice(gpa);
    var out = projection.Projection.empty(d);
    out.n = n;
    out.k = k;
    out.exl3 = p;
    return out;
}

/// cuda_projection_load.load's EXL3 branch: `prefix` one EXL3 matrix of n outputs and k inputs.
pub fn load(gpa: std.mem.Allocator, d: *cuda.Driver, store: *Store, prefix: []const u8, n: usize, k: usize) !projection.Projection {
    return wrap(gpa, d, store, prefix, &.{""}, n, k);
}

/// Engine.initRank's head for an EXL3 pack: one rank holds the whole vocabulary.
pub fn head(gpa: std.mem.Allocator, d: *cuda.Driver, store: *Store, world: usize, rank: usize) !?projection.Projection {
    if (!has(store, "language_model.lm_head")) return null;
    if (world != 1 or rank != 0) return error.UnsupportedQuantization;
    if (try @import("../../cuda_head.zig").enabled()) return error.BadHeadSharding;
    return try load(gpa, d, store, "language_model.lm_head", wt.vocab, wt.dims);
}

/// Whether any of `prefix`'s parts is an EXL3 matrix (cuda_projection_load.loadStitched's branch).
pub fn hasAny(store: *const Store, prefix: []const u8, names: []const []const u8) bool {
    var buf: [256]u8 = undefined;
    for (names) |name| if (has(store, std.fmt.bufPrint(&buf, "{s}{s}", .{ prefix, name }) catch return false)) return true;
    return false;
}

/// cuda_projection_load.loadStitched's EXL3 branch: parts side by side in one output row, EXL3 or dense BF16.
pub fn stitched(gpa: std.mem.Allocator, d: *cuda.Driver, store: *Store, prefix: []const u8, names: []const []const u8, n: usize, k: usize) !projection.Projection {
    return wrap(gpa, d, store, prefix, names, n, k);
}

/// cuda_prompt_load's PLE branch: an EXL3 pack's n-gram rows reach key_proj and value_proj as fp16 bit patterns
pub fn pleRows(gpa: std.mem.Allocator, d: *cuda.Driver, store: *Store, key: *projection.Projection, value: *projection.Projection) !void {
    if (store.exl3_ops == null) return; // not an EXL3 pack: the n-gram rows are bf16
    for ([_]*projection.Projection{ key, value }) |q| {
        const native = q.native orelse return error.UnexpectedTensor;
        const p = try gpa.create(Proj);
        p.* = .{ .gpa = gpa, .ops = try opsOf(store, d), .k = q.k, .n = q.n, .parts = &.{}, .x = .f16 };
        errdefer p.destroy();
        p.ops.max_f16_k = @max(p.ops.max_f16_k, q.k);
        p.ops.max_f16_n = @max(p.ops.max_f16_n, q.n);
        p.parts = try gpa.alloc(Part, 1);
        p.parts[0] = .{ .col = 0, .n = q.n, .kind = .{ .native = native } };
        try p.natives.append(gpa, q.*);
        var out = projection.Projection.empty(d);
        out.n = q.n;
        out.k = q.k;
        out.exl3 = p;
        q.* = out;
    }
}

/// One layer's routed experts and the shared expert (id `wt.experts_n`) for exl3_experts.Experts.
pub const Experts = struct {
    gpa: std.mem.Allocator,
    ops: *Ops,
    layer: cuda.exl3_experts.Layer,
    bufs: std.ArrayList(cuda.DeviceBuffer) = .empty,
    /// Each expert's and projection's trellis words on the device, for a pool to rewrite (gate, up, down; 513 each).
    trellis: [3][]u64 = .{ &.{}, &.{}, &.{} },
    halves: [3][]i32 = .{ &.{}, &.{}, &.{} },
    bank: ?cuda.HostBuffer = null,
    /// The pool's Source: every routed expert's gate/up/down bytes inside `bank`, and the shared expert's.
    routed: [][3][]const u8 = &.{},
    shared: [3][]const u8 = .{ &.{}, &.{}, &.{} },
    pool: ?*pool_.Pool = null,
    /// Host time the bank's copies took at load (ns).
    bank_ns: u64 = 0,

    pub fn destroy(x: *Experts) void {
        if (x.pool) |p| {
            p.deinit();
            x.gpa.destroy(p);
        }
        for (x.bufs.items) |*b| b.free();
        x.bufs.deinit(x.gpa);
        for (x.trellis) |t| x.gpa.free(t);
        for (x.halves) |h| x.gpa.free(h);
        x.gpa.free(x.routed);
        if (x.bank) |*b| b.free();
        x.ops.release();
        x.gpa.destroy(x);
    }

    fn up(x: *Experts, d: *const cuda.Driver, bytes: []const u8) !u64 {
        try x.bufs.ensureUnusedCapacity(x.gpa, 1);
        const b = try cuda.DeviceBuffer.fromHost(d, bytes);
        x.bufs.appendAssumeCapacity(b);
        return b.ptr;
    }

    /// Whether the routed experts wait for a pool (offloaded, none attached yet).
    pub fn needsPool(x: *const Experts) bool {
        return x.bank != null and x.pool == null;
    }

    /// Bytes of one pool line of this layer (gate, up and down of its widest expert, 256-aligned).
    pub fn lineBytes(x: *const Experts) usize {
        var bytes: [3]usize = .{ 0, 0, 0 };
        for (x.routed) |e| for (0..3) |j| {
            bytes[j] = @max(bytes[j], e[j].len);
        };
        return cuda.exl3_pool.Layout.init(bytes).line_bytes;
    }

    /// Makes the pool: `lines` device lines, `counts` the family's lifetime use table (null: the pool's own), `layer`
    pub fn attach(x: *Experts, io: std.Io, d: *const cuda.Driver, which: u8, lines: usize, max_rows: usize, counts: ?*[pool_.shape.layers][pool_.shape.experts]u32) !void {
        if (x.bank == null or x.pool != null) return error.Invalid;
        const p = try x.gpa.create(pool_.Pool);
        errdefer x.gpa.destroy(p);
        const l = x.layer;
        p.* = try pool_.Pool.init(x.gpa, io, d, .{ .routed = x.routed, .shared = if (x.shared[0].len == 0) null else x.shared, .pinned_bank = x.bank.?.bytes }, .{ .suh_g = l.suh_g, .suh_u = l.suh_u, .svh_g = l.svh_g, .svh_u = l.svh_u, .suh_d = l.suh_d, .svh_d = l.svh_d }, .{
            .layer = which,
            .lines = lines,
            .dims = wt.dims,
            .width = wt.moe_w,
            .max_rows = max_rows,
            .slots = wt.slots,
            .direct = true, // the bank is pinned: one H2D a projection on the copy engine, no staging memcpy
            .codebook = .mul1,
            .act_mode = l.act_mode,
            .max_passes = pool_.passesFor(pool_.min_lines),
        }, counts);
        x.pool = p;
        x.layer = p.tables();
    }
};

/// Whether `mlp` (".mlp" of a layer) holds EXL3 experts.
pub fn hasExperts(store: *const Store, mlp: []const u8) bool {
    var buf: [256]u8 = undefined;
    return store.has(std.fmt.bufPrint(&buf, "{s}.shared_expert.gate_proj.trellis", .{mlp}) catch return false);
}

/// TF_FLASHNEXT_EXL3_ROUTED=device keeps every routed expert on the device (cuda_exl3_memory.zig reads the same).
pub fn offloaded() bool {
    return @import("cuda_exl3_memory.zig").options().routed_on_host;
}

fn align256(n: usize) usize {
    return (n + 255) / 256 * 256;
}

/// Every routed expert of `mlp`, the shared expert as id 512 (the same 2560 x 640 x 2560 shape): resident, or with
pub fn experts(gpa: std.mem.Allocator, d: *cuda.Driver, store: *Store, mlp: []const u8, world: usize, rank: usize) !*Experts {
    if (world != 1 or rank != 0) return error.UnsupportedQuantization; // one rank holds all 513
    const count = wt.experts_n + 1;
    const offload = offloaded();
    const x = try gpa.create(Experts);
    x.* = .{ .gpa = gpa, .ops = try opsOf(store, d), .layer = undefined };
    errdefer x.destroy();
    for (&x.trellis, &x.halves) |*t, *h| {
        t.* = try gpa.alloc(u64, count);
        @memset(t.*, 0);
        h.* = try gpa.alloc(i32, count);
        @memset(h.*, 0);
    }
    const projs = [3][]const u8{ "gate_proj", "up_proj", "down_proj" };
    const ks = [3]usize{ wt.dims, wt.dims, wt.moe_w };
    const ns = [3]usize{ wt.moe_w, wt.moe_w, wt.dims };
    var scales: [6]std.ArrayList(u8) = @splat(.empty);
    defer for (&scales) |*l| l.deinit(gpa);
    var buf: [256]u8 = undefined;
    const nameOf = struct {
        fn f(b: []u8, m: []const u8, e: usize, proj: []const u8) ![]const u8 {
            if (e == wt.experts_n) return std.fmt.bufPrint(b, "{s}.shared_expert.{s}", .{ m, proj });
            return std.fmt.bufPrint(b, "{s}.experts.{d}.{s}", .{ m, e, proj });
        }
    }.f;
    var at: usize = 0;
    if (offload) {
        // the bank's size first (headers only), then one pinned region for the layer
        var total: usize = 0;
        for (0..count) |e| {
            for (projs) |proj| total += align256((try mat(store, try nameOf(&buf, mlp, e, proj))).trellis.len);
            store.release();
        }
        x.bank = try cuda.HostBuffer.alloc(d, total);
        x.routed = try gpa.alloc([3][]const u8, wt.experts_n);
    }
    const t0 = std.Io.Clock.awake.now(store.io).toNanoseconds();
    for (0..count) |e| {
        for (projs, 0..) |proj, j| {
            const m = try mat(store, try nameOf(&buf, mlp, e, proj));
            if (m.k != ks[j] or m.n != ns[j]) return error.UnexpectedTensor;
            if (x.bank) |bank| {
                const dst = bank.bytes[at..][0..m.trellis.len];
                @memcpy(dst, m.trellis);
                at += align256(m.trellis.len);
                if (e == wt.experts_n) x.shared[j] = dst else x.routed[e][j] = dst;
            } else x.trellis[j][e] = try x.up(d, m.trellis);
            x.halves[j][e] = @intCast(m.halves);
            try scales[2 * j].appendSlice(gpa, m.suh);
            try scales[2 * j + 1].appendSlice(gpa, m.svh);
        }
        // each expert's mapped pages are read once: release them so the 513 experts never pin a shard's worth
        store.release();
    }
    x.bank_ns = @intCast(std.Io.Clock.awake.now(store.io).toNanoseconds() - t0);
    x.layer = .{
        // offloaded: the pool's tables replace these at attach (0 until then; a pass refuses an unattached layer)
        .gate = if (offload) 0 else try x.up(d, std.mem.sliceAsBytes(x.trellis[0])),
        .up = if (offload) 0 else try x.up(d, std.mem.sliceAsBytes(x.trellis[1])),
        .down = if (offload) 0 else try x.up(d, std.mem.sliceAsBytes(x.trellis[2])),
        .gate_k2 = if (offload) 0 else try x.up(d, std.mem.sliceAsBytes(x.halves[0])),
        .up_k2 = if (offload) 0 else try x.up(d, std.mem.sliceAsBytes(x.halves[1])),
        .down_k2 = if (offload) 0 else try x.up(d, std.mem.sliceAsBytes(x.halves[2])),
        .suh_g = try x.up(d, scales[0].items),
        .svh_g = try x.up(d, scales[1].items),
        .suh_u = try x.up(d, scales[2].items),
        .svh_u = try x.up(d, scales[3].items),
        .suh_d = try x.up(d, scales[4].items),
        .svh_d = try x.up(d, scales[5].items),
        .experts = count,
        .dims = wt.dims,
        .width = wt.moe_w,
        .codebook = .mul1,
        // fp32 SwiGLU between the grouped GEMMs (GLM's mode 0 rounds to bf16 there)
        .act_mode = 1,
    };
    x.ops.moe_layer = x.layer;
    return x;
}

test "exl3: the mul1 marker value is format.py's" {
    try std.testing.expectEqual(@as(u32, 0x83DCD12D), mul1_magic);
}
