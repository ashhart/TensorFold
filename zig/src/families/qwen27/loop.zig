//! Looped SenseNova-Loop-9B weight loading, BF16/AWQ linear dispatch and two-stream mHC hooks.
const std = @import("std");
const mtl = @import("metal");
const core = @import("core");
const cp = core.checkpoint_metal;
const sn_cfg = core.sensenova_loop.config;
const sn_awq = core.sensenova_loop.awq;
const linear_gpu = core.sensenova_loop.linear_gpu;
const mhc_metal = core.sensenova_loop.mhc_metal;
const profile = core.gpu_profile;
const cfg = @import("config.zig");
const wts = @import("gpu_weights.zig");
const projection = @import("projection.zig");
const widths = @import("widths.zig");
const Frame = @import("gpu_frame.zig").Frame;

pub const SnMat = union(enum) { bf16: linear_gpu.Bf16Matrix, awq: linear_gpu.AwqMatrix };
pub const SnRun = struct { mat: SnMat, first: u32 };
pub const SnLinear = struct { runs: []const SnRun, n: u32, k: u32 };
pub const MhcLayer = struct { attn_w: cp.Tensor, attn_b: cp.Tensor, ffn_w: cp.Tensor, ffn_b: cp.Tensor };

const PartSpec = struct { name: []const u8, n: usize, width: widths.Width };
pub const TapHook = struct { ptr: *anyopaque, cb: *const fn (*anyopaque, mtl.ComputeEncoder, usize, projection.Ref, u32) anyerror!void };

const stacked_modules = [_][]const u8{
    ".mlp.gate_proj",
    ".mlp.up_proj",
    ".self_attn.q_proj",
    ".self_attn.k_proj",
    ".self_attn.v_proj",
    ".linear_attn.in_proj_qkv",
    ".linear_attn.in_proj_z",
    ".linear_attn.in_proj_b",
    ".linear_attn.in_proj_a",
};

/// Whether a tensor is repacked, dtype-rounded, or ignored during SenseNova shard loading.
pub fn repackedSenseNova(name: []const u8, _: usize) bool {
    if (sn_cfg.isSkippedTensor(name)) return true;
    const dot = std.mem.lastIndexOfScalar(u8, name, '.') orelse return false;
    const leaf = name[dot + 1 ..];
    if (std.mem.eql(u8, leaf, "qweight") or std.mem.eql(u8, leaf, "qzeros") or
        std.mem.eql(u8, leaf, "scales") or std.mem.eql(u8, leaf, "input_scale")) return true;
    if (std.mem.eql(u8, leaf, "A_log") or std.mem.eql(u8, leaf, "dt_bias")) return true;
    if (!std.mem.eql(u8, leaf, "weight")) return false;
    const module = name[0..dot];
    for (stacked_modules) |suffix| if (std.mem.endsWith(u8, module, suffix)) return true;
    return false;
}

/// Dispatch one SenseNova-Loop linear projection (BF16, AWQ, or mixed runs), optionally split across output buffers.
pub fn encodeSnSplit(sn_gpu: *linear_gpu.LinearGpu, profiler: ?*profile.Trace, e: mtl.ComputeEncoder, s: SnLinear, x: projection.Ref, xs_opt: ?linear_gpu.Sums, y: projection.Ref, rows: u32, ldy: u32, prompt: bool, splits: linear_gpu.OutSplits) !void {
    if (s.runs.len == 0) return error.BadCoreProjectionRows;
    const Seg = struct { lo: u32, hi: u32, ref: linear_gpu.Ref, ldy: u32 };
    var segs: [3]Seg = undefined;
    var seg_n: usize = 0;
    const s0_hi: u32 = if (splits.y1 != null) splits.split1 else s.n;
    segs[seg_n] = .{ .lo = 0, .hi = s0_hi, .ref = .{ .buf = y.buffer, .off = y.offset }, .ldy = ldy };
    seg_n += 1;
    if (splits.y1) |y1| {
        const s1_hi: u32 = if (splits.y2 != null) splits.split2 else s.n;
        segs[seg_n] = .{ .lo = splits.split1, .hi = s1_hi, .ref = y1, .ldy = splits.ldy1 };
        seg_n += 1;
        if (splits.y2) |y2| {
            segs[seg_n] = .{ .lo = splits.split2, .hi = s.n, .ref = y2, .ldy = splits.ldy2 };
            seg_n += 1;
        }
    }
    var span = profile.begin(profiler, e, .dense);
    const first_awq = s.runs[0].mat == .awq;
    const sk: usize = if (prompt) 1 else linear_gpu.decodeSk(sn_gpu.tensor_units, first_awq, s.n, s.k);
    span.shape = .{ .n = s.n, .k = s.k, .sk = sk, .bits = if (first_awq) 4 else 16 };
    errdefer span.cancel();
    const xr = linear_gpu.Ref{ .buf = x.buffer, .off = x.offset };
    for (s.runs) |run| {
        const start = run.first;
        const run_n: u32 = switch (run.mat) {
            .bf16 => |m| m.n,
            .awq => |m| m.n,
        };
        const end = start + run_n;
        var hit: [3]Seg = undefined;
        var hit_n: usize = 0;
        for (segs[0..seg_n]) |seg| if (seg.hi > start and seg.lo < end) {
            hit[hit_n] = seg;
            hit_n += 1;
        };
        if (hit_n == 0) return error.BadCoreProjectionRows;
        const ry = linear_gpu.Ref{ .buf = hit[0].ref.buf, .off = hit[0].ref.off + @as(usize, start - hit[0].lo) * 2 };
        var rsplits = linear_gpu.OutSplits{};
        if (hit_n > 1) {
            rsplits.split1 = hit[1].lo - start;
            rsplits.y1 = hit[1].ref;
            rsplits.ldy1 = hit[1].ldy;
        }
        if (hit_n > 2) {
            rsplits.split2 = hit[2].lo - start;
            rsplits.y2 = hit[2].ref;
            rsplits.ldy2 = hit[2].ldy;
        }
        switch (run.mat) {
            .bf16 => |m| try sn_gpu.encodeBf16Split(span.encoder, m, xr, ry, rows, hit[0].ldy, prompt, null, rsplits),
            .awq => |m| try sn_gpu.encodeAwqSplit(span.encoder, m, xr, xs_opt, ry, rows, hit[0].ldy, prompt, null, rsplits),
        }
    }
    try span.end();
}

/// Dispatch one SenseNova-Loop linear projection (BF16, AWQ, or mixed runs).
pub fn encodeSn(sn_gpu: *linear_gpu.LinearGpu, profiler: ?*profile.Trace, e: mtl.ComputeEncoder, s: SnLinear, x: projection.Ref, xs_opt: ?linear_gpu.Sums, y: projection.Ref, rows: u32, prompt: bool) !void {
    return encodeSnSplit(sn_gpu, profiler, e, s, x, xs_opt, y, rows, s.n, prompt, .{});
}

/// Dispatch a SenseNova-Loop `projection.Linear` across up to 3 output destinations using active fused `sn_xs128` sums.
pub fn quantSplit(kernels: anytype, e: mtl.ComputeEncoder, p: projection.Linear, x: projection.Ref, y: projection.Ref, rows: u32, ldy: u32, splits: linear_gpu.OutSplits) !void {
    const s = p.sn orelse return error.BadCoreProjectionRows;
    const sn_xs = if (kernels.sn_xs128) |xs128| if (std.meta.eql(xs128.x.buffer, x.buffer) and xs128.x.offset == x.offset) xs128.sums else null else null;
    return encodeSnSplit(&kernels.sn.?, kernels.profiler, e, s, x, sn_xs, y, rows, ldy, kernels.prompt, splits);
}

/// Dispatch attention Q/K/V via single stacked `OutSplits` GEMM/MV when `.kv` is folded, else split Q + KV + unstack.
pub fn encodeAttnProj(kernels: anytype, glue: anytype, e: mtl.ComputeEncoder, weights: wts.Attention, f: *Frame, rows: u32, q_dim: u32, kv_dim: u32) !void {
    if (weights.kv.n == 0) {
        const key = f.get(.key);
        const val = f.get(.value);
        try quantSplit(kernels, e, weights.q, f.get(.input), f.get(.q_gate), rows, q_dim, .{
            .split1 = q_dim,
            .y1 = .{ .buf = key.buffer, .off = key.offset },
            .ldy1 = kv_dim,
            .split2 = q_dim + kv_dim,
            .y2 = .{ .buf = val.buffer, .off = val.offset },
            .ldy2 = kv_dim,
        });
        e.barrier();
        return;
    }
    const sums = @import("forward.zig").sums(f, .sums, rows);
    try kernels.quant(e, weights.q, f.get(.input), sums, f.get(.dims), f.get(.q_gate), rows);
    try kernels.quant(e, weights.kv, f.get(.input), sums, f.get(.dims), f.get(.mixed), rows);
    e.barrier();
    try glue.unstack(e, f.get(.mixed), f.get(.key), .{ .rows = rows, .width = kv_dim, .stride = 2 * kv_dim, .offset = 0 });
    try glue.unstack(e, f.get(.mixed), f.get(.value), .{ .rows = rows, .width = kv_dim, .stride = 2 * kv_dim, .offset = kv_dim });
    e.barrier();
}

pub fn callCfg(c: cfg.Config, use_xs: bool, rows: u32) mhc_metal.PreNormCall {
    const stride = if (use_xs) @import("quant_gpu.zig").padded(rows) else 0;
    return .{ .iters = c.mhc_iterations, .shift_norm = c.shift_norms, .xs_group = if (use_xs) 128 else 0, .xs_stride = stride, .pre_scale = c.mhc_pre_scale, .eps = c.eps };
}

fn mref(r: projection.Ref) mhc_metal.Ref {
    return .{ .buf = r.buffer, .off = r.offset };
}

fn tref(t: cp.Tensor) mhc_metal.Ref {
    return .{ .buf = t.buffer, .off = t.offset };
}

fn setInputXs128(kernels: anytype, f: *Frame, rows: u32, enabled: bool) ?mhc_metal.Ref {
    if (!enabled) {
        kernels.sn_xs128 = null;
        return null;
    }
    const xr = f.get(.input);
    const sr = f.get(.sums);
    const stride = @import("quant_gpu.zig").padded(rows);
    kernels.sn_xs128 = .{ .x = xr, .sums = .{ .ref = .{ .buf = sr.buffer, .off = sr.offset }, .stride = stride } };
    return mref(sr);
}

/// Enter two-stream residual if `enter`, then compute mHC coefficients and collapsed pre-norm (unless fused from prev).
pub fn preMhc(mhc: *const mhc_metal.Pipelines, kernels: anytype, c: cfg.Config, index: usize, layer: wts.Layer, ml: MhcLayer, hidden: *projection.Ref, residual: *projection.Ref, pending: *?projection.Ref, f: *Frame, rows: u32, e: mtl.ComputeEncoder, tap: ?TapHook) !void {
    if (index > 0 and (c.mhc_roles[index] == .inner or c.mhc_roles[index] == .exit) and
        (c.mhc_roles[index - 1] == .enter or c.mhc_roles[index - 1] == .inner)) return;
    const h: u32 = @intCast(c.hidden);
    const want_xs = kernels.device.tensorUnits() and layer.gu.bits == 4;
    const xs = setInputXs128(kernels, f, rows, want_xs);
    var span = profile.begin(kernels.profiler, e, .norm);
    errdefer span.cancel();
    const enc = span.encoder;
    const enters = c.mhc_roles[index] == .enter or c.mhc_roles[index] == .both;
    const need_tap = enters and tap != null and pending.* != null;
    if (enters) {
        try mhc_metal.enter(mhc, enc, mref(hidden.*), if (pending.*) |p| mref(p) else null, mref(f.get(.h_streams)), if (need_tap) mref(residual.*) else null, rows, h);
        enc.barrier();
        pending.* = null;
    }
    try mhc_metal.preNorm(mhc, enc, mref(f.get(.h_streams)), tref(ml.attn_w), tref(ml.attn_b), tref(layer.norm), mref(f.get(.mhc_coeff)), mref(f.get(.input)), xs, rows, h, callCfg(c, want_xs, rows));
    enc.barrier();
    try span.end();
    if (need_tap) try tap.?.cb(tap.?.ptr, e, index - 1, residual.*, rows);
}

/// Fuse attention residual merge into two streams with FFN mHC coefficients and pre-norm.
pub fn postMhcMid(mhc: *const mhc_metal.Pipelines, kernels: anytype, c: cfg.Config, layer: wts.Layer, ml: MhcLayer, f: *Frame, rows: u32, e: mtl.ComputeEncoder) !void {
    const h: u32 = @intCast(c.hidden);
    const hs = mref(f.get(.h_streams));
    const co = mref(f.get(.mhc_coeff));
    const want_xs = kernels.device.tensorUnits() and layer.gu.bits == 4;
    const xs = setInputXs128(kernels, f, rows, want_xs);
    var span = profile.begin(kernels.profiler, e, .norm);
    errdefer span.cancel();
    const enc = span.encoder;
    try mhc_metal.postAndPreNorm(mhc, enc, hs, mref(f.get(.output)), co, tref(ml.ffn_w), tref(ml.ffn_b), tref(layer.post_norm), hs, co, mref(f.get(.input)), xs, rows, h, callCfg(c, want_xs, rows));
    enc.barrier();
    try span.end();
}

/// Merge FFN branch into next layer's pre-norm when continuing in two streams, else mean-collapse on exit.
pub fn postMhcEnd(mhc: *const mhc_metal.Pipelines, kernels: anytype, c: cfg.Config, index: usize, next_layer: ?wts.Layer, next_ml: ?MhcLayer, hidden: *projection.Ref, pending: *?projection.Ref, f: *Frame, rows: u32, e: mtl.ComputeEncoder, tap: ?TapHook) !void {
    const h: u32 = @intCast(c.hidden);
    const exit = c.mhc_roles[index] == .exit or c.mhc_roles[index] == .both;
    const fuse_next = !exit and next_layer != null and next_ml != null and index + 1 < c.layers and
        (c.mhc_roles[index + 1] == .inner or c.mhc_roles[index + 1] == .exit);
    var span = profile.begin(kernels.profiler, e, .norm);
    errdefer span.cancel();
    const enc = span.encoder;
    const hs = mref(f.get(.h_streams));
    const co = mref(f.get(.mhc_coeff));
    if (fuse_next) {
        const nl = next_layer.?;
        const nml = next_ml.?;
        const want_xs = kernels.device.tensorUnits() and nl.gu.bits == 4;
        const xs = setInputXs128(kernels, f, rows, want_xs);
        try mhc_metal.postAndPreNorm(mhc, enc, hs, mref(f.get(.output)), co, tref(nml.attn_w), tref(nml.attn_b), tref(nl.norm), hs, co, mref(f.get(.input)), xs, rows, h, callCfg(c, want_xs, rows));
    } else {
        kernels.sn_xs128 = null;
        const dst = if (exit) hidden.* else f.get(.h_streams);
        try mhc_metal.postMerge(mhc, enc, hs, mref(f.get(.output)), co, mref(dst), rows, h, exit);
    }
    enc.barrier();
    try span.end();
    pending.* = null;
    if (exit) if (tap) |t| try t.cb(t.ptr, e, index, hidden.*, rows);
}

fn namespace(ck: *const cp.Checkpoint) !struct { text: []const u8, head: []const u8 } {
    const text: []const u8 = if (ck.has("model.language_model.embed_tokens.weight"))
        "model.language_model."
    else if (ck.has("language_model.model.embed_tokens.weight"))
        "language_model.model."
    else if (ck.has("model.embed_tokens.weight"))
        "model."
    else
        return error.MissingTextNamespace;
    const head = if (ck.has("lm_head.weight") or ck.has("lm_head.qweight")) "lm_head" else try std.fmt.allocPrint(ck.allocator, "{s}lm_head", .{text});
    return .{ .text = text, .head = head };
}

fn checked(t: cp.Tensor, dtype: cp.DType, shape: []const usize) !cp.Tensor {
    if (t.dtype != dtype or t.rank != shape.len or !std.mem.eql(usize, t.shape[0..t.rank], shape)) return error.BadNativeTensor;
    return t;
}

fn checkedInt32(t: cp.Tensor, shape: []const usize) !cp.Tensor {
    if ((t.dtype != .i32 and t.dtype != .u32) or t.rank != shape.len or !std.mem.eql(usize, t.shape[0..t.rank], shape)) return error.BadNativeTensor;
    return t;
}

fn checkedConv(t: cp.Tensor, n: usize, kernel: usize) !cp.Tensor {
    if (t.dtype != .bf16) return error.BadNativeTensor;
    const ok = (t.rank == 2 and t.shape[0] == n and t.shape[1] == kernel) or
        (t.rank == 3 and t.shape[0] == n and ((t.shape[1] == kernel and t.shape[2] == 1) or (t.shape[1] == 1 and t.shape[2] == kernel)));
    if (!ok) return error.BadNativeTensor;
    var out = t;
    out.rank = 3;
    out.shape = .{ n, kernel, 1, 1, 1 };
    return out;
}

fn roundF32ToBf16(x: f32) u16 {
    const u: u32 = @bitCast(x);
    if ((u & 0x7fff_ffff) > 0x7f80_0000) return @truncate((u >> 16) | 0x0040);
    return @truncate((u +% 0x7fff +% ((u >> 16) & 1)) >> 16);
}

fn roundScalarBf16(a: std.mem.Allocator, device: mtl.Device, ck: *const cp.Checkpoint, stage: *cp.Stage, owned: *std.ArrayList(mtl.Buffer), name: []const u8, n: usize) !cp.Tensor {
    _ = a;
    try stage.reserve(ck, &.{name});
    defer stage.reset();
    const src = try stage.read(ck, name);
    if (src.rank != 1 or src.shape[0] != n or (src.dtype != .f32 and src.dtype != .bf16)) return error.BadNativeTensor;
    const buf = try device.buffer(n * 2, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    owned.append(ck.allocator, buf) catch |err| {
        buf.deinit();
        return err;
    };
    const dst = buf.slice(u16, n);
    if (src.dtype == .bf16) {
        @memcpy(dst, src.host(u16)[0..n]);
    } else {
        for (src.host(f32)[0..n], dst) |v, *d| d.* = roundF32ToBf16(v);
    }
    return .{ .buffer = buf, .offset = 0, .bytes = n * 2, .dtype = .bf16, .shape = .{ n, 1, 1, 1, 1 }, .rank = 1 };
}

fn scaleDType(dt: cp.DType) !sn_awq.ScaleDType {
    return switch (dt) {
        .f16 => .f16,
        .bf16 => .bf16,
        .f32 => .f32,
        else => error.BadNativeTensor,
    };
}

fn ownBuffer(a: std.mem.Allocator, owned: *std.ArrayList(mtl.Buffer), b: mtl.Buffer) !void {
    owned.append(a, b) catch |err| {
        b.deinit();
        return err;
    };
}

fn loadBf16Run(a: std.mem.Allocator, device: mtl.Device, ck: *const cp.Checkpoint, stage: *cp.Stage, owned: *std.ArrayList(mtl.Buffer), parts: []const PartSpec, k: usize) !linear_gpu.Bf16Matrix {
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const ta = arena.allocator();
    if (parts.len == 1) {
        const wname = try std.mem.concat(ta, u8, &.{ parts[0].name, ".weight" });
        if (ck.tensors.contains(wname)) {
            const t = try checked(try ck.get(wname), .bf16, &.{ parts[0].n, k });
            if (t.offset % 16 == 0) return linear_gpu.LinearGpu.prepareBf16(t.buffer, t.offset, @intCast(parts[0].n), @intCast(k));
        }
    }
    const names = try ta.alloc([]const u8, parts.len);
    var total_n: usize = 0;
    for (parts, names) |p, *nm| {
        nm.* = try std.mem.concat(ta, u8, &.{ p.name, ".weight" });
        total_n += p.n;
    }
    try stage.reserve(ck, names);
    defer stage.reset();
    const buf = try device.buffer(total_n * k * 2, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    try ownBuffer(a, owned, buf);
    var dst_row: usize = 0;
    const dst = buf.contents();
    for (parts, names) |p, nm| {
        const t = try checked(try stage.read(ck, nm), .bf16, &.{ p.n, k });
        const row_bytes = p.n * k * 2;
        @memcpy(dst[dst_row * k * 2 ..][0..row_bytes], t.buffer.contents()[t.offset..][0..row_bytes]);
        dst_row += p.n;
    }
    return linear_gpu.LinearGpu.prepareBf16(buf, 0, @intCast(total_n), @intCast(k));
}

fn loadAwqRun(a: std.mem.Allocator, ck: *const cp.Checkpoint, stage: *cp.Stage, sn_gpu: *const linear_gpu.LinearGpu, owned: *std.ArrayList(mtl.Buffer), parts: []const PartSpec, k: usize, out_runs: *std.ArrayList(SnRun), first: *u32) !void {
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const ta = arena.allocator();
    var names: std.ArrayList([]const u8) = .empty;
    const has_in_scale = try ta.alloc(bool, parts.len);
    for (parts, has_in_scale) |p, *hs| {
        for ([_][]const u8{ ".qweight", ".qzeros", ".scales" }) |suf| try names.append(ta, try std.mem.concat(ta, u8, &.{ p.name, suf }));
        const is_name = try std.mem.concat(ta, u8, &.{ p.name, ".input_scale" });
        hs.* = ck.has(is_name);
        if (hs.*) try names.append(ta, is_name);
    }
    try stage.reserve(ck, names.items);
    defer stage.reset();
    const awq_parts = try ta.alloc(sn_awq.Part, parts.len);
    for (parts, awq_parts, has_in_scale) |p, *ap, hs| {
        const qw_name = try std.mem.concat(ta, u8, &.{ p.name, ".qweight" });
        const qz_name = try std.mem.concat(ta, u8, &.{ p.name, ".qzeros" });
        const sc_name = try std.mem.concat(ta, u8, &.{ p.name, ".scales" });
        const qw = try checkedInt32(try stage.read(ck, qw_name), &.{ k, p.n / 8 });
        const qz = try checkedInt32(try stage.read(ck, qz_name), &.{ k / 128, p.n / 8 });
        const sc = try stage.read(ck, sc_name);
        const sdt = try scaleDType(sc.dtype);
        if (sc.rank != 2 or sc.shape[0] != k / 128 or sc.shape[1] != p.n) return error.BadNativeTensor;
        var in_scale: ?[]const f32 = null;
        if (hs) {
            const is_t = try checked(try stage.read(ck, try std.mem.concat(ta, u8, &.{ p.name, ".input_scale" })), .f32, &.{k});
            in_scale = is_t.host(f32);
        }
        ap.* = .{ .n = p.n, .k = k, .qweight = qw.host(u32), .qzeros = qz.host(u32), .scales = sc.buffer.contents()[sc.offset..][0..sc.bytes], .scales_dtype = sdt, .input_scale = in_scale };
    }
    const mats = try sn_gpu.prepareAwq(awq_parts);
    defer sn_gpu.allocator.free(mats);
    for (mats) |m| {
        var copy = m;
        copy.owns = false;
        owned.appendSlice(a, &.{ copy.w, copy.sb }) catch |err| {
            var tmp = m;
            tmp.deinit();
            return err;
        };
        if (copy.input_scale) |ib| owned.append(a, ib) catch |err| {
            ib.deinit();
            return err;
        };
        try out_runs.append(a, .{ .mat = .{ .awq = copy }, .first = first.* });
        first.* += copy.n;
    }
}

fn loadSnLinear(a: std.mem.Allocator, device: mtl.Device, ck: *const cp.Checkpoint, stage: *cp.Stage, sn_gpu: *const linear_gpu.LinearGpu, owned: *std.ArrayList(mtl.Buffer), sn_runs: *std.ArrayList([]SnRun), parts: []const PartSpec, k: usize) !projection.Linear {
    if (parts.len == 0) return error.BadProjectionShape;
    var runs: std.ArrayList(SnRun) = .empty;
    defer runs.deinit(a);
    var first: u32 = 0;
    var i: usize = 0;
    while (i < parts.len) {
        const kind = parts[i].width.kind;
        var j = i + 1;
        while (j < parts.len and parts[j].width.kind == kind) : (j += 1) {}
        switch (kind) {
            .bf16 => {
                const mat = try loadBf16Run(a, device, ck, stage, owned, parts[i..j], k);
                try runs.append(a, .{ .mat = .{ .bf16 = mat }, .first = first });
                first += mat.n;
            },
            .awq => try loadAwqRun(a, ck, stage, sn_gpu, owned, parts[i..j], k, &runs, &first),
            .affine => return error.UnqualifiedQwenFormat,
        }
        i = j;
    }
    const owned_runs = try runs.toOwnedSlice(a);
    sn_runs.append(a, owned_runs) catch |err| {
        a.free(owned_runs);
        return err;
    };
    const dummy: projection.Ref = .{ .buffer = owned.items[0] };
    return .{ .words = dummy, .pairs = dummy, .n = first, .k = @intCast(k), .tile = 32, .slices = 1, .bits = if (parts[0].width.kind == .awq) 4 else 16, .group = parts[0].width.group, .sn = .{ .runs = owned_runs, .n = first, .k = @intCast(k) } };
}

fn zeroSnLinear(owned: *const std.ArrayList(mtl.Buffer), k: usize) projection.Linear {
    const dummy: projection.Ref = .{ .buffer = owned.items[0] };
    return .{ .words = dummy, .pairs = dummy, .n = 0, .k = @intCast(k), .tile = 32, .slices = 1, .sn = .{ .runs = &.{}, .n = 0, .k = @intCast(k) } };
}

fn loadPhysicalLayer(a: std.mem.Allocator, device: mtl.Device, ck: *const cp.Checkpoint, stage: *cp.Stage, sn_gpu: *const linear_gpu.LinearGpu, owned: *std.ArrayList(mtl.Buffer), sn_runs: *std.ArrayList([]SnRun), c: cfg.Config, w: [17]widths.Width, prefix: []const u8, i: usize, phys_n: usize, need_mhc: bool) !struct { layer: wts.Layer, mhc: MhcLayer } {
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const ta = arena.allocator();
    const lp = try std.fmt.allocPrint(ta, "{s}layers.{d}.", .{ prefix, i });
    const cat = struct {
        fn s(alloc: std.mem.Allocator, p: []const u8, sub: []const u8) ![]const u8 {
            return std.mem.concat(alloc, u8, &.{ p, sub });
        }
    }.s;
    const M = widths.Member;
    const kind = c.physKind(i);
    const mixer: @FieldType(wts.Layer, "mixer") = switch (kind) {
        .linear => .{ .linear = .{
            .qkv = try loadSnLinear(a, device, ck, stage, sn_gpu, owned, sn_runs, &.{
                .{ .name = try cat(ta, lp, "linear_attn.in_proj_qkv"), .n = c.gdnQkvDim(), .width = w[@backingInt(M.qkv)] },
                .{ .name = try cat(ta, lp, "linear_attn.in_proj_z"), .n = c.gdnValueDim(), .width = w[@backingInt(M.z)] },
                .{ .name = try cat(ta, lp, "linear_attn.in_proj_b"), .n = c.v_heads, .width = w[@backingInt(M.b)] },
                .{ .name = try cat(ta, lp, "linear_attn.in_proj_a"), .n = c.v_heads, .width = w[@backingInt(M.a)] },
            }, c.hidden),
            .zba = zeroSnLinear(owned, c.hidden),
            .out = try loadSnLinear(a, device, ck, stage, sn_gpu, owned, sn_runs, &.{.{ .name = try cat(ta, lp, "linear_attn.out_proj"), .n = c.hidden, .width = w[@backingInt(M.out)] }}, c.gdnValueDim()),
            .conv = try checkedConv(try ck.get(try cat(ta, lp, "linear_attn.conv1d.weight")), c.gdnQkvDim(), c.conv_kernel),
            .a_log = try roundScalarBf16(a, device, ck, stage, owned, try cat(ta, lp, "linear_attn.A_log"), c.v_heads),
            .dt = try roundScalarBf16(a, device, ck, stage, owned, try cat(ta, lp, "linear_attn.dt_bias"), c.v_heads),
            .norm = try checked(try ck.get(try cat(ta, lp, "linear_attn.norm.weight")), .bf16, &.{c.dv}),
        } },
        .attention => if (i + 1 < phys_n) .{ .attention = .{
            .q = try loadSnLinear(a, device, ck, stage, sn_gpu, owned, sn_runs, &.{
                .{ .name = try cat(ta, lp, "self_attn.q_proj"), .n = c.qDim(), .width = w[@backingInt(M.q)] },
                .{ .name = try cat(ta, lp, "self_attn.k_proj"), .n = c.kvDim(), .width = w[@backingInt(M.k)] },
                .{ .name = try cat(ta, lp, "self_attn.v_proj"), .n = c.kvDim(), .width = w[@backingInt(M.v)] },
            }, c.hidden),
            .kv = zeroSnLinear(owned, c.hidden),
            .out = try loadSnLinear(a, device, ck, stage, sn_gpu, owned, sn_runs, &.{.{ .name = try cat(ta, lp, "self_attn.o_proj"), .n = c.hidden, .width = w[@backingInt(M.o)] }}, c.heads * c.head_dim),
            .q_norm = try checked(try ck.get(try cat(ta, lp, "self_attn.q_norm.weight")), .bf16, &.{c.head_dim}),
            .k_norm = try checked(try ck.get(try cat(ta, lp, "self_attn.k_norm.weight")), .bf16, &.{c.head_dim}),
        } } else .{ .attention = .{
            .q = try loadSnLinear(a, device, ck, stage, sn_gpu, owned, sn_runs, &.{.{ .name = try cat(ta, lp, "self_attn.q_proj"), .n = c.qDim(), .width = w[@backingInt(M.q)] }}, c.hidden),
            .kv = try loadSnLinear(a, device, ck, stage, sn_gpu, owned, sn_runs, &.{
                .{ .name = try cat(ta, lp, "self_attn.k_proj"), .n = c.kvDim(), .width = w[@backingInt(M.k)] },
                .{ .name = try cat(ta, lp, "self_attn.v_proj"), .n = c.kvDim(), .width = w[@backingInt(M.v)] },
            }, c.hidden),
            .out = try loadSnLinear(a, device, ck, stage, sn_gpu, owned, sn_runs, &.{.{ .name = try cat(ta, lp, "self_attn.o_proj"), .n = c.hidden, .width = w[@backingInt(M.o)] }}, c.heads * c.head_dim),
            .q_norm = try checked(try ck.get(try cat(ta, lp, "self_attn.q_norm.weight")), .bf16, &.{c.head_dim}),
            .k_norm = try checked(try ck.get(try cat(ta, lp, "self_attn.k_norm.weight")), .bf16, &.{c.head_dim}),
        } },
    };
    const gu = try loadSnLinear(a, device, ck, stage, sn_gpu, owned, sn_runs, &.{
        .{ .name = try cat(ta, lp, "mlp.gate_proj"), .n = c.intermediate, .width = w[@backingInt(M.gate)] },
        .{ .name = try cat(ta, lp, "mlp.up_proj"), .n = c.intermediate, .width = w[@backingInt(M.up)] },
    }, c.hidden);
    const down = try loadSnLinear(a, device, ck, stage, sn_gpu, owned, sn_runs, &.{.{ .name = try cat(ta, lp, "mlp.down_proj"), .n = c.hidden, .width = w[@backingInt(M.down)] }}, c.intermediate);
    const layer = wts.Layer{
        .norm = try checked(try ck.get(try cat(ta, lp, "input_layernorm.weight")), .bf16, &.{c.hidden}),
        .post_norm = try checked(try ck.get(try cat(ta, lp, "post_attention_layernorm.weight")), .bf16, &.{c.hidden}),
        .gu = gu,
        .down = down,
        .mixer = mixer,
    };
    const ml: MhcLayer = if (need_mhc) .{
        .attn_w = try checked(try ck.get(try cat(ta, lp, "mhc_attn.weight")), .f32, &.{ 8, 2 * c.hidden }),
        .attn_b = try checked(try ck.get(try cat(ta, lp, "mhc_attn.bias")), .f32, &.{8}),
        .ffn_w = try checked(try ck.get(try cat(ta, lp, "mhc_ffn.weight")), .f32, &.{ 8, 2 * c.hidden }),
        .ffn_b = try checked(try ck.get(try cat(ta, lp, "mhc_ffn.bias")), .f32, &.{8}),
    } else undefined;
    return .{ .layer = layer, .mhc = ml };
}

/// Load physical SenseNova-Loop-9B weights and build execution-step aliases.
pub fn loadWeights(a: std.mem.Allocator, device: mtl.Device, ck: *const cp.Checkpoint, c: cfg.Config, ws: *const widths.Widths) !wts.Weights {
    const ns = try namespace(ck);
    defer if (!std.mem.eql(u8, ns.head, "lm_head")) a.free(ns.head);
    var sn_gpu = linear_gpu.LinearGpu.init(a, device);
    defer sn_gpu.deinit();
    var owned: std.ArrayList(mtl.Buffer) = .empty;
    errdefer {
        for (owned.items) |b| b.deinit();
        owned.deinit(a);
    }
    var sn_runs: std.ArrayList([]SnRun) = .empty;
    errdefer {
        for (sn_runs.items) |r| a.free(r);
        sn_runs.deinit(a);
    }
    const phys_n = c.physicalLayers();
    const physical = try a.alloc(wts.Layer, phys_n);
    errdefer a.free(physical);
    const exec_layers = try a.alloc(wts.Layer, c.layers);
    errdefer a.free(exec_layers);
    const mhc = try a.alloc(MhcLayer, phys_n);
    errdefer a.free(mhc);
    var need_mhc = try a.alloc(bool, phys_n);
    defer a.free(need_mhc);
    @memset(need_mhc, false);
    for (0..c.layers) |exec| if (c.mhc_roles[exec] != .none) {
        need_mhc[c.exec_phys[exec]] = true;
    };
    const embed_name = try std.mem.concat(a, u8, &.{ ns.text, "embed_tokens.weight" });
    defer a.free(embed_name);
    const embed_bf16 = try checked(try ck.get(embed_name), .bf16, &.{ c.vocab, c.hidden });
    // Ensure owned[0] exists even when every unstacked weight borrows shard buffers.
    try ownBuffer(a, &owned, try device.buffer(16, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked));
    var stage = cp.Stage{ .device = device };
    defer stage.deinit();
    for (0..phys_n) |i| {
        const res = try loadPhysicalLayer(a, device, ck, &stage, &sn_gpu, &owned, &sn_runs, c, ws.layers[i], ns.text, i, phys_n, need_mhc[i]);
        physical[i] = res.layer;
        mhc[i] = res.mhc;
    }
    for (exec_layers, 0..) |*dst, exec| dst.* = physical[c.exec_phys[exec]];
    const norm_name = try std.mem.concat(a, u8, &.{ ns.text, "norm.weight" });
    defer a.free(norm_name);
    const norm = try checked(try ck.get(norm_name), .bf16, &.{c.hidden});
    const head = try loadSnLinear(a, device, ck, &stage, &sn_gpu, &owned, &sn_runs, &.{.{ .name = ns.head, .n = c.vocab, .width = ws.head }}, c.hidden);
    const dummy_embed: projection.Triple = .{ .weight = embed_bf16, .scales = embed_bf16, .biases = embed_bf16, .bits = 4, .group = 64 };
    return .{ .allocator = a, .layers = exec_layers, .physical_layers = physical, .embed = dummy_embed, .embed_bf16 = embed_bf16, .norm = norm, .head = head, .owned = owned, .sn_runs = sn_runs, .mhc = mhc };
}
