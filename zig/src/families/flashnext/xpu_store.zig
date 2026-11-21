//! Qwen3.8 Flash Next (qwen4_exp, MLX affine 4-bit group-32) Level Zero device & TP={1,2,4} weight loader.
const std = @import("std");
const ze = @import("ze");
const core = @import("core");
const norms = @import("cuda_norms.zig");
const kbin = @import("xpu_flashnext_kernels");

pub const layers_n: usize = 48;
pub const linear_n: usize = 36;
pub const attn_n: usize = 12;
pub const dims: usize = 2560;
pub const streams: usize = 4;
pub const wide: usize = streams * dims; // 10240
pub const low: usize = 320;
pub const eps: f32 = 1e-6;
pub const vocab: usize = 248320;
pub const experts_n: usize = 512;
pub const topk: usize = 10;
pub const slots: usize = 11;
pub const moe_w: usize = 640;
pub const q_heads: usize = 24;
pub const kv_heads: usize = 2;
pub const head_dim: usize = 256;
pub const index_heads: usize = 4;
pub const index_dim: usize = 128;
pub const half: usize = 32;
pub const index_ratio: usize = 4;
pub const index_top: usize = 512;
pub const index_idw: usize = index_top * index_ratio + index_ratio; // 2052
pub const max_attn_slices: usize = 65; // ceil(2051 / 32)
pub const ple_layer: usize = 1;
pub const ple_tail_n: usize = 9;
pub const gdn_taps: usize = 4;
pub const max_tp: usize = 4;
pub const max_verify_rows: usize = 16;
pub const chunk_rows: usize = 64;
pub const default_solo_rows: usize = 2048;
pub const max_chunk_rows: usize = 2048;
pub const attn_tile_rows: usize = 64;

pub const Tensor = core.safetensors.Tensor;

pub fn linearLayer(l: usize) bool {
    return l % 4 != 3;
}

pub fn linearIndex(l: usize) usize {
    return l - l / 4;
}

pub fn attnIndex(l: usize) usize {
    return l / 4;
}

pub fn promote(bits: u16) f32 {
    return @bitCast(@as(u32, bits) << 16);
}

pub const TpShape = struct {
    tp: usize,
    nk: usize,
    nv: usize,
    gdn_conv: usize,
    gdn_pw: usize,
    gdn_vdim: usize,
    nq: usize,
    nkv: usize,
    attn_pw: usize,
    attn_odim: usize,
    local_e: usize,
    sh_w: usize,
    vocab_per: usize,

    pub fn forTp(tp: usize) !TpShape {
        if (tp != 1 and tp != 2 and tp != 4) return error.InvalidTpDegree;
        const nk = 16 / tp;
        const nv = 48 / tp;
        const gdn_conv = 2 * nk * 128 + nv * 128;
        const gdn_vdim = nv * 128;
        const gdn_pw = gdn_conv + gdn_vdim + 2 * nv;
        const nq = 24 / tp;
        const nkv: usize = if (tp == 1) 2 else 1;
        const attn_pw = nq * 512 + 2 * nkv * 256 + 640;
        const attn_odim = nq * 256;
        return .{
            .tp = tp,
            .nk = nk,
            .nv = nv,
            .gdn_conv = gdn_conv,
            .gdn_pw = gdn_pw,
            .gdn_vdim = gdn_vdim,
            .nq = nq,
            .nkv = nkv,
            .attn_pw = attn_pw,
            .attn_odim = attn_odim,
            .local_e = experts_n / tp,
            .sh_w = moe_w / tp,
            .vocab_per = vocab / tp,
        };
    }

    pub fn draftSlice(self: TpShape, rk: usize, total: usize) struct { lo: usize, len: usize } {
        const per = (total + self.tp - 1) / self.tp;
        const lo = @min(total, rk * per);
        const hi = @min(total, lo + per);
        return .{ .lo = lo, .len = hi - lo };
    }
};

pub const Kernels = struct {
    embed_q4: ze.H,
    embed_bf16: ze.H,
    rmsnorm_rows: ze.H,
    add_streams: ze.H,
    hc_writeback_norm: ze.H,
    hc_writeback_only: ze.H,
    hc_down_1: ze.H,
    hc_down_wide: ze.H,
    hc_upmix_1: ze.H,
    hc_upmix_dpas: ze.H,
    qmv1_bf16: ze.H,
    qmv1_f32: ze.H,
    qmv16_bf16: ze.H,
    qmv16_f32: ze.H,
    qmm_bf16: ze.H,
    qmm_f32: ze.H,
    ple_gate: ze.H,
    ple_conv: ze.H,
    shift_window: ze.H,
    gdn_front: ze.H,
    gdn_rows: ze.H,
    gdn_norm: ze.H,
    attn_prep: ze.H,
    qsa_pool: ze.H,
    qsa_scores: ze.H,
    qsa_select: ze.H,
    attn_sparse: ze.H,
    attn_merge: ze.H,
    attn_gate: ze.H,
    router_1: ze.H,
    router_mm: ze.H,
    route_topk: ze.H,
    moe_compact: ze.H,
    moe_up_1: ze.H,
    moe_up_dpas: ze.H,
    moe_down_1: ze.H,
    moe_down_dpas: ze.H,
    moe_sh_up_1: ze.H,
    moe_sh_up_dpas: ze.H,
    moe_combine: ze.H,
    logits_part: ze.H,
    logits_best: ze.H,
    topk64_part: ze.H,
    bf16_to_f32: ze.H,
};

pub const Dev = struct {
    g: ze.Gpu,
    list: ze.H,
    k: Kernels,
    staging: []u8 = &.{},
    staging_used: usize = 0,
    allocated: usize = 0,

    pub fn initFromGpu(g: ze.Gpu) !Dev {
        var d: Dev = .{ .g = g, .list = null, .k = undefined };
        d.list = try d.g.immediate(0, 0);
        const m = try d.g.module(&kbin.flashnext);
        d.k = .{
            .embed_q4 = try d.loadK(m, "fn_embed_q4", 256),
            .embed_bf16 = try d.loadK(m, "fn_embed_bf16", 256),
            .rmsnorm_rows = try d.loadK(m, "fn_rmsnorm_rows", 256),
            .add_streams = try d.loadK(m, "fn_add_streams", 256),
            .hc_writeback_norm = try d.loadK(m, "fn_hc_writeback_norm", 256),
            .hc_writeback_only = try d.loadK(m, "fn_hc_writeback_only", 256),
            .hc_down_1 = try d.loadK(m, "fn_hc_down_1", 64),
            .hc_down_wide = try d.loadK(m, "fn_hc_down_wide", 128),
            .hc_upmix_1 = try d.loadK(m, "fn_hc_upmix_1", 64),
            .hc_upmix_dpas = try d.loadK(m, "fn_hc_upmix_dpas", 128),
            .qmv1_bf16 = try d.loadK(m, "fn_qmv1_dpas_bf16", 64),
            .qmv1_f32 = try d.loadK(m, "fn_qmv1_dpas_f32", 64),
            .qmv16_bf16 = try d.loadK(m, "fn_qmv16_dpas_bf16", 64),
            .qmv16_f32 = try d.loadK(m, "fn_qmv16_dpas_f32", 64),
            .qmm_bf16 = try d.loadK(m, "fn_qmm_tile_dpas_bf16", 128),
            .qmm_f32 = try d.loadK(m, "fn_qmm_tile_dpas_f32", 128),
            .ple_gate = try d.loadK(m, "fn_ple_gate", 512),
            .ple_conv = try d.loadK(m, "fn_ple_conv", 512),
            .shift_window = try d.loadK(m, "fn_shift_window", 256),
            .gdn_front = try d.loadK(m, "fn_gdn_front", 256),
            .gdn_rows = try d.loadK(m, "fn_gdn_rows", 16),
            .gdn_norm = try d.loadK(m, "fn_gdn_norm", 128),
            .attn_prep = try d.loadK(m, "fn_attn_prep", 256),
            .qsa_pool = try d.loadK(m, "fn_qsa_pool", 128),
            .qsa_scores = try d.loadK(m, "fn_qsa_scores", 256),
            .qsa_select = try d.loadK(m, "fn_qsa_select", 256),
            .attn_sparse = try d.loadK(m, "fn_attn_sparse", 16),
            .attn_merge = try d.loadK(m, "fn_attn_merge", 16),
            .attn_gate = try d.loadK(m, "fn_attn_gate", 256),
            .router_1 = try d.loadK(m, "fn_router_dpas_1", 64),
            .router_mm = try d.loadK(m, "fn_router_dpas_mm", 128),
            .route_topk = try d.loadK(m, "fn_route_topk", 128),
            .moe_compact = try d.loadK(m, "fn_moe_compact", 256),
            .moe_up_1 = try d.loadK(m, "fn_moe_up_1", 64),
            .moe_up_dpas = try d.loadK(m, "fn_moe_up_dpas", 256),
            .moe_down_1 = try d.loadK(m, "fn_moe_down_1", 64),
            .moe_down_dpas = try d.loadK(m, "fn_moe_down_dpas", 128),
            .moe_sh_up_1 = try d.loadK(m, "fn_moe_sh_up_1", 64),
            .moe_sh_up_dpas = try d.loadK(m, "fn_moe_sh_up_dpas", 128),
            .moe_combine = try d.loadK(m, "fn_moe_combine", 256),
            .logits_part = try d.loadK(m, "fn_logits_part", 256),
            .logits_best = try d.loadK(m, "fn_logits_best", 256),
            .topk64_part = try d.loadK(m, "fn_topk64_part", 64),
            .bf16_to_f32 = try d.loadK(m, "fn_bf16_to_f32", 256),
        };
        return d;
    }

    fn loadK(d: *Dev, m: ze.H, name: [*:0]const u8, wg: u32) !ze.H {
        const k = try d.g.kernel(m, name);
        try ze.check(d.g.api.zeKernelSetGroupSize(k, wg, 1, 1));
        try d.g.indirect(k);
        return k;
    }

    pub fn sync(d: *Dev) !void {
        try ze.check(d.g.api.zeCommandListHostSynchronize(d.list, std.math.maxInt(u64)));
    }

    pub fn allocBuf(d: *Dev, bytes: usize) !*anyopaque {
        const sz = @max(bytes, 64);
        d.allocated += sz;
        return d.g.deviceAlloc(sz);
    }

    pub fn freeBuf(d: *Dev, p: *anyopaque, bytes: usize) void {
        const sz = @max(bytes, 64);
        d.allocated -|= sz;
        d.g.free(p);
    }

    pub fn dropStaging(d: *Dev) void {
        if (d.staging.len > 0) {
            d.sync() catch {};
            d.g.free(d.staging.ptr);
            d.staging = &.{};
            d.staging_used = 0;
        }
    }

    pub fn copyIn(d: *Dev, dst: *anyopaque, bytes: []const u8) !void {
        if (bytes.len == 0) return;
        if (d.staging.len == 0) {
            const cap: usize = 256 << 20;
            d.staging = @as([*]u8, @ptrCast(try d.g.hostAlloc(cap)))[0..cap];
            d.staging_used = 0;
        }
        var done: usize = 0;
        while (done < bytes.len) {
            if (d.staging_used + 64 >= d.staging.len) {
                try d.sync();
                d.staging_used = 0;
            }
            const avail = d.staging.len - d.staging_used;
            const n = @min(avail, bytes.len - done);
            @memcpy(d.staging[d.staging_used..][0..n], bytes[done .. done + n]);
            const p: *anyopaque = @ptrFromInt(@intFromPtr(dst) + done);
            const s_ptr: *const anyopaque = @ptrCast(d.staging.ptr + d.staging_used);
            try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, p, s_ptr, n, null, 0, null));
            d.staging_used = std.mem.alignForward(usize, d.staging_used + n, 64);
            done += n;
        }
    }

    pub fn upload(d: *Dev, bytes: []const u8) !*anyopaque {
        const p = try d.allocBuf(bytes.len);
        try d.copyIn(p, bytes);
        return p;
    }

    pub fn zeroBuf(d: *Dev, dst: *anyopaque, bytes: usize) !void {
        if (bytes == 0) return;
        if (d.staging.len == 0) {
            const cap: usize = 256 << 20;
            d.staging = @as([*]u8, @ptrCast(try d.g.hostAlloc(cap)))[0..cap];
            d.staging_used = 0;
        }
        if (d.staging_used > 0) {
            try d.sync();
            d.staging_used = 0;
        }
        const z = @min(d.staging.len, bytes);
        @memset(d.staging[0..z], 0);
        var done: usize = 0;
        while (done < bytes) {
            const n = @min(z, bytes - done);
            const p: *anyopaque = @ptrFromInt(@intFromPtr(dst) + done);
            try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, p, d.staging.ptr, n, null, 0, null));
            done += n;
        }
        try d.sync();
    }

    pub fn allocZero(d: *Dev, bytes: usize) !*anyopaque {
        const p = try d.allocBuf(bytes);
        try d.zeroBuf(p, bytes);
        return p;
    }

    pub fn copy(d: *Dev, dst: *anyopaque, src: *const anyopaque, bytes: usize) !void {
        if (bytes == 0) return;
        try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, dst, src, bytes, null, 0, null));
    }

    pub fn launch(d: *Dev, k: ze.H, groups: ze.GroupCount, args: anytype) !void {
        inline for (args, 0..) |v, i| try d.g.arg(k, @intCast(i), v);
        try ze.check(d.g.api.zeCommandListAppendLaunchKernel(d.list, k, &groups, null, 0, null));
    }
};

pub fn off(p: *anyopaque, bytes: usize) *anyopaque {
    return @ptrFromInt(@intFromPtr(p) + bytes);
}

/// Discover the first Xe2 GPU's PCI device ID on the system (defaulting to 0xe211 B70 if XPU_DEVICE_ID is unset).
pub fn detectDeviceId() !u32 {
    if (try @import("sys").environment.read(?u32, "XPU_DEVICE_ID", null)) |id| return id;
    var lib = std.DynLib.open("libze_loader.so.1") catch return error.LoaderUnavailable;
    defer lib.close();
    const api = try ze.resolve(ze.Api, &lib);
    try ze.check(api.zeInit(1));
    var nd: u32 = 0;
    try ze.check(api.zeDriverGet(&nd, null));
    var drivers: [8]ze.H = @splat(null);
    nd = @min(nd, drivers.len);
    try ze.check(api.zeDriverGet(&nd, &drivers));
    for (drivers[0..nd]) |drv| {
        if (drv == null) continue;
        var nv: u32 = 0;
        try ze.check(api.zeDeviceGet(drv, &nv, null));
        var devs: [16]ze.H = @splat(null);
        nv = @min(nv, devs.len);
        try ze.check(api.zeDeviceGet(drv, &nv, &devs));
        for (devs[0..nv]) |dev| {
            if (dev == null) continue;
            var p: ze.DeviceProperties = .{};
            try ze.check(api.zeDeviceGetProperties(dev, &p));
            if (p.deviceId != 0) return p.deviceId;
        }
    }
    return error.NoDevice;
}

// Safetensors Store (drops non-PLE shard pages during load to preserve host RAM)

fn runtimePages(names: []const []const u8) bool {
    for (names) |name| if (std.mem.indexOf(u8, name, ".ple_embedding.ngram_embedding.") != null) return true;
    return false;
}

const Shard = struct {
    f: core.safetensors.File,
    io: std.Io,

    fn load(gpa: std.mem.Allocator, io: std.Io, path: []const u8) !Shard {
        return .{ .f = try core.safetensors.File.open(gpa, io, path), .io = io };
    }

    fn close(self: *Shard) void {
        if (!runtimePages(self.f.names.keys())) self.f.release(std.heap.page_allocator);
        self.f.close(self.io);
    }

    fn get(self: *const Shard, name: []const u8) ?Tensor {
        return self.f.get(name);
    }
};

const Held = struct { rel: []u8, file: Shard };

pub const Store = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    dir: []const u8,
    parsed: std.json.Parsed(std.json.Value),
    rel: []u8 = &.{},
    file: ?Shard = null,
    held: std.ArrayList(Held) = .empty,
    released: std.ArrayList(Held) = .empty,
    norm_offset: f32 = 0,

    pub fn open(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !Store {
        const index = try std.fs.path.join(gpa, &.{ dir, "model.safetensors.index.json" });
        defer gpa.free(index);
        const text = try std.Io.Dir.cwd().readFileAlloc(io, index, gpa, .limited(1 << 26));
        defer gpa.free(text);
        const parsed = try std.json.parseFromSlice(std.json.Value, gpa, text, .{});
        errdefer parsed.deinit();
        _ = try core.safetensors.index.weightMap(parsed.value);
        return .{ .gpa = gpa, .io = io, .dir = dir, .parsed = parsed };
    }

    pub fn deinit(self: *Store) void {
        self.release();
        self.held.deinit(self.gpa);
        for (self.released.items) |*h| {
            h.file.close();
            self.gpa.free(h.rel);
        }
        self.released.deinit(self.gpa);
        if (self.file) |*f| f.close();
        self.file = null;
        if (self.rel.len != 0) self.gpa.free(self.rel);
        self.rel = &.{};
        self.parsed.deinit();
    }

    pub fn detectNorms(self: *Store) !void {
        var means: [layers_n]f64 = undefined;
        var buf: [160]u8 = undefined;
        for (0..layers_n) |i| {
            const name = try std.fmt.bufPrint(&buf, "language_model.model.layers.{d}.attn_hyper_connection.hc_norm.weight", .{i});
            const values = try asF32(self.gpa, try self.tensor(name));
            defer self.gpa.free(values);
            if (values.len != wide) return error.UnexpectedTensor;
            var sum: f64 = 0;
            for (values) |v| {
                if (!std.math.isFinite(v)) return error.InvalidNormConvention;
                sum += v;
            }
            means[i] = sum / @as(f64, @floatFromInt(values.len));
        }
        self.norm_offset = try norms.offset(layers_n, means);
    }

    pub fn normScale(self: *const Store, gpa: std.mem.Allocator, t: Tensor) ![]f32 {
        const values = try asF32(gpa, t);
        errdefer gpa.free(values);
        for (values) |*v| {
            if (!std.math.isFinite(v.*)) return error.InvalidNormConvention;
            v.* += self.norm_offset;
        }
        return values;
    }

    pub fn release(self: *Store) void {
        for (self.held.items) |h| {
            self.released.append(self.gpa, h) catch {
                var kept = h.file;
                kept.close();
                self.gpa.free(h.rel);
            };
        }
        self.held.clearRetainingCapacity();
    }

    pub fn resolveKey(self: *const Store, requested: []const u8, buf: []u8) ![]const u8 {
        const wm = self.parsed.value.object.get("weight_map") orelse return error.MissingTensor;
        if (wm.object.contains(requested)) return requested;
        const aliases = .{
            .{ "language_model.model.", "model.language_model." },
            .{ "language_model.lm_head.", "lm_head." },
            .{ "language_model.mtp.", "mtp." },
        };
        inline for (aliases) |pair| {
            if (std.mem.startsWith(u8, requested, pair[0])) {
                const name = try std.fmt.bufPrint(buf, "{s}{s}", .{ pair[1], requested[pair[0].len..] });
                if (wm.object.contains(name)) return name;
            }
        }
        return error.MissingTensor;
    }

    pub fn has(self: *const Store, requested: []const u8) bool {
        var mapped: [512]u8 = undefined;
        _ = self.resolveKey(requested, &mapped) catch return false;
        return true;
    }

    fn shardName(self: *const Store, key: []const u8) ![]const u8 {
        const wm = self.parsed.value.object.get("weight_map") orelse return error.MissingTensor;
        const v = wm.object.get(key) orelse return error.MissingTensor;
        return v.string;
    }

    fn openShard(self: *Store, file: []const u8) !Held {
        const rel = try self.gpa.dupe(u8, file);
        errdefer self.gpa.free(rel);
        const path = try std.fs.path.join(self.gpa, &.{ self.dir, file });
        defer self.gpa.free(path);
        return .{ .rel = rel, .file = try Shard.load(self.gpa, self.io, path) };
    }

    pub fn hold(self: *Store, requested: []const u8) !Tensor {
        var mapped: [512]u8 = undefined;
        const key = try self.resolveKey(requested, &mapped);
        const file = try self.shardName(key);
        for (self.held.items) |*h| {
            if (std.mem.eql(u8, h.rel, file)) return h.file.get(key) orelse return error.MissingTensor;
        }
        try self.held.ensureUnusedCapacity(self.gpa, 1);
        for (self.released.items, 0..) |cached, i| {
            if (std.mem.eql(u8, cached.rel, file)) {
                self.held.appendAssumeCapacity(cached);
                _ = self.released.swapRemove(i);
                return self.held.items[self.held.items.len - 1].file.get(key) orelse return error.MissingTensor;
            }
        }
        self.held.appendAssumeCapacity(try self.openShard(file));
        return self.held.items[self.held.items.len - 1].file.get(key) orelse return error.MissingTensor;
    }

    fn use(self: *Store, key: []const u8) !void {
        const file = try self.shardName(key);
        if (self.file != null and std.mem.eql(u8, file, self.rel)) return;
        try self.released.ensureUnusedCapacity(self.gpa, 1);
        const next: Held = for (self.released.items, 0..) |cached, i| {
            if (std.mem.eql(u8, cached.rel, file)) break self.released.swapRemove(i);
        } else try self.openShard(file);
        if (self.file) |*f| {
            self.released.appendAssumeCapacity(.{ .rel = self.rel, .file = f.* });
        }
        self.rel = next.rel;
        self.file = next.file;
    }

    pub fn tensor(self: *Store, requested: []const u8) !Tensor {
        var mapped: [512]u8 = undefined;
        const key = try self.resolveKey(requested, &mapped);
        try self.use(key);
        return self.file.?.get(key) orelse return error.MissingTensor;
    }
};

pub fn asF32(gpa: std.mem.Allocator, t: Tensor) ![]f32 {
    if (t.rank != 1) return error.UnexpectedTensor;
    const out = try gpa.alloc(f32, t.dim(0));
    errdefer gpa.free(out);
    if (t.dtype == .f32 and t.bytes.len == out.len * 4) {
        for (out, 0..) |*o, i| o.* = @bitCast(std.mem.readInt(u32, t.bytes[4 * i ..][0..4], .little));
    } else if (t.dtype == .bf16 and t.bytes.len == out.len * 2) {
        for (out, 0..) |*o, i| o.* = promote(std.mem.readInt(u16, t.bytes[2 * i ..][0..2], .little));
    } else return error.UnexpectedTensor;
    return out;
}
