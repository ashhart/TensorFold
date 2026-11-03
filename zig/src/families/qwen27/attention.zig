//! M5 target attention owns scratch and cache movement; the model owns projections, caches and admission.
const std = @import("std");
const mtl = @import("metal");
const profile = @import("core").gpu_profile;
const Ref = @import("projection.zig").Ref;
const glue = @import("glue.zig");
pub const planning = @import("attention_plan.zig");
pub const Plan = planning.Plan;
pub const Cache = struct {
    keys: Ref,
    values: Ref,
    capacity: u32,
    key_stride: u32 = 0,
    value_stride: u32 = 0,
    /// TF_QWEN27_METAL_KV_PAGES: keys and values are this layer's part of every page; rows found through the table.
    pages: ?Pages = null,
};
/// A paged stream's table (u32 page ids, 256 rows a page) and the bytes from one page to the next in the arena.
pub const Pages = struct { table: Ref, page_bytes: u64 };
const CachePtrs = extern struct { keys: u64, values: u64 };
/// The paged kernels' read pointers (kernels/metal/qwen27/paged.zig): main's two and the stream's table and page size.
const PagedPtrs = extern struct { keys: u64, values: u64, table: u64, page_bytes: u64 };
/// The pack and append kernels' arguments (public for the kernel contract's io check).
pub const PackArgs = extern struct { rows: u32, packed_rows: u32, heads: u32 = 24, kv_heads: u32 = 4, dim: u32 = 256, group: u32 = 6 };
/// Keep's gather and scatter: rows kept, and the KV heads a staging row holds.
const KeepArgs = extern struct { kept: u32, kv_heads: u32 };
pub const PromptArgs = extern struct { start: i32, rows: i32, chunks: i32, rp: i32, key_stride: i32, value_stride: i32, scale2: f32, kv_heads: i32 };
pub const Limits = struct {
    rows: u32 = 128,
    streams: u32 = 8,
    keys: u32 = 32768,
    shared_prefix: bool = true,
    /// Numeric shape; shared_prefix above selects the allocation's execution mode.
    geometry: planning.Geometry = .{},
};
pub const Field = enum { tile_stream, q_rows, nodes, paths, positions, qa, qb, poa, pma, pla, pob, pmb, plb, keep_k, keep_v, pick };
pub const BindingField = enum { meta, caches, keep_map };
// Each layer owns immutable descriptors until its command buffer is terminal; arithmetic workspace is shared serially.
pub const Binding = struct {
    buffers: [3]mtl.Buffer,
    limits: Limits,
    uploaded: ?[32]u8 = null,
    streams: u32 = 0,
    pointers: [planning.max_streams]CachePtrs = @splat(.{ .keys = 0, .values = 0 }),
    paged: bool = false,
    strides: [planning.max_streams][2]u32 = @splat(.{ 0, 0 }),
    kept: ?u32 = null,
    pub fn init(device: mtl.Device, limits: Limits) !Binding {
        try validLimits(limits);
        var buffers: [3]mtl.Buffer = undefined;
        var made: usize = 0;
        errdefer for (buffers[0..made]) |buffer| buffer.deinit();
        for ([_]usize{ (planning.global_words + planning.max_streams * planning.stream_words) * 4, planning.max_streams * @sizeOf(PagedPtrs), limits.rows * 16 }, &buffers) |allocation_bytes, *buffer| {
            buffer.* = try device.buffer(allocation_bytes, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
            made += 1;
        }
        return .{ .buffers = buffers, .limits = limits };
    }
    pub fn deinit(b: *Binding) void {
        for (b.buffers) |buffer| buffer.deinit();
    }
    pub fn get(b: *const Binding, field: BindingField) Ref {
        return .{ .buffer = b.buffers[@backingInt(field)] };
    }
    pub fn upload(b: *Binding, plan: Plan, caches: []const Cache) !void {
        if (plan.rows > b.limits.rows or plan.streams > b.limits.streams or caches.len != plan.streams or !plan.geometry.sameShape(b.limits.geometry)) return error.AttentionLimits;
        var meta = plan.meta;
        var table: [planning.max_streams]CachePtrs = @splat(.{ .keys = 0, .values = 0 });
        var paged: [planning.max_streams]PagedPtrs = @splat(.{ .keys = 0, .values = 0, .table = 0, .page_bytes = 0 });
        const pages = caches.len > 0 and caches[0].pages != null;
        for (caches, 0..) |cache, i| {
            const at = 8 + i * 12;
            const start: u32 = @intCast(meta[at + 4]);
            const rows: u32 = @intCast(meta[at + 2]);
            if (cache.capacity == 0 or cache.capacity > std.math.maxInt(i32) / b.limits.geometry.dim or @as(u64, start) + rows > cache.capacity) return error.AttentionCapacity;
            if ((cache.pages != null) != pages) return error.AttentionMixedPages;
            if (cache.pages) |p| {
                // a page's head runs are [kv_heads][256][dim]: no stride, and no span past the table's pages
                try spanAligned(p.table, pageCount(cache.capacity) * 4, 4);
                if (cache.key_stride != 0 or cache.value_stride != 0 or p.page_bytes % 16 != 0) return error.AttentionCacheStride;
                table[i] = .{ .keys = cache.keys.buffer.gpuAddress() + cache.keys.offset, .values = cache.values.buffer.gpuAddress() + cache.values.offset };
                paged[i] = .{ .keys = table[i].keys, .values = table[i].values, .table = p.table.buffer.gpuAddress() + p.table.offset, .page_bytes = p.page_bytes };
                meta[at + 7] = @intCast(cache.capacity * b.limits.geometry.dim);
                meta[at + 8] = @intCast(cache.capacity * b.limits.geometry.dim);
                continue;
            }
            const ks = if (cache.key_stride == 0) cache.capacity * b.limits.geometry.dim else cache.key_stride;
            const vs = if (cache.value_stride == 0) cache.capacity * b.limits.geometry.dim else cache.value_stride;
            if (ks < @as(u64, cache.capacity) * b.limits.geometry.dim or vs < @as(u64, cache.capacity) * b.limits.geometry.dim or ks > std.math.maxInt(i32) or vs > std.math.maxInt(i32)) return error.AttentionCacheStride;
            try spanAligned(cache.keys, cacheBytes(cache, true, b.limits.geometry), 16);
            try spanAligned(cache.values, cacheBytes(cache, false, b.limits.geometry), 16);
            try disjoint(cache.keys, cacheBytes(cache, true, b.limits.geometry), cache.values, cacheBytes(cache, false, b.limits.geometry));
            for (caches[0..i]) |prior| {
                for ([_]bool{ true, false }) |key| for ([_]bool{ true, false }) |prior_key| try disjoint(if (key) cache.keys else cache.values, cacheBytes(cache, key, b.limits.geometry), if (prior_key) prior.keys else prior.values, cacheBytes(prior, prior_key, b.limits.geometry));
            }
            table[i] = .{ .keys = cache.keys.buffer.gpuAddress() + cache.keys.offset, .values = cache.values.buffer.gpuAddress() + cache.values.offset };
            meta[at + 7] = @intCast(ks);
            meta[at + 8] = @intCast(vs);
        }
        if (pages) @memcpy(b.get(.caches).buffer.slice(PagedPtrs, planning.max_streams), &paged) else @memcpy(b.get(.caches).buffer.slice(CachePtrs, planning.max_streams), &table);
        @memcpy(b.get(.meta).buffer.slice(i32, meta.len), &meta);
        b.uploaded = fingerprint(plan);
        b.streams = plan.streams;
        b.pointers = table;
        b.paged = pages;
        for (caches, 0..) |_, i| b.strides[i] = .{ @intCast(meta[8 + i * 12 + 7]), @intCast(meta[8 + i * 12 + 8]) };
        b.kept = null;
    }
    pub fn check(b: *const Binding, plan: Plan, caches: []const Cache) !void {
        const uploaded = b.uploaded orelse return error.AttentionNotUploaded;
        if (!std.mem.eql(u8, &uploaded, &fingerprint(plan))) return error.AttentionNotUploaded;
        try b.checkCaches(caches);
        for (caches, 0..) |cache, i| {
            const at = 8 + i * 12;
            if (@as(u64, @intCast(plan.meta[at + 4])) + @as(u32, @intCast(plan.meta[at + 2])) > cache.capacity) return error.AttentionCapacity;
        }
    }
    fn checkCaches(b: *const Binding, caches: []const Cache) !void {
        if (b.uploaded == null or caches.len != b.streams) return error.AttentionNotUploaded;
        for (caches, 0..) |cache, i| {
            if ((cache.pages != null) != b.paged) return error.AttentionNotUploaded;
            if (cache.pages != null) {
                if (cache.keys.buffer.gpuAddress() + cache.keys.offset != b.pointers[i].keys or cache.values.buffer.gpuAddress() + cache.values.offset != b.pointers[i].values) return error.AttentionNotUploaded;
                continue;
            }
            if (cache.capacity == 0 or cache.capacity > std.math.maxInt(i32) / b.limits.geometry.dim or cache.keys.buffer.gpuAddress() + cache.keys.offset != b.pointers[i].keys or cache.values.buffer.gpuAddress() + cache.values.offset != b.pointers[i].values) return error.AttentionNotUploaded;
            const ks = if (cache.key_stride == 0) cache.capacity * b.limits.geometry.dim else cache.key_stride;
            const vs = if (cache.value_stride == 0) cache.capacity * b.limits.geometry.dim else cache.value_stride;
            if (ks != b.strides[i][0] or vs != b.strides[i][1]) return error.AttentionNotUploaded;
        }
    }
    pub fn admitDeviceKeep(b: *Binding, plan: Plan, caches: []const Cache, dispatch_rows: u32) !void {
        try b.check(plan, caches);
        if (dispatch_rows == 0 or dispatch_rows > plan.rows or dispatch_rows > b.limits.rows) return error.AttentionKeep;
        b.kept = dispatch_rows;
    }
    pub fn uploadKeep(b: *Binding, plan: Plan, caches: []const Cache, paths: []const []const u32) !u32 {
        try b.check(plan, caches);
        const rows = try plan.keepRows(plan.gpa, paths);
        defer plan.gpa.free(rows);
        if (rows.len > b.limits.rows) return error.AttentionKeep;
        @memcpy(b.get(.keep_map).buffer.slice(planning.KeepRow, rows.len), rows);
        b.kept = @intCast(rows.len);
        return b.kept.?;
    }
};
pub const Scratch = struct {
    buffers: [std.enums.values(Field).len]mtl.Buffer,
    limits: Limits,
    tiles: u32,
    chunks: u32,
    tail_chunks: u32,
    uploaded: ?[32]u8 = null,
    pub fn init(device: mtl.Device, limits: Limits) !Scratch {
        try validLimits(limits);
        const layout = @import("attention_memory.zig").Layout.init(limits);
        const sizes = layout.sizes;
        comptime std.debug.assert(sizes.len == std.enums.values(Field).len);
        var buffers: [sizes.len]mtl.Buffer = undefined;
        var made: usize = 0;
        errdefer for (buffers[0..made]) |buffer| buffer.deinit();
        for (sizes, &buffers) |allocation_bytes, *buffer| {
            buffer.* = try device.buffer(@max(allocation_bytes, 16), mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
            made += 1;
        }
        @memset(buffers[@backingInt(Field.pick)].slice(u32, 12), 0); // the pick clears the flag after each read
        return .{ .buffers = buffers, .limits = limits, .tiles = layout.tiles, .chunks = layout.chunks, .tail_chunks = layout.tail_chunks };
    }
    pub fn deinit(s: *Scratch) void {
        for (s.buffers) |buffer| buffer.deinit();
    }
    pub fn get(s: *const Scratch, field: Field) Ref {
        return .{ .buffer = s.buffers[@backingInt(field)] };
    }
    pub fn bytes(s: Scratch) usize {
        var total: usize = 0;
        for (s.buffers) |buffer| total += buffer.length();
        return total;
    }
    pub fn upload(s: *Scratch, plan: Plan) !void {
        if (plan.rows > s.limits.rows or plan.streams > s.limits.streams or plan.tiles > s.tiles or plan.prefix_chunks > s.chunks or plan.tail_chunks > s.tail_chunks or plan.geometry.shared_prefix != s.limits.shared_prefix or !plan.geometry.sameShape(s.limits.geometry)) return error.AttentionLimits;
        @memcpy(s.get(.tile_stream).buffer.slice(i32, plan.tile_stream.len), plan.tile_stream);
        @memcpy(s.get(.q_rows).buffer.slice(i32, plan.q_rows.len), plan.q_rows);
        @memcpy(s.get(.nodes).buffer.slice(i32, plan.nodes.len), plan.nodes);
        @memcpy(s.get(.paths).buffer.slice(i32, plan.paths.len), plan.paths);
        @memcpy(s.get(.positions).buffer.slice(i32, plan.positions.len), plan.positions);
        s.uploaded = fingerprint(plan);
    }
    pub fn check(s: *const Scratch, plan: Plan) !void {
        const uploaded = s.uploaded orelse return error.AttentionNotUploaded;
        if (!std.mem.eql(u8, &uploaded, &fingerprint(plan))) return error.AttentionNotUploaded;
    }
};
fn validLimits(limits: Limits) !void {
    try limits.geometry.check();
    if (limits.rows == 0 or limits.rows > @import("gpu_frame.zig").max_rows or limits.streams == 0 or limits.streams > planning.max_streams or limits.keys == 0 or limits.keys > std.math.maxInt(i32) / limits.geometry.dim) return error.AttentionLimits;
}
fn span(r: Ref, bytes: usize) !void {
    try spanAligned(r, bytes, 2);
}
fn spanAligned(r: Ref, bytes: usize, alignment: usize) !void {
    if (r.offset > r.buffer.length() or bytes > r.buffer.length() - r.offset or r.offset % alignment != 0) return error.AttentionBuffer;
}
fn disjoint(a: Ref, an: usize, b: Ref, bn: usize) !void {
    if (a.buffer.id == b.buffer.id and a.offset < b.offset + bn and b.offset < a.offset + an) return error.AttentionAlias;
}
fn fingerprint(plan: Plan) [32]u8 {
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    // Preserve the canonical 27B's descriptor hash; any new numeric shape gets its own identity.
    if (!plan.geometry.sameShape(.{})) {
        const shape = [_]u32{ plan.geometry.heads, plan.geometry.kv_heads, plan.geometry.dim, plan.geometry.rotary };
        hash.update(std.mem.sliceAsBytes(&shape));
    }
    hash.update(std.mem.asBytes(&plan.meta));
    inline for (.{ plan.tile_stream, plan.q_rows, plan.nodes, plan.paths, plan.positions }) |values| hash.update(std.mem.sliceAsBytes(values));
    var digest: [32]u8 = undefined;
    hash.final(&digest);
    return digest;
}
fn cacheBytes(cache: Cache, key: bool, geometry: planning.Geometry) usize {
    const requested = if (key) cache.key_stride else cache.value_stride;
    const stride: usize = if (requested == 0) @as(usize, cache.capacity) * geometry.dim else requested;
    return ((@as(usize, geometry.kv_heads) - 1) * stride + @as(usize, cache.capacity) * geometry.dim) * 2;
}
/// An operand outside `cache`'s rows; a paged cache's arena holds only pages, so an operand in it at all is an alias.
fn apart(cache: Cache, geometry: planning.Geometry, operand: Ref, bytes: usize) !void {
    if (cache.pages != null) {
        if (operand.buffer.id == cache.keys.buffer.id or operand.buffer.id == cache.values.buffer.id) return error.AttentionAlias;
        return;
    }
    try disjoint(cache.keys, cacheBytes(cache, true, geometry), operand, bytes);
    try disjoint(cache.values, cacheBytes(cache, false, geometry), operand, bytes);
}
/// The caches' buffers, and a paged stream's table, resident for the pass's indirect reads.
fn useCaches(e: mtl.ComputeEncoder, caches: []const Cache) void {
    for (caches) |cache| {
        e.useResource(cache.keys.buffer, mtl.ResourceUsage.read | mtl.ResourceUsage.write);
        e.useResource(cache.values.buffer, mtl.ResourceUsage.read | mtl.ResourceUsage.write);
        if (cache.pages) |p| e.useResource(p.table.buffer, mtl.ResourceUsage.read);
    }
}
/// Pages a stream of `rows` rows takes (metal_kv_pages.page_tokens a page).
pub fn pageCount(rows: u32) usize {
    return (@as(usize, rows) + page_rows - 1) / page_rows;
}
/// Rows a page holds: kernels/metal/core/kv_pages.h's TF_PAGE_TOKENS.
pub const page_rows = 256;
fn bind(e: mtl.ComputeEncoder, refs: anytype) void {
    inline for (refs, 0..) |r, i| e.setBuffer(r.buffer, r.offset, i);
}
/// The prompt kernel's threads a group: eight simdgroups, both kernels (attention_prompt*.metal's SG = 8).
const prompt_threads = 256;
/// h_end and w_end above 0: an M-RoPE round, the h and w position planes `plane` rows past each other in `positions`.
pub const Preprocess = struct {
    qg: Ref,
    key: Ref,
    q_gain: Ref,
    k_gain: Ref,
    norm_q: Ref,
    norm_k: Ref,
    query: Ref,
    rotated_key: Ref,
    positions: Ref,
    rows: u32,
    eps: f32,
    theta: f64,
    gain: glue.Float = .bf16,
    h_end: u32 = 0,
    w_end: u32 = 0,
    plane: u32 = 0,
    geometry: planning.Geometry = .{},
    pub fn head(p: Preprocess, heads: u32, stride: u32) glue.Head {
        return .{ .width = p.geometry.dim, .rows = p.rows * heads, .stride = stride, .activation = .bf16, .gain = p.gain, .eps = p.eps };
    }
};
pub const Attention = struct {
    profiler: ?*profile.Trace = null,
    prefix: [32]mtl.Pipeline, // q27_attn_prefix_1 to _16, then q27_attn_prefix_masked_1 to _16
    io: [11]mtl.Pipeline,
    prompt: ?[6]mtl.Pipeline, // promptPipelines: tensor ops on the M5, simdgroup matrices before it
    units: bool,
    pub fn init(device: mtl.Device, library: mtl.Library) !Attention {
        var p: [32]mtl.Pipeline = undefined;
        var pn: usize = 0;
        errdefer for (p[0..pn]) |pipe| pipe.deinit();
        for (&p, 0..) |*pipe, i| {
            var name: [48]u8 = undefined;
            const kind = if (i < 16) "" else "masked_";
            pipe.* = try mtl.Pipeline.init(device, library, try std.fmt.bufPrint(&name, "q27_attn_prefix_{s}{d}", .{ kind, i % 16 + 1 }), false);
            pn += 1;
        }
        var io: [11]mtl.Pipeline = undefined;
        var n: usize = 0;
        errdefer for (io[0..n]) |pipe| pipe.deinit();
        for ([_][]const u8{ "q27_attn_pack_a", "q27_attn_pack_b", "q27_attn_append", "q27_attn_tail", "q27_attn_merge", "q27_attn_gather", "q27_attn_scatter", "q27_head_norm", "q27_rope", "q27_gate", "q27_attn_merge_single" }, &io) |name, *pipe| {
            pipe.* = try mtl.Pipeline.init(device, library, name, false);
            n += 1;
        }
        const units = device.tensorUnits();
        return .{ .prefix = p, .io = io, .prompt = promptPipelines(device, library, units) catch |err| if (units) return err else null, .units = units };
    }
    pub fn deinit(a: *Attention) void {
        for (a.prefix ++ a.io) |pipe| pipe.deinit();
        if (a.prompt) |pipes| for (pipes) |pipe| pipe.deinit();
    }
    /// Prompt kernel, pair and single merges, masked kernel, scan, pick; before M5 null below `prompt_threads`.
    pub fn promptPipelines(device: mtl.Device, library: mtl.Library, units: bool) !?[6]mtl.Pipeline {
        const names = if (units) [2][]const u8{ "q27_attn_prompt_8", "q27_attn_prompt_masked_8" } else [2][]const u8{ "q27_attn_prompt_sg_8", "q27_attn_prompt_sg_masked_8" };
        var out: [6]mtl.Pipeline = undefined;
        var n: usize = 0;
        errdefer for (out[0..n]) |pipe| pipe.deinit();
        for ([_][]const u8{ names[0], "q27_attn_prompt_merge", "q27_attn_prompt_merge_single", names[1], "q27_attn_prompt_scan", "q27_attn_prompt_pick" }, &out) |name, *pipe| {
            pipe.* = try mtl.Pipeline.init(device, library, name, false);
            n += 1;
        }
        const threads = @min(out[0].maxThreads(), out[3].maxThreads());
        if (!units and threads < prompt_threads) {
            std.log.info("qwen27: the prompt attention kernel runs {d} threads a group here, not {d}; prompt rows take the decode path", .{ threads, prompt_threads });
            for (out) |pipe| pipe.deinit();
            return null;
        }
        return out;
    }
    pub fn preprocess(a: *const Attention, input_encoder: mtl.ComputeEncoder, p: Preprocess) !void {
        try p.geometry.check();
        const g = p.geometry;
        const q_bytes = @as(usize, p.rows) * g.heads * g.dim * 2;
        const kv_bytes = @as(usize, p.rows) * g.kv_heads * g.dim * 2;
        const gain_bytes = @as(usize, g.dim) * (if (p.gain == .f32) @as(usize, 4) else 2);
        var timing = profile.begin(a.profiler, input_encoder, .attention_prepare);
        errdefer timing.cancel();
        const e = timing.encoder;

        if (p.rows == 0 or p.rows > @import("gpu_frame.zig").max_rows or !std.math.isFinite(p.eps) or p.eps <= 0 or !std.math.isFinite(p.theta) or p.theta <= 0) return error.TargetAttentionShape;
        try span(p.qg, 2 * q_bytes);
        try span(p.key, kv_bytes);
        try span(p.q_gain, gain_bytes);
        try span(p.k_gain, gain_bytes);
        try span(p.norm_q, q_bytes);
        try span(p.norm_k, kv_bytes);
        try span(p.query, q_bytes);
        try span(p.rotated_key, kv_bytes);
        if ((p.h_end > 0 or p.w_end > 0) and p.plane < p.rows) return error.TargetAttentionShape;
        const plane_rows: usize = if (p.h_end > 0 or p.w_end > 0) 2 * @as(usize, p.plane) + p.rows else p.rows;
        try spanAligned(p.positions, plane_rows * 4, 4);
        const outputs = [_]struct { ref: Ref, bytes: usize }{ .{ .ref = p.norm_q, .bytes = q_bytes }, .{ .ref = p.norm_k, .bytes = kv_bytes }, .{ .ref = p.query, .bytes = q_bytes }, .{ .ref = p.rotated_key, .bytes = kv_bytes } };
        const inputs = [_]struct { ref: Ref, bytes: usize }{ .{ .ref = p.qg, .bytes = 2 * q_bytes }, .{ .ref = p.key, .bytes = kv_bytes }, .{ .ref = p.q_gain, .bytes = gain_bytes }, .{ .ref = p.k_gain, .bytes = gain_bytes }, .{ .ref = p.positions, .bytes = plane_rows * 4 } };
        for (outputs, 0..) |dst, i| {
            for (inputs) |src| try disjoint(dst.ref, dst.bytes, src.ref, src.bytes);
            for (outputs[0..i]) |prior| try disjoint(dst.ref, dst.bytes, prior.ref, prior.bytes);
        }
        e.setPipeline(a.io[7]);
        bind(e, .{ p.qg, p.q_gain });
        e.setValue(p.head(g.heads, 2 * g.dim), 2);
        e.setBuffer(p.norm_q.buffer, p.norm_q.offset, 3);
        e.dispatchGroups(mtl.Size.of(p.rows * g.heads, 1, 1), mtl.Size.of(64, 1, 1));
        e.setPipeline(a.io[7]);
        bind(e, .{ p.key, p.k_gain });
        e.setValue(p.head(g.kv_heads, g.dim), 2);
        e.setBuffer(p.norm_k.buffer, p.norm_k.offset, 3);
        e.dispatchGroups(mtl.Size.of(p.rows * g.kv_heads, 1, 1), mtl.Size.of(64, 1, 1));
        e.barrier();
        for ([_]struct { x: Ref, y: Ref, heads: u32 }{ .{ .x = p.norm_q, .y = p.query, .heads = g.heads }, .{ .x = p.norm_k, .y = p.rotated_key, .heads = g.kv_heads } }) |r| {
            e.setPipeline(a.io[8]);
            bind(e, .{ r.x, p.positions });
            e.setValue(glue.Rope{ .width = g.dim, .heads = r.heads, .rows = p.rows, .rotary = g.rotary, .activation = .bf16, .theta = @floatCast(p.theta), .h_end = p.h_end, .w_end = p.w_end, .plane = p.plane }, 2);
            e.setBuffer(r.y.buffer, r.y.offset, 3);
            e.dispatchThreads(mtl.Size.of(g.dim, r.heads, p.rows), mtl.Size.of(32, 1, 1));
        }
        e.barrier();
        try timing.end();
    }
    /// The keys' half of `preprocess` and `encode`'s append: a pass whose rows reach only the cache.
    pub fn keysOnly(a: *const Attention, input_encoder: mtl.ComputeEncoder, scratch: *const Scratch, binding: *const Binding, plan: Plan, p: Preprocess, v: Ref, caches: []const Cache) !void {
        try plan.geometry.check();
        try p.geometry.check();
        const g = plan.geometry;
        const pg = p.geometry; // preprocess's geometry, as attentionForward passes it
        const kv_bytes = @as(usize, p.rows) * g.kv_heads * g.dim * 2;
        if (p.rows != plan.rows or p.rows == 0 or p.rows > @import("gpu_frame.zig").max_rows or !std.math.isFinite(p.eps) or p.eps <= 0 or !std.math.isFinite(p.theta) or p.theta <= 0) return error.TargetAttentionShape;
        if ((p.h_end > 0 or p.w_end > 0) and p.plane < p.rows) return error.TargetAttentionShape;
        const plane_rows: usize = if (p.h_end > 0 or p.w_end > 0) 2 * @as(usize, p.plane) + p.rows else p.rows;
        try span(p.key, kv_bytes);
        try span(p.k_gain, @as(usize, g.dim) * 2);
        try span(p.norm_k, kv_bytes);
        try span(p.rotated_key, kv_bytes);
        try span(v, kv_bytes);
        try spanAligned(p.positions, plane_rows * 4, 4);
        for ([_]Ref{ p.norm_k, p.rotated_key }) |dst| for ([_]Ref{ p.key, v }) |src| try disjoint(dst, kv_bytes, src, kv_bytes);
        try disjoint(p.norm_k, kv_bytes, p.rotated_key, kv_bytes);
        try scratch.check(plan);
        try binding.check(plan, caches);
        for (caches) |cache| for ([_]Ref{ p.rotated_key, v }) |operand| try apart(cache, g, operand, kv_bytes);
        var timing = profile.begin(a.profiler, input_encoder, .attention_prepare);
        errdefer timing.cancel();
        const e = timing.encoder;
        e.setPipeline(a.io[7]);
        bind(e, .{ p.key, p.k_gain });
        e.setValue(glue.Head{ .width = pg.dim, .rows = p.rows * pg.kv_heads, .stride = pg.dim, .activation = .bf16, .gain = .bf16, .eps = p.eps }, 2);
        e.setBuffer(p.norm_k.buffer, p.norm_k.offset, 3);
        e.dispatchGroups(mtl.Size.of(p.rows * pg.kv_heads, 1, 1), mtl.Size.of(64, 1, 1));
        e.barrier();
        e.setPipeline(a.io[8]);
        bind(e, .{ p.norm_k, p.positions });
        e.setValue(glue.Rope{ .width = pg.dim, .heads = pg.kv_heads, .rows = p.rows, .rotary = pg.rotary, .activation = .bf16, .theta = @floatCast(p.theta), .h_end = p.h_end, .w_end = p.w_end, .plane = p.plane }, 2);
        e.setBuffer(p.rotated_key.buffer, p.rotated_key.offset, 3);
        e.dispatchThreads(mtl.Size.of(pg.dim, pg.kv_heads, p.rows), mtl.Size.of(32, 1, 1));
        e.barrier();
        for (scratch.buffers) |buffer| e.useResource(buffer, mtl.ResourceUsage.read | mtl.ResourceUsage.write);
        for (binding.buffers) |buffer| e.useResource(buffer, mtl.ResourceUsage.read | mtl.ResourceUsage.write);
        useCaches(e, caches);
        const args = PackArgs{ .rows = plan.rows, .packed_rows = plan.tiles * 16, .heads = g.heads, .kv_heads = g.kv_heads, .dim = g.dim, .group = g.group() };
        e.setPipeline(a.io[2]);
        bind(e, .{ p.rotated_key, v, binding.get(.caches), binding.get(.meta), scratch.get(.nodes) });
        e.setValue(args, 5);
        e.dispatchThreads(mtl.Size.of(g.dim, g.kv_heads, plan.rows), mtl.Size.of(32, 4, 1));
        try timing.end();
    }

    /// `prompt` picks the prompt kernel; `split` reads the head count, constant when one Mac runs it.
    pub fn encode(a: *const Attention, input_encoder: mtl.ComputeEncoder, scratch: *const Scratch, binding: *const Binding, plan: Plan, q: Ref, k: Ref, v: Ref, out: Ref, caches: []const Cache, raw_gate: ?Ref, prompt: bool, split: bool) !void {
        try plan.geometry.check();
        const g = plan.geometry;
        const q_bytes = @as(usize, plan.rows) * g.heads * g.dim * 2;
        const kv_bytes = @as(usize, plan.rows) * g.kv_heads * g.dim * 2;
        var timing = profile.begin(a.profiler, input_encoder, .attention);
        errdefer timing.cancel();
        const e = timing.encoder;

        try scratch.check(plan);
        try binding.check(plan, caches);
        try spanAligned(q, q_bytes, 16);
        try span(k, kv_bytes);
        try span(v, kv_bytes);
        try span(out, q_bytes);
        if (raw_gate) |gate| {
            try span(gate, 2 * q_bytes);
            try disjoint(out, q_bytes, gate, 2 * q_bytes);
        }
        const operands = [_]struct { ref: Ref, bytes: usize }{ .{ .ref = q, .bytes = q_bytes }, .{ .ref = k, .bytes = kv_bytes }, .{ .ref = v, .bytes = kv_bytes }, .{ .ref = out, .bytes = q_bytes } };
        for (caches) |cache| for (operands) |operand| try apart(cache, plan.geometry, operand.ref, operand.bytes);
        for (scratch.buffers) |buffer| e.useResource(buffer, mtl.ResourceUsage.read | mtl.ResourceUsage.write);
        for (binding.buffers) |buffer| e.useResource(buffer, mtl.ResourceUsage.read | mtl.ResourceUsage.write);
        useCaches(e, caches);
        const args = PackArgs{ .rows = plan.rows, .packed_rows = plan.tiles * 16, .heads = g.heads, .kv_heads = g.kv_heads, .dim = g.dim, .group = g.group() };
        e.setPipeline(a.io[2]);
        bind(e, .{ k, v, binding.get(.caches), binding.get(.meta), scratch.get(.nodes) });
        e.setValue(args, 5);
        e.dispatchThreads(mtl.Size.of(g.dim, g.kv_heads, plan.rows), mtl.Size.of(32, 4, 1));
        e.setPipeline(a.io[0]);
        bind(e, .{ q, scratch.get(.q_rows), scratch.get(.qa) });
        e.setValue(args, 3);
        e.dispatchThreads(mtl.Size.of(g.dim, args.packed_rows, g.kv_heads), mtl.Size.of(32, 4, 1));
        if (prompt and a.prompt != null and (plan.geometry.shared_prefix or !a.units) and plan.streams == 1) {
            e.barrier();
            try a.promptRows(e, scratch, binding, plan, out, split);
            if (raw_gate) |gate| {
                e.setPipeline(a.io[9]);
                bind(e, .{ out, gate });
                e.setValue(glue.Element{ .width = g.dim, .heads = g.heads, .rows = plan.rows, .activation = .bf16 }, 2);
                e.setBuffer(out.buffer, out.offset, 3);
                e.dispatchThreads(mtl.Size.of(g.dim, g.heads, plan.rows), mtl.Size.of(32, 1, 1));
                e.barrier();
            }
        } else try a.treeRows(e, scratch, binding, plan, q, out, raw_gate, args, split);
        try timing.end();
    }
    /// `grid` at pick + 16 when every value some row must not see is finite, else at pick + 28; the other grid is 0.
    pub fn encodePick(e: mtl.ComputeEncoder, pipes: [2]mtl.Pipeline, caches: Ref, args: PromptArgs, pick: mtl.Buffer, grid: [3]u32) void {
        e.setPipeline(pipes[0]);
        bind(e, .{caches});
        e.setValue(args, 1);
        e.setBuffer(pick, 0, 2);
        e.dispatchGroups(mtl.Size.of(@intCast(args.kv_heads), @intCast(@max(@divFloor(args.rows + 31, 32), 1)), 1), mtl.Size.of(256, 1, 1));
        e.barrier();
        e.setPipeline(pipes[1]);
        e.setBuffer(pick, 0, 0);
        e.setValue(grid, 1);
        e.setBuffer(pick, 16, 2);
        e.dispatchGroups(mtl.Size.of(1, 1, 1), mtl.Size.of(1, 1, 1));
        e.barrier();
    }
    /// A prompt chunk's rows over absolute 512-key chunks (attention_prompt.metal), merged in key order.
    fn promptRows(a: *const Attention, e: mtl.ComputeEncoder, scratch: *const Scratch, binding: *const Binding, plan: Plan, out: Ref, split: bool) !void {
        const g = plan.geometry;
        const start: i32 = plan.meta[planning.global_words + 4];
        for (plan.positions, 0..) |position, i| if (position != start + @as(i32, @intCast(i))) return error.PromptRowsNotChain; // the kernel places row i at start + i
        const chunks: u32 = @intCast(@divFloor(start + @as(i32, @intCast(plan.rows)) + planning.chunk - 1, planning.chunk));
        if (chunks > scratch.chunks or plan.tiles > scratch.tiles) return error.AttentionLimits;
        // before M5 (no shared prefix) the partials take the tail's buffers, which hold every chunk's packed rows
        const parts = if (g.shared_prefix) [3]Field{ .poa, .pma, .pla } else [3]Field{ .pob, .pmb, .plb };
        if (!g.shared_prefix and @as(usize, chunks) * plan.tiles * 16 > @as(usize, scratch.tail_chunks) * scratch.limits.rows * 16) return error.AttentionLimits;
        const args = PromptArgs{ .start = start, .rows = @intCast(plan.rows), .chunks = @intCast(chunks), .rp = @intCast(plan.tiles * 16), .key_stride = @intCast(binding.strides[0][0]), .value_stride = @intCast(binding.strides[0][1]), .scale2 = 0.0625 * std.math.log2e, .kv_heads = @intCast(g.kv_heads) };
        const pairs = prompt_threads / 64; // simdgroup pairs, each halving the head dim of one 16-row tile
        const pick = scratch.get(.pick).buffer;
        encodePick(e, a.prompt.?[4..6].*, binding.get(.caches), args, pick, .{ g.kv_heads, chunks, (plan.tiles + pairs - 1) / pairs });
        for ([_]usize{ 0, 3 }, [_]usize{ 16, 28 }) |which, at| { // one of the two grids is empty: no barrier between
            e.setPipeline(a.prompt.?[which]);
            bind(e, .{ scratch.get(.qa), binding.get(.caches) });
            e.setValue(args, 2);
            e.setBuffer(scratch.get(parts[0]).buffer, 0, 3);
            e.setBuffer(scratch.get(parts[1]).buffer, 0, 4);
            e.setBuffer(scratch.get(parts[2]).buffer, 0, 5);
            e.dispatchIndirect(pick, at, mtl.Size.of(prompt_threads, 1, 1));
        }
        e.barrier();
        e.setPipeline(a.prompt.?[if (split) 1 else 2]);
        bind(e, .{ scratch.get(parts[0]), scratch.get(parts[1]), scratch.get(parts[2]) });
        e.setValue(args, 3);
        e.setBuffer(out.buffer, out.offset, 4);
        e.dispatchGroups(mtl.Size.of(g.kv_heads, plan.rows * g.group(), 1), mtl.Size.of(32, 1, 1));
        e.barrier();
    }
    /// Trees and decoded rows: cached keys by chunk (prefix), the round's own by path (tail), merged and gated.
    fn treeRows(a: *const Attention, e: mtl.ComputeEncoder, scratch: *const Scratch, binding: *const Binding, plan: Plan, q: Ref, out: Ref, raw_gate: ?Ref, args: PackArgs, split: bool) !void {
        const g = plan.geometry;
        if (!plan.fused) {
            e.setPipeline(a.io[1]);
            bind(e, .{ q, scratch.get(.qb) });
            e.setValue(args, 2);
            e.dispatchThreads(mtl.Size.of(g.dim, 16, g.kv_heads * plan.rows), mtl.Size.of(32, 4, 1));
        }
        e.barrier();
        const scale: f32 = 0.0625;
        if (plan.prefix_chunks > 0) {
            const sg: u32 = @min(plan.tiles, if (plan.prefix_chunks >= 64) @as(u32, 6) else 4); // a pair a tile; a long cache reads each chunk once for six tiles
            // a stream of one row sees every key its tile reads; only a stream of rows can mask a value
            e.setPipeline(a.prefix[if (plan.rows > plan.streams) 16 + sg - 1 else sg - 1]);
            bind(e, .{ scratch.get(.qa), binding.get(.caches) });
            e.setValue(scale, 2);
            e.setBuffer(binding.get(.meta).buffer, 0, 3);
            e.setBuffer(scratch.get(.tile_stream).buffer, 0, 4);
            e.setBuffer(scratch.get(.poa).buffer, 0, 5);
            e.setBuffer(scratch.get(.pma).buffer, 0, 6);
            e.setBuffer(scratch.get(.pla).buffer, 0, 7);
            e.setBuffer(scratch.get(.nodes).buffer, 0, 8);
            e.dispatchGroups(mtl.Size.of(g.kv_heads, plan.prefix_chunks, (plan.tiles + sg - 1) / sg), mtl.Size.of(64 * sg, 1, 1));
            e.barrier();
        }
        if (!plan.fused) {
            e.setPipeline(a.io[3]);
            bind(e, .{ scratch.get(.qb), binding.get(.caches) });
            e.setValue(scale, 2);
            e.setBuffer(binding.get(.meta).buffer, 0, 3);
            for ([_]Field{ .paths, .nodes, .poa, .pma, .pla, .pob, .pmb, .plb }, 4..) |field, i| {
                const r = scratch.get(field);
                e.setBuffer(r.buffer, 0, i);
            }
            e.dispatchGroups(mtl.Size.of(g.kv_heads, plan.tail_chunks, plan.rows), mtl.Size.of(128, 1, 1));
            e.barrier();
        }
        e.setPipeline(a.io[if (split) 4 else 10]);
        bind(e, .{ scratch.get(.poa), scratch.get(.pma), scratch.get(.pla), scratch.get(.pob), scratch.get(.pmb), scratch.get(.plb), binding.get(.meta), scratch.get(.nodes), out, raw_gate orelse out });
        e.setValue(@as(u32, @intFromBool(raw_gate != null)), 10);
        if (split) e.setValue(g.heads, 11);
        e.dispatchGroups(mtl.Size.of(g.kv_heads, plan.rows * g.group(), 1), mtl.Size.of(32, 1, 1));
        e.barrier();
    }
    pub fn keep(a: *const Attention, e: mtl.ComputeEncoder, scratch: *const Scratch, binding: *const Binding, kept: u32, caches: []const Cache) !void {
        if (kept > scratch.limits.rows or binding.kept == null or binding.kept.? != kept) return error.AttentionKeep;
        try binding.checkCaches(caches);
        const g = binding.limits.geometry;
        // the staging rows (scratch's keep_k, keep_v) hold scratch's KV heads; the caches' must be the same count
        if (scratch.limits.geometry.kv_heads != g.kv_heads) return error.AttentionKeep;
        const args = KeepArgs{ .kept = kept, .kv_heads = g.kv_heads };
        for (scratch.buffers) |buffer| e.useResource(buffer, mtl.ResourceUsage.read | mtl.ResourceUsage.write);
        for (binding.buffers) |buffer| e.useResource(buffer, mtl.ResourceUsage.read | mtl.ResourceUsage.write);
        if (kept == 0) return;
        useCaches(e, caches);
        e.setPipeline(a.io[5]);
        bind(e, .{ binding.get(.caches), binding.get(.meta), binding.get(.keep_map), scratch.get(.keep_k), scratch.get(.keep_v) });
        e.setValue(args, 5);
        e.dispatchThreads(mtl.Size.of(g.dim, g.kv_heads, kept), mtl.Size.of(32, 4, 1));
        e.barrier();
        e.setPipeline(a.io[6]);
        bind(e, .{ scratch.get(.keep_k), scratch.get(.keep_v), binding.get(.caches), binding.get(.meta), binding.get(.keep_map) });
        e.setValue(args, 5);
        e.dispatchThreads(mtl.Size.of(g.dim, g.kv_heads, kept), mtl.Size.of(32, 4, 1));
        e.barrier();
    }
};
test "cache and pack descriptors match native pointer and six-word ABI" {
    try std.testing.expectEqual(@as(usize, 16), @sizeOf(CachePtrs));
    try std.testing.expectEqual(@as(usize, 32), @sizeOf(PagedPtrs));
    try std.testing.expectEqual(@as(usize, 24), @sizeOf(PackArgs));
    try std.testing.expectEqual(@as(usize, 16), @sizeOf(planning.KeepRow));
}

test "plan fingerprint changes for an ancestor mutation rather than only counts" {
    var plan = try Plan.init(std.testing.allocator, &.{.{ .parents = &.{ -1, 0, 0 }, .start = 63, .capacity = 512 }}, .{});
    defer plan.deinit();
    const before = fingerprint(plan);
    plan.paths[2 * planning.path_width + 1] = 1;
    try std.testing.expect(!std.mem.eql(u8, &before, &fingerprint(plan)));
}

test "cache span includes padded head strides but no unused trailing padding" {
    const cache = Cache{ .keys = undefined, .values = undefined, .capacity = 8, .key_stride = 4096, .value_stride = 8192 };
    try std.testing.expectEqual(@as(usize, 28672), cacheBytes(cache, true, .{}));
    try std.testing.expectEqual(@as(usize, 53248), cacheBytes(cache, false, .{}));
    const contiguous = Cache{ .keys = undefined, .values = undefined, .capacity = 8 };
    try std.testing.expectEqual(@as(usize, 16384), cacheBytes(contiguous, true, .{}));
}

test "descriptor identity includes numeric geometry and limits still refuse new shaders" {
    var plan = try Plan.init(std.testing.allocator, &.{.{ .parents = &.{-1}, .start = 0, .capacity = 8 }}, .{});
    defer plan.deinit();
    const before = fingerprint(plan);
    plan.geometry.heads = 20; // GQA 5: no shader build takes it
    try std.testing.expect(!std.mem.eql(u8, &before, &fingerprint(plan)));
    try std.testing.expectError(error.TargetAttentionShape, validLimits(.{ .geometry = plan.geometry }));
}

test "GGUF attention: native gain defaults to BF16 and exact F32 gains reach both head bindings" {
    var p: Preprocess = .{ .qg = undefined, .key = undefined, .q_gain = undefined, .k_gain = undefined, .norm_q = undefined, .norm_k = undefined, .query = undefined, .rotated_key = undefined, .positions = undefined, .rows = 3, .eps = 1e-6, .theta = 1e7 };
    try std.testing.expectEqual(glue.Float.bf16, p.head(24, 512).gain);
    p.gain = .f32;
    const q = p.head(24, 512);
    const k = p.head(4, 256);
    try std.testing.expectEqual(glue.Float.f32, q.gain);
    try std.testing.expectEqual(glue.Float.f32, k.gain);
    try std.testing.expectEqual(@as(u32, 72), q.rows);
    try std.testing.expectEqual(@as(u32, 12), k.rows);
    try std.testing.expectEqual(@as(u32, 256), q.width);
    try std.testing.expectEqual(@as(u32, 512), q.stride);
}
