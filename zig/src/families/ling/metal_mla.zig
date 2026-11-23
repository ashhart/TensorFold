//! Ling MLA projections, latent cache and unchanged Kimi mixer with the three Ling RoPE/gate kernels.
const std = @import("std");
const mtl = @import("metal");
const kernels = @import("kimi_kernels");
const ck = @import("ling_checkpoint");
const Rows = @import("ling_rows").Rows;
const Ref = kernels.Ref;

pub const Mla = struct {
    library: mtl.Library,
    rope_cache: mtl.Pipeline,
    rope_q: mtl.Pipeline,
    gate_expand: mtl.Pipeline,
    cache: mtl.Buffer,
    positions: mtl.Buffer,
    angles: mtl.Buffer,
    qa: mtl.Buffer,
    qan: mtl.Buffer,
    q: mtl.Buffer,
    kv: mtl.Buffer,
    gate: mtl.Buffer,
    expanded: mtl.Buffer,
    qlat: mtl.Buffer,
    partial: mtl.Buffer,
    olat: mtl.Buffer,
    y: mtl.Buffer,
    heads: u32,
    capacity: u32,
    position: u32 = 0,

    pub fn init(rows: *Rows, heads: u32, capacity: u32) !Mla {
        const buffer_mark = rows.owned.items.len;
        errdefer rows.rollbackBuffers(buffer_mark);
        const library = try mtl.Library.fromSource(rows.device, @import("ling_sources").mla, mtl.CompileOptions.mlx());
        errdefer library.deinit();
        const rope_cache = try mtl.Pipeline.init(rows.device, library, "tf_ling_mla_rope_cache", false);
        errdefer rope_cache.deinit();
        const rope_q = try mtl.Pipeline.init(rows.device, library, "tf_ling_mla_rope_q", false);
        errdefer rope_q.deinit();
        const gate_expand = try mtl.Pipeline.init(rows.device, library, "tf_ling_mla_gate_expand", false);
        errdefer gate_expand.deinit();
        const r: usize = rows.max_rows;
        const cache = try rows.buffer(@as(usize, capacity) * 576 * 2);
        @memset(cache.slice(u16, capacity * 576), 0);
        const positions = try rows.buffer(r * @sizeOf(kernels.MlaRow));
        const angles = try rows.buffer(r * 64 * 4);
        const qa = try rows.buffer(r * 256 * 2);
        const qan = try rows.buffer(r * 256 * 2);
        const q = try rows.buffer(r * heads * 192 * 2);
        const kv = try rows.buffer(r * 576 * 2);
        const gate = try rows.buffer(r * heads * 2);
        const expanded = try rows.buffer(r * heads * 128 * 2);
        const qlat = try rows.buffer(r * heads * 512 * 4);
        const partial = try rows.buffer(r * heads * 16 * 514 * 4);
        const olat = try rows.buffer(r * heads * 512 * 4);
        const y = try rows.buffer(r * heads * 128 * 2);
        return .{
            .library = library,
            .rope_cache = rope_cache,
            .rope_q = rope_q,
            .gate_expand = gate_expand,
            .cache = cache,
            .heads = heads,
            .capacity = capacity,
            .positions = positions,
            .angles = angles,
            .qa = qa,
            .qan = qan,
            .q = q,
            .kv = kv,
            .gate = gate,
            .expanded = expanded,
            .qlat = qlat,
            .partial = partial,
            .olat = olat,
            .y = y,
        };
    }

    fn name(rows: *Rows, prefix: []const u8, part: []const u8) ![]u8 {
        return std.mem.concat(rows.allocator, u8, &.{ prefix, part });
    }

    fn project(self: *Mla, rows: *Rows, model: *ck.Checkpoint, prefix: []const u8, part: []const u8, x: Ref, y: mtl.Buffer, n: u32, width: u32, out_width: u32) !void {
        _ = self;
        const full = try name(rows, prefix, part);
        defer rows.allocator.free(full);
        try rows.linear(model, full, x, .{ .buf = y }, n, width, out_width, false);
    }

    pub fn forward(self: *Mla, rows: *Rows, model: *ck.Checkpoint, prefix: []const u8, x: Ref, out: Ref, n: u32, hidden: u32, eps: f32, theta: f64) !void {
        if (n == 0 or n > rows.max_rows or self.position > self.capacity or n > self.capacity - self.position) return error.ContextCapacity;
        try self.project(rows, model, prefix, "q_a_proj.weight", x, self.qa, n, hidden, 256);
        const qa_norm = try name(rows, prefix, "q_a_layernorm.weight");
        defer rows.allocator.free(qa_norm);
        try rows.norm(model, qa_norm, .{ .buf = self.qa }, .{ .buf = self.qan }, n, 256, eps);
        try self.project(rows, model, prefix, "q_b_proj.weight", .{ .buf = self.qan }, self.q, n, 256, self.heads * 192);
        try self.project(rows, model, prefix, "kv_a_proj_with_mqa.weight", x, self.kv, n, hidden, 576);
        try self.project(rows, model, prefix, "g_proj.weight", x, self.gate, n, hidden, self.heads);
        const wn = try name(rows, prefix, "kv_a_layernorm.weight");
        defer rows.allocator.free(wn);
        const w = try model.get(wn, .bf16, &.{512});
        const bn = try name(rows, prefix, "kv_b_proj.weight");
        defer rows.allocator.free(bn);
        const b = try model.get(bn, .bf16, &.{ self.heads * 256, 512 });
        for (self.positions.slice(kernels.MlaRow, n), 0..) |*p, t| {
            p.* = .{ .slot = 0, .pos = self.position + @as(u32, @intCast(t)) };
            for (0..32) |i| {
                const angle = @as(f64, @floatFromInt(p.pos)) / std.math.pow(f64, theta, @as(f64, @floatFromInt(2 * i)) / 64.0);
                self.angles.slice(f32, n * 64)[t * 64 + i * 2] = @floatCast(@cos(angle));
                self.angles.slice(f32, n * 64)[t * 64 + i * 2 + 1] = @floatCast(@sin(angle));
            }
        }
        const a = kernels.MlaArgs{ .cache = self.cache.gpuAddress(), .slot_keys = self.capacity, .heads = self.heads, .head0 = 0, .rows = n, .q_stride = self.heads * 192, .kv_stride = 576, .gate_stride = self.heads * 128, .out_stride = self.heads * 128, .eps = eps, .scale = 1.0 / @sqrt(@as(f32, 192)) };
        const cb = rows.queue.commandBuffer();
        const e = cb.compute(.serial);
        e.useResource(self.cache, mtl.ResourceUsage.read | mtl.ResourceUsage.write);
        rows.k.mlaCache(e, .{ .buf = self.kv }, .{ .buf = w.buffer, .off = w.offset }, .{ .buf = self.positions }, a);
        e.setPipeline(self.rope_cache);
        e.setBuffer(self.positions, 0, 0);
        e.setValue(a, 1);
        e.setBuffer(self.angles, 0, 2);
        e.dispatchGroups(mtl.Size.of(n, 1, 1), mtl.Size.of(32, 1, 1));
        rows.k.mlaQlat(e, .{ .buf = self.q }, .{ .buf = b.buffer, .off = b.offset }, .{ .buf = self.qlat }, a);
        e.setPipeline(self.rope_q);
        e.setBuffer(self.q, 0, 0);
        e.setValue(a, 1);
        e.setBuffer(self.angles, 0, 2);
        e.dispatchGroups(mtl.Size.of(self.heads, n, 1), mtl.Size.of(32, 1, 1));
        rows.k.mlaAttend(e, .{ .buf = self.q }, .{ .buf = self.qlat }, .{ .buf = self.positions }, .{ .buf = self.partial }, a);
        rows.k.mlaMerge(e, .{ .buf = self.partial }, .{ .buf = self.positions }, .{ .buf = self.olat }, a);
        e.setPipeline(self.gate_expand);
        e.setBuffer(self.gate, 0, 0);
        e.setBuffer(self.expanded, 0, 1);
        e.setValue(a, 2);
        e.dispatchThreads(mtl.Size.of(self.heads * 128, n, 1), mtl.Size.of(128, 1, 1));
        rows.k.mlaUv(e, .{ .buf = self.olat }, .{ .buf = b.buffer, .off = b.offset }, .{ .buf = self.expanded }, .{ .buf = self.y }, a);
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure() != null) return error.GpuFailed;
        const dense = try name(rows, prefix, "dense.weight");
        defer rows.allocator.free(dense);
        try rows.linear(model, dense, .{ .buf = self.y }, out, n, self.heads * 128, hidden, false);
        self.position += n;
    }

    pub fn deinit(self: *Mla) void {
        self.gate_expand.deinit();
        self.rope_q.deinit();
        self.rope_cache.deinit();
        self.library.deinit();
    }
};
