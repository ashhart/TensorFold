//! Folded Memory Weights v3 Metal adapter for Qwen3.8-27B and Qwen3.5-MoE (`qwen27_served.zig`).
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const q = tf.qwen27;
const living = @import("living");
const mem = living.memory;
const lb = @import("qwen27_lanes.zig");
const Allocator = std.mem.Allocator;

pub const Adapter = struct {
    gpa: Allocator,
    back: *lb.Metal,
    tap_lex: mtl.Buffer,
    tap_sem: mtl.Buffer,

    pub fn init(gpa: Allocator, back: *lb.Metal) !*Adapter {
        const a = try gpa.create(Adapter);
        errdefer gpa.destroy(a);
        const r = back.runner;
        const bytes = @as(usize, r.model.frame.capacity) * r.model.config.hidden * 2;
        const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;
        const tap_lex = try r.model.device.buffer(bytes, opts);
        errdefer tap_lex.deinit();
        const tap_sem = try r.model.device.buffer(bytes, opts);
        a.* = .{ .gpa = gpa, .back = back, .tap_lex = tap_lex, .tap_sem = tap_sem };
        return a;
    }

    pub fn deinit(self: *Adapter, gpa: Allocator) void {
        self.tap_lex.deinit();
        self.tap_sem.deinit();
        gpa.destroy(self);
    }

    pub fn deinitOpaque(ptr: *anyopaque, gpa: Allocator) void {
        const self: *Adapter = @ptrCast(@alignCast(ptr));
        self.deinit(gpa);
    }

    pub fn archSpec(back: *const lb.Metal) mem.ArchSpec {
        const r = back.runner;
        const c = r.model.config;
        const num_attn = r.caches.len / r.slots;
        const num_rec = r.gdn.budget.layers;
        return .{
            .family = .qwen27,
            .hidden = c.hidden,
            .num_layers = c.layers,
            .layer_lex = 1,
            .layer_sem = (2 * c.layers) / 3, // layer 42 of 64 on Qwen3.8-27B
            .num_attn = num_attn, // 16 (+1 if MTP head)
            .kv_heads = c.kv_heads, // 4 on 27B
            .head_dim = c.head_dim, // 256
            .rotary_dim = c.rope_dims, // 64
            .rope_theta = c.rope_theta, // 10_000_000.0
            .num_rec = num_rec, // 48
            .rec_heads = c.v_heads, // 48
            .rec_rows = c.dv, // 128
            .rec_cols = c.dk, // 128
            .conv_elems = r.gdn.budget.shape.conv,
        };
    }

    pub fn backend(self: *Adapter) mem.Backend {
        return .{
            .ctx = self,
            .arch = archSpec(self.back),
            .extractHidden = extractHiddenFn,
            .compileState = compileStateFn,
            .buildSnap = buildSnapFn,
            .dropSnap = dropSnapFn,
        };
    }

    fn of(ctx: *anyopaque) *Adapter {
        return @ptrCast(@alignCast(ctx));
    }

    /// Borrows an idle ready slot without allocating extra KV VRAM. Clears slot 0 standing state if used.
    fn pickIdleSlot(self: *Adapter) !u32 {
        const b = self.back;
        b.settle() catch {};
        b.dropRound();
        // Prefer any free ready slot > 0 first so slot 0's standing conversation stays untouched if possible
        var chosen: ?u32 = null;
        for (0..b.runner.ready) |i| {
            if (b.owners[i] == null) {
                chosen = @intCast(i);
                if (i > 0) break;
            }
        }
        const slot = chosen orelse return error.NoIdleMemorySlot;
        if (slot == 0) {
            b.standing.deinit(b.gpa);
            b.standing.rewind_at = 0;
        }
        b.runner.settleSlot(slot);
        b.runner.rope[slot] = .{};
        return slot;
    }

    fn releaseIdleSlot(self: *Adapter, slot: u32) void {
        const b = self.back;
        b.runner.settleSlot(slot);
        b.runner.reset(slot) catch {};
    }

    fn writeRawIntoSharedState(r: *const q.decode_round.Runner, st: *q.snapshot.State, src: *const mem.RawState) void {
        const c = r.model.config;
        const buf = st.buf.contents()[0..st.bytes];
        const st_bytes = r.gdn.budget.shape.state * 4;
        const cv_bytes = r.gdn.budget.shape.conv * 2;
        var off: usize = 0;
        for (0..r.gdn.budget.layers) |l| {
            @memcpy(buf[off..][0..st_bytes], std.mem.sliceAsBytes(src.rec_state[l]));
            off += st_bytes;
            @memcpy(buf[off..][0..cv_bytes], std.mem.sliceAsBytes(src.rec_conv[l]));
            off += cv_bytes;
        }
        const num_attn = r.caches.len / r.slots;
        const kv_bytes = c.kv_heads * @as(usize, src.len) * c.head_dim * 2;
        for (0..num_attn) |a| {
            @memcpy(buf[off..][0..kv_bytes], std.mem.sliceAsBytes(src.kv_k[a]));
            off += kv_bytes;
            @memcpy(buf[off..][0..kv_bytes], std.mem.sliceAsBytes(src.kv_v[a]));
            off += kv_bytes;
        }
        std.debug.assert(off == st.bytes);
    }

    fn readSliceFromSharedState(
        r: *const q.decode_round.Runner,
        st: *const q.snapshot.State,
        start: usize,
        len: usize,
        dst: *mem.RawState,
    ) void {
        const c = r.model.config;
        const buf = st.buf.contents()[0..st.bytes];
        const st_bytes = r.gdn.budget.shape.state * 4;
        const cv_bytes = r.gdn.budget.shape.conv * 2;
        var off: usize = 0;
        for (0..r.gdn.budget.layers) |l| {
            @memcpy(std.mem.sliceAsBytes(dst.rec_state[l]), buf[off..][0..st_bytes]);
            off += st_bytes;
            @memcpy(std.mem.sliceAsBytes(dst.rec_conv[l]), buf[off..][0..cv_bytes]);
            off += cv_bytes;
        }
        const num_attn = r.caches.len / r.slots;
        const total_at: usize = st.at;
        const head_src_bytes = total_at * c.head_dim * 2;
        const head_dst_bytes = len * c.head_dim * 2;
        const row_off_bytes = start * c.head_dim * 2;
        for (0..num_attn) |a| {
            // Keys: [kv_heads, total_at, head_dim] bf16
            for (0..c.kv_heads) |h| {
                const s_off = off + h * head_src_bytes + row_off_bytes;
                const d_slice = std.mem.sliceAsBytes(dst.kv_k[a])[h * head_dst_bytes ..][0..head_dst_bytes];
                @memcpy(d_slice, buf[s_off..][0..head_dst_bytes]);
            }
            off += c.kv_heads * head_src_bytes;
            // Values: [kv_heads, total_at, head_dim] bf16
            for (0..c.kv_heads) |h| {
                const s_off = off + h * head_src_bytes + row_off_bytes;
                const d_slice = std.mem.sliceAsBytes(dst.kv_v[a])[h * head_dst_bytes ..][0..head_dst_bytes];
                @memcpy(d_slice, buf[s_off..][0..head_dst_bytes]);
            }
            off += c.kv_heads * head_src_bytes;
        }
        std.debug.assert(off == st.bytes);
    }

    fn extractHiddenFn(
        ctx: *anyopaque,
        ids: []const u32,
        layer_lex: usize,
        layer_sem: usize,
        out_lex: []f32,
        out_sem: []f32,
    ) anyerror!void {
        const self = of(ctx);
        const r = self.back.runner;
        const hidden = r.model.config.hidden;
        if (ids.len == 0) return;
        std.debug.assert(out_lex.len == ids.len * hidden and out_sem.len == ids.len * hidden);

        const slot = try self.pickIdleSlot();
        defer self.releaseIdleSlot(slot);

        try r.reset(slot);
        try r.admit(slot, @intCast(ids.len));
        r.model.kernels.prompt = true;
        defer r.model.kernels.prompt = false;

        const step = r.promptRows(@intCast(@min(self.back.chunk, r.model.frame.capacity)));
        var at: usize = 0;
        while (at < ids.len) {
            const end = @min(ids.len, at + step);
            const part = ids[at..end];
            var round = try q.round_plan.Round.init(
                r.allocator,
                &.{.{ .slot = slot, .start = r.offsets[slot], .capacity = r.capacity, .ids = part }},
                @intCast(r.model.config.conv_kernel),
                r.slots,
                @intCast(r.model.shard.global.vocab),
            );
            defer round.deinit();
            try r.advancePartialTap(&round, layer_lex, layer_sem, self.tap_lex, self.tap_sem);
            const lex_bf16 = self.tap_lex.slice(u16, part.len * hidden);
            const sem_bf16 = self.tap_sem.slice(u16, part.len * hidden);
            mem.quant.bf16SliceToF32(lex_bf16, out_lex[at * hidden ..][0 .. part.len * hidden]);
            mem.quant.bf16SliceToF32(sem_bf16, out_sem[at * hidden ..][0 .. part.len * hidden]);
            at = end;
        }
    }

    fn compileStateFn(
        ctx: *anyopaque,
        base: ?*const mem.RawState,
        ids: []const u32,
        out: *mem.RawState,
    ) anyerror!void {
        const self = of(ctx);
        const r = self.back.runner;
        const base_len: usize = if (base) |b| b.len else 0;
        if (ids.len == 0) return error.EmptyCompile;

        const slot = try self.pickIdleSlot();
        defer self.releaseIdleSlot(slot);

        if (base) |b| {
            const st = try q.snapshot.blank(self.gpa, r, null, b.len);
            defer q.snapshot.drop(self.gpa, st);
            writeRawIntoSharedState(r, st, b);
            try q.snapshot.restoreSlot(self.gpa, r, slot, null, st);
        } else {
            try r.reset(slot);
            try r.admit(slot, @intCast(ids.len));
        }

        const sess = q.session.Session{ .runner = r, .slot = slot };
        try sess.prefill(ids, @min(self.back.chunk, r.model.frame.capacity));

        const total_at = r.offsets[slot];
        std.debug.assert(total_at == base_len + ids.len);
        const full_st = try q.snapshot.blank(self.gpa, r, null, total_at);
        defer q.snapshot.drop(self.gpa, full_st);
        try q.snapshot.copyShared(self.gpa, r, slot, full_st, true);
        readSliceFromSharedState(r, full_st, base_len, ids.len, out);
    }

    fn buildSnapFn(
        ctx: *anyopaque,
        composed: *const mem.RawState,
        suffix_ids: []const u32,
    ) anyerror!mem.SnapHandle {
        const self = of(ctx);
        const b = self.back;
        const r = b.runner;

        const slot = try self.pickIdleSlot();
        defer self.releaseIdleSlot(slot);

        {
            const st = try q.snapshot.blank(self.gpa, r, null, composed.len);
            defer q.snapshot.drop(self.gpa, st);
            writeRawIntoSharedState(r, st, composed);
            try q.snapshot.restoreSlot(self.gpa, r, slot, null, st);
        }

        if (suffix_ids.len > 0) {
            const sess = q.session.Session{ .runner = r, .slot = slot };
            try sess.prefill(suffix_ids, @min(b.chunk, r.model.frame.capacity));
        }

        const at = r.offsets[slot];
        var ring = b.ringOf(slot);
        if (ring) |*g| {
            try g.hold(at);
            @memset(g.buf.contents()[0..g.bytes(at)], 0);
            g.end.* = at;
        }
        const snap = try q.snapshot.saveSlot(self.gpa, r, slot, ring, at, null);
        errdefer q.snapshot.drop(self.gpa, snap);
        try snap.landed();
        r.settleSlot(slot);
        return .{ .saved = @ptrCast(snap), .at = snap.at };
    }

    fn dropSnapFn(ctx: *anyopaque, ptr: *anyopaque) void {
        const self = of(ctx);
        q.snapshot.drop(self.gpa, @ptrCast(@alignCast(ptr)));
    }
};
