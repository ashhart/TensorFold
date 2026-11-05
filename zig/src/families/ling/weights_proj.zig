//! Ling's linears and routed experts as stored (bf16, FP8 blocks, ModelOpt NVFP4) into qlinear and experts views.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const config = @import("config.zig");

const Quant = config.Quant;
const cuts = @import("weights_cut.zig");
const Span = cuts.Span;
const Tensor = core.checkpoint.Tensor;
const X8 = cuda.experts.Fp8Experts;
const X4 = cuda.experts.Nvfp4Experts;

/// One projection in the format its checkpoint stores: a lane matmul view (bf16, fp8g, nvfp4).
pub const Proj = cuda.qlinear.Weight;

/// Gate, up and down: the dense MLP, the shared expert, and one bf16 routed expert.
pub const Mlp = struct { gate: Proj, up: Proj, down: Proj };

/// Routed experts: the grouped kernels' layout (fp8g, nvfp4), or bf16 lane weights (no grouped bf16 kernel).
pub const Routed = union(enum) { grouped: cuda.experts.Layer, lanes: []Mlp };

/// One upload, by the checkpoint name it came from (tests find small tensors by it).
pub const Named = struct { name: []u8, ptr: u64, len: usize };

/// Every device buffer a load made, freed together.
pub const Store = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    buffers: std.ArrayList(cuda.DeviceBuffer) = .empty,
    named: std.ArrayList(Named) = .empty,
    lanes: std.ArrayList([]Mlp) = .empty,

    pub fn deinit(s: *Store) void {
        for (s.buffers.items) |*b| b.free();
        for (s.named.items) |n| s.gpa.free(n.name);
        for (s.lanes.items) |l| s.gpa.free(l);
        s.buffers.deinit(s.gpa);
        s.named.deinit(s.gpa);
        s.lanes.deinit(s.gpa);
    }

    /// A device buffer of `len` bytes the store owns, named `name`; on failure the store holds nothing new.
    pub fn alloc(s: *Store, len: usize, name: []const u8) !u64 {
        try s.buffers.ensureUnusedCapacity(s.gpa, 1);
        try s.named.ensureUnusedCapacity(s.gpa, 1);
        const owned = try s.gpa.dupe(u8, name);
        errdefer s.gpa.free(owned);
        const b = try cuda.DeviceBuffer.alloc(s.d, @max(len, 16));
        s.buffers.appendAssumeCapacity(b);
        s.named.appendAssumeCapacity(.{ .name = owned, .ptr = b.ptr, .len = len });
        return b.ptr;
    }

    /// `bytes` uploaded into a buffer the store owns.
    pub fn keep(s: *Store, bytes: []const u8, name: []const u8) !u64 {
        const p = try s.alloc(bytes.len, name);
        try s.buffers.items[s.buffers.items.len - 1].upload(0, bytes);
        return p;
    }

    /// `parts` back to back in one buffer the store owns (an EXL3 layer's expert trellises: one allocation, not 512).
    pub fn keepAll(s: *Store, parts: []const []const u8, name: []const u8) !u64 {
        var total: usize = 0;
        for (parts) |p| total += p.len;
        const base = try s.alloc(total, name);
        const buf = &s.buffers.items[s.buffers.items.len - 1];
        var at: usize = 0;
        for (parts) |p| {
            try buf.upload(at, p);
            at += p.len;
        }
        return base;
    }

    /// deinit, or when the device may still write them (a failed drain) the host lists alone: buffers leaked.
    pub fn drop(s: *Store, settled: bool) void {
        if (!settled) s.buffers.clearRetainingCapacity();
        s.deinit();
    }

    pub fn find(s: *const Store, name: []const u8) ?Named {
        for (s.named.items) |n| if (std.mem.eql(u8, n.name, name)) return n;
        return null;
    }
};

/// Whether `s` drained: false when its synchronize fails (work queued on it may still touch memory).
pub fn drains(s: cuda.Stream) bool {
    s.synchronize() catch return false;
    return true;
}

/// A device copy of `bytes`; a failed copy frees it after a context drain, else leaks it (fromHost exits instead).
pub fn uploaded(d: *const cuda.Driver, bytes: []const u8) !cuda.DeviceBuffer {
    var b = try cuda.DeviceBuffer.alloc(d, bytes.len);
    b.upload(0, bytes) catch |e| {
        if (d.synchronize()) |_| b.free() else |_| {}
        return e;
    };
    return b;
}

/// Host bf16 from fp32: round to nearest even, NaN kept quiet.
pub fn toBf16(v: f32) u16 {
    const b: u32 = @bitCast(v);
    if (std.math.isNan(v)) return @intCast((b >> 16) | 0x40);
    return @intCast((b + 0x7fff + ((b >> 16) & 1)) >> 16);
}

/// Element i of a bf16 or fp32 tensor.
pub fn value(t: Tensor, i: usize) !f32 {
    return switch (t.dtype) {
        .bf16 => @bitCast(@as(u32, std.mem.readInt(u16, t.bytes[2 * i ..][0..2], .little)) << 16),
        .f16 => @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, t.bytes[2 * i ..][0..2], .little)))), // EXL3 packs' plain tensors
        .f32 => @bitCast(std.mem.readInt(u32, t.bytes[4 * i ..][0..4], .little)),
        else => error.UnexpectedTensor,
    };
}

/// What reading a layer needs: the checkpoint, the config, the GPU packers, the store the uploads land in.
pub const Reader = struct {
    gpa: std.mem.Allocator,
    stream: cuda.Stream, // blocking: the packers wait for the uploads
    pack: *const cuda.modelopt.Packer,
    ck: *core.Checkpoint,
    c: *const config.Config,
    store: *Store,
    stage: usize = 512 << 20, // raw expert bytes staged at a time (weights.Options.stage)
    queued: bool = false, // an upload or packer launch not yet drained
    buf: [256]u8 = undefined,

    fn name(r: *Reader, module: []const u8, tail: []const u8) ![]const u8 {
        return std.fmt.bufPrint(&r.buf, "{s}.{s}", .{ module, tail });
    }

    /// The config gives `module` format `f`; the file disagrees (dtype or shape): refused, both named.
    fn wrong(module: []const u8, f: Quant, t: Tensor, n: usize, k: usize) error{UnexpectedTensor} {
        std.log.err("{s}: the config makes it {t} [{d}, {d}], the file holds {t} {any}", .{ module, f, n, k, t.dtype, t.shape[0..t.rank] });
        return error.UnexpectedTensor;
    }

    /// Linear `module` [n, k] in the format its config gives it; a file tensor of another format or shape is refused.
    pub fn linear(r: *Reader, module: []const u8, n: usize, k: usize) !Proj {
        if (k % cuda.qmmf.group != 0) return error.UnexpectedTensor;
        return switch (r.c.format(module)) {
            .bf16 => r.bf16(module, n, k),
            .fp8_block128 => r.fp8(module, n, k),
            .nvfp4 => r.fp4(module, n, k),
            // an EXL3 pack keeps what exllamav3 left unquantized (b_proj, kv_b_proj, the head gate) as bf16 weights
            .exl3 => if (r.ck.has(try r.name(module, "trellis"))) .{ .exl3 = try r.exl3Linear(module, n, k) } else r.bf16(module, n, k),
        };
    }

    /// One EXL3 group as main's exl3_load admits it: trellis int16 [k/16][n/16][16 bits], fp16 suh [k] and svh [n].
    pub const Group = struct { trellis: Tensor, suh: Tensor, svh: Tensor, halves: u32, codebook: cuda.exl3.Codebook, n: usize };

    pub fn exl3Group(r: *Reader, module: []const u8, n: usize, k: usize) !Group {
        const t = try r.ck.get(try r.name(module, "trellis"));
        const halves = if (t.rank == 3) t.shape[2] / 8 else 0;
        // exllamav3 pads a width off the 128 grid up to it (kv_a_proj_with_mqa's 576 is stored as 640)
        const np = if (n % 128 == 0) n else std.mem.alignForward(usize, n, 128);
        if (t.dtype != .i16 or t.rank != 3 or t.shape[0] != k / 16 or t.shape[1] != np / 16 or t.shape[2] % 8 != 0 or halves < 2 or halves > 16 or k % 128 != 0) return wrong(module, .exl3, t, n, k);
        var codebook: cuda.exl3.Codebook = .inst3;
        for (core.exl3_load.markers) |m| if (r.ck.has(try r.name(module, m.name))) {
            const v = try r.ck.get(try r.name(module, m.name));
            if (v.bytes.len != 4 or std.mem.readInt(u32, v.bytes[0..4], .little) != m.value) return wrong(module, .exl3, v, n, k);
            codebook = switch (m.codebook) {
                .inst3 => .inst3,
                .mcg => .mcg,
                .mul1 => .mul1,
            };
        };
        if (halves % 2 == 1 and codebook != .mul1) return wrong(module, .exl3, t, n, k); // half steps are mul1's alone
        const suh = try r.ck.get(try r.name(module, "suh"));
        const svh = try r.ck.get(try r.name(module, "svh"));
        if (!suh.is(.f16, &.{k}) or !svh.is(.f16, &.{np})) return wrong(module, .exl3, suh, n, k);
        return .{ .trellis = t, .suh = suh, .svh = svh, .halves = @intCast(halves), .codebook = codebook, .n = np };
    }

    /// An n-by-k EXL3 projection, its trellis as stored (exl3.Dense reads the stored layout), on qlinear's exl3 arm.
    fn exl3Linear(r: *Reader, module: []const u8, n: usize, k: usize) !cuda.exl3.Weight {
        const g = try r.exl3Group(module, n, k);
        const st = cuda.exl3.strides(k, g.n, g.halves, false);
        return .{
            .trellis = try r.store.keep(g.trellis.bytes, module),
            .suh = try r.store.keep(g.suh.bytes, try r.name(module, "suh")),
            .svh = try r.store.keep(g.svh.bytes, try r.name(module, "svh")),
            .k = @intCast(k),
            .n = @intCast(g.n), // the stored width: a padded one's extra columns are read past
            .halves = g.halves,
            .codebook = g.codebook,
            .stride_k = st.k,
            .stride_nb = st.nb,
        };
    }

    /// A temporary device copy of `bytes` for a packer, freed by the caller once settled().
    fn staged(r: *Reader, bytes: []const u8) !cuda.DeviceBuffer {
        return uploaded(r.store.d, bytes);
    }

    /// Nothing queued can still read or write memory: at once, or after a drain; false when the drain fails.
    pub fn settled(r: *Reader) bool {
        if (r.queued and !drains(r.stream)) return false;
        r.queued = false;
        return true;
    }

    fn drain(r: *Reader) !void {
        try r.stream.synchronize();
        r.queued = false;
    }

    fn bf16(r: *Reader, module: []const u8, n: usize, k: usize) !Proj {
        const t = try r.ck.get(try r.name(module, "weight"));
        // an EXL3 pack stores what it left unquantized as fp16: rounded to bf16 here, the lanes' plain format
        if (r.c.quant == .exl3 and t.is(.f16, &.{ n, k })) {
            var host: std.ArrayList(u8) = .empty;
            defer host.deinit(r.gpa);
            try append(r.gpa, &host, t, .bf16);
            var u = t;
            u.dtype = .bf16;
            u.bytes = host.items;
            return r.bf16Of(module, u, n, k);
        }
        if (!t.is(.bf16, &.{ n, k })) return wrong(module, .bf16, t, n, k);
        return r.bf16Of(module, t, n, k);
    }

    pub fn bf16Of(r: *Reader, module: []const u8, t: Tensor, n: usize, k: usize) !Proj {
        var src = try r.staged(t.bytes);
        defer if (r.settled()) src.free(); // else leaked: a packer may still read it
        const out = try r.store.alloc(cuda.modelopt.padded(n) * k * 2, module);
        r.queued = true;
        try r.pack.bf16Lane(r.stream, src.ptr, out, n, k);
        try r.drain();
        return .{ .bf16 = cuda.modelopt.bf16Weight(out, n, k) };
    }

    /// e4m3 [n, k] and fp32 scale_inv [n/128 up, k/128] checked against the config's shape.
    fn fp8Pair(r: *Reader, module: []const u8, n: usize, k: usize) !struct { w: Tensor, s: Tensor } {
        const w = try r.ck.get(try r.name(module, "weight"));
        if (!w.is(.f8_e4m3, &.{ n, k }) or k % cuda.fp8.block != 0) return wrong(module, .fp8_block128, w, n, k);
        const s = try r.ck.get(try r.name(module, "weight_scale_inv"));
        if (!s.is(.f32, &.{ (n + 127) / 128, k / 128 })) return wrong(module, .fp8_block128, s, n, k);
        return .{ .w = w, .s = s };
    }

    fn fp8(r: *Reader, module: []const u8, n: usize, k: usize) !Proj {
        const p = try r.fp8Pair(module, n, k);
        var src = try r.staged(p.w.bytes);
        defer if (r.settled()) src.free(); // else leaked: a packer may still read it
        const codes = try r.store.alloc(cuda.fp8.padded(n) * k, module);
        r.queued = true;
        try r.pack.fp8Lane(r.stream, src.ptr, codes, n, k);
        const inv = try r.gpa.alloc(f32, p.s.numel());
        defer r.gpa.free(inv);
        @memcpy(std.mem.sliceAsBytes(inv), p.s.bytes);
        const tiles = try r.gpa.alloc(f32, cuda.fp8.padded(n) * (k / cuda.fp8.group));
        defer r.gpa.free(tiles);
        cuda.fp8.packScales(tiles, inv, n, k);
        const scales = try r.store.keep(std.mem.sliceAsBytes(tiles), try r.name(module, "weight_scale_inv"));
        try r.drain();
        return .{ .fp8g = cuda.fp8.weight(codes, scales, n, k) };
    }

    /// ModelOpt NVFP4 [n, k]: a weight stored without its weight_scale is refused before core.nvfp4 reads it.
    fn fp4Read(r: *Reader, module: []const u8, n: usize, k: usize) !core.nvfp4.Linear {
        if (!r.ck.has(try r.name(module, "weight_scale"))) return wrong(module, .nvfp4, try r.ck.get(try r.name(module, "weight")), n, k);
        const l = try core.nvfp4.Linear.read(r.ck, module);
        if (l.n != n or l.k != k) return wrong(module, .nvfp4, try r.ck.get(try r.name(module, "weight")), n, k);
        return l;
    }

    fn fp4(r: *Reader, module: []const u8, n: usize, k: usize) !Proj {
        const l = try r.fp4Read(module, n, k);
        var codes = try r.staged(l.codes);
        defer if (r.settled()) codes.free(); // else both leaked
        var scales = try r.staged(l.scales);
        defer if (r.settled()) scales.free();
        const np = cuda.nvfp4.padded(n);
        const words = try r.store.alloc(np / 64 * (k / 64) * 512 * 4, module);
        const tiles = try r.store.alloc(np * (k / 16), try r.name(module, "weight_scale"));
        r.queued = true;
        try r.pack.fp4Lane(r.stream, codes.ptr, scales.ptr, words, tiles, n, k);
        try r.drain();
        return .{ .nvfp4 = cuda.nvfp4.weight(words, tiles, l.global, n, k) };
    }

    /// Gate, up [width, dims] and down [dims, width] under `prefix` (`prefix`gate_proj and so on).
    pub fn mlp(r: *Reader, prefix: []const u8, width: usize, dims: usize) !Mlp {
        var m: [3]Proj = undefined;
        inline for (.{ "gate_proj", "up_proj", "down_proj" }, 0..) |p, j| {
            var b: [192]u8 = undefined;
            const module = try std.fmt.bufPrint(&b, "{s}{s}", .{ prefix, p });
            m[j] = if (j < 2) try r.linear(module, width, dims) else try r.linear(module, dims, width);
        }
        return .{ .gate = m[0], .up = m[1], .down = m[2] };
    }

    /// Small tensor `tensor` (bf16 or fp32, `numel` long) as `to`, uploaded under its own name.
    pub fn small(r: *Reader, tensor: []const u8, numel: usize, to: core.checkpoint.DType) !u64 {
        const t = try r.ck.get(tensor);
        if (t.numel() != numel or (t.dtype != .bf16 and t.dtype != .f32 and !(t.dtype == .f16 and r.c.quant == .exl3))) {
            std.log.err("{s}: {t} {any}, expected {d} bf16 or fp32 values (fp16 in an EXL3 pack)", .{ tensor, t.dtype, t.shape[0..t.rank], numel });
            return error.UnexpectedTensor;
        }
        var host: std.ArrayList(u8) = .empty;
        defer host.deinit(r.gpa);
        try append(r.gpa, &host, t, to);
        return r.store.keep(host.items, tensor);
    }

    /// The three short-conv weights [n, 1, taps] as one fp32 [3n][taps]: #804 KDA's q|k|v channels.
    pub fn convs(r: *Reader, prefix: []const u8, n: usize, out_name: []const u8) !u64 {
        var host: std.ArrayList(u8) = .empty;
        defer host.deinit(r.gpa);
        inline for (.{ "q", "k", "v" }) |x| {
            var b: [192]u8 = undefined;
            const t = try r.ck.get(try std.fmt.bufPrint(&b, "{s}" ++ x ++ "_conv1d.weight", .{prefix}));
            if (!t.is(t.dtype, &.{ n, 1, r.c.conv }) or (t.dtype != .bf16 and t.dtype != .f32)) {
                std.log.err("{s}" ++ x ++ "_conv1d: {t} {any}, expected [{d}, 1, {d}]", .{ prefix, t.dtype, t.shape[0..t.rank], n, r.c.conv });
                return error.UnexpectedTensor;
            }
            try append(r.gpa, &host, t, .f32);
        }
        return r.store.keep(host.items, out_name);
    }

    /// Layer i's routed experts 0..count [width, dims], one format; `cut`: a TP rank's span of the width (EXL3 only).
    pub fn experts(r: *Reader, i: u32, count: usize, width: usize, dims: usize, limit: f32, cut: ?Span) !Routed {
        const names = @import("names.zig");
        var b: [192]u8 = undefined;
        const f = r.c.format(try names.moduleName(&b, i, .expert_gate, 0));
        for (0..count) |e| for ([_]names.Role{ .expert_gate, .expert_up, .expert_down }) |role| {
            const g = r.c.format(try names.moduleName(&b, i, role, @intCast(e)));
            if (g != f) {
                std.log.err("{s} is {t}, expert 0's gate {t}: the grouped kernels take one format a layer", .{ try names.moduleName(&b, i, role, @intCast(e)), g, f });
                return error.MixedExperts;
            }
        };
        var prefix: [96]u8 = undefined;
        if (cut != null and f != .exl3) return error.TpFormat;
        switch (f) {
            .bf16 => {
                const lanes = try r.gpa.alloc(Mlp, count);
                errdefer r.gpa.free(lanes);
                for (lanes, 0..) |*m, e| m.* = try r.mlp(try std.fmt.bufPrint(&prefix, "model.layers.{d}.mlp.experts.{d}.", .{ i, e }), width, dims);
                try r.store.lanes.append(r.gpa, lanes);
                return .{ .lanes = lanes };
            },
            .fp8_block128 => return .{ .grouped = try r.fp8Experts(i, count, width, dims, limit) },
            .nvfp4 => return .{ .grouped = try r.fp4Experts(i, count, width, dims, limit) },
            .exl3 => return .{ .grouped = try r.exl3Experts(i, count, width, dims, limit, cut) },
        }
    }

    /// experts.Layer's exl3 form (main's exl3_grouped): trellises as stored, pointers and half-bit widths per expert.
    fn exl3Experts(r: *Reader, i: u32, count: usize, ni: usize, d: usize, limit: f32, cut: ?Span) !cuda.experts.Layer {
        const w = if (cut) |s| s.len() else ni; // a rank's width: gate and up by tile columns, down by tile rows
        var arena = std.heap.ArenaAllocator.init(r.gpa);
        defer arena.deinit();
        var tables: [3]struct { ptrs: u64, k2: u64, suh: u64, svh: u64 } = undefined;
        var codebook: ?cuda.exl3.Codebook = null;
        var b: [192]u8 = undefined;
        var lb: [96]u8 = undefined;
        inline for (.{ "gate_proj", "up_proj", "down_proj" }, 0..) |proj, m| {
            const n = if (m < 2) ni else d;
            const k = if (m < 2) d else ni;
            const ptrs = try r.gpa.alloc(u64, count);
            defer r.gpa.free(ptrs);
            const k2 = try r.gpa.alloc(i32, count);
            defer r.gpa.free(k2);
            var su: std.ArrayList(u8) = .empty;
            defer su.deinit(r.gpa);
            var sv: std.ArrayList(u8) = .empty;
            defer sv.deinit(r.gpa);
            const parts = try r.gpa.alloc([]const u8, count);
            defer r.gpa.free(parts);
            var at: usize = 0;
            for (0..count) |e| {
                const g = try r.exl3Group(try expertModule(&b, i, e, proj), n, k);
                if (codebook) |c| (if (c != g.codebook) return error.MixedExperts) else codebook = g.codebook;
                parts[e] = if (cut) |s| try cuts.expertTrellis(arena.allocator(), g, k, m < 2, s) else g.trellis.bytes;
                ptrs[e] = at; // offsets until the one upload below
                at += parts[e].len;
                k2[e] = @intCast(g.halves);
                try su.appendSlice(r.gpa, if (cut != null and m == 2) g.suh.bytes[cut.?.lo * 2 .. cut.?.hi * 2] else g.suh.bytes);
                try sv.appendSlice(r.gpa, if (cut != null and m < 2) g.svh.bytes[cut.?.lo * 2 .. cut.?.hi * 2] else g.svh.bytes);
            }
            const base = try r.store.keepAll(parts, try label(&lb, i, proj ++ ".trellis"));
            for (ptrs) |*p| p.* += base;
            tables[m] = .{
                .ptrs = try r.store.keep(std.mem.sliceAsBytes(ptrs), try label(&lb, i, proj ++ ".trellis_ptrs")),
                .k2 = try r.store.keep(std.mem.sliceAsBytes(k2), try label(&lb, i, proj ++ ".k2")),
                .suh = try r.store.keep(su.items, try label(&lb, i, proj ++ ".suh")),
                .svh = try r.store.keep(sv.items, try label(&lb, i, proj ++ ".svh")),
            };
            r.ck.release(); // each projection's pages dropped: the GPU shares this memory
        }
        const lay: cuda.exl3_experts.Layer = .{
            .gate = tables[0].ptrs,
            .up = tables[1].ptrs,
            .down = tables[2].ptrs,
            .gate_k2 = tables[0].k2,
            .up_k2 = tables[1].k2,
            .down_k2 = tables[2].k2,
            .suh_g = tables[0].suh,
            .suh_u = tables[1].suh,
            .svh_g = tables[0].svh,
            .svh_u = tables[1].svh,
            .suh_d = tables[2].suh,
            .svh_d = tables[2].svh,
            .experts = count,
            .dims = d,
            .width = w,
            .codebook = codebook orelse return error.UnexpectedTensor,
            .limit = limit,
            .act_mode = 1, // fp32 silu(g) * u, nearest exllamav3's fp16 intermediate
        };
        // after: Ling's order, min(silu(g), L) * clip(u, -L, L), on the shared finish's twin (plan_finish_after)
        return .{ .format = .exl3, .exl3 = lay, .up = 0, .down = 0, .width = w, .dims = d, .experts = count, .limit = limit, .after = true };
    }

    fn expertModule(b: []u8, i: u32, e: usize, comptime proj: []const u8) ![]const u8 {
        return std.fmt.bufPrint(b, "model.layers.{d}.mlp.experts.{d}." ++ proj, .{ i, e });
    }

    fn label(b: []u8, i: u32, what: []const u8) ![]const u8 {
        return std.fmt.bufPrint(b, "model.layers.{d}.mlp.experts.{s}", .{ i, what });
    }

    /// Experts a stage holds when one expert's raw tensors take `one` bytes.
    fn batch(r: *const Reader, count: usize, one: usize) usize {
        return @max(1, @min(count, r.stage / one));
    }

    /// A batch's raw tensors: copied from the file into pinned memory by `threads` threads, uploaded, packed there.
    const Stage = struct {
        host: cuda.HostBuffer,
        dev: cuda.DeviceBuffer,
        jobs: std.ArrayList(Job) = .empty,

        const Job = struct { at: usize, bytes: []const u8 };
        const threads = 8; // page faults in flight: one reader leaves the disk (or the mount) idle between them

        fn init(r: *Reader, len: usize) !Stage {
            var host = try cuda.HostBuffer.alloc(r.store.d, len);
            errdefer host.free();
            return .{ .host = host, .dev = try cuda.DeviceBuffer.alloc(r.store.d, len) };
        }

        fn deinit(st: *Stage, r: *Reader) void {
            st.jobs.deinit(r.gpa);
            if (!r.settled()) return; // both leaked on purpose: the upload or a packer may still read them
            st.host.free();
            st.dev.free();
        }

        fn put(st: *Stage, r: *Reader, at: usize, bytes: []const u8) !void {
            try st.jobs.append(r.gpa, .{ .at = at, .bytes = bytes });
        }

        fn copy(st: *const Stage, first: usize) void {
            var j = first;
            while (j < st.jobs.items.len) : (j += threads) {
                const job = st.jobs.items[j];
                @memcpy(st.host.bytes[job.at..][0..job.bytes.len], job.bytes);
            }
        }

        /// The batch on the device; the packers run after it on `r.stream`.
        fn upload(st: *Stage, r: *Reader) !void {
            var pool: [threads - 1]?std.Thread = @splat(null);
            for (&pool, 1..) |*t, first| t.* = std.Thread.spawn(.{}, copy, .{ st, first }) catch null;
            for (pool, 1..) |t, first| if (t == null) st.copy(first); // no thread: this one copies its share
            st.copy(0);
            for (pool) |t| if (t) |x| x.join();
            st.jobs.clearRetainingCapacity();
            r.queued = true;
            try st.dev.uploadAsync(0, st.host.bytes, r.stream.handle);
        }

        /// The packers done (the stage is reused) and the batch's file pages dropped: the GPU shares this memory.
        fn drained(_: *Stage, r: *Reader) !void {
            try r.drain();
            r.ck.release();
        }
    };

    /// experts.Layer's fp8g form: batches of e4m3 bytes staged on the device, packed there (Packer.fp8Experts).
    fn fp8Experts(r: *Reader, i: u32, count: usize, ni: usize, d: usize, limit: f32) !cuda.experts.Layer {
        if (ni % X8.block != 0 or d % X8.block != 0) return error.UnexpectedTensor;
        const one = ni * d; // bytes of one e4m3 matrix
        const inv_n = (ni / 128) * (d / 128); // one matrix's scale_inv blocks
        var lb: [96]u8 = undefined;
        const up = try r.store.alloc(count * 2 * one, try label(&lb, i, "up"));
        const down = try r.store.alloc(count * one, try label(&lb, i, "down"));
        const inv = try r.gpa.alloc(f32, 3 * count * inv_n); // [E][gate, up] then [E][down], as stored
        defer r.gpa.free(inv);
        const nb = r.batch(count, 3 * one);
        var stage = try Stage.init(r, 3 * nb * one); // [gate, up, down][nb][n][k]
        defer stage.deinit(r);
        var b: [192]u8 = undefined;
        var e0: usize = 0;
        while (e0 < count) : (e0 += nb) {
            const n = @min(nb, count - e0);
            for (0..n) |j| inline for (.{ "gate_proj", "up_proj", "down_proj" }, 0..) |proj, m| {
                const e = e0 + j;
                const x = if (m < 2) try r.fp8Pair(try expertModule(&b, i, e, proj), ni, d) else try r.fp8Pair(try expertModule(&b, i, e, proj), d, ni);
                try stage.put(r, (m * nb + j) * one, x.w.bytes);
                @memcpy(std.mem.sliceAsBytes(inv[(if (m < 2) 2 * e + m else 2 * count + e) * inv_n ..][0..inv_n]), x.s.bytes);
            };
            try stage.upload(r);
            for (0..2) |m| try r.pack.fp8Experts(r.stream, stage.dev.ptr + m * nb * one, up + e0 * 2 * one, n, ni, d, 2, m);
            try r.pack.fp8Experts(r.stream, stage.dev.ptr + 2 * nb * one, down + e0 * one, n, d, ni, 1, 0);
            try stage.drained(r);
        }
        const up_scale = try r.store.keep(std.mem.sliceAsBytes(inv[0 .. 2 * count * inv_n]), try label(&lb, i, "up_scale"));
        const down_scale = try r.store.keep(std.mem.sliceAsBytes(inv[2 * count * inv_n ..]), try label(&lb, i, "down_scale"));
        return .{ .format = .fp8g, .up = up, .down = down, .up_scale = up_scale, .down_scale = down_scale, .width = ni, .dims = d, .experts = count, .limit = limit, .after = true };
    }

    /// experts.Layer's nvfp4 form: batches of codes and scales staged on the device, packed there (Packer.fp4Experts).
    fn fp4Experts(r: *Reader, i: u32, count: usize, ni: usize, d: usize, limit: f32) !cuda.experts.Layer {
        if (ni % X4.cols != 0 or d % 32 != 0) return error.UnexpectedTensor;
        // main's NVFP4 epilogue clamps g before silu; Ling's order (min(silu(g), L)) has no NVFP4 instance yet
        if (limit != 0) return error.Nvfp4ClampOrder;
        const up_n = X4.bytes(1, ni, d, 2);
        const down_n = X4.bytes(1, d, ni, 1);
        const cb = ni * d / 2; // code bytes of one matrix
        const sb = ni * d / 16; // its e4m3 scale bytes
        var lb: [96]u8 = undefined;
        const up = try r.store.alloc(count * up_n, try label(&lb, i, "up"));
        const down = try r.store.alloc(count * down_n, try label(&lb, i, "down"));
        const gu = try r.gpa.alloc(f32, 2 * count);
        defer r.gpa.free(gu);
        const gd = try r.gpa.alloc(f32, count);
        defer r.gpa.free(gd);
        const nb = r.batch(count, 3 * (cb + sb));
        var stage = try Stage.init(r, 3 * nb * (cb + sb)); // codes [3][nb], then scales [3][nb]
        defer stage.deinit(r);
        const codes = stage.dev.ptr;
        const scales = stage.dev.ptr + 3 * nb * cb;
        var b: [192]u8 = undefined;
        var e0: usize = 0;
        while (e0 < count) : (e0 += nb) {
            const n = @min(nb, count - e0);
            for (0..n) |j| inline for (.{ "gate_proj", "up_proj", "down_proj" }, 0..) |proj, m| {
                const e = e0 + j;
                const x = if (m < 2) try r.fp4Read(try expertModule(&b, i, e, proj), ni, d) else try r.fp4Read(try expertModule(&b, i, e, proj), d, ni);
                try stage.put(r, (m * nb + j) * cb, x.codes);
                try stage.put(r, 3 * nb * cb + (m * nb + j) * sb, x.scales);
                if (m < 2) gu[2 * e + m] = x.global else gd[e] = x.global;
            };
            try stage.upload(r);
            for (0..2) |m| try r.pack.fp4Experts(r.stream, codes + m * nb * cb, scales + m * nb * sb, up + e0 * up_n, n, ni, d, 0, d, 2, m);
            try r.pack.fp4Experts(r.stream, codes + 2 * nb * cb, scales + 2 * nb * sb, down + e0 * down_n, n, d, ni, 0, ni, 1, 0);
            try stage.drained(r);
        }
        const up_scale = try r.store.keep(std.mem.sliceAsBytes(gu), try label(&lb, i, "up_scale"));
        const down_scale = try r.store.keep(std.mem.sliceAsBytes(gd), try label(&lb, i, "down_scale"));
        return .{ .format = .nvfp4, .up = up, .down = down, .up_scale = up_scale, .down_scale = down_scale, .width = ni, .dims = d, .experts = count, .limit = limit, .after = true };
    }
};

/// `t`'s elements appended to `host` as bf16 or fp32 (bf16 from fp32 rounds to nearest even).
pub fn append(gpa: std.mem.Allocator, host: *std.ArrayList(u8), t: Tensor, to: core.checkpoint.DType) !void {
    if (t.dtype == to) return host.appendSlice(gpa, t.bytes);
    for (0..t.numel()) |i| {
        const v = try value(t, i);
        switch (to) {
            .f32 => try host.appendSlice(gpa, std.mem.asBytes(&v)),
            .bf16 => try host.appendSlice(gpa, std.mem.asBytes(&toBf16(v))),
            else => return error.UnexpectedTensor,
        }
    }
}
