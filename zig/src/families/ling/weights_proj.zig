//! Ling's linears and routed experts as stored (bf16, FP8 blocks, ModelOpt NVFP4) into qlinear and experts views.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const config = @import("config.zig");

const Quant = config.Quant;
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

    /// A device buffer of `len` bytes the store owns, named `name`.
    pub fn alloc(s: *Store, len: usize, name: []const u8) !u64 {
        const b = try cuda.DeviceBuffer.alloc(s.d, @max(len, 16));
        errdefer {
            var x = b;
            x.free();
        }
        try s.buffers.append(s.gpa, b);
        try s.named.append(s.gpa, .{ .name = try s.gpa.dupe(u8, name), .ptr = b.ptr, .len = len });
        return b.ptr;
    }

    /// `bytes` uploaded into a buffer the store owns.
    pub fn keep(s: *Store, bytes: []const u8, name: []const u8) !u64 {
        const p = try s.alloc(bytes.len, name);
        try s.buffers.items[s.buffers.items.len - 1].upload(0, bytes);
        return p;
    }

    pub fn find(s: *const Store, name: []const u8) ?Named {
        for (s.named.items) |n| if (std.mem.eql(u8, n.name, name)) return n;
        return null;
    }
};

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
        };
    }

    /// A temporary device copy of `bytes` for a packer, freed by the caller after the stream drains.
    fn staged(r: *Reader, bytes: []const u8) !cuda.DeviceBuffer {
        return cuda.DeviceBuffer.fromHost(r.store.d, bytes);
    }

    fn bf16(r: *Reader, module: []const u8, n: usize, k: usize) !Proj {
        const t = try r.ck.get(try r.name(module, "weight"));
        if (!t.is(.bf16, &.{ n, k })) return wrong(module, .bf16, t, n, k);
        var src = try r.staged(t.bytes);
        defer src.free();
        const out = try r.store.alloc(cuda.modelopt.padded(n) * k * 2, module);
        try r.pack.bf16Lane(r.stream, src.ptr, out, n, k);
        try r.stream.synchronize();
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
        defer src.free();
        const codes = try r.store.alloc(cuda.fp8.padded(n) * k, module);
        try r.pack.fp8Lane(r.stream, src.ptr, codes, n, k);
        const inv = try r.gpa.alloc(f32, p.s.numel());
        defer r.gpa.free(inv);
        @memcpy(std.mem.sliceAsBytes(inv), p.s.bytes);
        const tiles = try r.gpa.alloc(f32, cuda.fp8.padded(n) * (k / cuda.fp8.group));
        defer r.gpa.free(tiles);
        cuda.fp8.packScales(tiles, inv, n, k);
        const scales = try r.store.keep(std.mem.sliceAsBytes(tiles), try r.name(module, "weight_scale_inv"));
        try r.stream.synchronize();
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
        defer codes.free();
        var scales = try r.staged(l.scales);
        defer scales.free();
        const np = cuda.nvfp4.padded(n);
        const words = try r.store.alloc(np / 64 * (k / 64) * 512 * 4, module);
        const tiles = try r.store.alloc(np * (k / 16), try r.name(module, "weight_scale"));
        try r.pack.fp4Lane(r.stream, codes.ptr, scales.ptr, words, tiles, n, k);
        try r.stream.synchronize();
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
        if (t.numel() != numel or (t.dtype != .bf16 and t.dtype != .f32)) {
            std.log.err("{s}: {t} {any}, expected {d} bf16 or fp32 values", .{ tensor, t.dtype, t.shape[0..t.rank], numel });
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

    /// Layer i's routed experts 0..count [width, dims]: every projection of every expert in one format, or refused.
    pub fn experts(r: *Reader, i: u32, count: usize, width: usize, dims: usize, limit: f32) !Routed {
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
        }
    }

    fn expertModule(b: []u8, i: u32, e: usize, comptime proj: []const u8) ![]const u8 {
        return std.fmt.bufPrint(b, "model.layers.{d}.mlp.experts.{d}." ++ proj, .{ i, e });
    }

    /// A view of `ptr` the store owns, for uploads at an offset.
    fn view(r: *Reader, ptr: u64, len: usize) cuda.DeviceBuffer {
        return .{ .d = r.store.d, .ptr = ptr, .len = len };
    }

    fn label(b: []u8, i: u32, what: []const u8) ![]const u8 {
        return std.fmt.bufPrint(b, "model.layers.{d}.mlp.experts.{s}", .{ i, what });
    }

    /// experts.Layer's fp8g form: Fp8Experts.packGateUp and packOne per expert, scale_inv blocks as stored.
    fn fp8Experts(r: *Reader, i: u32, count: usize, ni: usize, d: usize, limit: f32) !cuda.experts.Layer {
        if (ni % X8.block != 0 or d % X8.block != 0) return error.UnexpectedTensor;
        const up_n = 2 * ni * d / 4; // u32 words of one expert's gate and up
        const down_n = d * ni / 4;
        const inv_n = (ni / 128) * (d / 128); // one matrix's scale_inv blocks
        var lb: [96]u8 = undefined;
        const up = try r.store.alloc(count * up_n * 4, try label(&lb, i, "up"));
        const down = try r.store.alloc(count * down_n * 4, try label(&lb, i, "down"));
        const up_scale = try r.store.alloc(count * 2 * inv_n * 4, try label(&lb, i, "up_scale"));
        const down_scale = try r.store.alloc(count * inv_n * 4, try label(&lb, i, "down_scale"));
        const hu = try r.gpa.alloc(u32, up_n);
        defer r.gpa.free(hu);
        const hd = try r.gpa.alloc(u32, down_n);
        defer r.gpa.free(hd);
        var b: [192]u8 = undefined;
        for (0..count) |e| {
            const g = try r.fp8Pair(try expertModule(&b, i, e, "gate_proj"), ni, d);
            const u = try r.fp8Pair(try expertModule(&b, i, e, "up_proj"), ni, d);
            const dn = try r.fp8Pair(try expertModule(&b, i, e, "down_proj"), d, ni);
            try X8.packGateUp(r.gpa, hu, g.w.bytes, u.w.bytes, ni, d);
            X8.packOne(hd, dn.w.bytes, d, ni);
            try r.view(up, count * up_n * 4).upload(e * up_n * 4, std.mem.sliceAsBytes(hu));
            try r.view(down, count * down_n * 4).upload(e * down_n * 4, std.mem.sliceAsBytes(hd));
            const us = r.view(up_scale, count * 2 * inv_n * 4);
            try us.upload(2 * e * inv_n * 4, g.s.bytes);
            try us.upload((2 * e + 1) * inv_n * 4, u.s.bytes);
            try r.view(down_scale, count * inv_n * 4).upload(e * inv_n * 4, dn.s.bytes);
        }
        return .{ .format = .fp8g, .up = up, .down = down, .up_scale = up_scale, .down_scale = down_scale, .width = ni, .dims = d, .experts = count, .limit = limit, .after = true };
    }

    /// experts.Layer's nvfp4 form: Nvfp4Experts.packGateUp and packOne per expert, weight_scale_2 as the globals.
    fn fp4Experts(r: *Reader, i: u32, count: usize, ni: usize, d: usize, limit: f32) !cuda.experts.Layer {
        if (ni % X4.cols != 0 or d % 32 != 0) return error.UnexpectedTensor;
        const up_n = X4.bytes(1, ni, d, 2) / 4;
        const down_n = X4.bytes(1, d, ni, 1) / 4;
        var lb: [96]u8 = undefined;
        const up = try r.store.alloc(count * up_n * 4, try label(&lb, i, "up"));
        const down = try r.store.alloc(count * down_n * 4, try label(&lb, i, "down"));
        const hu = try r.gpa.alloc(u32, up_n);
        defer r.gpa.free(hu);
        const hd = try r.gpa.alloc(u32, down_n);
        defer r.gpa.free(hd);
        const gu = try r.gpa.alloc(f32, 2 * count);
        defer r.gpa.free(gu);
        const gd = try r.gpa.alloc(f32, count);
        defer r.gpa.free(gd);
        var b: [192]u8 = undefined;
        for (0..count) |e| {
            const g = try r.fp4Read(try expertModule(&b, i, e, "gate_proj"), ni, d);
            const u = try r.fp4Read(try expertModule(&b, i, e, "up_proj"), ni, d);
            const dn = try r.fp4Read(try expertModule(&b, i, e, "down_proj"), d, ni);
            X4.packGateUp(hu, .{ g.codes, g.scales }, .{ u.codes, u.scales }, ni, d);
            X4.packOne(hd, dn.codes, dn.scales, d, ni);
            try r.view(up, count * up_n * 4).upload(e * up_n * 4, std.mem.sliceAsBytes(hu));
            try r.view(down, count * down_n * 4).upload(e * down_n * 4, std.mem.sliceAsBytes(hd));
            gu[2 * e] = g.global;
            gu[2 * e + 1] = u.global;
            gd[e] = dn.global;
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
