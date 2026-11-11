const std = @import("std");
const gpu = @import("tensor_metal");
const ops = gpu.ops;
const Store = @import("checkpoint.zig").Store;
pub const Tensor = gpu.Tensor;
pub const Feature = struct { tensor: Tensor, height: usize, width: usize, channels: usize };
const Size = gpu.mtl.Size;

pub const Runtime = struct {
    g: *gpu.Gpu,
    store: *Store,
    a: std.mem.Allocator,
    precision: enum { fp32, mixed } = .mixed,
    config: @import("config.zig").Config = .of(.sam31),
    vision_ready: bool = false,
    observer: ?@import("observe.zig").Observer = null,
    tensors: std.ArrayList(Tensor) = .empty,
    convolutions: std.StringHashMapUnmanaged(Tensor) = .empty,
    constants: std.StringHashMapUnmanaged(Tensor) = .empty,

    pub fn deinit(self: *Runtime) void {
        self.trim(0, &.{});
        self.tensors.deinit(self.g.allocator);
        var it = self.convolutions.iterator();
        while (it.next()) |e| {
            self.g.allocator.free(e.key_ptr.*);
            e.value_ptr.deinit();
        }
        self.convolutions.deinit(self.g.allocator);
        var constants = self.constants.iterator();
        while (constants.next()) |e| {
            self.g.allocator.free(e.key_ptr.*);
            e.value_ptr.deinit();
        }
        self.constants.deinit(self.g.allocator);
    }

    pub fn constant(self: *Runtime, key: []const u8) ?Tensor {
        var t = self.constants.get(key) orelse return null;
        t.owned = false;
        return t;
    }

    pub fn remember(self: *Runtime, key: []const u8, tensor: Tensor) !Tensor {
        std.debug.assert(self.g.batch == null);
        const name_ = try self.g.allocator.dupe(u8, key);
        errdefer self.g.allocator.free(name_);
        const limit = self.g.buffer_cache_limit;
        self.g.buffer_cache_limit = 0;
        defer self.g.buffer_cache_limit = limit;
        // A small persistent constant must not pin a much larger pooled scratch buffer.
        const stored = try self.g.alloc(tensor.count, tensor.dtype);
        errdefer stored.deinit();
        @memcpy(stored.slice(u8, tensor.count * tensor.dtype.size()), tensor.slice(u8, tensor.count * tensor.dtype.size()));
        try self.constants.put(self.g.allocator, name_, stored);
        for (self.tensors.items, 0..) |t, i| if (t.buffer.contents() == tensor.buffer.contents() and t.offset == tensor.offset) {
            t.deinit();
            _ = self.tensors.swapRemove(i);
            return self.constant(key).?;
        };
        unreachable;
    }

    pub fn name(self: *Runtime, comptime fmt: []const u8, args: anytype) ![]const u8 {
        return std.fmt.allocPrint(self.a, fmt, args);
    }
    pub fn mark(self: *Runtime) usize {
        return self.tensors.items.len;
    }
    pub fn release(self: *Runtime, tensor: Tensor) void {
        for (self.tensors.items, 0..) |t, i| if (t.buffer.contents() == tensor.buffer.contents() and t.offset == tensor.offset) {
            t.deinit();
            _ = self.tensors.swapRemove(i);
            return;
        };
        unreachable;
    }
    pub fn trim(self: *Runtime, start: usize, keep: []const Tensor) void {
        var at = start;
        for (self.tensors.items[start..]) |t| {
            var retain = false;
            for (keep) |k| retain = retain or (t.buffer.contents() == k.buffer.contents() and t.offset == k.offset);
            if (retain) {
                self.tensors.items[at] = t;
                at += 1;
            } else t.deinit();
        }
        self.tensors.shrinkRetainingCapacity(at);
    }
    pub fn own(self: *Runtime, t: Tensor) !Tensor {
        errdefer t.deinit();
        try self.tensors.append(self.g.allocator, t);
        return t;
    }
    pub fn alloc(self: *Runtime, n: usize) !Tensor {
        return self.own(try self.g.alloc(n, .f32));
    }
    pub fn device(self: *Runtime, n: usize) !Tensor {
        return self.own(try self.g.allocDevice(n, .f32));
    }
    pub fn upload(self: *Runtime, values: []const f32) !Tensor {
        return self.own(try self.g.upload(values, .f32));
    }
    pub fn weight(self: *Runtime, key: []const u8) !Tensor {
        return self.own(try self.g.weight(try self.store.get(key), .f32));
    }
    pub fn add(self: *Runtime, x: Tensor, y: Tensor) !Tensor {
        return self.own(try ops.point(self.g, x, y, if (x.count == y.count) 0 else 1, y.count));
    }
    pub fn mul(self: *Runtime, x: Tensor, y: Tensor) !Tensor {
        return self.own(try ops.point(self.g, x, y, if (x.count == y.count) 2 else 3, y.count));
    }
    pub fn scale(self: *Runtime, x: Tensor, factor: f32) !Tensor {
        const out = try self.device(x.count);
        try self.g.dispatch("im_sam_scale", &.{ x, out }, [2]f32{ @bitCast(@as(u32, @intCast(x.count))), factor }, Size.of(x.count, 1, 1), Size.of(256, 1, 1));
        return out;
    }
    pub fn unary(self: *Runtime, x: Tensor, kind: u32) !Tensor {
        const out = try self.device(x.count);
        try self.g.dispatch("im_sam_unary", &.{ x, out }, [2]u32{ @intCast(x.count), kind }, Size.of(x.count, 1, 1), Size.of(256, 1, 1));
        return out;
    }
    pub fn gelu(self: *Runtime, x: Tensor) !Tensor {
        return self.own(try ops.gelu(self.g, x, true));
    }
    pub fn norm(self: *Runtime, key: []const u8, x: Tensor, width: usize, eps: f32) !Tensor {
        const w = try self.weight(try self.name("{s}.weight", .{key}));
        const b = try self.weight(try self.name("{s}.bias", .{key}));
        return self.own(try ops.norm(self.g, x, w, b, width, false, eps));
    }
    pub fn visionNorm(self: *Runtime, key: []const u8, x: Tensor, windowed: bool) !Tensor {
        if (self.precision == .fp32) {
            const out = try self.norm(key, x, 1024, 1e-5);
            return if (windowed) self.window(out, 1024, false) else out;
        }
        const w = try self.weight(try self.name("{s}.weight", .{key}));
        const b = try self.weight(try self.name("{s}.bias", .{key}));
        const out = try self.own(try self.g.allocDevice(x.count, .f16));
        try self.g.dispatch("im_sam_vision_norm", &.{ x, w, b, out }, @as(u32, @intFromBool(windowed)), Size.of(x.count / 1024 * 256, 1, 1), Size.of(256, 1, 1));
        return out;
    }
    pub fn linear(self: *Runtime, key: []const u8, x: Tensor, width: usize) !Tensor {
        const w = try self.weight(try self.name("{s}.weight", .{key}));
        const b = try self.weight(try self.name("{s}.bias", .{key}));
        return self.own(try self.g.linear(x, w, b, x.count / width, b.count, width));
    }
    pub fn visionLinear(self: *Runtime, key: []const u8, x: Tensor, width: usize, activate: bool) !Tensor {
        if (self.precision == .fp32) {
            const out = try self.linear(key, x, width);
            return if (activate) self.gelu(out) else out;
        }
        return self.visionProjection(key, x, width, activate, null, false);
    }
    pub fn visionResidual(self: *Runtime, key: []const u8, x: Tensor, width: usize, residual: Tensor, windowed: bool) !Tensor {
        if (self.precision == .fp32) {
            const projected = try self.linear(key, x, width);
            return self.add(residual, if (windowed) try self.window(projected, 1024, true) else projected);
        }
        return self.visionProjection(key, x, width, false, residual, windowed);
    }
    fn visionProjection(self: *Runtime, key: []const u8, x: Tensor, width: usize, activate: bool, residual: ?Tensor, windowed: bool) !Tensor {
        const source = try self.store.get(try self.name("{s}.weight", .{key}));
        const w = try self.own(try self.g.weight(source, .f16));
        const b = try self.weight(try self.name("{s}.bias", .{key}));
        const input = if (x.dtype == .f16) x else try self.own(try ops.cast(self.g, x, .f16));
        const rows = x.count / width;
        if (x.count % width != 0 or rows % 64 != 0 or b.count % 128 != 0 or source.numel() != width * b.count) return error.SamProjectionShape;
        const out = try self.own(try self.g.allocDevice(rows * b.count, if (activate) .f16 else .f32));
        const Params = extern struct { rows: u32, cols: u32, depth: u32, activate: u32 };
        var tensors = [_]Tensor{ input, w, b, out, residual orelse out };
        if (residual) |t| if (t.count != out.count or t.dtype != .f32 or (windowed and rows != 5184)) return error.SamProjectionShape;
        try self.g.dispatch(if (residual != null) "im_sam_vision_residual" else if (activate) "im_sam_vision_gelu" else "im_sam_vision_linear", tensors[0..if (residual != null) @as(usize, 5) else 4], Params{ .rows = @intCast(rows), .cols = @intCast(b.count), .depth = @intCast(width), .activate = @intFromBool(if (residual != null) windowed else activate) }, Size.of(rows / 64 * 32, b.count / 128 * 4, 2), Size.of(32, 4, 2));
        return out;
    }
    pub fn project(self: *Runtime, key: []const u8, x: Tensor, width: usize, part: usize) !Tensor {
        const w = try self.weight(try self.name("{s}.in_proj_weight", .{key}));
        const b = try self.weight(try self.name("{s}.in_proj_bias", .{key}));
        if (w.count != 3 * width * width or b.count != 3 * width) return error.SamProjectionShape;
        return self.own(try self.g.linear(x, view(w, part * width * width, width * width), view(b, part * width, width), x.count / width, width, width));
    }
    pub fn columns(self: *Runtime, x: Tensor, rows: usize, width: usize, offset: usize, count: usize) !Tensor {
        if (x.count != rows * width or offset + count > width) return error.SamColumnsShape;
        const out = try self.device(rows * count);
        try self.g.dispatch("im_sam_columns", &.{ x, out }, [4]u32{ @intCast(rows), @intCast(width), @intCast(offset), @intCast(count) }, Size.of(out.count, 1, 1), Size.of(256, 1, 1));
        return out;
    }
    pub fn transpose(self: *Runtime, x: Tensor, rows: usize, width: usize) !Tensor {
        const out = try self.device(x.count);
        try self.g.dispatch("im_sam_transpose", &.{ x, out }, [2]u32{ @intCast(rows), @intCast(width) }, Size.of(x.count, 1, 1), Size.of(256, 1, 1));
        return out;
    }
    pub fn matmul(self: *Runtime, x: Tensor, w: Tensor, m: usize, n: usize, k: usize) !Tensor {
        return self.own(try self.g.matmul(x, w, m, n, k));
    }
    pub fn concat(self: *Runtime, parts: []const Tensor) !Tensor {
        var count: usize = 0;
        for (parts) |p| count += p.count;
        const out = try self.device(count);
        var at: usize = 0;
        for (parts) |p| {
            try self.g.dispatch("im_sam_copy", &.{ p, view(out, at, p.count) }, @as(u32, @intCast(p.count)), Size.of(p.count, 1, 1), Size.of(256, 1, 1));
            at += p.count;
        }
        return out;
    }
    pub fn mlp(self: *Runtime, key: []const u8, x: Tensor, width: usize, depth: usize) !Tensor {
        var h = x;
        var dim = width;
        for (0..depth) |i| {
            h = try self.linear(try self.name("{s}.layers.{d}", .{ key, i }), h, dim);
            dim = h.count / (x.count / width);
            if (i + 1 < depth) h = try self.unary(h, 0);
        }
        return h;
    }
    pub fn attend(self: *Runtime, q: Tensor, k: Tensor, v: Tensor, heads: usize, width: usize, bias: ?Tensor, causal: bool) !Tensor {
        const rows = q.count / width;
        const keys = k.count / width;
        const dim = width / heads;
        if (width % heads != 0 or k.count != v.count) return error.SamAttentionShape;
        if (self.precision == .mixed and bias == null and !causal and dim == 32) {
            if (rows < 5184 or keys < 5184) return self.own(try flash(self.g, q, k, v, rows, keys, heads, 1));
            const start = self.mark();
            const qh = try self.own(try ops.cast(self.g, q, .f16));
            const kh = try self.own(try ops.cast(self.g, k, .f16));
            const vh = try self.own(try ops.cast(self.g, v, .f16));
            const attended = try self.own(try flash(self.g, qh, kh, vh, rows, keys, heads, 1));
            const out = try self.own(try ops.cast(self.g, attended, .f32));
            self.trim(start, &.{out});
            return out;
        }
        if (bias == null and dim == 16) {
            const qp = try self.own(try ops.headsCopy(self.g, q, rows, heads, dim, 32, width, 0));
            const kp = try self.own(try ops.headsCopy(self.g, k, keys, heads, dim, 32, width, 0));
            const vp = try self.own(try ops.headsCopy(self.g, v, keys, heads, dim, 32, width, 0));
            const out = try self.own(try ops.attentionCross(self.g, qp, kp, vp, rows, keys, heads, 1 / @sqrt(@as(f32, @floatFromInt(dim)))));
            return self.own(try ops.headsCopy(self.g, out, rows, heads, 32, dim, heads * 32, 0));
        }
        if (bias == null and (dim == 32 or dim == 64 or dim == 128 or dim == 256)) return self.own(if (causal) try ops.attention(self.g, q, k, v, rows, heads, heads, true, 1 / @sqrt(@as(f32, @floatFromInt(dim)))) else try ops.attentionCross(self.g, q, k, v, rows, keys, heads, 1 / @sqrt(@as(f32, @floatFromInt(dim)))));
        if (causal or dim > 256) return error.SamAttentionShape;
        const out = try self.device(q.count);
        const pass = self.g.command();
        pass.enc.setPipeline(try self.g.pipeline("im_sam_attention"));
        for ([_]Tensor{ q, k, v, bias orelse q, out }, 0..) |t, i| pass.enc.setBuffer(t.buffer, t.offset, i);
        pass.enc.setValue([4]u32{ @intCast(rows), @intCast(keys), @intCast(heads), @intCast(dim) }, 5);
        pass.enc.setValue(@as(u32, @intFromBool(bias != null)), 6);
        pass.enc.dispatchGroups(Size.of(rows, heads, 1), Size.of(32, 1, 1));
        try self.g.finish(pass.cb, pass.enc);
        return out;
    }
    pub fn visionAttention(self: *Runtime, q: Tensor, k: Tensor, v: Tensor, global: bool) !Tensor {
        return self.own(if (self.precision == .mixed) try flash(self.g, q, k, v, if (global) 5184 else 576, if (global) 5184 else 576, 16, if (global) 1 else 9) else if (global) try ops.attention(self.g, q, k, v, 5184, 16, 16, false, 0.125) else try ops.attentionBatched(self.g, q, k, v, 576, 16, 16, false, 0.125, 9));
    }
    pub fn mha(self: *Runtime, key: []const u8, q: Tensor, k: Tensor, v: Tensor, heads: usize, width: usize, bias: ?Tensor) !Tensor {
        return self.linear(try self.name("{s}.out_proj", .{key}), try self.attend(try self.project(key, q, width, 0), try self.project(key, k, width, 1), try self.project(key, v, width, 2), heads, width, bias, false), width);
    }
    pub fn splitMha(self: *Runtime, key: []const u8, q: Tensor, k: Tensor, v: Tensor, heads: usize) !Tensor {
        const qp = try self.linear(try self.name("{s}.q_proj", .{key}), q, 256);
        const kp = try self.linear(try self.name("{s}.k_proj", .{key}), k, 256);
        const vp = try self.linear(try self.name("{s}.v_proj", .{key}), v, 256);
        const width = qp.count / (q.count / 256);
        return self.linear(try self.name("{s}.out_proj", .{key}), try self.attend(qp, kp, vp, heads, width, null, false), width);
    }
    pub fn resize(self: *Runtime, x: Feature, h: usize, w: usize, bilinear: bool) !Feature {
        const out = try self.device(h * w * x.channels);
        const pass = self.g.command();
        pass.enc.setPipeline(try self.g.pipeline("im_sam_resize"));
        pass.enc.setBuffer(x.tensor.buffer, x.tensor.offset, 0);
        pass.enc.setBuffer(out.buffer, out.offset, 1);
        pass.enc.setValue([4]u32{ @intCast(x.height), @intCast(x.width), @intCast(x.channels), @intFromBool(bilinear) }, 2);
        pass.enc.setValue([2]u32{ @intCast(h), @intCast(w) }, 3);
        pass.enc.dispatchThreads(Size.of(out.count, 1, 1), Size.of(256, 1, 1));
        try self.g.finish(pass.cb, pass.enc);
        return feature(out, h, w, x.channels);
    }
    pub fn conv(self: *Runtime, key: []const u8, x: Feature, stride: usize, pad: usize, transposed: bool, depthwise: bool) !Feature {
        const source = try self.store.get(try self.name("{s}.weight", .{key}));
        if (source.rank != 4) return error.SamConvShape;
        const k = source.dim(2);
        if (transposed and (k != 2 or stride != 2 or pad != 0)) return error.SamDeconvolutionShape;
        const n = if (transposed) source.dim(1) else source.dim(0);
        const oh = if (transposed) x.height * 2 else (x.height + 2 * pad - k) / stride + 1;
        const ow = if (transposed) x.width * 2 else (x.width + 2 * pad - k) / stride + 1;
        const raw = try self.weight(try self.name("{s}.weight", .{key}));
        const bias = if (self.store.has(try self.name("{s}.bias", .{key}))) try self.weight(try self.name("{s}.bias", .{key})) else zero: {
            const z = try self.alloc(n);
            @memset(z.slice(f32, n), 0);
            break :zero z;
        };
        const out = try self.device(oh * ow * n);
        var p = Conv{ .h = @intCast(x.height), .w = @intCast(x.width), .c = @intCast(x.channels), .n = @intCast(n), .k = @intCast(k), .oh = @intCast(oh), .ow = @intCast(ow), .stride = @intCast(stride), .pad = @intCast(pad), .first = 0, .rows = 0 };
        if (transposed and self.precision == .mixed) {
            const batch = self.g.batch == null;
            if (batch) try self.g.beginBatch();
            errdefer if (batch and self.g.batch != null) self.g.endBatch() catch {};
            const matrix = try self.convMatrix(key, raw, n, x.channels, k, true);
            const product = try self.matmul(x.tensor, matrix, x.height * x.width, 4 * n, x.channels);
            try self.g.dispatch("im_sam_deconv_shuffle", &.{ product, bias, out }, p, Size.of(out.count, 1, 1), Size.of(256, 1, 1));
            if (batch) try self.g.endBatch();
            return feature(out, oh, ow, n);
        }
        if (transposed or depthwise) {
            try self.g.dispatch(if (transposed) "im_sam_deconv" else "im_sam_depthwise", &.{ x.tensor, raw, bias, out }, p, Size.of(if (transposed) out.count / 4 else out.count, 1, 1), Size.of(256, 1, 1));
            return feature(out, oh, ow, n);
        }
        const batch = self.g.batch == null;
        if (batch) try self.g.beginBatch();
        errdefer if (batch and self.g.batch != null) self.g.endBatch() catch {};
        const matrix = try self.convMatrix(key, raw, n, x.channels, k, false);
        const width = k * k * x.channels;
        const chunk = @min(16384, @max(64, (32 * 1024 * 1024 / (width * 4)) / 64 * 64));
        var first: usize = 0;
        while (first < oh * ow) {
            const rows = @min(chunk, oh * ow - first);
            const mark_ = self.mark();
            const cols = if (k == 1 and stride == 1 and pad == 0) view(x.tensor, first * x.channels, rows * x.channels) else columns_: {
                const cols = try self.device(rows * width);
                p.first = @intCast(first);
                p.rows = @intCast(rows);
                try self.g.dispatch("im_sam_im2col", &.{ x.tensor, cols }, p, Size.of(cols.count, 1, 1), Size.of(256, 1, 1));
                break :columns_ cols;
            };
            const dst = view(out, first * n, rows * n);
            try self.g.matmulInto(cols, matrix, dst, rows, n, width);
            try self.g.dispatch("im_sam_bias", &.{ dst, bias }, [2]u32{ @intCast(dst.count), @intCast(n) }, Size.of(dst.count, 1, 1), Size.of(256, 1, 1));
            self.trim(mark_, &.{});
            first += rows;
        }
        if (batch) try self.g.endBatch();
        return feature(out, oh, ow, n);
    }
    fn convMatrix(self: *Runtime, key: []const u8, raw: Tensor, n: usize, c: usize, k: usize, transposed: bool) !Tensor {
        if (self.convolutions.get(key)) |cached| return cached;
        const limit = self.g.buffer_cache_limit;
        self.g.buffer_cache_limit = 0;
        defer self.g.buffer_cache_limit = limit;
        const t = try self.g.allocDevice(raw.count, .f32);
        errdefer t.deinit();
        try self.g.dispatch(if (transposed) "im_sam_deconv_weight" else "im_sam_conv_weight", &.{ raw, t }, [3]u32{ @intCast(n), @intCast(c), @intCast(k * k) }, Size.of(t.count, 1, 1), Size.of(256, 1, 1));
        const name_ = try self.g.allocator.dupe(u8, key);
        errdefer self.g.allocator.free(name_);
        try self.convolutions.put(self.g.allocator, name_, t);
        return t;
    }
    pub fn groupNorm(self: *Runtime, key: []const u8, x: Feature, groups: usize) !Feature {
        const w = try self.weight(try self.name("{s}.weight", .{key}));
        const b = try self.weight(try self.name("{s}.bias", .{key}));
        const out = try self.device(x.tensor.count);
        const pass = self.g.command();
        pass.enc.setPipeline(try self.g.pipeline("im_sam_group_norm"));
        for ([_]Tensor{ x.tensor, w, b, out }, 0..) |t, i| pass.enc.setBuffer(t.buffer, t.offset, i);
        pass.enc.setValue([3]u32{ @intCast(x.height * x.width), @intCast(x.channels), @intCast(groups) }, 4);
        pass.enc.dispatchGroups(Size.of(groups, 1, 1), Size.of(256, 1, 1));
        try self.g.finish(pass.cb, pass.enc);
        return feature(out, x.height, x.width, x.channels);
    }
    pub fn window(self: *Runtime, x: Tensor, width: usize, reverse: bool) !Tensor {
        const out = try self.device(x.count);
        try self.g.dispatch("im_sam_window", &.{ x, out }, [3]u32{ @intCast(width), @intFromBool(reverse), 0 }, Size.of(x.count, 1, 1), Size.of(256, 1, 1));
        return out;
    }
    pub fn rope(self: *Runtime, x: Tensor, heads: usize, width: usize, side: usize, scaled: bool, spatial: usize) !Tensor {
        const out = try self.device(x.count);
        const table = try self.rotaryTable(width / heads, side, scaled);
        const pass = self.g.command();
        pass.enc.setPipeline(try self.g.pipeline("im_sam_rope_cached"));
        pass.enc.setBuffer(x.buffer, x.offset, 0);
        pass.enc.setBuffer(table.buffer, table.offset, 1);
        pass.enc.setBuffer(out.buffer, out.offset, 2);
        pass.enc.setValue(Rope{ .rows = @intCast(x.count / width), .heads = @intCast(heads), .dim = @intCast(width / heads), .spatial = @intCast(spatial), .side = @intCast(side), .scale = if (scaled) 1.0 / 3.0 else 1 }, 3);
        pass.enc.dispatchThreads(Size.of(x.count, 1, 1), Size.of(256, 1, 1));
        try self.g.finish(pass.cb, pass.enc);
        return out;
    }
    pub fn rotaryTable(self: *Runtime, dim: usize, side: usize, scaled: bool) !Tensor {
        const key = try self.name("rotary.{d}.{d}.{}", .{ dim, side, scaled });
        if (self.constant(key)) |cached| return cached;
        const limit = self.g.buffer_cache_limit;
        self.g.buffer_cache_limit = 0;
        defer self.g.buffer_cache_limit = limit;
        const table = try self.g.allocDevice(side * side * dim, .f32);
        errdefer table.deinit();
        try self.g.dispatch("im_sam_rope_table", &.{table}, Rope{ .rows = 0, .heads = 0, .dim = @intCast(dim), .side = @intCast(side), .spatial = 0, .scale = if (scaled) 1.0 / 3.0 else 1 }, Size.of(table.count / 2, 1, 1), Size.of(256, 1, 1));
        const name_ = try self.g.allocator.dupe(u8, key);
        errdefer self.g.allocator.free(name_);
        try self.constants.put(self.g.allocator, name_, table);
        return self.constant(key).?;
    }
    pub fn visionQkv(self: *Runtime, x: Tensor, global: bool) ![3]Tensor {
        const side: usize = if (global) 72 else 24;
        const table = try self.rotaryTable(64, side, global);
        var out: [3]Tensor = undefined;
        const half = self.precision == .mixed and global;
        for (&out) |*t| t.* = try self.own(try self.g.allocDevice(5184 * 1024, if (half) .f16 else .f32));
        try self.g.dispatch(if (half) "im_sam_qkv_rope_h" else "im_sam_qkv_rope", &.{ x, table, out[0], out[1], out[2] }, @as(u32, @intCast(side)), Size.of(out[0].count, 1, 1), Size.of(256, 1, 1));
        return out;
    }
};
pub const Rope = extern struct { rows: u32, heads: u32, dim: u32, spatial: u32, side: u32, scale: f32 };
pub const Conv = extern struct { h: u32, w: u32, c: u32, n: u32, k: u32, oh: u32, ow: u32, stride: u32, pad: u32, first: u32, rows: u32 };
pub fn feature(t: Tensor, h: usize, w: usize, c: usize) Feature {
    return .{ .tensor = t, .height = h, .width = w, .channels = c };
}
pub fn view(t: Tensor, start: usize, count: usize) Tensor {
    return .{ .buffer = t.buffer, .offset = t.offset + start * t.dtype.size(), .dtype = t.dtype, .count = count, .owned = false, .host_data = t.host_data };
}

pub fn flash(g: *gpu.Gpu, q: Tensor, k: Tensor, v: Tensor, rows: usize, keys: usize, heads: usize, batch: usize) !Tensor {
    const out = try g.allocDevice(q.count, q.dtype);
    errdefer out.deinit();
    var name_: [64]u8 = undefined;
    const dim = q.count / rows / heads / batch;
    const tile: usize = if (q.dtype == .f16 and dim == 32) 64 else 32;
    const name = try std.fmt.bufPrint(&name_, "im_sam_attention_{d}_{d}{s}", .{ dim, tile, if (q.dtype == .f16) "_h" else "" });
    try g.dispatch(name, &.{ q, k, v, out }, [4]u32{ @intCast(rows), @intCast(keys), @intCast(heads), @intCast(batch) }, Size.of((rows + 63) / 64 * 32, heads * 4, batch), Size.of(32, 4, 1));
    return out;
}
