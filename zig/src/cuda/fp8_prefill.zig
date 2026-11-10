//! Prompt-only FP8 products, preserving the existing weight allocations and decode arithmetic.
const std = @import("std");
const Driver = @import("driver.zig").Driver;
const Module = @import("module.zig").Module;
const Function = @import("module.zig").Function;
const Buffer = @import("memory.zig").DeviceBuffer;
const Stream = @import("stream.zig").Stream;
const launch = @import("launch.zig");
const kernels = @import("kernels.zig");
const qmmf = @import("qmmf.zig");
const Affine4 = @import("qlinear.zig").Affine4;
pub const tile_geometry = @import("fp8_prefill_tiles.zig");
pub const affine_packing = @import("affine_fp8_prepared.zig");

pub fn selected(raw: ?[]const u8) !bool {
    const value = raw orelse return false;
    if (std.mem.eql(u8, value, "0")) return false;
    if (std.mem.eql(u8, value, "1")) return true;
    return error.InvalidFastPrefill;
}

pub fn requested() !bool {
    _ = try tile_geometry.requested();
    _ = try affine_packing.requested();
    return selected(try @import("sys").environment.given("TF_QWEN_FAST_PREFILL"));
}

pub fn enabled() bool {
    return requested() catch false;
}

pub fn compatible(active: bool, learning: bool, integer: bool, full: bool) !void {
    if (active and (learning or integer or full)) return error.IncompatibleFastPrefill;
}

pub const Native = struct {
    mod: Module,
    quant: Function,
    affine_quant: Function,
    row_quant: Function,
    dense: [2]Function,
    affine: [2]Function,
    row: [2]Function,
    scratch: Buffer,
    max_rows: usize,
    max_width: usize,
    tiles: ?tile_geometry.Tiles = null,
    affine_prepared: ?affine_packing.Prepared = null,

    pub fn load(d: *const Driver, rows: usize, width: usize) !Native {
        const mode = try tile_geometry.requested();
        const prepare_affine = try affine_packing.requested();
        var m = try Module.load(d, kernels.fp8_prefill);
        errdefer m.unload();
        var tiles = if (mode == .baseline) null else try tile_geometry.Tiles.load(d, mode);
        errdefer if (tiles) |*t| t.unload();
        var affine_prepared = if (prepare_affine) try affine_packing.Prepared.load(d) else null;
        errdefer if (affine_prepared) |*p| p.unload();
        return .{
            .mod = m,
            .quant = try m.function("tf_fp8_prompt_quant"),
            .affine_quant = try m.function("tf_affine4_fp8_quant"),
            .row_quant = try m.function("tf_fp8_row_quant"),
            .dense = .{ try m.function("tf_fp8_prompt"), try m.function("tf_fp8_prompt_f32") },
            .affine = .{ try m.function("tf_affine4_fp8_prompt"), try m.function("tf_affine4_fp8_prompt_f32") },
            .row = .{ try m.function("tf_fp8_row_prompt"), try m.function("tf_fp8_row_prompt_f32") },
            .scratch = try Buffer.alloc(d, rows * width + rows * (width / 64) * 8),
            .max_rows = rows,
            .max_width = width,
            .tiles = tiles,
            .affine_prepared = affine_prepared,
        };
    }

    pub fn unload(n: *Native) void {
        if (n.affine_prepared) |*p| p.unload();
        if (n.tiles) |*t| t.unload();
        n.scratch.free();
        n.mod.unload();
    }

    /// Reserves one reusable weight-code buffer from the actual affine projections before KV planning.
    pub fn reserveAffinePacked(n: *Native, bytes: usize) !void {
        if (n.affine_prepared) |*p| try p.reserve(bytes);
    }

    pub fn affinePackedBytes(n: *const Native) usize {
        return if (n.affine_prepared) |*p| p.capacity() else 0;
    }

    fn shape(n: *const Native, m: usize, k: usize, stride: usize) !void {
        if (m == 0 or m > n.max_rows or k == 0 or k > n.max_width or k % 64 != 0 or stride < k)
            return error.InvalidFp8PromptShape;
    }

    pub fn matmul(n: *const Native, s: Stream, x: u64, stride: usize, m: usize, w: qmmf.Weight, out: u64, kind: qmmf.Out) !void {
        try n.shape(m, w.k, stride);
        if (w.mode != .fp8 or !@import("nvfp4_prompt.zig").validScale(w.act)) return error.InvalidFp8PromptScale;
        var q: launch.Args = .{};
        q.add(x);
        q.add(n.scratch.ptr);
        for ([_]usize{ m, w.k, stride }) |v| q.add(@as(i32, @intCast(v)));
        q.add(@as(f32, 1.0 / w.act));
        try launch.launch(n.quant, .{ .grid = .{ .x = @intCast((m * (w.k / 16) + 255) / 256) }, .block = .{ .x = 256 } }, s, &q);
        if (n.tiles) |*t| return t.product(s, n.scratch.ptr, 0, m, w, out, kind, false);
        var a: launch.Args = .{};
        a.add(n.scratch.ptr);
        a.add(w.codes);
        a.add(out);
        for ([_]usize{ m, w.n, w.k, w.npad }) |v| a.add(@as(i32, @intCast(v)));
        a.add(@as(f32, w.act * w.scale));
        try launch.launch(n.dense[@intFromBool(kind == .f32)], .{ .grid = .{ .x = @intCast((w.n + 127) / 128), .y = @intCast((m + 63) / 64) }, .block = .{ .x = 256 } }, s, &a);
    }

    pub fn affine4(n: *Native, s: Stream, x: u64, m: usize, w: anytype, out: u64, kind: qmmf.Out) !void {
        try n.shape(m, w.k, w.k);
        if (n.affine_prepared) |*p| if (affine_packing.selectedRows(m) and p.fits(w)) {
            try n.validateAffine(x, w.k, m, &.{w}, &.{out}, kind);
            try p.admit(s);
        };
        const input = try n.quantizeAffine(s, x, w.k, m, w.k);
        return n.affineProduct(s, input, m, w, out, kind);
    }

    /// One K64-group quantization per call, with all projections consuming the same codes, scales and sums.
    pub fn affine4Many(n: *Native, s: Stream, x: u64, stride: usize, m: usize, weights: []const Affine4, outs: []const u64, kind: qmmf.Out) !void {
        try n.validateAffine(x, stride, m, weights, outs, kind);
        if (n.affine_prepared) |*p| if (affine_packing.selectedRows(m)) {
            for (weights) |w| if (p.fits(w)) {
                try p.admit(s);
                break;
            };
        };
        const input = try n.quantizeAffine(s, x, stride, m, weights[0].k);
        for (weights, outs) |w, out| try n.affineProduct(s, input, m, w, out, kind);
    }

    pub fn canShareAffine4(n: *const Native, x: u64, stride: usize, m: usize, weights: []const Affine4, outs: []const u64, kind: qmmf.Out) bool {
        n.validateAffine(x, stride, m, weights, outs, kind) catch return false;
        return true;
    }

    fn validateAffine(n: *const Native, x: u64, stride: usize, m: usize, weights: []const Affine4, outs: []const u64, kind: qmmf.Out) !void {
        if (weights.len == 0 or weights.len != outs.len) return error.InvalidFp8PromptShape;
        const k = weights[0].k;
        try n.shape(m, k, stride);
        const maximum = std.math.maxInt(i32);
        if (m > maximum or k > maximum or stride > maximum) return error.InvalidFp8PromptShape;
        const code_bytes = try std.math.mul(usize, m, k);
        const metadata_bytes = try std.math.mul(usize, try std.math.mul(usize, m, k / 64), 8);
        if (try std.math.add(usize, code_bytes, metadata_bytes) > n.scratch.len) return error.InvalidFp8PromptShape;
        const elements = try std.math.add(usize, try std.math.mul(usize, m - 1, stride), k);
        const input = try Span.init(x, elements, 2);
        const scratch = try Span.init(n.scratch.ptr, n.scratch.len, 1);
        if (input.overlaps(scratch)) return error.InvalidFp8PromptAlias;
        const prepared_span: ?Span = if (n.affine_prepared) |p|
            if (p.buffer) |buffer| try Span.init(buffer.ptr, buffer.len, 1) else null
        else
            null;
        if (prepared_span) |span| if (span.overlaps(input) or span.overlaps(scratch)) return error.InvalidFp8PromptAlias;
        for (weights, outs, 0..) |w, out, i| {
            if (w.k != k or w.n == 0 or w.n > maximum or w.npad > maximum or
                w.npad % 64 != 0 or w.npad < (w.n + 63) / 64 * 64) return error.InvalidFp8PromptShape;
            const output = try Span.init(out, try std.math.mul(usize, m, w.n), if (kind == .f32) 4 else 2);
            if (output.overlaps(input) or output.overlaps(scratch)) return error.InvalidFp8PromptAlias;
            if (prepared_span) |span| if (span.overlaps(output)) return error.InvalidFp8PromptAlias;
            for (weights) |source| {
                const weight_span = try Span.init(source.w, (try std.math.mul(usize, source.npad, source.k)) / 2, 1);
                const scales = try Span.init(source.s, try std.math.mul(usize, source.npad, source.k / 64), 2);
                const biases = try Span.init(source.b, try std.math.mul(usize, source.npad, source.k / 64), 2);
                for ([_]Span{ weight_span, scales, biases }) |read| {
                    if (output.overlaps(read) or scratch.overlaps(read)) return error.InvalidFp8PromptAlias;
                    if (prepared_span) |span| if (span.overlaps(read)) return error.InvalidFp8PromptAlias;
                }
            }
            for (weights[0..i], outs[0..i]) |previous, address| {
                const before = try Span.init(address, try std.math.mul(usize, m, previous.n), if (kind == .f32) 4 else 2);
                if (output.overlaps(before)) return error.InvalidFp8PromptAlias;
            }
        }
    }

    const AffineInput = struct { scales: u64, sums: u64 };
    fn quantizeAffine(n: *const Native, s: Stream, x: u64, stride: usize, m: usize, k: usize) !AffineInput {
        const scales = n.scratch.ptr + m * k;
        const sums = scales + m * (k / 64) * 4;
        var q: launch.Args = .{};
        for ([_]u64{ x, n.scratch.ptr, scales, sums }) |v| q.add(v);
        for ([_]usize{ m, k, stride }) |v| q.add(@as(i32, @intCast(v)));
        try launch.launch(n.affine_quant, .{ .grid = .{ .x = @intCast((m * (k / 64) + 7) / 8) }, .block = .{ .x = 256 } }, s, &q);
        return .{ .scales = scales, .sums = sums };
    }
    fn affineProduct(n: *Native, s: Stream, input: AffineInput, m: usize, w: anytype, out: u64, kind: qmmf.Out) !void {
        if (n.affine_prepared) |*p| if (affine_packing.selectedRows(m) and
            try p.matmul(s, n.scratch.ptr, input.scales, input.sums, m, w, out, kind)) return;
        var a: launch.Args = .{};
        for ([_]u64{ n.scratch.ptr, input.scales, input.sums, w.w, w.s, w.b, out }) |v| a.add(v);
        for ([_]usize{ m, w.n, w.k, w.npad }) |v| a.add(@as(i32, @intCast(v)));
        try launch.launch(n.affine[@intFromBool(kind == .f32)], .{ .grid = .{ .x = @intCast((w.n + 127) / 128), .y = @intCast((m + 63) / 64) }, .block = .{ .x = 256 } }, s, &a);
    }

    pub fn rowScaled(n: *const Native, s: Stream, x: u64, m: usize, w: qmmf.Weight, out: u64, kind: qmmf.Out) !void {
        try n.shape(m, w.k, w.k);
        if (w.mode != .fp8g) return error.InvalidFp8PromptScale;
        const scales = try n.quantizeRows(s, x, w.k, m, w.k);
        return n.rowProduct(s, scales, m, w, out, kind);
    }

    /// Quantizes this call's common input once; no prepared activation state survives the call.
    pub fn rowScaledMany(n: *const Native, s: Stream, x: u64, stride: usize, m: usize, weights: []const qmmf.Weight, outs: []const u64, kind: qmmf.Out) !void {
        try n.validateRows(x, stride, m, weights, outs, kind);
        const scales = try n.quantizeRows(s, x, stride, m, weights[0].k);
        for (weights, outs) |w, out| try n.rowProduct(s, scales, m, w, out, kind);
    }

    pub fn canShareRows(n: *const Native, x: u64, stride: usize, m: usize, weights: []const qmmf.Weight, outs: []const u64, kind: qmmf.Out) bool {
        n.validateRows(x, stride, m, weights, outs, kind) catch return false;
        return true;
    }

    fn validateRows(n: *const Native, x: u64, stride: usize, m: usize, weights: []const qmmf.Weight, outs: []const u64, kind: qmmf.Out) !void {
        if (weights.len == 0 or weights.len != outs.len) return error.InvalidFp8PromptShape;
        const k = weights[0].k;
        try n.shape(m, k, stride);
        const code_bytes = try std.math.mul(usize, m, k);
        if (try std.math.add(usize, code_bytes, try std.math.mul(usize, m, 4)) > n.scratch.len)
            return error.InvalidFp8PromptShape;
        const elements = try std.math.add(usize, try std.math.mul(usize, m - 1, stride), k);
        const input = try Span.init(x, elements, 2);
        const scratch = try Span.init(n.scratch.ptr, n.scratch.len, 1);
        if (input.overlaps(scratch)) return error.InvalidFp8PromptAlias;
        for (weights, outs, 0..) |w, out, i| {
            if (w.mode != .fp8g) return error.InvalidFp8PromptScale;
            if (w.k != k or w.n == 0 or w.npad % 64 != 0 or w.npad < (@as(u64, w.n) + 63) / 64 * 64)
                return error.InvalidFp8PromptShape;
            const output = try Span.init(out, try std.math.mul(usize, m, w.n), if (kind == .f32) 4 else 2);
            if (output.overlaps(input) or output.overlaps(scratch)) return error.InvalidFp8PromptAlias;
            for (weights) |source| {
                const codes = try Span.init(source.codes, try std.math.mul(usize, source.npad, source.k), 1);
                const scales = try Span.init(source.scales, try std.math.mul(usize, source.npad, source.k / 64), 4);
                if (output.overlaps(codes) or output.overlaps(scales)) return error.InvalidFp8PromptAlias;
            }
            for (weights[0..i], outs[0..i]) |previous, address| {
                const before = try Span.init(address, try std.math.mul(usize, m, previous.n), if (kind == .f32) 4 else 2);
                if (output.overlaps(before)) return error.InvalidFp8PromptAlias;
            }
        }
    }

    fn quantizeRows(n: *const Native, s: Stream, x: u64, stride: usize, m: usize, k: usize) !u64 {
        const scales = n.scratch.ptr + m * k;
        var q: launch.Args = .{};
        for ([_]u64{ x, n.scratch.ptr, scales }) |v| q.add(v);
        for ([_]usize{ m, k, stride }) |v| q.add(@as(i32, @intCast(v)));
        try launch.launch(n.row_quant, .{ .grid = .{ .x = @intCast(m) }, .block = .{ .x = 256 } }, s, &q);
        return scales;
    }

    fn rowProduct(n: *const Native, s: Stream, scales: u64, m: usize, w: qmmf.Weight, out: u64, kind: qmmf.Out) !void {
        if (n.tiles) |*t| return t.product(s, n.scratch.ptr, scales, m, w, out, kind, true);
        var a: launch.Args = .{};
        for ([_]u64{ n.scratch.ptr, scales, w.codes, w.scales, out }) |v| a.add(v);
        for ([_]usize{ m, w.n, w.k, w.npad }) |v| a.add(@as(i32, @intCast(v)));
        try launch.launch(n.row[@intFromBool(kind == .f32)], .{ .grid = .{ .x = @intCast((w.n + 127) / 128), .y = @intCast((m + 63) / 64) }, .block = .{ .x = 256 } }, s, &a);
    }
};

const Span = struct {
    begin: u64,
    end: u64,
    fn init(address: u64, count: usize, bytes: usize) !Span {
        if (address == 0) return error.InvalidFp8PromptAlias;
        return .{ .begin = address, .end = try std.math.add(u64, address, try std.math.mul(usize, count, bytes)) };
    }
    fn overlaps(a: Span, b: Span) bool {
        return a.begin < b.end and b.begin < a.end;
    }
};

test {
    _ = @import("fp8_prefill_test.zig");
}
