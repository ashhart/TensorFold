//! Common proposal head views over qlinear, retaining its stored format and projection recipe.
const std = @import("std");
const q = @import("qlinear.zig");
const qmmf = @import("qmmf.zig");
const memory = @import("memory.zig");
const Driver = @import("driver.zig").Driver;
const Stream = @import("stream.zig").Stream;
pub const Role = enum { decode, prompt };
pub const Head = struct {
    weight: q.Weight,
    role: Role,
    vocab: u32,
    logical: u32,
    stride: u32,
    ids: ?[]const u32 = null,
    generation: [32]u8 = @splat(0),
    queue: u64 = 0,
    pub fn full(weight: q.Weight, role: Role) Head {
        const n: u32 = @intCast(weight.outputs());
        return .{ .weight = weight, .role = role, .vocab = n, .logical = n, .stride = n };
    }
    pub fn project(h: Head, linear: q.Linear, s: Stream, in: q.Rows, out: u64, rows: usize) !void {
        return switch (h.role) {
            .decode => linear.decode(s, h.weight, in, out, rows),
            .prompt => linear.prompt(s, h.weight, in, out, rows),
        };
    }
};
pub const Packed = struct {
    gpa: std.mem.Allocator,
    words: []u16,
    ids: []const u32,
    template: qmmf.Weight,
    vocab: u32,
    role: Role,
    generation: [32]u8,
    queue: u64,
    pub fn deinit(p: *Packed) void {
        p.gpa.free(p.words);
        p.gpa.free(p.ids);
        p.* = undefined;
    }
    pub fn view(p: *const Packed, codes: u64) Head {
        var w = p.template;
        w.codes = codes;
        return .{ .weight = .{ .bf16 = w }, .role = p.role, .vocab = p.vocab, .logical = @intCast(p.ids.len), .stride = w.n, .ids = p.ids, .generation = p.generation, .queue = p.queue };
    }
    pub fn deviceBytes(p: *const Packed) !usize {
        return std.math.add(usize, try std.math.mul(usize, p.words.len, 2), try std.math.mul(usize, p.ids.len, 8));
    }
    pub fn uploadBytes(p: *const Packed) !usize {
        const words = try std.math.mul(usize, p.words.len, 2);
        const host_ids = try std.math.mul(usize, p.ids.len, 4);
        const staged_ids = try std.math.mul(usize, p.ids.len, 8);
        const custody = @min(@max(words, staged_ids), 64 << 20);
        return std.math.add(usize, try p.deviceBytes(), try std.math.add(usize, try std.math.add(usize, words, host_ids), try std.math.add(usize, staged_ids, custody)));
    }
    /// Success takes the nullable source; failure leaves its prepared generation intact.
    pub fn upload(source: *?Packed, d: *const Driver, budget: usize) !DeviceHead {
        const p = if (source.*) |*p| p else return error.ProposalAlreadyUploaded;
        if (try p.uploadBytes() > budget) return error.ProposalRowBudget;
        var codes = try memory.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(p.words));
        errdefer codes.free();
        const allocator = p.gpa;
        const ids = try allocator.alloc(i64, p.ids.len);
        defer allocator.free(ids);
        for (ids, p.ids) |*dst, id| dst.* = id;
        const mapped = try memory.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(ids));
        const result = DeviceHead{ .gpa = p.gpa, .codes = codes, .ids = mapped, .head = p.view(codes.ptr) };
        p.gpa.free(p.words);
        source.* = null;
        return result;
    }
};
pub const DeviceHead = struct {
    gpa: std.mem.Allocator,
    codes: memory.DeviceBuffer,
    ids: memory.DeviceBuffer,
    head: Head,
    captures: u32 = 0,
    pub fn retainCapture(h: *DeviceHead) !void {
        h.captures = try std.math.add(u32, h.captures, 1);
    }
    pub fn releaseCapture(h: *DeviceHead) !void {
        if (h.captures == 0) return error.ProposalCaptureUnderflow;
        h.captures -= 1;
    }
    /// Refuse live captured pointers, then prove all queues complete before releasing this immutable generation.
    pub fn deinit(h: *DeviceHead) !void {
        if (h.captures != 0) return error.ProposalCaptureLive;
        try h.codes.d.synchronize();
        h.codes.free();
        h.ids.free();
        h.gpa.free(h.head.ids.?);
        h.* = undefined;
    }
};

/// Exact BF16 permutations use qmmf's existing packer; changing a shape-selected reduction basis is refused.
pub fn bf16(gpa: std.mem.Allocator, selected: anytype, original: q.Weight) !Packed {
    const source = switch (original) {
        .bf16 => |w| w,
        else => return error.ProposalBasisUnsupported,
    };
    if (selected.output_f32 or selected.storage != .bf16 or selected.bits != 16 or selected.group != 0 or source.mode != .bf16 or selected.hidden != source.k or selected.vocab != source.n or
        source.k == 0 or source.k % 64 != 0 or selected.ids.len == 0) return error.ProposalBasisUnsupported;
    const role: Role = switch (selected.recipe) {
        .qlinear_decode => .decode,
        .qlinear_prompt => .prompt,
        else => return error.ProposalBasisUnsupported,
    };
    const n = try std.math.mul(usize, try std.math.divCeil(usize, selected.ids.len, 128), 128);
    if (n > std.math.maxInt(u32) or qmmf.splitK(source.n, source.k) != qmmf.splitK(n, source.k)) return error.ProposalBasisUnsupported;
    const cells = try std.math.mul(usize, selected.ids.len, source.k);
    if (selected.codes.len < try std.math.mul(usize, cells, 2)) return error.ProposalRowShape;
    const words = try gpa.alloc(u16, try std.math.mul(usize, n, source.k));
    errdefer gpa.free(words);
    const ids = try gpa.dupe(u32, selected.ids);
    qmmf.packBf16(words, @alignCast(std.mem.bytesAsSlice(u16, selected.codes[0 .. cells * 2])), selected.ids.len, source.k);
    var template = source;
    template.n = @intCast(n);
    template.npad = @intCast(n);
    template.codes = 0;
    return .{ .gpa = gpa, .words = words, .ids = ids, .template = template, .vocab = source.n, .role = role, .generation = selected.fingerprint, .queue = selected.queue };
}

/// Raw selected affine-4 planes go through the existing packer, never through dequantization or requantization.
pub fn affine4(kernels: *const q.Affine4Kernels, s: Stream, selected: anytype, original: q.Weight, raw: [3]u64, tiled: q.Affine4) !Head {
    const source = switch (original) {
        .affine4 => |w| w,
        else => return error.ProposalBasisUnsupported,
    };
    if (selected.output_f32 or selected.storage != .affine or selected.bits != 4 or selected.group != 64 or selected.hidden != source.k or selected.vocab != source.n or
        tiled.k != source.k or tiled.n != selected.physical or tiled.npad != tiled.n or tiled.n % 128 != 0 or
        q.splitK(source.n, source.k) != q.splitK(tiled.n, tiled.k)) return error.ProposalBasisUnsupported;
    const role: Role = switch (selected.recipe) {
        .qlinear_decode => .decode,
        .qlinear_prompt => .prompt,
        else => return error.ProposalBasisUnsupported,
    };
    try kernels.pack(s, raw[0], raw[1], raw[2], tiled);
    return .{ .weight = .{ .affine4 = tiled }, .role = role, .vocab = @intCast(source.n), .logical = @intCast(selected.ids.len), .stride = @intCast(tiled.n), .ids = selected.ids, .generation = selected.fingerprint, .queue = selected.queue };
}
