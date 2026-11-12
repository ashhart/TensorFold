//! Host-only Qwen27 GGUF preparation owns exact affine arrays and borrows the original dense bytes until deinit.
const std = @import("std");
const g = @import("core").gguf;
pub const metadata = @import("gguf_config.zig");
pub const names = @import("gguf_names.zig");
pub const Problem = metadata.Problem;
pub const Affine = struct { shape: g.affine.Layout, words: []const u32, scales: []const f32, biases: []const f32 };
pub const Tensor = struct {
    name: names.Name,
    source: g.Tensor,
    data: union(enum) { dense: []const u8, affine: Affine },
};
fn check(t: g.Tensor, n: names.Name, p: *Problem) !void {
    const rank: usize = if (n.role == .convolution) 2 else n.rank;
    if (t.rank != rank) return p.fail("{s}: tensor rank {d}, expected {d}", .{ t.name, t.rank, rank });
    for (0..rank) |i| {
        const dim = if (n.role == .convolution) n.dims[1 - i] else n.dims[rank - 1 - i];
        if (t.shape[i] != dim) return p.fail("{s}: dimension {d} is {d}, expected {d}", .{ t.name, i, t.shape[i], dim });
    }
    switch (t.kind) {
        .f32, .f16, .bf16 => {},
        .q4_0, .q4_1, .q5_0, .q5_1, .q8_0, .q2_k, .q3_k, .q4_k, .q5_k, .q6_k => if (n.role != .matrix) return p.fail("{s}: {s} is unsupported for a small tensor", .{ t.name, @tagName(t.kind) }),
        else => return p.fail("{s}: unsupported quantization {s} ({d})", .{ t.name, @tagName(t.kind), @backingInt(t.kind) }),
    }
}
pub const Adapter = struct {
    document: g.Document,
    config: @import("config.zig").Config,
    arena: std.heap.ArenaAllocator,
    tensors: []Tensor,
    pub fn open(a: std.mem.Allocator, bytes: []const u8, p: *Problem) !Adapter {
        var unsupported: g.UnsupportedTensor = .{};
        var doc = g.Document.parseDiagnosed(a, bytes, .{}, &unsupported) catch |err| switch (err) {
            error.OutOfMemory => return err,
            error.UnsupportedTensorType => return p.fail("{s}: unsupported GGML quantization type {d}", .{ unsupported.name, unsupported.kind }),
            else => return p.fail("GGUF reader: {s}", .{@errorName(err)}),
        };
        errdefer doc.deinit();
        const c = try metadata.parse(&doc, p);
        var arena: std.heap.ArenaAllocator = .init(a);
        errdefer arena.deinit();
        const mem = arena.allocator();
        const expected = try names.plan(mem, c);
        for (expected) |n| try check(doc.getTensor(n.gguf) orelse return p.fail("{s}: missing tensor", .{n.gguf}), n, p);
        for (doc.tensors) |t| {
            var found = false;
            for (expected) |n| found = found or std.mem.eql(u8, n.gguf, t.name);
            if (!found) return p.fail("{s}: unsupported tensor", .{t.name});
        }
        const tensors = try mem.alloc(Tensor, expected.len);
        for (tensors, expected) |*out, n| {
            const t = doc.getTensor(n.gguf).?;
            out.* = .{ .name = n, .source = t, .data = undefined };
            switch (t.kind) {
                .f32, .f16, .bf16 => out.data = .{ .dense = t.bytes },
                else => {
                    const q = g.affine.repack(mem, t.kind, t.bytes) catch |err| switch (err) {
                        error.OutOfMemory => return err,
                        else => return p.fail("{s}: {s}", .{ t.name, @errorName(err) }),
                    };
                    out.data = .{ .affine = .{ .shape = q.shape, .words = q.words, .scales = q.scales, .biases = q.biases } };
                },
            }
        }
        return .{ .document = doc, .config = c, .arena = arena, .tensors = tensors };
    }
    pub fn get(a: *const Adapter, native_name: []const u8) ?*const Tensor {
        for (a.tensors) |*t| if (std.mem.eql(u8, t.name.native, native_name)) return t;
        return null;
    }
    pub fn deinit(a: *Adapter) void {
        a.arena.deinit();
        a.document.deinit();
        a.* = undefined;
    }
};
