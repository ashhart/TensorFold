//! Optional original bf16 input rows for proposal generation; the target keeps its loaded embedding weights.
const std = @import("std");
const core = @import("core");
const cuda = @import("cuda");
const Engine = @import("cuda_engine.zig").Engine;

pub const Table = struct {
    buffer: cuda.DeviceBuffer,
    kernels: cuda.qwen35_bf16_rows.Kernels,

    pub fn deinit(t: *Table) void {
        t.kernels.unload();
        t.buffer.free();
    }

    pub fn encode(t: *const Table, e: *Engine, ids: u64, out: u64, rows: usize, hidden: usize) !void {
        try t.kernels.embedRows(e.stream, ids, t.buffer.ptr, out, rows, hidden, 0);
    }
};

pub fn load(gpa: std.mem.Allocator, io: std.Io, e: *Engine, vocab: usize, hidden: usize) !?Table {
    const dir = (try @import("sys").environment.given("TF_DRAFT_BF16_EMBED_DIR")) orelse return null;
    if (e.head != .a4 or !@import("cuda_geometry.zig").dense27(e.g))
        return error.UnsupportedDraftEmbedding;
    var ck = try core.Checkpoint.openModel(gpa, io, dir);
    defer core.checkpoint.cache.close(&ck);
    const tensor = try ck.expect("model.language_model.embed_tokens.weight", .bf16, &.{ vocab, hidden });
    var buffer = try cuda.DeviceBuffer.alloc(e.d, tensor.bytes.len);
    errdefer buffer.free();
    try buffer.upload(0, tensor.bytes);
    var kernels = try cuda.qwen35_bf16_rows.Kernels.load(e.d);
    errdefer kernels.unload();
    std.log.info("DFlash2 bf16 proposal embeddings: {d} bytes", .{buffer.len});
    return .{ .buffer = buffer, .kernels = kernels };
}
