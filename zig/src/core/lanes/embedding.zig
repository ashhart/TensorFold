//! Embedding batches borrow request storage until completion and declare each independent input's token boundary.
const std = @import("std");

pub const Info = struct {
    dimensions: u32,
    vocabulary: u32,
    max_batch_tokens: u32,
    max_inputs: u32 = 128,
    mrl_dimensions: []const u32 = &.{},

    pub fn dimension(i: Info, requested: ?u32) !u32 {
        const n = requested orelse return i.dimensions;
        if (n == 0 or n > i.dimensions) return error.UnsupportedEmbeddingDimension;
        if (i.mrl_dimensions.len == 0) return error.MrlNotDeclared;
        if (n == i.dimensions or std.mem.indexOfScalar(u32, i.mrl_dimensions, n) != null) return n;
        return error.UnsupportedEmbeddingDimension;
    }
};

pub const Batch = struct {
    ends: []const u32,
    dimensions: u32,
    output: []f32,

    pub fn validate(b: Batch, info: Info, tokens: []const u32, context: u32) !void {
        if (info.dimensions == 0 or info.vocabulary == 0 or info.max_batch_tokens == 0 or
            b.dimensions != info.dimensions or b.ends.len == 0 or b.ends.len > info.max_inputs or
            tokens.len == 0 or tokens.len > info.max_batch_tokens) return error.InvalidEmbeddingBatch;
        const values = try std.math.mul(usize, b.ends.len, info.dimensions);
        if (b.output.len != values) return error.InvalidEmbeddingBatch;
        var at: usize = 0;
        for (b.ends) |end| {
            if (end <= at or end > tokens.len or (context > 0 and end - at > context)) return error.InvalidEmbeddingBatch;
            at = end;
        }
        if (at != tokens.len) return error.InvalidEmbeddingBatch;
        for (tokens) |token| if (token >= info.vocabulary) return error.InvalidEmbeddingToken;
        const bytes = std.mem.sliceAsBytes(b.output);
        if (overlap(bytes, std.mem.sliceAsBytes(tokens)) or
            overlap(bytes, std.mem.sliceAsBytes(b.ends))) return error.EmbeddingOutputAliasesInput;
    }
};

fn overlap(a: []const u8, b: []const u8) bool {
    if (a.len == 0 or b.len == 0) return false;
    const ap = @intFromPtr(a.ptr);
    const bp = @intFromPtr(b.ptr);
    return if (ap <= bp) bp - ap < a.len else ap - bp < b.len;
}
