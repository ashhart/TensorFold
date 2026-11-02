//! Load-scope regressions: actual CUDA sources must defer checkpoint release until every borrowed read has ended.
const std = @import("std");
const t = std.testing;
const Ast = std.zig.Ast;

fn scope(comptime path: []const u8, function: []const u8, variable: []const u8) !void {
    const source = @embedFile("src/" ++ path);
    var tree = try Ast.parse(t.allocator, source, .{});
    defer tree.deinit(t.allocator);
    try t.expectEqual(@as(usize, 0), tree.errors.len);
    var found: usize = 0;
    for (0..tree.nodes.len) |i| {
        const node: Ast.Node.Index = @fromBackingInt(@intCast(i));
        if (tree.nodeTag(node) != .fn_decl) continue;
        var one: [1]Ast.Node.Index = undefined;
        const proto = tree.fullFnProto(&one, node).?;
        const name = proto.name_token orelse continue;
        if (!std.mem.eql(u8, function, tree.tokenSlice(name))) continue;
        const body = tree.nodeData(node).node_and_node[1];
        var two: [2]Ast.Node.Index = undefined;
        const statements = tree.blockStatements(&two, body).?;
        for (statements, 0..) |statement, at| {
            const decl = tree.fullVarDecl(statement) orelse continue;
            if (!std.mem.eql(u8, variable, tree.tokenSlice(decl.ast.mut_token + 1))) continue;
            found += 1;
            try t.expect(at + 1 < statements.len);
            const next = statements[at + 1];
            try t.expectEqual(.@"defer", tree.nodeTag(next));
            const call = tree.nodeData(next).node;
            if (std.mem.eql(u8, variable, "file")) {
                var block: [2]Ast.Node.Index = undefined;
                const cleanup = tree.blockStatements(&block, call).?;
                try t.expectEqual(@as(usize, 2), cleanup.len);
                try t.expectEqualStrings("file.release()", tree.getNodeSource(cleanup[0]));
                try t.expectEqualStrings("file.close(io)", tree.getNodeSource(cleanup[1]));
                continue;
            }
            var buf: [160]u8 = undefined;
            const want = try std.fmt.bufPrint(&buf, "core.checkpoint.cache.close(&{s})", .{if (std.mem.eql(u8, variable, "reader")) "reader.ck" else variable});
            try t.expectEqualStrings(want, tree.getNodeSource(call));
        }
    }
    try t.expectEqual(@as(usize, 1), found);
}

const qwen = "families/qwen3_5/";
const gemma = "families/gemma4/";
const nemo = "families/nemotron/";
const mm = "families/minimax_m3/";
const glm = "families/glm/";

test "CUDA load cache: Qwen dense backbone closes after all load reads" {
    try scope(qwen ++ "cuda_engine.zig", "init", "ck");
    try scope(qwen ++ "cuda_train_load.zig", "weights", "ck");
}
test "CUDA load cache: Qwen MoE backbone and its separate heads" {
    try scope(qwen ++ "cuda_engine.zig", "init", "ck");
    try scope(qwen ++ "cuda_mtp_weights.zig", "load", "ck");
    try scope(qwen ++ "cuda_mtp_bf16.zig", "load", "ck");
    try scope(qwen ++ "cuda_draft_weights.zig", "load", "ck");
}
test "CUDA load cache: Qwen EXL3 shares the backbone checkpoint owner" {
    try scope(qwen ++ "cuda_engine.zig", "init", "ck");
}
test "CUDA load cache: Gemma backbone, vision and both draft loaders" {
    try scope(gemma ++ "cuda_weights.zig", "load", "ck");
    try scope(gemma ++ "cuda_assistant_weights.zig", "load", "ck");
    try scope(gemma ++ "cuda_dflash.zig", "load", "ck");
}
test "CUDA load cache: Nemotron backbone and separate MTP" {
    try scope(nemo ++ "cuda_load.zig", "fill", "ck");
    try scope(nemo ++ "cuda_weights.zig", "loadMtp", "ck");
    try scope(nemo ++ "cuda_worker.zig", "loadLayers", "ck");
    try scope(nemo ++ "cuda_train.zig", "loadRaw", "ck");
}
test "CUDA load cache: Spark" {
    try scope("families/spark25/weights.zig", "load", "ck");
}
test "CUDA load cache: Kolibri" {
    try scope("families/kolibri1/weights.zig", "load", "ck");
}
test "CUDA load cache: MiniMax backbone, DSpark and vision" {
    try scope(mm ++ "cuda_weights.zig", "load", "ck");
    try scope(mm ++ "cuda_dspark.zig", "load", "ck");
    try scope(mm ++ "cuda_vision.zig", "load", "ck");
}
test "CUDA load cache: GLM final head and MTP reads after layer releases" {
    try scope(glm ++ "cuda_weights.zig", "load", "ck");
    try scope(glm ++ "cuda_weights.zig", "load", "head_ck");
    try scope(glm ++ "cuda_vision.zig", "load", "ck");
}
test "CUDA load cache: Dory" {
    try scope("families/dory/cuda_weights.zig", "load", "ck");
}
test "CUDA load cache: shared Qwen and Flash Next image tower" {
    try scope("vision/qwen_cuda_weights.zig", "load", "reader");
}

test "CUDA load cache: DeepSeek finishes backbone and DSpark before the pack sweep" {
    const s = @embedFile("src/families/deepseek_v41/cuda_model.zig");
    const drop = std.mem.indexOf(u8, s, "@import(\"core\").checkpoint.cache.files(io, m.pack.dir, m.pack.files.items)") orelse return error.NoFinalPackDrop;
    const last = std.mem.lastIndexOf(u8, s, "try m.weights.upload(&m.driver, &b)").?;
    try t.expect(last < drop);
    const simple = @embedFile("src/families/deepseek_v41/cuda_weights.zig");
    try t.expect(std.mem.indexOf(u8, simple, "try w.upload(d, &b);\n    @import(\"core\").checkpoint.cache.files(io, pack.dir, pack.files.items);") != null);
    const prepared = @embedFile("src/families/deepseek_v41/prepared.zig");
    try t.expectEqual(@as(usize, 2), std.mem.count(u8, prepared, "defer @import(\"core\").checkpoint.cache.drop(file)"));
}

test "CUDA load cache: Flash Next sweep follows MTP and retains live PLE files" {
    const s = @embedFile("src/families/flashnext/cuda_lanes_load.zig");
    const drop = std.mem.indexOf(u8, s, ".dropPageCache(io, dir, e.ngram.files.items)") orelse return error.NoPleExclusion;
    try t.expect(std.mem.indexOf(u8, s, "mtp.Head.init").? < drop);
    const owner = @embedFile("src/families/flashnext/cuda_prompt_wt.zig");
    try t.expect(std.mem.indexOf(u8, owner, "if (!runtimePages(self.f.names.keys())) self.f.release()") != null);
}

test "CUDA load cache: LFM2 decision model releases its direct-read fallback" {
    try scope("families/lfm2/weights.zig", "load", "file");
}
