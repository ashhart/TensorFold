//! BF16 projection plans and selected-token scoring for the CUDA Qwen3.5 engine.
const cuda = @import("cuda");
const Engine = @import("cuda_engine.zig").Engine;
const Proj = @import("cuda_layers.zig").Proj;
pub fn projectionPlan(e: *Engine, rows: usize, n: usize, k: usize) !*cuda.cublaslt.Linear {
    const key = (@as(u64, rows) << 48) | (@as(u64, n) << 24) | @as(u64, k);
    const item = try e.bf16_plans.getOrPut(e.gpa, key);
    errdefer {
        if (!item.found_existing) _ = e.bf16_plans.remove(key);
    }
    if (!item.found_existing) item.value_ptr.* = try cuda.cublaslt.Linear.init(&e.bf16_lt.?, rows, n, k, .bf16, e.bf16_workspace.?.len);
    return item.value_ptr;
}
pub fn prepare(e: *Engine, rows: usize, heads: usize) !void {
    for (e.layers) |l| {
        switch (l.mlp) {
            .dense => |m| {
                for ([_]Proj{ m.gate, m.up, m.down }) |p| {
                    _ = try projectionPlan(e, rows, p.outputs(), p.inputs());
                }
            },
            else => return error.UnsupportedQwenQuantization,
        }
        switch (l.mixer) {
            .delta => |m| {
                for ([_]Proj{ m.qkv, m.z, m.a, m.b, m.out }) |p| {
                    _ = try projectionPlan(e, rows, p.outputs(), p.inputs());
                }
            },
            .attention => |m| {
                for ([_]Proj{ m.q, m.k, m.v, m.o }) |p| {
                    _ = try projectionPlan(e, rows, p.outputs(), p.inputs());
                }
            },
        }
    }
    _ = try projectionPlan(e, heads, e.g.vocab, e.g.hidden);
    if (heads <= 4) _ = try scorePlan(e, heads);
}
fn scorePlan(e: *Engine, rows: usize) !*cuda.cublaslt.Linear {
    const item = try e.score_plans.getOrPut(e.gpa, rows);
    errdefer {
        if (!item.found_existing) _ = e.score_plans.remove(rows);
    }
    if (!item.found_existing) item.value_ptr.* = try cuda.cublaslt.Linear.init(&e.bf16_lt.?, rows, e.g.vocab, e.g.hidden, .f32, e.bf16_workspace.?.len);
    return item.value_ptr;
}
pub fn head(e: *Engine, x: u64, rows: usize) !void {
    const plan = try scorePlan(e, rows);
    const ws = e.bf16_workspace.?;
    try plan.run(x, e.head.bf16.w, e.score_logits, ws.ptr, ws.len, e.stream.handle);
    try e.run(e.fns.score_stats, .{ rows, 1, 1 }, .{ 256, 1, 1 }, .{ e.score_logits, @as(c_int, @intCast(e.g.vocab)), e.score_stats, e.score_labels });
}

pub fn init(e: *Engine, d: *const cuda.Driver, enabled: bool) !void {
    e.bf16_lt = null;
    e.bf16_workspace = null;
    e.bf16_embed = null;
    e.bf16_plans = .empty;
    e.score_plans = .empty;
    errdefer deinit(e);
    if (!enabled) return;
    e.bf16_lt = try cuda.cublaslt.Library.open();
    e.bf16_workspace = try cuda.DeviceBuffer.alloc(d, 32 * 1024 * 1024);
    e.bf16_embed = try cuda.embed_rows.EmbedRows.load(d);
}

pub fn deinit(e: *Engine) void {
    var plans = e.bf16_plans.valueIterator();
    while (plans.next()) |plan| plan.deinit();
    e.bf16_plans.deinit(e.gpa);
    var score_plans = e.score_plans.valueIterator();
    while (score_plans.next()) |plan| plan.deinit();
    e.score_plans.deinit(e.gpa);
    if (e.bf16_embed) |*embed_| embed_.unload();
    if (e.bf16_workspace) |*ws| ws.free();
    if (e.bf16_lt) |*lt| lt.close();
}

pub fn stageLabels(e: *Engine, parts: []const @import("cuda_engine.zig").Part) !void {
    const labels = e.host_score_logits.slice(u32);
    for (parts, 0..) |p_, i| {
        if (p_.cache.score_labels[0] > 16) return error.InvalidLabelCount;
        @memcpy(labels[i * 17 ..][0..17], &p_.cache.score_labels);
    }
    try e.d.check(e.d.api.cuMemcpyHtoDAsync_v2(e.score_labels, e.host_score_logits.bytes.ptr, parts.len * 17 * 4, e.stream.handle), "cuMemcpyHtoDAsync score labels");
}

pub fn read(e: *Engine, parts: []const @import("cuda_engine.zig").Part, heads: usize) !void {
    const out = e.host_score_logits.bytes[0 .. 32 + heads * 16 * 4];
    try e.d.check(e.d.api.cuMemcpyDtoHAsync_v2(out.ptr, e.score_stats, out.len, e.stream.handle), "cuMemcpyDtoHAsync selected scores");
    try e.stream.synchronize();
    for (parts, 0..) |p_, i| {
        p_.cache.score_lse = e.host_score_logits.slice(f32)[i * 2];
        p_.cache.score_token = e.host_score_logits.slice(u32)[i * 2 + 1];
        @memcpy(&p_.cache.scores, e.host_score_logits.slice(f32)[8 + i * 16 ..][0..16]);
    }
}

pub fn validateLabels(labels: *const [17]u32, vocab: usize) !void {
    if (labels[0] > 16) return error.InvalidLabelCount;
    for (labels[1..][0..labels[0]]) |id| if (id >= vocab) return error.InvalidLabel;
}
