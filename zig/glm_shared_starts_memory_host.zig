//! Allocation geometry from the actual eight-layer constructors, with address-only driver calls and no weights.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm_cuda");
const check = @import("check");

const Driver = struct {
    var next: u64 = 1 << 30;
    fn alloc(out: *u64, n: usize) callconv(.c) cuda.abi.Result {
        out.* = next;
        next += std.mem.alignForward(u64, n + 256, 256);
        return 0;
    }
    fn free(_: u64) callconv(.c) cuda.abi.Result {
        return 0;
    }
    fn clear(_: u64, _: u8, _: usize) callconv(.c) cuda.abi.Result {
        return 0;
    }
};

pub fn main() !void {
    const a = std.heap.page_allocator;
    var d: cuda.Driver = undefined;
    d.api.cuMemAlloc_v2 = Driver.alloc;
    d.api.cuMemFree_v2 = Driver.free;
    d.api.cuMemsetD8_v2 = Driver.clear;
    const c: glm.config.Config = .{};
    _ = try check.checkSettings(c, .{ .layers = check.layer_count });
    var layers: [check.layer_count]glm.weights.Layer = undefined;
    for (&layers, 0..) |*l, i| {
        l.index = @intCast(i);
        l.mixer = if (c.kind(@intCast(i)) == .mla) .{ .mla = undefined } else .{ .kda = undefined };
        if (l.mixer == .kda) l.mixer.kda.proj = .{ .bf16 = 0 };
        l.ffn = if (c.isMoe(@intCast(i))) .{ .moe = undefined } else .{ .dense = undefined };
        if (l.ffn == .moe) l.ffn.moe.exl3 = null;
    }
    const w: glm.weights.Weights = .{ .gpa = a, .config = c, .share = try glm.shard.Share.of(&c, 0, 1), .layers = &layers };
    var prompt = try glm.forward.Buffers.init(&d, &w, glm.forward.prefill_rows, true, check.capacity);
    defer prompt.deinit();
    var decode = try glm.forward.Buffers.init(&d, &w, glm.forward.window_rows, false, check.capacity);
    defer decode.deinit();
    var verify = try glm.forward.Buffers.init(&d, &w, glm.forward.window_rows, false, check.capacity);
    defer verify.deinit();
    var whole = try glm.forward.State.init(a, &d, &w, check.capacity);
    defer whole.deinit();
    var mla: usize = 0;
    for (layers) |l| mla += @intFromBool(l.mixer == .mla);
    const per_page = (mla + 1) * (512 * c.kv_lora * 2 + 128 * c.i_dim * 2);
    var states: [3]glm.forward.State = undefined;
    for (&states) |*s| s.* = try glm.forward.State.initPaged(a, &d, &w, check.capacity, .{ .kv = 1 << 28, .pages = glm.pages.tableLen(check.capacity), .page_bytes = per_page });
    defer for (&states) |*s| s.deinit();
    var seq: @typeInfo(@FieldType(glm.engine.Engine, "cur")).pointer.child = .{ .st = states[0] };
    var e: glm.engine.Engine = undefined;
    e.cur = &seq;
    e.c = c;
    e.capacity = check.capacity;
    e.share_pages = false;
    e.mtp = false;
    const snapshot_budget = 4 * e.stateBytes(check.capacity - 64);
    const pool_pages = 3 * glm.pages.tableLen(check.capacity) + 8;
    const pool_bytes = pool_pages * per_page;
    const buffers = prompt.arena.buf.len + decode.arena.buf.len + verify.arena.buf.len;
    const state_bytes = 3 * states[0].arena.buf.len + whole.arena.buf.len;
    const routed = (check.layer_count - c.dense_layers) * (cuda.experts.Nvfp4Experts.bytes(c.experts, c.moe_inter, c.hidden, 2) + cuda.experts.Nvfp4Experts.bytes(c.experts, c.hidden, c.moe_inter, 1) + 3 * c.experts * 4);
    const embedding_head = 2 * @as(u64, c.vocab) * c.hidden * 2;
    const counted = buffers + state_bytes + pool_bytes + snapshot_budget;
    std.debug.print("MEMORY_MODEL {f}\n", .{std.json.fmt(.{
        .scope = "default_geometry_no_checkpoint_no_device",
        .layers = check.layer_count,
        .prompt_buffer = prompt.arena.buf.len,
        .decode_buffer = decode.arena.buf.len,
        .paged_slot_fixed = states[0].arena.buf.len,
        .unpaged_slot = whole.arena.buf.len,
        .kv_page_bytes = per_page,
        .kv_pool_pages = pool_pages,
        .kv_pool_bytes = pool_bytes,
        .snapshot_budget = snapshot_budget,
        .counted_nonweight_bytes = counted,
        .routed_nvfp4_packed_bytes = routed,
        .bf16_embedding_and_head_bytes = embedding_head,
        .planning_allowance_other_weights_and_transients = @as(u64, 8 << 30),
        .planning_total = counted + routed + embedding_head + 8 * @as(u64, 1 << 30),
        .tracked_ceiling = check.memory_ceiling,
    }, .{})});
    if (counted + routed + embedding_head + 8 * @as(u64, 1 << 30) >= check.memory_ceiling) return error.SharedStartsModelExceedsCeiling;
}
