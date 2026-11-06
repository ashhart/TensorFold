//! Prompt probes use the production GPU selector; their baseline retains the original direct dispatch.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const rig = @import("rig.zig");
const K = tf.nemotron.prefill_kernels.Kernels;
const Masks = @typeInfo(@FieldType(K, "mask")).optional.child;

pub const State = struct {
    mask: ?Masks = null,
    pick: rig.Ref = undefined,
    baseline: bool = false,
    paged: bool = false,
    pages: rig.Ref = undefined,
    table: rig.Ref = undefined,
    page_count: usize = 0,
    originals: [4]?mtl.Pipeline = @splat(null),
    page_model: tf.nemotron.kv_pages.Pipelines = undefined,
    pub fn deinit(s: *State) void {
        if (s.mask) |m| m.deinit();
        for (s.originals) |pipe| if (pipe) |x| x.deinit();
    }
};

pub fn prepare(p: anytype) !void {
    if (p.control) return;
    if (p.prompt_run.mask == null) p.prompt_run.mask = try Masks.load(p.a, p.rig.device, p.prompt_full);
    p.prompt_run.pick = try p.rig.scratch(64);
    @memset(p.rig.bytes(p.prompt_run.pick, 64), 0);
    if (!p.prompt_run.paged) return;
    const state = &p.prompt_run;
    if (state.originals[0] == null) {
        const raw = @import("kernel_sources").nemotron_paged;
        const text = if (p.prompt_full) |on| try tf.frags.sourceMode(p.rig.device, p.a, raw.prefill, on) else try tf.frags.source(p.rig.device, p.a, raw.prefill);
        defer p.a.free(text);
        for (raw.prefill_names, &state.originals, &state.page_model.prompt) |name, *slot, *borrowed| {
            slot.* = try rig.pipeline(p.rig.device, text, name);
            borrowed.* = slot.*.?;
        }
    }
    state.page_count = (p.cap + 255) / 256;
    const ids = try p.a.alloc(u32, state.page_count);
    defer p.a.free(ids);
    for (ids, 0..) |*id, i| id.* = @intCast(ids.len - 1 - i);
    state.table = try p.rig.input(std.mem.sliceAsBytes(ids));
    const data = try p.a.alloc(u8, state.page_count * page_bytes);
    defer p.a.free(data);
    @memset(data, 0xff); // unmapped/past-capacity rows must not participate
    state.pages = try p.rig.input(data);
    stagePages(p);
}

const page_bytes = 4 * 256 * 128 * 2;

/// The same fixture in reversed physical pages, including both K and V planes and every KV head.
pub fn stagePages(p: anytype) void {
    const state = &p.prompt_run;
    if (!state.paged) return;
    const out = p.rig.bytes(state.pages, state.page_count * page_bytes);
    for (0..p.cap) |key| for (0..2) |plane| for (0..2) |h| {
        const physical = state.page_count - 1 - key / 256;
        const off = physical * page_bytes + ((plane * 2 + h) * 256 + key % 256) * 128 * 2;
        const source = p.at(plane == 1, h, key);
        @memcpy(out[off..][0 .. 128 * 2], std.mem.sliceAsBytes(p.host[source..][0..128]));
    };
}

pub fn run(p: anytype, e: mtl.ComputeEncoder) !void {
    const t = p.rows;
    const kl = p.len + t;
    const D = 128;
    const heads = 32;
    const kvh = 2;
    const row = heads * D;
    const kvs = kvh * p.cap * D;
    var pr: [32]i32 = @splat(0);
    const head = [_]usize{ 1, heads, t, kl, heads / kvh, (t + 63) / 64, (kl + 31) / 32, t / 64, kl / 32, t % 64, kl % 32, p.len };
    const strides = [12]usize{ t * row, D, row, kvs, p.cap * D, D, kvs, p.cap * D, D, t * row, D, row };
    for (head ++ strides, 0..) |x, i| pr[i] = @intCast(x);
    var fl: [16]f32 = @splat(0);
    fl[0] = 1 / @sqrt(@as(f32, D));
    const variant = @as(usize, @intFromBool(t % 64 == 0)) * 2 + @intFromBool(kl % 32 == 0);
    const original = p.prompts[variant].?;
    if (p.prompt_run.mask != null and !p.prompt_run.baseline) {
        var kernels: K = .{ .gpa = p.a, .mask = p.prompt_run.mask };
        defer kernels.pipelines.deinit(p.a);
        var buf: [160]u8 = undefined;
        const name = try std.fmt.bufPrint(&buf, "custom_kernel_tf_attention_nax_bf16_{c}_{c}_t_bfloat16_t_bfloat16_t_bfloat16_t_int32_t_float_bfloat16_t", .{ if (t % 64 == 0) @as(u8, 't') else 'n', if (kl % 32 == 0) @as(u8, 't') else 'n' });
        try kernels.pipelines.put(p.a, name, original);
        var enc = tf.nemotron.encoder.Enc{ .e = e, .concurrent = true };
        const launch = tf.nemotron.prefill_launch.Launch{ .k = &kernels, .e = &enc };
        const state = &p.prompt_run;
        const kv0 = if (state.paged) state.pages else p.kv[0];
        const kv1 = if (state.paged) rig.Ref{ .buf = state.pages.buf, .off = state.pages.off + 2 * 256 * D * 2 } else p.kv[1];
        const pages: ?tf.nemotron.prefill_launch.Paged = if (state.paged) .{ .pipes = &state.page_model, .table = .{ .b = state.table.buf, .off = state.table.off, .page_bytes = page_bytes } } else null;
        launch.attention(.{ .b = p.qref.buf, .off = p.qref.off }, .{ .b = kv0.buf, .off = kv0.off }, .{ .b = kv1.buf, .off = kv1.off }, .{ .b = p.out.buf, .off = p.out.off }, t, kl, heads, kvh, strides, fl[0], pages, .{ .b = state.pick.buf, .off = state.pick.off });
        return;
    }
    e.setPipeline(original);
    e.setBuffer(p.qref.buf, p.qref.off, 0);
    e.setBuffer(p.kv[0].buf, p.kv[0].off, 1);
    e.setBuffer(p.kv[1].buf, p.kv[1].off, 2);
    e.setValue(pr, 3);
    e.setValue(fl, 4);
    e.setBuffer(p.out.buf, p.out.off, 5);
    e.dispatchThreads(mtl.Size.of((t + 63) / 64 * 32, heads * 4, 1), mtl.Size.of(32, 4, 1));
}
