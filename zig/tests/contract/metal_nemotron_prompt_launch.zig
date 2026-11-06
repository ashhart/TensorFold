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
    pub fn deinit(s: *State) void {
        if (s.mask) |m| m.deinit();
    }
};

pub fn prepare(p: anytype) !void {
    if (p.control) return;
    if (p.prompt_run.mask == null) p.prompt_run.mask = try Masks.load(p.a, p.rig.device, p.prompt_full);
    p.prompt_run.pick = try p.rig.scratch(64);
    @memset(p.rig.bytes(p.prompt_run.pick, 64), 0);
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
        launch.attention(.{ .b = p.qref.buf, .off = p.qref.off }, .{ .b = p.kv[0].buf, .off = p.kv[0].off }, .{ .b = p.kv[1].buf, .off = p.kv[1].off }, .{ .b = p.out.buf, .off = p.out.off }, t, kl, heads, kvh, strides, fl[0], null, .{ .b = p.prompt_run.pick.buf, .off = p.prompt_run.pick.off });
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
