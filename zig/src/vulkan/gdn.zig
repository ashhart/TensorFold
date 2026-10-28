//! Qwen3.5 GatedDeltaNet GPU primitives over the shared set-0 storage /
//! push-constant pipeline model, matching the unchanged Ash8c Metal oracle
//! (zig/kernels/metal/qwen3_5/gdn_{pre,chain,post}.metal) op for op.
//!
//! Kernels (kernels/vulkan/, all compiled -DUSE_BF16 --target-env vulkan1.2):
//!   conv_dw1d.comp            prefill 4-tap conv (copied mlx-omarchy, MIT)
//!   gdn_conv_decode.comp      decode conv + window shift + SiLU (copied, MIT)
//!   gdn_split.comp            SiLU + Q/K L2 norm split + V + a/b gate (new)
//!   gated_delta_prefill_ash.comp   gated-delta prefill scan, u8 mask binding
//!                             removed (copied default kept alongside; MIT)
//!   gated_delta_decode.comp   gated-delta decode step, raw-gates flags
//!                             bit8|bit9 (copied, MIT)
//!   fast_norm_gated.comp      gated RMSNorm + SiLU (copied, MIT)
//!   copy_general.comp -DUSE_U16  contiguous bf16 copies (conv-input assembly)
//!
//! The gate formula is taken from the Metal oracle, not assumed: s is bf16,
//! softplus is bf16-rounded, g = exp(-exp(A_log_f32) * softplus) is stored
//! f32, and beta = sigmoid(b) is stored bf16. A_log rides f32, dt_bias bf16.
//! The decode kernel's bit9 flag both selects the f32 A_log read and the
//! kv-walk association the Metal chain uses (state*g then *k). Its
//! raw-gates prologue has an ASH_QWEN35 variant: softplus is the Metal
//! gdn_pre form — DIRECT log(1+exp(-|x|)), bf16-rounded (not the MLX
//! elementwise log1p ratio form, and not the C++ fallback's f32 sp).
//!
//! Binding counts vs dispatch.max_bindings = 10:
//!   conv_dw1d 3, gdn_conv_decode 5, gdn_split 8, fast_norm_gated 4,
//!   copy_u16 4, gated_delta_prefill_ash 10, gated_delta_decode 10.
//!
//! Shared memory: gdn_split 128 B; prefill scan 3*128 f32 (1.5 KiB); decode
//! 3*128 f32 + 32*129 f32 tile (about 18 KiB); fast_norm_gated 1 KiB;
//! gdn_conv_decode declares a 1 KiB block (used only by its unused flag-2
//! epilogue). The prefill scan uses subgroup relative shuffles: the device
//! must expose them for the compute stage and a subgroup size that is a
//! multiple of 4 (checked in init, refused fail-closed). 16-bit storage +
//! int16 are required everywhere.
//!
//! State lifetime: conv state ([3, 6144] bf16) and recurrence state
//! ([16*128*128] f32, plus hf and the prefill snapshot) are caller-owned
//! device buffers written out of place. They must stay alive until every
//! submission using them returned success; a Timeout/InFlight error must be
//! routed through commands.orExit (exit without freeing) rather than deinit.
//! 18 linear layers per model: 18 conv-state and 18 recurrence-state
//! buffers (plus per-stream hf where a caller snapshots mid-prefill).

const std = @import("std");
const dispatch = @import("dispatch.zig");
const memory = @import("memory.zig");
const params = @import("stage1_params");
const tensor = @import("tensor.zig");
const c = @import("vk");
const Device = @import("device.zig").Device;
const Commands = @import("commands.zig").Commands;
const Buffer = memory.Buffer;
const Error = dispatch.Error;
const View = tensor.View;

/// The fixed Qwen3.5 linear-layer geometry these primitives serve.
pub const Model = struct {
    pub const linear_layers = 18;
    pub const heads = 16;
    pub const head_dim = 128;
    pub const taps = 4;
    pub const conv_channels = 6144;
    pub const conv_state_rows = taps - 1;
    pub const linear_dim = heads * head_dim; // 2048
    pub const state_elements = heads * head_dim * head_dim; // 262144 f32
};

/// qk l2 norm epsilon term, as the Metal oracle writes it: 1e-6 / DK.
const qk_norm_eps: f32 = 1e-6 / @as(f32, Model.head_dim);

const dummy_bytes: u64 = 256;

comptime {
    if (Model.conv_channels != 2 * Model.heads * Model.head_dim + Model.heads * Model.head_dim)
        @compileError("conv channels drifted from 2*NK*DK + NV*DV");
}

pub const Shaders = struct {
    /// conv_dw1d.comp -DUSE_BF16
    conv_dw1d_bf16: []align(4) const u8,
    /// gdn_conv_decode.comp -DUSE_BF16 (no APPLE_REDUCTION: the Ash arm
    /// uses flag bit 0 only)
    gdn_conv_decode_bf16: []align(4) const u8,
    /// gdn_split.comp -DUSE_BF16
    gdn_split_bf16: []align(4) const u8,
    /// gated_delta_prefill_ash.comp -DUSE_BF16 (both passes dispatch this
    /// one image; the pass is a push-constant flag, not a variant)
    prefill_recur_bf16: []align(4) const u8,
    /// gated_delta_decode.comp -DUSE_BF16 (raw gates ride flags bit8|bit9)
    gated_delta_decode_bf16: []align(4) const u8,
    /// fast_norm_gated.comp -DUSE_BF16 (default build, mode 0 at runtime)
    fast_norm_gated_bf16: []align(4) const u8,
    /// copy_general.comp -DUSE_U16
    copy_u16: []align(4) const u8,
};

/// Push block sizes, each equal to the GLSL block glslang emits.
const conv_push_bytes: u32 = 36; // count..output_offset (9 words)
const conv_decode_push_bytes: u32 = 60; // count..flags (15 words)
const full_push_bytes: u32 = params.push_bytes; // 120
/// gated_delta_prefill_ash.comp appends `chunk` and `chunks` after the
/// standard Params: 32 words.
const prefill_push_bytes: u32 = 128;

/// Params plus the prefill scan's trailing chunk fields, matching the
/// shader's push block field for field.
const PrefillPush = extern struct {
    base: params.Params = .{},
    chunk: u32 = 0,
    chunks: u32 = 0,
};

comptime {
    if (@sizeOf(PrefillPush) != prefill_push_bytes) @compileError("prefill push block drifted");
    if (@offsetOf(PrefillPush, "chunk") != params.push_bytes) @compileError("prefill chunk field misaligned");
}
const norm_push_bytes: u32 = @offsetOf(params.Params, "beta") + 4; // 68

fn groups(count: u64, local: u32, max_x: u32) Error![3]u32 {
    const n = std.math.cast(u32, (count + local - 1) / local) orelse return error.TooLarge;
    return .{ @min(n, max_x), 1, 1 };
}

fn wantBf16(views: []const View) Error!void {
    for (views) |v| if (v.kind != .bf16) return error.Invalid;
}

fn wantF32(views: []const View) Error!void {
    for (views) |v| if (v.kind != .f32) return error.Invalid;
}

/// The gate split's three outputs, wherever split runs (standalone or inside
/// convPrefill/convDecodeStep): soa and beta are bf16, g is f32. One validator
/// so the compound calls cannot drift back to a bf16 g_out.
pub fn wantSplitOutputs(out_soa: View, g_out: View, beta_out: View) Error!void {
    try wantBf16(&.{ out_soa, beta_out });
    try wantF32(&.{g_out});
}

/// Refuses overlap between the views a kernel writes and the ones it reads
/// across dispatches of one composite op (Pipeline.bind already refuses
/// same-dispatch overlaps).
fn refuseOverlap(writes: []const View, reads: []const View) Error!void {
    for (writes) |w| for (reads) |r| {
        if (w.overlaps(r)) return error.Aliased;
    };
}

pub const Gdn = struct {
    dev: *const Device,
    copy_u16: dispatch.Pipeline,
    conv_bf16: dispatch.Pipeline,
    conv_decode_bf16: dispatch.Pipeline,
    split_bf16: dispatch.Pipeline,
    prefill_bf16: dispatch.Pipeline,
    decode_bf16: dispatch.Pipeline,
    norm_bf16: dispatch.Pipeline,
    /// Stand-in for declared bindings a dispatch never touches.
    dummy: Buffer,

    pub fn init(dev: *const Device, shaders: Shaders) Error!Gdn {
        if (!dev.features.storage16 or !dev.features.int16) return error.MissingFeature;
        // The prefill scan's row totals ride two subgroupShuffleDown calls:
        // relative shuffles must exist for the compute stage, and the row
        // lanes (4 per row) must sit in one subgroup of a multiple-of-4 size.
        if (dev.v11.subgroupSupportedStages & c.VK_SHADER_STAGE_COMPUTE_BIT == 0) return error.MissingFeature;
        if (dev.v11.subgroupSupportedOperations & c.VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT == 0) return error.MissingFeature;
        if (dev.v11.subgroupSize == 0 or dev.v11.subgroupSize % 4 != 0) return error.MissingFeature;
        var g: Gdn = .{
            .dev = dev,
            .copy_u16 = undefined,
            .conv_bf16 = undefined,
            .conv_decode_bf16 = undefined,
            .split_bf16 = undefined,
            .prefill_bf16 = undefined,
            .decode_bf16 = undefined,
            .norm_bf16 = undefined,
            .dummy = try Buffer.init(dev, .device, 2 * dummy_bytes),
        };
        errdefer g.dummy.deinit();
        g.copy_u16 = try dispatch.Pipeline.init(dev, shaders.copy_u16, .{ .bindings = 4, .push_bytes = full_push_bytes, .writes = 1 << 2 });
        errdefer g.copy_u16.deinit();
        g.conv_bf16 = try dispatch.Pipeline.init(dev, shaders.conv_dw1d_bf16, .{ .bindings = 3, .push_bytes = conv_push_bytes, .writes = 1 << 2 });
        errdefer g.conv_bf16.deinit();
        g.conv_decode_bf16 = try dispatch.Pipeline.init(dev, shaders.gdn_conv_decode_bf16, .{ .bindings = 5, .push_bytes = conv_decode_push_bytes, .writes = (1 << 3) | (1 << 4) });
        errdefer g.conv_decode_bf16.deinit();
        g.split_bf16 = try dispatch.Pipeline.init(dev, shaders.gdn_split_bf16, .{ .bindings = 8, .push_bytes = full_push_bytes, .writes = 0b1110_0000 });
        errdefer g.split_bf16.deinit();
        // 10 bindings each: refused with TooManyBindings while
        // dispatch.max_bindings is 8. See the module doc block.
        g.prefill_bf16 = try dispatch.Pipeline.init(dev, shaders.prefill_recur_bf16, .{ .bindings = 10, .push_bytes = prefill_push_bytes, .writes = (1 << 6) | (1 << 7) | (1 << 9) });
        errdefer g.prefill_bf16.deinit();
        g.decode_bf16 = try dispatch.Pipeline.init(dev, shaders.gated_delta_decode_bf16, .{ .bindings = 10, .push_bytes = full_push_bytes, .writes = (1 << 6) | (1 << 7) });
        errdefer g.decode_bf16.deinit();
        g.norm_bf16 = try dispatch.Pipeline.init(dev, shaders.fast_norm_gated_bf16, .{ .bindings = 4, .push_bytes = norm_push_bytes, .writes = 1 << 3 });
        return g;
    }

    pub fn deinit(m: *Gdn) void {
        m.copy_u16.deinit();
        m.conv_bf16.deinit();
        m.conv_decode_bf16.deinit();
        m.split_bf16.deinit();
        m.prefill_bf16.deinit();
        m.decode_bf16.deinit();
        m.norm_bf16.deinit();
        m.dummy.deinit();
    }

    fn dummyView(m: *const Gdn, kind: tensor.Kind) Error!View {
        return View.init(&m.dummy, kind, 0, dummy_bytes / kind.size());
    }

    fn snapshotDummyView(m: *const Gdn) Error!View {
        // Binding 9 is declared writable even when the single-chunk shader never touches it.
        return View.init(&m.dummy, .f32, dummy_bytes, dummy_bytes / tensor.Kind.f32.size());
    }

    fn bindPrefill(m: *Gdn, bindings: []const dispatch.Binding) Error!void {
        m.prefill_bf16.bind(bindings) catch |e| {
            if (e == error.Aliased) {
                const names = [_][]const u8{ "q", "k", "v", "g_bf16_dummy", "beta", "h0", "y", "hf", "g_f32", "snapshot" };
                for (bindings, 0..) |w, i| if (m.prefill_bf16.layout.writes >> @intCast(i) & 1 != 0) {
                    for (bindings, 0..) |r, j| {
                        if (i != j and w.buffer.handle == r.buffer.handle and w.offset < r.offset + r.len and r.offset < w.offset + w.len)
                            std.debug.print("GDN prefill alias: {s} binding={d} buffer={any} bytes=[{d},{d}) overlaps {s} binding={d} buffer={any} bytes=[{d},{d})\n", .{
                                names[i], i, w.buffer.handle, w.offset, w.offset + w.len, names[j], j, r.buffer.handle, r.offset, r.offset + r.len,
                            });
                    }
                };
            }
            return e;
        };
    }

    fn copy(m: *Gdn, cmds: *Commands, out: View, src: View, timeout_ns: u64) Error!void {
        const d = try m.dummyView(.u32);
        try m.copy_u16.bind(&.{ src.bind(), d.bind(), out.bind(), d.bind() });
        const count = try src.elemCount32();
        var p: params.Params = .{
            .count = count,
            .dims = 1,
            .shape = .{ count, 0, 0, 0 },
            .in_strides = .{ 1, 0, 0, 0 },
            .out_strides = .{ 1, 0, 0, 0 },
        };
        try m.copy_u16.run(cmds, std.mem.asBytes(&p), try groups(count, m.copy_u16.local[0], m.dev.properties.limits.maxComputeWorkGroupCount[0]), timeout_ns);
    }

    /// The gate split: raw conv output [tokens, C] -> SiLU, Q/K L2 norm,
    /// V passthrough (SoA [3, tokens, 16, 128] bf16 out) and the a/b gate
    /// (g f32, beta bf16, both [tokens, 16]).
    pub fn split(m: *Gdn, cmds: *Commands, out_soa: View, g_out: View, beta_out: View, conv_out: View, a: View, b: View, a_log: View, dt: View, tokens: u64, timeout_ns: u64) Error!void {
        try wantBf16(&.{ conv_out, a, b, dt, out_soa, beta_out });
        try wantF32(&.{ a_log, g_out });
        const plan = try splitPlan(conv_out.elements, out_soa.elements, g_out.elements, beta_out.elements, a.elements, b.elements, a_log.elements, dt.elements, tokens);
        try refuseOverlap(&.{ out_soa, g_out, beta_out }, &.{ conv_out, a, b, a_log, dt });
        try m.split_bf16.bind(&.{
            conv_out.bind(), a.bind(),     b.bind(),        a_log.bind(), dt.bind(),
            out_soa.bind(),  g_out.bind(), beta_out.bind(),
        });
        var p: params.Params = .{
            .count = plan.tokens,
            .lhs_size = plan.conv_channels,
            .rhs_size = plan.tokens,
            .output_size = 3 * Model.heads,
        };
        const most = m.dev.properties.limits.maxComputeWorkGroupCount;
        if (plan.tokens > most[0] or 3 * Model.heads > most[1]) return error.BadGroupCount;
        try m.split_bf16.run(cmds, std.mem.asBytes(&p), .{ plan.tokens, 3 * Model.heads, 1 }, timeout_ns);
    }

    /// Prefill conv: assemble the [tokens+3, C] conv input (conv state rows
    /// then the projection output), run the 4-tap conv, the gate split, and
    /// store the trailing [3, C] window as the next conv state.
    /// `a`/`b` are the gate projections [tokens, 16]; `a_log` f32 [16];
    /// `dt` bf16 [16]; writes are out of place.
    pub fn convPrefill(m: *Gdn, cmds: *Commands, out_soa: View, g_out: View, beta_out: View, conv_out: View, conv_input: View, conv_state_in: View, qkv: View, weight: View, a: View, b: View, a_log: View, dt: View, conv_state_out: View, tokens: u64, timeout_ns: u64) Error!void {
        try wantBf16(&.{ conv_out, conv_input, conv_state_in, qkv, weight, a, b, dt, conv_state_out });
        try wantSplitOutputs(out_soa, g_out, beta_out);
        try wantF32(&.{a_log});
        const plan = try convPrefillPlan(conv_input.elements, conv_state_in.elements, qkv.elements, weight.elements, conv_state_out.elements, tokens);
        try refuseOverlap(&.{ conv_input, conv_state_out }, &.{ conv_state_in, qkv, weight });
        try refuseOverlap(&.{ out_soa, g_out, beta_out, conv_out }, &.{ conv_input, a, b, a_log, dt });
        // rows 0..2 of the conv input are the stored conv state
        try m.copy(cmds, try conv_input.slice(0, plan.state_elements), conv_state_in, timeout_ns);
        // rows 3.. are the projection output
        try m.copy(cmds, try conv_input.slice(plan.state_elements, plan.tokens_elements), qkv, timeout_ns);
        try m.conv_bf16.bind(&.{ conv_input.bind(), weight.bind(), conv_out.bind() });
        var p: params.Params = .{
            .count = plan.count,
            .operation = Model.taps,
            .lhs_size = plan.l_in,
            .rhs_size = plan.tokens,
            .reduce_size = Model.conv_channels,
            .output_size = 1,
        };
        try m.conv_bf16.run(cmds, std.mem.asBytes(&p)[0..conv_push_bytes], try groups(plan.count, m.conv_bf16.local[0], m.dev.properties.limits.maxComputeWorkGroupCount[0]), timeout_ns);
        try m.split(cmds, out_soa, g_out, beta_out, conv_out, a, b, a_log, dt, tokens, timeout_ns);
        // the next conv state: conv input rows tokens..tokens+3
        try m.copy(cmds, conv_state_out, try conv_input.slice(plan.tokens_elements, plan.state_elements), timeout_ns);
    }

    /// Decode conv step: one token. The kernel folds the concat(state, x)
    /// read, the 4-tap conv and the shifted carry-out state into one dispatch
    /// and stores the RAW bf16 conv output (flags bit 0 clear); the gate
    /// split that follows on [1, C] owns the SINGLE SiLU activation, exactly
    /// like Metal gdn_pre (conv, then one activation inside the split).
    pub fn convDecodeStep(m: *Gdn, cmds: *Commands, out_soa: View, g_out: View, beta_out: View, activated: View, conv_state_in: View, x: View, weight: View, a: View, b: View, a_log: View, dt: View, conv_state_out: View, timeout_ns: u64) Error!void {
        try wantBf16(&.{ activated, conv_state_in, x, weight, a, b, dt, conv_state_out });
        try wantSplitOutputs(out_soa, g_out, beta_out);
        try wantF32(&.{a_log});
        const plan = try convDecodePlan(conv_state_in.elements, x.elements, weight.elements, activated.elements, conv_state_out.elements, out_soa.elements, g_out.elements, beta_out.elements, a.elements, b.elements, a_log.elements, dt.elements);
        try refuseOverlap(&.{ activated, conv_state_out }, &.{ conv_state_in, x, weight });
        try refuseOverlap(&.{ out_soa, g_out, beta_out }, &.{ activated, a, b, a_log, dt });
        try m.conv_decode_bf16.bind(&.{ conv_state_in.bind(), x.bind(), weight.bind(), activated.bind(), conv_state_out.bind() });
        var p: params.Params = .{
            .count = plan.count,
            .operation = Model.taps,
            .lhs_size = Model.conv_state_rows,
            .reduce_size = plan.conv_channels,
            .flags = 0, // raw conv store: the split owns the single SiLU (Metal gdn_pre)
        };
        try m.conv_decode_bf16.run(cmds, std.mem.asBytes(&p)[0..conv_decode_push_bytes], try groups(plan.count, m.conv_decode_bf16.local[0], m.dev.properties.limits.maxComputeWorkGroupCount[0]), timeout_ns);
        try m.split(cmds, out_soa, g_out, beta_out, activated, a, b, a_log, dt, 1, timeout_ns);
    }

    /// Gated-delta prefill recurrence over `tokens` tokens in chunks of
    /// `chunk`: pass 0 (skipped when a single chunk) writes the per-chunk
    /// snapshot, pass 1 writes the bf16 outputs and the terminal state.
    /// q/k/v ride one SoA buffer (the split's output shape), g is the f32
    /// gate the split produced, h0/hf are [16*128*128] f32 states. When
    /// `tokens <= chunk`, pass `snap = null` (the shader's snapshot slot is
    /// never touched and stays on the module dummy).
    pub fn prefillRecur(m: *Gdn, cmds: *Commands, soa: View, g: View, beta: View, h0: View, y: View, hf: View, snap: ?View, tokens: u64, chunk: u64, timeout_ns: u64) Error!void {
        try wantBf16(&.{ soa, beta, y });
        try wantF32(&.{ g, h0, hf });
        if (snap) |s| try wantF32(&.{s});
        const plan = try prefillPlan(soa.elements, g.elements, beta.elements, h0.elements, y.elements, hf.elements, if (snap) |s| s.elements else 0, tokens, chunk);
        const q = try soa.slice(0, plan.section_elements);
        const k = try soa.slice(plan.section_elements, plan.section_elements);
        const v = try soa.slice(2 * plan.section_elements, plan.section_elements);
        try refuseOverlap(&.{ y, hf }, &.{ soa, g, beta, h0 });
        if (snap) |s| try refuseOverlap(&.{s}, &.{ h0, hf, y });
        const scan_flags: u32 = 4; // bit2: g is f32
        const out_flags: u32 = 4 | 8; // bit2 + bit3: output pass
        if (plan.chunks > 1) {
            const snap_v = snap orelse return error.Invalid;
            const db = try m.dummyView(.bf16);
            try m.bindPrefill(&.{
                q.bind(),  k.bind(), v.bind(),  db.bind(), beta.bind(),
                h0.bind(), y.bind(), hf.bind(), g.bind(),  snap_v.bind(),
            });
            try m.runPrefillPass(cmds, scan_flags, plan, timeout_ns);
        }
        const snap_binding = snap orelse try m.snapshotDummyView();
        const db = try m.dummyView(.bf16);
        try m.bindPrefill(&.{
            q.bind(),  k.bind(), v.bind(),  db.bind(), beta.bind(),
            h0.bind(), y.bind(), hf.bind(), g.bind(),  snap_binding.bind(),
        });
        try m.runPrefillPass(cmds, out_flags, plan, timeout_ns);
    }

    fn runPrefillPass(m: *Gdn, cmds: *Commands, flags: u32, plan: PrefillPlan, timeout_ns: u64) Error!void {
        const base: params.Params = .{
            .dims = plan.tokens,
            .matrix_m = Model.head_dim, // Dk
            .matrix_n = Model.head_dim, // Dv
            .matrix_k = Model.heads, // Hv
            .flags = flags,
            .shape = .{ 0, 0, 0, 0 }, // views bind at their base
        };
        const p: PrefillPush = .{ .base = base, .chunk = plan.chunk, .chunks = plan.chunks };
        const grid_x = Model.heads * 4; // LANES == 4 row blocks per head
        const most = m.dev.properties.limits.maxComputeWorkGroupCount;
        if (grid_x > most[0] or plan.chunks > most[1]) return error.BadGroupCount;
        try m.prefill_bf16.run(cmds, std.mem.asBytes(&p), .{ grid_x, plan.chunks, 1 }, timeout_ns);
    }

    /// Gated-delta decode step (T == 1): raw-gates mode (flags bit8|bit9) —
    /// the kernel computes g and beta from a/b/A_log/dt_bias with the
    /// Metal-oracle rounding chain and f32 A_log, and the kv walk uses the
    /// Metal chain's association. a_log_u16 is the f32 A_log viewed as u16
    /// pairs (32 elements); state flows out of place.
    pub fn decodeRecur(m: *Gdn, cmds: *Commands, soa: View, a: View, b: View, a_log_u16: View, dt: View, state_in: View, state_out: View, y: View, timeout_ns: u64) Error!void {
        try wantBf16(&.{ soa, a, b, dt, y });
        try wantF32(&.{ state_in, state_out });
        if (a_log_u16.kind != .u16) return error.Invalid;
        _ = try decodePlan(soa.elements, a.elements, b.elements, a_log_u16.elements, dt.elements, state_in.elements, state_out.elements, y.elements);
        const q = try soa.slice(0, Model.linear_dim);
        const k = try soa.slice(Model.linear_dim, Model.linear_dim);
        const v = try soa.slice(2 * Model.linear_dim, Model.linear_dim);
        try refuseOverlap(&.{ state_out, y }, &.{ soa, a, b, a_log_u16, dt, state_in });
        try m.decode_bf16.bind(&.{
            q.bind(),        k.bind(), v.bind(),         a.bind(),         b.bind(),
            state_in.bind(), y.bind(), state_out.bind(), a_log_u16.bind(), dt.bind(),
        });
        var p: params.Params = .{
            .matrix_m = Model.head_dim,
            .matrix_n = Model.head_dim,
            .matrix_k = Model.heads,
            .flags = 0x300, // bit8 raw gates, bit9 f32 A_log + Metal kv order
            .shape = .{ 0, 0, 0, 0 },
            .in_strides = .{ 0, 0, Model.heads, 0 }, // [2] = Hv for head % Hv
        };
        const most = m.dev.properties.limits.maxComputeWorkGroupCount;
        if (Model.heads > most[0] or (Model.head_dim / 32) > most[1]) return error.BadGroupCount;
        try m.decode_bf16.run(cmds, std.mem.asBytes(&p), .{ Model.heads, Model.head_dim / 32, 1 }, timeout_ns);
    }

    /// Gated RMSNorm + SiLU epilogue (Ash8c Metal gdn_post): PER-HEAD 128-wide
    /// RMS, not a 2048 global row. out = bf16((gate/(1+exp(-gate))) *
    /// bf16(weight * (value * inv))). y/z/out are [tokens, 16, 128] bf16
    /// (tokens*2048 elements, head-major, exactly the recurrence's y layout);
    /// the weight is [128] bf16, indexed by the column within the head and
    /// repeated for every head; `rows` is the TOKEN count (the dispatch runs
    /// rows*16 one-workgroup-per-head-row). eps the model's norm epsilon.
    pub fn gatedNorm(m: *Gdn, cmds: *Commands, out: View, y: View, z: View, weight: View, rows: u64, eps: f32, timeout_ns: u64) Error!void {
        try wantBf16(&.{ out, y, z, weight });
        const plan = try gatedNormPlan(out.elements, y.elements, z.elements, weight.elements, rows, eps);
        try refuseOverlap(&.{out}, &.{ y, z, weight });
        try m.norm_bf16.bind(&.{ y.bind(), weight.bind(), z.bind(), out.bind() });
        var p: params.Params = .{
            .operation = 0, // gated mode
            .lhs_size = Model.head_dim, // per-column weight within the head (!= 1)
            .reduce_size = Model.head_dim, // the 128-wide per-head row
            .output_size = plan.rows, // tokens * 16 head-rows
        };
        p.alpha = eps;
        const g = try groups(plan.rows, 1, m.dev.properties.limits.maxComputeWorkGroupCount[0]);
        try m.norm_bf16.run(cmds, std.mem.asBytes(&p)[0..norm_push_bytes], g, timeout_ns);
    }
};

/// The gate split's derived geometry, or the named reason it cannot run.
pub const SplitPlan = struct { tokens: u32, conv_channels: u32, section_elements: u32 };

pub fn splitPlan(conv_out_elements: u64, soa_elements: u64, g_elements: u64, beta_elements: u64, a_elements: u64, b_elements: u64, a_log_elements: u64, dt_elements: u64, tokens: u64) Error!SplitPlan {
    if (tokens == 0) return error.Invalid;
    const section = std.math.mul(u64, tokens, Model.linear_dim) catch return error.TooLarge;
    const rows = std.math.mul(u64, tokens, Model.heads) catch return error.TooLarge;
    const soa = std.math.mul(u64, section, 3) catch return error.TooLarge;
    const conv = std.math.mul(u64, tokens, Model.conv_channels) catch return error.TooLarge;
    if (conv_out_elements != conv or soa_elements != soa) return error.Invalid;
    if (g_elements != rows or beta_elements != rows or a_elements != rows or b_elements != rows) return error.Invalid;
    if (a_log_elements != Model.heads or dt_elements != Model.heads) return error.Invalid;
    return .{
        .tokens = try cast32(tokens),
        .conv_channels = try cast32(Model.conv_channels),
        .section_elements = try cast32(section),
    };
}

pub const ConvPrefillPlan = struct { tokens: u32, count: u32, l_in: u32, tokens_elements: u64, state_elements: u64 };

pub fn convPrefillPlan(conv_input_elements: u64, conv_state_in_elements: u64, qkv_elements: u64, weight_elements: u64, conv_state_out_elements: u64, tokens: u64) Error!ConvPrefillPlan {
    if (tokens == 0) return error.Invalid;
    const rows = std.math.mul(u64, tokens, Model.conv_channels) catch return error.TooLarge;
    const input = std.math.add(u64, rows, std.math.mul(u64, Model.conv_state_rows, Model.conv_channels) catch return error.TooLarge) catch return error.TooLarge;
    const state = std.math.mul(u64, Model.conv_state_rows, Model.conv_channels) catch return error.TooLarge;
    if (conv_input_elements != input) return error.Invalid;
    if (conv_state_in_elements != state or conv_state_out_elements != state) return error.Invalid;
    if (qkv_elements != rows) return error.Invalid;
    if (weight_elements != Model.conv_channels * Model.taps) return error.Invalid;
    return .{
        .tokens = try cast32(tokens),
        .count = try cast32(rows),
        .l_in = try cast32(tokens + Model.conv_state_rows),
        .tokens_elements = rows,
        .state_elements = state,
    };
}

pub const ConvDecodePlan = struct { count: u32, conv_channels: u32 };

pub fn convDecodePlan(conv_state_in_elements: u64, x_elements: u64, weight_elements: u64, activated_elements: u64, conv_state_out_elements: u64, soa_elements: u64, g_elements: u64, beta_elements: u64, a_elements: u64, b_elements: u64, a_log_elements: u64, dt_elements: u64) Error!ConvDecodePlan {
    const state = std.math.mul(u64, Model.conv_state_rows, Model.conv_channels) catch return error.TooLarge;
    const channels: u64 = Model.conv_channels;
    if (conv_state_in_elements != state or conv_state_out_elements != state) return error.Invalid;
    if (x_elements != channels or activated_elements != channels) return error.Invalid;
    if (weight_elements != channels * Model.taps) return error.Invalid;
    if (soa_elements != 3 * Model.linear_dim) return error.Invalid;
    if (g_elements != Model.heads or beta_elements != Model.heads) return error.Invalid;
    if (a_elements != Model.heads or b_elements != Model.heads) return error.Invalid;
    if (a_log_elements != Model.heads or dt_elements != Model.heads) return error.Invalid;
    return .{ .count = try cast32(channels), .conv_channels = try cast32(channels) };
}

pub const PrefillPlan = struct { tokens: u32, chunk: u32, chunks: u32, section_elements: u32 };

pub fn prefillPlan(soa_elements: u64, g_elements: u64, beta_elements: u64, h0_elements: u64, y_elements: u64, hf_elements: u64, snap_elements: u64, tokens: u64, chunk: u64) Error!PrefillPlan {
    if (tokens == 0 or chunk == 0) return error.Invalid;
    const section = std.math.mul(u64, tokens, Model.linear_dim) catch return error.TooLarge;
    const soa = std.math.mul(u64, section, 3) catch return error.TooLarge;
    const rows = std.math.mul(u64, tokens, Model.heads) catch return error.TooLarge;
    if (soa_elements != soa or g_elements != rows or beta_elements != rows) return error.Invalid;
    if (h0_elements != Model.state_elements or hf_elements != Model.state_elements) return error.Invalid;
    if (y_elements != section) return error.Invalid;
    const chunks = (tokens + chunk - 1) / chunk;
    const snap = std.math.mul(u64, chunks - 1, Model.state_elements) catch return error.TooLarge;
    if (snap_elements != snap) return error.Invalid;
    return .{
        .tokens = try cast32(tokens),
        .chunk = try cast32(chunk),
        .chunks = try cast32(chunks),
        .section_elements = try cast32(section),
    };
}

pub const DecodePlan = struct {};

pub fn decodePlan(soa_elements: u64, a_elements: u64, b_elements: u64, a_log_u16_elements: u64, dt_elements: u64, state_in_elements: u64, state_out_elements: u64, y_elements: u64) Error!DecodePlan {
    if (soa_elements != 3 * Model.linear_dim) return error.Invalid;
    if (a_elements != Model.heads or b_elements != Model.heads or dt_elements != Model.heads) return error.Invalid;
    // f32 A_log viewed as u16 pairs: heads * 2
    if (a_log_u16_elements != 2 * Model.heads) return error.Invalid;
    if (state_in_elements != Model.state_elements or state_out_elements != Model.state_elements) return error.Invalid;
    if (y_elements != Model.linear_dim) return error.Invalid;
    return .{};
}

pub const GatedNormPlan = struct { rows: u32 };

/// `rows` is the TOKEN count; the plan dispatches rows*16 one-workgroup
/// head-rows of width head_dim=128 (the Ash gdn_post per-head norm). The
/// weight is the [128] per-head row, NOT a [2048] full-row vector.
pub fn gatedNormPlan(out_elements: u64, y_elements: u64, z_elements: u64, weight_elements: u64, rows: u64, eps: f32) Error!GatedNormPlan {
    if (!std.math.isFinite(eps) or eps <= 0) return error.Invalid;
    if (rows == 0) return error.Invalid;
    const total = std.math.mul(u64, rows, Model.linear_dim) catch return error.TooLarge;
    if (out_elements != total or y_elements != total or z_elements != total) return error.Invalid;
    if (weight_elements != Model.head_dim) return error.Invalid;
    const head_rows = std.math.mul(u64, rows, Model.heads) catch return error.TooLarge;
    return .{ .rows = try cast32(head_rows) };
}

fn cast32(v: u64) Error!u32 {
    return std.math.cast(u32, v) orelse error.TooLarge;
}

/// Independent f64 host reference for the gate formula the gdn_split.comp
/// shader and the decode kernel's raw-gates prologue implement. Pure host
/// math for assert checks and review; never a CPU fallback for tensors.
pub const GateF64 = struct { g: f64, beta: f64 };

pub fn gateReferenceF64(a: f64, b: f64, a_log: f64, dt: f64) GateF64 {
    const s = a + dt;
    const sp = @max(s, 0.0) + std.math.log1p(std.math.exp(-@abs(s)));
    return .{
        .g = @exp(-@exp(a_log) * sp),
        .beta = 1.0 / (1.0 + @exp(-b)),
    };
}

const testing = std.testing;

test "wantSplitOutputs requires bf16 soa/beta and f32 g on real View kinds" {
    // fake buffers, exactly like commands.zig's geometry tests: only the
    // views' kinds and extents matter here, no device.
    const dev: *const Device = @ptrFromInt(0x5000);
    const words: usize = 4096;
    const buf = Buffer{ .dev = dev, .handle = @ptrFromInt(0x10), .memory = @ptrFromInt(0x20), .len = words * 4, .kind = .device, .mapped = null, .coherent = false };
    const soa = try View.init(&buf, .bf16, 0, words / 2);
    const g = try View.init(&buf, .f32, 0, words / 4);
    const beta = try View.init(&buf, .bf16, 256, words / 2 - 128);
    try wantSplitOutputs(soa, g, beta);
    // a bf16 g_out — the regression this guards — must refuse
    const g_bf16 = try View.init(&buf, .bf16, 0, words / 2);
    try testing.expectError(error.Invalid, wantSplitOutputs(soa, g_bf16, beta));
    // so does an f32 beta or a u16 soa (u16 has the same extent as bf16, so
    // the refusal is genuinely the kind check)
    const beta_f32 = try View.init(&buf, .f32, 0, words / 4);
    try testing.expectError(error.Invalid, wantSplitOutputs(soa, g, beta_f32));
    const soa_u16 = try View.init(&buf, .u16, 0, words / 2);
    try testing.expectError(error.Invalid, wantSplitOutputs(soa_u16, g, beta));
}

test "gateReferenceF64 behaves like exp(-exp(A_log) softplus) and a sigmoid" {
    const g0 = gateReferenceF64(0, 0, 0, 0);
    try testing.expectApproxEqAbs(@as(f64, @exp(-@exp(0.0) * std.math.log1p(@as(f64, 1.0)))), g0.g, 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 0.5), g0.beta, 1e-12);
    // larger a (larger softplus) decays g; larger a_log decays it harder
    const up = gateReferenceF64(2, 0, 0, 0);
    try testing.expect(g0.g > up.g and up.g > 0 and up.g < 1);
    const loud = gateReferenceF64(2, 0, 2, 0);
    try testing.expect(loud.g < up.g);
    try testing.expect(gateReferenceF64(0, 4, 0, 0).beta > g0.beta);
    try testing.expect(gateReferenceF64(-40, 0, 0, 0).g > 0.999); // softplus saturates to 0
}

test "splitPlan derives the geometry and refuses wrong shapes and overflow" {
    const t: u64 = 7;
    const section = t * Model.linear_dim;
    try testing.expectEqual(@as(u32, 7), (try splitPlan(t * Model.conv_channels, 3 * section, t * Model.heads, t * Model.heads, t * Model.heads, t * Model.heads, Model.heads, Model.heads, t)).tokens);
    try testing.expectError(error.Invalid, splitPlan(t * Model.conv_channels + 1, 3 * section, t * Model.heads, t * Model.heads, t * Model.heads, t * Model.heads, Model.heads, Model.heads, t));
    try testing.expectError(error.Invalid, splitPlan(t * Model.conv_channels, 3 * section, t * Model.heads, t * Model.heads, t * Model.heads, t * Model.heads, Model.heads + 1, Model.heads, t));
    try testing.expectError(error.Invalid, splitPlan(t * Model.conv_channels, 3 * section, t * Model.heads, t * Model.heads, t * Model.heads, t * Model.heads, Model.heads, Model.heads, 0));
    // a CONSISTENT geometry whose section passes u32: tokens big enough that
    // cast32(section) overflows (tokens = 2^44 -> section = 2^55, soa 3*2^55,
    // conv rows 6144*2^44 all fit u64; the u32 casts fail with TooLarge).
    const big: u64 = 1 << 44;
    const big_section = big * Model.linear_dim;
    try testing.expectError(error.TooLarge, splitPlan(big * Model.conv_channels, 3 * big_section, big * Model.heads, big * Model.heads, big * Model.heads, big * Model.heads, Model.heads, Model.heads, big));
}

test "convPrefillPlan checks the conv input, state and weight geometry" {
    const t: u64 = 5;
    const ch: u64 = Model.conv_channels;
    const state = Model.conv_state_rows * ch;
    try testing.expectEqual(@as(u32, 8), (try convPrefillPlan((t + 3) * ch, state, t * ch, ch * 4, state, t)).l_in);
    try testing.expectError(error.Invalid, convPrefillPlan((t + 2) * ch, state, t * ch, ch * 4, state, t));
    try testing.expectError(error.Invalid, convPrefillPlan((t + 3) * ch, state, t * ch, ch * 5, state, t));
    try testing.expectError(error.Invalid, convPrefillPlan((t + 3) * ch, state + ch, t * ch, ch * 4, state, t));
}

test "convDecodePlan refuses any torn decode geometry" {
    const ch: u64 = Model.conv_channels;
    const state = Model.conv_state_rows * ch;
    const heads: u64 = Model.heads;
    const ok = try convDecodePlan(state, ch, ch * 4, ch, state, 3 * Model.linear_dim, heads, heads, heads, heads, heads, heads);
    try testing.expect(ok.count == ch);
    try testing.expectError(error.Invalid, convDecodePlan(state, ch + 2, ch * 4, ch, state, 3 * Model.linear_dim, heads, heads, heads, heads, heads, heads));
    try testing.expectError(error.Invalid, convDecodePlan(state, ch, ch * 4, ch, state, 3 * Model.linear_dim, heads, heads, heads, heads, heads, heads + 1));
}

test "prefillPlan derives chunks and the snapshot size, refusing snap mismatch" {
    const t: u64 = 130;
    const chunk: u64 = 64;
    const section = t * Model.linear_dim;
    const state: u64 = Model.state_elements;
    // chunks = 3 boundaries between them, and the shader stores one snapshot
    // per INTERIOR boundary (chunk c restores slot c-1): (chunks-1)*state.
    const plan = try prefillPlan(3 * section, t * Model.heads, t * Model.heads, state, section, state, 2 * state, t, chunk);
    try testing.expectEqual(@as(u32, 3), plan.chunks);
    try testing.expectEqual(@as(u32, 64), plan.chunk);
    try testing.expectError(error.Invalid, prefillPlan(3 * section, t * Model.heads, t * Model.heads, state, section, state, 3 * state, t, chunk)); // too many slots
    try testing.expectError(error.Invalid, prefillPlan(3 * section, t * Model.heads, t * Model.heads, state, section, state, 1 * state, t, chunk)); // too few
    try testing.expectError(error.Invalid, prefillPlan(3 * section, t * Model.heads, t * Model.heads, state, section, state, 0, t, chunk)); // chunks>1 needs snap
    try testing.expectError(error.Invalid, prefillPlan(3 * section, t * Model.heads, t * Model.heads, state, section, state, 0, t, 0));
    const single = try prefillPlan(3 * section, t * Model.heads, t * Model.heads, state, section, state, 0, t, t);
    try testing.expectEqual(@as(u32, 1), single.chunks);
}

test "decodePlan checks the u16 A_log window and the fixed decode shapes" {
    const heads: u64 = Model.heads;
    _ = try decodePlan(3 * Model.linear_dim, heads, heads, 2 * heads, heads, Model.state_elements, Model.state_elements, Model.linear_dim);
    try testing.expectError(error.Invalid, decodePlan(3 * Model.linear_dim, heads, heads, heads, heads, Model.state_elements, Model.state_elements, Model.linear_dim));
    try testing.expectError(error.Invalid, decodePlan(3 * Model.linear_dim, heads, heads, 2 * heads, heads, Model.state_elements + 4, Model.state_elements, Model.linear_dim));
}

test "gatedNormPlan is the per-head 128 norm and refuses torn geometry" {
    const rows: u64 = 3;
    const total = rows * Model.linear_dim;
    // rows = tokens; the plan dispatches tokens*16 head-rows, weight [128]
    try testing.expectEqual(@as(u32, 3 * Model.heads), (try gatedNormPlan(total, total, total, Model.head_dim, rows, 1e-6)).rows);
    // the old 2048 global-row weight is REFUSED
    try testing.expectError(error.Invalid, gatedNormPlan(total, total, total, Model.linear_dim, rows, 1e-6));
    try testing.expectError(error.Invalid, gatedNormPlan(total + 2, total, total, Model.head_dim, rows, 1e-6));
    try testing.expectError(error.Invalid, gatedNormPlan(total, total, total, Model.head_dim, rows, 0));
    try testing.expectError(error.Invalid, gatedNormPlan(total, total, total, Model.head_dim, 0, 1e-6));
}

test "the fixed model constants agree with the Metal kernel constexprs" {
    try testing.expectEqual(@as(usize, 16), Model.heads);
    try testing.expectEqual(@as(usize, 128), Model.head_dim);
    try testing.expectEqual(@as(usize, 6144), Model.conv_channels);
    try testing.expectEqual(@as(usize, 4), Model.taps);
    try testing.expectEqual(@as(usize, 18), Model.linear_layers);
    try testing.expectEqual(@as(usize, 262144), Model.state_elements);
}

test "single-chunk prefill read and written snapshot dummies are disjoint" {
    var m: Gdn = undefined;
    m.dummy = .{ .dev = undefined, .handle = null, .memory = null, .len = 2 * dummy_bytes, .kind = .device, .mapped = null, .coherent = false };
    const read = try m.dummyView(.bf16);
    const snapshot = try m.snapshotDummyView();
    try testing.expect(read.buffer == snapshot.buffer);
    try testing.expect(!read.overlaps(snapshot));
    try testing.expectEqual(@as(u64, 0), read.byte_offset);
    try testing.expectEqual(dummy_bytes, snapshot.byte_offset);
    try testing.expectEqual(dummy_bytes, read.byteLen());
    try testing.expectEqual(dummy_bytes, snapshot.byteLen());
}
