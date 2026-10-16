//! What a family hands Living Weights: where a pocket sits, how its neurons fire, how its shards hold them.
const std = @import("std");

/// What a pocket neuron does with its input x: relu(u x)^2 (Nemotron), or silu(g x) (w x) (a SwiGLU shared expert).
pub const Activation = enum { relu2, swiglu };

/// A pocket neuron's kind: its activation, whether a per-row gate scales its expert's output, and its input fit.
pub const Neuron = struct {
    act: Activation,
    /// The shared expert's output is scaled by a gate the trainer reads at each row, so a neuron adds gate act(x) v.
    gated: bool = false,
    /// The input fit's margin: rows a neuron stays silent on are fitted to u x at or below -margin.
    margin: f32 = 0.25,
    /// SwiGLU: a neuron fitted as u is written as gate row sharp * u and up row u / sharp (a power of two keeps u).
    sharp: f32 = 16,

    /// Input projections a neuron has a row in: one (up) or two (gate, then up).
    pub fn inputs(n: Neuron) usize {
        return switch (n.act) {
            .relu2 => 1,
            .swiglu => 2,
        };
    }
};

/// The shared expert's storage: MLX affine, `bits` a code in u32 words, a bf16 scale and bias every `group`.
pub const Quant = struct { bits: u8 = 4, group: usize = 64 };

/// The MTP head's shared expert, widened with zero neurons when the model's are, so its tensors stay one width.
pub const Mtp = struct {
    file: ?[]const u8 = null, // its own safetensors file in the folder; null: in the model's index beside the rest
    base: []const u8, // its shared expert's tensors' prefix, ending in "."
};

/// Tensor names: layer i's shared expert is `<layers><i>.<expert>`, then `<projection>.weight|scales|biases`.
pub const Names = struct {
    layers: []const u8,
    expert: []const u8,
    inputs: []const []const u8, // the input projections in Neuron.inputs order: a row a neuron each
    output: []const u8 = "down_proj", // the output projection: a column a neuron
    mtp: ?Mtp = null,
};

/// config.json keys: where the model's own keys sit, the shared expert's width, the pocket's width once written.
pub const Keys = struct {
    scope: ?[]const u8 = null, // the object holding the model's keys (Flash Next: "text_config"); null: the top level
    width: []const u8, // the shared expert's width, widened by the pocket on its first save
    pocket: []const u8 = "tensorfold_learned_neurons",
    layers: []const u8 = "num_hidden_layers",
    hidden: []const u8 = "hidden_size",
};

/// Which layers have a shared expert: those a layer pattern marks MoE, or every layer.
pub const Experts = enum { pattern, every };

pub const Family = struct {
    model_type: []const u8,
    names: Names,
    neuron: Neuron,
    quant: Quant,
    keys: Keys,
    experts: Experts,
    /// Pocket neurons at each layer: a whole number of quantization groups.
    neurons: usize = 64,

    /// Layer `layer`'s shared expert's tensor prefix, into `buf`.
    pub fn expertBase(f: Family, buf: []u8, layer: usize) ![]const u8 {
        return std.fmt.bufPrint(buf, "{s}{d}.{s}", .{ f.names.layers, layer, f.names.expert });
    }

    /// The layer a tensor name's shared expert sits at (null: not one of a layer's shared expert's tensors).
    pub fn expertLayer(f: Family, name: []const u8) ?usize {
        if (!std.mem.startsWith(u8, name, f.names.layers)) return null;
        const rest = name[f.names.layers.len..];
        const dot = std.mem.indexOfScalar(u8, rest, '.') orelse return null;
        if (!std.mem.startsWith(u8, rest[dot + 1 ..], f.names.expert)) return null;
        return std.fmt.parseInt(usize, rest[0..dot], 10) catch null;
    }

    /// Whether the description holds together: a neuron's rows match its inputs, the pocket a whole number of groups.
    pub fn check(f: Family) !void {
        if (f.names.inputs.len != f.neuron.inputs()) return error.BadFamily;
        if (32 % f.quant.bits != 0 or f.quant.group == 0 or f.neurons == 0 or f.neurons % f.quant.group != 0) return error.BadFamily;
        if (f.neuron.act == .swiglu and !(f.neuron.sharp > 0 and std.math.isPowerOfTwo(@as(u32, @intFromFloat(f.neuron.sharp))))) return error.BadFamily;
    }
};

pub const max_layers = 128;
pub const Layers = std.bit_set.Static(max_layers);

/// The served checkpoint as Living Weights reads it, built by the family (from its config) for its host.
pub const Layout = struct {
    hidden: usize,
    width: usize, // the shared expert's own neurons, a pocket written earlier left out
    layers: usize,
    widen: Layers = .empty, // layers whose shared expert carries the pocket in the shards (each with an expert)
    learn: Layers = .empty, // of those, the ones lessons put neurons in

    /// The layers set in `which`, in order, into `out`.
    pub fn list(lay: Layout, which: Layers, out: []usize) []usize {
        var n: usize = 0;
        for (0..lay.layers) |l| if (which.isSet(l)) {
            out[n] = l;
            n += 1;
        };
        return out[0..n];
    }

    /// Every layer `has` names carries one widened; all but the last learn (a nudge after it has no gradient to read).
    pub fn fromExperts(hidden: usize, width: usize, layers: usize, has: Layers) Layout {
        var lay: Layout = .{ .hidden = hidden, .width = width, .layers = layers, .widen = has, .learn = has };
        if (layers > 0) lay.learn.unset(layers - 1);
        return lay;
    }
};

test "a family finds its shared expert's layer in a tensor name and builds its prefix" {
    const f = @import("test_families.zig").swiglu;
    try f.check();
    var buf: [128]u8 = undefined;
    try std.testing.expectEqualStrings("language_model.model.layers.7.mlp.shared_expert.", try f.expertBase(&buf, 7));
    try std.testing.expectEqual(@as(?usize, 7), f.expertLayer("language_model.model.layers.7.mlp.shared_expert.up_proj.scales"));
    try std.testing.expectEqual(@as(?usize, null), f.expertLayer("language_model.model.layers.7.mlp.shared_expert_gate.scales"));
    try std.testing.expectEqual(@as(?usize, null), f.expertLayer("language_model.mtp.layers.0.mlp.shared_expert.up_proj.scales"));
    const n = @import("families.zig").nemotron_h;
    try n.check();
    try std.testing.expectEqual(@as(?usize, 13), n.expertLayer("backbone.layers.13.mixer.shared_experts.down_proj.weight"));
    var has: Layers = .empty;
    has.set(1);
    has.set(3);
    const lay = Layout.fromExperts(8, 16, 4, has);
    var out: [4]usize = undefined;
    try std.testing.expectEqualSlices(usize, &.{ 1, 3 }, lay.list(lay.widen, &out));
    try std.testing.expectEqualSlices(usize, &.{1}, lay.list(lay.learn, &out));
}
