//! Prompt prefixes kept on Metal: each Mamba layer's conv and scan state, each attention layer's keys and values.
const std = @import("std");
const mtl = @import("metal");
const cfg = @import("config.zig");
const At = @import("prefill_launch.zig").At;

const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;

/// Prefix slots a trainer holds at most.
pub const max_slots = 64;

pub const Store = struct {
    pub const most = max_slots;
    device: mtl.Device,
    slots: std.ArrayList(mtl.Buffer) = .empty,
    conv: usize, // a Mamba layer's conv state: its last three inputs, bf16 [3, conv]
    ssm: usize, // its scan state, f32 [heads, head dim, state]
    kv: usize, // an attention layer's keys (or values) for up to `rows` rows, bf16 [rows, kv heads x head dim]
    mambas: usize,
    attentions: usize,

    pub fn init(device: mtl.Device, c: cfg.Config, rows: usize) Store {
        return .{ .device = device, .conv = 3 * c.convDim() * 2, .ssm = c.mamba_heads * c.mamba_head_dim * c.state * 4, .kv = rows * c.kv_heads * c.head_dim * 2, .mambas = c.count(.mamba), .attentions = c.count(.attention) };
    }

    pub fn deinit(s: *Store, gpa: std.mem.Allocator) void {
        for (s.slots.items) |b| b.deinit();
        s.slots.deinit(gpa);
    }

    fn bytes(s: *const Store) usize {
        return s.mambas * (align256(s.conv) + align256(s.ssm)) + s.attentions * 2 * align256(s.kv);
    }

    /// Every slot freed.
    pub fn drop(s: *Store, gpa: std.mem.Allocator) void {
        for (s.slots.items) |b| b.deinit();
        s.slots.clearAndFree(gpa);
    }

    /// Room for `n` slots (at most max_slots), made on first use.
    pub fn ensure(s: *Store, gpa: std.mem.Allocator, n: usize) !void {
        if (n > max_slots) return error.TooManyPrefixes;
        while (s.slots.items.len < n) {
            const b = try s.device.buffer(s.bytes(), opts);
            errdefer b.deinit();
            try s.slots.append(gpa, b);
        }
    }

    /// Slot `slot`'s conv state of Mamba layer j (its index among the Mamba layers).
    pub fn convAt(s: *const Store, slot: u32, j: usize) At {
        return At.of(s.slots.items[slot]).plus(j * (align256(s.conv) + align256(s.ssm)));
    }

    /// Slot `slot`'s scan state of Mamba layer j.
    pub fn ssmAt(s: *const Store, slot: u32, j: usize) At {
        return s.convAt(slot, j).plus(align256(s.conv));
    }

    /// Slot `slot`'s keys of attention layer j (its index among the attention layers), then its values.
    pub fn keysAt(s: *const Store, slot: u32, j: usize, values: bool) At {
        const base = s.mambas * (align256(s.conv) + align256(s.ssm));
        return At.of(s.slots.items[slot]).plus(base + (2 * j + @intFromBool(values)) * align256(s.kv));
    }
};

fn align256(n: usize) usize {
    return std.mem.alignForward(usize, n, 256);
}
