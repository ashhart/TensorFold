//! Prompt prefixes kept on CUDA (train_prefix.zig on Metal): each Mamba layer's conv and scan state, keys and values.
const std = @import("std");
const cuda = @import("cuda");
const Config = @import("config.zig").Config;

/// Prefix slots a trainer holds at most.
pub const max_slots = 64;

pub const Store = struct {
    pub const most = max_slots;
    d: *const cuda.Driver,
    slots: std.ArrayList(cuda.DeviceBuffer) = .empty,
    conv: usize, // a Mamba layer's conv state: its last three inputs, bf16 [3, conv]
    ssm: usize, // its scan state, f32 [heads, head dim, state]
    kv: usize, // an attention layer's keys (or values) for up to `rows` rows, bf16 [rows, kv heads x head dim]
    mambas: usize,
    attentions: usize,

    pub fn init(d: *const cuda.Driver, c: Config, rows: usize) Store {
        return .{ .d = d, .conv = 3 * c.convDim() * 2, .ssm = c.mamba_heads * c.mamba_head_dim * c.state * 4, .kv = rows * c.kv_heads * c.head_dim * 2, .mambas = c.count(.mamba), .attentions = c.count(.attention) };
    }

    pub fn deinit(s: *Store, gpa: std.mem.Allocator) void {
        for (s.slots.items) |*b| b.free();
        s.slots.deinit(gpa);
    }

    pub fn bytes(s: *const Store) usize {
        return s.mambas * (align256(s.conv) + align256(s.ssm)) + s.attentions * 2 * align256(s.kv);
    }

    /// Every slot freed.
    pub fn drop(s: *Store, gpa: std.mem.Allocator) void {
        for (s.slots.items) |*b| b.free();
        s.slots.clearAndFree(gpa);
    }

    /// Room for `n` slots (at most max_slots), made on first use.
    pub fn ensure(s: *Store, gpa: std.mem.Allocator, n: usize) !void {
        if (n > max_slots) return error.TooManyPrefixes;
        while (s.slots.items.len < n) {
            var b = try cuda.DeviceBuffer.alloc(s.d, s.bytes());
            errdefer b.free();
            try s.slots.append(gpa, b);
        }
    }

    /// Slot `slot`'s conv state of Mamba layer j (its index among the Mamba layers).
    pub fn convAt(s: *const Store, slot: u32, j: usize) u64 {
        return s.slots.items[slot].ptr + j * (align256(s.conv) + align256(s.ssm));
    }

    /// Slot `slot`'s scan state of Mamba layer j.
    pub fn ssmAt(s: *const Store, slot: u32, j: usize) u64 {
        return s.convAt(slot, j) + align256(s.conv);
    }

    /// Slot `slot`'s keys of attention layer j (its index among the attention layers), or its values.
    pub fn keysAt(s: *const Store, slot: u32, j: usize, values: bool) u64 {
        const base = s.mambas * (align256(s.conv) + align256(s.ssm));
        return s.slots.items[slot].ptr + base + (2 * j + @intFromBool(values)) * align256(s.kv);
    }
};

fn align256(n: usize) usize {
    return std.mem.alignForward(usize, n, 256);
}
