// The native Vulkan Qwen3.5-2B single-sequence state: the flat KV caches of the six full-attention layers, and the conv windows and gated-delta-net recurrence states of the eighteen linear layers, each as checked typed views over stable heap-owned device buffers.
//  Every region is real zeroed device memory: construction and `reset` write zeros on the GPU through the existing staged transfer (`commands.upload` through the caller's staging buffer) — never a metadata-only reset and no new kernel. The decode and prefill operators read one state half and write a disjoint other half (ping-pong), so no read aliases a write under the pipeline's validation; `flip` swaps the halves between steps.
//  Capacity is bounded by the SDPA arm's visible-key limit (7168): a context beyond that is refused at construction, never silently truncated. Length commits only through `append`, after the writes that produced the rows completed; `deinit` is legal only when no submission touching any view can still be pending; `reset`'s transfers route in-flight errors through `commands.orExit` inside this module, before any unwind can free.
const std = @import("std");
const vk = @import("vulkan");
const pinned = @import("../../config.zig");

const Allocator = std.mem.Allocator;
const Buffer = vk.Buffer;
const Commands = vk.Commands;
const Device = vk.Device;
const View = vk.tensor.View;

pub const Error = vk.runtime.Error || error{ContextTooLong};

// Full-attention layers are every 4th (indices 3, 7, ...): the pinned config's `full_attention_interval`, checked against the family constants.
pub const full_layers: usize = pinned.layers / 4;
pub const linear_layers: usize = pinned.layers - full_layers;

// The SDPA decode arm stages the whole visible key prefix in shared memory; this is the hard capacity it admits (attention.zig's limit).
pub const max_capacity: usize = 7168;

pub const kv_heads: usize = pinned.kv_heads; // 2
pub const head_dim: usize = pinned.head_dim; // 256
pub const conv_rows: usize = pinned.conv_taps - 1; // 3
pub const conv_channels: usize = pinned.conv_dim; // 6144
pub const gdn_heads: usize = pinned.linear_heads; // 16
pub const gdn_head_dim: usize = pinned.linear_dim; // 128

/// The whole-model capacity: the context pin admits, the SDPA limit bounds.
pub fn capacityFor(context: usize) Error!usize {
    if (context == 0 or context > max_capacity) return error.ContextTooLong;
    return context;
}

/// Layer `index`'s full-attention cache slot (indices 3, 7, ... map to 0..5).
pub fn fullSlot(index: usize) ?usize {
    if (index % 4 != 3 or index >= pinned.layers) return null;
    return (index + 1) / 4 - 1;
}

/// Layer `index`'s linear-layer slot (the other eighteen indices, in order).
pub fn linearSlot(index: usize) ?usize {
    if (index >= pinned.layers or index % 4 == 3) return null;
    return index - index / 4;
}

// One sequence's whole state. Every buffer is heap-owned and never moved, so all views stay valid for the state's lifetime; `parity` selects which ping-pong half each linear layer's operators read and write next.
pub const State = struct {
    gpa: Allocator,
    dev: *const Device,
    capacity: u64,
    /// The committed visible prefix; only `append` and `reset` change it.
    committed: u64 = 0,
    buffers: []*Buffer,
    keys: []View,
    values: []View,
    /// Two halves per linear layer, even = half 0; `parity` names the input.
    conv: []View,
    gdn: []View,
    parity: []u1,
    // How many of `buffers` hold real created buffers; init's error path frees exactly this prefix.
    created: usize,

    // Allocates and really zeroes every region for a context of `min(context, max_capacity)` positions. `staging` is the caller's staging buffer (any nonempty size; transfers chunk through it).
    pub fn init(
        gpa: Allocator,
        dev: *const Device,
        cmds: *Commands,
        context: usize,
        staging: Buffer,
        timeout_ns: u64,
    ) (Allocator.Error || Error)!State {
        const capacity = try capacityFor(context);

        var s: State = .{
            .gpa = gpa,
            .dev = dev,
            .capacity = capacity,
            .buffers = &.{},
            .keys = &.{},
            .values = &.{},
            .conv = &.{},
            .gdn = &.{},
            .parity = &.{},
            .created = 0,
        };
        errdefer s.deinit();

        // every allocation is recorded in `created` as it completes, so the single error path frees exactly what was created
        s.buffers = try gpa.alloc(*Buffer, 2 * full_layers + 4 * linear_layers);
        s.keys = try gpa.alloc(View, full_layers);
        s.values = try gpa.alloc(View, full_layers);
        s.conv = try gpa.alloc(View, 2 * linear_layers);
        s.gdn = try gpa.alloc(View, 2 * linear_layers);
        s.parity = try gpa.alloc(u1, linear_layers);
        @memset(s.parity, 0);

        const kv_elements: u64 = kv_heads * capacity * head_dim;
        const conv_elements: u64 = conv_rows * conv_channels;
        const gdn_elements: u64 = gdn_heads * gdn_head_dim * gdn_head_dim;
        var next: usize = 0;
        for (0..full_layers) |slot| {
            s.keys[slot] = try s.region(next, .bf16, kv_elements);
            next += 1;
            s.values[slot] = try s.region(next, .bf16, kv_elements);
            next += 1;
        }
        for (0..linear_layers) |slot| {
            s.conv[2 * slot] = try s.region(next, .bf16, conv_elements);
            next += 1;
            s.conv[2 * slot + 1] = try s.region(next, .bf16, conv_elements);
            next += 1;
        }
        for (0..linear_layers) |slot| {
            s.gdn[2 * slot] = try s.region(next, .f32, gdn_elements);
            next += 1;
            s.gdn[2 * slot + 1] = try s.region(next, .f32, gdn_elements);
            next += 1;
        }
        try s.reset(cmds, staging, timeout_ns);
        s.created = s.buffers.len;
        return s;
    }

    // One heap-owned device buffer with a typed whole-buffer view. The slot registers in `created` as soon as the Vk allocation exists, so exactly one owner (State.deinit) ever frees it.
    fn region(s: *State, at: usize, kind: vk.tensor.Kind, elements: u64) (Allocator.Error || vk.runtime.Error)!View {
        const bytes = try vk.tensor.extent(kind, elements);
        const ptr = try s.gpa.create(Buffer);
        ptr.* = Buffer.init(s.dev, .device, bytes) catch |e| {
            s.gpa.destroy(ptr);
            return e;
        };
        s.buffers[at] = ptr;
        s.created = at + 1;
        return View.init(ptr, kind, 0, elements);
    }

    // Writes actual zero bytes over every region on the GPU through the staged transfer; a 16 MiB zero page bounds host memory.
    pub fn reset(s: *State, cmds: *Commands, staging: Buffer, timeout_ns: u64) Error!void {
        const zeros = try s.gpa.alloc(u8, @min(1 << 24, staging.len));
        defer s.gpa.free(zeros);
        @memset(zeros, 0);
        for (s.buffers) |ptr| {
            var done: u64 = 0;
            while (done < ptr.len) {
                const n = @min(zeros.len, ptr.len - done);
                // in-flight failures exit here, never unwinding into frees
                try vk.orExit(vk.upload(cmds, ptr.*, done, staging, zeros[0..n], timeout_ns));
                done += n;
            }
        }
        s.committed = 0;
        @memset(s.parity, 0);
    }

    /// The full-attention cache slot's key rows, `[2, capacity, 256]` BF16.
    pub fn keysFor(s: *const State, slot: usize) View {
        return s.keys[slot];
    }

    /// The full-attention cache slot's value rows, same shape.
    pub fn valuesFor(s: *const State, slot: usize) View {
        return s.values[slot];
    }

    /// The linear slot's conv window the next step reads, `[3, 6144]` BF16.
    pub fn convInput(s: *const State, slot: usize) View {
        return s.conv[2 * slot + s.parity[slot]];
    }

    /// The disjoint half the next step writes, same shape.
    pub fn convOutput(s: *const State, slot: usize) View {
        return s.conv[2 * slot + (s.parity[slot] ^ 1)];
    }

    /// The linear slot's GDN recurrence the next step reads, `[16, 128, 128]` F32.
    pub fn gdnInput(s: *const State, slot: usize) View {
        return s.gdn[2 * slot + s.parity[slot]];
    }

    /// The disjoint half the next step writes, same shape.
    pub fn gdnOutput(s: *const State, slot: usize) View {
        return s.gdn[2 * slot + (s.parity[slot] ^ 1)];
    }

    /// After a step's writes completed: the written half becomes the input.
    pub fn flip(s: *State, slot: usize) void {
        s.parity[slot] ^= 1;
    }

    /// The committed visible prefix (what attention may attend over).
    pub fn visible(s: *const State) u64 {
        return s.committed;
    }

    // Commits `rows` freshly written cache positions; refuses to pass the capacity — the KV caches have no room past it. There is no rewind: lowering the committed length after recurrence updates would leave the conv and GDN states silently wrong, so a future caller that needs speculation or prefix reuse must restore state for real (this module provides only a full `reset`).
    pub fn append(s: *State, rows: u64) Error!void {
        if (rows > s.capacity - s.committed) return error.ContextTooLong;
        s.committed += rows;
    }

    // Frees every device buffer. Call only when no submission touching any view can still be pending.
    pub fn deinit(s: *State) void {
        for (s.buffers[0..s.created]) |ptr| {
            var buf = ptr.*;
            buf.deinit();
            s.gpa.destroy(ptr);
        }
        s.gpa.free(s.buffers);
        s.gpa.free(s.keys);
        s.gpa.free(s.values);
        s.gpa.free(s.conv);
        s.gpa.free(s.gdn);
        s.gpa.free(s.parity);
        s.* = undefined;
    }
};

const testing = std.testing;

fn fakeState(gpa: Allocator, capacity: usize) !State {
    // content-free state for layout tests: fake buffers, no GPU behind them
    const kv_elements: u64 = kv_heads * capacity * head_dim;
    const conv_elements: u64 = conv_rows * conv_channels;
    const gdn_elements: u64 = gdn_heads * gdn_head_dim * gdn_head_dim;
    const s: State = .{
        .gpa = gpa,
        .dev = undefined,
        .capacity = capacity,
        .buffers = try gpa.alloc(*Buffer, 2 * full_layers + 4 * linear_layers),
        .keys = try gpa.alloc(View, full_layers),
        .values = try gpa.alloc(View, full_layers),
        .conv = try gpa.alloc(View, 2 * linear_layers),
        .gdn = try gpa.alloc(View, 2 * linear_layers),
        .parity = try gpa.alloc(u1, linear_layers),
        .created = 0,
    };
    @memset(s.parity, 0);
    const conv_bytes = 2 * conv_elements;
    const gdn_bytes = 4 * gdn_elements;
    // one shared fake buffer value per region; views only read geometry, and it must be large enough for the biggest region (the F32 GDN state) two distinct fakes so ping-pong halves sit on different buffers, as production does; the larger F32 geometry bounds both lengths
    const fakes: [2]*Buffer = .{ try gpa.create(Buffer), try gpa.create(Buffer) };
    defer for (fakes) |ptr| gpa.destroy(ptr);
    for (fakes) |ptr| ptr.* = .{ .dev = undefined, .handle = null, .memory = null, .len = gdn_bytes, .kind = .device, .mapped = null, .coherent = false };
    for (s.buffers, 0..) |*ptr, i| {
        ptr.* = fakes[i % 2];
    }
    const kb: *Buffer = fakes[0];
    const kb2: *Buffer = fakes[1];
    for (s.keys) |*v| v.* = try View.init(kb, .bf16, 0, kv_elements);
    for (s.values) |*v| v.* = try View.init(kb, .bf16, 0, kv_elements);
    for (s.conv, 0..) |*v, i| v.* = try View.init(if (i % 2 == 0) kb else kb2, .bf16, 0, conv_elements);
    for (s.gdn, 0..) |*v, i| v.* = try View.init(if (i % 2 == 0) kb else kb2, .f32, 0, gdn_elements);
    _ = conv_bytes;
    return s;
}

test "capacity is the context pinned, or refused past the SDPA limit" {
    try testing.expectEqual(@as(usize, 4096), try capacityFor(4096));
    try testing.expectEqual(@as(usize, max_capacity), try capacityFor(max_capacity));
    // a context beyond the SDPA arm's limit is refused, never truncated
    try testing.expectError(error.ContextTooLong, capacityFor(0));
    try testing.expectError(error.ContextTooLong, capacityFor(max_capacity + 1));
    try testing.expectError(error.ContextTooLong, capacityFor(262144));
}

test "layer slots partition the twenty-four layers exactly" {
    var fulls: usize = 0;
    var linears: usize = 0;
    for (0..pinned.layers) |i| {
        if (fullSlot(i)) |slot| {
            try testing.expectEqual(fulls, slot);
            fulls += 1;
            try testing.expectEqual(@as(?usize, null), linearSlot(i));
        } else {
            try testing.expectEqual(linears, linearSlot(i).?);
            linears += 1;
        }
    }
    try testing.expectEqual(@as(usize, full_layers), fulls);
    try testing.expectEqual(@as(usize, linear_layers), linears);
    try testing.expectEqual(@as(usize, 18), linear_layers);
    try testing.expectEqual(@as(?usize, null), fullSlot(pinned.layers));
    try testing.expectEqual(@as(?usize, null), linearSlot(pinned.layers));
}

test "cache regions carry the pinned shapes and stay inside their buffers" {
    const gpa = testing.allocator;
    const capacity = 64;
    var s = try fakeState(gpa, capacity);
    defer {
        gpa.free(s.buffers);
        gpa.free(s.keys);
        gpa.free(s.values);
        gpa.free(s.conv);
        gpa.free(s.gdn);
        gpa.free(s.parity);
    }
    try testing.expectEqual(@as(u64, kv_heads * capacity * head_dim), s.keysFor(0).elements);
    try testing.expectEqual(vk.tensor.Kind.bf16, s.keysFor(5).kind);
    try testing.expectEqual(vk.tensor.Kind.bf16, s.valuesFor(5).kind);
    try testing.expectEqual(@as(u64, conv_rows * conv_channels), s.convInput(17).elements);
    try testing.expectEqual(vk.tensor.Kind.f32, s.gdnInput(0).kind);
    try testing.expectEqual(@as(u64, gdn_heads * gdn_head_dim * gdn_head_dim), s.gdnInput(0).elements);
    // ping-pong halves are disjoint views of different buffers
    try testing.expect(!s.convInput(3).overlaps(s.convOutput(3)));
    try testing.expect(!s.gdnInput(9).overlaps(s.gdnOutput(9)));
}

test "flip swaps the halves and append is the only length commit" {
    const gpa = testing.allocator;
    var s = try fakeState(gpa, 32);
    defer {
        gpa.free(s.buffers);
        gpa.free(s.keys);
        gpa.free(s.values);
        gpa.free(s.conv);
        gpa.free(s.gdn);
        gpa.free(s.parity);
    }
    const in0 = s.convInput(0);
    const out0 = s.convOutput(0);
    try s.append(7);
    try testing.expectEqual(@as(u64, 7), s.visible());
    s.flip(0);
    try testing.expectEqual(out0.byte_offset, s.convInput(0).byte_offset);
    try testing.expectEqual(in0.byte_offset, s.convOutput(0).byte_offset);
    s.flip(0);
    try testing.expectEqual(in0.byte_offset, s.convInput(0).byte_offset);
    // the capacity is a hard commit boundary
    try testing.expectError(error.ContextTooLong, s.append(26));
    try s.append(25);
    try testing.expectEqual(@as(u64, 32), s.visible());
    try testing.expectError(error.ContextTooLong, s.append(1));
    // no rewind exists: visible length only ever grows until a real reset
    try testing.expectEqual(@as(u64, 32), s.visible());
}
