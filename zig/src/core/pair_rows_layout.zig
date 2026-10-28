//! The seam window's regions and the bf16 frame ABI, shared by the link and the host-copy Loopback.
const std = @import("std");
const mtl = @import("metal");
const Fence = @import("fence.zig").Fence;

pub const Ref = @import("exlr8_projection.zig").Ref;

pub const PAGE = 16384;
pub fn pages(n: usize) usize {
    return (n + PAGE - 1) / PAGE * PAGE;
}
/// A command's words at most: a 262,144-token prompt and its head.
pub const command_words = 262144 + 64;
pub const CMD_BYTES = pages(8 + 4 * command_words);
pub const JOBS = 4096;

/// Which parity slot x uses: rank r cannot finish x without the peer's x, so the peer is at most one exchange ahead.
pub const Slot = enum(u1) { a = 0, b = 1 };

pub fn slotOf(x: u32) Slot {
    return @fromBackingInt(@intCast(x % 2));
}

/// The window on both Macs: parts in/out by parity (a page before each out), commands, flags, host words.
pub const Layout = struct {
    slot: usize,

    pub fn recv(l: Layout, x: u32) usize {
        return @as(usize, @backingInt(slotOf(x))) * l.slot;
    }
    pub fn send(l: Layout, x: u32) usize {
        return 2 * l.slot + @as(usize, @backingInt(slotOf(x))) * (PAGE + l.slot) + PAGE;
    }
    pub fn cmdIn(l: Layout) usize {
        return 4 * l.slot + 2 * PAGE;
    }
    pub fn cmdOut(l: Layout) usize {
        return l.cmdIn() + CMD_BYTES + PAGE;
    }
    pub fn flags(l: Layout) usize {
        return l.cmdOut() + CMD_BYTES;
    }
    pub fn sync(l: Layout) usize {
        return l.flags() + PAGE;
    }
    pub fn window(l: Layout) usize {
        return l.sync() + PAGE;
    }
};
pub const FLAG = 0; // u64 in the flag page: the last exchange whose peer part has landed (written by the peer's relay)
pub const HELLO = 64;
pub const CMD_FLAG = 128; // u64, at rank 1: the last command rank 0 has written
pub const CMD_ACK = 192; // u64, at rank 0: the last command rank 1 has copied out
pub const READY = 256; // u64: the peer has loaded its model
pub const BEAT = 320; // u64: the peer's watchdog ran; a rank idle in no round still sees a dead peer through this

// The sync page's words shared with pair_rows.metal: POSTED last released exchange, FAILED give-ups, OBSERVED refusals.
pub const POSTED = 0;
pub const FAILED = 16;
pub const OBSERVED = 20;

pub const seam_failed: u64 = std.math.maxInt(u64);

pub const Seam = struct { x: u32, ref: Ref };

/// Elements in one 32-bit window word: the relay and the link move whole words, and a frame element is a bf16.
pub const elems_per_word: usize = 2;

/// `stage`'s outgoing copy: this rank's `cols` of `rows` rows, bf16 -- the frame seam's payload.
pub fn stageBytes(rows: u32, cols: u32) usize {
    return @as(usize, rows) * cols * @sizeOf(u16);
}

/// `gather`'s `out`: both halves of `rows` rows packed into the frame, so the frame row is `2 * cols` wide.
pub fn gatherBytes(rows: u32, cols: u32) usize {
    return @as(usize, rows) * 2 * cols * @sizeOf(u16);
}

/// The head seam's exchange: an (fp32 value, u32 id) a row; pair_pick compares logits, not words.
pub fn pickBytes(rows: u32) usize {
    return @as(usize, rows) * @sizeOf(f32) * 2;
}

/// A frame tensor's bytes: the bf16 grid `stage` reads and `gather` writes its halves into.
pub fn frameBytes(rows: u32, width: u32) usize {
    return @as(usize, rows) * width * @sizeOf(u16);
}

/// One fp32 value as the frame stores it (bf16, nearest even), the host side of the kernels' frame elements.
pub fn toFrame(v: f32) u16 {
    return bf16Bits(v);
}

/// One rounding to bf16, nearest-even: the host side of a frame element (pair_rows.metal copies the bits it is given).
fn bf16Bits(v: f32) u16 {
    const u: u32 = @bitCast(v);
    return @truncate((u +% 0x7FFF +% ((u >> 16) & 1)) >> 16);
}

/// The seed pattern as a frame element: 1 + m/128, exact in both bf16 and fp32, so expected bytes hold either way.
pub fn seedValue(r: u32, i: u32, salt: u16) f32 {
    const k = r *% 5120 +% i *% 37;
    const m: u32 = (salt & 0x1F) + ((k *% 53) & 0x1F);
    return 1.0 + @as(f32, @floatFromInt(m)) / 128.0;
}

/// One frame element at `at` of a bf16 payload: `stage`'s and `gather`'s slot is the frame's bits, two bytes each.
pub fn writeHalf(out: []u8, at: usize, v: f32) void {
    std.mem.writeInt(u16, out[at * @sizeOf(u16) ..][0..@sizeOf(u16)], toFrame(v), .little);
}

/// The kernels' argument block: four u32, so its layout and pair_rows.metal's struct stay one struct.
pub const PairRows = extern struct { width: u32, stride: u32, rank: u32, x: u32 };

/// Between encoders only (core/segments.zig:59-66): fence, close, posted, wait, reopen, wait -- a real hand-off.
pub fn seam(enc: *mtl.ComputeEncoder, fence: Fence, posted: mtl.SharedEvent, landed: mtl.SharedEvent, x: u32) void {
    const cb = enc.commandBuffer();
    const mode = enc.dispatchType();
    fence.update(enc.*);
    enc.end();
    cb.signal(posted, @as(u64, x));
    cb.waitFor(landed, @as(u64, x));
    enc.* = cb.compute(mode);
    fence.wait(enc.*);
}
