//! Streaming SHA256 selects optional compression instructions at runtime, with an explicit scalar fallback.
const std = @import("std");
const builtin = @import("builtin");
const blocks = @import("sha256_blocks.zig");
const linuxAuxval = if (builtin.link_libc) std.c.getauxval else std.os.linux.getauxval;
pub const Mode = enum { auto, software, hardware };
pub const Backend = enum { software, arm_sha2, x86_sha };

fn cpuid(leaf: u32) [4]u32 {
    var eax: u32 = undefined;
    var ebx: u32 = undefined;
    var ecx: u32 = undefined;
    var edx: u32 = undefined;
    asm volatile ("cpuid"
        : [_] "={eax}" (eax),
          [_] "={ebx}" (ebx),
          [_] "={ecx}" (ecx),
          [_] "={edx}" (edx),
        : [_] "{eax}" (leaf),
          [_] "{ecx}" (@as(u32, 0)),
    );
    return .{ eax, ebx, ecx, edx };
}

pub fn available() Backend {
    if (builtin.zig_backend == .stage2_c) return .software;
    if (builtin.cpu.arch == .aarch64) {
        if (builtin.os.tag == .linux) {
            if (linuxAuxval(std.elf.AT.HWCAP) & std.os.linux.HWCAP.SHA2 != 0) return .arm_sha2;
        } else if (builtin.os.tag == .macos) {
            var yes: c_int = 0;
            var len: usize = @sizeOf(c_int);
            if (std.c.sysctlbyname("hw.optional.arm.FEAT_SHA256", &yes, &len, null, 0) == 0 and
                len == @sizeOf(c_int) and yes == 1) return .arm_sha2;
        }
    } else if (builtin.cpu.arch == .x86_64 and builtin.zig_backend == .stage2_llvm) {
        if (cpuid(0)[0] >= 7 and cpuid(7)[1] & (@as(u32, 1) << 29) != 0) return .x86_sha;
    }
    return .software;
}

pub fn select(mode: Mode) !Backend {
    if (mode == .software) return .software;
    const backend = available();
    if (mode == .hardware and backend == .software) return error.Sha256HardwareUnavailable;
    return backend;
}

pub const Hash = struct {
    backend: Backend,
    standard: std.crypto.hash.sha2.Sha256 = .init(.{}),
    state: [8]u32 = .{ 0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19 },
    buffer: [64]u8 = undefined,
    used: usize = 0,
    bytes: u64 = 0,

    pub fn init(backend: Backend) Hash {
        return .{ .backend = backend };
    }

    fn block(h: *Hash, b: *const [64]u8) void {
        switch (h.backend) {
            .software => unreachable,
            .arm_sha2 => if (builtin.cpu.arch == .aarch64 and builtin.zig_backend != .stage2_c)
                blocks.arm(&h.state, b)
            else
                unreachable,
            .x86_sha => if (builtin.cpu.arch == .x86_64 and builtin.zig_backend == .stage2_llvm)
                blocks.x86(&h.state, b)
            else
                unreachable,
        }
    }

    pub fn update(h: *Hash, input: []const u8) void {
        if (h.backend == .software) return h.standard.update(input);
        h.bytes +%= input.len;
        var at: usize = 0;
        if (h.used > 0) {
            const n = @min(64 - h.used, input.len);
            @memcpy(h.buffer[h.used..][0..n], input[0..n]);
            h.used += n;
            at += n;
            if (h.used == 64) {
                h.block(&h.buffer);
                h.used = 0;
            }
        }
        while (input.len - at >= 64) : (at += 64) h.block(input[at..][0..64]);
        const tail = input[at..];
        @memcpy(h.buffer[h.used..][0..tail.len], tail);
        h.used += tail.len;
    }

    pub fn finalResult(h: *Hash) [32]u8 {
        if (h.backend == .software) return h.standard.finalResult();
        h.buffer[h.used] = 0x80;
        h.used += 1;
        @memset(h.buffer[h.used..], 0);
        if (h.used > 56) {
            h.block(&h.buffer);
            @memset(&h.buffer, 0);
        }
        std.mem.writeInt(u64, h.buffer[56..64], h.bytes *% 8, .big);
        h.block(&h.buffer);
        var out: [32]u8 = undefined;
        for (h.state, 0..) |word, i| std.mem.writeInt(u32, out[i * 4 ..][0..4], word, .big);
        return out;
    }
};

test "runtime SHA256: Linux auxv uses the entry point matching the executable startup" {
    comptime {
        if (builtin.os.tag == .linux and builtin.cpu.arch == .aarch64) {
            if (builtin.link_libc) {
                std.debug.assert(linuxAuxval == std.c.getauxval);
            } else {
                std.debug.assert(linuxAuxval == std.os.linux.getauxval);
            }
        }
    }
}
