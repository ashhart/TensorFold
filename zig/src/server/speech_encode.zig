//! Encoded speech replies from mono PCM16: MP3, Ogg Opus and AAC through libraries loaded on first use; FLAC here.
const std = @import("std");
const sys = @import("sys");
const log = @import("log.zig");
const loadDynLib = @import("jpeg_out.zig").loadDynLib;
const Allocator = std.mem.Allocator;

pub const Codec = enum { mp3, opus, aac, flac };
pub const Error = error{ OutOfMemory, NoLibrary, EncodeFailed };

const Gfp = ?*anyopaque;
const Lame = struct {
    lame_init: *const fn () callconv(.c) Gfp,
    lame_set_in_samplerate: *const fn (Gfp, c_int) callconv(.c) c_int,
    lame_set_num_channels: *const fn (Gfp, c_int) callconv(.c) c_int,
    lame_set_mode: *const fn (Gfp, c_int) callconv(.c) c_int,
    lame_set_brate: *const fn (Gfp, c_int) callconv(.c) c_int,
    lame_set_quality: *const fn (Gfp, c_int) callconv(.c) c_int,
    lame_set_bWriteVbrTag: *const fn (Gfp, c_int) callconv(.c) c_int,
    lame_init_params: *const fn (Gfp) callconv(.c) c_int,
    lame_encode_buffer: *const fn (Gfp, [*]const i16, ?[*]const i16, c_int, [*]u8, c_int) callconv(.c) c_int,
    lame_encode_flush: *const fn (Gfp, [*]u8, c_int) callconv(.c) c_int,
    lame_close: *const fn (Gfp) callconv(.c) c_int,
};

const Opus = struct {
    opus_encoder_create: *const fn (i32, c_int, c_int, *c_int) callconv(.c) ?*anyopaque,
    opus_encoder_ctl: *const fn (?*anyopaque, c_int, ...) callconv(.c) c_int,
    opus_encode: *const fn (?*anyopaque, [*]const i16, c_int, [*]u8, i32) callconv(.c) i32,
    opus_encoder_destroy: *const fn (?*anyopaque) callconv(.c) void,
};

fn Slot(comptime T: type) type {
    return union(enum) { unknown, missing, ready: T };
}

/// libfdk-aac's buffer list: `numBufs` buffers with their identifiers, byte sizes and element sizes.
const BufDesc = extern struct { numBufs: c_int, bufs: [*]?*anyopaque, bufferIdentifiers: [*]c_int, bufSizes: [*]c_int, bufElSizes: [*]c_int };
const InArgs = extern struct { numInSamples: c_int, numAncBytes: c_int = 0 };
const OutArgs = extern struct { numOutBytes: c_int = 0, numInSamples: c_int = 0, numAncBytes: c_int = 0, bitResState: c_int = 0 };
const Aac = struct {
    aacEncOpen: *const fn (*?*anyopaque, c_uint, c_uint) callconv(.c) c_int,
    aacEncClose: *const fn (*?*anyopaque) callconv(.c) c_int,
    aacEncoder_SetParam: *const fn (?*anyopaque, c_int, c_uint) callconv(.c) c_int,
    aacEncEncode: *const fn (?*anyopaque, ?*const BufDesc, ?*const BufDesc, ?*const InArgs, ?*OutArgs) callconv(.c) c_int,
};

var lame: Slot(Lame) = .unknown;
var opus: Slot(Opus) = .unknown;
var fdk: Slot(Aac) = .unknown;
var mutex: std.Io.Mutex = .init;

fn lameFns(a: Allocator) ?Lame {
    const io = log.io();
    mutex.lockUncancelable(io);
    defer mutex.unlock(io);
    if (lame == .unknown) lame = if (loadDynLib(Lame, .mp3lame, a)) |f| .{ .ready = f } else .missing;
    return if (lame == .ready) lame.ready else null;
}

fn opusFns(a: Allocator) ?Opus {
    const io = log.io();
    mutex.lockUncancelable(io);
    defer mutex.unlock(io);
    if (opus == .unknown) opus = if (loadDynLib(Opus, .opus, a)) |f| .{ .ready = f } else .missing;
    return if (opus == .ready) opus.ready else null;
}

fn aacFns(a: Allocator) ?Aac {
    const io = log.io();
    mutex.lockUncancelable(io);
    defer mutex.unlock(io);
    if (fdk == .unknown) fdk = if (loadDynLib(Aac, .fdkaac, a)) |f| .{ .ready = f } else .missing;
    return if (fdk == .ready) fdk.ready else null;
}

/// Whether `codec`'s library loads here; the first call tries.
pub fn available(codec: Codec, a: Allocator) bool {
    return switch (codec) {
        .mp3 => lameFns(a) != null,
        .opus => opusFns(a) != null,
        .aac => aacFns(a) != null,
        .flac => true,
    };
}

/// What a request for a codec whose library is missing is told.
pub fn missing(codec: Codec) []const u8 {
    return sys.optional_libraries.spec(switch (codec) {
        .opus => .opus,
        .aac => .fdkaac,
        else => .mp3lame,
    }).message;
}

pub fn encode(a: Allocator, codec: Codec, pcm: []const i16, sample_rate: u32) Error![]u8 {
    return switch (codec) {
        .mp3 => mp3(a, pcm, sample_rate),
        .opus => oggOpus(a, pcm, sample_rate),
        .aac => aacAdts(a, pcm, sample_rate),
        .flac => flac(a, pcm, sample_rate),
    };
}

/// AAC-LC at 96 kbit/s, mono, in ADTS frames (each frame its own header, so the bytes play as they are).
fn aacAdts(a: Allocator, pcm: []const i16, sample_rate: u32) Error![]u8 {
    const f = aacFns(a) orelse return error.NoLibrary;
    var h: ?*anyopaque = null;
    if (f.aacEncOpen(&h, 0, 1) != 0) return error.EncodeFailed;
    defer _ = f.aacEncClose(&h);
    // aacenc_lib.h: the object type (2: AAC-LC), bit rate, sample rate, channel mode (1: mono), transport (2: ADTS)
    const params = [_][2]c_uint{ .{ 0x0100, 2 }, .{ 0x0101, 96000 }, .{ 0x0103, sample_rate }, .{ 0x0106, 1 }, .{ 0x0300, 2 } };
    for (params) |p| if (f.aacEncoder_SetParam(h, @intCast(p[0]), p[1]) != 0) return error.EncodeFailed;
    if (f.aacEncEncode(h, null, null, null, null) != 0) return error.EncodeFailed;

    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    var packet: [8192]u8 = undefined;
    var out_ptr: ?*anyopaque = &packet;
    var out_id: c_int = 3; // OUT_BITSTREAM_DATA
    var out_size: c_int = packet.len;
    var out_el: c_int = 1;
    const out_desc: BufDesc = .{ .numBufs = 1, .bufs = @ptrCast(&out_ptr), .bufferIdentifiers = @ptrCast(&out_id), .bufSizes = @ptrCast(&out_size), .bufElSizes = @ptrCast(&out_el) };
    var at: usize = 0;
    const eof = 0x0080; // AACENC_ENCODE_EOF: the flush is done
    while (true) {
        const left = pcm[at..];
        const step = @min(left.len, 1 << 20);
        var in_ptr: ?*anyopaque = @constCast(left.ptr);
        var in_id: c_int = 0; // IN_AUDIO_DATA
        var in_size: c_int = @intCast(step * 2);
        var in_el: c_int = 2;
        const in_desc: BufDesc = .{ .numBufs = 1, .bufs = @ptrCast(&in_ptr), .bufferIdentifiers = @ptrCast(&in_id), .bufSizes = @ptrCast(&in_size), .bufElSizes = @ptrCast(&in_el) };
        // -1 samples asks the encoder to flush what it still holds
        const in_args: InArgs = .{ .numInSamples = if (step == 0) -1 else @intCast(step) };
        var got: OutArgs = .{};
        const rc = f.aacEncEncode(h, &in_desc, &out_desc, &in_args, &got);
        if (rc == eof) break;
        if (rc != 0 or got.numOutBytes < 0 or got.numInSamples < 0) return error.EncodeFailed;
        try out.appendSlice(a, packet[0..@intCast(got.numOutBytes)]);
        at += @intCast(got.numInSamples);
        if (step > 0 and got.numInSamples == 0 and got.numOutBytes == 0) return error.EncodeFailed; // no progress
    }
    return out.toOwnedSlice(a);
}

const Bits = struct {
    out: *std.ArrayList(u8),
    a: Allocator,
    acc: u64 = 0,
    n: u6 = 0,

    /// The low `count` bits of `value`, most significant first.
    fn put(b: *Bits, value: u64, count: u6) Allocator.Error!void {
        var left = count;
        while (left > 0) {
            const take: u6 = @min(left, 32);
            left -= take;
            b.acc = (b.acc << take) | ((value >> left) & ((@as(u64, 1) << take) - 1));
            b.n += take;
            while (b.n >= 8) {
                b.n -= 8;
                try b.out.append(b.a, @truncate(b.acc >> b.n));
            }
        }
    }

    fn alignByte(b: *Bits) Allocator.Error!void {
        if (b.n > 0) try b.put(0, 8 - b.n);
    }
};

const flac_block = 4096;

/// FLAC (RFC 9639), lossless: 4,096-sample blocks, each the best fixed predictor of order 0 to 4, one Rice parameter.
fn flac(a: Allocator, pcm: []const i16, sample_rate: u32) Error![]u8 {
    if (sample_rate == 0 or sample_rate >= 1 << 20) return error.EncodeFailed;
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    try out.appendSlice(a, "fLaC");
    var b: Bits = .{ .out = &out, .a = a };
    try b.put(0x80, 8); // the last metadata block, STREAMINFO
    try b.put(34, 24);
    try b.put(flac_block, 16);
    try b.put(flac_block, 16);
    try b.put(0, 24); // frame sizes: unknown
    try b.put(0, 24);
    try b.put(sample_rate, 20);
    try b.put(0, 3); // one channel
    try b.put(15, 5); // 16 bits a sample
    try b.put(pcm.len, 36);
    var md5: [16]u8 = undefined;
    std.crypto.hash.Md5.hash(std.mem.sliceAsBytes(pcm), &md5, .{}); // of the little-endian samples, as stored
    try out.appendSlice(a, &md5);

    const rate_code: u64 = switch (sample_rate) {
        8000 => 4,
        16000 => 5,
        22050 => 6,
        24000 => 7,
        32000 => 8,
        44100 => 9,
        48000 => 10,
        96000 => 11,
        else => 0, // read it from STREAMINFO
    };
    var number: u32 = 0;
    var at: usize = 0;
    while (at < pcm.len) : (number += 1) {
        const x = pcm[at..@min(pcm.len, at + flac_block)];
        at += x.len;
        const start = out.items.len;
        try b.put(0xfff8, 16); // sync, fixed block size
        try b.put(if (x.len == flac_block) 12 else 7, 4); // 4,096, or the size in 16 bits below
        try b.put(rate_code, 4);
        try b.put(0, 4); // mono
        try b.put(4, 3); // 16 bits
        try b.put(0, 1);
        try utf8(&b, number);
        if (x.len != flac_block) try b.put(x.len - 1, 16);
        try out.append(a, crc8(out.items[start..]));
        try subframe(&b, x);
        try b.alignByte();
        const sum = crc16(out.items[start..]);
        try b.put(sum, 16);
    }
    return out.toOwnedSlice(a);
}

/// One channel's block: a fixed predictor's warm-up samples and Rice-coded residuals, or the samples verbatim.
fn subframe(b: *Bits, x: []const i16) Allocator.Error!void {
    var best_order: usize = 0;
    var best_sum: u64 = std.math.maxInt(u64);
    for (0..@min(5, x.len)) |order| {
        var sum: u64 = 0;
        for (order..x.len) |i| sum += @abs(residual(x, i, order));
        if (sum < best_sum) {
            best_sum = sum;
            best_order = order;
        }
    }
    const n = x.len - best_order;
    // the Rice parameter near log2 of the mean magnitude; 15 is the escape code: a block that needs more goes verbatim
    var k: u6 = 0;
    while (k < 15 and (@as(u64, n) << k) < best_sum) k += 1;
    if (k >= 15 or n == 0) {
        try b.put(0b0_000001_0, 8);
        for (x) |v| try b.put(@as(u16, @bitCast(v)), 16);
        return;
    }
    try b.put(0b0_001000_0 | (@as(u64, best_order) << 1), 8);
    for (x[0..best_order]) |v| try b.put(@as(u16, @bitCast(v)), 16);
    try b.put(0, 2); // Rice parameters of 4 bits
    try b.put(0, 4); // one partition
    try b.put(k, 4);
    for (best_order..x.len) |i| {
        const r = residual(x, i, best_order);
        const u: u64 = @bitCast((r << 1) ^ (r >> 63));
        var q = u >> k;
        while (q >= 32) : (q -= 32) try b.put(0, 32);
        try b.put(1, @intCast(q + 1)); // q zeros, then a one
        if (k > 0) try b.put(u, k);
    }
}

/// Sample i less its fixed prediction of `order` (RFC 9639, section 9.2.5).
fn residual(x: []const i16, i: usize, order: usize) i64 {
    const s = struct {
        fn at(v: []const i16, j: usize) i64 {
            return v[j];
        }
    }.at;
    return switch (order) {
        0 => s(x, i),
        1 => s(x, i) - s(x, i - 1),
        2 => s(x, i) - 2 * s(x, i - 1) + s(x, i - 2),
        3 => s(x, i) - 3 * s(x, i - 1) + 3 * s(x, i - 2) - s(x, i - 3),
        else => s(x, i) - 4 * s(x, i - 1) + 6 * s(x, i - 2) - 4 * s(x, i - 3) + s(x, i - 4),
    };
}

/// A frame number in FLAC's UTF-8-like coding (up to 31 bits here).
fn utf8(b: *Bits, v: u32) Allocator.Error!void {
    if (v < 0x80) return b.put(v, 8);
    const w: u64 = v;
    const bytes: u6 = if (v < 0x800) 2 else if (v < 0x1_0000) 3 else if (v < 0x20_0000) 4 else if (v < 0x400_0000) 5 else 6;
    const lead: u64 = (@as(u64, 0xff00) >> bytes) & 0xff; // `bytes` leading ones, then a zero
    try b.put(lead | (w >> (6 * (bytes - 1))), 8);
    var left = bytes - 1;
    while (left > 0) {
        left -= 1;
        try b.put(0x80 | ((w >> (6 * left)) & 0x3f), 8);
    }
}

/// CRC-8 with polynomial 0x07 and a zero start: FLAC's frame header check.
pub fn crc8(bytes: []const u8) u8 {
    var c: u8 = 0;
    for (bytes) |byte| {
        c ^= byte;
        for (0..8) |_| c = if (c & 0x80 != 0) (c << 1) ^ 0x07 else c << 1;
    }
    return c;
}

/// CRC-16 with polynomial 0x8005, most significant bit first and a zero start: FLAC's frame check.
pub fn crc16(bytes: []const u8) u16 {
    var c: u16 = 0;
    for (bytes) |byte| {
        c ^= @as(u16, byte) << 8;
        for (0..8) |_| c = if (c & 0x8000 != 0) (c << 1) ^ 0x8005 else c << 1;
    }
    return c;
}

/// Constant 128 kbit/s mono MPEG audio without a LAME tag frame, so equal samples give equal bytes.
fn mp3(a: Allocator, pcm: []const i16, sample_rate: u32) Error![]u8 {
    const f = lameFns(a) orelse return error.NoLibrary;
    if (pcm.len > std.math.maxInt(c_int) / 2) return error.EncodeFailed;
    const g = f.lame_init() orelse return error.EncodeFailed;
    defer _ = f.lame_close(g);
    const mono = 3; // MPEG_mode MONO
    if (f.lame_set_in_samplerate(g, @intCast(sample_rate)) < 0 or f.lame_set_num_channels(g, 1) < 0 or f.lame_set_mode(g, mono) < 0 or
        f.lame_set_brate(g, 128) < 0 or f.lame_set_quality(g, 2) < 0 or f.lame_set_bWriteVbrTag(g, 0) < 0 or f.lame_init_params(g) < 0)
        return error.EncodeFailed;
    // lame.h: the worst case is 1.25 * samples + 7200 bytes, and a flush needs 7200 more
    const out = try a.alloc(u8, pcm.len + pcm.len / 4 + 14400);
    errdefer a.free(out);
    var n: usize = 0;
    if (pcm.len > 0) {
        const got = f.lame_encode_buffer(g, pcm.ptr, null, @intCast(pcm.len), out.ptr, @intCast(out.len));
        if (got < 0) return error.EncodeFailed;
        n = @intCast(got);
    }
    const tail = f.lame_encode_flush(g, out.ptr + n, @intCast(out.len - n));
    if (tail < 0) return error.EncodeFailed;
    n += @intCast(tail);
    return a.realloc(out, n);
}

const opus_rate = 48000;
const frame = 960; // 20 ms at 48 kHz
const application_audio = 2049;
const set_bitrate = 4002;
const get_lookahead = 4027;
const serial = 0x5446_5353; // one logical stream a file: a fixed serial keeps equal samples as equal bytes

/// Ogg Opus (RFC 7845) at 64 kbit/s: the samples resampled to 48 kHz, 20 ms packets, about a second a page.
fn oggOpus(a: Allocator, pcm: []const i16, sample_rate: u32) Error![]u8 {
    const f = opusFns(a) orelse return error.NoLibrary;
    const at48 = try resample(a, pcm, sample_rate, opus_rate);
    defer a.free(at48);
    var err: c_int = 0;
    const st = f.opus_encoder_create(opus_rate, 1, application_audio, &err) orelse return error.EncodeFailed;
    defer f.opus_encoder_destroy(st);
    var lookahead: i32 = 0;
    if (err != 0 or f.opus_encoder_ctl(st, set_bitrate, @as(c_int, 64000)) != 0 or f.opus_encoder_ctl(st, get_lookahead, &lookahead) != 0 or lookahead < 0)
        return error.EncodeFailed;
    const pre_skip: u16 = std.math.cast(u16, lookahead) orelse return error.EncodeFailed;

    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    var head: [19]u8 = undefined;
    @memcpy(head[0..8], "OpusHead");
    head[8] = 1; // version
    head[9] = 1; // channels
    std.mem.writeInt(u16, head[10..12], pre_skip, .little);
    std.mem.writeInt(u32, head[12..16], sample_rate, .little);
    std.mem.writeInt(u16, head[16..18], 0, .little); // output gain
    head[18] = 0; // mapping family: mono or stereo
    var seq: u32 = 0;
    try page(a, &out, 0x02, 0, &seq, &.{&head});
    const vendor = "TensorFold";
    var tags: [8 + 4 + vendor.len + 4]u8 = undefined;
    @memcpy(tags[0..8], "OpusTags");
    std.mem.writeInt(u32, tags[8..12], vendor.len, .little);
    @memcpy(tags[12..][0..vendor.len], vendor);
    std.mem.writeInt(u32, tags[12 + vendor.len ..][0..4], 0, .little); // no comments
    try page(a, &out, 0, 0, &seq, &.{&tags});

    // the encoder's delay is flushed with zeros: every sample must come back after the decoder drops pre_skip
    const total = at48.len + pre_skip;
    const packets = @max(1, (total + frame - 1) / frame);
    const a_page = 50;
    const store = try a.alloc(u8, a_page * 1275);
    defer a.free(store);
    var held: [a_page][]const u8 = undefined;
    var n_held: usize = 0;
    var used: usize = 0;
    var block: [frame]i16 = undefined;
    for (0..packets) |i| {
        const from = i * frame;
        const have = if (from < at48.len) @min(frame, at48.len - from) else 0;
        @memcpy(block[0..have], at48[from..][0..have]);
        @memset(block[have..], 0);
        const got = f.opus_encode(st, &block, frame, store.ptr + used, 1275);
        if (got <= 0) return error.EncodeFailed;
        held[n_held] = store[used..][0..@intCast(got)];
        n_held += 1;
        used += @intCast(got);
        const last = i + 1 == packets;
        if (n_held == a_page or last) {
            // a page's granule is the samples its last packet completes; the last page's ends the stream exactly
            const granule: u64 = if (last) total else (i + 1) * frame;
            try page(a, &out, if (last) 0x04 else 0, granule, &seq, held[0..n_held]);
            n_held = 0;
            used = 0;
        }
    }
    return out.toOwnedSlice(a);
}

/// One Ogg page holding whole packets (at most 255 lacing values).
fn page(a: Allocator, out: *std.ArrayList(u8), flags: u8, granule: u64, seq: *u32, packets: []const []const u8) Allocator.Error!void {
    var lacing: [255]u8 = undefined;
    var n: usize = 0;
    for (packets) |p| {
        var left = p.len;
        while (left >= 255) : (left -= 255) {
            lacing[n] = 255;
            n += 1;
        }
        lacing[n] = @intCast(left);
        n += 1;
    }
    const start = out.items.len;
    try out.appendSlice(a, "OggS");
    try out.append(a, 0); // stream structure version
    try out.append(a, flags);
    var word: [8]u8 = undefined;
    std.mem.writeInt(u64, &word, granule, .little);
    try out.appendSlice(a, &word);
    std.mem.writeInt(u32, word[0..4], serial, .little);
    try out.appendSlice(a, word[0..4]);
    std.mem.writeInt(u32, word[0..4], seq.*, .little);
    try out.appendSlice(a, word[0..4]);
    try out.appendSlice(a, &.{ 0, 0, 0, 0 }); // the checksum, filled below
    try out.append(a, @intCast(n));
    try out.appendSlice(a, lacing[0..n]);
    for (packets) |p| try out.appendSlice(a, p);
    std.mem.writeInt(u32, out.items[start + 22 ..][0..4], crc(out.items[start..]), .little);
    seq.* += 1;
}

/// Ogg's checksum: CRC-32 with polynomial 0x04c11db7, most significant bit first, no reflection, zero start and end.
pub fn crc(bytes: []const u8) u32 {
    var c: u32 = 0;
    for (bytes) |b| c = (c << 8) ^ crc_table[(c >> 24) ^ b];
    return c;
}

const crc_table = table: {
    @setEvalBranchQuota(10_000);
    var t: [256]u32 = undefined;
    for (&t, 0..) |*e, i| {
        var r: u32 = @as(u32, i) << 24;
        for (0..8) |_| r = if (r & 0x8000_0000 != 0) (r << 1) ^ 0x04c1_1db7 else r << 1;
        e.* = r;
    }
    break :table t;
};

const taps = 32;

/// `pcm` at rate `from` as samples at rate `to`: a Kaiser-windowed sinc below the lower Nyquist, 32 taps a sample.
pub fn resample(a: Allocator, pcm: []const i16, from: u32, to: u32) Allocator.Error![]i16 {
    if (from == to) return a.dupe(i16, pcm);
    const g = std.math.gcd(from, to);
    const up: u64 = to / g; // output samples for every `down` input samples
    const down: u64 = from / g;
    const n_out: usize = @intCast((@as(u64, pcm.len) * up + down - 1) / down);
    const out = try a.alloc(i16, n_out);
    errdefer a.free(out);
    // one filter a phase: output j reads the input around j * down / up
    const coef = try a.alloc(f32, @intCast(up * taps));
    defer a.free(coef);
    const cutoff: f64 = @min(1.0, @as(f64, @floatFromInt(to)) / @as(f64, @floatFromInt(from)));
    for (0..@intCast(up)) |phase| {
        const frac = @as(f64, @floatFromInt(phase)) / @as(f64, @floatFromInt(up));
        var sum: f64 = 0;
        var row: [taps]f64 = undefined;
        for (&row, 0..) |*h, k| {
            const x = @as(f64, @floatFromInt(@as(i64, @intCast(k)) - taps / 2 + 1)) - frac;
            h.* = cutoff * sinc(cutoff * x) * kaiser(x / (taps / 2));
            sum += h.*;
        }
        for (row, 0..) |h, k| coef[phase * taps + k] = @floatCast(h / sum);
    }
    for (out, 0..) |*o, j| {
        const at = @as(u64, j) * down;
        const base: i64 = @intCast(at / up);
        const h = coef[@intCast((at % up) * taps)..][0..taps];
        var acc: f32 = 0;
        for (h, 0..) |c, k| {
            const i = base + @as(i64, @intCast(k)) - taps / 2 + 1;
            if (i >= 0 and i < pcm.len) acc += c * @as(f32, @floatFromInt(pcm[@intCast(i)]));
        }
        o.* = @intFromFloat(std.math.clamp(@round(acc), -32768, 32767));
    }
    return out;
}

fn sinc(x: f64) f64 {
    if (x == 0) return 1;
    const y = std.math.pi * x;
    return @sin(y) / y;
}

/// A Kaiser window (beta 8) on [-1, 1], zero outside.
fn kaiser(x: f64) f64 {
    if (@abs(x) >= 1) return 0;
    return bessel0(8.0 * @sqrt(1 - x * x)) / bessel0(8.0);
}

fn bessel0(x: f64) f64 {
    var sum: f64 = 1;
    var term: f64 = 1;
    var k: f64 = 1;
    while (term > 1e-12 * sum) : (k += 1) {
        term *= (x / (2 * k)) * (x / (2 * k));
        sum += term;
    }
    return sum;
}

test "the Ogg checksum of a known page" {
    // RFC 3533's parameters: the check value of "123456789" under CRC-32/MPEG-2 without its start and end inversions
    try std.testing.expectEqual(@as(u32, 0x89a1897f), crc("123456789"));
}

test "a page laces a 600-byte packet as 255, 255, 90 and carries its own checksum" {
    const a = std.testing.allocator;
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(a);
    var seq: u32 = 7;
    const packet: [600]u8 = @splat(0xab);
    try page(a, &out, 0x04, 12345, &seq, &.{&packet});
    try std.testing.expectEqualSlices(u8, "OggS", out.items[0..4]);
    try std.testing.expectEqual(@as(u8, 0x04), out.items[5]);
    try std.testing.expectEqual(@as(u64, 12345), std.mem.readInt(u64, out.items[6..14], .little));
    try std.testing.expectEqual(@as(u32, 7), std.mem.readInt(u32, out.items[18..22], .little));
    try std.testing.expectEqualSlices(u8, &.{ 3, 255, 255, 90 }, out.items[26..30]);
    try std.testing.expectEqual(@as(usize, 30 + 600), out.items.len);
    try std.testing.expectEqual(@as(u32, 8), seq);
    const stored = std.mem.readInt(u32, out.items[22..26], .little);
    @memset(out.items[22..26], 0);
    try std.testing.expectEqual(crc(out.items), stored);
}

test "resampling 44.1 kHz to 48 kHz keeps a 1 kHz tone's level and length" {
    const a = std.testing.allocator;
    var x: [4410]i16 = undefined;
    for (&x, 0..) |*v, i| v.* = @intFromFloat(10000 * @sin(2 * std.math.pi * 1000.0 * @as(f64, @floatFromInt(i)) / 44100.0));
    const y = try resample(a, &x, 44100, 48000);
    defer a.free(y);
    try std.testing.expectEqual(@as(usize, 4800), y.len);
    // away from the ends, each output is the tone at its own time, within the filter's ripple
    for (y[200..4600], 200..) |v, j| {
        const want = 10000 * @sin(2 * std.math.pi * 1000.0 * @as(f64, @floatFromInt(j)) / 48000.0);
        try std.testing.expect(@abs(@as(f64, @floatFromInt(v)) - want) < 30);
    }
    const same = try resample(a, &x, 48000, 48000);
    defer a.free(same);
    try std.testing.expectEqualSlices(i16, &x, same);
}

test "FLAC's two checks on the standard check string" {
    try std.testing.expectEqual(@as(u8, 0xf4), crc8("123456789")); // CRC-8/SMBUS
    try std.testing.expectEqual(@as(u16, 0xfee8), crc16("123456789")); // CRC-16/UMTS
}

test "a silent FLAC block: the stream header, one frame of order-0 zeros, and its checks" {
    const a = std.testing.allocator;
    const x: [16]i16 = @splat(0);
    const out = try flac(a, &x, 44100);
    defer a.free(out);
    try std.testing.expectEqualSlices(u8, "fLaC", out[0..4]);
    try std.testing.expectEqualSlices(u8, &.{ 0x80, 0, 0, 34, 0x10, 0x00, 0x10, 0x00 }, out[4..12]);
    // 44,100 Hz in 20 bits, one channel, 16 bits, 16 samples in 36 bits
    try std.testing.expectEqualSlices(u8, &.{ 0x0a, 0xc4, 0x40, 0xf0, 0, 0, 0, 16 }, out[18..26]);
    const f = out[42..];
    // sync, a 16-bit block size, 44.1 kHz, mono 16-bit, frame 0, size - 1, the header's check
    try std.testing.expectEqualSlices(u8, &.{ 0xff, 0xf8, 0x79, 0x08, 0x00, 0x00, 0x0f }, f[0..7]);
    try std.testing.expectEqual(crc8(f[0..7]), f[7]);
    // a fixed predictor of order 0, Rice parameter 0: sixteen residuals of zero are sixteen one bits
    try std.testing.expectEqualSlices(u8, &.{ 0x10, 0x00, 0x3f, 0xff, 0xc0 }, f[8..13]);
    try std.testing.expectEqual(@as(usize, 15), f.len);
    try std.testing.expectEqual(crc16(f[0..13]), std.mem.readInt(u16, f[13..15], .big));
}

test "a frame number past 127 is coded in two bytes" {
    const a = std.testing.allocator;
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(a);
    var b: Bits = .{ .out = &out, .a = a };
    try utf8(&b, 0x7f);
    try utf8(&b, 0x80);
    try utf8(&b, 0x7ff);
    try std.testing.expectEqualSlices(u8, &.{ 0x7f, 0xc2, 0x80, 0xdf, 0xbf }, out.items);
}
