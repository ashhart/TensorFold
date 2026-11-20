//! Immutable source identities and a runtime-overlay cache namespace distinct from base projections.
const std = @import("std");
const rt = @import("runtime_lora_target.zig");
const Sha256 = std.crypto.hash.sha2.Sha256;

pub const BaseIdentity = struct {
    digest: [32]u8,

    pub fn fromDigest(digest: [32]u8) BaseIdentity {
        return .{ .digest = digest };
    }
};

pub const AdapterIdentity = struct {
    digest: [32]u8,

    pub fn fromBytes(config: []const u8, safetensors: []const u8) AdapterIdentity {
        var h = Sha256.init(.{});
        h.update("tensorfold.runtime_lora.adapter.v1\x00");
        updateBytes(&h, config);
        updateBytes(&h, safetensors);
        return .{ .digest = h.finalResult() };
    }
};

pub const CacheKey = struct {
    base: BaseIdentity,
    adapter: AdapterIdentity,
    digest: [32]u8,
};

pub const cache_namespace = "tensorfold.runtime_lora.overlay.v1\x00";
pub const base_projection_namespace = "tensorfold.projection.base.v1\x00";

pub fn baseProjectionKey(base: BaseIdentity) [32]u8 {
    var h = Sha256.init(.{});
    h.update(base_projection_namespace);
    h.update(&base.digest);
    return h.finalResult();
}

pub const KeyBuilder = struct {
    hash: Sha256,

    pub fn init(base: BaseIdentity, adapter: AdapterIdentity) KeyBuilder {
        var h = Sha256.init(.{});
        h.update(cache_namespace);
        h.update(&base.digest);
        h.update(&adapter.digest);
        return .{ .hash = h };
    }

    pub fn add(b: *KeyBuilder, module: []const u8, n: usize, k: usize, rank: usize, scale: f64, storage: rt.Storage) void {
        updateBytes(&b.hash, module);
        updateInt(&b.hash, n);
        updateInt(&b.hash, k);
        updateInt(&b.hash, rank);
        var scale_bytes: [8]u8 = undefined;
        std.mem.writeInt(u64, &scale_bytes, @bitCast(scale), .little);
        b.hash.update(&scale_bytes);
        switch (storage) {
            .dense => |dtype| {
                b.hash.update(&.{1});
                updateInt(&b.hash, @backingInt(dtype));
            },
            .affine => |q| {
                b.hash.update(&.{2});
                updateInt(&b.hash, q.bits);
                updateInt(&b.hash, q.group);
            },
            .unsupported => b.hash.update(&.{255}),
        }
    }

    pub fn finish(b: *KeyBuilder, base: BaseIdentity, adapter: AdapterIdentity) CacheKey {
        return .{ .base = base, .adapter = adapter, .digest = b.hash.finalResult() };
    }
};

fn updateInt(h: *Sha256, value: anytype) void {
    var bytes: [8]u8 = undefined;
    std.mem.writeInt(u64, &bytes, @intCast(value), .little);
    h.update(&bytes);
}

fn updateBytes(h: *Sha256, bytes: []const u8) void {
    updateInt(h, bytes.len);
    h.update(bytes);
}
