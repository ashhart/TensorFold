//! Prompt-ID handoff for translated Responses, bound to the exact local render and admitted worker identity.
const std = @import("std");
const json = @import("json");
const errors = @import("errors.zig");
const tokens = @import("tokens.zig");
const Server = @import("server.zig").Server;
const Allocator = std.mem.Allocator;

pub const Identity = struct {
    model: [32]u8,
    tokenizer: [32]u8,
    template: [32]u8,
    media: [32]u8,

    pub fn eql(a: Identity, b: Identity) bool {
        return std.mem.eql(u8, &a.model, &b.model) and std.mem.eql(u8, &a.tokenizer, &b.tokenizer) and
            std.mem.eql(u8, &a.template, &b.template) and std.mem.eql(u8, &a.media, &b.media);
    }
};

pub const Worker = struct {
    identity: Identity,
    render_options: [32]u8,
};

pub const Receipt = struct {
    ids: []const u32,
    token_sha256: [64]u8,
    identity: Identity,
    render_options: [32]u8,
    render_calls: u32,
    started_ns: i96,
    ready_ns: i96,

    /// The actual local render occurred once; no separate renderer or hidden retry is involved.
    pub fn renderDurationNs(r: Receipt) u64 {
        return positiveDelta(r.ready_ns - r.started_ns);
    }

    /// Caller supplies the observed prefill-start timestamp; a timestamp before the request is refused.
    pub fn inputToPrefillNs(r: Receipt, prefill_started_ns: i96) !u64 {
        if (prefill_started_ns < r.started_ns) return error.InvalidPrefillTimestamp;
        return positiveDelta(prefill_started_ns - r.started_ns);
    }

    pub fn admit(r: Receipt, worker: Worker) !void {
        if (!r.identity.eql(worker.identity)) return error.WorkerIdentityMismatch;
        if (!std.mem.eql(u8, &r.render_options, &worker.render_options)) return error.WorkerRenderOptionsMismatch;
    }

    pub fn describe(r: Receipt, w: *std.Io.Writer) !void {
        try std.json.Stringify.value(.{
            .schema = 1,
            .prompt_tokens = r.ids.len,
            .token_sha256 = r.token_sha256,
            .model = std.fmt.bytesToHex(r.identity.model, .lower),
            .tokenizer = std.fmt.bytesToHex(r.identity.tokenizer, .lower),
            .template = std.fmt.bytesToHex(r.identity.template, .lower),
            .media = std.fmt.bytesToHex(r.identity.media, .lower),
            .render_options = std.fmt.bytesToHex(r.render_options, .lower),
            .render_calls = r.render_calls,
            .render_ns = r.renderDurationNs(),
        }, .{}, w);
        try w.writeByte('\n');
    }
};

pub fn templateIdentity(srv: *Server) [32]u8 {
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(srv.text.templateSource(), &digest, .{});
    return digest;
}

/// Translate inline or saved-parent input first; both HTTP and socket paths then share this exact renderer.
pub fn render(srv: *Server, cx: *errors.Cx, translated_chat: json.Value, identity: Identity) !Receipt {
    const started = @import("clock.zig").nowNs(srv.io);
    const options = try optionDigest(cx.a, translated_chat);
    const ids = try tokens.messageIds(srv, cx, translated_chat);
    const ready = @import("clock.zig").nowNs(srv.io);
    return .{
        .ids = ids,
        .token_sha256 = tokens.tokenSha256(ids),
        .identity = identity,
        .render_options = options,
        .render_calls = 1,
        .started_ns = started,
        .ready_ns = ready,
    };
}

pub fn optionDigest(a: Allocator, translated_chat: json.Value) ![32]u8 {
    const canonical = try json.stringify(a, translated_chat, .{ .ascii = false, .compact = true });
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(canonical, &digest, .{});
    return digest;
}

fn positiveDelta(delta: i96) u64 {
    return @intCast(@max(delta, 0));
}

test {
    _ = @import("responses_render_test.zig");
}
