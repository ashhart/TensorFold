//! The gate every engine-using POST passes before its handler runs: counts, drain, admission, model kind, wake.
const std = @import("std");
const api = @import("engine_api");
const Server = @import("server.zig").Server;
const idle_unload = @import("idle_unload.zig");
const admission = @import("request_admission.zig");
const wait = @import("wait_reason.zig");
const Allocator = std.mem.Allocator;

pub const RouteGroup = api.load_ledger.RouteGroup;

/// What the route asks of the engine: the model-kind refusals read it.
pub const RouteKind = struct {
    /// A text route: speech, Messages, Responses, decisions, System One, chat or completions.
    text: bool,
    embedding: bool,
    rerank: bool,
    systemone: bool,
};

/// The counts and lease a request holds from `enter` until its reply ends.
pub const Held = struct {
    srv: *Server,
    group: RouteGroup,
    lease: ?admission.Lease = null,

    /// Undoes `enter` in reverse order: the lease, the idle window, the load ledger, then the in-progress count.
    pub fn release(h: *Held) void {
        if (h.lease) |*lease| lease.release();
        if (h.srv.idle_manager) |m| m.end();
        h.srv.load.leave(h.group);
        _ = h.srv.generating.fetchSub(1, .seq_cst);
    }
};

/// A parked engine that could not wake: the wait sent and the problem the route logs.
pub const Waiting = struct { wait: wait.Wait, problem: []const u8 };

pub const Refusal = union(enum) {
    /// A stop drains: 503 after the body is dropped.
    draining,
    /// Admission refused before the body is read: 429, and the body stays unread.
    overloaded: admission.Refusal,
    /// The loaded model cannot serve this route: the exact 400 JSON body.
    model_kind: []const u8,
    /// The idle wake failed: 503 after the body is dropped.
    waiting: Waiting,
};

pub const Entered = struct { held: Held, refusal: ?Refusal };

const embeddings_only = "{\"error\":{\"message\":\"this model produces embeddings; use POST /v1/embeddings\",\"type\":\"invalid_request_error\"}}";
const reranker_only = "{\"error\":{\"message\":\"this model is a reranker: use POST /v1/rerank\",\"type\":\"invalid_request_error\"}}";
const image_only = "{\"error\":{\"message\":\"this model is an image model: use POST /v1/images/generations\",\"type\":\"invalid_request_error\"}}";
const readout_only = "{\"error\":{\"message\":\"this model answers decisions only; use POST /v1/systemone\",\"type\":\"invalid_request_error\"}}";

/// Takes the counts and any lease, then checks drain, admission, model kind and the wake; `held` stays held either way.
pub fn enter(srv: *Server, a: Allocator, key_digest: ?[32]u8, group: RouteGroup, kind: RouteKind) Entered {
    // counted before the drain flag is read: a stop that sees no request in progress sees every later one refused
    _ = srv.generating.fetchAdd(1, .seq_cst);
    srv.load.enter(group);
    if (srv.idle_manager) |m| m.begin(); // engine work starts: the idle window restarts at this route's end
    var e: Entered = .{ .held = .{ .srv = srv, .group = group }, .refusal = null };
    if (@import("builtin").is_test) if (@import("routes.zig").counted_hook) |h| h();
    if (srv.draining.load(.seq_cst)) {
        e.refusal = .draining;
        return e;
    }
    // --max-inflight and the rest, when set: a reservation for the whole reply, taken before the body is read
    if (srv.config.admission.enabled()) switch (srv.admission.acquire(srv.io, srv.config.admission, key_digest)) {
        .accepted => |lease| e.held.lease = lease,
        .rejected => |refusal| {
            @import("refusals.zig").admitted(srv, refusal.reason); // its 429 carries the code the ledger counts
            e.refusal = .{ .overloaded = refusal };
            return e;
        },
    };
    e.refusal = modelKind(srv, kind) orelse wake(srv, a);
    return e;
}

fn modelKind(srv: *Server, kind: RouteKind) ?Refusal {
    if (srv.info.embeddings != null and (kind.text or kind.rerank)) return .{ .model_kind = embeddings_only };
    // a reranker's template reads query and document roles only: every other route would score or write nonsense
    if (srv.info.rerank != null and (kind.text or kind.embedding)) return .{ .model_kind = reranker_only };
    // an image-only engine answers only its own route: its text, embedding and rerank routes would return nothing
    if (srv.info.image != null and !srv.info.text_generation and (kind.text or kind.embedding or kind.rerank))
        return .{ .model_kind = image_only };
    // a readout engine answers System One decisions and nothing else
    if (srv.info.readouts != null and !kind.systemone) return .{ .model_kind = readout_only };
    return null;
}

/// The wake for every engine-using route: a parked server reloads under the `active` count this begin raised.
fn wake(srv: *Server, a: Allocator) ?Refusal {
    const m = srv.idle_manager orelse return null;
    const problem = idle_unload.wakeReload(m, srv, a) catch |err|
        return .{ .waiting = .{ .wait = wait.reload_failed, .problem = @errorName(err) } };
    if (problem.len == 0) return null;
    return .{ .waiting = .{ .wait = wait.reloadReason(problem), .problem = problem } };
}
