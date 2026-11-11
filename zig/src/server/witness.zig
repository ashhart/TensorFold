//! The opt-in execution witness (--execution-witness): one coherent snapshot of what serves, or why it is unknown.
const std = @import("std");
const api = @import("engine_api");
const cli = @import("cli.zig");
const auth = @import("auth.zig");
const parts = @import("witness_parts.zig");
const routes = @import("routes.zig");
const log = @import("log.zig");
const Server = @import("server.zig").Server;
const Conn = @import("http_conn.zig").Conn;
const Sha256 = std.crypto.hash.sha2.Sha256;
const Allocator = std.mem.Allocator;
const Io = std.Io;
pub const Reason = api.execution_witness.Reason;
const Facts = api.execution_witness.Facts;

/// The read route; with the flag off it answers 404, as any unknown path does.
pub const path = "/v1/execution/witness";

/// The component digests a known snapshot names, in the order `content` hashes them.
pub const components = [_][]const u8{ "engine", "binary", "text", "preparation", "environment" };

/// --execution-witness answers only authenticated readers: an API key, or the owner-only --unix-socket.
pub fn refusal(args: cli.Args, keys: bool) ?[]const u8 {
    if (!args.execution_witness or keys or args.unix_socket != null) return null;
    return "--execution-witness needs an API key (--api-key, --api-key-file or TENSORFOLD_API_KEY) or --unix-socket";
}

/// Whether the flags or TENSORFOLD_API_KEY give the key store a key before the load; serve refuses a malformed one.
pub fn keysNamed(a: Allocator, args: cli.Args, env_keys: []const u8) bool {
    var problem: []const u8 = "";
    var keys = auth.Store.init(a, args.api_key, env_keys, null, false, &problem) catch return true;
    defer keys.deinit();
    return keys.enabled() or args.api_key_file != null;
}

/// The first flag outside the witness's first path: learning, drafting, imprint, Cortheon, compaction, kept disk state.
pub fn excluded(args: cli.Args) ?[]const u8 {
    const checks = [_]struct { []const u8, bool }{
        .{ "--slide", args.slide },
        .{ "--learn-conversations", args.learn_conversations },
        .{ "--learn", args.learn },
        .{ "--spill-gib", (args.spill_gib orelse 0) > 0 },
        .{ "--compact-at", args.compact_auto or args.compact_fraction != null },
        .{ "--compact-memory", args.compact_memory != null },
        .{ "--cortheon", args.cortheon },
        .{ "--recall", args.recall },
        .{ "--drafter", args.drafter != null },
        .{ "--mtp-drafts", (args.mtp_drafts orelse 0) > 0 },
        .{ "--record", args.record != null },
        .{ "--vision", args.vision },
        .{ "--tp", args.tp > 1 },
        .{ "--speed-up", args.speed_up != null },
    };
    for (checks) |c| if (c[1]) return c[0];
    return null;
}

/// A known snapshot: `content` over the engine's path and the components, `witness` over content, instance and epoch.
pub const Composed = struct { path: []const u8, content: [32]u8, witness: [32]u8, parts: [components.len][32]u8 };

pub const Snapshot = struct {
    instance: [16]u8,
    epoch: u64,
    result: union(enum) { unknown: struct { reason: Reason, feature: ?[]const u8 = null }, known: Composed },

    fn unknown(s: Snapshot, reason: Reason, feature: ?[]const u8) Snapshot {
        var out = s;
        out.result = .{ .unknown = .{ .reason = reason, .feature = feature } };
        return out;
    }
};

/// Content is restart-stable; the witness also binds this process's instance and the engine's epoch.
pub fn compose(instance: [16]u8, epoch: u64, engine: api.execution_witness.Known, server: [components.len - 1][32]u8) Composed {
    var out: Composed = .{ .path = engine.path, .content = undefined, .witness = undefined, .parts = undefined };
    out.parts[0] = engine.digest;
    @memcpy(out.parts[1..], &server);
    var h = Sha256.init(.{});
    parts.field(&h, "tensorfold execution content v1");
    parts.field(&h, engine.path);
    for (components, out.parts) |name, digest| {
        parts.field(&h, name);
        parts.field(&h, &digest);
    }
    out.content = h.finalResult();
    var w = Sha256.init(.{});
    parts.field(&w, "tensorfold execution witness v1");
    parts.field(&w, &out.content);
    parts.field(&w, &instance);
    var e: [8]u8 = undefined;
    std.mem.writeInt(u64, &e, epoch, .little);
    parts.field(&w, &e);
    out.witness = w.finalResult();
    return out;
}

const Hashed = union(enum) { pending, failed: Reason, ready: [2][32]u8 };

/// The running binary, opened before the load so a later rename cannot swap in other bytes; null logs why.
pub fn openBinary(a: Allocator, io: Io) ?parts.Binary {
    return parts.Binary.openRunning(a, io) catch |e| {
        log.line("execution witness: the running binary cannot be opened ({s}); the witness reads hash_failed", .{@errorName(e)});
        return null;
    };
}

/// The text files' stamps before the text load reads them; null logs why.
pub fn stampText(gpa: Allocator, io: Io, dir: []const u8, template: ?[]const u8) ?*parts.Text {
    return parts.Text.stamp(gpa, io, dir, template) catch |e| {
        log.line("execution witness: the text files cannot be stamped ({s}); the witness reads hash_failed", .{@errorName(e)});
        return null;
    };
}

/// One process's witness: fixed parts from startup, the binary and text hashed after ready, the engine's epoch.
pub const State = struct {
    gpa: Allocator,
    io: Io,
    instance: [16]u8,
    excluded: ?[]const u8,
    preparation: [32]u8,
    environment: [32]u8,
    binary: ?parts.Binary,
    text: ?*const parts.Text,
    /// The engine's generation: each engine the server publishes takes the next, under the idle manager's lock.
    epoch: std.atomic.Value(u64) = .init(1),
    mutex: Io.Mutex = .init,
    hashed: Hashed = .pending,
    thread: ?std.Thread = null,
    stop: parts.Stop = .init(false),
    started: std.atomic.Value(bool) = .init(false),

    /// `binary` and `text` were opened and stamped before the load (null: unknown); the caller keeps and closes them.
    pub fn init(gpa: Allocator, io: Io, srv: *const Server, args: cli.Args, binary: ?parts.Binary, text: ?*const parts.Text) !*State {
        var instance: [16]u8 = undefined;
        try io.randomSecure(&instance);
        var arena: std.heap.ArenaAllocator = .init(gpa);
        defer arena.deinit();
        const a = arena.allocator();
        const s = try gpa.create(State);
        s.* = .{ .gpa = gpa, .io = io, .instance = instance, .excluded = excluded(args), .preparation = undefined, .environment = undefined, .binary = binary, .text = text };
        errdefer gpa.destroy(s);
        s.preparation = try parts.preparation(a, srv.config, srv.text);
        s.environment = try parts.environment(a, srv.config.environ);
        return s;
    }

    /// Hashes the binary and the text files on a thread of its own, and starts the engine's: call once ready.
    pub fn start(s: *State, srv: *Server) void {
        s.started.store(true, .release);
        if (srv.idle_manager) |m| {
            m.mutex.lockUncancelable(s.io);
            defer m.mutex.unlock(s.io);
            if (!m.parked) if (srv.info.witness) |w| w.start();
        } else if (srv.info.witness) |w| w.start();
        s.thread = std.Thread.spawn(.{}, work, .{s}) catch {
            s.publish(.{ .failed = .hash_failed });
            return;
        };
    }

    /// Stops a hash in progress at its next block, so a stop right after ready does not wait for it.
    pub fn deinit(s: *State) void {
        s.stop.store(true, .release);
        if (s.thread) |t| t.join();
        s.gpa.destroy(s);
    }

    /// A new engine is published; the caller holds the lock its readers take (the idle manager's).
    pub fn bump(s: *State, info: api.Info) void {
        _ = s.epoch.fetchAdd(1, .acq_rel);
        if (s.started.load(.acquire)) if (info.witness) |w| w.start(); // a reloaded engine hashes its weights again
    }

    fn work(s: *State) void {
        const binary = if (s.binary) |b| b.digest(s.io, &s.stop) catch |e| return s.publish(.{ .failed = reasonOf(e) }) else return s.publish(.{ .failed = .hash_failed });
        const text = if (s.text) |t| t.digest(s.io, &s.stop) catch |e| return s.publish(.{ .failed = reasonOf(e) }) else return s.publish(.{ .failed = .hash_failed });
        s.publish(.{ .ready = .{ binary, text } });
    }

    fn publish(s: *State, h: Hashed) void {
        s.mutex.lockUncancelable(s.io);
        defer s.mutex.unlock(s.io);
        s.hashed = h;
    }

    /// The engine's facts and epoch read together under the lock every engine swap takes; then the server parts.
    pub fn read(s: *State, srv: *Server) Snapshot {
        var snap: Snapshot = .{ .instance = s.instance, .epoch = s.epoch.load(.acquire), .result = undefined };
        if (s.excluded) |flag| return snap.unknown(.feature_excluded, flag);
        const engine = switch (s.engineFacts(srv, &snap.epoch)) {
            .pending => return snap.unknown(.pending, null),
            .unknown => |r| return snap.unknown(r, null),
            .known => |k| k,
        };
        if (engine.path.len == 0 or engine.path.len > 64) return snap.unknown(.hash_failed, null);
        s.mutex.lockUncancelable(s.io);
        const hashed = s.hashed;
        s.mutex.unlock(s.io);
        switch (hashed) {
            .pending => return snap.unknown(.pending, null),
            .failed => |r| return snap.unknown(r, null),
            .ready => |h| {
                snap.result = .{ .known = compose(s.instance, snap.epoch, engine, .{ h[0], h[1], s.preparation, s.environment }) };
                return snap;
            },
        }
    }

    fn engineFacts(s: *State, srv: *Server, epoch: *u64) Facts {
        const m = srv.idle_manager orelse {
            epoch.* = s.epoch.load(.acquire);
            return factsOf(srv.info);
        };
        m.mutex.lockUncancelable(s.io);
        defer m.mutex.unlock(s.io);
        epoch.* = s.epoch.load(.acquire);
        if (m.parked) return .{ .unknown = if (m.state == .loading) .reloading else .parked };
        return factsOf(srv.info);
    }
};

fn factsOf(info: api.Info) Facts {
    const source = info.witness orelse return .{ .unknown = .unsupported_path };
    return source.facts();
}

fn reasonOf(e: anyerror) Reason {
    return if (e == error.FilesChanged) .files_changed else .hash_failed;
}

/// GET /v1/execution/witness: a stored snapshot, never hashing on the request; 404 with the flag off.
pub fn route(srv: *Server, conn: *Conn, a: Allocator) void {
    const state = srv.witness orelse return routes.unknown(conn, a);
    if (srv.draining.load(.seq_cst)) return routes.sendDraining(conn, "{\"status\":\"draining\"}");
    conn.sendJson(200, render(a, state.read(srv)) catch return);
}

/// The snapshot's JSON: known with its digests, or unknown with its reason (and the excluded flag).
pub fn render(a: Allocator, snap: Snapshot) ![]u8 {
    const instance = std.fmt.bytesToHex(snap.instance, .lower);
    switch (snap.result) {
        .unknown => |u| return std.json.Stringify.valueAlloc(a, .{ .schema = 1, .state = "unknown", .reason = @tagName(u.reason), .feature = u.feature, .instance = instance[0..], .epoch = snap.epoch }, .{ .emit_null_optional_fields = false }),
        .known => |k| {
            var hex: [components.len][64]u8 = undefined;
            for (k.parts, &hex) |p, *x| x.* = std.fmt.bytesToHex(p, .lower);
            const witness = std.fmt.bytesToHex(k.witness, .lower);
            const content = std.fmt.bytesToHex(k.content, .lower);
            const named = .{ .engine = hex[0][0..], .binary = hex[1][0..], .text = hex[2][0..], .preparation = hex[3][0..], .environment = hex[4][0..] };
            return std.json.Stringify.valueAlloc(a, .{ .schema = 1, .state = "known", .witness = witness[0..], .content = content[0..], .instance = instance[0..], .epoch = snap.epoch, .path = k.path, .components = named }, .{});
        },
    }
}
