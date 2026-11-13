//! Engine readiness and status values, read without entering a device context.

/// Thread-safe backend readiness; no GPU context or blocking device work is allowed on HTTP threads.
pub const HealthSource = struct {
    ctx: *anyopaque,
    read: *const fn (*anyopaque) bool,
};

/// A backend's SSD-expert cache counts, copied by its engine thread; read under a lane host's lock, never the arena.
pub const FsrSource = struct {
    ctx: *anyopaque,
    read: *const fn (*anyopaque) ?Status.FsrCache,
};

pub const Memory = struct { active: u64 = 0, cache: u64 = 0, peak: u64 = 0, pinned: u64 = 0, limit: ?u64 = null };

/// A backend's own memory counts for LaneHost; read from HTTP threads, so it must not touch the GPU.
pub const MemorySource = struct {
    ctx: ?*anyopaque = null,
    read: *const fn (ctx: ?*anyopaque, reset_peak: bool) ?Memory,
};

pub const Status = struct {
    healthy: bool = true,
    /// Engine-thread publication; all retained states, including long-message interior cuts and spare storage.
    prompt_cache: ?PromptCache = null,
    /// The disk tier's copied snapshot (--learn-conversations): null when the host keeps none.
    disk: ?DiskCache = null,
    /// The FSR SSD-expert cache's copied snapshot: null when FSR is off.
    fsr: ?FsrCache = null,
    /// Requests in prefill or decode, and those waiting for a lane.
    running: u32 = 0,
    waiting: u32 = 0,
    decode_tokens_per_second: f64 = 0,
    prefill_tokens_per_second: f64 = 0,
    /// Requests that gave a lane up to a later one; null when the engine does not count them.
    preemptions: ?u64 = null,
    warming: bool = false,
    /// Live streams written to the caller's buffer: tokens each holds.
    streams: usize = 0,
    /// Generated tokens held by live streams, excluding their prompts.
    generation_tokens: u64 = 0,
    /// Filling prompts' total tokens and those still to prefill (null: no prompt is going in).
    fill_prompt_tokens: ?u32 = null,
    fill_tokens_left: ?u32 = null,
    /// A prompt pass is running whose backend reports no `fill_at`: the totals stay null rather than guessed.
    fill_unknown: bool = false,
    /// Changes whenever the filling set does: a reader pairs its drain samples with one fill.
    fill_run: u64 = 0,

    pub const PromptCache = struct {
        states: u64 = 0,
        bytes: u64 = 0,
        hits: u64 = 0,
        misses: u64 = 0,
        kept: u64 = 0,
        evicted: u64 = 0,
        refused: u64 = 0,
        failed: u64 = 0,
        budget_bytes: u64 = 0,
        spare_bytes: u64 = 0,
        hold: bool = false,
        last_access_ms: i64 = 0,
        entries: [64]EntryView = @splat(.{}),
        entry_count: u64 = 0,
        pub const EntryView = struct {
            id: [16]u8 = @splat(0), // the name stem (rows_chain.Stem): the `.st` file's name on a disk tier
            at: u32 = 0,
            bytes: u64 = 0,
            shared: bool = false,
            branched: bool = false,
            borrowers: u32 = 0,
            last_access_ms: i64 = 0,
            created_at_ms: i64 = 0,
            hits: u64 = 0,
        };
    };
    pub const DiskCache = struct {
        states: u64 = 0,
        held_bytes: u64 = 0,
        cap_bytes: u64 = 0,
        daily_cap_bytes: u64 = 0,
        queue_len: u64 = 0,
        read_rate_bps: f64 = 0,
        read_ms_last: f64 = 0,
        read_ms_ema: f64 = 0,
        write_ms_last: f64 = 0,
        write_ms_ema: f64 = 0,
        read_ms_p50: f64 = 0, // fixed-bucket histogram percentiles (bucket edges; 0: no reads yet)
        read_ms_p95: f64 = 0,
        write_ms_p50: f64 = 0,
        write_ms_p95: f64 = 0,
        last_read_ms: i64 = 0, // wall-clock ms since epoch of the last disk read (0: none)
        last_write_ms: i64 = 0, // wall-clock ms since epoch of the last disk write (0: none)
        written_bytes: u64 = 0,
        read_bytes: u64 = 0,
        reads: u64 = 0,
        writes: u64 = 0,
        skipped: u64 = 0,
        failed: u64 = 0,
        evicted: u64 = 0,
        verified: u64 = 0,
        differed: u64 = 0,
        failure_count: u64 = 0,
        last_error: []const u8 = "",
        admission_retry_at: u64 = 0,
        imprint_states: u64 = 0,
        imprint_bytes: u64 = 0,
        prefetch_made: u64 = 0,
        prefetch_used: u64 = 0,
        prefetch_freed: u64 = 0,
        prefetch_failed: u64 = 0,
        prefetch_cancelled: u64 = 0,
        prefetch_timeouts: u64 = 0,
        prefetch_held_reads: u64 = 0,
        prefetch_held_bytes: u64 = 0,
    };
    pub const FsrCache = struct {
        io_bytes: u64 = 0,
        io_calls: u64 = 0,
        io_size_checks: u64 = 0,
        io_logical_bytes: u64 = 0,
        demand_misses: u64 = 0,
        forecast_misses: u64 = 0,
        demand_keys: u64 = 0,
        unique_demand_keys: u64 = 0,
        worker_joins: u64 = 0,
        cancelled_forecasts: u64 = 0,
        occupancy: u64 = 0,
        capacity: u64 = 0,
        cache_hits: u64 = 0,
        evictions: u64 = 0,

        /// A backend's own counts struct with these fields (Flash Next's lives in a module without engine_api).
        pub fn from(counts: anytype) FsrCache {
            var out: FsrCache = .{};
            inline for (@typeInfo(FsrCache).@"struct".field_names) |name| @field(out, name) = @field(counts, name);
            return out;
        }
    };
};
