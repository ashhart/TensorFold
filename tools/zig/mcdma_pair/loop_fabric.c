// loop_fabric: a stand-in libmcdma-fabric (ABI 1), not MCDMA, for stub_proof.sh's two-process cases (README.md)
#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

#define API __attribute__((visibility("default")))
enum { OK = 0, INVALID = 1, UNSUPPORTED = 2, NOMEM = 3, DEVICE = 4, PEER = 5, TIMEOUT = 6, BOUNDS = 7 };
enum { M_WRITE = 1, M_SIGNAL = 2, M_BYE = 3 };
struct hdr { uint64_t type, off, len, value; };
// the first bytes each end sends: both must be loop_fabric of this run on this link
struct hello { char magic[8], token[40], name[24]; };
#define HELLO_MAGIC "loopfab2"

static long fault(const char *name) {
    const char *v = getenv(name);
    return v ? atol(v) : 0;
}
static atomic_long fault_writes, fault_signals;
static atomic_int fault_dropped, fault_drop_signal;

struct mcdma_fabric { uint8_t *win; size_t len; int kind; char dev[64]; };
struct mcdma_fabric_peer {
    struct mcdma_fabric *f;
    int fd;
    uint64_t peer_len;
    pthread_t rx;
    pthread_mutex_t tx;
    atomic_int claimed, down;
    char why[160];
};

static uint64_t now_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec;
}

static void set_down(struct mcdma_fabric_peer *p, const char *why) {
    int no = 0;
    if (!atomic_compare_exchange_strong(&p->claimed, &no, 1)) return;
    snprintf(p->why, sizeof p->why, "%s", why);
    atomic_store_explicit(&p->down, 1, memory_order_release);
    fprintf(stderr, "sock-fabric: peer down: %s\n", why);
}

static int read_full(int fd, void *buf, size_t n) {
    uint8_t *b = buf;
    while (n) {
        ssize_t got = read(fd, b, n);
        if (got < 0 && errno == EINTR) continue;
        if (got <= 0) return -1;
        b += got;
        n -= (size_t)got;
    }
    return 0;
}

static int write_full(int fd, const void *buf, size_t n) {
    const uint8_t *b = buf;
#ifdef MSG_NOSIGNAL
    const int flags = MSG_NOSIGNAL;
#else
    const int flags = 0;
#endif
    while (n) {
        ssize_t put = send(fd, b, n, flags);
        if (put < 0 && errno == EINTR) continue;
        if (put <= 0) return -1;
        b += put;
        n -= (size_t)put;
    }
    return 0;
}

static void *rx_main(void *arg) {
    struct mcdma_fabric_peer *p = arg;
    const long corrupt = fault("LOOP_FAULT_CORRUPT_NTH"), minlen = fault("LOOP_FAULT_CORRUPT_MINLEN");
    const long bad_signal = fault("LOOP_FAULT_SIGNAL_NTH");
    for (;;) {
        struct hdr h;
        if (read_full(p->fd, &h, sizeof h)) break;
        if (h.type == M_BYE) {
            set_down(p, "the other end said goodbye");
            return NULL;
        }
        if (h.off > p->f->len || h.len > p->f->len - h.off) {
            set_down(p, "the peer broke the protocol (bounds)");
            return NULL;
        }
        if (h.type == M_WRITE) {
            if (read_full(p->fd, p->f->win + h.off, h.len)) break;
            if (corrupt > 0 && h.len >= (uint64_t)(minlen > 0 ? minlen : 1) && atomic_fetch_add(&fault_writes, 1) + 1 == corrupt) {
                p->f->win[h.off + h.len / 2] ^= 1;
                fprintf(stderr, "loop-fabric fault: flipped one bit of received write %ld (%llu bytes)\n", corrupt, (unsigned long long)h.len);
            }
        } else if (h.type == M_SIGNAL) {
            uint64_t v = h.value;
            int hold = bad_signal > 0 && atomic_fetch_add(&fault_signals, 1) + 1 == bad_signal;
            if (hold) {
                v ^= 0xff0000;
                fprintf(stderr, "loop-fabric fault: received signal %ld corrupted: 0x%llx -> 0x%llx\n", bad_signal, (unsigned long long)h.value, (unsigned long long)v);
            }
            __atomic_store_n((uint64_t *)(void *)(p->f->win + h.off), v, __ATOMIC_RELEASE);
            if (hold) usleep((useconds_t)fault("LOOP_FAULT_SIGNAL_HOLD_MS") * 1000);
        }
    }
    if (fault("LOOP_FAULT_SILENT_EOF")) return NULL;
    set_down(p, "the socket closed: the other process is gone");
    return NULL;
}

// LOOP_FAULT_FLAP_MS: the link drops on its own this long after connect.
struct flap { int fd; long ms; };
static void *flap_main(void *arg) {
    struct flap *fl = arg;
    usleep((useconds_t)fl->ms * 1000);
    const char *at = getenv("LOOP_FAULT_DROPPED_AT");
    if (at && *at) {
        FILE *out = fopen(at, "w");
        if (out) {
            fprintf(out, "%ld\n", (long)time(NULL));
            fclose(out);
        }
    }
    fprintf(stderr, "loop-fabric fault: the link dropped %ld ms after connect\n", fl->ms);
    shutdown(fl->fd, SHUT_RDWR);
    free(fl);
    return NULL;
}

static void hello_of(struct hello *h, const char *name) {
    memset(h, 0, sizeof *h);
    memcpy(h->magic, HELLO_MAGIC, sizeof h->magic);
    const char *token = getenv("LOOP_FABRIC_TOKEN");
    snprintf(h->token, sizeof h->token, "%s", token ? token : "");
    snprintf(h->name, sizeof h->name, "%s", name);
}

// The other end's whole hello within `ms`: 1 when it is this run's loop_fabric on this link.
static int hello_ok(int fd, const char *name, int ms) {
    struct hello want, got;
    struct timeval tv = {.tv_sec = ms / 1000, .tv_usec = (ms % 1000) * 1000}, none = {0, 0};
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
    int bad = read_full(fd, &got, sizeof got);
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &none, sizeof none);
    hello_of(&want, name);
    return !bad && !memcmp(&want, &got, sizeof want);
}

static int sock_opts(int fd) {
    int one = 1;
#ifdef SO_NOSIGPIPE
    setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof one);
#endif
    return setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
}

API uint32_t mcdma_fabric_abi(void) { return 1; }

API int mcdma_fabric_open(const char *device, int gid_index, int path_mtu, void *window, size_t length, int dmabuf_fd,
                          uint64_t dmabuf_offset, uint32_t flags, struct mcdma_fabric **out) {
    (void)gid_index, (void)dmabuf_offset, (void)flags;
    if (!device || !window || !length || dmabuf_fd >= 0 || !out) return INVALID;
    if (path_mtu != 1024 && path_mtu != 2048 && path_mtu != 4096) return INVALID;
    struct mcdma_fabric *f = calloc(1, sizeof *f);
    if (!f) return NOMEM;
    f->win = window;
    f->len = length;
    f->kind = strncmp(device, "tb", 2) == 0 ? 2 : 1;
    snprintf(f->dev, sizeof f->dev, "%s", device);
    *out = f;
    return OK;
}

// The end with the lower port listens on it; the other connects to its peer_port, both on 127.0.0.1.
API int mcdma_fabric_connect(struct mcdma_fabric *f, const char *via, int port, int peer_port, const char *name,
                             uint64_t timeout_ns, struct mcdma_fabric_peer **out) {
    (void)via;
    if (!f || !out || port <= 0 || peer_port <= 0) return INVALID;
    const uint64_t deadline = now_ns() + timeout_ns;
    struct sockaddr_in a;
    memset(&a, 0, sizeof a);
    a.sin_family = AF_INET;
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    int fd = -1;
    // one host, one port number for both (as settings.py writes them): the device ending in 0 listens
    size_t dl = strlen(f->dev);
    struct hello me;
    hello_of(&me, name ? name : "");
    if (port < peer_port || (port == peer_port && dl && f->dev[dl - 1] == '0')) {
        a.sin_port = htons((uint16_t)port);
        // SO_REUSEADDR (for this run's own TIME_WAIT) could share a port another process listens on: refuse that
        int probe = socket(AF_INET, SOCK_STREAM, 0);
        int taken = probe >= 0 && connect(probe, (struct sockaddr *)&a, sizeof a) == 0;
        if (probe >= 0) close(probe);
        if (taken) {
            fprintf(stderr, "loop-fabric: 127.0.0.1:%d already has a listener (not this run's): refusing it\n", port);
            return DEVICE;
        }
        int l = socket(AF_INET, SOCK_STREAM, 0), one = 1;
        if (l >= 0) setsockopt(l, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
        if (l < 0 || bind(l, (struct sockaddr *)&a, sizeof a) || listen(l, 1)) {
            if (l >= 0) close(l);
            return DEVICE;
        }
        for (;;) {
            uint64_t now = now_ns();
            if (now >= deadline) break;
            struct pollfd pf = {.fd = l, .events = POLLIN};
            int ms = (int)((deadline - now) / 1000000ull) + 1;
            if (poll(&pf, 1, ms > 200 ? 200 : ms) <= 0) continue;
            fd = accept(l, NULL, NULL);
            if (fd < 0) continue;
            if (!write_full(fd, &me, sizeof me) && hello_ok(fd, me.name, 2000)) break;
            fprintf(stderr, "loop-fabric: a connection to 127.0.0.1:%d that is not this run's loop_fabric: closed\n", port);
            close(fd);
            fd = -1;
        }
        close(l);
    } else {
        a.sin_port = htons((uint16_t)peer_port);
        while (now_ns() < deadline) {
            fd = socket(AF_INET, SOCK_STREAM, 0);
            if (fd >= 0 && connect(fd, (struct sockaddr *)&a, sizeof a) == 0) break;
            if (fd >= 0) close(fd);
            fd = -1;
            usleep(20000);
        }
        // read before writing anything: a stranger's listener on the port gets no byte from this end
        if (fd >= 0 && !hello_ok(fd, me.name, 2000)) {
            fprintf(stderr, "loop-fabric: 127.0.0.1:%d is not this run's loop_fabric (another process's listener?): nothing sent\n", peer_port);
            close(fd);
            return DEVICE;
        }
        if (fd >= 0 && write_full(fd, &me, sizeof me)) {
            close(fd);
            return PEER;
        }
    }
    if (fd < 0) return TIMEOUT;
    sock_opts(fd);
    uint64_t mine = f->len, theirs = 0;
    if (write_full(fd, &mine, sizeof mine) || read_full(fd, &theirs, sizeof theirs)) {
        close(fd);
        return PEER;
    }
    struct mcdma_fabric_peer *p = calloc(1, sizeof *p);
    if (!p) {
        close(fd);
        return NOMEM;
    }
    p->f = f;
    p->fd = fd;
    p->peer_len = theirs;
    pthread_mutex_init(&p->tx, NULL);
    if (pthread_create(&p->rx, NULL, rx_main, p)) {
        close(fd);
        free(p);
        return NOMEM;
    }
    struct flap *fl = fault("LOOP_FAULT_FLAP_MS") > 0 ? malloc(sizeof *fl) : NULL;
    pthread_t flapper;
    if (fl) {
        fl->fd = fd;
        fl->ms = fault("LOOP_FAULT_FLAP_MS");
        if (pthread_create(&flapper, NULL, flap_main, fl)) free(fl);
        else pthread_detach(flapper);
    }
    *out = p;
    return OK;
}

API int mcdma_fabric_link(const struct mcdma_fabric_peer *p) { return p->f->kind; }
API uint64_t mcdma_fabric_peer_length(const struct mcdma_fabric_peer *p) { return p->peer_len; }

API int mcdma_fabric_peer_status(const struct mcdma_fabric_peer *p, char *why, size_t n) {
    if (!atomic_load_explicit(&((struct mcdma_fabric_peer *)p)->down, memory_order_acquire)) return OK;
    if (why && n) snprintf(why, n, "%s", p->why);
    return PEER;
}

static int post(struct mcdma_fabric_peer *p, const struct hdr *h, const void *payload) {
    if (atomic_load_explicit(&p->down, memory_order_acquire)) return PEER;
    pthread_mutex_lock(&p->tx);
    int bad = write_full(p->fd, h, sizeof *h) || (payload && write_full(p->fd, payload, h->len));
    pthread_mutex_unlock(&p->tx);
    if (bad) {
        // the receive thread may still be reading what came before the close (a BYE): its reason first
        for (int i = 0; i < 20 && !atomic_load_explicit(&p->down, memory_order_acquire); ++i) usleep(5000);
        set_down(p, "send failed: the other process is gone");
        return PEER;
    }
    return OK;
}

API int mcdma_fabric_write(struct mcdma_fabric_peer *p, uint64_t local_offset, uint64_t remote_offset, uint64_t length) {
    if (local_offset > p->f->len || length > p->f->len - local_offset || remote_offset > p->peer_len ||
        length > p->peer_len - remote_offset)
        return BOUNDS;
    const long drop = fault("LOOP_FAULT_DROP_LEN");
    if (drop > 0 && (long)length == drop && !atomic_exchange(&fault_dropped, 1)) {
        fprintf(stderr, "loop-fabric fault: lost a write of %llu bytes and the signal after it (link up)\n", (unsigned long long)length);
        atomic_store(&fault_drop_signal, 1);
        return OK;
    }
    struct hdr h = {M_WRITE, remote_offset, length, 0};
    return post(p, &h, p->f->win + local_offset);
}

API int mcdma_fabric_signal(struct mcdma_fabric_peer *p, uint64_t remote_offset, uint64_t value) {
    if (remote_offset % 8 || remote_offset > p->peer_len - 8) return BOUNDS;
    if (atomic_exchange(&fault_drop_signal, 0)) return OK;
    struct hdr h = {M_SIGNAL, remote_offset, 0, value};
    return post(p, &h, NULL);
}

// MCDMA refuses the fused call on one RoCE link; this does the same, and on "tb" devices writes then signals.
API int mcdma_fabric_write_signal(struct mcdma_fabric_peer *p, uint64_t local_offset, uint64_t remote_offset,
                                  uint64_t length, uint64_t signal_offset, uint64_t value) {
    if (p->f->kind == 1 && length) return UNSUPPORTED;
    if (length) {
        int st = mcdma_fabric_write(p, local_offset, remote_offset, length);
        if (st) return st;
    }
    return mcdma_fabric_signal(p, signal_offset, value);
}

API int mcdma_fabric_flush(struct mcdma_fabric_peer *p, uint64_t timeout_ns) {
    (void)timeout_ns;
    return atomic_load_explicit(&p->down, memory_order_acquire) ? PEER : OK;
}

API void mcdma_fabric_disconnect(struct mcdma_fabric_peer **pp) {
    struct mcdma_fabric_peer *p = pp ? *pp : NULL;
    if (!p) return;
    struct hdr h = {M_BYE, 0, 0, 0};
    if (!fault("LOOP_FAULT_NO_BYE")) post(p, &h, NULL);
    shutdown(p->fd, SHUT_RDWR);
    pthread_join(p->rx, NULL);
    close(p->fd);
    pthread_mutex_destroy(&p->tx);
    free(p);
    *pp = NULL;
}

API void mcdma_fabric_close(struct mcdma_fabric **ff) {
    if (!ff || !*ff) return;
    free(*ff);
    *ff = NULL;
}
