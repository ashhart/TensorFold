#include <stdint.h>
#ifdef _WIN32
#define API __declspec(dllexport)
#else
#define API __attribute__((visibility("default")))
#endif
#ifndef MOCK_abi
API uint32_t mcdma_rpc_abi(void) {
#ifdef MOCK_wrong
    return 2;
#else
    return 1;
#endif
}
#endif
#ifndef MOCK_wait
API uint64_t mcdma_rpc_wait_word(const volatile uint64_t *word, uint32_t seq, int equal, uint64_t spin, uint64_t timeout) {
    (void)seq; (void)equal; (void)spin; (void)timeout;
    return *word;
}
#endif
#ifndef MOCK_store
API void mcdma_rpc_store_word(volatile uint64_t *word, uint64_t value) { *word = value; }
#endif
