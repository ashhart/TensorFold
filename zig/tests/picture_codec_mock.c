#include <stddef.h>
#include <stdlib.h>

#ifdef _WIN32
#define API __declspec(dllexport)
#else
#define API __attribute__((visibility("default")))
#endif

#if MOCK_wrong
API const char codec_version[] = "libjpeg-turbo version 3.2.0 (build fixture)";
API int WebPGetDecoderVersion(void) { return 0x010601; }
#else
API const char codec_version[] = "libjpeg-turbo version 3.1.4.1 (build fixture)";
API int WebPGetDecoderVersion(void) { return 0x010600; }
#endif

API void *tj3Init(int type) { (void)type; abort(); }
API void tj3Destroy(void *h) { (void)h; abort(); }
API int tj3Set(void *h, int p, int v) { (void)h; (void)p; (void)v; abort(); }
API int tj3Get(void *h, int p) { (void)h; (void)p; abort(); }
API int tj3DecompressHeader(void *h, const unsigned char *p, size_t n) { (void)h; (void)p; (void)n; abort(); }
#if !MOCK_symbols
API int tj3Decompress8(void *h, const unsigned char *p, size_t n, unsigned char *out, int pitch, int pf) {
    (void)h; (void)p; (void)n; (void)out; (void)pitch; (void)pf; abort();
}
#endif
API int tj3GetErrorCode(void *h) { (void)h; abort(); }
API int WebPGetInfo(const unsigned char *p, size_t n, int *w, int *h) { (void)p; (void)n; (void)w; (void)h; abort(); }
#if !MOCK_symbols
API unsigned char *WebPDecodeRGBAInto(const unsigned char *p, size_t n, unsigned char *out, size_t len, int stride) {
    (void)p; (void)n; (void)out; (void)len; (void)stride; abort();
}
#endif
