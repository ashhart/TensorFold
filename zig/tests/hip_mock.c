/* Host-only HIP ABI fixture, never linked into the runtime. */
#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif
#ifndef OMIT_INIT
EXPORT int hipInit(unsigned int flags) {
    return flags == 0 ? INIT_RESULT : 1;
}
#endif
EXPORT int hipGetDeviceCount(int *count) {
    *count = 2;
    return 0;
}
