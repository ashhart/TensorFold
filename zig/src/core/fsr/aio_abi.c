#include <aio.h>
#include <stddef.h>
#include <stdint.h>
uint64_t tf_fsr_aio_abi(unsigned field) {
    const uint64_t values[] = {
        sizeof(struct aiocb), _Alignof(struct aiocb), offsetof(struct aiocb, aio_fildes),
        offsetof(struct aiocb, aio_offset), offsetof(struct aiocb, aio_buf), offsetof(struct aiocb, aio_nbytes),
        offsetof(struct aiocb, aio_reqprio), offsetof(struct aiocb, aio_sigevent),
        offsetof(struct aiocb, aio_lio_opcode), sizeof(struct sigevent), SIGEV_NONE
    };
    return field < sizeof(values) / sizeof(values[0]) ? values[field] : UINT64_MAX;
}
