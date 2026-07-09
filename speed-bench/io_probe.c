/* io_uring sequential-read throughput probe (O_DIRECT).
 * Keeps QD requests in flight over one sequential stream — the config that
 * reveals a fast NVMe's real ceiling, unlike QD1 dd.
 * usage: io_probe <file> <qd> <bs_bytes> <total_bytes> <start_off_bytes>
 * bs and off must be multiples of the device logical block size (O_DIRECT). */
#define _GNU_SOURCE
#include <liburing.h>
#include <fcntl.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <stdint.h>

int main(int argc, char **argv) {
    if (argc < 6) { fprintf(stderr, "usage: %s file qd bs total off\n", argv[0]); return 2; }
    const char *path = argv[1];
    int qd = atoi(argv[2]);
    long long bs_arg = atoll(argv[3]), total_arg = atoll(argv[4]), off_arg = atoll(argv[5]);
    if (qd < 1 || bs_arg <= 0 || total_arg <= 0 || off_arg < 0) {
        fprintf(stderr, "bad args: need qd>=1, bs>0, total>0, off>=0\n");
        return 2;
    }
    size_t bs = (size_t)bs_arg;
    uint64_t total = (uint64_t)total_arg, off = (uint64_t)off_arg;
    if (bs % 512)
        fprintf(stderr, "warning: bs=%zu is not a multiple of 512; O_DIRECT reads will likely fail\n", bs);

    int fd = open(path, O_RDONLY | O_DIRECT);
    if (fd < 0) { perror("open"); return 1; }

    struct io_uring ring;
    if (io_uring_queue_init(qd, &ring, 0) < 0) { perror("queue_init"); close(fd); return 1; }

    char **bufs = calloc((size_t)qd, sizeof(char *));
    if (!bufs) { fprintf(stderr, "calloc failed\n"); return 1; }
    for (int i = 0; i < qd; i++)
        if (posix_memalign((void **)&bufs[i], 4096, bs)) { perror("memalign"); return 1; }

    /* Drive by request COUNT, not summed bytes: issue ceil(total/bs) reads and
     * keep qd in flight. Terminating on completion count is immune to short
     * reads (0 < res < bs), which would otherwise leave the wait loop blocked
     * on an empty queue. Throughput uses the actual bytes returned. */
    uint64_t total_reqs = (total + bs - 1) / bs;
    uint64_t submitted = 0, done = 0, bytes = 0, cur = off;
    int align_err = 0;

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    for (int i = 0; i < qd && submitted < total_reqs; i++) {
        struct io_uring_sqe *sqe = io_uring_get_sqe(&ring);
        io_uring_prep_read(sqe, fd, bufs[i], bs, cur);
        io_uring_sqe_set_data(sqe, bufs[i]);
        cur += bs; submitted++;
    }
    io_uring_submit(&ring);

    while (done < total_reqs) {
        struct io_uring_cqe *cqe;
        if (io_uring_wait_cqe(&ring, &cqe) < 0) { perror("wait_cqe"); break; }
        int res = cqe->res;
        char *buf = io_uring_cqe_get_data(cqe);
        io_uring_cqe_seen(&ring, cqe);
        done++;
        if (res > 0) bytes += (uint64_t)res;
        else if (res == -EINVAL) align_err = 1;
        else if (res < 0) fprintf(stderr, "read failed: %s\n", strerror(-res));
        if (submitted < total_reqs) {
            struct io_uring_sqe *sqe = io_uring_get_sqe(&ring);
            io_uring_prep_read(sqe, fd, buf, bs, cur);
            io_uring_sqe_set_data(sqe, buf);
            cur += bs; submitted++;
            io_uring_submit(&ring);
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double secs = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
    double gb = bytes / 1e9;
    if (align_err)
        fprintf(stderr, "note: reads returned -EINVAL — bs (%zu) and off (%llu) must be "
                        "multiples of the device logical block size for O_DIRECT\n",
                bs, (unsigned long long)off);
    printf("QD=%-3d bs=%zuK : %.2f GB in %.3fs = %.2f GB/s\n",
           qd, bs / 1024, gb, secs, secs > 0 ? gb / secs : 0.0);

    io_uring_queue_exit(&ring);
    for (int i = 0; i < qd; i++) free(bufs[i]);
    free(bufs);
    close(fd);
    return 0;
}
