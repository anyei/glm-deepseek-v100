/* io_uring sequential-read throughput probe (O_DIRECT).
 * Keeps QD requests in flight over one sequential stream — the config that
 * reveals a fast NVMe's real ceiling, unlike QD1 dd.
 * usage: io_probe <file> <qd> <bs_bytes> <total_bytes> <start_off_bytes> */
#define _GNU_SOURCE
#include <liburing.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <stdint.h>

int main(int argc, char **argv) {
    if (argc < 6) { fprintf(stderr, "usage: %s file qd bs total off\n", argv[0]); return 2; }
    const char *path = argv[1];
    int qd = atoi(argv[2]);
    size_t bs = strtoull(argv[3], 0, 10);
    uint64_t total = strtoull(argv[4], 0, 10);
    uint64_t off = strtoull(argv[5], 0, 10);

    int fd = open(path, O_RDONLY | O_DIRECT);
    if (fd < 0) { perror("open"); return 1; }

    struct io_uring ring;
    if (io_uring_queue_init(qd, &ring, 0) < 0) { perror("queue_init"); return 1; }

    char **bufs = malloc(sizeof(char *) * qd);
    for (int i = 0; i < qd; i++)
        if (posix_memalign((void **)&bufs[i], 4096, bs)) { perror("memalign"); return 1; }

    uint64_t submitted = 0, completed = 0, cur = off;
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    for (int i = 0; i < qd && submitted < total; i++) {
        struct io_uring_sqe *sqe = io_uring_get_sqe(&ring);
        io_uring_prep_read(sqe, fd, bufs[i], bs, cur);
        io_uring_sqe_set_data(sqe, bufs[i]);
        cur += bs; submitted += bs;
    }
    io_uring_submit(&ring);

    while (completed < total) {
        struct io_uring_cqe *cqe;
        if (io_uring_wait_cqe(&ring, &cqe) < 0) { perror("wait_cqe"); break; }
        if (cqe->res <= 0) { fprintf(stderr, "read res=%d\n", cqe->res); io_uring_cqe_seen(&ring, cqe); break; }
        completed += cqe->res;
        char *buf = io_uring_cqe_get_data(cqe);
        io_uring_cqe_seen(&ring, cqe);
        if (submitted < total) {
            struct io_uring_sqe *sqe = io_uring_get_sqe(&ring);
            io_uring_prep_read(sqe, fd, buf, bs, cur);
            io_uring_sqe_set_data(sqe, buf);
            cur += bs; submitted += bs;
            io_uring_submit(&ring);
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double secs = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
    double gb = completed / 1e9;
    printf("QD=%-3d bs=%zuK : %.2f GB in %.3fs = %.2f GB/s\n", qd, bs / 1024, gb, secs, gb / secs);
    io_uring_queue_exit(&ring);
    close(fd);
    return 0;
}
