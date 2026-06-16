#ifndef TEST_UTILS_H
#define TEST_UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define EXPECT_TRUE(cond) \
    do { \
        if (!(cond)) { \
            fprintf(stderr, "%s:%d: EXPECT_TRUE failed: %s\n", __FILE__, __LINE__, #cond); \
            exit(1); \
        } \
    } while (0)

#define EXPECT_EQ_INT(expected, actual) \
    do { \
        long long e = (long long)(expected); \
        long long a = (long long)(actual); \
        if (e != a) { \
            fprintf(stderr, "%s:%d: EXPECT_EQ_INT failed: expected %lld, got %lld\n", __FILE__, __LINE__, e, a); \
            exit(1); \
        } \
    } while (0)

#define EXPECT_EQ_SIZE(expected, actual) \
    do { \
        size_t e = (size_t)(expected); \
        size_t a = (size_t)(actual); \
        if (e != a) { \
            fprintf(stderr, "%s:%d: EXPECT_EQ_SIZE failed: expected %zu, got %zu\n", __FILE__, __LINE__, e, a); \
            exit(1); \
        } \
    } while (0)

#define EXPECT_EQ_STATUS(expected, actual_status_expr) \
    do { \
        mlx_c_status_t actual_status = (actual_status_expr); \
        mlx_c_status_code_t e = (expected); \
        mlx_c_status_code_t a = actual_status.code; \
        if (e != a) { \
            fprintf(stderr, "%s:%d: EXPECT_EQ_STATUS failed: expected %d, got %d (%s: %s)\n", \
                __FILE__, __LINE__, e, a, actual_status.function, actual_status.message); \
            exit(1); \
        } \
    } while (0)

#define EXPECT_FLOAT_NEAR(expected, actual, tol) \
    do { \
        float e = (expected); \
        float a = (actual); \
        if (fabsf(e - a) > (tol)) { \
            fprintf(stderr, "%s:%d: EXPECT_FLOAT_NEAR failed: expected %f, got %f (diff > %f)\n", \
                __FILE__, __LINE__, e, a, (tol)); \
            exit(1); \
        } \
    } while (0)

#endif // TEST_UTILS_H
