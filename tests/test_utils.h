#ifndef TEST_UTILS_H
#define TEST_UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define EXPECT_TRUE(cond) \
    if (!(cond)) { \
        fprintf(stderr, "EXPECT_TRUE failed at %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        exit(1); \
    }

#define EXPECT_NEAR_FLOAT(a, b, tol) \
    if (fabs((a) - (b)) > (tol)) { \
        fprintf(stderr, "EXPECT_NEAR_FLOAT failed at %s:%d: %f vs %f\n", __FILE__, __LINE__, (float)(a), (float)(b)); \
        exit(1); \
    }

#endif
