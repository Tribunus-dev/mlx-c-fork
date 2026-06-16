#include <stdio.h>
#include <assert.h>
#include <mlx_c/mlx_c.h>
#include <stdlib.h>

void test_adversarial() {
    mlx_c_context_t* ctx = NULL;
    mlx_c_status_t status = mlx_c_context_create(&ctx);
    assert(mlx_c_status_is_ok(status));

    mlx_c_array_t* arr = NULL;
    float data[4] = {1.0, 2.0, 3.0, 4.0};
    int64_t shape[2] = {2, 2};

    // Test null output pointer
    status = mlx_c_array_create_from_f32(ctx, data, shape, 2, NULL);
    assert(!mlx_c_status_is_ok(status) && status.code == MLX_C_STATUS_NULL_POINTER);

    // Test null context
    status = mlx_c_array_create_from_f32(NULL, data, shape, 2, &arr);
    assert(!mlx_c_status_is_ok(status) && status.code == MLX_C_STATUS_NULL_POINTER);
    assert(arr == NULL);

    // Test null data
    status = mlx_c_array_create_from_f32(ctx, NULL, shape, 2, &arr);
    assert(!mlx_c_status_is_ok(status) && status.code == MLX_C_STATUS_NULL_POINTER);
    assert(arr == NULL);

    // Test zero ndim
    status = mlx_c_array_create_from_f32(ctx, data, shape, 0, &arr);
    assert(!mlx_c_status_is_ok(status) && status.code == MLX_C_STATUS_INVALID_ARGUMENT);
    assert(arr == NULL);

    // Test negative dimensions
    int64_t bad_shape1[2] = {2, -2};
    status = mlx_c_array_create_from_f32(ctx, data, bad_shape1, 2, &arr);
    assert(!mlx_c_status_is_ok(status) && status.code == MLX_C_STATUS_INVALID_ARGUMENT);
    assert(arr == NULL);

    // Test shape overflow (should fail gracefully)
    int64_t bad_shape2[2] = {10000000000LL, 10000000000LL};
    status = mlx_c_array_create_from_f32(ctx, data, bad_shape2, 2, &arr);
    assert(!mlx_c_status_is_ok(status) && status.code == MLX_C_STATUS_INVALID_ARGUMENT);
    assert(arr == NULL);

    mlx_c_context_free(ctx);
}

void test_roundtrip_and_ownership() {
    mlx_c_context_t* ctx = NULL;
    mlx_c_status_t status = mlx_c_context_create(&ctx);
    assert(mlx_c_status_is_ok(status));

    bool available = false;
    mlx_c_context_is_backend_available(ctx, &available);

    mlx_c_array_t* arr = NULL;
    float source_data[4] = {1.5f, 2.5f, 3.5f, 4.5f};
    int64_t shape[2] = {2, 2};

    status = mlx_c_array_create_from_f32(ctx, source_data, shape, 2, &arr);
    if (!available) {
        assert(!mlx_c_status_is_ok(status) && status.code == MLX_C_STATUS_BACKEND_UNAVAILABLE);
        mlx_c_context_free(ctx);
        return;
    }

    assert(mlx_c_status_is_ok(status));
    assert(arr != NULL);

    // Mutate source_data to verify copy semantics
    source_data[0] = 99.9f;

    // Inspect ndim
    size_t ndim = 0;
    status = mlx_c_array_ndim(arr, &ndim);
    assert(mlx_c_status_is_ok(status));
    assert(ndim == 2);

    // Inspect size
    size_t size = 0;
    status = mlx_c_array_size(arr, &size);
    assert(mlx_c_status_is_ok(status));
    assert(size == 4);

    // Inspect dtype
    mlx_c_dtype_t dtype;
    status = mlx_c_array_dtype(arr, &dtype);
    assert(mlx_c_status_is_ok(status));
    assert(dtype == MLX_C_DTYPE_FLOAT32);

    // Inspect shape - Two pass query
    size_t queried_ndim = 0;
    status = mlx_c_array_shape(arr, NULL, 0, &queried_ndim);
    assert(mlx_c_status_is_ok(status));
    assert(queried_ndim == 2);

    int64_t out_shape[2];
    status = mlx_c_array_shape(arr, out_shape, 2, &queried_ndim);
    assert(mlx_c_status_is_ok(status));
    assert(out_shape[0] == 2 && out_shape[1] == 2);

    // Array copy to host
    float copied_data[4] = {0};
    status = mlx_c_array_copy_to_f32(arr, copied_data, 4);
    assert(mlx_c_status_is_ok(status));

    // Verify values remain original (1.5) and not mutated (99.9)
    assert(copied_data[0] == 1.5f);
    assert(copied_data[1] == 2.5f);
    assert(copied_data[2] == 3.5f);
    assert(copied_data[3] == 4.5f);

    // Test too small capacity
    float tiny_data[2];
    status = mlx_c_array_copy_to_f32(arr, tiny_data, 2);
    assert(!mlx_c_status_is_ok(status) && status.code == MLX_C_STATUS_INVALID_ARGUMENT);

    mlx_c_array_free(arr);
    mlx_c_context_free(ctx);
}

void test_create_free_loop() {
    mlx_c_context_t* ctx = NULL;
    mlx_c_status_t status = mlx_c_context_create(&ctx);
    assert(mlx_c_status_is_ok(status));

    bool available = false;
    mlx_c_context_is_backend_available(ctx, &available);
    if (!available) {
        mlx_c_context_free(ctx);
        return;
    }

    float data[1] = {1.0f};
    int64_t shape[1] = {1};

    for (int i = 0; i < 1000; i++) {
        mlx_c_array_t* arr = NULL;
        status = mlx_c_array_create_from_f32(ctx, data, shape, 1, &arr);
        assert(mlx_c_status_is_ok(status));
        assert(arr != NULL);
        mlx_c_array_free(arr);
    }

    mlx_c_context_free(ctx);
}

int main() {
    test_adversarial();
    test_roundtrip_and_ownership();
    test_create_free_loop();

    printf("test_array passed\n");
    return 0;
}
