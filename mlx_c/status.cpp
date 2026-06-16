#include "mlx_c/status.h"
#include <string.h>

int mlx_c_status_is_ok(mlx_c_status_t status) {
    return status.code == MLX_C_STATUS_OK;
}

mlx_c_status_t mlx_c_status_ok(void) {
    mlx_c_status_t status;
    status.code = MLX_C_STATUS_OK;
    status.function = "mlx_c_status_ok";
    status.message[0] = '\0';
    return status;
}
