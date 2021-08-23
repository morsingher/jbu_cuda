#include "upsampling.h"

__global__ void UpsamplingKernel(const Parameters params, const TextureObj* tex, float* depth)
{
    const int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    const int rows = params.height_hr;
    const int cols = params.width_hr;
    const int center = p.y * cols + p.x;

    if (p.x >= cols || p.y >= rows) 
    {
        return;
    }

    const float scale = params.width_lr / (float)params.width_hr;
    const float sigma_d = 0.5;
    const float sigma_r = 25.5;
    const int win_size = (params.radius * params.radius + 1) / 2;

    const float o_y = p.y * scale;
    const float o_x = p.x * scale;
    const float ref_pix = tex2D<float>(tex->data[0], p.x + 0.5f, p.y + 0.5f);

    float tot = 0.0, norm = 0.0;

    for (int i = -win_size; i <= win_size; i++) {

        const int r_y = max(min(o_y + (float)i, params.height_lr - 1.0f), 0.0f);
        const int r_ys = max(min(p.y + (float)i, params.height_hr - 1.0f), 0.0f);

        for (int j = -win_size; j <= win_size; j++) {

            const int r_x = max(min(o_x + (float)j, params.width_lr - 1.0f), 0.0f);
            const int r_xs = max(min(p.x + (float)j, params.width_hr - 1.0f), 0.0f);

            const float src_pix = tex2D<float>(tex->data[1], r_x + 0.5f, r_y + 0.5f);
            const float near_pix = tex2D<float>(tex->data[0], r_xs + 0.5f, r_ys + 0.5f);

            const float dis = pow(o_x - r_x, 2) + pow(o_y - r_y, 2);
            const float sgauss = exp(-1.0 * dis / (2 * sigma_d * sigma_d));
            
            const float x_p = fabs(ref_pix - near_pix);
            const float rgauss = exp(-1.0 * (x_p * x_p) / (2 * sigma_r * sigma_r));

            norm += sgauss * rgauss;
            tot += (src_pix * sgauss * rgauss);
        }
    }

    depth[center] = tot / norm;
}

void JBU::RunCuda() {

	const int rows = params.height_hr;
    const int cols = params.width_hr;

    dim3 grid_size((cols + 16 - 1) / 16, (rows + 16 - 1) / 16, 1);
    dim3 block_size(16, 16, 1);

    UpsamplingKernel<<<grid_size, block_size>>>(params, tex_dev, data_dev);
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaMemcpy(data_host, data_dev, sizeof(float)*rows*cols, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaDeviceSynchronize());

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }
}