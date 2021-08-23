# Joint Bilateral Upsampling (JBU) with CUDA

This is a simple implementation of Joint Bilateral Upsampling (JBU) with C++ and CUDA. 

### Usage

Building the code follows the usual CMake pipeline and it requires both OpenCV and CUDA. Running the code is as simple as:

```
./upsampling <low_res_input> <high_res_input> <output_folder>
```

### Data Format

Since this is a typical module in modern Multi-View Stereo (MVS) algorithms (see for example: https://github.com/GhiXu/ACMMP), I assume that you want to upsample a low resolution depth map produced by COLMAP (or at least in its format), guided with the corresponding high resolution image (which can be in any format). Therefore, the output is a binary matrix in the COLMAP format, along with a `.jpg` for visualization purposes. It is trivial to modify this requirement, if needed.

### Software Style

I'm definitely not an expert with CUDA. If anyone wants to suggest coding improvements, you're welcome!