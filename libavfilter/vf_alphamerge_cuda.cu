/*
 * Copyright (c) 2025 Jorge Estrada <jestrada@atlassian.com>
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */


extern "C" {

    // CUDA implementation of av_image_copy_plane
    __global__ void AlphaMergePlanar(
        unsigned char* main_alpha_plane, // Pointer to the main inputs alpha plane
        int main_alpha_linesize,
        const unsigned char* alpha_mask_luma_plane, // Pointer to the alpha inputs luma plane
        int alpha_mask_luma_linesize,
        int width,
        int height)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x < width && y < height) {
            // Read luma from alpha_mask
            unsigned char luma_value = alpha_mask_luma_plane[y * alpha_mask_luma_linesize + x];

            // Write to main video's alpha plane
            main_alpha_plane[y * main_alpha_linesize + x] = luma_value;
        }
    }

}