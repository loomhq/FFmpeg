/*
 * Copyright (c) 2026
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

#include "libavutil/attributes.h"
#include "libavutil/cpu.h"
#include "libavutil/aarch64/cpu.h"
#include "libavfilter/vf_overlay.h"

int ff_overlay_row_44_neon(uint8_t *d, uint8_t *da, uint8_t *s, uint8_t *a,
                           int w, ptrdiff_t alinesize);
int ff_overlay_row_20_neon(uint8_t *d, uint8_t *da, uint8_t *s, uint8_t *a,
                           int w, ptrdiff_t alinesize);
int ff_overlay_row_22_neon(uint8_t *d, uint8_t *da, uint8_t *s, uint8_t *a,
                           int w, ptrdiff_t alinesize);

av_cold void ff_overlay_init_aarch64(AVFilterContext *ctx)
{
    OverlayContext *s = ctx->priv;
    const AVFilterLink *main = ctx->inputs[0];
    const AVFilterLink *overlay = ctx->inputs[1];
    const int cpu_flags = av_get_cpu_flags();

    if (!have_neon(cpu_flags))
        return;

    if ((s->format == OVERLAY_FORMAT_YUV444 || s->format == OVERLAY_FORMAT_GBRP) &&
        overlay->alpha_mode != AVALPHA_MODE_PREMULTIPLIED && !s->main_has_alpha) {
        s->blend_row[0] = ff_overlay_row_44_neon;
        s->blend_row[1] = ff_overlay_row_44_neon;
        s->blend_row[2] = ff_overlay_row_44_neon;
    }

    if (main->format == AV_PIX_FMT_YUV420P &&
        s->format == OVERLAY_FORMAT_YUV420 &&
        overlay->alpha_mode != AVALPHA_MODE_PREMULTIPLIED && !s->main_has_alpha) {
        s->blend_row[0] = ff_overlay_row_44_neon;
        s->blend_row[1] = ff_overlay_row_20_neon;
        s->blend_row[2] = ff_overlay_row_20_neon;
    }

    if (s->format == OVERLAY_FORMAT_YUV422 &&
        overlay->alpha_mode != AVALPHA_MODE_PREMULTIPLIED && !s->main_has_alpha) {
        s->blend_row[0] = ff_overlay_row_44_neon;
        s->blend_row[1] = ff_overlay_row_22_neon;
        s->blend_row[2] = ff_overlay_row_22_neon;
    }

}
