/*
 * Copyright (c) 2013 Paul B Mahol
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

#include <math.h>

#include "libavutil/eval.h"
#include "libavutil/imgutils.h"
#include "libavutil/mem.h"
#include "libavutil/opt.h"
#include "libavutil/pixdesc.h"
#include "avfilter.h"
#include "filters.h"
#include "video.h"

/* This filter builds each output pixel by interpolating (blending) nearby
 * source pixels. A "tap" is one such source pixel and its weight; more taps
 * blend a wider area, which gives a smoother result. 
 * These values are opninionated and meant to get a good balance of quality and perf */
#define BASE_TAPS  4    /* kernel taps at 1:1 and when upscaling */
#define MAX_TAPS   32   /* cap on the widened kernel used when reducing */
#define COEF_BITS  14   /* fixed point precision of the tap weights */
#define COEF_ONE   (1 << COEF_BITS)  /* the taps of one output sample sum to this */
#define H_SHIFT    8    /* horizontal pass scales down by this to fit int16 */
#define V_SHIFT    (COEF_BITS + COEF_BITS - H_SHIFT)  /* vertical pass removes the rest */
#define RING_ROWS  32   /* rows of horizontal output cached per thread, 2^n >= MAX_TAPS */

typedef struct Plane {
    uint8_t       *dst;
    const uint8_t *src;
    int dst_ls, src_ls;
    int dw, dh;
    int sw, sh;
    int taps_h, taps_v;
    int32_t *hpos, *vpos;
    int16_t *hcoef, *vcoef;
} Plane;

static const char *const var_names[] = {
    "in_w",   "iw",
    "in_h",   "ih",
    "out_w",  "ow",
    "out_h",  "oh",
    "in",
    "on",
    "duration",
    "pduration",
    "in_time", "it",
    "out_time", "time", "ot",
    "frame",
    "zoom",
    "pzoom",
    "x", "px",
    "y", "py",
    "a",
    "sar",
    "dar",
    "hsub",
    "vsub",
    NULL
};

enum var_name {
    VAR_IN_W,   VAR_IW,
    VAR_IN_H,   VAR_IH,
    VAR_OUT_W,  VAR_OW,
    VAR_OUT_H,  VAR_OH,
    VAR_IN,
    VAR_ON,
    VAR_DURATION,
    VAR_PDURATION,
    VAR_IN_TIME, VAR_IT,
    VAR_TIME, VAR_OUT_TIME, VAR_OT,
    VAR_FRAME,
    VAR_ZOOM,
    VAR_PZOOM,
    VAR_X, VAR_PX,
    VAR_Y, VAR_PY,
    VAR_A,
    VAR_SAR,
    VAR_DAR,
    VAR_HSUB,
    VAR_VSUB,
    VARS_NB
};

typedef struct ZPcontext {
    const AVClass *class;
    char *zoom_expr_str;
    char *x_expr_str;
    char *y_expr_str;
    char *duration_expr_str;

    AVExpr *zoom_expr, *x_expr, *y_expr;

    int w, h;
    double x, y;
    double prev_zoom;
    int prev_nb_frames;
    int64_t frame_count;
    const AVPixFmtDescriptor *desc;
    AVFrame *in;
    double var_values[VARS_NB];
    int nb_frames;
    int current_frame;
    int finished;
    AVRational framerate;

    Plane planes[4];
    int nb_planes;
    int32_t *tap_pos;            /* tap positions for all planes, both axes */
    int16_t *tap_coef;           /* tap weights, likewise */
    unsigned tap_pos_size, tap_coef_size;
    int16_t *ring;               /* per-job horizontal row cache */
    unsigned ring_size;
    int ring_stride;
    int32_t *acc;                /* per-job vertical accumulator */
    unsigned acc_size;
} ZPContext;

#define OFFSET(x) offsetof(ZPContext, x)
#define FLAGS AV_OPT_FLAG_VIDEO_PARAM|AV_OPT_FLAG_FILTERING_PARAM
static const AVOption zoompan_options[] = {
    { "zoom", "set the zoom expression", OFFSET(zoom_expr_str), AV_OPT_TYPE_STRING, {.str = "1" }, .flags = FLAGS },
    { "z", "set the zoom expression", OFFSET(zoom_expr_str), AV_OPT_TYPE_STRING, {.str = "1" }, .flags = FLAGS },
    { "x", "set the x expression", OFFSET(x_expr_str), AV_OPT_TYPE_STRING, {.str="0"}, .flags = FLAGS },
    { "y", "set the y expression", OFFSET(y_expr_str), AV_OPT_TYPE_STRING, {.str="0"}, .flags = FLAGS },
    { "d", "set the duration expression", OFFSET(duration_expr_str), AV_OPT_TYPE_STRING, {.str="90"}, .flags = FLAGS },
    { "s", "set the output image size", OFFSET(w), AV_OPT_TYPE_IMAGE_SIZE, {.str="hd720"}, .flags = FLAGS },
    { "fps", "set the output framerate", OFFSET(framerate), AV_OPT_TYPE_VIDEO_RATE, { .str = "25" }, 0, INT_MAX, .flags = FLAGS },
    { NULL }
};

AVFILTER_DEFINE_CLASS(zoompan);

static av_cold int init(AVFilterContext *ctx)
{
    ZPContext *s = ctx->priv;

    s->prev_zoom = 1;
    return 0;
}

static int config_output(AVFilterLink *outlink)
{
    AVFilterContext *ctx = outlink->src;
    FilterLink *l = ff_filter_link(outlink);
    ZPContext *s = ctx->priv;
    int ret;

    outlink->w = s->w;
    outlink->h = s->h;
    outlink->time_base = av_inv_q(s->framerate);
    l->frame_rate = s->framerate;
    s->desc = av_pix_fmt_desc_get(outlink->format);
    s->finished = 1;

    ret = av_expr_parse(&s->zoom_expr, s->zoom_expr_str, var_names, NULL, NULL, NULL, NULL, 0, ctx);
    if (ret < 0)
        return ret;

    ret = av_expr_parse(&s->x_expr, s->x_expr_str, var_names, NULL, NULL, NULL, NULL, 0, ctx);
    if (ret < 0)
        return ret;

    ret = av_expr_parse(&s->y_expr, s->y_expr_str, var_names, NULL, NULL, NULL, NULL, 0, ctx);
    if (ret < 0)
        return ret;

    return 0;
}

/* Interpolation weight for one tap, by its distance from the sample point -
 * the Catmull-Rom curve (Keys, 1981:
 * https://ncorr.com/download/publications/keysbicubic.pdf). FFmpeg already
 * exposes this same curve as NPPI_INTER_CUBIC2P_CATMULLROM in vf_scale_npp.c. */
static double catmull_rom(double d)
{
    d = fabs(d);
    if (d < 1.0)
        return  1.5 * d * d * d - 2.5 * d * d + 1.0;
    if (d < 2.0)
        return -0.5 * d * d * d + 2.5 * d * d - 4.0 * d + 2.0;
    return 0.0;
}

/* How many source pixels to interpolate per output pixel. 2 either side is
 * enough at 1:1 or when upscaling; shrinking needs a wider blend to avoid
 * aliasing, so this grows with the ratio (see initFilter() in
 * libswscale/utils.c for the same idea). */
static int kernel_taps(double a)
{
    if (a <= 1.0)
        return BASE_TAPS;
    return av_clip(BASE_TAPS * (int)ceil(a), BASE_TAPS, MAX_TAPS);
}

/* For one axis, work out which source pixels and weights each output pixel
 * interpolates from: output j reads around source coordinate
 * a * (j + 0.5) + b - 0.5 (a = scale, b = fractional offset). Same layout
 * libswscale uses: a position plus a set of weights per output pixel. */
static void build_taps(int dn, double a, double b, int sn, int taps,
                          int32_t *pos, int16_t *coef)
{
    const double r = a > 1.0 ? a : 1.0;
    int j, k;

    for (j = 0; j < dn; j++) {
        double t    = a * (j + 0.5) + b - 0.5;
        int    base = (int)floor(t) - (taps / 2 - 1);
        double c[MAX_TAPS];
        double norm = 0.0;
        int    sum  = 0;

        /* Keep the read range inside the plane; edge pixels repeat, as swscale does. */
        base = av_clip(base, 0, FFMAX(sn - taps, 0));

        for (k = 0; k < taps; k++) {
            c[k]  = catmull_rom(((base + k) - t) / r);
            norm += c[k];
        }
        if (norm == 0.0)
            norm = 1.0;

        pos[j] = base;
        for (k = 0; k < taps; k++) {
            coef[j * taps + k] = lrint(c[k] / norm * COEF_ONE);
            sum += coef[j * taps + k];
        }
        /* Make the weights add up to exactly 1, so a solid-color area is unchanged. */
        coef[j * taps + taps / 2] += COEF_ONE - sum;
    }
}

/* Interpolates one row horizontally. Same job as hScale8To15_c() in
 * libswscale/swscale.c. */
static void hscale_row(int16_t *restrict dst, const uint8_t *restrict src,
                          int dw, int taps, const int32_t *restrict pos,
                          const int16_t *restrict coef)
{
    int j, k;

    if (taps == BASE_TAPS) {
        for (j = 0; j < dw; j++) {
            const uint8_t *s = src + pos[j];
            const int c0 = coef[j * BASE_TAPS + 0], c1 = coef[j * BASE_TAPS + 1];
            const int c2 = coef[j * BASE_TAPS + 2], c3 = coef[j * BASE_TAPS + 3];

            dst[j] = (s[0] * c0 + s[1] * c1 + s[2] * c2 + s[3] * c3) >> H_SHIFT;
        }
        return;
    }

    for (j = 0; j < dw; j++) {
        const uint8_t *restrict s = src + pos[j];
        const int16_t *restrict c = coef + (ptrdiff_t)j * taps;
        int v = 0;

        for (k = 0; k < taps; k++)
            v += s[k] * c[k];
        dst[j] = v >> H_SHIFT;
    }
}

/* Interpolates one row vertically from 4 rows already interpolated
 * horizontally. Same as yuv2planeX_8_c() in libswscale/output.c. */
static void vscale_row(uint8_t *restrict dst, int dw,
                          const int16_t *restrict r0, const int16_t *restrict r1,
                          const int16_t *restrict r2, const int16_t *restrict r3,
                          const int16_t *restrict vc)
{
    const int c0 = vc[0], c1 = vc[1], c2 = vc[2], c3 = vc[3];
    int j;

    for (j = 0; j < dw; j++) {
        int v = (r0[j] * c0 + r1[j] * c1 + r2[j] * c2 + r3[j] * c3
                 + (1 << (V_SHIFT - 1))) >> V_SHIFT;

        dst[j] = av_clip_uint8(v);
    }
}

/* Same, for any number of taps: blend one row at a time into an accumulator,
 * keeping each inner loop a simple multiply-add. */
static void vscale_row_n(uint8_t *restrict dst, int dw, int taps,
                            const int16_t *const *rows,
                            const int16_t *restrict vc,
                            int32_t *restrict acc)
{
    int j, k;

    for (j = 0; j < dw; j++)
        acc[j] = 1 << (V_SHIFT - 1);

    for (k = 0; k < taps; k++) {
        const int16_t *restrict r = rows[k];
        const int c = vc[k];

        for (j = 0; j < dw; j++)
            acc[j] += r[j] * c;
    }

    for (j = 0; j < dw; j++)
        dst[j] = av_clip_uint8(acc[j] >> V_SHIFT);
}

/* Interpolates one band of output rows. A horizontal row is only computed
 * once a vertical tap actually needs it (tracked by next_row) - the same
 * lazy approach as lastInLumBuf in libswscale/swscale.c's scale_internal(). */
static void resample_band(const Plane *p, int i0, int i1, int16_t *ring,
                             int ring_stride, int32_t *acc)
{
    int next_row = -1;
    int i, k;

    for (i = i0; i < i1; i++) {
        const int base = p->vpos[i];

        if (next_row < base)
            next_row = base;
        while (next_row < base + p->taps_v) {
            hscale_row(ring + (next_row & (RING_ROWS - 1)) * (ptrdiff_t)ring_stride,
                          p->src + (ptrdiff_t)next_row * p->src_ls,
                          p->dw, p->taps_h, p->hpos, p->hcoef);
            next_row++;
        }

        if (p->taps_v == BASE_TAPS) {
            vscale_row(p->dst + (ptrdiff_t)i * p->dst_ls, p->dw,
                          ring + ((base + 0) & (RING_ROWS - 1)) * (ptrdiff_t)ring_stride,
                          ring + ((base + 1) & (RING_ROWS - 1)) * (ptrdiff_t)ring_stride,
                          ring + ((base + 2) & (RING_ROWS - 1)) * (ptrdiff_t)ring_stride,
                          ring + ((base + 3) & (RING_ROWS - 1)) * (ptrdiff_t)ring_stride,
                          p->vcoef + i * BASE_TAPS);
        } else {
            const int16_t *rows[MAX_TAPS];

            for (k = 0; k < p->taps_v; k++)
                rows[k] = ring + ((base + k) & (RING_ROWS - 1)) * (ptrdiff_t)ring_stride;
            vscale_row_n(p->dst + (ptrdiff_t)i * p->dst_ls, p->dw, p->taps_v,
                            rows, p->vcoef + (ptrdiff_t)i * p->taps_v, acc);
        }
    }
}

/* Splits the output rows across threads for ff_filter_execute() - same
 * pattern as filter_slice() in libavfilter/vf_hflip.c. */
static int resample_slice(AVFilterContext *ctx, void *arg, int jobnr, int nb_jobs)
{
    ZPContext *s = ctx->priv;
    int16_t *ring = s->ring + (ptrdiff_t)jobnr * RING_ROWS * s->ring_stride;
    int p;

    for (p = 0; p < s->nb_planes; p++) {
        const Plane *pl = &s->planes[p];
        int i0 = pl->dh *  jobnr      / nb_jobs;
        int i1 = pl->dh * (jobnr + 1) / nb_jobs;

        if (i1 > i0)
            resample_band(pl, i0, i1, ring, s->ring_stride,
                             s->acc + (ptrdiff_t)jobnr * s->ring_stride);
    }

    return 0;
}

/* Interpolates the fractional rectangle (rx, ry, rw, rh), in luma pixels,
 * into out */
static int resample_rect(AVFilterContext *ctx, AVFrame *out, const AVFrame *in,
                       double rx, double ry, double rw, double rh)
{
    ZPContext *s = ctx->priv;
    AVFilterLink *outlink = ctx->outputs[0];
    const AVPixFmtDescriptor *desc = s->desc;
    int nb_jobs, p, tap_pos_off = 0, tap_coef_off = 0, max_dw = 0;
    unsigned pos_needed = 0, coef_needed = 0;
    const int taps_h = kernel_taps(rw / outlink->w);
    const int taps_v = kernel_taps(rh / outlink->h);

    s->nb_planes = 0;
    for (p = 0; p < 4 && in->data[p]; p++)
        s->nb_planes++;

    for (p = 0; p < s->nb_planes; p++) {
        int sub_w = (p == 1 || p == 2) ? desc->log2_chroma_w : 0;
        int sub_h = (p == 1 || p == 2) ? desc->log2_chroma_h : 0;
        int dw = AV_CEIL_RSHIFT(outlink->w, sub_w);
        int dh = AV_CEIL_RSHIFT(outlink->h, sub_h);

        pos_needed  += (dw + dh) * sizeof(int32_t);
        coef_needed += (dw * taps_h + dh * taps_v) * sizeof(int16_t);
        max_dw       = FFMAX(max_dw, dw);
    }

    av_fast_mallocz(&s->tap_pos,  &s->tap_pos_size,  pos_needed);
    av_fast_mallocz(&s->tap_coef, &s->tap_coef_size, coef_needed);
    if (!s->tap_pos || !s->tap_coef)
        return AVERROR(ENOMEM);

    nb_jobs = FFMIN(AV_CEIL_RSHIFT(outlink->h, desc->log2_chroma_h),
                    ff_filter_get_nb_threads(ctx));
    nb_jobs = FFMAX(nb_jobs, 1);

    s->ring_stride = max_dw;
    av_fast_mallocz(&s->ring, &s->ring_size,
                    (size_t)nb_jobs * RING_ROWS * max_dw * sizeof(int16_t));
    av_fast_mallocz(&s->acc, &s->acc_size,
                    (size_t)nb_jobs * max_dw * sizeof(int32_t));
    if (!s->ring || !s->acc)
        return AVERROR(ENOMEM);

    for (p = 0; p < s->nb_planes; p++) {
        Plane *pl = &s->planes[p];
        int sub_w = (p == 1 || p == 2) ? desc->log2_chroma_w : 0;
        int sub_h = (p == 1 || p == 2) ? desc->log2_chroma_h : 0;

        pl->dst    = out->data[p];
        pl->dst_ls = out->linesize[p];
        pl->src    = in->data[p];
        pl->src_ls = in->linesize[p];
        pl->dw     = AV_CEIL_RSHIFT(outlink->w, sub_w);
        pl->dh     = AV_CEIL_RSHIFT(outlink->h, sub_h);
        pl->sw     = AV_CEIL_RSHIFT(in->width,  sub_w);
        pl->sh     = AV_CEIL_RSHIFT(in->height, sub_h);
        /* A plane can be smaller than the tap count (tiny frames, subsampled
         * chroma, big reductions); narrowing the taps keeps the read range
         * inside it instead of reading past the row. Buffers stay sized for
         * taps_h/v. */
        pl->taps_h = av_clip(taps_h, 1, pl->sw);
        pl->taps_v = av_clip(taps_v, 1, pl->sh);

        pl->hpos  = s->tap_pos  + tap_pos_off;
        pl->vpos  = s->tap_pos  + tap_pos_off + pl->dw;
        pl->hcoef = s->tap_coef + tap_coef_off;
        pl->vcoef = s->tap_coef + tap_coef_off + pl->dw * taps_h;
        tap_pos_off  += pl->dw + pl->dh;
        tap_coef_off += pl->dw * taps_h + pl->dh * taps_v;

        /* The rectangle is in luma samples; move it onto this plane's grid. */
        build_taps(pl->dw, rw / (double)(1 << sub_w) / pl->dw,
                      rx / (double)(1 << sub_w), pl->sw, pl->taps_h,
                      pl->hpos, pl->hcoef);
        build_taps(pl->dh, rh / (double)(1 << sub_h) / pl->dh,
                      ry / (double)(1 << sub_h), pl->sh, pl->taps_v,
                      pl->vpos, pl->vcoef);
    }

    ff_filter_execute(ctx, resample_slice, NULL, NULL, nb_jobs);

    return 0;
}

/* A whole-pixel 1:1 rectangle - what zoom 1 with no pan comes to - is copied
 * rather than interpolated, which would only cost time and soften the picture. */
static int is_plain_copy(AVFilterContext *ctx, const AVFrame *in,
                            double rx, double ry, double rw, double rh)
{
    ZPContext *s = ctx->priv;
    AVFilterLink *outlink = ctx->outputs[0];
    int ix = (int)lrint(rx), iy = (int)lrint(ry);

    return rw == outlink->w && rh == outlink->h &&
           rx == floor(rx) && ry == floor(ry) &&
           !(ix & ((1 << s->desc->log2_chroma_w) - 1)) &&
           !(iy & ((1 << s->desc->log2_chroma_h) - 1)) &&
           ix + outlink->w <= in->width && iy + outlink->h <= in->height;
}

static void copy_rect(AVFilterContext *ctx, AVFrame *out, const AVFrame *in,
                    double rx, double ry)
{
    ZPContext *s = ctx->priv;
    AVFilterLink *outlink = ctx->outputs[0];
    int p;

    for (p = 0; p < 4 && in->data[p]; p++) {
        int sub_w = (p == 1 || p == 2) ? s->desc->log2_chroma_w : 0;
        int sub_h = (p == 1 || p == 2) ? s->desc->log2_chroma_h : 0;
        int x = (int)lrint(rx) >> sub_w;
        int y = (int)lrint(ry) >> sub_h;

        av_image_copy_plane(out->data[p], out->linesize[p],
                            in->data[p] + (ptrdiff_t)y * in->linesize[p] + x,
                            in->linesize[p],
                            AV_CEIL_RSHIFT(outlink->w, sub_w),
                            AV_CEIL_RSHIFT(outlink->h, sub_h));
    }
}

static int output_single_frame(AVFilterContext *ctx, AVFrame *in, double *var_values, int i,
                               double *zoom, double *dx, double *dy)
{
    ZPContext *s = ctx->priv;
    AVFilterLink *outlink = ctx->outputs[0];
    FilterLink *outl = ff_filter_link(outlink);
    AVFilterLink *inlink = ctx->inputs[0];
    int64_t pts = s->frame_count;
    int ret = 0;
    double rw, rh;
    AVFrame *out;

    var_values[VAR_PX]    = s->x;
    var_values[VAR_PY]    = s->y;
    var_values[VAR_PZOOM] = s->prev_zoom;
    var_values[VAR_PDURATION] = s->prev_nb_frames;
    var_values[VAR_IN_TIME] = var_values[VAR_IT]  = in->pts == AV_NOPTS_VALUE ?
        NAN : in->pts * av_q2d(inlink->time_base);
    var_values[VAR_OUT_TIME] = pts * av_q2d(outlink->time_base);
    var_values[VAR_TIME] = var_values[VAR_OT] = var_values[VAR_OUT_TIME];
    var_values[VAR_FRAME] = i;
    var_values[VAR_ON] = outl->frame_count_in;

    *zoom = av_expr_eval(s->zoom_expr, var_values, NULL);

    *zoom = av_clipd(*zoom, 1, 10);
    var_values[VAR_ZOOM] = *zoom;

    /* Keep the source rectangle fractional: rounding it to whole samples is
     * what made the motion stutter. */
    rw = in->width  / *zoom;
    rh = in->height / *zoom;

    *dx = av_expr_eval(s->x_expr, var_values, NULL);

    *dx = av_clipd(*dx, 0, FFMAX(in->width - rw, 0));
    var_values[VAR_X] = *dx;

    *dy = av_expr_eval(s->y_expr, var_values, NULL);

    *dy = av_clipd(*dy, 0, FFMAX(in->height - rh, 0));
    var_values[VAR_Y] = *dy;

    out = ff_get_video_buffer(outlink, outlink->w, outlink->h);
    if (!out) {
        ret = AVERROR(ENOMEM);
        return ret;
    }

    if (is_plain_copy(ctx, in, *dx, *dy, rw, rh))
        copy_rect(ctx, out, in, *dx, *dy);
    else if ((ret = resample_rect(ctx, out, in, *dx, *dy, rw, rh)) < 0)
        goto error;

    out->pts = pts;
    s->frame_count++;

    ret = ff_filter_frame(outlink, out);
    s->current_frame++;

    if (s->current_frame >= s->nb_frames) {
        if (*dx != -1)
            s->x = *dx;
        if (*dy != -1)
            s->y = *dy;
        if (*zoom != -1)
            s->prev_zoom = *zoom;
        s->prev_nb_frames = s->nb_frames;
        s->nb_frames = 0;
        s->current_frame = 0;
        av_frame_free(&s->in);
        s->finished = 1;
    }
    return ret;
error:
    av_frame_free(&out);
    return ret;
}

static int activate(AVFilterContext *ctx)
{
    ZPContext *s = ctx->priv;
    AVFilterLink *inlink = ctx->inputs[0];
    FilterLink *inl = ff_filter_link(inlink);
    AVFilterLink *outlink = ctx->outputs[0];
    FilterLink *outl = ff_filter_link(outlink);
    int status, ret = 0;
    int64_t pts;

    FF_FILTER_FORWARD_STATUS_BACK(outlink, inlink);

    if (s->in && ff_outlink_frame_wanted(outlink)) {
        double zoom = -1, dx = -1, dy = -1;

        ret = output_single_frame(ctx, s->in, s->var_values, s->current_frame,
                                  &zoom, &dx, &dy);
        if (ret < 0)
            return ret;
    }

    if (!s->in && (ret = ff_inlink_consume_frame(inlink, &s->in)) > 0) {
        double zoom = -1, dx = -1, dy = -1, nb_frames;

        s->finished = 0;
        s->var_values[VAR_IN_W]  = s->var_values[VAR_IW] = s->in->width;
        s->var_values[VAR_IN_H]  = s->var_values[VAR_IH] = s->in->height;
        s->var_values[VAR_OUT_W] = s->var_values[VAR_OW] = s->w;
        s->var_values[VAR_OUT_H] = s->var_values[VAR_OH] = s->h;
        s->var_values[VAR_IN]    = inl->frame_count_out - 1;
        s->var_values[VAR_ON]    = outl->frame_count_in;
        s->var_values[VAR_PX]    = s->x;
        s->var_values[VAR_PY]    = s->y;
        s->var_values[VAR_X]     = 0;
        s->var_values[VAR_Y]     = 0;
        s->var_values[VAR_PZOOM] = s->prev_zoom;
        s->var_values[VAR_ZOOM]  = 1;
        s->var_values[VAR_PDURATION] = s->prev_nb_frames;
        s->var_values[VAR_A]     = (double) s->in->width / s->in->height;
        s->var_values[VAR_SAR]   = inlink->sample_aspect_ratio.num ?
            (double) inlink->sample_aspect_ratio.num / inlink->sample_aspect_ratio.den : 1;
        s->var_values[VAR_DAR]   = s->var_values[VAR_A] * s->var_values[VAR_SAR];
        s->var_values[VAR_HSUB]  = 1 << s->desc->log2_chroma_w;
        s->var_values[VAR_VSUB]  = 1 << s->desc->log2_chroma_h;

        if ((ret = av_expr_parse_and_eval(&nb_frames, s->duration_expr_str,
                                          var_names, s->var_values,
                                          NULL, NULL, NULL, NULL, NULL, 0, ctx)) < 0) {
            av_frame_free(&s->in);
            return ret;
        }

        s->var_values[VAR_DURATION] = s->nb_frames = nb_frames;

        ret = output_single_frame(ctx, s->in, s->var_values, s->current_frame,
                                  &zoom, &dx, &dy);
        if (ret < 0)
            return ret;
    }
    if (ret < 0) {
        return ret;
    } else if (s->finished && ff_inlink_acknowledge_status(inlink, &status, &pts)) {
        ff_outlink_set_status(outlink, status, pts);
        return 0;
    } else {
        if (ff_outlink_frame_wanted(outlink) && s->finished)
            ff_inlink_request_frame(inlink);
        return 0;
    }
}

static const enum AVPixelFormat pix_fmts[] = {
    AV_PIX_FMT_YUV444P,  AV_PIX_FMT_YUV422P,
    AV_PIX_FMT_YUV420P,  AV_PIX_FMT_YUV411P,
    AV_PIX_FMT_YUV410P,  AV_PIX_FMT_YUV440P,
    AV_PIX_FMT_YUVA444P, AV_PIX_FMT_YUVA422P,
    AV_PIX_FMT_YUVA420P,
    AV_PIX_FMT_YUVJ444P, AV_PIX_FMT_YUVJ440P,
    AV_PIX_FMT_YUVJ422P, AV_PIX_FMT_YUVJ420P,
    AV_PIX_FMT_YUVJ411P,
    AV_PIX_FMT_GBRP, AV_PIX_FMT_GBRAP,
    AV_PIX_FMT_GRAY8,
    AV_PIX_FMT_NONE
};

static av_cold void uninit(AVFilterContext *ctx)
{
    ZPContext *s = ctx->priv;

    av_expr_free(s->x_expr);
    av_expr_free(s->y_expr);
    av_expr_free(s->zoom_expr);
    av_frame_free(&s->in);
    av_freep(&s->tap_pos);
    av_freep(&s->tap_coef);
    av_freep(&s->ring);
    av_freep(&s->acc);
    s->tap_pos_size = s->tap_coef_size = s->ring_size = s->acc_size = 0;
}

static const AVFilterPad outputs[] = {
    {
        .name          = "default",
        .type          = AVMEDIA_TYPE_VIDEO,
        .config_props  = config_output,
    },
};

const FFFilter ff_vf_zoompan = {
    .p.name        = "zoompan",
    .p.description = NULL_IF_CONFIG_SMALL("Apply Zoom & Pan effect."),
    .p.priv_class  = &zoompan_class,
    .p.flags       = AVFILTER_FLAG_SLICE_THREADS,
    .priv_size     = sizeof(ZPContext),
    .init          = init,
    .uninit        = uninit,
    .activate      = activate,
    FILTER_INPUTS(ff_video_default_filterpad),
    FILTER_OUTPUTS(outputs),
    FILTER_PIXFMTS_ARRAY(pix_fmts),
};
