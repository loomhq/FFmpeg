/*
 * Copyright (c) 2025 The FFmpeg Project
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

/**
 * @file
 * NPP video cropping video filter
 */

#include <math.h>
#include <nppi.h>

#include "filters.h"
#include "internal.h"
#include "libavutil/avstring.h"
#include "libavutil/common.h"
#include "libavutil/cuda_check.h"
#include "libavutil/eval.h"
#include "libavutil/hwcontext.h"
#include "libavutil/hwcontext_cuda_internal.h"
#include "libavutil/imgutils.h"
#include "libavutil/opt.h"
#include "libavutil/pixdesc.h"

#define CHECK_CU(x) FF_CUDA_CHECK_DL(ctx, device_hwctx->internal->cuda_dl, x)

// clang-format off
static const enum AVPixelFormat supported_formats[] = {
    AV_PIX_FMT_YUV420P,
    AV_PIX_FMT_YUV444P,
    AV_PIX_FMT_NV12,
    AV_PIX_FMT_YUV422P,
    AV_PIX_FMT_YUVA420P,
    AV_PIX_FMT_YUVA444P,
    AV_PIX_FMT_NONE
};

typedef struct NPPCropContext {
    const AVClass* class;

    AVBufferRef* frames_ctx;

    int w, h;
    int x, y;

    int in_w, in_h;

    char* w_expr;
    char* h_expr;
    char* x_expr;
    char* y_expr;

    int keep_aspect;
    int exact;

    AVRational out_sar;
} NPPCropContext;

static const char* const var_names[] = {
    "in_w",  "iw",
    "in_h",  "ih",
    "out_w", "ow",
    "out_h", "oh",
    "x",
    "y",
    "a",
    "sar",
    "dar",
    "hsub",
    "vsub",
    NULL
};

enum {
    VAR_IN_W,  VAR_IW,
    VAR_IN_H,  VAR_IH,
    VAR_OUT_W, VAR_OW,
    VAR_OUT_H, VAR_OH,
    VAR_X,
    VAR_Y,
    VAR_A,
    VAR_SAR,
    VAR_DAR,
    VAR_HSUB,
    VAR_VSUB,
    VARS_NB
};
// clang-format on

// Mostly copied from the crop filter
static int nppcrop_eval_expr(AVFilterContext* ctx) {
    NPPCropContext* crop_ctx = ctx->priv;
    AVFilterLink* inlink = ctx->inputs[0];
    FilterLink *inl = ff_filter_link(inlink);
    AVHWFramesContext* in_hwfc = (AVHWFramesContext*)inl->hw_frames_ctx->data;
    const AVPixFmtDescriptor* desc = av_pix_fmt_desc_get(in_hwfc->sw_format);

    double var_values[VARS_NB], res;
    char* expr_str;
    int ret;

    var_values[VAR_IN_W]  = var_values[VAR_IW]  = crop_ctx->in_w;
    var_values[VAR_IN_H]  = var_values[VAR_IH]  = crop_ctx->in_h;
    var_values[VAR_OUT_W] = var_values[VAR_OW] = NAN;
    var_values[VAR_OUT_H] = var_values[VAR_OH] = NAN;
    var_values[VAR_X]     = NAN;
    var_values[VAR_Y]     = NAN;
    var_values[VAR_A]     = crop_ctx->in_h ? (double)crop_ctx->in_w / crop_ctx->in_h : 1.0;
    var_values[VAR_SAR]   = inlink->sample_aspect_ratio.num ?
                            (double)inlink->sample_aspect_ratio.num / inlink->sample_aspect_ratio.den : 1.0;
    var_values[VAR_DAR]   = var_values[VAR_A] * var_values[VAR_SAR];
    var_values[VAR_HSUB]  = 1 << desc->log2_chroma_w;
    var_values[VAR_VSUB]  = 1 << desc->log2_chroma_h;

#define PARSE_AND_EVAL_EXPR(ctx, expr, result_var, var_name, var_value_array) \
    do { \
        expr_str = expr; \
        if ((ret = av_expr_parse_and_eval(&res, expr_str, var_names, var_values, NULL, NULL, NULL, NULL, NULL, 0, ctx)) < 0) \
            goto fail; \
        result_var = lrint(res); \
        var_value_array = result_var; \
    } while (0)

    PARSE_AND_EVAL_EXPR(ctx, crop_ctx->w_expr, crop_ctx->w, VAR_OW, var_values[VAR_OUT_W]);
    var_values[VAR_OW] = crop_ctx->w;
    PARSE_AND_EVAL_EXPR(ctx, crop_ctx->h_expr, crop_ctx->h, VAR_OH, var_values[VAR_OUT_H]);
    var_values[VAR_OH] = crop_ctx->h;
    
    if (crop_ctx->w <= 0 || crop_ctx->h <= 0) {
        av_log(ctx, AV_LOG_ERROR, "Invalid non-positive crop dimensions w:%d h:%d\n", crop_ctx->w, crop_ctx->h);
        return AVERROR(EINVAL);
    }
    PARSE_AND_EVAL_EXPR(ctx, crop_ctx->x_expr, crop_ctx->x, VAR_X, var_values[VAR_X]);
    PARSE_AND_EVAL_EXPR(ctx, crop_ctx->y_expr, crop_ctx->y, VAR_Y, var_values[VAR_Y]);

    // Clamp crop window
    if (crop_ctx->w > crop_ctx->in_w) crop_ctx->w = crop_ctx->in_w;
    if (crop_ctx->h > crop_ctx->in_h) crop_ctx->h = crop_ctx->in_h;

    if (crop_ctx->x < 0) crop_ctx->x = 0;
    if (crop_ctx->y < 0) crop_ctx->y = 0;

    if ((unsigned)crop_ctx->x + (unsigned)crop_ctx->w > crop_ctx->in_w)
        crop_ctx->x = crop_ctx->in_w - crop_ctx->w;
    if ((unsigned)crop_ctx->y + (unsigned)crop_ctx->h > crop_ctx->in_h)
        crop_ctx->y = crop_ctx->in_h - crop_ctx->h;

    if (crop_ctx->w <= 0 || crop_ctx->h <= 0) {
        av_log(ctx, AV_LOG_ERROR, "Crop dimensions became non-positive after clamping: w:%d h:%d from input %dx%d, x:%d y:%d\n",
               crop_ctx->w, crop_ctx->h, crop_ctx->in_w, crop_ctx->in_h, crop_ctx->x, crop_ctx->y);
        return AVERROR(EINVAL);
    }

    // Force keeping the original aspect ratio if requested
    if (crop_ctx->keep_aspect && inlink->sample_aspect_ratio.num != 0 && crop_ctx->h != 0 && crop_ctx->w != 0) {
        AVRational dar_in = av_mul_q(inlink->sample_aspect_ratio, (AVRational){crop_ctx->in_w, crop_ctx->in_h});
        av_reduce(&crop_ctx->out_sar.num, &crop_ctx->out_sar.den,
                  (int64_t)dar_in.num * crop_ctx->h,
                  (int64_t)dar_in.den * crop_ctx->w,
                  INT_MAX);
    } else {
        crop_ctx->out_sar = inlink->sample_aspect_ratio;
    }

    /*
    We don't want an exact crop by default. The crop should take into account color format subsampling to prevent removing chroma samples and potentially corrupting the output
    The CPU crop filter behaves in a similar way
    */
    if (!crop_ctx->exact) {
        in_hwfc = (AVHWFramesContext*)ff_filter_link(ctx->inputs[0])->hw_frames_ctx->data;
        const AVPixFmtDescriptor* current_desc = av_pix_fmt_desc_get(in_hwfc->sw_format);
        int h_shift = current_desc->log2_chroma_w;
        int v_shift = current_desc->log2_chroma_h;

        // Align to even dimensions
        crop_ctx->x = (crop_ctx->x >> h_shift) << h_shift;
        crop_ctx->y = (crop_ctx->y >> v_shift) << v_shift;
        crop_ctx->w = (crop_ctx->w >> h_shift) << h_shift;
        crop_ctx->h = (crop_ctx->h >> v_shift) << v_shift;

        // After alignment, w/h could become 0 if they were small and odd.
        // Also, the right/bottom edges might have shifted. Re-clamp if needed
        if (crop_ctx->w <= 0) crop_ctx->w = (1 << h_shift); 
        if (crop_ctx->h <= 0) crop_ctx->h = (1 << v_shift);

        // Re-check boundaries
        if ((unsigned)crop_ctx->x + (unsigned)crop_ctx->w > crop_ctx->in_w) {
            crop_ctx->w = ((crop_ctx->in_w - crop_ctx->x) >> h_shift) << h_shift;
        }
        if ((unsigned)crop_ctx->y + (unsigned)crop_ctx->h > crop_ctx->in_h) {
            crop_ctx->h = ((crop_ctx->in_h - crop_ctx->y) >> v_shift) << v_shift;
        }
        if (crop_ctx->w <= 0 || crop_ctx->h <= 0) {
            av_log(ctx, AV_LOG_ERROR, "Crop dimensions non-positive after exact=0 alignment: w:%d h:%d\n", crop_ctx->w, crop_ctx->h);
            return AVERROR(EINVAL);
        }
    }

    return 0;
fail:
    av_log(ctx, AV_LOG_ERROR, "Error evaluating '%s': %s\n", expr_str, av_err2str(ret));
    return ret;
#undef PARSE_AND_EVAL_EXPR
}

static int nppcrop_alloc_out_frames_ctx(AVFilterContext *ctx, AVBufferRef **out_frames_ctx_ref, const int width, const int height) {
    NPPCropContext *s = ctx->priv;
    AVFilterLink *inlink = ctx->inputs[0];
    FilterLink *inl = ff_filter_link(inlink);
    AVHWFramesContext *in_frames_ctx = (AVHWFramesContext*)inl->hw_frames_ctx->data;
    int ret;

    av_buffer_unref(out_frames_ctx_ref);
    *out_frames_ctx_ref = av_hwframe_ctx_alloc(in_frames_ctx->device_ref);
    if (!*out_frames_ctx_ref) return AVERROR(ENOMEM);

    AVHWFramesContext *out_fc = (AVHWFramesContext*)(*out_frames_ctx_ref)->data;
    out_fc->format    = AV_PIX_FMT_CUDA;
    out_fc->sw_format = in_frames_ctx->sw_format;
    out_fc->width     = width;
    out_fc->height    = height;

    ret = av_hwframe_ctx_init(*out_frames_ctx_ref);
    if (ret < 0) {
        av_log(ctx, AV_LOG_ERROR, "Failed to init output hwframes ctx for crop: %s\n", av_err2str(ret));
        av_buffer_unref(out_frames_ctx_ref);
        return ret;
    }
    return 0;
}

static int nppcrop_init(AVFilterContext* ctx) {
    NPPCropContext* s = ctx->priv;
    s->out_sar = (AVRational){0, 1};
    return 0;
}

static void nppcrop_uninit(AVFilterContext* ctx) {
    NPPCropContext* s = ctx->priv;
    av_buffer_unref(&s->frames_ctx);
}

static int nppcrop_config_props(AVFilterLink* outlink) {
    AVFilterContext* ctx = outlink->src;
    FilterLink* outl = ff_filter_link(outlink);
    NPPCropContext* s = ctx->priv;
    AVFilterLink* inlink = ctx->inputs[0];
    FilterLink* inl = ff_filter_link(inlink);

    AVHWFramesContext* in_frames_ctx = (AVHWFramesContext*)inl->hw_frames_ctx->data;
    int format_supported = 0;
    int ret;

    if (!inl->hw_frames_ctx || in_frames_ctx->format != AV_PIX_FMT_CUDA) {
        av_log(ctx, AV_LOG_ERROR, "Input must be CUDA frames.\n");
        return AVERROR(EINVAL);
    }

    for (int i = 0; supported_formats[i] != AV_PIX_FMT_NONE; i++) {
        if (in_frames_ctx->sw_format == supported_formats[i]) {
            format_supported = 1;
            break;
        }
    }
    if (!format_supported) {
        av_log(ctx, AV_LOG_ERROR, "Unsupported input SW format: %s.\n",
               av_get_pix_fmt_name(in_frames_ctx->sw_format));
        return AVERROR(EINVAL);
    }

    s->in_w = inlink->w;
    s->in_h = inlink->h;

    ret = nppcrop_eval_expr(ctx);
    if (ret < 0) return ret;

    ret = nppcrop_alloc_out_frames_ctx(ctx, &s->frames_ctx, s->w, s->h);
    if (ret < 0) return ret;

    outl->hw_frames_ctx = av_buffer_ref(s->frames_ctx);
    if (!outl->hw_frames_ctx) return AVERROR(ENOMEM);

    outlink->w = s->w;
    outlink->h = s->h;
    outlink->sample_aspect_ratio = s->out_sar;
    outlink->time_base = inlink->time_base;
    outlink->format = AV_PIX_FMT_CUDA;

    return 0;
}

static int nppcrop_do_crop(AVFilterContext* ctx, AVFrame* out, const AVFrame* in) {
    NPPCropContext* s = ctx->priv;
    AVFilterLink* in_avlink = ctx->inputs[0];
    FilterLink* inl = ff_filter_link(in_avlink);
    AVHWFramesContext* hwctx_in = (AVHWFramesContext*)inl->hw_frames_ctx->data;
    const AVPixFmtDescriptor* desc_in = av_pix_fmt_desc_get(hwctx_in->sw_format);

    AVHWFramesContext* hwctx_out = (AVHWFramesContext*)s->frames_ctx->data;
    AVCUDADeviceContext* device_hwctx = hwctx_out->device_ctx->hwctx;

    NppStreamContext npp_stream_ctx;
    npp_stream_ctx.hStream = device_hwctx->stream;

    const int nb_planes = av_pix_fmt_count_planes(hwctx_in->sw_format);
    NppStatus npp_st;

    for (int plane = 0; plane < nb_planes; plane++) {
        // For YUV420P, YUV422P, YUV444P, YUVA420P, YUVA444P, hsub/vsub are per-plane from the input AVPixFmtDescriptor.
        // For NV12, the first plane, the Y plane, has no subsampling. The second plane, the interleaved UV plane, has hsub=1 and vsub=1.
        int hsub = (plane > 0) ? desc_in->log2_chroma_w : 0;
        int vsub = (plane > 0) ? desc_in->log2_chroma_h : 0;
        
        // For YUVA formats the alpha plane does not need to take into account subsampling
        if ((hwctx_in->sw_format == AV_PIX_FMT_YUVA420P || hwctx_in->sw_format == AV_PIX_FMT_YUVA444P) && plane == 3) {
            hsub = 0;
            vsub = 0;
        }


        int src_offset_x_comps = AV_CEIL_RSHIFT(s->x, hsub);
        int src_offset_y_lines = AV_CEIL_RSHIFT(s->y, vsub);
        int roi_w_comps        = AV_CEIL_RSHIFT(s->w, hsub);
        int roi_h_lines        = AV_CEIL_RSHIFT(s->h, vsub);

        NppiSize oSizeROI_npp;
        const Npp8u* pSrc_plane_roi_start;
        Npp8u* pDst_plane_start = (Npp8u*)out->data[plane];
        int src_offset_x_bytes;

        if (hwctx_in->sw_format == AV_PIX_FMT_NV12 && plane == 1) {
            // For NV12, U and V are interleaved.
            // Each UV pair is 2 bytes (U then V).
            // We need to adjust the target offset, width, and height so that this format works with the nppiCopy_8u_C1R_Ctx function we're using
            src_offset_x_bytes = src_offset_x_comps * 2; // 2 bytes per UV pair offset
            oSizeROI_npp.width = roi_w_comps * 2;      // Total bytes in width for the ROI
            oSizeROI_npp.height = roi_h_lines;
        } else {
            // All other supported formats we support are planar 8-bit (Y, U, V, or A)
            src_offset_x_bytes = src_offset_x_comps; // should be 1 byte per component
            oSizeROI_npp.width = roi_w_comps;
            oSizeROI_npp.height = roi_h_lines;
        }

        pSrc_plane_roi_start = (const Npp8u*)in->data[plane] +
                               (int64_t)src_offset_y_lines * in->linesize[plane] +
                               src_offset_x_bytes;

        if (oSizeROI_npp.width <= 0 || oSizeROI_npp.height <= 0) {
            av_log(ctx, AV_LOG_VERBOSE, "Skipping plane %d due to zero or negative NPP ROI dimension (%dx%d).\n", plane, oSizeROI_npp.width, oSizeROI_npp.height);
            continue;
        }

        // boundary check
        int plane_in_w_bytes;
        if (hwctx_in->sw_format == AV_PIX_FMT_NV12 && plane == 1) {
            plane_in_w_bytes = AV_CEIL_RSHIFT(in->width, desc_in->log2_chroma_w) * 2;
        } else {
            plane_in_w_bytes = AV_CEIL_RSHIFT(in->width, hsub);
        }

        if (src_offset_x_bytes + oSizeROI_npp.width > plane_in_w_bytes) {
            av_log(ctx, AV_LOG_ERROR, "NPP Crop ROI for plane %d (offset_bytes:%d + roi_bytes:%d = %d) exceeds source plane data width (%d bytes).\n",
                plane, src_offset_x_bytes, oSizeROI_npp.width, src_offset_x_bytes + oSizeROI_npp.width, plane_in_w_bytes);
            return AVERROR(EINVAL);
        }
        if (src_offset_y_lines + oSizeROI_npp.height > AV_CEIL_RSHIFT(in->height, vsub)) {
             av_log(ctx, AV_LOG_ERROR, "NPP Crop ROI for plane %d (offset_lines:%d, roi_lines:%d) exceeds source plane height (%d lines)\n",
                   plane, src_offset_y_lines, oSizeROI_npp.height, AV_CEIL_RSHIFT(in->height, vsub));
            return AVERROR(EINVAL);
        }

        npp_st = nppiCopy_8u_C1R_Ctx(pSrc_plane_roi_start, in->linesize[plane],
                                     pDst_plane_start, out->linesize[plane],
                                     oSizeROI_npp, npp_stream_ctx);

        if (npp_st != NPP_SUCCESS) {
            av_log(ctx, AV_LOG_ERROR, "NPP copy failed for plane %d with error %d. ROI_NPP: %dx%d, Format: %s\n",
                   plane, npp_st, oSizeROI_npp.width, oSizeROI_npp.height,
                   av_get_pix_fmt_name(hwctx_in->sw_format));
            return AVERROR_EXTERNAL;
        }
    }
    return 0;
}

static int nppcrop_filter_frame(AVFilterLink* inlink, AVFrame* in) {
    AVFilterContext* ctx = inlink->dst;
    NPPCropContext* s = ctx->priv;
    AVFilterLink* outlink = ctx->outputs[0];

    AVHWFramesContext *current_output_hw_frames_ctx = (AVHWFramesContext*)s->frames_ctx->data;
    AVCUDADeviceContext* device_hwctx = current_output_hw_frames_ctx->device_ctx->hwctx;
    int ret;

    s->in_w = in->width;
    s->in_h = in->height;

    AVFrame* out = av_frame_alloc();
    if (!out) {
        av_frame_free(&in);
        return AVERROR(ENOMEM);
    }

    out->format = in->format;
    ret = av_hwframe_get_buffer(s->frames_ctx, out, 0);
    if (ret < 0) {
        av_log(ctx, AV_LOG_ERROR, "Failed to get output GPU buffer: %s\n", av_err2str(ret));
        goto fail;
    }
    out->width = s->w;
    out->height = s->h;

    CUcontext cu_ctx_pop;
    ret = CHECK_CU(device_hwctx->internal->cuda_dl->cuCtxPushCurrent(device_hwctx->cuda_ctx));
    if (ret < 0) goto fail;

    ret = nppcrop_do_crop(ctx, out, in);

    CHECK_CU(device_hwctx->internal->cuda_dl->cuCtxPopCurrent(&cu_ctx_pop));

    if (ret < 0) goto fail;

    ret = av_frame_copy_props(out, in);
    if (ret < 0) goto fail;

    out->width = s->w;
    out->height = s->h;
    out->sample_aspect_ratio = s->out_sar;

    av_frame_free(&in);
    return ff_filter_frame(outlink, out);
fail:
    av_frame_free(&out);
    av_frame_free(&in);
    return ret;
}

#define OFFSET(x) offsetof(NPPCropContext, x)
#define FLAGS (AV_OPT_FLAG_FILTERING_PARAM | AV_OPT_FLAG_VIDEO_PARAM)

// clang-format off
static const AVOption nppcrop_options[] = {
    { "w",        "Output width",                 OFFSET(w_expr),      AV_OPT_TYPE_STRING, {.str = "iw"},             0,0, FLAGS },
    { "h",        "Output height",                OFFSET(h_expr),      AV_OPT_TYPE_STRING, {.str = "ih"},             0,0, FLAGS },
    { "x",        "X offset on input",            OFFSET(x_expr),      AV_OPT_TYPE_STRING, {.str = "(in_w-out_w)/2"}, 0,0, FLAGS },
    { "y",        "Y offset on input",            OFFSET(y_expr),      AV_OPT_TYPE_STRING, {.str = "(in_h-out_h)/2"}, 0,0, FLAGS },

    { "out_w",    "Output width (alias for w)",   OFFSET(w_expr),      AV_OPT_TYPE_STRING, {.str = NULL}, 0,0, FLAGS },
    { "out_h",    "Output height (alias for h)",  OFFSET(h_expr),      AV_OPT_TYPE_STRING, {.str = NULL}, 0,0, FLAGS },

    { "keep_aspect", "Keep display aspect ratio", OFFSET(keep_aspect), AV_OPT_TYPE_BOOL, {.i64=0}, 0, 1, FLAGS },
    { "exact", "Exact cropping for subsampled formats", OFFSET(exact), AV_OPT_TYPE_BOOL, {.i64=0}, 0, 1, FLAGS },
    { NULL }
};
// clang-format on

static const AVClass nppcrop_class = {
    .class_name = "crop_npp",
    .item_name  = av_default_item_name,
    .option     = nppcrop_options,
    .version    = LIBAVUTIL_VERSION_INT,
};

static const AVFilterPad nppcrop_inputs[] = {
    {
        .name         = "default",
        .type         = AVMEDIA_TYPE_VIDEO,
        .filter_frame = nppcrop_filter_frame,
    },
};

static const AVFilterPad nppcrop_outputs[] = {
    {
        .name         = "default",
        .type         = AVMEDIA_TYPE_VIDEO,
        .config_props = nppcrop_config_props,
    },
};

AVFilter ff_vf_crop_npp = {
    .name           = "crop_npp",
    .description    = NULL_IF_CONFIG_SMALL("NPP-based GPU video cropping filter."),
    .priv_size      = sizeof(NPPCropContext),
    .priv_class     = &nppcrop_class,
    .init           = nppcrop_init,
    .uninit         = nppcrop_uninit,
    FILTER_INPUTS(nppcrop_inputs),
    FILTER_OUTPUTS(nppcrop_outputs),
    FILTER_SINGLE_PIXFMT(AV_PIX_FMT_CUDA),
    .flags_internal = FF_FILTER_FLAG_HWFRAME_AWARE,
};