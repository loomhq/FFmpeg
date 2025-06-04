/*
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
 * NPP video padding video filter
 */

#include <float.h> /* DBL_MAX */
#include <nppi.h>

#include "filters.h"
#include "libavutil/internal.h"
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
    AV_PIX_FMT_NONE
};

typedef struct NPPPadContext {
    const AVClass* class;

    AVBufferRef* frames_ctx;

    int w, h; ///< output dimensions, a value of 0 will result in the input size
    int x, y; ///< offsets of the input area with respect to the padded area
    int in_w, in_h; ///< width and height for the padded input video, which has to be aligned to the chroma values in order to avoid chroma issues

    char* w_expr; ///< width  expression string
    char* h_expr; ///< height expression string
    char* x_expr; ///< width  expression string
    char* y_expr; ///< height expression string

    uint8_t rgba_color[4];   ///< color for the padding area
    uint8_t parsed_color[4]; ////< parsed color for use in npp functions
    AVRational aspect;

    int eval_mode;

    int last_out_w, last_out_h; ////< used to evaluate the prior output width and height with the incoming frame. a change in this would require GPU frame context to be reallocated
} NPPPadContext;
// clang-format on

static const char* const var_names[] = { "in_w",  "iw",   "in_h",  "ih",
                                         "out_w", "ow",   "out_h", "oh",
                                         "x",     "y",    "a",     "sar",
                                         "dar",   "hsub", "vsub",  NULL };

enum {
    VAR_IN_W,
    VAR_IW,
    VAR_IN_H,
    VAR_IH,
    VAR_OUT_W,
    VAR_OW,
    VAR_OUT_H,
    VAR_OH,
    VAR_X,
    VAR_Y,
    VAR_A,
    VAR_SAR,
    VAR_DAR,
    VAR_HSUB,
    VAR_VSUB,
    VARS_NB
};

enum EvalMode { EVAL_MODE_INIT, EVAL_MODE_FRAME, EVAL_MODE_NB };

/* Heavily borrowed from vf_pad.c config_input - Evaluates the user defined
 * padding expression
 */
static int eval_expr(AVFilterContext* ctx) {
    NPPPadContext* s = ctx->priv;
    AVFilterLink* inlink = ctx->inputs[0];
    const AVPixFmtDescriptor* desc = av_pix_fmt_desc_get(inlink->format);

    double var_values[VARS_NB], res;
    char* expr;
    int ret;

    var_values[VAR_IN_W] = var_values[VAR_IW] = s->in_w;
    var_values[VAR_IN_H] = var_values[VAR_IH] = s->in_h;
    var_values[VAR_OUT_W] = var_values[VAR_OW] = NAN;
    var_values[VAR_OUT_H] = var_values[VAR_OH] = NAN;
    var_values[VAR_A] = (double)s->in_w / s->in_h;
    var_values[VAR_SAR] = inlink->sample_aspect_ratio.num
                              ? (double)inlink->sample_aspect_ratio.num /
                                    inlink->sample_aspect_ratio.den
                              : 1;
    var_values[VAR_DAR] = var_values[VAR_A] * var_values[VAR_SAR];
    var_values[VAR_HSUB] = 1 << desc->log2_chroma_w;
    var_values[VAR_VSUB] = 1 << desc->log2_chroma_h;

    /* evaluate width */
    expr = s->w_expr;
    if ((ret = av_expr_parse_and_eval(&res, expr, var_names,
                                      var_values, NULL, NULL, NULL, NULL, NULL,
                                      0, ctx)) < 0)
        goto fail;

    s->w = res;
    if (s->w < 0) {
        av_log(ctx, AV_LOG_ERROR, "Width expression => negative.\n");
        ret = AVERROR(EINVAL);
        goto fail;
    }

    var_values[VAR_OUT_W] = var_values[VAR_OW] = s->w;

    /* evaluate height */
    expr = s->h_expr;
    if ((ret = av_expr_parse_and_eval(&res, expr, var_names, var_values, NULL, NULL, NULL, NULL, NULL, 0, ctx)) < 0)
        goto fail;

    s->h = res;
    if (s->h < 0) {
        av_log(ctx, AV_LOG_ERROR, "Height expression => negative.\n");
        ret = AVERROR(EINVAL);
        goto fail;
    }
    var_values[VAR_OUT_H] = var_values[VAR_OH] = s->h;

    if (!s->h)
        s->h = s->in_h;
    var_values[VAR_OUT_H] = var_values[VAR_OH] = s->h;

    /* evaluate the width again, as it may depend on the evaluated output height */
    expr = s->w_expr;
    if ((ret = av_expr_parse_and_eval(&res, expr, var_names, var_values, NULL, NULL, NULL, NULL, NULL, 0, ctx)) < 0)
        goto fail;

    s->w = res;
    if (s->w < 0) {
        av_log(ctx, AV_LOG_ERROR, "Width expression => negative.\n");
        ret = AVERROR(EINVAL);
        goto fail;
    }
    if (!s->w)
        s->w = s->in_w;
    var_values[VAR_OUT_W] = var_values[VAR_OW] = s->w;

    /* Evaluate x */
    expr = s->x_expr;
    if ((ret = av_expr_parse_and_eval(&res, expr, var_names, var_values, NULL, NULL, NULL, NULL, NULL, 0, ctx)) < 0)
        goto fail;

    s->x = res;

    /* Evaluate y */
    expr = s->y_expr;
    if ((ret = av_expr_parse_and_eval(&res, (expr = s->y_expr), var_names, var_values, NULL, NULL, NULL, NULL, NULL, 0, ctx)) < 0)
        goto fail;

    s->y = res;

    /* center x if needed */
    if (s->x < 0 || s->x + s->in_w > s->w) {
        s->x = (s->w - s->in_w) / 2;
        av_log(ctx, AV_LOG_INFO, "Centering X offset.\n");
    }

    /* center y if needed */
    if (s->y < 0 || s->y + s->in_h > s->h) {
        s->y = (s->h - s->in_h) / 2;
        av_log(ctx, AV_LOG_INFO, "Centering Y offset.\n");
    }

    s->w = av_clip(s->w, 1, INT_MAX);
    s->h = av_clip(s->h, 1, INT_MAX);

    /* sanity check params */
    if (s->w < s->in_w || s->h < s->in_h) {
        av_log(ctx, AV_LOG_ERROR, "Padded size < input size.\n");
        return AVERROR(EINVAL);
    }

    av_log(ctx, AV_LOG_DEBUG,
           "w:%d h:%d -> w:%d h:%d x:%d y:%d color:0x%02X%02X%02X%02X\n",
           inlink->w, inlink->h, s->w, s->h, s->x, s->y, s->rgba_color[0],
           s->rgba_color[1], s->rgba_color[2], s->rgba_color[3]);

    return 0;

fail:
    av_log(ctx, AV_LOG_ERROR, "Error evaluating '%s'\n", expr);
    return ret;
}

static int npppad_alloc_out_frames_ctx(AVFilterContext *ctx, AVBufferRef **out_frames_ctx, const int width, const int height)
{
    AVFilterLink *inlink = ctx->inputs[0];
    FilterLink* inl = ff_filter_link(inlink);
    AVHWFramesContext *in_frames_ctx = (AVHWFramesContext*)inl->hw_frames_ctx->data;
    int ret;

    *out_frames_ctx = av_hwframe_ctx_alloc(in_frames_ctx->device_ref);
    if (!*out_frames_ctx) {
        return AVERROR(ENOMEM);
    }

    AVHWFramesContext *out_fc = (AVHWFramesContext*)(*out_frames_ctx)->data;
    out_fc->format    = AV_PIX_FMT_CUDA;
    out_fc->sw_format = in_frames_ctx->sw_format;
    /* CUDA/NPP work best with 32-pixel alignment. Existing CUDA/CPP
   * filters do the same prior to calling GPU specific functions;
   * see vf_sharpen_npp.c */
    out_fc->width     = FFALIGN(width, 32);
    out_fc->height    = FFALIGN(height, 32);

    ret = av_hwframe_ctx_init(*out_frames_ctx);
    if (ret < 0) {
        av_log(ctx, AV_LOG_ERROR, "Failed to init output hwframes ctx: %s\n",
               av_err2str(ret));
        av_buffer_unref(out_frames_ctx);
        return ret;
    }

    return 0;
}

static int npppad_init(AVFilterContext* ctx) {
    NPPPadContext* npp_pad_context = ctx->priv;
    if (!npp_pad_context) {
        av_log(ctx, AV_LOG_ERROR, "Failed to allocate NPPPadContext.\n");
        return AVERROR(ENOMEM);
    }

    npp_pad_context->last_out_w = -1;
    npp_pad_context->last_out_h = -1;

    return 0;
}

static void npppad_uninit(AVFilterContext* ctx) {
    NPPPadContext* npp_pad_context = ctx->priv;
    av_buffer_unref(&npp_pad_context->frames_ctx);
}

static int npppad_config_props(AVFilterLink* outlink) {
    AVFilterContext* ctx = outlink->src;
    NPPPadContext* npp_pad_context = ctx->priv;

    AVFilterLink* inlink = ctx->inputs[0];
    FilterLink* inl = ff_filter_link(inlink);

    AVHWFramesContext* in_frames_ctx;
    int format_supported = 0;
    int ret;

    /* evaluate expressions for initial frame */
    npp_pad_context->in_w = inlink->w;
    npp_pad_context->in_h = inlink->h;
    ret = eval_expr(ctx);
    if (ret < 0)
        return ret;

    /* Must have CUDA frames */
    if (!inl->hw_frames_ctx) {
        av_log(ctx, AV_LOG_ERROR, "Input must be CUDA frames.\n");
        return AVERROR(EINVAL);
    }

    in_frames_ctx = (AVHWFramesContext*)inl->hw_frames_ctx->data;

    /* Check format */
    for (int i = 0; i < FF_ARRAY_ELEMS(supported_formats); i++) {
        if (in_frames_ctx->sw_format == supported_formats[i]) {
            format_supported = 1;
            break;
        }
    }
    if (!format_supported) {
        av_log(ctx, AV_LOG_ERROR, "Unsupported input format.\n");
        return AVERROR(EINVAL);
    }

    /* evaluate color value */
    uint8_t R = npp_pad_context->rgba_color[0];
    uint8_t G = npp_pad_context->rgba_color[1];
    uint8_t B = npp_pad_context->rgba_color[2];
    /* limited-range integer formula - couldn't find an implementation like
     * this in ffmpegs swscale.c or input.c. This was taken from
     * https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion. Asked
     * google Gemini to put the formula together
     */
    int Y = ((66 * R + 129 * G + 25 * B + 128) >> 8) + 16;
    int U = ((-38 * R - 74 * G + 112 * B + 128) >> 8) + 128;
    int V = ((112 * R - 94 * G - 18 * B + 128) >> 8) + 128;
    npp_pad_context->parsed_color[0] = av_clip_uint8(Y);
    npp_pad_context->parsed_color[1] = av_clip_uint8(U);
    npp_pad_context->parsed_color[2] = av_clip_uint8(V);
    npp_pad_context->parsed_color[3] = npp_pad_context->rgba_color[3]; // Keep alpha from input color

    /* Create output frame context */
    ret = npppad_alloc_out_frames_ctx(ctx, &npp_pad_context->frames_ctx, npp_pad_context->w, npp_pad_context->h);
    if (ret < 0)
        return ret;

    /* set outlink context and assign GPU frame context */
    {
        FilterLink* ol = ff_filter_link(outlink);
        ol->hw_frames_ctx = av_buffer_ref(npp_pad_context->frames_ctx);
        if (!ol->hw_frames_ctx)
            return AVERROR(ENOMEM);

        outlink->w = npp_pad_context->w;
        outlink->h = npp_pad_context->h;
        outlink->time_base = inlink->time_base;
        outlink->format = AV_PIX_FMT_CUDA;
    }

    /* we track these to detect changes in frames for frame based expression
     * evaluation */
    npp_pad_context->last_out_w = npp_pad_context->w;
    npp_pad_context->last_out_h = npp_pad_context->h;

    return 0;
}

static int npppad_pad(AVFilterContext* ctx, AVFrame* out, const AVFrame* in) {
    NPPPadContext* npp_pad_context = ctx->priv;
    FilterLink* inl = ff_filter_link(ctx->inputs[0]);

    AVHWFramesContext* in_frames_ctx =
        (AVHWFramesContext*)inl->hw_frames_ctx->data;
    const AVPixFmtDescriptor* desc_in =
        av_pix_fmt_desc_get(in_frames_ctx->sw_format);

    /* get format planes and pixel components - mostly pulled from
     * vf_colorlevels.c */
    const int nb_planes = av_pix_fmt_count_planes(in_frames_ctx->sw_format);
    const int nb_components = desc_in->nb_components;

    for (int plane = 0; plane < nb_planes; plane++) {
        /* Consider YUV chroma if applicable - taken from vf_sharpen_npp.c */
        int hsub = (plane == 1 || plane == 2) ? desc_in->log2_chroma_w : 0;
        int vsub = (plane == 1 || plane == 2) ? desc_in->log2_chroma_h : 0;

        // NV12 UV plane subsampling
        if (in_frames_ctx->sw_format == AV_PIX_FMT_NV12 && plane == 1) {
            // Both should be 1 for NV12
            hsub = desc_in->log2_chroma_w;
            vsub = desc_in->log2_chroma_h;
        }

        /*  values we need to calculate for nppiCopyConstBorder_8u_C1R
         * https://docs.nvidia.com/cuda/npp/image_data_exchange_and_initialization.html#group__image__copy__constant__border_1image_copy_constant_border
         */

        /* source plane size */
        int srcW = AV_CEIL_RSHIFT(npp_pad_context->in_w, hsub);
        int srcH = AV_CEIL_RSHIFT(npp_pad_context->in_h, vsub);

        /* destination plane size */
        int dstW = AV_CEIL_RSHIFT(npp_pad_context->w, hsub);
        int dstH = AV_CEIL_RSHIFT(npp_pad_context->h, vsub);

        /* x and y offset */
        int y_plane_offset = AV_CEIL_RSHIFT(npp_pad_context->y, vsub);
        int x_plane_offset = AV_CEIL_RSHIFT(npp_pad_context->x, hsub);

        /* sanity check boundaries */
        if (x_plane_offset + srcW > dstW || y_plane_offset + srcH > dstH) {
            av_log(ctx, AV_LOG_ERROR,
                   "ROI out of bounds in plane %d: offset=(%d,%d) in=(%dx%d) "
                   "out=(%dx%d)\n",
                   plane, x_plane_offset, y_plane_offset, srcW, srcH, dstW, dstH);
            return AVERROR(EINVAL);
        }

        NppStatus st;

        // UV plane (NV12)
        // There is no nppiCopyConstBorder function that can handle a UV pair so instead
        // we have to create the color plane with nppiSet_8u_C2R and then copy over the existing UV plane to our newly created plane
        if (in_frames_ctx->sw_format == AV_PIX_FMT_NV12 && plane == 1) {
            // srcW/dstW are in UV-pair counts for this plane. x_plane_offset is also in UV-pairs.
            Npp8u fillValUV[2] = {npp_pad_context->parsed_color[1], npp_pad_context->parsed_color[2]}; // U, V
            NppiSize oFullDstPlaneSizeUV = { dstW, dstH };

            st = nppiSet_8u_C2R(fillValUV,
                                     out->data[plane],
                                     out->linesize[plane],
                                     oFullDstPlaneSizeUV);


            if (st != NPP_SUCCESS) {
                av_log(ctx, AV_LOG_ERROR,
                       "nppiSet_8u_C2R plane=%d error=%d\n", plane,
                       st);
                return AVERROR_EXTERNAL;
            }


            if (srcW > 0 && srcH > 0) { // Only copy if there's source data for this plane
                NppiSize oSrcROISizeBytesNV12UV = { srcW * 2, srcH }; // Width in bytes for C1R copy
                Npp8u *pDstROIStart = out->data[plane] + (y_plane_offset * out->linesize[plane]) + (x_plane_offset * 2); // Byte offset

                st = nppiCopy_8u_C1R(
                    in->data[plane], in->linesize[plane],
                    pDstROIStart, out->linesize[plane],
                    oSrcROISizeBytesNV12UV);

                if (st != NPP_SUCCESS) {
                    av_log(ctx, AV_LOG_ERROR,
                           "nppiCopy_8u_C1R plane=%d error=%d\n", plane,
                           st);
                    return AVERROR_EXTERNAL;
                }
            }
        } else {
            // Y plane (all formats) or U/V planes for YUV420P/YUV444P
            Npp8u fillVal = npp_pad_context->parsed_color[plane];
            NppiSize oSrcSizeROI_C1 = { srcW, srcH };
            NppiSize oDstSizeROI_C1 = { dstW, dstH };

            st = nppiCopyConstBorder_8u_C1R(
                in->data[plane], in->linesize[plane], oSrcSizeROI_C1,
                out->data[plane], out->linesize[plane], oDstSizeROI_C1,
                y_plane_offset, x_plane_offset, fillVal);

            if (st != NPP_SUCCESS) {
                av_log(ctx, AV_LOG_ERROR,
                       "nppiCopyConstBorder_8u_C1R plane=%d error=%d\n", plane,
                       st);
                return AVERROR_EXTERNAL;
            }
        }
    }

    return 0;
}

static int npppad_filter_frame(AVFilterLink* inlink, AVFrame* in) {
    AVFilterContext* ctx = inlink->dst;
    NPPPadContext* npp_pad_context = ctx->priv;
    AVFilterLink* outlink = ctx->outputs[0];

    FilterLink* outl = ff_filter_link(outlink);

    AVHWFramesContext* out_frames_ctx =
        (AVHWFramesContext*)outl->hw_frames_ctx->data;
    AVCUDADeviceContext* device_hwctx = out_frames_ctx->device_ctx->hwctx;

    int ret;

    /* re-evaluate input expression for every frame if per-frame mode is
     * selected */
    if (npp_pad_context->eval_mode == EVAL_MODE_FRAME) {
        npp_pad_context->in_w = in->width;
        npp_pad_context->in_h = in->height;
        npp_pad_context->aspect = in->sample_aspect_ratio;

        ret = eval_expr(ctx);
        if (ret < 0) {
            av_frame_free(&in);
            return ret;
        }
    }

    /* passthrough if possible (no border requested) */
    if (npp_pad_context->x == 0 && npp_pad_context->y == 0 &&
        npp_pad_context->w == in->width &&
        npp_pad_context->h == in->height) {
        av_log(ctx, AV_LOG_DEBUG, "No border => pass frame unmodified.\n");
        npp_pad_context->last_out_w = npp_pad_context->w;
        npp_pad_context->last_out_h = npp_pad_context->h;
        return ff_filter_frame(outlink, in);
    }

    /* if width or height has changed re-initialize context */
    if (npp_pad_context->w != npp_pad_context->last_out_w ||
        npp_pad_context->h != npp_pad_context->last_out_h) {

        /* re allocate frame context */
        av_buffer_unref(&npp_pad_context->frames_ctx);

        ret = npppad_alloc_out_frames_ctx(ctx, &npp_pad_context->frames_ctx, npp_pad_context->w, npp_pad_context->h);
        if (ret < 0)
            return ret;

        /* update output HW frame context */
        av_buffer_unref(&outl->hw_frames_ctx);
        outl->hw_frames_ctx = av_buffer_ref(npp_pad_context->frames_ctx);
        if (!outl->hw_frames_ctx) {
            av_frame_free(&in);
            av_log(ctx, AV_LOG_ERROR, "Failed to allocate HW AVBuffer.\n");
            return AVERROR(ENOMEM);
        }
        outlink->w = npp_pad_context->w;
        outlink->h = npp_pad_context->h;

        npp_pad_context->last_out_w = npp_pad_context->w;
        npp_pad_context->last_out_h = npp_pad_context->h;
    }

    /* allocate output GPU frame */
    AVFrame* out = av_frame_alloc();
    if (!out) {
        av_frame_free(&in);
        av_log(ctx, AV_LOG_ERROR, "Failed to allocate output AVFrame.\n");
        return AVERROR(ENOMEM);
    }
    ret = av_hwframe_get_buffer(outl->hw_frames_ctx, out, 0);
    if (ret < 0) {
        av_log(ctx, AV_LOG_ERROR, "Unable to get output GPU buffer: %s\n",
               av_err2str(ret));
        av_frame_free(&out);
        av_frame_free(&in);
        return ret;
    }

    /* push CUDA context - required before running CUDA/NPP implementations on
     * the current frame
     */
    CUcontext dummy;
    ret = CHECK_CU(device_hwctx->internal->cuda_dl->cuCtxPushCurrent(
        device_hwctx->cuda_ctx));
    if (ret < 0) {
        av_frame_free(&out);
        av_frame_free(&in);
        return ret;
    }

    /* pad */
    ret = npppad_pad(ctx, out, in);

    /* pop CUDA context - prepare for the next frame */
    CHECK_CU(device_hwctx->internal->cuda_dl->cuCtxPopCurrent(&dummy));

    if (ret < 0) {
        av_frame_free(&out);
        av_frame_free(&in);
        return ret;
    }

    /* copy metadata and set output size */
    av_frame_copy_props(out, in);
    out->width = npp_pad_context->w;
    out->height = npp_pad_context->h;

    /*  adjust aspect ratio */
    av_reduce(&out->sample_aspect_ratio.num, &out->sample_aspect_ratio.den,
              (int64_t)in->sample_aspect_ratio.num * out->height * in->width,
              (int64_t)in->sample_aspect_ratio.den * out->width * in->height,
              INT_MAX);

    av_frame_free(&in);
    return ff_filter_frame(outlink, out);
}

#define OFFSET(x) offsetof(NPPPadContext, x)
#define FLAGS (AV_OPT_FLAG_FILTERING_PARAM | AV_OPT_FLAG_VIDEO_PARAM)

// clang-format off
/* options copied from vf_pad.c */
static const AVOption npppad_options[] = {
    { "width",  "set the pad area width expression",       OFFSET(w_expr), AV_OPT_TYPE_STRING, {.str = "iw"}, 0, 0, FLAGS },
    { "w",      "set the pad area width expression",       OFFSET(w_expr), AV_OPT_TYPE_STRING, {.str = "iw"}, 0, 0, FLAGS },
    { "height", "set the pad area height expression",      OFFSET(h_expr), AV_OPT_TYPE_STRING, {.str = "ih"}, 0, 0, FLAGS },
    { "h",      "set the pad area height expression",      OFFSET(h_expr), AV_OPT_TYPE_STRING, {.str = "ih"}, 0, 0, FLAGS },
    { "x",      "set the x offset expression for the input image position", OFFSET(x_expr), AV_OPT_TYPE_STRING, {.str = "0"}, 0, 0, FLAGS },
    { "y",      "set the y offset expression for the input image position", OFFSET(y_expr), AV_OPT_TYPE_STRING, {.str = "0"}, 0, 0, FLAGS },
    { "color",  "set the color of the padded area border", OFFSET(rgba_color), AV_OPT_TYPE_COLOR, {.str = "black"}, .flags = FLAGS },
    { "eval",   "specify when to evaluate expressions",    OFFSET(eval_mode), AV_OPT_TYPE_INT, {.i64 = EVAL_MODE_INIT}, 0, EVAL_MODE_NB-1, FLAGS, .unit = "eval" },
         { "init",  "eval expressions once during initialization", 0, AV_OPT_TYPE_CONST, {.i64=EVAL_MODE_INIT},  .flags = FLAGS, .unit = "eval" },
         { "frame", "eval expressions during initialization and per-frame", 0, AV_OPT_TYPE_CONST, {.i64=EVAL_MODE_FRAME}, .flags = FLAGS, .unit = "eval" },
    { "aspect",  "pad to fit an aspect instead of a resolution", OFFSET(aspect), AV_OPT_TYPE_RATIONAL, {.dbl = 0}, 0, DBL_MAX, FLAGS },
    { NULL }
};
// clang-format on

static const AVClass npppad_class = {
    .class_name = "pad_npp",
    .item_name = av_default_item_name,
    .option = npppad_options,
    .version = LIBAVUTIL_VERSION_INT,
};

static const AVFilterPad npppad_inputs[] = {
    {
        .name = "default",
        .type = AVMEDIA_TYPE_VIDEO,
        .filter_frame = npppad_filter_frame,
    },
};

static const AVFilterPad npppad_outputs[] = {
    {
        .name = "default",
        .type = AVMEDIA_TYPE_VIDEO,
        .config_props = npppad_config_props,
    },
};

AVFilter ff_vf_pad_npp = {
    .name = "pad_npp",
    .description = NULL_IF_CONFIG_SMALL("NPP-based GPU padding filter"),
    .init = npppad_init,
    .uninit = npppad_uninit,

    FILTER_INPUTS(npppad_inputs),
    FILTER_OUTPUTS(npppad_outputs),

    FILTER_SINGLE_PIXFMT(AV_PIX_FMT_CUDA),

    .priv_size = sizeof(NPPPadContext),
    .priv_class = &npppad_class,
    .flags_internal = FF_FILTER_FLAG_HWFRAME_AWARE,
};