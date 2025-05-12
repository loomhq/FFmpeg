/*
 * Copyright (c) 2024 The FFmpeg Project
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
 * Copy an alpha component from another video's luma using CUDA.
 */

#include "libavutil/opt.h"
#include "libavutil/pixdesc.h"
#include "libavutil/hwcontext.h"
#include "libavutil/hwcontext_cuda_internal.h"
#include "libavutil/cuda_check.h"
#include "libavutil/mem.h"

#include "avfilter.h"
#include "filters.h"
#include "formats.h"
#include "framesync.h"
#include "internal.h"
#include "video.h"

#include "cuda/load_helper.h"

#define CHECK_CU(call) FF_CUDA_CHECK_DL(avctx, ctx->hwctx->internal->cuda_dl, call)
#define DIV_UP(a, b) ( ((a) + (b) - 1) / (b) )

// CUDA kernel launch parameters copied from overlay_cuda
#define BLOCK_X 32
#define BLOCK_Y 16

#define MAIN_INPUT 0
#define ALPHA_INPUT 1

#define ALPHA_LAYER_IDX 3

// clang-format off
static const enum AVPixelFormat supported_main_formats[] = {
    AV_PIX_FMT_YUVA444P,
    AV_PIX_FMT_YUVA420P,
    AV_PIX_FMT_NONE,
};

static const enum AVPixelFormat supported_alpha_mask_formats[] = {
    /*
    The alphamerge CPU filter expects a grayscale (GRAY8) input format since it only needs the luminence layer.
    Hoewever, GPU decoding and format conversions don't easily support grayscale so instead we can accept the following inputs and take the Y plane (luminence) to achieve the same result
    */
    AV_PIX_FMT_YUV420P,
    AV_PIX_FMT_YUV444P,
    AV_PIX_FMT_YUVA420P,
    AV_PIX_FMT_YUVA444P,
    AV_PIX_FMT_NV12,
    AV_PIX_FMT_NONE,
};
// clang-format on

typedef struct AlphaMergeCUDAContext {
    const AVClass *class;

    enum AVPixelFormat sw_format_main;
    enum AVPixelFormat sw_format_alpha_mask;

    AVBufferRef *hw_device_ctx;
    AVCUDADeviceContext *hwctx;

    CUcontext cu_ctx;
    CUmodule cu_module;
    CUfunction cu_func_planar;
    CUstream cu_stream;

    FFFrameSync fs;

    int alpha_plane_idx;

} AlphaMergeCUDAContext;


static int format_is_supported(const enum AVPixelFormat supported_formats[], enum AVPixelFormat fmt)
{
    for (int i = 0; supported_formats[i] != AV_PIX_FMT_NONE; i++)
        if (supported_formats[i] == fmt)
            return 1;
    return 0;
}

// We need a custom function to query formats and colorrange support since FILTER_SINGLE_PIXFMT(AV_PIX_FMT_CUDA) causes the filter to fail on some defined color ranges (bt470bg)
static int query_formats(AVFilterContext *avctx)
{
    static const int pix_fmts_cuda[] = { AV_PIX_FMT_CUDA, AV_PIX_FMT_NONE };

    // Supported colorspaces for the alpha input
    static const int alpha_input_supported_csp[] = {
        AVCOL_SPC_UNSPECIFIED,
        AVCOL_SPC_BT709,
        AVCOL_SPC_BT470BG,
        AVCOL_SPC_SMPTE170M,
        AVCOL_SPC_SMPTE240M,
        -1,                  
    };

    // Supported color ranges for the alpha input
    static const int alpha_input_supported_cr[] = {
        AVCOL_RANGE_UNSPECIFIED,
        AVCOL_RANGE_MPEG,
        AVCOL_RANGE_JPEG,
        -1
    };

    // Supported colorspaces for the main input
    static const int main_io_supported_csp[] = {
        AVCOL_SPC_UNSPECIFIED, AVCOL_SPC_RGB, AVCOL_SPC_BT709, AVCOL_SPC_BT470BG,
        AVCOL_SPC_SMPTE170M, AVCOL_SPC_SMPTE240M, AVCOL_SPC_BT2020_NCL, -1
    };

    // Supported color ranges for the main input
    static const int main_io_supported_cr[] = {
        AVCOL_RANGE_UNSPECIFIED, AVCOL_RANGE_MPEG, AVCOL_RANGE_JPEG, -1
    };

    AVFilterFormats *formats_list = NULL;
    int ret;

    // Macro to copy formats across main input, alpha input, and output
#define ASSIGN_FORMAT_LIST(target_list_ptr, source_array) do { \
    formats_list = ff_make_format_list(source_array); \
    if (!formats_list) { ret = AVERROR(ENOMEM); goto end; } \
    if ((ret = ff_formats_ref(formats_list, target_list_ptr)) < 0) { \
        av_freep(&formats_list); \
        goto end; \
    } \
    formats_list = NULL; \
} while (0)

    // Configure main input pad
    ASSIGN_FORMAT_LIST(&avctx->inputs[MAIN_INPUT]->outcfg.formats, pix_fmts_cuda);
    ASSIGN_FORMAT_LIST(&avctx->inputs[MAIN_INPUT]->outcfg.color_spaces, main_io_supported_csp);
    ASSIGN_FORMAT_LIST(&avctx->inputs[MAIN_INPUT]->outcfg.color_ranges, main_io_supported_cr);

    // Configure alpha input pad
    ASSIGN_FORMAT_LIST(&avctx->inputs[ALPHA_INPUT]->outcfg.formats, pix_fmts_cuda);
    ASSIGN_FORMAT_LIST(&avctx->inputs[ALPHA_INPUT]->outcfg.color_spaces, alpha_input_supported_csp);
    ASSIGN_FORMAT_LIST(&avctx->inputs[ALPHA_INPUT]->outcfg.color_ranges, alpha_input_supported_cr);

    // Configure output pad
    ASSIGN_FORMAT_LIST(&avctx->outputs[0]->incfg.formats, pix_fmts_cuda);
    ASSIGN_FORMAT_LIST(&avctx->outputs[0]->incfg.color_spaces, main_io_supported_csp);
    ASSIGN_FORMAT_LIST(&avctx->outputs[0]->incfg.color_ranges, main_io_supported_cr);

end:
    if (ret < 0) {
        av_freep(&formats_list);
    }
    return ret;
#undef ASSIGN_FORMAT_LIST
}


static int do_alphamerge_cuda(FFFrameSync *fs)
{
    AVFilterContext *avctx = fs->parent;
    AlphaMergeCUDAContext *ctx = avctx->priv;
    AVFilterLink *outlink = avctx->outputs[0];

    CudaFunctions *cu = ctx->hwctx->internal->cuda_dl;
    CUcontext dummy_cu_ctx;

    AVFrame *main_frame = NULL;
    AVFrame *alpha_mask_frame = NULL;
    int ret = 0;

    // Get synchronized frames from both inputs
    ret = ff_framesync_dualinput_get_writable(fs, &main_frame, &alpha_mask_frame);
    if (ret < 0)
        return ret;
    if (!main_frame)
        return AVERROR_BUG;

    // If the alpha input frames have ended we can just return the main input frame
    if (!alpha_mask_frame) {
        return ff_filter_frame(outlink, main_frame);
    }

    // Push the CUDA context
    ret = CHECK_CU(cu->cuCtxPushCurrent(ctx->cu_ctx));
    if (ret < 0) {
        av_frame_free(&main_frame);
        av_frame_free(&alpha_mask_frame);
        return ret;
    }

    // Prep CUDA launch params
    int width = main_frame->width;
    int height = main_frame->height;
    unsigned int grid_x = DIV_UP(width, BLOCK_X);
    unsigned int grid_y = DIV_UP(height, BLOCK_Y);

    // Get the luminence layer of the alpha input
    CUdeviceptr d_alpha_mask_luma_plane = (CUdeviceptr)alpha_mask_frame->data[0];
    int alpha_mask_luma_linesize = alpha_mask_frame->linesize[0];

    // Error copied from alphamerge
    if (alpha_mask_frame->color_range == AVCOL_RANGE_MPEG) {
        av_log(ctx, AV_LOG_WARNING, "alpha plane color range tagged as %s, output will be wrong!\n", 
            av_color_range_name(alpha_mask_frame->color_range));
    }

    // Get alpha plane from main input
    CUdeviceptr d_main_alpha_plane = (CUdeviceptr)main_frame->data[ctx->alpha_plane_idx];
    int main_alpha_linesize = main_frame->linesize[ctx->alpha_plane_idx];

    void *kernel_args[] = {
        &d_main_alpha_plane,
        &main_alpha_linesize,
        &d_alpha_mask_luma_plane,
        &alpha_mask_luma_linesize,
        &width,
        &height
    };
    ret = CHECK_CU(cu->cuLaunchKernel(ctx->cu_func_planar, grid_x, grid_y, 1,
                                        BLOCK_X, BLOCK_Y, 1,
                                        0, ctx->cu_stream, kernel_args, NULL));

    if (ret < 0) {
        av_log(avctx, AV_LOG_ERROR, "Failed to launch CUDA kernel\n");
        CHECK_CU(cu->cuCtxPopCurrent(&dummy_cu_ctx));
        av_frame_free(&main_frame);
        av_frame_free(&alpha_mask_frame);
        return ret;
    }

    // Pop the CUDA context
    ret = CHECK_CU(cu->cuCtxPopCurrent(&dummy_cu_ctx));
    if (ret < 0) {
        av_frame_free(&main_frame);
        av_frame_free(&alpha_mask_frame);
        return ret;
    }

    return ff_filter_frame(outlink, main_frame);
}


static int alphamerge_cuda_config_output(AVFilterLink *outlink)
{

    extern const unsigned char ff_vf_alphamerge_cuda_ptx_data[];
    extern const unsigned int ff_vf_alphamerge_cuda_ptx_len;

    AVFilterContext *avctx = outlink->src;
    AlphaMergeCUDAContext *ctx = avctx->priv;
    CudaFunctions *cu;
    CUcontext dummy_cu_ctx;
    int ret = 0;

    AVFilterLink *main_inlink = avctx->inputs[MAIN_INPUT];
    FilterLink *main_inl = ff_filter_link(main_inlink);
    AVHWFramesContext *main_frames_ctx = (AVHWFramesContext*)main_inl->hw_frames_ctx->data;

    AVFilterLink *alpha_inlink = avctx->inputs[ALPHA_INPUT];
    FilterLink *alpha_inl = ff_filter_link(alpha_inlink);
    AVHWFramesContext *alpha_frames_ctx = (AVHWFramesContext*)alpha_inl->hw_frames_ctx->data;

    // Validate main input
    if (!main_frames_ctx || main_frames_ctx->format != AV_PIX_FMT_CUDA) {
        av_log(avctx, AV_LOG_ERROR, "Main input requires CUDA HW frames.\n");
        return AVERROR(EINVAL);
    }
    ctx->sw_format_main = main_frames_ctx->sw_format;
    if (!format_is_supported(supported_main_formats, ctx->sw_format_main)) {
        av_log(avctx, AV_LOG_ERROR, "Unsupported main input software pixel format: %s\n",
               av_get_pix_fmt_name(ctx->sw_format_main));
        return AVERROR(ENOSYS);
    }

    // Validate alpha mask input
    if (!alpha_frames_ctx || alpha_frames_ctx->format != AV_PIX_FMT_CUDA) {
        av_log(avctx, AV_LOG_ERROR, "Alpha mask input requires CUDA HW frames.\n");
        return AVERROR(EINVAL);
    }
    ctx->sw_format_alpha_mask = alpha_frames_ctx->sw_format;
    if (!format_is_supported(supported_alpha_mask_formats, ctx->sw_format_alpha_mask)) {
        av_log(avctx, AV_LOG_ERROR, "Unsupported alpha mask input software pixel format: %s.\n",
               av_get_pix_fmt_name(ctx->sw_format_alpha_mask));
        return AVERROR(ENOSYS);
    }

    // Check for matching dimensions
    if (main_inlink->w != alpha_inlink->w || main_inlink->h != alpha_inlink->h) {
        av_log(avctx, AV_LOG_ERROR, "Input frame sizes do not match (%dx%d vs %dx%d).\n",
               main_inlink->w, main_inlink->h, alpha_inlink->w, alpha_inlink->h);
        return AVERROR(EINVAL);
    }

    // Set output properties
    outlink->w = main_inlink->w;
    outlink->h = main_inlink->h;
    outlink->time_base = main_inlink->time_base;
    outlink->sample_aspect_ratio = main_inlink->sample_aspect_ratio;
    FilterLink* outl_fl = ff_filter_link(outlink);
    FilterLink* main_inlink_fl = ff_filter_link(main_inlink);
    outl_fl->frame_rate = main_inlink_fl->frame_rate;


    // Initialize CUDA context and load module/kernels
    ctx->hw_device_ctx = av_buffer_ref(main_frames_ctx->device_ref);
    if (!ctx->hw_device_ctx)
        return AVERROR(ENOMEM);
    ctx->hwctx = ((AVHWDeviceContext*)ctx->hw_device_ctx->data)->hwctx;
    ctx->cu_ctx = ctx->hwctx->cuda_ctx;
    ctx->cu_stream = ctx->hwctx->stream;
    cu = ctx->hwctx->internal->cuda_dl;

    ret = CHECK_CU(cu->cuCtxPushCurrent(ctx->cu_ctx));
    if (ret < 0) return ret;

    ret = ff_cuda_load_module(avctx, ctx->hwctx, &ctx->cu_module,
                              ff_vf_alphamerge_cuda_ptx_data, ff_vf_alphamerge_cuda_ptx_len);
    if (ret < 0) {
        CHECK_CU(cu->cuCtxPopCurrent(&dummy_cu_ctx));
        return ret;
    }

    // Get CUDA kernel function
    ret = CHECK_CU(cu->cuModuleGetFunction(&ctx->cu_func_planar, ctx->cu_module, "AlphaMergePlanar"));
    if (ret < 0) {
        av_log(avctx, AV_LOG_ERROR, "Failed to get AlphaMergePlanar_Kernel function\n");
        CHECK_CU(cu->cuCtxPopCurrent(&dummy_cu_ctx));
        return ret;
    }

    CHECK_CU(cu->cuCtxPopCurrent(&dummy_cu_ctx));


    const AVPixFmtDescriptor *main_desc = av_pix_fmt_desc_get(ctx->sw_format_main);
    if (!main_desc) {
        av_log(avctx, AV_LOG_ERROR, "Could not get pixel format descriptor for %s\n",
               av_get_pix_fmt_name(ctx->sw_format_main));
        return AVERROR(EINVAL);
    }

    if ((main_desc->flags & AV_PIX_FMT_FLAG_PLANAR) && (main_desc->flags & AV_PIX_FMT_FLAG_ALPHA)) {
        if (main_desc->nb_components == 4 && main_desc->comp[ALPHA_LAYER_IDX].plane >= 0) {
            ctx->alpha_plane_idx = main_desc->comp[ALPHA_LAYER_IDX].plane;
        } else {
            av_log(avctx, AV_LOG_ERROR, "Unsupported planar format %s: expected 4 components, with component at index %d being the alpha plane. Found %d components.\n",
                 main_desc->name, ALPHA_LAYER_IDX, main_desc->nb_components);
            return AVERROR(ENOSYS);
        }
    } else {
        av_log(avctx, AV_LOG_ERROR, "Main input sw_format %s (%s) is not a recognized format with alpha\n",
               av_get_pix_fmt_name(ctx->sw_format_main), main_desc->name);
        return AVERROR(EINVAL);
    }

    // Output HW frames context
    outl_fl->hw_frames_ctx = av_buffer_ref(main_inl->hw_frames_ctx);
    if (!outl_fl->hw_frames_ctx)
        return AVERROR(ENOMEM);

    // framesync
    ctx->fs.time_base = main_inlink->time_base;
    ret = ff_framesync_init_dualinput(&ctx->fs, avctx);
    if (ret < 0)
        return ret;

    return ff_framesync_configure(&ctx->fs);
}


static av_cold int alphamerge_cuda_init(AVFilterContext *avctx)
{
    AlphaMergeCUDAContext *ctx = avctx->priv;
    ctx->fs.on_event = &do_alphamerge_cuda;
    return 0;
}


static av_cold void alphamerge_cuda_uninit(AVFilterContext *avctx)
{
    AlphaMergeCUDAContext *ctx = avctx->priv;
    CudaFunctions *cu = NULL;

    ff_framesync_uninit(&ctx->fs);

    // Sync stream and unload CUDA kernel
    if (ctx->hwctx && ctx->hwctx->internal) {
        cu = ctx->hwctx->internal->cuda_dl;
        if (cu && ctx->cu_ctx) {
            if (CHECK_CU(cu->cuCtxPushCurrent(ctx->cu_ctx)) == 0) {
                if (ctx->cu_stream) {
                    CHECK_CU(cu->cuStreamSynchronize(ctx->cu_stream));
                }
                if (ctx->cu_module) {
                    CHECK_CU(cu->cuModuleUnload(ctx->cu_module));
                }
                CUcontext dummy_cu_ctx_pop;
                CHECK_CU(cu->cuCtxPopCurrent(&dummy_cu_ctx_pop));
            } else {
                 av_log(ctx, AV_LOG_ERROR, "Failed to push CUDA context in uninit.\n");
            }
        }
    }
    ctx->cu_module = NULL; 

    av_buffer_unref(&ctx->hw_device_ctx);
    ctx->hw_device_ctx = NULL; 

    ctx->hwctx = NULL;         
    ctx->cu_ctx = NULL;        
    ctx->cu_stream = NULL;     
}

static int alphamerge_cuda_activate(AVFilterContext *avctx)
{
    AlphaMergeCUDAContext *ctx = avctx->priv;
    return ff_framesync_activate(&ctx->fs);
}


static int config_input_alpha_cuda(AVFilterLink *inlink)
{
    AVFilterContext *avctx = inlink->dst;
    FilterLink *inl = ff_filter_link(inlink);


    if (!inl->hw_frames_ctx) {
        av_log(avctx, AV_LOG_ERROR, "Alpha input pad '%s' requires a HW frames context.\n", inlink->dstpad->name);
        return AVERROR(EINVAL);
    }
    if (inlink->format != AV_PIX_FMT_CUDA) {
         av_log(avctx, AV_LOG_ERROR, "Alpha input pad '%s' must be AV_PIX_FMT_CUDA, got %s.\n",
                inlink->dstpad->name, av_get_pix_fmt_name(inlink->format));
        return AVERROR(EINVAL);
    }

    return 0;
}


#define OFFSET(x) offsetof(AlphaMergeCUDAContext, fs.opt_##x)
#define FLAGS (AV_OPT_FLAG_FILTERING_PARAM | AV_OPT_FLAG_VIDEO_PARAM)

static const AVOption alphamerge_cuda_options[] = {
    { NULL },
};

FRAMESYNC_DEFINE_CLASS(alphamerge_cuda, AlphaMergeCUDAContext, fs);


static const AVFilterPad alphamerge_cuda_inputs[] = {
    {
        .name = "main",
        .type = AVMEDIA_TYPE_VIDEO,
    },
    {
        .name = "alpha",
        .type = AVMEDIA_TYPE_VIDEO,
        .config_props = config_input_alpha_cuda,
    }
};


static const AVFilterPad alphamerge_cuda_outputs[] = {
    {
        .name = "default",
        .type = AVMEDIA_TYPE_VIDEO,
        .config_props = &alphamerge_cuda_config_output,
    }
};

const AVFilter ff_vf_alphamerge_cuda = {
    .name          = "alphamerge_cuda",
    .description   = NULL_IF_CONFIG_SMALL("Copy the luma value of the second input into the alpha channel of the first input using CUDA."),
    .priv_size     = sizeof(AlphaMergeCUDAContext),
    .priv_class    = &alphamerge_cuda_class,
    .init          = &alphamerge_cuda_init,
    .uninit        = &alphamerge_cuda_uninit,
    .activate      = &alphamerge_cuda_activate,
    FILTER_QUERY_FUNC(query_formats), 
    FILTER_INPUTS(alphamerge_cuda_inputs),
    FILTER_OUTPUTS(alphamerge_cuda_outputs),
    .preinit       = alphamerge_cuda_framesync_preinit,
    .flags_internal = FF_FILTER_FLAG_HWFRAME_AWARE,
};
