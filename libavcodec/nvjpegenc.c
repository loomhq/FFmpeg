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
 * NVIDIA nvJPEG encoder
 */

#include <nvjpeg.h>

#include "codec_internal.h"
#include "libavcodec/avcodec.h"
#include "libavutil/frame.h"
#include "libavutil/hwcontext.h"
#include "libavutil/imgutils.h"
#include "libavutil/log.h"
#include "libavutil/mem.h"
#include "libavutil/opt.h"

static inline int ff_nvjpeg_check(void* avctx, nvjpegStatus_t err,
                                  const char* func) {
    av_log(avctx, AV_LOG_TRACE, "Calling %s\n", func);

    if (err == NVJPEG_STATUS_SUCCESS)
        return 0;

    av_log(avctx, AV_LOG_ERROR, "%s failed with error code: %d\n", func, err);

    return AVERROR_EXTERNAL;
}

#define CHECK_NVJPEG(x)                                                        \
    do {                                                                       \
        nvjpegStatus_t status = (x);                                           \
        int ret = ff_nvjpeg_check(avctx, status, #x);                          \
        if (ret < 0)                                                           \
            return ret;                                                        \
    } while (0)

typedef struct NvjpegContext {
    const AVClass* class;
    nvjpegHandle_t nvjpeg_handle;
    nvjpegEncoderState_t encoder_state;
    nvjpegEncoderParams_t encode_params;
    nvjpegJpegState_t jpeg_state;
    int width;
    int height;
    nvjpegChromaSubsampling_t subsampling;
    int quality;
    int huffman_optimize;
} NvjpegContext;

typedef enum NvjpegInputType {
    NVJPEG_INPUT_TYPE_YUV,
    NVJPEG_INPUT_TYPE_RGB,
    NVJPEG_INPUT_TYPE_UNSUPPORTED
} NvjpegInputType;

static NvjpegInputType get_nvjpeg_input_format(enum AVPixelFormat pix_fmt) {
    switch (pix_fmt) {
    case AV_PIX_FMT_YUV420P:
    case AV_PIX_FMT_YUVJ420P:
    case AV_PIX_FMT_YUV422P:
    case AV_PIX_FMT_YUVJ422P:
    case AV_PIX_FMT_YUV444P:
    case AV_PIX_FMT_YUVJ444P:
        return NVJPEG_INPUT_TYPE_YUV;
    case AV_PIX_FMT_RGB24:
    case AV_PIX_FMT_BGR24:
    case AV_PIX_FMT_RGBA:
    case AV_PIX_FMT_BGRA:
        return NVJPEG_INPUT_TYPE_RGB;
    default:
        return NVJPEG_INPUT_TYPE_UNSUPPORTED;
    }
}

static int nvjpeg_init(AVCodecContext* avctx) {
    NvjpegContext* ctx = avctx->priv_data;

    // NULL passed in for memory allocator. This will make us use the CUDA
    // defaults cudaMalloc and cudaFree:
    // https://docs.nvidia.com/cuda/nvjpeg/index.html#nvjpeg-device-memory-allocator-interface
    CHECK_NVJPEG(
        nvjpegCreate(NVJPEG_BACKEND_DEFAULT, NULL, &ctx->nvjpeg_handle));

    // Create encoder and set parameters
    CHECK_NVJPEG(nvjpegEncoderStateCreate(ctx->nvjpeg_handle,
                                          &ctx->encoder_state, NULL));

    CHECK_NVJPEG(nvjpegEncoderParamsCreate(ctx->nvjpeg_handle,
                                           &ctx->encode_params, NULL));

    CHECK_NVJPEG(nvjpegJpegStateCreate(ctx->nvjpeg_handle, &ctx->jpeg_state));

    CHECK_NVJPEG(
        nvjpegEncoderParamsSetQuality(ctx->encode_params, ctx->quality, NULL));

    CHECK_NVJPEG(nvjpegEncoderParamsSetOptimizedHuffman(
        ctx->encode_params, ctx->huffman_optimize, NULL));

    CHECK_NVJPEG(nvjpegEncoderParamsSetSamplingFactors(ctx->encode_params,
                                                       ctx->subsampling, NULL));

    return 0;
}

static int nvjpeg_encode_frame(AVCodecContext* avctx, AVPacket* pkt,
                               const AVFrame* frame, int* got_packet) {
    NvjpegContext* ctx = avctx->priv_data;
    nvjpegImage_t nv_image;
    size_t out_buf_size;
    uint8_t* out_buf = NULL;
    int ret = 0;
    int i;

    AVHWFramesContext* hw_frames_ctx =
        (AVHWFramesContext*)frame->hw_frames_ctx->data;
    enum AVPixelFormat sw_format = hw_frames_ctx->sw_format;

    // Check the CUDA sw format
    int input_format = get_nvjpeg_input_format(sw_format);
    if (input_format == NVJPEG_INPUT_TYPE_UNSUPPORTED) {
        av_log(avctx, AV_LOG_ERROR, "Unsupported pixel format: %s\n",
               av_get_pix_fmt_name(sw_format));
        return AVERROR(EINVAL);
    }

    // Initialize default channel and pitch
    for (i = 0; i < NVJPEG_MAX_COMPONENT; i++) {
        nv_image.channel[i] = NULL;
        nv_image.pitch[i] = 0;
    }

    // Handle YUV input formats and populate nv_image data
    if (sw_format == AV_PIX_FMT_YUV420P || sw_format == AV_PIX_FMT_YUVJ420P
        || AV_PIX_FMT_YUV422P || AV_PIX_FMT_YUVJ422P) {
        nv_image.channel[0] = frame->data[0];
        nv_image.pitch[0] = frame->linesize[0];
        nv_image.channel[1] = frame->data[1];
        nv_image.pitch[1] = frame->linesize[1];
        nv_image.channel[2] = frame->data[2];
        nv_image.pitch[2] = frame->linesize[2];
    } else if (input_format != NVJPEG_INPUT_TYPE_RGB) {
        // Handle BGR/RGB input formats
        for (i = 0; i < NVJPEG_MAX_COMPONENT; i++) {
            nv_image.channel[i] = frame->data[i];
            nv_image.pitch[i] = frame->linesize[i];
        }
    } else {
        av_log(avctx, AV_LOG_ERROR, "Unsupported pixel format: %s\n",
               av_get_pix_fmt_name(sw_format));
        return AVERROR(EINVAL);
    }

    // Handle subsampling if applicable and encode
    if (input_format == NVJPEG_INPUT_TYPE_YUV) {
        nvjpegChromaSubsampling_t subsampling;
        switch (ctx->subsampling) {
        case NVJPEG_CSS_444:
            subsampling = NVJPEG_CSS_444;
            break;
        case NVJPEG_CSS_422:
            subsampling = NVJPEG_CSS_422;
            break;
        case NVJPEG_CSS_420:
            subsampling = NVJPEG_CSS_420;
            break;
        case NVJPEG_CSS_411:
            subsampling = NVJPEG_CSS_411;
            break;
        case NVJPEG_CSS_GRAY:
            subsampling = NVJPEG_CSS_GRAY;
            break;
        case NVJPEG_CSS_410:
            subsampling = NVJPEG_CSS_410;
            break;
        default:
            subsampling = NVJPEG_CSS_UNKNOWN;
            break;
        }
        if (subsampling == NVJPEG_CSS_UNKNOWN) {
            av_log(avctx, AV_LOG_ERROR, "Unsupported chroma subsampling: %d\n",
                   ctx->subsampling);
            return AVERROR(EINVAL);
        }

        CHECK_NVJPEG(nvjpegEncodeYUV(ctx->nvjpeg_handle, ctx->encoder_state,
                                     ctx->encode_params, &nv_image, subsampling,
                                     ctx->width, ctx->height, NULL));
    } else {
        // RGB/BGR format
        CHECK_NVJPEG(nvjpegEncodeImage(ctx->nvjpeg_handle, ctx->encoder_state,
                                       ctx->encode_params, &nv_image,
                                       (nvjpegInputFormat_t)input_format,
                                       ctx->width, ctx->height, NULL));
    }

    // Retrieve the bitstream and get the output buffer size
    CHECK_NVJPEG(nvjpegEncodeRetrieveBitstream(
        ctx->nvjpeg_handle, ctx->encoder_state, NULL, &out_buf_size, NULL));

    out_buf = av_malloc(out_buf_size);
    if (!out_buf) {
        av_log(avctx, AV_LOG_ERROR, "Failed to allocate output buffer\n");
        return AVERROR(ENOMEM);
    }

    // Retrieve the bitstream again to populate output buffer
    ret = CHECK_NVJPEG(nvjpegEncodeRetrieveBitstream(
        ctx->nvjpeg_handle, ctx->encoder_state, out_buf, &out_buf_size, NULL));

    if (ret < 0) {
    	av_free(out_buf);
		return AVERROR_EXTERNAL;
    }

    ret = av_packet_from_data(pkt, out_buf, out_buf_size);
    if (ret < 0) {
        av_free(out_buf);
        av_log(avctx, AV_LOG_ERROR, "av_packet_from_data failed\n");
        return ret;
    }

    *got_packet = 1;
    return 0;
}

static av_cold int nvjpeg_close(AVCodecContext* avctx) {
    NvjpegContext* ctx = avctx->priv_data;

    if (ctx->encoder_state) {
        CHECK_NVJPEG(nvjpegEncoderStateDestroy(ctx->encoder_state));
    }

    if (ctx->encode_params) {
        CHECK_NVJPEG(nvjpegEncoderParamsDestroy(ctx->encode_params));
    }

    if (ctx->jpeg_state) {
        CHECK_NVJPEG(nvjpegJpegStateDestroy(ctx->jpeg_state));
    }

    if (ctx->nvjpeg_handle) {
        CHECK_NVJPEG(nvjpegDestroy(ctx->nvjpeg_handle));
    }
    return 0;
}

#define OFFSET(x) offsetof(NvjpegContext, x)
#define VE AV_OPT_FLAG_VIDEO_PARAM | AV_OPT_FLAG_ENCODING_PARAM

// clang-format off
static const AVOption options[] = {
    {"quality", "Set JPEG quality (0-100)", OFFSET(quality), AV_OPT_TYPE_INT, {.i64 = 75}, 0, 100, VE},
    {"huffman_optimize", "Enable Huffman optimization", OFFSET(huffman_optimize), AV_OPT_TYPE_BOOL, {.i64 = 0}, 0, 1, VE},
    {"subsampling", "Set chroma subsampling", OFFSET(subsampling), AV_OPT_TYPE_INT, {.i64 = NVJPEG_CSS_420}, 0, 5, VE, "subsampling"},
        {"444", "YUV 4:4:4", 0, AV_OPT_TYPE_CONST, {.i64 = NVJPEG_CSS_444}, INT_MIN, INT_MAX, VE, "subsampling"},
        {"422", "YUV 4:2:2", 0, AV_OPT_TYPE_CONST, {.i64 = NVJPEG_CSS_422}, INT_MIN, INT_MAX, VE, "subsampling"},
        {"420","YUV 4:2:0", 0, AV_OPT_TYPE_CONST, {.i64 = NVJPEG_CSS_420}, INT_MIN, INT_MAX, VE, "subsampling"},
        {"411", "YUV 4:1:1", 0, AV_OPT_TYPE_CONST, {.i64 = NVJPEG_CSS_411}, INT_MIN, INT_MAX, VE, "subsampling"},
        {"gray", "Grayscale", 0, AV_OPT_TYPE_CONST, {.i64 = NVJPEG_CSS_GRAY}, INT_MIN, INT_MAX, VE, "subsampling"},
        {"410", "YUV 4:1:0", 0, AV_OPT_TYPE_CONST, {.i64 = NVJPEG_CSS_410}, INT_MIN, INT_MAX, VE, "subsampling"},
    {NULL}
};
// clang-format on

static const AVClass nvjpeg_class = {
    .class_name = "nvjpeg",
    .item_name = av_default_item_name,
    .option = options,
    .version = LIBAVUTIL_VERSION_INT,
};

static av_cold int nvjpeg_encode_init(AVCodecContext* avctx) {
    NvjpegContext* ctx = avctx->priv_data;
    int ret;

    if (avctx->width <= 0 || avctx->height <= 0) {
        av_log(avctx, AV_LOG_ERROR, "dimensions not set\n");
        return AVERROR(EINVAL);
    }

    if (avctx->pix_fmt != AV_PIX_FMT_CUDA) {
        av_log(avctx, AV_LOG_ERROR, "Unsupported pixel format: %s\n",
               av_get_pix_fmt_name(avctx->pix_fmt));
        return AVERROR(EINVAL);
    }

    ctx->width = avctx->width;
    ctx->height = avctx->height;
    if ((ret = nvjpeg_init(avctx)) < 0)
        return ret;

    return 0;
}

const FFCodec ff_nvjpeg_encoder = {
    .p.name = "nvjpeg",
    CODEC_LONG_NAME("nvJPEG Encoder"),
    .p.type = AVMEDIA_TYPE_VIDEO,
    .p.id = AV_CODEC_ID_MJPEG,
    .init = nvjpeg_encode_init,
    FF_CODEC_ENCODE_CB(nvjpeg_encode_frame),
    .close = nvjpeg_close,
    .priv_data_size = sizeof(NvjpegContext),
    .p.priv_class = &nvjpeg_class,
    .p.pix_fmts =
        (const enum AVPixelFormat[]){
            AV_PIX_FMT_CUDA,
            AV_PIX_FMT_NONE,
        },
    .p.capabilities = AV_CODEC_CAP_HARDWARE | AV_CODEC_CAP_ENCODER_FLUSH |
                      AV_CODEC_CAP_DR1 | AV_CODEC_CAP_ENCODER_REORDERED_OPAQUE,
    .caps_internal =
        FF_CODEC_CAP_NOT_INIT_THREADSAFE | FF_CODEC_CAP_INIT_CLEANUP,

};
