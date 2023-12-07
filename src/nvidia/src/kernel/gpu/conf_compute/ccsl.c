/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#define NVOC_CCSL_H_PRIVATE_ACCESS_ALLOWED

#include "core/prelude.h"
#include "core/locks.h"
#include "rmconfig.h"
#include "kernel/gpu/conf_compute/ccsl.h"
#include "kernel/gpu/fifo/kernel_channel.h"
#include "nvport/nvport.h"
#include "rmapi/rmapi.h"
#include "ctrl/ctrlc56f.h"
#include <stddef.h>
// This guard is here until we fix CONF_COMPUTE and SPDM guards across whole RM
#include "kernel/gpu/spdm/libspdm_includes.h"
#include <hal/library/cryptlib.h>
#include <nvspdm_cryptlib_extensions.h>

#include "cc_drv.h"

static void
ccslSplit32(NvU8 *dst, NvU32 num)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3111);
    dst[3] = (NvU8) (num >> 24);
    dst[2] = (NvU8) (num >> 16);
    dst[1] = (NvU8) (num >> 8);
    dst[0] = (NvU8) (num);
}

static void
ccslSplit64(NvU8 *dst, NvU64 num)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3112);
    dst[7] = (NvU8) (num >> 56);
    dst[6] = (NvU8) (num >> 48);
    dst[5] = (NvU8) (num >> 40);
    dst[4] = (NvU8) (num >> 32);
    dst[3] = (NvU8) (num >> 24);
    dst[2] = (NvU8) (num >> 16);
    dst[1] = (NvU8) (num >> 8);
    dst[0] = (NvU8) (num);
}

static void
writeKmbToContext
(
    pCcslContext  pCtx,
    CC_KMB       *kmb
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3113);
    for (NvU32 index = 0; index < CC_AES_256_GCM_KEY_SIZE_DWORD; index++)
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3114);
        ccslSplit32(pCtx->keyOut + 4 * index, kmb->encryptBundle.key[index]);
    }

    for (NvU32 index = 0; index < CC_AES_256_GCM_IV_SIZE_DWORD; index++)
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3115);
        ccslSplit32(pCtx->ivOut + 4 * index, kmb->encryptBundle.iv[index]);
        ccslSplit32(pCtx->ivMaskOut + 4 * index, kmb->encryptBundle.ivMask[index]);
    }

    if (kmb->bIsWorkLaunch)
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3116);
        for (NvU32 index = 0; index < CC_AES_256_GCM_KEY_SIZE_DWORD; index++)
        {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3117);
            ccslSplit32(pCtx->keyIn + 4 * index, kmb->hmacBundle.key[index]);
        }

        for (NvU32 index = 0; index < CC_HMAC_NONCE_SIZE_DWORD; index++)
        {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3118);
            ccslSplit32(pCtx->nonce + 4 * index, kmb->hmacBundle.nonce[index]);
        }
    }
    else
    {
        for (NvU32 index = 0; index < CC_AES_256_GCM_KEY_SIZE_DWORD; index++)
        {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3119);
            ccslSplit32(pCtx->keyIn + 4 * index, kmb->decryptBundle.key[index]);
        }

        for (NvU32 index = 0; index < CC_AES_256_GCM_IV_SIZE_DWORD; index++)
        {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3120);
            ccslSplit32(pCtx->ivMaskIn + 4 * index, kmb->decryptBundle.ivMask[index]);
            ccslSplit32(pCtx->ivIn + 4 * index, kmb->decryptBundle.iv[index]);
        }
    }
}

NV_STATUS
ccslIncrementCounter_IMPL
(
    pCcslContext  pCtx,
    NvU8         *ctr,
    NvU64         increment
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3121);
    NvU32 msgCounterLo = NvU32_BUILD(ctr[3], ctr[2], ctr[1], ctr[0]);

    switch (pCtx->msgCounterSize)
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3122);
        case CSL_MSG_CTR_32:
            if (increment > NV_U32_MAX)
            {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3123);
                return NV_ERR_INVALID_ARGUMENT;
            }

            if (msgCounterLo > (NV_U32_MAX - increment))
            {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3124);
                return NV_ERR_INSUFFICIENT_RESOURCES;
            }

            msgCounterLo += increment;
            ccslSplit32(ctr, msgCounterLo);
            break;
        case CSL_MSG_CTR_64:
        {
            NvU32 msgCounterHi = NvU32_BUILD(ctr[7], ctr[6], ctr[5], ctr[4]);
            NvU64 msgCounter = ((NvU64) msgCounterHi << 32) | msgCounterLo;

            if (msgCounterLo > (NV_U64_MAX - increment))
            {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3125);
                return NV_ERR_INSUFFICIENT_RESOURCES;
            }

            msgCounter += increment;
            ccslSplit64(ctr, msgCounter);
            break;
        }
    }

    return NV_OK;
}

NV_STATUS
ccslContextInitViaChannel_IMPL
(
    pCcslContext *ppCtx,
    NvHandle      hClient,
    NvHandle      hChannel
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3126);
    CC_KMB *kmb;

    OBJSYS    *pSys        = SYS_GET_INSTANCE();
    OBJGPUMGR *pGpuMgr     = SYS_GET_GPUMGR(pSys);
    OBJGPU    *pGpu;
    NvU32      gpuMask;
    NvU32      gpuInstance = 0;
    RM_API    *pRmApi      = NULL;
    NV_STATUS  status;

    NVC56F_CTRL_CMD_GET_KMB_PARAMS getKmbParams;

    NV_PRINTF(LEVEL_INFO, "Initializing CCSL context via channel.\n");

    // This function must be redesigned for multi-gpu
    if(!pGpuMgr->ccCaps.bHccFeatureCapable)
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3127);
        return NV_ERR_NOT_SUPPORTED;
    }

    if (ppCtx == NULL)
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3128);
        return NV_ERR_INVALID_PARAMETER;
    }

    pCcslContext pCtx = portMemAllocNonPaged(sizeof(*pCtx));
    if (pCtx == NULL)
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3129);
        return NV_ERR_NO_MEMORY;
    }
    *ppCtx = pCtx;

    if (!libspdm_aead_gcm_prealloc(&pCtx->openrmCtx))
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3130);
        portMemFree(pCtx);
        return NV_ERR_NO_MEMORY;
    }

    pCtx->hClient = hClient;
    pCtx->hChannel = hChannel;

    (void)gpumgrGetGpuAttachInfo(NULL, &gpuMask);

    while ((pGpu = gpumgrGetNextGpu(gpuMask, &gpuInstance)) != NULL)
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3131);
        if (IS_GSP_CLIENT(pGpu))
        {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3132);
            if (rmGpuLockIsOwner())
            {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3133);
                pRmApi = rmapiGetInterface(RMAPI_GPU_LOCK_INTERNAL);
            }
            else
            {
                pRmApi = rmapiGetInterface(RMAPI_EXTERNAL_KERNEL);
            }
        }
        else
        {
            pRmApi = GPU_GET_PHYSICAL_RMAPI(pGpu);
        }
        portMemSet(&getKmbParams, 0, sizeof(getKmbParams));

        status = pRmApi->Control(pRmApi, hClient, hChannel,
                                 NVC56F_CTRL_CMD_GET_KMB, &getKmbParams,
                                 sizeof(getKmbParams));
        if (status != NV_OK)
        {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3134);
            libspdm_aead_free(pCtx->openrmCtx);
            portMemFree(pCtx);
            return status;
        }

        pCtx->msgCounterSize = CSL_MSG_CTR_32;

        kmb = &getKmbParams.kmb;

        writeKmbToContext(pCtx, kmb);
    }

    return NV_OK;
}

NV_STATUS
ccslContextInitViaKeyId_KERNEL
(
    ConfidentialCompute *pConfCompute,
    pCcslContext        *ppCtx,
    NvU32                globalKeyId
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3135);
    NV_STATUS  status;
    CC_KMB     kmb;
    OBJSYS    *pSys    = SYS_GET_INSTANCE();
    OBJGPUMGR *pGpuMgr = SYS_GET_GPUMGR(pSys);

    NV_PRINTF(LEVEL_INFO, "Initializing CCSL context via globak key ID.\n");

    // This function must be redesigned for multi-gpu
    if(!pGpuMgr->ccCaps.bHccFeatureCapable)
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3136);
        return NV_ERR_NOT_SUPPORTED;
    }

    if (ppCtx == NULL)
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3137);
        return NV_ERR_INVALID_PARAMETER;
    }

    pCcslContext pCtx = portMemAllocNonPaged(sizeof(*pCtx));
    if (pCtx == NULL)
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3138);
        return NV_ERR_NO_MEMORY;
    }
    *ppCtx = pCtx;
    if (!libspdm_aead_gcm_prealloc(&pCtx->openrmCtx))
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3139);
        portMemFree(pCtx);
        return NV_ERR_NO_MEMORY;
    }

    status = confComputeKeyStoreRetrieveViaKeyId_HAL(pConfCompute,
                                                     globalKeyId,
                                                     ROTATE_IV_ALL_VALID,
                                                     NV_TRUE,
                                                     &kmb);
    if (status != NV_OK)
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3140);
    libspdm_aead_free(pCtx->openrmCtx);
        portMemFree(pCtx);
        return status;
    }

    // For now assume any call to this function uses a 64-bit message counter.
    pCtx->msgCounterSize = CSL_MSG_CTR_64;
    writeKmbToContext(pCtx, &kmb);

    return NV_OK;
}

void
ccslContextClear_IMPL
(
    pCcslContext pCtx
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3141);
    NV_PRINTF(LEVEL_INFO, "Clearing the CCSL context.\n");

    if (pCtx == NULL)
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3142);
        return;
    }

    libspdm_aead_free(pCtx->openrmCtx);
    portMemFree(pCtx);
}

NV_STATUS
ccslRotateIv_IMPL
(
    pCcslContext pCtx,
    NvU8         direction
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3143);
    OBJGPU *pGpu;
    NvU32   gpuMask;
    NvU32   gpuInstance = 0;
    RM_API *pRmApi      = NULL;

    NVC56F_CTRL_ROTATE_SECURE_CHANNEL_IV_PARAMS rotateIvParams;

    if ((direction != CCSL_DIR_HOST_TO_DEVICE) && (direction != CCSL_DIR_DEVICE_TO_HOST))
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3144);
        return NV_ERR_INVALID_ARGUMENT;
    }

    portMemSet(&rotateIvParams, 0, sizeof(rotateIvParams));

    if (direction == CCSL_DIR_HOST_TO_DEVICE)
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3145);
        rotateIvParams.rotateIvType = ROTATE_IV_ENCRYPT;
    }
    else
    {
        rotateIvParams.rotateIvType = ROTATE_IV_DECRYPT;
    }

    (void)gpumgrGetGpuAttachInfo(NULL, &gpuMask);
    while ((pGpu = gpumgrGetNextGpu(gpuMask, &gpuInstance)) != NULL)
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3146);
        if (IS_GSP_CLIENT(pGpu))
        {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3147);
            pRmApi = rmapiGetInterface(RMAPI_EXTERNAL_KERNEL);
        }
        else
        {
            pRmApi = GPU_GET_PHYSICAL_RMAPI(pGpu);
        }
        NV_ASSERT_OK_OR_RETURN(pRmApi->Control(pRmApi,
                                               pCtx->hClient,
                                               pCtx->hChannel,
                                               NVC56F_CTRL_ROTATE_SECURE_CHANNEL_IV,
                                               &rotateIvParams,
                                               sizeof(rotateIvParams)));
    }

    switch (direction)
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3148);
        case CCSL_DIR_HOST_TO_DEVICE:
            portMemCopy(pCtx->keyOut, sizeof(pCtx->keyOut),
                        &rotateIvParams.updatedKmb.encryptBundle.key,
                        sizeof(rotateIvParams.updatedKmb.encryptBundle.key));
            portMemCopy(pCtx->ivMaskOut, sizeof(pCtx->ivMaskOut),
                        &rotateIvParams.updatedKmb.encryptBundle.ivMask,
                        sizeof(rotateIvParams.updatedKmb.encryptBundle.ivMask));
            portMemCopy(pCtx->ivOut, sizeof(pCtx->ivOut),
                        &rotateIvParams.updatedKmb.encryptBundle.iv,
                        sizeof(rotateIvParams.updatedKmb.encryptBundle.iv));
            break;

        case CCSL_DIR_DEVICE_TO_HOST:
            portMemCopy(pCtx->keyIn, sizeof(pCtx->keyIn),
                        &rotateIvParams.updatedKmb.decryptBundle.key,
                        sizeof(rotateIvParams.updatedKmb.decryptBundle.key));
            portMemCopy(pCtx->ivMaskIn, sizeof(pCtx->ivMaskIn),
                        &rotateIvParams.updatedKmb.decryptBundle.ivMask,
                        sizeof(rotateIvParams.updatedKmb.decryptBundle.ivMask));
            portMemCopy(pCtx->ivIn, sizeof(pCtx->ivIn),
                        &rotateIvParams.updatedKmb.decryptBundle.iv,
                        sizeof(rotateIvParams.updatedKmb.decryptBundle.iv));
            break;
    }

    return NV_OK;
}

NV_STATUS
ccslEncryptWithIv_IMPL
(
    pCcslContext  pCtx,
    NvU32         bufferSize,
    NvU8 const   *inputBuffer,
    NvU8         *encryptIv,
    NvU8 const   *aadBuffer,
    NvU32         aadSize,
    NvU8         *outputBuffer,
    NvU8         *authTagBuffer
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3149);
    NvU8   iv[CC_AES_256_GCM_IV_SIZE_BYTES] = {0};
    size_t outputBufferSize                 = bufferSize;

    // The "freshness" bit is right after the IV
    if (encryptIv[CC_AES_256_GCM_IV_SIZE_BYTES] != 1)
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3150);
        return NV_ERR_INVALID_ARGUMENT;
    }

    encryptIv[CC_AES_256_GCM_IV_SIZE_BYTES] = 0;

    for (NvU64 i = 0; i < CC_AES_256_GCM_IV_SIZE_BYTES; i++)
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3151);
        iv[i] = encryptIv[i] ^ pCtx->ivMaskOut[i];
    }

    if(!libspdm_aead_aes_gcm_encrypt_prealloc(pCtx->openrmCtx,
        (NvU8 *)pCtx->keyOut, CC_AES_256_GCM_KEY_SIZE_BYTES,
        iv, CC_AES_256_GCM_IV_SIZE_BYTES, aadBuffer, aadSize,
        inputBuffer, bufferSize, authTagBuffer, 16,
        outputBuffer, &outputBufferSize))
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3152);
        return NV_ERR_GENERIC;
    }

    return NV_OK;
}

NV_STATUS
ccslEncrypt_KERNEL
(
    pCcslContext  pCtx,
    NvU32         bufferSize,
    NvU8 const   *inputBuffer,
    NvU8 const   *aadBuffer,
    NvU32         aadSize,
    NvU8         *outputBuffer,
    NvU8         *authTagBuffer
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3153);
    NvU8   iv[CC_AES_256_GCM_IV_SIZE_BYTES] = {0};
    size_t outputBufferSize                 = bufferSize;

    if (ccslIncrementCounter(pCtx, pCtx->ivOut, 1) != NV_OK)
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3154);
        return NV_ERR_INSUFFICIENT_RESOURCES;
    }

    for (NvU64 i = 0; i < CC_AES_256_GCM_IV_SIZE_BYTES; i++)
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3155);
        iv[i] = pCtx->ivOut[i] ^ pCtx->ivMaskOut[i];
    }

    if(!libspdm_aead_aes_gcm_encrypt_prealloc(pCtx->openrmCtx,
        (NvU8 *)pCtx->keyOut, CC_AES_256_GCM_KEY_SIZE_BYTES,
        iv, CC_AES_256_GCM_IV_SIZE_BYTES, aadBuffer, aadSize,
        inputBuffer, bufferSize, authTagBuffer, 16,
        outputBuffer, &outputBufferSize))
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3156);
        return NV_ERR_GENERIC;
    }

    return NV_OK;
}

NV_STATUS
ccslDecrypt_KERNEL
(
    pCcslContext  pCtx,
    NvU32         bufferSize,
    NvU8 const   *inputBuffer,
    NvU8 const   *decryptIv,
    NvU8 const   *aadBuffer,
    NvU32         aadSize,
    NvU8         *outputBuffer,
    NvU8 const   *authTagBuffer
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3157);
    NvU8   iv[CC_AES_256_GCM_IV_SIZE_BYTES] = {0};
    size_t outputBufferSize = bufferSize;

    if ((bufferSize == 0) || ((aadBuffer != NULL) && (aadSize == 0)))
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3158);
        return NV_ERR_INVALID_ARGUMENT;
    }

    if (decryptIv == NULL)
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3159);
        if (ccslIncrementCounter(pCtx, pCtx->ivIn, 1) != NV_OK)
        {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3160);
            return NV_ERR_INSUFFICIENT_RESOURCES;
        }

        for (NvU64 i = 0; i < CC_AES_256_GCM_IV_SIZE_BYTES; i++)
        {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3161);
            iv[i] = pCtx->ivIn[i] ^ pCtx->ivMaskIn[i];
        }
    }
    else
    {
        for (NvU64 i = 0; i < CC_AES_256_GCM_IV_SIZE_BYTES; i++)
        {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3162);
            iv[i] = decryptIv[i] ^ pCtx->ivMaskIn[i];
        }
    }

    if(!libspdm_aead_aes_gcm_decrypt_prealloc(pCtx->openrmCtx,
        (NvU8 *)pCtx->keyIn, CC_AES_256_GCM_KEY_SIZE_BYTES,
        iv, CC_AES_256_GCM_IV_SIZE_BYTES, aadBuffer, aadSize,
        inputBuffer, bufferSize, (NvU8 *) authTagBuffer, 16,
        outputBuffer, &outputBufferSize))
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3163);
        return NV_ERR_INVALID_DATA;
    }

    return NV_OK;
}

static NV_STATUS ccslIncrementCounter192(NvU8 *ctr)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3164);
    NvU8 carry = 1;
    NvU64 i;
    NvBool overflow = NV_TRUE;

    for (i = 0; i < 192 / 8; ++i)
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3165);
        if (ctr[i] != 0xFF)
        {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3166);
            overflow = NV_FALSE;
            break;
        }
    }
    if (overflow)
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3167);
        return NV_ERR_INSUFFICIENT_RESOURCES;
    }

    for (i = 0; i < 192 / 8; ++i)
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3168);
        ctr[i] += carry;
        carry = (ctr[i] == 0) && (carry == 1);
    }

    return NV_OK;
}

NV_STATUS
ccslSign_IMPL
(
    pCcslContext  pCtx,
    NvU32         bufferSize,
    NvU8 const   *inputBuffer,
    NvU8         *authTagBuffer
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3169);
    void *hmac_ctx;

    if (bufferSize == 0)
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3170);
        return NV_ERR_INVALID_PARAMETER;
    }

    if (ccslIncrementCounter192(pCtx->nonce) != NV_OK)
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3171);
        return NV_ERR_INSUFFICIENT_RESOURCES;
    }

    hmac_ctx = libspdm_hmac_sha256_new();
    if (hmac_ctx == NULL)
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3172);
        return NV_ERR_GENERIC;
    }

    if (!libspdm_hmac_sha256_set_key(hmac_ctx, pCtx->keyIn, CC_AES_256_GCM_KEY_SIZE_BYTES))
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3173);
        libspdm_hmac_sha256_free(hmac_ctx);
        return NV_ERR_GENERIC;
    }

    if (!libspdm_hmac_sha256_update(hmac_ctx, inputBuffer, bufferSize))
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3174);
        libspdm_hmac_sha256_free(hmac_ctx);
        return NV_ERR_GENERIC;
    }

    if (!libspdm_hmac_sha256_update(hmac_ctx, pCtx->nonce, CC_HMAC_NONCE_SIZE_BYTES))
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3175);
        libspdm_hmac_sha256_free(hmac_ctx);
        return NV_ERR_GENERIC;
    }

    if (!libspdm_hmac_sha256_final(hmac_ctx, authTagBuffer))
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3176);
        libspdm_hmac_sha256_free(hmac_ctx);
        return NV_ERR_GENERIC;
    }

    libspdm_hmac_sha256_free(hmac_ctx);

    return NV_OK;
}

static NvU64
getMessageCounterAndLimit
(
    pCcslContext  pCtx,
    NvU8         *iv,
    NvU64        *limit
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3177);
    NvU32 msgCounterLo = NvU32_BUILD(iv[3], iv[2], iv[1], iv[0]);
    NvU32 msgCounterHi = NvU32_BUILD(iv[7], iv[6], iv[5], iv[4]);

    switch (pCtx->msgCounterSize)
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3178);
        case CSL_MSG_CTR_32:
            *limit = NV_U32_MAX;
            return msgCounterLo;
        case CSL_MSG_CTR_64:
            *limit = NV_U64_MAX;
            return ((((NvU64) msgCounterHi) << 32) | msgCounterLo);
        default:
            return 0;
    }
}

NV_STATUS
ccslQueryMessagePool_IMPL
(
    pCcslContext  pCtx,
    NvU8          direction,
    NvU64        *messageNum
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3179);
    NvU64 limit;
    NvU64 messageCounter;

    switch (direction)
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3180);
        case CCSL_DIR_HOST_TO_DEVICE:
            messageCounter = getMessageCounterAndLimit(pCtx, pCtx->ivOut, &limit);
            break;
        case CCSL_DIR_DEVICE_TO_HOST:
            messageCounter = getMessageCounterAndLimit(pCtx, pCtx->ivIn, &limit);
            break;
        default:
            return NV_ERR_INVALID_ARGUMENT;
    }

    *messageNum = limit - messageCounter;

    return NV_OK;
}

NV_STATUS
ccslIncrementIv_IMPL
(
    pCcslContext  pCtx,
    NvU8          direction,
    NvU64         increment,
    NvU8         *iv
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3181);
    NV_STATUS status;
    void *ivPtr;

    switch (direction)
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3182);
        case CCSL_DIR_HOST_TO_DEVICE:
            ivPtr = pCtx->ivOut;
            break;
        case CCSL_DIR_DEVICE_TO_HOST:
            ivPtr = pCtx->ivIn;
            break;
        default:
            return NV_ERR_INVALID_ARGUMENT;
    }

    status = ccslIncrementCounter(pCtx, ivPtr, increment);

    if (status != NV_OK)
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3183);
        return status;
    }

    if (iv != NULL) {
        portMemCopy(iv, CC_AES_256_GCM_IV_SIZE_BYTES, ivPtr, CC_AES_256_GCM_IV_SIZE_BYTES);

        if (direction == CCSL_DIR_HOST_TO_DEVICE)
        {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3184);
            // The "freshness" bit is right after the IV.
            iv[CC_AES_256_GCM_IV_SIZE_BYTES] = 1;
        }
        else
        {
            // Decrypt IV cannot be used for encryption.
            iv[CC_AES_256_GCM_IV_SIZE_BYTES] = 0;
        }
    }

    return NV_OK;
}
