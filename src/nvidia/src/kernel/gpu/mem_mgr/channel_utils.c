/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "kernel/gpu/intr/intr.h"
#include "kernel/gpu/fifo/kernel_fifo.h"
#include "kernel/gpu/mig_mgr/kernel_mig_manager.h"
#include "utils/nvassert.h"
#include "core/locks.h"
#include "gpu/mem_mgr/mem_mgr.h"

#include "kernel/gpu/mem_mgr/ce_utils_sizes.h"
#include "kernel/gpu/mem_mgr/channel_utils.h"
#include "class/clcba2.h"


#define SEC2_WL_METHOD_ARRAY_SIZE 16
#define SHA_256_HASH_SIZE_BYTE  32

/* Static helper functions */
static void channelPushAddress(OBJCHANNEL *pChannel, CHANNEL_PB_INFO *pChannelPbInfo, NvU32 **ppPtr);
static NvU32 channelPushMemoryProperties(OBJCHANNEL *pChannel, CHANNEL_PB_INFO *pChannelPbInfo, NvU32 **ppPtr);
static void channelPushMethod(OBJCHANNEL *pChannel, CHANNEL_PB_INFO *pChannelPbInfo,
                              NvBool bPipelined, NvBool bInsertFinishPayload,
                              NvU32 launchType, NvU32 semaValue, NvU32 **ppPtr);

/* Public APIs */
NV_STATUS
channelSetupIDs
(
    OBJCHANNEL *pChannel,
    OBJGPU     *pGpu,
    NvBool      bUseVasForCeCopy,
    NvBool      bMIGInUse
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 2720);
    NV_CHECK_OK_OR_RETURN(LEVEL_ERROR,
                          serverutilGenResourceHandle(pChannel->hClient, &pChannel->physMemId));

    NV_CHECK_OK_OR_RETURN(LEVEL_ERROR,
                          serverutilGenResourceHandle(pChannel->hClient, &pChannel->channelId));

    NV_CHECK_OK_OR_RETURN(LEVEL_ERROR,
                          serverutilGenResourceHandle(pChannel->hClient, &pChannel->errNotifierIdVirt));

    NV_CHECK_OK_OR_RETURN(LEVEL_ERROR,
                          serverutilGenResourceHandle(pChannel->hClient, &pChannel->errNotifierIdPhys));

    NV_CHECK_OK_OR_RETURN(LEVEL_ERROR,
                          serverutilGenResourceHandle(pChannel->hClient, &pChannel->engineObjectId));

    NV_CHECK_OK_OR_RETURN(LEVEL_ERROR,
                          serverutilGenResourceHandle(pChannel->hClient, &pChannel->eventId));

    NV_CHECK_OK_OR_RETURN(LEVEL_ERROR,
                          serverutilGenResourceHandle(pChannel->hClient, &pChannel->pushBufferId));

    NV_CHECK_OK_OR_RETURN(LEVEL_ERROR,
                          serverutilGenResourceHandle(pChannel->hClient, &pChannel->doorbellRegionHandle));

    NV_CHECK_OK_OR_RETURN(LEVEL_ERROR,
                          serverutilGenResourceHandle(pChannel->hClient, &pChannel->hUserD));

    // For physical CE channels, we will use RM internal VAS to map channel buffers
    NV_ASSERT(pChannel->hVASpaceId == NV01_NULL_OBJECT);

    if (bUseVasForCeCopy || (IS_GSP_CLIENT(pGpu) && bMIGInUse))
    {
        NV_CHECK_OK_OR_RETURN(LEVEL_ERROR,
                              serverutilGenResourceHandle(pChannel->hClient, &pChannel->hVASpaceId));
    }

    return NV_OK;
}

void
channelSetupChannelBufferSizes
(
    OBJCHANNEL *pChannel
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 2721);
    NV_ASSERT_OR_RETURN_VOID(pChannel != NULL);
    NV_ASSERT_OR_RETURN_VOID(pChannel->type < MAX_CHANNEL_TYPE);

    if (pChannel->type == SWL_SCRUBBER_CHANNEL)
    {
        pChannel->methodSizePerBlock = SEC2_METHOD_SIZE_PER_BLOCK;
    }
    else
    {
        pChannel->methodSizePerBlock = CE_METHOD_SIZE_PER_BLOCK;
        if (pChannel->type == FAST_SCRUBBER_CHANNEL)
        {
            pChannel->methodSizePerBlock = FAST_SCRUBBER_METHOD_SIZE_PER_BLOCK;
        }
    }
    pChannel->channelNotifierSize = CHANNEL_NOTIFIER_SIZE;
    pChannel->channelNumGpFifioEntries = NUM_COPY_BLOCKS;

    // These sizes depend on size of each block of pushbuffer methods
    pChannel->channelPbSize = pChannel->methodSizePerBlock * NUM_COPY_BLOCKS;
    pChannel->channelSize = pChannel->channelPbSize + GPFIFO_SIZE + 
                            CHANNEL_HOST_SEMAPHORE_SIZE + CHANNEL_ENGINE_SEMAPHORE_SIZE;
    if (pChannel->type == SWL_SCRUBBER_CHANNEL)
    {
        pChannel->channelSize = pChannel->channelSize + SEC2_AUTH_TAG_BUF_SEMAPHORE_SIZE;
    }
    pChannel->semaOffset = pChannel->channelPbSize + GPFIFO_SIZE;
    pChannel->finishPayloadOffset = pChannel->semaOffset + CHANNEL_HOST_SEMAPHORE_SIZE;
    pChannel->authTagBufSemaOffset = pChannel->finishPayloadOffset + CHANNEL_ENGINE_SEMAPHORE_SIZE;
}

NvU32
channelReadChannelMemdesc
(
    OBJCHANNEL *pChannel,
    NvU32       offset
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 2722);
    NV_ASSERT_OR_RETURN(pChannel != NULL, 0);
    NV_ASSERT_OR_RETURN(pChannel->pGpu != NULL, 0);

    MemoryManager *pMemoryManager = GPU_GET_MEMORY_MANAGER(pChannel->pGpu);
    NvBool bReleaseMapping = NV_FALSE;
    NvU32 result = 0;

    //
    // Use BAR1 if CPU access is allowed, otherwise allocate and init shadow
    // buffer for DMA access
    //
    NvU32 transferFlags = (TRANSFER_FLAGS_USE_BAR1     |
                           TRANSFER_FLAGS_SHADOW_ALLOC |
                           TRANSFER_FLAGS_SHADOW_INIT_MEM);

    if (pChannel->pbCpuVA == NULL)
    {
        pChannel->pbCpuVA = memmgrMemDescBeginTransfer(pMemoryManager, pChannel->pChannelBufferMemdesc,
                                                       transferFlags);
        bReleaseMapping = NV_TRUE;
    }

    NV_ASSERT_OR_RETURN(pChannel->pbCpuVA != NULL, 0);

    result = MEM_RD32((NvU8*)pChannel->pbCpuVA + offset);

    if (bReleaseMapping)
    {
        memmgrMemDescEndTransfer(pMemoryManager, pChannel->pChannelBufferMemdesc, transferFlags);
        pChannel->pbCpuVA = NULL;
    }

    return result;
}

/*!
 * Service any interrupts which may block the CE or SEC2 scrubber associated
 * with this channel from making forward progress.  That includes interrupts
 * from the CE/SEC2 engine as well as host/esched that is processing the
 * pushbuffer.
 */
void
channelServiceScrubberInterrupts(
    OBJCHANNEL *pChannel
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 2723);
    OBJGPU *pGpu = pChannel->pGpu;
    Intr *pIntr = GPU_GET_INTR(pGpu);
    KernelFifo *pKernelFifo = GPU_GET_KERNEL_FIFO(pGpu);
    const NvU32 ceId = pChannel->ceId;
    MC_ENGINE_BITVECTOR engines;
    NvU32 mcIndex;
    NvU32 engineType;

    if (pChannel->type == SWL_SCRUBBER_CHANNEL)
    {
        mcIndex = MC_ENGINE_IDX_SEC2;
        engineType = RM_ENGINE_TYPE_SEC2;
    }
    else
    {
        mcIndex = MC_ENGINE_IDX_CE(pChannel->ceId);
        engineType = RM_ENGINE_TYPE_COPY(ceId);
    }

    bitVectorClrAll(&engines);
    bitVectorSet(&engines, mcIndex);
    if (kfifoIsHostEngineExpansionSupported(pKernelFifo))
    {
        NvU32 runlistId;
        NV_ASSERT_OK(kfifoEngineInfoXlate_HAL(pGpu, pKernelFifo,
            ENGINE_INFO_TYPE_RM_ENGINE_TYPE, engineType,
            ENGINE_INFO_TYPE_RUNLIST, &runlistId));
        bitVectorSet(&engines, MC_ENGINE_IDX_ESCHEDn(runlistId));
    }
    else
    {
        bitVectorSet(&engines, MC_ENGINE_IDX_FIFO);
    }
    intrServiceStallList_HAL(pGpu, pIntr, &engines, NV_FALSE);
}

NV_STATUS
channelWaitForFinishPayload
(
    OBJCHANNEL *pChannel,
    NvU64       targetPayload
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 2724);
    NV_ASSERT_OR_RETURN(pChannel != NULL, NV_ERR_INVALID_STATE);

    OBJGPU *pGpu = pChannel->pGpu;
    RMTIMEOUT timeout;
    NV_STATUS status = NV_OK;

    gpuSetTimeout(pGpu, GPU_TIMEOUT_DEFAULT, &timeout, GPU_TIMEOUT_FLAGS_BYPASS_THREAD_STATE);
    while (1) 
    {
        if (READ_CHANNEL_PAYLOAD_SEMA(pChannel) >= targetPayload)
        {
            break;
        }
        
        status = gpuCheckTimeout(pGpu, &timeout);
        if (status == NV_ERR_TIMEOUT)
        {
            break;
        }

        if (rmDeviceGpuLockIsOwner(pGpu->gpuInstance))
        {
            channelServiceScrubberInterrupts(pChannel);
        }
        else
        {
            osSchedule();
        }
    }

    return status;
}


/** helper function which waits for a PB & GPFIO entry to be read by HOST.
 *  After the HOST reads GPFIFO and PB entry, the semaphore will be released.
 */
NV_STATUS
channelWaitForFreeEntry
(
    OBJCHANNEL *pChannel,
    NvU32      *pPutIndex
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 2725);
    NV_ASSERT_OR_RETURN(pChannel != NULL, NV_ERR_INVALID_STATE);

    OBJGPU *pGpu = pChannel->pGpu;
    RMTIMEOUT timeout;
    NV_STATUS status = NV_OK;

    NvU32 putIndex = 0;
    NvU32 getIndex = 0;

    putIndex = (pChannel->lastSubmittedEntry + 1) % pChannel->channelNumGpFifioEntries;

    gpuSetTimeout(pGpu, GPU_TIMEOUT_DEFAULT, &timeout, GPU_TIMEOUT_FLAGS_BYPASS_THREAD_STATE);
    do
    {
        getIndex = READ_CHANNEL_PB_SEMA(pChannel);

        NV_PRINTF(LEVEL_INFO, "Get Index: %x, PayloadIndex: %x\n", getIndex,
                              READ_CHANNEL_PAYLOAD_SEMA(pChannel));

        if (getIndex != putIndex)
        {
            break;
        }
        
        status = gpuCheckTimeout(pGpu, &timeout);
        if (status == NV_ERR_TIMEOUT)
        {
            break;
        }
        
        if (rmDeviceGpuLockIsOwner(pGpu->gpuInstance))
        {
            channelServiceScrubberInterrupts(pChannel);
        }
        else
        {
            osSchedule();
        }
    } while(1);

    if (status == NV_OK)
    {
        NV_ASSERT_OR_RETURN(pPutIndex != NULL, NV_ERR_INVALID_STATE);
        *pPutIndex = putIndex;
    }

    return status;
}


/** helper function to fill GPFIFO entry with a pushbuffer segment. and kick
 *  off the executiion by HOST.
 */
NV_STATUS
channelFillGpFifo
(
    OBJCHANNEL *pChannel,
    NvU32       putIndex,
    NvU32       methodsLength
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 2726);
    NvU32  *pGpEntry;
    NvU32   GpEntry0;
    NvU32   GpEntry1;
    NvU64   pbPutOffset;
    OBJGPU *pGpu;
    KernelBus *pKernelBus;
    MemoryManager *pMemoryManager;
    NvBool bReleaseMapping = NV_FALSE;

    //
    // Use BAR1 if CPU access is allowed, otherwise allocate and init shadow
    // buffer for DMA access
    //
    NvU32 transferFlags = (TRANSFER_FLAGS_USE_BAR1     |
                           TRANSFER_FLAGS_SHADOW_ALLOC |
                           TRANSFER_FLAGS_SHADOW_INIT_MEM);

    NV_ASSERT_OR_RETURN(putIndex < pChannel->channelNumGpFifioEntries, NV_ERR_INVALID_STATE);
    NV_ASSERT_OR_RETURN(pChannel != NULL, NV_ERR_INVALID_STATE);

    pGpu = pChannel->pGpu;
    NV_ASSERT_OR_RETURN(pGpu != NULL, NV_ERR_INVALID_STATE);

    pKernelBus = GPU_GET_KERNEL_BUS(pGpu);
    pMemoryManager = GPU_GET_MEMORY_MANAGER(pGpu);

    if (pChannel->pbCpuVA == NULL)
    {
        pChannel->pbCpuVA = memmgrMemDescBeginTransfer(pMemoryManager, pChannel->pChannelBufferMemdesc,
                                                       transferFlags);
        bReleaseMapping = NV_TRUE;
    }

    NV_ASSERT_OR_RETURN(pChannel->pbCpuVA != NULL, NV_ERR_GENERIC);

    pbPutOffset = (pChannel->pbGpuVA + (putIndex * pChannel->methodSizePerBlock));

    GpEntry0 = DRF_DEF(906F, _GP_ENTRY0, _NO_CONTEXT_SWITCH, _FALSE) | 
               DRF_NUM(906F, _GP_ENTRY0, _GET, NvU64_LO32(pbPutOffset) >> 2);

    GpEntry1 = DRF_NUM(906F, _GP_ENTRY1, _GET_HI, NvU64_HI32(pbPutOffset)) | 
               DRF_NUM(906F, _GP_ENTRY1, _LENGTH, methodsLength >> 2) |
               DRF_DEF(906F, _GP_ENTRY1, _LEVEL, _MAIN);

    pGpEntry = (NvU32 *)(((NvU8 *)pChannel->pbCpuVA) + pChannel->channelPbSize + 
                (pChannel->lastSubmittedEntry * NV906F_GP_ENTRY__SIZE));

    MEM_WR32(&pGpEntry[0], GpEntry0);
    MEM_WR32(&pGpEntry[1], GpEntry1);

    if (bReleaseMapping)
    {
        memmgrMemDescEndTransfer(pMemoryManager, pChannel->pChannelBufferMemdesc, 
                                 transferFlags);
        pChannel->pbCpuVA = NULL;
    }

    //  need to flush WRC buffer
    osFlushCpuWriteCombineBuffer();

    // write GP put
    if (pChannel->pControlGPFifo == NULL)
    {
        pChannel->pControlGPFifo = 
            (void *)memmgrMemDescBeginTransfer(pMemoryManager, pChannel->pUserdMemdesc,
                                               transferFlags);
        NV_ASSERT_OR_RETURN(pChannel->pControlGPFifo != NULL, NV_ERR_INVALID_STATE);
        bReleaseMapping = NV_TRUE;
    }

    MEM_WR32(&pChannel->pControlGPFifo->GPPut, putIndex);

    if (bReleaseMapping)
    {
        memmgrMemDescEndTransfer(pMemoryManager, pChannel->pUserdMemdesc, transferFlags);
        pChannel->pControlGPFifo = NULL;
    }

    osFlushCpuWriteCombineBuffer();
    
    //
    // On some architectures, if doorbell is mapped via bar0, we need to send
    // an extra flush
    //
    if (kbusFlushPcieForBar0Doorbell_HAL(pGpu, pKernelBus) != NV_OK)
    {
        NV_PRINTF(LEVEL_ERROR, "Busflush failed in _scrubFillGpFifo\n");
        return NV_ERR_GENERIC;
    }

    //
    // removing the FIFO Lite Mode handling
    // Refer older _ceChannelUpdateGpFifo_GF100 code for implementation
    //

    // Update doorbell with work submission token
    if (pChannel->bUseDoorbellRegister)
    {
        if (pChannel->pTokenFromNotifier == NULL)
        {
            NvU8 *pErrNotifierCpuVA = 
                (void *)memmgrMemDescBeginTransfer(pMemoryManager, 
                    pChannel->pErrNotifierMemdesc, transferFlags);

            NV_ASSERT_OR_RETURN(pErrNotifierCpuVA != NULL, NV_ERR_INVALID_STATE);

            pChannel->pTokenFromNotifier =
                (NvNotification *)(pErrNotifierCpuVA +
                               (NV_CHANNELGPFIFO_NOTIFICATION_TYPE_WORK_SUBMIT_TOKEN *
                                sizeof(NvNotification)));
            bReleaseMapping = NV_TRUE;
        }

        // Use the token from notifier memory for VM migration support.
        MEM_WR32(pChannel->pDoorbellRegisterOffset, 
                 MEM_RD32(&(pChannel->pTokenFromNotifier->info32)));

        if (bReleaseMapping)
        {
            memmgrMemDescEndTransfer(pMemoryManager, pChannel->pErrNotifierMemdesc, transferFlags);
            pChannel->pTokenFromNotifier = NULL;
        }
    }

    return NV_OK;
}

NvU32
channelFillPbFastScrub
(
    OBJCHANNEL      *pChannel,
    NvU32            putIndex, 
    NvBool           bPipelined,
    NvBool           bInsertFinishPayload,
    CHANNEL_PB_INFO *pChannelPbInfo
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 2727);
    NvU32   pipelinedValue = 0;
    NvU32   flushValue     = 0;
    NvU32  *pPtr           = (NvU32 *)((NvU8*)pChannel->pbCpuVA + (putIndex * pChannel->methodSizePerBlock));
    NvU32  *pStartPtr      = pPtr;
    NvU32   semaValue      = 0;
    NvU32   data           = 0;
    NvU64   pSemaAddr      = 0;

    NV_PRINTF(LEVEL_INFO, "PutIndex: %x, PbOffset: %x\n", putIndex,
               putIndex * pChannel->methodSizePerBlock);
    // SET OBJECT
    NV_PUSH_INC_1U(RM_SUBCHANNEL, NVC86F_SET_OBJECT, pChannel->classEngineID);

    // Set Pattern for Memset
    NV_PUSH_INC_1U(RM_SUBCHANNEL, NVC8B5_SET_REMAP_CONST_A, pChannelPbInfo->pattern);
    NV_PUSH_INC_1U(RM_SUBCHANNEL, NVC8B5_SET_REMAP_CONST_B, pChannelPbInfo->pattern);

    // Set Component Size to 1
    NV_PUSH_INC_1U(RM_SUBCHANNEL, NVC8B5_SET_REMAP_COMPONENTS,
                  DRF_DEF(C8B5, _SET_REMAP_COMPONENTS, _DST_X, _CONST_A)          |
                  DRF_DEF(C8B5, _SET_REMAP_COMPONENTS, _DST_X, _CONST_B)          |
                  DRF_DEF(C8B5, _SET_REMAP_COMPONENTS, _COMPONENT_SIZE, _ONE)     |
                  DRF_DEF(C8B5, _SET_REMAP_COMPONENTS, _NUM_DST_COMPONENTS, _ONE));

    NV_PUSH_INC_1U(RM_SUBCHANNEL, NVC8B5_SET_DST_PHYS_MODE,
                  DRF_DEF(C8B5, _SET_DST_PHYS_MODE, _TARGET, _LOCAL_FB));

    semaValue = (bInsertFinishPayload) ?
        DRF_DEF(C8B5, _LAUNCH_DMA, _SEMAPHORE_TYPE, _RELEASE_ONE_WORD_SEMAPHORE) : 0;

    if (bPipelined)
        pipelinedValue = DRF_DEF(C8B5, _LAUNCH_DMA, _DATA_TRANSFER_TYPE, _PIPELINED);
    else
        pipelinedValue = DRF_DEF(C8B5, _LAUNCH_DMA, _DATA_TRANSFER_TYPE, _NON_PIPELINED);

    if (bInsertFinishPayload)
        flushValue = DRF_DEF(B0B5, _LAUNCH_DMA, _FLUSH_ENABLE, _TRUE);
    else
        flushValue = DRF_DEF(B0B5, _LAUNCH_DMA, _FLUSH_ENABLE, _FALSE);

    NV_PUSH_INC_2U(RM_SUBCHANNEL, NVC8B5_OFFSET_OUT_UPPER,
                   DRF_NUM(C8B5, _OFFSET_OUT_UPPER, _UPPER, NvU64_HI32(pChannelPbInfo->dstAddr)),
                   NVC8B5_OFFSET_OUT_LOWER,
                   DRF_NUM(C8B5, _OFFSET_OUT_LOWER, _VALUE,NvU64_LO32(pChannelPbInfo->dstAddr)));

    NV_PUSH_INC_1U(RM_SUBCHANNEL, NVC8B5_LINE_LENGTH_IN, pChannelPbInfo->size);

    if (semaValue)
    {
        NV_PUSH_INC_3U(RM_SUBCHANNEL, NVC8B5_SET_SEMAPHORE_A,
            DRF_NUM(C8B5, _SET_SEMAPHORE_A, _UPPER, NvU64_HI32(pChannel->pbGpuVA + pChannel->finishPayloadOffset)),
            NVC8B5_SET_SEMAPHORE_B,
            DRF_NUM(C8B5, _SET_SEMAPHORE_B, _LOWER, NvU64_LO32(pChannel->pbGpuVA + pChannel->finishPayloadOffset)),
            NVC8B5_SET_SEMAPHORE_PAYLOAD,
            pChannelPbInfo->payload);
    }

    NV_PUSH_INC_1U(RM_SUBCHANNEL, NVC8B5_SET_MEMORY_SCRUB_PARAMETERS,
                       DRF_DEF(C8B5, _SET_MEMORY_SCRUB_PARAMETERS, _DISCARDABLE, _FALSE));

    NV_PUSH_INC_1U(RM_SUBCHANNEL, NVC8B5_LAUNCH_DMA,
            DRF_DEF(C8B5, _LAUNCH_DMA, _SRC_MEMORY_LAYOUT, _PITCH)    |
            DRF_DEF(C8B5, _LAUNCH_DMA, _DST_MEMORY_LAYOUT, _PITCH)    |
            DRF_DEF(C8B5, _LAUNCH_DMA, _REMAP_ENABLE, _FALSE)         |
            DRF_DEF(C8B5, _LAUNCH_DMA, _MULTI_LINE_ENABLE, _FALSE)    |
            DRF_DEF(C8B5, _LAUNCH_DMA, _MEMORY_SCRUB_ENABLE, _TRUE)   |
            DRF_DEF(C8B5, _LAUNCH_DMA, _DISABLE_PLC, _TRUE)           |
            DRF_DEF(C8B5, _LAUNCH_DMA, _DST_TYPE, _PHYSICAL)          |
            DRF_DEF(C8B5, _LAUNCH_DMA, _SRC_TYPE, _PHYSICAL)          |
            pipelinedValue                                            |
            flushValue                                                |
            semaValue);

    //
    // This should always be at the bottom the push buffer segment, since this
    // denotes that HOST has read all the methods needed for this memory operation
    // and safely assume that this GPFIFO and PB entry can be reused.
    //
    data =  DRF_DEF(C86F, _SEM_EXECUTE, _OPERATION, _RELEASE) |
            DRF_DEF(C86F, _SEM_EXECUTE, _PAYLOAD_SIZE, _32BIT) |
            DRF_DEF(C86F, _SEM_EXECUTE, _RELEASE_WFI, _DIS);

    pSemaAddr = (pChannel->pbGpuVA+pChannel->semaOffset);

    NV_PUSH_INC_4U(RM_SUBCHANNEL, NVC86F_SEM_ADDR_LO,
            DRF_NUM(C86F, _SEM_ADDR_LO, _OFFSET, NvU64_LO32(pSemaAddr) >> 2),
            NVC86F_SEM_ADDR_HI,
            DRF_NUM(C86F, _SEM_ADDR_HI, _OFFSET, NvU64_HI32(pSemaAddr)),
            NVC86F_SEM_PAYLOAD_LO, putIndex,
            NVC86F_SEM_PAYLOAD_HI, 0);

    NV_PUSH_INC_1U(RM_SUBCHANNEL, NVC86F_SEM_EXECUTE, data);

    NvU32 methodSize = (NvU32)((NvU8*)pPtr - (NvU8*)pStartPtr);
    NV_ASSERT_OR_RETURN(methodSize <= pChannel->methodSizePerBlock, 0);
    return methodSize;
}

static void
channelAddHostSema
(
    OBJCHANNEL *pChannel,
    NvU32       putIndex, 
    NvU32     **ppPtr
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 2728);
    NvU32  *pPtr = *ppPtr;
    NvU32  data       = 0;
    NvU64  pSemaAddr  = 0;

    // Release HOST semaphore after after gpfifo/pb segments are processed by esched
    data = DRF_DEF(906F, _SEMAPHORED, _OPERATION, _RELEASE) |
           DRF_DEF(906F, _SEMAPHORED, _RELEASE_SIZE, _4BYTE) |
           DRF_DEF(906F, _SEMAPHORED, _RELEASE_WFI, _DIS);

    pSemaAddr = (pChannel->pbGpuVA + pChannel->semaOffset);

    //
    // This should always be at the bottom the push buffer segment, since this
    // denotes that HOST has read all the methods needed for this memory operation
    // and safely assume that this GPFIFO and PB entry can be reused.
    //
    NV_PUSH_INC_4U(RM_SUBCHANNEL, NV906F_SEMAPHOREA,
                   DRF_NUM(906F, _SEMAPHOREA_OFFSET, _UPPER, NvU64_HI32(pSemaAddr)),
                   NV906F_SEMAPHOREB,
                   DRF_NUM(906F, _SEMAPHOREB_OFFSET, _LOWER, NvU64_LO32(pSemaAddr) >> 2),
                   NV906F_SEMAPHOREC,
                   putIndex,
                   NV906F_SEMAPHORED, 
                   data);
    *ppPtr = pPtr;
}

/** single helper function to fill the push buffer with the methods needed for
 *  memsetting using CE. This function is much more efficient in the sense it
 *  decouples the mem(set/copy) operation from managing channel resources.
 **/

NvU32
channelFillCePb
(
    OBJCHANNEL      *pChannel,
    NvU32            putIndex, 
    NvBool           bPipelined,
    NvBool           bInsertFinishPayload,
    CHANNEL_PB_INFO *pChannelPbInfo
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 2729);
    NvU32  launchType = 0;
    NvU32 *pPtr       = (NvU32 *)((NvU8 *)pChannel->pbCpuVA + (putIndex * pChannel->methodSizePerBlock));
    NvU32 *pStartPtr  = pPtr;
    NvU32  semaValue  = 0;

    NV_PRINTF(LEVEL_INFO, "PutIndex: %x, PbOffset: %x\n", putIndex, putIndex * pChannel->methodSizePerBlock);

    NV_PUSH_INC_1U(RM_SUBCHANNEL, NV906F_SET_OBJECT, pChannel->classEngineID);

    // Side effect - pushed target aperture and REMAP method for memset
    launchType = channelPushMemoryProperties(pChannel, pChannelPbInfo, &pPtr);

    // Side effect - pushed addresses
    channelPushAddress(pChannel, pChannelPbInfo, &pPtr);
    NV_PUSH_INC_1U(RM_SUBCHANNEL, NVB0B5_LINE_LENGTH_IN, pChannelPbInfo->size);

    if (bInsertFinishPayload)
    {
        semaValue = DRF_DEF(B0B5, _LAUNCH_DMA, _SEMAPHORE_TYPE, _RELEASE_ONE_WORD_SEMAPHORE);
        
        // Do not support client semaphore for now
        NV_ASSERT(pChannelPbInfo->clientSemaAddr == 0);

        NV_PUSH_INC_3U(RM_SUBCHANNEL, NVB0B5_SET_SEMAPHORE_A,
                       DRF_NUM(B0B5, _SET_SEMAPHORE_A, _UPPER, NvU64_HI32(pChannel->pbGpuVA + pChannel->finishPayloadOffset)),
                       NVB0B5_SET_SEMAPHORE_B,
                       DRF_NUM(B0B5, _SET_SEMAPHORE_B, _LOWER, NvU64_LO32(pChannel->pbGpuVA + pChannel->finishPayloadOffset)),
                       NVB0B5_SET_SEMAPHORE_PAYLOAD,
                       pChannelPbInfo->payload);
    }

    // Side effect - pushed LAUNCH_DMA methods
    channelPushMethod(pChannel, pChannelPbInfo, bPipelined, bInsertFinishPayload, launchType, semaValue, &pPtr);

    channelAddHostSema(pChannel, putIndex, &pPtr);

    NvU32 methodSize = (NvU32)((NvU8*)pPtr - (NvU8*)pStartPtr);
    NV_ASSERT_OR_RETURN(methodSize <= pChannel->methodSizePerBlock, 0);
    return methodSize;
}

static NV_STATUS
addMethodsToMethodBuf
(
    NvU32 command,
    NvU32 data,
    NvU32 *pMethodBuf,
    NvU32  index
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 2730);

    NV_ASSERT_OR_RETURN((index < SEC2_WL_METHOD_ARRAY_SIZE), NV_ERR_INVALID_ARGUMENT);
    pMethodBuf[index*2 + 0] = command;
    pMethodBuf[index*2 + 1] = data;
    return NV_OK;
}

/** single helper function to fill the push buffer with the methods needed for
 *  memsetting using SEC2. This function is much more efficient in the sense it
 *  decouples the mem(set/copy) operation from managing channel resources.
 **/
NV_STATUS
channelFillSec2Pb
(
    OBJCHANNEL        *pChannel,
    NvU32              putIndex,
    NvBool             bInsertFinishPayload,
    CHANNEL_PB_INFO   *pChannelPbInfo,
    CCSL_CONTEXT      *pCcslCtx,
    MEMORY_DESCRIPTOR *pAuthTagBufMemDesc,
    MEMORY_DESCRIPTOR *pSemaMemDesc,
    NvU64              scrubMthdAuthTagBufGpuVA,
    NvU32              scrubAuthTagBufIndex,
    NvU64              semaMthdAuthTagBufGpuVA,
    NvU32              semaAuthTagBufIndex,
    NvU32             *pMethodLength
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 2731);
    NvU32          *pPtr                  = NULL;
    NvU32          *pStartPtr             = NULL;
    NvU32           execute               = 0;
    NvU32           methodIdx             = 0;
    NvU32          *pMethods              = NULL;
    NV_STATUS       status                = NV_OK;
    MemoryManager  *pMemoryManager        = NULL;
    NvU32           scrubAuthTagBufoffset = scrubAuthTagBufIndex * SHA_256_HASH_SIZE_BYTE;
    NvU32           semaAuthTagBufoffset  = semaAuthTagBufIndex * SHA_256_HASH_SIZE_BYTE;

    NV_ASSERT_OR_RETURN((pChannel != NULL), NV_ERR_INVALID_ARGUMENT);
    NV_ASSERT_OR_RETURN((pChannelPbInfo != NULL), NV_ERR_INVALID_ARGUMENT);
    NV_ASSERT_OR_RETURN((pCcslCtx != NULL), NV_ERR_INVALID_ARGUMENT);
    NV_ASSERT_OR_RETURN((pAuthTagBufMemDesc != NULL), NV_ERR_INVALID_ARGUMENT);
    NV_ASSERT_OR_RETURN((pSemaMemDesc != NULL), NV_ERR_INVALID_ARGUMENT);
    NV_ASSERT_OR_RETURN((pMethodLength != NULL), NV_ERR_INVALID_ARGUMENT);

    pPtr = (NvU32 *)((NvU8 *)pChannel->pbCpuVA + (putIndex * pChannel->methodSizePerBlock));
    pStartPtr = pPtr;
    pMemoryManager = GPU_GET_MEMORY_MANAGER(pChannel->pGpu);

    NV_PRINTF(LEVEL_INFO, "PutIndex: %x, PbOffset: %x\n", putIndex, putIndex * pChannel->methodSizePerBlock);
    NvU32 transferFlags = (TRANSFER_FLAGS_USE_BAR1     |
                           TRANSFER_FLAGS_SHADOW_ALLOC |
                           TRANSFER_FLAGS_SHADOW_INIT_MEM);
    NvU8* pScrubMethdAuthTagBuf = (NvU8 *)memmgrMemDescBeginTransfer(pMemoryManager, pAuthTagBufMemDesc, transferFlags);
    NV_ASSERT_OR_RETURN((pScrubMethdAuthTagBuf != NULL), NV_ERR_INVALID_STATE);

    NvU8* pSemaAuthTagBuf = (NvU8 *)memmgrMemDescBeginTransfer(pMemoryManager, pSemaMemDesc, transferFlags);
    NV_ASSERT_OR_RETURN((pSemaAuthTagBuf != NULL), NV_ERR_INVALID_STATE);

    pMethods = (NvU32*)portMemAllocNonPaged(sizeof(NvU32) * SEC2_WL_METHOD_ARRAY_SIZE * 2);
    NV_ASSERT_OR_RETURN(pMethods != NULL, NV_ERR_NO_MEMORY);
    portMemSet(pMethods, 0, (sizeof(NvU32) * SEC2_WL_METHOD_ARRAY_SIZE * 2));

    NV_ASSERT_OK_OR_GOTO(status, addMethodsToMethodBuf(NV906F_SET_OBJECT, pChannel->classEngineID, pMethods, methodIdx++), cleanup);
    NV_ASSERT_OK_OR_GOTO(status, addMethodsToMethodBuf(NVCBA2_DECRYPT_COPY_DST_ADDR_HI, NvU64_HI32(pChannelPbInfo->dstAddr), pMethods, methodIdx++), cleanup);
    NV_ASSERT_OK_OR_GOTO(status, addMethodsToMethodBuf(NVCBA2_DECRYPT_COPY_DST_ADDR_LO, NvU64_LO32(pChannelPbInfo->dstAddr), pMethods, methodIdx++), cleanup);
    NV_ASSERT_OK_OR_GOTO(status, addMethodsToMethodBuf(NVCBA2_DECRYPT_COPY_SIZE, pChannelPbInfo->size, pMethods, methodIdx++), cleanup);
    NV_ASSERT_OK_OR_GOTO(status, addMethodsToMethodBuf(NVCBA2_METHOD_STREAM_AUTH_TAG_ADDR_HI, NvU64_HI32(scrubMthdAuthTagBufGpuVA + scrubAuthTagBufoffset), pMethods, methodIdx++), cleanup);
    NV_ASSERT_OK_OR_GOTO(status, addMethodsToMethodBuf(NVCBA2_METHOD_STREAM_AUTH_TAG_ADDR_LO, NvU64_LO32(scrubMthdAuthTagBufGpuVA + scrubAuthTagBufoffset), pMethods, methodIdx++), cleanup);

    // Use this semaphore release to track the current method stream auth tag buffer being used
    execute |= FLD_SET_DRF(CBA2, _EXECUTE, _NOTIFY, _ENABLE, execute);
    execute |= FLD_SET_DRF(CBA2, _EXECUTE, _NOTIFY_ON, _END, execute);
    NV_ASSERT_OK_OR_GOTO(status, addMethodsToMethodBuf(NVCBA2_SEMAPHORE_A, NvU64_HI32(pChannel->pbGpuVA + pChannel->authTagBufSemaOffset), pMethods, methodIdx++), cleanup);
    NV_ASSERT_OK_OR_GOTO(status, addMethodsToMethodBuf(NVCBA2_SEMAPHORE_B, NvU64_LO32(pChannel->pbGpuVA + pChannel->authTagBufSemaOffset), pMethods, methodIdx++), cleanup);
    NV_ASSERT_OK_OR_GOTO(status, addMethodsToMethodBuf(NVCBA2_SET_SEMAPHORE_PAYLOAD_LOWER, scrubAuthTagBufIndex, pMethods, methodIdx++), cleanup);

    execute |= FLD_SET_DRF(CBA2, _EXECUTE, _PHYSICAL_SCRUBBER, _ENABLE, execute);
    NV_ASSERT_OK_OR_GOTO(status, addMethodsToMethodBuf(NVCBA2_EXECUTE, execute, pMethods, methodIdx++), cleanup);

    NvU32 hmacBufferSizeBytes = 2 * methodIdx * sizeof(NvU32); 
    NvU8  hmacDigest[SHA_256_HASH_SIZE_BYTE] = {0};
    NvU8* pBufScrub = &pScrubMethdAuthTagBuf[scrubAuthTagBufoffset];

    ccslSign(pCcslCtx, hmacBufferSizeBytes, (NvU8 *)pMethods, hmacDigest);

    portMemCopy((void*)pBufScrub, SHA_256_HASH_SIZE_BYTE, (const void*)&hmacDigest[0], SHA_256_HASH_SIZE_BYTE);

    if (methodIdx > SEC2_WL_METHOD_ARRAY_SIZE)
    {
        status = NV_ERR_INVALID_ARGUMENT;
        goto cleanup;
    }

    for (NvU32 i = 0; i < methodIdx; i++)
    {
        NV_PUSH_INC_1U(RM_SUBCHANNEL, pMethods[i*2 + 0], pMethods[i*2 + 1]);
    }

    // insert standalone semaphore to track sub operation completion
    if (bInsertFinishPayload)
    {
        NvU32 semaD = 0;
        methodIdx = 0;
        portMemSet(pMethods, 0, (sizeof(NvU32) * SEC2_WL_METHOD_ARRAY_SIZE * 2));
        semaD |= FLD_SET_DRF(CBA2, _SEMAPHORE_D, _FLUSH_DISABLE, _FALSE, execute);
        NV_ASSERT_OK_OR_GOTO(status, addMethodsToMethodBuf(NVCBA2_METHOD_STREAM_AUTH_TAG_ADDR_HI, NvU64_HI32(semaMthdAuthTagBufGpuVA + semaAuthTagBufoffset), pMethods, methodIdx++), cleanup);
        NV_ASSERT_OK_OR_GOTO(status, addMethodsToMethodBuf(NVCBA2_METHOD_STREAM_AUTH_TAG_ADDR_LO, NvU64_LO32(semaMthdAuthTagBufGpuVA + semaAuthTagBufoffset), pMethods, methodIdx++), cleanup);
        NV_ASSERT_OK_OR_GOTO(status, addMethodsToMethodBuf(NVCBA2_SEMAPHORE_A, NvU64_HI32(pChannel->pbGpuVA + pChannel->finishPayloadOffset), pMethods, methodIdx++), cleanup);
        NV_ASSERT_OK_OR_GOTO(status, addMethodsToMethodBuf(NVCBA2_SEMAPHORE_B, NvU64_LO32(pChannel->pbGpuVA + pChannel->finishPayloadOffset), pMethods, methodIdx++), cleanup);
        NV_ASSERT_OK_OR_GOTO(status, addMethodsToMethodBuf(NVCBA2_SET_SEMAPHORE_PAYLOAD_LOWER, pChannelPbInfo->payload, pMethods, methodIdx++), cleanup);
        NV_ASSERT_OK_OR_GOTO(status, addMethodsToMethodBuf(NVCBA2_SEMAPHORE_D, semaD, pMethods, methodIdx++), cleanup);

        hmacBufferSizeBytes = 2 * methodIdx * sizeof(NvU32); 
        portMemSet(&hmacDigest[0], 0, SHA_256_HASH_SIZE_BYTE);
        NvU8* pBufSema = &pSemaAuthTagBuf[semaAuthTagBufoffset];

        ccslSign(pCcslCtx, hmacBufferSizeBytes, (NvU8 *)pMethods, hmacDigest);

        portMemCopy((void*)pBufSema, SHA_256_HASH_SIZE_BYTE, (const void*)&hmacDigest[0], SHA_256_HASH_SIZE_BYTE);

        if (methodIdx > SEC2_WL_METHOD_ARRAY_SIZE)
        {
            status = NV_ERR_INVALID_ARGUMENT;
            goto cleanup;
        }

        for (NvU32 i = 0; i < methodIdx; i++)
        {
            NV_PUSH_INC_1U(RM_SUBCHANNEL, pMethods[i*2 + 0], pMethods[i*2 + 1]);
        }        
    }

    channelAddHostSema(pChannel, putIndex, &pPtr);

    *pMethodLength = 0;
    NvU32 methodSize = (NvU32)((NvU8*)pPtr - (NvU8*)pStartPtr);
    NV_ASSERT_OR_RETURN(methodSize <= pChannel->methodSizePerBlock, NV_ERR_INVALID_STATE);
    *pMethodLength = methodSize;

cleanup:
    if (pSemaAuthTagBuf != NULL)
    {
        memmgrMemDescEndTransfer(pMemoryManager, pSemaMemDesc, transferFlags);
    }
    if (pScrubMethdAuthTagBuf != NULL)
    {
        memmgrMemDescEndTransfer(pMemoryManager, pAuthTagBufMemDesc, transferFlags);
    }
    portMemFree(pMethods);
    return status;
}

/*** Implementation for static methods ***/
static NvU32 
channelPushMemoryProperties
(
    OBJCHANNEL      *pChannel,
    CHANNEL_PB_INFO *pChannelPbInfo,
    NvU32           **ppPtr
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 2732);
    NV_ADDRESS_SPACE dstAddressSpace = pChannelPbInfo->dstAddressSpace;
    NV_ADDRESS_SPACE srcAddressSpace = pChannelPbInfo->srcAddressSpace;
    NvU32 dstCpuCacheAttrib = pChannelPbInfo->dstCpuCacheAttrib;
    NvU32 srcCpuCacheAttrib = pChannelPbInfo->srcCpuCacheAttrib;

    NvU32 data = 0;
    NvU32 retVal = 0;
    NvU32 *pPtr = *ppPtr;
    
    if (!pChannelPbInfo->bCeMemcopy)
    {
        // If memset, push remap components
        NV_PUSH_INC_1U(RM_SUBCHANNEL, NVB0B5_SET_REMAP_CONST_A, pChannelPbInfo->pattern);
        NV_PUSH_INC_1U(RM_SUBCHANNEL, NVB0B5_SET_REMAP_COMPONENTS,
                       DRF_DEF(B0B5, _SET_REMAP_COMPONENTS, _DST_X, _CONST_A)          |
                       DRF_DEF(B0B5, _SET_REMAP_COMPONENTS, _COMPONENT_SIZE, _ONE)     |
                       DRF_DEF(B0B5, _SET_REMAP_COMPONENTS, _NUM_DST_COMPONENTS, _ONE));
    }
    else
    {
        // If memcopy, push src addr properties
        if (srcAddressSpace == ADDR_FBMEM)
        {
            data = DRF_DEF(B0B5, _SET_SRC_PHYS_MODE, _TARGET, _LOCAL_FB);
        }
        else if (srcCpuCacheAttrib == NV_MEMORY_CACHED)
        {
            data = DRF_DEF(B0B5, _SET_SRC_PHYS_MODE, _TARGET, _COHERENT_SYSMEM);
        }
        else
        {
            data = DRF_DEF(B0B5, _SET_SRC_PHYS_MODE, _TARGET, _NONCOHERENT_SYSMEM);
        }
        
        NV_PUSH_INC_1U(RM_SUBCHANNEL, NVB0B5_SET_SRC_PHYS_MODE, data);
    }

    // Push dst addr properties
    if (dstAddressSpace == ADDR_FBMEM)
    {
        data = DRF_DEF(B0B5, _SET_DST_PHYS_MODE, _TARGET, _LOCAL_FB);
    }
    else if (dstCpuCacheAttrib == NV_MEMORY_CACHED)
    {
        data = DRF_DEF(B0B5, _SET_DST_PHYS_MODE, _TARGET, _COHERENT_SYSMEM);
    }
    else
    {
        data = DRF_DEF(B0B5, _SET_DST_PHYS_MODE, _TARGET, _NONCOHERENT_SYSMEM);
    }
    
    NV_PUSH_INC_1U(RM_SUBCHANNEL, NVB0B5_SET_DST_PHYS_MODE, data);
    
    if (pChannel->bUseVasForCeCopy)
    {
        retVal = DRF_DEF(B0B5, _LAUNCH_DMA, _DST_TYPE, _VIRTUAL) | 
                 DRF_DEF(B0B5, _LAUNCH_DMA, _SRC_TYPE, _VIRTUAL);
    }
    else
    {
        retVal = DRF_DEF(B0B5, _LAUNCH_DMA, _DST_TYPE, _PHYSICAL) |
                 DRF_DEF(B0B5, _LAUNCH_DMA, _SRC_TYPE, _PHYSICAL);
    }

    *ppPtr = pPtr;
    return retVal;
}


static void
channelPushAddress
(
    OBJCHANNEL      *pChannel,
    CHANNEL_PB_INFO *pChannelPbInfo,
    NvU32           **ppPtr
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 2733);
    NvU64 dstAddr = pChannelPbInfo->dstAddr;
    NvU64 srcAddr = pChannelPbInfo->srcAddr;
    NvU32 *pPtr = *ppPtr;

    // Linear translation to map the PA to VA
    if (pChannel->bUseVasForCeCopy)
    {
        dstAddr = dstAddr + pChannel->fbAliasVA - pChannel->startFbOffset;
        srcAddr = srcAddr + pChannel->fbAliasVA - pChannel->startFbOffset;
    }

    // Set dst addr 
    NV_PUSH_INC_2U(RM_SUBCHANNEL, NVB0B5_OFFSET_OUT_UPPER,
                   DRF_NUM(B0B5, _OFFSET_OUT_UPPER, _UPPER, NvU64_HI32(dstAddr)),
                   NVB0B5_OFFSET_OUT_LOWER,
                   DRF_NUM(B0B5, _OFFSET_OUT_LOWER, _VALUE, NvU64_LO32(dstAddr)));

    // Set src addr if memcopy
    if (pChannelPbInfo->bCeMemcopy)
    {
        NV_PUSH_INC_2U(RM_SUBCHANNEL, NVB0B5_OFFSET_IN_UPPER,
                       DRF_NUM(B0B5, _OFFSET_IN_UPPER, _UPPER, NvU64_HI32(srcAddr)),
                       NVB0B5_OFFSET_IN_LOWER,
                       DRF_NUM(B0B5, _OFFSET_IN_LOWER, _VALUE, NvU64_LO32(srcAddr)));
    }

    *ppPtr = pPtr;
}


static void
channelPushMethod
(
    OBJCHANNEL      *pChannel,
    CHANNEL_PB_INFO *pChannelPbInfo,
    NvBool           bPipelined,
    NvBool           bInsertFinishPayload,
    NvU32            launchType,
    NvU32            semaValue,
    NvU32          **ppPtr
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 2734);
    NvU32 pipelinedValue = 0;
    NvU32 flushValue = 0;
    NvU32 disablePlcKind = 0;
    NvU32 launchParams = 0;
    NvU32 *pPtr = *ppPtr;

    if (bPipelined)
    {
        pipelinedValue = DRF_DEF(B0B5, _LAUNCH_DMA, _DATA_TRANSFER_TYPE, _PIPELINED);
    }
    else
    {
        pipelinedValue = DRF_DEF(B0B5, _LAUNCH_DMA, _DATA_TRANSFER_TYPE, _NON_PIPELINED);
    }

    if (bInsertFinishPayload)
    {
        flushValue = DRF_DEF(B0B5, _LAUNCH_DMA, _FLUSH_ENABLE, _TRUE);
    }
    else
    {
        flushValue = DRF_DEF(B0B5, _LAUNCH_DMA, _FLUSH_ENABLE, _FALSE);
    }

    if (!pChannelPbInfo->bCeMemcopy)
    {
        switch (pChannel->hTdCopyClass)
        {
            case MAXWELL_DMA_COPY_A:
            case PASCAL_DMA_COPY_A:
            case PASCAL_DMA_COPY_B:
            case VOLTA_DMA_COPY_A:
                disablePlcKind = 0;
                break;
            default: // For anything after Turing, set the kind
                disablePlcKind = DRF_DEF(C5B5, _LAUNCH_DMA, _DISABLE_PLC, _TRUE);
            break;
        }

        launchParams =  DRF_DEF(B0B5, _LAUNCH_DMA, _REMAP_ENABLE, _TRUE) | disablePlcKind;
    }

    NV_PUSH_INC_1U(RM_SUBCHANNEL, NVB0B5_LAUNCH_DMA, 
                   launchParams | 
                   DRF_DEF(B0B5, _LAUNCH_DMA, _SRC_MEMORY_LAYOUT, _PITCH) |
                   DRF_DEF(B0B5, _LAUNCH_DMA, _DST_MEMORY_LAYOUT, _PITCH) |
                   DRF_DEF(B0B5, _LAUNCH_DMA, _MULTI_LINE_ENABLE, _FALSE) |
                   launchType |
                   pipelinedValue |
                   flushValue |
                   semaValue);
    *ppPtr = pPtr;
}
