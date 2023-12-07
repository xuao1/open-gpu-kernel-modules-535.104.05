/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "gpu/mem_mgr/mem_mgr.h"
#include "gpu/mem_mgr/heap_base.h"
#include "gpu/mem_mgr/mem_utils.h"
#include "gpu/mem_mgr/virt_mem_allocator_common.h"
#include "os/nv_memory_type.h"
#include "core/locks.h"
#include "ctrl/ctrl2080.h"
#include "rmapi/rs_utils.h"
#include "gpu/subdevice/subdevice.h"

#include "kernel/gpu/fifo/kernel_fifo.h"
#include "kernel/gpu/fifo/kernel_channel.h"

#include "gpu/bus/kern_bus.h"

#include "kernel/gpu/conf_compute/ccsl.h"

#include "class/cl0005.h"      // NV01_EVENT

#include "ctrl/ctrla06f/ctrla06fgpfifo.h"

// Memory copy block size for if we need to cut up a mapping
#define MEMORY_COPY_BLOCK_SIZE 1024 * 1024

/* ------------------------ Private functions --------------------------------------- */

/*!
 * @brief This utility routine helps in determining the appropriate
 *        memory transfer technique to be used
 */
static TRANSFER_TYPE
memmgrGetMemTransferType
(
    MemoryManager    *pMemoryManager,
    TRANSFER_SURFACE *pDst,
    TRANSFER_SURFACE *pSrc
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3056);
    TRANSFER_TYPE transferType        = TRANSFER_TYPE_PROCESSOR;
    OBJGPU    *pGpu       = ENG_GET_GPU(pMemoryManager);
    KernelBus *pKernelBus = GPU_GET_KERNEL_BUS(pGpu);

    //
    // In case of copy, both dest and src will be passed
    // In case of memset/memread/memwrite either dest or src will be passed
    //
    if ((pDst != NULL) && (pSrc != NULL) &&
        (memdescGetAddressSpace(pDst->pMemDesc) == ADDR_SYSMEM) &&
        (memdescGetAddressSpace(pSrc->pMemDesc) == ADDR_SYSMEM))
    {
        transferType = TRANSFER_TYPE_PROCESSOR;
    }
    else if (((pDst != NULL) &&
             (memdescGetAddressSpace(pDst->pMemDesc) == ADDR_SYSMEM)) ||
             ((pSrc != NULL) &&
             (memdescGetAddressSpace(pSrc->pMemDesc) == ADDR_SYSMEM)))
    {
        transferType = TRANSFER_TYPE_PROCESSOR;
    }
    else if (kbusIsBarAccessBlocked(pKernelBus))
    {
        transferType = TRANSFER_TYPE_GSP_DMA;
    }
    return transferType;
}

static NV_STATUS
_memmgrAllocAndMapSurface
(
    OBJGPU             *pGpu,
    NvU64               size,
    MEMORY_DESCRIPTOR **ppMemDesc,
    void              **ppMap,
    void              **ppPriv
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3057);
    NV_STATUS status;
    NvU64 flags = 0;

    NV_ASSERT_OR_RETURN(ppMemDesc != NULL, NV_ERR_INVALID_ARGUMENT);
    NV_ASSERT_OR_RETURN(ppMap != NULL, NV_ERR_INVALID_ARGUMENT);
    NV_ASSERT_OR_RETURN(ppPriv != NULL, NV_ERR_INVALID_ARGUMENT);

    flags = MEMDESC_FLAGS_ALLOC_IN_UNPROTECTED_MEMORY;

    NV_ASSERT_OK_OR_RETURN(
        memdescCreate(ppMemDesc, pGpu, size, RM_PAGE_SIZE, NV_TRUE,
                      ADDR_SYSMEM, NV_MEMORY_UNCACHED, flags));

    NV_ASSERT_OK_OR_GOTO(status, memdescAlloc(*ppMemDesc), failed);

    NV_ASSERT_OK_OR_GOTO(status,
        memdescMapOld(*ppMemDesc, 0, size, NV_TRUE, NV_PROTECT_READ_WRITE,
                      ppMap, ppPriv),
        failed);

    // Clear surface before use
    portMemSet(*ppMap, 0, size);

    return NV_OK;
failed:
    memdescFree(*ppMemDesc);
    memdescDestroy(*ppMemDesc);

    *ppMemDesc = NULL;
    *ppMap = NULL;
    *ppPriv = NULL;

    return status;
}

static void
_memmgrUnmapAndFreeSurface
(
    MEMORY_DESCRIPTOR *pMemDesc,
    void              *pMap,
    void              *pPriv
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3058);
    memdescUnmapOld(pMemDesc, NV_TRUE, 0, pMap, pPriv);

    memdescFree(pMemDesc);
    memdescDestroy(pMemDesc);
}

/*!
 * @brief This function is used for writing/reading data to/from a client
 *        provided buffer from/to some source region in vidmem
 *
 * @param[in] pDst    TRANSFER_SURFACE info for destination region
 * @param[in] pBuf    Client provided buffer
 * @param[in] size    Size in bytes of the memory transfer
 * @param[in] bRead   TRUE for read and FALSE for write
 */
static NV_STATUS
_memmgrMemReadOrWriteWithGsp
(
    OBJGPU           *pGpu,
    TRANSFER_SURFACE *pDst,
    void             *pBuf,
    NvU64             size,
    NvBool            bRead
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3059);
    NV2080_CTRL_INTERNAL_MEMMGR_MEMORY_TRANSFER_WITH_GSP_PARAMS gspParams;
    NV_STATUS status;
    MEMORY_DESCRIPTOR *pStagingBuf = NULL;
    void *pStagingBufMap = NULL;
    void *pStagingBufPriv = NULL;
    RM_API *pRmApi = GPU_GET_PHYSICAL_RMAPI(pGpu);
    ConfidentialCompute *pConfCompute = GPU_GET_CONF_COMPUTE(pGpu);
    if (gpuIsCCFeatureEnabled(pGpu))
    {
        NV_ASSERT_OR_RETURN(pConfCompute->getProperty(pCC,
                                    PDB_PROP_CONFCOMPUTE_ENCRYPT_ENABLED),
                            NV_ERR_INVALID_STATE);
    }

    // Do not expect GSP to be used for reading/writing from/to sysmem
    if (memdescGetAddressSpace(pDst->pMemDesc) == ADDR_SYSMEM)
        return NV_ERR_NOT_SUPPORTED;

    // Allocate and map the staging buffer
    NV_ASSERT_OK_OR_RETURN(
        _memmgrAllocAndMapSurface(pGpu, size, &pStagingBuf, &pStagingBufMap,
                                  &pStagingBufPriv));

    // Setup control call params
    portMemSet(&gspParams, 0, sizeof(gspParams));

    // Copy the data to staging buffer before poking GSP for copying
    if (!bRead)
    {
        if (gpuIsCCFeatureEnabled(pGpu))
        {
            NV_ASSERT_OK_OR_GOTO(status,
                ccslEncrypt_HAL(pConfCompute->pDmaCcslCtx, size, pBuf, NULL, 0,
                                pStagingBufMap, gspParams.authTag),
                failed);
        }
        else
        {
            portMemCopy(pStagingBufMap, size, pBuf, size);
        }
    }

    gspParams.memop = NV2080_CTRL_MEMMGR_MEMORY_OP_MEMCPY;
    gspParams.transferSize = size;

    if (bRead)
    {
        // Source surface in vidmem
        gspParams.src.baseAddr = memdescGetPhysAddr(pDst->pMemDesc, AT_GPU, 0);
        gspParams.src.size = memdescGetSize(pDst->pMemDesc);
        gspParams.src.offset = pDst->offset;
        gspParams.src.cpuCacheAttrib = memdescGetCpuCacheAttrib(pDst->pMemDesc);
        gspParams.src.aperture = memdescGetAddressSpace(pDst->pMemDesc);

        // Destination surface in unprotected sysmem
        gspParams.dst.baseAddr = memdescGetPhysAddr(pStagingBuf, AT_GPU, 0);
        gspParams.dst.size = memdescGetSize(pStagingBuf);
        gspParams.dst.offset = 0;
        gspParams.dst.cpuCacheAttrib = memdescGetCpuCacheAttrib(pStagingBuf);
        gspParams.dst.aperture = memdescGetAddressSpace(pStagingBuf);
    }
    else
    {
        // Source surface in unprotected sysmem
        gspParams.src.baseAddr = memdescGetPhysAddr(pStagingBuf, AT_GPU, 0);
        gspParams.src.size = memdescGetSize(pStagingBuf);
        gspParams.src.offset = 0;
        gspParams.src.cpuCacheAttrib = memdescGetCpuCacheAttrib(pStagingBuf);
        gspParams.src.aperture = memdescGetAddressSpace(pStagingBuf);

        // Destination surface in vidmem
        gspParams.dst.baseAddr = memdescGetPhysAddr(pDst->pMemDesc, AT_GPU, 0);
        gspParams.dst.size = memdescGetSize(pDst->pMemDesc);
        gspParams.dst.offset = pDst->offset;
        gspParams.dst.cpuCacheAttrib = memdescGetCpuCacheAttrib(pDst->pMemDesc);
        gspParams.dst.aperture = memdescGetAddressSpace(pDst->pMemDesc);
    }

    // Send the control call
    NV_ASSERT_OK_OR_GOTO(status,
        pRmApi->Control(pRmApi,
                        pGpu->hInternalClient,
                        pGpu->hInternalSubdevice,
                        NV2080_CTRL_CMD_INTERNAL_MEMMGR_MEMORY_TRANSFER_WITH_GSP,
                        &gspParams,
                        sizeof(gspParams)),
        failed);

    // Read contents from staging buffer after GSP is done copying
    if (bRead)
    {
        if (gpuIsCCFeatureEnabled(pGpu))
        {
            NV_ASSERT_OK_OR_GOTO(status,
                ccslDecrypt_HAL(pConfCompute->pDmaCcslCtx, size, pStagingBufMap,
                                NULL, NULL, 0, pBuf, gspParams.authTag),
                failed);
        }
        else
        {
            portMemCopy(pBuf, size, pStagingBufMap, size);
        }
    }

failed:
    _memmgrUnmapAndFreeSurface(pStagingBuf, pStagingBufMap, pStagingBufPriv);
    return status;
}

/*!
 * @brief This function is used for copying data b/w two memory regions
 *        using GSP.
 *
 * @param[in] pDst    TRANSFER_SURFACE info for destination region
 * @param[in] pSrc    TRANSFER_SURFACE info for source region
 * @param[in] size    Size in bytes of the memory transfer
 */
static NV_STATUS
_memmgrMemcpyWithGsp
(
    OBJGPU           *pGpu,
    TRANSFER_SURFACE *pDst,
    TRANSFER_SURFACE *pSrc,
    NvU64             size
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3060);
    NV2080_CTRL_INTERNAL_MEMMGR_MEMORY_TRANSFER_WITH_GSP_PARAMS gspParams;
    NV_STATUS status;
    MEMORY_DESCRIPTOR *pStagingBuf = NULL;
    void *pStagingBufMap = NULL;
    void *pStagingBufPriv = NULL;
    NvU8 *pMap = NULL;
    void *pPriv = NULL;
    RM_API *pRmApi = GPU_GET_PHYSICAL_RMAPI(pGpu);

    //
    // Do not expect GSP to be used for copying data b/w two surfaces
    // in sysmem. For SPT, there is no non-CPR vidmem. So, allow vidmem
    // to vidmem copies in plain text. For copies b/w CPR and non-CPR
    // vidmem, encryption/decryption needs to happen at the endpoints.
    //
    if (memdescGetAddressSpace(pSrc->pMemDesc) == ADDR_SYSMEM &&
        memdescGetAddressSpace(pDst->pMemDesc) == ADDR_SYSMEM)
    {
        return NV_ERR_NOT_SUPPORTED;
    }

    // Allocate and map the bounce buffer
    NV_ASSERT_OK_OR_RETURN(
        _memmgrAllocAndMapSurface(pGpu, size, &pStagingBuf, &pStagingBufMap,
                                  &pStagingBufPriv));

    // Setup control call params
    portMemSet(&gspParams, 0, sizeof(gspParams));

    gspParams.memop = NV2080_CTRL_MEMMGR_MEMORY_OP_MEMCPY;
    gspParams.transferSize = size;

    if (memdescGetAddressSpace(pSrc->pMemDesc) == ADDR_SYSMEM)
    {
        NV_ASSERT_OK_OR_GOTO(status,
            memdescMapOld(pSrc->pMemDesc, 0, size, NV_TRUE,
                          NV_PROTECT_READ_WRITE, (void**)&pMap, &pPriv),
            failed);

        // Copy to staging buffer
        portMemCopy(pStagingBufMap, size, pMap + pSrc->offset, size);

        memdescUnmapOld(pSrc->pMemDesc, NV_TRUE, 0, (void*)pMap, pPriv);

        // Source surface in unprotected sysmem
        gspParams.src.baseAddr = memdescGetPhysAddr(pStagingBuf, AT_GPU, 0);
        gspParams.src.size = memdescGetSize(pStagingBuf);
        gspParams.src.offset = 0;
        gspParams.src.cpuCacheAttrib = memdescGetCpuCacheAttrib(pStagingBuf);
        gspParams.src.aperture = memdescGetAddressSpace(pStagingBuf);

        // Destination surface in vidmem
        gspParams.dst.baseAddr = memdescGetPhysAddr(pDst->pMemDesc, AT_GPU, 0);
        gspParams.dst.size = memdescGetSize(pDst->pMemDesc);
        gspParams.dst.offset = pDst->offset;
        gspParams.dst.cpuCacheAttrib = memdescGetCpuCacheAttrib(pDst->pMemDesc);
        gspParams.dst.aperture = memdescGetAddressSpace(pDst->pMemDesc);
    }
    else
    {
        // Source surface in vidmem
        gspParams.src.baseAddr = memdescGetPhysAddr(pSrc->pMemDesc, AT_GPU, 0);
        gspParams.src.size = memdescGetSize(pSrc->pMemDesc);
        gspParams.src.offset = pSrc->offset;
        gspParams.src.cpuCacheAttrib = memdescGetCpuCacheAttrib(pSrc->pMemDesc);
        gspParams.src.aperture = memdescGetAddressSpace(pSrc->pMemDesc);

        if (memdescGetAddressSpace(pDst->pMemDesc) == ADDR_FBMEM)
        {
            // Destination surface in vidmem
            gspParams.dst.baseAddr = memdescGetPhysAddr(pDst->pMemDesc, AT_GPU, 0);
            gspParams.dst.size = memdescGetSize(pDst->pMemDesc);
            gspParams.dst.offset = pDst->offset;
            gspParams.dst.cpuCacheAttrib = memdescGetCpuCacheAttrib(pDst->pMemDesc);
            gspParams.dst.aperture = memdescGetAddressSpace(pDst->pMemDesc);
        }
        else
        {
            // Destination surface in unprotected sysmem
            gspParams.dst.baseAddr = memdescGetPhysAddr(pStagingBuf, AT_GPU, 0);
            gspParams.dst.size = memdescGetSize(pStagingBuf);
            gspParams.dst.offset = 0;
            gspParams.dst.cpuCacheAttrib = memdescGetCpuCacheAttrib(pStagingBuf);
            gspParams.dst.aperture = memdescGetAddressSpace(pStagingBuf);
        }
    }

    // Send the control call
    NV_ASSERT_OK_OR_GOTO(status,
        pRmApi->Control(pRmApi,
                        pGpu->hInternalClient,
                        pGpu->hInternalSubdevice,
                        NV2080_CTRL_CMD_INTERNAL_MEMMGR_MEMORY_TRANSFER_WITH_GSP,
                        &gspParams,
                        sizeof(gspParams)),
        failed);

    // Copy from staging buffer to destination
    if (memdescGetAddressSpace(pDst->pMemDesc) == ADDR_SYSMEM)
    {
        NV_ASSERT_OK_OR_GOTO(status,
            memdescMapOld(pDst->pMemDesc, 0, size, NV_TRUE,
                          NV_PROTECT_READ_WRITE, (void**)&pMap, &pPriv),
            failed);

        portMemCopy(pMap + pDst->offset, size, pStagingBufMap, size);

        memdescUnmapOld(pDst->pMemDesc, NV_TRUE, 0, (void*)pMap, pPriv);
    }

failed:
    _memmgrUnmapAndFreeSurface(pStagingBuf, pStagingBufMap, pStagingBufPriv);
    return status;
}

static NV_STATUS
_memmgrMemsetWithGsp
(
    OBJGPU           *pGpu,
    TRANSFER_SURFACE *pDst,
    NvU32             value,
    NvU64             size
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3061);
    NV2080_CTRL_INTERNAL_MEMMGR_MEMORY_TRANSFER_WITH_GSP_PARAMS gspParams;
    RM_API *pRmApi = GPU_GET_PHYSICAL_RMAPI(pGpu);

    // Do not expect to use GSP to memset surfaces in sysmem
    if (memdescGetAddressSpace(pDst->pMemDesc) == ADDR_SYSMEM)
        return NV_ERR_NOT_SUPPORTED;

    portMemSet(&gspParams, 0, sizeof(gspParams));

    gspParams.memop = NV2080_CTRL_MEMMGR_MEMORY_OP_MEMSET;
    gspParams.transferSize = size;
    gspParams.value = value;
    gspParams.dst.baseAddr = memdescGetPhysAddr(pDst->pMemDesc, AT_GPU, 0);
    gspParams.dst.size = memdescGetSize(pDst->pMemDesc);
    gspParams.dst.offset = pDst->offset;
    gspParams.dst.cpuCacheAttrib = memdescGetCpuCacheAttrib(pDst->pMemDesc);
    gspParams.dst.aperture = memdescGetAddressSpace(pDst->pMemDesc);

    // Send the control call
    NV_ASSERT_OK_OR_RETURN(
        pRmApi->Control(pRmApi,
                        pGpu->hInternalClient,
                        pGpu->hInternalSubdevice,
                        NV2080_CTRL_CMD_INTERNAL_MEMMGR_MEMORY_TRANSFER_WITH_GSP,
                        &gspParams,
                        sizeof(gspParams)));

    return NV_OK;
}

/*!
 * @brief This function is used for copying data b/w two memory regions
 *        using the specified memory transfer technique. Both memory regions
 *        can be in the same aperture or in different apertures.
 *
 * @param[in] pDstInfo      TRANSFER_SURFACE info for destination region
 * @param[in] pSrcInfo      TRANSFER_SURFACE info for source region
 * @param[in] size          Size in bytes of the memory transfer
 * @param[in] transferType  Memory transfer technique to be used
 * @param[in] flags         Flags
 */
static NV_STATUS
memmgrMemCopyWithTransferType
(
    MemoryManager    *pMemoryManager,
    TRANSFER_SURFACE *pDstInfo,
    TRANSFER_SURFACE *pSrcInfo,
    NvU32             size,
    TRANSFER_TYPE     transferType,
    NvU32             flags
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3062);
    OBJGPU *pGpu = ENG_GET_GPU(pMemoryManager);
    NvU8 *pSrc;
    NvU8 *pDst;

    // Sanitize the input
    NV_ASSERT_OR_RETURN(pDstInfo != NULL, NV_ERR_INVALID_ARGUMENT);
    NV_ASSERT_OR_RETURN(pSrcInfo != NULL, NV_ERR_INVALID_ARGUMENT);
    NV_ASSERT_OR_RETURN(pDstInfo->pMemDesc != NULL, NV_ERR_INVALID_ARGUMENT);
    NV_ASSERT_OR_RETURN(pSrcInfo->pMemDesc != NULL, NV_ERR_INVALID_ARGUMENT);
    NV_ASSERT_OR_RETURN(!memdescDescIsEqual(pDstInfo->pMemDesc, pSrcInfo->pMemDesc),
                        NV_ERR_INVALID_ARGUMENT);

    switch (transferType)
    {
        case TRANSFER_TYPE_PROCESSOR:
            pDst = memdescMapInternal(pGpu, pDstInfo->pMemDesc, TRANSFER_FLAGS_NONE);
            NV_ASSERT_OR_RETURN(pDst != NULL, NV_ERR_INSUFFICIENT_RESOURCES);
            pSrc = memdescMapInternal(pGpu, pSrcInfo->pMemDesc, TRANSFER_FLAGS_NONE);
            if (pSrc == NULL)
            {
                memdescUnmapInternal(pGpu, pDstInfo->pMemDesc, 0);
                NV_ASSERT_OR_RETURN(0, NV_ERR_INSUFFICIENT_RESOURCES);
            }

            portMemCopy(pDst + pDstInfo->offset, size, pSrc + pSrcInfo->offset, size);

            memdescUnmapInternal(pGpu, pSrcInfo->pMemDesc, TRANSFER_FLAGS_NONE);
            memdescUnmapInternal(pGpu, pDstInfo->pMemDesc, flags);
            break;
        case TRANSFER_TYPE_GSP_DMA:
            if (IS_GSP_CLIENT(pGpu))
            {
                NV_ASSERT_OK_OR_RETURN(
                    _memmgrMemcpyWithGsp(pGpu, pDstInfo, pSrcInfo, size));
            }
            else
            {
                NV_ASSERT_OR_RETURN(0, NV_ERR_INVALID_ARGUMENT);
            }
            break;
        case TRANSFER_TYPE_CE:
            NV_PRINTF(LEVEL_INFO, "Add call to CE\n");
            break;
    }

    return NV_OK;
}

/*!
 * @brief This function is used for setting a memory region to a constant state
 *        using a specified memory transfer technique
 *
 * @param[in] pDstInfo      TRANSFER_SURFACE info for destination region
 * @param[in] value         Value to be written to the region
 * @param[in] size          Size in bytes of the memory to be initialized
 * @param[in] transferType  Memory transfer technique to be used
 * @param[in] flags         Flags
 */
static NV_STATUS
memmgrMemSetWithTransferType
(
    MemoryManager    *pMemoryManager,
    TRANSFER_SURFACE *pDstInfo,
    NvU32             value,
    NvU32             size,
    TRANSFER_TYPE     transferType,
    NvU32             flags
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3063);
    OBJGPU *pGpu = ENG_GET_GPU(pMemoryManager);
    NvU8 *pDst;

    // Sanitize the input
    NV_ASSERT_OR_RETURN(pDstInfo != NULL, NV_ERR_INVALID_ARGUMENT);
    NV_ASSERT_OR_RETURN(pDstInfo->pMemDesc != NULL, NV_ERR_INVALID_ARGUMENT);
    NV_ASSERT_OR_RETURN(size > 0, NV_ERR_INVALID_ARGUMENT);
    NV_ASSERT_OR_RETURN(pDstInfo->offset + size <= pDstInfo->pMemDesc->Size, NV_ERR_INVALID_ARGUMENT);

    switch (transferType)
    {
        case TRANSFER_TYPE_PROCESSOR:
            pDst = memdescMapInternal(pGpu, pDstInfo->pMemDesc, TRANSFER_FLAGS_NONE);
            NV_ASSERT_OR_RETURN(pDst != NULL, NV_ERR_INSUFFICIENT_RESOURCES);

            portMemSet(pDst + pDstInfo->offset, value, size);

            memdescUnmapInternal(pGpu, pDstInfo->pMemDesc, flags);
            break;
        case TRANSFER_TYPE_GSP_DMA:
            if (IS_GSP_CLIENT(pGpu))
            {
                NV_ASSERT_OK_OR_RETURN(
                    _memmgrMemsetWithGsp(pGpu, pDstInfo, value, size));
            }
            else
            {
                NV_ASSERT_OR_RETURN(0, NV_ERR_INVALID_ARGUMENT);
            }
            break;
        case TRANSFER_TYPE_CE:
            NV_PRINTF(LEVEL_INFO, "Add call to CE\n");
            break;
    }

    return NV_OK;
}

/*!
 * @brief This function is used to map the appropriate memory descriptor,
 *        copy the memory from the given buffer, and then unmap.
 *
 * @param[in] pMemDesc Memory descriptor of buffer to write
 * @param[in] pBuf     Buffer allocated by caller
 * @param[in] offset   Offset of buffer to write
 * @param[in] size     Size in bytes of the buffer
 * @param[in] flags    Flags
 */
static NV_STATUS
memmgrMemWriteMapAndCopy
(
    MemoryManager     *pMemoryManager,
    MEMORY_DESCRIPTOR *pMemDesc,
    void              *pBuf,
    NvU64              offset,
    NvU64              size,
    NvU32              flags
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3064);
    NvU8   *pDst = NULL;
    OBJGPU *pGpu = ENG_GET_GPU(pMemoryManager);

    pDst = memdescMapInternal(pGpu, pMemDesc, TRANSFER_FLAGS_NONE);
    NV_CHECK_OR_RETURN(LEVEL_SILENT, pDst != NULL, NV_ERR_INSUFFICIENT_RESOURCES);

    portMemCopy(pDst + offset, size, pBuf, size);
    memdescUnmapInternal(pGpu, pMemDesc, flags);

    return NV_OK;
}

/*!
 * @brief This function is used for writing data placed in a caller passed buffer
 *        to a given memory region while only mapping regions as large as the given
 *        block size.
 *
 * @param[in] pMemDesc   Memory descriptor of buffer to write
 * @param[in] pBuf       Buffer allocated by caller
 * @param[in] baseOffset Offset of entire buffer to write
 * @param[in] size       Size in bytes of the buffer
 * @param[in] flags      Flags
 * @param[in] blockSize  Maximum size of a mapping to use
 */
static NV_STATUS
memmgrMemWriteInBlocks
(
    MemoryManager     *pMemoryManager,
    MEMORY_DESCRIPTOR *pMemDesc,
    void              *pBuf,
    NvU64              baseOffset,
    NvU64              size,
    NvU32              flags,
    NvU32              blockSize
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3065);
    NV_STATUS  status    = NV_OK;
    OBJGPU    *pGpu      = ENG_GET_GPU(pMemoryManager);
    NvU64      remaining = size;
    NvU64      offset    = 0;

    while ((remaining > 0) && (status == NV_OK))
    {
        MEMORY_DESCRIPTOR *pSubMemDesc = NULL;
        NvU32              mapSize     = NV_MIN(blockSize, remaining);

        NV_CHECK_OK_OR_RETURN(LEVEL_SILENT, memdescCreateSubMem(&pSubMemDesc, pMemDesc, pGpu, offset + baseOffset, mapSize));

        // Set the offset to 0, as the sub descriptor already starts at the offset
        status = memmgrMemWriteMapAndCopy(pMemoryManager, pSubMemDesc, (NvU8 *)pBuf + offset,
                                          0, mapSize, flags);

        memdescFree(pSubMemDesc);
        memdescDestroy(pSubMemDesc);

        offset += mapSize;
        remaining -= mapSize;
    }

    return status;
}

/*!
 * @brief This function is used for writing data placed in a caller passed buffer
 *        to a given memory region using the specified memory transfer technique
 *
 * @param[in] pDstInfo      TRANSFER_SURFACE info for the destination region
 * @param[in] pBuf          Buffer allocated by caller
 * @param[in] size          Size in bytes of the buffer
 * @param[in] transferType  Memory transfer technique to be used
 * @param[in] flags         Flags
 */
static NV_STATUS
memmgrMemWriteWithTransferType
(
    MemoryManager    *pMemoryManager,
    TRANSFER_SURFACE *pDstInfo,
    void             *pBuf,
    NvU64             size,
    TRANSFER_TYPE     transferType,
    NvU32             flags
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3066);
    NvU8 *pMapping = memdescGetKernelMapping(pDstInfo->pMemDesc);
    OBJGPU *pGpu = ENG_GET_GPU(pMemoryManager);

    // Sanitize the input
    NV_ASSERT_OR_RETURN(pDstInfo != NULL, NV_ERR_INVALID_ARGUMENT);
    NV_ASSERT_OR_RETURN(pDstInfo->pMemDesc != NULL, NV_ERR_INVALID_ARGUMENT);
    NV_ASSERT_OR_RETURN(pBuf != NULL, NV_ERR_INVALID_ARGUMENT);
    NV_ASSERT_OR_RETURN(size > 0, NV_ERR_INVALID_ARGUMENT);
    NV_ASSERT_OR_RETURN(pDstInfo->offset + size <= pDstInfo->pMemDesc->Size, NV_ERR_INVALID_ARGUMENT);

    if (pMapping != NULL)
    {
        portMemCopy(pMapping + pDstInfo->offset, size, pBuf, size);
        return NV_OK;
    }

    switch (transferType)
    {
        case TRANSFER_TYPE_PROCESSOR:
            if (memmgrMemWriteMapAndCopy(pMemoryManager, pDstInfo->pMemDesc, pBuf, pDstInfo->offset, size, flags) != NV_OK)
            {
                // If we fail to map a block large enough for the entire transfer, split up the mapping.
                NV_ASSERT_OK_OR_RETURN(memmgrMemWriteInBlocks(pMemoryManager, pDstInfo->pMemDesc, pBuf,
                                                              pDstInfo->offset, size, flags, MEMORY_COPY_BLOCK_SIZE));
            }
            break;
        case TRANSFER_TYPE_GSP_DMA:
            if (IS_GSP_CLIENT(pGpu))
            {
                NV_PRINTF(LEVEL_INFO, "Calling GSP DMA task\n");
                NV_ASSERT_OK_OR_RETURN(
                    _memmgrMemReadOrWriteWithGsp(pGpu, pDstInfo, pBuf, size,
                                                 NV_FALSE /* bRead */));
            }
            else
            {
                NV_ASSERT_OR_RETURN(0, NV_ERR_INVALID_ARGUMENT);
            }
            break;
        case TRANSFER_TYPE_CE:
            NV_PRINTF(LEVEL_INFO, "Add call to CE\n");
            break;
    }

    return NV_OK;
}

/*!
 * @brief This function is used for reading specified number of bytes from
 *        a source memory region into a caller passed buffer using a specified
 *        memory transfer technique
 *
 * @param[in] pSrcInfo      TRANSFER_SURFACE info for the source region
 * @param[in] pBuf          Caller allocated buffer
 * @param[in] size          Size in bytes of the buffer
 * @param[in] transferType  Memory transfer technique to be used
 * @param[in] flags         Flags
 */
static NV_STATUS
memmgrMemReadWithTransferType
(
    MemoryManager    *pMemoryManager,
    TRANSFER_SURFACE *pSrcInfo,
    void             *pBuf,
    NvU64             size,
    TRANSFER_TYPE     transferType,
    NvU32             flags
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3067);
    OBJGPU *pGpu = ENG_GET_GPU(pMemoryManager);
    NvU8   *pSrc;
    NvU8   *pMapping = memdescGetKernelMapping(pSrcInfo->pMemDesc);


    // Sanitize the input
    NV_ASSERT_OR_RETURN(pSrcInfo != NULL, NV_ERR_INVALID_ARGUMENT);
    NV_ASSERT_OR_RETURN(pSrcInfo->pMemDesc != NULL, NV_ERR_INVALID_ARGUMENT);
    NV_ASSERT_OR_RETURN(pBuf != NULL, NV_ERR_INVALID_ARGUMENT);
    NV_ASSERT_OR_RETURN(size > 0, NV_ERR_INVALID_ARGUMENT);
    NV_ASSERT_OR_RETURN(pSrcInfo->offset + size <= pSrcInfo->pMemDesc->Size, NV_ERR_INVALID_ARGUMENT);

    if (pMapping != NULL)
    {
        portMemCopy(pBuf, size, pMapping + pSrcInfo->offset, size);
        return NV_OK;
    }

    switch (transferType)
    {
        case TRANSFER_TYPE_PROCESSOR:
            pSrc = memdescMapInternal(pGpu, pSrcInfo->pMemDesc, TRANSFER_FLAGS_NONE);
            NV_ASSERT_OR_RETURN(pSrc != NULL, NV_ERR_INSUFFICIENT_RESOURCES);

            portMemCopy(pBuf, size, pSrc + pSrcInfo->offset, size);

            memdescUnmapInternal(pGpu, pSrcInfo->pMemDesc, 0);
            break;
        case TRANSFER_TYPE_GSP_DMA:
            if (IS_GSP_CLIENT(pGpu))
            {
                NV_ASSERT_OK_OR_RETURN(
                    _memmgrMemReadOrWriteWithGsp(pGpu, pSrcInfo, pBuf, size,
                                                 NV_TRUE /* bRead */));
            }
            else
            {
                NV_ASSERT_OR_RETURN(0, NV_ERR_INVALID_ARGUMENT);
            }
            break;
        case TRANSFER_TYPE_CE:
            NV_PRINTF(LEVEL_INFO, "Add call to CE\n");
            break;
    }

    return NV_OK;
}

/* ------------------------ Public functions --------------------------------------- */

NvU64 memUtilsLeastCommonAlignment(NvU64 align1, NvU64 align2)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3068);
    NvU64 a, b;  // For Euclid's algorithm
    NvU64 lcm;   // Least Common Multiple of align1 and align2
    NvU64 maxAlignment = NV_U64_MAX;

    // WOLOG, make sure align1 >= align2.
    //
    if (align2 > align1)
    {
        NvU64 tmp = align1;
        align1 = align2;
        align2 = tmp;
    }

    // If align2 is 0, return min(align1, maxAlignment)
    //
    if (align2 == 0)
    {
        return align1 < maxAlignment ? align1 : maxAlignment;
    }

    // Use Euclid's algorithm (GCD(a, b) = GCD(b, a % b)) to find the
    // GCD of the two alignments, and use the GCD to find the LCM.
    //
    a = align1;
    b = align2;
    while (b != 0)
    {
        NvU64 old_a = a;
        a = b;
        b = old_a % b;
        NV_ASSERT(a > b);  // Ensure termination.  Should never fail.
    }
    lcm = align1 * (align2 / a);  // May overflow

    // Return min(lcm, maxAlignment).  Also return maxAlignment if the
    // lcm calculation overflowed, since that means it must have been
    // much bigger than maxAlignment.
    //
    if (lcm > maxAlignment || lcm < align1 ||
        0 != (lcm % align1) || 0 != (lcm % align2))
    {
        NV_CHECK_FAILED(LEVEL_ERROR, "Alignment limit exceeded");
        return maxAlignment;
    }
    return lcm;
}

void memUtilsInitFBAllocInfo
(
    NV_MEMORY_ALLOCATION_PARAMS *pAllocParams,
    FB_ALLOC_INFO *pFbAllocInfo,
    NvHandle hClient,
    NvHandle hDevice
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3069);
    pFbAllocInfo->pageFormat->type  = pAllocParams->type;
    pFbAllocInfo->owner             = pAllocParams->owner;
    pFbAllocInfo->hwResId           = 0;
    pFbAllocInfo->pad               = 0;
    pFbAllocInfo->alignPad          = 0;
    pFbAllocInfo->height            = pAllocParams->height;
    pFbAllocInfo->width             = pAllocParams->width;
    pFbAllocInfo->pitch             = pAllocParams->pitch;
    pFbAllocInfo->size              = pAllocParams->size;
    pFbAllocInfo->origSize          = pAllocParams->size;
    pFbAllocInfo->adjustedSize      = pAllocParams->size;
    pFbAllocInfo->offset            = ~0;
    pFbAllocInfo->pageFormat->flags = pAllocParams->flags;
    pFbAllocInfo->pageFormat->attr  = pAllocParams->attr;
    pFbAllocInfo->retAttr           = pAllocParams->attr;
    pFbAllocInfo->pageFormat->attr2 = pAllocParams->attr2;
    pFbAllocInfo->retAttr2          = pAllocParams->attr2;
    pFbAllocInfo->format            = pAllocParams->format;
    pFbAllocInfo->comprCovg         = pAllocParams->comprCovg;
    pFbAllocInfo->zcullCovg         = 0;
    pFbAllocInfo->ctagOffset        = pAllocParams->ctagOffset;
    pFbAllocInfo->bIsKernelAlloc    = NV_FALSE;
    pFbAllocInfo->internalflags     = 0;
    pFbAllocInfo->hClient           = hClient;
    pFbAllocInfo->hDevice           = hDevice;

    if ((pAllocParams->flags & NVOS32_ALLOC_FLAGS_ALIGNMENT_HINT) ||
        (pAllocParams->flags & NVOS32_ALLOC_FLAGS_ALIGNMENT_FORCE))
        pFbAllocInfo->align = pAllocParams->alignment;
    else
        pFbAllocInfo->align = RM_PAGE_SIZE;

    if (pAllocParams->flags & NVOS32_ALLOC_FLAGS_FIXED_ADDRESS_ALLOCATE)
    {
        pFbAllocInfo->offset = pAllocParams->offset;
        pFbAllocInfo->desiredOffset = pAllocParams->offset;
    }
}


MEMORY_DESCRIPTOR *
memmgrMemUtilsGetMemDescFromHandle_IMPL
(
    MemoryManager *pMemoryManager,
    NvHandle hClient,
    NvHandle hMemory
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3070);
    RsResourceRef *pMemoryRef;
    Memory        *pMemory;

    if (serverutilGetResourceRef(hClient, hMemory, &pMemoryRef) != NV_OK)
    {
        return NULL;
    }

    pMemory = dynamicCast(pMemoryRef->pResource, Memory);
    if (pMemory == NULL)
    {
        return NULL;
    }
    return pMemory->pMemDesc;
}

/*!
 * @brief This function is used for copying data b/w two memory regions
 *        Both memory regions can be in the same aperture of different apertures
 *
 * @param[in] pDstInfo  TRANSFER_SURFACE info for destination region
 * @param[in] pSrcInfo  TRANSFER_SURFACE info for source region
 * @param[in] size      Size in bytes of the memory transfer
 * @param[in] flags     Flags
 */
NV_STATUS
memmgrMemCopy_IMPL
(
    MemoryManager    *pMemoryManager,
    TRANSFER_SURFACE *pDstInfo,
    TRANSFER_SURFACE *pSrcInfo,
    NvU32             size,
    NvU32             flags
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3071);
    TRANSFER_TYPE transferType = memmgrGetMemTransferType(pMemoryManager,
                                                          pDstInfo, pSrcInfo);

    return memmgrMemCopyWithTransferType(pMemoryManager, pDstInfo, pSrcInfo,
                                         size, transferType, flags);
}

/*!
 * @brief This function is used for setting a memory region to a constant state
 *
 * @param[in] pDstInfo  TRANSFER_SURFACE info for the destination region
 * @param[in] value     Value to be written to the region
 * @param[in] size      Size in bytes of the memory to be initialized
 * @param[in] flags     Flags
 */
NV_STATUS
memmgrMemSet_IMPL
(
    MemoryManager    *pMemoryManager,
    TRANSFER_SURFACE *pDstInfo,
    NvU32             value,
    NvU32             size,
    NvU32             flags
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3072);
    TRANSFER_TYPE transferType = memmgrGetMemTransferType(pMemoryManager,
                                                          pDstInfo, NULL);

    return memmgrMemSetWithTransferType(pMemoryManager, pDstInfo, value,
                                        size, transferType, flags);
}

/*!
 * @brief This function is used for setting a memory region to a constant state
 *
 * @param[in] pMemDesc  Memory descriptor to end transfer to
 * @param[in] value     Value to be written to the region
 * @param[in] flags     Flags
 */
NV_STATUS
memmgrMemDescMemSet_IMPL
(
    MemoryManager     *pMemoryManager,
    MEMORY_DESCRIPTOR *pMemDesc,
    NvU32              value,
    NvU32              flags
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3073);
    TRANSFER_SURFACE transferSurface = {.offset = 0, .pMemDesc = pMemDesc};
    TRANSFER_TYPE    transferType = memmgrGetMemTransferType(pMemoryManager,
                                                             &transferSurface, NULL);

    return memmgrMemSetWithTransferType(pMemoryManager, &transferSurface, value,
                                        (NvU32)memdescGetSize(pMemDesc),
                                        transferType, flags);
}

/*!
 * @brief This function is used for writing data placed in a user buffer
 *        to a given memory region
 *
 * @param[in] pDstInfo  TRANSFER_SURFACE info for the destination region
 * @param[in] pBuf      Buffer allocated by caller
 * @param[in] size      Size in bytes of the buffer
 * @param[in] flags     Flags
 */
NV_STATUS
memmgrMemWrite_IMPL
(
    MemoryManager    *pMemoryManager,
    TRANSFER_SURFACE *pDstInfo,
    void             *pBuf,
    NvU64             size,
    NvU32             flags
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3074);
    TRANSFER_TYPE transferType = memmgrGetMemTransferType(pMemoryManager,
                                                          pDstInfo, NULL);

    return memmgrMemWriteWithTransferType(pMemoryManager, pDstInfo, pBuf,
                                          size, transferType, flags);
}

/*!
 * @brief This function is used for reading specified number of bytes from
 *        a source memory region into a caller passed buffer
 *
 * @param[in] pSrcInfo  TRANSFER_SURFACE info for the source region
 * @param[in] pBuf      Caller allocated buffer
 * @param[in] size      Size in bytes of the buffer
 * @param[in] flags     Flags
 */
NV_STATUS
memmgrMemRead_IMPL
(
    MemoryManager    *pMemoryManager,
    TRANSFER_SURFACE *pSrcInfo,
    void             *pBuf,
    NvU64             size,
    NvU32             flags
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3075);
    TRANSFER_TYPE transferType = memmgrGetMemTransferType(pMemoryManager,
                                                          NULL, pSrcInfo);

    return memmgrMemReadWithTransferType(pMemoryManager, pSrcInfo, pBuf,
                                         size, transferType, flags);
}

/*!
 * @brief This helper function can be used to begin transfers
 *
 * @param[in] pTransferInfo      Transfer information
 * @param[in] shadowBufSize      Size of allocated shadow buffer in case of shadow mapping
 * @param[in] flags              Flags
 */
NvU8 *
memmgrMemBeginTransfer_IMPL
(
    MemoryManager     *pMemoryManager,
    TRANSFER_SURFACE  *pTransferInfo,
    NvU64              shadowBufSize,
    NvU32              flags
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3076);
    TRANSFER_TYPE      transferType = memmgrGetMemTransferType(pMemoryManager,
                                                               pTransferInfo, NULL);
    MEMORY_DESCRIPTOR *pMemDesc     = pTransferInfo->pMemDesc;
    NvU64              offset       = pTransferInfo->offset;
    OBJGPU            *pGpu         = ENG_GET_GPU(pMemoryManager);
    NvU8              *pPtr         = NULL;
    NvU64              memSz        = 0;

    NV_ASSERT_OR_RETURN(pMemDesc != NULL, NULL);
    NV_ASSERT_OR_RETURN((memSz = memdescGetSize(pMemDesc)) >= shadowBufSize, NULL);
    NV_ASSERT_OR_RETURN(memdescGetKernelMapping(pMemDesc) == NULL, NULL);

    memSz = shadowBufSize == 0 ? memSz : shadowBufSize;

    switch (transferType)
    {
        case TRANSFER_TYPE_PROCESSOR:
            if (flags & TRANSFER_FLAGS_USE_BAR1)
            {
                NvP64 pPriv;
                NvU32 protect = NV_PROTECT_READ_WRITE;

                if (flags & TRANSFER_FLAGS_MAP_PROTECT_READABLE)
                {
                    protect = NV_PROTECT_READABLE;
                }
                else if (flags & TRANSFER_FLAGS_MAP_PROTECT_WRITEABLE)
                {
                    protect = NV_PROTECT_WRITEABLE;
                }

                NV_ASSERT_OR_RETURN(memdescMap(pMemDesc, offset, memSz, NV_TRUE, protect,
                    (NvP64*) &pPtr, &pPriv) == NV_OK, NULL);
                memdescSetKernelMappingPriv(pMemDesc, pPriv);
                break;
            }
            NV_ASSERT_OR_RETURN((pPtr = memdescMapInternal(pGpu, pMemDesc, flags)) != NULL, NULL);
            pPtr = &pPtr[offset];

            break;
        case TRANSFER_TYPE_GSP_DMA:
        case TRANSFER_TYPE_CE:
            if (flags & TRANSFER_FLAGS_SHADOW_ALLOC)
            {
                NV_ASSERT_OR_RETURN((pPtr = portMemAllocNonPaged(memSz)), NULL);
                if (flags & TRANSFER_FLAGS_SHADOW_INIT_MEM)
                {
                    NV_ASSERT_OK(memmgrMemRead(pMemoryManager, pTransferInfo, pPtr, memSz, flags));
                }
            }
            break;
        default:
            NV_ASSERT(0);
    }
    memdescSetKernelMapping(pMemDesc, pPtr);
    return pPtr;
}

/*!
 * @brief This helper function can be used to end transfers
 *
 * @param[in] pTransferInfo      Transfer information
 * @param[in] shadowBufSize      Size of allocated shadow buffer in case of shadow mapping
 * @param[in] flags              Flags
 */
void
memmgrMemEndTransfer_IMPL
(
    MemoryManager     *pMemoryManager,
    TRANSFER_SURFACE  *pTransferInfo,
    NvU64              shadowBufSize,
    NvU32              flags
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3077);
    TRANSFER_TYPE      transferType = memmgrGetMemTransferType(pMemoryManager,
                                                               pTransferInfo, NULL);
    MEMORY_DESCRIPTOR *pMemDesc     = pTransferInfo->pMemDesc;
    NvU64              offset       = pTransferInfo->offset;
    OBJGPU            *pGpu         = ENG_GET_GPU(pMemoryManager);
    NvU64              memSz        = 0;
    NvU8              *pMapping     = NULL;

    NV_ASSERT_OR_RETURN_VOID(pMemDesc != NULL);
    pMapping = memdescGetKernelMapping(pMemDesc);

    NV_ASSERT_OR_RETURN_VOID((memSz = memdescGetSize(pMemDesc)) >= (shadowBufSize + offset) );
    memSz = shadowBufSize == 0 ? memSz : shadowBufSize;

    memdescSetKernelMapping(pMemDesc, NULL);

    switch (transferType)
    {
        case TRANSFER_TYPE_PROCESSOR:
            if (flags & TRANSFER_FLAGS_USE_BAR1)
            {
                NvP64 pPriv = memdescGetKernelMappingPriv(pMemDesc);
                memdescSetKernelMappingPriv(pMemDesc, NULL);
                if (pMapping != NULL)
                {
                    memdescUnmap(pMemDesc, NV_TRUE, 0, pMapping, pPriv);
                }
                return;
            }
            memdescUnmapInternal(pGpu, pMemDesc, flags);
            return;
        case TRANSFER_TYPE_GSP_DMA:
        case TRANSFER_TYPE_CE:
            if (pMapping != NULL)
            {
                NV_ASSERT_OK(memmgrMemWrite(pMemoryManager, pTransferInfo, pMapping, memSz, flags));
                portMemFree(pMapping);
            }
            return;
        default:
            NV_ASSERT(0);
    }
    return;
}

/*!
 * @brief Helper function that ends transfers to a memdesc with default offset/size
 *
 * @param[in] pMemDesc           Memory descriptor to end transfer to
 * @param[in] flags              Flags
 */
void
memmgrMemDescEndTransfer_IMPL
(
    MemoryManager *pMemoryManager,
    MEMORY_DESCRIPTOR *pMemDesc,
    NvU32 flags
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3078);
    if (pMemDesc == NULL)
    {
        return;
    }

    TRANSFER_SURFACE transferSurface = {.offset = 0, .pMemDesc = pMemDesc};
    memmgrMemEndTransfer(pMemoryManager, &transferSurface, memdescGetSize(pMemDesc), flags);
}

/*!
 * @brief Helper function that begins transfers to a memdesc with default offset/size
 *
 * @param[in] pMemDesc           Memory descriptor to begin transfer to
 * @param[in] flags              Flags
 */
NvU8 *
memmgrMemDescBeginTransfer_IMPL
(
    MemoryManager *pMemoryManager,
    MEMORY_DESCRIPTOR *pMemDesc,
    NvU32 flags
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3079);
    NV_ASSERT_OR_RETURN(pMemDesc != NULL, NULL);
    TRANSFER_SURFACE transferSurface = {.offset = 0, .pMemDesc = pMemDesc};
    return memmgrMemBeginTransfer(pMemoryManager, &transferSurface, memdescGetSize(pMemDesc), flags);
}

/*!
 * @brief This function is used to allocate common resources across memory
 *        classes, and must be used before memory-specific resource alloc.
 *
 * @param[in/out] pAllocRequest     User-provided alloc request struct
 * @param[in/out] pFbAllocInfo      Initialized FB_ALLOC_INFO struct to alloc
 */
NV_STATUS
memmgrAllocResources_IMPL
(
    OBJGPU                      *pGpu,
    MemoryManager               *pMemoryManager,
    MEMORY_ALLOCATION_REQUEST   *pAllocRequest,
    FB_ALLOC_INFO               *pFbAllocInfo
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3080);
    NV_STATUS                    status        = NV_OK;
    NvU64                        alignment     = 0;
    NV_MEMORY_ALLOCATION_PARAMS *pVidHeapAlloc = pAllocRequest->pUserParams;
    NV_ADDRESS_SPACE             addrSpace     = memmgrAllocGetAddrSpace(pMemoryManager, pVidHeapAlloc->flags,
                                                                         pFbAllocInfo->retAttr);

    NvU64                        pageSize      = 0;
    NvBool                       bAllocedHwRes = NV_FALSE;

    // IRQL TEST:  must be running at equivalent of passive-level
    IRQL_ASSERT_AND_RETURN(!osIsRaisedIRQL());

    //
    // Check for valid size.
    //
    if (pVidHeapAlloc->size == 0)
        return NV_ERR_INVALID_ARGUMENT;

    //
    // Ensure a valid allocation pVidHeapAlloc->type was passed in
    //
    if (pVidHeapAlloc->type > NVOS32_NUM_MEM_TYPES - 1)
        return NV_ERR_INVALID_ARGUMENT;

    if (ADDR_VIRTUAL != addrSpace)
    {
        // If vidmem not requested explicitly, decide on the physical location.
        if (FLD_TEST_DRF(OS32, _ATTR, _LOCATION, _PCI, pFbAllocInfo->retAttr) ||
            FLD_TEST_DRF(OS32, _ATTR, _LOCATION, _ANY, pFbAllocInfo->retAttr))
        {
            if (ADDR_FBMEM == addrSpace)
            {
                pFbAllocInfo->retAttr = FLD_SET_DRF(OS32, _ATTR, _LOCATION, _VIDMEM, pFbAllocInfo->retAttr);
            }
            else
            {
                pFbAllocInfo->retAttr = FLD_SET_DRF(OS32, _ATTR, _LOCATION, _PCI, pFbAllocInfo->retAttr);
            }
        }
    }
    else // Virtual
    {
        // Clear location to ANY since virtual does not associate with location.
        pFbAllocInfo->retAttr = FLD_SET_DRF(OS32, _ATTR, _LOCATION, _ANY, pFbAllocInfo->retAttr);
    }

    // Fetch RM page size
    pageSize = memmgrDeterminePageSize(pMemoryManager, pFbAllocInfo->hClient, pFbAllocInfo->size,
                                       pFbAllocInfo->format, pFbAllocInfo->pageFormat->flags,
                                       &pFbAllocInfo->retAttr, &pFbAllocInfo->retAttr2);
    if (!IsAMODEL(pGpu) && pageSize == 0)
    {
        status = NV_ERR_INVALID_STATE;
        NV_PRINTF(LEVEL_ERROR, "memmgrDeterminePageSize failed, status: 0x%x\n", status);
        goto failed;
    }

    // Fetch memory alignment
    status = memmgrAllocDetermineAlignment_HAL(pGpu, pMemoryManager, &pFbAllocInfo->size, &pFbAllocInfo->align,
                                               pFbAllocInfo->alignPad, pFbAllocInfo->pageFormat->flags,
                                               pFbAllocInfo->retAttr, pFbAllocInfo->retAttr2, 0);
    if (status != NV_OK)
    {
        NV_PRINTF(LEVEL_ERROR, "memmgrAllocDetermineAlignment failed, status: 0x%x\n", status);
        goto failed;
    }

    //
    // Call into HAL to reserve any hardware resources for
    // the specified memory pVidHeapAlloc->type.
    // If the alignment was changed due to a HW limitation, and the
    // flag NVOS32_ALLOC_FLAGS_ALIGNMENT_FORCE is set, bad_argument
    // will be passed back from the HAL
    //
    status = memmgrAllocHwResources(pGpu, pMemoryManager, pFbAllocInfo);
    bAllocedHwRes = NV_TRUE;

    pVidHeapAlloc->attr  = pFbAllocInfo->retAttr;
    pVidHeapAlloc->attr2 = pFbAllocInfo->retAttr2;
    pVidHeapAlloc->format = pFbAllocInfo->format;
    pVidHeapAlloc->comprCovg = pFbAllocInfo->comprCovg;
    pVidHeapAlloc->zcullCovg = pFbAllocInfo->zcullCovg;

    if (status != NV_OK)
    {
        //
        // probably means we passed in a bogus pVidHeapAlloc->type or no tiling resources available
        // when tiled memory attribute was set to REQUIRED
        //
        NV_PRINTF(LEVEL_ERROR, "fbAlloc failure!\n");
        goto failed;
    }

    // call HAL to set resources
    status = memmgrSetAllocParameters_HAL(pGpu, pMemoryManager, pFbAllocInfo);

    if (status != NV_OK)
    {
        //
        // Two possibilties: either some attribute was set to REQUIRED, ran out of resources,
        // or unaligned address / size was passed down. Free up memory and fail this call.
        // heapFree will fix up heap pointers.
        //
        goto failed;
    }

    //
    // for fixed allocation check if the alignment needs to adjusted.
    // some hardware units request allocation aligned to smaller than
    // page sizes which can be handled through alignPad
    //
    if (pVidHeapAlloc->flags & NVOS32_ALLOC_FLAGS_FIXED_ADDRESS_ALLOCATE)
    {
        //
        // is our desired offset suitably aligned?
        // if not adjust alignment using alignPad(offset into a page), the
        // allocation is page size aligned as required for swizzling.
        //
        if (pFbAllocInfo->desiredOffset % (pFbAllocInfo->align + 1))
        {
           pFbAllocInfo->alignPad = pFbAllocInfo->desiredOffset % (pFbAllocInfo->align + 1);
           pFbAllocInfo->desiredOffset -= pFbAllocInfo->alignPad;
        }
    }

    //
    // Refresh search parameters.
    //
    pFbAllocInfo->adjustedSize = pFbAllocInfo->size - pFbAllocInfo->alignPad;
    pVidHeapAlloc->height = pFbAllocInfo->height;
    pVidHeapAlloc->pitch  = pFbAllocInfo->pitch;

    //
    // The api takes alignment-1 (used to be a mask).
    //
    alignment = pFbAllocInfo->align + 1;
    pVidHeapAlloc->alignment = pFbAllocInfo->align + 1;      // convert mask to size

    //
    // Allow caller to request host page alignment to make it easier
    // to move things around with host os VM subsystem
    //
    if ((pVidHeapAlloc->flags & NVOS32_ALLOC_FLAGS_FORCE_ALIGN_HOST_PAGE) &&
        (addrSpace == ADDR_FBMEM))
    {
        OBJSYS *pSys         = SYS_GET_INSTANCE();
        NvU64   hostPageSize = pSys->cpuInfo.hostPageSize;

        // hostPageSize *should* always be set, but....
        if (hostPageSize == 0)
            hostPageSize = RM_PAGE_SIZE;

        alignment = memUtilsLeastCommonAlignment(alignment, hostPageSize);
    }

    pVidHeapAlloc->alignment = alignment;
    pFbAllocInfo->align = alignment - 1;

    return status;

failed:
    if (bAllocedHwRes)
    {
        memmgrFreeHwResources(pGpu, pMemoryManager, pFbAllocInfo);
    }

    return status;
}

/*!
 * @brief This function is used to create a memory descriptor if needed.
 *
 * @param[in/out] pAllocRequest     User-provided alloc request struct
 * @param[in/out] pFbAllocInfo      Initialized FB_ALLOC_INFO struct to alloc
 * @param[out]    ppMemDesc         Double pointer to created descriptor
 * @param[in]     pHeap             Heap pointer to store in descriptor
 * @param[in]     addrSpace         Address space identifier
 * @param[in]     memDescFlags      Memory descriptor alloc flags
 * @param[out]    bAllocedMemDesc   NV_TRUE if a descriptor was created
 */
NV_STATUS
memUtilsAllocMemDesc
(
    OBJGPU                     *pGpu,
    MEMORY_ALLOCATION_REQUEST  *pAllocRequest,
    FB_ALLOC_INFO              *pFbAllocInfo,
    MEMORY_DESCRIPTOR         **ppMemDesc,
    Heap                       *pHeap,
    NV_ADDRESS_SPACE            addrSpace,
    NvBool                      bContig,
    NvBool                     *bAllocedMemDesc
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3081);
    NV_STATUS status = NV_OK;

    //
    // Allocate a memory descriptor if needed. We do this after the fbHwAllocResources() call
    // so we have the updated size information.  Linear callers like memdescAlloc() can live with
    // only having access to the requested size in bytes, but block linear callers really do
    // need to allocate after fbAlloc() rounding takes place.
    //
    if (pAllocRequest->pMemDesc == NULL)
    {
        NvU64 memDescFlags = MEMDESC_FLAGS_SKIP_RESOURCE_COMPUTE;

        //
        // Allocate a contig vidmem descriptor now; if needed we'll
        // allocate a new noncontig memdesc later
        //
        status = memdescCreate(&pAllocRequest->pMemDesc, pGpu, pFbAllocInfo->adjustedSize, 0, bContig,
                               addrSpace, NV_MEMORY_UNCACHED, memDescFlags);

        if (status != NV_OK)
        {
            NV_PRINTF(LEVEL_ERROR, "cannot alloc memDesc!\n");
            return status;
        }

        *bAllocedMemDesc = NV_TRUE;
    }

    *ppMemDesc = pAllocRequest->pMemDesc;
    (*ppMemDesc)->pHeap = pHeap;

    // Set attributes tracked by the memdesc
    memdescSetPteKind(*ppMemDesc, pFbAllocInfo->format);
    memdescSetHwResId(*ppMemDesc, pFbAllocInfo->hwResId);

    return status;
}

/*!
 * Memsets the memory for the given memory descriptor with the given value.
 * This function assumes that BAR2 is not yet available. Thus either the BAR0
 * window to FB or a memmap to SYSMEM will be used, depending on the memory
 * location.
 *
 * @param[in] pGpu      GPU object pointer
 * @param[in] pMemDesc  Memory descriptor for the memory to memset
 * @param[in] value     Value to memset to.
 */
NV_STATUS
memUtilsMemSetNoBAR2(OBJGPU *pGpu, PMEMORY_DESCRIPTOR pMemDesc, NvU8 value)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3082);
    KernelBus  *pKernelBus = GPU_GET_KERNEL_BUS(pGpu);
    NvU8       *pMap  = NULL;
    void       *pPriv = NULL;
    RmPhysAddr  physAddr;
    RmPhysAddr  physAddrOrig;
    NvU64       sizeInDWord;
    NvU32       sizeOfDWord = sizeof(NvU32);
    NvU32       bar0Addr;
    NvU32       i;

    NV_ASSERT((pMemDesc != NULL) &&
              (pMemDesc->Size & (sizeOfDWord-1)) == 0);
    sizeInDWord = pMemDesc->Size / sizeOfDWord;

    //
    // BAR2 is not yet initialized. Thus use either the BAR0 window or
    // memmap to initialize the given surface.
    //
    NV_ASSERT(pKernelBus->virtualBar2[GPU_GFID_PF].pCpuMapping == NULL);
    switch (memdescGetAddressSpace(pMemDesc))
    {
        case ADDR_FBMEM:
            if (KBUS_BAR0_PRAMIN_DISABLED(pGpu))
            {
                NvU8 *pMap = kbusMapRmAperture_HAL(pGpu, pMemDesc);
                NV_ASSERT_OR_RETURN(pMap != NULL, NV_ERR_INSUFFICIENT_RESOURCES);
                portMemSet(pMap, value, pMemDesc->Size);
                kbusUnmapRmAperture_HAL(pGpu, pMemDesc, &pMap, NV_TRUE);

                break;
            }
            //
            // Set the BAR0 window to encompass the given surface while
            // saving off the location to where the BAR0 window was
            // previously pointing.
            //
            physAddr = memdescGetPhysAddr(pMemDesc, AT_GPU, 0);
            NV_ASSERT((physAddr & (sizeOfDWord-1)) == 0);

            physAddrOrig = kbusGetBAR0WindowVidOffset_HAL(pGpu, pKernelBus);
            NV_ASSERT_OK_OR_RETURN(
                kbusSetBAR0WindowVidOffset_HAL(pGpu,
                                               pKernelBus,
                                               physAddr & ~0xffffULL));
            bar0Addr =
                NvU64_LO32(kbusGetBAR0WindowAddress_HAL(pKernelBus) +
                          (physAddr - kbusGetBAR0WindowVidOffset_HAL(pGpu, pKernelBus)));

            //
            // Iterate and initialize the given surface with BAR0
            // writes.
            //
            for (i = 0; i < sizeInDWord; i++)
            {
                GPU_REG_WR32(pGpu,
                             bar0Addr + (sizeOfDWord * i),
                             value);
            }

            //
            // Restore where the BAR0 window was previously pointing
            // to.
            //
            NV_ASSERT_OK_OR_RETURN(
                kbusSetBAR0WindowVidOffset_HAL(pGpu, pKernelBus, physAddrOrig));

            break;

        case ADDR_SYSMEM:
            // Plain old memmap.
            NV_ASSERT_OK_OR_RETURN(
                memdescMapOld(pMemDesc, 0,
                              pMemDesc->Size,
                              NV_TRUE, // kernel,
                              NV_PROTECT_READ_WRITE,
                              (void **)&pMap,
                              &pPriv));
            portMemSet(pMap, value, NvU64_LO32(pMemDesc->Size));
            memdescUnmapOld(pMemDesc, 1, 0, pMap, pPriv);
            break;

        default:
            // Should not happen.
            NV_ASSERT(0);
            break;
    }

    return NV_OK;
}

/*!
 * Registers the callback specified in clientHeap.callback for the channel
 * driven scrub.  The callback is triggered by NV906F_NON_STALL_INTERRUPT.
 */
static NV_STATUS
_memmgrMemUtilsScrubInitRegisterCallback
(
    OBJGPU       *pGpu,
    OBJCHANNEL   *pChannel
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3083);
    NV0005_ALLOC_PARAMETERS nv0005AllocParams;
    NV2080_CTRL_EVENT_SET_NOTIFICATION_PARAMS nv2080EventNotificationParams;
    NV_STATUS rmStatus;
    NvHandle subDeviceHandle = 0;
    Subdevice *pSubDevice;
    RM_API *pRmApi = rmapiGetInterface(RMAPI_GPU_LOCK_INTERNAL);
    NvU32 subdeviceInstance = gpumgrGetSubDeviceInstanceFromGpu(pGpu);

    rmStatus = subdeviceGetByInstance(pChannel->pRsClient, pChannel->deviceId,
                                      subdeviceInstance, &pSubDevice);
    if (rmStatus != NV_OK)
    {
        NV2080_ALLOC_PARAMETERS nv2080AllocParams;

        NV_PRINTF(LEVEL_WARNING, "Unable to get subdevice handle. Allocating subdevice\n");

        // Allocate a sub device if we dont have it created before hand
        portMemSet(&nv2080AllocParams, 0, sizeof(NV2080_ALLOC_PARAMETERS));
        nv2080AllocParams.subDeviceId = subdeviceInstance;

        rmStatus = pRmApi->AllocWithHandle(pRmApi,
                                           pChannel->hClient,
                                           pChannel->deviceId,
                                           pChannel->subdeviceId,
                                           NV20_SUBDEVICE_0,
                                           &nv2080AllocParams,
                                           sizeof(nv2080AllocParams));
        if (rmStatus != NV_OK)
        {
            NV_PRINTF(LEVEL_ERROR, "Unable to allocate a subdevice.\n");
            return NV_ERR_GENERIC;
        }

        // Set newly created subdevice's handle
        subDeviceHandle = pChannel->subdeviceId;
    }
    else
    {
        GPU_RES_SET_THREAD_BC_STATE(pSubDevice);

        subDeviceHandle = RES_GET_HANDLE(pSubDevice);
    }

    // Register callback
    portMemSet(&nv0005AllocParams, 0, sizeof(NV0005_ALLOC_PARAMETERS));
    nv0005AllocParams.hParentClient = pChannel->hClient;
    nv0005AllocParams.hClass        = NV01_EVENT_KERNEL_CALLBACK_EX;
    nv0005AllocParams.notifyIndex   = NV2080_NOTIFIERS_FIFO_EVENT_MTHD | NV01_EVENT_NONSTALL_INTR ;
    nv0005AllocParams.data          = NV_PTR_TO_NvP64(&pChannel->callback);

    rmStatus = pRmApi->AllocWithHandle(pRmApi,
                                       pChannel->hClient,
                                       subDeviceHandle,
                                       pChannel->eventId,
                                       NV01_EVENT_KERNEL_CALLBACK_EX,
                                       &nv0005AllocParams,
                                       sizeof(nv0005AllocParams));

    if (rmStatus != NV_OK)
    {
        NV_PRINTF(LEVEL_ERROR, "event allocation failed\n");
        return NV_ERR_GENERIC;
    }

    // Setup periodic event notification
    portMemSet(&nv2080EventNotificationParams, 0, sizeof(NV2080_CTRL_EVENT_SET_NOTIFICATION_PARAMS));
    nv2080EventNotificationParams.event = NV2080_NOTIFIERS_FIFO_EVENT_MTHD;
    nv2080EventNotificationParams.action = NV2080_CTRL_EVENT_SET_NOTIFICATION_ACTION_REPEAT;

    rmStatus = pRmApi->Control(pRmApi,
                               pChannel->hClient,
                               subDeviceHandle,
                               NV2080_CTRL_CMD_EVENT_SET_NOTIFICATION,
                               &nv2080EventNotificationParams,
                               sizeof(NV2080_CTRL_EVENT_SET_NOTIFICATION_PARAMS));

    if (rmStatus != NV_OK)
    {
        NV_PRINTF(LEVEL_ERROR, "event notification control failed\n");
        return NV_ERR_GENERIC;
    }

    return NV_OK;
}

/*!
 * Schedules the scrubber channel for execution.
 */
static NV_STATUS
_memmgrMemUtilsScrubInitScheduleChannel
(
    OBJGPU       *pGpu,
    OBJCHANNEL   *pChannel
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3084);
    NV_STATUS rmStatus;
    NVA06F_CTRL_GPFIFO_SCHEDULE_PARAMS nvA06fScheduleParams;
    RM_API *pRmApi = rmapiGetInterface(RMAPI_GPU_LOCK_INTERNAL);

    if (pChannel->bUseVasForCeCopy)
    {
        NVA06F_CTRL_BIND_PARAMS bindParams;
        portMemSet(&bindParams, 0, sizeof(bindParams));

        bindParams.engineType = gpuGetNv2080EngineType(pChannel->engineType);

        rmStatus = pRmApi->Control(pRmApi,
                                   pChannel->hClient,
                                   pChannel->channelId,
                                   NVA06F_CTRL_CMD_BIND,
                                   &bindParams,
                                   sizeof(bindParams));
        if (rmStatus != NV_OK)
        {
            NV_PRINTF(LEVEL_ERROR, "Unable to bind Channel, status: %x\n", rmStatus);
            return rmStatus;
        }
    }

    portMemSet(&nvA06fScheduleParams, 0, sizeof(NVA06F_CTRL_GPFIFO_SCHEDULE_PARAMS));
    nvA06fScheduleParams.bEnable = NV_TRUE;

    rmStatus = pRmApi->Control(pRmApi,
                               pChannel->hClient,
                               pChannel->channelId,
                               NVA06F_CTRL_CMD_GPFIFO_SCHEDULE,
                               &nvA06fScheduleParams,
                               sizeof(NVA06F_CTRL_GPFIFO_SCHEDULE_PARAMS));

    if (rmStatus != NV_OK)
    {
        NV_PRINTF(LEVEL_ERROR, "Unable to schedule channel, status: %x\n", rmStatus);
        return NV_ERR_GENERIC;
    }

    return NV_OK;
}

NV_STATUS
memmgrMemUtilsChannelSchedulingSetup_IMPL
(
    OBJGPU        *pGpu,
    MemoryManager *pMemoryManager,
    OBJCHANNEL    *pChannel
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3085);
    NvU32           classID;
    RM_ENGINE_TYPE  engineID;
    KernelFifo     *pKernelFifo = GPU_GET_KERNEL_FIFO(pGpu);
    KernelChannel  *pFifoKernelChannel = NULL;

    // schedulechannel
    NV_ASSERT_OK_OR_RETURN(_memmgrMemUtilsScrubInitScheduleChannel(pGpu, pChannel));

    // Determine classEngineID for SetObject usage
    NV_ASSERT_OK_OR_RETURN(CliGetKernelChannelWithDevice(pChannel->pRsClient,
                                                         pChannel->deviceId,
                                                         pChannel->channelId,
                                                        &pFifoKernelChannel));


    NV_ASSERT_OK_OR_RETURN(kchannelGetClassEngineID_HAL(pGpu,
                                                        pFifoKernelChannel,
                                                        pChannel->engineObjectId,
                                                       &pChannel->classEngineID,
                                                       &classID,
                                                       &engineID));

    NV_ASSERT_OK_OR_RETURN(_memmgrMemUtilsScrubInitRegisterCallback(pGpu, pChannel));

    NV_ASSERT_OK_OR_RETURN(kfifoRmctrlGetWorkSubmitToken_HAL(pKernelFifo,
                                                             pChannel->hClient,
                                                             pChannel->channelId,
                                                            &pChannel->workSubmitToken));

    // initialize the channel parameters (should be done by the parent object)
    pChannel->channelPutOffset = 0;

    if (pChannel->pbCpuVA != NULL)
    {
        MEM_WR32(pChannel->pbCpuVA + pChannel->semaOffset, 0);
    }
    return NV_OK;
}
