/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "core/core.h"
#include "core/locks.h"
#include "core/system.h"
#include "os/os.h"
#include "tls/tls.h"
#include "gpu_mgr/gpu_mgr.h"
#include "gpu/gpu.h"

static NvBool s_bRmLocksAllocated = NV_FALSE;

NV_STATUS
rmLocksAlloc(OBJSYS *pSys)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 228);
    NV_STATUS status;

    s_bRmLocksAllocated = NV_FALSE;

    // legacy lock model : RM system semaphore
    status = osAllocRmSema(&pSys->pSema);
    if (status != NV_OK)
        return status;

    // RM_BASIC_LOCK_MODEL : GPU lock info (ISR/DPC synchronization)
    status = rmGpuLockInfoInit();
    if (status != NV_OK)
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 229);
        osFreeRmSema(&pSys->pSema);
        return status;
    }
    rmInitLockMetering();

    s_bRmLocksAllocated = NV_TRUE;

    return status;
}

void
rmLocksFree(OBJSYS *pSys)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 230);
    if (s_bRmLocksAllocated)
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 231);
        rmDestroyLockMetering();
        rmGpuLockInfoDestroy();
        osFreeRmSema(pSys->pSema);

        s_bRmLocksAllocated = NV_FALSE;
    }
}

/*!
 * @brief Acquires all of the locks necessary to execute RM code safely
 *
 * Other threads and client APIs will be blocked from executing while the locks
 * are held, so the locks should not be held longer than necessary.  The locks
 * should not be held across long HW delays.
 *
 * @returns NV_OK if locks are acquired successfully
 *          NV_ERR_INVALID_LOCK_STATE if locks cannot be acquired
 */
NV_STATUS
rmLocksAcquireAll(NvU32 module)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 232);
    OBJSYS    *pSys = SYS_GET_INSTANCE();

    if (osAcquireRmSemaForced(pSys->pSema) != NV_OK)
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 233);
        NV_PRINTF(LEVEL_ERROR, "Failed to acquire the RM lock!\n");
        return NV_ERR_INVALID_LOCK_STATE;
    }

    if (rmapiLockAcquire(API_LOCK_FLAGS_NONE, module) != NV_OK)
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 234);
        NV_PRINTF(LEVEL_ERROR, "Failed to acquire the API lock!\n");
        osReleaseRmSema(pSys->pSema, NULL);
        return NV_ERR_INVALID_LOCK_STATE;
    }

    if (rmGpuLocksAcquire(GPUS_LOCK_FLAGS_NONE, module) != NV_OK)
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 235);
        NV_PRINTF(LEVEL_ERROR, "Failed to acquire the GPU lock!\n");
        rmapiLockRelease();
        osReleaseRmSema(pSys->pSema, NULL);
        return NV_ERR_INVALID_LOCK_STATE;
    }

    return NV_OK;
}

/*!
 * @brief Releases the locks acquired by rmLocksAcquireAll
 */
void
rmLocksReleaseAll(void)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 236);
    OBJSYS    *pSys = SYS_GET_INSTANCE();

    rmGpuLocksRelease(GPUS_LOCK_FLAGS_NONE, NULL);
    rmapiLockRelease();
    osReleaseRmSema(pSys->pSema, NULL);
}


NV_STATUS
workItemLocksAcquire(NvU32 gpuInstance, NvU32 flags, NvU32 *pReleaseLocks, NvU32 *pGpuMask)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 237);
    OBJSYS   *pSys = SYS_GET_INSTANCE();
    OBJGPU   *pGpu;
    NvU32     grp;
    NV_STATUS status = NV_OK;

    *pReleaseLocks = 0;
    *pGpuMask = 0;

    if (flags & OS_QUEUE_WORKITEM_FLAGS_LOCK_SEMA)
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 238);
        status = osAcquireRmSema(pSys->pSema);
        if (status != NV_OK)
            goto done;

        *pReleaseLocks |= OS_QUEUE_WORKITEM_FLAGS_LOCK_SEMA;
    }

    if ((flags & OS_QUEUE_WORKITEM_FLAGS_LOCK_API_RW) ||
        (flags & OS_QUEUE_WORKITEM_FLAGS_LOCK_API_RO))
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 239);
        NvU32 apiLockFlags = RMAPI_LOCK_FLAGS_NONE;
        NvU32 releaseFlags = OS_QUEUE_WORKITEM_FLAGS_LOCK_API_RW;

        if (flags & OS_QUEUE_WORKITEM_FLAGS_LOCK_API_RO)
        {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 240);
            apiLockFlags = RMAPI_LOCK_FLAGS_READ;
            releaseFlags = OS_QUEUE_WORKITEM_FLAGS_LOCK_API_RO;
        }

        status = rmapiLockAcquire(apiLockFlags, RM_LOCK_MODULES_WORKITEM);
        if (status != NV_OK)
            goto done;

        *pReleaseLocks |= releaseFlags;
    }

    if ((flags & OS_QUEUE_WORKITEM_FLAGS_LOCK_GPUS_RW) ||
        (flags & OS_QUEUE_WORKITEM_FLAGS_LOCK_GPUS_RO) ||
        (flags & OS_QUEUE_WORKITEM_FLAGS_LOCK_GPU_GROUP_DEVICE_RW) ||
        (flags & OS_QUEUE_WORKITEM_FLAGS_LOCK_GPU_GROUP_DEVICE_RO) ||
        (flags & OS_QUEUE_WORKITEM_FLAGS_LOCK_GPU_GROUP_SUBDEVICE_RW) ||
        (flags & OS_QUEUE_WORKITEM_FLAGS_LOCK_GPU_GROUP_SUBDEVICE_RO))
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 241);
        NvU32 gpuLockFlags = GPUS_LOCK_FLAGS_NONE;
        NvU32 releaseFlags = OS_QUEUE_WORKITEM_FLAGS_LOCK_GPUS_RW;

        if (((flags & OS_QUEUE_WORKITEM_FLAGS_LOCK_GPUS_RO) ||
             (flags & OS_QUEUE_WORKITEM_FLAGS_LOCK_GPU_GROUP_DEVICE_RO) ||
             (flags & OS_QUEUE_WORKITEM_FLAGS_LOCK_GPU_GROUP_SUBDEVICE_RO)) &&
            (pSys->gpuLockModuleMask & RM_LOCK_MODULE_GRP(RM_LOCK_MODULES_WORKITEM)))
        {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 242);
            gpuLockFlags = GPU_LOCK_FLAGS_READ;
            releaseFlags = OS_QUEUE_WORKITEM_FLAGS_LOCK_GPUS_RO;
        }

        if (flags & OS_QUEUE_WORKITEM_FLAGS_LOCK_GPUS_RW)
            grp = GPU_LOCK_GRP_ALL;
        else if (flags & OS_QUEUE_WORKITEM_FLAGS_LOCK_GPU_GROUP_DEVICE_RW)
            grp = GPU_LOCK_GRP_DEVICE;
        else // (flags & OS_QUEUE_WORKITEM_FLAGS_LOCK_GPU_GROUP_SUBDEVICE_RW)
            grp = GPU_LOCK_GRP_SUBDEVICE;

        status = rmGpuGroupLockAcquire(gpuInstance, grp, gpuLockFlags,
                                       RM_LOCK_MODULES_WORKITEM, pGpuMask);
        if (status != NV_OK)
            goto done;

        // All of these call into the same function, just share the flag
        *pReleaseLocks |= releaseFlags;

        pGpu = gpumgrGetGpu(gpuInstance);
        if (pGpu == NULL)
        {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 243);
            status = NV_ERR_INVALID_ARGUMENT;
            goto done;
        }

        if (flags & OS_QUEUE_WORKITEM_FLAGS_FULL_GPU_SANITY)
        {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 244);
            if (!FULL_GPU_SANITY_CHECK(pGpu) ||
                !pGpu->getProperty(pGpu, PDB_PROP_GPU_STATE_INITIALIZED))
            {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 245);
                status = NV_ERR_INVALID_STATE;
                NV_PRINTF(LEVEL_ERROR,
                          "GPU isn't full power! gpuInstance = 0x%x.\n",
                          gpuInstance);
                goto done;
            }
        }

        if (flags & OS_QUEUE_WORKITEM_FLAGS_FOR_PM_RESUME)
        {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 246);
            if (!FULL_GPU_SANITY_FOR_PM_RESUME(pGpu))
            {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 247);
                status = NV_ERR_INVALID_STATE;
                NV_PRINTF(LEVEL_ERROR,
                          "GPU isn't full power and isn't in resume codepath! gpuInstance = 0x%x.\n",
                          gpuInstance);
                goto done;
            }
        }
    }

done:
    if (status != NV_OK)
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 248);
        workItemLocksRelease(*pReleaseLocks, *pGpuMask);
        *pReleaseLocks = 0;
    }
    return status;
}

void
workItemLocksRelease(NvU32 releaseLocks, NvU32 gpuMask)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 249);
    OBJSYS *pSys = SYS_GET_INSTANCE();

    if (releaseLocks & OS_QUEUE_WORKITEM_FLAGS_LOCK_GPUS_RW)
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 250);
        rmGpuGroupLockRelease(gpuMask, GPUS_LOCK_FLAGS_NONE);
    }

    if (releaseLocks & OS_QUEUE_WORKITEM_FLAGS_LOCK_GPUS_RO)
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 251);
        rmGpuGroupLockRelease(gpuMask, GPU_LOCK_FLAGS_READ);
    }

    if ((releaseLocks & OS_QUEUE_WORKITEM_FLAGS_LOCK_API_RW) ||
        (releaseLocks & OS_QUEUE_WORKITEM_FLAGS_LOCK_API_RO))
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 252);
        rmapiLockRelease();
    }

    if (releaseLocks & OS_QUEUE_WORKITEM_FLAGS_LOCK_SEMA)
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 253);
        osReleaseRmSema(pSys->pSema, NULL);
    }
}

//
// rmGpuGroupLockGetMask
//
// Given a GPU group ID this function returns the MASK for all GPUS in that group
// We skip the lookup for GPU_LOCK_GRP_MASK as that implies that the caller is aware of the mask
//
NV_STATUS
rmGpuGroupLockGetMask(NvU32 gpuInst, GPU_LOCK_GRP_ID gpuGrpId, GPU_MASK *pGpuMask)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 254);
    switch (gpuGrpId)
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 255);
        case GPU_LOCK_GRP_SUBDEVICE:
            *pGpuMask = NVBIT(gpuInst);
            break;

        case GPU_LOCK_GRP_DEVICE:
            *pGpuMask = gpumgrGetGrpMaskFromGpuInst(gpuInst);
            break;

        case GPU_LOCK_GRP_MASK:
            break;

        case GPU_LOCK_GRP_ALL:
            *pGpuMask = GPUS_LOCK_ALL;
            break;

        default:
            NV_ASSERT_FAILED("Unexpected gpuGrpId in gpu lock get mask");
            return NV_ERR_INVALID_ARGUMENT;
    }
    return NV_OK;
}


void threadPriorityStateAlloc(void)          {}
void threadPriorityStateFree(void)           {}
void threadPriorityThrottle(void)            {}
void threadPriorityBoost(NvU64 *p, NvU64 *o) {}
void threadPriorityRestore(void)             {}

