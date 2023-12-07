/*
 * SPDX-FileCopyrightText: Copyright (c) 2009-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "nvlimits.h"
#include "nvlog/nvlog.h"
#include "nvrm_registry.h"
#include "os/os.h"
#include "diagnostics/tracer.h"
#include "tls/tls.h"
#include "core/locks.h"

//
// Buffer push method declarations
//
NvBool nvlogRingBufferPush  (NVLOG_BUFFER *pBuffer, NvU8 *pData, NvU32 dataSize);
NvBool nvlogNowrapBufferPush(NVLOG_BUFFER *pBuffer, NvU8 *pData, NvU32 dataSize);
NvBool nvlogStringBufferPush(NVLOG_BUFFER *unused,  NvU8 *pData, NvU32 dataSize);
NvBool nvlogKernelLogPush(NVLOG_BUFFER *unused, NvU8 *pData, NvU32 dataSize);

static void _printBase64(NvU8 *pData, NvU32 dataSize);
static NV_STATUS _allocateNvlogBuffer(NvU32 size, NvU32 flags, NvU32 tag,
                                      NVLOG_BUFFER **ppBuffer);
static void _deallocateNvlogBuffer(NVLOG_BUFFER *pBuffer);

volatile NvU32 nvlogInitCount;
static void *nvlogRegRoot;

// Zero (null) buffer definition.
static NVLOG_BUFFER _nvlogZeroBuffer =
{
    {nvlogStringBufferPush},
    0,
    NvU32_BUILD('l','l','u','n'),
    0,
    0,
    0
};

NVLOG_LOGGER NvLogLogger =
{
    NVLOG_LOGGER_VERSION,

    // Default buffers
    {
        // The 0th buffer just prints to the screen in debug builds.
        &_nvlogZeroBuffer
    },

    // Next available slot
    1,

    // Free slots
    NVLOG_MAX_BUFFERS-1,

    // Main lock, must be allocated at runtime.
    NULL
};

#define NVLOG_IS_VALID_BUFFER_HANDLE(hBuffer)                                  \
  ((hBuffer < NVLOG_MAX_BUFFERS) && (NvLogLogger.pBuffers[hBuffer] != NULL))

typedef struct
{
    void (*pCb)(void *);
    void *pData;
} NvlogFlushCb;

#define NVLOG_MAX_FLUSH_CBS 32

// At least one callback for each OBJGPU's KernelGsp
ct_assert(NVLOG_MAX_FLUSH_CBS >= NV_MAX_DEVICES);

static NvlogFlushCb nvlogFlushCbs[NVLOG_MAX_FLUSH_CBS];

NV_STATUS
nvlogInit(void *pData)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 888);
    NV_STATUS status = NV_OK;

    nvlogRegRoot = pData;
    portInitialize();
    NvLogLogger.mainLock = portSyncSpinlockCreate(portMemAllocatorGetGlobalNonPaged());
    if (NvLogLogger.mainLock == NULL)
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 889);
        return NV_ERR_INSUFFICIENT_RESOURCES;
    }
    NvLogLogger.buffersLock = portSyncMutexCreate(portMemAllocatorGetGlobalNonPaged());
    if (NvLogLogger.buffersLock == NULL)
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 890);
        return NV_ERR_INSUFFICIENT_RESOURCES;
    }
    NvLogLogger.flushCbsLock = portSyncRwLockCreate(portMemAllocatorGetGlobalNonPaged());
    if (NvLogLogger.flushCbsLock == NULL)
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 891);
        return NV_ERR_INSUFFICIENT_RESOURCES;
    }
    tlsInitialize();

    portMemSet(nvlogFlushCbs, '\0', sizeof(nvlogFlushCbs));
    return status;
}

void nvlogUpdate(void) {
}

NV_STATUS
nvlogDestroy(void)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 892);
    NV_STATUS status = NV_OK;
    NvU32 i;

    for (i = 0; i < NVLOG_MAX_BUFFERS; i++)
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 893);
        nvlogDeallocBuffer(i, NV_TRUE);
    }

    if (NvLogLogger.mainLock != NULL)
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 894);
        portSyncSpinlockDestroy(NvLogLogger.mainLock);
        NvLogLogger.mainLock = NULL;
    }
    if (NvLogLogger.buffersLock != NULL)
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 895);
        portSyncMutexDestroy(NvLogLogger.buffersLock);
        NvLogLogger.buffersLock = NULL;
    }
    if (NvLogLogger.flushCbsLock != NULL)
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 896);
        portSyncRwLockDestroy(NvLogLogger.flushCbsLock);
        NvLogLogger.flushCbsLock = NULL;
    }

    tlsShutdown();
    /// @todo Destructor should return void.
    portShutdown();

    return status;
}

static NV_STATUS
_allocateNvlogBuffer
(
    NvU32          size,
    NvU32          flags,
    NvU32          tag,
    NVLOG_BUFFER **ppBuffer
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 897);
    NVLOG_BUFFER          *pBuffer;
    NVLOG_BUFFER_PUSHFUNC  pushfunc;

    // Sanity check on some invalid combos:
    if (FLD_TEST_DRF(LOG_BUFFER, _FLAGS, _EXPANDABLE, _YES, flags))
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 898);
        // Only nonwrapping buffers can be expanded
        if (!FLD_TEST_DRF(LOG_BUFFER, _FLAGS, _TYPE, _NOWRAP, flags))
            return NV_ERR_INVALID_ARGUMENT;
        // Full locking required to expand the buffer.
        if (!FLD_TEST_DRF(LOG_BUFFER, _FLAGS, _LOCKING, _FULL, flags))
            return NV_ERR_INVALID_ARGUMENT;
    }

    if (FLD_TEST_DRF(LOG_BUFFER, _FLAGS, _TYPE, _SYSTEMLOG, flags))
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 899);
        // System log does not need to allocate memory for buffer.
        pushfunc = (NVLOG_BUFFER_PUSHFUNC) nvlogKernelLogPush;
        size = 0;
    }
    else
    {
        NV_ASSERT_OR_RETURN(size > 0, NV_ERR_INVALID_ARGUMENT);

        if (FLD_TEST_DRF(LOG_BUFFER, _FLAGS, _TYPE, _RING, flags))
        {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 900);
            pushfunc = (NVLOG_BUFFER_PUSHFUNC) nvlogRingBufferPush;
        }
        else if (FLD_TEST_DRF(LOG_BUFFER, _FLAGS, _TYPE, _NOWRAP, flags))
        {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 901);
            pushfunc = (NVLOG_BUFFER_PUSHFUNC) nvlogNowrapBufferPush;
        }
        else
        {
            return NV_ERR_INVALID_ARGUMENT;
        }
    }

    if (FLD_TEST_DRF(LOG_BUFFER, _FLAGS, _NONPAGED, _YES, flags))
        pBuffer = portMemAllocNonPaged(sizeof(*pBuffer) + size);
    else
        pBuffer = portMemAllocPaged(sizeof(*pBuffer) + size);

    if (!pBuffer)
        return NV_ERR_NO_MEMORY;

    portMemSet(pBuffer, 0, sizeof(*pBuffer) + size);
    if (FLD_TEST_DRF(LOG_BUFFER, _FLAGS, _OCA, _YES, flags))
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 902);
        osAddRecordForCrashLog(pBuffer, NV_OFFSETOF(NVLOG_BUFFER, data) + size);
    }

    pBuffer->push.fn  = pushfunc;
    pBuffer->size     = size;
    pBuffer->flags    = flags;
    pBuffer->tag      = tag;

    *ppBuffer = pBuffer;

    return NV_OK;
}

static void
_deallocateNvlogBuffer
(
    NVLOG_BUFFER *pBuffer
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 903);
    if (FLD_TEST_DRF(LOG_BUFFER, _FLAGS, _OCA, _YES, pBuffer->flags))
        osDeleteRecordForCrashLog(pBuffer);

    portMemFree(pBuffer);
}

NV_STATUS
nvlogAllocBuffer
(
    NvU32                size,
    NvU32                flags,
    NvU32                tag,
    NVLOG_BUFFER_HANDLE *pBufferHandle,
    ...
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 904);
    NVLOG_BUFFER *pBuffer;
    NV_STATUS     status;

    if (FLD_TEST_DRF(LOG_BUFFER, _FLAGS, _TYPE, _SYSTEMLOG, flags))
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 905);
    }
    else
    {
        NV_ASSERT_OR_RETURN(NvLogLogger.totalFree > 0,
                          NV_ERR_INSUFFICIENT_RESOURCES);
    }

    status = _allocateNvlogBuffer(size, flags, tag, &pBuffer);

    if (status != NV_OK)
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 906);
        return status;
    }

    portSyncMutexAcquire(NvLogLogger.buffersLock);
    portSyncSpinlockAcquire(NvLogLogger.mainLock);

    if (NvLogLogger.nextFree < NVLOG_MAX_BUFFERS)
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 907);
        NvLogLogger.pBuffers[NvLogLogger.nextFree] = pBuffer;
        *pBufferHandle = NvLogLogger.nextFree++;
        NvLogLogger.totalFree--;
    }
    else
    {
        status = NV_ERR_INSUFFICIENT_RESOURCES;
    }

    // Find the next slot in the buffers array
    while (NvLogLogger.nextFree < NVLOG_MAX_BUFFERS)
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 908);
        if (NvLogLogger.pBuffers[NvLogLogger.nextFree] != NULL)
            NvLogLogger.nextFree++;
        else break;
    }
    portSyncSpinlockRelease(NvLogLogger.mainLock);
    portSyncMutexRelease(NvLogLogger.buffersLock);

    if (status != NV_OK)
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 909);
        portMemFree(pBuffer);
    }

    return status;
}

void
nvlogDeallocBuffer
(
    NVLOG_BUFFER_HANDLE hBuffer,
    NvBool bDeallocPreserved
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 910);
    NVLOG_BUFFER *pBuffer;

    if ((hBuffer == 0) || !NVLOG_IS_VALID_BUFFER_HANDLE(hBuffer))
        return;

    pBuffer = NvLogLogger.pBuffers[hBuffer];

    if (FLD_TEST_DRF(LOG_BUFFER, _FLAGS, _PRESERVE, _YES, pBuffer->flags) &&
        !bDeallocPreserved)
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 911);
        return;
    }

    pBuffer->flags = FLD_SET_DRF(LOG_BUFFER, _FLAGS, _DISABLED,
                                 _YES, pBuffer->flags);

    while (pBuffer->threadCount > 0) { /*spin*/ }
    portSyncMutexAcquire(NvLogLogger.buffersLock);
    portSyncSpinlockAcquire(NvLogLogger.mainLock);
      NvLogLogger.pBuffers[hBuffer] = NULL;
      NvLogLogger.nextFree = NV_MIN(hBuffer, NvLogLogger.nextFree);
      NvLogLogger.totalFree++;
    portSyncSpinlockRelease(NvLogLogger.mainLock);
    portSyncMutexRelease(NvLogLogger.buffersLock);

    _deallocateNvlogBuffer(pBuffer);
}

NV_STATUS
nvlogWriteToBuffer
(
    NVLOG_BUFFER_HANDLE hBuffer,
    NvU8 *pData,
    NvU32 size
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 912);
    NvBool status;
    NVLOG_BUFFER *pBuffer;

    NV_ASSERT_OR_RETURN(size > 0,                    NV_ERR_INVALID_ARGUMENT);
    NV_ASSERT_OR_RETURN(pData != NULL,               NV_ERR_INVALID_POINTER);

    NV_ASSERT_OR_RETURN(NVLOG_IS_VALID_BUFFER_HANDLE(hBuffer),
                      NV_ERR_INVALID_OBJECT_HANDLE);

    pBuffer = NvLogLogger.pBuffers[hBuffer];

    // Normal condition when fetching nvLog from NV0000_CTRL_CMD_NVD_GET_NVLOG.
    if (FLD_TEST_DRF(LOG_BUFFER, _FLAGS, _DISABLED, _YES, pBuffer->flags))
        return NV_ERR_NOT_READY;

    portAtomicIncrementS32(&pBuffer->threadCount);
    status = pBuffer->push.fn(pBuffer, pData, size);
    // Get pBuffer from the handle again, as it might have realloc'd
    portAtomicDecrementS32(&NvLogLogger.pBuffers[hBuffer]->threadCount);

    return (status == NV_TRUE) ? NV_OK : NV_ERR_BUFFER_TOO_SMALL;
}



NV_STATUS
nvlogExtractBufferChunk
(
    NVLOG_BUFFER_HANDLE hBuffer,
    NvU32               chunkNum,
    NvU32              *pChunkSize,
    NvU8               *pDest
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 913);
    NVLOG_BUFFER *pBuffer;
    NvU32 index;

    NV_ASSERT_OR_RETURN(*pChunkSize > 0, NV_ERR_INVALID_ARGUMENT);
    NV_ASSERT_OR_RETURN(pDest != NULL,   NV_ERR_INVALID_POINTER);

    NV_ASSERT_OR_RETURN(NVLOG_IS_VALID_BUFFER_HANDLE(hBuffer),
                      NV_ERR_INVALID_OBJECT_HANDLE);

    pBuffer = NvLogLogger.pBuffers[hBuffer];

    index = chunkNum * (*pChunkSize);
    NV_ASSERT_OR_RETURN(index <= pBuffer->size,   NV_ERR_OUT_OF_RANGE);
    *pChunkSize = NV_MIN(*pChunkSize, (pBuffer->size - index));

    portSyncSpinlockAcquire(NvLogLogger.mainLock);
    portMemCopy(pDest, *pChunkSize, &pBuffer->data[index], *pChunkSize);
    portSyncSpinlockRelease(NvLogLogger.mainLock);

    return NV_OK;
}


NV_STATUS
nvlogGetBufferSize
(
    NVLOG_BUFFER_HANDLE hBuffer,
    NvU32 *pSize
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 914);
    NV_ASSERT_OR_RETURN(pSize != NULL, NV_ERR_INVALID_POINTER);

    NV_ASSERT_OR_RETURN(NVLOG_IS_VALID_BUFFER_HANDLE(hBuffer),
                      NV_ERR_INVALID_OBJECT_HANDLE);

    *pSize = NvLogLogger.pBuffers[hBuffer]->size;
    return NV_OK;
}

NV_STATUS
nvlogGetBufferTag
(
    NVLOG_BUFFER_HANDLE hBuffer,
    NvU32 *pTag
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 915);
    NV_ASSERT_OR_RETURN(pTag != NULL, NV_ERR_INVALID_POINTER);

    NV_ASSERT_OR_RETURN(NVLOG_IS_VALID_BUFFER_HANDLE(hBuffer),
                      NV_ERR_INVALID_OBJECT_HANDLE);

    *pTag = NvLogLogger.pBuffers[hBuffer]->tag;
    return NV_OK;
}

NV_STATUS
nvlogGetBufferFlags
(
    NVLOG_BUFFER_HANDLE hBuffer,
    NvU32 *pFlags
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 916);
    NV_ASSERT_OR_RETURN(pFlags != NULL, NV_ERR_INVALID_POINTER);

    NV_ASSERT_OR_RETURN(NVLOG_IS_VALID_BUFFER_HANDLE(hBuffer),
                      NV_ERR_INVALID_OBJECT_HANDLE);

    *pFlags = NvLogLogger.pBuffers[hBuffer]->flags;
    return NV_OK;
}


NV_STATUS
nvlogPauseLoggingToBuffer
(
    NVLOG_BUFFER_HANDLE hBuffer,
    NvBool bPause
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 917);
    NVLOG_BUFFER *pBuffer;

    NV_ASSERT_OR_RETURN(NVLOG_IS_VALID_BUFFER_HANDLE(hBuffer),
                      NV_ERR_INVALID_OBJECT_HANDLE);

    pBuffer = NvLogLogger.pBuffers[hBuffer];

    pBuffer->flags = (bPause)
        ? FLD_SET_DRF(LOG, _BUFFER_FLAGS, _DISABLED, _YES, pBuffer->flags)
        : FLD_SET_DRF(LOG, _BUFFER_FLAGS, _DISABLED, _NO,  pBuffer->flags);

    return NV_OK;
}


NV_STATUS
nvlogPauseAllLogging
(
    NvBool bPause
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 918);
    return NV_OK;
}

NV_STATUS
nvlogGetBufferHandleFromTag
(
    NvU32 tag,
    NVLOG_BUFFER_HANDLE *pBufferHandle
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 919);
    NvU32 i;

    NV_ASSERT_OR_RETURN(pBufferHandle != NULL, NV_ERR_INVALID_POINTER);

    for (i = 0; i < NVLOG_MAX_BUFFERS; i++)
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 920);
        if (NvLogLogger.pBuffers[i] != NULL)
        {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 921);
            if (NvLogLogger.pBuffers[i]->tag == tag)
            {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 922);
                *pBufferHandle = i;
                return NV_OK;
            }
        }
    }
    return NV_ERR_OBJECT_NOT_FOUND;
}

NV_STATUS
nvlogGetBufferSnapshot
(
    NVLOG_BUFFER_HANDLE hBuffer,
    NvU8               *pDest,
    NvU32               destSize
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 923);
    NVLOG_BUFFER *pBuffer;

    NV_ASSERT_OR_RETURN(NVLOG_IS_VALID_BUFFER_HANDLE(hBuffer),
                      NV_ERR_INVALID_OBJECT_HANDLE);

    NV_ASSERT_OR_RETURN(pDest != NULL, NV_ERR_INVALID_POINTER);

    pBuffer = NvLogLogger.pBuffers[hBuffer];

    NV_ASSERT_OR_RETURN(destSize >= NVLOG_BUFFER_SIZE(pBuffer),
                        NV_ERR_BUFFER_TOO_SMALL);

    portSyncSpinlockAcquire(NvLogLogger.mainLock);
    portMemCopy(pDest, NVLOG_BUFFER_SIZE(pBuffer), pBuffer, NVLOG_BUFFER_SIZE(pBuffer));
    portSyncSpinlockRelease(NvLogLogger.mainLock);

    return NV_OK;
}



NvBool
nvlogRingBufferPush
(
    NVLOG_BUFFER *pBuffer,
    NvU8         *pData,
    NvU32        dataSize
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 924);
    NvU32 writeSize;
    NvU32 oldPos;
    NvU32 lock = DRF_VAL(LOG, _BUFFER_FLAGS, _LOCKING, pBuffer->flags);

    if (lock != NVLOG_BUFFER_FLAGS_LOCKING_NONE)
        portSyncSpinlockAcquire(NvLogLogger.mainLock);

    oldPos = pBuffer->pos;
    pBuffer->extra.ring.overflow += (pBuffer->pos + dataSize) / pBuffer->size;
    pBuffer->pos                  = (pBuffer->pos + dataSize) % pBuffer->size;

    // State locking does portMemCopy unlocked.
    if (lock == NVLOG_BUFFER_FLAGS_LOCKING_STATE)
        portSyncSpinlockRelease(NvLogLogger.mainLock);

    while (dataSize > 0)
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 925);
        writeSize = NV_MIN(pBuffer->size - oldPos, dataSize);
        portMemCopy(&pBuffer->data[oldPos], writeSize, pData, writeSize);
        oldPos = 0;
        dataSize -= writeSize;
        pData    += writeSize;
    }

    if (lock == NVLOG_BUFFER_FLAGS_LOCKING_FULL)
        portSyncSpinlockRelease(NvLogLogger.mainLock);

    return NV_TRUE;
}

NvBool
nvlogNowrapBufferPush
(
    NVLOG_BUFFER *pBuffer,
    NvU8 *pData,
    NvU32 dataSize
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 926);
    NvU32 oldPos;
    NvU32 lock = DRF_VAL(LOG, _BUFFER_FLAGS, _LOCKING, pBuffer->flags);

    if (pBuffer->pos + dataSize >= pBuffer->size)
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 927);
        NvBool bExpandable = FLD_TEST_DRF(LOG_BUFFER, _FLAGS, _EXPANDABLE, _YES, pBuffer->flags);
        NvBool bNonPaged   = FLD_TEST_DRF(LOG_BUFFER, _FLAGS, _NONPAGED,   _YES, pBuffer->flags);

        // Expandable buffer, and we are at IRQL where we can do realloc
        if (bExpandable &&
            ((bNonPaged && portMemExSafeForNonPagedAlloc()) || (!bNonPaged && portMemExSafeForPagedAlloc())))
        {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 928);
            NVLOG_BUFFER *pNewBuffer;
            NvU32 i;
            NvU32 newSize = pBuffer->size * 2;
            NvU32 allocSize = sizeof(*pBuffer) + newSize;

            pNewBuffer = bNonPaged ? portMemAllocNonPaged(allocSize) : portMemAllocPaged(allocSize);
            if (pNewBuffer == NULL)
                return NV_FALSE;

            //
            // Two threads couid have entered this block at the same time, and
            // both will have allocated their own bigger buffer. Only the one
            // that takes the spinlock first should do the copy and the swap.
            //
            portSyncSpinlockAcquire(NvLogLogger.mainLock);
              // Check if this buffer is still there and was not swapped for a bigger one
              for (i = 0; i < NVLOG_MAX_BUFFERS; i++)
              {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 929);
                  if (NvLogLogger.pBuffers[i] == pBuffer)
                    break;
              }
              if (i == NVLOG_MAX_BUFFERS)
              {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 930);
                  // Another thread has already expanded the buffer, bail out.
                  // TODO: Maybe we could store the handle and then try again?
                  portSyncSpinlockRelease(NvLogLogger.mainLock);
                  portMemFree(pNewBuffer);
                  return NV_FALSE;
              }

              portMemCopy(pNewBuffer, allocSize, pBuffer, sizeof(*pBuffer)+pBuffer->size);
              pNewBuffer->size = newSize;
              for (i = 0; i < NVLOG_MAX_BUFFERS; i++)
              {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 931);
                  if (NvLogLogger.pBuffers[i] == pBuffer)
                      NvLogLogger.pBuffers[i] = pNewBuffer;
              }
            portSyncSpinlockRelease(NvLogLogger.mainLock);

            //
            // Before we can free this buffer, we need to make sure any threads
            // that were still accessing it are done. Spin on volatile threadCount
            // NOTE: threadCount includes the current thread too.
            //
            while (pBuffer->threadCount > 1) { /*spin*/ }
            portMemFree(pBuffer);
            pBuffer = pNewBuffer;
        }
        else
        {
            return NV_FALSE;
        }
    }

    if (lock != NVLOG_BUFFER_FLAGS_LOCKING_NONE)
        portSyncSpinlockAcquire(NvLogLogger.mainLock);

      oldPos = pBuffer->pos;
      pBuffer->pos = oldPos + dataSize;

    // State locking does portMemCopy unlocked.
    if (lock == NVLOG_BUFFER_FLAGS_LOCKING_STATE)
        portSyncSpinlockRelease(NvLogLogger.mainLock);

    portMemCopy(&pBuffer->data[oldPos], dataSize, pData, dataSize);

    if (lock == NVLOG_BUFFER_FLAGS_LOCKING_FULL)
        portSyncSpinlockRelease(NvLogLogger.mainLock);

    return NV_TRUE;
}

NvBool
nvlogStringBufferPush
(
    NVLOG_BUFFER *unused,
    NvU8         *pData,
    NvU32         dataSize
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 932);
    return NV_TRUE;
}

//
// Prints the buffer encoded as base64, with a prefix for easy grepping.
// Base64 allows the padding characters ('=') to appear anywhere, not just at
// the end, so it is fine to print buffers one at a time without merging.
//
static void _printBase64(NvU8 *pData, NvU32 dataSize)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 933);
    const NvU8 base64_key[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    NvU8 output[64+1]; // 64 bas64 characters per line of output
    NvU32 i;

    do
    {
        i = 0;
        while (i < (sizeof(output)-1) && (dataSize > 0))
        {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 934);
            output[i++] = base64_key[pData[0] >> 2];
            if (dataSize == 1)
            {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 935);
                output[i++] = base64_key[(pData[0] << 4) & 0x3F];
                output[i++] = '=';
                output[i++] = '=';
                dataSize = 0;
                break;
            }

            output[i++] = base64_key[((pData[0] << 4) & 0x3F) | (pData[1] >> 4)];
            if (dataSize == 2)
            {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 936);
                output[i++] = base64_key[(pData[1] << 2) & 0x3F];
                output[i++] = '=';
                dataSize = 0;
                break;
            }

            output[i++] = base64_key[((pData[1] << 2) & 0x3F) | (pData[2] >> 6)];
            output[i++] = base64_key[pData[2] & 0x3F];

            pData += 3;
            dataSize -= 3;
        }
        output[i] = 0;
        portDbgPrintf("nvrm-nvlog: %s\n", output);
    } while (dataSize > 0);
}

NvBool nvlogKernelLogPush(NVLOG_BUFFER *unused, NvU8 *pData, NvU32 dataSize)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 937);
    PORT_UNREFERENCED_VARIABLE(unused);
    _printBase64(pData, dataSize);
    return NV_TRUE;
}

void nvlogDumpToKernelLog(NvBool bDumpUnchangedBuffersOnlyOnce)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 938);
    NvU32 i;
    static NvU32 lastDumpPos[NVLOG_MAX_BUFFERS];

    for (i = 0; i < NVLOG_MAX_BUFFERS; i++)
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 939);
        NVLOG_BUFFER *pBuf = NvLogLogger.pBuffers[i];

        if (pBuf && pBuf->size)
        {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 940);
            if (bDumpUnchangedBuffersOnlyOnce)
            {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 941);
                NvU32 pos = pBuf->pos + (pBuf->size * pBuf->extra.ring.overflow);

                //Dump the buffer only if it's contents have changed
                if (lastDumpPos[i] != pos)
                {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 942);
                    lastDumpPos[i] = pos;
                    _printBase64((NvU8*)pBuf, NVLOG_BUFFER_SIZE(pBuf));
                }
            }
            else
            {
                _printBase64((NvU8*)pBuf, NVLOG_BUFFER_SIZE(pBuf));
            }
        }
    }
}

void nvlogDumpToKernelLogIfEnabled(void)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 943);
    NvU32 dumpNvlogValue;

    // Debug and develop builds already dump everything as it happens.
#if defined(DEBUG) || defined(DEVELOP)
    return;
#endif

    // Enable only if the regkey has been set
    if (osReadRegistryDword(NULL, NV_REG_STR_RM_DUMP_NVLOG, &dumpNvlogValue) != NV_OK)
        return;

    if (dumpNvlogValue != NV_REG_STR_RM_DUMP_NVLOG_ENABLE)
        return;

    nvlogDumpToKernelLog(NV_FALSE);
}

NV_STATUS nvlogRegisterFlushCb(void (*pCb)(void*), void *pData)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 944);
    NV_STATUS status = NV_ERR_INSUFFICIENT_RESOURCES;
    portSyncRwLockAcquireWrite(NvLogLogger.flushCbsLock);

    for (NvU32 i = 0; i < NV_ARRAY_ELEMENTS(nvlogFlushCbs); i++)
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 945);
        // The same callback should not be registered twice
        NV_ASSERT(nvlogFlushCbs[i].pCb != pCb || nvlogFlushCbs[i].pData != pData);

        if (nvlogFlushCbs[i].pCb == NULL)
        {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 946);
            nvlogFlushCbs[i].pCb = pCb;
            nvlogFlushCbs[i].pData = pData;

            status = NV_OK;
            goto done;
        }
    }

done:
    portSyncRwLockReleaseWrite(NvLogLogger.flushCbsLock);
    return status;
}

void nvlogDeregisterFlushCb(void (*pCb)(void*), void *pData)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 947);
    portSyncRwLockAcquireWrite(NvLogLogger.flushCbsLock);

    for (NvU32 i = 0; i < NV_ARRAY_ELEMENTS(nvlogFlushCbs); i++)
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 948);
        if (nvlogFlushCbs[i].pCb == pCb && nvlogFlushCbs[i].pData == pData)
        {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 949);
            nvlogFlushCbs[i] = (NvlogFlushCb){0};
            goto done;
        }
    }

done:
    portSyncRwLockReleaseWrite(NvLogLogger.flushCbsLock);
}

void nvlogRunFlushCbs(void)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 950);
    portSyncRwLockAcquireRead(NvLogLogger.flushCbsLock);
    for (NvU32 i = 0; i < NV_ARRAY_ELEMENTS(nvlogFlushCbs); i++)
        if (nvlogFlushCbs[i].pCb != NULL)
            nvlogFlushCbs[i].pCb(nvlogFlushCbs[i].pData);
    portSyncRwLockReleaseRead(NvLogLogger.flushCbsLock);
}
