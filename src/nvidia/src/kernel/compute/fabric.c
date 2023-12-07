/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

/*!
 * @file
 * @brief This file contains the functions managing the NVLink fabric
 */

#include "os/os.h"
#include "compute/fabric.h"

static NV_STATUS
_fabricCacheInsert
(
    FabricCache  *pCache,
    NvU64         key1,
    NvU64         key2,
    NvU64         key3,
    void         *pData
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 1);
    FabricCacheSubmap *pInsertedSubmap = NULL;
    FabricCacheEntry *pInsertedEntry = NULL;
    FabricCacheEntry *pEntry;
    FabricCacheMapEntry *pMapEntry;

    pEntry = multimapFindItem(pCache, key1, key2);
    if (pEntry != NULL)
        goto insert;

    if (multimapFindSubmap(pCache, key1) == NULL)
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 2);
        pInsertedSubmap = multimapInsertSubmap(pCache, key1);
        if (pInsertedSubmap == NULL)
            goto fail;
    }

    pInsertedEntry = multimapInsertItemNew(pCache, key1, key2);
    if (pInsertedEntry == NULL)
        goto fail;

    mapInit(&pInsertedEntry->map, portMemAllocatorGetGlobalNonPaged());
    pEntry = pInsertedEntry;

insert:
    pMapEntry = mapInsertNew(&pEntry->map, key3);
    if (pMapEntry == NULL)
        goto fail;

    pMapEntry->pData = pData;

    return NV_OK;

fail:
    if (pInsertedEntry != NULL)
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3);
        mapDestroy(&pInsertedEntry->map);
        multimapRemoveItem(pCache, pInsertedEntry);
    }

    if (pInsertedSubmap != NULL)
        multimapRemoveSubmap(pCache, pInsertedSubmap);

    return NV_ERR_INVALID_STATE;
}

static void
_fabricCacheDelete
(
    FabricCache  *pCache,
    NvU64         key1,
    NvU64         key2,
    NvU64         key3
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 4);
    FabricCacheSubmap *pSubmap;
    FabricCacheEntry *pEntry;

    pEntry = multimapFindItem(pCache, key1, key2);
    if (pEntry == NULL)
        return;

    mapRemoveByKey(&pEntry->map, key3);
    if (mapCount(&pEntry->map) > 0)
        return;

    mapDestroy(&pEntry->map);
    multimapRemoveItem(pCache, pEntry);

    pSubmap = multimapFindSubmap(pCache, key1);
    NV_ASSERT_OR_RETURN_VOID(pSubmap != NULL);

    if (multimapCountSubmapItems(pCache, pSubmap) > 0)
        return;

    multimapRemoveSubmap(pCache, pSubmap);
}

static FabricCacheMapEntry*
_fabricCacheFind
(
    FabricCache  *pCache,
    NvU64         key1,
    NvU64         key2,
    NvU64         key3
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 5);
    FabricCacheEntry *pEntry;
    FabricCacheMapEntry *pMapEntry;

    pEntry = multimapFindItem(pCache, key1, key2);
    if (pEntry == NULL)
        return NULL;

    pMapEntry = mapFind(&pEntry->map, key3);
    if (pMapEntry == NULL)
        return NULL;

    return pMapEntry;
}

void
fabricMulticastFabricOpsMutexAcquire_IMPL
(
    Fabric *pFabric
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 6);
    portSyncMutexAcquire(pFabric->pMulticastFabricOpsMutex);
}

void
fabricMulticastFabricOpsMutexRelease_IMPL
(
    Fabric *pFabric
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 7);
    portSyncMutexRelease(pFabric->pMulticastFabricOpsMutex);
}

NV_STATUS
fabricMulticastSetupCacheInsertUnderLock_IMPL
(
    Fabric *pFabric,
    NvU64   requestId,
    void   *pData
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 8);
    return _fabricCacheInsert(&pFabric->fabricMulticastCache,
                              0, requestId, 0, pData);
}

void
fabricMulticastSetupCacheDeleteUnderLock_IMPL
(
    Fabric *pFabric,
    NvU64   requestId
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 9);
    _fabricCacheDelete(&pFabric->fabricMulticastCache,
                       0, requestId, 0);
}

void*
fabricMulticastSetupCacheGetUnderLock_IMPL
(
    Fabric *pFabric,
    NvU64   requestId
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 10);
    FabricCacheMapEntry *pMapEntry;

    pMapEntry = _fabricCacheFind(&pFabric->fabricMulticastCache,
                                 0, requestId, 0);

    return (pMapEntry == NULL ? NULL : pMapEntry->pData);
}

NV_STATUS
fabricMulticastCleanupCacheInsertUnderLock_IMPL
(
    Fabric *pFabric,
    NvU64   requestId,
    void   *pData
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 11);
    return _fabricCacheInsert(&pFabric->fabricMulticastCache,
                              1, requestId, 0, pData);
}

void
fabricMulticastCleanupCacheDeleteUnderLock_IMPL
(
    Fabric *pFabric,
    NvU64   requestId
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 12);
    _fabricCacheDelete(&pFabric->fabricMulticastCache,
                       1, requestId, 0);
}

void*
fabricMulticastCleanupCacheGetUnderLock_IMPL
(
    Fabric *pFabric,
    NvU64   requestId
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 13);
    FabricCacheMapEntry *pMapEntry;

    pMapEntry = _fabricCacheFind(&pFabric->fabricMulticastCache,
                                 1, requestId, 0);

    return (pMapEntry == NULL ? NULL : pMapEntry->pData);
}

void
fabricSetFmSessionFlags_IMPL
(
    Fabric  *pFabric,
    NvU32   flags
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 14);
    pFabric->flags = flags;
}

NvU32
fabricGetFmSessionFlags_IMPL
(
    Fabric  *pFabric
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 15);
    return pFabric->flags;
}

NV_STATUS
fabricConstruct_IMPL
(
    Fabric *pFabric
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 16);
    NV_STATUS status = NV_OK;

    pFabric->pMulticastFabricOpsMutex = portSyncMutexCreate(portMemAllocatorGetGlobalNonPaged());
    if (pFabric->pMulticastFabricOpsMutex == NULL)
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 17);
        status = NV_ERR_NO_MEMORY;
        goto fail;
    }

    multimapInit(&pFabric->fabricMulticastCache, portMemAllocatorGetGlobalNonPaged());

    return NV_OK;

//TODO: Remove the WAR to suppress unused label warning
goto fail;
fail:
    fabricDestruct_IMPL(pFabric);
    return status;
}

void
fabricDestruct_IMPL
(
    Fabric *pFabric
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 18);
    NV_ASSERT(multimapCountItems(&pFabric->fabricMulticastCache) == 0);

    multimapDestroy(&pFabric->fabricMulticastCache);

    if (pFabric->pMulticastFabricOpsMutex != NULL)
        portSyncMutexDestroy(pFabric->pMulticastFabricOpsMutex);

}

NV_STATUS
fabricInitInbandMsgHdr
(
    nvlink_inband_msg_header_t *pMsgHdr,
    NvU32 type,
    NvU32 len
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 19);
    NV_STATUS status;

    portMemSet(pMsgHdr, 0, sizeof(*pMsgHdr));

    status = osGetRandomBytes((NvU8 *)&pMsgHdr->requestId,
                              sizeof(pMsgHdr->requestId));
    if (status != NV_OK)
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 20);
        return status;
    }

    pMsgHdr->magicId = NVLINK_INBAND_MSG_MAGIC_ID_FM;
    pMsgHdr->type = type;
    pMsgHdr->length = len;

    return NV_OK;
}

void
fabricMulticastWaitOnTeamCleanupCallback
(
    void *pCbData
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 21);
    NvU64 inbandReqId = (NvU64)pCbData;
    Fabric *pFabric = SYS_GET_FABRIC(SYS_GET_INSTANCE());
    OS_WAIT_QUEUE *pWq;

    fabricMulticastFabricOpsMutexAcquire(pFabric);

    pWq = (OS_WAIT_QUEUE *)fabricMulticastCleanupCacheGetUnderLock_IMPL(pFabric,
                                                                        inbandReqId);
    fabricMulticastFabricOpsMutexRelease(pFabric);

    if (pWq == NULL)
        return;

    osWaitInterruptible(pWq);

    fabricMulticastFabricOpsMutexAcquire(pFabric);

    fabricMulticastCleanupCacheDeleteUnderLock_IMPL(pFabric,
                                                    inbandReqId);

    fabricMulticastFabricOpsMutexRelease(pFabric);

    osFreeWaitQueue(pWq);
}
