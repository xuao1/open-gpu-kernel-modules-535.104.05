/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "rmapi/rs_utils.h"
#include "rmapi/rmapi.h"
#include "core/locks.h"

NV_STATUS
serverutilGetResourceRef
(
    NvHandle                 hClient,
    NvHandle                 hObject,
    RsResourceRef          **ppResourceRef
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 5734);
    RsResourceRef    *pResourceRef;
    RsClient         *pRsClient;
    NV_STATUS         status;

    *ppResourceRef = NULL;

    status = serverGetClientUnderLock(&g_resServ, hClient, &pRsClient);
    if (status != NV_OK)
        return NV_ERR_INVALID_CLIENT;

    status = clientGetResourceRef(pRsClient, hObject, &pResourceRef);
    if (status != NV_OK)
        return status;

    *ppResourceRef = pResourceRef;

    return NV_OK;
}

NV_STATUS
serverutilGetResourceRefWithType
(
    NvHandle            hClient,
    NvHandle            hObject,
    NvU32               internalClassId,
    RsResourceRef     **ppResourceRef
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 5735);
    if (serverutilGetResourceRef(hClient, hObject, ppResourceRef) != NV_OK)
    {
        return NV_ERR_OBJECT_NOT_FOUND;
    }

    if (!objDynamicCastById((*ppResourceRef)->pResource, internalClassId))
    {
        return NV_ERR_INVALID_OBJECT_HANDLE;
    }

    return NV_OK;
}

NV_STATUS
serverutilGetResourceRefWithParent
(
    NvHandle            hClient,
    NvHandle            hParent,
    NvHandle            hObject,
    NvU32               internalClassId,
    RsResourceRef     **ppResourceRef
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 5736);
    NvHandle hFoundParent;

    if (serverutilGetResourceRef(hClient, hObject, ppResourceRef) != NV_OK)
    {
        return NV_ERR_OBJECT_NOT_FOUND;
    }

    hFoundParent = (*ppResourceRef)->pParentRef ? (*ppResourceRef)->pParentRef->hResource : 0;

    if (!objDynamicCastById((*ppResourceRef)->pResource, internalClassId) ||
        hFoundParent != hParent)
    {
        return NV_ERR_INVALID_OBJECT_HANDLE;
    }

    return NV_OK;
}

RmClient
*serverutilGetClientUnderLock
(
    NvHandle hClient
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 5737);
    NV_STATUS status;
    RsClient *pRsClient;

    status = serverGetClientUnderLock(&g_resServ, hClient, &pRsClient);
    if (status != NV_OK)
        return NULL;

    return dynamicCast(pRsClient, RmClient);
}

RmClient
**serverutilGetFirstClientUnderLock
(
    void
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 5738);
    RmClient **ppClient;

    //
    // Resource server's client list is not protected by any RM locks
    // so, as a WAR, we access a lock-protected shadow client list. This avoids
    // the race condition where a client is freed while a DPC is iterating
    // through the client list.
    //
    ppClient = listHead(&g_clientListBehindGpusLock);
    if (NULL == ppClient)
        return NULL;

    return ppClient;
}

RmClient
**serverutilGetNextClientUnderLock
(
    RmClient **ppClient
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 5739);
    //
    // Resource server's client list is not protected by any RM locks
    // so, as a WAR, we access a lock-protected shadow client list. This avoids
    // the race condition where a client is freed while a DPC is iterating
    // through the client list.
    //
    ppClient = listNext(&g_clientListBehindGpusLock, ppClient);
    if (NULL == ppClient)
        return NULL;

    return ppClient;
}

RsResourceRef *
serverutilFindChildRefByType
(
    NvHandle hClient,
    NvHandle hParent,
    NvU32 internalClassId,
    NvBool bExactMatch
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 5740);
    NV_STATUS status;
    RsClient *pRsClient;
    RsResourceRef *pResourceRef;
    RsResourceRef *pParentRef;

    status = serverGetClientUnderLock(&g_resServ, hClient, &pRsClient);
    if (status != NV_OK)
        return NULL;

    status = clientGetResourceRef(pRsClient, hParent, &pParentRef);
    if (status != NV_OK)
    {
        return NULL;
    }

    status = refFindChildOfType(pParentRef, internalClassId, bExactMatch, &pResourceRef);
    if (status != NV_OK)
    {
        return NULL;
    }

    return pResourceRef;
}

RS_ITERATOR
serverutilRefIter
(
    NvHandle hClient,
    NvHandle hScopedObject,
    NvU32 internalClassId,
    RS_ITER_TYPE iterType,
    NvBool bExactMatch
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 5741);
    NV_STATUS      status;
    RsClient      *pRsClient;
    RsResourceRef *pScopedRef = NULL;
    RS_ITERATOR    it;

    portMemSet(&it, 0, sizeof(it));

    status = serverGetClientUnderLock(&g_resServ, hClient, &pRsClient);
    if (status != NV_OK)
        return it;

    if (hScopedObject != NV01_NULL_OBJECT)
    {
        status = clientGetResourceRef(pRsClient, hScopedObject, &pScopedRef);
        if (status != NV_OK)
        {
            return it;
        }
    }

    return clientRefIter(pRsClient, pScopedRef, internalClassId, iterType, bExactMatch);
}

NvBool
serverutilValidateNewResourceHandle
(
    NvHandle hClient,
    NvHandle hObject
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 5742);
    RmClient *pClient = serverutilGetClientUnderLock(hClient);

    return ((pClient != NULL) &&
            (NV_OK == clientValidateNewResourceHandle(staticCast(pClient, RsClient), hObject, NV_TRUE)));
}

NV_STATUS
serverutilGenResourceHandle
(
    NvHandle    hClient,
    NvHandle   *returnHandle
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 5743);
    NV_STATUS status;
    RmClient *pClient;

    // LOCK TEST: we should have the API lock here
    LOCK_ASSERT_AND_RETURN(rmapiLockIsOwner());

    pClient = serverutilGetClientUnderLock(hClient);

    if (pClient == NULL)
        return NV_ERR_INVALID_CLIENT;

    status = clientGenResourceHandle(staticCast(pClient, RsClient), returnHandle);
    return status;
}

RS_SHARE_ITERATOR
serverutilShareIter
(
    NvU32 internalClassId
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 5744);
    return serverShareIter(&g_resServ, internalClassId);
}

NvBool
serverutilShareIterNext
(
    RS_SHARE_ITERATOR* pIt
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 5745);
    return serverShareIterNext(pIt);
}

NV_STATUS
serverutilGetClientHandlesFromPid
(
    NvU32               procID,
    NvU32               subProcessID,
    ClientHandlesList   *pClientList
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 5746);
    RmClient **ppClient;
    RmClient  *pClient;

    // If the list passed in has old elements, lets clear its elements.
    if (listCount(pClientList))
    {
        // Clear & free nodes in temp list
        listDestroy(pClientList);
    }

    for (ppClient = serverutilGetFirstClientUnderLock();
         ppClient;
         ppClient = serverutilGetNextClientUnderLock(ppClient))
    {
        RsClient *pRsClient;

        pClient = *ppClient;
        pRsClient = staticCast(pClient, RsClient);

        if ((pClient->ProcID == procID) &&
            (pClient->SubProcessID == subProcessID))
        {
             if (listAppendValue(pClientList,
                                 &pRsClient->hClient) == NULL)
             {
                listClear(pClientList);
                return NV_ERR_INSUFFICIENT_RESOURCES;
             }
        }
    }

    return NV_OK;
}

NvBool
serverutilMappingFilterCurrentUserProc
(
    RsCpuMapping *pMapping
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 5747);
    return (!pMapping->pPrivate->bKernel &&
            (pMapping->processId == osGetCurrentProcess()));
}

NvBool
serverutilMappingFilterKernel
(
    RsCpuMapping *pMapping
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 5748);
    return pMapping->pPrivate->bKernel;
}


NV_STATUS
serverutilAcquireClient
(
    NvHandle hClient,
    LOCK_ACCESS_TYPE access,
    RmClient **ppClient
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 5749);
    RsClient *pRsClient;
    RmClient *pClient;

    // LOCK TEST: we should have the API lock here
    LOCK_ASSERT_AND_RETURN(rmapiLockIsOwner());

    if (NV_OK != serverAcquireClient(&g_resServ, hClient, access, &pRsClient))
        return NV_ERR_INVALID_CLIENT;

    pClient = dynamicCast(pRsClient, RmClient);
    if (pClient == NULL)
    {
        serverReleaseClient(&g_resServ, access, pRsClient);
        return NV_ERR_INVALID_CLIENT;
    }

    *ppClient = pClient;
    return NV_OK;
}

void
serverutilReleaseClient
(
    LOCK_ACCESS_TYPE access,
    RmClient *pClient
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 5750);
    serverReleaseClient(&g_resServ, access, staticCast(pClient, RsClient));
}
