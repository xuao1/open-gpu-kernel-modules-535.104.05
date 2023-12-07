/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2020 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "core/hal_mgr.h"
#include "core/hal.h"

#include "g_hal_archimpl.h"

// NOTE: string arguments only get used when NV_PRINTF_STRINGS_ALLOWED is true.
#if NV_PRINTF_STRINGS_ALLOWED
static const char *_halmgrGetStringRepForHalImpl(OBJHALMGR *pHalMgr, HAL_IMPLEMENTATION);
#endif

NV_STATUS
halmgrConstruct_IMPL
(
    OBJHALMGR *pHalMgr
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 81);
    HAL_IMPLEMENTATION i;

    //
    // Make sure all the possible handles to the Hal Objects
    // have been zeroed out. Also initialize the implementation
    // and public id's in the hal descriptor list.
    //
    for (i = 0; i < HAL_IMPL_MAXIMUM; i++)
        pHalMgr->pHalList[i] = NULL;

    return NV_OK;
}

void
halmgrDestruct_IMPL
(
    OBJHALMGR *pHalMgr
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 82);
    NvU32 i;

    for (i = 0; i < HAL_IMPL_MAXIMUM; i++)
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 83);
        objDelete(pHalMgr->pHalList[i]);
        pHalMgr->pHalList[i] = NULL;
    }
}

NV_STATUS
halmgrCreateHal_IMPL
(
    OBJHALMGR *pHalMgr,
    NvU32      halImpl
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 84);
    OBJHAL    *pHal;
    NV_STATUS  status;

    NV_ASSERT_OR_RETURN(halImpl < HAL_IMPL_MAXIMUM, NV_ERR_INVALID_ARGUMENT);

    status = objCreate(&pHal, pHalMgr, OBJHAL);
    if (status != NV_OK)
        return status;

    // Store away the object pointer for this particular HAL object
    pHalMgr->pHalList[halImpl] = pHal;

    return NV_OK;
}

POBJHAL
halmgrGetHal_IMPL
(
    OBJHALMGR *pHalMgr,
    NvU32      halImpl
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 85);
    if (halImpl < HAL_IMPL_MAXIMUM)
        return pHalMgr->pHalList[halImpl];
    else
        return NULL;
}

static NvBool
_halmgrIsTegraSupported
(
    NvU32 publicHalID,
    NvU32 socChipID
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 86);
    NvU32 chipid, majorRev;

    chipid = DRF_VAL(_PAPB_MISC, _GP_HIDREV, _CHIPID, socChipID);
    majorRev = DRF_VAL(_PAPB_MISC, _GP_HIDREV, _MAJORREV, socChipID);

    // WAR: The majorrev of t234 shows 0xa on fmodel instead of 0x4
    if ((chipid == 0x23) && (majorRev == 0xa))
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 87);
        majorRev = 0x4;
    }

    // Convert to the HIDREV field format of chip-config
    return ((chipid << 4) | majorRev) == chipID[publicHalID].hidrev;
}

static NvBool
_halmgrIsChipSupported
(
    OBJHALMGR *pHalMgr,
    NvU32      publicHalID,
    NvU32      pPmcBoot0,
    NvU32      pPmcBoot42
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 88);
    NvBool retVal = NV_FALSE;

    if (chipID[publicHalID].hidrev)
        return _halmgrIsTegraSupported(publicHalID, pPmcBoot0);

    if (pPmcBoot42)
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 89);
        if ((DRF_VAL(_PMC, _BOOT_42, _ARCHITECTURE, pPmcBoot42) == chipID[publicHalID].arch) &&
            (DRF_VAL(_PMC, _BOOT_42, _IMPLEMENTATION, pPmcBoot42) == chipID[publicHalID].impl))
        {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 90);
            retVal = NV_TRUE;
        }
    }
    else
    {
        // Fail safely on older GPUs where pPmcBoot42 is not supported
        retVal = NV_FALSE;
    }

    return retVal;
}

NV_STATUS
halmgrGetHalForGpu_IMPL
(
    OBJHALMGR *pHalMgr,
    NvU32      pPmcBoot0,
    NvU32      pPmcBoot42,
    NvU32      *pHalImpl
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 91);
    HAL_IMPLEMENTATION  halImpl;
    OBJHAL             *pHal;

    for (halImpl = 0; halImpl < HAL_IMPL_MAXIMUM; halImpl++)
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 92);
        pHal = pHalMgr->pHalList[halImpl];

        // skip impls that have no hal object
        if (pHal == NULL)
            continue;

        if (_halmgrIsChipSupported(pHalMgr, halImpl, pPmcBoot0, pPmcBoot42))
        {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 93);
            *pHalImpl = halImpl;

#if NV_PRINTF_STRINGS_ALLOWED
            NV_PRINTF(LEVEL_INFO,
                      "Matching %s = 0x%x to HAL_IMPL_%s\n",
                      pPmcBoot42 ? "PMC_BOOT_42" : "PMC_BOOT_0",
                      pPmcBoot42 ? pPmcBoot42 : pPmcBoot0,
                      _halmgrGetStringRepForHalImpl(pHalMgr, halImpl));
#else // NV_PRINTF_STRINGS_ALLOWED
            NV_PRINTF(LEVEL_INFO,
                      "Matching 0x%x to %u\n",
                      pPmcBoot42 ? "PMC_BOOT_42" : "PMC_BOOT_0",
                      halImpl);
#endif // NV_PRINTF_STRINGS_ALLOWED

            return NV_OK;
        }
    }

    return NV_ERR_NOT_SUPPORTED;
}

// NOTE: string arguments only get used when NV_PRINTF_STRINGS_ALLOWED is true.
#if NV_PRINTF_STRINGS_ALLOWED
static const char *
_halmgrGetStringRepForHalImpl
(
    OBJHALMGR *pHalMgr,
    HAL_IMPLEMENTATION halImpl
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 94);
    const char *chipName = "UNKNOWN";
    static const struct
    {
        HAL_IMPLEMENTATION halImpl;
        const char *name;
    } halImplNames[] = { HAL_IMPL_NAME_LIST }; // generated by rmconfig into g_hal.h
    NvU32 i;

    for (i = 0; i < NV_ARRAY_ELEMENTS(halImplNames); i++)
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 95);
        if (halImplNames[i].halImpl == halImpl)
        {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 96);
            chipName = halImplNames[i].name;
            break;
        }
    }

    return chipName;
}
#endif

