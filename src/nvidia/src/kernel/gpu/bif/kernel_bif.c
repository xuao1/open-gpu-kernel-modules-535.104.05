/*
 * SPDX-FileCopyrightText: Copyright (c) 2013-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * @brief  Provides KERNEL only functions for OBJBIF
 */

/* ------------------------ Includes ---------------------------------------- */
#include "gpu/bif/kernel_bif.h"
#include "gpu/bus/kern_bus.h"
#include "gpu/gpu.h"
#include "gpu/intr/intr.h"
#include "os/os.h"
#include "platform/chipset/chipset.h"
#include "core/locks.h"
#include "nvrm_registry.h"
#include "diagnostics/tracer.h"
#include "nvpcie.h"
#include "vgpu/vgpu_events.h"

/* ------------------------ Macros ------------------------------------------ */
/* ------------------------ Compile Time Checks ----------------------------- */
/* ------------------------ Static Function Prototypes ---------------------- */
static void _kbifInitRegistryOverrides(OBJGPU *, KernelBif *);
static void _kbifCheckIfGpuExists(OBJGPU *, void*);
static NV_STATUS _kbifSetPcieRelaxedOrdering(OBJGPU *, KernelBif *, NvBool);

/* ------------------------ Public Functions -------------------------------- */

/*!
 * @brief KernelBif Constructor
 *
 * @param[in] pGpu        GPU object pointer
 * @param[in] pKernelBif  BIF object pointer
 * @param[in] engDesc     Engine descriptor
 */
NV_STATUS
kbifConstructEngine_IMPL
(
    OBJGPU        *pGpu,
    KernelBif     *pKernelBif,
    ENGDESCRIPTOR  engDesc
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 1336);
    // Initialize registry overrides
    _kbifInitRegistryOverrides(pGpu, pKernelBif);

    // WAR for Bug 3208922 - disables P2P on Ampere NB
    kbifApplyWARBug3208922_HAL(pGpu, pKernelBif);

    // Disables P2P on VF
    kbifDisableP2PTransactions_HAL(pGpu, pKernelBif);

    // Cache MNOC interface support
    kbifIsMnocSupported_HAL(pGpu, pKernelBif);

    // Cache VF info
    kbifCacheVFInfo_HAL(pGpu, pKernelBif);

    // Used to track when the link has gone into Recovery, which can cause CEs.
    pKernelBif->EnteredRecoverySinceErrorsLastChecked = NV_FALSE;

    return NV_OK;
}

/*!
 * @brief KernelBif Constructor
 *
 * @param[in] pGpu        GPU object pointer
 * @param[in] pKernelBif  BIF object pointer
 */
NV_STATUS
kbifStateInitLocked_IMPL
(
    OBJGPU    *pGpu,
    KernelBif *pKernelBif
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 1337);
    OBJSYS    *pSys   = SYS_GET_INSTANCE();
    OBJOS     *pOS    = SYS_GET_OS(pSys);
    OBJCL     *pCl    = SYS_GET_CL(pSys);
    NV_STATUS  status = NV_OK;

    // Return early if GPU is connected to an unsupported chipset
    if (pCl->getProperty(pCl, PDB_PROP_CL_UNSUPPORTED_CHIPSET))
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 1338);
        return NV_ERR_NOT_COMPATIBLE;
    }

    // Initialize OS mapping and core logic
    status = osInitMapping(pGpu);
    if (status != NV_OK)
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 1339);
        return status;
    }

    // Initialize BIF static info
    kbifStaticInfoInit(pGpu, pKernelBif);

    // Initialize DMA caps
    kbifInitDmaCaps(pGpu, pKernelBif);

    // Check for OS w/o usable PAT support
    if ((kbifGetBusIntfType_HAL(pKernelBif) ==
         NV2080_CTRL_BUS_INFO_TYPE_PCI_EXPRESS) &&
        pOS->getProperty(pOS, PDB_PROP_OS_PAT_UNSUPPORTED))
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 1340);
        NV_PRINTF(LEVEL_INFO,
                  "BIF disabling noncoherent on OS w/o usable PAT support\n");

        pKernelBif->setProperty(pKernelBif, PDB_PROP_KBIF_SUPPORT_NONCOHERENT, NV_FALSE);
    }

    return status;
}

/*!
 * @brief KernelBif state load
 *
 * @param[in] pGpu        GPU object pointer
 * @param[in] pKernelBif  BIF object pointer
 * @param[in] flags       GPU state flag
 */
NV_STATUS
kbifStateLoad_IMPL
(
    OBJGPU    *pGpu,
    KernelBif *pKernelBif,
    NvU32      flags
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 1341);
    NV_PRINTF(LEVEL_INFO, "BIF DMA Caps: %08x\n", kbifGetDmaCaps(pGpu, pKernelBif));

    // Apply C73 chipset WAR
    kbifExecC73War_HAL(pGpu, pKernelBif);

    // Check for stale PCI-E dev ctrl/status errors and AER errors
    kbifClearConfigErrors(pGpu, pKernelBif, NV_TRUE, KBIF_CLEAR_XVE_AER_ALL_MASK);

    //
    // A vGPU cannot disappear and these accesses are
    // particularly expensive on vGPUs
    //
    if (pKernelBif->getProperty(pKernelBif, PDB_PROP_KBIF_CHECK_IF_GPU_EXISTS_DEF) &&
        !IS_VIRTUAL(pGpu))
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 1342);
        osSchedule1SecondCallback(pGpu, _kbifCheckIfGpuExists, NULL, NV_OS_1HZ_REPEAT);
    }

    return NV_OK;
}

/*!
 * @brief Configure PCIe Relaxed Ordering in BIF
 *
 * @param[in] pGpu        GPU object pointer
 * @param[in] pKernelBif  KBIF object pointer
 * @param[in] enableRo    Enable/disable RO
 */
static NV_STATUS
_kbifSetPcieRelaxedOrdering
(
    OBJGPU    *pGpu,
    KernelBif *pKernelBif,
    NvBool    enableRo
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 1343);
    NV2080_CTRL_INTERNAL_BIF_SET_PCIE_RO_PARAMS pcieRo;
    RM_API    *pRmApi = GPU_GET_PHYSICAL_RMAPI(pGpu);
    NV_STATUS  status;

    pcieRo.enableRo = enableRo;

    status = pRmApi->Control(pRmApi, pGpu->hInternalClient, pGpu->hInternalSubdevice,
                             NV2080_CTRL_CMD_INTERNAL_BIF_SET_PCIE_RO,
                             &pcieRo, sizeof(pcieRo));
    if (status != NV_OK) {
        NV_PRINTF(LEVEL_ERROR, "NV2080_CTRL_CMD_INTERNAL_BIF_SET_PCIE_RO failed %s (0x%x)\n",
                  nvstatusToString(status), status);
        return status;
    }

    return NV_OK;
}

/*!
 * @brief KernelBif state post-load
 *
 * @param[in] pGpu        GPU object pointer
 * @param[in] pKernelBif  KBIF object pointer
 * @param[in] flags       GPU state flag
 */
NV_STATUS
kbifStatePostLoad_IMPL
(
    OBJGPU      *pGpu,
    KernelBif   *pKernelBif,
    NvU32       flags
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 1344);
    NV_STATUS status;

    kbifInitRelaxedOrderingFromEmulatedConfigSpace(pGpu, pKernelBif);
    if (pKernelBif->getProperty(pKernelBif, PDB_PROP_KBIF_PCIE_RELAXED_ORDERING_SET_IN_EMULATED_CONFIG_SPACE)) {
        //
        // This is done from StatePostLoad() to guarantee that BIF's StateLoad()
        // is already completed for both monolithic RM and GSP RM.
        //
        status = _kbifSetPcieRelaxedOrdering(pGpu, pKernelBif, NV_TRUE);
        if (status != NV_OK)
            return NV_OK;
    }

    return NV_OK;
}

/*!
 * @brief KernelBif state unload
 *
 * @param[in] pGpu        GPU object pointer
 * @param[in] pKernelBif  BIF object pointer
 * @param[in] flags       GPU state flag
 */
NV_STATUS
kbifStateUnload_IMPL
(
    OBJGPU    *pGpu,
    KernelBif *pKernelBif,
    NvU32      flags
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 1345);

    return NV_OK;
}

/*!
 * @brief Initialize DMA caps
 *
 * @param[in] pGpu        GPU object pointer
 * @param[in] pKernelBif  BIF object pointer
 */
void
kbifInitDmaCaps_IMPL
(
    OBJGPU    *pGpu,
    KernelBif *pKernelBif
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 1346);
    OBJSYS *pSys = SYS_GET_INSTANCE();
    OBJCL  *pCl  = SYS_GET_CL(pSys);

    pKernelBif->dmaCaps = REF_DEF(BIF_DMA_CAPS_NOSNOOP, _CTXDMA);

    // Set the coherency cap on host RM based on the chipset
    if (IsAMODEL(pGpu) ||
        pCl->getProperty(pCl, PDB_PROP_CL_IS_CHIPSET_IO_COHERENT))
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 1347);
        pKernelBif->dmaCaps |= REF_DEF(BIF_DMA_CAPS_SNOOP, _CTXDMA);
    }
}

NvU32
kbifGetDmaCaps_IMPL
(
    OBJGPU    *pGpu,
    KernelBif *pKernelBif
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 1348);
    NvU32 retval;

    // Start out with system specific DMA caps
    retval = pKernelBif->dmaCaps;

    // If noncoherent support is disabled, mask out SNOOP caps
    if (!pKernelBif->getProperty(pKernelBif, PDB_PROP_KBIF_SUPPORT_NONCOHERENT))
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 1349);
        retval &= ~DRF_SHIFTMASK(BIF_DMA_CAPS_NOSNOOP);
    }

    return retval;
}

/*!
 * @brief Initialize BIF static info in Kernel object through RPC
 *
 * @param[in] pGpu        GPU object pointer
 * @param[in] pKernelBif  BIF object pointer
 */
NV_STATUS
kbifStaticInfoInit_IMPL
(
    OBJGPU    *pGpu,
    KernelBif *pKernelBif
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 1350);
    NV2080_CTRL_INTERNAL_BIF_GET_STATIC_INFO_PARAMS *pStaticInfo;
    RM_API    *pRmApi = GPU_GET_PHYSICAL_RMAPI(pGpu);
    NV_STATUS  status = NV_OK;

    // Allocate memory for the command parameter
    pStaticInfo = portMemAllocNonPaged(sizeof(*pStaticInfo));
    if (pStaticInfo == NULL)
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 1351);
        NV_PRINTF(LEVEL_ERROR, "Could not allocate pStaticInfo for KernelBif");
        status = NV_ERR_NO_MEMORY;
        goto kBifStaticInfoInit_IMPL_exit;
    }
    portMemSet(pStaticInfo, 0, sizeof(*pStaticInfo));

    // Send the command
    NV_CHECK_OK_OR_GOTO(status, LEVEL_ERROR,
                        pRmApi->Control(pRmApi, pGpu->hInternalClient, pGpu->hInternalSubdevice,
                                        NV2080_CTRL_CMD_INTERNAL_BIF_GET_STATIC_INFO,
                                        pStaticInfo, sizeof(*pStaticInfo)),
                        kBifStaticInfoInit_IMPL_exit);

    // Initialize Kernel object fields with RPC response
    pKernelBif->setProperty(pKernelBif, PDB_PROP_KBIF_PCIE_GEN4_CAPABLE,
                            pStaticInfo->bPcieGen4Capable);
    pKernelBif->setProperty(pKernelBif, PDB_PROP_KBIF_IS_C2C_LINK_UP,
                            pStaticInfo->bIsC2CLinkUp);
    pKernelBif->dmaWindowStartAddress = pStaticInfo->dmaWindowStartAddress;

kBifStaticInfoInit_IMPL_exit:
    portMemFree(pStaticInfo);

    return status;
}

/*!
 * @brief Initialize PCI-E config space bits based on chipset and GPU support.
 */
void
kbifInitPcieDeviceControlStatus
(
    OBJGPU    *pGpu,
    KernelBif *pKernelBif
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 1352);
    OBJSYS *pSys = SYS_GET_INSTANCE();
    OBJCL  *pCl  = SYS_GET_CL(pSys);

    kbifEnableExtendedTagSupport_HAL(pGpu, pKernelBif);

    //
    // Bug 382675 and 482867: Many SBIOSes default to disabling relaxed
    // ordering on GPUs, we want to always force it back on unless
    // the upstream root port is known to be broken with respect to this
    // feature.
    //
    if (!pCl->getProperty(pCl, PDB_PROP_CL_RELAXED_ORDERING_NOT_CAPABLE))
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 1353);
        kbifPcieConfigEnableRelaxedOrdering_HAL(pGpu, pKernelBif);
    }
    else
    {
        kbifPcieConfigDisableRelaxedOrdering_HAL(pGpu, pKernelBif);
    }

    //
    // WAR for bug 3661529. All GH100 SKUs will need the NoSnoop WAR.
    // But currently GSP-RM does not detect this correctly,
    //
    if (IsGH100(pGpu))
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 1354);
        pCl->setProperty(pCl, PDB_PROP_CL_ROOTPORT_NEEDS_NOSNOOP_WAR, NV_TRUE);
    }

    if (!pCl->getProperty(pCl, PDB_PROP_CL_NOSNOOP_NOT_CAPABLE) &&
        !pCl->getProperty(pCl, PDB_PROP_CL_ROOTPORT_NEEDS_NOSNOOP_WAR))
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 1355);
        // Bug 393398 - Re-enable DEVICE_CONTROL_STATUS_ENABLE_NO_SNOOP
        kbifEnableNoSnoop_HAL(pGpu, pKernelBif, NV_TRUE);
    }
    else
    {
        //
        // Check for NO_SNOOP P2P bug on specific chipset.  More info in bug 332764.
        // Check for NO_SNOOP enabled by default on specific CPU. Refer bug 1511622.
        //
        kbifEnableNoSnoop_HAL(pGpu, pKernelBif, NV_FALSE);
    }
}

/*!
 * @brief Check and rearm MSI
 *
 * @param[in]   pGpu          GPU object pointer
 * @param[in]   pKernelBif    BIF object pointer
 *
 * @return NV_TRUE   if MSI is enabled
 *         NV_FALSE  if MSI is disabled
 */
void
kbifCheckAndRearmMSI_IMPL
(
    OBJGPU    *pGpu,
    KernelBif *pKernelBif
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 1356);
    Intr *pIntr = GPU_GET_INTR(pGpu);

    if (kbifIsMSIEnabled(pGpu, pKernelBif))
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 1357);
        if (!IS_VIRTUAL(pGpu))
        {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 1358);
            // Send EOI to rearm
            if (pKernelBif->getProperty(pKernelBif, PDB_PROP_KBIF_USE_CONFIG_SPACE_TO_REARM_MSI))
            {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 1359);
                kbifRearmMSI_HAL(pGpu, pKernelBif);
            }
            else
            {
                intrRetriggerTopLevel_HAL(pGpu, pIntr);
            }
        }
    }
    else if (kbifIsMSIXEnabled(pGpu, pKernelBif))
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 1360);
        intrRetriggerTopLevel_HAL(pGpu, pIntr);
    }
}

/*!
 * @brief Checks if MSI is enabled. Prefers to check the SW cache, but if
 * uncached, checks HW state and updates the SW cache for future use
 *
 * @param[in]   pGpu          GPU object pointer
 * @param[in]   pKernelBif    BIF object pointer
 *
 * @return NV_TRUE   if MSI is enabled
 *         NV_FALSE  if MSI is disabled
 */
NvBool
kbifIsMSIEnabled_IMPL
(
    OBJGPU    *pGpu,
    KernelBif *pKernelBif
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 1361);
    //
    // Bug 418883: We shall rely upon value cached at boot, for the value
    // should not change during execution. If however, we must ever change
    // this back to be read at every ISR, we *must* read the value through
    // PCI CFG cycles.
    //
    if (!pKernelBif->getProperty(pKernelBif, PDB_PROP_KBIF_IS_MSI_CACHED))
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 1362);
        if (kbifIsMSIEnabledInHW_HAL(pGpu, pKernelBif))
        {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 1363);
            pKernelBif->setProperty(pKernelBif, PDB_PROP_KBIF_IS_MSI_ENABLED, NV_TRUE);

            if (IS_VIRTUAL(pGpu))
            {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 1364);
                // vGPU guests want an FYI print that re-arming is not required
                NV_PRINTF(LEVEL_WARNING,
                          "MSI is enabled for vGPU, but no need to re-ARM\n");
            }
        }
        else
        {
            pKernelBif->setProperty(pKernelBif, PDB_PROP_KBIF_IS_MSI_ENABLED, NV_FALSE);
        }
        pKernelBif->setProperty(pKernelBif, PDB_PROP_KBIF_IS_MSI_CACHED, NV_TRUE);
    }

    return pKernelBif->getProperty(pKernelBif, PDB_PROP_KBIF_IS_MSI_ENABLED);
}

/*!
 * @brief Checks if MSI-X is enabled. Prefers to check the SW cache, but if
 * uncached, checks HW state and updates the SW cache for future use
 *
 * @param[in]   pGpu          GPU object pointer
 * @param[in]   pKernelBif    BIF object pointer
 *
 * @return NV_TRUE   if MSI is enabled
 *         NV_FALSE  if MSI is disabled
 */
NvBool
kbifIsMSIXEnabled_IMPL
(
    OBJGPU    *pGpu,
    KernelBif *pKernelBif
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 1365);
    if (!pKernelBif->getProperty(pKernelBif, PDB_PROP_KBIF_IS_MSIX_CACHED))
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 1366);
        if (kbifIsMSIXEnabledInHW_HAL(pGpu, pKernelBif))
        {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 1367);
            pKernelBif->setProperty(pKernelBif, PDB_PROP_KBIF_IS_MSIX_ENABLED, NV_TRUE);
        }
        else
        {
            pKernelBif->setProperty(pKernelBif, PDB_PROP_KBIF_IS_MSIX_ENABLED, NV_FALSE);
        }
        pKernelBif->setProperty(pKernelBif, PDB_PROP_KBIF_IS_MSIX_CACHED, NV_TRUE);
    }
    return pKernelBif->getProperty(pKernelBif, PDB_PROP_KBIF_IS_MSIX_ENABLED);
}

/*!
 * @brief Clear PCIe HW PCIe config space error counters.
 * All of these should be cleared using config cycles.
 *
 * @param[in]   pGpu          GPU object pointer
 * @param[in]   pKernelBif    BIF object pointer
 */
void
kbifClearConfigErrors_IMPL
(
    OBJGPU    *pGpu,
    KernelBif *pKernelBif,
    NvBool     bClearStatus,
    NvU32      xveAerFlagsMask
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 1368);
    NvU32 xveStatusFlags = 0;
    NvU32 xveStatus      = 0;
    NvU32 xveAerFlags    = 0;

    if ((bClearStatus) &&
        (kbifGetXveStatusBits_HAL(pGpu, pKernelBif, &xveStatusFlags, &xveStatus) == NV_OK) &&
        (xveStatusFlags != 0))
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 1369);
        NV_PRINTF(LEVEL_WARNING, "PCI-E device status errors pending (%08X):\n",
                  xveStatusFlags);
#ifdef DEBUG
        if ( xveStatusFlags & NV2080_CTRL_BUS_INFO_PCIE_LINK_ERRORS_CORR_ERROR )
        {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 1370);
            NV_PRINTF(LEVEL_WARNING, "     _CORR_ERROR_DETECTED\n");
        }
        if ( xveStatusFlags & NV2080_CTRL_BUS_INFO_PCIE_LINK_ERRORS_NON_FATAL_ERROR )
        {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 1371);
            NV_PRINTF(LEVEL_WARNING, "     _NON_FATAL_ERROR_DETECTED\n");
        }
        if ( xveStatusFlags & NV2080_CTRL_BUS_INFO_PCIE_LINK_ERRORS_FATAL_ERROR )
        {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 1372);
            NV_PRINTF(LEVEL_WARNING, "     _FATAL_ERROR_DETECTED\n");
        }
        if ( xveStatusFlags & NV2080_CTRL_BUS_INFO_PCIE_LINK_ERRORS_UNSUPP_REQUEST )
        {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 1373);
            NV_PRINTF(LEVEL_WARNING, "     _UNSUPP_REQUEST_DETECTED\n");
        }
#endif
        NV_PRINTF(LEVEL_WARNING, "Clearing these errors..\n");
        kbifClearXveStatus_HAL(pGpu, pKernelBif, &xveStatus);
    }

    if ((xveAerFlagsMask) &&
        (kbifGetXveAerBits_HAL(pGpu, pKernelBif, &xveAerFlags) == NV_OK))
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 1374);
        xveAerFlags &= xveAerFlagsMask;

        if (xveAerFlags != 0)
        {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 1375);
            NV_PRINTF(LEVEL_WARNING,
                      "PCI-E device AER errors pending (%08X):\n",
                      xveAerFlags);
#ifdef DEBUG
            if (xveAerFlags & NV2080_CTRL_BUS_INFO_PCIE_LINK_AER_UNCORR_TRAINING_ERR)
            {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 1376);
                NV_PRINTF(LEVEL_WARNING, "     _AER_UNCORR_TRAINING_ERR\n");
            }
            if (xveAerFlags & NV2080_CTRL_BUS_INFO_PCIE_LINK_AER_UNCORR_DLINK_PROTO_ERR)
            {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 1377);
                NV_PRINTF(LEVEL_WARNING, "     _AER_UNCORR_DLINK_PROTO_ERR\n");
            }
            if (xveAerFlags & NV2080_CTRL_BUS_INFO_PCIE_LINK_AER_UNCORR_POISONED_TLP)
            {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 1378);
                NV_PRINTF(LEVEL_WARNING, "     _AER_UNCORR_POISONED_TLP\n");
            }
            if (xveAerFlags & NV2080_CTRL_BUS_INFO_PCIE_LINK_AER_UNCORR_FC_PROTO_ERR)
            {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 1379);
                NV_PRINTF(LEVEL_WARNING, "     _AER_UNCORR_FC_PROTO_ERR\n");
            }
            if (xveAerFlags & NV2080_CTRL_BUS_INFO_PCIE_LINK_AER_UNCORR_CPL_TIMEOUT)
            {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 1380);
                NV_PRINTF(LEVEL_WARNING, "     _AER_UNCORR_CPL_TIMEOUT\n");
            }
            if (xveAerFlags & NV2080_CTRL_BUS_INFO_PCIE_LINK_AER_UNCORR_CPL_ABORT)
            {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 1381);
                NV_PRINTF(LEVEL_WARNING, "     _AER_UNCORR_CPL_ABORT\n");
            }
            if (xveAerFlags & NV2080_CTRL_BUS_INFO_PCIE_LINK_AER_UNCORR_UNEXP_CPL)
            {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 1382);
                NV_PRINTF(LEVEL_WARNING, "     _AER_UNCORR_UNEXP_CPL\n");
            }
            if (xveAerFlags & NV2080_CTRL_BUS_INFO_PCIE_LINK_AER_UNCORR_RCVR_OVERFLOW)
            {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 1383);
                NV_PRINTF(LEVEL_WARNING, "     _AER_UNCORR_RCVR_OVERFLOW\n");
            }
            if (xveAerFlags & NV2080_CTRL_BUS_INFO_PCIE_LINK_AER_UNCORR_MALFORMED_TLP)
            {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 1384);
                NV_PRINTF(LEVEL_WARNING, "     _AER_UNCORR_MALFORMED_TLP\n");
            }
            if (xveAerFlags & NV2080_CTRL_BUS_INFO_PCIE_LINK_AER_UNCORR_ECRC_ERROR)
            {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 1385);
                NV_PRINTF(LEVEL_WARNING, "     _AER_UNCORR_ECRC_ERROR\n");
            }
            if (xveAerFlags & NV2080_CTRL_BUS_INFO_PCIE_LINK_AER_UNCORR_UNSUPPORTED_REQ)
            {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 1386);
                NV_PRINTF(LEVEL_WARNING, "     _AER_UNCORR_UNSUPPORTED_REQ\n");
            }
            if (xveAerFlags & NV2080_CTRL_BUS_INFO_PCIE_LINK_AER_CORR_RCV_ERR)
            {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 1387);
                NV_PRINTF(LEVEL_WARNING, "     _AER_CORR_RCV_ERR\n");
            }
            if (xveAerFlags & NV2080_CTRL_BUS_INFO_PCIE_LINK_AER_CORR_BAD_TLP)
            {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 1388);
                NV_PRINTF(LEVEL_WARNING, "     _AER_CORR_BAD_TLP\n");
            }
            if (xveAerFlags & NV2080_CTRL_BUS_INFO_PCIE_LINK_AER_CORR_BAD_DLLP)
            {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 1389);
                NV_PRINTF(LEVEL_WARNING, "     _AER_CORR_BAD_DLLP\n");
            }
            if (xveAerFlags & NV2080_CTRL_BUS_INFO_PCIE_LINK_AER_CORR_RPLY_ROLLOVER)
            {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 1390);
                NV_PRINTF(LEVEL_WARNING, "     _AER_CORR_RPLY_ROLLOVER\n");
            }
            if (xveAerFlags & NV2080_CTRL_BUS_INFO_PCIE_LINK_AER_CORR_RPLY_TIMEOUT)
            {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 1391);
                NV_PRINTF(LEVEL_WARNING, "     _AER_CORR_RPLY_TIMEOUT\n");
            }
            if (xveAerFlags & NV2080_CTRL_BUS_INFO_PCIE_LINK_AER_CORR_ADVISORY_NONFATAL)
            {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 1392);
                NV_PRINTF(LEVEL_WARNING, "     _AER_CORR_ADVISORY_NONFATAL\n");
            }
#endif
            NV_PRINTF(LEVEL_WARNING, "Clearing these errors..\n");
            kbifClearXveAer_HAL(pGpu, pKernelBif, xveAerFlags);
        }
    }
}

/*!
 * @brief The PCI bus family means it has the concept of bus/dev/func
 *        and compatible PCI config space.
 */
NvBool
kbifIsPciBusFamily_IMPL
(
    KernelBif *pKernelBif
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 1393);
    NvU32 busType = kbifGetBusIntfType_HAL(pKernelBif);

    return ((busType == NV2080_CTRL_BUS_INFO_TYPE_PCI) ||
            (busType == NV2080_CTRL_BUS_INFO_TYPE_PCI_EXPRESS) ||
            (busType == NV2080_CTRL_BUS_INFO_TYPE_FPCI));
}

/*!
 * @brief Regkey Overrides for Bif
 *
 * @param[in]   pGpu          GPU object pointer
 * @param[in]   pKernelBif    BIF object pointer
 */
static void
_kbifInitRegistryOverrides
(
    OBJGPU    *pGpu,
    KernelBif *pKernelBif
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 1394);
    NvU32 data32;

    // P2P Override
    pKernelBif->p2pOverride = BIF_P2P_NOT_OVERRIDEN;
    if (osReadRegistryDword(pGpu, NV_REG_STR_CL_FORCE_P2P, &data32) == NV_OK)
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 1395);
        pKernelBif->p2pOverride = data32;
        pKernelBif->setProperty(pKernelBif, PDB_PROP_KBIF_P2P_READS_DISABLED, FLD_TEST_DRF(_REG_STR, _CL_FORCE_P2P, _READ, _DISABLE, data32));
        pKernelBif->setProperty(pKernelBif, PDB_PROP_KBIF_P2P_WRITES_DISABLED, FLD_TEST_DRF(_REG_STR, _CL_FORCE_P2P, _WRITE, _DISABLE, data32));
    }

    // P2P force type override
    pKernelBif->forceP2PType = NV_REG_STR_RM_FORCE_P2P_TYPE_DEFAULT;
    if (osReadRegistryDword(pGpu, NV_REG_STR_RM_FORCE_P2P_TYPE, &data32) == NV_OK &&
        (data32 <= NV_REG_STR_RM_FORCE_P2P_TYPE_MAX))
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 1396);
        pKernelBif->forceP2PType = data32;
    }

    // Peer Mapping override
    pKernelBif->peerMappingOverride = NV_REG_STR_PEERMAPPING_OVERRIDE_DEFAULT;
    if (osReadRegistryDword(pGpu, NV_REG_STR_PEERMAPPING_OVERRIDE, &data32) == NV_OK)
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 1397);
        NV_PRINTF(LEVEL_INFO, "allow peermapping reg key = %d\n", data32);
        pKernelBif->peerMappingOverride = !!data32;
    }

}

/*!
 * Callback function to check if GPU exists
 *
 * @param[in]  pGpu    GPU object pointer
 * @param[in]  rsvd    Reserved  field
 */
static void
_kbifCheckIfGpuExists
(
    OBJGPU *pGpu,
    void   *rsvd
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 1398);
    if (FULL_GPU_SANITY_CHECK(pGpu))
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 1399);
        if (gpuVerifyExistence_HAL(pGpu) != NV_OK)
        {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 1400);
            osRemove1SecondRepeatingCallback(pGpu, _kbifCheckIfGpuExists, NULL);
        }
    }
}

NvU32
kbifGetGpuLinkCapabilities_IMPL
(
    OBJGPU    *pGpu,
    KernelBif *pKernelBif
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 1401);
    NvU32 addrLinkCap = 0;
    NvU32 data        = 0;

    if (NV_OK != kbifGetBusOptionsAddr_HAL(pGpu, pKernelBif, BUS_OPTIONS_LINK_CAPABILITIES, &addrLinkCap))
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 1402);
        return 0;
    }

    if (NV_OK != GPU_BUS_CFG_RD32(pGpu, addrLinkCap, &data))
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 1403);
        NV_PRINTF(LEVEL_ERROR, "Unable to read %x\n", addrLinkCap);
        return 0;
    }

    return data;
}

NvU32
kbifGetGpuLinkControlStatus_IMPL
(
    OBJGPU    *pGpu,
    KernelBif *pKernelBif
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 1404);
    NvU32 addrLinkControlStatus = 0;
    NvU32 data                  = 0;

    if (NV_OK != kbifGetBusOptionsAddr_HAL(pGpu, pKernelBif, BUS_OPTIONS_LINK_CONTROL_STATUS, &addrLinkControlStatus))
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 1405);
        return 0;
    }

    if (NV_OK != GPU_BUS_CFG_RD32(pGpu, addrLinkControlStatus, &data ))
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 1406);
        NV_PRINTF(LEVEL_ERROR, "Unable to read %x\n", addrLinkControlStatus);
        return 0;
    }

    return data;
}

static NvBool
_doesBoardHaveMultipleGpusAndSwitch(OBJGPU *pGpu)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 1407);
    if (((gpuIsMultiGpuBoard(pGpu)) ||
        (pGpu->getProperty(pGpu, PDB_PROP_GPU_IS_GEMINI)))&&
        ((pGpu->getProperty(pGpu, PDB_PROP_GPU_IS_PLX_PRESENT))  ||
         (pGpu->getProperty(pGpu, PDB_PROP_GPU_IS_BR03_PRESENT)) ||
         (pGpu->getProperty(pGpu, PDB_PROP_GPU_IS_BR04_PRESENT))))
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 1408);
        return NV_TRUE;
    }
    else
    {
        return NV_FALSE;
    }
}

NV_STATUS
kbifControlGetPCIEInfo_IMPL
(
    OBJGPU               *pGpu,
    KernelBif            *pKernelBif,
    NV2080_CTRL_BUS_INFO *pBusInfo
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 1409);
    OBJSYS *pSys  = SYS_GET_INSTANCE();
    OBJCL  *pCl   = SYS_GET_CL(pSys);
    NvU32   index = pBusInfo->index;
    NvU32   data  = 0;

    if (kbifGetBusIntfType_HAL(pKernelBif) != NV2080_CTRL_BUS_INFO_TYPE_PCI_EXPRESS)
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 1410);
        // KMD cannot handle error codes for this ctrl call, hence returning
        // NV_OK, once KMD fixes the bug:3545197, RM can return NV_ERR_NOT_SUPPORTED
        return NV_OK;
    }

    switch (index)
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 1411);
        case NV2080_CTRL_BUS_INFO_INDEX_PCIE_GPU_LINK_CAPS:
        {
            data = kbifGetGpuLinkCapabilities(pGpu, pKernelBif);
            break;
        }
        case NV2080_CTRL_BUS_INFO_INDEX_PCIE_UPSTREAM_LINK_CAPS:
        case NV2080_CTRL_BUS_INFO_INDEX_PCIE_BOARD_LINK_CAPS:
        {
            if (_doesBoardHaveMultipleGpusAndSwitch(pGpu))
            {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 1412);
                if (clPcieReadPortConfigReg(pGpu, pCl,
                                            &pGpu->gpuClData.boardUpstreamPort,
                                            CL_PCIE_LINK_CAP, &data) != NV_OK)
                {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 1413);
                    data = 0;
                }
            }
            else
            {
                data = kbifGetGpuLinkCapabilities(pGpu, pKernelBif);
            }
            break;
        }
        case NV2080_CTRL_BUS_INFO_INDEX_PCIE_UPSTREAM_GEN_INFO:
        case NV2080_CTRL_BUS_INFO_INDEX_PCIE_BOARD_GEN_INFO:
        {
            NvU32 temp;

            if (_doesBoardHaveMultipleGpusAndSwitch(pGpu))
            {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 1414);
                if (clPcieReadPortConfigReg(pGpu, pCl,
                                            &pGpu->gpuClData.boardUpstreamPort,
                                            CL_PCIE_LINK_CTRL_STATUS, &temp) != NV_OK)
                {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 1415);
                    data = 0;
                    break;
                }
                else
                {
                    temp = REF_VAL(NV2080_CTRL_BUS_INFO_PCIE_LINK_CTRL_STATUS_LINK_SPEED, temp);
                    if (temp == NV2080_CTRL_BUS_INFO_PCIE_LINK_CTRL_STATUS_LINK_SPEED_64000MBPS)
                    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 1416);
                        data = FLD_SET_DRF(2080, _CTRL_BUS_INFO_PCIE_LINK_CAP,
                                           _CURR_LEVEL, _GEN6, data);
                    }
                    else if (temp == NV2080_CTRL_BUS_INFO_PCIE_LINK_CTRL_STATUS_LINK_SPEED_32000MBPS)
                    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 1417);
                        data = FLD_SET_DRF(2080, _CTRL_BUS_INFO_PCIE_LINK_CAP,
                                           _CURR_LEVEL, _GEN5, data);
                    }
                    else if (temp == NV2080_CTRL_BUS_INFO_PCIE_LINK_CTRL_STATUS_LINK_SPEED_16000MBPS)
                    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 1418);
                        data = FLD_SET_DRF(2080, _CTRL_BUS_INFO_PCIE_LINK_CAP,
                                           _CURR_LEVEL, _GEN4, data);
                    }
                    else if (temp == NV2080_CTRL_BUS_INFO_PCIE_LINK_CTRL_STATUS_LINK_SPEED_8000MBPS)
                    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 1419);
                        data = FLD_SET_DRF(2080, _CTRL_BUS_INFO_PCIE_LINK_CAP,
                                           _CURR_LEVEL, _GEN3, data);
                    }
                    else if (temp == NV2080_CTRL_BUS_INFO_PCIE_LINK_CTRL_STATUS_LINK_SPEED_5000MBPS)
                    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 1420);
                        data = FLD_SET_DRF(2080, _CTRL_BUS_INFO_PCIE_LINK_CAP,
                                           _CURR_LEVEL, _GEN2, data);
                    }
                    else
                    {
                        data = FLD_SET_DRF(2080, _CTRL_BUS_INFO_PCIE_LINK_CAP,
                                           _CURR_LEVEL, _GEN1, data);
                    }
                }

                if (clPcieReadPortConfigReg(pGpu, pCl,
                                            &pGpu->gpuClData.boardUpstreamPort,
                                            CL_PCIE_LINK_CAP, &temp) != NV_OK)
                {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 1421);
                    data = 0;
                    break;
                }
                else
                {
                    temp = REF_VAL(NV2080_CTRL_BUS_INFO_PCIE_LINK_CAP_MAX_SPEED, temp);
                    if (temp == NV2080_CTRL_BUS_INFO_PCIE_LINK_CAP_MAX_SPEED_64000MBPS)
                    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 1422);
                        data = FLD_SET_DRF(2080, _CTRL_BUS_INFO_PCIE_LINK_CAP,
                                           _GEN, _GEN6, data);
                    }
                    else if (temp == NV2080_CTRL_BUS_INFO_PCIE_LINK_CAP_MAX_SPEED_32000MBPS)
                    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 1423);
                        data = FLD_SET_DRF(2080, _CTRL_BUS_INFO_PCIE_LINK_CAP,
                                           _GEN, _GEN5, data);
                    }
                    else if (temp == NV2080_CTRL_BUS_INFO_PCIE_LINK_CAP_MAX_SPEED_16000MBPS)
                    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 1424);
                        data = FLD_SET_DRF(2080, _CTRL_BUS_INFO_PCIE_LINK_CAP,
                                           _GEN, _GEN4, data);
                    }
                    else if (temp == NV2080_CTRL_BUS_INFO_PCIE_LINK_CAP_MAX_SPEED_8000MBPS)
                    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 1425);
                        data = FLD_SET_DRF(2080, _CTRL_BUS_INFO_PCIE_LINK_CAP,
                                           _GEN, _GEN3, data);
                    }
                    else if (temp == NV2080_CTRL_BUS_INFO_PCIE_LINK_CAP_MAX_SPEED_5000MBPS)
                    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 1426);
                        data = FLD_SET_DRF(2080, _CTRL_BUS_INFO_PCIE_LINK_CAP,
                                           _GEN, _GEN2, data);
                    }
                    else
                    {
                        data = FLD_SET_DRF(2080, _CTRL_BUS_INFO_PCIE_LINK_CAP,
                                           _GEN, _GEN1, data);
                    }
                }
            }
            else
            {
                if (IS_VIRTUAL(pGpu) || IS_GSP_CLIENT(pGpu))
                {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 1427);
                    NV2080_CTRL_BUS_INFO busInfo = {0};
                    NV_STATUS rmStatus = NV_OK;

                    busInfo.index = NV2080_CTRL_BUS_INFO_INDEX_PCIE_GEN_INFO;

                    if ((rmStatus = kbusSendBusInfo(pGpu, GPU_GET_KERNEL_BUS(pGpu), &busInfo)) != NV_OK)
                    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 1428);
                        NV_PRINTF(LEVEL_INFO, "Squashing rmStatus: %x \n", rmStatus);
                        rmStatus = NV_OK;
                        busInfo.data = 0;
                    }
                    data = busInfo.data;
                }
            }
            break;
        }
        case NV2080_CTRL_BUS_INFO_INDEX_PCIE_ROOT_LINK_CAPS:
        {
            if (clPcieReadPortConfigReg(pGpu, pCl,
                                        &pGpu->gpuClData.rootPort,
                                        CL_PCIE_LINK_CAP, &data) != NV_OK)
            {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 1429);
                data = 0;
            }
            break;
        }
        case NV2080_CTRL_BUS_INFO_INDEX_PCIE_DOWNSTREAM_LINK_CAPS:
        {
            if (pGpu->getProperty(pGpu, PDB_PROP_GPU_IS_BR03_PRESENT) ||
                pGpu->getProperty(pGpu, PDB_PROP_GPU_IS_BR04_PRESENT))
            {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 1430);
                if (clPcieReadPortConfigReg(pGpu, pCl,
                                            &pGpu->gpuClData.boardDownstreamPort,
                                            CL_PCIE_LINK_CAP, &data) != NV_OK)
                {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 1431);
                    data = 0;
                }
            }
            else
            {
                // no br03/br04, same as link from RC
                if (clPcieReadPortConfigReg(pGpu, pCl,
                                            &pGpu->gpuClData.rootPort,
                                            CL_PCIE_LINK_CAP, &data) != NV_OK)
                {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 1432);
                    data = 0;
                }
            }
            break;
        }
        case NV2080_CTRL_BUS_INFO_INDEX_PCIE_GPU_LINK_CTRL_STATUS:
        {
            data = kbifGetGpuLinkControlStatus(pGpu, pKernelBif);
            break;
        }
        case NV2080_CTRL_BUS_INFO_INDEX_PCIE_UPSTREAM_LINK_CTRL_STATUS:
        case NV2080_CTRL_BUS_INFO_INDEX_PCIE_BOARD_LINK_CTRL_STATUS:
        {
            if (_doesBoardHaveMultipleGpusAndSwitch(pGpu))
            {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 1433);
                if (clPcieReadPortConfigReg(pGpu, pCl,
                                            &pGpu->gpuClData.boardUpstreamPort,
                                            CL_PCIE_LINK_CTRL_STATUS, &data) != NV_OK)
                {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 1434);
                    data = 0;
                }
            }
            else
            {
                data = kbifGetGpuLinkControlStatus(pGpu, pKernelBif);
            }
            break;
        }
        case NV2080_CTRL_BUS_INFO_INDEX_PCIE_ROOT_LINK_CTRL_STATUS:
        {
            if (clPcieReadPortConfigReg(pGpu, pCl,
                                        &pGpu->gpuClData.rootPort,
                                        CL_PCIE_LINK_CTRL_STATUS,
                                        &data) != NV_OK)
            {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 1435);
                data = 0;
            }
            break;
        }
        case NV2080_CTRL_BUS_INFO_INDEX_PCIE_DOWNSTREAM_LINK_CTRL_STATUS:
        {
            if (pGpu->getProperty(pGpu, PDB_PROP_GPU_IS_BR03_PRESENT) ||
                pGpu->getProperty(pGpu, PDB_PROP_GPU_IS_BR04_PRESENT))
            {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 1436);
                if (clPcieReadPortConfigReg(pGpu, pCl,
                                            &pGpu->gpuClData.boardDownstreamPort,
                                            CL_PCIE_LINK_CTRL_STATUS, &data) != NV_OK)
                {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 1437);
                    data = 0;
                }
            }
            else
            {
                // no br03/br04, same as link from RC
                if (clPcieReadPortConfigReg(pGpu, pCl,
                                            &pGpu->gpuClData.rootPort,
                                            CL_PCIE_LINK_CTRL_STATUS,
                                            &data) != NV_OK)
                {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 1438);
                    data = 0;
                }
            }
            break;
        }
        case NV2080_CTRL_BUS_INFO_INDEX_PCIE_GPU_LINK_ERRORS:
        {
            NvU32 xveStatus = 0;

            if (pKernelBif != NULL)
            {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 1439);
                if (kbifGetXveStatusBits_HAL(pGpu, pKernelBif, &data, &xveStatus) != NV_OK)
                {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 1440);
                    data = 0;
                    break;
                }
                if (kbifClearXveStatus_HAL(pGpu, pKernelBif, &xveStatus) != NV_OK)
                {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 1441);
                    data = 0;
                }
            }
            break;
        }
        case NV2080_CTRL_BUS_INFO_INDEX_PCIE_ROOT_LINK_ERRORS:
        {
            NvU32 clStatus = 0;

            if (clPcieReadDevCtrlStatus(pGpu, pCl, &data, &clStatus) != NV_OK)
            {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 1442);
                data = 0;
                break;
            }
            if (clPcieClearDevCtrlStatus(pGpu, pCl, &clStatus) != NV_OK)
            {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 1443);
                data = 0;
            }
            break;
        }
        case NV2080_CTRL_BUS_INFO_INDEX_PCIE_GPU_LINK_AER:
        {
            if (pKernelBif != NULL)
            {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 1444);
                if (kbifGetXveAerBits_HAL(pGpu, pKernelBif, &data) != NV_OK)
                {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 1445);
                    data = 0;
                    break;
                }
                if (kbifClearXveAer_HAL(pGpu, pKernelBif, data) != NV_OK)
                {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 1446);
                    data = 0;
                }
            }
            break;
        }
        case NV2080_CTRL_BUS_INFO_INDEX_MSI_INFO:
        {
            if (kbifIsMSIEnabledInHW_HAL(pGpu, pKernelBif))
            {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 1447);
                data = FLD_SET_DRF(2080, _CTRL_BUS_INFO_MSI,
                                   _STATUS, _ENABLED, data);
            }
            else
            {
                data = FLD_SET_DRF(2080, _CTRL_BUS_INFO_MSI,
                                   _STATUS, _DISABLED, data);
            }
            break;
        }

        default:
            break;
    }

    pBusInfo->data = data;
    return NV_OK;
}

/*!
 * @brief To ensure GPU is back on bus and accessible by polling device ID
 *
 * @param[in]  pGpu        GPU object pointer
 * @param[in]  pKernelBif  Kernel BIF object pointer
 *
 * @returns NV_OK
 * @returns NV_ERR_TIMEOUT
 */
NV_STATUS
kbifPollDeviceOnBus_IMPL
(
    OBJGPU     *pGpu,
    KernelBif  *pKernelBif
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 1448);
    RMTIMEOUT timeout;

    gpuSetTimeout(pGpu, GPU_TIMEOUT_DEFAULT, &timeout, 0);

    while (osPciInitHandle(gpuGetDomain(pGpu),
                           gpuGetBus(pGpu),
                           gpuGetDevice(pGpu), 0, NULL, NULL) == NULL)
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 1449);
        if (gpuCheckTimeout(pGpu, &timeout) == NV_ERR_TIMEOUT)
        {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 1450);
            NV_PRINTF(LEVEL_ERROR, "Timeout polling GPU back on bus\n");
            DBG_BREAKPOINT();
            return NV_ERR_TIMEOUT;
        }
        osDelayUs(100);
    }

    return NV_OK;
}

