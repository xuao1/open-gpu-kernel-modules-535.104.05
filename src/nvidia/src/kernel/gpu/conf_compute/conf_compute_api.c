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

/***************************************************************************\
 *                                                                          *
 *      Confidential Compute API Object Module                              *
 *                                                                          *
 \**************************************************************************/

#include "core/locks.h"
#include "rmapi/rs_utils.h"
#include "core/system.h"
#include "gpu/gpu.h"
#include "gpu/mem_mgr/mem_mgr.h"
#include "gpu/mem_mgr/heap.h"
#include "kernel/gpu/mig_mgr/kernel_mig_manager.h"
#include "kernel/gpu/fifo/kernel_fifo.h"
#include "gpu/conf_compute/conf_compute_api.h"
#include "gpu/subdevice/subdevice.h"
#include "class/clcb33.h" // NV_CONFIDENTIAL_COMPUTE

NV_STATUS
confComputeApiConstruct_IMPL
(
    ConfidentialComputeApi       *pConfComputeApi,
    CALL_CONTEXT                 *pCallContext,
    RS_RES_ALLOC_PARAMS_INTERNAL *pParams
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3231);
    OBJSYS    *pSys = SYS_GET_INSTANCE();
    OBJGPUMGR *pGpuMgr = SYS_GET_GPUMGR(pSys);

    pConfComputeApi->pCcCaps = &pGpuMgr->ccCaps;

    return NV_OK;
}

void
confComputeApiDestruct_IMPL
(
    ConfidentialComputeApi *pConfComputeApi
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3232);
}

NV_STATUS
confComputeApiCtrlCmdSystemGetCapabilities_IMPL
(
    ConfidentialComputeApi                                  *pConfComputeApi,
    NV_CONF_COMPUTE_CTRL_CMD_SYSTEM_GET_CAPABILITIES_PARAMS *pParams
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3233);
    OBJSYS    *pSys = SYS_GET_INSTANCE();
    CONF_COMPUTE_CAPS *pCcCaps = pConfComputeApi->pCcCaps;

    LOCK_ASSERT_AND_RETURN(rmapiLockIsOwner() && rmGpuLockIsOwner());

    pParams->cpuCapability = NV_CONF_COMPUTE_SYSTEM_CPU_CAPABILITY_NONE;
    if ((sysGetStaticConfig(pSys))->bOsCCEnabled)
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3234);
        pParams->cpuCapability = NV_CONF_COMPUTE_SYSTEM_CPU_CAPABILITY_AMD_SEV;
        if ((sysGetStaticConfig(pSys))->bOsCCTdxEnabled)
        {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3235);
            pParams->cpuCapability = NV_CONF_COMPUTE_SYSTEM_CPU_CAPABILITY_INTEL_TDX;
        }
    }

    pParams->gpusCapability = NV_CONF_COMPUTE_SYSTEM_GPUS_CAPABILITY_NONE;
    pParams->environment = NV_CONF_COMPUTE_SYSTEM_ENVIRONMENT_UNAVAILABLE;
    pParams->ccFeature = NV_CONF_COMPUTE_SYSTEM_FEATURE_DISABLED;
    pParams->devToolsMode = NV_CONF_COMPUTE_SYSTEM_DEVTOOLS_MODE_DISABLED;

    if (pCcCaps->bApmFeatureCapable)
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3236);
        pParams->gpusCapability = NV_CONF_COMPUTE_SYSTEM_GPUS_CAPABILITY_APM;
    }
    else if (pCcCaps->bHccFeatureCapable)
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3237);
        pParams->gpusCapability = NV_CONF_COMPUTE_SYSTEM_GPUS_CAPABILITY_HCC;
    }

    if (pCcCaps->bCCFeatureEnabled)
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3238);
        if (pCcCaps->bApmFeatureCapable)
        {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3239);
            pParams->ccFeature = NV_CONF_COMPUTE_SYSTEM_FEATURE_APM_ENABLED;
        }
        else if (pCcCaps->bHccFeatureCapable)
        {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3240);
            pParams->ccFeature = NV_CONF_COMPUTE_SYSTEM_FEATURE_HCC_ENABLED;
        }
    }

    if (pCcCaps->bDevToolsModeEnabled)
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3241);
        pParams->devToolsMode = NV_CONF_COMPUTE_SYSTEM_DEVTOOLS_MODE_ENABLED;
    }

    if (pParams->ccFeature != NV_CONF_COMPUTE_SYSTEM_FEATURE_DISABLED)
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3242);
        pParams->environment = NV_CONF_COMPUTE_SYSTEM_ENVIRONMENT_SIM;
    }

    return NV_OK;
}

NV_STATUS
confComputeApiCtrlCmdSystemGetGpusState_IMPL
(
    ConfidentialComputeApi                                *pConfComputeApi,
    NV_CONF_COMPUTE_CTRL_CMD_SYSTEM_GET_GPUS_STATE_PARAMS *pParams
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3243);
    LOCK_ASSERT_AND_RETURN(rmapiLockIsOwner() && rmGpuLockIsOwner());

    pParams->bAcceptClientRequest = pConfComputeApi->pCcCaps->bAcceptClientRequest;

    return NV_OK;
}

NV_STATUS
confComputeApiCtrlCmdSystemSetGpusState_IMPL
(
    ConfidentialComputeApi                                *pConfComputeApi,
    NV_CONF_COMPUTE_CTRL_CMD_SYSTEM_SET_GPUS_STATE_PARAMS *pParams
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3244);
    LOCK_ASSERT_AND_RETURN(rmapiLockIsOwner() && rmGpuLockIsOwner());

    pConfComputeApi->pCcCaps->bAcceptClientRequest = pParams->bAcceptClientRequest;

    return NV_OK;
}

NV_STATUS
confComputeApiCtrlCmdGpuGetVidmemSize_IMPL
(
    ConfidentialComputeApi                              *pConfComputeApi,
    NV_CONF_COMPUTE_CTRL_CMD_GPU_GET_VIDMEM_SIZE_PARAMS *pParams
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3245);
    Subdevice        *pSubdevice            = NULL;
    OBJGPU           *pGpu                  = NULL;
    Heap             *pHeap                 = NULL;
    Heap             *pMemoryPartitionHeap  = NULL;
    KernelMIGManager *pKernelMIGManager     = NULL;
    NvU64             totalProtectedBytes   = 0;
    NvU64             totalUnprotectedBytes = 0;

    LOCK_ASSERT_AND_RETURN(rmapiLockIsOwner() && rmGpuLockIsOwner());

    NV_CHECK_OK_OR_RETURN(LEVEL_INFO,
        subdeviceGetByHandle(RES_GET_CLIENT(pConfComputeApi),
                             pParams->hSubDevice, &pSubdevice));

    pGpu = GPU_RES_GET_GPU(pSubdevice);
    pHeap = GPU_GET_HEAP(pGpu);
    pKernelMIGManager = GPU_GET_KERNEL_MIG_MANAGER(pGpu);

    //
    // If MIG-GPU-Instancing is enabled, we check for GPU instance subscription
    // and provide GPU instance local info
    //
    if (IS_MIG_IN_USE(pGpu))
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3246);
        NV_CHECK_OK_OR_RETURN(LEVEL_INFO,
            kmigmgrGetMemoryPartitionHeapFromDevice(pGpu, pKernelMIGManager,
                                                    GPU_RES_GET_DEVICE(pSubdevice),
                                                    &pMemoryPartitionHeap));
        //
        // If client is associated with a GPU instance then point pHeap
        // to client's memory partition heap
        //
        if (pMemoryPartitionHeap != NULL)
             pHeap = pMemoryPartitionHeap;
    }

    pmaGetTotalProtectedMemory(&pHeap->pmaObject, &totalProtectedBytes);
    pmaGetTotalUnprotectedMemory(&pHeap->pmaObject, &totalUnprotectedBytes);

    pParams->protectedMemSizeInKb = totalProtectedBytes >> 10;
    pParams->unprotectedMemSizeInKb = totalUnprotectedBytes >> 10;

    return NV_OK;
}

NV_STATUS
confComputeApiCtrlCmdGpuSetVidmemSize_IMPL
(
    ConfidentialComputeApi                              *pConfComputeApi,
    NV_CONF_COMPUTE_CTRL_CMD_GPU_SET_VIDMEM_SIZE_PARAMS *pParams
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3247);
    LOCK_ASSERT_AND_RETURN(rmapiLockIsOwner() && rmGpuLockIsOwner());

    return NV_OK;
}

NV_STATUS
confComputeApiCtrlCmdGetGpuCertificate_IMPL
(
    ConfidentialComputeApi                              *pConfComputeApi,
    NV_CONF_COMPUTE_CTRL_CMD_GET_GPU_CERTIFICATE_PARAMS *pParams
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3248);
    Subdevice           *pSubdevice   = NULL;
    OBJGPU              *pGpu         = NULL;
    ConfidentialCompute *pConfCompute = NULL;

    LOCK_ASSERT_AND_RETURN(rmapiLockIsOwner() && rmGpuLockIsOwner());

    NV_CHECK_OK_OR_RETURN(LEVEL_INFO,
        subdeviceGetByHandle(RES_GET_CLIENT(pConfComputeApi),
                             pParams->hSubDevice, &pSubdevice));
    pGpu         = GPU_RES_GET_GPU(pSubdevice);
    pConfCompute = GPU_GET_CONF_COMPUTE(pGpu);

    if (pConfCompute != NULL && pConfCompute->pSpdm != NULL &&
        pConfCompute->getProperty(pConfCompute, PDB_PROP_CONFCOMPUTE_SPDM_ENABLED))
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3249);
        // Set max size of certificate buffers before calling SPDM.
        pParams->certChainSize            = NV_CONF_COMPUTE_CERT_CHAIN_MAX_SIZE;
        pParams->attestationCertChainSize = NV_CONF_COMPUTE_ATTESTATION_CERT_CHAIN_MAX_SIZE;

        return spdmGetCertChains_HAL(pGpu,
                                 pConfCompute->pSpdm,
                                 pParams->certChain,
                                 &pParams->certChainSize,
                                 pParams->attestationCertChain,
                                 &pParams->attestationCertChainSize);
    }

    return NV_ERR_OBJECT_NOT_FOUND;
}

NV_STATUS
confComputeApiCtrlCmdGetGpuAttestationReport_IMPL
(
    ConfidentialComputeApi                                     *pConfComputeApi,
    NV_CONF_COMPUTE_CTRL_CMD_GET_GPU_ATTESTATION_REPORT_PARAMS *pParams
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3250);
    Subdevice           *pSubdevice   = NULL;
    OBJGPU              *pGpu         = NULL;
    ConfidentialCompute *pConfCompute = NULL;

    LOCK_ASSERT_AND_RETURN(rmapiLockIsOwner() && rmGpuLockIsOwner());

    NV_CHECK_OK_OR_RETURN(LEVEL_INFO,
        subdeviceGetByHandle(RES_GET_CLIENT(pConfComputeApi),
                             pParams->hSubDevice, &pSubdevice));
    pGpu         = GPU_RES_GET_GPU(pSubdevice);
    pConfCompute = GPU_GET_CONF_COMPUTE(pGpu);

    if (pConfCompute != NULL && pConfCompute->pSpdm != NULL &&
        pConfCompute->getProperty(pConfCompute, PDB_PROP_CONFCOMPUTE_SPDM_ENABLED))
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3251);
        // Set max size of report buffers before calling SPDM.
        pParams->attestationReportSize    = NV_CONF_COMPUTE_GPU_ATTESTATION_REPORT_MAX_SIZE;
        pParams->cecAttestationReportSize = NV_CONF_COMPUTE_GPU_CEC_ATTESTATION_REPORT_MAX_SIZE;

        return spdmGetAttestationReport(pGpu,
                                        pConfCompute->pSpdm,
                                        pParams->nonce,
                                        pParams->attestationReport,
                                        &pParams->attestationReportSize,
                                        &pParams->isCecAttestationReportPresent,
                                        pParams->cecAttestationReport,
                                        &pParams->cecAttestationReportSize);
    }

    return NV_ERR_OBJECT_NOT_FOUND;
}

NV_STATUS
confComputeApiCtrlCmdGpuGetNumSecureChannels_IMPL
(
    ConfidentialComputeApi                                     *pConfComputeApi,
    NV_CONF_COMPUTE_CTRL_CMD_GPU_GET_NUM_SECURE_CHANNELS_PARAMS *pParams
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3252);
    Subdevice  *pSubdevice;
    OBJGPU     *pGpu;
    KernelFifo *pKernelFifo;

    LOCK_ASSERT_AND_RETURN(rmapiLockIsOwner() && rmGpuLockIsOwner());

    NV_CHECK_OK_OR_RETURN(LEVEL_INFO,
        subdeviceGetByHandle(RES_GET_CLIENT(pConfComputeApi),
        pParams->hSubDevice, &pSubdevice));

    pGpu = GPU_RES_GET_GPU(pSubdevice);
    pKernelFifo = GPU_GET_KERNEL_FIFO(pGpu);

    pParams->maxSec2Channels = pKernelFifo->maxSec2SecureChannels;
    pParams->maxCeChannels = pKernelFifo->maxCeSecureChannels;

    return NV_OK;
}
