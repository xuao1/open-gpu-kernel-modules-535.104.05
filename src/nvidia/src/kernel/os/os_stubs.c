/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


/***************************** HW State Routines ***************************\
*                                                                           *
* Module: os_stubs.c                                                        *
*           Stubs for all the public stub routines                          *
\***************************************************************************/

#include "os/os_stub.h"

//
// Here's a little debugging tool. It is possible that some code is stubbed for
// certain OS's that shouldn't be. In debug mode, the stubs below will dump out
// a stub 'number' to help you identify any stubs that are getting called. You
// can then evaluate whether or not that is correct.
//
// Highest used STUB_CHECK is 237.
//
#if defined(DEBUG)
#define STUB_CHECK(n) _stubCallCheck(n)

int enableOsStubCallCheck = 0;

static void _stubCallCheck(int funcNumber)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 4781);
    if (enableOsStubCallCheck) {
        NV_PRINTF(LEVEL_INFO, "STUB CALL: %d \r\n", funcNumber);
    }
}

#else
#define STUB_CHECK(n)
#endif // DEBUG

struct OBJCL;

void stubOsQADbgRegistryInit(OBJOS *pOS)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 4782);
    STUB_CHECK(61);
}

NvU32 stubOsnv_rdcr4(OBJOS *pOS)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 4783);
    STUB_CHECK(76);
    return 0;
}

NvU64 stubOsnv_rdxcr0(OBJOS *pOs)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 4784);
    STUB_CHECK(237);
    return 0;
}

int stubOsnv_cpuid(OBJOS *pOS, int arg1, int arg2, NvU32 *arg3,
                   NvU32 *arg4, NvU32 *arg5, NvU32 *arg6)
{
    STUB_CHECK(77);
    return 0;
}

NvU32 stubOsnv_rdmsr(OBJOS *pOS, NvU32 arg1, NvU32 *arg2, NvU32 *arg3)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 4785);
    STUB_CHECK(122);
    return 0;
}

NvU32 stubOsnv_wrmsr(OBJOS *pOS, NvU32 arg1, NvU32 arg2, NvU32 arg3)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 4786);
    STUB_CHECK(123);
    return 0;
}

NvU32 stubOsRobustChannelsDefaultState(OBJOS *pOS)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 4787);
    STUB_CHECK(128);
    return 0;
}

NV_STATUS stubOsQueueWorkItem(OBJGPU *pGpu, OSWorkItemFunction pFunction, void * pParms)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 4788);
    STUB_CHECK(180);
    return NV_ERR_NOT_SUPPORTED;
}

NV_STATUS stubOsQueueSystemWorkItem(OSSystemWorkItemFunction pFunction, void *pParms)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 4789);
    STUB_CHECK(181);
    return NV_ERR_NOT_SUPPORTED;
}

NV_STATUS stubOsQueueWorkItemWithFlags(OBJGPU *pGpu, OSWorkItemFunction pFunction, void * pParms, NvU32 flags)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 4790);
    STUB_CHECK(182);
    return NV_ERR_NOT_SUPPORTED;
}

NV_STATUS stubOsSimEscapeWrite(OBJGPU *pGpu, const char *path, NvU32 Index, NvU32 Size, NvU32 Value)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 4791);
    STUB_CHECK(195);
    return NV_ERR_GENERIC;
}

NV_STATUS stubOsSimEscapeWriteBuffer(OBJGPU *pGpu, const char *path, NvU32 Index, NvU32 Size, void* pBuffer)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 4792);
    STUB_CHECK(197);
    return NV_ERR_GENERIC;
}

NV_STATUS stubOsSimEscapeRead(OBJGPU *pGpu, const char *path, NvU32 Index, NvU32 Size, NvU32 *Value)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 4793);
    STUB_CHECK(196);
    return NV_ERR_GENERIC;
}

NV_STATUS stubOsSimEscapeReadBuffer(OBJGPU *pGpu, const char *path, NvU32 Index, NvU32 Size, void* pBuffer)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 4794);
    STUB_CHECK(198);
    return NV_ERR_GENERIC;
}

NV_STATUS osCallACPI_MXMX(OBJGPU *pGpu, NvU32 AcpiId, NvU8 *pInOut)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 4795);
    return NV_ERR_NOT_SUPPORTED;
}

NV_STATUS osCallACPI_BCL(OBJGPU *pGpu, NvU32 acpiId, NvU32 *pOut, NvU16 *size)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 4796);
    return NV_ERR_NOT_SUPPORTED;
}

NV_STATUS osCallACPI_ON(OBJGPU *pGpu, NvU32 uAcpiId)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 4797);
    return NV_ERR_NOT_SUPPORTED;
}

NV_STATUS osCallACPI_OFF(OBJGPU *pGpu, NvU32 uAcpiId)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 4798);
    return NV_ERR_NOT_SUPPORTED;
}

NV_STATUS osCallACPI_OPTM_GPUON(OBJGPU *pGpu)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 4799);
    return NV_ERR_NOT_SUPPORTED;
}

NV_STATUS osCallACPI_NVHG_GPUON(OBJGPU *pGpu, NvU32 *pInOut)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 4800);
    return NV_ERR_NOT_SUPPORTED;
}

NV_STATUS osCallACPI_NVHG_GPUOFF(OBJGPU *pGpu, NvU32 *pInOut)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 4801);
    return NV_ERR_NOT_SUPPORTED;
}

NV_STATUS stubOsCallWMI_NVHG_GPUSTA(OBJGPU *pGpu, NvU32 *pInOut)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 4802);
    //STUB_CHECK(227);
    return NV_ERR_NOT_SUPPORTED;
}

NV_STATUS stubOsCallWMI_NVHG_MXDS(OBJGPU *pGpu, NvU32 AcpiId, NvU32 *pInOut)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 4803);
    //STUB_CHECK(228);
    return NV_ERR_NOT_SUPPORTED;
}

NV_STATUS stubOsCallWMI_NVHG_MXMX(OBJGPU *pGpu, NvU32 AcpiId, NvU32 *pInOut)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 4804);
    //STUB_CHECK(229);
    return NV_ERR_NOT_SUPPORTED;
}

NV_STATUS stubOsCallWMI_NVHG_DOS(OBJGPU *pGpu, NvU32 AcpiId, NvU32 *pInOut)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 4805);
    //STUB_CHECK(230);
    return NV_ERR_NOT_SUPPORTED;
}

NV_STATUS stubOsCallWMI_NVHG_DCS(OBJGPU *pGpu, NvU32 AcpiId, NvU32 *pInOut)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 4806);
    //STUB_CHECK(232);
    return NV_ERR_NOT_SUPPORTED;
}

NV_STATUS osCallACPI_MXID(OBJGPU *pGpu, NvU32 ulAcpiId, NvU32 *pInOut)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 4807);
    return NV_ERR_NOT_SUPPORTED;
}

NV_STATUS osCallACPI_LRST(OBJGPU *pGpu, NvU32 ulAcpiId, NvU32 *pInOut)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 4808);
    return NV_ERR_NOT_SUPPORTED;
}

NvBool stubOsCheckCallback(OBJGPU *pGpu)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 4809);
    return NV_FALSE;
}

RC_CALLBACK_STATUS
stubOsRCCallback
(
    OBJGPU  *pGpu,
    NvHandle hClient,    // IN The client RC is on
    NvHandle hDevice,    // IN The device RC is on
    NvHandle hFifo,      // IN The channel or TSG RC is on
    NvHandle hChannel,   // IN The channel RC is on
    NvU32    errorLevel, // IN Error Level
    NvU32    errorType,  // IN Error type
    NvU32   *data,      // IN/OUT context of RC handler
    void    *pfnRmRCReenablePusher
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 4810);
    return RC_CALLBACK_IGNORE;
}

NV_STATUS stubOsSetupVBlank(OBJGPU *pGpu, void * pProc,
                       void * pParm1, void * pParm2, NvU32 Head, void * pParm3)
{
    return NV_OK;
}

NV_STATUS stubOsObjectEventNotification(NvHandle hClient, NvHandle hObject, NvU32 hClass, PEVENTNOTIFICATION pNotifyEvent,
                                    NvU32 notifyIndex, void * pEventData, NvU32 eventDataSize)
{
    return NV_ERR_NOT_SUPPORTED;
}

RmPhysAddr
stubOsPageArrayGetPhysAddr(OS_GPU_INFO *pOsGpuInfo, void* pPageData, NvU32 pageIndex)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 4811);
    NV_ASSERT(0);
    return 0;
}

void stubOsInternalReserveAllocCallback(NvU64 offset, NvU64 size, NvU32 gpuId)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 4812);
    return;
}

void stubOsInternalReserveFreeCallback(NvU64 offset, NvU32 gpuId)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 4813);
    return;
}

NV_STATUS osGetCurrentProcessGfid(NvU32 *pGfid)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 4814);
    return NV_ERR_NOT_SUPPORTED;
}

#if !(RMCFG_FEATURE_PLATFORM_UNIX || RMCFG_FEATURE_PLATFORM_DCE) || \
    (RMCFG_FEATURE_PLATFORM_UNIX && !RMCFG_FEATURE_TEGRA_SOC_NVDISPLAY)
NV_STATUS osTegraSocGpioGetPinState(
    OS_GPU_INFO  *pArg1,
    NvU32         arg2,
    NvU32        *pArg3
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 4815);
    return NV_ERR_NOT_SUPPORTED;
}

void osTegraSocGpioSetPinState(
    OS_GPU_INFO  *pArg1,
    NvU32         arg2,
    NvU32         arg3
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 4816);
}

NV_STATUS osTegraSocGpioSetPinDirection(
    OS_GPU_INFO  *pArg1,
    NvU32         arg2,
    NvU32         arg3
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 4817);
    return NV_ERR_NOT_SUPPORTED;
}

NV_STATUS osTegraSocGpioGetPinDirection(
    OS_GPU_INFO  *pArg1,
    NvU32         arg2,
    NvU32        *pArg3
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 4818);
    return NV_ERR_NOT_SUPPORTED;
}

NV_STATUS osTegraSocGpioGetPinNumber(
    OS_GPU_INFO  *pArg1,
    NvU32         arg2,
    NvU32        *pArg3
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 4819);
    return NV_ERR_NOT_SUPPORTED;
}

NV_STATUS osTegraSocGpioGetPinInterruptStatus(
    OS_GPU_INFO  *pArg1,
    NvU32         arg2,
    NvU32         arg3,
    NvBool       *pArg4
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 4820);
    return NV_ERR_NOT_SUPPORTED;
}

NV_STATUS osTegraSocGpioSetPinInterrupt(
    OS_GPU_INFO  *pArg1,
    NvU32         arg2,
    NvU32         arg3
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 4821);
    return NV_ERR_NOT_SUPPORTED;
}

NV_STATUS
osTegraSocResetMipiCal
(
    OS_GPU_INFO *pOsGpuInfo
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 4822);
    return NV_ERR_NOT_SUPPORTED;
}

NV_STATUS osGetValidWindowHeadMask
(
    OS_GPU_INFO *pArg1,
    NvU64 *pWindowHeadMask
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 4823);
    return NV_ERR_NOT_SUPPORTED;
}

NvBool
osTegraSocIsDsiPanelConnected
(
    OS_GPU_INFO *pOsGpuInfo
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 4824);
    return NV_FALSE;
}

NV_STATUS
osTegraSocDsiParsePanelProps
(
    OS_GPU_INFO *pOsGpuInfo,
    void        *dsiPanelInfo
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 4825);
    return NV_ERR_NOT_SUPPORTED;
}

NV_STATUS
osTegraSocDsiPanelEnable
(
    OS_GPU_INFO *pOsGpuInfo,
    void        *dsiPanelInfo
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 4826);
    return NV_ERR_NOT_SUPPORTED;
}

NV_STATUS
osTegraSocDsiPanelReset
(
    OS_GPU_INFO *pOsGpuInfo,
    void        *dsiPanelInfo
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 4827);
    return NV_ERR_NOT_SUPPORTED;
}

void
osTegraSocDsiPanelDisable
(
    OS_GPU_INFO *pOsGpuInfo,
    void        *dsiPanelInfo
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 4828);
    return;
}

void
osTegraSocDsiPanelCleanup
(
    OS_GPU_INFO *pOsGpuInfo,
    void        *dsiPanelInfo
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 4829);
    return;
}

NV_STATUS
osTegraSocHspSemaphoreAcquire
(
    NvU32 ownerId,
    NvBool bAcquire,
    NvU64 timeout
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 4830);
    return NV_ERR_NOT_SUPPORTED;
}

NvBool
osTegraSocGetHdcpEnabled(OS_GPU_INFO *pOsGpuInfo)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 4831);
    return NV_TRUE;
}
#endif

NV_STATUS
osTegraSocParseFixedModeTimings
(
    OS_GPU_INFO *pOsGpuInfo,
    NvU32 dcbIndex,
    OS_FIXED_MODE_TIMINGS *pFixedModeTimings
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 4832);
    return NV_OK;
}

NV_STATUS osLockPageableDataSection(RM_PAGEABLE_SECTION *pSection)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 4833);
    return NV_OK;
}

NV_STATUS osUnlockPageableDataSection(RM_PAGEABLE_SECTION *pSection)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 4834);
    return NV_OK;
}

NV_STATUS osIsKernelBuffer(void *pArg1, NvU32 arg2)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 4835);
    return NV_OK;
}

NV_STATUS osMapViewToSection(OS_GPU_INFO  *pArg1,
                             void         *pSectionHandle,
                             void         **ppAddress,
                             NvU64         actualSize,
                             NvU64         sectionOffset,
                             NvBool        bIommuEnabled)
{
    return NV_ERR_NOT_SUPPORTED;
}

NV_STATUS osUnmapViewFromSection(OS_GPU_INFO *pArg1,
                                 void *pAddress,
                                 NvBool bIommuEnabled)
{
    return NV_ERR_NOT_SUPPORTED;
}

NV_STATUS osSrPinSysmem(
    OS_GPU_INFO  *pArg1,
    NvU64         commitSize,
    void         *pMdl
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 4836);
    return NV_ERR_NOT_SUPPORTED;
}

NV_STATUS osSrUnpinSysmem(OS_GPU_INFO  *pArg1)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 4837);
    return NV_ERR_NOT_SUPPORTED;
}

NV_STATUS osCreateMemFromOsDescriptorInternal(
    OBJGPU       *pGpu,
    void         *pAddress,
    NvU32         flags,
    NvU64         size,
    MEMORY_DESCRIPTOR **ppMemDesc,
    NvBool        bCachedKernel,
    RS_PRIV_LEVEL privilegeLevel
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 4838);
    return NV_ERR_NOT_SUPPORTED;
}

NV_STATUS osReserveCpuAddressSpaceUpperBound(void **ppSectionHandle,
                                             NvU64 maxSectionSize)
{
    return NV_ERR_NOT_SUPPORTED;
}

void osReleaseCpuAddressSpaceUpperBound(void *pSectionHandle)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 4839);
}

void osIoWriteDword(
    NvU32         port,
    NvU32         data
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 4840);
}

NvU32 osIoReadDword(
    NvU32         port
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 4841);
    return 0;
}

NvBool osIsVga(
    OS_GPU_INFO  *pArg1,
    NvBool        bIsGpuPrimaryDevice
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 4842);
    return bIsGpuPrimaryDevice;
}

void osInitOSHwInfo(
    OBJGPU       *pGpu
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 4843);
}

void osDestroyOSHwInfo(
    OBJGPU       *pGpu
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 4844);
}

NV_STATUS osDoFunctionLevelReset(
    OBJGPU *pGpu
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 4845);
    return NV_ERR_NOT_SUPPORTED;
}

NvBool osGrService(
    OS_GPU_INFO    *pOsGpuInfo,
    NvU32           grIdx,
    NvU32           intr,
    NvU32           nstatus,
    NvU32           addr,
    NvU32           dataLo
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 4846);
    return NV_FALSE;
}

NvBool osDispService(
    NvU32         Intr0,
    NvU32         Intr1
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 4847);
    return NV_FALSE;
}

NV_STATUS osDeferredIsr(
    OBJGPU       *pGpu
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 4848);
    return NV_OK;
}

void osSetSurfaceName(
    void *pDescriptor,
    char *name
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 4849);
}

NV_STATUS osGetAcpiTable(
    NvU32         tableSignature,
    void         **ppTable,
    NvU32         tableSize,
    NvU32        *retSize
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 4850);
    return NV_ERR_NOT_SUPPORTED;
}

NV_STATUS osInitGetAcpiTable(void)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 4851);
    return NV_ERR_NOT_SUPPORTED;
}

void osDbgBugCheckOnAssert(void)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 4852);
    return;
}

NvBool osQueueDpc(OBJGPU *pGpu)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 4853);
    return NV_FALSE;
}

NvBool osBugCheckOnTimeoutEnabled(void)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 4854);
    return NV_FALSE;
}

NV_STATUS osNvifMethod(
    OBJGPU       *pGpu,
    NvU32         func,
    NvU32         subFunc,
    void         *pInParam,
    NvU16         inParamSize,
    NvU32        *pOutStatus,
    void         *pOutData,
    NvU16        *pOutDataSize
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 4855);
    return NV_ERR_NOT_SUPPORTED;
}

NV_STATUS osNvifInitialize(
    OBJGPU       *pGpu
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 4856);
    return NV_ERR_NOT_SUPPORTED;
}

NV_STATUS
stubOsGetUefiVariable
(
    OBJGPU *pGpu,
    char   *pName,
    LPGUID  pGuid,
    NvU8   *pBuffer,
    NvU32  *pSize,
    NvU32  *pAttributes
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 4857);
    return NV_ERR_NOT_SUPPORTED;
}

NV_STATUS
osGetNvGlobalRegistryDword
(
    OBJGPU     *pGpu,
    const char *pRegParmStr,
    NvU32      *pData
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 4858);
    return NV_ERR_NOT_SUPPORTED;
}

#if !RMCFG_FEATURE_PLATFORM_DCE /* dce_core_rm_clk_reset.c */ && \
    (!RMCFG_FEATURE_PLATFORM_UNIX || !RMCFG_FEATURE_TEGRA_SOC_NVDISPLAY || \
     RMCFG_FEATURE_DCE_CLIENT_RM /* osSocNvDisp.c */ )
NV_STATUS
osTegraSocEnableClk
(
    OS_GPU_INFO             *pOsGpuInfo,
    NvU32     whichClkRM
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 4859);
    return NV_ERR_NOT_SUPPORTED;
}

NV_STATUS
osTegraSocDisableClk
(
    OS_GPU_INFO             *pOsGpuInfo,
    NvU32                   whichClkRM
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 4860);
    return NV_ERR_NOT_SUPPORTED;
}

NV_STATUS
osTegraSocGetCurrFreqKHz
(
    OS_GPU_INFO             *pOsGpuInfo,
    NvU32                   whichClkRM,
    NvU32                   *pCurrFreqKHz
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 4861);
    return NV_ERR_NOT_SUPPORTED;
}

NV_STATUS
osTegraSocGetMaxFreqKHz
(
    OS_GPU_INFO             *pOsGpuInfo,
    NvU32                    whichClkRM,
    NvU32                   *pMaxFreqKHz
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 4862);
    return NV_ERR_NOT_SUPPORTED;
}

NV_STATUS
osTegraSocGetMinFreqKHz
(
    OS_GPU_INFO             *pOsGpuInfo,
    NvU32                    whichClkRM,
    NvU32                   *pMinFreqKHz
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 4863);
    return NV_ERR_NOT_SUPPORTED;
}

NV_STATUS
osTegraSocSetFreqKHz
(
    OS_GPU_INFO             *pOsGpuInfo,
    NvU32                    whichClkRM,
    NvU32                    reqFreqKHz
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 4864);
    return NV_ERR_NOT_SUPPORTED;
}

NV_STATUS
osTegraSocSetParent
(
    OS_GPU_INFO             *pOsGpuInfo,
    NvU32                    whichClkRMsource,
    NvU32                    whichClkRMparent
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 4865);
    return NV_ERR_NOT_SUPPORTED;
}

NV_STATUS
osTegraSocGetParent
(
    OS_GPU_INFO             *pOsGpuInfo,
    NvU32                    whichClkRMsource,
    NvU32                   *pWhichClkRMparent
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 4866);
    return NV_ERR_NOT_SUPPORTED;
}

NV_STATUS
osTegraSocDeviceReset
(
    OS_GPU_INFO *pOsGpuInfo
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 4867);
    return NV_ERR_NOT_SUPPORTED;
}

NV_STATUS
osTegraSocPmPowergate
(
    OS_GPU_INFO *pOsGpuInfo
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 4868);
    return NV_ERR_NOT_SUPPORTED;
}

NV_STATUS
osTegraSocPmUnpowergate
(
    OS_GPU_INFO *pOsGpuInfo
)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 4869);
    return NV_ERR_NOT_SUPPORTED;
}

NvU32
osTegraSocFuseRegRead(NvU32 addr)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 4870);
    return 0;
}
#endif

#if !(RMCFG_FEATURE_PLATFORM_UNIX) || \
    (RMCFG_FEATURE_PLATFORM_UNIX && !RMCFG_FEATURE_TEGRA_SOC_NVDISPLAY)
NV_STATUS
osTegraSocDpUphyPllInit(OS_GPU_INFO *pOsGpuInfo, NvU32 link_rate, NvU32 lanes)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 4871);
    return NV_ERR_NOT_SUPPORTED;
}

NV_STATUS
osTegraSocDpUphyPllDeInit(OS_GPU_INFO *pOsGpuInfo)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 4872);
    return NV_ERR_NOT_SUPPORTED;
}

#endif


