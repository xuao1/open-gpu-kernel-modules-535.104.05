/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "gpu/external_device/gsync.h"
#include "gpu/external_device/external_device.h"
#include "os/os.h"
#include "ctrl/ctrl402c.h"

#include "gpu/external_device/dac_p2060.h"
#include "dev_p2060.h"

NV_STATUS
gpuExtdevConstruct_GK104(OBJGPU *pGpu)
{
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3950);
    NvU32    index;
    NvBool   foundDevice = NV_FALSE;

    pfextdevConstruct pConstruct[] =
    {
        extdevConstruct_P2060,
        0                                   // terminates list
    };

    union
    {
        DACP2060EXTERNALDEVICE P2060;
        DACEXTERNALDEVICE ExtDevice;
    } *pExtDeviceTest = NULL;

    pExtDeviceTest = portMemAllocNonPaged(sizeof(*pExtDeviceTest));
    if (pExtDeviceTest == NULL)
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3951);
        NV_PRINTF(LEVEL_ERROR, "Out of memory.\n");
        return NV_ERR_NO_MEMORY;
    }

    if (!(IS_EMULATION(pGpu)))
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3952);
        for (index = 0; pConstruct[index]; index++)
        {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3953);
            portMemSet(pExtDeviceTest, 0, sizeof(*pExtDeviceTest));

            //
            // Map this extdev i2c port to pGpu.
            // It will be needed to detect external device/access extdev registers.
            //

            if ((pConstruct[index])(pGpu, &pExtDeviceTest->ExtDevice))
            {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3954);
                if (pExtDeviceTest->ExtDevice.pI->setI2cHandles(pGpu, &pExtDeviceTest->ExtDevice))
                {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3955);
                    if (pExtDeviceTest->ExtDevice.pI->GetDevice(pGpu, &pExtDeviceTest->ExtDevice))
                    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3956);
                        NV_PRINTF(LEVEL_INFO,
                                  "EXTDEV: device is connecting\n");
                        if (pExtDeviceTest->ExtDevice.pI->Attach(pGpu, (PDACEXTERNALDEVICE *) &pExtDeviceTest))
                        {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3957);
                            if (pExtDeviceTest->ExtDevice.pI->Init(pGpu, &pExtDeviceTest->ExtDevice))
                            {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3958);
                                foundDevice = NV_TRUE;
                                break;
                            }
                        }
                    }
                }

                if (!foundDevice)
                {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3959);
                    pExtDeviceTest->ExtDevice.pI->Destroy(pGpu, &pExtDeviceTest->ExtDevice);
                }
            }
        }
    }

    if (!foundDevice)
    {
    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\n", 3960);
        portMemFree(pExtDeviceTest);

        // Set back extdev i2c to default value.
        pGpu->i2cPortForExtdev = NV402C_CTRL_NUM_I2C_PORTS;
    }

    return NV_OK;
}

