# escape.c

## 1 概述

本文件主要是实现了 RmIoctl

## 2 RmIoctl

### 2.1 函数定义和参数

```c
NV_STATUS RmIoctl(
    nv_state_t *nv,
    nv_file_private_t *nvfp,
    NvU32 cmd,
    void *data,
    NvU32 dataSize
)
```

- `nv_state_t *nv`: 指向 NVIDIA 设备状态的指针。
- `nv_file_private_t *nvfp`: 指向文件私有数据的指针。
- `NvU32 cmd`: IOCTL 命令代码。`arg_cmd`
- `void *data`: 指向数据的指针，通常是一个结构体。`arg_copy`
- `NvU32 dataSize`: 数据的大小。`arg_size`

### 2.2 switch(cmd)

函数体中，除了刚开始设置了返回状态和安全信息结构体，剩下的内容均为根据 cmd 进行处理。

非常结构化，基本上每个 case 都是先进行设备检查、参数有效性检查，然后再调用具体的操作函数。

![image-20231127191339408](C:\Users\15370\AppData\Roaming\Typora\typora-user-images\image-20231127191339408.png)

#### 2.2.1 NV_ESC_RM_ALLOC_MEMORY

内存分配

- 根据 `hClass` 的值执行不同的内存分配逻辑。
- 对于特定类型的内存（如系统内存），可能还涉及映射上下文（mmap context）的创建。

```c
RmAllocOsDescriptor(pParms, secInfo);
Nv01AllocMemoryWithSecInfo(pParms, secInfo);
rm_create_mmap_context(pParms->hRoot, pParms->hObjectParent, pParms->hObjectNew, pParms->pMemory, pParms->limit + 1, 0, NV_MEMORY_DEFAULT, pApi->fd)
```

#### 2.2.2 NV_ESC_RM_ALLOC_OBJECT

object 分配

```c
Nv01AllocObjectWithSecInfo(pApi, secInfo);
```

#### 2.2.3 NV_ESC_RM_ALLOC

内存分配

根据 dataSize == sizeof(NVOS64_PARAMETERS) 判断执行哪种内存分配函数

```c
Nv04AllocWithSecInfo(pApi, secInfo);
Nv04AllocWithAccessSecInfo(pApiAccess, secInfo);
```

#### 2.2.4 NV_ESC_RM_FREE

释放之前分配的资源

如果 `pApi` 的状态为 `NV_OK` 并且 `hObjectOld` 等于 `hRoot`，则执行额外的释放操作

```c
Nv01FreeWithSecInfo(pApi, secInfo);
rm_client_free_os_events(pApi->hRoot);
```

#### 2.2.5 NV_ESC_RM_VID_HEAP_CONTROL

根据 `pApi` 的 `function` 字段执行相应的操作，例如分配描述符或控制视频堆

```c
RmCreateOsDescriptor(pApi, secInfo);
Nv04VidHeapControlWithSecInfo(pApi, secInfo);
```

#### 2.2.6 NV_ESC_RM_I2C_ACCESS

执行 I2C 访问操作

```c
Nv04I2CAccessWithSecInfo(pApi, secInfo);
```

#### 2.2.7 NV_ESC_RM_IDLE_CHANNELS

执行闲置 GPU 通道的操作

```c
Nv04IdleChannelsWithSecInfo(pApi, secInfo);
```

#### 2.2.8 NV_ESC_RM_MAP_MEMORY

执行内存映射操作:

- 设置内存映射标志并执行映射操作。
- 如果映射成功，创建内存映射上下文。
- 如果创建上下文失败，执行解映射操作。

```c
Nv04MapMemoryWithSecInfo(pParms, secInfo);
rm_create_mmap_context(pParms->hClient, pParms->hDevice, pParms->hMemory, pParms->pLinearAddress, pParms->length, pParms->offset, DRF_VAL(OS33, _FLAGS, _CACHING_TYPE, pParms->flags), pApi->fd);
portMemSet(&params, 0, sizeof(NVOS34_PARAMETERS));
Nv04UnmapMemoryWithSecInfo(&params, secInfo);
```

#### 2.2.9 NV_ESC_RM_UNMAP_MEMORY

执行内存解映射操作

```c
Nv04UnmapMemoryWithSecInfo(pApi, secInfo);
```

#### 2.2.10 NV_ESC_RM_ACCESS_REGISTRY

执行注册表访问操作

```c
rm_access_registry(pApi->hClient, pApi->hObject, pApi->AccessType, pApi->pDevNode, pApi->DevNodeLength, pApi->pParmStr, pApi->ParmStrLength, pApi->pBinaryData, &pApi->BinaryDataLength, &pApi->Data, &pApi->Entry);
```

#### 2.2.11 NV_ESC_RM_ALLOC_CONTEXT_DMA2

执行上下文 DMA 分配操作

```c
Nv04AllocContextDmaWithSecInfo(pApi, secInfo);
```

#### 2.2.12 NV_ESC_RM_BIND_CONTEXT_DMA

执行上下文 DMA 绑定操作

```c
Nv04BindContextDmaWithSecInfo(pApi, secInfo);
```

#### 2.2.13 NV_ESC_RM_MAP_MEMORY_DMA

执行内存 DMA 映射操作

```c
Nv04MapMemoryDmaWithSecInfo(pApi, secInfo);
```

#### 2.2.14 NV_ESC_RM_UNMAP_MEMORY_DMA

执行内存 DMA 解映射操作

```c
Nv04UnmapMemoryDmaWithSecInfo(pApi, secInfo);
```

#### 2.2.15 NV_ESC_RM_DUP_OBJECT

执行对象复制操作

```c
Nv04DupObjectWithSecInfo(pApi, secInfo);
```

#### 2.2.16 NV_ESC_RM_SHARE



```c
Nv04ShareWithSecInfo(pApi, secInfo);
```

#### 2.2.17 NV_ESC_ALLOC_OS_EVENT



```c
rm_alloc_os_event(pApi->hClient, nvfp, pApi->fd);
```

#### 2.2.18 NV_ESC_FREE_OS_EVENT



```c
rm_free_os_event(pApi->hClient, pApi->fd);
```

#### 2.2.19 NV_ESC_RM_GET_EVENT_DATA



```c
rm_get_event_data(nvfp, pApi->pEvent, &pApi->MoreEvents);
```

#### 2.2.20 NV_ESC_STATUS_CODE



```c
nv_get_adapter_state(pApi->domain, pApi->bus, pApi->slot);
rm_get_adapter_status(pNv, &pApi->status);
```

#### 2.2.21 NV_ESC_RM_CONTROL



```c
RmIsDeviceRefNeeded(pApi)
RmGetDeviceFd(pApi, &fd);
nv_get_file_private(fd, NV_FALSE, &priv);
portAtomicCompareAndSwapU32(&dev_nvfp->register_or_refcount, NVFP_TYPE_REFCOUNTED, NVFP_TYPE_NONE))
nv_put_file_private(priv);
Nv04ControlWithSecInfo(pApi, secInfo);
```

#### 2.2.22 NV_ESC_RM_UPDATE_DEVICE_MAPPING_INFO



```c
rm_update_device_mapping_info(pApi->hClient, pApi->hDevice, pApi->hMemory, pOldCpuAddress, pNewCpuAddress);
```

#### 2.2.23 NV_ESC_RM_NVLOG_CTRL



```c
osIsAdministrator();
cliresCtrlCmdNvdGetNvlogInfo_IMPL(NULL, &pParams->params.getNvlogInfo);
cliresCtrlCmdNvdGetNvlogBufferInfo_IMPL(NULL, &pParams->params.getNvlogBufferInfo);
cliresCtrlCmdNvdGetNvlog_IMPL(NULL, &pParams->params.getNvlog);
```

#### 2.2.24 NV_ESC_REGISTER_FD



```c
rmapiLockAcquire(API_LOCK_FLAGS_NONE, RM_LOCK_MODULES_OSAPI);
rmapiLockRelease();
nv_get_file_private(params->ctl_fd, NV_TRUE, /* require ctl fd */ &priv);
nv_put_file_private(priv);
portAtomicCompareAndSwapU32(&nvfp->register_or_refcount, NVFP_TYPE_REGISTERED, NVFP_TYPE_NONE)
portAtomicSetSize(&nvfp->ctl_nvfp, ctl_nvfp);
```

#### 2.2.25 default

报错，未知的 ioctl

## 3 其他函数

### 3.1 RmIsDeviceRefNeeded

分析 pApi->cmd，如果是 `NV00FD_CTRL_CMD_ATTACH_GPU`，则为 NV_TRUE，否则为 NV_FALSE.

### 3.2 RmGetDeviceFd

### 3.3 RmCreateOsDescriptor

### 3.4 RmAllocOsDescriptor