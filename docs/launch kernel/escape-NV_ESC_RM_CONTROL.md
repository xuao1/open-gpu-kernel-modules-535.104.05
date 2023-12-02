# NV_ESC_RM_CONTROL

escape.c 中 cmd 进入的宏，testkernel 的 launch kernel 时共执行了 62 次 ioctl，这个宏进入了 55 次

## 0 执行路径

![image-20231202205944474](..\..\img\ioctl-NV_ESC_RM_CONTROL.png)

## 1 escape.c

在 escape.c 中，并没有进入 if，所以核心调用就是 `Nv04ControlWithSecInfo(pApi, secInfo);`

这个函数位于 entry_points.c

## 2 entry_points.c

在这个代码文件中，`Nv04ControlWithSecInfo` 调用了 `_nv04ControlWithSecInfo`

而后者也是在本代码文件中实现的

#### 2.1 _nv04ControlWithSecInfo

处理 Nvidia 驱动程序的控制命令

首先尝试获取一个 Deprecated 的 Handler，如果获取成功，则创建一个 context 并调用这个 Handler，如果获取不成功，则调用 `rmapiGetInterface`

在 testkernel 中，每次都是进入 else，即调用 `rmapiGetInterface`，且每次传的参数都是 0，即 `RMAPI_EXTERNAL`

关键代码：

```c
RM_API *pRmApi = rmapiGetInterface(bInternalCall ? RMAPI_MODS_LOCK_BYPASS : RMAPI_EXTERNAL);

pArgs->status = pRmApi->ControlWithSecInfo(pRmApi, pArgs->hClient, pArgs->hObject, pArgs->cmd, pArgs->params, pArgs->paramsSize, pArgs->flags, &secInfo);
```

所以，接下来需要分析：

+ rmapiGetInterface(RMAPI_EXTERNAL)
+ pRmApi->ControlWithSecInfo

## 3 rmapiGetInterface(RMAPI_EXTERNAL)

位于 rmapi.c

只有一行代码：

```c
return &g_RmApiList[rmapiType];
```

`rmapiType` 是传入的参数，在此处为 RMAPI_EXTERNAL，即 0

那么返回的其实是：

> 阅读代码可知，g_RmApiList 的内容是在函数 `_rmapiInitInterface` 中填充的
>
> 对于 RMAPI_EXTERNAL 的填充是
>
> ```c
> _rmapiInitInterface(&g_RmApiList[RMAPI_EXTERNAL],         
>                     NULL	 /* pDefaultSecInfo */,   
>                     NV_FALSE /* bTlsInternal */,  
>                     NV_FALSE /* bApiLockInternal */, 
>                     NV_FALSE /* bGpuLockInternal */);
> ```
>
> 总之是定义了一堆 API 的入口

所以 rmapiGetInterface 就是返回了一些 API 的入口

而即将被调用的 `pRmApi->ControlWithSecInfo`，则是：

`rmapiControlWithSecInfoTls`

## 4 rmapiControlWithSecInfoTls

本来此处应该分析 `pRmApi->ControlWithSecInfo`

根据上述分析，此处实际是函数 `rmapiDupObjectWithSecInfoTls`，所以直接分析此函数：

该函数首先检查是否可以执行非分页内存分配，如果不可以，则函数调用 `_rmapiControlWithSecInfoTlsIRQL` 并 return。如果可以，使用 `threadStateInit` 初始化线程状态，调用 `rmapiControlWithSecInfo`，传递所有参数来执行实际的控制命令，使用 `threadStateFree` 清理线程状态。

位于 control.c

此处并没有进入 if，即「可以执行非分页内存分配」，所以核心代码为：

```c
threadStateInit(&threadState, THREAD_STATE_FLAGS_NONE);
status = rmapiControlWithSecInfo(pRmApi, hClient, hObject, cmd, pParams, paramsSize, flags, pSecInfo);
threadStateFree(&threadState, THREAD_STATE_FLAGS_NONE);
```

那么接下来需要分析 `rmapiControlWithSecInfo`

## 5 rmapiControlWithSecInfo

位于 control.c

关键代码就一句：

```c
status = _rmapiRmControl(hClient, hObject, cmd, pParams, paramsSize, flags, pRmApi, pSecInfo);
```

而且每次的返回值都是 `NV_OK`

所以继续分析 `_rmapiRmControl`

## 6 _rmapiRmControl

位于 control.c

处理 Nvidia 设备的各种控制命令

> 函数中有几个 `if` 语句块，用于处理特定类型的命令（例如，提升的 IRQL 命令或绕过锁定的命令）
>
> // Three separate rmctrl command modes:
>
> //  mode#1: lock bypass rmctrl request
>
> //  mode#2: raised-irql rmctrl request
>
> //  mode#3: normal rmctrl request

打了 log，发现全部 55 次执行流程均一致：（标记在源码中）

简单来说就是并没有进入特殊类型的命令。中间会有一次 `portMemSet(&rmCtrlParams, 0, sizeof(rmCtrlParams));`。最后在 else 中（源码中的注释表明这里是「Normal rmctrl request」）。而且在 else 中也没有进入 if.

所以被调用的函数有：

+ `serverutilGetClientUnderLock(hClient)`：返回值不为 NULL
+ `rmapiutilGetControlInfo(cmd, &ctrlFlags, &ctrlAccessRight);`：返回值为 NV_OK
+ `rmapiPrologue(pRmApi, &rmApiContext);`
+ `serverControl(&g_resServ, &rmCtrlParams);`
+ `rmapiEpilogue(pRmApi, &rmApiContext);`

目前感觉应该是第四个是关键，先分析他。

## 7 serverControl

位于 re_server.c

#### 执行逻辑：

验证锁信息（pLockInfo）并设置各种本地变量。

它执行一系列锁定操作，以确保对资源的线程安全访问。这包括顶层锁定、客户端锁定和会话锁定。 

它验证客户端和资源引用，确保它们处于活动状态且未失效

设置调用上下文（callContext），其中包括有关资源、客户端、服务器和控制参数的信息。

在控制命令执行期间，将当前线程的本地存储（TLS）调用上下文与新上下文进行交换。

控制命令执行，调用 `resControl(pResourceRef->pResource, &callContext, pParams);`

在执行命令后，还原原始的TLS调用上下文

执行后清理，包括释放锁定和处理任何必要的执行后逻辑

综上，关键的是 `resControl`

根据 log，55 次执行过程均一致，都顺利执行到 `resControl`