# cmd

在 testKernel 被 launch 时，进入过的三个 cmd.

都是沿 ioctl 进入的，共 62 次 ioctl

以下每个标题依次为：cmd 名称，调用次数

## NV_ESC_RM_CONTROL	55

包括验证参数、管理设备引用、执行控制命令以及处理执行后的状态

**管理设备引用**：如果 `RmIsDeviceRefNeeded(pApi)` 返回 true，表示需要设备引用，代码尝试使用 `RmGetDeviceFd(pApi, &fd)` 获取设备的文件描述符。 然后，它使用 `nv_get_file_private(fd, NV_FALSE, &priv)` 获取设备的文件私有数据。

**执行控制命令**：`Nv04ControlWithSecInfo(pApi, secInfo);`

执行控制命令后，如果 pApi->status 不是 NV_OK 并且 priv 不为 NULL，则该函数释放文件私有数据并清除 secInfo.gpuOsInfo.

**根本没进入各种 if**

所以只需要分析 `Nv04ControlWithSecInfo(pApi, secInfo);`