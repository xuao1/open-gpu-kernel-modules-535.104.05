# nvidia-ioctl

## 1 概述

该函数位于 kernel-open/nvdia/nv.c 文件中，是 nv-frontend.c 中的 `nvidia_frontend_ioctl` 调用的 `module->ioctl` 的具体实现。

```c
int
nvidia_ioctl(
    struct inode *inode,
    struct file *file,
    unsigned int cmd,
    unsigned long i_arg)
{
```

## 2 参数

- `struct inode *inode`：表示文件系统节点，包含与设备文件相关的信息。
- `struct file *file`：表示打开的设备文件。
- `unsigned int cmd`：指定 ioctl 操作的命令。
- `unsigned long i_arg`：传递给 ioctl 的参数，通常是一个指向用户空间数据的指针。

## 3 函数流程

1. **获取设备状态**:
   - 通过 `NV_GET_NVL_FROM_FILEP(file)` 获取设备的状态。
   - 检查设备的当前状态，如果设备已丢失，返回错误。
2. **处理不同的 ioctl 请求**:
   - 根据 `cmd` 的值，代码会执行不同的操作。
   - 对于不同的 ioctl 命令，代码可能会读取或修改设备状态、获取信息、或执行其他设备特定操作。
3. **内存操作**:
   - 使用 `NV_COPY_FROM_USER` 从用户空间复制数据到内核空间。
   - 为处理 ioctl 请求分配内核内存，处理完成后释放。
4. **错误处理**:
   - 在执行过程中遇到错误时，会打印相关错误消息并在必要时释放资源。
5. **完成请求**:
   - 执行完 ioctl 操作后，将结果从内核空间复制回用户空间（如果需要）。
   - 释放分配的资源，并返回操作的结果（成功或错误代码）。

## 4 执行 cmd 前的处理

1. **定义和初始化变量**:

   - `NV_STATUS rmStatus; int status = 0;` 定义 NVIDIA 状态和通用状态变量。
   - `nv_linux_state_t *nvl = NV_GET_NVL_FROM_FILEP(file);` 从文件指针获取 NVIDIA Linux 状态。
   - `nv_state_t *nv = NV_STATE_PTR(nvl);` 获取通用 NVIDIA 状态。
   - `nv_linux_file_private_t *nvlfp = NV_GET_LINUX_FILE_PRIVATE(file);` 获取文件的私有数据。
   - **`nvidia_stack_t *sp = NULL;` 初始化 NVIDIA 栈指针为 NULL。**
   - **`nv_ioctl_xfer_t ioc_xfer;` 定义一个 IOCTL 传输结构。**
   - `void *arg_ptr = (void *) i_arg;` 将 IOCTL 参数转换为指针。
   - `void *arg_copy = NULL; size_t arg_size = 0; int arg_cmd;` 初始化用于存储复制数据的变量。

2. **打印更多调试信息**:

   - 使用 `nv_printf` 打印 IOCTL 请求的详细信息。

3. **获取系统电源管理锁**:

   - `nv_down_read_interruptible(&nv_system_pm_lock);` 尝试获取读取电源管理锁。如果失败，则返回错误状态。

4. **获取 NVIDIA 内核栈**:

   - **`sp = nv_nvlfp_get_sp(nvlfp, NV_FOPS_STACK_INDEX_IOCTL);` 获取与 IOCTL 操作相关的内核栈。**

5. **检查 GPU 状态**:

   - `rmStatus = nv_check_gpu_state(nv);` 检查 GPU 的当前状态。
   - 如果 GPU 已丢失 (`NV_ERR_GPU_IS_LOST`)，则打印信息并设置错误状态，跳转到 `done`。

6. **处理 IOCTL 命令和参数**:

   - 获取 IOCTL 命令的大小和编号。

   - 如果是特殊的传输命令

     ```c
     NV_ESC_IOCTL_XFER_CMD
     ```

     则进行进一步处理：

     - 检查传输结构的大小是否正确。
     - 从用户空间复制传输数据到 `ioc_xfer`。
     - 更新命令和参数的大小和指针，以处理实际的 IOCTL 请求。
     - 如果参数大小超过最大限制，则设置错误状态。

7. **为 IOCTL 数据分配内存并从用户空间复制**:

   - 为 `arg_copy` 分配内核空间内存。
   - 如果内存分配失败，则设置错误状态并跳转到 `done`。
   - 从用户空间复制数据到分配的内存 `arg_copy`。
   - 如果复制失败，则设置错误状态并跳转到 `done`。

## 5 cmd

这一部分由 `switch (arg_cmd)` 展开

### 5.1 NV_ESC_QUERY_DEVICE_INTR

从设备的寄存器映射中读取中断状态，并将其存储在 `query_intr->intrStatus` 中

### 5.2 NV_ESC_CARD_INFO

调用 `nvidia_read_card_info` 函数读取图形卡信息，并将结果存储在 `arg_copy` 指向的内存中

### 5.3 NV_ESC_ATTACH_GPUS_TO_FD

1. **计算 GPU 数量**:
   - 根据参数大小和 `NvU32` 类型的大小计算 GPU 数量。
2. **检查 GPU 数量和有效性**:
   - 如果 GPU 数量为零，或者已附加 GPU 数量不为零，或者参数大小不是 `NvU32` 大小的倍数，则设置错误状态并跳转到 `done`。
3. **为 GPU ID 分配内存并复制**:
   - 为 GPU ID 数组分配内存并从 `arg_copy` 复制数据到这个数组。
4. **处理每个 GPU ID**:
   - 遍历 GPU ID 数组，对每个非零 ID 执行 `nvidia_dev_get` 函数。
   - 如果 `nvidia_dev_get` 失败，执行清理操作：释放已分配的内存并将已附加 GPU 数量设置为零。

### 5.4 NV_ESC_CHECK_VERSION_STR

调用 `rm_perform_version_check` 函数，检查版本是否符合要求

### 5.5 NV_ESC_SYS_PARAMS

设置 NUMA 内存块大小:

- 如果 `nvl->numa_memblock_size` 为 0，设置为 `api->memblock_size`。
- 如果已经设置过，确保提供的大小与已设置的大小一致；不一致则设置错误状态 `-EBUSY` 并跳转到 `done`

### 5.6 NV_ESC_NUMA_INFO

1. **设置 NUMA 信息结构体**:
   - 设置 `api->offline_addresses.numEntries` 为地址数组的大小。
   - 调用 `rm_get_gpu_numa_info` 获取 GPU 的 NUMA 信息。
2. **检查 NUMA 信息获取结果**:
   - 如果 `rmStatus` 不是 `NV_OK`，设置错误状态 `-EBUSY` 并跳转到 `done`。
3. **设置 NUMA 状态和配置**:
   - 设置 NUMA 状态、自动上线配置和内存块大小。

### 5.7 NV_ESC_SET_NUMA_STATUS

处理 NUMA（非一致性内存访问）状态的设置

1. **锁定设备状态**:
   - `down(&nvl->ldata_lock);`：锁定设备状态，防止在修改 NUMA 状态时其他进程对设备进行打开或关闭操作。
2. **检查和设置 NUMA 状态**:
   - `if (nv_get_numa_status(nvl) != api->status)`：如果当前 NUMA 状态与请求的状态不同，根据 `api->status` 的值进行相应操作。
   - 如果请求将 NUMA 状态设置为 `NV_IOCTL_NUMA_STATUS_OFFLINE_IN_PROGRESS`，则检查是否只有一个客户端正在使用设备。如果有多个客户端，则设置错误状态 `-EBUSY`。
   - 如果可以安全地将 NUMA 状态设置为 offline，调用 `rm_gpu_numa_offline`。如果调用失败，设置错误状态 `-EBUSY`。
3. **更改 NUMA 状态**:
   - 调用 `nv_set_numa_status` 以更新 NUMA 状态。如果状态设置失败，尝试恢复原始状态。
4. **处理 NUMA 状态为在线**:
   - 如果新的 NUMA 状态为 `NV_IOCTL_NUMA_STATUS_ONLINE`，调用 `rm_gpu_numa_online` 将设备重新上线。
5. **解锁设备状态**:
   - `up(&nvl->ldata_lock);`：解锁设备状态，允许其他进程进行打开或关闭操作。

### 5.8 NV_ESC_EXPORT_TO_DMABUF_FD

执行 DMA 缓冲区导出操作:

- `params->status = nv_dma_buf_export(nv, params);`：调用 `nv_dma_buf_export` 函数执行 DMA 缓冲区的导出操作，结果状态存储在 `params->status` 中

### 5.9 default

1. **处理未识别的 IOCTL 请求**:
   - `rmStatus = rm_ioctl(sp, nv, &nvlfp->nvfp, arg_cmd, arg_copy, arg_size);`：这里 `rm_ioctl` 可能是一个通用的 IOCTL 处理函数，处理各种未在 switch 语句中显式处理的命令。
2. **设置状态**:
   - `status = ((rmStatus == NV_OK) ? 0 : -EINVAL);`：根据 `rmStatus` 设置操作的最终状态。

## 6 cmd 结束以后

还剩一个 done

cmd 结束后，或者是 nvidia_ioctl 函数执行过程中出错，跳转到的

1. **释放内核栈**:
   - `nv_nvlfp_put_sp(nvlfp, NV_FOPS_STACK_INDEX_IOCTL);`：释放或清理之前获取的内核栈。
2. **释放读锁**:
   - `up_read(&nv_system_pm_lock);`：释放之前获取的读锁。
3. **将结果复制回用户空间**:
   - 如果 `arg_copy` 不为空，且状态不是 `-EFAULT`（复制失败），则尝试将数据从内核空间复制回用户空间。
   - 如果复制失败，打印错误信息并设置错误状态。
4. **释放内存**:
   - `NV_KFREE(arg_copy, arg_size);`：释放之前为 `arg_copy` 分配的内存。
5. **返回状态**:
   - `return status;`：返回处理结果的状态码。