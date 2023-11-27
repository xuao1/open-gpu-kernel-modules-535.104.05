# rm_ioctl

在 /src/nvidia/arch/nvalloc/unix/src，主要是阅读本代码中实现的 `rm_ioctl`

```c
NV_STATUS NV_API_CALL rm_ioctl(
    nvidia_stack_t     *sp,
    nv_state_t         *pNv,
    nv_file_private_t  *nvfp,
    NvU32               Command,
    void               *pData,
    NvU32               dataSize
)
```

## 1 参数

- `nvidia_stack_t *sp`: 指向 NVIDIA 内核栈的指针。
- `nv_state_t *pNv`: 指向 NVIDIA 设备状态的指针。
- `nv_file_private_t *nvfp`: 指向与文件相关的私有数据结构的指针。
- `NvU32 Command`: 表示 IOCTL 命令的整数。即 `arg_cmd`
- `void *pData`: 指向 IOCTL 数据的指针。即 `arg_copy`
- `NvU32 dataSize`: IOCTL 数据的大小。即 `arg_size`

## 2 流程

1. **初始化变量**:
   - `NV_STATUS rmStatus;`: 定义一个 NVIDIA 状态变量 `rmStatus`。
   - `THREAD_STATE_NODE threadState;`: 定义一个线程状态节点 `threadState`。
   - `void *fp;`: 定义一个指针 `fp`。
2. **进入 NVIDIA 运行时**:
   - `NV_ENTER_RM_RUNTIME(sp,fp);`: 这是一个宏，用于设置 NVIDIA 运行时的上下文。
3. **初始化线程状态**:
   - `threadStateInit(&threadState, THREAD_STATE_FLAGS_NONE);`: 初始化线程状态，可能与线程的同步或状态追踪有关。
4. **执行实际的 IOCTL 操作**:
   - `rmStatus = RmIoctl(pNv, nvfp, Command, pData, dataSize);`: 调用 `RmIoctl` 函数执行实际的 IOCTL 操作，传递 NVIDIA 设备状态、文件私有数据、命令、数据和数据大小。
5. **释放线程状态**:
   - `threadStateFree(&threadState, THREAD_STATE_FLAGS_NONE);`: 释放或清理线程状态。
6. **退出 NVIDIA 运行时**:
   - `NV_EXIT_RM_RUNTIME(sp,fp);`: 这是一个宏，用于清理 NVIDIA 运行时的上下文。
7. **返回状态**:
   - `return rmStatus;`: 返回 IOCTL 操作的结果状态