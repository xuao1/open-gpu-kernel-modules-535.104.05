1-3：使用 cuda-gdb，打 log 的地方包括从 nv-frontend.c 沿 ioctl 到 escape.c

4-5：只保留 7

6：只保留 `NV_ESC_RM_CONTROL` 的 7，同时在 `NV_ESC_RM_CONTROL` 内部，分别打 log

7：分析 宏 `NV_ESC_RM_CONTROL` 下一步调用的函数

8：分析 control.c 中的函数 `rmapiControlWithSecInfoTls`

9：分析 control.c 中的函数 `rmapiControlWithSecInfo`

10：分析 control 中的函数 `_rmapiRmControl`

11：分析 control 中的函数 `_rmapiRmControl` 的 else 部分

12：分析 control 中的函数 `_rmapiRmControl` 能否执行到最后

13：分析 re_server.c 中的函数 `serverControl`

14：分析哪些 `__resControl__` 被调用了

15：分析哪些 `__gpuresControl__`  被调用了

16：分析 `resControl_IMPL` 内部执行

17：分析 `resControl_IMPL`  else 的内部执行，以及尝试输出 pFunc

18：分析 `resControl_IMPL`  else 的内部执行，以及尝试输出 pFunc 和 methodId

19：根据 18 的 methodId，在可能的 pFunc 函数入口添加 log

20：在 `_nv83deCtrlCmdDebugAccessMemory` 函数加 log

21：不使用 cuda-gdb 的 log

22：初始运行 `NV_ESC_RM_GET_EVENT_DATA` 和 `NV_ESC_RM_DUP_OBJECT` 的 log

23：不使用 cuda-gdb 的 log

24-31：不使用 cuda-gdb，运行两个代码，区别是有没有 launch kernel

32-39：不使用 cuda-gdb，在 kernel-open/nvidia 的每个函数增加 log，运行两个代码，区别是有没有 launch kernel

40-41：不使用 cuda-gdb，在 kernel-open/nvidia-uvm 的每个函数增加 log，运行两个代码，区别是有没有 launch kernel

42-43：不使用 cuda-gdb，在 kernel-open/nvidia-uvm 的每个函数增加 log，运行两个代码，区别是有没有 memcopy

