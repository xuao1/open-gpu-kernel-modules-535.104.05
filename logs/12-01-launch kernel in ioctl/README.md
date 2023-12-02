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
