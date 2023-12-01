1-3：使用 cuda-gdb，打 log 的地方包括从 nv-frontend.c 沿 ioctl 到 escape.c

4-5：只保留 7

6：只保留 `NV_ESC_RM_CONTROL` 的 7，同时在 `NV_ESC_RM_CONTROL` 内部，分别打 log
