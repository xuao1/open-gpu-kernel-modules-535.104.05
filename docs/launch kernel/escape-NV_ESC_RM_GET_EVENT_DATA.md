# NV_ESC_RM_GET_EVENT_DATA

escape.c 中 cmd 进入的宏，testkernel 的 launch kernel 时共执行了 62 次 ioctl，这个宏进入了 3 次

## 0 执行路径

直接运行 testkernel，不使用 cuda-gdb，没有这个宏的输出

## 1 escape.c

核心是调用了 `rm_get_event_data`
