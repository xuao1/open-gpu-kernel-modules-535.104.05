被注释掉的，包括会被后台反复执行的函数，以及两个轮询代码

### 450	nvidia_frontend_poll

对 Nvidia 设备文件进行轮询操作

主要是调用 `mask = module->poll(file, wait)`

### 716	nvidia_poll

对 Nvidia 设备文件进行轮询操作

检查 event 队列中是否有任何事件（event_data_head）或是否有待处理的无数据事件。如果有，它将 mask 设置为 POLLPRI | POLLIN，表示有数据可读或有高优先级事件，还会重置 dataless_event_pending 标志。

### 727	nvidia_rc_timer_callback

定时器回调函数，旨在定期检查 Nvidia GPU 的状态，并根据该状态可能执行某些操作

如果 GPU 处于有效状态，它调用 `rm_run_rc_callback`，这很可能是运行与 GPU 相关的某个资源控制（RC）回调的函数

如果 `rm_run_rc_callback` 返回 NV_OK，函数会重新调度定时器，在一定时间后再次运行（1秒）

### 882	os_semaphore_may_sleep

检查当前执行上下文是否允许睡眠操作

### 895	os_mem_set

memset 函数的自定义实现。它包含逻辑，用于处理特定架构上不同类型的内存

### 899	os_get_current_time

获取当前系统时间

### 901	os_get_current_tick

检索当前系统的滴答数并以纳秒为单位返回

### 910	os_get_current_thread

检索当前线程的标识并将其返回，如果在中断上下文中，则函数将 threadId 设置为 0

### 928	os_flush_cpu_write_combine_buffer

刷新 CPU 的写组合缓冲区

写组合是一种用于 CPU 内存访问的性能优化技术。启用写组合时，CPU 将连续内存地址的多个写操作组合成单个较大的写操作。这可以在写入某些类型的内存时显着提高性能

### 937	os_get_cpu_number

获取当前执行代码的 CPU 的标识符

### 946	os_acquire_spinlock

获取自旋锁并返回在获取时中断标志（IF）的状态

### 947	os_release_spinlock

释放先前获取的自旋锁并将中断标志（IF）恢复到其原始状态