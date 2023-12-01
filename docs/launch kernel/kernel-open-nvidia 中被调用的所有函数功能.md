testkernel 在 launch kernel 时，kernel-open/nvidia 目录下的所有被调用过的函数（不含后台反复调用的函数，不含两个轮询函数）

总调用次数 5801

执行次数相同的有：

不按调用先后顺寻，标题名依次为 log 序号、函数名、调用次数

### 216	getUvmEvents(void)	7

返回一个 UvmOpsUvmEvents 类型的结构体，应该是与 UVM 相关的

### 266	nvUvmInterfaceHasPendingNonReplayableFaults	7

用于检查 GPU 中是否存在待处理的非可重放故障（pending non-replayable faults in a GPU）

### 278	nv_uvm_event_interrupt	7

会调用 `getUvmEvents` 获取一个 event，如果 event->isrTopHalf 回调不存在，函数将返回 NV_ERR_NO_INTR_PENDING，表示没有相关的中断待处理。如果回调存在并被执行，函数将返回回调返回值。

### 451	nvidia_frontend_ioctl	62

ioctl 相关

### 452	nvidia_frontend_unlocked_ioctl	62

ioctl 相关，会调用上面的 `nvidia_frontend_ioctl`

### 670	nv_vm_map_pages	15

进行初始检查，然后调用 `nv_vmap`，可能是将页面实际映射到虚拟内存中

返回值为 `nv_vmap` 的结果，是虚拟地址（NvUPtr）

### 671	nv_vm_unmap_pages	15

进行初始检查，然后使用虚拟地址 virt_addr 和计数 count 调用 `nv_vunmap`，可能是从给定虚拟地址开始的指定页数的取消映射

### 720	nvidia_ioctl	62

ioctl 相关

### 721	nvidia_isr_msix	7

处理 MSI-X（Message Signaled Interrupts Extended）中断

使用了自旋锁，中间调用了 nvidia_isr，确保了 nvidia_isr(irq, arg) 的执行在不同的 CPU 核上是序列化的

### 722	nvidia_isr	7

处理与 Nvidia GPU 相关的中断的中断服务例程（ISR）

处理 MMU（内存管理单元）故障，与 UVM 驱动程序交互，管理常规中断处理，跟踪和记录未处理的中断，并根据需要安排进一步处理

### 724	nvidia_isr_msix_kthread_bh	7

会调用 `nvidia_isr_common_bh`

处理 NVIDIA GPU 驱动程序中的 MSI-X（消息信号中断扩展）中断的中断服务例程。

它确保对共享资源的同步访问，会调用 `os_acquire_mutex` 和 `os_release_mutex`

### 725	nvidia_isr_common_bh	7

用作中断服务例程（ISR）的通用底半处理程序

检查 GPU 的状态，如果 GPU 正常运行，调用 rm_isr_bh(sp, nv) 进行进一步处理。这表明 rm_isr_bh 负责实际处理中断的底半部分

将中断处理分为顶半部分和底半部分有助于管理 Linux 内核中中断处理的时间敏感性

### 743	nv_alloc_kernel_mapping	15

负责分配和映射 kernel 内存页，特别是处理不同类型的内存（如用户分配的或 guest 内存），并确保它们在 kernel 的虚拟地址空间中正确映射

可能会调用 `nv_vm_map_pages`，或者 `nv_map_guest_pages`

### 744	nv_free_kernel_mapping	15

负责释放 NVIDIA GPU 驱动程序中先前映射的内核内存。

它处理不同类型的内存分配（如客户端内存），并确保内存正确取消映射并从内核的虚拟地址空间中释放

可能会调用 `nv_iounmap` 或者 `nv_vm_unmap_pages`

### 749	nv_post_event	8

事件处理机制的一部分，处理发布和排队事件，并通知系统的其他部分有关这些事件

如果 data_valid 为 true，它为新的 nvidia_event_t 结构分配内存，然后将此事件添加到链表的尾部。这个链表似乎用于跟踪与 NVIDIA GPU 相关的事件。 然后，它使用提供的参数（event、handle、index、info32、info16）填充此事件结构。 

如果 data_valid 为 false，它会设置一个标志（dataless_event_pending），以指示挂起了一个没有数据的事件。这可能用于向驱动程序的其他部分发出事件已发生的信号，但它不携带额外的数据。

之后，会调用 `wake_up_interruptible(&nvlfp->waitqueue)`

### 755	nv_get_event	3

从与特定文件私有数据结构（nv_linux_file_private_t）关联的事件队列中检索事件

从队头取一个事件，设置 pending 参数以指示队列中是否有更多的事件

取到的事件通过参数 event 返回

### 782	nv_get_ctl_state	16

返回 Nvidia 控制设备的状态指针，可能用于在驱动程序中管理或查询设备的状态

### 866	os_acquire_mutex	7

获取互斥锁

down 函数是 Linux 内核中用于获取信号量或互斥锁的函数。如果互斥锁已被锁定，调用线程将阻塞，直到互斥锁可用.

### 868	os_release_mutex	7

释放先前获取的互斥锁

up 函数是 Linux 内核中用于释放信号量或互斥锁的函数。当调用此函数时，如果有任何线程因等待该互斥锁而被阻塞，则其中一个将被唤醒并允许获取互斥锁。

### 876	os_acquire_rwlock_read	20

获取读写锁（rwlock）上的读锁

down_read 函数是 Linux 内核中用于获取读锁的函数，多个读取者可以同时持有锁，但如果写入者持有锁，则它们将被阻塞

### 877	os_acquire_rwlock_write	195

获取读写锁（rwlock）上的写锁

### 880	os_release_rwlock_read	20

释放先前获取的读写锁上的读锁

### 881	os_release_rwlock_write	195

释放先前获取的读写锁上的写锁

### 883	os_is_isr	2879

检查当前的执行上下文是否是中断服务例程（ISR）。确定当前代码是否在响应硬件中断。

通过调用 `in_irq()` 函数来实现这一点。在 Linux 内核编程中，`in_irq()` 是一个内核函数，用于检查当前代码是否在中断上下文中运行。

### 884	os_is_administrator	62

检查当前用户是否具有管理员（或超级用户）特权

### 892	os_mem_copy	274

是一个自定义的内存复制函数，处理各种情况和优化。

它考虑了系统架构和内核配置，以选择最适合复制内存的方法

### 893	os_memcpy_from_user	68

安全地从用户空间复制数据到内核空间

### 894	os_memcpy_to_user	73

安全地从内核空间复制数据到用户空间

### 897	os_alloc_mem	657

在内核空间中分配内存，考虑了 Nvidia 驱动程序的资源管理器（RM）特定的约束和要求

它根据分配大小和当前上下文（系统是否可以休眠）智能地在 kmalloc 和 vmalloc 之间做出选择

### 898	os_free_mem	645

释放先前在内核空间分配的内存

### 902	os_get_tick_resolution	130

返回系统滴答的分辨率（以纳秒为单位）

### 908	os_get_current_process	70

返回当前进程的标识符

### 912	nv_printf	62

输出消息到内核日志

### 922	xen_support_fully_virtualized_kernel	20

主要目的是确定当前的内核环境是否在完全虚拟化的设置下运行，特别是在 Xen 虚拟机监控程序或类似技术的上下文中

### 923	os_map_kernel_space	20

会调用 `xen_support_fully_virtualized_kernel()`

将指定的物理内存范围映射到内核的虚拟地址空间

根据模式参数，函数选择适当的映射函数

### 924	os_unmap_kernel_space	20

从内核的虚拟地址空间中取消映射先前映射的一段物理内存范围

### 950	os_is_vgx_hyper	55

确定当前系统是否运行在特定的虚拟化环境下，特别是由 NV_VGX_HYPER 定义的环境
