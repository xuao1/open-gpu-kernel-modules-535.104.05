# nv-frontend.c

## 1 概述

`nv-frontend.c` 文件是 NVIDIA GPU 驱动程序的一部分，实现了 NVIDIA GPU 驱动程序的前端部分

主要功能是提供一个接口，**允许用户空间程序通过设备文件（如 `/dev/nvidia0`）与 NVIDIA GPU 进行交互**。

通过这个接口，用户空间程序可以执行各种操作，如配置 GPU、执行计算任务等。这个文件的实现确保了 NVIDIA GPU 可以作为一个字符设备在 Linux 系统中使用，从而允许用户空间程序通过标准的文件操作与 GPU 交互。

功能：

1. **模块管理**: 代码提供了 NVIDIA 设备驱动模块的注册、注销、添加设备和删除设备的功能。
2. **字符设备操作**: 实现了字符设备的标准操作，如打开、关闭、IO 控制等，以便用户空间程序可以通过设备文件与驱动程序交互。

## 2 概念区分

#### Instance（实例）

- **含义**：`instance` 通常指的是驱动程序的一个特定实例。在多 GPU 系统中，每个 GPU 可能对应一个不同的驱动程序实例。
- **作用**：它用于区分系统中的不同 GPU 设备。例如，如果有多个 NVIDIA GPU，每个 GPU 将由一个单独的实例表示。

#### Device（设备）

- **含义**：`device` 通常指的是一个具体的硬件设备，在这个上下文中，它指的是一个具体的 NVIDIA GPU。
- **作用**：`device` 结构体通常包含了与该 GPU 相关的信息，如设备状态、硬件参数等。它是驱动程序与具体硬件交互的基础。

#### Module（模块）

- **含义**：`module` 在这里指的是加载到内核中的驱动程序模块。在 Linux 中，模块是可以动态加载和卸载的内核代码片段。
- **作用**：`module` 提供了驱动程序的功能实现。它可能包含了一系列的函数和数据结构，用于与硬件设备进行交互。

#### 关系和联系

- **实例与设备**：每个 `instance` 通常与一个 `device` 相关联。在多 GPU 系统中，每个 GPU 设备（`device`）由一个单独的驱动程序实例（`instance`）管理。
- **模块与实例**：`module` 可以被视为实现一个或多个 `instance` 的代码和数据结构的集合。在多实例（多GPU）的情况下，每个 `instance` 可能共享相同的 `module`，但代表不同的物理设备。
- **模块与设备**：`module` 通过其提供的接口和功能与具体的 `device` 进行交互，实现对设备的控制和管理。

## 3 字符设备操作的实现

写在前面：

基本上都是调用 module 对应的操作，获取 module 的指针是通过：

```c
NvU32 minor_num = NV_FRONTEND_MINOR_NUMBER(inode);
module = nv_minor_num_table[minor_num];
```

即通过 inode.

定义一系列函数来处理操作系统和用户程序对字符设备文件的操作请求，包括：

#### nvidia_frontend_open(struct inode *inode, struct file *file)

- **功能**：当用户程序尝试打开与 NVIDIA 设备相关联的设备文件时（例如 `/dev/nvidia0`），此函数被调用。
- 参数：
  - `struct inode *inode`: 表示文件系统中的节点，包含了与设备文件相关的元数据。
  - `struct file *file`: 表示一个打开的文件描述符，包含了文件操作的状态。
- 操作：
  - 检查与给定设备文件关联的 module（如 `nvidia_module_t`），并调用其 `open` 方法。
  - 增加 module 的引用计数，防止在文件打开时 module 被卸载。

#### nvidia_frontend_close(struct inode *inode, struct file *file)

- **功能**：当用户程序关闭设备文件时调用。
- 操作：
  - 调用与设备文件关联的 module 的 `close` 方法。
  - 减少 module 的引用计数。

#### nvidia_frontend_poll(struct file \*file, poll_table *wait)

- **功能**：实现轮询接口，允许用户程序检查设备的状态
- 操作：
  - 调用相应 module 的 `poll` 方法来获取设备状态。

#### nvidia_frontend_ioctl(struct inode *inode, struct file *file, unsigned int cmd, unsigned long i_arg)

- **功能**：处理 I/O 控制请求，用于执行特定于设备的操作。
- 操作：
  - 根据提供的命令和参数调用相应 module 的 `ioctl` 方法。

#### nvidia_frontend_unlocked_ioctl(struct file *file, unsigned int cmd, unsigned long i_arg)

- **功能**：是 `ioctl` 的一个变种，用于处理从用户空间传递到内核空间的命令。
- 操作：
  - 调用 `nvidia_frontend_ioctl` 来实际处理请求。

#### nvidia_frontend_compat_ioctl(struct file *file, unsigned int cmd, unsigned long i_arg)

- **功能**：处理兼容模式的 I/O 控制请求，主要用于在 64 位系统上支持 32 位应用程序。
- 操作：
  - 同样通过调用 `nvidia_frontend_ioctl` 来处理兼容模式的请求。

#### nvidia_frontend_mmap(struct file *file, struct vm_area_struct *vma)

- **功能**：允许用户程序将设备内存映射到其自己的地址空间，常用于高效数据传输。
- 操作：
  - 调用相关模块的 `mmap` 方法来设置内存映射。