运行 launchkernel.cu 代码，只运行 testkernel，因为里面的 cudalaunch 会报错

在 cuda-gdb 中加入断点，只截取第一次运行 continue 的 log

多次截取，目的是查看 ioctrl 调用次数是不是稳定