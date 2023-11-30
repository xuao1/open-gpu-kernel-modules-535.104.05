运行 launchkernel.cu 代码，只运行 testkernel，因为里面的 cudalaunch 会报错

在 cuda-gdb 中加入断点，只截取第一次运行 continue 的 log

多次截取，目的是查看 ioctrl 调用次数是不是稳定

1-5 是 testKernel 的

6-8 是 Matrix 的前几次运行（此时需要多次 continue 才能执行完，保存的是一次运行的多次 continue）

之后减小了 Matrix 的 block num，此时可以一次 continue 就执行完，多次运行，保存为 9-11