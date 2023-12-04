# 在 resControl_IMPL 中 pEntry 的赋值

在分析宏 `NV_ESC_RM_CONTROL` 的调用逻辑时，分析到了函数 `resCOntrol_IMPL`，这个函数中的核心执行逻辑是调用了 `pEntry->pFunc`，且均为有参数的调用。接下来的重点就是，函数指针 `pFunc` 具体函数是哪个。

直接全局搜目前看起来不现实，因为 `pFunc` 的出现次数比较多。

所以分析在 `resControl_IMPL` 中对 `pEntry` 的赋值

## 1 pEntry 的结构

是结构体 `NVOC_EXPORTED_METHOD_DEF`：

```c
struct NVOC_EXPORTED_METHOD_DEF
{
    void (*pFunc) (void);                         // Pointer to the method itself
    NvU32 flags;                                  // Export flags used for permission, method attribute verification (eg. NO_LOCK, PRIVILEGED...)
    NvU32 accessRight;                            // Access rights required for this method
    NvU32 methodId;                               // Id of the method in the class. Used for method identification.
    NvU32 paramSize;                              // Size of the parameter structure that the method takes as the argument (0 if it takes no arguments)
    const NVOC_CLASS_INFO* pClassInfo;            // Class info for the parent class of the method

#if NV_PRINTF_STRINGS_ALLOWED
    const char  *func;                            // Debug info
#endif
};
```

看起来 `methodId` 可以指示当前 method，尝试输出一下，确实可以