1. ~~查看被注释的函数的具体功能~~
2. ~~将 1.log 中的数字替换为实际的函数名，阅读执行过程~~
3. 追踪 ioctl
   1. ~~追踪到 escape.c~~
   2. 追踪宏 `NV_ESC_RM_CONTROL`
   3. 追踪宏 `NV_ESC_RM_GET_EVENT_DATA`
   5. 追踪宏 `NV_ESC_RM_DUP_OBJECT`
4. 追踪 event 相关的函数