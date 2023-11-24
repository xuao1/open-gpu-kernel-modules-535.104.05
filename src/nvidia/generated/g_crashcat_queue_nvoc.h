#ifndef _G_CRASHCAT_QUEUE_NVOC_H_
#define _G_CRASHCAT_QUEUE_NVOC_H_
#include "nvoc/runtime.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "g_crashcat_queue_nvoc.h"

#ifndef CRASHCAT_QUEUE_H
#define CRASHCAT_QUEUE_H

#include "nvoc/object.h"
#include "nv-crashcat.h"
#include "crashcat/crashcat_wayfinder.h" // for CrashCatWayfinderHal spec

struct CrashCatEngine;

#ifndef __NVOC_CLASS_CrashCatEngine_TYPEDEF__
#define __NVOC_CLASS_CrashCatEngine_TYPEDEF__
typedef struct CrashCatEngine CrashCatEngine;
#endif /* __NVOC_CLASS_CrashCatEngine_TYPEDEF__ */

#ifndef __nvoc_class_id_CrashCatEngine
#define __nvoc_class_id_CrashCatEngine 0x654166
#endif /* __nvoc_class_id_CrashCatEngine */


struct CrashCatReport;

#ifndef __NVOC_CLASS_CrashCatReport_TYPEDEF__
#define __NVOC_CLASS_CrashCatReport_TYPEDEF__
typedef struct CrashCatReport CrashCatReport;
#endif /* __NVOC_CLASS_CrashCatReport_TYPEDEF__ */

#ifndef __nvoc_class_id_CrashCatReport
#define __nvoc_class_id_CrashCatReport 0xde4777
#endif /* __nvoc_class_id_CrashCatReport */



typedef struct
{
    NV_CRASHCAT_MEM_APERTURE aperture;
    NvU32 size;
    NvU64 offset;

    NvU32 putRegOffset;
    NvU32 getRegOffset;
} CrashCatQueueConfig;

#ifdef NVOC_CRASHCAT_QUEUE_H_PRIVATE_ACCESS_ALLOWED
#define PRIVATE_FIELD(x) x
#else
#define PRIVATE_FIELD(x) NVOC_PRIVATE_FIELD(x)
#endif
struct CrashCatQueue {
    const struct NVOC_RTTI *__nvoc_rtti;
    struct Object __nvoc_base_Object;
    struct Object *__nvoc_pbase_Object;
    struct CrashCatQueue *__nvoc_pbase_CrashCatQueue;
    CrashCatQueueConfig PRIVATE_FIELD(config);
    struct CrashCatEngine *PRIVATE_FIELD(pEngine);
    void *PRIVATE_FIELD(pMapping);
};

#ifndef __NVOC_CLASS_CrashCatQueue_TYPEDEF__
#define __NVOC_CLASS_CrashCatQueue_TYPEDEF__
typedef struct CrashCatQueue CrashCatQueue;
#endif /* __NVOC_CLASS_CrashCatQueue_TYPEDEF__ */

#ifndef __nvoc_class_id_CrashCatQueue
#define __nvoc_class_id_CrashCatQueue 0xbaa900
#endif /* __nvoc_class_id_CrashCatQueue */

extern const struct NVOC_CLASS_DEF __nvoc_class_def_CrashCatQueue;

#define __staticCast_CrashCatQueue(pThis) \
    ((pThis)->__nvoc_pbase_CrashCatQueue)

#ifdef __nvoc_crashcat_queue_h_disabled
#define __dynamicCast_CrashCatQueue(pThis) ((CrashCatQueue*)NULL)
#else //__nvoc_crashcat_queue_h_disabled
#define __dynamicCast_CrashCatQueue(pThis) \
    ((CrashCatQueue*)__nvoc_dynamicCast(staticCast((pThis), Dynamic), classInfo(CrashCatQueue)))
#endif //__nvoc_crashcat_queue_h_disabled


NV_STATUS __nvoc_objCreateDynamic_CrashCatQueue(CrashCatQueue**, Dynamic*, NvU32, va_list);

NV_STATUS __nvoc_objCreate_CrashCatQueue(CrashCatQueue**, Dynamic*, NvU32, CrashCatQueueConfig * arg_pQueueConfig);
#define __objCreate_CrashCatQueue(ppNewObj, pParent, createFlags, arg_pQueueConfig) \
    __nvoc_objCreate_CrashCatQueue((ppNewObj), staticCast((pParent), Dynamic), (createFlags), arg_pQueueConfig)

struct CrashCatReport *crashcatQueueConsumeNextReport_V1(struct CrashCatQueue *arg0);


#ifdef __nvoc_crashcat_queue_h_disabled
static inline struct CrashCatReport *crashcatQueueConsumeNextReport(struct CrashCatQueue *arg0) {
    NV_ASSERT_FAILED_PRECOMP("CrashCatQueue was disabled!");
    return NULL;
}
#else //__nvoc_crashcat_queue_h_disabled
#define crashcatQueueConsumeNextReport(arg0) crashcatQueueConsumeNextReport_V1(arg0)
#endif //__nvoc_crashcat_queue_h_disabled

#define crashcatQueueConsumeNextReport_HAL(arg0) crashcatQueueConsumeNextReport(arg0)

NV_STATUS crashcatQueueConstruct_IMPL(struct CrashCatQueue *arg_, CrashCatQueueConfig *arg_pQueueConfig);

#define __nvoc_crashcatQueueConstruct(arg_, arg_pQueueConfig) crashcatQueueConstruct_IMPL(arg_, arg_pQueueConfig)
void crashcatQueueDestruct_IMPL(struct CrashCatQueue *arg0);

#define __nvoc_crashcatQueueDestruct(arg0) crashcatQueueDestruct_IMPL(arg0)
#undef PRIVATE_FIELD


#endif // CRASHCAT_QUEUE_H

#ifdef __cplusplus
} // extern "C"
#endif
#endif // _G_CRASHCAT_QUEUE_NVOC_H_
