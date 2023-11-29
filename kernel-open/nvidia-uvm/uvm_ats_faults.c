/*******************************************************************************
    Copyright (c) 2018 NVIDIA Corporation

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to
    deal in the Software without restriction, including without limitation the
    rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
    sell copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

        The above copyright notice and this permission notice shall be
        included in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
    THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE SOFTWARE.
*******************************************************************************/

#include "uvm_api.h"
#include "uvm_tools.h"
#include "uvm_va_range.h"
#include "uvm_ats.h"
#include "uvm_ats_faults.h"
#include "uvm_migrate_pageable.h"
#include <linux/nodemask.h>
#include <linux/mempolicy.h>
#include <linux/mmu_notifier.h>

#if UVM_ATS_PREFETCH_SUPPORTED()
#include <linux/hmm.h>
#endif

static NV_STATUS service_ats_faults(uvm_gpu_va_space_t *gpu_va_space,
                                    struct vm_area_struct *vma,
                                    NvU64 start,
                                    size_t length,
                                    uvm_fault_access_type_t access_type,
                                    uvm_ats_fault_context_t *ats_context)
{
    printk(KERN_ERR "=====================================   %d\n", 108);
    uvm_va_space_t *va_space = gpu_va_space->va_space;
    struct mm_struct *mm = va_space->va_space_mm.mm;
    bool write = (access_type >= UVM_FAULT_ACCESS_TYPE_WRITE);
    NV_STATUS status;
    NvU64 user_space_start;
    NvU64 user_space_length;

    // Request uvm_migrate_pageable() to touch the corresponding page after
    // population.
    // Under virtualization ATS provides two translations:
    // 1) guest virtual -> guest physical
    // 2) guest physical -> host physical
    //
    // The overall ATS translation will fault if either of those translations is
    // invalid. The pin_user_pages() call within uvm_migrate_pageable() call
    // below handles translation #1, but not #2. We don't know if we're running
    // as a guest, but in case we are we can force that translation to be valid
    // by touching the guest physical address from the CPU. If the translation
    // is not valid then the access will cause a hypervisor fault. Note that
    // dma_map_page() can't establish mappings used by GPU ATS SVA translations.
    // GPU accesses to host physical addresses obtained as a result of the
    // address translation request uses the CPU address space instead of the
    // IOMMU address space since the translated host physical address isn't
    // necessarily an IOMMU address. The only way to establish guest physical to
    // host physical mapping in the CPU address space is to touch the page from
    // the CPU.
    //
    // We assume that the hypervisor mappings are all VM_PFNMAP, VM_SHARED, and
    // VM_WRITE, meaning that the mappings are all granted write access on any
    // fault and that the kernel will never revoke them.
    // drivers/vfio/pci/vfio_pci_nvlink2.c enforces this. Thus we can assume
    // that a read fault is always sufficient to also enable write access on the
    // guest translation.

    uvm_migrate_args_t uvm_migrate_args =
    {
        .va_space                       = va_space,
        .mm                             = mm,
        .dst_id                         = ats_context->residency_id,
        .dst_node_id                    = ats_context->residency_node,
        .start                          = start,
        .length                         = length,
        .populate_permissions           = write ? UVM_POPULATE_PERMISSIONS_WRITE : UVM_POPULATE_PERMISSIONS_ANY,
        .touch                          = true,
        .skip_mapped                    = true,
        .populate_on_cpu_alloc_failures = true,
        .user_space_start               = &user_space_start,
        .user_space_length              = &user_space_length,
    };

    UVM_ASSERT(uvm_ats_can_service_faults(gpu_va_space, mm));

    // We are trying to use migrate_vma API in the kernel (if it exists) to
    // populate and map the faulting region on the GPU. We want to do this only
    // on the first touch. That is, pages which are not already mapped. So, we
    // set skip_mapped to true. For pages already mapped, this will only handle
    // PTE upgrades if needed.
    status = uvm_migrate_pageable(&uvm_migrate_args);
    if (status == NV_WARN_NOTHING_TO_DO)
        status = NV_OK;

    UVM_ASSERT(status != NV_ERR_MORE_PROCESSING_REQUIRED);

    return status;
}

static void flush_tlb_write_faults(uvm_gpu_va_space_t *gpu_va_space,
                                   NvU64 addr,
                                   size_t size,
                                   uvm_fault_client_type_t client_type)
{
    printk(KERN_ERR "=====================================   %d\n", 109);
    uvm_ats_fault_invalidate_t *ats_invalidate;

    if (client_type == UVM_FAULT_CLIENT_TYPE_GPC)
        ats_invalidate = &gpu_va_space->gpu->parent->fault_buffer_info.replayable.ats_invalidate;
    else
        ats_invalidate = &gpu_va_space->gpu->parent->fault_buffer_info.non_replayable.ats_invalidate;

    if (!ats_invalidate->write_faults_in_batch) {
        uvm_tlb_batch_begin(&gpu_va_space->page_tables, &ats_invalidate->write_faults_tlb_batch);
        ats_invalidate->write_faults_in_batch = true;
    }

    uvm_tlb_batch_invalidate(&ats_invalidate->write_faults_tlb_batch, addr, size, PAGE_SIZE, UVM_MEMBAR_NONE);
}

static void ats_batch_select_residency(uvm_gpu_va_space_t *gpu_va_space,
                                       struct vm_area_struct *vma,
                                       uvm_ats_fault_context_t *ats_context)
{
    printk(KERN_ERR "=====================================   %d\n", 110);
    uvm_gpu_t *gpu = gpu_va_space->gpu;
    int residency = uvm_gpu_numa_node(gpu);

#if defined(NV_MEMPOLICY_HAS_UNIFIED_NODES)
    struct mempolicy *vma_policy = vma_policy(vma);
    unsigned short mode;

    ats_context->prefetch_state.has_preferred_location = false;

    // It's safe to read vma_policy since the mmap_lock is held in at least read
    // mode in this path.
    uvm_assert_mmap_lock_locked(vma->vm_mm);

    if (!vma_policy)
        goto done;

    mode = vma_policy->mode;

    if ((mode == MPOL_BIND) || (mode == MPOL_PREFERRED_MANY) || (mode == MPOL_PREFERRED)) {
        int home_node = NUMA_NO_NODE;

#if defined(NV_MEMPOLICY_HAS_HOME_NODE)
        if ((mode != MPOL_PREFERRED) && (vma_policy->home_node != NUMA_NO_NODE))
            home_node = vma_policy->home_node;
#endif

        // Prefer home_node if set. Otherwise, prefer the faulting GPU if it's
        // in the list of preferred nodes, else prefer the closest_cpu_numa_node
        // to the GPU if closest_cpu_numa_node is in the list of preferred
        // nodes. Fallback to the faulting GPU if all else fails.
        if (home_node != NUMA_NO_NODE) {
            residency = home_node;
        }
        else if (!node_isset(residency, vma_policy->nodes)) {
            int closest_cpu_numa_node = gpu->parent->closest_cpu_numa_node;

            if ((closest_cpu_numa_node != NUMA_NO_NODE) && node_isset(closest_cpu_numa_node, vma_policy->nodes))
                residency = gpu->parent->closest_cpu_numa_node;
            else
                residency = first_node(vma_policy->nodes);
        }

        if (!nodes_empty(vma_policy->nodes))
            ats_context->prefetch_state.has_preferred_location = true;
    }

    // Update gpu if residency is not the faulting gpu.
    if (residency != uvm_gpu_numa_node(gpu))
        gpu = uvm_va_space_find_gpu_with_memory_node_id(gpu_va_space->va_space, residency);

done:
#else
    ats_context->prefetch_state.has_preferred_location = false;
#endif

    ats_context->residency_id = gpu ? gpu->parent->id : UVM_ID_CPU;
    ats_context->residency_node = residency;
}

static void get_range_in_vma(struct vm_area_struct *vma, NvU64 base, NvU64 *start, NvU64 *end)
{
    printk(KERN_ERR "=====================================   %d\n", 111);
    *start = max(vma->vm_start, (unsigned long) base);
    *end = min(vma->vm_end, (unsigned long) (base + UVM_VA_BLOCK_SIZE));
}

static uvm_page_index_t uvm_ats_cpu_page_index(NvU64 base, NvU64 addr)
{
    printk(KERN_ERR "=====================================   %d\n", 112);
    UVM_ASSERT(addr >= base);
    UVM_ASSERT(addr <= (base + UVM_VA_BLOCK_SIZE));

    return (addr - base) / PAGE_SIZE;
}

// start and end must be aligned to PAGE_SIZE and must fall within
// [base, base + UVM_VA_BLOCK_SIZE]
static uvm_va_block_region_t uvm_ats_region_from_start_end(NvU64 start, NvU64 end)
{
    printk(KERN_ERR "=====================================   %d\n", 113);
    // base can be greater than, less than or equal to the start of a VMA.
    NvU64 base = UVM_VA_BLOCK_ALIGN_DOWN(start);

    UVM_ASSERT(start < end);
    UVM_ASSERT(PAGE_ALIGNED(start));
    UVM_ASSERT(PAGE_ALIGNED(end));
    UVM_ASSERT(IS_ALIGNED(base, UVM_VA_BLOCK_SIZE));

    return uvm_va_block_region(uvm_ats_cpu_page_index(base, start), uvm_ats_cpu_page_index(base, end));
}

static uvm_va_block_region_t uvm_ats_region_from_vma(struct vm_area_struct *vma, NvU64 base)
{
    printk(KERN_ERR "=====================================   %d\n", 114);
    NvU64 start;
    NvU64 end;

    get_range_in_vma(vma, base, &start, &end);

    return uvm_ats_region_from_start_end(start, end);
}

#if UVM_ATS_PREFETCH_SUPPORTED()

static bool uvm_ats_invalidate_notifier(struct mmu_interval_notifier *mni, unsigned long cur_seq)
{
    printk(KERN_ERR "=====================================   %d\n", 115);
    uvm_ats_fault_context_t *ats_context = container_of(mni, uvm_ats_fault_context_t, prefetch_state.notifier);
    uvm_va_space_t *va_space = ats_context->prefetch_state.va_space;

    // The following write lock protects against concurrent invalidates while
    // hmm_range_fault() is being called in ats_compute_residency_mask().
    uvm_down_write(&va_space->ats.lock);

    mmu_interval_set_seq(mni, cur_seq);

    uvm_up_write(&va_space->ats.lock);

    return true;
}

static bool uvm_ats_invalidate_notifier_entry(struct mmu_interval_notifier *mni,
                                              const struct mmu_notifier_range *range,
                                              unsigned long cur_seq)
{
    printk(KERN_ERR "=====================================   %d\n", 116);
    UVM_ENTRY_RET(uvm_ats_invalidate_notifier(mni, cur_seq));
}

static const struct mmu_interval_notifier_ops uvm_ats_notifier_ops =
{
    .invalidate = uvm_ats_invalidate_notifier_entry,
};

#endif

static NV_STATUS ats_compute_residency_mask(uvm_gpu_va_space_t *gpu_va_space,
                                            struct vm_area_struct *vma,
                                            NvU64 base,
                                            uvm_ats_fault_context_t *ats_context)
{
    printk(KERN_ERR "=====================================   %d\n", 117);
    NV_STATUS status = NV_OK;

#if UVM_ATS_PREFETCH_SUPPORTED()
    int ret;
    NvU64 start;
    NvU64 end;
    uvm_page_mask_t *residency_mask = &ats_context->prefetch_state.residency_mask;
    struct hmm_range range;
    uvm_page_index_t page_index;
    uvm_va_block_region_t vma_region;
    uvm_va_space_t *va_space = gpu_va_space->va_space;
    struct mm_struct *mm = va_space->va_space_mm.mm;

    uvm_assert_rwsem_locked_read(&va_space->lock);

    ats_context->prefetch_state.first_touch = true;

    uvm_page_mask_zero(residency_mask);

    get_range_in_vma(vma, base, &start, &end);

    vma_region = uvm_ats_region_from_start_end(start, end);

    range.notifier = &ats_context->prefetch_state.notifier;
    range.start = start;
    range.end = end;
    range.hmm_pfns = ats_context->prefetch_state.pfns;
    range.default_flags = 0;
    range.pfn_flags_mask = 0;
    range.dev_private_owner = NULL;

    ats_context->prefetch_state.va_space = va_space;

    // mmu_interval_notifier_insert() will try to acquire mmap_lock for write
    // and will deadlock since mmap_lock is already held for read in this path.
    // This is prevented by calling __mmu_notifier_register() during va_space
    // creation. See the comment in uvm_mmu_notifier_register() for more
    // details.
    ret = mmu_interval_notifier_insert(range.notifier, mm, start, end, &uvm_ats_notifier_ops);
    if (ret)
        return errno_to_nv_status(ret);

    while (true) {
        range.notifier_seq = mmu_interval_read_begin(range.notifier);
        ret = hmm_range_fault(&range);
        if (ret == -EBUSY)
            continue;
        if (ret) {
            status = errno_to_nv_status(ret);
            UVM_ASSERT(status != NV_OK);
            break;
        }

        uvm_down_read(&va_space->ats.lock);

        // Pages may have been freed or re-allocated after hmm_range_fault() is
        // called. So the PTE might point to a different page or nothing. In the
        // memory hot-unplug case it is not safe to call page_to_nid() on the
        // page as the struct page itself may have been freed. To protect
        // against these cases, uvm_ats_invalidate_entry() blocks on va_space
        // ATS write lock for concurrent invalidates since va_space ATS lock is
        // held for read in this path.
        if (!mmu_interval_read_retry(range.notifier, range.notifier_seq))
            break;

        uvm_up_read(&va_space->ats.lock);
    }

    if (status == NV_OK) {
        for_each_va_block_page_in_region(page_index, vma_region) {
            unsigned long pfn = ats_context->prefetch_state.pfns[page_index - vma_region.first];

            if (pfn & HMM_PFN_VALID) {
                struct page *page = hmm_pfn_to_page(pfn);

                if (page_to_nid(page) == ats_context->residency_node)
                    uvm_page_mask_set(residency_mask, page_index);

                ats_context->prefetch_state.first_touch = false;
            }
        }

        uvm_up_read(&va_space->ats.lock);
    }

    mmu_interval_notifier_remove(range.notifier);

#endif

    return status;
}

static void ats_expand_fault_region(uvm_gpu_va_space_t *gpu_va_space,
                                    struct vm_area_struct *vma,
                                    uvm_ats_fault_context_t *ats_context,
                                    uvm_va_block_region_t max_prefetch_region,
                                    uvm_page_mask_t *faulted_mask)
{
    printk(KERN_ERR "=====================================   %d\n", 118);
    uvm_page_mask_t *read_fault_mask = &ats_context->read_fault_mask;
    uvm_page_mask_t *write_fault_mask = &ats_context->write_fault_mask;
    uvm_page_mask_t *residency_mask = &ats_context->prefetch_state.residency_mask;
    uvm_page_mask_t *prefetch_mask = &ats_context->prefetch_state.prefetch_pages_mask;
    uvm_perf_prefetch_bitmap_tree_t *bitmap_tree = &ats_context->prefetch_state.bitmap_tree;

    if (uvm_page_mask_empty(faulted_mask))
        return;

    uvm_perf_prefetch_compute_ats(gpu_va_space->va_space,
                                  faulted_mask,
                                  uvm_va_block_region_from_mask(NULL, faulted_mask),
                                  max_prefetch_region,
                                  residency_mask,
                                  bitmap_tree,
                                  prefetch_mask);

    uvm_page_mask_or(read_fault_mask, read_fault_mask, prefetch_mask);

    if (vma->vm_flags & VM_WRITE)
        uvm_page_mask_or(write_fault_mask, write_fault_mask, prefetch_mask);
}

static NV_STATUS ats_fault_prefetch(uvm_gpu_va_space_t *gpu_va_space,
                                    struct vm_area_struct *vma,
                                    NvU64 base,
                                    uvm_ats_fault_context_t *ats_context)
{
    printk(KERN_ERR "=====================================   %d\n", 119);
    NV_STATUS status = NV_OK;
    uvm_page_mask_t *read_fault_mask = &ats_context->read_fault_mask;
    uvm_page_mask_t *write_fault_mask = &ats_context->write_fault_mask;
    uvm_page_mask_t *faulted_mask = &ats_context->faulted_mask;
    uvm_page_mask_t *prefetch_mask = &ats_context->prefetch_state.prefetch_pages_mask;
    uvm_va_block_region_t max_prefetch_region = uvm_ats_region_from_vma(vma, base);

    if (!uvm_perf_prefetch_enabled(gpu_va_space->va_space))
        return status;

    if (uvm_page_mask_empty(faulted_mask))
        return status;

    status = ats_compute_residency_mask(gpu_va_space, vma, base, ats_context);
    if (status != NV_OK)
        return status;

    // Prefetch the entire region if none of the pages are resident on any node
    // and if preferred_location is the faulting GPU.
    if (ats_context->prefetch_state.has_preferred_location &&
        ats_context->prefetch_state.first_touch &&
        uvm_id_equal(ats_context->residency_id, gpu_va_space->gpu->parent->id)) {

        uvm_page_mask_init_from_region(prefetch_mask, max_prefetch_region, NULL);
        uvm_page_mask_or(read_fault_mask, read_fault_mask, prefetch_mask);

        if (vma->vm_flags & VM_WRITE)
            uvm_page_mask_or(write_fault_mask, write_fault_mask, prefetch_mask);

        return status;
    }

    ats_expand_fault_region(gpu_va_space, vma, ats_context, max_prefetch_region, faulted_mask);

    return status;
}

NV_STATUS uvm_ats_service_faults(uvm_gpu_va_space_t *gpu_va_space,
                                 struct vm_area_struct *vma,
                                 NvU64 base,
                                 uvm_ats_fault_context_t *ats_context)
{
    printk(KERN_ERR "=====================================   %d\n", 120);
    NV_STATUS status = NV_OK;
    uvm_va_block_region_t subregion;
    uvm_va_block_region_t region = uvm_va_block_region(0, PAGES_PER_UVM_VA_BLOCK);
    uvm_page_mask_t *read_fault_mask = &ats_context->read_fault_mask;
    uvm_page_mask_t *write_fault_mask = &ats_context->write_fault_mask;
    uvm_page_mask_t *faults_serviced_mask = &ats_context->faults_serviced_mask;
    uvm_page_mask_t *reads_serviced_mask = &ats_context->reads_serviced_mask;
    uvm_fault_client_type_t client_type = ats_context->client_type;

    UVM_ASSERT(vma);
    UVM_ASSERT(IS_ALIGNED(base, UVM_VA_BLOCK_SIZE));
    UVM_ASSERT(g_uvm_global.ats.enabled);
    UVM_ASSERT(gpu_va_space);
    UVM_ASSERT(gpu_va_space->ats.enabled);
    UVM_ASSERT(uvm_gpu_va_space_state(gpu_va_space) == UVM_GPU_VA_SPACE_STATE_ACTIVE);

    uvm_page_mask_zero(faults_serviced_mask);
    uvm_page_mask_zero(reads_serviced_mask);

    if (!(vma->vm_flags & VM_READ))
        return status;

    if (!(vma->vm_flags & VM_WRITE)) {
        // If VMA doesn't have write permissions, all write faults are fatal.
        // Try servicing such faults for read iff they are also present in
        // read_fault_mask. This is because for replayable faults, if there are
        // pending read accesses on the same page, we have to service them
        // before we can cancel the write/atomic faults. So we try with read
        // fault access type even though these write faults are fatal.
        if (ats_context->client_type == UVM_FAULT_CLIENT_TYPE_GPC)
            uvm_page_mask_and(write_fault_mask, write_fault_mask, read_fault_mask);
        else
            uvm_page_mask_zero(write_fault_mask);
    }

    ats_batch_select_residency(gpu_va_space, vma, ats_context);

    ats_fault_prefetch(gpu_va_space, vma, base, ats_context);

    for_each_va_block_subregion_in_mask(subregion, write_fault_mask, region) {
        NvU64 start = base + (subregion.first * PAGE_SIZE);
        size_t length = uvm_va_block_region_num_pages(subregion) * PAGE_SIZE;
        uvm_fault_access_type_t access_type = (vma->vm_flags & VM_WRITE) ?
                                                                          UVM_FAULT_ACCESS_TYPE_WRITE :
                                                                          UVM_FAULT_ACCESS_TYPE_READ;

        UVM_ASSERT(start >= vma->vm_start);
        UVM_ASSERT((start + length) <= vma->vm_end);

        status = service_ats_faults(gpu_va_space, vma, start, length, access_type, ats_context);
        if (status != NV_OK)
            return status;

        if (vma->vm_flags & VM_WRITE) {
            uvm_page_mask_region_fill(faults_serviced_mask, subregion);

            // The Linux kernel never invalidates TLB entries on mapping
            // permission upgrade. This is a problem if the GPU has cached
            // entries with the old permission. The GPU will re-fetch the entry
            // if the PTE is invalid and page size is not 4K (this is the case
            // on P9). However, if a page gets upgraded from R/O to R/W and GPU
            // has the PTEs cached with R/O permissions we will enter an
            // infinite loop because we just forward the fault to the Linux
            // kernel and it will see that the permissions in the page table are
            // correct. Therefore, we flush TLB entries on ATS write faults.
            flush_tlb_write_faults(gpu_va_space, start, length, client_type);
        }
        else {
            uvm_page_mask_region_fill(reads_serviced_mask, subregion);
        }
    }

    // Remove write faults from read_fault_mask
    uvm_page_mask_andnot(read_fault_mask, read_fault_mask, write_fault_mask);

    for_each_va_block_subregion_in_mask(subregion, read_fault_mask, region) {
        NvU64 start = base + (subregion.first * PAGE_SIZE);
        size_t length = uvm_va_block_region_num_pages(subregion) * PAGE_SIZE;
        uvm_fault_access_type_t access_type = UVM_FAULT_ACCESS_TYPE_READ;

        UVM_ASSERT(start >= vma->vm_start);
        UVM_ASSERT((start + length) <= vma->vm_end);

        status = service_ats_faults(gpu_va_space, vma, start, length, access_type, ats_context);
        if (status != NV_OK)
            return status;

        uvm_page_mask_region_fill(faults_serviced_mask, subregion);
    }

    return status;
}

bool uvm_ats_check_in_gmmu_region(uvm_va_space_t *va_space, NvU64 address, uvm_va_range_t *next)
{
    printk(KERN_ERR "=====================================   %d\n", 121);
    uvm_va_range_t *prev;
    NvU64 gmmu_region_base = UVM_ALIGN_DOWN(address, UVM_GMMU_ATS_GRANULARITY);

    UVM_ASSERT(va_space);

    if (next) {
        if (next->node.start <= gmmu_region_base + UVM_GMMU_ATS_GRANULARITY - 1)
            return true;

        prev = uvm_va_range_container(uvm_range_tree_prev(&va_space->va_range_tree, &next->node));
    }
    else {
        // No VA range exists after address, so check the last VA range in the
        // tree.
        prev = uvm_va_range_container(uvm_range_tree_last(&va_space->va_range_tree));
    }

    return prev && (prev->node.end >= gmmu_region_base);
}

NV_STATUS uvm_ats_invalidate_tlbs(uvm_gpu_va_space_t *gpu_va_space,
                                  uvm_ats_fault_invalidate_t *ats_invalidate,
                                  uvm_tracker_t *out_tracker)
{
    printk(KERN_ERR "=====================================   %d\n", 122);
    NV_STATUS status;
    uvm_push_t push;

    if (!ats_invalidate->write_faults_in_batch)
        return NV_OK;

    UVM_ASSERT(gpu_va_space);
    UVM_ASSERT(gpu_va_space->ats.enabled);

    status = uvm_push_begin(gpu_va_space->gpu->channel_manager,
                            UVM_CHANNEL_TYPE_MEMOPS,
                            &push,
                            "Invalidate ATS entries");

    if (status == NV_OK) {
        uvm_tlb_batch_end(&ats_invalidate->write_faults_tlb_batch, &push, UVM_MEMBAR_NONE);
        uvm_push_end(&push);

        // Add this push to the GPU's tracker so that fault replays/clears can
        // wait on it
        status = uvm_tracker_add_push_safe(out_tracker, &push);
    }

    ats_invalidate->write_faults_in_batch = false;

    return status;
}

