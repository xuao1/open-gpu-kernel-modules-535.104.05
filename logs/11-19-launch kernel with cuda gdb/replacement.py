# 映射关系字典
mapping = {
    216: 'getUvmEvents(void)',
    266: 'nvUvmInterfaceHasPendingNonReplayableFaults',
    278: 'nv_uvm_event_interrupt',
    451: 'nvidia_frontend_ioctl',
    452: 'nvidia_frontend_unlocked_ioctl',
    670: 'nv_vm_map_pages',
    671: 'nv_vm_unmap_pages',
    720: 'nvidia_ioctl',
    721: 'nvidia_isr_msix',
    722: 'nvidia_isr',
    724: 'nvidia_isr_msix_kthread_bh',
    725: 'nvidia_isr_common_bh',
    743: 'nv_alloc_kernel_mapping',
    744: 'nv_free_kernel_mapping',
    749: 'nv_post_event',
    755: 'nv_get_event',
    782: 'nv_get_ctl_state',
    866: 'os_acquire_mutex',
    868: 'os_release_mutex',
    876: 'os_acquire_rwlock_read',
    877: 'os_acquire_rwlock_write',
    880: 'os_release_rwlock_read',
    881: 'os_release_rwlock_write',
    883: 'os_is_isr',
    884: 'os_is_administrator',
    892: 'os_mem_copy',
    893: 'os_memcpy_from_user',
    894: 'os_memcpy_to_user',
    897: 'os_alloc_mem',
    898: 'os_free_mem',
    902: 'os_get_tick_resolution',
    908: 'os_get_current_process',
    912: 'nv_printf',
    922: 'xen_support_fully_virtualized_kernel',
    923: 'os_map_kernel_space',
    924: 'os_unmap_kernel_space',
    950: 'os_is_vgx_hyper',
}

def replace_numbers_in_place(file_name):
    with open(file_name, 'r+') as file:
        content = file.read()
        for number, replacement in mapping.items():
            content = content.replace(str(number), replacement)
        file.seek(0)
        file.write(content)
        file.truncate()

file_name = input('Input file name: ')

replace_numbers_in_place(file_name)

print('Done.')
