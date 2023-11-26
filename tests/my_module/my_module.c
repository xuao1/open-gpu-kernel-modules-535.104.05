#include <linux/init.h>
#include <linux/module.h>
#include <linux/kernel.h>

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Your Name");
MODULE_DESCRIPTION("A simple kernel module with printk");
MODULE_VERSION("0.1");

static int __init my_module_init(void) {
    printk(KERN_ERR "Hello from my_module!\n");
    return 0;
}

static void __exit my_module_exit(void) {
    printk(KERN_ERR "Goodbye from my_module!\n");
}

module_init(my_module_init);
module_exit(my_module_exit);
