#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/init.h>
#include <linux/bio.h>
#include <linux/blkdev.h>
#include <linux/bootmem.h>
#include "blk.h"

static struct kmem_cache *iocontext_cachep;

static void cfq_dtor(struct io_context *ioc) {
    if (!hlist_empty(&ioc->cic_list)) {
        struct cfq_io_context *cic = list_entry(ioc->cic_list.first, struct cfq_io_context, cic_list);
        cic->dtor(ioc);
    }
}

int put_io_context(struct io_context *ioc) {
    if (ioc == NULL)
        return 1;

    BUG_ON(atomic_long_read(&ioc->refcount) == 0);

    if (atomic_long_dec_and_test(&ioc->refcount)) {
        cfq_dtor(ioc);
        kmem_cache_free(iocontext_cachep, ioc);
    }

    return 0;
}
EXPORT_SYMBOL(put_io_context);

struct io_context *alloc_io_context(gfp_t gfp_flags) {
    struct io_context *ret = kmem_cache_alloc(iocontext_cachep, gfp_flags);
    if (ret) {
        atomic_set(&ret->refcount, 1);
        // Initialize other fields...
    }
    return ret;
}

struct io_context *current_io_context(gfp_t gfp_flags) {
    struct task_struct *tsk = current;
    struct io_context *ret;

    ret = tsk->io_context;
    if (likely(ret))
        return ret;

    ret = alloc_io_context(gfp_flags);
    if (ret) {
        smp_wmb();
        tsk->io_context = ret;
    }

    return ret;
}

struct io_context *get_io_context(gfp_t gfp_flags) {
    struct io_context *ret = NULL;

    do {
        ret = current_io_context(gfp_flags);
        if (unlikely(!ret))
            break;
    } while (!atomic_long_inc_not_zero(&ret->refcount));

    return ret;
}
EXPORT_SYMBOL(get_io_context);

void copy_io_context(struct io_context **pdst, struct io_context *src) {
    if (src) {
        BUG_ON(atomic_long_read(&src->refcount) == 0);
        atomic_long_inc(&src->refcount);
        put_io_context(*pdst);
        *pdst = src;
    }
}
EXPORT_SYMBOL(copy_io_context);

static int __init blk_ioc_init(void) {
    iocontext_cachep = kmem_cache_create("blkdev_ioc", sizeof(struct io_context), 0, SLAB_PANIC, NULL);
    return 0;
}
subsys_initcall(blk_ioc_init);
