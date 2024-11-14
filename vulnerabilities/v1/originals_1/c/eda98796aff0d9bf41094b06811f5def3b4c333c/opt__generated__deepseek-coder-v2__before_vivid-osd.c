   dev->video_vbase = kzalloc(dev->video_buffer_size, GFP_KERNEL | GFP_DMA32);
   if (dev->video_vbase == NULL)
       return -ENOMEM;
   dev->video_pbase = virt_to_phys(dev->video_vbase);
   