   bos = kzalloc(sizeof(struct usb_bos_descriptor), GFP_KERNEL);
   if (!bos)
       return -ENOMEM;
   