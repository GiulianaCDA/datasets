     if (ngranules <= MAXOBJGRANULES) {
         BZERO((word *)p + 1, sz - sizeof(word));
         flh = &(ok->ok_freelist[ngranules]);
         obj_link(p) = *flh;
         *flh = (ptr_t)p;
     }
     