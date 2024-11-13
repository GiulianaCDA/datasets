 /*
     * Read colormapped image.
     */
     number_colormaps=ReadBlobMSBShort(image);
     if (number_colormaps != 1)
       {
         ThrowFileException(exception,CorruptImageError,"ImproperColormapSize",
           image->filename);
         break;
       }
     colormap=(unsigned char *) AcquireMagickMemory(768*sizeof(*colormap));
     if (colormap == (unsigned char *) NULL)
       {
         ThrowFileException(exception,ResourceLimitError,"MemoryAllocationFailed",
           image->filename);
         break;
       }
     count=ReadBlob(image,768,(unsigned char *) colormap);
     if (count != 768)
       {
         ThrowFileException(exception,CorruptImageError,"ImproperColormapSize",
           image->filename);
         break;
       }
     image->colors=256;
     image->colormap=(PixelInfo *) AcquireMagickMemory(image->colors*
       sizeof(*image->colormap));
     if (image->colormap == (PixelInfo *) NULL)
       {
         ThrowFileException(exception,ResourceLimitError,"MemoryAllocationFailed",
           image->filename);
         break;
       }
     for (i=0; i < 256; i++)
     {
       QuantumAny
         scale;

       scale=(QuantumAny) MaxRGB/255.0;
       image->colormap[i].red=scale*(*colormap++);
       image->colormap[i].green=scale*(*colormap++);
       image->colormap[i].blue=scale*(*colormap++);
     }
   else
     {
       /*
         * Read RGB formatted image.
         */
       if (ReadBlobByte(image) == EOF)
         ThrowFileException(exception,CorruptImageError,"UnexpectedEndOfFile",
           image->filename);
     }
   pixel_info=AcquireVirtualMemory((size_t) image->columns*image->rows*
     sizeof(*pixel_info));
   if (pixel_info == (unsigned char *) NULL)
     {
       ThrowFileException(exception,ResourceLimitError,"MemoryAllocationFailed",
         image->filename);
       break;
     }
   /*
     Read RLE-coded image a scanline at a time.
   */
   for (y=0; y < (ssize_t) image->rows; y++)
   {
     register long
       pixel;

     ssize_t
       x;

     unsigned char
       *q;

     q=(unsigned char *) pixel_info+((image->rows-y-1)*image->columns);
     for (x=0; x < (ssize_t) image->columns; )
     {
       int
         tag;

       if ((tag=ReadBlobByte(image)) == EOF)
         {
           ThrowFileException(exception,CorruptImageError,"UnexpectedEndOfFile",
             image->filename);
           break;
         }
       switch (tag)
       {
         case 0:
         default:
         {
           if ((x+tag) > (ssize_t) image->columns)
             {
               ThrowFileException(exception,CorruptImageError,"ImproperImageHeader",
                 image->filename);
               break;
             }
           for (i=0; i < tag; i++)
           {
             pixel=ReadBlobByte(image);
             if (pixel == EOF)
               {
                 ThrowFileException(exception,CorruptImageError,
                   "UnexpectedEndOfFile",image->filename);
                 break;
               }
             *q++=(unsigned char) pixel;
           }
           x+=tag;
           continue;
         }
         case 1:
         {
           unsigned char
             color;

           if ((color=ReadBlobByte(image)) == EOF)
             {
               ThrowFileException(exception,CorruptImageError,"UnexpectedEndOfFile",
                 image->filename);
               break;
             }
           for (i=0; i <= tag; i++)
             *q++=color;
           x+=tag+1;
           continue;
         }
       }
     }
   }
   if ((image_format == RGBFormat) && (quantum_type != CharPixel))
     {
       /*
         * Convert RGB image to DirectClass pixel format.
         */
       unsigned char
         *p;

       p=(unsigned char *) pixel_info;
#if defined(HAVE_OPENMP)
       #pragma omp parallel for schedule(static,4) shared(stdout,image,quantum_type,progress,p,pixel_info)
#endif
       for (i=0; i < (ssize_t) image->rows; i++)
       {
         register int
           x;

         register PixelPacket
           *q;

         q=SetImagePixels(image,0,image->rows-i-1,image->columns,1);
         if (q == (PixelPacket *) NULL)
           break;
         for (x=0; x < (ssize_t) image->columns; x++)
         {
           q->red=(QuantumAny) ScaleCharToQuantum(*p++);
           q->green=(QuantumAny) ScaleCharToQuantum(*p++);
           q->blue=(QuantumAny) ScaleCharToQuantum(*p++);
           q++;
         }
         if (SyncImagePixels(image) == MagickFalse)
           break;
         if (image->previous == (Image *) NULL)
           {
             status=SetImageProgress(image,LoadImagesTag,(MagickOffsetType) i,
               image->rows);
             if (status == MagickFalse)
               break;
           }
       }
     }
   else
     {
       /*
         * Convert RGB image to PseudoClass pixel format.
         */
       unsigned char
         *p;

       p=(unsigned char *) pixel_info;
#if defined(HAVE_OPENMP)
       #pragma omp parallel for schedule(static,4) shared(stdout,image,quantum_type,progress,p,pixel_info)
#endif
       for (i=0; i < (ssize_t) image->rows; i++)
       {
         register int
           x;

         register IndexPacket
           *indexes;

         indexes=GetIndexes(image);
         if ((pixel_info != (unsigned char *) NULL) && (p != (unsigned char *) NULL))
           for (x=0; x < (ssize_t) image->columns; x++)
           {
             unsigned char
               red,
               green,
               blue;

             red=(*p++);
             green=(*p++);
             blue=(*p++);
             indexes[x]=GetColorFromPixel(image,&red,&green,&blue);
           }
         if (SyncImagePixels(image) == MagickFalse)
           break;
         if (image->previous == (Image *) NULL)
           {
             status=SetImageProgress(image,LoadImagesTag,(MagickOffsetType) i,
               image->rows);
             if (status == MagickFalse)
               break;
           }
       }
     }