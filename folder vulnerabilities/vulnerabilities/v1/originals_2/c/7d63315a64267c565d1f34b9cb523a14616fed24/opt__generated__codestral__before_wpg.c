 /* Read WPG image */
static Image *ReadWPGImage(const ImageInfo *image_info, ExceptionInfo *exception)
{
    const char *filename;
    Image *image = NULL;
    MagickBooleanType end_of_file = MagickFalse;
    ssize_t scene = 0;
    unsigned char magic[8];
    BlobInfo *blob;
    unsigned short header_length, version;
    HeaderRec1 StartWPG;
    RectangleInfo CTM;
    unsigned long WPG2Flags;
    ExceptionInfo local_exception;

    /* Allocate exception info if necessary */
    if (exception == NULL) {
        AcquireExceptionInfo(&local_exception);
        exception = &local_exception;
    }

    filename = image_info->filename;

    /* Open file */
    blob = OpenBlob(image_info, ReadBinaryBlobMode, exception);
    if (blob == NULL) {
        ThrowReaderException(FileOpenError, "UnableToOpenFile");
    }

    /* Check for WPG format */
    if (ReadBlobLSBShort(blob) != 0x9AC6) {
        CloseBlob(blob);
        ThrowReaderException(CorruptImageError, "NotAValidWPGFile");
    }

    /* Read header */
    StartWPG.StartTag = ReadBlobLSBShort(blob);
    if (StartWPG.StartTag != 0x01) {
        CloseBlob(blob);
        ThrowReaderException(CorruptImageError, "InvalidWPGFileHeader");
    }

    header_length = ReadBlobLSBShort(blob);
    version = ReadBlobLSBShort(blob);
    if (version != 0x03 && version != 0x05) {
        CloseBlob(blob);
        ThrowReaderException(CorruptImageError, "UnsupportedWPGFileVersion");
    }

    /* Skip reserved fields */
    SeekBlob(blob, 12L, SEEK_CUR);

    StartWPG.PosSizePrecision = ReadBlobLSBLong(blob);
    StartWPG.Reserved1 = ReadBlobLSBLong(blob);
    StartWPG.HuffmanTableLength = ReadBlobLSBShort(blob);
    StartWPG.StartOffset = ReadBlobLSBLong(blob);
    StartWPG.EndOfHeaderTag = ReadBlobLSBShort(blob);
    if (StartWPG.EndOfHeaderTag != 0xFFFF) {
        CloseBlob(blob);
        ThrowReaderException(CorruptImageError, "InvalidWPGFileHeader");
    }

    /* Skip huffman table */
    SeekBlob(blob, StartWPG.HuffmanTableLength, SEEK_CUR);

    /* Process records */
    while (!end_of_file) {
        HeaderRec2 Rec2;
        size_t record_length = 0;
        long next_record_offset = 0;
        ssize_t offset_from_current_position = 0;
        unsigned short data_encoding_scheme = 0;

        /* Read record header */
        Rec2.RecordType = ReadBlobLSBShort(blob);
        switch (Rec2.RecordType) {
            case 0x0D:
                end_of_file = MagickTrue;
                break;
            default:
                Rec2.RecordLength = ReadBlobLSBLong(blob);
                record_length = Rec2.RecordLength - 6; /* Subtract header length */
                next_record_offset = TellBlob(blob) + record_length;

                data_encoding_scheme = ReadBlobLSBShort(blob);
                switch (data_encoding_scheme) {
                    case 0x12: /* Postscript WPG2 */
                        /* Skip PS header in the WPG2 */
                        offset_from_current_position = Rec2.RecordLength > 2 ? ReadBlobLSBShort(blob) : 0;
                        if (Rec2.RecordLength > (unsigned int)offset_from_current_position) {
                            image = ExtractPostscript(image, image_info, TellBlob(blob) + offset_from_current_position, Rec2.RecordLength - offset_from_current_position - 2, exception);
                        }
                        break;
                    case 0x19: /* WPG2 */
                        if (Rec2.RecordType == 0x03) {
                            image = ReadWPG2Image(blob, image_info, &CTM, exception);
                            scene++;
                        } else if (Rec2.RecordType == 0x0F) {
                            WPG2Flags = LoadWPG2Flags(blob, Rec2.RecordLength - 2);
                        }
                        break;
                    default:
                        /* Unsupported data encoding scheme */
                        SeekBlob(blob, record_length, SEEK_CUR);
                }

                SeekBlob(blob, next_record_offset, SEEK_SET);
                break;
        }
    }

    CloseBlob(blob);

    /* Return images */
    return image;
}