#include <stdlib.h>
#include <string.h>

// A function to convert a plist to a binary property list (bplist)
void plist_to_bplist(const plist_t *plist, uint8_t **plist_bin, size_t *length) {
    // Initialize buffers and hash table
    ptr_array_t *objects = ptr_array_new(10);
    hash_table_t *ref_table = hash_table_create(10);

    // Write the header
    byte_array_t *bplist_buff = byte_array_new(sizeof(uint32_t) + sizeof(bplist_trailer_t));
    bplist_header_t header;
    memset(&header, 0, sizeof(header));
    header.magic = htobe64(BPLIST_MAGIC);
    header.version = htobe32(BPLIST_VERSION);
    byte_array_append(bplist_buff, (uint8_t *)&header, sizeof(bplist_header_t));

    // Write the offsets
    uint64_t num_objects = plist->num_objects;
    uint64_t *offsets = malloc(sizeof(uint64_t) * num_objects);
    for (size_t i = 0; i < num_objects; i++) {
        offsets[i] = bplist_buff->len;
        plist_data_t data = plist_get_data(ptr_array_index(objects, i));
        switch (data->type) {
            case PLIST_BOOLEAN:
                buff = malloc(sizeof(uint8_t));
                buff[0] = data->boolval ? BPLIST_TRUE : BPLIST_FALSE;
                byte_array_append(bplist_buff, buff, sizeof(uint8_t));
                free(buff);
                break;
            case PLIST_UINT:
                if (data->length == 16) {
                    write_uint(bplist_buff, data->intval);
                } else {
                    write_int(bplist_buff, data->intval);
                }
                break;
            case PLIST_REAL:
                write_real(bplist_buff, data->realval);
                break;
            case PLIST_KEY:
            case PLIST_STRING:
                len = strlen(data->strval);
                if (is_ascii_string(data->strval, len)) {
                    write_string(bplist_buff, data->strval);
                } else {
                    unicodestr = plist_utf8_to_utf16(data->strval, len, &items_read, &items_written);
                    write_unicode(bplist_buff, unicodestr, items_written);
                    free(unicodestr);
                }
                break;
            case PLIST_DATA:
                write_data(bplist_buff, data->buff, data->length);
                break;
            case PLIST_ARRAY:
                write_array(bplist_buff, ptr_array_index(objects, i), ref_table, ref_size);
                break;
            case PLIST_DICT:
                write_dict(bplist_buff, ptr_array_index(objects, i), ref_table, ref_size);
                break;
            case PLIST_DATE:
                write_date(bplist_buff, data->realval);
                break;
            case PLIST_UID:
                write_uid(bplist_buff, data->intval);
                break;
            default:
                break;
        }
    }

    // Free intermediate objects
    ptr_array_free(objects);
    hash_table_destroy(ref_table);

    // Write offsets
    buff_len = bplist_buff->len;
    offset_size = get_needed_bytes(buff_len);
    offset_table_index = bplist_buff->len;
    for (size_t i = 0; i < num_objects; i++) {
        offsets[i] = htobe64(offsets[i]);
        byte_array_append(bplist_buff, (uint8_t *)&offsets[i], offset_size);
    }
    free(offsets);

    // Write trailer
    bplist_trailer_t trailer;
    memset(&trailer, 0, sizeof(trailer));
    trailer.offset_table_offset = htobe64(offset_table_index);
    byte_array_append(bplist_buff, (uint8_t *)&trailer, sizeof(bplist_trailer_t));

    // Return the binary plist
    *plist_bin = bplist_buff->data;
    *length = bplist_buff->len;
}
