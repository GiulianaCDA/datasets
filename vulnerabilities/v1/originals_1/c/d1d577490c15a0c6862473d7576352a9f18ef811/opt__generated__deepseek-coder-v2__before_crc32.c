uLong crc1 = 0xFFFFFFFF; // Initial CRC value
uLong crc2 = calculate_crc32(data2, length2); // Calculate CRC for second part
z_off64_t len2 = length2; // Length of the second data segment

uLong combined_crc = crc32_combine64(crc1, crc2, len2);
