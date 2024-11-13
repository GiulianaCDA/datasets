[PYTHON]
import os
from typing import Union

def imagetoraw_common(image: opj_image_t, outfile: str, inverted: bool) -> int:
    """
    Common function for converting an image to raw format.

    Args:
        image (opj_image_t): Image to be converted.
        outfile (str): Output file path.
        inverted (bool): Invert the pixel values if set to True.

    Returns:
        int: Non-zero on failure, zero otherwise.
    """
    # Number of components in the image
    numcomps = image.numcomps

    # Width and height of each component
    w = image.comps[0].w
    h = image.comps[0].h

    # Open output file
    rawFile = open(outfile, "wb")

    # Fails by default
    fails = 1

    if (rawFile is not None):
        # Loop over each component of the image
        for compno in range(numcomps):
            mask = 0
            ptr = None
            curr = 0
            uc = 0
            res = 0

            # Check precision of the component
            if (image.comps[compno].prec <= 8):
                if (image.comps[compno].sgnd == 1):
                    mask = (1 << image.comps[compno].prec) - 1
                    ptr = image.comps[compno].data
                    for line in range(h):
                        for row in range(w):
                            curr = ptr[line * w + row]
                            if (inverted):
                                curr = 255 - curr
                            uc = int((curr & mask))
                            res = rawFile.write(bytes([uc]))
                            if (res < 1):
                                print("failed to write 1 byte for " + outfile)
                                goto fin
                else:
                    mask = (1 << image.comps[compno].prec) - 1
                    ptr = image.comps[compno].data
                    for line in range(h):
                        for row in range(w):
                            curr = ptr[line * w + row]
                            if (inverted):
                                curr = 255 - curr
                            if (curr > 255):
                                curr = 255
                            elif (curr < 0):
                                curr = 0
                            uc = int((curr & mask))
                            res = rawFile.write(bytes([uc]))
                            if (res < 1):
                                print("failed to write 1 byte for " + outfile)
                                goto fin
            elif (image.comps[compno].prec <= 16):
                union = [0, 0]
                mask = (1 << image.comps[compno].prec) - 1
                ptr = image.comps[compno].data
                for line in range(h):
                    for row in range(w):
                        curr = ptr[line * w + row]
                        if (inverted):
                            curr = 32767 - curr
                        union[0] = int((curr & mask))
                        res = rawFile.write(bytes([union[1], union[0]]))
                        if (res < 2):
                            print("failed to write 2 byte for " + outfile)
                            goto fin
            elif (image.comps[compno].prec <= 32):
                print("More than 16 bits per component not handled yet")
                goto fin
            else:
                print("Error: invalid precision: " + str(image.comps[compno].prec))
                goto fin
    fails = 0
fin:
    rawFile.close()
    return fails

def imagetoraw(image: opj_image_t, outfile: str) -> int:
    """
    Convert an image to raw format and apply the inverted pixel values.

    Args:
        image (opj_image_t): Image to be converted.
        outfile (str): Output file path.

    Returns:
        int: Non-zero on failure, zero otherwise.
    """
    return imagetoraw_common(image, outfile, inverted=True)

def imagetoraw(image: opj_image_t, outfile: str) -> int:
    """
    Convert an image to raw format and apply the inverted pixel values.

    Args:
        image (opj_image_t): Image to be converted.
        outfile (str): Output file path.

    Returns:
        int: Non-zero on failure, zero otherwise.
    """
    return imagetoraw_common(image, outfile, inverted=False)
[/PYTHON]
