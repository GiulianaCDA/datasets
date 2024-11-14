  // [TUTORIAL] WriteMNGImage
[PYTHON]
from magickwand import MagickWand
import image

with MagickWand(image) as wand:
    wand.write('result.mng')

# Or, without a context manager:
wand = MagickWand(image)
wand.write('result.mng')
del wand
[/PYTHON]
[TESTS]
# Test for MNG support in the ImageMagick installation.
def test_has_png_support(Image):
    import os
    version = Image.VERSION.split('-')[0]
    prefix = 'ImageMagick 7.' if version >= '7' else 'Wand ';
    assert os.environ['MAGICK_CODER_MODULE_PATH'].find(prefix + 'mng') != -1

# Write image and compare with expected size.
def test_write_image(Image, util_save_image):
    image = Image('rose:')
    util_save_image(image, 'tmp/test.mng')
    assert os.path.getsize('tmp/test.mng') > 0
[/TESTS]
