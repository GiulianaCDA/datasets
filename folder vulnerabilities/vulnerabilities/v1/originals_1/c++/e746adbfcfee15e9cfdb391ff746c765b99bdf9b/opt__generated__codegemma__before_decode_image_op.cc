# Decode a JPEG image with dimensions 224x224 and 3 color channels.
image_tensor = tf.io.decode_image(image_data, shape=[224, 224, 3])

# Decode a GIF image with animation expansion enabled.
gif_tensor = tf.io.decode_image(gif_data, expand_animations=True)

# Decode a BMP image with 4 color channels.
bmp_tensor = tf.io.decode_image(bmp_data, channels=4)
