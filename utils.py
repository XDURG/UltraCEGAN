import tensorflow as tf

def color_histogram_loss(y_true, y_pred, num_bins=256, intensity_factor=0.1):
    hist1 = tf.histogram_fixed_width(int((y_true+1)*127.5), value_range=(0, 256), nbins=num_bins)
    hist2 = tf.histogram_fixed_width(int((y_pred+1)*127.5), value_range=(0, 256), nbins=num_bins)
    hist1 = tf.cast(hist1, tf.float32)
    hist2 = tf.cast(hist2, tf.float32)
    chisq_distance = tf.reduce_sum(tf.square(hist1 - hist2) / (hist1 + hist2 + 1e-10))
    intersection = tf.reduce_sum(tf.minimum(hist1, hist2))
    loss = intensity_factor * (1.0 - (chisq_distance + intersection) / (2.0 * num_bins))
    return tf.cast(loss, tf.float32)

def ssim_loss(y_true, y_pred):
    ssim = tf.image.ssim(y_true, y_pred, max_val=1)
    return 1 - ssim
