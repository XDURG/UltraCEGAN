import tensorflow as tf

def ssim_metric(y_true, y_pred):
    return tf.image.ssim(y_true, y_pred, max_val=1.0)

def psnr_metric(y_true, y_pred):
    return tf.image.psnr(y_true, y_pred, max_val=1.0)

def mape_metric(y_true, y_pred):
    return tf.reduce_mean(tf.abs((y_true - y_pred) / y_true))

def mse_metric(y_true, y_pred):
    return tf.keras.backend.mean(tf.keras.backend.square(y_true - y_pred)).numpy()

def mae_metric(y_true, y_pred):
    return tf.keras.backend.mean(tf.keras.backend.abs(y_true - y_pred)).numpy()