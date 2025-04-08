import os
import warnings
import tensorflow as tf

def suppress_tf_warnings():
    """
    Suppress common TensorFlow deprecation warnings to keep the output clean
    """
    # Disable tensorflow deprecation warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=info, 2=warning, 3=error
    
    # Filter out specific deprecation warnings
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    
    # Check TF version 
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Eager execution is {'enabled' if tf.executing_eagerly() else 'disabled'}")
    
    # Enable memory growth to avoid GPU memory allocation errors
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Found {len(gpus)} GPU(s), memory growth enabled")
    except:
        print("No GPU devices found or error setting memory growth")

def get_compatible_placeholder(shape=None, dtype=None, name=None):
    """
    Get a placeholder that works with both TF 1.x and 2.x
    
    In TF 2.x, use Variable or just tensors directly when possible
    """
    if tf.__version__.startswith('2'):
        # For TF 2.x, return a Variable instead of placeholder
        return tf.Variable(
            initial_value=tf.zeros(shape, dtype=dtype),
            trainable=False,
            name=name
        )
    else:
        # For TF 1.x compatibility
        return tf.compat.v1.placeholder(dtype=dtype, shape=shape, name=name)

# Example usage of tf.compat.v1 instead of deprecated functions
def example_compat_usage():
    """
    Examples of replacing deprecated TF functions with their compat.v1 counterparts
    """
    # Instead of tf.placeholder
    placeholder = tf.compat.v1.placeholder(tf.float32, shape=[None, 784])
    
    # Instead of tf.Session
    with tf.compat.v1.Session() as sess:
        pass  # Session code here
    
    # Instead of tf.global_variables_initializer
    init = tf.compat.v1.global_variables_initializer()
