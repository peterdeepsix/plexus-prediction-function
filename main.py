import tensorflow
import tensorflow as tf
from tensorflow.python.keras import models
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from google.cloud import storage
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import layers
from tensorflow.python.keras import losses
from tensorflow.python.keras.preprocessing import image as kp_image
import IPython.display
import os
import functools
import time
from PIL import Image
import numpy as np
import numpy


import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (10, 10)
mpl.rcParams['axes.grid'] = False


style_path = '/content/drive/Shared drives/Deep Six Design/Deep Six Design/Projects/Neural Art Process/Iterations/Contact/style.jpg'
content_path = '/content/drive/Shared drives/Deep Six Design/Deep Six Design/Projects/Neural Art Process/Iterations/Contact/content.jpg'
output_path = '/content/drive/Shared drives/Deep Six Design/Deep Six Design/Projects/Neural Art Process/Iterations/Contact/output.png'

# We keep model as global variable so we don't have to reload it in case of warm invocations
model = None


class CustomModel(Model):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.conv1 = Conv2D(32, 3, activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(10, activation='softmax')

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)


def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    blob.download_to_filename(destination_file_name)

    print('Blob {} downloaded to {}.'.format(
        source_blob_name,
        destination_file_name))


def handler(request):
    global model
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    # Model load which only happens during cold starts
    if model is None:
        download_blob('<your_bucket_name>', 'tensorflow/fashion_mnist_weights.index',
                      '/tmp/fashion_mnist_weights.index')
        download_blob('<your_bucket_name>', 'tensorflow/fashion_mnist_weights.data-00000-of-00001',
                      '/tmp/fashion_mnist_weights.data-00000-of-00001')
        model = CustomModel()
        model.load_weights('/tmp/fashion_mnist_weights')

    download_blob('<your_bucket_name>', 'tensorflow/test.png', '/tmp/test.png')
    image = numpy.array(Image.open('/tmp/test.png'))
    input_np = (numpy.array(Image.open('/tmp/test.png')) /
                255)[numpy.newaxis, :, :, numpy.newaxis]
    predictions = model.call(input_np)
    print(predictions)
    print("Image is "+class_names[numpy.argmax(predictions)])

    return class_names[numpy.argmax(predictions)]

##########################################################################
#----------------------------------Python Notebook-----------------------#
##########################################################################


def load_img(path_to_img):
    max_dim = 1024
    img = Image.open(path_to_img)
    long = max(img.size)
    scale = max_dim/long
    img = img.resize(
        (round(img.size[0]*scale), round(img.size[1]*scale)), Image.ANTIALIAS)

    img = kp_image.img_to_array(img)

    # Broadcast the image array such that it has a batch dimension
    img = np.expand_dims(img, axis=0)
    return img


def imshow(img, title=None):
    # Remove the batch dimension
    out = np.squeeze(img, axis=0)
    # Normalize for display
    out = out.astype('uint8')
    plt.imshow(out)
    if title is not None:
        plt.title(title)
    plt.imshow(out)

# Load preprocess images according to that VGG train process. each channel is normalized by mean = [103.939, 116.779, 123.68] and with channels BGR.


def load_and_process_img(path_to_img):
    img = load_img(path_to_img)
    img = tf.keras.applications.vgg19.preprocess_input(img)
    return img

# Inverse the preprocess step to view the ouptuts of the optimization. clip values to 0-255 from infinity


def deprocess_img(processed_img):
    x = processed_img.copy()
    if len(x.shape) == 4:
        x = np.squeeze(x, 0)
    assert len(x.shape) == 3, ("Input to deprocess image must be an image of "
                               "dimension [1, height, width, channel] or [height, width, channel]")
    if len(x.shape) != 3:
        raise ValueError("Invalid input to deprocessing image")

    # perform the inverse of the preprocessiing step
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]

    x = np.clip(x, 0, 255).astype('uint8')
    return x


# pull feature maps from these content layers
content_layers = ['block5_conv2']

# interested in these style layers
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1'
                ]

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

# Access intermediate layers of style and features by getting corresponding outputs with Keras
# define the model via functional api like this... model = Model(inputs, outputs)


def get_model():
    # Load pretrained VGG, trained on imagenet data
    vgg = tf.keras.applications.vgg19.VGG19(
        include_top=False, weights='imagenet')
    vgg.trainable = False
    # Get output layers corresponding to style and content layers
    style_outputs = [vgg.get_layer(name).output for name in style_layers]
    content_outputs = [vgg.get_layer(name).output for name in content_layers]
    model_outputs = style_outputs + content_outputs
    # Build model
    return models.Model(vgg.input, model_outputs)

# add content losses to each layer


def get_content_loss(base_content, target):
    return tf.reduce_mean(tf.square(base_content - target))

# implement style lose as a distance metric


def gram_matrix(input_tensor):
    # Make image channels first
    channels = int(input_tensor.shape[-1])
    a = tf.reshape(input_tensor, [-1, channels])
    n = tf.shape(a)[0]
    gram = tf.matmul(a, a, transpose_a=True)
    return gram / tf.cast(n, tf.float32)


def get_style_loss(base_style, gram_target):
    """Expects two images of dimension h, w, c"""
    # height, width, num filters of each layer
    # Scale the loss at a given layer by the size of the feature map and the number of filters
    height, width, channels = base_style.get_shape().as_list()
    gram_style = gram_matrix(base_style)

    # / (4. * (channels ** 2) * (width * height) ** 2)
    return tf.reduce_mean(tf.square(gram_style - gram_target))

# load content and style, feed them through the network and output the representations


def get_feature_representations(model, content_path, style_path):
    # Load our images in
    content_image = load_and_process_img(content_path)
    style_image = load_and_process_img(style_path)

    # batch compute content and style features
    style_outputs = model(style_image)
    content_outputs = model(content_image)

    # Get the style and content feature representations from our model
    style_features = [style_layer[0]
                      for style_layer in style_outputs[:num_style_layers]]
    content_features = [content_layer[0]
                        for content_layer in content_outputs[num_style_layers:]]
    return style_features, content_features

# compute the loss and gradients


def compute_loss(model, loss_weights, init_image, gram_style_features, content_features):
    style_weight, content_weight = loss_weights

    # Feed our init image through our model. returns the content and style representations at our desired layers
    model_outputs = model(init_image)

    style_output_features = model_outputs[:num_style_layers]
    content_output_features = model_outputs[num_style_layers:]

    style_score = 0
    content_score = 0

    # Accumulate style losses from all layers. equally weighting each contribution of each loss layer
    weight_per_style_layer = 1.0 / float(num_style_layers)
    for target_style, comb_style in zip(gram_style_features, style_output_features):
        style_score += weight_per_style_layer * \
            get_style_loss(comb_style[0], target_style)

    # Accumulate content losses from all layers
    weight_per_content_layer = 1.0 / float(num_content_layers)
    for target_content, comb_content in zip(content_features, content_output_features):
        content_score += weight_per_content_layer * \
            get_content_loss(comb_content[0], target_content)

    style_score *= style_weight
    content_score *= content_weight

    # Get total loss
    loss = style_score + content_score
    return loss, style_score, content_score


def compute_grads(cfg):
    with tf.GradientTape() as tape:
        all_loss = compute_loss(**cfg)
    # Compute gradients wrt input image
    total_loss = all_loss[0]
    return tape.gradient(total_loss, cfg['init_image']), all_loss


def run_style_transfer(content_path,
                       style_path,
                       num_iterations=1000,
                       content_weight=1e3,
                       style_weight=1e-2):
    # not training so set false
    model = get_model()
    for layer in model.layers:
        layer.trainable = False

    # Get the style and content feature representations from specified intermediate layers
    style_features, content_features = get_feature_representations(
        model, content_path, style_path)
    gram_style_features = [gram_matrix(style_feature)
                           for style_feature in style_features]

    # Set initial image
    init_image = load_and_process_img(content_path)
    init_image = tf.Variable(init_image, dtype=tf.float32)
    # Create optimizer
    opt = tf.optimizers.Adam(learning_rate=5, beta_1=0.99, epsilon=1e-1)

    # For displaying intermediate images
    iter_count = 1

    # Store best result
    best_loss, best_img = float('inf'), None

    # Create a config
    loss_weights = (style_weight, content_weight)
    cfg = {
        'model': model,
        'loss_weights': loss_weights,
        'init_image': init_image,
        'gram_style_features': gram_style_features,
        'content_features': content_features
    }

    # For displaying
    num_rows = 2
    num_cols = 5
    display_interval = num_iterations/(num_rows*num_cols)
    start_time = time.time()
    global_start = time.time()

    norm_means = np.array([103.939, 116.779, 123.68])
    min_vals = -norm_means
    max_vals = 255 - norm_means

    imgs = []
    for i in range(num_iterations):
        grads, all_loss = compute_grads(cfg)
        loss, style_score, content_score = all_loss
        opt.apply_gradients([(grads, init_image)])
        clipped = tf.clip_by_value(init_image, min_vals, max_vals)
        init_image.assign(clipped)
        end_time = time.time()

        if loss < best_loss:
            # Update best loss and best image from total loss.
            best_loss = loss
            best_img = deprocess_img(init_image.numpy())

        if i % display_interval == 0:
            start_time = time.time()

            # Use the .numpy() method to get the solid numpy array
            plot_img = init_image.numpy()
            plot_img = deprocess_img(plot_img)
            imgs.append(plot_img)
            IPython.display.clear_output(wait=True)
            IPython.display.display_png(Image.fromarray(plot_img))
            print('Iteration: {}'.format(i))
            print('Total loss: {:.4e}, '
                  'style loss: {:.4e}, '
                  'content loss: {:.4e}, '
                  'time: {:.4f}s'.format(loss, style_score, content_score, time.time() - start_time))
    print('Total time: {:.4f}s'.format(time.time() - global_start))
    IPython.display.clear_output(wait=True)
    plt.figure(figsize=(14, 4))
    for i, img in enumerate(imgs):
        plt.subplot(num_rows, num_cols, i+1)
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])

    return best_img, best_loss

# depreocess the output image to remove processing


def show_results(best_img, content_path, style_path, output_path, show_large_final=True):
    plt.figure(figsize=(10, 5))
    content = load_img(content_path)
    style = load_img(style_path)

    plt.subplot(1, 2, 1)
    imshow(content, 'Content Image')

    plt.subplot(1, 2, 2)
    imshow(style, 'Style Image')

    if show_large_final:
        plt.figure(figsize=(40, 40))

        plt.imshow(best_img)
        plt.imsave(output_path, best_img)
        plt.title('Output Image')
        plt.show()

# Run style transfer and show the results


def best_results():
    best, best_loss = run_style_transfer(
        content_path, style_path, num_iterations=1000)
    Image.fromarray(best)
    show_results(best, content_path, style_path, output_path)
