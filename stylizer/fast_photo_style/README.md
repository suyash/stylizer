# A Closed-Form Solution to Photorealistic Image Stylization

https://arxiv.org/abs/1802.06474

| Style                       | Content                     | Result (no smoothing)          | Result (smoothing)    | Result (smooth filter)       |
|:---------------------------:|:---------------------------:|:------------------------------:|:---------------------:|:----------------------------:|
|![](/images/style_targets/louvre_udnie.jpg)|![](/images/content_targets/katya.jpg)       |![](/images/fast_photo_style/results/styled_nosmooth.jpg)|![](/images/fast_photo_style/results/styled.jpg)|![](/images/fast_photo_style/results/smooth_filter.jpg)|

__WORK IN PROGRESS__

This implements a training script for training decoders to invert features at different levels for VGG19.

Once trained decoders and their saved_models are available, the `fast_photo_style_model.py` script can be used to create a saved_model for styling a content image using a style image.

Currently, photorealistic Smoothing is not implemented inside the model. For smoothing, a `photo_smooth.py` script is provided, which is the same script as in the NVIDIA implementation, just modified to take image paths as input when run as a script, as well as export a function to run as a module (see `demo.ipynb` notebook). Cannot port that into TensorFlow currently because `linalg` operations currently do not support sparse tensors. See https://github.com/tensorflow/tensorflow/issues/27380

The final GPU based smoothing filter is not provided/used here. Please refer to the file in the NVIDIA implementation (https://github.com/NVIDIA/FastPhotoStyle/blob/master/smooth_filter.py). I tried it after on the smoothed image in a Google Colab Notebook, and have added the result above.

Once a serving compatible saved_model is available, it can be saved and deployed anywhere, it basically takes two image inputs and returns the styled image. See the demo notebook (`demo.ipynb`) for more details.

Styling different sections/regions differently using semantic label maps (manual or automatic) is also not implemented.

`tf.nn.max_pool_with_argmax` is used instead of the keras `MaxPool2D` layer to create pooling masks, and fed into a custom `Unpool` layer.
