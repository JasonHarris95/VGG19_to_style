```python
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow import keras
from keras.layers import Dense, InputLayer
from google.colab import files
from io import BytesIO
from PIL import Image
```


```python
upl = files.upload()
img = Image.open(BytesIO(upl['img.jpg']))
img_style = Image.open(BytesIO(upl['img_style.jpg']))
```



     <input type="file" id="files-1a852a84-02ce-4c24-b079-f1dcc6bc32f1" name="files[]" multiple disabled
        style="border:none" />
     <output id="result-1a852a84-02ce-4c24-b079-f1dcc6bc32f1">
      Upload widget is only available when the cell has been executed in the
      current browser session. Please rerun this cell to enable.
      </output>
      <script>// Copyright 2017 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/**
 * @fileoverview Helpers for google.colab Python module.
 */
(function(scope) {
function span(text, styleAttributes = {}) {
  const element = document.createElement('span');
  element.textContent = text;
  for (const key of Object.keys(styleAttributes)) {
    element.style[key] = styleAttributes[key];
  }
  return element;
}

// Max number of bytes which will be uploaded at a time.
const MAX_PAYLOAD_SIZE = 100 * 1024;

function _uploadFiles(inputId, outputId) {
  const steps = uploadFilesStep(inputId, outputId);
  const outputElement = document.getElementById(outputId);
  // Cache steps on the outputElement to make it available for the next call
  // to uploadFilesContinue from Python.
  outputElement.steps = steps;

  return _uploadFilesContinue(outputId);
}

// This is roughly an async generator (not supported in the browser yet),
// where there are multiple asynchronous steps and the Python side is going
// to poll for completion of each step.
// This uses a Promise to block the python side on completion of each step,
// then passes the result of the previous step as the input to the next step.
function _uploadFilesContinue(outputId) {
  const outputElement = document.getElementById(outputId);
  const steps = outputElement.steps;

  const next = steps.next(outputElement.lastPromiseValue);
  return Promise.resolve(next.value.promise).then((value) => {
    // Cache the last promise value to make it available to the next
    // step of the generator.
    outputElement.lastPromiseValue = value;
    return next.value.response;
  });
}

/**
 * Generator function which is called between each async step of the upload
 * process.
 * @param {string} inputId Element ID of the input file picker element.
 * @param {string} outputId Element ID of the output display.
 * @return {!Iterable<!Object>} Iterable of next steps.
 */
function* uploadFilesStep(inputId, outputId) {
  const inputElement = document.getElementById(inputId);
  inputElement.disabled = false;

  const outputElement = document.getElementById(outputId);
  outputElement.innerHTML = '';

  const pickedPromise = new Promise((resolve) => {
    inputElement.addEventListener('change', (e) => {
      resolve(e.target.files);
    });
  });

  const cancel = document.createElement('button');
  inputElement.parentElement.appendChild(cancel);
  cancel.textContent = 'Cancel upload';
  const cancelPromise = new Promise((resolve) => {
    cancel.onclick = () => {
      resolve(null);
    };
  });

  // Wait for the user to pick the files.
  const files = yield {
    promise: Promise.race([pickedPromise, cancelPromise]),
    response: {
      action: 'starting',
    }
  };

  cancel.remove();

  // Disable the input element since further picks are not allowed.
  inputElement.disabled = true;

  if (!files) {
    return {
      response: {
        action: 'complete',
      }
    };
  }

  for (const file of files) {
    const li = document.createElement('li');
    li.append(span(file.name, {fontWeight: 'bold'}));
    li.append(span(
        `(${file.type || 'n/a'}) - ${file.size} bytes, ` +
        `last modified: ${
            file.lastModifiedDate ? file.lastModifiedDate.toLocaleDateString() :
                                    'n/a'} - `));
    const percent = span('0% done');
    li.appendChild(percent);

    outputElement.appendChild(li);

    const fileDataPromise = new Promise((resolve) => {
      const reader = new FileReader();
      reader.onload = (e) => {
        resolve(e.target.result);
      };
      reader.readAsArrayBuffer(file);
    });
    // Wait for the data to be ready.
    let fileData = yield {
      promise: fileDataPromise,
      response: {
        action: 'continue',
      }
    };

    // Use a chunked sending to avoid message size limits. See b/62115660.
    let position = 0;
    do {
      const length = Math.min(fileData.byteLength - position, MAX_PAYLOAD_SIZE);
      const chunk = new Uint8Array(fileData, position, length);
      position += length;

      const base64 = btoa(String.fromCharCode.apply(null, chunk));
      yield {
        response: {
          action: 'append',
          file: file.name,
          data: base64,
        },
      };

      let percentDone = fileData.byteLength === 0 ?
          100 :
          Math.round((position / fileData.byteLength) * 100);
      percent.textContent = `${percentDone}% done`;

    } while (position < fileData.byteLength);
  }

  // All done.
  yield {
    response: {
      action: 'complete',
    }
  };
}

scope.google = scope.google || {};
scope.google.colab = scope.google.colab || {};
scope.google.colab._files = {
  _uploadFiles,
  _uploadFilesContinue,
};
})(self);
</script> 


    Saving img.jpg to img (5).jpg
    Saving img_style.jpg to img_style (1).jpg
    


```python
plt.subplot(1, 2, 1)
plt.imshow( img )
plt.subplot(1, 2, 2)
plt.imshow( img_style )
plt.show()
```


    
![png](output_2_0.png)
    



```python
x_img = keras.applications.vgg19.preprocess_input(np.expand_dims(img, axis=0))                #conversion to BGR format
x_style = keras.applications.vgg19.preprocess_input(np.expand_dims(img_style, axis=0))

def deprocess_img(processed_img):                     #to return the image to the original RGB format, we immediately define the following function
  x = processed_img.copy()
  if len(x.shape) == 4:
    x = np.squeeze(x, 0)
  assert len(x.shape) == 3, ("Input to deprocess image must be an image of"
                             "dimension [1, height, width, channel] or [height, width, channel]")
  if len(x.shape) != 3:
    raise ValueError("Invalid input to deprocessing image")
  
  # perform the inverse of the preprocessing step
  x[:, :, 0] += 103.939
  x[:, :, 1] += 116.779
  x[:, :, 2] += 123.68
  x = x[:, :, ::-1]
 
  x = np.clip(x, 0, 255).astype('uint8')
  return x
```


```python
vgg = keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')     #load the trained VGG19 network, but without a fully connected neural network at its end (we don’t need it)
vgg.trainable = False
```


```python
inputs = keras.Input(shape=(784,), name='img')                     #we will need to work with this network, feed images to its input and take computed feature maps at the outputs of certain layers
x = keras.layers.Dense(64, activation='relu')(inputs)
x = keras.layers.Dense(64, activation='relu')(x)
outputs = keras.layers.Dense(10, activation='softmax')(x)

model = keras.models.Model(inputs, outputs)

content_layers=['block5_conv2']                                             #define auxiliary collections with layer names
style_layers=['block1_conv1',
             'block2_conv1',
             'block3_conv1',
             'block4_conv1',
             'block5_conv1']

print(vgg.summary())
```

    Model: "vgg19"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     input_3 (InputLayer)        [(None, None, None, 3)]   0         
                                                                     
     block1_conv1 (Conv2D)       (None, None, None, 64)    1792      
                                                                     
     block1_conv2 (Conv2D)       (None, None, None, 64)    36928     
                                                                     
     block1_pool (MaxPooling2D)  (None, None, None, 64)    0         
                                                                     
     block2_conv1 (Conv2D)       (None, None, None, 128)   73856     
                                                                     
     block2_conv2 (Conv2D)       (None, None, None, 128)   147584    
                                                                     
     block2_pool (MaxPooling2D)  (None, None, None, 128)   0         
                                                                     
     block3_conv1 (Conv2D)       (None, None, None, 256)   295168    
                                                                     
     block3_conv2 (Conv2D)       (None, None, None, 256)   590080    
                                                                     
     block3_conv3 (Conv2D)       (None, None, None, 256)   590080    
                                                                     
     block3_conv4 (Conv2D)       (None, None, None, 256)   590080    
                                                                     
     block3_pool (MaxPooling2D)  (None, None, None, 256)   0         
                                                                     
     block4_conv1 (Conv2D)       (None, None, None, 512)   1180160   
                                                                     
     block4_conv2 (Conv2D)       (None, None, None, 512)   2359808   
                                                                     
     block4_conv3 (Conv2D)       (None, None, None, 512)   2359808   
                                                                     
     block4_conv4 (Conv2D)       (None, None, None, 512)   2359808   
                                                                     
     block4_pool (MaxPooling2D)  (None, None, None, 512)   0         
                                                                     
     block5_conv1 (Conv2D)       (None, None, None, 512)   2359808   
                                                                     
     block5_conv2 (Conv2D)       (None, None, None, 512)   2359808   
                                                                     
     block5_conv3 (Conv2D)       (None, None, None, 512)   2359808   
                                                                     
     block5_conv4 (Conv2D)       (None, None, None, 512)   2359808   
                                                                     
     block5_pool (MaxPooling2D)  (None, None, None, 512)   0         
                                                                     
    =================================================================
    Total params: 20,024,384
    Trainable params: 0
    Non-trainable params: 20,024,384
    _________________________________________________________________
    None
    


```python
num_content_layers=len(content_layers)
num_style_layers=len(style_layers)

style_outputs = [vgg.get_layer(name).output for name in style_layers]       #get layers for our model
content_outputs = [vgg.get_layer(name).output for name in content_layers]

model_outputs = style_outputs + content_outputs
 
print(vgg.input)
for m in model_outputs:
  print(m)
```

    KerasTensor(type_spec=TensorSpec(shape=(None, None, None, 3), dtype=tf.float32, name='input_3'), name='input_3', description="created by layer 'input_3'")
    KerasTensor(type_spec=TensorSpec(shape=(None, None, None, 64), dtype=tf.float32, name=None), name='block1_conv1/Relu:0', description="created by layer 'block1_conv1'")
    KerasTensor(type_spec=TensorSpec(shape=(None, None, None, 128), dtype=tf.float32, name=None), name='block2_conv1/Relu:0', description="created by layer 'block2_conv1'")
    KerasTensor(type_spec=TensorSpec(shape=(None, None, None, 256), dtype=tf.float32, name=None), name='block3_conv1/Relu:0', description="created by layer 'block3_conv1'")
    KerasTensor(type_spec=TensorSpec(shape=(None, None, None, 512), dtype=tf.float32, name=None), name='block4_conv1/Relu:0', description="created by layer 'block4_conv1'")
    KerasTensor(type_spec=TensorSpec(shape=(None, None, None, 512), dtype=tf.float32, name=None), name='block5_conv1/Relu:0', description="created by layer 'block5_conv1'")
    KerasTensor(type_spec=TensorSpec(shape=(None, None, None, 512), dtype=tf.float32, name=None), name='block5_conv2/Relu:0', description="created by layer 'block5_conv2'")
    


```python
model = keras.models.Model(vgg.input, model_outputs)                        #model
for layer in model.layers:
    layer.trainable = False 
print(model.summary())
```

    Model: "model_6"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     input_3 (InputLayer)        [(None, None, None, 3)]   0         
                                                                     
     block1_conv1 (Conv2D)       (None, None, None, 64)    1792      
                                                                     
     block1_conv2 (Conv2D)       (None, None, None, 64)    36928     
                                                                     
     block1_pool (MaxPooling2D)  (None, None, None, 64)    0         
                                                                     
     block2_conv1 (Conv2D)       (None, None, None, 128)   73856     
                                                                     
     block2_conv2 (Conv2D)       (None, None, None, 128)   147584    
                                                                     
     block2_pool (MaxPooling2D)  (None, None, None, 128)   0         
                                                                     
     block3_conv1 (Conv2D)       (None, None, None, 256)   295168    
                                                                     
     block3_conv2 (Conv2D)       (None, None, None, 256)   590080    
                                                                     
     block3_conv3 (Conv2D)       (None, None, None, 256)   590080    
                                                                     
     block3_conv4 (Conv2D)       (None, None, None, 256)   590080    
                                                                     
     block3_pool (MaxPooling2D)  (None, None, None, 256)   0         
                                                                     
     block4_conv1 (Conv2D)       (None, None, None, 512)   1180160   
                                                                     
     block4_conv2 (Conv2D)       (None, None, None, 512)   2359808   
                                                                     
     block4_conv3 (Conv2D)       (None, None, None, 512)   2359808   
                                                                     
     block4_conv4 (Conv2D)       (None, None, None, 512)   2359808   
                                                                     
     block4_pool (MaxPooling2D)  (None, None, None, 512)   0         
                                                                     
     block5_conv1 (Conv2D)       (None, None, None, 512)   2359808   
                                                                     
     block5_conv2 (Conv2D)       (None, None, None, 512)   2359808   
                                                                     
    =================================================================
    Total params: 15,304,768
    Trainable params: 0
    Non-trainable params: 15,304,768
    _________________________________________________________________
    None
    


```python
outs = model(x_img)
```


```python
def get_feature_representations(model):
  # batch compute content and style features
  style_outputs = model(x_style)
  content_outputs = model(x_img)
 
  # Get the style and content feature representations from our model  
  style_features = [style_layer[0] for style_layer in style_outputs[:num_style_layers]]
  content_features = [content_layer[0] for content_layer in content_outputs[num_style_layers:]]
  return style_features, content_features

def get_content_loss(base_content, target):                     #Mean square difference across content can be calculated using the built-in methods of the Tensorflow package
  return tf.reduce_mean(tf.square(base_content - target))

def gram_matrix(input_tensor):
  # We make the image channels first 
  channels = int(input_tensor.shape[-1])
  a = tf.reshape(input_tensor, [-1, channels])
  n = tf.shape(a)[0]
  gram = tf.matmul(a, a, transpose_a=True)
  return gram / tf.cast(n, tf.float32)

def get_style_loss(base_style, gram_target):                  #implementation of a function that will calculate the size of the squares of the mismatch between the style maps of the generated image and the style image
  gram_style = gram_matrix(base_style)
 
  return tf.reduce_mean(tf.square(gram_style - gram_target))

def compute_loss(model, loss_weights, init_image, gram_style_features, content_features):  #general function to calculate all losses
  style_weight, content_weight = loss_weights
 
  model_outputs = model(init_image)
 
  style_output_features = model_outputs[:num_style_layers]
  content_output_features = model_outputs[num_style_layers:]
 
  style_score = 0
  content_score = 0
 
  weight_per_style_layer = 1.0 / float(num_style_layers)
  for target_style, comb_style in zip(gram_style_features, style_output_features):
    style_score += weight_per_style_layer * get_style_loss(comb_style[0], target_style)
 
  weight_per_content_layer = 1.0 / float(num_content_layers)
  for target_content, comb_content in zip(content_features, content_output_features):
    content_score += weight_per_content_layer* get_content_loss(comb_content[0], target_content)
 
  style_score *= style_weight
  content_score *= content_weight
 
  # Get total loss
  loss = style_score + content_score 
  return loss, style_score, content_score

num_iterations=100         #determine the number of iterations of the algorithm and parameters to take into account the weight of the content in the generated image and styles
content_weight=1e3
style_weight=1e-2

style_features, content_features = get_feature_representations(model)                   #compute style and content maps for initial images
gram_style_features = [gram_matrix(style_feature) for style_feature in style_features]

init_image = np.copy(x_img)                                  #compute style and content maps for initial images
init_image = tf.Variable(init_image, dtype=tf.float32)

#optimizer, current iteration number, variables to store the image with the least loss and best styling, and a tuple of alpha and beta parameters.
opt = tf.compat.v1.train.AdamOptimizer(learning_rate=2, beta1=0.99, epsilon=1e-1)  
iter_count = 1
 
best_loss, best_img = float('inf'), None
loss_weights = (style_weight, content_weight)

cfg = {                                                     #Let's create a configuration dictionary
      'model': model,
      'loss_weights': loss_weights,
      'init_image': init_image,
      'gram_style_features': gram_style_features,
      'content_features': content_features
}

norm_means = np.array([103.939, 116.779, 123.68])             #Auxiliary variables for converting generated images to RGB format
min_vals = -norm_means
max_vals = 255 - norm_means
imgs = []
```


```python
for i in range(num_iterations):                               #running the gradient descent algorithm itself
    with tf.GradientTape() as tape: 
       all_loss = compute_loss(**cfg)
 
    loss, style_score, content_score = all_loss
    grads = tape.gradient(loss, init_image)
 
    opt.apply_gradients([(grads, init_image)])
    clipped = tf.clip_by_value(init_image, min_vals, max_vals)
    init_image.assign(clipped)
 
    if loss < best_loss:
      best_loss = loss
      best_img = deprocess_img(init_image.numpy())
 
      plot_img = deprocess_img(init_image.numpy())
      imgs.append(plot_img)
      print('Iteration: {}'.format(i))
```

    Iteration: 0
    Iteration: 1
    Iteration: 2
    Iteration: 3
    Iteration: 4
    Iteration: 5
    Iteration: 6
    Iteration: 7
    Iteration: 8
    Iteration: 9
    Iteration: 10
    Iteration: 11
    Iteration: 12
    Iteration: 13
    Iteration: 14
    Iteration: 15
    Iteration: 16
    Iteration: 17
    Iteration: 18
    Iteration: 19
    Iteration: 20
    Iteration: 21
    Iteration: 22
    Iteration: 23
    Iteration: 24
    Iteration: 25
    Iteration: 26
    Iteration: 27
    Iteration: 28
    Iteration: 29
    Iteration: 30
    Iteration: 31
    Iteration: 32
    Iteration: 33
    Iteration: 34
    Iteration: 35
    Iteration: 36
    Iteration: 37
    Iteration: 38
    Iteration: 39
    Iteration: 40
    Iteration: 41
    Iteration: 42
    Iteration: 43
    Iteration: 44
    Iteration: 45
    Iteration: 46
    Iteration: 47
    Iteration: 48
    Iteration: 49
    Iteration: 50
    Iteration: 51
    Iteration: 52
    Iteration: 53
    Iteration: 54
    Iteration: 55
    Iteration: 56
    Iteration: 57
    Iteration: 58
    Iteration: 59
    Iteration: 60
    Iteration: 61
    Iteration: 62
    Iteration: 63
    Iteration: 64
    Iteration: 65
    Iteration: 66
    Iteration: 67
    Iteration: 68
    Iteration: 69
    Iteration: 70
    Iteration: 71
    Iteration: 72
    Iteration: 73
    Iteration: 74
    Iteration: 75
    Iteration: 76
    Iteration: 77
    Iteration: 78
    Iteration: 79
    Iteration: 80
    Iteration: 81
    Iteration: 82
    Iteration: 83
    Iteration: 84
    Iteration: 85
    Iteration: 86
    Iteration: 87
    Iteration: 88
    Iteration: 89
    Iteration: 90
    Iteration: 91
    Iteration: 92
    Iteration: 93
    Iteration: 94
    Iteration: 95
    Iteration: 96
    Iteration: 97
    Iteration: 98
    Iteration: 99
    


```python
plt.imshow(best_img)
print(best_loss)
```

    tf.Tensor(13000847.0, shape=(), dtype=float32)
    


    
![png](output_11_1.png)
    



```python
image = Image.fromarray(best_img.astype('uint8'), 'RGB')
image.save('out.jpg')
files.download('out.jpg')
```
