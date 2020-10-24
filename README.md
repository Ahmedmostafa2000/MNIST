# MNIST
The MNIST database contains 60,000 training images and 10,000 testing images.
Usually it is used as an example for image classification since any good model will work on it.

### importing the necessary models
```javascript
from tensorflow import keras
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.datasets import mnist
from keras.layers import Conv2D, Activation, MaxPooling2D,Flatten,Dense
from keras.callbacks import ModelCheckpoint
```

### loading the mnist datasets
```javascript
(x_train,y_train), (x_test,y_test) = mnist.load_data()
```

### reshaping and normalization of the data
```javascript
x_train = x_train.reshape(-1,28,28,1)
x_test=x_test.reshape(-1,28,28,1)
x_train = x_train/255
x_test = x_test/255
```

### Adding the convolutional network
```javascript
def cnn():
    model = keras.models.Sequential()
    model.add(Conv2D(32,(3,3),input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Conv2D(32,(3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Flatten())
    
    model.add(Dense( 10))
    model.add(Activation('softmax'))
    
    model.compile(optimizer = 'adam',
                 loss = keras.losses.SparseCategoricalCrossentropy(),
                 metrics = ["accuracy"])
    return model
    
model = cnn()
```
### Training the model
Nots: I am using the checkpointer to determine the best point of convergence

```javascript
checkpointer = ModelCheckpoint(filepath='best_mnist_model.hdf5', 
                               verbose=1, save_best_only=True)
model.fit(x_train,y_train,epochs=2, batch_size=128, verbose=1)
```

### And finally evaluating the model
```javascript
model.evaluate(x_test,y_test)
```

my model got 
`a loss of: 0.0644 and an accuracy of: 0.9790`
