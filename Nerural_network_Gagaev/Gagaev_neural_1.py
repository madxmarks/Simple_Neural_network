# example of progressively loading images from file
#Gagaev proj 1 20.01.2020
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import array_to_img
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.utils import plot_model
import matplotlib.pyplot as plt
from matplotlib import pyplot
from matplotlib.image import imread
import json
from keras.preprocessing import image

k= 1

# create generator
datagen = ImageDataGenerator()
# prepare an iterators for each dataset
train_it = datagen.flow_from_directory('C:/Users/Nik_g/Ch2/Simple_MNIST_classifier/Images/lego-brick-images/train/', class_mode='binary',target_size=(28,28),batch_size=3949,color_mode='grayscale')
#val_it = datagen.flow_from_directory('C:/Users/Nik_g/Ch2/Simple_MNIST_classifier/Images/train/', class_mode='binary',target_size=(28,28),batch_size=20,color_mode='grayscale')
test_it = datagen.flow_from_directory('C:/Users/Nik_g/Ch2/Simple_MNIST_classifier/Images/lego-brick-images/test/', class_mode='binary',target_size=(28,28),batch_size=618,color_mode='grayscale',shuffle="false")

# confirm the iterator works
TrainImagelArr, TrainLablArr = train_it.next()
print('Batch shape=%s, min=%.3f, max=%.3f' % (TrainImagelArr.shape, TrainImagelArr.min(), TrainImagelArr.max()))
print("   ")
TestImageArr, TestLablArr = test_it.next()


print('Batch shape=%s, min=%.3f, max=%.3f' % (TestImageArr.shape, TestImageArr.min(), TestImageArr.max()))



TrainImagelArr = (TrainImagelArr / 255) - 0.5
TestImageArr = (TestImageArr / 255) - 0.5
name = TestLablArr[k]

# Flatten the images.
TrainImagelArr = TrainImagelArr.reshape((-1, 784))
TestImageArr = TestImageArr.reshape((-1, 784))

# Build the model.
model = Sequential([
  Dense(64, activation='relu', input_shape=(784,)),
  Dense(64, activation='relu'),
 #  Dense(64, activation='relu'),
  Dense(10, activation='softmax'),
])

# Compile the model.
model.compile(
  optimizer='adam',
  loss='categorical_crossentropy',
  metrics=['accuracy'],
)


# Train the model.
history = model.fit(
  TrainImagelArr,
  to_categorical(TrainLablArr),
  validation_split=0.10,
  epochs=500,
  batch_size=300,
)

# Evaluate the model.
model.evaluate(
  TestImageArr,
  to_categorical(TestLablArr)
)
image1 = TestImageArr[k:k+1]


#saving the weights
model.save_weights('model.h5')
#saving the graph representation of network structure
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)




#    =============================================================================
# datagen1 = ImageDataGenerator()
## prepare an iterators for each dataset
#train_IT = datagen.flow_from_directory('Images/train/', class_mode='categorical')
##
### confirm the iterator works
#TrainImagelArr, TrainLablArr = train_IT.next()
#print('Batch shape=%s, min=%.3f, max=%.3f' % (TrainImagelArr.shape, TrainImagelArr.min(), TrainImagelArr.max()))

# pyplot.show()
# 
# #for i in range(9):
#  
# # show the figure
# for i in range(9):
#    # pyplot.show()
#     pyplot.subplot(330 + 1 + i)
#     aa=TrainImagelArr[i]
#     bb=array_to_img(aa)
#     pyplot.imshow(bb)
#     print(type(bb))
# 
# =============================================================================
#bb.show()



predictions = model.predict(TestImageArr[:])
PredArr = np.argmax(predictions, axis=1)
print(PredArr) # [7, 2, 1, 0, 4]
TestLablArr = TestLablArr.astype(int)  
print(TestLablArr[:]) # [7, 2, 1, 0, 4]



with open('prediction_comparison.txt', 'w') as f:
    for i in range(len(TestLablArr)):
        f.write(str(TestLablArr[i].astype(int)))  
        f.write('\t | \t')
        f.write(str(PredArr[i]))
        f.write('\n')


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('accuracy_graph.png')
plt.show()

#creating the loss plot
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('loss_graph.png')
plt.show()

#Due to a reason that i have a lot of images, I pick one image and try to 
#classify it and try to look what is happening if we have no labels
predictionsOne = model.predict(image1)
PredArrOne = np.argmax(predictionsOne, axis=1)
print("expected class ", name)
print(PredArrOne)
