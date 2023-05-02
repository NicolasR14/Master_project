import numpy as np
import keras
import cv2



class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, path, batch_size, dim, n_channels=1, shuffle=True):
        'Initialization'
        self.dim = dim
        self.path = path
        self.batch_size = batch_size
        # self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        # self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the 
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        HEIGHT = 128
        WIDTH = 128
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        # X = np.empty((self.batch_size, *self.dim, self.n_channels))
        roi_GRAY = []
        # y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            # X[i,] = np.load(f'{self.path}{ID}.jpg')
    
            roi = cv2.imread(f'{self.path}{ID}.jpg')
            img_resized = cv2.resize(roi, (HEIGHT,WIDTH))
            # roi_GRAY = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
            roi_GRAY.append(cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY))
            # X[i,] = np.load(roi_GRAY)
            # # Store class
            # y[i] = self.labels[ID]
        X = np.reshape(roi_GRAY,(self.batch_size,*self.dim,self.n_channels))
        X = X.astype("float32") / 255.0

        return X,X
        # , keras.utils.to_categorical(y, num_classes=self.n_classes)
