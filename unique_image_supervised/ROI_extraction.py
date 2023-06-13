import numpy as np
import keras
import cv2

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size, dim=(40,40), n_channels=1,
                 n_classes=3, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
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
        X = []
        y = np.empty((self.batch_size), dtype=int)
        for i, ID in enumerate(list_IDs_temp):
            image = cv2.imread(ID)
            img_gamma_correct = self.correct_gamma(image)
            ROI = self.extract_ROI(img_gamma_correct)
            ROI = image[ROI['roi_y']:ROI['roi_y']+ROI['roi_size'], ROI['roi_x']:ROI['roi_x']+ROI['roi_size']]
            # ROI = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
            X.append(ROI)
            # Store class
            y[i] = self.labels[ID]

        X = np.reshape(X,(self.batch_size,*self.dim,self.n_channels))
        X = X.astype("float32") / 255.0
        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)

    def correct_gamma(self,image):
        # Convert image to float and normalize to range 0-1
        image_normalized = image.astype(float) / 255.0

        # Calculate mean R intensity
        meanRimg = np.mean(image_normalized[:, :, 2])  # Image is in BGR format
        
        # Calculate G value
        G = 0.74 * np.exp(-3.97 * meanRimg)
        
        # Apply transformation
        transformed_image = np.power(image_normalized, 1 / G)
        img_float32 = np.float32(transformed_image)
        return img_float32

    def extract_ROI(self,original_image):
        # Convert to grayscale
        gray_image = cv2.cvtColor((original_image*255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        
        # # Apply histogram normalization
        # normalized_image = cv2.equalizeHist(gray_image)
        
        # Apply median filtering
        filtered_image = cv2.medianBlur(gray_image, 5)
        
        # Apply Otsu's thresholding
        _, thresholded_image = cv2.threshold(filtered_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Apply morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed_image = cv2.morphologyEx(thresholded_image, cv2.MORPH_CLOSE, kernel)
        opened_image = cv2.morphologyEx(closed_image, cv2.MORPH_OPEN, kernel)

        # Find contours in the processed image
        contours, _ = cv2.findContours(opened_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find the contour with the largest area
        contour = max(contours, key=cv2.contourArea)
        
        # Calculate the moments of the contour
        M = cv2.moments(contour)
        
        # Calculate the center of the contour
        center_x = int(M["m10"] / M["m00"])
        center_y = int(M["m01"] / M["m00"])
        
        # Calculate the coordinates of the square ROI
        roi_size = 40
        roi_x = center_x - roi_size // 2
        roi_y = center_y - roi_size // 2
        
        return {'contours':contours,'roi_x':roi_x,'roi_y':roi_y,'roi_size':roi_size}