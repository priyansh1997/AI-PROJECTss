import numpy as np
from numpy import asarray
import cv2 #opencv lib
from keras.preprocessing.image import load_img, img_to_array
#load_img=loads an image into PIL (Pillow)format 
#img_to_array=converts the image to  numpy array
import os
#os lib can provide some function of dir cmd prompt like mkdir, dir etc.
from keras.layers import Conv3D, ConvLSTM2D, Conv3DTranspose
#Conv3D=This layer creates a convolution kernel that is convolved with the layer input to produce a tensor of outputs.
#ConvLSTM2D=It is similar to an LSTM layer, but the input transformations and recurrent transformations are both convolutional.
#Conv3DTranspose=The need for transposed convolutions generally arises from the desire to use a 
#transformation going in the opposite direction of a normal convolution, i.e., from something that has the shape of the output of some convolution to something that has the shape of its input while maintaining a connectivity pattern that is compatible with said convolution.
from keras.models import Sequential
#A Sequential model is appropriate for a plain stack of layers where each layer has exactly one input tensor and one output tensor
from keras.callbacks import ModelCheckpoint, EarlyStopping
#Callback=To save the Keras model or model weights at some frequency.
#ModelCheckpoint=ModelCheckpoint callback is used in conjunction with training using model.
#fit() to save a model or weights (in a checkpoint file) at some interval, so the model or weights can be loaded later to continue the training from the state saved.
#EarlyStopping=Stop training when a monitored metric has stopped improving.
#import imutils
#imutils=A series of convenience functions to make basic image processing functions such as translation, rotation, resizing, skeletonization, displaying Matplotlib images, sorting contours, detecting edges, and much more easier with OpenCV 


train_img=[]
train_path='E://AI//object_detection//UCSD_Anomaly_Dataset//UCSD_Anomaly_Dataset.v1p2//UCSDped1//Train//frames'

#train_videos=os.listdir(train_path)
#train_images_path=train_path+'/frames'
#os.makedirs(train_images_path+'/frames')
#A folder named frames is created at the given path by os.makedir()

i=0
video=cv2.VideoCapture(0)
while(True):
    _,frame=video.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    cv2.imshow('video',gray)
    image=cv2.resize(gray,(227,227))
    #because we cannot train the model as the model wants 5d array
    cv2.imwrite("gray%d.jpg"%i,image)
    train_img.append(image)
    i=i+1
    if(cv2.waitKey(1)==ord('q')):
        break
video.release()
cv2.destroyAllWindows()


train_image=np.array(train_img,dtype=np.uint8)
a,b,c=train_image.shape
train_image.resize(b,c,a)
#now we resize an train_image matrix 
#np.save('training.npy',store_image)

model=Sequential()

model.add(Conv3D(filters=128,kernel_size=(11,11,1),strides=(4,4,1),padding='valid',input_shape=(227,227,10,1),activation='tanh'))
#5+D tensor with shape: batch_shape + (channels, conv_dim1, conv_dim2, conv_dim3) if data_format='channels_first' 
#5+D tensor with shape: batch_shape + (conv_dim1, conv_dim2, conv_dim3, channels) if data_format='channels_last'

model.add(Conv3D(filters=64,kernel_size=(5,5,1),strides=(2,2,1),padding='valid',activation='tanh'))

model.add(ConvLSTM2D(filters=64,kernel_size=(3,3),strides=1,padding='same',dropout=0.2,return_sequences=True))

model.add(ConvLSTM2D(filters=32,kernel_size=(3,3),strides=1,padding='same',dropout=0.3,return_sequences=True))

model.add(ConvLSTM2D(filters=64,kernel_size=(3,3),strides=1,padding='same',dropout=0.2,return_sequences=True))

model.add(Conv3DTranspose(filters=128,kernel_size=(5,5,1),strides=(2,2,1),padding='valid',activation='tanh'))

model.add(Conv3DTranspose(filters=1,kernel_size=(11,11,1),strides=(4,4,1),padding='valid',activation='tanh'))

model.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])





scaled_images=(train_image-train_image.mean())/train_image.std()
clipped_images=np.clip(scaled_images,0,1)


frm=clipped_images.shape[2]
frm=frm-frm%10
train_image1=clipped_images[:,:,:frm]
train_image2=train_image1.reshape(-1,227,227,10)
train_image3=np.expand_dims(train_image2,axis=4)

target_data=train_image3

callback_save=ModelCheckpoint("saved_wts.hdf5",monitor="mean_squared_error",mode='auto',save_best_only=True)
#save the best wts for the further use
callback_earlyStopping=EarlyStopping(monitor="accuracy",patience=3,mode='auto')
#stops the epochs when there is no further changes in the accuracy  or loss

model.fit(train_image3,target_data,batch_size=1,epochs=5,callbacks=[callback_save,callback_earlyStopping]).history


def mean_squared_loss(x1,x2):
    difference=x1-x2
    a,b,c,d,e=difference.shape
    n_samples=a*b*c*d*e
    sq_difference=difference**2
    Sum=sq_difference.sum()
    distance=np.sqrt(Sum)
    mean_distance=distance/n_samples
    return mean_distance



imagedump=[]
cap=cv2.VideoCapture(0)
while (True):
    
    _,frame1=cap.read()
    gray2=cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    cv2.imshow('video1',gray2)
    image1=cv2.resize(gray2,(227,227))
    
    
    scaled_test_images=(image1-image1.mean())/image1.std()
    clipped_test_images=np.clip(scaled_test_images,0,1)
    imagedump.append(clipped_test_images)
    imagedump1=np.array(imagedump)
    imagedump1.resize(227,227,10)
    imagedump2=np.expand_dims(imagedump1,axis=0)
    imagedump3=np.expand_dims(imagedump2,axis=4)

    output=model.predict(imagedump3)
    loss=mean_squared_loss(imagedump3,output)
    
    
    if frame1.any()==None:
        print("none")    
        
          
    if loss>0.09:
        print('Abnormal Event Detected')
        cv2.putText(frame1,"Abnormal Event",(100,80),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),4)
    

   
    imagedump.append(image1)
    if(cv2.waitKey(1)==ord('q')):
        break
  
cap.release()
cv2.destroyAllWindows()









































