from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy import vstack
from numpy.random import randn
from numpy.random import randint
from keras.datasets.mnist import load_data
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Dropout
from IPython.display import clear_output

import os
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def inv_sigmoid(x):
    return np.log(x/(1-x))


def access_image(img_list,path,length):
    pixels = []
    imgs   = []

    for i in range(length):
        img = Image.open(path+"\\"+img_list[i],'r')
        baseWidht = 100
        img = img.resize((baseWidht,baseWidht),Image.ANTIALIAS)
        pix = np.array(img.getdata())
        pixels.append(pix.reshape(100,100,3))
        imgs.append(img)
    return np.array(pixels),imgs

def show_image(pix_list):
    array = np.array(pix_list.reshape(100,100,3),dtype=np.uint8)
    new_image = Image.fromarray(array)
    new_image.show()

def define_discriminator(in_shape=(100,100,3)):
    model = Sequential()
    model.add(Conv2D(64,(3,3),(2,2),"same",input_shape=in_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))

    model.add(Conv2D(64,(3,3),(2,2),"same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(1,activation='sigmoid'))
    opt = Adam(learning_rate=0.0002,beta_1=0.5)
    model.compile(loss='binary_crossentropy',optimizer=opt,metrics=['accuracy'])
    model.summary()
    return model 

def define_generator(latent_dim):
    model = Sequential()
    n_nodes = 128 * 25  * 25
    model.add(Dense(n_nodes,input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((25,25,128)))

    model.add(Conv2DTranspose(128,(4,4),strides=(2,2),padding="same"))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2DTranspose(128,(4,4),strides=(2,2),padding="same"))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(3,(7,7),padding="same"))
    model.summary()
    return model
    
def define_gan(g_model,d_model):
    d_model.trainable = False
    model = Sequential()
    model.add(g_model)
    model.add(d_model)
    opt = Adam(learning_rate=0.0002,beta_1=0.5)
    model.compile(loss="binary_crossentropy",optimizer=opt)
    model.summary()
    return model

def generate_real_samples(dataset,n_samples):
    ix = randint(0,dataset.shape[0],n_samples)
    X = dataset[ix]
    y = ones((n_samples,1))
    return X,y

def generate_latent_points(latent_dim,n_samples):
    x_input = randn(latent_dim*n_samples)
    x_input = x_input.reshape(n_samples,latent_dim)
    return x_input

def generate_fake_samples(g_model,latent_dim,n_samples):
    x_input = generate_latent_points(latent_dim,n_samples)
    x = g_model.predict(x_input)
    y = zeros((n_samples,1))
    return x,y

def summarize_performance(epoch,g_model,d_model,dataset,latent_dim,n_samples=100):
    x_real, y_real = generate_real_samples(dataset,n_samples)
    _, acc_real = d_model.evaluate(x_real,y_real,verbose=0)
    x_fake, y_fake = generate_fake_samples(g_model, latent_dim, n_samples)
    _, acc_fake = d_model.evaluate(x_fake,y_fake,verbos=0)
    print("Real Accuracy: {:.2f}, Fake Accuracy: {:.2f}".format(acc_real*100,acc_fake*100))
    filename = 'generator_model_{}.h5'.format(epoch+1)
    g_model.save(filename)
    
def train(g_model,d_model,gan_model,dataset,latent_dim,n_epochs=100,n_batch=10):
    bat_per_epo = int(dataset.shape[0]/n_batch)
    print(dataset.shape[0])
    half_batch = int(n_batch/2)

    for i in range(n_epochs):
        for j in range(bat_per_epo):
            x_real, y_real = generate_real_samples(dataset,half_batch)
            x_fake, y_fake = generate_fake_samples(g_model,latent_dim,half_batch)
            x,y = vstack((x_real,x_fake)), vstack((y_real,y_fake))

            d_loss, _ = d_model.train_on_batch(x,y)
            x_gan = generate_latent_points(latent_dim,n_batch)
            y_gan = ones((n_batch,1))
            g_loss = gan_model.train_on_batch(x_gan,y_gan)
            print("For Epoch {}, step {} of {}: d_loss: {:.6f}  gan_loss: {:.6f}".format(i+1,j+1,bat_per_epo,d_loss,g_loss))
        if (i+1)%10 == 0:
            summarize_performance(i,g_model,d_model,dataset,latent_dim)
            clear_output()


path = "image"
os.getcwd()
img_list = os.listdir(path)
latent_dim = 100

print("[INFO] Building Discriminator")
d_model = define_discriminator()

print("[INFO] Building Generator")
g_model = define_generator(latent_dim)


print("[INFO] Building GAN")
gan_model = define_gan(g_model,d_model)


print("[INFO] Accessing Dataset")
pixels,imgs = access_image(img_list,path,100)
print(pixels.shape)

print("[INFO] Training the Model")
train(g_model,d_model,gan_model,np.array(pixels),latent_dim)