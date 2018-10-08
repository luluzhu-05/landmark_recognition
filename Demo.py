
# coding: utf-8

# In[5]:


file_name = input('please input file name:')


# In[2]:


from keras.preprocessing import image
import numpy as np

# file_path = '/Users/lulu/Desktop/landmark/demo/6b847c554504d347.png'  
  
img = image.load_img('/Users/lulu/Desktop/landmark/demo/{}'.format(file_name), target_size=(224, 224))  
x = image.img_to_array(img)  
x = np.expand_dims(x, axis=0) 


# In[ ]:


print('------------- finish loading picture -------------------')


# In[7]:


# load best model trained 
from keras.models import load_model
model = load_model('/Users/lulu/Desktop/landmark/demo/vgg16.h5')


# In[ ]:


label_map_ = {
    'Kurhaus of Scheveningen':0,
    'Rosary Basilica':1,
    'Innsbruck':2,
    'Chepstow Castle':3,
    'Tiger Leaping Gorge':4,
    "Clifford's Tower, York":5,
    "saint peter's basilica, statue of saint gregory the illuminator":6,
    "statue of decebalus":7,
    "Galleria Borghese":8,
    "rotunda of mosta":9,
    "baochu pagoda":10,
    "latvian academy of sciences":11,
    "delaware memorial bridge":12,
    "St. Francis Xavier Chapel":13,
    "vall de núria":14,
    "Eltz Castle":15,
    "chinatown":16,
    "lácar lake":17,
    "london coliseum":18,
    "lubart's castle":19
    
}


# In[11]:


# predict
y_prob = model.predict(x) 
y_classes = list(label_map_.keys())[int(y_prob.argmax(axis=-1))]

print('------------- finish prediction -------------------')

print('the landmark in picture is:', y_classes)

