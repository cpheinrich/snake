import numpy as np
from PIL import Image

IMG_SIZE = None


# Functions
def preprocess_observation(obs):
    global IMG_SIZE
    # Convert to gray-scale and resize it
    image = Image.fromarray(obs, 'RGB')
    #image.save('./sample-raw','JPEG')
    image = Image.fromarray(obs, 'RGB').convert('L').resize(IMG_SIZE)
    # Convert image to array and return it
    #image.save('./sample-preprocess','JPEG')
    new_obs = np.asarray(image.getdata(), dtype=np.uint8).reshape(image.size[1],
                                                               image.size[0])
    #print(new_obs)
    return new_obs


def get_next_state(current, obs):
    # Next state is composed by the last 3 images of the previous state and the
    # new observation
    return np.append(current[1:], [obs], axis=0)

def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray
