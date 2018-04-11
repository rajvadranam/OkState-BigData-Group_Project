# NeuPY Hopfield Example
# An implementation of Hopefield Network using NeuPy


import numpy as np
from neupy import algorithms
from random import randint


def draw_bin_image(image_mat):
    for row in image_mat.tolist ():
        for val in row:
            if val == 1:
                print ( '* ', end='' )
            else:
                print ( '  ', end='' )

        print ()


def generateNoise(noise_mat):
    noise_mat = np.matrix ( np.copy ( noise_mat ) )

    for row in noise_mat.tolist ():
        # print(row , len(row))
        for i in range ( 0, len ( row ) // 2 ):
            row[i] = 1
            # print(row)

    # print(noise_mat)

    return noise_mat


def displayHelper(images):
    for img in images:
        draw_bin_image ( img.reshape ( (6, 5) ) )
        print ()


zero = np.matrix (
    [0, 1, 1, 1, 0,
     1, 0, 0, 0, 1,
     1, 0, 0, 0, 1,
     1, 0, 0, 0, 1,
     1, 0, 0, 0, 1,
     0, 1, 1, 1, 0
     ] )

one = np.matrix (
    [0, 1, 1, 0, 0,
     0, 0, 1, 0, 0,
     0, 0, 1, 0, 0,
     0, 0, 1, 0, 0,
     0, 0, 1, 0, 0,
     0, 0, 1, 0, 0
     ] )

two = np.matrix ( [
    1, 1, 1, 0, 0,
    0, 0, 0, 1, 0,
    0, 0, 0, 1, 0,
    0, 1, 1, 0, 0,
    1, 0, 0, 0, 0,
    1, 1, 1, 1, 1
] )
print(zero.reshape(6 , 5))


images = [zero, one, two]

data = np.concatenate ( [zero, one, two], axis=0 )
print(data)
displayHelper ( images )

hopfield_net = algorithms.DiscreteHopfieldNetwork ( mode='async', n_times=200 )
hopfield_net.train ( data )

noise_two = np.rot90 ( np.matrix ( np.copy ( two ) ) )
displayHelper ( [noise_two] )

shape = two.shape
noise_two = noise_two.reshape ( shape )
half_two = np.matrix ( [
    0, 1, 1, 0, 0,
    0, 0, 1, 1, 0,
    0, 0, 1, 1, 0,
    0, 0, 1, 0, 0,
    0, 1, 1, 0, 0,
    1, 1, 1, 1, 1
] )
displayHelper ( [half_two] )
print(two.shape , noise_two.shape)
half_two = half_two.reshape ( shape )

predict_pattern = hopfield_net.predict ( half_two )

print ()
print ( 'Recovered pattern....' )
displayHelper ( [predict_pattern] )