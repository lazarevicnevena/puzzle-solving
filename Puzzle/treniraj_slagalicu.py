import tensorflow as tf
import numpy as np
import pickle
import random
import cv2

from podaci import *
from tensorflow.examples.tutorials.mnist import input_data



def model_neuronske_mreze(data):
    hidden_1_layer = {'weights': tf.Variable(tf.random_normal([len(train_x[0]), n_nodes_hl1])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                    'biases': tf.Variable(tf.random_normal([n_classes]))}

    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']

    return output

def treniraj_neuronsku_mrezu(x):
    prediction = model_neuronske_mreze(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            print('Epoch started:', epoch + 1)
            epoch_loss = 0
            i=0
            proc=0
            train_x, train_y = zip(*cr)
            while i < len(train_x):
                if proc!=procenat(i,len(train_x)):
                    proc=procenat(i,len(train_x))
                start=i
                end=i+batch_size
                batch_x=np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
                epoch_loss += c
                i+=batch_size

            print('Epoch', epoch+1, 'completed out of', hm_epochs, 'loss:', epoch_loss)
            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            saver.save(sess, "/tmp/model.ckpt")
            print('Accuracy:', accuracy.eval({x: test_x, y: test_y}))

#odgovor=input("Da li zelite da trenirate?")

def pronadji_resenje(vektor1):
    povratno_resenje=[0]*len(vektor1[0])
    for i in range(1):
        rez,drugo=test(vektor1)
        indeks=rez[0]
        povratno_resenje[indeks]+=1
    print(povratno_resenje)
    max_el=povratno_resenje[0]
    max_index=0
    for i in range(len(povratno_resenje)):
        if max_el<povratno_resenje[i]:
            max_el=povratno_resenje[i]
            max_index=i

    return  max_index


if __name__ == '__main__':
    with open('slike_treniranje.pickle', 'rb') as f:
        x = pickle.load(f)
    train_x, train_y, test_x, test_y = x
    cr = list(zip(train_x, train_y))
    random.shuffle(cr)
    train_x, train_y = zip(*cr)
    # print(len(train_x[0]), len(train_y))
    n_nodes_hl1 = 800
    n_nodes_hl2 = 800
    n_nodes_hl3 = 800
    v1 = tf.get_variable("v1", shape=[3])
    saver = tf.train.Saver()
    n_classes = 5
    batch_size = 100
    odgovor = 'ne'
    hm_epochs = 50
    x = tf.placeholder('float', [None, len(train_x[0])])
    y = tf.placeholder('float')
    treniraj_neuronsku_mrezu(x)
