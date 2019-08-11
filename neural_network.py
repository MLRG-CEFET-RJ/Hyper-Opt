import datetime
opening = datetime.datetime.now() #execution time start
import os, logging
import tensorflow as tf
import numpy as np
import data_utils as du
import sys
import math
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

'''
System args: LAYER1, LAYER2, LAYER3, LR, BETA
LAYER1 = Neurons number in the first hidden layer
LAYER2 = Neurons number in the second hidden layer
LAYER3 = Neurons number in the third hidden layer
LR = learning rate
BETA = beta parameter to regularization. 0 to ignore
Example: python neural_network.py 100 200 300 0.001 0
'''

#Train/dev/test path whitespace separated file without header. Target in the last column (data_utils)
train_path =  "/home/mserqueira/BRKGA/src/python/datasets/mnist_train.csv"
test_path = "/home/mserqueira/BRKGA/src/python/datasets/mnist_val.csv"

epochs_no = 300 #Number of epochs
batch_size = 32 #Batch size
patience = 13 #Number of trials to reduce epoch loss based on the min_delta
min_delta = 0.01 #min MSE reduction in each epoch
patience_cnt = 0
prior = float("inf")

trainX, trainY, predX, predY, n_classes = du.csv_to_numpy_array(train_path, test_path)
num_x = trainX.shape[1]
num_y = trainY.shape[1]
learning_rate = float(sys.argv[4])
beta = float(sys.argv[5])

x = tf.placeholder(tf.float32, [None, num_x])
y = tf.placeholder(tf.float32, [None, num_y])
lr = tf.placeholder(tf.float32)

def neural_network(LAYER1, LAYER2, LAYER3):
   with tf.device('/device:GPU:0'):
    units_layer1 = int(LAYER1)
    units_layer2 = int(LAYER2)
    units_layer3 = int(LAYER3)
    
    W1 = tf.Variable(tf.random_normal([num_x, units_layer1]))
    b1 = tf.Variable(tf.random_normal([units_layer1]))
    
    W2 = tf.Variable(tf.random_normal([units_layer1, units_layer2]))
    b2 = tf.Variable(tf.random_normal([units_layer2]))
    
    W3 = tf.Variable(tf.random_normal([units_layer2, units_layer3]))
    b3 = tf.Variable(tf.random_normal([units_layer3]))
    
    W_out = tf.Variable(tf.random_normal([units_layer3, n_classes]))
    b_out = tf.Variable(tf.random_normal([n_classes]))
    
    Z1 = tf.add(tf.matmul(x, W1), b1)
    A1 = tf.nn.relu(Z1)
    
    Z2 = tf.add(tf.matmul(A1, W2), b2)
    A2 = tf.nn.relu(Z2)
    
    Z3 = tf.add(tf.matmul(A2, W3), b3)
    A3 = tf.nn.relu(Z3)
    
    Z_out = tf.matmul(A3, W_out) + b_out

    return Z_out, W_out

def models_predictions(prediction, sess):
  pred_model =  tf.argmax(prediction, 1)
  pred_model = sess.run(pred_model, feed_dict={x:predX, y:predY})
  pred_true =  tf.argmax(y, 1)
  pred_true = sess.run(pred_true, feed_dict={x:predX, y:predY})

  train_model = tf.argmax(prediction, 1)
  train_model = sess.run(train_model, feed_dict={x:trainX, y:trainY})
  train_true =  tf.argmax(y, 1)
  train_true = sess.run(train_true, feed_dict={x:trainX, y:trainY})

  return pred_model, pred_true, train_model, train_true

def early_stop(epoch, epoch_loss):
  global patience_cnt
  global prior
  if epoch > 0 and prior - epoch_loss > min_delta:
    patience_cnt = 0
    flag = False
  else:
    patience_cnt += 1
    flag = False
  if patience_cnt > patience:
    flag = True
  prior = epoch_loss
  return flag

def learning_plot(loss_train, loss_test):
  epoch_plot = range(0,len(loss_train))
  epoch_plot
  loss_test
  loss_train
  plt.plot(epoch_plot, loss_train, label="Train error")
  plt.plot(epoch_plot, loss_test, label="Test error")
  plt.legend(loc='lower left', bbox_to_anchor= (0.0, 1.01), ncol=2, 
    borderaxespad=0, frameon=False)
  #plt.legend(loc='upper right')
  plt.ylabel('Error')
  plt.xlabel('Epoch')
  plt.savefig('learning_plot.pdf')
  pass

def get_error(prediction, y, sess, labelX, labelY, x):
  correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
  error = 1-sess.run(accuracy, feed_dict={x: labelX, y: labelY})

  return error

def build_graph():    
    prediction, weights = neural_network(sys.argv[1], sys.argv[2], sys.argv[3])
    
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y)) #Ou reduce_sum
    regularizer = tf.nn.l2_loss(weights) #L2
    loss = tf.reduce_mean(loss + beta * regularizer)
    
    optimizer = tf.train.AdamOptimizer(lr).minimize(loss)
    loss_train = []
    loss_test = []

    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      for epoch in range(epochs_no):
        epoch_loss = 0
        i=0
        while i < len(trainX):
          start = i
          end = i+batch_size
          batch_x = np.array(trainX[start:end])
          batch_y = np.array(trainY[start:end])
          _, c = sess.run([optimizer, loss], feed_dict={x: batch_x, y: batch_y, lr: learning_rate})
          epoch_loss += c
          i+=batch_size
        
        early_flag = early_stop(epoch, epoch_loss)
        if early_flag == False:
          print('Epoch ', epoch+1, 'of ', epochs_no, '| loss: ', epoch_loss)
        else:
          print("Early stopping... ", 'Epoch', epoch+1, 'of', epochs_no, '| loss:', epoch_loss)
          break
        loss_train.append(get_error(prediction, y, sess, trainX, trainY, x))
        loss_test.append(get_error(prediction, y, sess, predX, predY, x))

      pred_model, pred_true, train_model, train_true = models_predictions(prediction, sess)
      prec, rec, f1, acc = du.nn_performance_metrics(pred_model, pred_true, train_model, train_true)

      sess.close()
      final = datetime.datetime.now()
      time = final-opening
      print("\nExecution time: ", time)
      learning_plot(loss_train, loss_test)
      return acc

def main():
  acc = build_graph()
  return 0

if __name__ == "__main__":
  main()