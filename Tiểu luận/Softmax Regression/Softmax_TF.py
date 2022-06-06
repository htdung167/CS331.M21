import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def generate_dataset(size, noise=0.5):

    label= np.zeros(size)

    x_l_t= (np.random.rand(size)+1)
    y_l_t= x_l_t + 2 + np.random.rand(size)*noise

    x_l_b= (np.random.rand(size)+1)
    y_l_b= x_l_t + 1+ np.random.rand(size)*noise

    x_r_t= (np.random.rand(size)+2)
    y_r_t= x_l_t + 2 + np.random.rand(size)*noise

    x_r_b= (np.random.rand(size)+2)
    y_r_b= x_l_t + 1+ np.random.rand(size)*noise

    x= np.append(x_l_t,[x_l_b,x_r_t,x_r_b])

    y= np.append(y_l_t,[y_l_b,y_r_t,y_r_b])

    labels= np.append(label+0,[label+1,label+2,label+3])

    x = x.reshape(len(x), 1)
    y = y.reshape(len(y), 1)
    labels = labels.reshape(len(labels), 1)

    data = np.hstack((x, y))
    return data, labels

def loss_f(X,W,Y):
    h= tf.matmul(X,W)
    entropy = tf.nn.softmax_cross_entropy_with_logits(logits=h, labels=Y, name='loss')
    loss= tf.reduce_mean(entropy) 
    return loss

def optimizer(lr):
    return tf.optimizers.SGD(learning_rate=lr)

def fit(X,Y,W,iters=25,lr=0.1):
    for i in range (iters):
        with tf.GradientTape(persistent=True) as g:
            loss= loss_f(X,W,Y)
            gradients=g.gradient(loss,[W])
            optimize=optimizer(lr)
            optimize.apply_gradients(zip(gradients,[W]))
    return W

def visual(W,X,Y,ax):
    for i in range (4):
        temp=[]
        for j in range(len(Y)):
            if Y[j]==i:
                temp.append(X[j])
        x_vis= np.array([min(X[:,0]),max(X[:,0])])    
        y_vis= -(W[1,i]*x_vis+W[0,i])/ W[2,i]
        ax.plot(x_vis,y_vis)

X,Y=generate_dataset(25,0.5)
Y= Y.reshape(len(Y))
Y_onehot=tf.one_hot(Y,4)
#print(Y_onehot.numpy())
bias= np.ones((len(X),1))
X_bias= np.hstack((bias,X))
X_bias_tf=tf.cast(X_bias,tf.float32)

W = tf.Variable(tf.random.normal((3,4)), trainable = True, name='weight')

iter=10000
lr=0.01


W= fit(X_bias_tf,Y_onehot,W,iters=iter,lr=lr).numpy()
print(W)

fig1, ax = plt.subplots()
ax.scatter(X[:,0], X[:,1],c=Y[:])
visual(W,X,Y,ax)
plt.show()