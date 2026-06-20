import numpy as np
def softmax(x):
    exp_x=np.exp(x-np.max(x,axis=-1,keepdims=True))
    return exp_x/np.sum(exp_x,axis=-1,keepdims=True)
rows=np.random.randint(1,100)
cols=np.random.randint(1,100)
x=np.random.rand(rows,cols)
print(softmax(x))