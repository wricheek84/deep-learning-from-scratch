import numpy as np
def cross_entropy(pred,ans):
    pred=np.clip(pred,1e-1,1-1e-15)
    return -np.sum(ans*np.log(pred)/ans.shape[0])
rows=np.random.randint(1,100)
cols=np.random.randint(1,100)
random_val=np.random.randint(rows,cols)
exp_x=np.exp(random_val-np.max(random_val,axis=-1,keepdims=True))
soft_x=exp_x/np.sum(exp_x,axis=-1,keepdims=True)
sim_pred=np.zeros((rows,cols))
for r in range (rows):
    sim_pred[r][np.random.randint(0,cols)]=1
loss=cross_entropy(soft_x,sim_pred)
print(loss)