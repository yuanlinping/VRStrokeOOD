import numpy as np

data = np.loadtxt('./stroke/3DMadLab_processed.csv',delimiter=',',usecols=(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21),skiprows=1)
# print(a.shape)
print(data.shape)
data = data.reshape(int(int(data.shape[0])/64),64,-1)
data = data.transpose((0,2,1))
print(data.shape)
feature = data[:,0:2,:]
feature_add = data[:,6:24,:]
print(data.shape)
feature = np.concatenate((feature,feature_add),axis=1)
print(f'feature.shape:{feature.shape}')
# label = data[:,4:6,:]
sy = np.zeros((data.shape[0],))
z = data[:,2,:]
t = data[:,3,:]
z_class = data[:,25,0]

z_random_class = np.random.randint(10,size = z_class.shape)
print(f'z_random:{z_random_class.shape}')
print(t.shape)
class_labels =data[:,4,0] -1
random_class = np.random.randint(40,size = class_labels.shape)
print(f'z_random:{random_class}')
person_labels = data[:,5,0] -1
np.save('./stroke/stroke_x.npy',feature)
print('save x to ./stroke/stroke_x.npy')

np.save('./stroke/stroke_z.npy',z)
print('save y to ./stroke/stroke_z.npy')

np.save('./stroke/stroke_zclass.npy',z_class)
print(np.unique(z_class))
print('save y to ./stroke/stroke_zclass.npy')

np.save('./stroke/stroke_t.npy',t)
print('save y to ./stroke/stroke_t.npy')

np.save('./stroke/stroke_person.npy',person_labels)
print('save y to ./stroke/stroke_person.npy')

np.save('./stroke/stroke_class.npy',class_labels)
print('save y to ./stroke/stroke_class.npy')

np.save('./stroke/stroke_sy.npy',sy)
print('save y to ./stroke/stroke_sy.npy')

np.save('./stroke/stroke_zrandom.npy',z_random_class)
print('save y to ./stroke/stroke_zrandom.npy')

np.save('./stroke/stroke_classrandom.npy',random_class)
print('save y to ./stroke/stroke_classrandom.npy')
