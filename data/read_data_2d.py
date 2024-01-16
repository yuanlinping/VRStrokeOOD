import numpy as np

data = np.loadtxt('./stroke2d/DigiLeTs_Digits_combined.csv',delimiter=',',usecols=(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20),skiprows=1)
# a = np.load('./act/emg/emg_x.npy')
# print(a.shape)
print(data.shape)
data = data.reshape(int(int(data.shape[0])/64),64,-1)
data = data.transpose((0,2,1))
print(data.shape)
feature = data[:,0:2,:]#xy coordinates
feature_add = data[:,5:23,:] # other features
print(data.shape)
feature = np.concatenate((feature,feature_add),axis=1)
print(f'feature.shape:{feature.shape}')
# label = data[:,4:6,:]
sy = np.zeros((data.shape[0],))
t = data[:,2,:]

print(t.shape)
class_labels =data[:,3,0] -1
print(f'class:{np.unique(class_labels)}')
random_class = np.random.randint(10,size = class_labels.shape)
print(f'random_class:{random_class}')
person_labels = data[:,4,0] -1
print(f"person:{np.unique(person_labels)}")

np.save('./stroke2d/stroke2d_x.npy',feature)
print('save to ./stroke2d/stroke2d_x.npy')
print(feature.shape)

np.save('./stroke2d/stroke2d_t.npy',t)
print('save to ./stroke2d/stroke2d_t.npy')

np.save('./stroke2d/stroke2d_person.npy',person_labels)
print('save to ./stroke2d/stroke2d_person.npy')

np.save('./stroke2d/stroke2d_class.npy',class_labels)
print('save to ./stroke2d/stroke2d_class.npy')

np.save('./stroke2d/stroke2d_zrandom.npy',random_class)
print('save to ./stroke2d/stroke2d_zrandom.npy')

np.save('./stroke2d/stroke2d_classrandom.npy',random_class)
print('save to ./stroke2d/stroke2d_classrandom.npy')