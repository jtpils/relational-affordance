import tensorflow as tf
import numpy as np
import h5py
import os 
import random
import sys
import matplotlib.pyplot as plt
from plyfile import (PlyData, PlyElement, make2d, PlyParseError, PlyProperty)



BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '3rd_party/utils'))

import show3d_balls
import rela_aff_model as model


pointclouds_pl, labels_pl = model.placeholder_inputs(1,4000)
is_training_pl = tf.placeholder(tf.bool, shape=())
num_class = 5
sess = tf.Session()
pred, end_points = model.get_model(pointclouds_pl,is_training_pl)
saver = tf.train.Saver(var_list=tf.global_variables())
saver.restore(sess, "log/model.ckpt")

def export_ply(pc, filename):
    vertex = np.zeros(pc.shape[0], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red','uint8'),('green','uint8'),('blue','uint8')])
    for i in range(pc.shape[0]):
        vertex[i] = (pc[i][0], pc[i][1], pc[i][2],pc[i][3], pc[i][4], pc[i][5])
    ply_out = PlyData([PlyElement.describe(vertex, 'vertex', comments=['vertices'])],text=True)
    ply_out.write(filename)


def load_dataset(h5_file):
    f = h5py.File(h5_file)
    data = f['data'][:,:,:4]
    label = f['data'][:,:,4]
    return data, label


def shuffle_data(data,label,batch):
    idx = np.arange(len(data))
    np.random.shuffle(idx)
    return data[idx, ...], label[idx,...]

def predict(data):
    feed_dict = {pointclouds_pl: [data],
                 is_training_pl: False}
    logits = sess.run(pred,feed_dict=feed_dict)
    return np.argmax(logits,2)

if __name__ == '__main__':
    
    data, label = load_dataset("test_set/scene_p12.h5")
    #data, label = load_dataset("scene_h4.h5")
    

    cmap = plt.cm.get_cmap("hsv", num_class)
    cmap = np.array([cmap(i) for i in range(num_class)])[:,:3]
    total_acc = 0
    total_prec = 0
    total_recall = 0
    for i in range(30):
        #random_sample = random.sample(range(len(data)),1)        
        #ps = data[random_sample[0]]
        ps = data[i]



        #seg = label[random_sample[0]]        
        seg = label[i]                
        segp = predict(ps)        

        correct = np.sum(segp == seg)
        segp = segp.squeeze()
        seg = seg.astype(int)

        #debug
        TP = 0
        FP = 0
        FN = 0
        cls_num = 1
        for i in range(seg.shape[0]):
            if seg[i] == cls_num and segp[i] == cls_num:
                TP = TP + 1
            if segp[i] == cls_num  and seg[i] != cls_num:
                FP = FP + 1
            if segp[i] != cls_num and seg[i] == cls_num:
                FN = FN + 1
        if (TP+FP) == 0:
            precision = 0
        else:
            precision = float(TP)/(TP+FP)
        recall = float(TP)/(TP+FN)

        #print "precision", precision
        #print "recall", recall
        total_prec = total_prec + precision
        total_recall = total_recall + recall

        #-------------------
        acc = correct/4000.
        #print acc
        gt = cmap[seg, :]
        result = cmap[segp, :]
        ps_show = ps[:,0:3]
        show3d_balls.showpoints(ps_show, gt, result, ballradius=3)
        total_acc += acc
        fakecolor = np.zeros((ps_show.shape[0],3)) 
        fakecolor[:,1] = 255
        predict_result = np.c_[ps_show,fakecolor]
        gt_result = np.copy(predict_result)
        predict_result[segp > 0.9, 3] = 0
        predict_result[segp > 0.9, 4] = 80
        predict_result[segp > 0.9, 5] = 255
        gt_result[seg > 0.9, 3] = 0
        gt_result[seg > 0.9, 4] = 80
        gt_result[seg > 0.9, 5] = 255


        input_ps = np.c_[ps_show,np.ones((ps_show.shape[0],3))]
        input_ps[ps[:,3]>0,3] = 255
        #export_ply(input_ps,"input.ply")
        #export_ply(predict_result,"predict2h2.ply")
        #export_ply(gt_result,"groundtruth2.ply")
    print "average accuracy",total_acc/30.
    aver_prec = total_prec/30
    aver_recall = total_recall/30
    print "average precision", aver_prec
    print "average recall", aver_recall
    print "F1", (2*aver_prec*aver_recall)/(aver_prec + aver_recall)
