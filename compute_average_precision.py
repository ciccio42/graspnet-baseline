import sys
import numpy as np
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--dump_dir', type=str, default="/home/ciccio/Desktop/Tesi/graspnet-baseline/logs/dump_rs_test_3/accuracy", help='Dump dir to save outputs')
parser.add_argument('--split', type=str, default="test", help='Split of test set')
cfgs = parser.parse_args()

TEST = range(100, 190)
TEST_SEEN = range(100, 130)
TEST_SIMILAR = range(130, 160)
TEST_NOVEL = range(160, 190)

if cfgs.split == "test_seen":
    scene_accuracy = np.zeros((len(TEST_SEEN), 256, 50, 6))
    print(f"Scene accuracy shape: {scene_accuracy.shape}")
    for scene_indx, scene_id in enumerate(TEST_SEEN):
        for scene_ann_indx, scene_ann_id in enumerate(range(0, 256)):
            scene_ann_accuracy_file = cfgs.dump_dir + "/scene_%04d/%04d.npy" % (scene_id, scene_ann_id)
            try:
                scene_accuracy[scene_indx][scene_ann_indx] = np.load(scene_ann_accuracy_file)
            except:
                scene_accuracy[scene_indx][scene_ann_indx] = 0.0              
    print(f"TEST SEEN: Average Precision: {100*np.mean(scene_accuracy)}, Average Precision 0.8: {100*np.mean(scene_accuracy[:,:,:,3])}, Average Precision 0.4: {100*np.mean(scene_accuracy[:,:,:,1])}")

elif cfgs.split == "test_similar":
    scene_accuracy = np.zeros((len(TEST_SIMILAR), 256, 50, 6))
    print(f"Scene accuracy shape: {scene_accuracy.shape}")
    for scene_indx, scene_id in enumerate(TEST_SIMILAR):
        for scene_ann_indx, scene_ann_id in enumerate(range(0, 256)):
            scene_ann_accuracy_file = cfgs.dump_dir + "/scene_%04d/%04d.npy" % (scene_id, scene_ann_id)
            try:
                scene_accuracy[scene_indx][scene_ann_indx] = np.load(scene_ann_accuracy_file)
            except:
                scene_accuracy[scene_indx][scene_ann_indx] = 0.0      
    print(f"TEST SIMILAR: Average Precision: {100*np.mean(scene_accuracy)}, Average Precision 0.8: {100*np.mean(scene_accuracy[:,:,:,3])}, Average Precision 0.4: {100*np.mean(scene_accuracy[:,:,:,1])}")

elif cfgs.split == "test_novel":
    scene_accuracy = np.zeros((len(TEST_NOVEL), 256, 50, 6))
    print(f"Scene accuracy shape: {scene_accuracy.shape}")
    for scene_indx, scene_id in enumerate(TEST_NOVEL):
        for scene_ann_indx, scene_ann_id in enumerate(range(0, 256)):
            scene_ann_accuracy_file = cfgs.dump_dir + "/scene_%04d/%04d.npy" % (scene_id, scene_ann_id)
            try:
                scene_accuracy[scene_indx][scene_ann_indx] = np.load(scene_ann_accuracy_file)
            except:
                scene_accuracy[scene_indx][scene_ann_indx] = 0.0
    print(f"TEST NOVEL: Average Precision: {100*np.mean(scene_accuracy)}, Average Precision 0.8: {100*np.mean(scene_accuracy[:,:,:,3])}, Average Precision 0.4: {100*np.mean(scene_accuracy[:,:,:,1])}")

elif cfgs.split == "test":
    scene_accuracy = np.zeros((len(TEST), 256, 50, 6))
    print(f"Scene accuracy shape: {scene_accuracy.shape}")
    for scene_indx, scene_id in enumerate(TEST):
        for scene_ann_indx, scene_ann_id in enumerate(range(0, 256)):
            scene_ann_accuracy_file = cfgs.dump_dir + "/scene_%04d/%04d.npy" % (scene_id, scene_ann_id)
            try:
                scene_accuracy[scene_indx][scene_ann_indx] = np.load(scene_ann_accuracy_file)
            except:
                scene_accuracy[scene_indx][scene_ann_indx] = 0.0
    print(f"TEST: Average Precision: {100*np.mean(scene_accuracy[:,:,:,:])}, Average Precision 0.8: {100*np.mean(scene_accuracy[:,:,:,3])}, Average Precision 0.4: {100*np.mean(scene_accuracy[:,:,:,1])}")
    print(f"TEST SEEN: Average Precision: {100*np.mean(scene_accuracy[:30,:,:,:])}, Average Precision 0.8: {100*np.mean(scene_accuracy[:30,:,:,3])}, Average Precision 0.4: {100*np.mean(scene_accuracy[:30,:,:,1])}")
    print(f"TEST SIMILAR: Average Precision: {100*np.mean(scene_accuracy[30:60,:,:,:])}, Average Precision 0.8: {100*np.mean(scene_accuracy[30:60,:,:,3])}, Average Precision 0.4: {100*np.mean(scene_accuracy[30:60,:,:,1])}")
    print(f"TEST NOVEL: Average Precision: {100*np.mean(scene_accuracy[60:90,:,:,:])}, Average Precision 0.8: {100*np.mean(scene_accuracy[60:90,:,:,3])}, Average Precision 0.4: {100*np.mean(scene_accuracy[60:90,:,:,1])}")

