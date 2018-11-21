## MEnet 

This repository contains a Dockerfile and scripts to build and run our Metric Expression Network for Salient Object Segmentation in Docker containers. We also provide some example data to test the networks. 


If you use this project or parts of it in your research, please cite the original paper of MEnet:

    @InProceedings{MEnet,
      author       = "Shulian Cai, Jiabin Huang, Delu Zeng, Xinghao Ding, John Paisley",
      title        = "MEnet: A Metric Expression Network for Salient Object Segmentation",
      booktitle    = "Proceedings of the Twenty-Seventh International Joint Conference on Artificial Intelligence(IJCAI)",
      month        = "May",
      year         = "2018",
      url          = "https://doi.org/10.24963/ijcai.2018/83"
    }


See the [paper website](https://www.ijcai.org/proceedings/2018/0083.pdf)for more details.

# 1 Requirements
caffe:gpu


# 2 Train
First, you can use your own datasets to train the MEnet, and the train_data_dir is './0408data/Images_train',
train_labels_dir is './0408data/GT_train'.

After that, you can start training with:
        $python ./tools/training.py 

Models will be saved in './model/snapshot/' every 10000 iters, you can get the final model and solverstate which named 'attention_iter_110000.caffemdel' and 'attention_iter_110000.solverstate' in the same path.

# 3 Test
First, you can change the path 'GV.test_dir' in testing.py where you want to put your test images,
After that, you can start testing with:
        $python ./tools/testing.py 

Test results will be saved in this dir.






