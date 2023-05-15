export PYTHONPATH=$PYTHONPATH:/home/ciccio/Desktop/catkin_ws/src/Demo/graspnet/graspnet-baseline/models
CUDA_VISIBLE_DEVICES=0 python3 demo.py --checkpoint_path /home/ciccio/Desktop/catkin_ws/src/Demo/graspnet/graspnet-baseline/logs/logs_rs/checkpoint-rs.tar
