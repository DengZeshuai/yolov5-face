# bash evaluation.sh | tee -a ./test_logs/test_0831.log
root="/mnt/cephfs/home/dengzeshuai/code/Detection/yolov5-face"
cd ${root}/widerface_evaluate

# echo python evaluation.py -p widerface_txt/ -g ground_truth/
# python evaluation.py -p widerface_txt/ -g ground_truth/

echo python evaluation.py -p widerface_v5le_5x5/ -g ground_truth/ 
python evaluation.py -p widerface_v5le_5x5/ -g ground_truth/ 

############ ablation study ##########

echo python evaluation.py -p widerface_v5le_5x5_wopre/ -g ground_truth/ 
python evaluation.py -p widerface_v5le_5x5_wopre/ -g ground_truth/ 

echo python evaluation.py -p widerface_v5le_5x5_patch/ -g ground_truth/ 
python evaluation.py -p widerface_v5le_5x5_patch/ -g ground_truth/ 

echo python evaluation.py -p widerface_v5le_5x5_stem/ -g ground_truth/ 
python evaluation.py -p widerface_v5le_5x5_stem/ -g ground_truth/ 


echo python evaluation.py -p widerface_v5le/ -g ground_truth/ 
python evaluation.py -p widerface_v5le/ -g ground_truth/ 

echo python evaluation.py -p widerface_v5le_patch/ -g ground_truth/ 
python evaluation.py -p widerface_v5le_patch/ -g ground_truth/ 

echo python evaluation.py -p widerface_v5le_5x5_stem_coco_pre/ -g ground_truth/ 
python evaluation.py -p widerface_v5le_5x5_stem_coco_pre/ -g ground_truth/ 

cd $root