# CUDA_VISIBLE_DEVICES=1 \
python test_widerface.py --weights runs/train/yolov5s_coco_pre_widerface_b64/weights/best.pt --img-size 640 --device 1 --save_folder ./widerface_evaluate/widerface_txt 

python test_widerface.py --weights runs/train/yolov5le_5x5_wopre_p800b64/weights/best.pt --img-size 640 --device 3 --save_folder ./widerface_evaluate/widerface_v5le_5x5_wopre/


python test_widerface.py --weights runs/train/yolov5le_wopre_p800b64/weights/best.pt --img-size 640 --device 1 --save_folder ./widerface_evaluate/widerface_v5le

python test_widerface.py --weights runs/train/yolov5le_patch_wopre_p800b64/weights/best.pt --img-size 640 --device 4 --save_folder ./widerface_evaluate/widerface_v5le_patch 

python test_widerface.py --weights runs/train/yolov5le_5x5_stem_coco_pre_p800b64/weights/best.pt --img-size 640 --device 7 --save_folder ./widerface_evaluate/widerface_v5le_5x5_stem_coco_pre