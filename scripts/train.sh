CUDA_VISIBLE_DEVICES="4,5,6,7" python train.py --data data/widerface.yaml --cfg models/yolov5s.yaml --weights /mnt/cephfs/home/dengzeshuai/code/Detection/CodeDetection/yolov5_obb/pretrain/yolov5s.pt --name yolov5s_coco_pre_widerface_b64 --batch-size 64

CUDA_VISIBLE_DEVICES="4,5,6,7" python train.py --data data/widerface.yaml --cfg models/yolov5l-e-5x5.yaml --weights /mnt/cephfs/home/dengzeshuai/code/Detection/CodeDetection/yolov5_obb/pretrain/v5le-coco-5x5-baseline.pt --name yolov5le_5x5_coco_pre_p800b128 --batch-size 128 --workers 16

CUDA_VISIBLE_DEVICES="0,1" python train.py --data data/widerface.yaml --cfg models/yolov5l-e-5x5-stem.yaml --weights /mnt/cephfs/home/dengzeshuai/code/Detection/CodeDetection/yolov5_obb/pretrain/v5le-coco-5x5-baseline.pt --name yolov5le_5x5_stem_coco_pre_p800b128 --batch-size 64 --workers 8
