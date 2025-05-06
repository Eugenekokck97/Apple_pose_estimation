from mmcv import Config
from mmpose.apis import inference_bottom_up_pose_model, init_pose_model, vis_pose_result
from utils import localize_apple, orientation_computation, visualizer_3d

import argparse
import os
import cv2
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='Apple pose detection using MMPose')
    parser.add_argument('--config', default='config.py', help='Path to config file')
    parser.add_argument('--checkpoint', default='model/best.pth', help='Path to pose checkpoint')
    parser.add_argument('--data_folder', default='data', help='Path to data folder')
    parser.add_argument('--visualize', action='store_true', help='Whether to visualize the results')
    return parser.parse_args()

def main():
    args = parse_args()

    # Load config and model
    cfg = Config.fromfile(args.config)
    pose_model = init_pose_model(cfg, args.checkpoint)

    # Set up folders
    image_folder = os.path.join(args.data_folder, 'images')
    mask_folder = os.path.join(args.data_folder, 'masks')

    for id in os.listdir(image_folder):
        img_id = os.path.join(image_folder, id)
        mask_id = os.path.join(mask_folder, id)

        # Perform pose inference
        pose_results, _ = inference_bottom_up_pose_model(
            pose_model,
            img_id,
            dataset='BottomUpCocoDataset',
            dataset_info=None,
            pose_nms_thr=0.9,
            return_heatmap=False,
            outputs=None
        )

        if pose_results:
            # Update keypoints with scores
            for pose in pose_results:
                for i in range(len(pose["keypoints"])):
                    pose["keypoints"][i][2] = pose["score"][i]

            # Visualize pose results
            vis_result = vis_pose_result(
                pose_model,
                img_id,
                pose_results,
                kpt_score_thr=0.2,
                dataset='BottomUpCocoDataset',
                show=False
            )

            # Localize apple
            center, radius = localize_apple(mask_id)

            # Compute orientation
            vec, _type = orientation_computation(center, radius, pose_results[0], 0.3)
            if _type == 'stem':
                vec *= -1
            
            # Visualization
            if args.visualize:
                img = cv2.circle(vis_result, center, radius, (0, 255, 0), 1)
                cv2.imshow('result', img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                visualizer_3d(vec, _type, radius)

if __name__ == '__main__':
    main()