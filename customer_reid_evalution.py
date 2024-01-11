import numpy as np
from pathlib import Path
import tqdm

import argparse
from transformers import CLIPProcessor, CLIPModel,CLIPVisionModel
from PIL import Image
import os
import torch
import torch.nn as nn
from transformers import (
    VisionTextDualEncoderModel,
    VisionTextDualEncoderProcessor,
    AutoImageProcessor,
    AutoTokenizer,
)
from transformers import AutoImageProcessor, Swinv2Model

parser = argparse.ArgumentParser(description='Test')
parser.add_argument('--usePretrain', default=False, type=bool)
parser.add_argument('--test_dir',default='./rawdata/Market-1501-v15.09.15/pytorch',type=str, help='./test_data')
parser.add_argument('--name', default='clip-vit-base-patch16-finetuned', type=str, help='save model path')
parser.add_argument('--useSwin', default=False, type=bool)

opt = parser.parse_args()


class ReidEvalution:
    def __init__(
        self, query_dir: str, gallery_dir: str, threshold=0.75, result_path=None,
    model=None, processor=None):
        self.query_path = Path(query_dir)
        self.gallery_path = Path(gallery_dir)
        self.threshold = threshold
        
        self.result_path = result_path
        if self.result_path is None:
            self.result_path = Path.cwd() / "reid_result.txt"
        
        self.model = model
        self.processor = processor
        
        print("Extracting lib features ...")
        self.all_hb_lib_feats, self.index_to_person_id_map = self.get_gallery_feat()
        print("Extracting lib features done")


    def get_gallery_feat(self):
        all_hb_feat = {}
        hb_image_index = {}
        index_to_person_id_map = {}  # {hb_name: {index: person_id}}
        gallery_image_paths = list(self.gallery_path.glob("*/reidpic/*/*.jpg")) + list(
            self.gallery_path.glob("*/reidpic/*/*.png")
        )
        for img_path in tqdm.tqdm(gallery_image_paths):
            hb_name = img_path.parent.parent.parent.name
            if hb_name not in all_hb_feat:
                all_hb_feat[hb_name] = []
            if hb_name not in index_to_person_id_map:
                index_to_person_id_map[hb_name] = {}
            if hb_name not in hb_image_index:
                hb_image_index[hb_name] = 0

            person_id = img_path.parent.name
            feat = self.extract_reid_feat(img_path)
            if feat is None:
                continue
            all_hb_feat[hb_name].append(feat)
            image_index = hb_image_index[hb_name]
            index_to_person_id_map[hb_name][image_index] = person_id
            hb_image_index[hb_name] += 1

        for hb_name in all_hb_feat:
            all_hb_feat[hb_name] = np.concatenate(all_hb_feat[hb_name], axis=0)
        return all_hb_feat, index_to_person_id_map

    def test_all_videos(self):
        tp = 0
        fp = 0
        gt = 0
        with open(self.result_path, "w") as f_result:
            for hb3_dir in self.query_path.iterdir():
                print(f"Processing hb3: {hb3_dir}")
                video_dirs = list(hb3_dir.glob("*"))
                for video_dir in tqdm.tqdm(video_dirs):
                    if not video_dir.is_dir():
                        print(f"{video_dir} is not dir")
                        continue
                    gt += 1
                    video_result, pred_person_ids, labels = self.test_one_video(
                        video_dir
                    )
                    result_line = f"{video_dir.name} pred_person_ids: {pred_person_ids} labels: {labels}\n"
                    f_result.write(result_line)
                    if video_result == "true":
                        tp += 1
                    elif video_result == "false":
                        fp += 1
        print("Precision: {:d} / {:d} = {:.2f}".format(tp, tp + fp, tp / (tp + fp)))
        print("Recall: {:d} / {:d} = {:.2f}".format(tp, gt, tp / gt))

    def test_one_video(self, video_dir: Path):
        hb_name = video_dir.parent.name
        lib_feats = self.all_hb_lib_feats[hb_name]
        index_to_person_id_map = self.index_to_person_id_map[hb_name]
        video_result = "null"  # null, true, false
        pred_ids = []
        labels = []
        for track_dir in video_dir.iterdir():
            if not track_dir.is_dir():
                print(f"track_dir: {track_dir} is not dir")
                continue
            
            try:
                # 尝试将目录名解析为整数标签
                label = int(track_dir.name.split("_")[-1].strip("_"))
            except ValueError:
                # 如果转换失败，返回-1
                label = -1
                print(track_dir)
            labels.append(label)
            track_images = list(track_dir.glob("*.jpg")) + list(track_dir.glob("*.png"))
            if len(track_images) == 0:
                print(f"track_dir: {track_dir} is empty")
                continue
            track_feats = self.extract_reid_feat_batch(track_images)
            distance_matric = (np.dot(track_feats, lib_feats.T) + 1) / 2
            # find the max score and index
            max_score = np.max(distance_matric)
            if max_score >= self.threshold:
                max_index = np.where(distance_matric == max_score)
                max_col_index = max_index[1][0]
                if max_col_index not in index_to_person_id_map:
                    print(
                        f"max_col_index: {max_col_index} is not in index_to_person_id_map"
                    )
                    print(index_to_person_id_map)
                    continue
                pred_person_id = int(index_to_person_id_map[max_col_index])
                pred_ids.append(pred_person_id)
                if label == pred_person_id:
                    video_result = "true"
                else:
                    # only one track is wrong, the video is false
                    video_result = "false"
                    break
        return video_result, pred_ids, labels

    def extract_reid_feat(self, img_path) -> np.ndarray:
        images = Image.open(img_path)
        inputs = self.processor(images=images, return_tensors="pt").to('cuda')
        outputs = self.model(**inputs)
        feat = outputs.pooler_output
        feat_np = feat.cpu().detach().numpy()
        
        return feat_np
        # raise NotImplementedError(
        #     "Please implement this function, return a numpy array. For example: np.random.rand(1, 512)"
        # )

    def extract_reid_feat_batch(self, img_path_list) -> np.ndarray:
        feat_list = []
        for img_path in img_path_list:
            feat = self.extract_reid_feat(img_path)
            if feat is not None:
                feat_list.append(feat)
        return np.concatenate(feat_list, axis=0)

    def l2norm(self, feat: np.ndarray) -> np.ndarray:
        feat /= np.linalg.norm(feat, axis=1).reshape(-1, 1)
        return feat


if __name__ == "__main__":
    save_path = os.path.join('./model',opt.name) 
    if opt.usePretrain:
        save_path = opt.name
    if not opt.useSwin:
        model = CLIPModel.from_pretrained(save_path)
        processor = CLIPProcessor.from_pretrained(save_path)
        vision_model = CLIPVisionModel.from_pretrained(save_path)
    else:
        if opt.usePretrain: 
            processor = AutoImageProcessor.from_pretrained(save_path)
            vision_model = Swinv2Model.from_pretrained(save_path)
        else:
            tokenizer = AutoTokenizer.from_pretrained("roberta-base")
            processor =  AutoImageProcessor.from_pretrained(save_path)
            processor_ = VisionTextDualEncoderProcessor(processor, tokenizer)
            Model = VisionTextDualEncoderModel.from_pretrained(save_path)
            vision_model = Model.vision_model
    if torch.cuda.device_count() > 1:
        print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
        vision_model = nn.DataParallel(vision_model)
    vision_model.to('cuda')
    
    anker_reid_evalutor = ReidEvalution(
        query_dir="rawdata/customer/images/", gallery_dir="rawdata/customer/libs/",model=vision_model,processor=processor
    )
    anker_reid_evalutor.test_all_videos()
