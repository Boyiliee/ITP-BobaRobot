def load_model_hf(repo_id, filename, ckpt_config_filename):
    from huggingface_hub import hf_hub_download

    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)
    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    return cache_config_file, cache_file


def get_ovod_model(device="cuda"):
    from groundingdino.util.inference import Model

    ckpt_repo_id = "ShilongLiu/GroundingDINO"
    ckpt_filenmae = "groundingdino_swinb_cogcoor.pth"
    ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"
    config, ckpt = load_model_hf(ckpt_repo_id, ckpt_filenmae, ckpt_config_filename)
    return Model(config, ckpt, device=device)


def main():
    import cv2
    import supervision as sv

    image = cv2.imread("boba_robot/cameras/test_example3/image.png")
    model = get_ovod_model(device="cpu")
    CLASSES = ["red cup", "purple cup", "boba cup", "water cup"]
    BOX_THRESHOLD = 0.3
    TEXT_THRESHOLD = 0.3

    detections = model.predict_with_classes(
        image=image,
        classes=CLASSES,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
    )

    box_annotator = sv.BoxAnnotator()
    annotated_image = box_annotator.annotate(scene=image, detections=detections)
    cv2.imshow("annotated_image", annotated_image)


if __name__ == "__main__":
    main()
