from cellotype.trainer import *
from detectron2.utils.visualizer import Visualizer
from tqdm import tqdm
from detectron2.utils.visualizer import ColorMode
import json
from skimage import io
from skimage.exposure import equalize_adapthist
from skimage.exposure import rescale_intensity

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()

    # for poly lr schedule
    add_deeplab_config(cfg)
    add_maskdino_config(cfg)
    args.config_file = './configs/maskdino_R50_bs16_50ep_4s_dowsample1_2048.yaml'
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    cfg.MODEL.IN_CHANS = 92
    cfg.DATASETS.TRAIN = ("cell_train",)
    cfg.DATASETS.TEST = ('cell_test',)
    cfg.OUTPUT_DIR = 'output/codex'
    cfg.SOLVER.AMP.ENABLED = False
    cfg.MODEL.WEIGHTS = './models/crc_model_0005999.pth'
    cfg.TEST.DETECTIONS_PER_IMAGE = 1000
    cfg.MODEL.PIXEL_MEAN = [128 for _ in range(92)]
    cfg.MODEL.PIXEL_STD = [11 for _ in range(92)]

    cfg.freeze()
    default_setup(cfg, args)
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="cellotype")
    return cfg


def main(args):
    data_dir = 'data/example_codex_crc'
    meta_to_id = json.load(open('data/example_codex_crc/ct2num.json'))

    for d in ["test"]:
        DatasetCatalog.register("cell_" + d, lambda d=d: np.load(os.path.join(data_dir, 'dataset_dicts_patch_{}_ct.npy'.format(d)), allow_pickle=True))
        MetadataCatalog.get("cell_" + d).set(thing_classes=list(meta_to_id.keys()))

    args.resume = True

    cfg = setup(args)
    print("Command cfg:", cfg)
    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        checkpointer = DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR)
        checkpointer.resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    model = Trainer.build_model(cfg)
    model.eval()
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)

    height, width = [512, 512]

    balloon_metadata = MetadataCatalog.get("cell_test")
    ds_dict = np.load('data/example_codex_crc/dataset_dicts_patch_test_ct.npy', allow_pickle=True)
   
    rst = []
    k = 0
    for i,d in enumerate(tqdm(ds_dict)):
        img = io.imread(d["file_name"])
        im = np.transpose(img, (2, 0, 1))
        im = torch.as_tensor(im.astype("float32"))
        inputs = {"image": im, "height": height, "width": width}
        outputs = model([inputs])[0]
        instances = outputs["instances"].to("cpu")
        confident_detections = instances[instances.scores > 0.3]


        if k < 2:
            show_img = img[:, :, [29,33,0]]
            show_img[:, :, 1] = 0
            show_img = equalize_adapthist(show_img)
            show_img = rescale_intensity(show_img, out_range=(0, 255))
            v = Visualizer(show_img,
                        metadata=balloon_metadata, 
                        scale=2,  
                        instance_mode=ColorMode.SEGMENTATION   # remove the colors of unsegmented pixels. This option is only available for segmentation models
            )
            v_gt = Visualizer(show_img,
                        metadata=balloon_metadata, 
                        scale=2,
                        instance_mode=ColorMode.SEGMENTATION)
            v_pred = Visualizer(show_img,
                        metadata=balloon_metadata, 
                        scale=2,
                        instance_mode=ColorMode.SEGMENTATION)

            # confident_detections = instances
            out = v.draw_instance_predictions(confident_detections)

            out_gt = v_gt.draw_dataset_dict(d)

            out.save(os.path.join('output/codex', '{}_pred.png'.format(d['image_id'])))
            out_gt.save(os.path.join('output/codex', '{}_gt.png'.format(d['image_id'])))


            k += 1

        else: quit()

        mask_array = confident_detections.pred_masks.numpy().copy()
        num_instances = mask_array.shape[0]
        output = np.zeros(mask_array.shape[1:])
        pred_cls = confident_detections.pred_classes.numpy().copy()

        for i in range(num_instances):
            # output[mask_array[i,:,:]==True] = pred_cls[i] + 1
            output[mask_array[i,:,:]==True] = i + 1

        output = output.astype(int)
        rst.append(output)


    rst = np.array(rst)
    np.save('output/codex/pred.npy', rst)



if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--EVAL_FLAG', type=int, default=1)
    args = parser.parse_args()
    
    main(args)





