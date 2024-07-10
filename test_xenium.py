from cellotype.trainer import *
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from tqdm import tqdm
import cv2
from detectron2.utils.visualizer import ColorMode

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()

    # for poly lr schedule
    add_deeplab_config(cfg)
    add_maskdino_config(cfg)
    args.config_file = 'cellotype/configs/maskdino_R50_bs16_50ep_4s_dowsample1_2048.yaml'
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.MODEL.IN_CHANS = 3
    cfg.DATASETS.TRAIN = ("cell_train",)
    cfg.DATASETS.TEST = ('cell_test',)
    cfg.OUTPUT_DIR = 'output/xenium'
    cfg.SOLVER.AMP.ENABLED = False
    cfg.MODEL.WEIGHTS = 'cellotype/models/xenium_model_0001499.pth'
    cfg.TEST.DETECTIONS_PER_IMAGE = 500
    cfg.MODEL.DEVICE = 'cuda'


    cfg.freeze()
    default_setup(cfg, args)
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="cellotype")
    return cfg


def main(args):
    data_dir = 'data/example_xenium'

    for d in ["train","val", "test"]:
        DatasetCatalog.register("cell_" + d, lambda d=d: np.load(os.path.join(data_dir, 'dataset_dicts_{}.npy'.format(d)), allow_pickle=True))
        MetadataCatalog.get("cell_" + d).set(thing_classes=["cell"])
    
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
    
    predictor = DefaultPredictor(cfg)

    balloon_metadata = MetadataCatalog.get("cell_test")
    ds_dict = np.load('data/example_xenium/dataset_dicts_test.npy', allow_pickle=True)

    rst = []
    k = 0
    for d in tqdm(ds_dict):
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        instances = outputs["instances"].to("cpu")
        confident_detections = instances[instances.scores > 0.3]
        if k < 2:
            v = Visualizer(im[:, :, ::-1],
                        metadata=balloon_metadata, 
                        scale=2, 
                        instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
            )
           
            out = v.draw_instance_predictions(confident_detections)
            cv2.imwrite(os.path.join('output/xenium', '{}_pred.png'.format(k)), out.get_image()[:, :, ::-1])
            k += 1
        else: quit()

        mask_array = confident_detections.pred_masks.numpy().copy()
        num_instances = mask_array.shape[0]
        output = np.zeros(mask_array.shape[1:])

        for i in range(num_instances):
            output[mask_array[i,:,:]==True] = i+1

        output = output.astype(int)
        rst.append(output)

    rst = np.array(rst)
    np.save('output/xenium/pred.npy', rst)
    



if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--EVAL_FLAG', type=int, default=1)
    args = parser.parse_args()

    main(args)





