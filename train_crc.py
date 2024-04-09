from cellotype.trainer import *

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

    cfg.MODEL.IN_CHANS = 92
    cfg.DATASETS.TRAIN = ("cell_train",)

    cfg.DATASETS.TEST = ('cell_test',)
    cfg.OUTPUT_DIR = 'output/codex'
    cfg.SOLVER.AMP.ENABLED = False
    cfg.MODEL.PIXEL_MEAN = [128 for _ in range(92)]
    cfg.MODEL.PIXEL_STD = [11 for _ in range(92)]
    cfg.MODEL.WEIGHTS = 'cellotype/models/maskdino_swinl_50ep_300q_hid2048_3sd1_instance_maskenhanced_mask52.3ap_box59.0ap.pth'
    cfg.freeze()
    default_setup(cfg, args)
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="cellotype")
    return cfg


def main(args):
    data_dir = 'data/example_codex_crc'
    meta_to_id = json.load(open('data/example_codex_crc/ct2num.json'))

    for d in ["train", "test"]:
        if d == 'train':
            DatasetCatalog.register("cell_" + d, lambda d=d: np.load(os.path.join(data_dir, 'dataset_dicts_patch_{}_ct.npy'.format(d)), allow_pickle=True))
        else:
            DatasetCatalog.register("cell_" + d, lambda d=d: np.load(os.path.join(data_dir, 'dataset_dicts_patch_{}_ct.npy'.format(d)), allow_pickle=True))
        MetadataCatalog.get("cell_" + d).set(thing_classes=list(meta_to_id.keys()))
    
    args.resume = False


    cfg = setup(args)
    print("Command cfg:", cfg)
    if args.eval_only:
        model = Trainer_CODEX.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        checkpointer = DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR)
        checkpointer.resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer_CODEX.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer_CODEX.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer_CODEX(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--EVAL_FLAG', type=int, default=1)
    args = parser.parse_args()
    # random port
    port = random.randint(1000, 20000)
    args.dist_url = 'tcp://127.0.0.1:' + str(port)
    print("Command Line Args:", args)
    print("pwd:", os.getcwd())
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
