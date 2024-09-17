from cellotype.trainer import *

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
    cfg.MODEL.WEIGHTS = './models/maskdino_swinl_50ep_300q_hid2048_3sd1_instance_maskenhanced_mask52.3ap_box59.0ap.pth'
    
    cfg.MODEL.IN_CHANS = 3
    cfg.DATASETS.TRAIN = ("cell_train",)
    cfg.DATASETS.VAL = ("cell_val",)
    cfg.DATASETS.TEST = ('cell_test',)
    cfg.OUTPUT_DIR = 'output/tissuenet'
    cfg.SOLVER.AMP.ENABLED = False

    cfg.freeze()
    default_setup(cfg, args)
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="cellotype")
    return cfg


def main(args):
    data_dir = 'data/example_tissuenet'

    for d in ["train","val", "test"]:
        DatasetCatalog.register("cell_" + d, lambda d=d: np.load(os.path.join(data_dir, 'dataset_dicts_cell_{}.npy'.format(d)), allow_pickle=True))
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

    trainer = Trainer(cfg)
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
