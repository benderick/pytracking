from rootutils import find_root
class EnvironmentSettings:
    def __init__(self):
        root = str(find_root(__file__))
        self.workspace_dir = root + '/workspace'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = self.workspace_dir + '/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = self.workspace_dir + '/pretrained_networks'
        self.pregenerated_masks = ''
        self.lasot_dir = ''
        self.got10k_dir = root + '/data/got10k/train'
        self.trackingnet_dir = ''
        self.coco_dir = ''
        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenet_dir = ''
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''
        self.lasot_candidate_matching_dataset_path = ''
