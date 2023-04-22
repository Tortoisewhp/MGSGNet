from .metrics import averageMeter, runningScore
from .log import get_logger
from .utils import ClassWeight, save_ckpt, load_ckpt, class_to_RGB, \
    compute_speed, setup_seed, group_weight_decay

# 加载数据集
def get_dataset(cfg):
    assert cfg['dataset'] in ['nyuv2', 'sunrgbd', 'cityscapes', 'camvid', 'irseg', 'pst900']
    if cfg['dataset'] == 'irseg':
        # 导入数据集
        from .datasets.irseg import IRSeg
        return IRSeg(cfg, mode='train'), IRSeg(cfg, mode='val'), IRSeg(cfg, mode='test')

def get_model(cfg):
    # 根据配置文件里的model_name标签，匹配相应模型文件

    if cfg['model_name'] == 'MGSGNet-teacher':
        from toolbox.models.MGSGNet.MGSGNet_teacher import MGSGNet_teacher
        return MGSGNet_teacher()

    elif cfg['model_name'] == 'MGSGNet-student':
        from toolbox.models.MGSGNet.MGSGNet_student import MGSGNet_student
        return MGSGNet_student()
