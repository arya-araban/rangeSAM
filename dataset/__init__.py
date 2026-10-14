from .kitti360 import KITTI360
from .semantic_kitti import SemanticKitti
from .apolloscape import ApolloScape
from .range_view_loader import RangeViewLoader

DATASETS = {
    'KITTI360': KITTI360,
    'SemanticKitti': SemanticKitti,
    'ApolloScape': ApolloScape,
}
