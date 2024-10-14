import sys 
import logging
from common_utils.logconfig import * 
import argparse


log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


class NoduleAnalysisApp:
    def __init__(self, sys_argv=None):
        if sys_argv is None:
            log.debug(sys.argv)
            sys_argv = sys.argv[1:]

        parser = argparse.ArgumentParser()

        parser.add_argument('--segmentation-path',
            help="Path to the saved segmentation model",
            nargs='?',
            default='',
        )

        parser.add_argument('--cls-model',
            help="What to model class name to use for the classifier.",
            action='store',
            default='LunaModel',
        )

        parser.add_argument('--classification-path',
            help="Path to the saved classification model",
            nargs='?',
            default='data/part2/models/cls_2020-02-06_14.16.55_final-nodule-nonnodule.best.state',
        )

        parser.add_argument('--malignancy-model',
            help="What to model class name to use for the malignancy classifier.",
            action='store',
            default='LunaModel',
        )
        parser.add_argument('--malignancy-path',
            help="Path to the saved malignancy classification model",
            nargs='?',
            default=None,
        )

        parser.add_argument('series_uid',
            nargs='?',
            default=None,
            help="Series UID to use.",
        )

        parser.add_argument('--tb-prefix',
            default='nodule-analysis',
            help="Data prefix to use for Tensorboard run.",
        )
