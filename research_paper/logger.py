import logging
from research_paper import multiprocessing_utils

logging.basicConfig(filename='research_paper/logs/exp.log', level=logging.ERROR,
                   format='%(asctime)s %(levelname)s %(name)s %(message)s')


def get_logger(experiment_manager):
    return logging.getLogger(multiprocessing_utils.get_desc(experiment_manager))
