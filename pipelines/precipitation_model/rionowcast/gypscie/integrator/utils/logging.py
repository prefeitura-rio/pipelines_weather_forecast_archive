# -*- coding: utf-8 -*-
# pylint: disable= invalid-name, inconsistent-return-statements, broad-except
"""
ADD
"""

import logging


class Logger(object):

    name = "logger"
    levels = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }

    @classmethod
    def add_log_parameters(cls, parser, root_basename):
        """Add predefined logging parameters to an argparse.ArgumentParser object

        Args:
            parser (argparse.ArgumentParser): an argparse.ArgumentParser object
            root_basename (str): basename string of root script that invokes Logger.init(). \
                Get basename string with command os.path.basename(__file__).

        Returns:
            argparse.ArgumentParser: the argparse.ArgumentParser object with logging parameters added
        """
        logfile = (root_basename.split(".")[0]) + ".log"
        parser.add_argument(
            "--logfile",
            type=str,
            default=logfile,
            help="Log filename. Default __file__ basename",
        )
        parser.add_argument(
            "--loglevel",
            type=str,
            default="info",
            help=f"Log level. Level must be one of {list(Logger.levels.keys())}",
        )
        parser.add_argument(
            "--verbose",
            action="store_true",
            help="Enable verbose execution on stdout. Default is False.",
        )
        return parser

    @classmethod
    def init(cls, filename, level="info", verbose=False):
        """Method that setup and start logging

        Args:
            filename (str): log filename for FileHandler. File will be stored at logs folder.
            level (str, optional): a python logging level. Defaults to 'info'.
            verbose (bool, optional): if True, output logging on stdout. Defaults to False.
        """
        assert level in cls.levels.keys(), f"level must be one of {list(cls.levels.keys())}"
        logger = cls.get_logger()
        logger.setLevel(cls.levels[level])
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        fl = logging.FileHandler(f"logs/{filename}")
        fl.setLevel(cls.levels[level])
        fl.setFormatter(formatter)
        logger.addHandler(fl)
        if verbose:
            sh = logging.StreamHandler()
            sh.setLevel(cls.levels[level])
            sh.setFormatter(formatter)
            logger.addHandler(sh)

    @classmethod
    def get_logger(cls):
        return logging.getLogger(cls.name)
