import logging

_MAX_ALLOWED_NO_HITS_MESSAGES_ = 3

class NoHitsFilter(logging.Filter):
    def filter(self, record):
        # add other fields if you need more granular comparison, depends on your app
        current_log = record.msg
        last_log = getattr(self, "last_log", "")

        if 'hits is empty' in current_log:
            self.hits_is_empty_counter = getattr(self, "hits_is_empty_counter", 0) + 1

            if getattr(self, "hits_is_empty_counter", 0) > _MAX_ALLOWED_NO_HITS_MESSAGES_:
                return False
            else:
                if getattr(self, "hits_is_empty_counter", 0) == _MAX_ALLOWED_NO_HITS_MESSAGES_:
                    record.msg += ' \nFURTHER LOGS WILL BE SURPRESSED!'
                self.last_log = current_log
                return True
        else:
            return True


def enable_logging(lvl=None):
    """Use this function to enable logging at the given level.

    Usage:

        from logs import enable_logging

        enable_logging()
    """
    # do not create multiple loggers
    lvl = lvl or logging.INFO
    logging.basicConfig(level=lvl,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(message)s',
                        datefmt='%y-%m-%d %H:%M')

    logger = logging.getLogger()
    logger.handlers = []
    formatter = logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(message)s')

    # TODO read file based logging
    # file_handler = logging.FileHandler(config.logging_path())
    # file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    # logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    logger.addFilter(NoHitsFilter())

    logging.info('logging enabled for level: {0}'.format(lvl))
