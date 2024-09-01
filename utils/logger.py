import logging
from tqdm import tqdm

class TqdmToLogger(object):
    def __init__(self, logger, level=logging.INFO):
        self.logger = logger
        self.level = level
        self.terminal = tqdm(total=0, position=0, file=open('/dev/null', 'w'))  # 使用/dev/null丢弃默认的输出

    def write(self, message):
        # tqdm 组件会在末尾添加 '\r' 用于回到行首，因此在写入日志时需要剔除
        if message.rstrip() != '':
            self.logger.log(self.level, message.rstrip('\r'))

    def flush(self):
        pass

