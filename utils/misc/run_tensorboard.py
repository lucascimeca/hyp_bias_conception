from torch.utils.tensorboard import SummaryWriter
from tensorboard import program
import webbrowser
import time

TENSORBOARD_FOLDER = "./../runs/"
RUN_NUMBER = 0

writer = SummaryWriter(log_dir='')

# --- TensorBoard ----
tb = program.TensorBoard()
tb.configure(argv=[None, '--host', 'localhost',
                   '--reload_interval', '15',
                   '--port', '8080',
                   '--logdir', TENSORBOARD_FOLDER])
url = tb.launch()
webbrowser.open(url + '#timeseries')
while True:
    time.sleep(1)
