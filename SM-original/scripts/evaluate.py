
from metrics import Metric
from utils.utils import getLogger
from metric_parameters import main as calculate_CBKetc_metrics

# List of metrics to be calculated using the Metric class 
metrics = ['snr', 'si_sdr', 'pesq', 'stoi']

# Paths
ckpt_dir = '/ghome/fewahab/Sun-Models/Mod-3/T71a1/ckpt'
est_path = '/gdata/fewahab/Sun-Models/Mod-3/T71a1'
ckpt_dir_CBKetc = '/ghome/fewahab/Sun-Models/Mod-3/T71a1/ckpt/CBKetc' 

logger = getLogger(__name__)

def main():
    for metric in metrics:
        print(metric)
        # Use the Metric class for the specified metrics
        m = Metric(est_path, ckpt_dir, metric)
        m.evaluate()

    # Calculate CBKetc metrics
    calculate_CBKetc_metrics(est_path, ckpt_dir_CBKetc)  

if __name__ == '__main__':
    main()
