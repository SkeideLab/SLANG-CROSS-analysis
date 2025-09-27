# %%
import json
from pathlib import Path
from scripts.helpers import submit_job

# %%
analy_dir      = Path.cwd()
log_dir        = analy_dir / 'logs'
script_dir     = analy_dir / 'scripts'
with open(analy_dir / 'run_params.json', 'r') as fp:
    run_params = json.load(fp)

bids_dir       = Path(run_params['bids_dir'])
conda_env      = run_params['conda_env']
conda_env_name = run_params['conda_env_name']
# %%
script = script_dir / 'univariate.sh'
args = [script, bids_dir, analy_dir, conda_env, conda_env_name]
job_id = submit_job(args, cpus=1, mem=24000,log_dir=log_dir,
                    job_name='univariate')
# %%
