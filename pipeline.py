from typing import Dict

from zenml.steps import Output, step
from zenml.pipelines import pipeline
from zenml.integrations.mlflow.steps.mlflow_registry import mlflow_register_model_step

@step
def get_data_from_dvc() -> Output(DataFolder=str):
    from dvc.api import DVCFileSystem
    fs = DVCFileSystem('https://github.com/MRotaKalpa/mlops78000.git')
    print('Downloading data from DVC...')
    fs.get('data', 'data_from_dvc', recursive=True)
    return 'data_from_dvc'

@step
def trainer(DataFolder: str) -> Output(Model=dict):
    import subprocess
    from collections import namedtuple
    from pathlib import Path
    OutCmd = namedtuple('Command', ['stdout', 'stderr'])
    def run_cmd(command: str) -> OutCmd:
        """Run given command and returns generated stdout and stderr"""
        proc = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)
        proc.wait()
        stdout, stderr = map(
            lambda x: x.decode('utf-8').strip(),
            proc.communicate())
        return OutCmd(stdout=stdout, stderr=stderr)
    out = run_cmd('/bin/bash train_pipeline.sh') 
    match_str = 'Log file for this run: '
    logs = out.stderr
    begin = logs.rfind(match_str)
    if begin == -1:
        raise RuntimeError('Cannot find logs path')
    end = logs[begin:].find('\n')
    end = begin + end if end != -1 else None
    logs_path = Path(logs[begin+len(match_str):end])
    logs_path = str(logs_path.absolute().parent)
    return {
        'stdout': out.stdout,
        'logs_path': logs_path,
    }

@step
def builder(model_data: Dict) -> Output(Model=bytes):
    import boto3
    from pathlib import Path
    s3 = boto3.client('s3')
    logs_path = model_data['logs_path']
    model_path = Path(logs_path) / 'qat_best.pth.tar'
    bucket = 'mlops-dvc-remote'
    key = model_path.name
    with open(model_path, 'rb') as fp:
        s3.upload_fileobj(Fileobj=fp, Bucket=bucket, Key=key)
    waiter = s3.get_waiter('object_exists')
    waiter.wait(Bucket=bucket, Key=key)
    with open(model_path, 'rb') as fp:
        data = fp.read()
    return data

@pipeline
def max78000_pipeline(importer, trainer, builder):
    DataFolder = importer()
    model = trainer(DataFolder=DataFolder)
    artifact = builder(model)

pipeline_instance = max78000_pipeline(get_data_from_dvc(), trainer(), builder())

if __name__ == '__main__':
    pipeline_instance.run()
