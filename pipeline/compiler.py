# Install Kubeflow Pipelines SDK. Add the --user argument if you get permission errors.
#!PIP_DISABLE_PIP_VERSION_CHECK=1 python3 -m pip install 'kfp>=0.1.32.2' --quiet --user

import os
import kfp
from kfp.components import InputPath, InputTextFile, OutputPath, OutputTextFile
from kfp.components import func_to_container_op
from typing import NamedTuple

preprocess_train = kfp.components.load_component_from_file(os.path.join('preprocess_train_component.yaml'))
preprocess_test = kfp.components.load_component_from_file(os.path.join('preprocess_test_component.yaml'))
train = kfp.components.load_component_from_file(os.path.join('train_component.yaml'))
test = kfp.components.load_component_from_file(os.path.join('test_component.yaml'))

@kfp.dsl.pipeline(
    name='DDoS Attacks Classifier',
    description='End-to-end training of a traffic classifier'
)
def pipeline(hidden_layers_1: str = '60,30,20', hidden_layers_2: str = '60,40,30,20'):
    preproc_train_task = preprocess_train()
    preproc_train_task.set_memory_request('10G')
    preproc_test_task = preprocess_test()
    preproc_test_task.set_memory_request('10G')
    train_task = train(hidden_layers_1, preproc_train_task.output)
    train_task.set_memory_request('10G') 
    train_task_2 = train(hidden_layers_2, preproc_train_task.output)
    train_task_2.set_memory_request('10G')
    test_task = test(preproc_test_task.output, train_task.output)
    test_task.set_memory_request('10G')
    test_task_2 = test(preproc_test_task.output, train_task_2.output)
    test_task_2.set_memory_request('10G')

kfp.compiler.Compiler().compile(pipeline, 'pipeline.zip')
