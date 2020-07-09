import os
import kfp
from kfp.components import InputPath, InputTextFile, OutputPath, OutputTextFile
from kfp.components import func_to_container_op
from typing import NamedTuple

preprocess_train_ds = kfp.components.load_component_from_file(os.path.join('preprocess_train_ds_component.yaml'))
preprocess_test_ds = kfp.components.load_component_from_file(os.path.join('preprocess_test_ds_component.yaml'))
train_test = kfp.components.load_component_from_file(os.path.join('train_test_component.yaml'))

@kfp.dsl.pipeline(
    name='DDoS Attacks Classifier',
    description='End-to-end training of a traffic classifier'
)
def pipeline():
    preproc_train_ds_task = preprocess_train_ds()
    preproc_test_ds_task = preprocess_test_ds()
    train_test_task = train_test(preproc_train_ds_task.output, preproc_test_ds_task.output)

kfp.compiler.Compiler().compile(pipeline, 'pipeline.zip')
