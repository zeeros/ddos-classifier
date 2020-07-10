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
def pipeline():
    preproc_train_task = preprocess_train_()
    preproc_test_task = preprocess_test()
    train_task = train(preproc_train_task.output)
    test_task = test(preproc_test_task.output, train_task.output)

kfp.compiler.Compiler().compile(pipeline, 'pipeline.zip')
