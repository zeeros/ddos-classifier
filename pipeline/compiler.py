import os
import kfp
from kfp.components import InputPath, InputTextFile, OutputPath, OutputTextFile
from kfp.components import func_to_container_op
from typing import NamedTuple

preprocess = kfp.components.load_component_from_file(os.path.join('preprocess_component.yaml'))
train = kfp.components.load_component_from_file(os.path.join('train_component.yaml'))
test = kfp.components.load_component_from_file(os.path.join('test_component.yaml'))

@kfp.dsl.pipeline(
    name='DDoS Attacks Classifier',
    description='End-to-end training of a traffic classifier'
)
def pipeline():
    preprocess_task = preprocess()
    train_task = train(preprocess_task.output)
    test_task = test(train_task.output)

kfp.compiler.Compiler().compile(pipeline, 'pipeline.zip')
