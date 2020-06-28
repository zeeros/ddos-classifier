from typing import NamedTuple

import kfp
from kfp.components import InputPath, InputTextFile, OutputPath, OutputTextFile
from kfp.components import func_to_container_op


# Load two more components for importing and exporting the data:
preprocess = kfp.components.load_component_from_file(os.path.join('preprocess/component.yaml'))
#train = kfp.components.load_component_from_file(os.path.join('train/component.yaml'))

@dsl.pipeline(
    name='DDoS Attacks Classifier',
    description='End-to-end training of a traffic classifier'
)
def pipeline():
    preprocess_task = preprocess(data_url)
    #train(preprocess_task.output)

kfp.compiler.Compiler().compile(pipeline,
  'my-pipelinea.zip')
