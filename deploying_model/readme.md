## Packaging your Code

We are able to package model objects into a .tar.gz and access this file from s3. When a model is created \ deployed this s3 location will be referenced. 

### Structure 

This structure is what ended up working for us.

```
|----- inference.py
|----- requirements.txt        
|----- preprocessing.joblib
|----- model.joblib
```

There seems to be some support for this structure but we were not able to get this to work.  Depending on the container and model class you are using this may work for you. 

```       
|----- preprocessing.joblib
|----- model.joblib
|----- code
    |----- inference.py
    |----- requirements.txt 
```
### Creating your inference.py 

Here is a sample inference.py that accepts json and converts to pandas for passing to the model objects. 

```
import os 
import joblib
import pandas as pd
import xgboost as xgb_
import json 
from io import StringIO

# defines the model
def model_fn(model_dir):
    """Used to load model objects 

    Args:
        model_dir (string): os path where model files are loaded 

    Returns:
        models 
    """

    preprocessor = joblib.load(os.path.join(model_dir, 'preprocessor.joblib'))
    model_xgb = joblib.load(os.path.join(model_dir,'model_xgb.joblib'))

    return (preprocessor,model_xgb)


def predict_fn(input_data, model):
    """Takes input data and models loaded previously and outputs the data with the prediction added to it. 

    Args:
        input_data (pandas df): pandas data frame with feature required for scoring 
        model : models loaded from the model_fn

    Returns:
        pandas dataframe: data frame with predictions added 
    """

    preprocessor, model_xgb = model
    
    df = input_data
    
    preprocessed = preprocessor.transform(df)
    df['pred'] = model_xgb.predict(xgb_.DMatrix(preprocessed)) 

    beta = 0.0203214454457681
    df['prediction'] = [(x * beta) / ((x * beta) - x + 1) for x in df['pred']]

    df = df[['id', 'prediction']]
            
    return df


def input_fn(serialized_input_data, content_type):
    """Given  a  JSON input, this function formats the input
    data into a pandas data frame. 

    Args:
        serialized_input_data (string): input from the API
        content_type (string): content type of the input 

    Raises:
        ValueError: raises when an unsuported content_type is used 

    Returns:
        pandas dataframe: input converted to a pandas dataframe for scoring
    """
    input_data = {}
    
    if content_type == 'application/json':
        input_data = json.loads(serialized_input_data)    
        input_data = pd.DataFrame([input_data])
    else:
        raise ValueError("{} not supported by script!".format(content_type) )

    return input_data
    
    
def output_fn(prediction_output, accept):
    """Takes a prediction output that is a pandas dataframe and converts it to a dictionary for returning back to client. 

    Args:
        prediction_output (pandas dataframe): dataframe which includes the data we want to return to the client
        accept (string): accept content type

    Raises:
        Exception: Raises if an unsupported accept content type is provided 

    Returns:
        string: serialized output to return to the client
    """
    
    if accept in ['application/json']:
        output = {'Prediction': prediction_output['prediction'][0], 'Id': prediction_output['id'][0]}
        output = json.dumps(output)
        return output, accept
    
    raise Exception('Requested unsupported ContentType in Accept: ' + accept)
```

### Zipping your files 

ex. 

```
tar czvf model.tar.gz. model_xgb.joblib preprocessor.joblib inference.py requirements.txt
```

### Upload to S3

Upload your zip file to s3 and ensure that the role that will be accessing them has access to read them. 

## Deploying your model

### Validate Model is working as expected

Loan your model files into your environment and test them

```
%%time

import os
import boto3
import sagemaker

from sagemaker import get_execution_role

region = boto3.Session().region_name

role = get_execution_role()

print(role)

#pip install your requirements 
#pip install xgboost--.090

import joblib
import pandas as pd
import xgboost as xgb_

model = joblib.load('model_xgb.joblib')
preprocessor = joblib.load('preprocessor.joblib')

df= pd.read_csv('input_features.csv')

preprocessed_df= preprocessor.transform(df)
df['pred'] = 1 - model.predict(xgb_.DMatrix(preprocessed_df))
print(test['pred'])

```

### Deploy your model 



```
from sagemaker.amazon.amazon_estimator import get_image_uri
from sagemaker.model import Model
from sagemaker.pipeline import PipelineModel

container=sagemaker.image_uris.retrieve("xgboost", boto3.Session().region_name, "0.90-1")
print(container)

model_name = 'some_model_123'
xgb_model_location = "s3://somebucket/somemodelpath/someversion/model.tar.gz"

```

This was the secret - through using the sagemaker.pipeline.Framework model found that these were being set when using entry_point. Potential risk of using these if they change in the future. 

```
env = {
    'SAGEMAKER_REQUIREMENTS': 'requirements.txt',
    'SAGEMAKER_PROGRAM': 'inference.py',
    'SAGEMAKER_SUBMIT_DIRECTORY' : xgb_model_location
    }

xgb_model = Model(
    model_data=xgb_model_location, 
    image_uri =container,
    env = env
)

endpoint_name = model_name

pipeline_model = PipelineModel(name=model_name,
                               role=role,
                               models=[
                                    xgb_model
                               ])

pm = pipeline_model.deploy(initial_instance_count=1, instance_type="ml.c4.xlarge", endpoint_name=endpoint_name)

```

### Validate model endpoint 

```
import json 

file = open("feature_sample.json")
test = json.load(file)
payload = json.dumps(test)
ENDPOINT_NAME = model_name

runtime= boto3.client('runtime.sagemaker')
response = runtime.invoke_endpoint(EndpointName=endpoint_name,
                                   ContentType='application/json',
                                   Body=payload)

# print(response)
result = json.loads(response['Body'].read().decode())
print(result)

```