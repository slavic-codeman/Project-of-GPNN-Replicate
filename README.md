# Project-of-GPNN-Replicate
To run our project, please read this file carefully for detailed instructions.

## Environments
We recommend using Anaconda to create a Python 3.10 environment.
```
conda create -n gpnn python=3.10
conda activate gpnn
```

Then to install dependencies, use
```
pip install -r requirements.txt
```
## Paths
When running codes, please make sure that your terminal is one level above the folder ```my_gpnn```, to ensure that results can be generated in correct paths. This is to say, please ensure that folders ```my_gpnn```, ```tmp```, and ```v-coco-master``` are all in one directory, where you open your terminal for running codes.

## Data and Weights

Click [here](https://disk.pku.edu.cn/link/AA165E5BE67089441E8DF401A1A1234178) to download data and weights. You will get a ```tmp``` folder, and please put it in the right path.



## Running codes
Files for training and evaluation of different models include
```
my_gpnn/vcoco.py
my_gpnn/hico.py
my_gpnn/cad120.py
```
These files have similar structures. First, they all have a ```parse_arguments()``` function for parsing arguments, and you can read their instructions and  modify default values or specify arguments in the terminal. Second, in the ```main()``` function, you have to specify model arguments in the dictionary 
```
model_args = 
{
    ......
'update_type':'gru'}
```
The ```update_type``` here decides the type of update function, and you can change it by adding arguments ```--transformer``` when running the code to use transformer rather than gpu. Saving and loading trained model file with transformer architecture is correspondingly set automately in 
```
./my_gpnn/datasets/utils.py
```
Model file with name ```model_best_transformer.pth``` refers to model with transformer-type update function. Feel free to customize it.
### HICO-DET

If you want to visualize HOI relationships on HICO-DET dataset when testing, you have to download test images from HICO-DET [here](https://disk.pku.edu.cn/link/AA3D8A83A919BC4092BC81A39A3B129D21) and change the default path in ``` ./my_gpnn/hico.py ```to the folder of these images, i.e., 
```
parser.add_argument('--image-root', default="Your path to HICO-DET images")
```
and keep the argument ```visualize``` as True.
. Visulizations will be saved to
```
./tmp/results/HICO/detections/visualization
```

For training and evaluation, please specify arguments and run
```
python my_gpnn/hico.py
```

### VCOCO

If you want to visualize HOI relationships on VCOCO dataset when testing, you have to download MS-COCO [val2014](http://images.cocodataset.org/zips/val2014.zip) and change the ```val_path``` in ```validate()``` function in ```./my_gpnn/vcoco.py``` to the folder of these images, and keep the argument ```visualize``` as True. Visulizations will be saved to
```
./tmp/results/VCOCO/detections
```

For training and evaluation, please specify arguments and run
```
python my_gpnn/vcoco.py
```


### CAD-120

For training and evaluation, run
```
python my_gpnn/cad120.py --task="prediction" #For HOI anticipation
python my_gpnn/cad120.py --task="parsing" #For HOI detection
```
Visualizations will be saved to 
```
./tmp/results/CAD
```