# thirdeye
>
> *An effort to ensure seeing remains believing*

Thirdeye is a comprehensive system for deepfake video detection developed by [Mahesha Kulatunga](http://maheshak.com/) for a MSc Data Analytics dissertation while at The University of Warwick. This code base handles the preprocessing, training and evaluating required when creating neural networks for video classification. Included are 5 pre-trained 3D CNN architectures that can be used for uknown video classification.

```mermaid
graph LR
  %% main components
  user(User) --> cli(Command Line Interface)
  cli --> thirdeye(Thirdeye Core System)

  %% configuration
  config(Config Manager) --> thirdeye

  %% core modules
  thirdeye --> preprocess(Preprocessor)
  thirdeye --> network(Deepfake Detection Network)
  thirdeye --> classify(Video Classifier)
  thirdeye --> evaluate(Evaluator)

  %% data flows
  raw(Raw Videos) --> preprocess
  preprocess --> |Face Extraction| processed(Processed Videos)
  processed --> network
  processed --> classify

  %% utilities
  utils(Utilities) --- preprocess
  utils --- network
  utils --- classify
  utils --- evaluate

  %% network components
  network --> |Training| factory(Network Factory)
  factory --> models(Pre-trained Models)
  models --> network
  models --> classify
  models --> evaluate

  %% classification
  classify --> |Creates| ensemble(Ensemble Classifier)
  classify --> results(Classification Results)

  %% evaluation
  evaluate --> metrics(Evaluation Metrics)
  evaluate --> viz(Performance Visualizations)

  %% subgraphs
  subgraph "Preprocessing Pipeline"
    video_proc(Video Processor)
    face_det(Face Detection)
    motion_ex(Motion Vector Extraction)
    video_proc --> face_det --> motion_ex
  end
  preprocess --> "Preprocessing Pipeline"

  subgraph "Network Model Variants"
    legacy(Legacy Models)
    modern(Modern Architectures)
    legacy --> models
    modern --> models
  end
  factory --> "Network Model Variants"

  %% styling
  classDef mainComp fill:#f2f2f2,stroke:#333,stroke-width:2px;
  classDef module   fill:#cce5ff,stroke:#0062cc,stroke-width:1px;
  classDef dataFlow fill:#e6ffe6,stroke:#2d8f2d,stroke-width:1px,stroke-dasharray:5 5;
  classDef utility  fill:#ffe6e6,stroke:#cc0000,stroke-width:1px,stroke-dasharray:2 2;

  class thirdeye mainComp;
  class preprocess,network,classify,evaluate,factory module;
  class raw,processed,models,results,metrics,viz dataFlow;
  class config,utils utility;
```

For further questions on the network designs or Thirdeye in general, please contact <mahesha.kulatunga@gmail.com>.

## Requirements

- python ≥ 3.8

- ffmpeg installed and on your PATH

- disk space ≥ 9 gb (includes saved models + datasets)

- os support: ubuntu ≥ 16.04 │ windows ≥ 7 │ macos ≥ 10.12.6 │ raspbian ≥ 9.0

- see requirements.txt for full list. core python packages include:

-In addition, around 9GB disk space is required including saved models and the DFD dataset.

### Dependencies

### install via bundled script

./install.sh

### or manually via pip

pip install -r requirements.txt

## Set-up

- Ensure all dependencies are satisfied.
- Navigate to the thirdeye directory in the command line.  
- Run the *example.py* file to run an example classification of sample videos included in the data folder.

```mrkdown

cd thirdeye

python example.py

```

### Alternatively

```python
import thirdeye
t = thirdeye.Thirdeye()
t.perform_preprocessing()
t.set_network('providence_v1')
t.train()
```

## Usage

Selecting a network: Use t.set_network(<code_name>) with one of:

code_names: providence_v1, providence_v2, odin_v1, odin_v2, horus (original and updated)

code_names: resnet3d_18, resnet3d_34, resnet3d_50, slowfast, efficient3d, vivit (alternatives)

## How do I train a network?

```python
import thirdeye
t = thirdeye.Thirdeye()
t.perform_preprocessing()
t.set_network('resnet3d_34')
t.train()
```

Or one shot with:

```python
t = thirdeye.Thirdeye(network='slowfast', pre_p=True, force_t=True)
```

### Classifying unknown videos

```python
t = thirdeye.Thirdeye(network='efficient3d')
t.classify()
```

### Single video inference

```python
result = t.process_video('/path/to/video.mp4')
print(f"real: {result['Real']*100:.2f}%, deepfake: {result['Deepfake']*100:.2f}%")
```

## Added functionality in 2025

### Ensemble classification

```python
ensemble = t.create_ensemble(['odin_v1','providence_v2','resnet3d_18'])
results = ensemble.classify_folder('./Data/UNKNOWN/UNKNOWN_SAMPLES')
```

### Model comparison by industry metrics 

```python
t.compare_networks(['odin_v1','providence_v2','resnet3d_18','slowfast'])
```

### Visualise the above

```python
t.evaluate()
metrics = t.evaluator.evaluate_model(test_x, test_y, 'model-comparison')
```