# ImagenHeap

ImagenHeap is a powerful Python package designed for chaining image processing models to efficiently process images from datasets. It provides a flexible and extensible framework for creating custom image processing pipelines.

## Features

- Easy configuration using YAML files
- Customizable components for various image processing tasks
- Runtime configuration overrides
- Visualization tools for processed results
- Extensible architecture for adding new components

## Installation

```bash
pip install imagenheap
```

## Quick Start

1. Import the ImagenHeap class:

```python
from imagenheap import ImagenHeap
```

2. Create an instance of ImagenHeap:

```python
ih = ImagenHeap()
```

3. Load the configuration from a YAML file:

```python
ih.load_config("/path/to/your/config.yml")
```

4. (Optional) Override configurations:

```python
ih.add_args("detect").text_prompt = ["person", "glasses"]
```

5. Run the processing pipeline:

```python
results = ih.run()
```

6. Visualize the results:

```python
ih.visualize()
```

## Creating Custom Components

To create a custom component for ImagenHeap, inherit from the `Component` class and implement the required abstract methods:

```python
from imagenheap import Component

class MyCustomComponent(Component):
    def load_model(self):
        # Implement model loading logic
        pass

    def process(self, input_data):
        # Implement processing logic
        pass
```

## Configuration

ImagenHeap uses YAML files for configuration. Here's an example structure:

```yaml
components:
  - name: DetectDino
  - name: SegmentSam

pipeline:
  - detect
  - segment

dataset:
  path: /path/to/images or huggingface/dataset

batch_size: 10

component_args:
  detect:
    text_prompt:
      - hair
      - face
      - neck
      - arm
      - hand
      - back
      - leg
      - foot
      - outfit
      - phone
      - hat
      - shoe
    box_threshold: 0.3
    text_threshold: 0.25
    iou_threshold: 0.8
  segment:
    return_tensors: true
```

## Advanced Usage

### Training Models

ImagenHeap is developing a `train` method to allow users to train different types of models on the processed results. For example, training a YOLO model:

```python
ih.train(model_type="yolo", data=results, epochs=100)
```

### Visualization Options

ImagenHeap offers various visualization options:

```python
# Visualize all results
ih.visualize()

# Visualize a specific result
ih.visualize(result_index=5)

# Visualize with custom options
ih.visualize(show_bounding_boxes=True, show_segmentation=False)
```

## Contributing

We welcome contributions to ImagenHeap! Please see our [CONTRIBUTING.md](CONTRIBUTING.md) file for details on how to get started.

## License

ImagenHeap is released under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Support

For questions, bug reports, or feature requests, please open an issue on our [GitHub repository](https://github.com/yourusername/imagenheap).

## Acknowledgments

- List any libraries, tools, or contributors you'd like to acknowledge here.

---

Happy image processing with ImagenHeap!