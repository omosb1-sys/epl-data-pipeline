# Huggingface - Transformers

**Pages:** 50

---

## Transformers

**URL:** https://huggingface.co/docs/transformers/v5.0.0/en/trainer

**Contents:**
- Transformers
- Trainer
- Checkpoints
- Logging
- Customize
  - Callbacks
- Accelerate
- Optimizations
  - torch.compile
  - GaLore

Transformers documentation

and get access to the augmented documentation experience

Trainer is a complete training and evaluation loop for Transformers’ PyTorch models. Plug a model, preprocessor, dataset, and training arguments into Trainer and let it handle the rest to start training faster.

Trainer is also powered by Accelerate, a library for handling large models for distributed training.

This guide will show you how Trainer works and how to customize it for your use case with a callback.

Trainer contains all the necessary components of a training loop.

Manually coding this training loop every time can be inconvenient or a barrier if you’re just getting started with machine learning. Trainer abstracts this process, allowing you to focus on the model, dataset, and training design choices.

Configure your training with hyperparameters and options from TrainingArguments which supports many features such as distributed training, torch.compile, mixed precision training, and saving the model to the Hub.

The number of available parameters available in TrainingArguments may be intimidating at first. If there is a specific hyperparameter or feature you want to use, try searching for it directly. Otherwise, feel free to start with the default values and gradually customize them as you become more familiar with the training process.

The example below demonstrates an example of TrainingArguments that evaluates and saves the model at the end of each epoch. It also loads the best model found during training and pushes it to the Hub.

Pass your model, dataset, preprocessor, and TrainingArguments to Trainer, and call train() to start training.

Refer to the Fine-tuning guide for a more complete overview of the training process.

Trainer saves checkpoints (the optimizer state is not saved by default) to the directory in output_dir in TrainingArguments to a subfolder named checkpoint-000. The number at the end is the training step at which the checkpoint was saved.

Saving checkpoints are useful for resuming training or recovering your training progress if you encounter an error. Set the resume_from_checkpoint parameter in train() to resume training from the last checkpoint or a specific checkpoint.

Checkpoints can be saved to the Hub by setting push_to_hub=True in TrainingArguments. The default method ("every_save") saves a checkpoint to the Hub every time a model is saved, which is typically the final model at the end of training. Some other options for deciding how to save checkpoints to the Hub include the following.

Trainer attempts to maintain the same Python, NumPy, and PyTorch RNG states when you resume training from a checkpoint. But PyTorch has various non-deterministic settings which can’t guarantee the RNG states are identical. To enable full determinism, refer to the Controlling sources of randomness guide to learn what settings to adjust to make training fully deterministic (some settings may result in slower training).

Trainer is set to logging.INFO by default to report errors, warnings, and other basic information. Use log_level() to change the logging level and log verbosity.

The example below sets the main code and modules to use the same log level.

In a distributed environment, Trainer replicas are set to logging.WARNING to only report errors and warnings. Use log_level_replica() to change the logging level and log verbosity. To configure the log level for each node, use log_on_each_node() to determine whether to use a specific log level on each node or only the main node.

Use different combinations of log_level and log_level_replica to configure what gets logged on each node.

The log level is separately set for each node in the __init__() method. Consider setting this sooner if you’re using other Transformers functionalities before creating the Trainer instance.

Tailor Trainer to your use case by subclassing or overriding its methods to support the functionality you want to add or use, without rewriting the entire training loop from scratch. The table below lists some of the methods that can be customized.

For example, to use weighted loss, rewrite compute_loss() inside Trainer.

Callbacks are another way to customize Trainer, but they don’t change anything inside the training loop. Instead, a callback inspects the training loop state and executes some action (early stopping, logging, etc.) depending on the state. For example, you can’t implement a custom loss function with a callback because that requires overriding compute_loss().

To use a callback, create a class that inherits from TrainerCallback and implements the functionality you want. Then pass the callback to the callback parameter in Trainer. The example below implements an early stopping callback that stops training after 10 steps.

Accelerate is a library that simplifies training in distributed environments and across different hardware. Its integration with Trainer means Trainer supports distributed training frameworks like Fully Sharded Data Parallel (FSDP) and DeepSpeed.

Learn more about FSDP sharding strategies, CPU offloading, and more with Trainer in the Fully Sharded Data Parallel guide.

To use Accelerate with Trainer, run the accelerate_config command to configure your training environment. This command creates a config_file.yaml file that stores the configuration settings of your training environment and it’s used whenever you launch your training script. Some example distributed training configurations are shown below.

Run accelerate_launch to start training with the configurations set in config_file.yaml. This file is saved to the Accelerate cache folder and automatically loaded when you run accelerate_launch.

The example below launches the run_glue.py script with the FSDP configuration shown earlier. Parameters from the config_file.yaml file can also be directly set in the command line.

Refer to the Launching your Accelerate scripts tutorial to learn more about accelerate_launch and custom configurations.

Trainer supports various optimizations to improve training performance - reduce memory and increase training speed - and model performance.

torch.compile can significantly speed up training and reduce computational overhead. Configure your torch.compile settings in TrainingArguments. Set torch_compile to True, and select a backend and compile mode.

Gradient Low-Rank Projection (GaLore) significantly reduces memory usage when training large language models (LLMs). One of GaLores key benefits is full-parameter learning, unlike low-rank adaptation methods like LoRA, which produces better model performance.

Install the GaLore and TRL libraries.

Pick a GaLore optimizer ("galore_adamw", "galore_adafactor", "galore_adamw_8bit”) and pass it to the optim parameter in trl.SFTConfig. Use the optim_target_modules parameter to specify which modules to adapt (can be a list of strings, regex, or a full path).

Extra parameters supported by GaLore, rank, update_proj_gap, and scale, should be passed to the optim_args parameter in trl.SFTConfig.

The example below enables GaLore with SFTTrainer that targets the attn and mlp layers with regex.

It can take some time before training starts (~3 minutes for a 2B model on a NVIDIA A100).

Only linear layers that are considered GaLore layers can be trained with low-rank decomposition. The rest of the model layers are optimized in the usual way.

Liger Kernel is a collection of layers such as RMSNorm, RoPE, SwiGLU, CrossEntropy, FusedLinearCrossEntropy, and more that have been fused into a single Triton kernel for training LLMs. These kernels are also compatible with FlashAttention, FSDP, and DeepSpeed. As a result, Liger Kernel can increase multi-GPU training throughput and reduce memory usage. This is useful for multi-head training and supporting larger vocabulary sizes, larger batch sizes, and longer context lengths.

Enable Liger Kernel for training by setting use_liger_kernel=True in TrainingArguments. This patches the corresponding layers in the model with Ligers kernels.

Liger Kernel supports Llama, Gemma, Mistral, and Mixtral models. Refer to the patching list for the latest list of supported models.

You can also configure which specific kernels to apply using the liger_kernel_config parameter. This dict is passed as keyword arguments to the _apply_liger_kernel_to_instance function, allowing fine-grained control over kernel usage. Available options vary by model but typically include: rope, swiglu, cross_entropy, fused_linear_cross_entropy, rms_norm, etc.

NEFTune adds noise to the embedding vectors during training to improve model performance. Enable it in Trainer with the neftune_noise_alpha parameter in TrainingArguments to control how much noise is added.

The original embedding layer is restored after training to avoid any unexpected behavior.

---

## Transformers

**URL:** https://huggingface.co/docs/transformers/v5.0.0/en/main_classes/configuration

**Contents:**
- Transformers
- Configuration
- PreTrainedConfig
  - class transformers.PreTrainedConfig
    - push_to_hub
    - dict_dtype_to_str
    - from_dict
    - from_json_file
    - from_pretrained
    - get_config_dict

Transformers documentation

and get access to the augmented documentation experience

The base class PreTrainedConfig implements the common methods for loading/saving a configuration either from a local file or directory, or from a pretrained model configuration provided by the library (downloaded from HuggingFace’s AWS S3 repository).

Each derived config class implements model specific attributes. Common attributes present in all config classes are: hidden_size, num_attention_heads, and num_hidden_layers. Text models further implement: vocab_size.

( output_hidden_states: bool = False output_attentions: bool = False return_dict: bool = True dtype: typing.Union[str, ForwardRef('torch.dtype'), NoneType] = None chunk_size_feed_forward: int = 0 is_encoder_decoder: bool = False architectures: list[str] | None = None id2label: dict[int, str] | None = None label2id: dict[str, int] | None = None num_labels: int | None = None problem_type: str | None = None **kwargs )

Parameters for fine-tuning tasks

PyTorch specific parameters

Base class for all configuration classes. Handles a few parameters common to all models’ configurations as well as methods for loading/downloading/saving configurations.

A configuration file can be loaded and saved to disk. Loading the configuration file and using this file to initialize a model does not load the model weights. It only affects the model’s configuration.

Class attributes (overridden by derived classes):

Common attributes (present in all subclasses):

Setting parameters for sequence generation in the model config is deprecated. For backward compatibility, loading some of them will still be possible, but attempting to overwrite them will throw an exception — you should set them in a [~transformers.GenerationConfig]. Check the documentation of [~transformers.GenerationConfig] for more information about the individual parameters.

( repo_id: str commit_message: str | None = None commit_description: str | None = None private: bool | None = None token: bool | str | None = None revision: str | None = None create_pr: bool = False max_shard_size: int | str | None = '50GB' tags: list[str] | None = None )

Upload the configuration file to the 🤗 Model Hub.

Checks whether the passed dictionary and its nested dicts have a dtype key and if it’s not None, converts torch.dtype to a string of just the type. For example, torch.float32 get converted into “float32” string, which can then be stored in the json format.

( config_dict: dict **kwargs ) → PreTrainedConfig

The configuration object instantiated from those parameters.

The configuration object instantiated from those parameters.

Instantiates a PreTrainedConfig from a Python dictionary of parameters.

( json_file: str | os.PathLike ) → PreTrainedConfig

The configuration object instantiated from that JSON file.

The configuration object instantiated from that JSON file.

Instantiates a PreTrainedConfig from the path to a JSON file of parameters.

( pretrained_model_name_or_path: str | os.PathLike cache_dir: str | os.PathLike | None = None force_download: bool = False local_files_only: bool = False token: str | bool | None = None revision: str = 'main' **kwargs ) → PreTrainedConfig

To test a pull request you made on the Hub, you can pass revision="refs/pr/<pr_number>".

If True, then this functions returns a Tuple(config, unused_kwargs) where unused_kwargs is a dictionary consisting of the key/value pairs whose keys are not configuration attributes: i.e., the part of kwargs which has not been used to update config and is otherwise ignored.

The configuration object instantiated from this pretrained model.

The configuration object instantiated from this pretrained model.

Instantiate a PreTrainedConfig (or a derived class) from a pretrained model configuration.

( pretrained_model_name_or_path: str | os.PathLike **kwargs ) → tuple[Dict, Dict]

The dictionary(ies) that will be used to instantiate the configuration object.

The dictionary(ies) that will be used to instantiate the configuration object.

From a pretrained_model_name_or_path, resolve to a dictionary of parameters, to be used for instantiating a PreTrainedConfig using from_dict.

( decoder = None encoder = None )

Returns the text config related to the text input (encoder) or text output (decoder) of the model. The decoder and encoder input arguments can be used to specify which end of the model we are interested in, which is useful on models that have both text input and output modalities.

There are three possible outcomes of using this method:

( auto_class = 'AutoConfig' )

Register this class with a given auto class. This should only be used for custom configurations as the ones in the library are already mapped with AutoConfig.

( save_directory: str | os.PathLike push_to_hub: bool = False **kwargs )

Save a configuration object to the directory save_directory, so that it can be re-loaded using the from_pretrained() class method.

Dictionary of all the attributes that make up this configuration instance.

Dictionary of all the attributes that make up this configuration instance.

Serializes this instance to a Python dictionary.

Dictionary of all the attributes that make up this configuration instance.

Dictionary of all the attributes that make up this configuration instance.

Removes all attributes from the configuration that correspond to the default config attributes for better readability, while always retaining the config attribute from the class. Serializes to a Python dictionary.

( json_file_path: str | os.PathLike use_diff: bool = True )

Save this instance to a JSON file.

( use_diff: bool = True ) → str

String containing all the attributes that make up this configuration instance in JSON format.

String containing all the attributes that make up this configuration instance in JSON format.

Serializes this instance to a JSON string.

( config_dict: dict )

Updates attributes of this class with attributes from config_dict.

Updates attributes of this class with attributes from update_str.

The expected format is ints, floats and strings as is, and for booleans use true or false. For example: “n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index”

The keys to change have to already exist in the config object.

---

## Transformers

**URL:** https://huggingface.co/docs/transformers/v5.0.0/en/main_classes/image_processor

**Contents:**
- Transformers
- Image Processor
- ImageProcessingMixin
  - class transformers.ImageProcessingMixin
    - from_pretrained
    - save_pretrained
- BatchFeature
  - class transformers.BatchFeature
    - convert_to_tensors
    - to

Transformers documentation

and get access to the augmented documentation experience

An image processor is in charge of loading images (optionally), preparing input features for vision models and post processing their outputs. This includes transformations such as resizing, normalization, and conversion to PyTorch and Numpy tensors. It may also include model specific post-processing such as converting logits to segmentation masks. Fast image processors are available for a few models and more will be added in the future. They are based on the torchvision library and provide a significant speed-up, especially when processing on GPU. They have the same API as the base image processors and can be used as drop-in replacements. To use a fast image processor, you need to install the torchvision library, and set the use_fast argument to True when instantiating the image processor:

Note that use_fast will be set to True by default in a future release.

When using a fast image processor, you can also set the device argument to specify the device on which the processing should be done. By default, the processing is done on the same device as the inputs if the inputs are tensors, or on the CPU otherwise.

Here are some speed comparisons between the base and fast image processors for the DETR and RT-DETR models, and how they impact overall inference time:

These benchmarks were run on an AWS EC2 g5.2xlarge instance, utilizing an NVIDIA A10G Tensor Core GPU.

This is an image processor mixin used to provide saving/loading functionality for sequential and image feature extractors.

( pretrained_model_name_or_path: str | os.PathLike cache_dir: str | os.PathLike | None = None force_download: bool = False local_files_only: bool = False token: str | bool | None = None revision: str = 'main' **kwargs )

Instantiate a type of ImageProcessingMixin from an image processor.

( save_directory: str | os.PathLike push_to_hub: bool = False **kwargs )

Save an image processor object to the directory save_directory, so that it can be re-loaded using the from_pretrained() class method.

( data: dict[str, typing.Any] | None = None tensor_type: None | str | transformers.utils.generic.TensorType = None skip_tensor_conversion: list[str] | set[str] | None = None )

Holds the output of the pad() and feature extractor specific __call__ methods.

This class is derived from a python dictionary and can be used as a dictionary.

( tensor_type: str | transformers.utils.generic.TensorType | None = None skip_tensor_conversion: list[str] | set[str] | None = None )

Convert the inner content to tensors.

Note: Values that don’t have an array-like structure (e.g., strings, dicts, lists of strings) are automatically skipped and won’t be converted to tensors. Ragged arrays (lists of arrays with different lengths) are still attempted, though they may raise errors during conversion.

( *args **kwargs ) → BatchFeature

The same instance after modification.

The same instance after modification.

Send all values to device by calling v.to(*args, **kwargs) (PyTorch only). This should support casting in different dtypes and sending the BatchFeature to a different device.

( image: ndarray size: dict data_format: str | transformers.image_utils.ChannelDimension | None = None input_data_format: str | transformers.image_utils.ChannelDimension | None = None **kwargs )

Center crop an image to (size["height"], size["width"]). If the input size is smaller than crop_size along any edge, the image is padded with 0’s and then center cropped.

( image: ndarray mean: float | collections.abc.Iterable[float] std: float | collections.abc.Iterable[float] data_format: str | transformers.image_utils.ChannelDimension | None = None input_data_format: str | transformers.image_utils.ChannelDimension | None = None **kwargs ) → np.ndarray

The normalized image.

The normalized image.

Normalize an image. image = (image - image_mean) / image_std.

( image: ndarray scale: float data_format: str | transformers.image_utils.ChannelDimension | None = None input_data_format: str | transformers.image_utils.ChannelDimension | None = None **kwargs ) → np.ndarray

Rescale an image by a scale factor. image = image * scale.

( **kwargs: typing_extensions.Unpack[transformers.processing_utils.ImagesKwargs] )

( image: torch.Tensor size: SizeDict **kwargs ) → torch.Tensor

The center cropped image.

The center cropped image.

Note: override torchvision’s center_crop to have the same behavior as the slow processor. Center crop an image to (size["height"], size["width"]). If the input size is smaller than crop_size along any edge, the image is padded with 0’s and then center cropped.

( image: torch.Tensor new_size: tuple interpolation: typing.Optional[ForwardRef('tvF.InterpolationMode')] = None antialias: bool = True )

A wrapper around tvF.resize so that it is compatible with torch.compile when the image is a uint8 tensor.

( image: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']] ) → ImageInput

Converts an image to RGB format. Only converts if the image is of type PIL.Image.Image, otherwise returns the image as is.

Filter out the unused kwargs from the kwargs dictionary.

( image: torch.Tensor mean: float | collections.abc.Iterable[float] std: float | collections.abc.Iterable[float] **kwargs ) → torch.Tensor

The normalized image.

The normalized image.

Normalize an image. image = (image - image_mean) / image_std.

( images: list pad_size: SizeDict = None fill_value: int | None = 0 padding_mode: str | None = 'constant' return_mask: bool = False disable_grouping: bool | None = False is_nested: bool | None = False **kwargs ) → Union[tuple[torch.Tensor, torch.Tensor], torch.Tensor]

Union[tuple[torch.Tensor, torch.Tensor], torch.Tensor]

The padded images and pixel masks if return_mask is True.

The padded images and pixel masks if return_mask is True.

Pads images to (pad_size["height"], pad_size["width"]) or to the largest size in the batch.

( images: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']] *args **kwargs: typing_extensions.Unpack[transformers.processing_utils.ImagesKwargs] ) → <class 'transformers.image_processing_base.BatchFeature'>

<class 'transformers.image_processing_base.BatchFeature'>

data (dict) — Dictionary of lists/arrays/tensors returned by the call method (‘pixel_values’, etc.). tensor_type (Union[None, str, TensorType], optional) — You can give a tensor_type here to convert the lists of integers in PyTorch/Numpy Tensors at initialization.

( image: torch.Tensor scale: float **kwargs ) → torch.Tensor

Rescale an image by a scale factor. image = image * scale.

( images: torch.Tensor do_rescale: bool rescale_factor: float do_normalize: bool image_mean: float | list[float] image_std: float | list[float] )

Rescale and normalize images.

( image: torch.Tensor size: SizeDict interpolation: typing.Optional[ForwardRef('tvF.InterpolationMode')] = None antialias: bool = True **kwargs ) → torch.Tensor

Resize an image to (size["height"], size["width"]).

---

## Transformers

**URL:** https://huggingface.co/docs/transformers/v5.0.0/en/add_new_pipeline

**Contents:**
- Transformers
- Adding a new pipeline
- Design choices
- Create a pipeline
- Register a pipeline
- Share your pipeline
  - Upload to the Hub
  - Add to Transformers

Transformers documentation

Adding a new pipeline

and get access to the augmented documentation experience

Make Pipeline your own by subclassing it and implementing a few methods. Share the code with the community on the Hub and register the pipeline with Transformers so that everyone can quickly and easily use it.

This guide will walk you through the process of adding a new pipeline to Transformers.

At a minimum, you only need to provide Pipeline with an appropriate input for a task. This is also where you should begin when designing your pipeline.

Decide what input types Pipeline can accept. It can be strings, raw bytes, dictionaries, and so on. Try to keep the inputs in pure Python where possible because it’s more compatible. Next, decide on the output Pipeline should return. Again, keeping the output in Python is the simplest and best option because it’s easier to work with.

Keeping the inputs and outputs simple, and ideally JSON-serializable, makes it easier for users to run your Pipeline without needing to learn new object types. It’s also common to support many different input types for even greater ease of use. For example, making an audio file acceptable from a filename, URL, or raw bytes gives the user more flexibility in how they provide the audio data.

With an input and output decided, you can start implementing Pipeline. Your pipeline should inherit from the base Pipeline class and include 4 methods.

For example, add a top_k parameter in postprocess to return the top 5 most likely classes. Then in _sanitize_parameters, check if the user passed in top_k and add it to postprocess_kwargs.

Now the pipeline can return the top most likely labels if a user chooses to.

Register the new task your pipeline supports in the PIPELINE_REGISTRY. The registry defines:

Share your pipeline with the community on the Hub or you can add it directly to Transformers.

It’s faster to upload your pipeline code to the Hub because it doesn’t require a review from the Transformers team. Adding the pipeline to Transformers may be slower because it requires a review and you need to add tests to ensure your Pipeline works.

Add your pipeline code to the Hub in a Python file.

For example, a custom pipeline for sentence pair classification might look like the following code below.

Save the code in a file named pair_classification.py, and import and register it as shown below.

The register_pipeline function registers the pipeline details (task type, pipeline class, supported backends) to a models config.json file.

Call push_to_hub() to push the pipeline to the Hub. The Python file containing the code is copied to the Hub, and the pipelines model and tokenizer are also saved and pushed to the Hub. Your pipeline should now be available on the Hub under your namespace.

To use the pipeline, add trust_remote_code=True when loading the pipeline.

Adding a custom pipeline to Transformers requires adding tests to make sure everything works as expected, and requesting a review from the Transformers team.

Add your pipeline code as a new module to the pipelines submodule, and add it to the list of tasks defined in pipelines/init.py.

Next, add a new test for the pipeline in transformers/tests/pipelines. You can look at the other tests for examples of how to test your pipeline.

The run_pipeline_test function should be very generic and run on the models defined in model_mapping. This is important for testing future compatibility with new models.

You’ll also notice ANY is used throughout the run_pipeline_test function. The models are random, so you can’t check the actual values. Using ANY allows the test to match the output of the pipeline type instead.

Finally, you should also implement the following 4 tests.

---

## Transformers

**URL:** https://huggingface.co/docs/transformers/v5.0.0/en/model_doc/bert

**Contents:**
- Transformers
- BERT
- Notes
- BertConfig
  - class transformers.BertConfig
- BertTokenizer
  - class transformers.BertTokenizer
    - get_special_tokens_mask
    - save_vocabulary
- BertTokenizerLegacy

Transformers documentation

and get access to the augmented documentation experience

This model was released on 2018-10-11 and added to Hugging Face Transformers on 2020-11-16.

BERT is a bidirectional transformer pretrained on unlabeled text to predict masked tokens in a sentence and to predict whether one sentence follows another. The main idea is that by randomly masking some tokens, the model can train on text to the left and right, giving it a more thorough understanding. BERT is also very versatile because its learned language representations can be adapted for other NLP tasks by fine-tuning an additional layer or head.

You can find all the original BERT checkpoints under the BERT collection.

Click on the BERT models in the right sidebar for more examples of how to apply BERT to different language tasks.

The example below demonstrates how to predict the [MASK] token with Pipeline, AutoModel, and from the command line.

( vocab_size = 30522 hidden_size = 768 num_hidden_layers = 12 num_attention_heads = 12 intermediate_size = 3072 hidden_act = 'gelu' hidden_dropout_prob = 0.1 attention_probs_dropout_prob = 0.1 max_position_embeddings = 512 type_vocab_size = 2 initializer_range = 0.02 layer_norm_eps = 1e-12 pad_token_id = 0 use_cache = True classifier_dropout = None is_decoder = False add_cross_attention = False bos_token_id = None eos_token_id = None tie_word_embeddings = True **kwargs )

This is the configuration class to store the configuration of a BertModel. It is used to instantiate a BERT model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the BERT google-bert/bert-base-uncased architecture.

Configuration objects inherit from PreTrainedConfig and can be used to control the model outputs. Read the documentation from PreTrainedConfig for more information.

( vocab: str | dict[str, int] | None = None do_lower_case: bool = False unk_token: str = '[UNK]' sep_token: str = '[SEP]' pad_token: str = '[PAD]' cls_token: str = '[CLS]' mask_token: str = '[MASK]' tokenize_chinese_chars: bool = True strip_accents: bool | None = None **kwargs )

Construct a BERT tokenizer (backed by HuggingFace’s tokenizers library). Based on WordPiece.

This tokenizer inherits from TokenizersBackend which contains most of the main methods. Users should refer to this superclass for more information regarding those methods.

( token_ids_0: list[int] token_ids_1: list[int] | None = None already_has_special_tokens: bool = False ) → A list of integers in the range [0, 1]

A list of integers in the range [0, 1]

1 for a special token, 0 for a sequence token.

1 for a special token, 0 for a sequence token.

Retrieve sequence ids from a token list that has no special tokens added.

For fast tokenizers, data collators call this with already_has_special_tokens=True to build a mask over an already-formatted sequence. In that case, we compute the mask by checking membership in all_special_ids.

( save_directory: str filename_prefix: str | None = None )

( vocab_file do_lower_case = True do_basic_tokenize = True never_split = None unk_token = '[UNK]' sep_token = '[SEP]' pad_token = '[PAD]' cls_token = '[CLS]' mask_token = '[MASK]' tokenize_chinese_chars = True strip_accents = None clean_up_tokenization_spaces = True **kwargs )

This should likely be deactivated for Japanese (see this issue).

Construct a BERT tokenizer. Based on WordPiece.

This tokenizer inherits from PreTrainedTokenizer which contains most of the main methods. Users should refer to this superclass for more information regarding those methods.

( token_ids_0: list token_ids_1: list[int] | None = None ) → List[int]

List of input IDs with the appropriate special tokens.

List of input IDs with the appropriate special tokens.

Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and adding special tokens. A BERT sequence has the following format:

Converts a sequence of tokens (string) in a single string.

( token_ids_0: list token_ids_1: list[int] | None = None already_has_special_tokens: bool = False ) → List[int]

A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.

A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.

Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding special tokens using the tokenizer prepare_for_model method.

( vocab: str | dict[str, int] | None = None do_lower_case: bool = False unk_token: str = '[UNK]' sep_token: str = '[SEP]' pad_token: str = '[PAD]' cls_token: str = '[CLS]' mask_token: str = '[MASK]' tokenize_chinese_chars: bool = True strip_accents: bool | None = None **kwargs )

Construct a BERT tokenizer (backed by HuggingFace’s tokenizers library). Based on WordPiece.

This tokenizer inherits from TokenizersBackend which contains most of the main methods. Users should refer to this superclass for more information regarding those methods.

( config add_pooling_layer = True )

The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of cross-attention is added between the self-attention layers, following the architecture described in Attention is all you need by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

To behave as an decoder the model needs to be initialized with the is_decoder argument of the configuration set to True. To be used in a Seq2Seq model, the model needs to initialized with both is_decoder argument and add_cross_attention set to True; an encoder_hidden_states is then expected as an input to the forward pass.

This model inherits from PreTrainedModel. Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch torch.nn.Module subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

( input_ids: torch.Tensor | None = None attention_mask: torch.Tensor | None = None token_type_ids: torch.Tensor | None = None position_ids: torch.Tensor | None = None inputs_embeds: torch.Tensor | None = None encoder_hidden_states: torch.Tensor | None = None encoder_attention_mask: torch.Tensor | None = None past_key_values: transformers.cache_utils.Cache | None = None use_cache: bool | None = None cache_position: torch.Tensor | None = None **kwargs: typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs] ) → transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions or tuple(torch.FloatTensor)

Indices can be obtained using AutoTokenizer. See PreTrainedTokenizer.encode() and PreTrainedTokenizer.call() for details.

What are attention masks?

What are token type IDs?

What are position IDs?

Only Cache instance is allowed as input, see our kv cache guide. If no past_key_values are passed, DynamicCache will be initialized by default.

The model will output the same cache format that is fed as input.

If past_key_values are used, the user is expected to input only unprocessed input_ids (those that don’t have their past key value states given to this model) of shape (batch_size, unprocessed_length) instead of all input_ids of shape (batch_size, sequence_length).

transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions or tuple(torch.FloatTensor)

A transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions or a tuple of torch.FloatTensor (if return_dict=False is passed or when config.return_dict=False) comprising various elements depending on the configuration (BertConfig) and inputs. last_hidden_state (torch.FloatTensor of shape (batch_size, sequence_length, hidden_size)) — Sequence of hidden-states at the output of the last layer of the model. pooler_output (torch.FloatTensor of shape (batch_size, hidden_size)) — Last layer hidden-state of the first token of the sequence (classification token) after further processing through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns the classification token after processing through a linear layer and a tanh activation function. The linear layer weights are trained from the next sentence prediction (classification) objective during pretraining. hidden_states (tuple(torch.FloatTensor), optional, returned when output_hidden_states=True is passed or when config.output_hidden_states=True) — Tuple of torch.FloatTensor (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape (batch_size, sequence_length, hidden_size). Hidden-states of the model at the output of each layer plus the optional initial embedding outputs. attentions (tuple(torch.FloatTensor), optional, returned when output_attentions=True is passed or when config.output_attentions=True) — Tuple of torch.FloatTensor (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length). Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads. cross_attentions (tuple(torch.FloatTensor), optional, returned when output_attentions=True and config.add_cross_attention=True is passed or when config.output_attentions=True) — Tuple of torch.FloatTensor (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length). Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the weighted average in the cross-attention heads. past_key_values (Cache, optional, returned when use_cache=True is passed or when config.use_cache=True) — It is a Cache instance. For more details, see our kv cache guide. Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if config.is_encoder_decoder=True in the cross-attention blocks) that can be used (see past_key_values input) to speed up sequential decoding.

A transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions or a tuple of torch.FloatTensor (if return_dict=False is passed or when config.return_dict=False) comprising various elements depending on the configuration (BertConfig) and inputs.

last_hidden_state (torch.FloatTensor of shape (batch_size, sequence_length, hidden_size)) — Sequence of hidden-states at the output of the last layer of the model.

pooler_output (torch.FloatTensor of shape (batch_size, hidden_size)) — Last layer hidden-state of the first token of the sequence (classification token) after further processing through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns the classification token after processing through a linear layer and a tanh activation function. The linear layer weights are trained from the next sentence prediction (classification) objective during pretraining.

hidden_states (tuple(torch.FloatTensor), optional, returned when output_hidden_states=True is passed or when config.output_hidden_states=True) — Tuple of torch.FloatTensor (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape (batch_size, sequence_length, hidden_size).

Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.

attentions (tuple(torch.FloatTensor), optional, returned when output_attentions=True is passed or when config.output_attentions=True) — Tuple of torch.FloatTensor (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length).

Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

cross_attentions (tuple(torch.FloatTensor), optional, returned when output_attentions=True and config.add_cross_attention=True is passed or when config.output_attentions=True) — Tuple of torch.FloatTensor (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length).

Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the weighted average in the cross-attention heads.

past_key_values (Cache, optional, returned when use_cache=True is passed or when config.use_cache=True) — It is a Cache instance. For more details, see our kv cache guide.

Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if config.is_encoder_decoder=True in the cross-attention blocks) that can be used (see past_key_values input) to speed up sequential decoding.

The BertModel forward method, overrides the __call__ special method.

Although the recipe for forward pass needs to be defined within this function, one should call the Module instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Bert Model with two heads on top as done during the pretraining: a masked language modeling head and a next sentence prediction (classification) head.

This model inherits from PreTrainedModel. Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch torch.nn.Module subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

( input_ids: torch.Tensor | None = None attention_mask: torch.Tensor | None = None token_type_ids: torch.Tensor | None = None position_ids: torch.Tensor | None = None inputs_embeds: torch.Tensor | None = None labels: torch.Tensor | None = None next_sentence_label: torch.Tensor | None = None **kwargs: typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs] ) → transformers.models.bert.modeling_bert.BertForPreTrainingOutput or tuple(torch.FloatTensor)

Indices can be obtained using AutoTokenizer. See PreTrainedTokenizer.encode() and PreTrainedTokenizer.call() for details.

What are attention masks?

What are token type IDs?

What are position IDs?

transformers.models.bert.modeling_bert.BertForPreTrainingOutput or tuple(torch.FloatTensor)

A transformers.models.bert.modeling_bert.BertForPreTrainingOutput or a tuple of torch.FloatTensor (if return_dict=False is passed or when config.return_dict=False) comprising various elements depending on the configuration (BertConfig) and inputs. loss (*optional*, returned when labels is provided, torch.FloatTensor of shape (1,)) — Total loss as the sum of the masked language modeling loss and the next sequence prediction (classification) loss. prediction_logits (torch.FloatTensor of shape (batch_size, sequence_length, config.vocab_size)) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax). seq_relationship_logits (torch.FloatTensor of shape (batch_size, 2)) — Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation before SoftMax). hidden_states (tuple[torch.FloatTensor] | None.hidden_states, returned when output_hidden_states=True is passed or when config.output_hidden_states=True) — Tuple of torch.FloatTensor (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape (batch_size, sequence_length, hidden_size). Hidden-states of the model at the output of each layer plus the optional initial embedding outputs. attentions (tuple[torch.FloatTensor] | None.attentions, returned when output_attentions=True is passed or when config.output_attentions=True) — Tuple of torch.FloatTensor (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length). Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

A transformers.models.bert.modeling_bert.BertForPreTrainingOutput or a tuple of torch.FloatTensor (if return_dict=False is passed or when config.return_dict=False) comprising various elements depending on the configuration (BertConfig) and inputs.

loss (*optional*, returned when labels is provided, torch.FloatTensor of shape (1,)) — Total loss as the sum of the masked language modeling loss and the next sequence prediction (classification) loss.

prediction_logits (torch.FloatTensor of shape (batch_size, sequence_length, config.vocab_size)) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).

seq_relationship_logits (torch.FloatTensor of shape (batch_size, 2)) — Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation before SoftMax).

hidden_states (tuple[torch.FloatTensor] | None.hidden_states, returned when output_hidden_states=True is passed or when config.output_hidden_states=True) — Tuple of torch.FloatTensor (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape (batch_size, sequence_length, hidden_size).

Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.

attentions (tuple[torch.FloatTensor] | None.attentions, returned when output_attentions=True is passed or when config.output_attentions=True) — Tuple of torch.FloatTensor (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length).

Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

The BertForPreTraining forward method, overrides the __call__ special method.

Although the recipe for forward pass needs to be defined within this function, one should call the Module instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Bert Model with a language modeling head on top for CLM fine-tuning.

This model inherits from PreTrainedModel. Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch torch.nn.Module subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

( input_ids: torch.Tensor | None = None attention_mask: torch.Tensor | None = None token_type_ids: torch.Tensor | None = None position_ids: torch.Tensor | None = None inputs_embeds: torch.Tensor | None = None encoder_hidden_states: torch.Tensor | None = None encoder_attention_mask: torch.Tensor | None = None labels: torch.Tensor | None = None past_key_values: transformers.cache_utils.Cache | None = None use_cache: bool | None = None cache_position: torch.Tensor | None = None logits_to_keep: int | torch.Tensor = 0 **kwargs: typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs] ) → transformers.modeling_outputs.CausalLMOutputWithCrossAttentions or tuple(torch.FloatTensor)

Indices can be obtained using AutoTokenizer. See PreTrainedTokenizer.encode() and PreTrainedTokenizer.call() for details.

What are attention masks?

What are token type IDs?

What are position IDs?

Only Cache instance is allowed as input, see our kv cache guide. If no past_key_values are passed, DynamicCache will be initialized by default.

The model will output the same cache format that is fed as input.

If past_key_values are used, the user is expected to input only unprocessed input_ids (those that don’t have their past key value states given to this model) of shape (batch_size, unprocessed_length) instead of all input_ids of shape (batch_size, sequence_length).

transformers.modeling_outputs.CausalLMOutputWithCrossAttentions or tuple(torch.FloatTensor)

A transformers.modeling_outputs.CausalLMOutputWithCrossAttentions or a tuple of torch.FloatTensor (if return_dict=False is passed or when config.return_dict=False) comprising various elements depending on the configuration (BertConfig) and inputs. loss (torch.FloatTensor of shape (1,), optional, returned when labels is provided) — Language modeling loss (for next-token prediction). logits (torch.FloatTensor of shape (batch_size, sequence_length, config.vocab_size)) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax). hidden_states (tuple(torch.FloatTensor), optional, returned when output_hidden_states=True is passed or when config.output_hidden_states=True) — Tuple of torch.FloatTensor (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape (batch_size, sequence_length, hidden_size). Hidden-states of the model at the output of each layer plus the optional initial embedding outputs. attentions (tuple(torch.FloatTensor), optional, returned when output_attentions=True is passed or when config.output_attentions=True) — Tuple of torch.FloatTensor (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length). Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads. cross_attentions (tuple(torch.FloatTensor), optional, returned when output_attentions=True is passed or when config.output_attentions=True) — Tuple of torch.FloatTensor (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length). Cross attentions weights after the attention softmax, used to compute the weighted average in the cross-attention heads. past_key_values (Cache, optional, returned when use_cache=True is passed or when config.use_cache=True) — It is a Cache instance. For more details, see our kv cache guide. Contains pre-computed hidden-states (key and values in the attention blocks) that can be used (see past_key_values input) to speed up sequential decoding.

A transformers.modeling_outputs.CausalLMOutputWithCrossAttentions or a tuple of torch.FloatTensor (if return_dict=False is passed or when config.return_dict=False) comprising various elements depending on the configuration (BertConfig) and inputs.

loss (torch.FloatTensor of shape (1,), optional, returned when labels is provided) — Language modeling loss (for next-token prediction).

logits (torch.FloatTensor of shape (batch_size, sequence_length, config.vocab_size)) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).

hidden_states (tuple(torch.FloatTensor), optional, returned when output_hidden_states=True is passed or when config.output_hidden_states=True) — Tuple of torch.FloatTensor (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape (batch_size, sequence_length, hidden_size).

Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.

attentions (tuple(torch.FloatTensor), optional, returned when output_attentions=True is passed or when config.output_attentions=True) — Tuple of torch.FloatTensor (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length).

Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

cross_attentions (tuple(torch.FloatTensor), optional, returned when output_attentions=True is passed or when config.output_attentions=True) — Tuple of torch.FloatTensor (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length).

Cross attentions weights after the attention softmax, used to compute the weighted average in the cross-attention heads.

past_key_values (Cache, optional, returned when use_cache=True is passed or when config.use_cache=True) — It is a Cache instance. For more details, see our kv cache guide.

Contains pre-computed hidden-states (key and values in the attention blocks) that can be used (see past_key_values input) to speed up sequential decoding.

The BertLMHeadModel forward method, overrides the __call__ special method.

Although the recipe for forward pass needs to be defined within this function, one should call the Module instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

The Bert Model with a language modeling head on top.”

This model inherits from PreTrainedModel. Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch torch.nn.Module subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

( input_ids: torch.Tensor | None = None attention_mask: torch.Tensor | None = None token_type_ids: torch.Tensor | None = None position_ids: torch.Tensor | None = None inputs_embeds: torch.Tensor | None = None encoder_hidden_states: torch.Tensor | None = None encoder_attention_mask: torch.Tensor | None = None labels: torch.Tensor | None = None **kwargs: typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs] ) → transformers.modeling_outputs.MaskedLMOutput or tuple(torch.FloatTensor)

Indices can be obtained using AutoTokenizer. See PreTrainedTokenizer.encode() and PreTrainedTokenizer.call() for details.

What are attention masks?

What are token type IDs?

What are position IDs?

transformers.modeling_outputs.MaskedLMOutput or tuple(torch.FloatTensor)

A transformers.modeling_outputs.MaskedLMOutput or a tuple of torch.FloatTensor (if return_dict=False is passed or when config.return_dict=False) comprising various elements depending on the configuration (BertConfig) and inputs. loss (torch.FloatTensor of shape (1,), optional, returned when labels is provided) — Masked language modeling (MLM) loss. logits (torch.FloatTensor of shape (batch_size, sequence_length, config.vocab_size)) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax). hidden_states (tuple(torch.FloatTensor), optional, returned when output_hidden_states=True is passed or when config.output_hidden_states=True) — Tuple of torch.FloatTensor (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape (batch_size, sequence_length, hidden_size). Hidden-states of the model at the output of each layer plus the optional initial embedding outputs. attentions (tuple(torch.FloatTensor), optional, returned when output_attentions=True is passed or when config.output_attentions=True) — Tuple of torch.FloatTensor (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length). Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

A transformers.modeling_outputs.MaskedLMOutput or a tuple of torch.FloatTensor (if return_dict=False is passed or when config.return_dict=False) comprising various elements depending on the configuration (BertConfig) and inputs.

loss (torch.FloatTensor of shape (1,), optional, returned when labels is provided) — Masked language modeling (MLM) loss.

logits (torch.FloatTensor of shape (batch_size, sequence_length, config.vocab_size)) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).

hidden_states (tuple(torch.FloatTensor), optional, returned when output_hidden_states=True is passed or when config.output_hidden_states=True) — Tuple of torch.FloatTensor (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape (batch_size, sequence_length, hidden_size).

Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.

attentions (tuple(torch.FloatTensor), optional, returned when output_attentions=True is passed or when config.output_attentions=True) — Tuple of torch.FloatTensor (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length).

Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

The BertForMaskedLM forward method, overrides the __call__ special method.

Although the recipe for forward pass needs to be defined within this function, one should call the Module instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Bert Model with a next sentence prediction (classification) head on top.

This model inherits from PreTrainedModel. Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch torch.nn.Module subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

( input_ids: torch.Tensor | None = None attention_mask: torch.Tensor | None = None token_type_ids: torch.Tensor | None = None position_ids: torch.Tensor | None = None inputs_embeds: torch.Tensor | None = None labels: torch.Tensor | None = None **kwargs: typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs] ) → transformers.modeling_outputs.NextSentencePredictorOutput or tuple(torch.FloatTensor)

Indices can be obtained using AutoTokenizer. See PreTrainedTokenizer.encode() and PreTrainedTokenizer.call() for details.

What are attention masks?

What are token type IDs?

What are position IDs?

transformers.modeling_outputs.NextSentencePredictorOutput or tuple(torch.FloatTensor)

A transformers.modeling_outputs.NextSentencePredictorOutput or a tuple of torch.FloatTensor (if return_dict=False is passed or when config.return_dict=False) comprising various elements depending on the configuration (BertConfig) and inputs. loss (torch.FloatTensor of shape (1,), optional, returned when next_sentence_label is provided) — Next sequence prediction (classification) loss. logits (torch.FloatTensor of shape (batch_size, 2)) — Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation before SoftMax). hidden_states (tuple(torch.FloatTensor), optional, returned when output_hidden_states=True is passed or when config.output_hidden_states=True) — Tuple of torch.FloatTensor (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape (batch_size, sequence_length, hidden_size). Hidden-states of the model at the output of each layer plus the optional initial embedding outputs. attentions (tuple(torch.FloatTensor), optional, returned when output_attentions=True is passed or when config.output_attentions=True) — Tuple of torch.FloatTensor (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length). Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

A transformers.modeling_outputs.NextSentencePredictorOutput or a tuple of torch.FloatTensor (if return_dict=False is passed or when config.return_dict=False) comprising various elements depending on the configuration (BertConfig) and inputs.

loss (torch.FloatTensor of shape (1,), optional, returned when next_sentence_label is provided) — Next sequence prediction (classification) loss.

logits (torch.FloatTensor of shape (batch_size, 2)) — Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation before SoftMax).

hidden_states (tuple(torch.FloatTensor), optional, returned when output_hidden_states=True is passed or when config.output_hidden_states=True) — Tuple of torch.FloatTensor (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape (batch_size, sequence_length, hidden_size).

Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.

attentions (tuple(torch.FloatTensor), optional, returned when output_attentions=True is passed or when config.output_attentions=True) — Tuple of torch.FloatTensor (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length).

Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

The BertForNextSentencePrediction forward method, overrides the __call__ special method.

Although the recipe for forward pass needs to be defined within this function, one should call the Module instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Bert Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled output) e.g. for GLUE tasks.

This model inherits from PreTrainedModel. Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch torch.nn.Module subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

( input_ids: torch.Tensor | None = None attention_mask: torch.Tensor | None = None token_type_ids: torch.Tensor | None = None position_ids: torch.Tensor | None = None inputs_embeds: torch.Tensor | None = None labels: torch.Tensor | None = None **kwargs: typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs] ) → transformers.modeling_outputs.SequenceClassifierOutput or tuple(torch.FloatTensor)

Indices can be obtained using AutoTokenizer. See PreTrainedTokenizer.encode() and PreTrainedTokenizer.call() for details.

What are attention masks?

What are token type IDs?

What are position IDs?

transformers.modeling_outputs.SequenceClassifierOutput or tuple(torch.FloatTensor)

A transformers.modeling_outputs.SequenceClassifierOutput or a tuple of torch.FloatTensor (if return_dict=False is passed or when config.return_dict=False) comprising various elements depending on the configuration (BertConfig) and inputs. loss (torch.FloatTensor of shape (1,), optional, returned when labels is provided) — Classification (or regression if config.num_labels==1) loss. logits (torch.FloatTensor of shape (batch_size, config.num_labels)) — Classification (or regression if config.num_labels==1) scores (before SoftMax). hidden_states (tuple(torch.FloatTensor), optional, returned when output_hidden_states=True is passed or when config.output_hidden_states=True) — Tuple of torch.FloatTensor (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape (batch_size, sequence_length, hidden_size). Hidden-states of the model at the output of each layer plus the optional initial embedding outputs. attentions (tuple(torch.FloatTensor), optional, returned when output_attentions=True is passed or when config.output_attentions=True) — Tuple of torch.FloatTensor (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length). Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

A transformers.modeling_outputs.SequenceClassifierOutput or a tuple of torch.FloatTensor (if return_dict=False is passed or when config.return_dict=False) comprising various elements depending on the configuration (BertConfig) and inputs.

loss (torch.FloatTensor of shape (1,), optional, returned when labels is provided) — Classification (or regression if config.num_labels==1) loss.

logits (torch.FloatTensor of shape (batch_size, config.num_labels)) — Classification (or regression if config.num_labels==1) scores (before SoftMax).

hidden_states (tuple(torch.FloatTensor), optional, returned when output_hidden_states=True is passed or when config.output_hidden_states=True) — Tuple of torch.FloatTensor (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape (batch_size, sequence_length, hidden_size).

Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.

attentions (tuple(torch.FloatTensor), optional, returned when output_attentions=True is passed or when config.output_attentions=True) — Tuple of torch.FloatTensor (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length).

Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

The BertForSequenceClassification forward method, overrides the __call__ special method.

Although the recipe for forward pass needs to be defined within this function, one should call the Module instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example of single-label classification:

Example of multi-label classification:

The Bert Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a softmax) e.g. for RocStories/SWAG tasks.

This model inherits from PreTrainedModel. Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch torch.nn.Module subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

( input_ids: torch.Tensor | None = None attention_mask: torch.Tensor | None = None token_type_ids: torch.Tensor | None = None position_ids: torch.Tensor | None = None inputs_embeds: torch.Tensor | None = None labels: torch.Tensor | None = None **kwargs: typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs] ) → transformers.modeling_outputs.MultipleChoiceModelOutput or tuple(torch.FloatTensor)

Indices can be obtained using AutoTokenizer. See PreTrainedTokenizer.encode() and PreTrainedTokenizer.call() for details.

What are attention masks?

What are token type IDs?

What are position IDs?

transformers.modeling_outputs.MultipleChoiceModelOutput or tuple(torch.FloatTensor)

A transformers.modeling_outputs.MultipleChoiceModelOutput or a tuple of torch.FloatTensor (if return_dict=False is passed or when config.return_dict=False) comprising various elements depending on the configuration (BertConfig) and inputs. loss (torch.FloatTensor of shape (1,), optional, returned when labels is provided) — Classification loss. logits (torch.FloatTensor of shape (batch_size, num_choices)) — num_choices is the second dimension of the input tensors. (see input_ids above). Classification scores (before SoftMax). hidden_states (tuple(torch.FloatTensor), optional, returned when output_hidden_states=True is passed or when config.output_hidden_states=True) — Tuple of torch.FloatTensor (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape (batch_size, sequence_length, hidden_size). Hidden-states of the model at the output of each layer plus the optional initial embedding outputs. attentions (tuple(torch.FloatTensor), optional, returned when output_attentions=True is passed or when config.output_attentions=True) — Tuple of torch.FloatTensor (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length). Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

A transformers.modeling_outputs.MultipleChoiceModelOutput or a tuple of torch.FloatTensor (if return_dict=False is passed or when config.return_dict=False) comprising various elements depending on the configuration (BertConfig) and inputs.

loss (torch.FloatTensor of shape (1,), optional, returned when labels is provided) — Classification loss.

logits (torch.FloatTensor of shape (batch_size, num_choices)) — num_choices is the second dimension of the input tensors. (see input_ids above).

Classification scores (before SoftMax).

hidden_states (tuple(torch.FloatTensor), optional, returned when output_hidden_states=True is passed or when config.output_hidden_states=True) — Tuple of torch.FloatTensor (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape (batch_size, sequence_length, hidden_size).

Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.

attentions (tuple(torch.FloatTensor), optional, returned when output_attentions=True is passed or when config.output_attentions=True) — Tuple of torch.FloatTensor (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length).

Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

The BertForMultipleChoice forward method, overrides the __call__ special method.

Although the recipe for forward pass needs to be defined within this function, one should call the Module instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

The Bert transformer with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks.

This model inherits from PreTrainedModel. Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch torch.nn.Module subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

( input_ids: torch.Tensor | None = None attention_mask: torch.Tensor | None = None token_type_ids: torch.Tensor | None = None position_ids: torch.Tensor | None = None inputs_embeds: torch.Tensor | None = None labels: torch.Tensor | None = None **kwargs: typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs] ) → transformers.modeling_outputs.TokenClassifierOutput or tuple(torch.FloatTensor)

Indices can be obtained using AutoTokenizer. See PreTrainedTokenizer.encode() and PreTrainedTokenizer.call() for details.

What are attention masks?

What are token type IDs?

What are position IDs?

transformers.modeling_outputs.TokenClassifierOutput or tuple(torch.FloatTensor)

A transformers.modeling_outputs.TokenClassifierOutput or a tuple of torch.FloatTensor (if return_dict=False is passed or when config.return_dict=False) comprising various elements depending on the configuration (BertConfig) and inputs. loss (torch.FloatTensor of shape (1,), optional, returned when labels is provided) — Classification loss. logits (torch.FloatTensor of shape (batch_size, sequence_length, config.num_labels)) — Classification scores (before SoftMax). hidden_states (tuple(torch.FloatTensor), optional, returned when output_hidden_states=True is passed or when config.output_hidden_states=True) — Tuple of torch.FloatTensor (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape (batch_size, sequence_length, hidden_size). Hidden-states of the model at the output of each layer plus the optional initial embedding outputs. attentions (tuple(torch.FloatTensor), optional, returned when output_attentions=True is passed or when config.output_attentions=True) — Tuple of torch.FloatTensor (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length). Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

A transformers.modeling_outputs.TokenClassifierOutput or a tuple of torch.FloatTensor (if return_dict=False is passed or when config.return_dict=False) comprising various elements depending on the configuration (BertConfig) and inputs.

loss (torch.FloatTensor of shape (1,), optional, returned when labels is provided) — Classification loss.

logits (torch.FloatTensor of shape (batch_size, sequence_length, config.num_labels)) — Classification scores (before SoftMax).

hidden_states (tuple(torch.FloatTensor), optional, returned when output_hidden_states=True is passed or when config.output_hidden_states=True) — Tuple of torch.FloatTensor (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape (batch_size, sequence_length, hidden_size).

Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.

attentions (tuple(torch.FloatTensor), optional, returned when output_attentions=True is passed or when config.output_attentions=True) — Tuple of torch.FloatTensor (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length).

Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

The BertForTokenClassification forward method, overrides the __call__ special method.

Although the recipe for forward pass needs to be defined within this function, one should call the Module instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

The Bert transformer with a span classification head on top for extractive question-answering tasks like SQuAD (a linear layer on top of the hidden-states output to compute span start logits and span end logits).

This model inherits from PreTrainedModel. Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch torch.nn.Module subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

( input_ids: torch.Tensor | None = None attention_mask: torch.Tensor | None = None token_type_ids: torch.Tensor | None = None position_ids: torch.Tensor | None = None inputs_embeds: torch.Tensor | None = None start_positions: torch.Tensor | None = None end_positions: torch.Tensor | None = None **kwargs: typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs] ) → transformers.modeling_outputs.QuestionAnsweringModelOutput or tuple(torch.FloatTensor)

Indices can be obtained using AutoTokenizer. See PreTrainedTokenizer.encode() and PreTrainedTokenizer.call() for details.

What are attention masks?

What are token type IDs?

What are position IDs?

transformers.modeling_outputs.QuestionAnsweringModelOutput or tuple(torch.FloatTensor)

A transformers.modeling_outputs.QuestionAnsweringModelOutput or a tuple of torch.FloatTensor (if return_dict=False is passed or when config.return_dict=False) comprising various elements depending on the configuration (BertConfig) and inputs. loss (torch.FloatTensor of shape (1,), optional, returned when labels is provided) — Total span extraction loss is the sum of a Cross-Entropy for the start and end positions. start_logits (torch.FloatTensor of shape (batch_size, sequence_length)) — Span-start scores (before SoftMax). end_logits (torch.FloatTensor of shape (batch_size, sequence_length)) — Span-end scores (before SoftMax). hidden_states (tuple(torch.FloatTensor), optional, returned when output_hidden_states=True is passed or when config.output_hidden_states=True) — Tuple of torch.FloatTensor (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape (batch_size, sequence_length, hidden_size). Hidden-states of the model at the output of each layer plus the optional initial embedding outputs. attentions (tuple(torch.FloatTensor), optional, returned when output_attentions=True is passed or when config.output_attentions=True) — Tuple of torch.FloatTensor (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length). Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

A transformers.modeling_outputs.QuestionAnsweringModelOutput or a tuple of torch.FloatTensor (if return_dict=False is passed or when config.return_dict=False) comprising various elements depending on the configuration (BertConfig) and inputs.

loss (torch.FloatTensor of shape (1,), optional, returned when labels is provided) — Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.

start_logits (torch.FloatTensor of shape (batch_size, sequence_length)) — Span-start scores (before SoftMax).

end_logits (torch.FloatTensor of shape (batch_size, sequence_length)) — Span-end scores (before SoftMax).

hidden_states (tuple(torch.FloatTensor), optional, returned when output_hidden_states=True is passed or when config.output_hidden_states=True) — Tuple of torch.FloatTensor (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape (batch_size, sequence_length, hidden_size).

Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.

attentions (tuple(torch.FloatTensor), optional, returned when output_attentions=True is passed or when config.output_attentions=True) — Tuple of torch.FloatTensor (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length).

Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

The BertForQuestionAnswering forward method, overrides the __call__ special method.

Although the recipe for forward pass needs to be defined within this function, one should call the Module instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

( loss: torch.FloatTensor | None = None prediction_logits: torch.FloatTensor | None = None seq_relationship_logits: torch.FloatTensor | None = None hidden_states: tuple[torch.FloatTensor] | None = None attentions: tuple[torch.FloatTensor] | None = None )

Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.

Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

Output type of BertForPreTraining.

---

## Transformers

**URL:** https://huggingface.co/docs/transformers/v5.0.0/en/model_doc/align

**Contents:**
- Transformers
- ALIGN
- Notes
- Resources
- AlignConfig
  - class transformers.AlignConfig
- AlignTextConfig
  - class transformers.AlignTextConfig
- AlignVisionConfig
  - class transformers.AlignVisionConfig

Transformers documentation

and get access to the augmented documentation experience

This model was released on 2021-02-11 and added to Hugging Face Transformers on 2023-03-01.

ALIGN is pretrained on a noisy 1.8 billion alt‑text and image pair dataset to show that scale can make up for the noise. It uses a dual‑encoder architecture, EfficientNet for images and BERT for text, and a contrastive loss to align similar image–text embeddings together while pushing different embeddings apart. Once trained, ALIGN can encode any image and candidate captions into a shared vector space for zero‑shot retrieval or classification without requiring extra labels. This scale‑first approach reduces dataset curation costs and powers state‑of‑the‑art image–text retrieval and zero‑shot ImageNet classification.

You can find all the original ALIGN checkpoints under the Kakao Brain organization.

Click on the ALIGN models in the right sidebar for more examples of how to apply ALIGN to different vision and text related tasks.

The example below demonstrates zero-shot image classification with Pipeline or the AutoModel class.

ALIGN projects the text and visual features into latent space and the dot product between the projected image and text features is used as the similarity score. The example below demonstrates how to calculate the image-text similarity score with AlignProcessor and AlignModel.

( text_config = None vision_config = None projection_dim = 640 temperature_init_value = 1.0 initializer_range = 0.02 **kwargs )

AlignConfig is the configuration class to store the configuration of a AlignModel. It is used to instantiate a ALIGN model according to the specified arguments, defining the text model and vision model configs. Instantiating a configuration with the defaults will yield a similar configuration to that of the ALIGN kakaobrain/align-base architecture.

Configuration objects inherit from PreTrainedConfig and can be used to control the model outputs. Read the documentation from PreTrainedConfig for more information.

( vocab_size = 30522 hidden_size = 768 num_hidden_layers = 12 num_attention_heads = 12 intermediate_size = 3072 hidden_act = 'gelu' hidden_dropout_prob = 0.1 attention_probs_dropout_prob = 0.1 max_position_embeddings = 512 type_vocab_size = 2 initializer_range = 0.02 layer_norm_eps = 1e-12 pad_token_id = 0 bos_token_id = None eos_token_id = None **kwargs )

This is the configuration class to store the configuration of a AlignTextModel. It is used to instantiate a ALIGN text encoder according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the text encoder of the ALIGN kakaobrain/align-base architecture. The default values here are copied from BERT.

Configuration objects inherit from PreTrainedConfig and can be used to control the model outputs. Read the documentation from PreTrainedConfig for more information.

( num_channels: int = 3 image_size: int = 600 width_coefficient: float = 2.0 depth_coefficient: float = 3.1 depth_divisor: int = 8 kernel_sizes: list = [3, 3, 5, 3, 5, 5, 3] in_channels: list = [32, 16, 24, 40, 80, 112, 192] out_channels: list = [16, 24, 40, 80, 112, 192, 320] depthwise_padding: list = [] strides: list = [1, 2, 2, 2, 1, 2, 1] num_block_repeats: list = [1, 2, 2, 3, 3, 4, 1] expand_ratios: list = [1, 6, 6, 6, 6, 6, 6] squeeze_expansion_ratio: float = 0.25 hidden_act: str = 'swish' hidden_dim: int = 2560 pooling_type: str = 'mean' initializer_range: float = 0.02 batch_norm_eps: float = 0.001 batch_norm_momentum: float = 0.99 drop_connect_rate: float = 0.2 **kwargs )

This is the configuration class to store the configuration of a AlignVisionModel. It is used to instantiate a ALIGN vision encoder according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the vision encoder of the ALIGN kakaobrain/align-base architecture. The default values are copied from EfficientNet (efficientnet-b7)

Configuration objects inherit from PreTrainedConfig and can be used to control the model outputs. Read the documentation from PreTrainedConfig for more information.

( image_processor tokenizer )

Constructs a AlignProcessor which wraps a image processor and a tokenizer into a single processor.

AlignProcessor offers all the functionalities of EfficientNetImageProcessorFast and BertTokenizer. See the ~EfficientNetImageProcessorFast and ~BertTokenizer for more information.

( images: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor'], NoneType] = None text: str | list[str] | list[list[str]] | None = None videos: typing.Union[list['PIL.Image.Image'], numpy.ndarray, ForwardRef('torch.Tensor'), list[numpy.ndarray], list['torch.Tensor'], list[list['PIL.Image.Image']], list[list[numpy.ndarray]], list[list['torch.Tensor']], transformers.video_utils.URL, list[transformers.video_utils.URL], list[list[transformers.video_utils.URL]], transformers.video_utils.Path, list[transformers.video_utils.Path], list[list[transformers.video_utils.Path]], NoneType] = None audio: typing.Union[numpy.ndarray, ForwardRef('torch.Tensor'), collections.abc.Sequence[numpy.ndarray], collections.abc.Sequence['torch.Tensor'], NoneType] = None **kwargs: typing_extensions.Unpack[transformers.processing_utils.ProcessingKwargs] ) → BatchFeature

A BatchFeature object with processed inputs in a dict format.

A BatchFeature object with processed inputs in a dict format.

Main method to prepare for model inputs. This method forwards the each modality argument to its own processor along with kwargs. Please refer to the docstring of the each processor attributes for more information.

( config: AlignConfig )

The bare Align Model outputting raw hidden-states without any specific head on top.

This model inherits from PreTrainedModel. Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch torch.nn.Module subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

( input_ids: torch.LongTensor | None = None pixel_values: torch.FloatTensor | None = None attention_mask: torch.Tensor | None = None token_type_ids: torch.Tensor | None = None position_ids: torch.Tensor | None = None inputs_embeds: torch.Tensor | None = None return_loss: bool | None = None output_attentions: bool | None = None output_hidden_states: bool | None = None return_dict: bool | None = None **kwargs ) → transformers.models.align.modeling_align.AlignOutput or tuple(torch.FloatTensor)

Indices can be obtained using AutoTokenizer. See PreTrainedTokenizer.encode() and PreTrainedTokenizer.call() for details.

What are attention masks?

What are token type IDs?

What are position IDs?

transformers.models.align.modeling_align.AlignOutput or tuple(torch.FloatTensor)

A transformers.models.align.modeling_align.AlignOutput or a tuple of torch.FloatTensor (if return_dict=False is passed or when config.return_dict=False) comprising various elements depending on the configuration (AlignConfig) and inputs. loss (torch.FloatTensor of shape (1,), optional, returned when return_loss is True) — Contrastive loss for image-text similarity. logits_per_image (torch.FloatTensor of shape (image_batch_size, text_batch_size)) — The scaled dot product scores between image_embeds and text_embeds. This represents the image-text similarity scores. logits_per_text (torch.FloatTensor of shape (text_batch_size, image_batch_size)) — The scaled dot product scores between text_embeds and image_embeds. This represents the text-image similarity scores. text_embeds (torch.FloatTensor of shape (batch_size, output_dim) — The text embeddings obtained by applying the projection layer to the pooled output of AlignTextModel. image_embeds (torch.FloatTensor of shape (batch_size, output_dim) — The output of AlignVisionModel. text_model_output (<class '~modeling_outputs.BaseModelOutputWithPooling'>.text_model_output, defaults to None) — The output of the AlignTextModel. vision_model_output (<class '~modeling_outputs.BaseModelOutputWithPoolingAndNoAttention'>.vision_model_output, defaults to None) — The output of the AlignVisionModel.

A transformers.models.align.modeling_align.AlignOutput or a tuple of torch.FloatTensor (if return_dict=False is passed or when config.return_dict=False) comprising various elements depending on the configuration (AlignConfig) and inputs.

The AlignModel forward method, overrides the __call__ special method.

Although the recipe for forward pass needs to be defined within this function, one should call the Module instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

( input_ids: torch.Tensor | None = None attention_mask: torch.Tensor | None = None token_type_ids: torch.Tensor | None = None position_ids: torch.Tensor | None = None inputs_embeds: torch.Tensor | None = None **kwargs: typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs] ) → transformers.modeling_outputs.BaseModelOutputWithPooling or tuple(torch.FloatTensor)

Indices can be obtained using AutoTokenizer. See PreTrainedTokenizer.encode() and PreTrainedTokenizer.call() for details.

What are attention masks?

What are token type IDs?

What are position IDs?

transformers.modeling_outputs.BaseModelOutputWithPooling or tuple(torch.FloatTensor)

A transformers.modeling_outputs.BaseModelOutputWithPooling or a tuple of torch.FloatTensor (if return_dict=False is passed or when config.return_dict=False) comprising various elements depending on the configuration (AlignConfig) and inputs. last_hidden_state (torch.FloatTensor of shape (batch_size, sequence_length, hidden_size)) — Sequence of hidden-states at the output of the last layer of the model. pooler_output (torch.FloatTensor of shape (batch_size, hidden_size)) — Last layer hidden-state of the first token of the sequence (classification token) after further processing through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns the classification token after processing through a linear layer and a tanh activation function. The linear layer weights are trained from the next sentence prediction (classification) objective during pretraining. hidden_states (tuple(torch.FloatTensor), optional, returned when output_hidden_states=True is passed or when config.output_hidden_states=True) — Tuple of torch.FloatTensor (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape (batch_size, sequence_length, hidden_size). Hidden-states of the model at the output of each layer plus the optional initial embedding outputs. attentions (tuple(torch.FloatTensor), optional, returned when output_attentions=True is passed or when config.output_attentions=True) — Tuple of torch.FloatTensor (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length). Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

A transformers.modeling_outputs.BaseModelOutputWithPooling or a tuple of torch.FloatTensor (if return_dict=False is passed or when config.return_dict=False) comprising various elements depending on the configuration (AlignConfig) and inputs.

last_hidden_state (torch.FloatTensor of shape (batch_size, sequence_length, hidden_size)) — Sequence of hidden-states at the output of the last layer of the model.

pooler_output (torch.FloatTensor of shape (batch_size, hidden_size)) — Last layer hidden-state of the first token of the sequence (classification token) after further processing through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns the classification token after processing through a linear layer and a tanh activation function. The linear layer weights are trained from the next sentence prediction (classification) objective during pretraining.

hidden_states (tuple(torch.FloatTensor), optional, returned when output_hidden_states=True is passed or when config.output_hidden_states=True) — Tuple of torch.FloatTensor (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape (batch_size, sequence_length, hidden_size).

Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.

attentions (tuple(torch.FloatTensor), optional, returned when output_attentions=True is passed or when config.output_attentions=True) — Tuple of torch.FloatTensor (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length).

Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

( pixel_values: FloatTensor **kwargs: typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs] ) → transformers.modeling_outputs.BaseModelOutputWithPooling or tuple(torch.FloatTensor)

transformers.modeling_outputs.BaseModelOutputWithPooling or tuple(torch.FloatTensor)

A transformers.modeling_outputs.BaseModelOutputWithPooling or a tuple of torch.FloatTensor (if return_dict=False is passed or when config.return_dict=False) comprising various elements depending on the configuration (AlignConfig) and inputs. last_hidden_state (torch.FloatTensor of shape (batch_size, sequence_length, hidden_size)) — Sequence of hidden-states at the output of the last layer of the model. pooler_output (torch.FloatTensor of shape (batch_size, hidden_size)) — Last layer hidden-state of the first token of the sequence (classification token) after further processing through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns the classification token after processing through a linear layer and a tanh activation function. The linear layer weights are trained from the next sentence prediction (classification) objective during pretraining. hidden_states (tuple(torch.FloatTensor), optional, returned when output_hidden_states=True is passed or when config.output_hidden_states=True) — Tuple of torch.FloatTensor (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape (batch_size, sequence_length, hidden_size). Hidden-states of the model at the output of each layer plus the optional initial embedding outputs. attentions (tuple(torch.FloatTensor), optional, returned when output_attentions=True is passed or when config.output_attentions=True) — Tuple of torch.FloatTensor (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length). Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

A transformers.modeling_outputs.BaseModelOutputWithPooling or a tuple of torch.FloatTensor (if return_dict=False is passed or when config.return_dict=False) comprising various elements depending on the configuration (AlignConfig) and inputs.

last_hidden_state (torch.FloatTensor of shape (batch_size, sequence_length, hidden_size)) — Sequence of hidden-states at the output of the last layer of the model.

pooler_output (torch.FloatTensor of shape (batch_size, hidden_size)) — Last layer hidden-state of the first token of the sequence (classification token) after further processing through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns the classification token after processing through a linear layer and a tanh activation function. The linear layer weights are trained from the next sentence prediction (classification) objective during pretraining.

hidden_states (tuple(torch.FloatTensor), optional, returned when output_hidden_states=True is passed or when config.output_hidden_states=True) — Tuple of torch.FloatTensor (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape (batch_size, sequence_length, hidden_size).

Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.

attentions (tuple(torch.FloatTensor), optional, returned when output_attentions=True is passed or when config.output_attentions=True) — Tuple of torch.FloatTensor (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length).

Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

( config: AlignTextConfig add_pooling_layer: bool = True )

The text model from ALIGN without any head or projection on top.

This model inherits from PreTrainedModel. Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch torch.nn.Module subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

( input_ids: torch.Tensor | None = None attention_mask: torch.Tensor | None = None token_type_ids: torch.Tensor | None = None position_ids: torch.Tensor | None = None inputs_embeds: torch.Tensor | None = None output_attentions: bool | None = None output_hidden_states: bool | None = None return_dict: bool | None = None **kwargs ) → transformers.modeling_outputs.BaseModelOutputWithPooling or tuple(torch.FloatTensor)

Indices can be obtained using AutoTokenizer. See PreTrainedTokenizer.encode() and PreTrainedTokenizer.call() for details.

What are attention masks?

What are token type IDs?

What are position IDs?

transformers.modeling_outputs.BaseModelOutputWithPooling or tuple(torch.FloatTensor)

A transformers.modeling_outputs.BaseModelOutputWithPooling or a tuple of torch.FloatTensor (if return_dict=False is passed or when config.return_dict=False) comprising various elements depending on the configuration (AlignConfig) and inputs. last_hidden_state (torch.FloatTensor of shape (batch_size, sequence_length, hidden_size)) — Sequence of hidden-states at the output of the last layer of the model. pooler_output (torch.FloatTensor of shape (batch_size, hidden_size)) — Last layer hidden-state of the first token of the sequence (classification token) after further processing through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns the classification token after processing through a linear layer and a tanh activation function. The linear layer weights are trained from the next sentence prediction (classification) objective during pretraining. hidden_states (tuple(torch.FloatTensor), optional, returned when output_hidden_states=True is passed or when config.output_hidden_states=True) — Tuple of torch.FloatTensor (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape (batch_size, sequence_length, hidden_size). Hidden-states of the model at the output of each layer plus the optional initial embedding outputs. attentions (tuple(torch.FloatTensor), optional, returned when output_attentions=True is passed or when config.output_attentions=True) — Tuple of torch.FloatTensor (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length). Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

A transformers.modeling_outputs.BaseModelOutputWithPooling or a tuple of torch.FloatTensor (if return_dict=False is passed or when config.return_dict=False) comprising various elements depending on the configuration (AlignConfig) and inputs.

last_hidden_state (torch.FloatTensor of shape (batch_size, sequence_length, hidden_size)) — Sequence of hidden-states at the output of the last layer of the model.

pooler_output (torch.FloatTensor of shape (batch_size, hidden_size)) — Last layer hidden-state of the first token of the sequence (classification token) after further processing through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns the classification token after processing through a linear layer and a tanh activation function. The linear layer weights are trained from the next sentence prediction (classification) objective during pretraining.

hidden_states (tuple(torch.FloatTensor), optional, returned when output_hidden_states=True is passed or when config.output_hidden_states=True) — Tuple of torch.FloatTensor (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape (batch_size, sequence_length, hidden_size).

Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.

attentions (tuple(torch.FloatTensor), optional, returned when output_attentions=True is passed or when config.output_attentions=True) — Tuple of torch.FloatTensor (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length).

Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

The AlignTextModel forward method, overrides the __call__ special method.

Although the recipe for forward pass needs to be defined within this function, one should call the Module instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

( config: AlignVisionConfig )

The vision model from ALIGN without any head or projection on top.

This model inherits from PreTrainedModel. Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch torch.nn.Module subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

( pixel_values: torch.FloatTensor | None = None output_hidden_states: bool | None = None return_dict: bool | None = None **kwargs ) → transformers.modeling_outputs.BaseModelOutputWithPoolingAndNoAttention or tuple(torch.FloatTensor)

transformers.modeling_outputs.BaseModelOutputWithPoolingAndNoAttention or tuple(torch.FloatTensor)

A transformers.modeling_outputs.BaseModelOutputWithPoolingAndNoAttention or a tuple of torch.FloatTensor (if return_dict=False is passed or when config.return_dict=False) comprising various elements depending on the configuration (AlignConfig) and inputs. last_hidden_state (torch.FloatTensor of shape (batch_size, num_channels, height, width)) — Sequence of hidden-states at the output of the last layer of the model. pooler_output (torch.FloatTensor of shape (batch_size, hidden_size)) — Last layer hidden-state after a pooling operation on the spatial dimensions. hidden_states (tuple(torch.FloatTensor), optional, returned when output_hidden_states=True is passed or when config.output_hidden_states=True) — Tuple of torch.FloatTensor (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape (batch_size, num_channels, height, width). Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.

A transformers.modeling_outputs.BaseModelOutputWithPoolingAndNoAttention or a tuple of torch.FloatTensor (if return_dict=False is passed or when config.return_dict=False) comprising various elements depending on the configuration (AlignConfig) and inputs.

last_hidden_state (torch.FloatTensor of shape (batch_size, num_channels, height, width)) — Sequence of hidden-states at the output of the last layer of the model.

pooler_output (torch.FloatTensor of shape (batch_size, hidden_size)) — Last layer hidden-state after a pooling operation on the spatial dimensions.

hidden_states (tuple(torch.FloatTensor), optional, returned when output_hidden_states=True is passed or when config.output_hidden_states=True) — Tuple of torch.FloatTensor (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape (batch_size, num_channels, height, width).

Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.

The AlignVisionModel forward method, overrides the __call__ special method.

Although the recipe for forward pass needs to be defined within this function, one should call the Module instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

---

## Transformers

**URL:** https://huggingface.co/docs/transformers/v5.0.0/en/main_classes/callback

**Contents:**
- Transformers
- Callbacks
- Available Callbacks
  - class transformers.integrations.CometCallback
    - setup
  - class transformers.DefaultFlowCallback
  - class transformers.PrinterCallback
  - class transformers.ProgressCallback
  - class transformers.EarlyStoppingCallback
  - class transformers.integrations.TensorBoardCallback

Transformers documentation

and get access to the augmented documentation experience

Callbacks are objects that can customize the behavior of the training loop in the PyTorch Trainer that can inspect the training loop state (for progress reporting, logging on TensorBoard or other ML platforms…) and take decisions (like early stopping).

Callbacks are “read only” pieces of code, apart from the TrainerControl object they return, they cannot change anything in the training loop. For customizations that require changes in the training loop, you should subclass Trainer and override the methods you need (see trainer for examples).

By default, TrainingArguments.report_to is set to "all", so a Trainer will use the following callbacks.

If a package is installed but you don’t wish to use the accompanying integration, you can change TrainingArguments.report_to to a list of just those integrations you want to use (e.g. ["azure_ml", "wandb"]).

The main class that implements callbacks is TrainerCallback. It gets the TrainingArguments used to instantiate the Trainer, can access that Trainer’s internal state via TrainerState, and can take some actions on the training loop via TrainerControl.

Here is the list of the available TrainerCallback in the library:

A TrainerCallback that sends the logs to Comet ML.

Setup the optional Comet integration.

For a number of configurable items in the environment, see here.

A TrainerCallback that handles the default flow of the training loop for logs, evaluation and checkpoints.

A bare TrainerCallback that just prints the logs.

( max_str_len: int = 100 )

A TrainerCallback that displays the progress of training or evaluation. You can modify max_str_len to control how long strings are truncated when logging.

( early_stopping_patience: int = 1 early_stopping_threshold: float | None = 0.0 )

A TrainerCallback that handles early stopping.

This callback depends on TrainingArguments argument load_best_model_at_end functionality to set best_metric in TrainerState. Note that if the TrainingArguments argument save_steps differs from eval_steps, the early stopping will not occur until the next save step.

A TrainerCallback that sends the logs to TensorBoard.

A TrainerCallback that logs metrics to Trackio.

It records training metrics, model (and PEFT) configuration, and GPU memory usage. If nvidia-ml-py is installed, GPU power consumption is also tracked.

( args state model **kwargs )

Setup the optional Trackio integration.

To customize the setup you can also set the arguments project, trackio_space_id and hub_private_repo in TrainingArguments. Please refer to the docstring of for more details.

A TrainerCallback that logs metrics, media, model checkpoints to Weight and Biases.

( args state model **kwargs )

Setup the optional Weights & Biases (wandb) integration.

One can subclass and override this method to customize the setup if needed. Find more information here. You can also override the following environment variables:

A TrainerCallback that sends the logs to MLflow. Can be disabled by setting environment variable DISABLE_MLFLOW_INTEGRATION = TRUE.

Setup the optional MLflow integration.

( azureml_run = None )

A TrainerCallback that sends the logs to AzureML.

A TrainerCallback that tracks the CO2 emission of training.

A TrainerCallback that sends the logs to ClearML.

A TrainerCallback that logs to DagsHub. Extends MLflowCallback

Setup the DagsHub’s Logging integration.

( save_log_history: bool = True sync_checkpoints: bool = True )

A TrainerCallback that sends the logs to Flyte. NOTE: This callback only works within a Flyte task.

( live: typing.Optional[typing.Any] = None log_model: typing.Union[typing.Literal['all'], bool, NoneType] = None **kwargs )

A TrainerCallback that sends the logs to DVCLive.

Use the environment variables below in setup to configure the integration. To customize this callback beyond those environment variables, see here.

Setup the optional DVCLive integration. To customize this callback beyond the environment variables below, see here.

A TrainerCallback that logs metrics, media, model checkpoints to SwanLab.

( args state model **kwargs )

Setup the optional SwanLab (swanlab) integration.

One can subclass and override this method to customize the setup if needed. Find more information here.

You can also override the following environment variables. Find more information about environment variables here

SWANLAB_API_KEY (str, optional, defaults to None): Cloud API Key. During login, this environment variable is checked first. If it doesn’t exist, the system checks if the user is already logged in. If not, the login process is initiated.

SWANLAB_PROJECT (str, optional, defaults to None): Set this to a custom string to store results in a different project. If not specified, the name of the current running directory is used.

SWANLAB_LOG_DIR (str, optional, defaults to swanlog): This environment variable specifies the storage path for log files when running in local mode. By default, logs are saved in a folder named swanlog under the working directory.

SWANLAB_MODE (Literal["local", "cloud", "disabled"], optional, defaults to cloud): SwanLab’s parsing mode, which involves callbacks registered by the operator. Currently, there are three modes: local, cloud, and disabled. Note: Case-sensitive. Find more information here

SWANLAB_LOG_MODEL (str, optional, defaults to None): SwanLab does not currently support the save mode functionality.This feature will be available in a future release

SWANLAB_WEB_HOST (str, optional, defaults to None): Web address for the SwanLab cloud environment for private version (its free)

SWANLAB_API_HOST (str, optional, defaults to None): API address for the SwanLab cloud environment for private version (its free)

Those are only accessible in the event on_evaluate.

Those are only accessible in the event on_log.

A class for objects that will inspect the state of the training loop at some events and take some decisions. At each of those events the following arguments are available:

The control object is the only one that can be changed by the callback, in which case the event that changes it should return the modified version.

The argument args, state and control are positionals for all events, all the others are grouped in kwargs. You can unpack the ones you need in the signature of the event using them. As an example, see the code of the simple PrinterCallback.

( args: TrainingArguments state: TrainerState control: TrainerControl **kwargs )

Event called at the beginning of an epoch.

( args: TrainingArguments state: TrainerState control: TrainerControl **kwargs )

Event called at the end of an epoch.

( args: TrainingArguments state: TrainerState control: TrainerControl **kwargs )

Event called after an evaluation phase.

( args: TrainingArguments state: TrainerState control: TrainerControl **kwargs )

Event called at the end of the initialization of the Trainer.

( args: TrainingArguments state: TrainerState control: TrainerControl **kwargs )

Event called after logging the last logs.

( args: TrainingArguments state: TrainerState control: TrainerControl **kwargs )

Event called after the optimizer step but before gradients are zeroed out. Useful for monitoring gradients.

( args: TrainingArguments state: TrainerState control: TrainerControl **kwargs )

Event called before the optimizer step but after gradient clipping. Useful for monitoring gradients.

( args: TrainingArguments state: TrainerState control: TrainerControl metrics **kwargs )

Event called after a successful prediction.

( args: TrainingArguments state: TrainerState control: TrainerControl **kwargs )

Event called after a prediction step.

( args: TrainingArguments state: TrainerState control: TrainerControl **kwargs )

Event called before pushing the model to the hub, at the beginning of Trainer.push_to_hub and Trainer._push_from_checkpoint.

( args: TrainingArguments state: TrainerState control: TrainerControl **kwargs )

Event called after a checkpoint save.

( args: TrainingArguments state: TrainerState control: TrainerControl **kwargs )

Event called at the beginning of a training step. If using gradient accumulation, one training step might take several inputs.

( args: TrainingArguments state: TrainerState control: TrainerControl **kwargs )

Event called at the end of a training step. If using gradient accumulation, one training step might take several inputs.

( args: TrainingArguments state: TrainerState control: TrainerControl **kwargs )

Event called at the end of an substep during gradient accumulation.

( args: TrainingArguments state: TrainerState control: TrainerControl **kwargs )

Event called at the beginning of training.

( args: TrainingArguments state: TrainerState control: TrainerControl **kwargs )

Event called at the end of training.

Here is an example of how to register a custom callback with the PyTorch Trainer:

Another way to register a callback is to call trainer.add_callback() as follows:

( epoch: float | None = None global_step: int = 0 max_steps: int = 0 logging_steps: int = 500 eval_steps: int = 500 save_steps: int = 500 train_batch_size: int | None = None num_train_epochs: int = 0 num_input_tokens_seen: int = 0 total_flos: float = 0 log_history: list = None best_metric: float | None = None best_global_step: int | None = None best_model_checkpoint: str | None = None is_local_process_zero: bool = True is_world_process_zero: bool = True is_hyper_param_search: bool = False trial_name: str | None = None trial_params: dict[str, str | float | int | bool] | None = None stateful_callbacks: list['TrainerCallback'] | None = None )

A class containing the Trainer inner state that will be saved along the model and optimizer when checkpointing and passed to the TrainerCallback.

In all this class, one step is to be understood as one update step. When using gradient accumulation, one update step may require several forward and backward passes: if you use gradient_accumulation_steps=n, then one update step requires going through n batches.

Calculates and stores the absolute value for logging, eval, and save steps based on if it was a proportion or not.

( trainer max_steps num_train_epochs trial )

Stores the initial training references needed in self

Create an instance from the content of json_path.

Save the content of this instance in JSON format inside json_path.

( should_training_stop: bool = False should_epoch_stop: bool = False should_save: bool = False should_evaluate: bool = False should_log: bool = False )

If True, this variable will not be set back to False. The training will just stop.

If True, this variable will be set back to False at the beginning of the next epoch.

If True, this variable will be set back to False at the beginning of the next step.

If True, this variable will be set back to False at the beginning of the next step.

If True, this variable will be set back to False at the beginning of the next step.

A class that handles the Trainer control flow. This class is used by the TrainerCallback to activate some switches in the training loop.

---

## Transformers

**URL:** https://huggingface.co/docs/transformers/v5.0.0/en/main_classes/data_collator

**Contents:**
- Transformers
- Data Collator
- Default data collator
    - transformers.default_data_collator
- DefaultDataCollator
  - class transformers.DefaultDataCollator
- DataCollatorWithPadding
  - class transformers.DataCollatorWithPadding
- DataCollatorForTokenClassification
  - class transformers.DataCollatorForTokenClassification

Transformers documentation

and get access to the augmented documentation experience

Data collators are objects that will form a batch by using a list of dataset elements as input. These elements are of the same type as the elements of train_dataset or eval_dataset.

To be able to build batches, data collators may apply some processing (like padding). Some of them (like DataCollatorForLanguageModeling) also apply some random data augmentation (like random masking) on the formed batch.

Examples of use can be found in the example scripts or example notebooks.

( features: list return_tensors = 'pt' )

Very simple data collator that simply collates batches of dict-like objects and performs special handling for potential keys named:

Does not do any additional preprocessing: property names of the input object will be used as corresponding inputs to the model. See glue and ner for example of how it’s useful.

( return_tensors: str = 'pt' )

Very simple data collator that simply collates batches of dict-like objects and performs special handling for potential keys named:

Does not do any additional preprocessing: property names of the input object will be used as corresponding inputs to the model. See glue and ner for example of how it’s useful.

This is an object (like other data collators) rather than a pure function like default_data_collator. This can be helpful if you need to set a return_tensors value at initialization.

( tokenizer: PreTrainedTokenizerBase padding: bool | str | transformers.utils.generic.PaddingStrategy = True max_length: int | None = None pad_to_multiple_of: int | None = None return_tensors: str = 'pt' )

This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.0 (Volta).

Data collator that will dynamically pad the inputs received.

( tokenizer: PreTrainedTokenizerBase padding: bool | str | transformers.utils.generic.PaddingStrategy = True max_length: int | None = None pad_to_multiple_of: int | None = None label_pad_token_id: int = -100 return_tensors: str = 'pt' )

This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.0 (Volta).

Data collator that will dynamically pad the inputs received, as well as the labels.

( tokenizer: PreTrainedTokenizerBase model: typing.Optional[typing.Any] = None padding: bool | str | transformers.utils.generic.PaddingStrategy = True max_length: int | None = None pad_to_multiple_of: int | None = None label_pad_token_id: int = -100 return_tensors: str = 'pt' )

This is useful when using label_smoothing to avoid calculating loss twice.

This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.0 (Volta).

Data collator that will dynamically pad the inputs received, as well as the labels.

( tokenizer: PreTrainedTokenizerBase mlm: bool = True whole_word_mask: bool = False mlm_probability: float | None = 0.15 mask_replace_prob: float = 0.8 random_replace_prob: float = 0.1 pad_to_multiple_of: int | None = None return_tensors: str = 'pt' seed: int | None = None )

Data collator used for language modeling. Inputs are dynamically padded to the maximum length of a batch if they are not all of the same length.

For best performance, this data collator should be used with a dataset having items that are dictionaries or BatchEncoding, with the "special_tokens_mask" key, as returned by a PreTrainedTokenizer or a PreTrainedTokenizerFast with the argument return_special_tokens_mask=True.

All masked tokens replaced by [MASK]:

No [MASK] replacement, only random tokens:

Balanced replacement:

Note: The sum of mask_replace_prob and random_replace_prob must not exceed 1. If their sum is less than 1, the remaining proportion will consist of masked tokens left unchanged.

( inputs: typing.Any special_tokens_mask: typing.Optional[typing.Any] = None offset_mapping: typing.Optional[typing.Any] = None )

Prepare masked tokens inputs/labels for masked language modeling.

( inputs: typing.Any special_tokens_mask: typing.Optional[typing.Any] = None offset_mapping: typing.Optional[typing.Any] = None )

Prepare masked tokens inputs/labels for masked language modeling.

Data collator used for language modeling that masks entire words.

( inputs: typing.Any special_tokens_mask: typing.Optional[typing.Any] = None offset_mapping: typing.Optional[typing.Any] = None )

Prepare masked tokens inputs/labels for masked language modeling.

( inputs: typing.Any special_tokens_mask: typing.Optional[typing.Any] = None offset_mapping: typing.Optional[typing.Any] = None )

Prepare masked tokens inputs/labels for masked language modeling.

( tokenizer: PreTrainedTokenizerBase plm_probability: float = 0.16666666666666666 max_span_length: int = 5 return_tensors: str = 'pt' )

Data collator used for permutation language modeling.

( inputs: typing.Any )

The masked tokens to be predicted for a particular sequence are determined by the following algorithm:

( inputs: typing.Any )

The masked tokens to be predicted for a particular sequence are determined by the following algorithm:

( *args return_position_ids = True separator_id = -100 return_flash_attn_kwargs = False return_seq_idx = False **kwargs )

Data collator used for padding free approach. Does the following:

Using DataCollatorWithFlattening will flatten the entire mini batch into single long sequence. Make sure your attention computation is able to handle it!

( tokenizer: PreTrainedTokenizerBase padding: bool | str | transformers.utils.generic.PaddingStrategy = True max_length: int | None = None pad_to_multiple_of: int | None = None return_tensors: str = 'pt' )

This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).

Data collator that dynamically pads a batch of nested examples for multiple choice, so that all choices of all examples have the same length.

---

## Transformers

**URL:** https://huggingface.co/docs/transformers/v5.0.0/en/main_classes/feature_extractor

**Contents:**
- Transformers
- Feature Extractor
- FeatureExtractionMixin
  - class transformers.FeatureExtractionMixin
    - from_pretrained
    - save_pretrained
- SequenceFeatureExtractor
  - class transformers.SequenceFeatureExtractor
    - pad
- BatchFeature

Transformers documentation

and get access to the augmented documentation experience

A feature extractor is in charge of preparing input features for audio models. This includes feature extraction from sequences, e.g., pre-processing audio files to generate Log-Mel Spectrogram features, and conversion to NumPy and PyTorch tensors.

This is a feature extraction mixin used to provide saving/loading functionality for sequential and audio feature extractors.

( pretrained_model_name_or_path: str | os.PathLike cache_dir: str | os.PathLike | None = None force_download: bool = False local_files_only: bool = False token: str | bool | None = None revision: str = 'main' **kwargs )

Instantiate a type of FeatureExtractionMixin from a feature extractor, e.g. a derived class of SequenceFeatureExtractor.

( save_directory: str | os.PathLike push_to_hub: bool = False **kwargs )

Save a feature_extractor object to the directory save_directory, so that it can be re-loaded using the from_pretrained() class method.

( feature_size: int sampling_rate: int padding_value: float **kwargs )

This is a general feature extraction class for speech recognition.

( processed_features: transformers.feature_extraction_utils.BatchFeature | list[transformers.feature_extraction_utils.BatchFeature] | dict[str, transformers.feature_extraction_utils.BatchFeature] | dict[str, list[transformers.feature_extraction_utils.BatchFeature]] | list[dict[str, transformers.feature_extraction_utils.BatchFeature]] padding: bool | str | transformers.utils.generic.PaddingStrategy = True max_length: int | None = None truncation: bool = False pad_to_multiple_of: int | None = None return_attention_mask: bool | None = None return_tensors: str | transformers.utils.generic.TensorType | None = None )

Instead of list[float] you can have tensors (numpy arrays or PyTorch tensors), see the note above for the return type.

This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta), or on TPUs which benefit from having sequence lengths be a multiple of 128.

What are attention masks?

Pad input values / input vectors or a batch of input values / input vectors up to predefined length or to the max sequence length in the batch.

Padding side (left/right) padding values are defined at the feature extractor level (with self.padding_side, self.padding_value)

If the processed_features passed are dictionary of numpy arrays or PyTorch tensors the result will use the same type unless you provide a different tensor type with return_tensors. In the case of PyTorch tensors, you will lose the specific device of your tensors however.

( data: dict[str, typing.Any] | None = None tensor_type: None | str | transformers.utils.generic.TensorType = None skip_tensor_conversion: list[str] | set[str] | None = None )

Holds the output of the pad() and feature extractor specific __call__ methods.

This class is derived from a python dictionary and can be used as a dictionary.

( tensor_type: str | transformers.utils.generic.TensorType | None = None skip_tensor_conversion: list[str] | set[str] | None = None )

Convert the inner content to tensors.

Note: Values that don’t have an array-like structure (e.g., strings, dicts, lists of strings) are automatically skipped and won’t be converted to tensors. Ragged arrays (lists of arrays with different lengths) are still attempted, though they may raise errors during conversion.

( *args **kwargs ) → BatchFeature

The same instance after modification.

The same instance after modification.

Send all values to device by calling v.to(*args, **kwargs) (PyTorch only). This should support casting in different dtypes and sending the BatchFeature to a different device.

Mixin that contain utilities for preparing image features.

( image size ) → new_image

A center cropped PIL.Image.Image or np.ndarray or torch.Tensor of shape: (n_channels, height, width).

A center cropped PIL.Image.Image or np.ndarray or torch.Tensor of shape: (n_channels, height, width).

Crops image to the given size using a center crop. Note that if the image is too small to be cropped to the size given, it will be padded (so the returned result has the size asked).

Converts PIL.Image.Image to RGB format.

Expands 2-dimensional image to 3 dimensions.

Flips the channel order of image from RGB to BGR, or vice versa. Note that this will trigger a conversion of image to a NumPy array if it’s a PIL Image.

( image mean std rescale = False )

Normalizes image with mean and std. Note that this will trigger a conversion of image to a NumPy array if it’s a PIL Image.

( image: ndarray scale: float | int )

Rescale a numpy image by scale amount

( image size resample = None default_to_square = True max_size = None ) → image

If size is an int and default_to_square is True, then image will be resized to (size, size). If size is an int and default_to_square is False, then smaller edge of the image will be matched to this number. i.e, if height > width, then image will be rescaled to (size * height / width, size).

A resized PIL.Image.Image.

A resized PIL.Image.Image.

Resizes image. Enforces conversion of input to PIL.Image.

( image angle resample = None expand = 0 center = None translate = None fillcolor = None ) → image

A rotated PIL.Image.Image.

A rotated PIL.Image.Image.

Returns a rotated copy of image. This method returns a copy of image, rotated the given number of degrees counter clockwise around its centre.

( image rescale = None channel_first = True )

Converts image to a numpy array. Optionally rescales it and puts the channel dimension as the first dimension.

( image rescale = None )

Converts image to a PIL Image. Optionally rescales it and puts the channel dimension back as the last axis if needed.

---

## Transformers

**URL:** https://huggingface.co/docs/transformers/v5.0.0/en/model_doc/apertus

**Contents:**
- Transformers
- Apertus
- Overview
- ApertusConfig
  - class transformers.ApertusConfig
- ApertusModel
  - class transformers.ApertusModel
    - forward
- ApertusForCausalLM
  - class transformers.ApertusForCausalLM

Transformers documentation

and get access to the augmented documentation experience

This model was released on 2025-09-02 and added to Hugging Face Transformers on 2025-08-28.

Apertus is a family of large language models from the Swiss AI Initiative.

The example below demonstrates how to generate text with Pipeline or the AutoModel, and from the command line.

( vocab_size: int | None = 131072 hidden_size: int | None = 4096 intermediate_size: int | None = 14336 num_hidden_layers: int | None = 32 num_attention_heads: int | None = 32 num_key_value_heads: int | None = None hidden_act: str | None = 'xielu' max_position_embeddings: int | None = 65536 initializer_range: float | None = 0.02 rms_norm_eps: float | None = 1e-05 use_cache: bool | None = True pad_token_id: int | None = 3 bos_token_id: int | None = 1 eos_token_id: int | None = 2 tie_word_embeddings: bool | None = False rope_parameters: transformers.modeling_rope_utils.RopeParameters | None = {'rope_type': 'llama3', 'rope_theta': 12000000.0, 'factor': 8.0, 'original_max_position_embeddings': 8192, 'low_freq_factor': 1.0, 'high_freq_factor': 4.0} attention_bias: bool | None = False attention_dropout: float | None = 0.0 **kwargs )

This is the configuration class to store the configuration of a ApertusModel. It is used to instantiate a Apertus model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the Apertus-8B. e.g. swiss-ai/Apertus-8B

Configuration objects inherit from PreTrainedConfig and can be used to control the model outputs. Read the documentation from PreTrainedConfig for more information.

( config: ApertusConfig )

The bare Apertus Model outputting raw hidden-states without any specific head on top.

This model inherits from PreTrainedModel. Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch torch.nn.Module subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

( input_ids: torch.LongTensor | None = None attention_mask: torch.Tensor | None = None position_ids: torch.LongTensor | None = None past_key_values: transformers.cache_utils.Cache | None = None inputs_embeds: torch.FloatTensor | None = None cache_position: torch.LongTensor | None = None use_cache: bool | None = None **kwargs: typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs] ) → transformers.modeling_outputs.BaseModelOutputWithPast or tuple(torch.FloatTensor)

Indices can be obtained using AutoTokenizer. See PreTrainedTokenizer.encode() and PreTrainedTokenizer.call() for details.

What are attention masks?

What are position IDs?

Only Cache instance is allowed as input, see our kv cache guide. If no past_key_values are passed, DynamicCache will be initialized by default.

The model will output the same cache format that is fed as input.

If past_key_values are used, the user is expected to input only unprocessed input_ids (those that don’t have their past key value states given to this model) of shape (batch_size, unprocessed_length) instead of all input_ids of shape (batch_size, sequence_length).

transformers.modeling_outputs.BaseModelOutputWithPast or tuple(torch.FloatTensor)

A transformers.modeling_outputs.BaseModelOutputWithPast or a tuple of torch.FloatTensor (if return_dict=False is passed or when config.return_dict=False) comprising various elements depending on the configuration (ApertusConfig) and inputs. last_hidden_state (torch.FloatTensor of shape (batch_size, sequence_length, hidden_size)) — Sequence of hidden-states at the output of the last layer of the model. If past_key_values is used only the last hidden-state of the sequences of shape (batch_size, 1, hidden_size) is output. past_key_values (Cache, optional, returned when use_cache=True is passed or when config.use_cache=True) — It is a Cache instance. For more details, see our kv cache guide. Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if config.is_encoder_decoder=True in the cross-attention blocks) that can be used (see past_key_values input) to speed up sequential decoding. hidden_states (tuple(torch.FloatTensor), optional, returned when output_hidden_states=True is passed or when config.output_hidden_states=True) — Tuple of torch.FloatTensor (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape (batch_size, sequence_length, hidden_size). Hidden-states of the model at the output of each layer plus the optional initial embedding outputs. attentions (tuple(torch.FloatTensor), optional, returned when output_attentions=True is passed or when config.output_attentions=True) — Tuple of torch.FloatTensor (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length). Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

A transformers.modeling_outputs.BaseModelOutputWithPast or a tuple of torch.FloatTensor (if return_dict=False is passed or when config.return_dict=False) comprising various elements depending on the configuration (ApertusConfig) and inputs.

last_hidden_state (torch.FloatTensor of shape (batch_size, sequence_length, hidden_size)) — Sequence of hidden-states at the output of the last layer of the model.

If past_key_values is used only the last hidden-state of the sequences of shape (batch_size, 1, hidden_size) is output.

past_key_values (Cache, optional, returned when use_cache=True is passed or when config.use_cache=True) — It is a Cache instance. For more details, see our kv cache guide.

Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if config.is_encoder_decoder=True in the cross-attention blocks) that can be used (see past_key_values input) to speed up sequential decoding.

hidden_states (tuple(torch.FloatTensor), optional, returned when output_hidden_states=True is passed or when config.output_hidden_states=True) — Tuple of torch.FloatTensor (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape (batch_size, sequence_length, hidden_size).

Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.

attentions (tuple(torch.FloatTensor), optional, returned when output_attentions=True is passed or when config.output_attentions=True) — Tuple of torch.FloatTensor (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length).

Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

The ApertusModel forward method, overrides the __call__ special method.

Although the recipe for forward pass needs to be defined within this function, one should call the Module instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

The Apertus Model for causal language modeling.

This model inherits from PreTrainedModel. Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch torch.nn.Module subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

( input_ids: torch.LongTensor | None = None attention_mask: torch.Tensor | None = None position_ids: torch.LongTensor | None = None past_key_values: transformers.cache_utils.Cache | None = None inputs_embeds: torch.FloatTensor | None = None labels: torch.LongTensor | None = None use_cache: bool | None = None cache_position: torch.LongTensor | None = None logits_to_keep: int | torch.Tensor = 0 **kwargs: typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs] ) → transformers.modeling_outputs.CausalLMOutputWithPast or tuple(torch.FloatTensor)

Indices can be obtained using AutoTokenizer. See PreTrainedTokenizer.encode() and PreTrainedTokenizer.call() for details.

What are attention masks?

What are position IDs?

Only Cache instance is allowed as input, see our kv cache guide. If no past_key_values are passed, DynamicCache will be initialized by default.

The model will output the same cache format that is fed as input.

If past_key_values are used, the user is expected to input only unprocessed input_ids (those that don’t have their past key value states given to this model) of shape (batch_size, unprocessed_length) instead of all input_ids of shape (batch_size, sequence_length).

transformers.modeling_outputs.CausalLMOutputWithPast or tuple(torch.FloatTensor)

A transformers.modeling_outputs.CausalLMOutputWithPast or a tuple of torch.FloatTensor (if return_dict=False is passed or when config.return_dict=False) comprising various elements depending on the configuration (ApertusConfig) and inputs. loss (torch.FloatTensor of shape (1,), optional, returned when labels is provided) — Language modeling loss (for next-token prediction). logits (torch.FloatTensor of shape (batch_size, sequence_length, config.vocab_size)) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax). past_key_values (Cache, optional, returned when use_cache=True is passed or when config.use_cache=True) — It is a Cache instance. For more details, see our kv cache guide. Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see past_key_values input) to speed up sequential decoding. hidden_states (tuple(torch.FloatTensor), optional, returned when output_hidden_states=True is passed or when config.output_hidden_states=True) — Tuple of torch.FloatTensor (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape (batch_size, sequence_length, hidden_size). Hidden-states of the model at the output of each layer plus the optional initial embedding outputs. attentions (tuple(torch.FloatTensor), optional, returned when output_attentions=True is passed or when config.output_attentions=True) — Tuple of torch.FloatTensor (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length). Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

A transformers.modeling_outputs.CausalLMOutputWithPast or a tuple of torch.FloatTensor (if return_dict=False is passed or when config.return_dict=False) comprising various elements depending on the configuration (ApertusConfig) and inputs.

loss (torch.FloatTensor of shape (1,), optional, returned when labels is provided) — Language modeling loss (for next-token prediction).

logits (torch.FloatTensor of shape (batch_size, sequence_length, config.vocab_size)) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).

past_key_values (Cache, optional, returned when use_cache=True is passed or when config.use_cache=True) — It is a Cache instance. For more details, see our kv cache guide.

Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see past_key_values input) to speed up sequential decoding.

hidden_states (tuple(torch.FloatTensor), optional, returned when output_hidden_states=True is passed or when config.output_hidden_states=True) — Tuple of torch.FloatTensor (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape (batch_size, sequence_length, hidden_size).

Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.

attentions (tuple(torch.FloatTensor), optional, returned when output_attentions=True is passed or when config.output_attentions=True) — Tuple of torch.FloatTensor (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length).

Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

The ApertusForCausalLM forward method, overrides the __call__ special method.

Although the recipe for forward pass needs to be defined within this function, one should call the Module instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

( input_ids: torch.LongTensor | None = None attention_mask: torch.Tensor | None = None position_ids: torch.LongTensor | None = None past_key_values: transformers.cache_utils.Cache | None = None inputs_embeds: torch.FloatTensor | None = None labels: torch.LongTensor | None = None use_cache: bool | None = None **kwargs: typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs] ) → transformers.modeling_outputs.TokenClassifierOutput or tuple(torch.FloatTensor)

Indices can be obtained using AutoTokenizer. See PreTrainedTokenizer.encode() and PreTrainedTokenizer.call() for details.

What are attention masks?

What are position IDs?

Only Cache instance is allowed as input, see our kv cache guide. If no past_key_values are passed, DynamicCache will be initialized by default.

The model will output the same cache format that is fed as input.

If past_key_values are used, the user is expected to input only unprocessed input_ids (those that don’t have their past key value states given to this model) of shape (batch_size, unprocessed_length) instead of all input_ids of shape (batch_size, sequence_length).

transformers.modeling_outputs.TokenClassifierOutput or tuple(torch.FloatTensor)

A transformers.modeling_outputs.TokenClassifierOutput or a tuple of torch.FloatTensor (if return_dict=False is passed or when config.return_dict=False) comprising various elements depending on the configuration (None) and inputs. loss (torch.FloatTensor of shape (1,), optional, returned when labels is provided) — Classification loss. logits (torch.FloatTensor of shape (batch_size, sequence_length, config.num_labels)) — Classification scores (before SoftMax). hidden_states (tuple(torch.FloatTensor), optional, returned when output_hidden_states=True is passed or when config.output_hidden_states=True) — Tuple of torch.FloatTensor (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape (batch_size, sequence_length, hidden_size). Hidden-states of the model at the output of each layer plus the optional initial embedding outputs. attentions (tuple(torch.FloatTensor), optional, returned when output_attentions=True is passed or when config.output_attentions=True) — Tuple of torch.FloatTensor (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length). Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

A transformers.modeling_outputs.TokenClassifierOutput or a tuple of torch.FloatTensor (if return_dict=False is passed or when config.return_dict=False) comprising various elements depending on the configuration (None) and inputs.

loss (torch.FloatTensor of shape (1,), optional, returned when labels is provided) — Classification loss.

logits (torch.FloatTensor of shape (batch_size, sequence_length, config.num_labels)) — Classification scores (before SoftMax).

hidden_states (tuple(torch.FloatTensor), optional, returned when output_hidden_states=True is passed or when config.output_hidden_states=True) — Tuple of torch.FloatTensor (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape (batch_size, sequence_length, hidden_size).

Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.

attentions (tuple(torch.FloatTensor), optional, returned when output_attentions=True is passed or when config.output_attentions=True) — Tuple of torch.FloatTensor (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length).

Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

The GenericForTokenClassification forward method, overrides the __call__ special method.

Although the recipe for forward pass needs to be defined within this function, one should call the Module instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

---

## Transformers

**URL:** https://huggingface.co/docs/transformers/v5.0.0/en/models_timeline

**Contents:**
- Transformers
- Models Timeline

Transformers documentation

and get access to the augmented documentation experience

The Models Timeline is an interactive chart of how architectures in Transformers have changed over time. You can scroll through models in order, spanning text, vision, audio, video, and multimodal use cases.

Use the filters to narrow models by modality or task. Set custom date ranges to focus on models added during specific periods. Click a model card to see its capabilities, supported tasks, and documentation.

---

## Transformers

**URL:** https://huggingface.co/docs/transformers/v5.0.0/en/pipeline_tutorial

**Contents:**
- Transformers
- Pipeline
- Tasks
- Parameters
  - Device
  - Batch inference
  - Task-specific parameters
- Chunk batching
- Large datasets
- Large models

Transformers documentation

and get access to the augmented documentation experience

The Pipeline is a simple but powerful inference API that is readily available for a variety of machine learning tasks with any model from the Hugging Face Hub.

Tailor the Pipeline to your task with task specific parameters such as adding timestamps to an automatic speech recognition (ASR) pipeline for transcribing meeting notes. Pipeline supports GPUs, Apple Silicon, and half-precision weights to accelerate inference and save memory.

Transformers has two pipeline classes, a generic Pipeline and many individual task-specific pipelines like TextGenerationPipeline or VisualQuestionAnsweringPipeline. Load these individual pipelines by setting the task identifier in the task parameter in Pipeline. You can find the task identifier for each pipeline in their API documentation.

Each task is configured to use a default pretrained model and preprocessor, but this can be overridden with the model parameter if you want to use a different model.

For example, to use the TextGenerationPipeline with Gemma 2, set task="text-generation" and model="google/gemma-2-2b".

When you have more than one input, pass them as a list.

This guide will introduce you to the Pipeline, demonstrate its features, and show how to configure its various parameters.

Pipeline is compatible with many machine learning tasks across different modalities. Pass an appropriate input to the pipeline and it will handle the rest.

Here are some examples of how to use Pipeline for different tasks and modalities.

At a minimum, Pipeline only requires a task identifier, model, and the appropriate input. But there are many parameters available to configure the pipeline with, from task-specific parameters to optimizing performance.

This section introduces you to some of the more important parameters.

Pipeline is compatible with many hardware types, including GPUs, CPUs, Apple Silicon, and more. Configure the hardware type with the device parameter. By default, Pipeline runs on a CPU which is given by device=-1.

To run Pipeline on a GPU, set device to the associated CUDA device id. For example, device=0 runs on the first GPU.

You could also let Accelerate, a library for distributed training, automatically choose how to load and store the model weights on the appropriate device. This is especially useful if you have multiple devices. Accelerate loads and stores the model weights on the fastest device first, and then moves the weights to other devices (CPU, hard drive) as needed. Set device_map="auto" to let Accelerate choose the device.

Make sure have Accelerate is installed.

Pipeline can also process batches of inputs with the batch_size parameter. Batch inference may improve speed, especially on a GPU, but it isn’t guaranteed. Other variables such as hardware, data, and the model itself can affect whether batch inference improves speed. For this reason, batch inference is disabled by default.

In the example below, when there are 4 inputs and batch_size is set to 2, Pipeline passes a batch of 2 inputs to the model at a time.

Another good use case for batch inference is for streaming data in Pipeline.

Keep the following general rules of thumb in mind for determining whether batch inference can help improve performance.

Pipeline accepts any parameters that are supported by each individual task pipeline. Make sure to check out each individual task pipeline to see what type of parameters are available. If you can’t find a parameter that is useful for your use case, please feel free to open a GitHub issue to request it!

The examples below demonstrate some of the task-specific parameters available.

Pass the return_timestamps="word" parameter to Pipeline to return when each word was spoken.

There are some instances where you need to process data in chunks.

The ChunkPipeline class is designed to handle these use cases. Both pipeline classes are used in the same way, but since ChunkPipeline can automatically handle batching, you don’t need to worry about the number of forward passes your inputs trigger. Instead, you can optimize batch_size independently of the inputs.

The example below shows how it differs from Pipeline.

For inference with large datasets, you can iterate directly over the dataset itself. This avoids immediately allocating memory for the entire dataset, and you don’t need to worry about creating batches yourself. Try Batch inference with the batch_size parameter to see if it improves performance.

Other ways to run inference on large datasets with Pipeline include using an iterator or generator.

Accelerate enables a couple of optimizations for running large models with Pipeline. Make sure Accelerate is installed first.

The device_map="auto" setting is useful for automatically distributing the model across the fastest devices (GPUs) first before dispatching to other slower devices if available (CPU, hard drive).

Pipeline supports half-precision weights (torch.float16), which can be significantly faster and save memory. Performance loss is negligible for most models, especially for larger ones. If your hardware supports it, you can enable torch.bfloat16 instead for more range.

Inputs are internally converted to torch.float16 and it only works for models with a PyTorch backend.

Lastly, Pipeline also accepts quantized models to reduce memory usage even further. Make sure you have the bitsandbytes library installed first, and then add quantization_config to model_kwargs in the pipeline.

---

## Transformers

**URL:** https://huggingface.co/docs/transformers/v5.0.0/en/internal/tokenization_utils

**Contents:**
- Transformers
- Utilities for Tokenizers
- PreTrainedTokenizerBase
  - class transformers.PreTrainedTokenizerBase
    - __call__
    - add_special_tokens
    - add_tokens
    - apply_chat_template
    - batch_decode
    - convert_ids_to_tokens

Transformers documentation

Utilities for Tokenizers

and get access to the augmented documentation experience

This page lists all the utility functions used by the tokenizers, mainly the class PreTrainedTokenizerBase that implements the common methods between PreTrainedTokenizer and PreTrainedTokenizerFast.

Most of those are only useful if you are studying the code of the tokenizers in the library.

Base class for all tokenizer backends.

Class attributes (overridden by derived classes)

( text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None text_pair: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None text_target: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None text_pair_target: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None add_special_tokens: bool = True padding: bool | str | PaddingStrategy = False truncation: bool | str | TruncationStrategy | None = None max_length: int | None = None stride: int = 0 is_split_into_words: bool = False pad_to_multiple_of: int | None = None padding_side: str | None = None return_tensors: str | TensorType | None = None return_token_type_ids: bool | None = None return_attention_mask: bool | None = None return_overflowing_tokens: bool = False return_special_tokens_mask: bool = False return_offsets_mapping: bool = False return_length: bool = False verbose: bool = True tokenizer_kwargs: dict[str, Any] | None = None **kwargs ) → BatchEncoding

If left unset or set to None, this will use the predefined model maximum length if a maximum length is required by one of the truncation/padding parameters. If the model has no specific maximum input length (like XLNet) truncation/padding to a maximum length will be deactivated.

What are token type IDs?

What are attention masks?

This is only available on fast tokenizers inheriting from PreTrainedTokenizerFast, if using Python’s tokenizer, this method will raise NotImplementedError.

A BatchEncoding with the following fields: input_ids — List of token ids to be fed to a model. What are input IDs? token_type_ids — List of token type ids to be fed to a model (when return_token_type_ids=True or if “token_type_ids” is in self.model_input_names). What are token type IDs? attention_mask — List of indices specifying which tokens should be attended to by the model (when return_attention_mask=True or if “attention_mask” is in self.model_input_names). What are attention masks? overflowing_tokens — List of overflowing tokens sequences (when a max_length is specified and return_overflowing_tokens=True). num_truncated_tokens — Number of tokens truncated (when a max_length is specified and return_overflowing_tokens=True). special_tokens_mask — List of 0s and 1s, with 1 specifying added special tokens and 0 specifying regular sequence tokens (when add_special_tokens=True and return_special_tokens_mask=True). length — The length of the inputs (when return_length=True)

A BatchEncoding with the following fields:

input_ids — List of token ids to be fed to a model.

token_type_ids — List of token type ids to be fed to a model (when return_token_type_ids=True or if “token_type_ids” is in self.model_input_names).

What are token type IDs?

attention_mask — List of indices specifying which tokens should be attended to by the model (when return_attention_mask=True or if “attention_mask” is in self.model_input_names).

What are attention masks?

overflowing_tokens — List of overflowing tokens sequences (when a max_length is specified and return_overflowing_tokens=True).

num_truncated_tokens — Number of tokens truncated (when a max_length is specified and return_overflowing_tokens=True).

special_tokens_mask — List of 0s and 1s, with 1 specifying added special tokens and 0 specifying regular sequence tokens (when add_special_tokens=True and return_special_tokens_mask=True).

length — The length of the inputs (when return_length=True)

Main method to tokenize and prepare for the model one or several sequence(s) or one or several pair(s) of sequences.

( special_tokens_dict: dict[str, str | AddedToken | Sequence[str | AddedToken]] replace_extra_special_tokens = True ) → int

Tokens are only added if they are not already in the vocabulary (tested by checking if the tokenizer assign the index of the unk_token to them).

Number of tokens added to the vocabulary.

Number of tokens added to the vocabulary.

Add a dictionary of special tokens (eos, pad, cls, etc.) to the encoder and link them to class attributes. If special tokens are NOT in the vocabulary, they are added to it (indexed starting from the last index of the current vocabulary).

When adding new tokens to the vocabulary, you should make sure to also resize the token embedding matrix of the model so that its embedding matrix matches the tokenizer.

In order to do that, please use the resize_token_embeddings() method.

Using add_special_tokens will ensure your special tokens can be used in several ways:

When possible, special tokens are already registered for provided pretrained models (for instance BertTokenizer cls_token is already registered to be '[CLS]' and XLM’s one is also registered to be '</s>').

( new_tokens: str | AddedToken | Sequence[str | AddedToken] special_tokens: bool = False ) → int

Number of tokens added to the vocabulary.

Number of tokens added to the vocabulary.

#TODO remove this from here! PreTrainedTOkeniuzerBase should be agnostic of AddedToken.

Add a list of new tokens. If the new tokens are not in the vocabulary, they are added to the end. Added tokens and tokens from the vocabulary of the tokenization algorithm are therefore not treated in the same way.

( conversation: list[dict[str, str]] | list[list[dict[str, str]]] tools: list[dict | Callable] | None = None documents: list[dict[str, str]] | None = None chat_template: str | None = None add_generation_prompt: bool = False continue_final_message: bool = False tokenize: bool = True padding: bool | str | PaddingStrategy = False truncation: bool = False max_length: int | None = None return_tensors: str | TensorType | None = None return_dict: bool = True return_assistant_tokens_mask: bool = False tokenizer_kwargs: dict[str, Any] | None = None **kwargs ) → Union[list[int], Dict]

Union[list[int], Dict]

A list of token ids representing the tokenized chat so far, including control tokens. This output is ready to pass to the model, either directly or via methods like generate(). If return_dict is set, will return a dict of tokenizer outputs instead.

A list of token ids representing the tokenized chat so far, including control tokens. This output is ready to pass to the model, either directly or via methods like generate(). If return_dict is set, will return a dict of tokenizer outputs instead.

Converts a list of dictionaries with "role" and "content" keys to a list of token ids. This method is intended for use with chat models, and will read the tokenizer’s chat_template attribute to determine the format and control tokens to use when converting.

( sequences: list[int] | list[list[int]] | np.ndarray | torch.Tensor skip_special_tokens: bool = False clean_up_tokenization_spaces: bool | None = None **kwargs ) → list[str]

The list of decoded sentences.

The list of decoded sentences.

Convert a list of lists of token ids into a list of strings by calling decode.

This method is provided for backwards compatibility. The decode method now handles batched input natively, so you can use decode directly instead of batch_decode.

( ids: int | list[int] skip_special_tokens: bool = False ) → str or list[str]

The decoded token(s).

The decoded token(s).

Converts a single index or a sequence of indices in a token or a sequence of tokens, using the vocabulary and added tokens.

( tokens: str | list[str] ) → int or list[int]

The token id or list of token ids.

The token id or list of token ids.

Converts a token string (or a sequence of tokens) in a single integer id (or a sequence of ids), using the vocabulary.

( tokens: list[str] ) → str

Converts a sequence of tokens in a single string. The most simple way to do it is " ".join(tokens) but we often want to remove sub-word tokenization artifacts at the same time.

( token_ids: int | list[int] | list[list[int]] | np.ndarray | torch.Tensor skip_special_tokens: bool = False **kwargs ) → Union[str, list[str]]

Union[str, list[str]]

The decoded string for a single sequence, or a list of decoded strings for a batch of sequences.

The decoded string for a single sequence, or a list of decoded strings for a batch of sequences.

Converts a sequence of ids into a string, or a list of sequences into a list of strings, using the tokenizer and vocabulary with options to remove special tokens and clean up tokenization spaces.

Similar to doing self.convert_tokens_to_string(self.convert_ids_to_tokens(token_ids)).

( text: TextInput | PreTokenizedInput | EncodedInput text_pair: TextInput | PreTokenizedInput | EncodedInput | None = None add_special_tokens: bool = True padding: bool | str | PaddingStrategy = False truncation: bool | str | TruncationStrategy | None = None max_length: int | None = None stride: int = 0 padding_side: str | None = None return_tensors: str | TensorType | None = None **kwargs ) → list[int], torch.Tensor, or np.ndarray

If left unset or set to None, this will use the predefined model maximum length if a maximum length is required by one of the truncation/padding parameters. If the model has no specific maximum input length (like XLNet) truncation/padding to a maximum length will be deactivated.

list[int], torch.Tensor, or np.ndarray

The tokenized ids of the text.

The tokenized ids of the text.

Converts a string to a sequence of ids (integer), using the tokenizer and vocabulary.

Same as doing self.convert_tokens_to_ids(self.tokenize(text)).

( message: dict[str, str] conversation_history: list[dict[str, str]] | None = None **kwargs ) → list[int]

A list of token ids representing the tokenized message.

A list of token ids representing the tokenized message.

Tokenize a single message. This method is a convenience wrapper around apply_chat_template that allows you to tokenize messages one by one. This is useful for things like token-by-token streaming. This method is not guaranteed to be perfect. For some models, it may be impossible to robustly tokenize single messages. For example, if the chat template adds tokens after each message, but also has a prefix that is added to the entire chat, it will be impossible to distinguish a chat-start-token from a message-start-token. In these cases, this method will do its best to find the correct tokenization, but it may not be perfect. Note: This method does not support add_generation_prompt. If you want to add a generation prompt, you should do it separately after tokenizing the conversation.

( pretrained_model_name_or_path: str | os.PathLike *init_inputs cache_dir: str | os.PathLike | None = None force_download: bool = False local_files_only: bool = False token: str | bool | None = None revision: str = 'main' trust_remote_code = False **kwargs )

Instantiate a PreTrainedTokenizerBase (or a derived class) from a predefined tokenizer.

Passing token=True is required when you want to use a private model.

( chat_template: str | None = None tools: list[dict] | None = None ) → str

The chat template string.

The chat template string.

Retrieve the chat template string used for tokenizing chat messages. This template is used internally by the apply_chat_template method and can also be used externally to retrieve the model’s chat template for better generation tracking.

( token_ids_0: list[int] token_ids_1: list[int] | None = None already_has_special_tokens: bool = False ) → A list of integers in the range [0, 1]

A list of integers in the range [0, 1]

1 for a special token, 0 for a sequence token.

1 for a special token, 0 for a sequence token.

Retrieve sequence ids from a token list that has no special tokens added.

For fast tokenizers, data collators call this with already_has_special_tokens=True to build a mask over an already-formatted sequence. In that case, we compute the mask by checking membership in all_special_ids.

Returns the vocabulary as a dictionary of token to index.

tokenizer.get_vocab()[token] is equivalent to tokenizer.convert_tokens_to_ids(token) when token is in the vocab.

( encoded_inputs: BatchEncoding | list[BatchEncoding] | dict[str, EncodedInput] | dict[str, list[EncodedInput]] | list[dict[str, EncodedInput]] padding: bool | str | PaddingStrategy = True max_length: int | None = None pad_to_multiple_of: int | None = None padding_side: str | None = None return_attention_mask: bool | None = None return_tensors: str | TensorType | None = None verbose: bool = True )

Instead of list[int] you can have tensors (numpy arrays, or PyTorch tensors), see the note above for the return type.

This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).

What are attention masks?

Pad a single encoded input or a batch of encoded inputs up to predefined length or to the max sequence length in the batch.

Padding side (left/right) padding token ids are defined at the tokenizer level (with self.padding_side, self.pad_token_id and self.pad_token_type_id).

Please note that with a fast tokenizer, using the __call__ method is faster than using a method to encode the text followed by a call to the pad method to get a padded encoding.

If the encoded_inputs passed are dictionary of numpy arrays, or PyTorch tensors, the result will use the same type unless you provide a different tensor type with return_tensors. In the case of PyTorch tensors, you will lose the specific device of your tensors however.

( response: str | list[str | int | list[int]] | np.ndarray | torch.Tensor schema: list | dict | None = None )

Converts an output string created by generating text from a model into a parsed message dictionary. This method is intended for use with chat models, and will read the tokenizer’s response_schema attribute to control parsing, although this can be overridden by passing a response_schema argument directly.

This method is currently highly experimental and the schema specification is likely to change in future! We recommend not building production code on top of it just yet.

( repo_id: str commit_message: str | None = None commit_description: str | None = None private: bool | None = None token: bool | str | None = None revision: str | None = None create_pr: bool = False max_shard_size: int | str | None = '50GB' tags: list[str] | None = None )

Upload the tokenizer files to the 🤗 Model Hub.

( auto_class = 'AutoTokenizer' )

Register this class with a given auto class. This should only be used for custom tokenizers as the ones in the library are already mapped with AutoTokenizer.

( save_directory: str | os.PathLike tokenizer_config: dict filename_prefix: str | None save_jinja_files: bool )

Writes chat templates out to the save directory if we’re using the new format, and removes them from the tokenizer config if present. If we’re using the legacy format, it doesn’t write any files, and instead writes the templates to the tokenizer config in the correct format.

( save_directory: str | os.PathLike legacy_format: bool | None = None filename_prefix: str | None = None push_to_hub: bool = False **kwargs ) → A tuple of str

If False, will only save the tokenizer in the unified JSON format. This format is incompatible with “slow” tokenizers (not powered by the tokenizers library), so the tokenizer will not be able to be loaded in the corresponding “slow” tokenizer.

If True, will save the tokenizer in legacy format. If the “slow” tokenizer doesn’t exits, a value error is raised.

Save the full tokenizer state.

This method make sure the full tokenizer can then be re-loaded using the ~tokenization_utils_base.PreTrainedTokenizer.from_pretrained class method..

Warning,None This won’t save modifications you may have applied to the tokenizer after the instantiation (for instance, modifying tokenizer.do_lower_case after creation).

( save_directory: str filename_prefix: str | None = None ) → tuple(str)

Paths to the files saved.

Paths to the files saved.

Save only the vocabulary of the tokenizer (vocabulary + added tokens).

This method won’t save the configuration and special token mappings of the tokenizer. Use _save_pretrained() to save the whole state of the tokenizer.

( text: str pair: str | None = None add_special_tokens: bool = False **kwargs ) → list[str]

Converts a string into a sequence of tokens, replacing unknown tokens with the unk_token.

( value names = None module = None qualname = None type = None start = 1 )

Possible values for the truncation argument in PreTrainedTokenizerBase.call(). Useful for tab-completion in an IDE.

( start: ForwardRef('int') end: ForwardRef('int') )

Character span in the original string.

( start: ForwardRef('int') end: ForwardRef('int') )

Token span in an encoded string (list of tokens).

---

## Transformers

**URL:** https://huggingface.co/docs/transformers/v5.0.0/en/weightconverter

**Contents:**
- Transformers
- Dynamic weight loading
- Conversion operations
  - Chunk
  - Concatenate
  - MergeModulelist
  - SplitModulelist
  - PermuteForRope
- Fast and efficient model loading
- Reusing the dynamic loading building blocks

Transformers documentation

Dynamic weight loading

and get access to the augmented documentation experience

Checkpoints are often serialized in a format that does not match what a model expects at runtime. Quantization and parallelism frequently require reshaping, splitting, or merging tensors into the expected model format instead of loading weights as-is.

Dynamic weight loading addresses this by applying scheduled, reversible operations to checkpoint tensors as they are loaded. Transformers makes this available through WeightConverter, which maps one or more source keys to target keys by running a list of composable conversion operations. This approach adapts to new weight layouts, and supports loading quantized mixture-of-experts (MoEs) or enabling tensor parallelism and MoEs.

This guide demonstrates how to use the WeightConverter to convert tensors. Your WeightConverter should be added inside _build_checkpoint_conversion_mapping() in the conversion_mapping.py file.

The WeightConverter class has several operations that are executed when from_pretrained() is called for transforming checkpoint source tensors into model target tensors.

Operations are fully reversible. Saving reverses the conversions and returns the original checkpoint so you can easily work across different frameworks.

The Chunk operation is used to split a tensor. For example, if a model expects Q, K, and V as three separate tensors instead of a single tensor.

The Concatenate operation allows you to fuse separate tensors into a single tensor. For example, if a model expects Q, K, and V as a single tensor instead of separate tensors.

MergeModulelist merges a list of tensors into a single tensor. For example, you can compose MergeModulelist with Concatenate to stack the experts in a MoE and pack them into one tensor.

SplitModulelist splits a tensor back into a list of tensors. For example, you can split a stack of experts back into individual experts.

PermuteForRope converts weights from the interleaved format to use the sin/cos format. For example, you can compose Chunk with PermuteForRope to split a fused QKV tensor and apply the sin/cos RoPE permutation to Q and K.

Loading a model is faster and uses less memory because the loader knows which tensors are required for operations and schedules their materialization lazily.

The loader scans the checkpoint once to discover pattern matches and collect tensors. It stores them as Future objects and submits them to a thread pool for asynchronous loading without blocking the GIL. A parameter starts loading as soon as a thread becomes available to it.

If your system runs other heavy processes, multiple threads may slow down loading instead of accelerating it. In this case, set the environment variable HF_DEACTIVATE_ASYNC_LOAD=1 to load weights sequentially.

The default is 4 threads for asynchronous parameter loading. This provides the best trade-off across loading scenarios and hardware. The work is mostly I/O bound, but depending on accelerator hardware and the dtype required at loading, it can become CPU/GPU-bound if the dtype differs from the serialized one (this requires an additional copy operation).

When converting a weight, the converter waits for all required tensors to materialize if they haven’t loaded yet. For example, the MergeModulelist operation requires all weights in ModuleList to be loaded before merging.

Concatenating tensors requires a temporary copy, so operations like MergeModulelist and Concatenate need 2x the memory of the underlying tensors during conversion. Once merged, only the resulting tensor stays in memory. The theoretical worst-case memory peak is the model size plus the tensors required for the largest MergeModulelist or Concatenate operation.

This worst case only occurs when all other parameters have loaded before the demanding conversion runs. Two scenarios trigger this.

For example, a MoE model using MergeModulelist for experts on each layer, the theoretical worst-case memory peak is model size plus experts on one layer.

These worst-case scenarios are uncommon. The actual memory peak tends to stay close to the model size.

Dynamic weight loading is not limited to full model checkpoints. The same building blocks let you load any set of weights as long as you can describe how checkpoint keys map to parameters and ensure the target modules exist.

At a high level, the contract looks like this:

These APIs are expose to allow you to handle custom code, custom weight format, but also make sure you benefit from the highest and most efficient weight loading, sharding and good quality of life of transformers API!

---

## Transformers

**URL:** https://huggingface.co/docs/transformers/v5.0.0/en/model_doc/albert

**Contents:**
- Transformers
- ALBERT
- Notes
- Resources
- AlbertConfig
  - class transformers.AlbertConfig
- AlbertTokenizer
- AlbertTokenizerFast
  - class transformers.AlbertTokenizer
- Albert specific outputs

Transformers documentation

and get access to the augmented documentation experience

This model was released on 2019-09-26 and added to Hugging Face Transformers on 2020-11-16.

ALBERT is designed to address memory limitations of scaling and training of BERT. It adds two parameter reduction techniques. The first, factorized embedding parametrization, splits the larger vocabulary embedding matrix into two smaller matrices so you can grow the hidden size without adding a lot more parameters. The second, cross-layer parameter sharing, allows layer to share parameters which keeps the number of learnable parameters lower.

ALBERT was created to address problems like — GPU/TPU memory limitations, longer training times, and unexpected model degradation in BERT. ALBERT uses two parameter-reduction techniques to lower memory consumption and increase the training speed of BERT:

ALBERT uses absolute position embeddings (like BERT) so padding is applied at right. Size of embeddings is 128 While BERT uses 768. ALBERT can processes maximum 512 token at a time.

You can find all the original ALBERT checkpoints under the ALBERT community organization.

Click on the ALBERT models in the right sidebar for more examples of how to apply ALBERT to different language tasks.

The example below demonstrates how to predict the [MASK] token with Pipeline, AutoModel, and from the command line.

The resources provided in the following sections consist of a list of official Hugging Face and community (indicated by 🌎) resources to help you get started with AlBERT. If you’re interested in submitting a resource to be included here, please feel free to open a Pull Request and we’ll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.

AlbertForSequenceClassification is supported by this example script.

Check the Text classification task guide on how to use the model.

AlbertForTokenClassification is supported by this example script.

Token classification chapter of the 🤗 Hugging Face Course.

Check the Token classification task guide on how to use the model.

( vocab_size = 30000 embedding_size = 128 hidden_size = 4096 num_hidden_layers = 12 num_hidden_groups = 1 num_attention_heads = 64 intermediate_size = 16384 inner_group_num = 1 hidden_act = 'gelu_new' hidden_dropout_prob = 0 attention_probs_dropout_prob = 0 max_position_embeddings = 512 type_vocab_size = 2 initializer_range = 0.02 layer_norm_eps = 1e-12 classifier_dropout_prob = 0.1 pad_token_id = 0 bos_token_id = 2 eos_token_id = 3 tie_word_embeddings = True **kwargs )

This is the configuration class to store the configuration of a AlbertModel. It is used to instantiate an ALBERT model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the ALBERT albert/albert-xxlarge-v2 architecture.

Configuration objects inherit from PreTrainedConfig and can be used to control the model outputs. Read the documentation from PreTrainedConfig for more information.

[[autodoc]] AlbertTokenizer - build_inputs_with_special_tokens - get_special_tokens_mask - create_token_type_ids_from_sequences - save_vocabulary

( vocab: str | list[tuple[str, float]] | None = None do_lower_case: bool = True keep_accents: bool = False bos_token: str = '[CLS]' eos_token: str = '[SEP]' unk_token: str = '<unk>' sep_token: str = '[SEP]' pad_token: str = '<pad>' cls_token: str = '[CLS]' mask_token: str = '[MASK]' add_prefix_space: bool = True trim_offsets: bool = True **kwargs )

When building a sequence using special tokens, this is not the token that is used for the beginning of sequence. The token used is the cls_token.

Construct a “fast” ALBERT tokenizer (backed by HuggingFace’s tokenizers library). Based on Unigram. This tokenizer inherits from PreTrainedTokenizerFast which contains most of the main methods. Users should refer to this superclass for more information regarding those methods

( loss: torch.FloatTensor | None = None prediction_logits: torch.FloatTensor | None = None sop_logits: torch.FloatTensor | None = None hidden_states: tuple[torch.FloatTensor] | None = None attentions: tuple[torch.FloatTensor] | None = None )

Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.

Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

Output type of AlbertForPreTraining.

[[autodoc]] AlbertModel - forward

[[autodoc]] AlbertForPreTraining - forward

[[autodoc]] AlbertForMaskedLM - forward

[[autodoc]] AlbertForSequenceClassification - forward

( config: AlbertConfig )

The Albert Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a softmax) e.g. for RocStories/SWAG tasks.

This model inherits from PreTrainedModel. Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch torch.nn.Module subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

( input_ids: torch.LongTensor | None = None attention_mask: torch.FloatTensor | None = None token_type_ids: torch.LongTensor | None = None position_ids: torch.LongTensor | None = None inputs_embeds: torch.FloatTensor | None = None labels: torch.LongTensor | None = None **kwargs: typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs] ) → transformers.models.albert.modeling_albert.AlbertForPreTrainingOutput or tuple(torch.FloatTensor)

Indices can be obtained using AutoTokenizer. See PreTrainedTokenizer.call() and PreTrainedTokenizer.encode() for details.

What are attention masks?

What are token type IDs?

What are position IDs?

transformers.models.albert.modeling_albert.AlbertForPreTrainingOutput or tuple(torch.FloatTensor)

A transformers.models.albert.modeling_albert.AlbertForPreTrainingOutput or a tuple of torch.FloatTensor (if return_dict=False is passed or when config.return_dict=False) comprising various elements depending on the configuration (AlbertConfig) and inputs. loss (*optional*, returned when labels is provided, torch.FloatTensor of shape (1,)) — Total loss as the sum of the masked language modeling loss and the next sequence prediction (classification) loss. prediction_logits (torch.FloatTensor of shape (batch_size, sequence_length, config.vocab_size)) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax). sop_logits (torch.FloatTensor of shape (batch_size, 2)) — Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation before SoftMax). hidden_states (tuple[torch.FloatTensor] | None.hidden_states, returned when output_hidden_states=True is passed or when config.output_hidden_states=True) — Tuple of torch.FloatTensor (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape (batch_size, sequence_length, hidden_size). Hidden-states of the model at the output of each layer plus the optional initial embedding outputs. attentions (tuple[torch.FloatTensor] | None.attentions, returned when output_attentions=True is passed or when config.output_attentions=True) — Tuple of torch.FloatTensor (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length). Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

A transformers.models.albert.modeling_albert.AlbertForPreTrainingOutput or a tuple of torch.FloatTensor (if return_dict=False is passed or when config.return_dict=False) comprising various elements depending on the configuration (AlbertConfig) and inputs.

loss (*optional*, returned when labels is provided, torch.FloatTensor of shape (1,)) — Total loss as the sum of the masked language modeling loss and the next sequence prediction (classification) loss.

prediction_logits (torch.FloatTensor of shape (batch_size, sequence_length, config.vocab_size)) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).

sop_logits (torch.FloatTensor of shape (batch_size, 2)) — Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation before SoftMax).

hidden_states (tuple[torch.FloatTensor] | None.hidden_states, returned when output_hidden_states=True is passed or when config.output_hidden_states=True) — Tuple of torch.FloatTensor (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape (batch_size, sequence_length, hidden_size).

Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.

attentions (tuple[torch.FloatTensor] | None.attentions, returned when output_attentions=True is passed or when config.output_attentions=True) — Tuple of torch.FloatTensor (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length).

Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

The AlbertForMultipleChoice forward method, overrides the __call__ special method.

Although the recipe for forward pass needs to be defined within this function, one should call the Module instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

[[autodoc]] AlbertForTokenClassification - forward

[[autodoc]] AlbertForQuestionAnswering - forward

---

## Transformers

**URL:** https://huggingface.co/docs/transformers/v5.0.0/en/main_classes/deepspeed

**Contents:**
- Transformers
- DeepSpeed
- HfDeepSpeedConfig
  - class transformers.integrations.HfDeepSpeedConfig

Transformers documentation

and get access to the augmented documentation experience

DeepSpeed, powered by Zero Redundancy Optimizer (ZeRO), is an optimization library for training and fitting very large models onto a GPU. It is available in several ZeRO stages, where each stage progressively saves more GPU memory by partitioning the optimizer state, gradients, parameters, and enabling offloading to a CPU or NVMe. DeepSpeed is integrated with the Trainer class and most of the setup is automatically taken care of for you.

However, if you want to use DeepSpeed without the Trainer, Transformers provides a HfDeepSpeedConfig class.

Learn more about using DeepSpeed with Trainer in the DeepSpeed guide.

( config_file_or_dict )

This object contains a DeepSpeed configuration dictionary and can be quickly queried for things like zero stage.

A weakref of this object is stored in the module’s globals to be able to access the config from areas where things like the Trainer object is not available (e.g. from_pretrained and _get_resized_embeddings). Therefore it’s important that this object remains alive while the program is still running.

Trainer uses the HfTrainerDeepSpeedConfig subclass instead. That subclass has logic to sync the configuration with values of TrainingArguments by replacing special placeholder values: "auto". Without this special logic the DeepSpeed configuration is not modified in any way.

---

## Transformers

**URL:** https://huggingface.co/docs/transformers/v5.0.0/en/main_classes/model

**Contents:**
- Transformers
- Models
- PreTrainedModel
  - class transformers.PreTrainedModel
    - push_to_hub
    - add_model_tags
    - can_generate
    - dequantize
    - disable_input_require_grads
    - enable_input_require_grads

Transformers documentation

and get access to the augmented documentation experience

The base class PreTrainedModel implements the common methods for loading/saving a model either from a local file or directory, or from a pretrained model configuration provided by the library (downloaded from HuggingFace’s Hub).

PreTrainedModel also implements a few methods which are common among all the models to:

The other methods that are common to each model are defined in ModuleUtilsMixin and GenerationMixin.

( config: PreTrainedConfig *inputs **kwargs )

Base class for all models.

PreTrainedModel takes care of storing the configuration of the models and handles methods for loading, downloading and saving models as well as a few methods common to all models to:

Class attributes (overridden by derived classes):

( repo_id: str commit_message: str | None = None commit_description: str | None = None private: bool | None = None token: bool | str | None = None revision: str | None = None create_pr: bool = False max_shard_size: int | str | None = '50GB' tags: list[str] | None = None )

Upload the model file to the 🤗 Model Hub.

( tags: list[str] | str )

Add custom tags into the model that gets pushed to the Hugging Face Hub. Will not overwrite existing tags in the model.

Whether this model can generate sequences with .generate().

Whether this model can generate sequences with .generate().

Returns whether this model can generate sequences with .generate() from the GenerationMixin.

Under the hood, on classes where this function returns True, some generation-specific changes are triggered: for instance, the model instance will have a populated generation_config attribute.

Potentially dequantize the model in case it has been quantized by a quantization method that support dequantization.

Removes the _require_grads_hook.

Enables the gradients for the input embeddings. This is useful for fine-tuning adapter weights while keeping the model weights fixed.

( pretrained_model_name_or_path: str | os.PathLike | None *model_args config: transformers.configuration_utils.PreTrainedConfig | str | os.PathLike | None = None cache_dir: str | os.PathLike | None = None ignore_mismatched_sizes: bool = False force_download: bool = False local_files_only: bool = False token: str | bool | None = None revision: str = 'main' use_safetensors: bool | None = None weights_only: bool = True **kwargs )

Configuration for the model to use instead of an automatically loaded configuration. Configuration can be automatically loaded when:

This option can be used if you want to create a model from a pretrained configuration but load your own weights. In this case though, you should check if using save_pretrained() and from_pretrained() is not a simpler option.

To test a pull request you made on the Hub, you can pass revision="refs/pr/<pr_number>".

Accept HF kernel references in the form:

Examples that match: “org/model” “org/model@main” “org/model:custom_kernel” “org/model@v1.2.3:custom_kernel”

By default, if available, grouped_mm will be used for torch>=2.9.0. The default is otherwise the sequential "eager" implementation.

Parameters for big model inference

torch.float16 or torch.bfloat16 or torch.float: load in a specified dtype, ignoring the model’s config.dtype if one exists. If not specified

"auto" - A dtype or torch_dtype entry in the config.json file of the model will be attempted to be used. If this entry isn’t found then next check the dtype of the first weight in the checkpoint that’s of a floating point type and use that as dtype. This will load the model using the dtype it was saved in at the end of the training. It can’t be used as an indicator of how the model was trained. Since it could be trained in one of half precision dtypes, but saved in fp32.

A string that is a valid torch.dtype. E.g. “float32” loads the model in torch.float32, “float16” loads in torch.float16 etc.

For some models the dtype they were trained in is unknown - you may try to check the model’s paper or reach out to the authors and ask them to add this information to the model’s card and to insert the dtype or torch_dtype entry in config.json on the hub.

To have Accelerate compute the most optimized device_map automatically, set device_map="auto". For more information about each option see designing a device map.

Instantiate a pretrained pytorch model from a pre-trained model configuration.

The model is set in evaluation mode by default using model.eval() (Dropout modules are deactivated). To train the model, you should first set it back in training mode with model.train().

The warning Weights from XXX not initialized from pretrained model means that the weights of XXX do not come pretrained with the rest of the model. It is up to you to train those weights with a downstream fine-tuning task.

The warning Weights from XXX not used in YYY means that the layer XXX is not used by YYY, therefore those weights are discarded.

Activate the special “offline-mode” to use this method in a firewalled environment.

( compile_config: transformers.generation.configuration_utils.CompileConfig | None )

Return a torch.compile‘d version of self.__call__. This is useful to dynamically choose between non-compiled/compiled forward during inference, especially to switch between prefill (where we don’t want to use compiled version to avoid recomputing the graph with new shapes) and iterative decoding (where we want the speed-ups of compiled version with static shapes).

Best-effort lookup of the decoder module.

Order of attempts (covers ~85 % of current usages):

( modality: str | None = None )

Best-effort lookup of the encoder module. If provided with modality argument, it looks for a modality-specific encoder in multimodal models (e.g. “image_encoder”) By default the function returns model’s text encoder if any, and otherwise returns self.

Possible modality values are “image”, “video” and “audio”.

( all_submodels: bool = False )

Return the expanded tied weight keys (in case they contain modules or regex patterns) for only the current model, or recursively for all submodels if all_submodels=True (i.e. it will re-check the config values for all submodels).

For almost all models, we only require to tie the embeddings, so the model has an internal property _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}. In this case, the mapping is already “expanded”, i.e. it already contains full parameters, and this function will simply return a copy of the property. For more complex patterns, e.g. for DFineForObjectDetection, we have the following attribute

returning the following:

( return_buffers = True )

Get the memory footprint of a model. This will return the memory footprint of the current model in bytes. Useful to benchmark the memory footprint of the current model and design some tests. Solution inspired from the PyTorch discussions: https://discuss.pytorch.org/t/gpu-memory-that-model-uses/56822/2

Return the parameter or buffer given by target if it exists, otherwise throw an error. This combines get_parameter() and get_buffer() in a single handy function. If the target is an _extra_state attribute, it will return the extra state provided by the module. Note that it only work if target is a leaf of the model.

Deactivates gradient checkpointing for the current model.

( gradient_checkpointing_kwargs = None )

Activates gradient checkpointing for the current model.

We pass the __call__ method of the modules instead of forward because __call__ attaches all the hooks of the module. https://discuss.pytorch.org/t/any-different-between-model-input-and-model-forward-input/3690/2

Maybe initializes weights. If using a custom PreTrainedModel, you need to implement any initialization logic in _init_weights.

This is equivalent to calling self.apply(self._initialize_weights), but correctly handles composite models. This function dynamically dispatches the correct init_weights function to the modules as we advance in the module graph along the recursion. It can handle an arbitrary number of sub-models. Without it, every composite model would have to recurse a second time on all sub-models explicitly in the outer-most _init_weights, which is extremely error prone and inefficient.

Adds the _is_hf_initialized flag on parameters that will be tied, in order to avoid initializing them later as they will be tied (overwritten) anyway. This is very important as most embeddings are tied, and they are huge params (vocabularies are often 256k), so running inits on them is very costly.

( recurse: bool = True remove_duplicate: bool = True )

Similar to named_buffers, but only yield non-persistent ones. It is handy as it’s not perfectly straightforward to know if they are persistent or not

A method executed at the end of each Transformer model initialization, to execute code that needs the model’s modules properly initialized (such as weight initialization).

( auto_class = 'AutoModel' )

Register this class with a given auto class. This should only be used for custom models as the ones in the library are already mapped with an auto class.

( new_num_tokens: int | None = None pad_to_multiple_of: int | None = None mean_resizing: bool = True ) → torch.nn.Embedding

This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta), or on TPUs which benefit from having sequence lengths be a multiple of 128. For more details about this, or help on choosing the correct value for resizing, refer to this guide: https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#requirements-tc

Setting mean_resizing to True is useful when increasing the size of the embeddings of causal language models, where the generated tokens’ probabilities won’t be affected by the added embeddings because initializing the new embeddings with the old embeddings’ mean will reduce the kl-divergence between the next token probability before and after adding the new embeddings. Refer to this article for more information: https://nlp.stanford.edu/~johnhew/vocab-expansion.html

Pointer to the input tokens Embeddings Module of the model.

Pointer to the input tokens Embeddings Module of the model.

Resizes input token embeddings matrix of the model if new_num_tokens != config.vocab_size.

Takes care of tying weights embeddings afterwards if the model class has a tie_weights() method.

( save_directory: str | os.PathLike is_main_process: bool = True state_dict: dict | None = None push_to_hub: bool = False max_shard_size: int | str = '50GB' variant: str | None = None token: str | bool | None = None save_peft_format: bool = True save_original_format: bool = True **kwargs )

If a single weight of the model is bigger than max_shard_size, it will be in its own checkpoint shard which will be bigger than max_shard_size.

Save a model and its configuration file to a directory, so that it can be re-loaded using the from_pretrained() class method.

( attn_implementation: str | dict )

Set the requested attn_implementation for this model.

Symmetric setter. Mirrors the lookup logic used in get_decoder.

( encoder modality: str | None = None )

Symmetric setter. Mirrors the lookup logic used in get_encoder.

( experts_implementation: str | dict )

Set the requested experts_implementation for this model.

( use_kernels kernel_config: transformers.utils.kernel_config.KernelConfig | None = None )

Set whether or not to use the kernels library to kernelize some layers of the model.

( missing_keys: set[str] | None = None recompute_mapping: bool = True )

Tie the model weights. If recompute_mapping=False (default when called internally), it will rely on the model.all_tied_weights_keys attribute, containing the {target: source} mapping for the tied params. If recompute_mapping=True, it will re-check all internal submodels and their config to determine the params that need to be tied. This is the default when model.tie_weights() is called on its own, outside of __init__, and from_pretrained, in case the config values were changed somewhere.

Note that during from_pretrained, tying is symmetric: if the mapping says “tie target -> source” but source is missing in the checkpoint while target exists, we swap source and target so we can still tie everything to the parameter that actually exists.

( input_ids attention_mask )

Shows a one-time warning if the input_ids appear to contain padding and no attention mask was given.

Custom models should also include a _supports_assign_param_buffer, which determines if superfast init can apply on the particular model. Signs that your model needs this are if test_save_and_load_from_pretrained fails. If so, set this to False.

A few utilities for torch.nn.Modules, to be used as a mixin.

( attention_mask: Tensor input_shape: tuple dtype: torch.dtype | None = None )

Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

( encoder_attention_mask: Tensor ) → torch.Tensor

The inverted attention mask.

The inverted attention mask.

Invert an attention mask (e.g., switches 0. and 1.).

( only_trainable: bool = False exclude_embeddings: bool = False ) → int

The number of parameters.

The number of parameters.

Get number of (optionally, trainable or non-embeddings) parameters in the module.

A Mixin containing the functionality to push a model or tokenizer to the hub.

( repo_id: str commit_message: str | None = None commit_description: str | None = None private: bool | None = None token: bool | str | None = None revision: str | None = None create_pr: bool = False max_shard_size: int | str | None = '50GB' tags: list[str] | None = None )

Upload the {object_files} to the 🤗 Model Hub.

---

## Transformers

**URL:** https://huggingface.co/docs/transformers/v5.0.0/en/main_classes/text_generation

**Contents:**
- Transformers
- Generation
- GenerationConfig
  - class transformers.GenerationConfig
    - from_pretrained
    - from_model_config
    - save_pretrained
    - update
    - validate
    - get_generation_mode

Transformers documentation

and get access to the augmented documentation experience

Each framework has a generate method for text generation implemented in their respective GenerationMixin class:

You can parameterize the generate method with a GenerationConfig class instance. Please refer to this class for the complete list of generation parameters, which control the behavior of the generation method.

To learn how to inspect a model’s generation configuration, what are the defaults, how to change the parameters ad hoc, and how to create and save a customized generation configuration, refer to the text generation strategies guide. The guide also explains how to use related features, like token streaming.

Parameters that control the length of the output

Parameters that control the generation strategy used

Parameters that control the cache

If none is specified, we will use the default cache for the model (which is often DynamicCache). See our cache documentation for further information.

Parameters for manipulation of the model output logits

Parameters that define the output variables of generate

Special tokens that can be used at generation time

Generation parameters exclusive to encoder-decoder models

Generation parameters exclusive to assistant generation

Parameters related to performances and compilation

Class that holds a configuration for a generation task. A generate call supports the following generation methods for text-decoder, text-to-text, speech-to-text, and vision-to-text models:

To learn more about decoding strategies refer to the text generation strategies guide.

A large number of these flags control the logits or the stopping criteria of the generation. Make sure you check the generate-related classes for a full description of the possible manipulations, as well as examples of their usage.

Note: the configuration field that are still None will be overriden by GenerationConfig._get_default_generation_params() during the generation loop. If you want to use different values for these fields, make sure to explicitly set them in the generation config.

( pretrained_model_name: str | os.PathLike config_file_name: str | os.PathLike | None = None cache_dir: str | os.PathLike | None = None force_download: bool = False local_files_only: bool = False token: str | bool | None = None revision: str = 'main' **kwargs ) → GenerationConfig

To test a pull request you made on the Hub, you can pass revision="refs/pr/<pr_number>".

If True, then this functions returns a Tuple(config, unused_kwargs) where unused_kwargs is a dictionary consisting of the key/value pairs whose keys are not configuration attributes: i.e., the part of kwargs which has not been used to update config and is otherwise ignored.

The configuration object instantiated from this pretrained model.

The configuration object instantiated from this pretrained model.

Instantiate a GenerationConfig from a generation configuration file.

( model_config: typing.Union[ForwardRef('PreTrainedConfig'), dict] ) → GenerationConfig

The configuration object instantiated from those parameters.

The configuration object instantiated from those parameters.

Instantiates a GenerationConfig from a PreTrainedConfig. This function is useful to convert legacy PreTrainedConfig objects, which may contain generation parameters, into a stand-alone GenerationConfig.

( save_directory: str | os.PathLike config_file_name: str | os.PathLike | None = None push_to_hub: bool = False **kwargs )

Save a generation configuration object to the directory save_directory, so that it can be re-loaded using the from_pretrained() class method.

( defaults_only = False allow_custom_entries = False **kwargs ) → dict[str, Any]

Dictionary containing all the key-value pairs that were not used to update the instance.

Dictionary containing all the key-value pairs that were not used to update the instance.

Updates attributes of this class instance with attributes from kwargs if they match existing attributes, returning all the unused kwargs.

Validates the values of the attributes of the GenerationConfig instance. Raises exceptions in the presence of parameterization that can be detected as incorrect from the configuration instance alone.

Note that some parameters not validated here are best validated at generate runtime, as they may depend on other inputs and/or the model, such as parameters related to the generation length.

( assistant_model: typing.Optional[ForwardRef('PreTrainedModel')] = None ) → GenerationMode

The generation mode triggered by the instance.

The generation mode triggered by the instance.

Returns the generation mode triggered by the GenerationConfig instance.

A class containing all functions for auto-regressive text generation, to be used as a mixin in model classes. Inheriting from this class causes the model to have special generation-related behavior, such as loading a GenerationConfig at initialization time or ensuring generate-related tests are run in transformers CI.

A model class should inherit from GenerationMixin to enable calling methods like generate, or when it has defined a custom generate method that relies on GenerationMixin, directly or indirectly, which approximately shares the same interface to public methods like generate. Three examples:

The class exposes generate(), which can be used for:

To learn more about decoding strategies refer to the text generation strategies guide.

( inputs: torch.Tensor | None = None generation_config: transformers.generation.configuration_utils.GenerationConfig | None = None logits_processor: transformers.generation.logits_process.LogitsProcessorList | None = None stopping_criteria: transformers.generation.stopping_criteria.StoppingCriteriaList | None = None prefix_allowed_tokens_fn: collections.abc.Callable[[int, torch.Tensor], list[int]] | None = None synced_gpus: bool | None = None assistant_model: typing.Optional[ForwardRef('PreTrainedModel')] = None streamer: typing.Optional[ForwardRef('BaseStreamer')] = None negative_prompt_ids: torch.Tensor | None = None negative_prompt_attention_mask: torch.Tensor | None = None custom_generate: str | collections.abc.Callable | None = None **kwargs ) → ModelOutput or torch.LongTensor

ModelOutput or torch.LongTensor

A ModelOutput (if return_dict_in_generate=True or when config.return_dict_in_generate=True) or a torch.LongTensor. If the model is not an encoder-decoder model (model.config.is_encoder_decoder=False), the possible ModelOutput types are: GenerateDecoderOnlyOutput, GenerateBeamDecoderOnlyOutput If the model is an encoder-decoder model (model.config.is_encoder_decoder=True), the possible ModelOutput types are: GenerateEncoderDecoderOutput, GenerateBeamEncoderDecoderOutput

A ModelOutput (if return_dict_in_generate=True or when config.return_dict_in_generate=True) or a torch.LongTensor.

If the model is not an encoder-decoder model (model.config.is_encoder_decoder=False), the possible ModelOutput types are:

If the model is an encoder-decoder model (model.config.is_encoder_decoder=True), the possible ModelOutput types are:

Generates sequences of token ids for models with a language modeling head.

Most generation-controlling parameters are set in generation_config which, if not passed, will be set to the model’s default generation configuration. You can override any generation_config by passing the corresponding parameters to generate(), e.g. .generate(inputs, num_beams=4, do_sample=True).

For an overview of generation strategies and code examples, check out the following guide.

( sequences: Tensor scores: tuple beam_indices: torch.Tensor | None = None normalize_logits: bool = False ) → torch.Tensor

A torch.Tensor of shape (batch_size*num_return_sequences, sequence_length) containing the transition scores (logits)

A torch.Tensor of shape (batch_size*num_return_sequences, sequence_length) containing the transition scores (logits)

Computes the transition scores of sequences given the generation scores (and beam indices, if beam search was used). This is a convenient method to quickly obtain the scores of the selected tokens at generation time.

Mixin class for models to add continuous batching capabilities.

( inputs: list generation_config: transformers.generation.configuration_utils.GenerationConfig | None = None num_q_padding_intervals: int = 0 num_kv_padding_intervals: int = 0 allow_block_sharing: bool = True record_timestamps: bool = False progress_bar: bool = True **kwargs ) → dict[str, GenerationOutput]

dict[str, GenerationOutput]

a dictionary of request ids to GenerationOutput objects

a dictionary of request ids to GenerationOutput objects

Generate sequences for a batch of prompts using continuous batching.

( generation_config: transformers.generation.configuration_utils.GenerationConfig | None = None manual_eviction: bool = False max_queue_size: int = 0 num_q_padding_intervals: int = 0 num_kv_padding_intervals: int = 0 allow_block_sharing: bool = True ) → ContinuousBatchingManager

ContinuousBatchingManager

The manager instance to add requests and retrieve results.

The manager instance to add requests and retrieve results.

Initialize a manager for continuous batching inference.

( model: Module generation_config: GenerationConfig manual_eviction: bool = False max_queue_size: int = 0 num_q_padding_intervals: int = 0 num_kv_padding_intervals: int = 0 allow_block_sharing: bool = True )

Manager for handling continuous batching of generation requests.

This class provides the user interface for submitting generation requests, retrieving results, and managing the background generation thread.

( input_ids: list request_id: str | None = None max_new_tokens: int | None = None streaming: bool = False record_timestamps: bool = False ) → str

Add a new generation request to the queue.

Cancel a request by its ID.

Evict a request from the cache. It is assumed that the request is already finished.

( request_id: str | None = None timeout: float | None = None ) → Optional[GenerationOutput]

Optional[GenerationOutput]

The result data or None if timeout

The result data or None if timeout

Retrieve one result from the output queue.

Check if the background generation thread is running.

( stop_trigger_time: float timeout: float | None = None )

Wait for the background thread to finish.

Iterate over results matching a specific request id as they become available.

Start the background generation thread.

( block: bool = True timeout: float | None = None )

Signal the background thread to stop.

( cache: PagedAttentionCache retain_cache_on_finish: bool = False )

Abstract base class for scheduling requests in the continuous batch processor. Schedulers manage the lifecycle of requests from when they are added to the waiting queue to when they are scheduled for processing. Different schedulers implement different strategies for prioritizing and batching requests.

( state: RequestState )

Adds a request to the waiting list.

Remove all cancelled requests from active and waiting queues.

( request_id: str evict_from_cache: bool = True )

Completes processing of a request and optionally frees its allocated cache blocks. This method is called when a request has finished generation or encountered an error.

Gets generated tokens for an active request.

Checks if there are requests ready to be processed.

Checks if a request has been cancelled or removed.

( token_budget: int cache_budget: int )

Schedules requests for the next batch based on available token and cache budgets. This method selects which requests should be processed in the current batch, considering the budgets and the scheduler’s prioritization rules. The token_budget is the maximum number of tokens that can be processed in a batch, and the cache_budget is the maximum number of KV cache entries that can be read in a batch.

Marks a request for cancellation.

( cache: PagedAttentionCache retain_cache_on_finish: bool = False safety_margin: float = 0.2 )

This scheduler processes requests in the order they arrive, meaning decoding requests has priority over prefilling requests. Additionally, it includes a safety margin mechanism to prevent cache exhaustion. By default, when 80% of the cache is full, new requests will not be scheduled to prioritize decoding active requests.

( cache: PagedAttentionCache retain_cache_on_finish: bool = False )

Scheduler that prioritizes split prefill requests over decoding requests. This scheduler ensures that split prefill requests (which are continuations of partially processed prompts) are completed before processing new decoding requests.

---

## Transformers

**URL:** https://huggingface.co/docs/transformers/v5.0.0/en/internal/file_utils

**Contents:**
- Transformers
- General Utilities
- Enums and namedtuples
  - class transformers.utils.ExplicitEnum
  - class transformers.utils.PaddingStrategy
  - class transformers.TensorType
- Special Decorators
    - transformers.add_start_docstrings
    - transformers.utils.add_start_docstrings_to_model_forward
    - transformers.add_end_docstrings

Transformers documentation

and get access to the augmented documentation experience

This page lists all of Transformers general utility functions that are found in the file utils.py.

Most of those are only useful if you are studying the general code in the library.

( value names = None module = None qualname = None type = None start = 1 )

Enum with more explicit error message for missing values.

( value names = None module = None qualname = None type = None start = 1 )

Possible values for the padding argument in PreTrainedTokenizerBase.call(). Useful for tab-completion in an IDE.

( value names = None module = None qualname = None type = None start = 1 )

Possible values for the return_tensors argument in PreTrainedTokenizerBase.call(). Useful for tab-completion in an IDE.

( *docstr processor_class = None checkpoint = None output_type = None config_class = None mask = '[MASK]' qa_target_start_index = 14 qa_target_end_index = 15 model_cls = None modality = None expected_output = None expected_loss = None real_checkpoint = None revision = None )

( output_type = None config_class = None )

( name: str module_file: str import_structure: dict module_spec: _frozen_importlib.ModuleSpec | None = None extra_objects: dict[str, object] | None = None explicit_import_shortcut: dict[str, list[str]] | None = None )

Module class that surfaces all objects but only performs associated imports when the objects are requested.

---

## Transformers

**URL:** https://huggingface.co/docs/transformers/v5.0.0/en/generation_strategies

**Contents:**
- Transformers
- Generation strategies
- Basic decoding methods
  - Greedy search
  - Sampling
  - Beam search
- Custom generation methods
  - Creating a custom generation method
    - Adding the base model
    - generate.py

Transformers documentation

Generation strategies

and get access to the augmented documentation experience

A decoding strategy informs how a model should select the next generated token. There are many types of decoding strategies, and choosing the appropriate one has a significant impact on the quality of the generated text.

This guide will help you understand the different decoding strategies available in Transformers and how and when to use them.

These are well established decoding methods, and should be your starting point for text generation tasks.

Greedy search is the default decoding strategy. It selects the next most likely token at each step. Unless specified in GenerationConfig, this strategy generates a maximum of 20 new tokens.

Greedy search works well for tasks with relatively short outputs where creativity is not a priority. However, it breaks down when generating longer sequences because it begins to repeat itself.

Sampling, or multinomial sampling, randomly selects a token based on the probability distribution over the entire model’s vocabulary (as opposed to the most likely token, as in greedy search). This means every token with a non-zero probability has a chance to be selected. Sampling strategies reduce repetition and can generate more creative and diverse outputs.

Enable multinomial sampling with do_sample=True and num_beams=1.

Beam search keeps track of several generated sequences (beams) at each time step. After a certain number of steps, it selects the sequence with the highest overall probability. Unlike greedy search, this strategy can “look ahead” and pick a sequence with a higher probability overall even if the initial tokens have a lower probability. It is best suited for input-grounded tasks, like describing an image or speech recognition. You can also use do_sample=True with beam search to sample at each step, but beam search will still greedily prune out low probability sequences between steps.

Check out the beam search visualizer to see how beam search works.

Enable beam search with the num_beams parameter (should be greater than 1 otherwise it’s equivalent to greedy search).

Custom generation methods enable specialized behavior such as:

We enable custom generation methods through model repositories, assuming a specific model tag and file structure (see subsection below). This feature is an extension of custom modeling code and, like such, requires setting trust_remote_code=True.

If a model repository holds a custom generation method, the easiest way to try it out is to load the model and generate with it:

Model repositories with custom generation methods have a special property: their generation method can be loaded from any model through generate()’s custom_generate argument. This means anyone can create and share their custom generation method to potentially work with any Transformers model, without requiring users to install additional Python packages.

You should read the README.md file of the repository containing the custom generation strategy to see what the new arguments and output type differences are, if they exist. Otherwise, you can assume it works like the base generate() method.

You can find all custom generation methods by searching for their custom tag., custom_generate.

Consider the Hub repository transformers-community/custom_generate_example as an example. The README.md states that it has an additional input argument, left_padding, which adds a number of padding tokens before the prompt.

If the custom method has pinned Python requirements that your environment doesn’t meet, you’ll get an exception about missing requirements. For instance, transformers-community/custom_generate_bad_requirements has an impossible set of requirements defined in its custom_generate/requirements.txt file, and you’ll see the error message below if you try to run it.

Updating your Python requirements accordingly will remove this error message.

To create a new generation method, you need to create a new Model repository and push a few files into it.

After you’ve added all required files, your repository should look like this

The starting point for your custom generation method is a model repository just like any other. The model to add to this repository should be the model you’ve designed your method with, and it is meant to be part of a working self-contained model-generate pair. When the model in this repository is loaded, your custom generation method will override generate. Don’t worry — your generation method can still be loaded with any other Transformers model, as explained in the section above.

If you simply want to copy an existing model, you can do

This is the core of your generation method. It must contain a method named generate, and this method must contain a model argument as its first argument. model is the model instance, which means you have access to all attributes and methods in the model, including the ones defined in GenerationMixin (like the base generate method).

generate.py must be placed in a folder named custom_generate, and not at the root level of the repository. The file paths for this feature are hardcoded.

Under the hood, when the base generate() method is called with a custom_generate argument, it first checks its Python requirements (if any), then locates the custom generate method in generate.py, and finally calls the custom generate. All received arguments and model are forwarded to your custom generate method, with the exception of the arguments used to trigger the custom generation (trust_remote_code and custom_generate).

This means your generate can have a mix of original and custom arguments (as well as a different output type) as shown below.

Follow the recommended practices below to ensure your custom generation method works as expected.

Your custom generate method can relative import code from the custom_generate folder. For example, if you have a utils.py file, you can import it like this:

Only relative imports from the same-level custom_generate folder are supported. Parent/sibling folder imports are not valid. The custom_generate argument also works locally with any directory that contains a custom_generate structure. This is the recommended workflow for developing your custom generation method.

You can optionally specify additional Python requirements in a requirements.txt file inside the custom_generate folder. These are checked at runtime and an exception will be thrown if they’re missing, nudging users to update their environment accordingly.

The root level README.md in the model repository usually describes the model therein. However, since the focus of the repository is the custom generation method, we highly recommend to shift its focus towards describing the custom generation method. In addition to a description of the method, we recommend documenting any input and/or output differences to the original generate(). This way, users can focus on what’s new, and rely on Transformers docs for generic implementation details.

For discoverability, we highly recommend you to add the custom_generate tag to your repository. To do so, the top of your README.md file should look like the example below. After you push the file, you should see the tag in your repository!

Recommended practices:

If you’re adding a new decoding loop, you might want to preserve the input preparation present in generate (batch expansion, attention masks, logits processors, stopping criteria, etc.). You can also pass a callable to custom_generate to reuse generate()’s full preparation pipeline while overriding only the decoding loop.

If you publish a custom_generate repository, your generate implementation can itself define a callable and pass it to model.generate(). This lets you customize the decoding loop while still benefiting from Transformers’ built-in input preparation logic.

You can find all custom generation methods by searching for their custom tag., custom_generate. In addition to the tag, we curate two collections of custom_generate methods:

Read the How to generate text: using different decoding methods for language generation with Transformers blog post for an explanation of how common decoding strategies work.

---

## Transformers

**URL:** https://huggingface.co/docs/transformers

**Contents:**
- Transformers
- Transformers
- Features
- Design
- Learn

Transformers documentation

and get access to the augmented documentation experience

Transformers acts as the model-definition framework for state-of-the-art machine learning models in text, computer vision, audio, video, and multimodal model, for both inference and training.

It centralizes the model definition so that this definition is agreed upon across the ecosystem. transformers is the pivot across frameworks: if a model definition is supported, it will be compatible with the majority of training frameworks (Axolotl, Unsloth, DeepSpeed, FSDP, PyTorch-Lightning, …), inference engines (vLLM, SGLang, TGI, …), and adjacent modeling libraries (llama.cpp, mlx, …) which leverage the model definition from transformers.

We pledge to help support new state-of-the-art models and democratize their usage by having their model definition be simple, customizable, and efficient.

There are over 1M+ Transformers model checkpoints on the Hugging Face Hub you can use.

Explore the Hub today to find a model and use Transformers to help you get started right away.

Explore the Models Timeline to discover the latest text, vision, audio and multimodal model architectures in Transformers.

Transformers provides everything you need for inference or training with state-of-the-art pretrained models. Some of the main features include:

Read our Philosophy to learn more about Transformers’ design principles.

Transformers is designed for developers and machine learning engineers and researchers. Its main design principles are:

If you’re new to Transformers or want to learn more about transformer models, we recommend starting with the LLM course. This comprehensive course covers everything from the fundamentals of how transformer models work to practical applications across various tasks. You’ll learn the complete workflow, from curating high-quality datasets to fine-tuning large language models and implementing reasoning capabilities. The course contains both theoretical and hands-on exercises to build a solid foundational knowledge of transformer models as you learn.

---

## Transformers

**URL:** https://huggingface.co/docs/transformers/v5.0.0/en/main_classes/tokenizer

**Contents:**
- Transformers
- Tokenizer
- Multimodal Tokenizer
- PreTrainedTokenizer
  - class transformers.PythonBackend
    - __call__
    - add_tokens
    - add_special_tokens
    - apply_chat_template
    - batch_decode

Transformers documentation

and get access to the augmented documentation experience

A tokenizer is in charge of preparing the inputs for a model. The library contains tokenizers for all the models. Most of the tokenizers are available in two flavors: a full python implementation and a “Fast” implementation based on the Rust library 🤗 Tokenizers. The “Fast” implementations allows:

The base classes PreTrainedTokenizer and PreTrainedTokenizerFast implement the common methods for encoding string inputs in model inputs (see below) and instantiating/saving python and “Fast” tokenizers either from a local file or directory or from a pretrained tokenizer provided by the library (downloaded from HuggingFace’s AWS S3 repository). They both rely on PreTrainedTokenizerBase that contains the common methods.

PreTrainedTokenizer and PreTrainedTokenizerFast thus implement the main methods for using all the tokenizers:

BatchEncoding holds the output of the PreTrainedTokenizerBase’s encoding methods (__call__, encode_plus and batch_encode_plus) and is derived from a Python dictionary. When the tokenizer is a pure python tokenizer, this class behaves just like a standard python dictionary and holds the various model inputs computed by these methods (input_ids, attention_mask…). When the tokenizer is a “Fast” tokenizer (i.e., backed by HuggingFace tokenizers library), this class provides in addition several advanced alignment methods which can be used to map between the original string (character and words) and the token space (e.g., getting the index of the token comprising a given character or the span of characters corresponding to a given token).

Apart from that each tokenizer can be a “multimodal” tokenizer which means that the tokenizer will hold all relevant special tokens as part of tokenizer attributes for easier access. For example, if the tokenizer is loaded from a vision-language model like LLaVA, you will be able to access tokenizer.image_token_id to obtain the special image token used as a placeholder.

To enable extra special tokens for any type of tokenizer, you have to add the following lines and save the tokenizer. Extra special tokens do not have to be modality related and can be anything that the model often needs access to. In the below code, tokenizer at output_dir will have direct access to three more special tokens.

Base class for all slow tokenizers.

Inherits from PreTrainedTokenizerBase.

Handle all the shared methods for tokenization and special tokens as well as methods downloading/caching/loading pretrained tokenizers as well as adding tokens to the vocabulary.

This class also contain the added tokens in a unified way on top of all tokenizers so we don’t have to handle the specific vocabulary augmentation methods of the various underlying dictionary structures (BPE, sentencepiece…).

Class attributes (overridden by derived classes)

( text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None text_pair: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None text_target: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None text_pair_target: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None add_special_tokens: bool = True padding: bool | str | PaddingStrategy = False truncation: bool | str | TruncationStrategy | None = None max_length: int | None = None stride: int = 0 is_split_into_words: bool = False pad_to_multiple_of: int | None = None padding_side: str | None = None return_tensors: str | TensorType | None = None return_token_type_ids: bool | None = None return_attention_mask: bool | None = None return_overflowing_tokens: bool = False return_special_tokens_mask: bool = False return_offsets_mapping: bool = False return_length: bool = False verbose: bool = True tokenizer_kwargs: dict[str, Any] | None = None **kwargs ) → BatchEncoding

If left unset or set to None, this will use the predefined model maximum length if a maximum length is required by one of the truncation/padding parameters. If the model has no specific maximum input length (like XLNet) truncation/padding to a maximum length will be deactivated.

What are token type IDs?

What are attention masks?

This is only available on fast tokenizers inheriting from PreTrainedTokenizerFast, if using Python’s tokenizer, this method will raise NotImplementedError.

A BatchEncoding with the following fields: input_ids — List of token ids to be fed to a model. What are input IDs? token_type_ids — List of token type ids to be fed to a model (when return_token_type_ids=True or if “token_type_ids” is in self.model_input_names). What are token type IDs? attention_mask — List of indices specifying which tokens should be attended to by the model (when return_attention_mask=True or if “attention_mask” is in self.model_input_names). What are attention masks? overflowing_tokens — List of overflowing tokens sequences (when a max_length is specified and return_overflowing_tokens=True). num_truncated_tokens — Number of tokens truncated (when a max_length is specified and return_overflowing_tokens=True). special_tokens_mask — List of 0s and 1s, with 1 specifying added special tokens and 0 specifying regular sequence tokens (when add_special_tokens=True and return_special_tokens_mask=True). length — The length of the inputs (when return_length=True)

A BatchEncoding with the following fields:

input_ids — List of token ids to be fed to a model.

token_type_ids — List of token type ids to be fed to a model (when return_token_type_ids=True or if “token_type_ids” is in self.model_input_names).

What are token type IDs?

attention_mask — List of indices specifying which tokens should be attended to by the model (when return_attention_mask=True or if “attention_mask” is in self.model_input_names).

What are attention masks?

overflowing_tokens — List of overflowing tokens sequences (when a max_length is specified and return_overflowing_tokens=True).

num_truncated_tokens — Number of tokens truncated (when a max_length is specified and return_overflowing_tokens=True).

special_tokens_mask — List of 0s and 1s, with 1 specifying added special tokens and 0 specifying regular sequence tokens (when add_special_tokens=True and return_special_tokens_mask=True).

length — The length of the inputs (when return_length=True)

Main method to tokenize and prepare for the model one or several sequence(s) or one or several pair(s) of sequences.

( new_tokens: str | AddedToken | Sequence[str | AddedToken] special_tokens: bool = False ) → int

Number of tokens added to the vocabulary.

Number of tokens added to the vocabulary.

#TODO remove this from here! PreTrainedTOkeniuzerBase should be agnostic of AddedToken.

Add a list of new tokens. If the new tokens are not in the vocabulary, they are added to the end. Added tokens and tokens from the vocabulary of the tokenization algorithm are therefore not treated in the same way.

( special_tokens_dict: dict[str, str | AddedToken | Sequence[str | AddedToken]] replace_extra_special_tokens = True ) → int

Tokens are only added if they are not already in the vocabulary (tested by checking if the tokenizer assign the index of the unk_token to them).

Number of tokens added to the vocabulary.

Number of tokens added to the vocabulary.

Add a dictionary of special tokens (eos, pad, cls, etc.) to the encoder and link them to class attributes. If special tokens are NOT in the vocabulary, they are added to it (indexed starting from the last index of the current vocabulary).

When adding new tokens to the vocabulary, you should make sure to also resize the token embedding matrix of the model so that its embedding matrix matches the tokenizer.

In order to do that, please use the resize_token_embeddings() method.

Using add_special_tokens will ensure your special tokens can be used in several ways:

When possible, special tokens are already registered for provided pretrained models (for instance BertTokenizer cls_token is already registered to be '[CLS]' and XLM’s one is also registered to be '</s>').

( conversation: list[dict[str, str]] | list[list[dict[str, str]]] tools: list[dict | Callable] | None = None documents: list[dict[str, str]] | None = None chat_template: str | None = None add_generation_prompt: bool = False continue_final_message: bool = False tokenize: bool = True padding: bool | str | PaddingStrategy = False truncation: bool = False max_length: int | None = None return_tensors: str | TensorType | None = None return_dict: bool = True return_assistant_tokens_mask: bool = False tokenizer_kwargs: dict[str, Any] | None = None **kwargs ) → Union[list[int], Dict]

Union[list[int], Dict]

A list of token ids representing the tokenized chat so far, including control tokens. This output is ready to pass to the model, either directly or via methods like generate(). If return_dict is set, will return a dict of tokenizer outputs instead.

A list of token ids representing the tokenized chat so far, including control tokens. This output is ready to pass to the model, either directly or via methods like generate(). If return_dict is set, will return a dict of tokenizer outputs instead.

Converts a list of dictionaries with "role" and "content" keys to a list of token ids. This method is intended for use with chat models, and will read the tokenizer’s chat_template attribute to determine the format and control tokens to use when converting.

( sequences: list[int] | list[list[int]] | np.ndarray | torch.Tensor skip_special_tokens: bool = False clean_up_tokenization_spaces: bool | None = None **kwargs ) → list[str]

The list of decoded sentences.

The list of decoded sentences.

Convert a list of lists of token ids into a list of strings by calling decode.

This method is provided for backwards compatibility. The decode method now handles batched input natively, so you can use decode directly instead of batch_decode.

( token_ids: int | list[int] | list[list[int]] | np.ndarray | torch.Tensor skip_special_tokens: bool = False **kwargs ) → Union[str, list[str]]

Union[str, list[str]]

The decoded string for a single sequence, or a list of decoded strings for a batch of sequences.

The decoded string for a single sequence, or a list of decoded strings for a batch of sequences.

Converts a sequence of ids into a string, or a list of sequences into a list of strings, using the tokenizer and vocabulary with options to remove special tokens and clean up tokenization spaces.

Similar to doing self.convert_tokens_to_string(self.convert_ids_to_tokens(token_ids)).

( text: TextInput | PreTokenizedInput | EncodedInput text_pair: TextInput | PreTokenizedInput | EncodedInput | None = None add_special_tokens: bool = True padding: bool | str | PaddingStrategy = False truncation: bool | str | TruncationStrategy | None = None max_length: int | None = None stride: int = 0 padding_side: str | None = None return_tensors: str | TensorType | None = None **kwargs ) → list[int], torch.Tensor, or np.ndarray

If left unset or set to None, this will use the predefined model maximum length if a maximum length is required by one of the truncation/padding parameters. If the model has no specific maximum input length (like XLNet) truncation/padding to a maximum length will be deactivated.

list[int], torch.Tensor, or np.ndarray

The tokenized ids of the text.

The tokenized ids of the text.

Converts a string to a sequence of ids (integer), using the tokenizer and vocabulary.

Same as doing self.convert_tokens_to_ids(self.tokenize(text)).

( repo_id: str commit_message: str | None = None commit_description: str | None = None private: bool | None = None token: bool | str | None = None revision: str | None = None create_pr: bool = False max_shard_size: int | str | None = '50GB' tags: list[str] | None = None )

Upload the tokenizer files to the 🤗 Model Hub.

( token_ids_0: list token_ids_1: list[int] | None = None ) → list[int]

List of input IDs with the appropriate special tokens.

List of input IDs with the appropriate special tokens.

Build model inputs from a sequence or a pair of sequences by adding special tokens.

This method dynamically builds inputs based on the tokenizer’s special_tokens_pattern:

( token_ids_0: list token_ids_1: list[int] | None = None ) → list[int]

Token type IDs according to the configured pattern.

Token type IDs according to the configured pattern.

Create a mask from the two sequences passed to be used in a sequence-pair classification task.

This method dynamically builds the token type IDs based on the tokenizer’s configuration attributes:

Returns the added tokens in the vocabulary as a dictionary of token to index. Results might be different from the fast call because for now we always add the tokens even if they are already in the vocabulary. This is something we should change.

( token_ids_0: list token_ids_1: list | None = None already_has_special_tokens: bool = False ) → A list of integers in the range [0, 1]

A list of integers in the range [0, 1]

1 for a special token, 0 for a sequence token.

1 for a special token, 0 for a sequence token.

Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding special tokens using the tokenizer prepare_for_model or encode_plus methods.

This method dynamically builds the special tokens mask based on the tokenizer’s special_tokens_pattern:

( pair: bool = False ) → int

Number of special tokens added to sequences.

Number of special tokens added to sequences.

Returns the number of added tokens when encoding a sequence with special tokens.

This encodes a dummy input and checks the number of added tokens, and is therefore not efficient. Do not put this inside your training loop.

( ids: list pair_ids: list[int] | None = None add_special_tokens: bool = True padding: bool | str | transformers.utils.generic.PaddingStrategy = False truncation: bool | str | transformers.tokenization_utils_base.TruncationStrategy = False max_length: int | None = None stride: int = 0 pad_to_multiple_of: int | None = None padding_side: str | None = None return_tensors: str | transformers.utils.generic.TensorType | None = None return_token_type_ids: bool | None = None return_attention_mask: bool | None = None return_overflowing_tokens: bool = False return_special_tokens_mask: bool = False return_length: bool = False verbose: bool = True prepend_batch_axis: bool = False **kwargs )

Prepares a sequence of input ids so it can be used by the model. Adds special tokens, truncates, and pads.

( text: str is_split_into_words: bool = False **kwargs ) → tuple[str, dict[str, Any]]

tuple[str, dict[str, Any]]

The prepared text and the unused kwargs.

The prepared text and the unused kwargs.

Performs any necessary transformations before tokenization.

This method should pop the arguments from kwargs and return the remaining kwargs as well. We test the kwargs at the end of the encoding process to be sure all the arguments have been used.

( save_directory: str filename_prefix: str | None = None ) → tuple[str, ...]

Paths to the files saved, or empty tuple if no files saved.

Paths to the files saved, or empty tuple if no files saved.

Default implementation for common vocabulary saving patterns. Saves self.encoder/self.vocab as JSON, optionally with self.bpe_ranks as merges. Returns empty tuple if no vocabulary exists.

Override this method if your tokenizer needs custom saving logic (e.g., SentencePiece models, multiple vocabulary files, or special file formats).

( text: str **kwargs )

Converts a string into a sequence of tokens, using the tokenizer.

( ids: list pair_ids: list[int] | None = None num_tokens_to_remove: int = 0 truncation_strategy: str | transformers.tokenization_utils_base.TruncationStrategy = 'longest_first' stride: int = 0 )

Truncates sequences according to the specified strategy.

The PreTrainedTokenizerFast depend on the tokenizers library. The tokenizers obtained from the 🤗 tokenizers library can be loaded very simply into 🤗 transformers. Take a look at the Using tokenizers from 🤗 tokenizers page to understand how this is done.

Base class for all fast tokenizers (wrapping HuggingFace tokenizers library).

Inherits from PreTrainedTokenizerBase.

Handles all the shared methods for tokenization and special tokens, as well as methods for downloading/caching/loading pretrained tokenizers, as well as adding tokens to the vocabulary.

This class also contains the added tokens in a unified way on top of all tokenizers so we don’t have to handle the specific vocabulary augmentation methods of the various underlying dictionary structures (BPE, sentencepiece…).

Class attributes (overridden by derived classes)

( text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None text_pair: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None text_target: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None text_pair_target: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None add_special_tokens: bool = True padding: bool | str | PaddingStrategy = False truncation: bool | str | TruncationStrategy | None = None max_length: int | None = None stride: int = 0 is_split_into_words: bool = False pad_to_multiple_of: int | None = None padding_side: str | None = None return_tensors: str | TensorType | None = None return_token_type_ids: bool | None = None return_attention_mask: bool | None = None return_overflowing_tokens: bool = False return_special_tokens_mask: bool = False return_offsets_mapping: bool = False return_length: bool = False verbose: bool = True tokenizer_kwargs: dict[str, Any] | None = None **kwargs ) → BatchEncoding

If left unset or set to None, this will use the predefined model maximum length if a maximum length is required by one of the truncation/padding parameters. If the model has no specific maximum input length (like XLNet) truncation/padding to a maximum length will be deactivated.

What are token type IDs?

What are attention masks?

This is only available on fast tokenizers inheriting from PreTrainedTokenizerFast, if using Python’s tokenizer, this method will raise NotImplementedError.

A BatchEncoding with the following fields: input_ids — List of token ids to be fed to a model. What are input IDs? token_type_ids — List of token type ids to be fed to a model (when return_token_type_ids=True or if “token_type_ids” is in self.model_input_names). What are token type IDs? attention_mask — List of indices specifying which tokens should be attended to by the model (when return_attention_mask=True or if “attention_mask” is in self.model_input_names). What are attention masks? overflowing_tokens — List of overflowing tokens sequences (when a max_length is specified and return_overflowing_tokens=True). num_truncated_tokens — Number of tokens truncated (when a max_length is specified and return_overflowing_tokens=True). special_tokens_mask — List of 0s and 1s, with 1 specifying added special tokens and 0 specifying regular sequence tokens (when add_special_tokens=True and return_special_tokens_mask=True). length — The length of the inputs (when return_length=True)

A BatchEncoding with the following fields:

input_ids — List of token ids to be fed to a model.

token_type_ids — List of token type ids to be fed to a model (when return_token_type_ids=True or if “token_type_ids” is in self.model_input_names).

What are token type IDs?

attention_mask — List of indices specifying which tokens should be attended to by the model (when return_attention_mask=True or if “attention_mask” is in self.model_input_names).

What are attention masks?

overflowing_tokens — List of overflowing tokens sequences (when a max_length is specified and return_overflowing_tokens=True).

num_truncated_tokens — Number of tokens truncated (when a max_length is specified and return_overflowing_tokens=True).

special_tokens_mask — List of 0s and 1s, with 1 specifying added special tokens and 0 specifying regular sequence tokens (when add_special_tokens=True and return_special_tokens_mask=True).

length — The length of the inputs (when return_length=True)

Main method to tokenize and prepare for the model one or several sequence(s) or one or several pair(s) of sequences.

( new_tokens: str | AddedToken | Sequence[str | AddedToken] special_tokens: bool = False ) → int

Number of tokens added to the vocabulary.

Number of tokens added to the vocabulary.

#TODO remove this from here! PreTrainedTOkeniuzerBase should be agnostic of AddedToken.

Add a list of new tokens. If the new tokens are not in the vocabulary, they are added to the end. Added tokens and tokens from the vocabulary of the tokenization algorithm are therefore not treated in the same way.

( special_tokens_dict: dict[str, str | AddedToken | Sequence[str | AddedToken]] replace_extra_special_tokens = True ) → int

Tokens are only added if they are not already in the vocabulary (tested by checking if the tokenizer assign the index of the unk_token to them).

Number of tokens added to the vocabulary.

Number of tokens added to the vocabulary.

Add a dictionary of special tokens (eos, pad, cls, etc.) to the encoder and link them to class attributes. If special tokens are NOT in the vocabulary, they are added to it (indexed starting from the last index of the current vocabulary).

When adding new tokens to the vocabulary, you should make sure to also resize the token embedding matrix of the model so that its embedding matrix matches the tokenizer.

In order to do that, please use the resize_token_embeddings() method.

Using add_special_tokens will ensure your special tokens can be used in several ways:

When possible, special tokens are already registered for provided pretrained models (for instance BertTokenizer cls_token is already registered to be '[CLS]' and XLM’s one is also registered to be '</s>').

( conversation: list[dict[str, str]] | list[list[dict[str, str]]] tools: list[dict | Callable] | None = None documents: list[dict[str, str]] | None = None chat_template: str | None = None add_generation_prompt: bool = False continue_final_message: bool = False tokenize: bool = True padding: bool | str | PaddingStrategy = False truncation: bool = False max_length: int | None = None return_tensors: str | TensorType | None = None return_dict: bool = True return_assistant_tokens_mask: bool = False tokenizer_kwargs: dict[str, Any] | None = None **kwargs ) → Union[list[int], Dict]

Union[list[int], Dict]

A list of token ids representing the tokenized chat so far, including control tokens. This output is ready to pass to the model, either directly or via methods like generate(). If return_dict is set, will return a dict of tokenizer outputs instead.

A list of token ids representing the tokenized chat so far, including control tokens. This output is ready to pass to the model, either directly or via methods like generate(). If return_dict is set, will return a dict of tokenizer outputs instead.

Converts a list of dictionaries with "role" and "content" keys to a list of token ids. This method is intended for use with chat models, and will read the tokenizer’s chat_template attribute to determine the format and control tokens to use when converting.

( sequences: list[int] | list[list[int]] | np.ndarray | torch.Tensor skip_special_tokens: bool = False clean_up_tokenization_spaces: bool | None = None **kwargs ) → list[str]

The list of decoded sentences.

The list of decoded sentences.

Convert a list of lists of token ids into a list of strings by calling decode.

This method is provided for backwards compatibility. The decode method now handles batched input natively, so you can use decode directly instead of batch_decode.

( token_ids: int | list[int] | list[list[int]] | np.ndarray | torch.Tensor skip_special_tokens: bool = False **kwargs ) → Union[str, list[str]]

Union[str, list[str]]

The decoded string for a single sequence, or a list of decoded strings for a batch of sequences.

The decoded string for a single sequence, or a list of decoded strings for a batch of sequences.

Converts a sequence of ids into a string, or a list of sequences into a list of strings, using the tokenizer and vocabulary with options to remove special tokens and clean up tokenization spaces.

Similar to doing self.convert_tokens_to_string(self.convert_ids_to_tokens(token_ids)).

( text: TextInput | PreTokenizedInput | EncodedInput text_pair: TextInput | PreTokenizedInput | EncodedInput | None = None add_special_tokens: bool = True padding: bool | str | PaddingStrategy = False truncation: bool | str | TruncationStrategy | None = None max_length: int | None = None stride: int = 0 padding_side: str | None = None return_tensors: str | TensorType | None = None **kwargs ) → list[int], torch.Tensor, or np.ndarray

If left unset or set to None, this will use the predefined model maximum length if a maximum length is required by one of the truncation/padding parameters. If the model has no specific maximum input length (like XLNet) truncation/padding to a maximum length will be deactivated.

list[int], torch.Tensor, or np.ndarray

The tokenized ids of the text.

The tokenized ids of the text.

Converts a string to a sequence of ids (integer), using the tokenizer and vocabulary.

Same as doing self.convert_tokens_to_ids(self.tokenize(text)).

( repo_id: str commit_message: str | None = None commit_description: str | None = None private: bool | None = None token: bool | str | None = None revision: str | None = None create_pr: bool = False max_shard_size: int | str | None = '50GB' tags: list[str] | None = None )

Upload the tokenizer files to the 🤗 Model Hub.

( trust_remote_code = False **kwargs )

s Build a tokenizers.Tokenizer backend from the available serialization files (tokenizer.json, sentencepiece models, tekken.json, vocab/merges).

Returns the added tokens in the vocabulary as a dictionary of token to index.

( pair: bool = False ) → int

Number of special tokens added to sequences.

Number of special tokens added to sequences.

Returns the number of added tokens when encoding a sequence with special tokens.

This encodes a dummy input and checks the number of added tokens, and is therefore not efficient. Do not put this inside your training loop.

( padding_strategy: PaddingStrategy truncation_strategy: TruncationStrategy max_length: int stride: int pad_to_multiple_of: int | None padding_side: str | None )

Define the truncation and the padding strategies for fast tokenizers (provided by HuggingFace tokenizers library) and restore the tokenizer settings afterwards.

The provided tokenizer has no padding / truncation strategy before the managed section. If your tokenizer set a padding / truncation strategy before, then it will be reset to no padding / truncation when exiting the managed section.

( text_iterator vocab_size length = None new_special_tokens = None special_tokens_map = None **kwargs ) → PreTrainedTokenizerFast

PreTrainedTokenizerFast

A new tokenizer of the same type as the original one, trained on text_iterator.

A new tokenizer of the same type as the original one, trained on text_iterator.

Trains a tokenizer on a new corpus with the same defaults (in terms of special tokens or tokenization pipeline) as the current one.

Updates the underlying post processor with the current bos_token and eos_token.

Base class for all slow tokenizers.

Inherits from PreTrainedTokenizerBase.

Handle all the shared methods for tokenization and special tokens as well as methods downloading/caching/loading pretrained tokenizers as well as adding tokens to the vocabulary.

This class also contain the added tokens in a unified way on top of all tokenizers so we don’t have to handle the specific vocabulary augmentation methods of the various underlying dictionary structures (BPE, sentencepiece…).

Class attributes (overridden by derived classes)

( token_ids_0: list token_ids_1: list[int] | None = None ) → list[int]

List of input IDs with the appropriate special tokens.

List of input IDs with the appropriate special tokens.

Build model inputs from a sequence or a pair of sequences by adding special tokens.

This method dynamically builds inputs based on the tokenizer’s special_tokens_pattern:

( token_ids_0: list token_ids_1: list[int] | None = None ) → list[int]

Token type IDs according to the configured pattern.

Token type IDs according to the configured pattern.

Create a mask from the two sequences passed to be used in a sequence-pair classification task.

This method dynamically builds the token type IDs based on the tokenizer’s configuration attributes:

Returns the added tokens in the vocabulary as a dictionary of token to index. Results might be different from the fast call because for now we always add the tokens even if they are already in the vocabulary. This is something we should change.

( token_ids_0: list token_ids_1: list | None = None already_has_special_tokens: bool = False ) → A list of integers in the range [0, 1]

A list of integers in the range [0, 1]

1 for a special token, 0 for a sequence token.

1 for a special token, 0 for a sequence token.

Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding special tokens using the tokenizer prepare_for_model or encode_plus methods.

This method dynamically builds the special tokens mask based on the tokenizer’s special_tokens_pattern:

( pair: bool = False ) → int

Number of special tokens added to sequences.

Number of special tokens added to sequences.

Returns the number of added tokens when encoding a sequence with special tokens.

This encodes a dummy input and checks the number of added tokens, and is therefore not efficient. Do not put this inside your training loop.

( ids: list pair_ids: list[int] | None = None add_special_tokens: bool = True padding: bool | str | transformers.utils.generic.PaddingStrategy = False truncation: bool | str | transformers.tokenization_utils_base.TruncationStrategy = False max_length: int | None = None stride: int = 0 pad_to_multiple_of: int | None = None padding_side: str | None = None return_tensors: str | transformers.utils.generic.TensorType | None = None return_token_type_ids: bool | None = None return_attention_mask: bool | None = None return_overflowing_tokens: bool = False return_special_tokens_mask: bool = False return_length: bool = False verbose: bool = True prepend_batch_axis: bool = False **kwargs )

Prepares a sequence of input ids so it can be used by the model. Adds special tokens, truncates, and pads.

( text: str is_split_into_words: bool = False **kwargs ) → tuple[str, dict[str, Any]]

tuple[str, dict[str, Any]]

The prepared text and the unused kwargs.

The prepared text and the unused kwargs.

Performs any necessary transformations before tokenization.

This method should pop the arguments from kwargs and return the remaining kwargs as well. We test the kwargs at the end of the encoding process to be sure all the arguments have been used.

( save_directory: str filename_prefix: str | None = None ) → tuple[str, ...]

Paths to the files saved, or empty tuple if no files saved.

Paths to the files saved, or empty tuple if no files saved.

Default implementation for common vocabulary saving patterns. Saves self.encoder/self.vocab as JSON, optionally with self.bpe_ranks as merges. Returns empty tuple if no vocabulary exists.

Override this method if your tokenizer needs custom saving logic (e.g., SentencePiece models, multiple vocabulary files, or special file formats).

( text: str **kwargs )

Converts a string into a sequence of tokens, using the tokenizer.

( ids: list pair_ids: list[int] | None = None num_tokens_to_remove: int = 0 truncation_strategy: str | transformers.tokenization_utils_base.TruncationStrategy = 'longest_first' stride: int = 0 )

Truncates sequences according to the specified strategy.

Base class for all fast tokenizers (wrapping HuggingFace tokenizers library).

Inherits from PreTrainedTokenizerBase.

Handles all the shared methods for tokenization and special tokens, as well as methods for downloading/caching/loading pretrained tokenizers, as well as adding tokens to the vocabulary.

This class also contains the added tokens in a unified way on top of all tokenizers so we don’t have to handle the specific vocabulary augmentation methods of the various underlying dictionary structures (BPE, sentencepiece…).

Class attributes (overridden by derived classes)

( trust_remote_code = False **kwargs )

s Build a tokenizers.Tokenizer backend from the available serialization files (tokenizer.json, sentencepiece models, tekken.json, vocab/merges).

Returns the added tokens in the vocabulary as a dictionary of token to index.

( pair: bool = False ) → int

Number of special tokens added to sequences.

Number of special tokens added to sequences.

Returns the number of added tokens when encoding a sequence with special tokens.

This encodes a dummy input and checks the number of added tokens, and is therefore not efficient. Do not put this inside your training loop.

( padding_strategy: PaddingStrategy truncation_strategy: TruncationStrategy max_length: int stride: int pad_to_multiple_of: int | None padding_side: str | None )

Define the truncation and the padding strategies for fast tokenizers (provided by HuggingFace tokenizers library) and restore the tokenizer settings afterwards.

The provided tokenizer has no padding / truncation strategy before the managed section. If your tokenizer set a padding / truncation strategy before, then it will be reset to no padding / truncation when exiting the managed section.

( text_iterator vocab_size length = None new_special_tokens = None special_tokens_map = None **kwargs ) → PreTrainedTokenizerFast

PreTrainedTokenizerFast

A new tokenizer of the same type as the original one, trained on text_iterator.

A new tokenizer of the same type as the original one, trained on text_iterator.

Trains a tokenizer on a new corpus with the same defaults (in terms of special tokens or tokenization pipeline) as the current one.

Updates the underlying post processor with the current bos_token and eos_token.

Base class for SentencePiece-based tokenizers that load from sentencepiece.model files.

Inherits from PreTrainedTokenizer.

Handle all the shared methods for tokenization and special tokens as well as methods downloading/caching/loading pretrained tokenizers as well as adding tokens to the vocabulary.

This class also contain the added tokens in a unified way on top of all tokenizers so we don’t have to handle the specific vocabulary augmentation methods of the various underlying dictionary structures (BPE, sentencepiece…).

Class attributes (overridden by derived classes)

Converts a sequence of tokens (string) in a single string.

Returns vocab as a dict

( save_directory: str filename_prefix: str | None = None ) → tuple(str)

Paths to the files saved.

Paths to the files saved.

Save the sentencepiece vocabulary (copy original file) to a directory.

( data: dict[str, Any] | None = None encoding: EncodingFast | Sequence[EncodingFast] | None = None tensor_type: None | str | TensorType = None prepend_batch_axis: bool = False n_sequences: int | None = None )

Holds the output of the call(), ~tokenization_utils_base.PreTrainedTokenizerBase.encode_plus and ~tokenization_utils_base.PreTrainedTokenizerBase.batch_encode_plus methods (tokens, attention_masks, etc).

This class is derived from a python dictionary and can be used as a dictionary. In addition, this class exposes utility methods to map from word/character space to token space.

( batch_or_char_index: int char_index: int | None = None sequence_index: int = 0 ) → int

Index of the token, or None if the char index refers to a whitespace only token and whitespace is trimmed with trim_offsets=True.

Index of the token, or None if the char index refers to a whitespace only token and whitespace is trimmed with trim_offsets=True.

Get the index of the token in the encoded output comprising a character in the original string for a sequence of the batch.

This method is particularly suited when the input sequences are provided as pre-tokenized sequences (i.e. words are defined by the user). In this case it allows to easily associate encoded tokens with provided tokenized words.

( batch_or_char_index: int char_index: int | None = None sequence_index: int = 0 ) → int or list[int]

Index or indices of the associated encoded token(s).

Index or indices of the associated encoded token(s).

Get the word in the original string corresponding to a character in the original string of a sequence of the batch.

This method is particularly suited when the input sequences are provided as pre-tokenized sequences (i.e. words are defined by the user). In this case it allows to easily associate encoded tokens with provided tokenized words.

( tensor_type: str | TensorType | None = None prepend_batch_axis: bool = False )

Convert the inner content to tensors.

( batch_index: int = 0 ) → list[Optional[int]]

A list indicating the sequence id corresponding to each token. Special tokens added by the tokenizer are mapped to None and other tokens are mapped to the index of their corresponding sequence.

A list indicating the sequence id corresponding to each token. Special tokens added by the tokenizer are mapped to None and other tokens are mapped to the index of their corresponding sequence.

Return a list mapping the tokens to the id of their original sentences:

( device: str | torch.device non_blocking: bool = False ) → BatchEncoding

The same instance after modification.

The same instance after modification.

Send all values to device by calling v.to(device, non_blocking=non_blocking) (PyTorch only).

( batch_or_token_index: int token_index: int | None = None ) → CharSpan

Span of characters in the original string, or None, if the token (e.g. , ) doesn’t correspond to any chars in the origin string.

Span of characters in the original string, or None, if the token (e.g. , ) doesn’t correspond to any chars in the origin string.

Get the character span corresponding to an encoded token in a sequence of the batch.

Character spans are returned as a CharSpan with:

( batch_or_token_index: int token_index: int | None = None ) → int

Index of the word in the input sequence.

Index of the word in the input sequence.

Get the index of the sequence represented by the given token. In the general use case, this method returns 0 for a single sequence or the first sequence of a pair, and 1 for the second sequence of a pair

This method is particularly suited when the input sequences are provided as pre-tokenized sequences (i.e., words are defined by the user). In this case it allows to easily associate encoded tokens with provided tokenized words.

( batch_or_token_index: int token_index: int | None = None ) → int

Index of the word in the input sequence.

Index of the word in the input sequence.

Get the index of the word corresponding (i.e. comprising) to an encoded token in a sequence of the batch.

This method is particularly suited when the input sequences are provided as pre-tokenized sequences (i.e., words are defined by the user). In this case it allows to easily associate encoded tokens with provided tokenized words.

( batch_index: int = 0 ) → list[str]

The list of tokens at that index.

The list of tokens at that index.

Return the list of tokens (sub-parts of the input strings after word/subword splitting and before conversion to integer indices) at a given batch index (only works for the output of a fast tokenizer).

( batch_index: int = 0 ) → list[Optional[int]]

A list indicating the word corresponding to each token. Special tokens added by the tokenizer are mapped to None and other tokens are mapped to the index of their corresponding word (several tokens will be mapped to the same word index if they are parts of that word).

A list indicating the word corresponding to each token. Special tokens added by the tokenizer are mapped to None and other tokens are mapped to the index of their corresponding word (several tokens will be mapped to the same word index if they are parts of that word).

Return a list mapping the tokens to their actual word in the initial sentence for a fast tokenizer.

( batch_or_word_index: int word_index: int | None = None sequence_index: int = 0 ) → CharSpan or list[CharSpan]

CharSpan or list[CharSpan]

Span(s) of the associated character or characters in the string. CharSpan are NamedTuple with: start: index of the first character associated to the token in the original string end: index of the character following the last character associated to the token in the original string

Span(s) of the associated character or characters in the string. CharSpan are NamedTuple with:

Get the character span in the original string corresponding to given word in a sequence of the batch.

Character spans are returned as a CharSpan NamedTuple with:

( batch_or_word_index: int word_index: int | None = None sequence_index: int = 0 ) → (TokenSpan, optional)

(TokenSpan, optional)

Span of tokens in the encoded sequence. Returns None if no tokens correspond to the word. This can happen especially when the token is a special token that has been used to format the tokenization. For example when we add a class token at the very beginning of the tokenization.

Span of tokens in the encoded sequence. Returns None if no tokens correspond to the word. This can happen especially when the token is a special token that has been used to format the tokenization. For example when we add a class token at the very beginning of the tokenization.

Get the encoded token span corresponding to a word in a sequence of the batch.

Token spans are returned as a TokenSpan with:

This method is particularly suited when the input sequences are provided as pre-tokenized sequences (i.e. words are defined by the user). In this case it allows to easily associate encoded tokens with provided tokenized words.

---

## Transformers

**URL:** https://huggingface.co/docs/transformers/v5.0.0/en/internal/pipelines_utils

**Contents:**
- Transformers
- Utilities for pipelines
- Argument handling
  - class transformers.pipelines.ArgumentHandler
  - class transformers.pipelines.ZeroShotClassificationArgumentHandler
  - class transformers.pipelines.QuestionAnsweringArgumentHandler
- Data format
  - class transformers.PipelineDataFormat
    - from_str
    - save

Transformers documentation

Utilities for pipelines

and get access to the augmented documentation experience

This page lists all the utility functions the library provides for pipelines.

Most of those are only useful if you are studying the code of the models in the library.

Base interface for handling arguments for each Pipeline.

Handles arguments for zero-shot for text classification by turning each possible label into an NLI premise/hypothesis pair.

QuestionAnsweringPipeline requires the user to provide multiple arguments (i.e. question & context) to be mapped to internal SquadExample.

QuestionAnsweringArgumentHandler manages all the possible to create a SquadExample from the command-line supplied arguments.

( output_path: str | None input_path: str | None column: str | None overwrite: bool = False )

Base class for all the pipeline supported data format both for reading and writing. Supported data formats currently includes:

PipelineDataFormat also includes some utilities to work with multi-columns like mapping from datasets columns to pipelines keyword arguments through the dataset_kwarg_1=dataset_column_1 format.

( format: str output_path: str | None input_path: str | None column: str | None overwrite = False ) → PipelineDataFormat

The proper data format.

The proper data format.

Creates an instance of the right subclass of PipelineDataFormat depending on format.

( data: dict | list[dict] )

Save the provided data object with the representation for the current PipelineDataFormat.

( data: dict | list[dict] ) → str

Path where the data has been saved.

Path where the data has been saved.

Save the provided data object as a pickle-formatted binary data on the disk.

( output_path: str | None input_path: str | None column: str | None overwrite = False )

Support for pipelines using CSV data format.

Save the provided data object with the representation for the current PipelineDataFormat.

( output_path: str | None input_path: str | None column: str | None overwrite = False )

Support for pipelines using JSON file format.

Save the provided data object in a json file.

( output_path: str | None input_path: str | None column: str | None overwrite: bool = False )

Read data from piped input to the python process. For multi columns data, columns should separated by

If columns are provided, then the output will be a dictionary with {column_x: value_x}

( task: str model: str reason: str )

Raised by a Pipeline when handling call.

---

## Transformers

**URL:** https://huggingface.co/docs/transformers/v5.0.0/en/llm_tutorial

**Contents:**
- Transformers
- Text generation
- Default generate
- Generation configuration
  - Saving
- Common Options
- Pitfalls
  - Output length
  - Decoding strategy
  - Padding side

Transformers documentation

and get access to the augmented documentation experience

Text generation is the most popular application for large language models (LLMs). A LLM is trained to generate the next word (token) given some initial text (prompt) along with its own generated outputs up to a predefined length or when it reaches an end-of-sequence (EOS) token.

In Transformers, the generate() API handles text generation, and it is available for all models with generative capabilities. This guide will show you the basics of text generation with generate() and some common pitfalls to avoid.

For the following commands, please make sure transformers serve is running.

Before you begin, it’s helpful to install bitsandbytes to quantize really large models to reduce their memory usage.

Bitsandbytes supports multiple backends in addition to CUDA-based GPUs. Refer to the multi-backend installation guide to learn more.

Load a LLM with from_pretrained() and add the following two parameters to reduce the memory requirements.

Tokenize your input, and set the padding_side() parameter to "left" because a LLM is not trained to continue generation from padding tokens. The tokenizer returns the input ids and attention mask.

Process more than one prompt at a time by passing a list of strings to the tokenizer. Batch the inputs to improve throughput at a small cost to latency and memory.

Pass the inputs to generate() to generate tokens, and batch_decode() the generated tokens back to text.

All generation settings are contained in GenerationConfig. In the example above, the generation settings are derived from the generation_config.json file of mistralai/Mistral-7B-v0.1. A default decoding strategy is used when no configuration is saved with a model.

Inspect the configuration through the generation_config attribute. It only shows values that are different from the default configuration, in this case, the bos_token_id and eos_token_id.

You can customize generate() by overriding the parameters and values in GenerationConfig. See this section below for commonly adjusted parameters.

generate() can also be extended with external libraries or custom code:

Refer to the Generation strategies guide to learn more about search, sampling, and decoding strategies.

Create an instance of GenerationConfig and specify the decoding parameters you want.

Use save_pretrained() to save a specific generation configuration and set the push_to_hub parameter to True to upload it to the Hub.

Leave the config_file_name parameter empty. This parameter should be used when storing multiple generation configurations in a single directory. It gives you a way to specify which generation configuration to load. You can create different configurations for different generative tasks (creative text generation with sampling, summarization with beam search) for use with a single model.

generate() is a powerful tool that can be heavily customized. This can be daunting for a new users. This section contains a list of popular generation options that you can define in most text generation tools in Transformers: generate(), GenerationConfig, pipelines, the chat CLI, …

The section below covers some common issues you may encounter during text generation and how to solve them.

generate() returns up to 20 tokens by default unless otherwise specified in a models GenerationConfig. It is highly recommended to manually set the number of generated tokens with the max_new_tokens parameter to control the output length. Decoder-only models returns the initial prompt along with the generated tokens.

The default decoding strategy in generate() is greedy search, which selects the next most likely token, unless otherwise specified in a models GenerationConfig. While this decoding strategy works well for input-grounded tasks (transcription, translation), it is not optimal for more creative use cases (story writing, chat applications).

For example, enable a multinomial sampling strategy to generate more diverse outputs. Refer to the Generation strategy guide for more decoding strategies.

Inputs need to be padded if they don’t have the same length. But LLMs aren’t trained to continue generation from padding tokens, which means the padding_side() parameter needs to be set to the left of the input.

Some models and tasks expect a certain input prompt format, and if the format is incorrect, the model returns a suboptimal output. You can learn more about prompting in the prompt engineering guide.

For example, a chat model expects the input as a chat template. Your prompt should include a role and content to indicate who is participating in the conversation. If you try to pass your prompt as a single string, the model doesn’t always return the expected output.

Take a look below for some more specific and specialized text generation libraries.

---

## Transformers

**URL:** https://huggingface.co/docs/transformers/v5.0.0/en/main_classes/peft

**Contents:**
- Transformers
- PEFT
  - class transformers.integrations.PeftAdapterMixin
    - load_adapter
    - add_adapter
    - set_adapter
    - disable_adapters
    - enable_adapters
    - active_adapters
    - get_adapter_state_dict

Transformers documentation

and get access to the augmented documentation experience

The PeftAdapterMixin provides functions from the PEFT library for managing adapters with Transformers. This mixin currently supports LoRA, IA3, and AdaLora. Prefix tuning methods (prompt tuning, prompt learning) aren’t supported because they can’t be injected into a torch module.

A class containing all functions for loading and using adapters weights that are supported in PEFT library. For more details about adapters and injecting them on a transformer-based model, check out the documentation of PEFT library: https://huggingface.co/docs/peft/index

Currently supported PEFT methods are all non-prompt learning methods (LoRA, IA³, etc.). Other PEFT models such as prompt tuning, prompt learning are out of scope as these adapters are not “injectable” into a torch module. For using these methods, please refer to the usage guide of PEFT library.

With this mixin, if the correct PEFT version is installed (>= 0.18.0), it is possible to:

( peft_model_id: str | None = None adapter_name: str | None = None peft_config: dict[str, typing.Any] | None = None adapter_state_dict: dict[str, 'torch.Tensor'] | None = None low_cpu_mem_usage: bool = False is_trainable: bool = False hotswap: typing.Union[bool, typing.Literal['auto']] = 'auto' local_files_only: bool = False adapter_kwargs: dict[str, typing.Any] | None = None load_config: typing.Optional[ForwardRef('LoadStateDictConfig')] = None **kwargs )

If the new adapter and the old adapter have different ranks and/or LoRA alphas (i.e. scaling), you need to call an additional method before loading the adapter:

Load adapter weights from file or remote Hub folder. If you are not familiar with adapters and PEFT methods, we invite you to read more about them on PEFT official documentation: https://huggingface.co/docs/peft

Requires PEFT to be installed as a backend to load the adapter weights.

( adapter_config adapter_name: str | None = None )

If you are not familiar with adapters and PEFT methods, we invite you to read more about them on the PEFT official documentation: https://huggingface.co/docs/peft

Adds a fresh new adapter to the current model for training purpose. If no adapter name is passed, a default name is assigned to the adapter to follow the convention of PEFT library (in PEFT we use “default” as the default adapter name).

Note that the newly added adapter is not automatically activated. To activate it, use model.set_adapter.

( adapter_name: list[str] | str )

If you are not familiar with adapters and PEFT methods, we invite you to read more about them on the PEFT official documentation: https://huggingface.co/docs/peft

Sets a specific adapter by forcing the model to use a that adapter and disable the other adapters.

If you are not familiar with adapters and PEFT methods, we invite you to read more about them on the PEFT official documentation: https://huggingface.co/docs/peft

Disable all adapters that are attached to the model. This leads to inferring with the base model only.

If you are not familiar with adapters and PEFT methods, we invite you to read more about them on the PEFT official documentation: https://huggingface.co/docs/peft

Enable adapters that are attached to the model.

If you are not familiar with adapters and PEFT methods, we invite you to read more about them on the PEFT official documentation: https://huggingface.co/docs/peft

Gets the current active adapters of the model. In case of multi-adapter inference (combining multiple adapters for inference) returns the list of all active adapters so that users can deal with them accordingly.

For previous PEFT versions (that does not support multi-adapter inference), module.active_adapter will return a single string.

( adapter_name: str | None = None state_dict: dict | None = None )

If you are not familiar with adapters and PEFT methods, we invite you to read more about them on the PEFT official documentation: https://huggingface.co/docs/peft

Gets the adapter state dict that should only contain the weights tensors of the specified adapter_name adapter. If no adapter_name is passed, the active adapter is used.

---

## Transformers

**URL:** https://huggingface.co/docs/transformers/v5.0.0/en/quicktour

**Contents:**
- Transformers
- Quickstart
- Set up
- Pretrained models
- Pipeline
- Trainer
- Next steps

Transformers documentation

and get access to the augmented documentation experience

Transformers is designed to be fast and easy to use so that everyone can start learning or building with transformer models.

The number of user-facing abstractions is limited to only three classes for instantiating a model, and two APIs for inference or training. This quickstart introduces you to Transformers’ key features and shows you how to:

To start, we recommend creating a Hugging Face account. An account lets you host and access version controlled models, datasets, and Spaces on the Hugging Face Hub, a collaborative platform for discovery and building.

Create a User Access Token and log in to your account.

Paste your User Access Token into notebook_login when prompted to log in.

Then install an up-to-date version of Transformers and some additional libraries from the Hugging Face ecosystem for accessing datasets and vision models, evaluating training, and optimizing training for large models.

Each pretrained model inherits from three base classes.

We recommend using the AutoClass API to load models and preprocessors because it automatically infers the appropriate architecture for each task and machine learning framework based on the name or path to the pretrained weights and configuration file.

Use from_pretrained() to load the weights and configuration file from the Hub into the model and preprocessor class.

When you load a model, configure the following parameters to ensure the model is optimally loaded.

Tokenize the text and return PyTorch tensors with the tokenizer. Move the model to an accelerator if it’s available to accelerate inference.

The model is now ready for inference or training.

For inference, pass the tokenized inputs to generate() to generate text. Decode the token ids back into text with batch_decode().

Skip ahead to the Trainer section to learn how to fine-tune a model.

The Pipeline class is the most convenient way to inference with a pretrained model. It supports many tasks such as text generation, image segmentation, automatic speech recognition, document question answering, and more.

Refer to the Pipeline API reference for a complete list of available tasks.

Create a Pipeline object and select a task. By default, Pipeline downloads and caches a default pretrained model for a given task. Pass the model name to the model parameter to choose a specific model.

Use Accelerator to automatically detect an available accelerator for inference.

Prompt Pipeline with some initial text to generate more text.

Trainer is a complete training and evaluation loop for PyTorch models. It abstracts away a lot of the boilerplate usually involved in manually writing a training loop, so you can start training faster and focus on training design choices. You only need a model, dataset, a preprocessor, and a data collator to build batches of data from the dataset.

Use the TrainingArguments class to customize the training process. It provides many options for training, evaluation, and more. Experiment with training hyperparameters and features like batch size, learning rate, mixed precision, torch.compile, and more to meet your training needs. You could also use the default training parameters to quickly produce a baseline.

Load a model, tokenizer, and dataset for training.

Create a function to tokenize the text and convert it into PyTorch tensors. Apply this function to the whole dataset with the map method.

Load a data collator to create batches of data and pass the tokenizer to it.

Next, set up TrainingArguments with the training features and hyperparameters.

Finally, pass all these separate components to Trainer and call train() to start.

Share your model and tokenizer to the Hub with push_to_hub().

Congratulations, you just trained your first model with Transformers!

Now that you have a better understanding of Transformers and what it offers, it’s time to keep exploring and learning what interests you the most.

---

## Transformers

**URL:** https://huggingface.co/docs/transformers/v5.0.0/en/main_classes/video_processor

**Contents:**
- Transformers
- Video Processor
  - Usage Example
    - Sampling behavior
- BaseVideoProcessor
  - class transformers.BaseVideoProcessor
    - convert_to_rgb
    - fetch_videos
    - from_dict
    - from_json_file

Transformers documentation

and get access to the augmented documentation experience

A Video Processor is a utility responsible for preparing input features for video models, as well as handling the post-processing of their outputs. It provides transformations such as resizing, normalization, and conversion into PyTorch. Along ith transformations the VideoProcessor class handles video decoding from local paths or URLs (requires torchcodec) and frame sampling according to model-specific strategies.

The video processor extends the functionality of image processors by allowing Vision Large Language Models (VLMs) to handle videos with a distinct set of arguments compared to images. It serves as the bridge between raw video data and the model, ensuring that input features are optimized for the VLM.

When adding a new VLM or updating an existing one to enable distinct video preprocessing, saving and reloading the processor configuration will store the video related arguments in a dedicated file named video_preprocessing_config.json. Don’t worry if you haven’t updated your VLM, the processor will try to load video related configurations from a file named preprocessing_config.json.

Here’s an example of how to load a video processor with llava-hf/llava-onevision-qwen2-0.5b-ov-hf model:

Currently, if using base image processor for videos, it processes video data by treating each frame as an individual image and applying transformations frame-by-frame. While functional, this approach is not highly efficient. Using AutoVideoProcessor allows us to take advantage of fast video processors, leveraging the torchvision library. Fast processors handle the whole batch of videos at once, without iterating over each video or frame. These updates introduce GPU acceleration and significantly enhance processing speed, especially for tasks requiring high throughput.

Fast video processors are available for all models and are loaded by default when an AutoVideoProcessor is initialized. When using a fast video processor, you can also set the device argument to specify the device on which the processing should be done. By default, the processing is done on the same device as the inputs if the inputs are tensors, or on the CPU otherwise. For even more speed improvement, we can compile the processor when using ‘cuda’ as device.

The video processor can also sample video frames using the technique best suited for the given model. Sampling behavior is controlled with the do_sample_frames argument and can be configured through model-specific parameters such as num_frames or fps (the rate at which the video will be sampled). If the input video is given as a local path or URL (str), the processor will decode it automatically. To obtain metadata about the decoded video, such as sampled frame indices, original dimensions, duration, and fps, pass return_metadata=True to the processor.

Specifying num_frames does not guarantee the output will contain exactly that number of frames. Depending on the model, the sampler may enforce minimum or maximum frame limits.

The default decoder is torchcodec, which must be installed.

If you pass an already decoded video array but still want to enable model-specific frame sampling, it is strongly recommended to provide video_metadata. This allows the sampler to know the original video’s duration and FPS. You can pass metadata as a VideoMetadata object or as a plain dict.

( **kwargs: typing_extensions.Unpack[transformers.processing_utils.VideosKwargs] )

Constructs a base VideoProcessor.

( video: torch.Tensor ) → torch.Tensor

Converts a video to RGB format.

( video_url_or_urls: str | list[str] | list[list[str]] sample_indices_fn = None )

Convert a single or a list of urls into the corresponding np.array objects.

If a single url is passed, the return value will be a single object. If a list is passed a list of objects is returned.

( video_processor_dict: dict **kwargs ) → ~video_processing_utils.VideoProcessorBase

~video_processing_utils.VideoProcessorBase

The video processor object instantiated from those parameters.

The video processor object instantiated from those parameters.

Instantiates a type of ~video_processing_utils.VideoProcessorBase from a Python dictionary of parameters.

( json_file: str | os.PathLike ) → A video processor of type ~video_processing_utils.VideoProcessorBase

A video processor of type ~video_processing_utils.VideoProcessorBase

The video_processor object instantiated from that JSON file.

The video_processor object instantiated from that JSON file.

Instantiates a video processor of type ~video_processing_utils.VideoProcessorBase from the path to a JSON file of parameters.

( pretrained_model_name_or_path: str | os.PathLike cache_dir: str | os.PathLike | None = None force_download: bool = False local_files_only: bool = False token: str | bool | None = None revision: str = 'main' **kwargs )

Instantiate a type of ~video_processing_utils.VideoProcessorBase from an video processor.

( pretrained_model_name_or_path: str | os.PathLike **kwargs ) → tuple[Dict, Dict]

The dictionary(ies) that will be used to instantiate the video processor object.

The dictionary(ies) that will be used to instantiate the video processor object.

From a pretrained_model_name_or_path, resolve to a dictionary of parameters, to be used for instantiating a video processor of type ~video_processing_utils.VideoProcessorBase using from_dict.

( videos: typing.Union[list['PIL.Image.Image'], numpy.ndarray, ForwardRef('torch.Tensor'), list[numpy.ndarray], list['torch.Tensor'], list[list['PIL.Image.Image']], list[list[numpy.ndarray]], list[list['torch.Tensor']], transformers.video_utils.URL, list[transformers.video_utils.URL], list[list[transformers.video_utils.URL]], transformers.video_utils.Path, list[transformers.video_utils.Path], list[list[transformers.video_utils.Path]]] **kwargs: typing_extensions.Unpack[transformers.processing_utils.VideosKwargs] )

( auto_class = 'AutoVideoProcessor' )

Register this class with a given auto class. This should only be used for custom video processors as the ones in the library are already mapped with AutoVideoProcessor .

This API is experimental and may have some slight breaking changes in the next releases.

( metadata: VideoMetadata num_frames: int | None = None fps: int | float | None = None **kwargs ) → np.ndarray

Indices to sample video frames.

Indices to sample video frames.

Default sampling function which uniformly samples the desired number of frames between 0 and total number of frames. If fps is passed along with metadata, fps frames per second are sampled uniformty. Arguments num_frames and fps are mutually exclusive.

( save_directory: str | os.PathLike push_to_hub: bool = False **kwargs )

Save an video processor object to the directory save_directory, so that it can be re-loaded using the ~video_processing_utils.VideoProcessorBase.from_pretrained class method.

Dictionary of all the attributes that make up this video processor instance.

Dictionary of all the attributes that make up this video processor instance.

Serializes this instance to a Python dictionary.

---

## Transformers

**URL:** https://huggingface.co/docs/transformers/v5.0.0/en/main_classes/logging

**Contents:**
- Transformers
- Logging
- logging vs warnings
    - transformers.utils.logging.captureWarnings
- Base setters
    - transformers.utils.logging.set_verbosity_error
    - transformers.utils.logging.set_verbosity_warning
    - transformers.utils.logging.set_verbosity_info
    - transformers.utils.logging.set_verbosity_debug
- Other functions

Transformers documentation

and get access to the augmented documentation experience

🤗 Transformers has a centralized logging system, so that you can setup the verbosity of the library easily.

Currently the default verbosity of the library is WARNING.

To change the level of verbosity, just use one of the direct setters. For instance, here is how to change the verbosity to the INFO level.

You can also use the environment variable TRANSFORMERS_VERBOSITY to override the default verbosity. You can set it to one of the following: debug, info, warning, error, critical, fatal. For example:

Additionally, some warnings can be disabled by setting the environment variable TRANSFORMERS_NO_ADVISORY_WARNINGS to a true value, like 1. This will disable any warning that is logged using logger.warning_advice. For example:

Here is an example of how to use the same logger as the library in your own module or script:

All the methods of this logging module are documented below, the main ones are logging.get_verbosity() to get the current level of verbosity in the logger and logging.set_verbosity() to set the verbosity to the level of your choice. In order (from the least verbose to the most verbose), those levels (with their corresponding int values in parenthesis) are:

By default, tqdm progress bars will be displayed during model download. logging.disable_progress_bar() and logging.enable_progress_bar() can be used to suppress or unsuppress this behavior.

Python has two logging systems that are often used in conjunction: logging, which is explained above, and warnings, which allows further classification of warnings in specific buckets, e.g., FutureWarning for a feature or path that has already been deprecated and DeprecationWarning to indicate an upcoming deprecation.

We use both in the transformers library. We leverage and adapt logging’s captureWarnings method to allow management of these warning messages by the verbosity setters above.

What does that mean for developers of the library? We should respect the following heuristics:

See reference of the captureWarnings method below.

Calls the captureWarnings method from the logging library to enable management of the warnings emitted by the warnings library.

Read more about this method here: https://docs.python.org/3/library/logging.html#integration-with-the-warnings-module

All warnings will be logged through the py.warnings logger.

Careful: this method also adds a handler to this logger if it does not already have one, and updates the logging level of that logger to the library’s root logger.

Set the verbosity to the ERROR level.

Set the verbosity to the WARNING level.

Set the verbosity to the INFO level.

Set the verbosity to the DEBUG level.

Return the current level for the 🤗 Transformers’s root logger as an int.

🤗 Transformers has following logging levels:

Set the verbosity level for the 🤗 Transformers’s root logger.

( name: str | None = None )

Return a logger with the specified name.

This function is not supposed to be directly accessed unless you are writing a custom transformers module.

Enable the default handler of the HuggingFace Transformers’s root logger.

Disable the default handler of the HuggingFace Transformers’s root logger.

Enable explicit formatting for every HuggingFace Transformers’s logger. The explicit formatter is as follows:

Resets the formatting for HuggingFace Transformers’s loggers.

All handlers currently bound to the root logger are affected by this method.

Enable tqdm progress bar.

Disable tqdm progress bar.

---

## Transformers

**URL:** https://huggingface.co/docs/transformers/v5.0.0/en/philosophy

**Contents:**
- Transformers
- Philosophy
- Who this library is for
- What you can expect
- Core tenets
- Main classes

Transformers documentation

and get access to the augmented documentation experience

Transformers is a PyTorch-first library. It provides models that are faithful to their papers, easy to use, and easy to hack.

A longer, in-depth article with examples, visualizations and timelines is available here as our canonical reference.

Our philosophy evolves through practice. What follows are our current, stable principles.

Three core classes are required for each model: configuration, models, and a preprocessing class. Tokenizers handle NLP, image processors handle images, video processors handle videos, feature extractors handle audio, and processors handle multimodal inputs.

All of these classes can be initialized in a simple and unified way from pretrained instances by using a common from_pretrained() method which downloads (if needed), caches and loads the related class instance and associated data (configurations’ hyperparameters, tokenizers’ vocabulary, processors’ parameters and models’ weights) from a pretrained checkpoint provided on Hugging Face Hub or your own saved checkpoint.

On top of those three base classes, the library provides two APIs: pipeline() for quickly using a model for inference on a given task and Trainer to quickly train or fine-tune a PyTorch model.

The following tenets solidified over time, and they’re detailed in our new philosophy blog post. They guide maintainer decisions when reviewing PRs and contributions.

Configuration classes store the hyperparameters required to build a model. These include the number of layers and hidden size. You don’t always need to instantiate these yourself. When using a pretrained model without modification, creating the model automatically instantiates the configuration.

Model classes are PyTorch models (torch.nn.Module), wrapped by at least a PreTrainedModel.

Modular transformers. Contributors write a small modular_*.py shard that declares reuse from existing components. The library auto-expands this into the visible modeling_*.py file that users read/debug. Maintainers review the shard; users hack the expanded file. This preserves “One Model, One File” without boilerplate drift. See the contributing documentation for more information.

Preprocessing classes convert the raw data into a format accepted by the model. A tokenizer stores the vocabulary for each model and provides methods for encoding and decoding strings in a list of token embedding indices. Image processors preprocess vision inputs, video processors preprocess videos inputs, feature extractors preprocess audio inputs, and processors preprocess multimodal inputs.

All these classes can be instantiated from pretrained instances, saved locally, and shared on the Hub with three methods:

---

## Transformers

**URL:** https://huggingface.co/docs/transformers/v5.0.0/en/internal/image_processing_utils

**Contents:**
- Transformers
- Utilities for Image Processors
- Image Transformations
    - transformers.image_transforms.center_crop
    - transformers.image_transforms.center_to_corners_format
    - transformers.image_transforms.corners_to_center_format
    - transformers.image_transforms.id_to_rgb
    - transformers.image_transforms.normalize
    - transformers.image_transforms.pad
    - transformers.image_transforms.rgb_to_id

Transformers documentation

Utilities for Image Processors

and get access to the augmented documentation experience

This page lists all the utility functions used by the image processors, mainly the functional transformations used to process the images.

Most of those are only useful if you are studying the code of the image processors in the library.

( image: ndarray size: tuple data_format: str | transformers.image_utils.ChannelDimension | None = None input_data_format: str | transformers.image_utils.ChannelDimension | None = None ) → np.ndarray

Crops the image to the specified size using a center crop. Note that if the image is too small to be cropped to the size given, it will be padded (so the returned result will always be of size size).

( bboxes_center: TensorType )

Converts bounding boxes from center format to corners format.

center format: contains the coordinate for the center of the box and its width, height dimensions (center_x, center_y, width, height) corners format: contains the coordinates for the top-left and bottom-right corners of the box (top_left_x, top_left_y, bottom_right_x, bottom_right_y)

( bboxes_corners: TensorType )

Converts bounding boxes from corners format to center format.

corners format: contains the coordinates for the top-left and bottom-right corners of the box (top_left_x, top_left_y, bottom_right_x, bottom_right_y) center format: contains the coordinate for the center of the box and its the width, height dimensions (center_x, center_y, width, height)

Converts unique ID to RGB color.

( image: ndarray mean: float | collections.abc.Collection[float] std: float | collections.abc.Collection[float] data_format: transformers.image_utils.ChannelDimension | None = None input_data_format: str | transformers.image_utils.ChannelDimension | None = None )

Normalizes image using the mean and standard deviation specified by mean and std.

image = (image - mean) / std

( image: ndarray padding: int | tuple[int, int] | collections.abc.Iterable[tuple[int, int]] mode: PaddingMode = <PaddingMode.CONSTANT: 'constant'> constant_values: float | collections.abc.Iterable[float] = 0.0 data_format: str | transformers.image_utils.ChannelDimension | None = None input_data_format: str | transformers.image_utils.ChannelDimension | None = None ) → np.ndarray

Pads the image with the specified (height, width) padding and mode.

Converts RGB color to unique ID.

( image: ndarray scale: float data_format: transformers.image_utils.ChannelDimension | None = None dtype: dtype = <class 'numpy.float32'> input_data_format: str | transformers.image_utils.ChannelDimension | None = None ) → np.ndarray

Rescales image by scale.

( image: ndarray size: tuple resample: typing.Optional[ForwardRef('PILImageResampling')] = None reducing_gap: int | None = None data_format: transformers.image_utils.ChannelDimension | None = None return_numpy: bool = True input_data_format: str | transformers.image_utils.ChannelDimension | None = None ) → np.ndarray

Resizes image to (height, width) specified by size using the PIL library.

( image: typing.Union[numpy.ndarray, ForwardRef('PIL.Image.Image'), ForwardRef('torch.Tensor')] do_rescale: bool | None = None image_mode: str | None = None input_data_format: str | transformers.image_utils.ChannelDimension | None = None ) → PIL.Image.Image

Converts image to a PIL Image. Optionally rescales it and puts the channel dimension back as the last axis if needed.

This is an image processor mixin used to provide saving/loading functionality for sequential and image feature extractors.

( image_url_or_urls: str | list[str] | list[list[str]] )

Convert a single or a list of urls into the corresponding PIL.Image objects.

If a single url is passed, the return value will be a single object. If a list is passed a list of objects is returned.

( image_processor_dict: dict **kwargs ) → ImageProcessingMixin

The image processor object instantiated from those parameters.

The image processor object instantiated from those parameters.

Instantiates a type of ImageProcessingMixin from a Python dictionary of parameters.

( json_file: str | os.PathLike ) → A image processor of type ImageProcessingMixin

A image processor of type ImageProcessingMixin

The image_processor object instantiated from that JSON file.

The image_processor object instantiated from that JSON file.

Instantiates a image processor of type ImageProcessingMixin from the path to a JSON file of parameters.

( pretrained_model_name_or_path: str | os.PathLike cache_dir: str | os.PathLike | None = None force_download: bool = False local_files_only: bool = False token: str | bool | None = None revision: str = 'main' **kwargs )

Instantiate a type of ImageProcessingMixin from an image processor.

( pretrained_model_name_or_path: str | os.PathLike **kwargs ) → tuple[Dict, Dict]

The dictionary(ies) that will be used to instantiate the image processor object.

The dictionary(ies) that will be used to instantiate the image processor object.

From a pretrained_model_name_or_path, resolve to a dictionary of parameters, to be used for instantiating a image processor of type ~image_processor_utils.ImageProcessingMixin using from_dict.

( repo_id: str commit_message: str | None = None commit_description: str | None = None private: bool | None = None token: bool | str | None = None revision: str | None = None create_pr: bool = False max_shard_size: int | str | None = '50GB' tags: list[str] | None = None )

Upload the image processor file to the 🤗 Model Hub.

( auto_class = 'AutoImageProcessor' )

Register this class with a given auto class. This should only be used for custom image processors as the ones in the library are already mapped with AutoImageProcessor .

( save_directory: str | os.PathLike push_to_hub: bool = False **kwargs )

Save an image processor object to the directory save_directory, so that it can be re-loaded using the from_pretrained() class method.

Dictionary of all the attributes that make up this image processor instance.

Dictionary of all the attributes that make up this image processor instance.

Serializes this instance to a Python dictionary.

( json_file_path: str | os.PathLike )

Save this instance to a JSON file.

String containing all the attributes that make up this feature_extractor instance in JSON format.

String containing all the attributes that make up this feature_extractor instance in JSON format.

Serializes this instance to a JSON string.

---

## Transformers

**URL:** https://huggingface.co/docs/transformers/v5.0.0/en/index

**Contents:**
- Transformers
- Transformers
- Features
- Design
- Learn

Transformers documentation

and get access to the augmented documentation experience

Transformers acts as the model-definition framework for state-of-the-art machine learning models in text, computer vision, audio, video, and multimodal model, for both inference and training.

It centralizes the model definition so that this definition is agreed upon across the ecosystem. transformers is the pivot across frameworks: if a model definition is supported, it will be compatible with the majority of training frameworks (Axolotl, Unsloth, DeepSpeed, FSDP, PyTorch-Lightning, …), inference engines (vLLM, SGLang, TGI, …), and adjacent modeling libraries (llama.cpp, mlx, …) which leverage the model definition from transformers.

We pledge to help support new state-of-the-art models and democratize their usage by having their model definition be simple, customizable, and efficient.

There are over 1M+ Transformers model checkpoints on the Hugging Face Hub you can use.

Explore the Hub today to find a model and use Transformers to help you get started right away.

Explore the Models Timeline to discover the latest text, vision, audio and multimodal model architectures in Transformers.

Transformers provides everything you need for inference or training with state-of-the-art pretrained models. Some of the main features include:

Read our Philosophy to learn more about Transformers’ design principles.

Transformers is designed for developers and machine learning engineers and researchers. Its main design principles are:

If you’re new to Transformers or want to learn more about transformer models, we recommend starting with the LLM course. This comprehensive course covers everything from the fundamentals of how transformer models work to practical applications across various tasks. You’ll learn the complete workflow, from curating high-quality datasets to fine-tuning large language models and implementing reasoning capabilities. The course contains both theoretical and hands-on exercises to build a solid foundational knowledge of transformer models as you learn.

---

## Transformers

**URL:** https://huggingface.co/docs/transformers/v5.0.0/en/main_classes/pipelines

**Contents:**
- Transformers
- Pipelines
- The pipeline abstraction
    - transformers.pipeline
- Pipeline batching
- Pipeline chunk batching
- Pipeline FP16 inference
- Pipeline custom code
- Implementing a pipeline
- Audio

Transformers documentation

and get access to the augmented documentation experience

The pipelines are a great and easy way to use models for inference. These pipelines are objects that abstract most of the complex code from the library, offering a simple API dedicated to several tasks, including Named Entity Recognition, Masked Language Modeling, Sentiment Analysis, Feature Extraction and Question Answering. See the task summary for examples of use.

There are two categories of pipeline abstractions to be aware about:

The pipeline abstraction is a wrapper around all the other available pipelines. It is instantiated as any other pipeline but can provide additional quality of life.

Simple call on one item:

If you want to use a specific model from the hub you can ignore the task if the model on the hub already defines it:

To call a pipeline on many items, you can call it with a list.

To iterate over full datasets it is recommended to use a dataset directly. This means you don’t need to allocate the whole dataset at once, nor do you need to do batching yourself. This should work just as fast as custom loops on GPU. If it doesn’t don’t hesitate to create an issue.

For ease of use, a generator is also possible:

( task: str | None = None model: str | PreTrainedModel | None = None config: str | PreTrainedConfig | None = None tokenizer: str | PreTrainedTokenizer | PreTrainedTokenizerFast | None = None feature_extractor: str | PreTrainedFeatureExtractor | None = None image_processor: str | BaseImageProcessor | None = None processor: str | ProcessorMixin | None = None revision: str | None = None use_fast: bool = True token: str | bool | None = None device: int | str | torch.device | None = None device_map: str | dict[str, int | str] | None = None dtype: str | torch.dtype | None = 'auto' trust_remote_code: bool | None = None model_kwargs: dict[str, Any] | None = None pipeline_class: Any | None = None **kwargs: Any ) → Pipeline

If not provided, the default for the task will be loaded.

If not provided, the default configuration file for the requested model will be used. That means that if model is given, its default configuration will be used. However, if model is not supplied, this task’s default model’s config is used instead.

If not provided, the default tokenizer for the given model will be loaded (if it is a string). If model is not specified or not a string, then the default tokenizer for config is loaded (if it is a string). However, if config is also not given or not a string, then the default tokenizer for the given task will be loaded.

Feature extractors are used for non-NLP models, such as Speech or Vision models as well as multi-modal models. Multi-modal models will also require a tokenizer to be passed.

If not provided, the default feature extractor for the given model will be loaded (if it is a string). If model is not specified or not a string, then the default feature extractor for config is loaded (if it is a string). However, if config is also not given or not a string, then the default feature extractor for the given task will be loaded.

Image processors are used for Vision models and multi-modal models that require image inputs. Multi-modal models will also require a tokenizer to be passed.

If not provided, the default image processor for the given model will be loaded (if it is a string). If model is not specified or not a string, then the default image processor for config is loaded (if it is a string).

Processors are used for multi-modal models that require multi-modal inputs, for example, a model that requires both text and image inputs.

If not provided, the default processor for the given model will be loaded (if it is a string). If model is not specified or not a string, then the default processor for config is loaded (if it is a string).

Do not use device_map AND device at the same time as they will conflict

A suitable pipeline for the task.

A suitable pipeline for the task.

Utility factory method to build a Pipeline.

A pipeline consists of:

All pipelines can use batching. This will work whenever the pipeline uses its streaming ability (so when passing lists or Dataset or generator).

However, this is not automatically a win for performance. It can be either a 10x speedup or 5x slowdown depending on hardware, data and the actual model being used.

Example where it’s mostly a speedup:

Example where it’s most a slowdown:

This is a occasional very long sentence compared to the other. In that case, the whole batch will need to be 400 tokens long, so the whole batch will be [64, 400] instead of [64, 4], leading to the high slowdown. Even worse, on bigger batches, the program simply crashes.

There are no good (general) solutions for this problem, and your mileage may vary depending on your use cases. Rule of thumb:

For users, a rule of thumb is:

Measure performance on your load, with your hardware. Measure, measure, and keep measuring. Real numbers are the only way to go.

If you are latency constrained (live product doing inference), don’t batch.

If you are using CPU, don’t batch.

If you are using throughput (you want to run your model on a bunch of static data), on GPU, then:

As soon as you enable batching, make sure you can handle OOMs nicely.

zero-shot-classification and question-answering are slightly specific in the sense, that a single input might yield multiple forward pass of a model. Under normal circumstances, this would yield issues with batch_size argument.

In order to circumvent this issue, both of these pipelines are a bit specific, they are ChunkPipeline instead of regular Pipeline. In short:

This should be very transparent to your code because the pipelines are used in the same way.

This is a simplified view, since the pipeline can handle automatically the batch to ! Meaning you don’t have to care about how many forward passes you inputs are actually going to trigger, you can optimize the batch_size independently of the inputs. The caveats from the previous section still apply.

Models can be run in FP16 which can be significantly faster on GPU while saving memory. Most models will not suffer noticeable performance loss from this. The larger the model, the less likely that it will.

To enable FP16 inference, you can simply pass dtype=torch.float16 or dtype='float16' to the pipeline constructor. Note that this only works for models with a PyTorch backend. Your inputs will be converted to FP16 internally.

If you want to override a specific pipeline.

Don’t hesitate to create an issue for your task at hand, the goal of the pipeline is to be easy to use and support most cases, so transformers could maybe support your use case.

If you want to try simply you can:

That should enable you to do all the custom code you want.

Implementing a new pipeline

Pipelines available for audio tasks include the following.

Audio classification pipeline using any AutoModelForAudioClassification. This pipeline predicts the class of a raw waveform or an audio file. In case of an audio file, ffmpeg should be installed to support multiple audio formats.

Learn more about the basics of using a pipeline in the pipeline tutorial

This pipeline can currently be loaded from pipeline() using the following task identifier: "audio-classification".

See the list of available models on huggingface.co/models.

( inputs: numpy.ndarray | bytes | str | dict **kwargs: typing.Any ) → A list of dict with the following keys

A list of dict with the following keys

label (str) — The label predicted. score (float) — The corresponding probability.

Classify the sequence(s) given as inputs. See the AutomaticSpeechRecognitionPipeline documentation for more information.

( model: PreTrainedModel feature_extractor: typing.Union[ForwardRef('SequenceFeatureExtractor'), str, NoneType] = None tokenizer: transformers.tokenization_python.PythonBackend | None = None decoder: typing.Union[ForwardRef('BeamSearchDecoderCTC'), str, NoneType] = None device: typing.Union[int, ForwardRef('torch.device'), NoneType] = None **kwargs )

For more information on how to effectively use chunk_length_s, please have a look at the ASR chunking blog post.

For more information on how to effectively use stride_length_s, please have a look at the ASR chunking blog post.

Pipeline that aims at extracting spoken text contained within some audio.

The input can be either a raw waveform or a audio file. In case of the audio file, ffmpeg should be installed for to support multiple audio formats

Unless the model you’re using explicitly sets these generation parameters in its configuration files (generation_config.json), the following default values will be used:

Learn more about the basics of using a pipeline in the pipeline tutorial

( inputs: numpy.ndarray | bytes | str | dict **kwargs: typing.Any ) → Dict

For CTC models, timestamps can take one of two formats:

For the Whisper model, timestamps can take one of two formats:

A dictionary with the following keys: text (str): The recognized text. chunks (optional(, list[Dict]) When using return_timestamps, the chunks will become a list containing all the various text chunks identified by the model, e.g.* [{"text": "hi ", "timestamp": (0.5, 0.9)}, {"text": "there", "timestamp": (1.0, 1.5)}]. The original full text can roughly be recovered by doing "".join(chunk["text"] for chunk in output["chunks"]).

A dictionary with the following keys:

Transcribe the audio sequence(s) given as inputs to text. See the AutomaticSpeechRecognitionPipeline documentation for more information.

( *args vocoder = None sampling_rate = None **kwargs )

Text-to-audio generation pipeline using any AutoModelForTextToWaveform or AutoModelForTextToSpectrogram. This pipeline generates an audio file from an input text and optional other conditional inputs.

Unless the model you’re using explicitly sets these generation parameters in its configuration files (generation_config.json), the following default values will be used:

Learn more about the basics of using a pipeline in the pipeline tutorial

You can specify parameters passed to the model by using TextToAudioPipeline.__call__.forward_params or TextToAudioPipeline.__call__.generate_kwargs.

This pipeline can currently be loaded from pipeline() using the following task identifiers: "text-to-speech" or "text-to-audio".

See the list of available models on huggingface.co/models.

( text_inputs **forward_params ) → AudioOutput or a list of AudioOutput, which is a TypedDict with two keys

AudioOutput or a list of AudioOutput, which is a TypedDict with two keys

audio (np.ndarray of shape (nb_channels, audio_length)) — The generated audio waveform. sampling_rate (int) — The sampling rate of the generated audio waveform.

Generates speech/audio from the inputs. See the TextToAudioPipeline documentation for more information.

Zero shot audio classification pipeline using ClapModel. This pipeline predicts the class of an audio when you provide an audio and a set of candidate_labels.

The default hypothesis_template is : "This is a sound of {}.". Make sure you update it for your usage.

Learn more about the basics of using a pipeline in the pipeline tutorial This audio classification pipeline can currently be loaded from pipeline() using the following task identifier: "zero-shot-audio-classification". See the list of available models on huggingface.co/models.

( audios: numpy.ndarray | bytes | str | dict **kwargs: typing.Any )

Assign labels to the audio(s) passed as inputs.

Pipelines available for computer vision tasks include the following.

Depth estimation pipeline using any AutoModelForDepthEstimation. This pipeline predicts the depth of an image.

Learn more about the basics of using a pipeline in the pipeline tutorial

This depth estimation pipeline can currently be loaded from pipeline() using the following task identifier: "depth-estimation".

See the list of available models on huggingface.co/models.

( inputs: typing.Union[str, list[str], ForwardRef('Image.Image'), list['Image.Image']] **kwargs: typing.Any )

The pipeline accepts either a single image or a batch of images, which must then be passed as a string. Images in a batch must all be in the same format: all as http links, all as local paths, or all as PIL images.

Predict the depth(s) of the image(s) passed as inputs.

Image classification pipeline using any AutoModelForImageClassification. This pipeline predicts the class of an image.

Learn more about the basics of using a pipeline in the pipeline tutorial

This image classification pipeline can currently be loaded from pipeline() using the following task identifier: "image-classification".

See the list of available models on huggingface.co/models.

( inputs: typing.Union[str, list[str], ForwardRef('Image.Image'), list['Image.Image']] **kwargs: typing.Any )

The pipeline accepts either a single image or a batch of images, which must then be passed as a string. Images in a batch must all be in the same format: all as http links, all as local paths, or all as PIL images.

If this argument is not specified, then it will apply the following functions according to the number of labels:

Assign labels to the image(s) passed as inputs.

Image segmentation pipeline using any AutoModelForXXXSegmentation. This pipeline predicts masks of objects and their classes.

This image segmentation pipeline can currently be loaded from pipeline() using the following task identifier: "image-segmentation".

See the list of available models on huggingface.co/models.

( inputs: typing.Union[str, ForwardRef('Image.Image'), list[str], list['Image.Image']] **kwargs: typing.Any )

The pipeline accepts either a single image or a batch of images. Images in a batch must all be in the same format: all as HTTP(S) links, all as local paths, or all as PIL images.

Perform segmentation (detect masks & classes) in the image(s) passed as inputs.

Image to Image pipeline using any AutoModelForImageToImage. This pipeline generates an image based on a previous image input.

This image to image pipeline can currently be loaded from pipeline() using the following task identifier: "image-to-image".

See the list of available models on huggingface.co/models.

( images: typing.Union[str, list[str], ForwardRef('Image.Image'), list['Image.Image']] **kwargs: typing.Any )

The pipeline accepts either a single image or a batch of images, which must then be passed as a string. Images in a batch must all be in the same format: all as http links, all as local paths, or all as PIL images.

Transform the image(s) passed as inputs.

Keypoint matching pipeline using any AutoModelForKeypointMatching. This pipeline matches keypoints between two images.

( inputs: list[collections.abc.Sequence[typing.Union[ForwardRef('Image.Image'), str]]] | collections.abc.Sequence[typing.Union[ForwardRef('Image.Image'), str]] threshold: float = 0.0 **kwargs: typing.Any ) → Union[list[Match], list[list[Match]]]

The pipeline accepts either a single pair of images or a batch of image pairs, which must then be passed as a string. Images in a batch must all be in the same format: all as http links, all as local paths, or all as PIL images.

Union[list[Match], list[list[Match]]]

A list of matches or a list if a single image pair is provided, or of lists of matches if a batch of image pairs is provided. Each match is a dictionary containing the following keys: keypoint_image_0 (Keypoint): The keypoint in the first image (x, y coordinates). keypoint_image_1 (Keypoint): The keypoint in the second image (x, y coordinates). score (float): The matching score between the two keypoints.

A list of matches or a list if a single image pair is provided, or of lists of matches if a batch of image pairs is provided. Each match is a dictionary containing the following keys:

Find matches between keypoints in two images.

Object detection pipeline using any AutoModelForObjectDetection. This pipeline predicts bounding boxes of objects and their classes.

Learn more about the basics of using a pipeline in the pipeline tutorial

This object detection pipeline can currently be loaded from pipeline() using the following task identifier: "object-detection".

See the list of available models on huggingface.co/models.

The pipeline accepts either a single image or a batch of images. Images in a batch must all be in the same format: all as HTTP(S) links, all as local paths, or all as PIL images.

Detect objects (bounding boxes & classes) in the image(s) passed as inputs.

Video classification pipeline using any AutoModelForVideoClassification. This pipeline predicts the class of a video.

This video classification pipeline can currently be loaded from pipeline() using the following task identifier: "video-classification".

See the list of available models on huggingface.co/models.

( inputs: str | list[str] | None **kwargs )

The pipeline accepts either a single video or a batch of videos, which must then be passed as a string. Videos in a batch must all be in the same format: all as http links or all as local paths.

Assign labels to the video(s) passed as inputs.

Zero shot image classification pipeline using CLIPModel. This pipeline predicts the class of an image when you provide an image and a set of candidate_labels.

Learn more about the basics of using a pipeline in the pipeline tutorial

This image classification pipeline can currently be loaded from pipeline() using the following task identifier: "zero-shot-image-classification".

See the list of available models on huggingface.co/models.

( image: typing.Union[str, list[str], ForwardRef('Image.Image'), list['Image.Image']] candidate_labels: list **kwargs: typing.Any )

Assign labels to the image(s) passed as inputs.

Zero shot object detection pipeline using OwlViTForObjectDetection. This pipeline predicts bounding boxes of objects when you provide an image and a set of candidate_labels.

Learn more about the basics of using a pipeline in the pipeline tutorial

This object detection pipeline can currently be loaded from pipeline() using the following task identifier: "zero-shot-object-detection".

See the list of available models on huggingface.co/models.

( image: typing.Union[str, ForwardRef('Image.Image'), list[dict[str, typing.Any]]] candidate_labels: str | list[str] | None = None **kwargs: typing.Any )

You can use this parameter to send directly a list of images, or a dataset or a generator like so:

Detect objects (bounding boxes & classes) in the image(s) passed as inputs.

Pipelines available for natural language processing tasks include the following.

( model: PreTrainedModel tokenizer: PreTrainedTokenizer | None = None feature_extractor: PreTrainedFeatureExtractor | None = None image_processor: BaseImageProcessor | None = None processor: ProcessorMixin | None = None task: str = '' device: int | torch.device | None = None binary_output: bool = False **kwargs )

( inputs: str | list[str] **kwargs: typing.Any ) → A list or a list of list of dict

A list or a list of list of dict

Each result comes as list of dictionaries with the following keys: sequence (str) — The corresponding input with the mask token prediction. score (float) — The corresponding probability. token (int) — The predicted token id (to replace the masked one). token_str (str) — The predicted token (to replace the masked one).

Each result comes as list of dictionaries with the following keys:

Fill the masked token in the text(s) given as inputs.

( model: PreTrainedModel tokenizer: PythonBackend task: str = '' **kwargs )

Question Answering pipeline using any ModelForQuestionAnswering. See the question answering examples for more information.

Learn more about the basics of using a pipeline in the pipeline tutorial

This question answering pipeline can currently be loaded from pipeline() using the following task identifier: "question-answering".

The models that this pipeline can use are models that have been fine-tuned on a question answering task. See the up-to-date list of available models on huggingface.co/models.

( **kwargs ) → A dict or a list of dict

A dict or a list of dict

Each result comes as a dictionary with the following keys: score (float) — The probability associated to the answer. start (int) — The character start index of the answer (in the tokenized version of the input). end (int) — The character end index of the answer (in the tokenized version of the input). answer (str) — The answer to the question.

Each result comes as a dictionary with the following keys:

Answer the question(s) given as inputs by using the context(s).

( question: str | list[str] context: str | list[str] ) → One or a list of SquadExample

One or a list of SquadExample

The corresponding SquadExample grouping question and context.

The corresponding SquadExample grouping question and context.

QuestionAnsweringPipeline leverages the SquadExample internally. This helper method encapsulate all the logic for converting question(s) and context(s) to SquadExample.

We currently support extractive question answering.

( text: str start: int end: int ) → Dictionary like `{‘answer’

Dictionary like `{‘answer’

str, ‘start’: int, ‘end’: int}`

str, ‘start’: int, ‘end’: int}`

When decoding from token probabilities, this method maps token indexes to actual word in the initial context.

( args_parser = <transformers.pipelines.table_question_answering.TableQuestionAnsweringArgumentHandler object at 0x7f872fcdb6d0> **kwargs )

Table Question Answering pipeline using a ModelForTableQuestionAnswering. This pipeline is only available in PyTorch.

Unless the model you’re using explicitly sets these generation parameters in its configuration files (generation_config.json), the following default values will be used:

Learn more about the basics of using a pipeline in the pipeline tutorial

This tabular question answering pipeline can currently be loaded from pipeline() using the following task identifier: "table-question-answering".

The models that this pipeline can use are models that have been fine-tuned on a tabular question answering task. See the up-to-date list of available models on huggingface.co/models.

( *args **kwargs ) → A dictionary or a list of dictionaries containing results

A dictionary or a list of dictionaries containing results

Each result is a dictionary with the following keys: answer (str) — The answer of the query given the table. If there is an aggregator, the answer will be preceded by AGGREGATOR >. coordinates (list[tuple[int, int]]) — Coordinates of the cells of the answers. cells (list[str]) — List of strings made up of the answer cell values. aggregator (str) — If the model has an aggregator, this returns the aggregator.

Each result is a dictionary with the following keys:

Answers queries according to a table. The pipeline accepts several types of inputs which are detailed below:

The table argument should be a dict or a DataFrame built from that dict, containing the whole table:

This dictionary can be passed in as such, or can be converted to a pandas DataFrame:

Text classification pipeline using any ModelForSequenceClassification. See the sequence classification examples for more information.

Learn more about the basics of using a pipeline in the pipeline tutorial

This text classification pipeline can currently be loaded from pipeline() using the following task identifier: "sentiment-analysis" (for classifying sequences according to positive or negative sentiments).

If multiple classification labels are available (model.config.num_labels >= 2), the pipeline will run a softmax over the results. If there is a single label, the pipeline will run a sigmoid over the result. In case of regression tasks (model.config.problem_type == "regression"), will not apply any function on the output.

The models that this pipeline can use are models that have been fine-tuned on a sequence classification task. See the up-to-date list of available models on huggingface.co/models.

( inputs: str | list[str] | dict[str, str] | list[dict[str, str]] **kwargs: typing.Any ) → A list of dict

If this argument is not specified, then it will apply the following functions according to the number of labels:

Each result comes as list of dictionaries with the following keys: label (str) — The label predicted. score (float) — The corresponding probability. If top_k is used, one such dictionary is returned per label.

Each result comes as list of dictionaries with the following keys:

If top_k is used, one such dictionary is returned per label.

Classify the text(s) given as inputs.

Language generation pipeline using any ModelWithLMHead or ModelForCausalLM. This pipeline predicts the words that will follow a specified text prompt. When the underlying model is a conversational model, it can also accept one or more chats, in which case the pipeline will operate in chat mode and will continue the chat(s) by adding its response(s). Each chat takes the form of a list of dicts, where each dict contains “role” and “content” keys.

Unless the model you’re using explicitly sets these generation parameters in its configuration files (generation_config.json), the following default values will be used:

Learn more about the basics of using a pipeline in the pipeline tutorial. You can pass text generation parameters to this pipeline to control stopping criteria, decoding strategy, and more. Learn more about text generation parameters in Text generation strategies and Text generation.

This language generation pipeline can currently be loaded from pipeline() using the following task identifier: "text-generation".

The models that this pipeline can use are models that have been trained with an autoregressive language modeling objective. See the list of available text completion models and the list of conversational models on [huggingface.co/models].

( text_inputs **kwargs ) → A list or a list of lists of dict

A list or a list of lists of dict

Returns one of the following dictionaries (cannot return a combination of both generated_text and generated_token_ids): generated_text (str, present when return_text=True) — The generated text. generated_token_ids (torch.Tensor, present when return_tensors=True) — The token ids of the generated text.

Returns one of the following dictionaries (cannot return a combination of both generated_text and generated_token_ids):

Complete the prompt(s) given as inputs.

( args_parser = <transformers.pipelines.token_classification.TokenClassificationArgumentHandler object at 0x7f872fb62950> **kwargs )

Named Entity Recognition pipeline using any ModelForTokenClassification. See the named entity recognition examples for more information.

Learn more about the basics of using a pipeline in the pipeline tutorial

This token recognition pipeline can currently be loaded from pipeline() using the following task identifier: "ner" (for predicting the classes of tokens in a sequence: person, organisation, location or miscellaneous).

The models that this pipeline can use are models that have been fine-tuned on a token classification task. See the up-to-date list of available models on huggingface.co/models.

( inputs: str | list[str] **kwargs: typing.Any ) → A list or a list of list of dict

A list or a list of list of dict

Each result comes as a list of dictionaries (one for each token in the corresponding input, or each entity if this pipeline was instantiated with an aggregation_strategy) with the following keys: word (str) — The token/word classified. This is obtained by decoding the selected tokens. If you want to have the exact string in the original sentence, use start and end. score (float) — The corresponding probability for entity. entity (str) — The entity predicted for that token/word (it is named entity_group when aggregation_strategy is not "none". index (int, only present when aggregation_strategy="none") — The index of the corresponding token in the sentence. start (int, optional) — The index of the start of the corresponding entity in the sentence. Only exists if the offsets are available within the tokenizer end (int, optional) — The index of the end of the corresponding entity in the sentence. Only exists if the offsets are available within the tokenizer

Each result comes as a list of dictionaries (one for each token in the corresponding input, or each entity if this pipeline was instantiated with an aggregation_strategy) with the following keys:

Classify each token of the text(s) given as inputs.

( entities: list aggregation_strategy: AggregationStrategy )

Override tokens from a given word that disagree to force agreement on word boundaries.

Example: micro|soft| com|pany| B-ENT I-NAME I-ENT I-ENT will be rewritten with first strategy as microsoft| company| B-ENT I-ENT

( sentence: str input_ids: ndarray scores: ndarray offset_mapping: list[tuple[int, int]] | None special_tokens_mask: ndarray aggregation_strategy: AggregationStrategy word_ids: list[int | None] | None = None word_to_chars_map: list[tuple[int, int]] | None = None )

Fuse various numpy arrays into dicts with all the information needed for aggregation

Find and group together the adjacent tokens with the same entity predicted.

Group together the adjacent tokens with the same entity predicted.

( args_parser = <transformers.pipelines.zero_shot_classification.ZeroShotClassificationArgumentHandler object at 0x7f872fb80ac0> **kwargs )

NLI-based zero-shot classification pipeline using a ModelForSequenceClassification trained on NLI (natural language inference) tasks. Equivalent of text-classification pipelines, but these models don’t require a hardcoded number of potential classes, they can be chosen at runtime. It usually means it’s slower but it is much more flexible.

Any combination of sequences and labels can be passed and each combination will be posed as a premise/hypothesis pair and passed to the pretrained model. Then, the logit for entailment is taken as the logit for the candidate label being valid. Any NLI model can be used, but the id of the entailment label must be included in the model config’s :attr:~transformers.PreTrainedConfig.label2id.

Learn more about the basics of using a pipeline in the pipeline tutorial

This NLI pipeline can currently be loaded from pipeline() using the following task identifier: "zero-shot-classification".

The models that this pipeline can use are models that have been fine-tuned on an NLI task. See the up-to-date list of available models on huggingface.co/models.

( sequences: str | list[str] *args **kwargs ) → A dict or a list of dict

A dict or a list of dict

Each result comes as a dictionary with the following keys: sequence (str) — The sequence for which this is the output. labels (list[str]) — The labels sorted by order of likelihood. scores (list[float]) — The probabilities for each of the labels.

Each result comes as a dictionary with the following keys:

Classify the sequence(s) given as inputs. See the ZeroShotClassificationPipeline documentation for more information.

Pipelines available for multimodal tasks include the following.

Document Question Answering pipeline using any AutoModelForDocumentQuestionAnswering. The inputs/outputs are similar to the (extractive) question answering pipeline; however, the pipeline takes an image (and optional OCR’d words/boxes) as input instead of text context.

Unless the model you’re using explicitly sets these generation parameters in its configuration files (generation_config.json), the following default values will be used:

Learn more about the basics of using a pipeline in the pipeline tutorial

This document question answering pipeline can currently be loaded from pipeline() using the following task identifier: "document-question-answering".

The models that this pipeline can use are models that have been fine-tuned on a document question answering task. See the up-to-date list of available models on huggingface.co/models.

( image: typing.Union[ForwardRef('Image.Image'), str, list[dict[str, typing.Any]]] question: str | None = None word_boxes: tuple[str, list[float]] | None = None **kwargs: typing.Any ) → A dict or a list of dict

The pipeline accepts either a single image or a batch of images. If given a single image, it can be broadcasted to multiple questions.

A dict or a list of dict

Each result comes as a dictionary with the following keys: score (float) — The probability associated to the answer. start (int) — The start word index of the answer (in the OCR’d version of the input or provided word_boxes). end (int) — The end word index of the answer (in the OCR’d version of the input or provided word_boxes). answer (str) — The answer to the question. words (list[int]) — The index of each word/box pair that is in the answer

Each result comes as a dictionary with the following keys:

Answer the question(s) given as inputs by using the document(s). A document is defined as an image and an optional list of (word, box) tuples which represent the text in the document. If the word_boxes are not provided, it will use the Tesseract OCR engine (if available) to extract the words and boxes automatically for LayoutLM-like models which require them as input. For Donut, no OCR is run.

You can invoke the pipeline several ways:

( model: PreTrainedModel tokenizer: PreTrainedTokenizer | None = None feature_extractor: PreTrainedFeatureExtractor | None = None image_processor: BaseImageProcessor | None = None processor: ProcessorMixin | None = None task: str = '' device: int | torch.device | None = None binary_output: bool = False **kwargs )

Feature extraction pipeline uses no model head. This pipeline extracts the hidden states from the base transformer, which can be used as features in downstream tasks.

Learn more about the basics of using a pipeline in the pipeline tutorial

This feature extraction pipeline can currently be loaded from pipeline() using the task identifier: "feature-extraction".

All models may be used for this pipeline. See a list of all models, including community-contributed models on huggingface.co/models.

( *args: str | list[str] **kwargs: typing.Any ) → A nested list of float

A nested list of float

The features computed by the model.

The features computed by the model.

Extract the features of the input(s) text.

( model: PreTrainedModel tokenizer: PreTrainedTokenizer | None = None feature_extractor: PreTrainedFeatureExtractor | None = None image_processor: BaseImageProcessor | None = None processor: ProcessorMixin | None = None task: str = '' device: int | torch.device | None = None binary_output: bool = False **kwargs )

Image feature extraction pipeline uses no model head. This pipeline extracts the hidden states from the base transformer, which can be used as features in downstream tasks.

Learn more about the basics of using a pipeline in the pipeline tutorial

This image feature extraction pipeline can currently be loaded from pipeline() using the task identifier: "image-feature-extraction".

All vision models may be used for this pipeline. See a list of all models, including community-contributed models on huggingface.co/models.

( *args: typing.Union[str, ForwardRef('Image.Image'), list['Image.Image'], list[str]] **kwargs: typing.Any ) → A nested list of float

The pipeline accepts either a single image or a batch of images, which must then be passed as a string. Images in a batch must all be in the same format: all as http links, all as local paths, or all as PIL images.

A nested list of float

The features computed by the model.

The features computed by the model.

Extract the features of the input(s).

Image-text-to-text pipeline using an AutoModelForImageTextToText. This pipeline generates text given an image and text. When the underlying model is a conversational model, it can also accept one or more chats, in which case the pipeline will operate in chat mode and will continue the chat(s) by adding its response(s). Each chat takes the form of a list of dicts, where each dict contains “role” and “content” keys.

Unless the model you’re using explicitly sets these generation parameters in its configuration files (generation_config.json), the following default values will be used:

Learn more about the basics of using a pipeline in the pipeline tutorial

This image-text to text pipeline can currently be loaded from pipeline() using the following task identifier: “image-text-to-text”.

See the list of available models on huggingface.co/models.

( images: typing.Union[str, list[str], list[list[str]], ForwardRef('Image.Image'), list['Image.Image'], list[list['Image.Image']], list[dict], NoneType] = None text: str | list[str] | list[dict] | None = None **kwargs ) → A list or a list of list of dict

The pipeline accepts either a single image or a batch of images. Finally, this pipeline also supports the chat format (see text) containing images and text in this argument.

A list or a list of list of dict

Each result comes as a dictionary with the following key (cannot return a combination of both generated_text and generated_token_ids): generated_text (str, present when return_text=True) — The generated text. generated_token_ids (torch.Tensor, present when return_tensors=True) — The token ids of the generated text. input_text (str) — The input text.

Each result comes as a dictionary with the following key (cannot return a combination of both generated_text and generated_token_ids):

Generate a text given text and the image(s) passed as inputs.

Multimodal Generation pipeline using an AutoModelForMultimodalLM. This pipeline generates text given any combination of multimodal data and text.When the underlying model is a conversational model, it can also accept one or more chats, in which case the pipeline will operate in chat mode and will continue the chat(s) by adding its response(s). Each chat takes the form of a list of dicts, where each dict contains “role” and “content” keys.

Unless the model you’re using explicitly sets these generation parameters in its configuration files (generation_config.json), the following default values will be used:

Learn more about the basics of using a pipeline in the pipeline tutorial

This multimodal pipeline can currently be loaded from pipeline() using the following task identifier: “any-to-any”.

See the list of available models on huggingface.co/models.

( text: str | list[str] | list[dict] images: typing.Union[str, list[str], list[list[str]], ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor'], NoneType] = None videos: typing.Union[str, list[str], list['PIL.Image.Image'], numpy.ndarray, ForwardRef('torch.Tensor'), list[numpy.ndarray], list['torch.Tensor'], list[list['PIL.Image.Image']], list[list[numpy.ndarray]], list[list['torch.Tensor']], transformers.video_utils.URL, list[transformers.video_utils.URL], list[list[transformers.video_utils.URL]], transformers.video_utils.Path, list[transformers.video_utils.Path], list[list[transformers.video_utils.Path]], NoneType] = None audio: typing.Union[str, list[str], numpy.ndarray, ForwardRef('torch.Tensor'), collections.abc.Sequence[numpy.ndarray], collections.abc.Sequence['torch.Tensor'], NoneType] = None **kwargs ) → A list or a list of list of dict

The pipeline accepts either a single image or a batch of images. Finally, this pipeline also supports the chat format (see text) containing images and text in this argument.

The pipeline accepts either a single video or a batch of videos. Finally, this pipeline also supports the chat format (see text) containing videos and text in this argument.

The pipeline accepts either a single audios or a batch of audios. Finally, this pipeline also supports the chat format (see text) containing audios and text in this argument.

A list or a list of list of dict

Each result comes as a dictionary with the following key (cannot return a combination of both generated_text and generated_token_ids): generated_text (str, present when return_text=True and `generation_mode=“text”) — The generated text. generated_audio (np.ndarray, present when `generation_mode=“audio”) — The generated audio. generated_image (PIL.Image.Image, present when `generation_mode=“image”) — The generated image. generated_token_ids (torch.Tensor, present when return_tensors=True and `generation_mode=“text”) — The token ids of the generated text. input_text (str) — The input text.

Each result comes as a dictionary with the following key (cannot return a combination of both generated_text and generated_token_ids):

Generate a text given text and optionally multimodal data passed as inputs.

Automatic mask generation for images using SamForMaskGeneration. This pipeline predicts binary masks for an image, given an image. It is a ChunkPipeline because you can separate the points in a mini-batch in order to avoid OOM issues. Use the points_per_batch argument to control the number of points that will be processed at the same time. Default is 64.

The pipeline works in 3 steps:

preprocess: A grid of 1024 points evenly separated is generated along with bounding boxes and point labels. For more details on how the points and bounding boxes are created, check the _generate_crop_boxes function. The image is also preprocessed using the image_processor. This function yields a minibatch of points_per_batch.

forward: feeds the outputs of preprocess to the model. The image embedding is computed only once. Calls both self.model.get_image_embeddings and makes sure that the gradients are not computed, and the tensors and models are on the same device.

postprocess: The most important part of the automatic mask generation happens here. Three steps are induced:

Learn more about the basics of using a pipeline in the pipeline tutorial

This segmentation pipeline can currently be loaded from pipeline() using the following task identifier: "mask-generation".

See the list of available models on huggingface.co/models.

( image: typing.Union[str, ForwardRef('Image.Image'), list[str], list['Image.Image']] *args: typing.Any **kwargs: typing.Any ) → Dict

A dictionary with the following keys: mask (PIL.Image) — A binary mask of the detected object as a PIL Image of shape (width, height) of the original image. Returns a mask filled with zeros if no object is found. score (optional float) — Optionally, when the model is capable of estimating a confidence of the “object” described by the label and the mask.

A dictionary with the following keys:

Generates binary segmentation masks

Visual Question Answering pipeline using a AutoModelForVisualQuestionAnswering. This pipeline is currently only available in PyTorch.

Unless the model you’re using explicitly sets these generation parameters in its configuration files (generation_config.json), the following default values will be used:

Learn more about the basics of using a pipeline in the pipeline tutorial

This visual question answering pipeline can currently be loaded from pipeline() using the following task identifiers: "visual-question-answering", "vqa".

The models that this pipeline can use are models that have been fine-tuned on a visual question answering task. See the up-to-date list of available models on huggingface.co/models.

( image: typing.Union[ForwardRef('Image.Image'), str, list['Image.Image'], list[str], ForwardRef('KeyDataset')] question: str | list[str] | None = None **kwargs ) → A dictionary or a list of dictionaries containing the result. The dictionaries contain the following keys

The pipeline accepts either a single image or a batch of images. If given a single image, it can be broadcasted to multiple questions. For dataset: the passed in dataset must be of type transformers.pipelines.pt_utils.KeyDataset Example:

A dictionary or a list of dictionaries containing the result. The dictionaries contain the following keys

label (str) — The label identified by the model. score (int) — The score attributed by the model for that label.

Answers open-ended questions about images. The pipeline accepts several types of inputs which are detailed below:

( model: PreTrainedModel tokenizer: PreTrainedTokenizer | None = None feature_extractor: PreTrainedFeatureExtractor | None = None image_processor: BaseImageProcessor | None = None processor: ProcessorMixin | None = None task: str = '' device: int | torch.device | None = None binary_output: bool = False **kwargs )

The Pipeline class is the class from which all pipelines inherit. Refer to this class for methods shared across different pipelines.

Base class implementing pipelined operations. Pipeline workflow is defined as a sequence of the following operations:

Input -> Tokenization -> Model Inference -> Post-Processing (task dependent) -> Output

Pipeline supports running on CPU or GPU through the device argument (see below).

Some pipeline, like for instance FeatureExtractionPipeline ('feature-extraction') output large tensor object as nested-lists. In order to avoid dumping such large structure as textual data we provide the binary_output constructor argument. If set to True, the output will be stored in the pickle format.

( supported_models: list[str] | dict )

Check if the model class is in supported by the pipeline.

Context Manager allowing tensor allocation on the user-specified device.

( **inputs ) → dict[str, torch.Tensor]

dict[str, torch.Tensor]

The same as inputs but on the proper device.

The same as inputs but on the proper device.

Ensure PyTorch tensors are on the specified device.

( model_outputs: ModelOutput **postprocess_parameters: dict )

Postprocess will receive the raw outputs of the _forward method, generally tensors, and reformat them into something more friendly. Generally it will output a list or a dict or results (containing just strings and numbers).

Scikit / Keras interface to transformers’ pipelines. This method will forward to call().

( input_: Any **preprocess_parameters: dict )

Preprocess will take the input_ of a specific pipeline and return a dictionary of everything necessary for _forward to run properly. It should contain at least one tensor, but might have arbitrary other items.

( repo_id: str commit_message: str | None = None commit_description: str | None = None private: bool | None = None token: bool | str | None = None revision: str | None = None create_pr: bool = False max_shard_size: int | str | None = '50GB' tags: list[str] | None = None )

Upload the pipeline file to the 🤗 Model Hub.

( save_directory: str | os.PathLike **kwargs: Any )

Save the pipeline’s model and tokenizer.

Scikit / Keras interface to transformers’ pipelines. This method will forward to call().

---

## Transformers

**URL:** https://huggingface.co/docs/transformers/v5.0.0/en/main_classes/kernels

**Contents:**
- Transformers
- Kernels
  - KernelConfig
  - class transformers.KernelConfig
    - create_compatible_mapping
    - sanitize_kernel_mapping

Transformers documentation

and get access to the augmented documentation experience

This page documents the kernels configuration utilities.

( kernel_mapping = {} use_local_kernel = False )

Kernel configuration class. This class is used to configure the kernel mapping for a model.

( model compile = False )

Transforms a simple kernel_mapping of the form: { “RMSNorm”: “kernels-community/layer_norm:LlamaRMSNorm”, … },

{ “RMSNorm”: “/home/user/liger_kernels:LigerRMSNorm”, … },

into a nested mapping:

{ “RMSNorm”: { “cuda”: { Mode.INFERENCE: LayerRepository( repo_id=“kernels-community/layer_norm”, layer_name=“LlamaRMSNorm”, ) } } }

{ “RMSNorm”: { “cuda”: { Mode.INFERENCE: LocalLayerRepository( repo_path=Path(“/home/user/liger_kernels”), package_name=“liger_kernels”, layer_name=“LigerRMSNorm”, ) } } }

that’s compatible with the kernels library.

The device is inferred from the model’s parameters if not provided. The Mode is inferred from the model’s training state.

ValueError — If a layer_name is not registered in the model, if a device is not supported, or if a repo_name is not a valid ‘org/repo:layer_name’ string.

Validates the kernel_mapping to ensure that:

---

## Transformers

**URL:** https://huggingface.co/docs/transformers/v5.0.0/en/internal/trainer_utils

**Contents:**
- Transformers
- Utilities for Trainer
- Utilities
  - class transformers.EvalPrediction
  - class transformers.IntervalStrategy
    - transformers.enable_full_determinism
    - transformers.set_seed
    - transformers.torch_distributed_zero_first
- Callbacks internals
  - class transformers.trainer_callback.CallbackHandler

Transformers documentation

Utilities for Trainer

and get access to the augmented documentation experience

This page lists all the utility functions used by Trainer.

Most of those are only useful if you are studying the code of the Trainer in the library.

( predictions: numpy.ndarray | tuple[numpy.ndarray] label_ids: numpy.ndarray | tuple[numpy.ndarray] inputs: numpy.ndarray | tuple[numpy.ndarray] | None = None losses: numpy.ndarray | tuple[numpy.ndarray] | None = None )

Evaluation output (always contains labels), to be used to compute metrics.

( value names = None module = None qualname = None type = None start = 1 )

( seed: int warn_only: bool = False )

Helper function for reproducible behavior during distributed training. See https://pytorch.org/docs/stable/notes/randomness.html for pytorch

( seed: int deterministic: bool = False )

Helper function for reproducible behavior to set the seed in random, numpy, torch (if installed).

Decorator to make all processes in distributed training wait for each local_master to do something.

( callbacks model processing_class optimizer lr_scheduler )

Internal class that just calls the list of callbacks in order.

( dataclass_types: typing.Union[transformers.hf_argparser.DataClassType, collections.abc.Iterable[transformers.hf_argparser.DataClassType], NoneType] = None **kwargs )

This subclass of argparse.ArgumentParser uses type hints on dataclasses to generate arguments.

The class is designed to play well with the native argparse. In particular, you can add more (non-dataclass backed) arguments to the parser after initialization and you’ll get the output back after parsing as an additional namespace. Optional: To create sub argument groups use the _argument_group_name attribute in the dataclass.

( args = None return_remaining_strings = False look_for_args_file = True args_filename = None args_file_flag = None ) → Tuple consisting of

the dataclass instances in the same order as they were passed to the initializer.abspath if applicable, an additional namespace for more (non-dataclass backed) arguments added to the parser after initialization. The potential list of remaining argument strings. (same as argparse.ArgumentParser.parse_known_args)

Parse command-line args into instances of the specified dataclass types.

This relies on argparse’s ArgumentParser.parse_known_args. See the doc at: docs.python.org/3/library/argparse.html#argparse.ArgumentParser.parse_args

( args: dict allow_extra_keys: bool = False ) → Tuple consisting of

the dataclass instances in the same order as they were passed to the initializer.

Alternative helper method that does not use argparse at all, instead uses a dict and populating the dataclass types.

( json_file: str | os.PathLike allow_extra_keys: bool = False ) → Tuple consisting of

the dataclass instances in the same order as they were passed to the initializer.

Alternative helper method that does not use argparse at all, instead loading a json file and populating the dataclass types.

( yaml_file: str | os.PathLike allow_extra_keys: bool = False ) → Tuple consisting of

the dataclass instances in the same order as they were passed to the initializer.

Alternative helper method that does not use argparse at all, instead loading a yaml file and populating the dataclass types.

( model max_frames_to_save = 21 trace_batch_nums = [] abort_after_batch_num = None )

This debug class helps detect and understand where the model starts getting very large or very small, and more importantly nan or inf weight and activation elements.

There are 2 working modes:

Mode 1: Underflow/overflow detection

To activate the underflow/overflow detection, initialize the object with the model :

then run the training as normal and if nan or inf gets detected in at least one of the weight, input or output elements this module will throw an exception and will print max_frames_to_save frames that lead to this event, each frame reporting

For example, here is the header and the last few frames in detection report for google/mt5-small run in fp16

You can see here, that T5DenseGatedGeluDense.forward resulted in output activations, whose absolute max value was around 62.7K, which is very close to fp16’s top limit of 64K. In the next frame we have Dropout which renormalizes the weights, after it zeroed some of the elements, which pushes the absolute max value to more than 64K, and we get an overflow.

As you can see it’s the previous frames that we need to look into when the numbers start going into very large for fp16 numbers.

The tracking is done in a forward hook, which gets invoked immediately after forward has completed.

By default the last 21 frames are printed. You can change the default to adjust for your needs. For example :

To validate that you have set up this debugging feature correctly, and you intend to use it in a training that may take hours to complete, first run it with normal tracing enabled for one of a few batches as explained in the next section.

Mode 2. Specific batch absolute min/max tracing without detection

The second work mode is per-batch tracing with the underflow/overflow detection feature turned off.

Let’s say you want to watch the absolute min and max values for all the ingredients of each forward call of a

given batch, and only do that for batches 1 and 3. Then you instantiate this class as :

And now full batches 1 and 3 will be traced using the same format as explained above. Batches are 0-indexed.

This is helpful if you know that the program starts misbehaving after a certain batch number, so you can fast-forward right to that area.

You can also specify the batch number after which to stop the training, with :

This feature is mainly useful in the tracing mode, but you can use it for any mode.

As this module measures absolute min/`max of each weight of the model on every forward it’ll slow the training down. Therefore remember to turn it off once the debugging needs have been met.

---

## 

**URL:** https://huggingface.co/docs/transformers/v5.0.0/en/task_summary

---

## Transformers

**URL:** https://huggingface.co/docs/transformers/v5.0.0/en/main_classes/trainer

**Contents:**
- Transformers
- Trainer
- Trainer
  - class transformers.Trainer
    - add_callback
    - autocast_smart_context_manager
    - compute_loss
    - compute_loss_context_manager
    - create_model_card
    - create_optimizer

Transformers documentation

and get access to the augmented documentation experience

The Trainer class provides an API for feature-complete training in PyTorch, and it supports distributed training on multiple GPUs/TPUs, mixed precision for NVIDIA GPUs, AMD GPUs, and torch.amp for PyTorch. Trainer goes hand-in-hand with the TrainingArguments class, which offers a wide range of options to customize how a model is trained. Together, these two classes provide a complete training API.

Seq2SeqTrainer and Seq2SeqTrainingArguments inherit from the Trainer and TrainingArguments classes and they’re adapted for training models for sequence-to-sequence tasks such as summarization or translation.

The Trainer class is optimized for 🤗 Transformers models and can have surprising behaviors when used with other models. When using it with your own model, make sure:

( model: transformers.modeling_utils.PreTrainedModel | torch.nn.modules.module.Module | None = None args: transformers.training_args.TrainingArguments | None = None data_collator: collections.abc.Callable[[list[typing.Any]], dict[str, typing.Any]] | None = None train_dataset: typing.Union[torch.utils.data.dataset.Dataset, torch.utils.data.dataset.IterableDataset, ForwardRef('datasets.Dataset'), NoneType] = None eval_dataset: typing.Union[torch.utils.data.dataset.Dataset, dict[str, torch.utils.data.dataset.Dataset], ForwardRef('datasets.Dataset'), NoneType] = None processing_class: transformers.tokenization_utils_base.PreTrainedTokenizerBase | transformers.image_processing_utils.BaseImageProcessor | transformers.feature_extraction_utils.FeatureExtractionMixin | transformers.processing_utils.ProcessorMixin | None = None model_init: collections.abc.Callable[..., transformers.modeling_utils.PreTrainedModel] | None = None compute_loss_func: collections.abc.Callable | None = None compute_metrics: collections.abc.Callable[[transformers.trainer_utils.EvalPrediction], dict] | None = None callbacks: list[transformers.trainer_callback.TrainerCallback] | None = None optimizers: tuple = (None, None) optimizer_cls_and_kwargs: tuple[type[torch.optim.optimizer.Optimizer], dict[str, typing.Any]] | None = None preprocess_logits_for_metrics: collections.abc.Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None )

Trainer is optimized to work with the PreTrainedModel provided by the library. You can still use your own models defined as torch.nn.Module as long as they work the same way as the 🤗 Transformers models.

Note that if it’s a torch.utils.data.IterableDataset with some randomization and you are training in a distributed fashion, your iterable dataset should either use a internal attribute generator that is a torch.Generator for the randomization that must be identical on all processes (and the Trainer will manually set the seed of this generator at each epoch) or have a set_epoch() method that internally sets the seed of the RNGs used.

The function may have zero argument, or a single one containing the optuna/Ray Tune trial object, to be able to choose different architectures according to hyper parameters (such as layer count, sizes of inner layers, dropout probabilities etc).

If you want to remove one of the default callbacks used, use the Trainer.remove_callback() method.

Unlike optimizers, this argument avoids the need to place model parameters on the correct devices before initializing the Trainer.

Note that the labels (second parameter) will be None if the dataset does not have them.

Trainer is a simple but feature-complete training and eval loop for PyTorch, optimized for 🤗 Transformers.

Important attributes:

Add a callback to the current list of TrainerCallback.

( cache_enabled: bool | None = True )

A helper wrapper that creates an appropriate context manager for autocast while feeding it the desired arguments, depending on the situation. We rely on accelerate for autocast, hence we do nothing here.

( model: Module inputs: dict return_outputs: bool = False num_items_in_batch: torch.Tensor | None = None )

How the loss is computed by Trainer. By default, all models return the loss in the first element.

Subclass and override for custom behavior. If you are not using num_items_in_batch when computing your loss, make sure to overwrite self.model_accepts_loss_kwargs to False. Otherwise, the loss calculating might be slightly inaccurate when performing gradient accumulation.

A helper wrapper to group together context managers.

( language: str | None = None license: str | None = None tags: str | list[str] | None = None model_name: str | None = None finetuned_from: str | None = None tasks: str | list[str] | None = None dataset_tags: str | list[str] | None = None dataset: str | list[str] | None = None dataset_args: str | list[str] | None = None )

Creates a draft of a model card using the information available to the Trainer.

We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the Trainer’s init through optimizers, or subclass and override this method in a subclass.

( num_training_steps: int )

Setup the optimizer and the learning rate scheduler.

We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the Trainer’s init through optimizers, or subclass and override this method (or create_optimizer and/or create_scheduler) in a subclass.

( num_training_steps: int optimizer: Optimizer = None )

Setup the scheduler. The optimizer of the trainer must have been set up either before this method is called or passed as an argument.

( eval_dataset: torch.utils.data.dataset.Dataset | dict[str, torch.utils.data.dataset.Dataset] | None = None ignore_keys: list[str] | None = None metric_key_prefix: str = 'eval' )

If you pass a dictionary with names of datasets as keys and datasets as values, evaluate will run separate evaluations on each dataset. This can be useful to monitor how training affects other datasets or simply to get a more fine-grained evaluation. When used with load_best_model_at_end, make sure metric_for_best_model references exactly one of the datasets. If you, for example, pass in {"data1": data1, "data2": data2} for two datasets data1 and data2, you could specify metric_for_best_model="eval_data1_loss" for using the loss on data1 and metric_for_best_model="eval_data2_loss" for the loss on data2.

Run evaluation and returns metrics.

The calling script will be responsible for providing a method to compute metrics, as they are task-dependent (pass it to the init compute_metrics argument).

You can also subclass and override this method to inject custom behavior.

( dataloader: DataLoader description: str prediction_loss_only: bool | None = None ignore_keys: list[str] | None = None metric_key_prefix: str = 'eval' )

Prediction/evaluation loop, shared by Trainer.evaluate() and Trainer.predict().

Works both with or without labels.

( inputs: dict ) → int

The number of floating-point operations.

The number of floating-point operations.

For models that inherit from PreTrainedModel, uses that method to compute the number of floating point operations for every backward + forward pass. If using another model, either implement such a method in the model or subclass and override this method.

( epoch_iterator: Iterator num_batches: int device: device )

Collects a specified number of batches from the epoch iterator and optionally counts the number of items in the batches to properly scale the loss.

Get the context parallel size

Get all parameter names that weight decay will be applied to.

This function filters out parameters in two ways:

( eval_dataset: str | torch.utils.data.dataset.Dataset | None = None )

Returns the evaluation ~torch.utils.data.DataLoader.

Subclass and override this method if you want to inject some custom behavior.

Returns the learning rate of each parameter from self.optimizer.

Get the number of trainable parameters.

( args: TrainingArguments model: transformers.modeling_utils.PreTrainedModel | None = None )

Returns the optimizer class and optimizer parameters based on the training arguments.

( param: str | torch.nn.parameter.Parameter | None = None )

Returns optimizer group for a parameter if given, else returns all optimizer groups for params.

Get the sequence parallel size

( test_dataset: Dataset )

Returns the test ~torch.utils.data.DataLoader.

Subclass and override this method if you want to inject some custom behavior.

Calculates total batch size (micro_batch grad_accum dp_world_size).

Accounts for all parallelism dimensions: TP, CP, and SP.

Formula: dp_world_size = world_size // (tp_size cp_size sp_size)

All dimensions are separate and multiplicative: world_size = dp_size tp_size cp_size * sp_size

Get the tensor parallel size from either the model or DeepSpeed config.

Returns the training ~torch.utils.data.DataLoader.

Will use no sampler if train_dataset does not implement __len__, a random sampler (adapted to distributed training if necessary) otherwise.

Subclass and override this method if you want to inject some custom behavior.

( hp_space: collections.abc.Callable[['optuna.Trial'], dict[str, float]] | None = None compute_objective: collections.abc.Callable[[dict[str, float]], float] | None = None n_trials: int = 20 direction: str | list[str] = 'minimize' backend: typing.Union[ForwardRef('str'), transformers.trainer_utils.HPSearchBackend, NoneType] = None hp_name: collections.abc.Callable[['optuna.Trial'], str] | None = None **kwargs ) → [trainer_utils.BestRun or list[trainer_utils.BestRun]]

[trainer_utils.BestRun or list[trainer_utils.BestRun]]

All the information about the best run or best runs for multi-objective optimization. Experiment summary can be found in run_summary attribute for Ray backend.

All the information about the best run or best runs for multi-objective optimization. Experiment summary can be found in run_summary attribute for Ray backend.

Launch an hyperparameter search using optuna or Ray Tune. The optimized quantity is determined by compute_objective, which defaults to a function returning the evaluation loss when no metric is provided, the sum of all metrics otherwise.

To use this method, you need to have provided a model_init when initializing your Trainer: we need to reinitialize the model at each new run. This is incompatible with the optimizers argument, so you need to subclass Trainer and override the method create_optimizer_and_scheduler() for custom optimizer/scheduler.

( token: str | None = None )

Initializes a git repo in self.args.hub_model_id.

Whether or not this process is the local (e.g., on one machine if training in a distributed fashion on several machines) main process.

Whether or not this process is the global main process (when training in a distributed fashion on several machines, this is only going to be True for one process).

( logs: dict start_time: float | None = None )

Log logs on the various objects watching training.

Subclass and override this method to inject custom behavior.

Log metrics in a specially formatted way.

Under distributed environment this is done only for a process with rank 0.

Notes on memory reports:

In order to get memory usage report you need to install psutil. You can do that with pip install psutil.

Now when this method is run, you will see a report that will include:

Understanding the reports:

The reporting happens only for process of rank 0 and gpu 0 (if there is a gpu). Typically this is enough since the main process does the bulk of work, but it could be not quite so if model parallel is used and then other GPUs may use a different amount of gpu memory. This is also not the same under DataParallel where gpu0 may require much more memory than the rest since it stores the gradient and optimizer states for all participating GPUs. Perhaps in the future these reports will evolve to measure those too.

The CPU RAM metric measures RSS (Resident Set Size) includes both the memory which is unique to the process and the memory shared with other processes. It is important to note that it does not include swapped out memory, so the reports could be imprecise.

The CPU peak memory is measured using a sampling thread. Due to python’s GIL it may miss some of the peak memory if that thread didn’t get a chance to run when the highest memory was used. Therefore this report can be less than reality. Using tracemalloc would have reported the exact peak memory, but it doesn’t report memory allocations outside of python. So if some C++ CUDA extension allocated its own memory it won’t be reported. And therefore it was dropped in favor of the memory sampling approach, which reads the current process memory usage.

The GPU allocated and peak memory reporting is done with torch.cuda.memory_allocated() and torch.cuda.max_memory_allocated(). This metric reports only “deltas” for pytorch-specific allocations, as torch.cuda memory management system doesn’t track any memory allocated outside of pytorch. For example, the very first cuda call typically loads CUDA kernels, which may take from 0.5 to 2GB of GPU memory.

Note that this tracker doesn’t account for memory allocations outside of Trainer’s __init__, train, evaluate and predict calls.

Because evaluation calls may happen during train, we can’t handle nested invocations because torch.cuda.max_memory_allocated is a single counter, so if it gets reset by a nested eval call, train’s tracker will report incorrect info. If this pytorch issue gets resolved it will be possible to change this class to be re-entrant. Until then we will only track the outer level of train, evaluate and predict methods. Which means that if eval is called during train, it’s the latter that will account for its memory usage and that of the former.

This also means that if any other tool that is used along the Trainer calls torch.cuda.reset_peak_memory_stats, the gpu peak memory stats could be invalid. And the Trainer will disrupt the normal behavior of any such tools that rely on calling torch.cuda.reset_peak_memory_stats themselves.

For best performance you may want to consider turning the memory profiling off for production runs.

( metrics: dict ) → metrics (dict[str, float])

metrics (dict[str, float])

The reformatted metrics

The reformatted metrics

Reformat Trainer metrics values to a human-readable format.

( dataloader: DataLoader )

Helper to get number of samples in a ~torch.utils.data.DataLoader by accessing its dataset. When dataloader.dataset does not exist or has no length, estimates as best it can

( train_dl: DataLoader max_steps: int | None = None )

Helper to get number of tokens in a ~torch.utils.data.DataLoader by enumerating dataloader.

( callback ) → TrainerCallback

The callback removed, if found.

The callback removed, if found.

Remove a callback from the current list of TrainerCallback and returns it.

If the callback is not found, returns None (and no error is raised).

( test_dataset: Dataset ignore_keys: list[str] | None = None metric_key_prefix: str = 'test' )

Run prediction and returns predictions and potential metrics.

Depending on the dataset and your use case, your test dataset may contain labels. In that case, this method will also return metrics, like in evaluate().

If your predictions or labels have different sequence length (for instance because you’re doing dynamic padding in a token classification task) the predictions will be padded (on the right) to allow for concatenation into one array. The padding index is -100.

Returns: NamedTuple A namedtuple with the following keys:

( model: Module inputs: dict prediction_loss_only: bool ignore_keys: list[str] | None = None ) → tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]

The dictionary will be unpacked before being fed to the model. Most models expect the targets under the argument labels. Check your model’s documentation for all accepted arguments.

tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]

A tuple with the loss, logits and labels (each being optional).

A tuple with the loss, logits and labels (each being optional).

Perform an evaluation step on model using inputs.

Subclass and override to inject custom behavior.

( auto_find_batch_size = False )

Sets values in the deepspeed plugin based on the Trainer args

( commit_message: str | None = 'End of training' blocking: bool = True token: str | None = None revision: str | None = None **kwargs )

Upload self.model and self.processing_class to the 🤗 model hub on the repo self.args.hub_model_id.

Remove a callback from the current list of TrainerCallback.

( split metrics combined = True )

Save metrics into a json file for that split, e.g. train_results.json.

Under distributed environment this is done only for a process with rank 0.

To understand the metrics please read the docstring of log_metrics(). The only difference is that raw unformatted numbers are saved in the current method.

( output_dir: str | None = None _internal_call: bool = False )

Will save the model, so you can reload it using from_pretrained().

Will only save from the main process.

Saves the Trainer state, since Trainer.save_model saves only the tokenizer with the model.

Under distributed environment this is done only for a process with rank 0.

( args: TrainingArguments dataloader: DataLoader total_train_batch_size: int )

Calculates and returns the following values:

( resume_from_checkpoint: str | bool | None = None trial: typing.Union[ForwardRef('optuna.Trial'), dict[str, typing.Any], NoneType] = None ignore_keys_for_eval: list[str] | None = None )

Main training entry point.

( model: Module inputs: dict num_items_in_batch: torch.Tensor | None = None ) → torch.Tensor

The dictionary will be unpacked before being fed to the model. Most models expect the targets under the argument labels. Check your model’s documentation for all accepted arguments.

The tensor with training loss on this batch.

The tensor with training loss on this batch.

Perform a training step on a batch of inputs.

Subclass and override to inject custom behavior.

( model: typing.Union[ForwardRef('PreTrainedModel'), torch.nn.modules.module.Module, NoneType] = None args: typing.Optional[ForwardRef('TrainingArguments')] = None data_collator: typing.Optional[ForwardRef('DataCollator')] = None train_dataset: typing.Union[torch.utils.data.dataset.Dataset, ForwardRef('IterableDataset'), ForwardRef('datasets.Dataset'), NoneType] = None eval_dataset: torch.utils.data.dataset.Dataset | dict[str, torch.utils.data.dataset.Dataset] | None = None processing_class: typing.Union[ForwardRef('PreTrainedTokenizerBase'), ForwardRef('BaseImageProcessor'), ForwardRef('FeatureExtractionMixin'), ForwardRef('ProcessorMixin'), NoneType] = None model_init: collections.abc.Callable[[], 'PreTrainedModel'] | None = None compute_loss_func: collections.abc.Callable | None = None compute_metrics: collections.abc.Callable[['EvalPrediction'], dict] | None = None callbacks: list['TrainerCallback'] | None = None optimizers: tuple = (None, None) preprocess_logits_for_metrics: collections.abc.Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None )

( eval_dataset: torch.utils.data.dataset.Dataset | None = None ignore_keys: list[str] | None = None metric_key_prefix: str = 'eval' **gen_kwargs )

Run evaluation and returns metrics.

The calling script will be responsible for providing a method to compute metrics, as they are task-dependent (pass it to the init compute_metrics argument).

You can also subclass and override this method to inject custom behavior.

( test_dataset: Dataset ignore_keys: list[str] | None = None metric_key_prefix: str = 'test' **gen_kwargs )

Run prediction and returns predictions and potential metrics.

Depending on the dataset and your use case, your test dataset may contain labels. In that case, this method will also return metrics, like in evaluate().

If your predictions or labels have different sequence lengths (for instance because you’re doing dynamic padding in a token classification task) the predictions will be padded (on the right) to allow for concatenation into one array. The padding index is -100.

Returns: NamedTuple A namedtuple with the following keys:

( output_dir: str | None = None do_train: bool = False do_eval: bool = False do_predict: bool = False eval_strategy: transformers.trainer_utils.IntervalStrategy | str = 'no' prediction_loss_only: bool = False per_device_train_batch_size: int = 8 per_device_eval_batch_size: int = 8 gradient_accumulation_steps: int = 1 eval_accumulation_steps: int | None = None eval_delay: float = 0 torch_empty_cache_steps: int | None = None learning_rate: float = 5e-05 weight_decay: float = 0.0 adam_beta1: float = 0.9 adam_beta2: float = 0.999 adam_epsilon: float = 1e-08 max_grad_norm: float = 1.0 num_train_epochs: float = 3.0 max_steps: int = -1 lr_scheduler_type: transformers.trainer_utils.SchedulerType | str = 'linear' lr_scheduler_kwargs: dict | str | None = None warmup_ratio: float | None = None warmup_steps: float = 0 log_level: str = 'passive' log_level_replica: str = 'warning' log_on_each_node: bool = True logging_dir: str | None = None logging_strategy: transformers.trainer_utils.IntervalStrategy | str = 'steps' logging_first_step: bool = False logging_steps: float = 500 logging_nan_inf_filter: bool = True save_strategy: transformers.trainer_utils.SaveStrategy | str = 'steps' save_steps: float = 500 save_total_limit: int | None = None enable_jit_checkpoint: bool = False save_on_each_node: bool = False save_only_model: bool = False restore_callback_states_from_checkpoint: bool = False use_cpu: bool = False seed: int = 42 data_seed: int | None = None bf16: bool = False fp16: bool = False bf16_full_eval: bool = False fp16_full_eval: bool = False tf32: bool | None = None local_rank: int = -1 ddp_backend: str | None = None debug: str | list[transformers.debug_utils.DebugOption] = '' dataloader_drop_last: bool = False eval_steps: float | None = None dataloader_num_workers: int = 0 dataloader_prefetch_factor: int | None = None run_name: str | None = None disable_tqdm: bool | None = None remove_unused_columns: bool = True label_names: list[str] | None = None load_best_model_at_end: bool = False metric_for_best_model: str | None = None greater_is_better: bool | None = None ignore_data_skip: bool = False fsdp: list[transformers.trainer_utils.FSDPOption] | str | None = None fsdp_config: dict[str, typing.Any] | str | None = None accelerator_config: dict | str | None = None parallelism_config: accelerate.parallelism_config.ParallelismConfig | None = None deepspeed: dict | str | None = None label_smoothing_factor: float = 0.0 optim: transformers.training_args.OptimizerNames | str = 'adamw_torch_fused' optim_args: str | None = None group_by_length: bool = False length_column_name: str = 'length' report_to: None | str | list[str] = 'none' project: str = 'huggingface' trackio_space_id: str | None = 'trackio' ddp_find_unused_parameters: bool | None = None ddp_bucket_cap_mb: int | None = None ddp_broadcast_buffers: bool | None = None dataloader_pin_memory: bool = True dataloader_persistent_workers: bool = False skip_memory_metrics: bool = True push_to_hub: bool = False resume_from_checkpoint: str | None = None hub_model_id: str | None = None hub_strategy: transformers.trainer_utils.HubStrategy | str = 'every_save' hub_token: str | None = None hub_private_repo: bool | None = None hub_always_push: bool = False hub_revision: str | None = None gradient_checkpointing: bool = False gradient_checkpointing_kwargs: dict[str, typing.Any] | str | None = None include_for_metrics: list = <factory> eval_do_concat_batches: bool = True auto_find_batch_size: bool = False full_determinism: bool = False ddp_timeout: int = 1800 torch_compile: bool = False torch_compile_backend: str | None = None torch_compile_mode: str | None = None include_num_input_tokens_seen: str | bool = 'no' neftune_noise_alpha: float | None = None optim_target_modules: None | str | list[str] = None batch_eval_metrics: bool = False eval_on_start: bool = False use_liger_kernel: bool = False liger_kernel_config: dict[str, bool] | None = None eval_use_gather_object: bool = False average_tokens_across_devices: bool = True use_cache: bool = False )

When using gradient accumulation, one step is counted as one step with backward pass. Therefore, logging, evaluation, save will be conducted every gradient_accumulation_steps * xxx_step training examples.

This can help avoid CUDA out-of-memory errors by lowering peak VRAM usage at a cost of about 10% slower performance.

logging_nan_inf_filter only influences the logging of loss values, it does not change the behavior the gradient is computed or applied to the model.

If "epoch" or "steps" is chosen, saving will also be performed at the very end of training, always.

This should not be activated when the different nodes use the same storage as the files will be saved with the same names for each node.

Will eventually default to the list of argument names accepted by the model that contain the word “label”, except if the model used is one of the XxxForQuestionAnswering in which case it will also include the ["start_positions", "end_positions"] keys.

You should only specify label_names if you’re using custom label names or if your model’s forward consumes multiple label tensors (e.g., extractive QA).

When set to True, the parameters save_strategy needs to be the same as eval_strategy, and in the case it is “steps”, save_steps must be a round multiple of eval_steps.

If not specified, this will default to "loss" when either load_best_model_at_end == True or lr_scheduler_type == SchedulerType.REDUCE_ON_PLATEAU (to use the evaluation loss).

If you set this value, greater_is_better will default to True unless the name ends with “loss”. Don’t forget to set it to False if your metric is better when lower.

A list of options along the following:

A List of config and its options:

fsdp_version (int, optional, defaults to 1): The version of FSDP to use. Defaults to 1.

min_num_params (int, optional, defaults to 0): FSDP’s minimum number of parameters for Default Auto Wrapping. (useful only when fsdp field is passed).

transformer_layer_cls_to_wrap (list[str], optional): List of transformer layer class names (case-sensitive) to wrap, e.g, BertLayer, GPTJBlock, T5Block … (useful only when fsdp flag is passed).

backward_prefetch (str, optional) FSDP’s backward prefetch mode. Controls when to prefetch next set of parameters (useful only when fsdp field is passed).

A list of options along the following:

forward_prefetch (bool, optional, defaults to False) FSDP’s forward prefetch mode (useful only when fsdp field is passed). If "True", then FSDP explicitly prefetches the next upcoming all-gather while executing in the forward pass.

limit_all_gathers (bool, optional, defaults to False) FSDP’s limit_all_gathers (useful only when fsdp field is passed). If "True", FSDP explicitly synchronizes the CPU thread to prevent too many in-flight all-gathers.

use_orig_params (bool, optional, defaults to True) If "True", allows non-uniform requires_grad during init, which means support for interspersed frozen and trainable parameters. Useful in cases such as parameter-efficient fine-tuning. Please refer this [blog](https://dev-discuss.pytorch.org/t/rethinking-pytorch-fully-sharded-data-parallel-fsdp-from-first-principles/1019

sync_module_states (bool, optional, defaults to True) If "True", each individually wrapped FSDP unit will broadcast module parameters from rank 0 to ensure they are the same across all ranks after initialization

cpu_ram_efficient_loading (bool, optional, defaults to False) If "True", only the first process loads the pretrained model checkpoint while all other processes have empty weights. When this setting as "True", sync_module_states also must to be "True", otherwise all the processes except the main process would have random weights leading to unexpected behaviour during training.

activation_checkpointing (bool, optional, defaults to False): If "True", activation checkpointing is a technique to reduce memory usage by clearing activations of certain layers and recomputing them during a backward pass. Effectively, this trades extra computation time for reduced memory usage.

xla (bool, optional, defaults to False): Whether to use PyTorch/XLA Fully Sharded Data Parallel Training. This is an experimental feature and its API may evolve in the future.

xla_fsdp_settings (dict, optional) The value is a dictionary which stores the XLA FSDP wrapping parameters.

For a complete list of options, please see here.

xla_fsdp_grad_ckpt (bool, optional, defaults to False): Will use gradient checkpointing over each nested XLA FSDP wrapped layer. This setting can only be used when the xla flag is set to true, and an auto wrapping policy is specified through fsdp_min_num_params or fsdp_transformer_layer_cls_to_wrap.

A list of config and its options:

Possible options are:

The options should be separated by whitespaces.

If output_dir exists, it needs to be a local clone of the repository to which the Trainer will be pushed.

Will default to the name of output_dir.

This will use the best defaults for the torch.compile API. You can customize the defaults with the argument torch_compile_backend and torch_compile_mode but we don’t guarantee any of them will work as the support is progressively rolled in in PyTorch.

This flag and the whole compile API is experimental and subject to change in future releases.

Refer to the PyTorch doc for possible values and note that they may change across PyTorch versions.

This flag is experimental and subject to change in future releases.

Refer to the PyTorch doc for possible values and note that they may change across PyTorch versions.

This flag is experimental and subject to change in future releases.

TrainingArguments is the subset of the arguments we use in our example scripts which relate to the training loop itself.

Using HfArgumentParser we can turn this class into argparse arguments that can be specified on the command line.

Returns the log level to be used depending on whether this process is the main process of node 0, main process of node non-0, or a non-main process.

For the main process the log level defaults to the logging level set (logging.WARNING if you didn’t do anything) unless overridden by log_level argument.

For the replica processes the log level defaults to logging.WARNING unless overridden by log_level_replica argument.

The choice between the main and replica process settings is made according to the return value of should_log.

( num_training_steps: int )

Get number of steps used for a linear warmup.

( local = True desc = 'work' )

A context manager for torch distributed environment where on needs to do something on the main process, while blocking replicas, and when it’s finished releasing the replicas.

One such use is for datasets’s map feature which to be efficient should be run once on the main process, which upon completion saves a cached version of results and which then automatically gets loaded by the replicas.

( train_batch_size: int = 8 eval_batch_size: int = 8 drop_last: bool = False num_workers: int = 0 pin_memory: bool = True persistent_workers: bool = False prefetch_factor: int | None = None auto_find_batch_size: bool = False ignore_data_skip: bool = False sampler_seed: int | None = None )

A method that regroups all arguments linked to the dataloaders creation.

( strategy: str | transformers.trainer_utils.IntervalStrategy = 'no' steps: int = 500 batch_size: int = 8 accumulation_steps: int | None = None delay: float | None = None loss_only: bool = False )

Setting a strategy different from "no" will set self.do_eval to True.

A method that regroups all arguments linked to evaluation.

( strategy: str | transformers.trainer_utils.IntervalStrategy = 'steps' steps: int = 500 report_to: str | list[str] = 'none' level: str = 'passive' first_step: bool = False nan_inf_filter: bool = False on_each_node: bool = False replica_level: str = 'passive' )

nan_inf_filter only influences the logging of loss values, it does not change the behavior the gradient is computed or applied to the model.

A method that regroups all arguments linked to logging.

( name: str | transformers.trainer_utils.SchedulerType = 'linear' num_epochs: float = 3.0 max_steps: int = -1 warmup_steps: float = 0 warmup_ratio: float | None = None )

A method that regroups all arguments linked to the learning rate scheduler and its hyperparameters.

( name: str | transformers.training_args.OptimizerNames = 'adamw_torch' learning_rate: float = 5e-05 weight_decay: float = 0 beta1: float = 0.9 beta2: float = 0.999 epsilon: float = 1e-08 args: str | None = None )

A method that regroups all arguments linked to the optimizer and its hyperparameters.

( model_id: str strategy: str | transformers.trainer_utils.HubStrategy = 'every_save' token: str | None = None private_repo: bool | None = None always_push: bool = False revision: str | None = None )

A method that regroups all arguments linked to synchronizing checkpoints with the Hub.

Calling this method will set self.push_to_hub to True, which means the output_dir will begin a git directory synced with the repo (determined by model_id) and the content will be pushed each time a save is triggered (depending on your self.save_strategy). Calling save_model() will also trigger a push.

( strategy: str | transformers.trainer_utils.IntervalStrategy = 'steps' steps: int = 500 total_limit: int | None = None on_each_node: bool = False )

This should not be activated when the different nodes use the same storage as the files will be saved with the same names for each node.

A method that regroups all arguments linked to checkpoint saving.

( batch_size: int = 8 loss_only: bool = False )

A method that regroups all basic arguments linked to testing on a held-out dataset.

Calling this method will automatically set self.do_predict to True.

( learning_rate: float = 5e-05 batch_size: int = 8 weight_decay: float = 0 num_epochs: float = 3 max_steps: int = -1 gradient_accumulation_steps: int = 1 seed: int = 42 gradient_checkpointing: bool = False )

When using gradient accumulation, one step is counted as one step with backward pass. Therefore, logging, evaluation, save will be conducted every gradient_accumulation_steps * xxx_step training examples.

A method that regroups all basic arguments linked to the training.

Calling this method will automatically set self.do_train to True.

Serializes this instance while replace Enum by their values (for JSON serialization support). It obfuscates the token values by removing their value.

Serializes this instance to a JSON string.

Sanitized serialization to use with TensorBoard’s hparams

( output_dir: str | None = None do_train: bool = False do_eval: bool = False do_predict: bool = False eval_strategy: transformers.trainer_utils.IntervalStrategy | str = 'no' prediction_loss_only: bool = False per_device_train_batch_size: int = 8 per_device_eval_batch_size: int = 8 gradient_accumulation_steps: int = 1 eval_accumulation_steps: int | None = None eval_delay: float = 0 torch_empty_cache_steps: int | None = None learning_rate: float = 5e-05 weight_decay: float = 0.0 adam_beta1: float = 0.9 adam_beta2: float = 0.999 adam_epsilon: float = 1e-08 max_grad_norm: float = 1.0 num_train_epochs: float = 3.0 max_steps: int = -1 lr_scheduler_type: transformers.trainer_utils.SchedulerType | str = 'linear' lr_scheduler_kwargs: dict | str | None = None warmup_ratio: float | None = None warmup_steps: float = 0 log_level: str = 'passive' log_level_replica: str = 'warning' log_on_each_node: bool = True logging_dir: str | None = None logging_strategy: transformers.trainer_utils.IntervalStrategy | str = 'steps' logging_first_step: bool = False logging_steps: float = 500 logging_nan_inf_filter: bool = True save_strategy: transformers.trainer_utils.SaveStrategy | str = 'steps' save_steps: float = 500 save_total_limit: int | None = None enable_jit_checkpoint: bool = False save_on_each_node: bool = False save_only_model: bool = False restore_callback_states_from_checkpoint: bool = False use_cpu: bool = False seed: int = 42 data_seed: int | None = None bf16: bool = False fp16: bool = False bf16_full_eval: bool = False fp16_full_eval: bool = False tf32: bool | None = None local_rank: int = -1 ddp_backend: str | None = None debug: str | list[transformers.debug_utils.DebugOption] = '' dataloader_drop_last: bool = False eval_steps: float | None = None dataloader_num_workers: int = 0 dataloader_prefetch_factor: int | None = None run_name: str | None = None disable_tqdm: bool | None = None remove_unused_columns: bool = True label_names: list[str] | None = None load_best_model_at_end: bool = False metric_for_best_model: str | None = None greater_is_better: bool | None = None ignore_data_skip: bool = False fsdp: list[transformers.trainer_utils.FSDPOption] | str | None = None fsdp_config: dict[str, typing.Any] | str | None = None accelerator_config: dict | str | None = None parallelism_config: accelerate.parallelism_config.ParallelismConfig | None = None deepspeed: dict | str | None = None label_smoothing_factor: float = 0.0 optim: transformers.training_args.OptimizerNames | str = 'adamw_torch_fused' optim_args: str | None = None group_by_length: bool = False length_column_name: str = 'length' report_to: None | str | list[str] = 'none' project: str = 'huggingface' trackio_space_id: str | None = 'trackio' ddp_find_unused_parameters: bool | None = None ddp_bucket_cap_mb: int | None = None ddp_broadcast_buffers: bool | None = None dataloader_pin_memory: bool = True dataloader_persistent_workers: bool = False skip_memory_metrics: bool = True push_to_hub: bool = False resume_from_checkpoint: str | None = None hub_model_id: str | None = None hub_strategy: transformers.trainer_utils.HubStrategy | str = 'every_save' hub_token: str | None = None hub_private_repo: bool | None = None hub_always_push: bool = False hub_revision: str | None = None gradient_checkpointing: bool = False gradient_checkpointing_kwargs: dict[str, typing.Any] | str | None = None include_for_metrics: list = <factory> eval_do_concat_batches: bool = True auto_find_batch_size: bool = False full_determinism: bool = False ddp_timeout: int = 1800 torch_compile: bool = False torch_compile_backend: str | None = None torch_compile_mode: str | None = None include_num_input_tokens_seen: str | bool = 'no' neftune_noise_alpha: float | None = None optim_target_modules: None | str | list[str] = None batch_eval_metrics: bool = False eval_on_start: bool = False use_liger_kernel: bool = False liger_kernel_config: dict[str, bool] | None = None eval_use_gather_object: bool = False average_tokens_across_devices: bool = True use_cache: bool = False sortish_sampler: bool = False predict_with_generate: bool = False generation_max_length: int | None = None generation_num_beams: int | None = None generation_config: str | pathlib.Path | transformers.generation.configuration_utils.GenerationConfig | None = None )

When using gradient accumulation, one step is counted as one step with backward pass. Therefore, logging, evaluation, save will be conducted every gradient_accumulation_steps * xxx_step training examples.

This can help avoid CUDA out-of-memory errors by lowering peak VRAM usage at a cost of about 10% slower performance.

logging_nan_inf_filter only influences the logging of loss values, it does not change the behavior the gradient is computed or applied to the model.

If "epoch" or "steps" is chosen, saving will also be performed at the very end of training, always.

This should not be activated when the different nodes use the same storage as the files will be saved with the same names for each node.

Will eventually default to the list of argument names accepted by the model that contain the word “label”, except if the model used is one of the XxxForQuestionAnswering in which case it will also include the ["start_positions", "end_positions"] keys.

You should only specify label_names if you’re using custom label names or if your model’s forward consumes multiple label tensors (e.g., extractive QA).

When set to True, the parameters save_strategy needs to be the same as eval_strategy, and in the case it is “steps”, save_steps must be a round multiple of eval_steps.

If not specified, this will default to "loss" when either load_best_model_at_end == True or lr_scheduler_type == SchedulerType.REDUCE_ON_PLATEAU (to use the evaluation loss).

If you set this value, greater_is_better will default to True unless the name ends with “loss”. Don’t forget to set it to False if your metric is better when lower.

A list of options along the following:

A List of config and its options:

fsdp_version (int, optional, defaults to 1): The version of FSDP to use. Defaults to 1.

min_num_params (int, optional, defaults to 0): FSDP’s minimum number of parameters for Default Auto Wrapping. (useful only when fsdp field is passed).

transformer_layer_cls_to_wrap (list[str], optional): List of transformer layer class names (case-sensitive) to wrap, e.g, BertLayer, GPTJBlock, T5Block … (useful only when fsdp flag is passed).

backward_prefetch (str, optional) FSDP’s backward prefetch mode. Controls when to prefetch next set of parameters (useful only when fsdp field is passed).

A list of options along the following:

forward_prefetch (bool, optional, defaults to False) FSDP’s forward prefetch mode (useful only when fsdp field is passed). If "True", then FSDP explicitly prefetches the next upcoming all-gather while executing in the forward pass.

limit_all_gathers (bool, optional, defaults to False) FSDP’s limit_all_gathers (useful only when fsdp field is passed). If "True", FSDP explicitly synchronizes the CPU thread to prevent too many in-flight all-gathers.

use_orig_params (bool, optional, defaults to True) If "True", allows non-uniform requires_grad during init, which means support for interspersed frozen and trainable parameters. Useful in cases such as parameter-efficient fine-tuning. Please refer this [blog](https://dev-discuss.pytorch.org/t/rethinking-pytorch-fully-sharded-data-parallel-fsdp-from-first-principles/1019

sync_module_states (bool, optional, defaults to True) If "True", each individually wrapped FSDP unit will broadcast module parameters from rank 0 to ensure they are the same across all ranks after initialization

cpu_ram_efficient_loading (bool, optional, defaults to False) If "True", only the first process loads the pretrained model checkpoint while all other processes have empty weights. When this setting as "True", sync_module_states also must to be "True", otherwise all the processes except the main process would have random weights leading to unexpected behaviour during training.

activation_checkpointing (bool, optional, defaults to False): If "True", activation checkpointing is a technique to reduce memory usage by clearing activations of certain layers and recomputing them during a backward pass. Effectively, this trades extra computation time for reduced memory usage.

xla (bool, optional, defaults to False): Whether to use PyTorch/XLA Fully Sharded Data Parallel Training. This is an experimental feature and its API may evolve in the future.

xla_fsdp_settings (dict, optional) The value is a dictionary which stores the XLA FSDP wrapping parameters.

For a complete list of options, please see here.

xla_fsdp_grad_ckpt (bool, optional, defaults to False): Will use gradient checkpointing over each nested XLA FSDP wrapped layer. This setting can only be used when the xla flag is set to true, and an auto wrapping policy is specified through fsdp_min_num_params or fsdp_transformer_layer_cls_to_wrap.

A list of config and its options:

Possible options are:

The options should be separated by whitespaces.

If output_dir exists, it needs to be a local clone of the repository to which the Trainer will be pushed.

Will default to the name of output_dir.

This will use the best defaults for the torch.compile API. You can customize the defaults with the argument torch_compile_backend and torch_compile_mode but we don’t guarantee any of them will work as the support is progressively rolled in in PyTorch.

This flag and the whole compile API is experimental and subject to change in future releases.

Refer to the PyTorch doc for possible values and note that they may change across PyTorch versions.

This flag is experimental and subject to change in future releases.

Refer to the PyTorch doc for possible values and note that they may change across PyTorch versions.

This flag is experimental and subject to change in future releases.

TrainingArguments is the subset of the arguments we use in our example scripts which relate to the training loop itself.

Using HfArgumentParser we can turn this class into argparse arguments that can be specified on the command line.

Serializes this instance while replace Enum by their values and GenerationConfig by dictionaries (for JSON serialization support). It obfuscates the token values by removing their value.

---

## Transformers

**URL:** https://huggingface.co/docs/transformers/v5.0.0/en/model_doc/altclip

**Contents:**
- Transformers
- AltCLIP
- Notes
- AltCLIPConfig
  - class transformers.AltCLIPConfig
- AltCLIPTextConfig
  - class transformers.AltCLIPTextConfig
- AltCLIPVisionConfig
  - class transformers.AltCLIPVisionConfig
- AltCLIPModel

Transformers documentation

and get access to the augmented documentation experience

This model was released on 2022-11-12 and added to Hugging Face Transformers on 2023-01-04.

AltCLIP replaces the CLIP text encoder with a multilingual XLM-R encoder and aligns image and text representations with teacher learning and contrastive learning.

You can find all the original AltCLIP checkpoints under the AltClip collection.

Click on the AltCLIP models in the right sidebar for more examples of how to apply AltCLIP to different tasks.

The examples below demonstrates how to calculate similarity scores between an image and one or more captions with the AutoModel class.

Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the Quantization overview for more available quantization backends.

The example below uses torchao to only quantize the weights to int4.

( text_config = None vision_config = None projection_dim = 768 logit_scale_init_value = 2.6592 **kwargs )

This is the configuration class to store the configuration of a AltCLIPModel. It is used to instantiate an AltCLIP model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the AltCLIP BAAI/AltCLIP architecture.

Configuration objects inherit from PreTrainedConfig and can be used to control the model outputs. Read the documentation from PreTrainedConfig for more information.

( vocab_size = 250002 hidden_size = 1024 num_hidden_layers = 24 num_attention_heads = 16 intermediate_size = 4096 hidden_act = 'gelu' hidden_dropout_prob = 0.1 attention_probs_dropout_prob = 0.1 max_position_embeddings = 514 type_vocab_size = 1 initializer_range = 0.02 initializer_factor = 0.02 layer_norm_eps = 1e-05 pad_token_id = 1 bos_token_id = 0 eos_token_id = 2 project_dim = 768 **kwargs )

This is the configuration class to store the configuration of a AltCLIPTextModel. It is used to instantiate a AltCLIP text model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the AltCLIP BAAI/AltCLIP architecture.

Configuration objects inherit from PreTrainedConfig and can be used to control the model outputs. Read the documentation from PreTrainedConfig for more information.

( hidden_size = 768 intermediate_size = 3072 projection_dim = 512 num_hidden_layers = 12 num_attention_heads = 12 num_channels = 3 image_size = 224 patch_size = 32 hidden_act = 'quick_gelu' layer_norm_eps = 1e-05 attention_dropout = 0.0 initializer_range = 0.02 initializer_factor = 1.0 **kwargs )

This is the configuration class to store the configuration of a AltCLIPModel. It is used to instantiate an AltCLIP model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the AltCLIP BAAI/AltCLIP architecture.

Configuration objects inherit from PreTrainedConfig and can be used to control the model outputs. Read the documentation from PreTrainedConfig for more information.

( config: AltCLIPConfig )

( input_ids: torch.LongTensor | None = None pixel_values: torch.FloatTensor | None = None attention_mask: torch.Tensor | None = None position_ids: torch.LongTensor | None = None token_type_ids: torch.Tensor | None = None return_loss: bool | None = None output_attentions: bool | None = None output_hidden_states: bool | None = None interpolate_pos_encoding: bool = False return_dict: bool | None = None **kwargs ) → transformers.models.altclip.modeling_altclip.AltCLIPOutput or tuple(torch.FloatTensor)

Indices can be obtained using AutoTokenizer. See PreTrainedTokenizer.encode() and PreTrainedTokenizer.call() for details.

What are attention masks?

What are position IDs?

What are token type IDs?

transformers.models.altclip.modeling_altclip.AltCLIPOutput or tuple(torch.FloatTensor)

A transformers.models.altclip.modeling_altclip.AltCLIPOutput or a tuple of torch.FloatTensor (if return_dict=False is passed or when config.return_dict=False) comprising various elements depending on the configuration (AltCLIPConfig) and inputs. loss (torch.FloatTensor of shape (1,), optional, returned when return_loss is True) — Contrastive loss for image-text similarity. logits_per_image (torch.FloatTensor of shape (image_batch_size, text_batch_size)) — The scaled dot product scores between image_embeds and text_embeds. This represents the image-text similarity scores. logits_per_text (torch.FloatTensor of shape (text_batch_size, image_batch_size)) — The scaled dot product scores between text_embeds and image_embeds. This represents the text-image similarity scores. text_embeds (torch.FloatTensor of shape (batch_size, output_dim) — The text embeddings obtained by applying the projection layer to the pooled output of AltCLIPTextModel. image_embeds (torch.FloatTensor of shape (batch_size, output_dim) — The image embeddings obtained by applying the projection layer to the pooled output of AltCLIPVisionModel. text_model_output (<class '~modeling_outputs.BaseModelOutputWithPooling'>.text_model_output, defaults to None) — The output of the AltCLIPTextModel. vision_model_output (<class '~modeling_outputs.BaseModelOutputWithPooling'>.vision_model_output, defaults to None) — The output of the AltCLIPVisionModel.

A transformers.models.altclip.modeling_altclip.AltCLIPOutput or a tuple of torch.FloatTensor (if return_dict=False is passed or when config.return_dict=False) comprising various elements depending on the configuration (AltCLIPConfig) and inputs.

The AltCLIPModel forward method, overrides the __call__ special method.

Although the recipe for forward pass needs to be defined within this function, one should call the Module instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

( pixel_values: FloatTensor interpolate_pos_encoding: bool = False **kwargs: typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs] ) → transformers.modeling_outputs.BaseModelOutputWithPooling or tuple(torch.FloatTensor)

transformers.modeling_outputs.BaseModelOutputWithPooling or tuple(torch.FloatTensor)

A transformers.modeling_outputs.BaseModelOutputWithPooling or a tuple of torch.FloatTensor (if return_dict=False is passed or when config.return_dict=False) comprising various elements depending on the configuration (AltCLIPConfig) and inputs. last_hidden_state (torch.FloatTensor of shape (batch_size, sequence_length, hidden_size)) — Sequence of hidden-states at the output of the last layer of the model. pooler_output (torch.FloatTensor of shape (batch_size, hidden_size)) — Last layer hidden-state of the first token of the sequence (classification token) after further processing through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns the classification token after processing through a linear layer and a tanh activation function. The linear layer weights are trained from the next sentence prediction (classification) objective during pretraining. hidden_states (tuple(torch.FloatTensor), optional, returned when output_hidden_states=True is passed or when config.output_hidden_states=True) — Tuple of torch.FloatTensor (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape (batch_size, sequence_length, hidden_size). Hidden-states of the model at the output of each layer plus the optional initial embedding outputs. attentions (tuple(torch.FloatTensor), optional, returned when output_attentions=True is passed or when config.output_attentions=True) — Tuple of torch.FloatTensor (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length). Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

A transformers.modeling_outputs.BaseModelOutputWithPooling or a tuple of torch.FloatTensor (if return_dict=False is passed or when config.return_dict=False) comprising various elements depending on the configuration (AltCLIPConfig) and inputs.

last_hidden_state (torch.FloatTensor of shape (batch_size, sequence_length, hidden_size)) — Sequence of hidden-states at the output of the last layer of the model.

pooler_output (torch.FloatTensor of shape (batch_size, hidden_size)) — Last layer hidden-state of the first token of the sequence (classification token) after further processing through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns the classification token after processing through a linear layer and a tanh activation function. The linear layer weights are trained from the next sentence prediction (classification) objective during pretraining.

hidden_states (tuple(torch.FloatTensor), optional, returned when output_hidden_states=True is passed or when config.output_hidden_states=True) — Tuple of torch.FloatTensor (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape (batch_size, sequence_length, hidden_size).

Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.

attentions (tuple(torch.FloatTensor), optional, returned when output_attentions=True is passed or when config.output_attentions=True) — Tuple of torch.FloatTensor (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length).

Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

( input_ids: Tensor attention_mask: torch.Tensor | None = None position_ids: torch.Tensor | None = None token_type_ids: torch.Tensor | None = None **kwargs: typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs] ) → transformers.modeling_outputs.BaseModelOutputWithPooling or tuple(torch.FloatTensor)

Indices can be obtained using AutoTokenizer. See PreTrainedTokenizer.encode() and PreTrainedTokenizer.call() for details.

What are attention masks?

What are position IDs?

What are token type IDs?

transformers.modeling_outputs.BaseModelOutputWithPooling or tuple(torch.FloatTensor)

A transformers.modeling_outputs.BaseModelOutputWithPooling or a tuple of torch.FloatTensor (if return_dict=False is passed or when config.return_dict=False) comprising various elements depending on the configuration (AltCLIPConfig) and inputs. last_hidden_state (torch.FloatTensor of shape (batch_size, sequence_length, hidden_size)) — Sequence of hidden-states at the output of the last layer of the model. pooler_output (torch.FloatTensor of shape (batch_size, hidden_size)) — Last layer hidden-state of the first token of the sequence (classification token) after further processing through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns the classification token after processing through a linear layer and a tanh activation function. The linear layer weights are trained from the next sentence prediction (classification) objective during pretraining. hidden_states (tuple(torch.FloatTensor), optional, returned when output_hidden_states=True is passed or when config.output_hidden_states=True) — Tuple of torch.FloatTensor (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape (batch_size, sequence_length, hidden_size). Hidden-states of the model at the output of each layer plus the optional initial embedding outputs. attentions (tuple(torch.FloatTensor), optional, returned when output_attentions=True is passed or when config.output_attentions=True) — Tuple of torch.FloatTensor (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length). Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

A transformers.modeling_outputs.BaseModelOutputWithPooling or a tuple of torch.FloatTensor (if return_dict=False is passed or when config.return_dict=False) comprising various elements depending on the configuration (AltCLIPConfig) and inputs.

last_hidden_state (torch.FloatTensor of shape (batch_size, sequence_length, hidden_size)) — Sequence of hidden-states at the output of the last layer of the model.

pooler_output (torch.FloatTensor of shape (batch_size, hidden_size)) — Last layer hidden-state of the first token of the sequence (classification token) after further processing through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns the classification token after processing through a linear layer and a tanh activation function. The linear layer weights are trained from the next sentence prediction (classification) objective during pretraining.

hidden_states (tuple(torch.FloatTensor), optional, returned when output_hidden_states=True is passed or when config.output_hidden_states=True) — Tuple of torch.FloatTensor (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape (batch_size, sequence_length, hidden_size).

Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.

attentions (tuple(torch.FloatTensor), optional, returned when output_attentions=True is passed or when config.output_attentions=True) — Tuple of torch.FloatTensor (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length).

Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

( input_ids: torch.Tensor | None = None attention_mask: torch.Tensor | None = None token_type_ids: torch.Tensor | None = None position_ids: torch.Tensor | None = None inputs_embeds: torch.Tensor | None = None output_attentions: bool | None = None return_dict: bool | None = None output_hidden_states: bool | None = None **kwargs ) → transformers.modeling_outputs.BaseModelOutputWithPoolingAndProjection or tuple(torch.FloatTensor)

Indices can be obtained using AutoTokenizer. See PreTrainedTokenizer.encode() and PreTrainedTokenizer.call() for details.

What are attention masks?

What are token type IDs?

What are position IDs?

transformers.modeling_outputs.BaseModelOutputWithPoolingAndProjection or tuple(torch.FloatTensor)

A transformers.modeling_outputs.BaseModelOutputWithPoolingAndProjection or a tuple of torch.FloatTensor (if return_dict=False is passed or when config.return_dict=False) comprising various elements depending on the configuration (AltCLIPConfig) and inputs. last_hidden_state (torch.FloatTensor of shape (batch_size, sequence_length, hidden_size)) — Sequence of hidden-states at the output of the last layer of the model. pooler_output (torch.FloatTensor of shape (batch_size, hidden_size)) — Last layer hidden-state of the first token of the sequence (classification token) after further processing through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns the classification token after processing through a linear layer and a tanh activation function. The linear layer weights are trained from the next sentence prediction (classification) objective during pretraining. hidden_states (tuple(torch.FloatTensor), optional, returned when output_hidden_states=True is passed or when config.output_hidden_states=True) — Tuple of torch.FloatTensor (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape (batch_size, sequence_length, hidden_size). Hidden-states of the model at the output of each layer plus the optional initial embedding outputs. attentions (tuple(torch.FloatTensor), optional, returned when output_attentions=True is passed or when config.output_attentions=True) — Tuple of torch.FloatTensor (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length). Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads. projection_state (tuple(torch.FloatTensor), returned when output_attentions=True is passed or when config.output_attentions=True) — Tuple of torch.FloatTensor of shape (batch_size,config.project_dim). Text embeddings before the projection layer, used to mimic the last hidden state of the teacher encoder.

A transformers.modeling_outputs.BaseModelOutputWithPoolingAndProjection or a tuple of torch.FloatTensor (if return_dict=False is passed or when config.return_dict=False) comprising various elements depending on the configuration (AltCLIPConfig) and inputs.

last_hidden_state (torch.FloatTensor of shape (batch_size, sequence_length, hidden_size)) — Sequence of hidden-states at the output of the last layer of the model.

pooler_output (torch.FloatTensor of shape (batch_size, hidden_size)) — Last layer hidden-state of the first token of the sequence (classification token) after further processing through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns the classification token after processing through a linear layer and a tanh activation function. The linear layer weights are trained from the next sentence prediction (classification) objective during pretraining.

hidden_states (tuple(torch.FloatTensor), optional, returned when output_hidden_states=True is passed or when config.output_hidden_states=True) — Tuple of torch.FloatTensor (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape (batch_size, sequence_length, hidden_size).

Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.

attentions (tuple(torch.FloatTensor), optional, returned when output_attentions=True is passed or when config.output_attentions=True) — Tuple of torch.FloatTensor (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length).

Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

projection_state (tuple(torch.FloatTensor), returned when output_attentions=True is passed or when config.output_attentions=True) — Tuple of torch.FloatTensor of shape (batch_size,config.project_dim).

Text embeddings before the projection layer, used to mimic the last hidden state of the teacher encoder.

The AltCLIPTextModel forward method, overrides the __call__ special method.

Although the recipe for forward pass needs to be defined within this function, one should call the Module instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

( config: AltCLIPVisionConfig )

( pixel_values: torch.FloatTensor | None = None output_attentions: bool | None = None output_hidden_states: bool | None = None interpolate_pos_encoding: bool = False return_dict: bool | None = None **kwargs ) → transformers.modeling_outputs.BaseModelOutputWithPooling or tuple(torch.FloatTensor)

transformers.modeling_outputs.BaseModelOutputWithPooling or tuple(torch.FloatTensor)

A transformers.modeling_outputs.BaseModelOutputWithPooling or a tuple of torch.FloatTensor (if return_dict=False is passed or when config.return_dict=False) comprising various elements depending on the configuration (AltCLIPConfig) and inputs. last_hidden_state (torch.FloatTensor of shape (batch_size, sequence_length, hidden_size)) — Sequence of hidden-states at the output of the last layer of the model. pooler_output (torch.FloatTensor of shape (batch_size, hidden_size)) — Last layer hidden-state of the first token of the sequence (classification token) after further processing through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns the classification token after processing through a linear layer and a tanh activation function. The linear layer weights are trained from the next sentence prediction (classification) objective during pretraining. hidden_states (tuple(torch.FloatTensor), optional, returned when output_hidden_states=True is passed or when config.output_hidden_states=True) — Tuple of torch.FloatTensor (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape (batch_size, sequence_length, hidden_size). Hidden-states of the model at the output of each layer plus the optional initial embedding outputs. attentions (tuple(torch.FloatTensor), optional, returned when output_attentions=True is passed or when config.output_attentions=True) — Tuple of torch.FloatTensor (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length). Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

A transformers.modeling_outputs.BaseModelOutputWithPooling or a tuple of torch.FloatTensor (if return_dict=False is passed or when config.return_dict=False) comprising various elements depending on the configuration (AltCLIPConfig) and inputs.

last_hidden_state (torch.FloatTensor of shape (batch_size, sequence_length, hidden_size)) — Sequence of hidden-states at the output of the last layer of the model.

pooler_output (torch.FloatTensor of shape (batch_size, hidden_size)) — Last layer hidden-state of the first token of the sequence (classification token) after further processing through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns the classification token after processing through a linear layer and a tanh activation function. The linear layer weights are trained from the next sentence prediction (classification) objective during pretraining.

hidden_states (tuple(torch.FloatTensor), optional, returned when output_hidden_states=True is passed or when config.output_hidden_states=True) — Tuple of torch.FloatTensor (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape (batch_size, sequence_length, hidden_size).

Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.

attentions (tuple(torch.FloatTensor), optional, returned when output_attentions=True is passed or when config.output_attentions=True) — Tuple of torch.FloatTensor (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length).

Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

The AltCLIPVisionModel forward method, overrides the __call__ special method.

Although the recipe for forward pass needs to be defined within this function, one should call the Module instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

( image_processor = None tokenizer = None )

Constructs a AltCLIPProcessor which wraps a image processor and a tokenizer into a single processor.

AltCLIPProcessor offers all the functionalities of CLIPImageProcessorFast and tokenizer_class. See the ~CLIPImageProcessorFast and ~tokenizer_class for more information.

( images: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor'], NoneType] = None text: str | list[str] | list[list[str]] | None = None videos: typing.Union[list['PIL.Image.Image'], numpy.ndarray, ForwardRef('torch.Tensor'), list[numpy.ndarray], list['torch.Tensor'], list[list['PIL.Image.Image']], list[list[numpy.ndarray]], list[list['torch.Tensor']], transformers.video_utils.URL, list[transformers.video_utils.URL], list[list[transformers.video_utils.URL]], transformers.video_utils.Path, list[transformers.video_utils.Path], list[list[transformers.video_utils.Path]], NoneType] = None audio: typing.Union[numpy.ndarray, ForwardRef('torch.Tensor'), collections.abc.Sequence[numpy.ndarray], collections.abc.Sequence['torch.Tensor'], NoneType] = None **kwargs: typing_extensions.Unpack[transformers.processing_utils.ProcessingKwargs] ) → BatchFeature

A BatchFeature object with processed inputs in a dict format.

A BatchFeature object with processed inputs in a dict format.

Main method to prepare for model inputs. This method forwards the each modality argument to its own processor along with kwargs. Please refer to the docstring of the each processor attributes for more information.

---

## Transformers

**URL:** https://huggingface.co/docs/transformers/v5.0.0/en/model_doc/aimv2

**Contents:**
- Transformers
- AIMv2
- Overview
- Usage Example
- Aimv2Config
  - class transformers.Aimv2Config
- Aimv2TextConfig
  - class transformers.Aimv2TextConfig
- Aimv2VisionConfig
  - class transformers.Aimv2VisionConfig

Transformers documentation

and get access to the augmented documentation experience

This model was released on 2024-11-21 and added to Hugging Face Transformers on 2025-07-08.

The AIMv2 model was proposed in Multimodal Autoregressive Pre-training of Large Vision Encoders by Enrico Fini, Mustafa Shukor, Xiujun Li, Philipp Dufter, Michal Klein, David Haldimann, Sai Aitharaju, Victor Guilherme Turrisi da Costa, Louis Béthune, Zhe Gan, Alexander T Toshev, Marcin Eichner, Moin Nabi, Yinfei Yang, Joshua M. Susskind, Alaaeldin El-Nouby.

The abstract from the paper is the following:

We introduce a novel method for pre-training of large-scale vision encoders. Building on recent advancements in autoregressive pre-training of vision models, we extend this framework to a multimodal setting, i.e., images and text. In this paper, we present AIMV2, a family of generalist vision encoders characterized by a straightforward pre-training process, scalability, and remarkable performance across a range of downstream tasks. This is achieved by pairing the vision encoder with a multimodal decoder that autoregressively generates raw image patches and text tokens. Our encoders excel not only in multimodal evaluations but also in vision benchmarks such as localization, grounding, and classification. Notably, our AIMV2-3B encoder achieves 89.5% accuracy on ImageNet-1k with a frozen trunk. Furthermore, AIMV2 consistently outperforms state-of-the-art contrastive models (e.g., CLIP, SigLIP) in multimodal image understanding across diverse settings.

This model was contributed by Yaswanth Gali. The original code can be found here.

Here is an example of Image Feature Extraction using specific checkpoints on resized images and native resolution images:

Here is an example of a checkpoint performing zero-shot classification:

( text_config = None vision_config = None projection_dim = 512 logit_scale_init_value = 2.6592 **kwargs )

Aimv2Config is the configuration class to store the configuration of a Aimv2Model. It is used to instantiate a AIMv2 model according to the specified arguments, defining the text model and vision model configs. Instantiating a configuration with the defaults will yield a similar configuration to that of the AIMv2 apple/aimv2-large-patch14-224-lit architecture.

Configuration objects inherit from PreTrainedConfig and can be used to control the model outputs. Read the documentation from PreTrainedConfig for more information.

( vocab_size: int = 49408 hidden_size: int = 768 intermediate_size: int = 2048 num_hidden_layers: int = 12 num_attention_heads: int = 6 rms_norm_eps: float = 1e-05 attention_dropout: float = 0.0 qkv_bias: bool = False mlp_bias: bool = False hidden_act: str = 'silu' eos_token_id: int = 49407 max_position_embeddings: int = 77 initializer_range: bool = 0.02 **kwargs )

This is the configuration class to store the configuration of a Aimv2TextModel. It is used to instantiate a AIMv2 text encoder according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the text encoder of the AIMv2 apple/aimv2-large-patch14-224-lit architecture.

Configuration objects inherit from PreTrainedConfig and can be used to control the model outputs. Read the documentation from PreTrainedConfig for more information.

( hidden_size: int = 1024 intermediate_size: int = 2816 num_hidden_layers: int = 24 num_attention_heads: int = 8 num_channels: int = 3 image_size: int = 224 patch_size: int = 14 rms_norm_eps: float = 1e-05 attention_dropout: float = 0.0 qkv_bias: bool = False mlp_bias: bool = False hidden_act: str = 'silu' initializer_range: float = 0.02 use_head: bool = True is_native: bool = False **kwargs )

This is the configuration class to store the configuration of a Aimv2VisionModel. It is used to instantiate a AIMv2 vision encoder according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the vision encoder of the AIMv2 apple/aimv2-large-patch14-224 architecture.

Configuration objects inherit from PreTrainedConfig and can be used to control the model outputs. Read the documentation from PreTrainedConfig for more information.

( config: Aimv2Config )

The bare Aimv2 Model outputting raw hidden-states without any specific head on top.

This model inherits from PreTrainedModel. Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch torch.nn.Module subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

( input_ids: torch.LongTensor | None = None pixel_values: torch.FloatTensor | None = None attention_mask: torch.Tensor | None = None **kwargs: typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs] ) → transformers.models.aimv2.modeling_aimv2.Aimv2Output or tuple(torch.FloatTensor)

Indices can be obtained using AutoTokenizer. See PreTrainedTokenizer.encode() and PreTrainedTokenizer.call() for details.

What are attention masks?

transformers.models.aimv2.modeling_aimv2.Aimv2Output or tuple(torch.FloatTensor)

A transformers.models.aimv2.modeling_aimv2.Aimv2Output or a tuple of torch.FloatTensor (if return_dict=False is passed or when config.return_dict=False) comprising various elements depending on the configuration (None) and inputs. loss (torch.FloatTensor of shape (1,), optional, returned when return_loss is True) — Contrastive loss for image-text similarity. logits_per_image (torch.FloatTensor of shape (image_batch_size, text_batch_size)) — The scaled dot product scores between image_embeds and text_embeds. This represents the image-text similarity scores. logits_per_text (torch.FloatTensor of shape (text_batch_size, image_batch_size)) — The scaled dot product scores between text_embeds and image_embeds. This represents the text-image similarity scores. text_embeds (torch.FloatTensor of shape (batch_size, output_dim) — The text embeddings obtained by applying the projection layer to the pooled output of Aimv2TextModel. image_embeds (torch.FloatTensor of shape (batch_size, output_dim) — The image embeddings obtained by applying the projection layer to the pooled output of Aimv2VisionModel. text_model_output (<class '~modeling_outputs.BaseModelOutputWithPooling'>.text_model_output, defaults to None) — The output of the Aimv2TextModel. vision_model_output (<class '~modeling_outputs.BaseModelOutputWithPooling'>.vision_model_output, defaults to None) — The output of the Aimv2VisionModel.

A transformers.models.aimv2.modeling_aimv2.Aimv2Output or a tuple of torch.FloatTensor (if return_dict=False is passed or when config.return_dict=False) comprising various elements depending on the configuration (None) and inputs.

The Aimv2Model forward method, overrides the __call__ special method.

Although the recipe for forward pass needs to be defined within this function, one should call the Module instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

( input_ids: Tensor attention_mask: torch.Tensor | None = None position_ids: torch.Tensor | None = None **kwargs: typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs] ) → transformers.modeling_outputs.BaseModelOutputWithPooling or tuple(torch.FloatTensor)

Indices can be obtained using AutoTokenizer. See PreTrainedTokenizer.encode() and PreTrainedTokenizer.call() for details.

What are attention masks?

What are position IDs?

transformers.modeling_outputs.BaseModelOutputWithPooling or tuple(torch.FloatTensor)

A transformers.modeling_outputs.BaseModelOutputWithPooling or a tuple of torch.FloatTensor (if return_dict=False is passed or when config.return_dict=False) comprising various elements depending on the configuration (Aimv2Config) and inputs. last_hidden_state (torch.FloatTensor of shape (batch_size, sequence_length, hidden_size)) — Sequence of hidden-states at the output of the last layer of the model. pooler_output (torch.FloatTensor of shape (batch_size, hidden_size)) — Last layer hidden-state of the first token of the sequence (classification token) after further processing through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns the classification token after processing through a linear layer and a tanh activation function. The linear layer weights are trained from the next sentence prediction (classification) objective during pretraining. hidden_states (tuple(torch.FloatTensor), optional, returned when output_hidden_states=True is passed or when config.output_hidden_states=True) — Tuple of torch.FloatTensor (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape (batch_size, sequence_length, hidden_size). Hidden-states of the model at the output of each layer plus the optional initial embedding outputs. attentions (tuple(torch.FloatTensor), optional, returned when output_attentions=True is passed or when config.output_attentions=True) — Tuple of torch.FloatTensor (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length). Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

A transformers.modeling_outputs.BaseModelOutputWithPooling or a tuple of torch.FloatTensor (if return_dict=False is passed or when config.return_dict=False) comprising various elements depending on the configuration (Aimv2Config) and inputs.

last_hidden_state (torch.FloatTensor of shape (batch_size, sequence_length, hidden_size)) — Sequence of hidden-states at the output of the last layer of the model.

pooler_output (torch.FloatTensor of shape (batch_size, hidden_size)) — Last layer hidden-state of the first token of the sequence (classification token) after further processing through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns the classification token after processing through a linear layer and a tanh activation function. The linear layer weights are trained from the next sentence prediction (classification) objective during pretraining.

hidden_states (tuple(torch.FloatTensor), optional, returned when output_hidden_states=True is passed or when config.output_hidden_states=True) — Tuple of torch.FloatTensor (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape (batch_size, sequence_length, hidden_size).

Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.

attentions (tuple(torch.FloatTensor), optional, returned when output_attentions=True is passed or when config.output_attentions=True) — Tuple of torch.FloatTensor (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length).

Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

( pixel_values: FloatTensor interpolate_pos_encoding: bool = False **kwargs: typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs] ) → transformers.modeling_outputs.BaseModelOutputWithPooling or tuple(torch.FloatTensor)

transformers.modeling_outputs.BaseModelOutputWithPooling or tuple(torch.FloatTensor)

A transformers.modeling_outputs.BaseModelOutputWithPooling or a tuple of torch.FloatTensor (if return_dict=False is passed or when config.return_dict=False) comprising various elements depending on the configuration (Aimv2Config) and inputs. last_hidden_state (torch.FloatTensor of shape (batch_size, sequence_length, hidden_size)) — Sequence of hidden-states at the output of the last layer of the model. pooler_output (torch.FloatTensor of shape (batch_size, hidden_size)) — Last layer hidden-state of the first token of the sequence (classification token) after further processing through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns the classification token after processing through a linear layer and a tanh activation function. The linear layer weights are trained from the next sentence prediction (classification) objective during pretraining. hidden_states (tuple(torch.FloatTensor), optional, returned when output_hidden_states=True is passed or when config.output_hidden_states=True) — Tuple of torch.FloatTensor (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape (batch_size, sequence_length, hidden_size). Hidden-states of the model at the output of each layer plus the optional initial embedding outputs. attentions (tuple(torch.FloatTensor), optional, returned when output_attentions=True is passed or when config.output_attentions=True) — Tuple of torch.FloatTensor (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length). Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

A transformers.modeling_outputs.BaseModelOutputWithPooling or a tuple of torch.FloatTensor (if return_dict=False is passed or when config.return_dict=False) comprising various elements depending on the configuration (Aimv2Config) and inputs.

last_hidden_state (torch.FloatTensor of shape (batch_size, sequence_length, hidden_size)) — Sequence of hidden-states at the output of the last layer of the model.

pooler_output (torch.FloatTensor of shape (batch_size, hidden_size)) — Last layer hidden-state of the first token of the sequence (classification token) after further processing through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns the classification token after processing through a linear layer and a tanh activation function. The linear layer weights are trained from the next sentence prediction (classification) objective during pretraining.

hidden_states (tuple(torch.FloatTensor), optional, returned when output_hidden_states=True is passed or when config.output_hidden_states=True) — Tuple of torch.FloatTensor (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape (batch_size, sequence_length, hidden_size).

Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.

attentions (tuple(torch.FloatTensor), optional, returned when output_attentions=True is passed or when config.output_attentions=True) — Tuple of torch.FloatTensor (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length).

Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

( config: Aimv2VisionConfig )

The Vision model from AIMv2 without any head or projection on top.

This model inherits from PreTrainedModel. Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch torch.nn.Module subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

( pixel_values **kwargs: typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs] ) → transformers.modeling_outputs.BaseModelOutputWithPooling or tuple(torch.FloatTensor)

transformers.modeling_outputs.BaseModelOutputWithPooling or tuple(torch.FloatTensor)

A transformers.modeling_outputs.BaseModelOutputWithPooling or a tuple of torch.FloatTensor (if return_dict=False is passed or when config.return_dict=False) comprising various elements depending on the configuration (Aimv2Config) and inputs. last_hidden_state (torch.FloatTensor of shape (batch_size, sequence_length, hidden_size)) — Sequence of hidden-states at the output of the last layer of the model. pooler_output (torch.FloatTensor of shape (batch_size, hidden_size)) — Last layer hidden-state of the first token of the sequence (classification token) after further processing through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns the classification token after processing through a linear layer and a tanh activation function. The linear layer weights are trained from the next sentence prediction (classification) objective during pretraining. hidden_states (tuple(torch.FloatTensor), optional, returned when output_hidden_states=True is passed or when config.output_hidden_states=True) — Tuple of torch.FloatTensor (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape (batch_size, sequence_length, hidden_size). Hidden-states of the model at the output of each layer plus the optional initial embedding outputs. attentions (tuple(torch.FloatTensor), optional, returned when output_attentions=True is passed or when config.output_attentions=True) — Tuple of torch.FloatTensor (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length). Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

A transformers.modeling_outputs.BaseModelOutputWithPooling or a tuple of torch.FloatTensor (if return_dict=False is passed or when config.return_dict=False) comprising various elements depending on the configuration (Aimv2Config) and inputs.

last_hidden_state (torch.FloatTensor of shape (batch_size, sequence_length, hidden_size)) — Sequence of hidden-states at the output of the last layer of the model.

pooler_output (torch.FloatTensor of shape (batch_size, hidden_size)) — Last layer hidden-state of the first token of the sequence (classification token) after further processing through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns the classification token after processing through a linear layer and a tanh activation function. The linear layer weights are trained from the next sentence prediction (classification) objective during pretraining.

hidden_states (tuple(torch.FloatTensor), optional, returned when output_hidden_states=True is passed or when config.output_hidden_states=True) — Tuple of torch.FloatTensor (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape (batch_size, sequence_length, hidden_size).

Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.

attentions (tuple(torch.FloatTensor), optional, returned when output_attentions=True is passed or when config.output_attentions=True) — Tuple of torch.FloatTensor (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length).

Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

The Aimv2VisionModel forward method, overrides the __call__ special method.

Although the recipe for forward pass needs to be defined within this function, one should call the Module instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

( config: Aimv2TextConfig )

The text model from AIMv2 without any head or projection on top.

This model inherits from PreTrainedModel. Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch torch.nn.Module subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

( input_ids attention_mask: torch.Tensor | None = None **kwargs: typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs] ) → transformers.modeling_outputs.BaseModelOutputWithPooling or tuple(torch.FloatTensor)

Indices can be obtained using AutoTokenizer. See PreTrainedTokenizer.encode() and PreTrainedTokenizer.call() for details.

What are attention masks?

transformers.modeling_outputs.BaseModelOutputWithPooling or tuple(torch.FloatTensor)

A transformers.modeling_outputs.BaseModelOutputWithPooling or a tuple of torch.FloatTensor (if return_dict=False is passed or when config.return_dict=False) comprising various elements depending on the configuration (Aimv2Config) and inputs. last_hidden_state (torch.FloatTensor of shape (batch_size, sequence_length, hidden_size)) — Sequence of hidden-states at the output of the last layer of the model. pooler_output (torch.FloatTensor of shape (batch_size, hidden_size)) — Last layer hidden-state of the first token of the sequence (classification token) after further processing through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns the classification token after processing through a linear layer and a tanh activation function. The linear layer weights are trained from the next sentence prediction (classification) objective during pretraining. hidden_states (tuple(torch.FloatTensor), optional, returned when output_hidden_states=True is passed or when config.output_hidden_states=True) — Tuple of torch.FloatTensor (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape (batch_size, sequence_length, hidden_size). Hidden-states of the model at the output of each layer plus the optional initial embedding outputs. attentions (tuple(torch.FloatTensor), optional, returned when output_attentions=True is passed or when config.output_attentions=True) — Tuple of torch.FloatTensor (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length). Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

A transformers.modeling_outputs.BaseModelOutputWithPooling or a tuple of torch.FloatTensor (if return_dict=False is passed or when config.return_dict=False) comprising various elements depending on the configuration (Aimv2Config) and inputs.

last_hidden_state (torch.FloatTensor of shape (batch_size, sequence_length, hidden_size)) — Sequence of hidden-states at the output of the last layer of the model.

pooler_output (torch.FloatTensor of shape (batch_size, hidden_size)) — Last layer hidden-state of the first token of the sequence (classification token) after further processing through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns the classification token after processing through a linear layer and a tanh activation function. The linear layer weights are trained from the next sentence prediction (classification) objective during pretraining.

hidden_states (tuple(torch.FloatTensor), optional, returned when output_hidden_states=True is passed or when config.output_hidden_states=True) — Tuple of torch.FloatTensor (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape (batch_size, sequence_length, hidden_size).

Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.

attentions (tuple(torch.FloatTensor), optional, returned when output_attentions=True is passed or when config.output_attentions=True) — Tuple of torch.FloatTensor (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length).

Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

The Aimv2TextModel forward method, overrides the __call__ special method.

Although the recipe for forward pass needs to be defined within this function, one should call the Module instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

---

## Transformers

**URL:** https://huggingface.co/docs/transformers/v5.0.0/en/main_classes/output

**Contents:**
- Transformers
- Model outputs
- ModelOutput
  - class transformers.utils.ModelOutput
    - to_tuple
- BaseModelOutput
  - class transformers.modeling_outputs.BaseModelOutput
- BaseModelOutputWithPooling
  - class transformers.modeling_outputs.BaseModelOutputWithPooling
- BaseModelOutputWithCrossAttentions

Transformers documentation

and get access to the augmented documentation experience

All models have outputs that are instances of subclasses of ModelOutput. Those are data structures containing all the information returned by the model, but that can also be used as tuples or dictionaries.

Let’s see how this looks in an example:

The outputs object is a SequenceClassifierOutput, as we can see in the documentation of that class below, it means it has an optional loss, a logits, an optional hidden_states and an optional attentions attribute. Here we have the loss since we passed along labels, but we don’t have hidden_states and attentions because we didn’t pass output_hidden_states=True or output_attentions=True.

When passing output_hidden_states=True you may expect the outputs.hidden_states[-1] to match outputs.last_hidden_state exactly. However, this is not always the case. Some models apply normalization or subsequent process to the last hidden state when it’s returned.

You can access each attribute as you would usually do, and if that attribute has not been returned by the model, you will get None. Here for instance outputs.loss is the loss computed by the model, and outputs.attentions is None.

When considering our outputs object as tuple, it only considers the attributes that don’t have None values. Here for instance, it has two elements, loss then logits, so

will return the tuple (outputs.loss, outputs.logits) for instance.

When considering our outputs object as dictionary, it only considers the attributes that don’t have None values. Here for instance, it has two keys that are loss and logits.

We document here the generic model outputs that are used by more than one model type. Specific output types are documented on their corresponding model page.

Base class for all model outputs as dataclass. Has a __getitem__ that allows indexing by integer or slice (like a tuple) or strings (like a dictionary) that will ignore the None attributes. Otherwise behaves like a regular python dictionary.

You can’t unpack a ModelOutput directly. Use the to_tuple() method to convert it to a tuple before.

Convert self to a tuple containing all the attributes/keys that are not None.

( last_hidden_state: torch.FloatTensor | None = None hidden_states: tuple[torch.FloatTensor, ...] | None = None attentions: tuple[torch.FloatTensor, ...] | None = None )

Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.

Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

Base class for model’s outputs, with potential hidden states and attentions.

( last_hidden_state: torch.FloatTensor | None = None pooler_output: torch.FloatTensor | None = None hidden_states: tuple[torch.FloatTensor, ...] | None = None attentions: tuple[torch.FloatTensor, ...] | None = None )

Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.

Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

Base class for model’s outputs that also contains a pooling of the last hidden states.

( last_hidden_state: torch.FloatTensor | None = None hidden_states: tuple[torch.FloatTensor, ...] | None = None attentions: tuple[torch.FloatTensor, ...] | None = None cross_attentions: tuple[torch.FloatTensor, ...] | None = None )

Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.

Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the weighted average in the cross-attention heads.

Base class for model’s outputs, with potential hidden states and attentions.

( last_hidden_state: torch.FloatTensor | None = None pooler_output: torch.FloatTensor | None = None hidden_states: tuple[torch.FloatTensor, ...] | None = None past_key_values: transformers.cache_utils.Cache | None = None attentions: tuple[torch.FloatTensor, ...] | None = None cross_attentions: tuple[torch.FloatTensor, ...] | None = None )

Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.

Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the weighted average in the cross-attention heads.

Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if config.is_encoder_decoder=True in the cross-attention blocks) that can be used (see past_key_values input) to speed up sequential decoding.

Base class for model’s outputs that also contains a pooling of the last hidden states.

( last_hidden_state: torch.FloatTensor | None = None past_key_values: transformers.cache_utils.Cache | None = None hidden_states: tuple[torch.FloatTensor, ...] | None = None attentions: tuple[torch.FloatTensor, ...] | None = None )

If past_key_values is used only the last hidden-state of the sequences of shape (batch_size, 1, hidden_size) is output.

Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if config.is_encoder_decoder=True in the cross-attention blocks) that can be used (see past_key_values input) to speed up sequential decoding.

Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.

Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

Base class for model’s outputs that may also contain a past key/values (to speed up sequential decoding).

( last_hidden_state: torch.FloatTensor | None = None past_key_values: transformers.cache_utils.Cache | None = None hidden_states: tuple[torch.FloatTensor, ...] | None = None attentions: tuple[torch.FloatTensor, ...] | None = None cross_attentions: tuple[torch.FloatTensor, ...] | None = None )

If past_key_values is used only the last hidden-state of the sequences of shape (batch_size, 1, hidden_size) is output.

Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if config.is_encoder_decoder=True in the cross-attention blocks) that can be used (see past_key_values input) to speed up sequential decoding.

Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.

Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the weighted average in the cross-attention heads.

Base class for model’s outputs that may also contain a past key/values (to speed up sequential decoding).

( last_hidden_state: torch.FloatTensor | None = None past_key_values: transformers.cache_utils.EncoderDecoderCache | None = None decoder_hidden_states: tuple[torch.FloatTensor, ...] | None = None decoder_attentions: tuple[torch.FloatTensor, ...] | None = None cross_attentions: tuple[torch.FloatTensor, ...] | None = None encoder_last_hidden_state: torch.FloatTensor | None = None encoder_hidden_states: tuple[torch.FloatTensor, ...] | None = None encoder_attentions: tuple[torch.FloatTensor, ...] | None = None )

If past_key_values is used only the last hidden-state of the sequences of shape (batch_size, 1, hidden_size) is output.

Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention blocks) that can be used (see past_key_values input) to speed up sequential decoding.

Hidden-states of the decoder at the output of each layer plus the optional initial embedding outputs.

Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the self-attention heads.

Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the weighted average in the cross-attention heads.

Hidden-states of the encoder at the output of each layer plus the optional initial embedding outputs.

Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the self-attention heads.

Base class for model encoder’s outputs that also contains : pre-computed hidden states that can speed up sequential decoding.

( loss: torch.FloatTensor | None = None logits: torch.FloatTensor | None = None hidden_states: tuple[torch.FloatTensor, ...] | None = None attentions: tuple[torch.FloatTensor, ...] | None = None )

Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.

Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

Base class for causal language model (or autoregressive) outputs.

( loss: torch.FloatTensor | None = None logits: torch.FloatTensor | None = None past_key_values: transformers.cache_utils.Cache | None = None hidden_states: tuple[torch.FloatTensor, ...] | None = None attentions: tuple[torch.FloatTensor, ...] | None = None cross_attentions: tuple[torch.FloatTensor, ...] | None = None )

Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.

Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

Cross attentions weights after the attention softmax, used to compute the weighted average in the cross-attention heads.

Contains pre-computed hidden-states (key and values in the attention blocks) that can be used (see past_key_values input) to speed up sequential decoding.

Base class for causal language model (or autoregressive) outputs.

( loss: torch.FloatTensor | None = None logits: torch.FloatTensor | None = None past_key_values: transformers.cache_utils.Cache | None = None hidden_states: tuple[torch.FloatTensor, ...] | None = None attentions: tuple[torch.FloatTensor, ...] | None = None )

Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see past_key_values input) to speed up sequential decoding.

Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.

Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

Base class for causal language model (or autoregressive) outputs.

( loss: torch.FloatTensor | None = None logits: torch.FloatTensor | None = None hidden_states: tuple[torch.FloatTensor, ...] | None = None attentions: tuple[torch.FloatTensor, ...] | None = None )

Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.

Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

Base class for masked language models outputs.

( loss: torch.FloatTensor | None = None logits: torch.FloatTensor | None = None past_key_values: transformers.cache_utils.EncoderDecoderCache | None = None decoder_hidden_states: tuple[torch.FloatTensor, ...] | None = None decoder_attentions: tuple[torch.FloatTensor, ...] | None = None cross_attentions: tuple[torch.FloatTensor, ...] | None = None encoder_last_hidden_state: torch.FloatTensor | None = None encoder_hidden_states: tuple[torch.FloatTensor, ...] | None = None encoder_attentions: tuple[torch.FloatTensor, ...] | None = None )

Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention blocks) that can be used (see past_key_values input) to speed up sequential decoding.

Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.

Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the self-attention heads.

Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the weighted average in the cross-attention heads.

Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.

Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the self-attention heads.

Base class for sequence-to-sequence language models outputs.

( loss: torch.FloatTensor | None = None logits: torch.FloatTensor | None = None hidden_states: tuple[torch.FloatTensor, ...] | None = None attentions: tuple[torch.FloatTensor, ...] | None = None )

Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.

Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

Base class for outputs of models predicting if two sentences are consecutive or not.

( loss: torch.FloatTensor | None = None logits: torch.FloatTensor | None = None hidden_states: tuple[torch.FloatTensor, ...] | None = None attentions: tuple[torch.FloatTensor, ...] | None = None )

Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.

Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

Base class for outputs of sentence classification models.

( loss: torch.FloatTensor | None = None logits: torch.FloatTensor | None = None past_key_values: transformers.cache_utils.EncoderDecoderCache | None = None decoder_hidden_states: tuple[torch.FloatTensor, ...] | None = None decoder_attentions: tuple[torch.FloatTensor, ...] | None = None cross_attentions: tuple[torch.FloatTensor, ...] | None = None encoder_last_hidden_state: torch.FloatTensor | None = None encoder_hidden_states: tuple[torch.FloatTensor, ...] | None = None encoder_attentions: tuple[torch.FloatTensor, ...] | None = None )

Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention blocks) that can be used (see past_key_values input) to speed up sequential decoding.

Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.

Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the self-attention heads.

Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the weighted average in the cross-attention heads.

Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.

Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the self-attention heads.

Base class for outputs of sequence-to-sequence sentence classification models.

( loss: torch.FloatTensor | None = None logits: torch.FloatTensor | None = None hidden_states: tuple[torch.FloatTensor, ...] | None = None attentions: tuple[torch.FloatTensor, ...] | None = None )

Classification scores (before SoftMax).

Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.

Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

Base class for outputs of multiple choice models.

( loss: torch.FloatTensor | None = None logits: torch.FloatTensor | None = None hidden_states: tuple[torch.FloatTensor, ...] | None = None attentions: tuple[torch.FloatTensor, ...] | None = None )

Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.

Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

Base class for outputs of token classification models.

( loss: torch.FloatTensor | None = None start_logits: torch.FloatTensor | None = None end_logits: torch.FloatTensor | None = None hidden_states: tuple[torch.FloatTensor, ...] | None = None attentions: tuple[torch.FloatTensor, ...] | None = None )

Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.

Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

Base class for outputs of question answering models.

( loss: torch.FloatTensor | None = None start_logits: torch.FloatTensor | None = None end_logits: torch.FloatTensor | None = None past_key_values: transformers.cache_utils.EncoderDecoderCache | None = None decoder_hidden_states: tuple[torch.FloatTensor, ...] | None = None decoder_attentions: tuple[torch.FloatTensor, ...] | None = None cross_attentions: tuple[torch.FloatTensor, ...] | None = None encoder_last_hidden_state: torch.FloatTensor | None = None encoder_hidden_states: tuple[torch.FloatTensor, ...] | None = None encoder_attentions: tuple[torch.FloatTensor, ...] | None = None )

Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention blocks) that can be used (see past_key_values input) to speed up sequential decoding.

Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.

Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the self-attention heads.

Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the weighted average in the cross-attention heads.

Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.

Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the self-attention heads.

Base class for outputs of sequence-to-sequence question answering models.

( loss: torch.FloatTensor | None = None spectrogram: torch.FloatTensor | None = None past_key_values: transformers.cache_utils.EncoderDecoderCache | None = None decoder_hidden_states: tuple[torch.FloatTensor, ...] | None = None decoder_attentions: tuple[torch.FloatTensor, ...] | None = None cross_attentions: tuple[torch.FloatTensor, ...] | None = None encoder_last_hidden_state: torch.FloatTensor | None = None encoder_hidden_states: tuple[torch.FloatTensor, ...] | None = None encoder_attentions: tuple[torch.FloatTensor, ...] | None = None )

Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention blocks) that can be used (see past_key_values input) to speed up sequential decoding.

Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.

Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the self-attention heads.

Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the weighted average in the cross-attention heads.

Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.

Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the self-attention heads.

Base class for sequence-to-sequence spectrogram outputs.

( loss: torch.FloatTensor | None = None logits: torch.FloatTensor | None = None hidden_states: tuple[torch.FloatTensor, ...] | None = None attentions: tuple[torch.FloatTensor, ...] | None = None )

The logits returned do not necessarily have the same size as the pixel_values passed as inputs. This is to avoid doing two interpolations and lose some quality when a user needs to resize the logits to the original image size as post-processing. You should always check your logits shape and resize as needed.

Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.

Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

Base class for outputs of semantic segmentation models.

( loss: torch.FloatTensor | None = None logits: torch.FloatTensor | None = None hidden_states: tuple[torch.FloatTensor, ...] | None = None attentions: tuple[torch.FloatTensor, ...] | None = None )

Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

Base class for outputs of image classification models.

( loss: torch.FloatTensor | None = None logits: torch.FloatTensor | None = None hidden_states: tuple[torch.FloatTensor, ...] | None = None )

Base class for outputs of image classification models.

( loss: torch.FloatTensor | None = None predicted_depth: torch.FloatTensor | None = None hidden_states: tuple[torch.FloatTensor, ...] | None = None attentions: tuple[torch.FloatTensor, ...] | None = None )

Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.

Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

Base class for outputs of depth estimation models.

( last_hidden_state: torch.FloatTensor | None = None extract_features: torch.FloatTensor | None = None hidden_states: tuple[torch.FloatTensor, ...] | None = None attentions: tuple[torch.FloatTensor, ...] | None = None )

Hidden-states of the model at the output of each layer plus the initial embedding outputs.

Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

Base class for models that have been trained with the Wav2Vec2 loss objective.

( loss: torch.FloatTensor | None = None logits: torch.FloatTensor | None = None embeddings: torch.FloatTensor | None = None hidden_states: tuple[torch.FloatTensor, ...] | None = None attentions: tuple[torch.FloatTensor, ...] | None = None )

Hidden-states of the model at the output of each layer plus the initial embedding outputs.

Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

Output type of Wav2Vec2ForXVector.

( last_hidden_state: torch.FloatTensor | None = None past_key_values: transformers.cache_utils.EncoderDecoderCache | None = None decoder_hidden_states: tuple[torch.FloatTensor, ...] | None = None decoder_attentions: tuple[torch.FloatTensor, ...] | None = None cross_attentions: tuple[torch.FloatTensor, ...] | None = None encoder_last_hidden_state: torch.FloatTensor | None = None encoder_hidden_states: tuple[torch.FloatTensor, ...] | None = None encoder_attentions: tuple[torch.FloatTensor, ...] | None = None loc: torch.FloatTensor | None = None scale: torch.FloatTensor | None = None static_features: torch.FloatTensor | None = None )

If past_key_values is used only the last hidden-state of the sequences of shape (batch_size, 1, hidden_size) is output.

Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention blocks) that can be used (see past_key_values input) to speed up sequential decoding.

Hidden-states of the decoder at the output of each layer plus the optional initial embedding outputs.

Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the self-attention heads.

Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the weighted average in the cross-attention heads.

Hidden-states of the encoder at the output of each layer plus the optional initial embedding outputs.

Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the self-attention heads.

Base class for time series model’s encoder outputs that also contains pre-computed hidden states that can speed up sequential decoding.

( loss: torch.FloatTensor | None = None params: tuple[torch.FloatTensor, ...] | None = None past_key_values: transformers.cache_utils.EncoderDecoderCache | None = None decoder_hidden_states: tuple[torch.FloatTensor, ...] | None = None decoder_attentions: tuple[torch.FloatTensor, ...] | None = None cross_attentions: tuple[torch.FloatTensor, ...] | None = None encoder_last_hidden_state: torch.FloatTensor | None = None encoder_hidden_states: tuple[torch.FloatTensor, ...] | None = None encoder_attentions: tuple[torch.FloatTensor, ...] | None = None loc: torch.FloatTensor | None = None scale: torch.FloatTensor | None = None static_features: torch.FloatTensor | None = None )

Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention blocks) that can be used (see past_key_values input) to speed up sequential decoding.

Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.

Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the self-attention heads.

Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the weighted average in the cross-attention heads.

Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.

Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the self-attention heads.

Base class for time series model’s decoder outputs that also contain the loss as well as the parameters of the chosen distribution.

( sequences: torch.FloatTensor | None = None )

Base class for time series model’s predictions outputs that contains the sampled values from the chosen distribution.

---

## Transformers

**URL:** https://huggingface.co/docs/transformers/v5.0.0/en/main_classes/optimizer_schedules

**Contents:**
- Transformers
- Optimization
- AdaFactor
  - class transformers.Adafactor
    - step
- Schedules
  - SchedulerType
  - class transformers.SchedulerType
  - get_scheduler
    - transformers.get_scheduler

Transformers documentation

and get access to the augmented documentation experience

The .optimization module provides:

( params lr = None eps = (1e-30, 0.001) clip_threshold = 1.0 decay_rate = -0.8 beta1 = None weight_decay = 0.0 scale_parameter = True relative_step = True warmup_init = False )

AdaFactor pytorch implementation can be used as a drop in replacement for Adam original fairseq code: https://github.com/pytorch/fairseq/blob/master/fairseq/optim/adafactor.py

Paper: Adafactor: Adaptive Learning Rates with Sublinear Memory Cost https://huggingface.co/papers/1804.04235 Note that this optimizer internally adjusts the learning rate depending on the scale_parameter, relative_step and warmup_init options. To use a manual (external) learning rate schedule you should set scale_parameter=False and relative_step=False.

This implementation handles low-precision (FP16, bfloat) values, but we have not thoroughly tested.

Recommended T5 finetuning settings (https://discuss.huggingface.co/t/t5-finetuning-tips/684/3):

Training without LR warmup or clip_threshold is not recommended.

Disable relative updates

Use scale_parameter=False

Additional optimizer operations like gradient clipping should not be used alongside Adafactor

Others reported the following combination to work well:

When using lr=None with Trainer you will most likely need to use AdafactorSchedule

scheduler as following:

Performs a single optimization step

( value names = None module = None qualname = None type = None start = 1 )

Scheduler names for the parameter lr_scheduler_type in TrainingArguments. By default, it uses “linear”. Internally, this retrieves get_linear_schedule_with_warmup scheduler from Trainer. Scheduler types:

( name: str | transformers.trainer_utils.SchedulerType optimizer: Optimizer num_warmup_steps: int | None = None num_training_steps: int | None = None scheduler_specific_kwargs: dict | None = None )

Unified API to get any scheduler from its name.

( optimizer: Optimizer last_epoch: int = -1 )

Create a schedule with a constant learning rate, using the learning rate set in optimizer.

( optimizer: Optimizer num_warmup_steps: int last_epoch: int = -1 )

Create a schedule with a constant learning rate preceded by a warmup period during which the learning rate increases linearly between 0 and the initial lr set in the optimizer.

( optimizer: Optimizer num_warmup_steps: int num_training_steps: int num_cycles: float = 0.5 last_epoch: int = -1 )

Create a schedule with a learning rate that decreases following the values of the cosine function between the initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the initial lr set in the optimizer.

( optimizer: Optimizer num_warmup_steps: int num_training_steps: int num_cycles: int = 1 last_epoch: int = -1 )

Create a schedule with a learning rate that decreases following the values of the cosine function between the initial lr set in the optimizer to 0, with several hard restarts, after a warmup period during which it increases linearly between 0 and the initial lr set in the optimizer.

( optimizer: Optimizer num_warmup_steps: int num_training_steps: int num_cycles: float = 0.5 last_epoch: int = -1 min_lr: float | None = None min_lr_rate: float | None = None )

Create a schedule with a learning rate that decreases following the values of the cosine function between the initial lr set in the optimizer to min_lr, after a warmup period during which it increases linearly between 0 and the initial lr set in the optimizer.

( optimizer: Optimizer num_warmup_steps: int num_training_steps: int num_cycles: float = 0.5 last_epoch: int = -1 min_lr: float | None = None min_lr_rate: float | None = None warmup_lr_rate: float | None = None )

Create a schedule with a learning rate that decreases following the values of the cosine function between the initial lr set in the optimizer to min_lr, after a warmup period during which it increases linearly between 0 and the initial lr set in the optimizer.

( optimizer num_warmup_steps num_training_steps last_epoch = -1 )

Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.

( optimizer num_warmup_steps num_training_steps lr_end = 1e-07 power = 1.0 last_epoch = -1 )

Create a schedule with a learning rate that decreases as a polynomial decay from the initial lr set in the optimizer to end lr defined by lr_end, after a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.

Note: power defaults to 1.0 as in the fairseq implementation, which in turn is based on the original BERT implementation at https://github.com/google-research/bert/blob/f39e881b169b9d53bea03d2d341b31707a6c052b/optimization.py#L37

( optimizer: Optimizer num_warmup_steps: int timescale: int | None = None last_epoch: int = -1 )

Create a schedule with an inverse square-root learning rate, from the initial lr set in the optimizer, after a warmup period which increases lr linearly from 0 to the initial lr set in the optimizer.

( optimizer: Optimizer **kwargs )

Create a schedule with a constant learning rate that decreases when a metric has stopped improving.

( optimizer: Optimizer num_warmup_steps: int num_decay_steps: int num_training_steps: int | None = None num_stable_steps: int | None = None warmup_type: str = 'linear' decay_type: str = 'cosine' min_lr_ratio: float = 0 num_cycles: float = 0.5 last_epoch: int = -1 )

Create a schedule with a learning rate that has three stages:

---

## Transformers

**URL:** https://huggingface.co/docs/transformers/v5.0.0/en/model_doc/llama2

**Contents:**
- Transformers
- Llama 2
- Notes
- LlamaConfig
  - class transformers.LlamaConfig
- LlamaTokenizer
  - class transformers.LlamaTokenizer
    - get_special_tokens_mask
    - save_vocabulary
- LlamaTokenizerFast

Transformers documentation

and get access to the augmented documentation experience

This model was released on 2023-07-18 and added to Hugging Face Transformers on 2023-07-18.

Llama 2 is a family of large language models, Llama 2 and Llama 2-Chat, available in 7B, 13B, and 70B parameters. The Llama 2 model mostly keeps the same architecture as Llama, but it is pretrained on more tokens, doubles the context length, and uses grouped-query attention (GQA) in the 70B model to improve inference.

Llama 2-Chat is trained with supervised fine-tuning (SFT), and reinforcement learning with human feedback (RLHF) - rejection sampling and proximal policy optimization (PPO) - is applied to the fine-tuned model to align the chat model with human preferences.

You can find all the original Llama 2 checkpoints under the Llama 2 Family collection.

Click on the Llama 2 models in the right sidebar for more examples of how to apply Llama to different language tasks.

The example below demonstrates how to generate text with Pipeline, AutoModel, and how to chat with Llama 2-Chat from the command line.

Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the Quantization overview for more available quantization backends.

The example below uses torchao to only quantize the weights to int4.

Use the AttentionMaskVisualizer to better understand what tokens the model can and cannot attend to.

Setting config.pretraining_tp to a value besides 1 activates a more accurate but slower computation of the linear layers. This matches the original logits better.

The original model uses pad_id = -1 to indicate a padding token. The Transformers implementation requires adding a padding token and resizing the token embedding accordingly.

It is recommended to initialize the embed_tokens layer with the following code to ensure encoding the padding token outputs zeros.

The tokenizer is a byte-pair encoding model based on SentencePiece. During decoding, if the first token is the start of the word (for example, “Banana”), the tokenizer doesn’t prepend the prefix space to the string.

Don’t use the dtype parameter in from_pretrained() if you’re using FlashAttention-2 because it only supports fp16 or bf16. You should use Automatic Mixed Precision, set fp16 or bf16 to True if using Trainer, or use torch.autocast.

( vocab_size: int | None = 32000 hidden_size: int | None = 4096 intermediate_size: int | None = 11008 num_hidden_layers: int | None = 32 num_attention_heads: int | None = 32 num_key_value_heads: int | None = None hidden_act: str | None = 'silu' max_position_embeddings: int | None = 2048 initializer_range: float | None = 0.02 rms_norm_eps: int | None = 1e-06 use_cache: bool | None = True pad_token_id: int | None = None bos_token_id: int | None = 1 eos_token_id: int | None = 2 pretraining_tp: int | None = 1 tie_word_embeddings: bool | None = False rope_parameters: transformers.modeling_rope_utils.RopeParameters | dict[str, transformers.modeling_rope_utils.RopeParameters] | None = None attention_bias: bool | None = False attention_dropout: float | None = 0.0 mlp_bias: bool | None = False head_dim: int | None = None **kwargs )

This is the configuration class to store the configuration of a LlamaModel. It is used to instantiate an LLaMA model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the LLaMA-7B. e.g. meta-llama/Llama-2-7b-hf

Configuration objects inherit from PreTrainedConfig and can be used to control the model outputs. Read the documentation from PreTrainedConfig for more information.

( vocab: str | dict | list | None = None merges: str | list | None = None clean_up_tokenization_spaces = False unk_token = '<unk>' bos_token = '<s>' eos_token = '</s>' use_default_system_prompt = False legacy = False add_prefix_space = None **kwargs )

Construct a Llama tokenizer. Based on byte-level Byte-Pair-Encoding.

This uses notably ByteFallback and no normalization.

If you want to change the bos_token or the eos_token, make sure to specify them when initializing the model, or call tokenizer.update_post_processor() to make sure that the post-processing is correctly done (otherwise the values of the first token and final token of an encoded sequence will not be correct). For more details, checkout [post-processors] (https://huggingface.co/docs/tokenizers/api/post-processors) documentation.

This tokenizer inherits from PreTrainedTokenizerFast which contains most of the main methods. Users should refer to this superclass for more information regarding those methods.

( token_ids_0: list[int] token_ids_1: list[int] | None = None already_has_special_tokens: bool = False ) → A list of integers in the range [0, 1]

A list of integers in the range [0, 1]

1 for a special token, 0 for a sequence token.

1 for a special token, 0 for a sequence token.

Retrieve sequence ids from a token list that has no special tokens added.

For fast tokenizers, data collators call this with already_has_special_tokens=True to build a mask over an already-formatted sequence. In that case, we compute the mask by checking membership in all_special_ids.

( save_directory: str filename_prefix: str | None = None )

( vocab: str | dict | list | None = None merges: str | list | None = None clean_up_tokenization_spaces = False unk_token = '<unk>' bos_token = '<s>' eos_token = '</s>' use_default_system_prompt = False legacy = False add_prefix_space = None **kwargs )

Construct a Llama tokenizer. Based on byte-level Byte-Pair-Encoding.

This uses notably ByteFallback and no normalization.

If you want to change the bos_token or the eos_token, make sure to specify them when initializing the model, or call tokenizer.update_post_processor() to make sure that the post-processing is correctly done (otherwise the values of the first token and final token of an encoded sequence will not be correct). For more details, checkout [post-processors] (https://huggingface.co/docs/tokenizers/api/post-processors) documentation.

This tokenizer inherits from PreTrainedTokenizerFast which contains most of the main methods. Users should refer to this superclass for more information regarding those methods.

( token_ids_0: list[int] token_ids_1: list[int] | None = None already_has_special_tokens: bool = False ) → A list of integers in the range [0, 1]

A list of integers in the range [0, 1]

1 for a special token, 0 for a sequence token.

1 for a special token, 0 for a sequence token.

Retrieve sequence ids from a token list that has no special tokens added.

For fast tokenizers, data collators call this with already_has_special_tokens=True to build a mask over an already-formatted sequence. In that case, we compute the mask by checking membership in all_special_ids.

Updates the underlying post processor with the current bos_token and eos_token.

( save_directory: str filename_prefix: str | None = None )

( config: LlamaConfig )

The bare Llama Model outputting raw hidden-states without any specific head on top.

This model inherits from PreTrainedModel. Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch torch.nn.Module subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

( input_ids: torch.LongTensor | None = None attention_mask: torch.Tensor | None = None position_ids: torch.LongTensor | None = None past_key_values: transformers.cache_utils.Cache | None = None inputs_embeds: torch.FloatTensor | None = None cache_position: torch.LongTensor | None = None use_cache: bool | None = None **kwargs: typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs] ) → transformers.modeling_outputs.BaseModelOutputWithPast or tuple(torch.FloatTensor)

Indices can be obtained using AutoTokenizer. See PreTrainedTokenizer.encode() and PreTrainedTokenizer.call() for details.

What are attention masks?

What are position IDs?

Only Cache instance is allowed as input, see our kv cache guide. If no past_key_values are passed, DynamicCache will be initialized by default.

The model will output the same cache format that is fed as input.

If past_key_values are used, the user is expected to input only unprocessed input_ids (those that don’t have their past key value states given to this model) of shape (batch_size, unprocessed_length) instead of all input_ids of shape (batch_size, sequence_length).

transformers.modeling_outputs.BaseModelOutputWithPast or tuple(torch.FloatTensor)

A transformers.modeling_outputs.BaseModelOutputWithPast or a tuple of torch.FloatTensor (if return_dict=False is passed or when config.return_dict=False) comprising various elements depending on the configuration (LlamaConfig) and inputs. last_hidden_state (torch.FloatTensor of shape (batch_size, sequence_length, hidden_size)) — Sequence of hidden-states at the output of the last layer of the model. If past_key_values is used only the last hidden-state of the sequences of shape (batch_size, 1, hidden_size) is output. past_key_values (Cache, optional, returned when use_cache=True is passed or when config.use_cache=True) — It is a Cache instance. For more details, see our kv cache guide. Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if config.is_encoder_decoder=True in the cross-attention blocks) that can be used (see past_key_values input) to speed up sequential decoding. hidden_states (tuple(torch.FloatTensor), optional, returned when output_hidden_states=True is passed or when config.output_hidden_states=True) — Tuple of torch.FloatTensor (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape (batch_size, sequence_length, hidden_size). Hidden-states of the model at the output of each layer plus the optional initial embedding outputs. attentions (tuple(torch.FloatTensor), optional, returned when output_attentions=True is passed or when config.output_attentions=True) — Tuple of torch.FloatTensor (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length). Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

A transformers.modeling_outputs.BaseModelOutputWithPast or a tuple of torch.FloatTensor (if return_dict=False is passed or when config.return_dict=False) comprising various elements depending on the configuration (LlamaConfig) and inputs.

last_hidden_state (torch.FloatTensor of shape (batch_size, sequence_length, hidden_size)) — Sequence of hidden-states at the output of the last layer of the model.

If past_key_values is used only the last hidden-state of the sequences of shape (batch_size, 1, hidden_size) is output.

past_key_values (Cache, optional, returned when use_cache=True is passed or when config.use_cache=True) — It is a Cache instance. For more details, see our kv cache guide.

Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if config.is_encoder_decoder=True in the cross-attention blocks) that can be used (see past_key_values input) to speed up sequential decoding.

hidden_states (tuple(torch.FloatTensor), optional, returned when output_hidden_states=True is passed or when config.output_hidden_states=True) — Tuple of torch.FloatTensor (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape (batch_size, sequence_length, hidden_size).

Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.

attentions (tuple(torch.FloatTensor), optional, returned when output_attentions=True is passed or when config.output_attentions=True) — Tuple of torch.FloatTensor (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length).

Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

The LlamaModel forward method, overrides the __call__ special method.

Although the recipe for forward pass needs to be defined within this function, one should call the Module instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

The Llama Model for causal language modeling.

This model inherits from PreTrainedModel. Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch torch.nn.Module subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

( input_ids: torch.LongTensor | None = None attention_mask: torch.Tensor | None = None position_ids: torch.LongTensor | None = None past_key_values: transformers.cache_utils.Cache | None = None inputs_embeds: torch.FloatTensor | None = None labels: torch.LongTensor | None = None use_cache: bool | None = None cache_position: torch.LongTensor | None = None logits_to_keep: int | torch.Tensor = 0 **kwargs: typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs] ) → transformers.modeling_outputs.CausalLMOutputWithPast or tuple(torch.FloatTensor)

Indices can be obtained using AutoTokenizer. See PreTrainedTokenizer.encode() and PreTrainedTokenizer.call() for details.

What are attention masks?

What are position IDs?

Only Cache instance is allowed as input, see our kv cache guide. If no past_key_values are passed, DynamicCache will be initialized by default.

The model will output the same cache format that is fed as input.

If past_key_values are used, the user is expected to input only unprocessed input_ids (those that don’t have their past key value states given to this model) of shape (batch_size, unprocessed_length) instead of all input_ids of shape (batch_size, sequence_length).

transformers.modeling_outputs.CausalLMOutputWithPast or tuple(torch.FloatTensor)

A transformers.modeling_outputs.CausalLMOutputWithPast or a tuple of torch.FloatTensor (if return_dict=False is passed or when config.return_dict=False) comprising various elements depending on the configuration (LlamaConfig) and inputs. loss (torch.FloatTensor of shape (1,), optional, returned when labels is provided) — Language modeling loss (for next-token prediction). logits (torch.FloatTensor of shape (batch_size, sequence_length, config.vocab_size)) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax). past_key_values (Cache, optional, returned when use_cache=True is passed or when config.use_cache=True) — It is a Cache instance. For more details, see our kv cache guide. Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see past_key_values input) to speed up sequential decoding. hidden_states (tuple(torch.FloatTensor), optional, returned when output_hidden_states=True is passed or when config.output_hidden_states=True) — Tuple of torch.FloatTensor (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape (batch_size, sequence_length, hidden_size). Hidden-states of the model at the output of each layer plus the optional initial embedding outputs. attentions (tuple(torch.FloatTensor), optional, returned when output_attentions=True is passed or when config.output_attentions=True) — Tuple of torch.FloatTensor (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length). Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

A transformers.modeling_outputs.CausalLMOutputWithPast or a tuple of torch.FloatTensor (if return_dict=False is passed or when config.return_dict=False) comprising various elements depending on the configuration (LlamaConfig) and inputs.

loss (torch.FloatTensor of shape (1,), optional, returned when labels is provided) — Language modeling loss (for next-token prediction).

logits (torch.FloatTensor of shape (batch_size, sequence_length, config.vocab_size)) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).

past_key_values (Cache, optional, returned when use_cache=True is passed or when config.use_cache=True) — It is a Cache instance. For more details, see our kv cache guide.

Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see past_key_values input) to speed up sequential decoding.

hidden_states (tuple(torch.FloatTensor), optional, returned when output_hidden_states=True is passed or when config.output_hidden_states=True) — Tuple of torch.FloatTensor (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape (batch_size, sequence_length, hidden_size).

Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.

attentions (tuple(torch.FloatTensor), optional, returned when output_attentions=True is passed or when config.output_attentions=True) — Tuple of torch.FloatTensor (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length).

Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

The LlamaForCausalLM forward method, overrides the __call__ special method.

Although the recipe for forward pass needs to be defined within this function, one should call the Module instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

( input_ids: torch.LongTensor | None = None attention_mask: torch.Tensor | None = None position_ids: torch.LongTensor | None = None past_key_values: transformers.cache_utils.Cache | None = None inputs_embeds: torch.FloatTensor | None = None labels: torch.LongTensor | None = None use_cache: bool | None = None **kwargs: typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs] ) → transformers.modeling_outputs.SequenceClassifierOutputWithPast or tuple(torch.FloatTensor)

Indices can be obtained using AutoTokenizer. See PreTrainedTokenizer.encode() and PreTrainedTokenizer.call() for details.

What are attention masks?

What are position IDs?

Only Cache instance is allowed as input, see our kv cache guide. If no past_key_values are passed, DynamicCache will be initialized by default.

The model will output the same cache format that is fed as input.

If past_key_values are used, the user is expected to input only unprocessed input_ids (those that don’t have their past key value states given to this model) of shape (batch_size, unprocessed_length) instead of all input_ids of shape (batch_size, sequence_length).

transformers.modeling_outputs.SequenceClassifierOutputWithPast or tuple(torch.FloatTensor)

A transformers.modeling_outputs.SequenceClassifierOutputWithPast or a tuple of torch.FloatTensor (if return_dict=False is passed or when config.return_dict=False) comprising various elements depending on the configuration (None) and inputs. loss (torch.FloatTensor of shape (1,), optional, returned when labels is provided) — Classification (or regression if config.num_labels==1) loss. logits (torch.FloatTensor of shape (batch_size, config.num_labels)) — Classification (or regression if config.num_labels==1) scores (before SoftMax). past_key_values (Cache, optional, returned when use_cache=True is passed or when config.use_cache=True) — It is a Cache instance. For more details, see our kv cache guide. Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see past_key_values input) to speed up sequential decoding. hidden_states (tuple(torch.FloatTensor), optional, returned when output_hidden_states=True is passed or when config.output_hidden_states=True) — Tuple of torch.FloatTensor (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape (batch_size, sequence_length, hidden_size). Hidden-states of the model at the output of each layer plus the optional initial embedding outputs. attentions (tuple(torch.FloatTensor), optional, returned when output_attentions=True is passed or when config.output_attentions=True) — Tuple of torch.FloatTensor (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length). Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

A transformers.modeling_outputs.SequenceClassifierOutputWithPast or a tuple of torch.FloatTensor (if return_dict=False is passed or when config.return_dict=False) comprising various elements depending on the configuration (None) and inputs.

loss (torch.FloatTensor of shape (1,), optional, returned when labels is provided) — Classification (or regression if config.num_labels==1) loss.

logits (torch.FloatTensor of shape (batch_size, config.num_labels)) — Classification (or regression if config.num_labels==1) scores (before SoftMax).

past_key_values (Cache, optional, returned when use_cache=True is passed or when config.use_cache=True) — It is a Cache instance. For more details, see our kv cache guide.

Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see past_key_values input) to speed up sequential decoding.

hidden_states (tuple(torch.FloatTensor), optional, returned when output_hidden_states=True is passed or when config.output_hidden_states=True) — Tuple of torch.FloatTensor (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape (batch_size, sequence_length, hidden_size).

Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.

attentions (tuple(torch.FloatTensor), optional, returned when output_attentions=True is passed or when config.output_attentions=True) — Tuple of torch.FloatTensor (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length).

Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

The GenericForSequenceClassification forward method, overrides the __call__ special method.

Although the recipe for forward pass needs to be defined within this function, one should call the Module instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

---

## Transformers

**URL:** https://huggingface.co/docs/transformers/v5.0.0/en/main_classes/quantization

**Contents:**
- Transformers
- Quantization
- QuantoConfig
  - class transformers.QuantoConfig
    - post_init
- AqlmConfig
  - class transformers.AqlmConfig
    - post_init
- VptqConfig
  - class transformers.VptqConfig

Transformers documentation

and get access to the augmented documentation experience

Quantization techniques reduce memory and computational costs by representing weights and activations with lower-precision data types like 8-bit integers (int8). This enables loading larger models you normally wouldn’t be able to fit into memory, and speeding up inference. Transformers supports the AWQ and GPTQ quantization algorithms and it supports 8-bit and 4-bit quantization with bitsandbytes.

Quantization techniques that aren’t supported in Transformers can be added with the HfQuantizer class.

Learn how to quantize models in the Quantization guide.

( weights = 'int8' activations = None modules_to_not_convert: list | None = None **kwargs )

This is a wrapper class about all possible attributes and features that you can play with a model that has been loaded using quanto.

Safety checker that arguments are correct

( in_group_size: int = 8 out_group_size: int = 1 num_codebooks: int = 1 nbits_per_codebook: int = 16 linear_weights_not_to_quantize: list[str] | None = None **kwargs )

This is a wrapper class about aqlm parameters.

Safety checker that arguments are correct - also replaces some NoneType arguments with their default values.

( enable_proxy_error: bool = False config_for_layers: dict = {} shared_layer_config: dict = {} modules_to_not_convert: list | None = None **kwargs )

This is a wrapper class about vptq parameters.

Safety checker that arguments are correct

( bits: int = 4 group_size: int = 128 zero_point: bool = True backend: AwqBackend = <AwqBackend.AUTO: 'auto'> modules_to_not_convert: list | None = None **kwargs )

This is a wrapper class about all possible attributes and features that you can play with a model that has been loaded using auto-awq library awq quantization relying on auto_awq backend.

( weights: str = 'int8' modules_to_not_convert: list | None = None **kwargs )

This is a wrapper class about all possible attributes and features that you can play with a model that has been loaded using eetq.

Safety checker that arguments are correct

( bits: int tokenizer: typing.Any = None dataset: list[str] | str | None = None group_size: int = 128 damp_percent: float = 0.1 desc_act: bool = False act_group_aware: bool = True sym: bool = True true_sequential: bool = True format: str = 'gptq' meta: dict[str, typing.Any] | None = None backend: str | None = None model_seqlen: int | None = None block_name_to_quantize: str | None = None module_name_preceding_first_block: list[str] | None = None batch_size: int = 1 pad_token_id: int | None = None max_input_length: int | None = None cache_block_outputs: bool = True modules_in_block_to_quantize: list[list[str]] | None = None **kwargs )

This is a wrapper class about all possible attributes and features that you can play with a model that has been loaded using optimum api for GPTQ quantization relying on the gptqmodel backend.

Get compatible class with optimum gptq config dict

Safety checker that arguments are correct

Get compatible dict for optimum gptq config

( load_in_8bit = False load_in_4bit = False llm_int8_threshold = 6.0 llm_int8_skip_modules = None llm_int8_enable_fp32_cpu_offload = False llm_int8_has_fp16_weight = False bnb_4bit_compute_dtype = None bnb_4bit_quant_type = 'fp4' bnb_4bit_use_double_quant = False bnb_4bit_quant_storage = None **kwargs )

This is a wrapper class about all possible attributes and features that you can play with a model that has been loaded using bitsandbytes.

Currently only supports LLM.int8(), FP4, and NF4 quantization. If more methods are added to bitsandbytes, then more arguments will be added to this class.

Returns True if the model is quantizable, False otherwise.

Safety checker that arguments are correct - also replaces some NoneType arguments with their default values.

This method returns the quantization method used for the model. If the model is not quantizable, it returns None.

Dictionary of all the attributes that make up this configuration instance,

Dictionary of all the attributes that make up this configuration instance,

Removes all attributes from config which correspond to the default config attributes for better readability and serializes to a Python dictionary.

( quantization_config: QuantizationConfigMixin **kwargs )

Abstract class of the HuggingFace quantizer. Supports for now quantizing HF transformers models for inference and/or quantization. This class is used only for transformers.PreTrainedModel.from_pretrained and cannot be easily used outside the scope of that method yet.

Attributes quantization_config (transformers.utils.quantization_config.QuantizationConfigMixin): The quantization config that defines the quantization parameters of your model that you want to quantize. requires_calibration (bool): Whether the quantization method requires to calibrate the model before using it.

adjust max_memory argument for infer_auto_device_map() if extra memory is needed for quantization

( model dtype = None )

Potentially dequantize the model to retrieve the original model, with some loss in accuracy / performance. Note not all quantization schemes support this.

Override this method if you want to adjust the param_name.

Get state dict and metadata. Useful when we need to modify a bit the state dict due to quantization

( model: PreTrainedModel param_name: str **kwargs )

Check whether a given param needs to be quantized.

( model: PreTrainedModel **kwargs )

Post-process the model post weights loading. Make sure to override the abstract method _process_model_after_weight_loading.

( model: PreTrainedModel dtype = None **kwargs )

Setting model attributes and/or converting model before weights loading. At this point the model should be initialized on the meta device so you can freely manipulate the skeleton of the model in order to replace modules in-place. Make sure to override the abstract method _process_model_before_weight_loading.

Remove the quantization config from the model.

( device_map: dict[str, typing.Any] | None )

Override this method if you want to pass a override the existing device map with a new one. E.g. for bitsandbytes, since accelerate is a hard requirement, if no device_map is passed, the device_map is set to `“auto”“

( dtype: torch.dtype )

Some quantization methods require to explicitly set the dtype of the model to a target dtype. You need to override this method in case you want to make sure that behavior is preserved

updates the tp plan for the scales

updates the tp plan for the scales

This method is used to potentially check for potential conflicts with arguments that are passed in from_pretrained. You need to define it for all future quantizers that are integrated with transformers. If no explicit check are needed, simply return nothing.

( bits: int = 4 p: int = 2 modules_to_not_convert: list[str] | None = None hadamard_size: int = 512 group_size: int = 256 tune_metadata: dict[str, typing.Any] | None = None **kwargs )

HiggsConfig is a configuration class for quantization using the HIGGS method.

Safety checker that arguments are correct - also replaces some NoneType arguments with their default values.

( nbits: int = 4 group_size: int = 64 view_as_float: bool = False axis: int | None = None dynamic_config: dict | None = None skip_modules: list = ['lm_head'] **kwargs )

This is wrapper around hqq’s BaseQuantizeConfig.

Override from_dict, used in AutoQuantizationConfig.from_dict in quantizers/auto.py

Safety checker that arguments are correct - also replaces some NoneType arguments with their default values.

Dictionary of all the attributes that make up this configuration instance,

Dictionary of all the attributes that make up this configuration instance,

Removes all attributes from config which correspond to the default config attributes for better readability and serializes to a Python dictionary.

( modules_to_not_convert: list | None = None dequantize: bool = False **kwargs )

This is a wrapper class about all possible attributes and features that you can play with a model that has been loaded using mxfp4 quantization.

( activation_scale_ub: float = 1200.0 modules_to_not_convert: list | None = None **kwargs )

This is a wrapper class about all possible attributes and features that you can play with a model that has been loaded using fbgemm fp8 quantization.

( config_groups: dict[str, typing.Union[ForwardRef('QuantizationScheme'), list[str]]] | None = None format: str = 'dense' quantization_status: QuantizationStatus = 'initialized' kv_cache_scheme: typing.Optional[ForwardRef('QuantizationArgs')] = None global_compression_ratio: float | None = None ignore: list[str] | None = None sparsity_config: dict[str, typing.Any] | None = None quant_method: str = 'compressed-tensors' run_compressed: bool = True **kwargs )

This is a wrapper class that handles compressed-tensors quantization config options. It is a wrapper around compressed_tensors.QuantizationConfig

( config_dict return_unused_kwargs = False **kwargs ) → QuantizationConfigMixin

QuantizationConfigMixin

The configuration object instantiated from those parameters.

The configuration object instantiated from those parameters.

Instantiates a CompressedTensorsConfig from a Python dictionary of parameters. Optionally unwraps any args from the nested quantization_config

Quantization config to be added to config.json

Serializes this instance to a Python dictionary. Returns: dict[str, Any]: Dictionary of all the attributes that make up this configuration instance.

Dictionary of all the attributes that make up this configuration instance,

Dictionary of all the attributes that make up this configuration instance,

Removes all attributes from config which correspond to the default config attributes for better readability and serializes to a Python dictionary.

( quant_type: typing.Union[str, ForwardRef('AOBaseConfig')] modules_to_not_convert: list | None = None include_input_output_embeddings: bool = False untie_embedding_weights: bool = False **kwargs )

( config_dict return_unused_kwargs = False **kwargs )

Create configuration from a dictionary.

Create the appropriate quantization method based on configuration.

Validate configuration and set defaults.

Convert configuration to a dictionary.

( modules_to_not_convert: list | None = None linear_class: str = 'bitlinear' quantization_mode: str = 'offline' use_rms_norm: bool = False rms_norm_eps: float | None = 1e-06 **kwargs )

Configuration class for applying BitNet quantization.

Safety checker that arguments are correct

( bits: int = 3 beta1: int = 16 beta2: int = 16 shapes: dict[str, int] | None = None modules_to_not_convert: list[str] | None = None **kwargs )

This is a wrapper class about spqr parameters. Refer to the original publication for more details.

Safety checker that arguments are correct - also replaces some NoneType arguments with their default values.

( activation_scheme: str = 'dynamic' weight_block_size: tuple = (128, 128) dequantize: bool = False modules_to_not_convert: list | None = None **kwargs )

FineGrainedFP8Config is a configuration class for fine-grained FP8 quantization used mainly for deepseek models.

Safety checker that arguments are correct

( forward_dtype: str = 'nvfp4' forward_method: str = 'abs_max' backward_dtype: str = 'bf16' store_master_weights: bool = False hadamard_group_size: int | None = None pseudoquantization: bool = False transform_init: str = 'hadamard' modules_to_not_convert: list[str] | None = None **kwargs )

FPQuantConfig is a configuration class for quantization using the FPQuant method.

Safety checker that arguments are correct - also replaces some NoneType arguments with their default values.

( bits: int = 4 group_size: int = 128 sym: bool = True backend: str = 'auto' **kwargs )

This is a wrapper class about all possible attributes and features that you can play with a model that has been loaded AutoRound quantization.

Safety checker that arguments are correct.

---

## Transformers

**URL:** https://huggingface.co/docs/transformers/v5.0.0/en/model_doc/auto

**Contents:**
- Transformers
- Auto Classes
- Extending the Auto Classes
- AutoConfig
  - class transformers.AutoConfig
    - from_pretrained
    - register
- AutoTokenizer
  - class transformers.AutoTokenizer
    - from_pretrained

Transformers documentation

and get access to the augmented documentation experience

In many cases, the architecture you want to use can be guessed from the name or the path of the pretrained model you are supplying to the from_pretrained() method. AutoClasses are here to do this job for you so that you automatically retrieve the relevant model given the name/path to the pretrained weights/config/vocabulary.

Instantiating one of AutoConfig, AutoModel, and AutoTokenizer will directly create a class of the relevant architecture. For instance

will create a model that is an instance of BertModel.

There is one class of AutoModel for each task.

Each of the auto classes has a method to be extended with your custom classes. For instance, if you have defined a custom class of model NewModel, make sure you have a NewModelConfig then you can add those to the auto classes like this:

You will then be able to use the auto classes like you would usually do!

If your NewModelConfig is a subclass of PreTrainedConfig, make sure its model_type attribute is set to the same key you use when registering the config (here "new-model").

Likewise, if your NewModel is a subclass of PreTrainedModel, make sure its config_class attribute is set to the same class you use when registering the model (here NewModelConfig).

This is a generic configuration class that will be instantiated as one of the configuration classes of the library when created with the from_pretrained() class method.

This class cannot be instantiated directly using __init__() (throws an error).

( pretrained_model_name_or_path: str | os.PathLike[str] **kwargs )

If True, then this functions returns a Tuple(config, unused_kwargs) where unused_kwargs is a dictionary consisting of the key/value pairs whose keys are not configuration attributes: i.e., the part of kwargs which has not been used to update config and is otherwise ignored.

Instantiate one of the configuration classes of the library from a pretrained model configuration.

The configuration class to instantiate is selected based on the model_type property of the config object that is loaded, or when it’s missing, by falling back to using pattern matching on pretrained_model_name_or_path:

( model_type config exist_ok = False )

Register a new configuration for this class.

This is a generic tokenizer class that will be instantiated as one of the tokenizer classes of the library when created with the AutoTokenizer.from_pretrained() class method.

This class cannot be instantiated directly using __init__() (throws an error).

( pretrained_model_name_or_path *inputs **kwargs )

Instantiate one of the tokenizer classes of the library from a pretrained model vocabulary.

The tokenizer class to instantiate is selected based on the model_type property of the config object (either passed as an argument or loaded from pretrained_model_name_or_path if possible), or when it’s missing, by falling back to using pattern matching on pretrained_model_name_or_path:

( config_class tokenizer_class = None slow_tokenizer_class = None fast_tokenizer_class = None exist_ok = False )

Register a new tokenizer in this mapping.

This is a generic feature extractor class that will be instantiated as one of the feature extractor classes of the library when created with the AutoFeatureExtractor.from_pretrained() class method.

This class cannot be instantiated directly using __init__() (throws an error).

( pretrained_model_name_or_path **kwargs )

Instantiate one of the feature extractor classes of the library from a pretrained model vocabulary.

The feature extractor class to instantiate is selected based on the model_type property of the config object (either passed as an argument or loaded from pretrained_model_name_or_path if possible), or when it’s missing, by falling back to using pattern matching on pretrained_model_name_or_path:

Passing token=True is required when you want to use a private model.

( config_class feature_extractor_class exist_ok = False )

Register a new feature extractor for this class.

This is a generic image processor class that will be instantiated as one of the image processor classes of the library when created with the AutoImageProcessor.from_pretrained() class method.

This class cannot be instantiated directly using __init__() (throws an error).

( pretrained_model_name_or_path *inputs **kwargs )

Instantiate one of the image processor classes of the library from a pretrained model vocabulary.

The image processor class to instantiate is selected based on the model_type property of the config object (either passed as an argument or loaded from pretrained_model_name_or_path if possible), or when it’s missing, by falling back to using pattern matching on pretrained_model_name_or_path:

Passing token=True is required when you want to use a private model.

( config_class slow_image_processor_class = None fast_image_processor_class = None exist_ok = False )

Register a new image processor for this class.

This is a generic video processor class that will be instantiated as one of the video processor classes of the library when created with the AutoVideoProcessor.from_pretrained() class method.

This class cannot be instantiated directly using __init__() (throws an error).

( pretrained_model_name_or_path *inputs **kwargs )

Instantiate one of the video processor classes of the library from a pretrained model vocabulary.

The video processor class to instantiate is selected based on the model_type property of the config object (either passed as an argument or loaded from pretrained_model_name_or_path if possible), or when it’s missing, by falling back to using pattern matching on pretrained_model_name_or_path:

Passing token=True is required when you want to use a private model.

( config_class video_processor_class exist_ok = False )

Register a new video processor for this class.

This is a generic processor class that will be instantiated as one of the processor classes of the library when created with the AutoProcessor.from_pretrained() class method.

This class cannot be instantiated directly using __init__() (throws an error).

( pretrained_model_name_or_path **kwargs )

Instantiate one of the processor classes of the library from a pretrained model vocabulary.

The processor class to instantiate is selected based on the model_type property of the config object (either passed as an argument or loaded from pretrained_model_name_or_path if possible):

Passing token=True is required when you want to use a private model.

( config_class processor_class exist_ok = False )

Register a new processor for this class.

The following auto classes are available for instantiating a base model class without a specific head.

This is a generic model class that will be instantiated as one of the base model classes of the library when created with the from_pretrained() class method or the from_config() class method.

This class cannot be instantiated directly using __init__() (throws an error).

Instantiates one of the base model classes of the library from a configuration.

Note: Loading a model from its configuration file does not load the model weights. It only affects the model’s configuration. Use from_pretrained() to load the model weights.

( *model_args **kwargs )

This option can be used if you want to create a model from a pretrained configuration but load your own weights. In this case though, you should check if using save_pretrained() and from_pretrained() is not a simpler option.

Instantiate one of the base model classes of the library from a pretrained model.

The model class to instantiate is selected based on the model_type property of the config object (either passed as an argument or loaded from pretrained_model_name_or_path if possible), or when it’s missing, by falling back to using pattern matching on pretrained_model_name_or_path:

The model is set in evaluation mode by default using model.eval() (so for instance, dropout modules are deactivated). To train the model, you should first set it back in training mode with model.train()

The following auto classes are available for instantiating a model with a pretraining head.

This is a generic model class that will be instantiated as one of the model classes of the library (with a pretraining head) when created with the from_pretrained() class method or the from_config() class method.

This class cannot be instantiated directly using __init__() (throws an error).

Instantiates one of the model classes of the library (with a pretraining head) from a configuration.

Note: Loading a model from its configuration file does not load the model weights. It only affects the model’s configuration. Use from_pretrained() to load the model weights.

( *model_args **kwargs )

This option can be used if you want to create a model from a pretrained configuration but load your own weights. In this case though, you should check if using save_pretrained() and from_pretrained() is not a simpler option.

Instantiate one of the model classes of the library (with a pretraining head) from a pretrained model.

The model class to instantiate is selected based on the model_type property of the config object (either passed as an argument or loaded from pretrained_model_name_or_path if possible), or when it’s missing, by falling back to using pattern matching on pretrained_model_name_or_path:

The model is set in evaluation mode by default using model.eval() (so for instance, dropout modules are deactivated). To train the model, you should first set it back in training mode with model.train()

The following auto classes are available for the following natural language processing tasks.

This is a generic model class that will be instantiated as one of the model classes of the library (with a causal language modeling head) when created with the from_pretrained() class method or the from_config() class method.

This class cannot be instantiated directly using __init__() (throws an error).

Instantiates one of the model classes of the library (with a causal language modeling head) from a configuration.

Note: Loading a model from its configuration file does not load the model weights. It only affects the model’s configuration. Use from_pretrained() to load the model weights.

( *model_args **kwargs )

This option can be used if you want to create a model from a pretrained configuration but load your own weights. In this case though, you should check if using save_pretrained() and from_pretrained() is not a simpler option.

Instantiate one of the model classes of the library (with a causal language modeling head) from a pretrained model.

The model class to instantiate is selected based on the model_type property of the config object (either passed as an argument or loaded from pretrained_model_name_or_path if possible), or when it’s missing, by falling back to using pattern matching on pretrained_model_name_or_path:

The model is set in evaluation mode by default using model.eval() (so for instance, dropout modules are deactivated). To train the model, you should first set it back in training mode with model.train()

This is a generic model class that will be instantiated as one of the model classes of the library (with a masked language modeling head) when created with the from_pretrained() class method or the from_config() class method.

This class cannot be instantiated directly using __init__() (throws an error).

Instantiates one of the model classes of the library (with a masked language modeling head) from a configuration.

Note: Loading a model from its configuration file does not load the model weights. It only affects the model’s configuration. Use from_pretrained() to load the model weights.

( *model_args **kwargs )

This option can be used if you want to create a model from a pretrained configuration but load your own weights. In this case though, you should check if using save_pretrained() and from_pretrained() is not a simpler option.

Instantiate one of the model classes of the library (with a masked language modeling head) from a pretrained model.

The model class to instantiate is selected based on the model_type property of the config object (either passed as an argument or loaded from pretrained_model_name_or_path if possible), or when it’s missing, by falling back to using pattern matching on pretrained_model_name_or_path:

The model is set in evaluation mode by default using model.eval() (so for instance, dropout modules are deactivated). To train the model, you should first set it back in training mode with model.train()

This is a generic model class that will be instantiated as one of the model classes of the library (with a sequence-to-sequence language modeling head) when created with the from_pretrained() class method or the from_config() class method.

This class cannot be instantiated directly using __init__() (throws an error).

Instantiates one of the model classes of the library (with a sequence-to-sequence language modeling head) from a configuration.

Note: Loading a model from its configuration file does not load the model weights. It only affects the model’s configuration. Use from_pretrained() to load the model weights.

( *model_args **kwargs )

This option can be used if you want to create a model from a pretrained configuration but load your own weights. In this case though, you should check if using save_pretrained() and from_pretrained() is not a simpler option.

Instantiate one of the model classes of the library (with a sequence-to-sequence language modeling head) from a pretrained model.

The model class to instantiate is selected based on the model_type property of the config object (either passed as an argument or loaded from pretrained_model_name_or_path if possible), or when it’s missing, by falling back to using pattern matching on pretrained_model_name_or_path:

The model is set in evaluation mode by default using model.eval() (so for instance, dropout modules are deactivated). To train the model, you should first set it back in training mode with model.train()

This is a generic model class that will be instantiated as one of the model classes of the library (with a sequence classification head) when created with the from_pretrained() class method or the from_config() class method.

This class cannot be instantiated directly using __init__() (throws an error).

Instantiates one of the model classes of the library (with a sequence classification head) from a configuration.

Note: Loading a model from its configuration file does not load the model weights. It only affects the model’s configuration. Use from_pretrained() to load the model weights.

( *model_args **kwargs )

This option can be used if you want to create a model from a pretrained configuration but load your own weights. In this case though, you should check if using save_pretrained() and from_pretrained() is not a simpler option.

Instantiate one of the model classes of the library (with a sequence classification head) from a pretrained model.

The model class to instantiate is selected based on the model_type property of the config object (either passed as an argument or loaded from pretrained_model_name_or_path if possible), or when it’s missing, by falling back to using pattern matching on pretrained_model_name_or_path:

The model is set in evaluation mode by default using model.eval() (so for instance, dropout modules are deactivated). To train the model, you should first set it back in training mode with model.train()

This is a generic model class that will be instantiated as one of the model classes of the library (with a multiple choice head) when created with the from_pretrained() class method or the from_config() class method.

This class cannot be instantiated directly using __init__() (throws an error).

Instantiates one of the model classes of the library (with a multiple choice head) from a configuration.

Note: Loading a model from its configuration file does not load the model weights. It only affects the model’s configuration. Use from_pretrained() to load the model weights.

( *model_args **kwargs )

This option can be used if you want to create a model from a pretrained configuration but load your own weights. In this case though, you should check if using save_pretrained() and from_pretrained() is not a simpler option.

Instantiate one of the model classes of the library (with a multiple choice head) from a pretrained model.

The model class to instantiate is selected based on the model_type property of the config object (either passed as an argument or loaded from pretrained_model_name_or_path if possible), or when it’s missing, by falling back to using pattern matching on pretrained_model_name_or_path:

The model is set in evaluation mode by default using model.eval() (so for instance, dropout modules are deactivated). To train the model, you should first set it back in training mode with model.train()

This is a generic model class that will be instantiated as one of the model classes of the library (with a next sentence prediction head) when created with the from_pretrained() class method or the from_config() class method.

This class cannot be instantiated directly using __init__() (throws an error).

Instantiates one of the model classes of the library (with a next sentence prediction head) from a configuration.

Note: Loading a model from its configuration file does not load the model weights. It only affects the model’s configuration. Use from_pretrained() to load the model weights.

( *model_args **kwargs )

This option can be used if you want to create a model from a pretrained configuration but load your own weights. In this case though, you should check if using save_pretrained() and from_pretrained() is not a simpler option.

Instantiate one of the model classes of the library (with a next sentence prediction head) from a pretrained model.

The model class to instantiate is selected based on the model_type property of the config object (either passed as an argument or loaded from pretrained_model_name_or_path if possible), or when it’s missing, by falling back to using pattern matching on pretrained_model_name_or_path:

The model is set in evaluation mode by default using model.eval() (so for instance, dropout modules are deactivated). To train the model, you should first set it back in training mode with model.train()

This is a generic model class that will be instantiated as one of the model classes of the library (with a token classification head) when created with the from_pretrained() class method or the from_config() class method.

This class cannot be instantiated directly using __init__() (throws an error).

Instantiates one of the model classes of the library (with a token classification head) from a configuration.

Note: Loading a model from its configuration file does not load the model weights. It only affects the model’s configuration. Use from_pretrained() to load the model weights.

( *model_args **kwargs )

This option can be used if you want to create a model from a pretrained configuration but load your own weights. In this case though, you should check if using save_pretrained() and from_pretrained() is not a simpler option.

Instantiate one of the model classes of the library (with a token classification head) from a pretrained model.

The model class to instantiate is selected based on the model_type property of the config object (either passed as an argument or loaded from pretrained_model_name_or_path if possible), or when it’s missing, by falling back to using pattern matching on pretrained_model_name_or_path:

The model is set in evaluation mode by default using model.eval() (so for instance, dropout modules are deactivated). To train the model, you should first set it back in training mode with model.train()

This is a generic model class that will be instantiated as one of the model classes of the library (with a question answering head) when created with the from_pretrained() class method or the from_config() class method.

This class cannot be instantiated directly using __init__() (throws an error).

Instantiates one of the model classes of the library (with a question answering head) from a configuration.

Note: Loading a model from its configuration file does not load the model weights. It only affects the model’s configuration. Use from_pretrained() to load the model weights.

( *model_args **kwargs )

This option can be used if you want to create a model from a pretrained configuration but load your own weights. In this case though, you should check if using save_pretrained() and from_pretrained() is not a simpler option.

Instantiate one of the model classes of the library (with a question answering head) from a pretrained model.

The model class to instantiate is selected based on the model_type property of the config object (either passed as an argument or loaded from pretrained_model_name_or_path if possible), or when it’s missing, by falling back to using pattern matching on pretrained_model_name_or_path:

The model is set in evaluation mode by default using model.eval() (so for instance, dropout modules are deactivated). To train the model, you should first set it back in training mode with model.train()

The following auto classes are available for the following computer vision tasks.

This is a generic model class that will be instantiated as one of the model classes of the library (with a depth estimation head) when created with the from_pretrained() class method or the from_config() class method.

This class cannot be instantiated directly using __init__() (throws an error).

Instantiates one of the model classes of the library (with a depth estimation head) from a configuration.

Note: Loading a model from its configuration file does not load the model weights. It only affects the model’s configuration. Use from_pretrained() to load the model weights.

( *model_args **kwargs )

This option can be used if you want to create a model from a pretrained configuration but load your own weights. In this case though, you should check if using save_pretrained() and from_pretrained() is not a simpler option.

Instantiate one of the model classes of the library (with a depth estimation head) from a pretrained model.

The model class to instantiate is selected based on the model_type property of the config object (either passed as an argument or loaded from pretrained_model_name_or_path if possible), or when it’s missing, by falling back to using pattern matching on pretrained_model_name_or_path:

The model is set in evaluation mode by default using model.eval() (so for instance, dropout modules are deactivated). To train the model, you should first set it back in training mode with model.train()

This is a generic model class that will be instantiated as one of the model classes of the library (with a image classification head) when created with the from_pretrained() class method or the from_config() class method.

This class cannot be instantiated directly using __init__() (throws an error).

Instantiates one of the model classes of the library (with a image classification head) from a configuration.

Note: Loading a model from its configuration file does not load the model weights. It only affects the model’s configuration. Use from_pretrained() to load the model weights.

( *model_args **kwargs )

This option can be used if you want to create a model from a pretrained configuration but load your own weights. In this case though, you should check if using save_pretrained() and from_pretrained() is not a simpler option.

Instantiate one of the model classes of the library (with a image classification head) from a pretrained model.

The model class to instantiate is selected based on the model_type property of the config object (either passed as an argument or loaded from pretrained_model_name_or_path if possible), or when it’s missing, by falling back to using pattern matching on pretrained_model_name_or_path:

The model is set in evaluation mode by default using model.eval() (so for instance, dropout modules are deactivated). To train the model, you should first set it back in training mode with model.train()

This is a generic model class that will be instantiated as one of the model classes of the library (with a video classification head) when created with the from_pretrained() class method or the from_config() class method.

This class cannot be instantiated directly using __init__() (throws an error).

Instantiates one of the model classes of the library (with a video classification head) from a configuration.

Note: Loading a model from its configuration file does not load the model weights. It only affects the model’s configuration. Use from_pretrained() to load the model weights.

( *model_args **kwargs )

This option can be used if you want to create a model from a pretrained configuration but load your own weights. In this case though, you should check if using save_pretrained() and from_pretrained() is not a simpler option.

Instantiate one of the model classes of the library (with a video classification head) from a pretrained model.

The model class to instantiate is selected based on the model_type property of the config object (either passed as an argument or loaded from pretrained_model_name_or_path if possible), or when it’s missing, by falling back to using pattern matching on pretrained_model_name_or_path:

The model is set in evaluation mode by default using model.eval() (so for instance, dropout modules are deactivated). To train the model, you should first set it back in training mode with model.train()

This is a generic model class that will be instantiated as one of the model classes of the library (with a masked image modeling head) when created with the from_pretrained() class method or the from_config() class method.

This class cannot be instantiated directly using __init__() (throws an error).

Instantiates one of the model classes of the library (with a masked image modeling head) from a configuration.

Note: Loading a model from its configuration file does not load the model weights. It only affects the model’s configuration. Use from_pretrained() to load the model weights.

( *model_args **kwargs )

This option can be used if you want to create a model from a pretrained configuration but load your own weights. In this case though, you should check if using save_pretrained() and from_pretrained() is not a simpler option.

Instantiate one of the model classes of the library (with a masked image modeling head) from a pretrained model.

The model class to instantiate is selected based on the model_type property of the config object (either passed as an argument or loaded from pretrained_model_name_or_path if possible), or when it’s missing, by falling back to using pattern matching on pretrained_model_name_or_path:

The model is set in evaluation mode by default using model.eval() (so for instance, dropout modules are deactivated). To train the model, you should first set it back in training mode with model.train()

This is a generic model class that will be instantiated as one of the model classes of the library (with a object detection head) when created with the from_pretrained() class method or the from_config() class method.

This class cannot be instantiated directly using __init__() (throws an error).

Instantiates one of the model classes of the library (with a object detection head) from a configuration.

Note: Loading a model from its configuration file does not load the model weights. It only affects the model’s configuration. Use from_pretrained() to load the model weights.

( *model_args **kwargs )

This option can be used if you want to create a model from a pretrained configuration but load your own weights. In this case though, you should check if using save_pretrained() and from_pretrained() is not a simpler option.

Instantiate one of the model classes of the library (with a object detection head) from a pretrained model.

The model class to instantiate is selected based on the model_type property of the config object (either passed as an argument or loaded from pretrained_model_name_or_path if possible), or when it’s missing, by falling back to using pattern matching on pretrained_model_name_or_path:

The model is set in evaluation mode by default using model.eval() (so for instance, dropout modules are deactivated). To train the model, you should first set it back in training mode with model.train()

This is a generic model class that will be instantiated as one of the model classes of the library (with a image segmentation head) when created with the from_pretrained() class method or the from_config() class method.

This class cannot be instantiated directly using __init__() (throws an error).

Instantiates one of the model classes of the library (with a image segmentation head) from a configuration.

Note: Loading a model from its configuration file does not load the model weights. It only affects the model’s configuration. Use from_pretrained() to load the model weights.

( *model_args **kwargs )

This option can be used if you want to create a model from a pretrained configuration but load your own weights. In this case though, you should check if using save_pretrained() and from_pretrained() is not a simpler option.

Instantiate one of the model classes of the library (with a image segmentation head) from a pretrained model.

The model class to instantiate is selected based on the model_type property of the config object (either passed as an argument or loaded from pretrained_model_name_or_path if possible), or when it’s missing, by falling back to using pattern matching on pretrained_model_name_or_path:

The model is set in evaluation mode by default using model.eval() (so for instance, dropout modules are deactivated). To train the model, you should first set it back in training mode with model.train()

This is a generic model class that will be instantiated as one of the model classes of the library (with a semantic segmentation head) when created with the from_pretrained() class method or the from_config() class method.

This class cannot be instantiated directly using __init__() (throws an error).

Instantiates one of the model classes of the library (with a semantic segmentation head) from a configuration.

Note: Loading a model from its configuration file does not load the model weights. It only affects the model’s configuration. Use from_pretrained() to load the model weights.

( *model_args **kwargs )

This option can be used if you want to create a model from a pretrained configuration but load your own weights. In this case though, you should check if using save_pretrained() and from_pretrained() is not a simpler option.

Instantiate one of the model classes of the library (with a semantic segmentation head) from a pretrained model.

The model class to instantiate is selected based on the model_type property of the config object (either passed as an argument or loaded from pretrained_model_name_or_path if possible), or when it’s missing, by falling back to using pattern matching on pretrained_model_name_or_path:

The model is set in evaluation mode by default using model.eval() (so for instance, dropout modules are deactivated). To train the model, you should first set it back in training mode with model.train()

This is a generic model class that will be instantiated as one of the model classes of the library (with a instance segmentation head) when created with the from_pretrained() class method or the from_config() class method.

This class cannot be instantiated directly using __init__() (throws an error).

Instantiates one of the model classes of the library (with a instance segmentation head) from a configuration.

Note: Loading a model from its configuration file does not load the model weights. It only affects the model’s configuration. Use from_pretrained() to load the model weights.

( *model_args **kwargs )

This option can be used if you want to create a model from a pretrained configuration but load your own weights. In this case though, you should check if using save_pretrained() and from_pretrained() is not a simpler option.

Instantiate one of the model classes of the library (with a instance segmentation head) from a pretrained model.

The model class to instantiate is selected based on the model_type property of the config object (either passed as an argument or loaded from pretrained_model_name_or_path if possible), or when it’s missing, by falling back to using pattern matching on pretrained_model_name_or_path:

The model is set in evaluation mode by default using model.eval() (so for instance, dropout modules are deactivated). To train the model, you should first set it back in training mode with model.train()

This is a generic model class that will be instantiated as one of the model classes of the library (with a universal image segmentation head) when created with the from_pretrained() class method or the from_config() class method.

This class cannot be instantiated directly using __init__() (throws an error).

Instantiates one of the model classes of the library (with a universal image segmentation head) from a configuration.

Note: Loading a model from its configuration file does not load the model weights. It only affects the model’s configuration. Use from_pretrained() to load the model weights.

( *model_args **kwargs )

This option can be used if you want to create a model from a pretrained configuration but load your own weights. In this case though, you should check if using save_pretrained() and from_pretrained() is not a simpler option.

Instantiate one of the model classes of the library (with a universal image segmentation head) from a pretrained model.

The model class to instantiate is selected based on the model_type property of the config object (either passed as an argument or loaded from pretrained_model_name_or_path if possible), or when it’s missing, by falling back to using pattern matching on pretrained_model_name_or_path:

The model is set in evaluation mode by default using model.eval() (so for instance, dropout modules are deactivated). To train the model, you should first set it back in training mode with model.train()

This is a generic model class that will be instantiated as one of the model classes of the library (with a zero-shot image classification head) when created with the from_pretrained() class method or the from_config() class method.

This class cannot be instantiated directly using __init__() (throws an error).

Instantiates one of the model classes of the library (with a zero-shot image classification head) from a configuration.

Note: Loading a model from its configuration file does not load the model weights. It only affects the model’s configuration. Use from_pretrained() to load the model weights.

( *model_args **kwargs )

This option can be used if you want to create a model from a pretrained configuration but load your own weights. In this case though, you should check if using save_pretrained() and from_pretrained() is not a simpler option.

Instantiate one of the model classes of the library (with a zero-shot image classification head) from a pretrained model.

The model class to instantiate is selected based on the model_type property of the config object (either passed as an argument or loaded from pretrained_model_name_or_path if possible), or when it’s missing, by falling back to using pattern matching on pretrained_model_name_or_path:

The model is set in evaluation mode by default using model.eval() (so for instance, dropout modules are deactivated). To train the model, you should first set it back in training mode with model.train()

This is a generic model class that will be instantiated as one of the model classes of the library (with a zero-shot object detection head) when created with the from_pretrained() class method or the from_config() class method.

This class cannot be instantiated directly using __init__() (throws an error).

Instantiates one of the model classes of the library (with a zero-shot object detection head) from a configuration.

Note: Loading a model from its configuration file does not load the model weights. It only affects the model’s configuration. Use from_pretrained() to load the model weights.

( *model_args **kwargs )

This option can be used if you want to create a model from a pretrained configuration but load your own weights. In this case though, you should check if using save_pretrained() and from_pretrained() is not a simpler option.

Instantiate one of the model classes of the library (with a zero-shot object detection head) from a pretrained model.

The model class to instantiate is selected based on the model_type property of the config object (either passed as an argument or loaded from pretrained_model_name_or_path if possible), or when it’s missing, by falling back to using pattern matching on pretrained_model_name_or_path:

The model is set in evaluation mode by default using model.eval() (so for instance, dropout modules are deactivated). To train the model, you should first set it back in training mode with model.train()

The following auto classes are available for the following audio tasks.

This is a generic model class that will be instantiated as one of the model classes of the library (with a audio classification head) when created with the from_pretrained() class method or the from_config() class method.

This class cannot be instantiated directly using __init__() (throws an error).

Instantiates one of the model classes of the library (with a audio classification head) from a configuration.

Note: Loading a model from its configuration file does not load the model weights. It only affects the model’s configuration. Use from_pretrained() to load the model weights.

( *model_args **kwargs )

This option can be used if you want to create a model from a pretrained configuration but load your own weights. In this case though, you should check if using save_pretrained() and from_pretrained() is not a simpler option.

Instantiate one of the model classes of the library (with a audio classification head) from a pretrained model.

The model class to instantiate is selected based on the model_type property of the config object (either passed as an argument or loaded from pretrained_model_name_or_path if possible), or when it’s missing, by falling back to using pattern matching on pretrained_model_name_or_path:

The model is set in evaluation mode by default using model.eval() (so for instance, dropout modules are deactivated). To train the model, you should first set it back in training mode with model.train()

This is a generic model class that will be instantiated as one of the model classes of the library (with a audio frame (token) classification head) when created with the from_pretrained() class method or the from_config() class method.

This class cannot be instantiated directly using __init__() (throws an error).

Instantiates one of the model classes of the library (with a audio frame (token) classification head) from a configuration.

Note: Loading a model from its configuration file does not load the model weights. It only affects the model’s configuration. Use from_pretrained() to load the model weights.

( *model_args **kwargs )

This option can be used if you want to create a model from a pretrained configuration but load your own weights. In this case though, you should check if using save_pretrained() and from_pretrained() is not a simpler option.

Instantiate one of the model classes of the library (with a audio frame (token) classification head) from a pretrained model.

The model class to instantiate is selected based on the model_type property of the config object (either passed as an argument or loaded from pretrained_model_name_or_path if possible), or when it’s missing, by falling back to using pattern matching on pretrained_model_name_or_path:

The model is set in evaluation mode by default using model.eval() (so for instance, dropout modules are deactivated). To train the model, you should first set it back in training mode with model.train()

This is a generic model class that will be instantiated as one of the model classes of the library (with a connectionist temporal classification head) when created with the from_pretrained() class method or the from_config() class method.

This class cannot be instantiated directly using __init__() (throws an error).

Instantiates one of the model classes of the library (with a connectionist temporal classification head) from a configuration.

Note: Loading a model from its configuration file does not load the model weights. It only affects the model’s configuration. Use from_pretrained() to load the model weights.

( *model_args **kwargs )

This option can be used if you want to create a model from a pretrained configuration but load your own weights. In this case though, you should check if using save_pretrained() and from_pretrained() is not a simpler option.

Instantiate one of the model classes of the library (with a connectionist temporal classification head) from a pretrained model.

The model class to instantiate is selected based on the model_type property of the config object (either passed as an argument or loaded from pretrained_model_name_or_path if possible), or when it’s missing, by falling back to using pattern matching on pretrained_model_name_or_path:

The model is set in evaluation mode by default using model.eval() (so for instance, dropout modules are deactivated). To train the model, you should first set it back in training mode with model.train()

This is a generic model class that will be instantiated as one of the model classes of the library (with a sequence-to-sequence speech-to-text modeling head) when created with the from_pretrained() class method or the from_config() class method.

This class cannot be instantiated directly using __init__() (throws an error).

Instantiates one of the model classes of the library (with a sequence-to-sequence speech-to-text modeling head) from a configuration.

Note: Loading a model from its configuration file does not load the model weights. It only affects the model’s configuration. Use from_pretrained() to load the model weights.

( *model_args **kwargs )

This option can be used if you want to create a model from a pretrained configuration but load your own weights. In this case though, you should check if using save_pretrained() and from_pretrained() is not a simpler option.

Instantiate one of the model classes of the library (with a sequence-to-sequence speech-to-text modeling head) from a pretrained model.

The model class to instantiate is selected based on the model_type property of the config object (either passed as an argument or loaded from pretrained_model_name_or_path if possible), or when it’s missing, by falling back to using pattern matching on pretrained_model_name_or_path:

The model is set in evaluation mode by default using model.eval() (so for instance, dropout modules are deactivated). To train the model, you should first set it back in training mode with model.train()

This is a generic model class that will be instantiated as one of the model classes of the library (with a audio retrieval via x-vector head) when created with the from_pretrained() class method or the from_config() class method.

This class cannot be instantiated directly using __init__() (throws an error).

Instantiates one of the model classes of the library (with a audio retrieval via x-vector head) from a configuration.

Note: Loading a model from its configuration file does not load the model weights. It only affects the model’s configuration. Use from_pretrained() to load the model weights.

( *model_args **kwargs )

This option can be used if you want to create a model from a pretrained configuration but load your own weights. In this case though, you should check if using save_pretrained() and from_pretrained() is not a simpler option.

Instantiate one of the model classes of the library (with a audio retrieval via x-vector head) from a pretrained model.

The model class to instantiate is selected based on the model_type property of the config object (either passed as an argument or loaded from pretrained_model_name_or_path if possible), or when it’s missing, by falling back to using pattern matching on pretrained_model_name_or_path:

The model is set in evaluation mode by default using model.eval() (so for instance, dropout modules are deactivated). To train the model, you should first set it back in training mode with model.train()

This is a generic model class that will be instantiated as one of the model classes of the library (with a audio tokenization through codebooks head) when created with the from_pretrained() class method or the from_config() class method.

This class cannot be instantiated directly using __init__() (throws an error).

Instantiates one of the model classes of the library (with a audio tokenization through codebooks head) from a configuration.

Note: Loading a model from its configuration file does not load the model weights. It only affects the model’s configuration. Use from_pretrained() to load the model weights.

( *model_args **kwargs )

This option can be used if you want to create a model from a pretrained configuration but load your own weights. In this case though, you should check if using save_pretrained() and from_pretrained() is not a simpler option.

Instantiate one of the model classes of the library (with a audio tokenization through codebooks head) from a pretrained model.

The model class to instantiate is selected based on the model_type property of the config object (either passed as an argument or loaded from pretrained_model_name_or_path if possible), or when it’s missing, by falling back to using pattern matching on pretrained_model_name_or_path:

The model is set in evaluation mode by default using model.eval() (so for instance, dropout modules are deactivated). To train the model, you should first set it back in training mode with model.train()

The following auto classes are available for the following multimodal tasks.

This is a generic model class that will be instantiated as one of the model classes of the library (with a multimodal generation head) when created with the from_pretrained() class method or the from_config() class method.

This class cannot be instantiated directly using __init__() (throws an error).

Instantiates one of the model classes of the library (with a multimodal generation head) from a configuration.

Note: Loading a model from its configuration file does not load the model weights. It only affects the model’s configuration. Use from_pretrained() to load the model weights.

( *model_args **kwargs )

This option can be used if you want to create a model from a pretrained configuration but load your own weights. In this case though, you should check if using save_pretrained() and from_pretrained() is not a simpler option.

Instantiate one of the model classes of the library (with a multimodal generation head) from a pretrained model.

The model class to instantiate is selected based on the model_type property of the config object (either passed as an argument or loaded from pretrained_model_name_or_path if possible), or when it’s missing, by falling back to using pattern matching on pretrained_model_name_or_path:

The model is set in evaluation mode by default using model.eval() (so for instance, dropout modules are deactivated). To train the model, you should first set it back in training mode with model.train()

This is a generic model class that will be instantiated as one of the model classes of the library (with a table question answering head) when created with the from_pretrained() class method or the from_config() class method.

This class cannot be instantiated directly using __init__() (throws an error).

Instantiates one of the model classes of the library (with a table question answering head) from a configuration.

Note: Loading a model from its configuration file does not load the model weights. It only affects the model’s configuration. Use from_pretrained() to load the model weights.

( *model_args **kwargs )

This option can be used if you want to create a model from a pretrained configuration but load your own weights. In this case though, you should check if using save_pretrained() and from_pretrained() is not a simpler option.

Instantiate one of the model classes of the library (with a table question answering head) from a pretrained model.

The model class to instantiate is selected based on the model_type property of the config object (either passed as an argument or loaded from pretrained_model_name_or_path if possible), or when it’s missing, by falling back to using pattern matching on pretrained_model_name_or_path:

The model is set in evaluation mode by default using model.eval() (so for instance, dropout modules are deactivated). To train the model, you should first set it back in training mode with model.train()

This is a generic model class that will be instantiated as one of the model classes of the library (with a document question answering head) when created with the from_pretrained() class method or the from_config() class method.

This class cannot be instantiated directly using __init__() (throws an error).

Instantiates one of the model classes of the library (with a document question answering head) from a configuration.

Note: Loading a model from its configuration file does not load the model weights. It only affects the model’s configuration. Use from_pretrained() to load the model weights.

( *model_args **kwargs )

This option can be used if you want to create a model from a pretrained configuration but load your own weights. In this case though, you should check if using save_pretrained() and from_pretrained() is not a simpler option.

Instantiate one of the model classes of the library (with a document question answering head) from a pretrained model.

The model class to instantiate is selected based on the model_type property of the config object (either passed as an argument or loaded from pretrained_model_name_or_path if possible), or when it’s missing, by falling back to using pattern matching on pretrained_model_name_or_path:

The model is set in evaluation mode by default using model.eval() (so for instance, dropout modules are deactivated). To train the model, you should first set it back in training mode with model.train()

This is a generic model class that will be instantiated as one of the model classes of the library (with a visual question answering head) when created with the from_pretrained() class method or the from_config() class method.

This class cannot be instantiated directly using __init__() (throws an error).

Instantiates one of the model classes of the library (with a visual question answering head) from a configuration.

Note: Loading a model from its configuration file does not load the model weights. It only affects the model’s configuration. Use from_pretrained() to load the model weights.

( *model_args **kwargs )

This option can be used if you want to create a model from a pretrained configuration but load your own weights. In this case though, you should check if using save_pretrained() and from_pretrained() is not a simpler option.

Instantiate one of the model classes of the library (with a visual question answering head) from a pretrained model.

The model class to instantiate is selected based on the model_type property of the config object (either passed as an argument or loaded from pretrained_model_name_or_path if possible), or when it’s missing, by falling back to using pattern matching on pretrained_model_name_or_path:

The model is set in evaluation mode by default using model.eval() (so for instance, dropout modules are deactivated). To train the model, you should first set it back in training mode with model.train()

This is a generic model class that will be instantiated as one of the model classes of the library (with a image-text-to-text modeling head) when created with the from_pretrained() class method or the from_config() class method.

This class cannot be instantiated directly using __init__() (throws an error).

Instantiates one of the model classes of the library (with a image-text-to-text modeling head) from a configuration.

Note: Loading a model from its configuration file does not load the model weights. It only affects the model’s configuration. Use from_pretrained() to load the model weights.

( *model_args **kwargs )

This option can be used if you want to create a model from a pretrained configuration but load your own weights. In this case though, you should check if using save_pretrained() and from_pretrained() is not a simpler option.

Instantiate one of the model classes of the library (with a image-text-to-text modeling head) from a pretrained model.

The model class to instantiate is selected based on the model_type property of the config object (either passed as an argument or loaded from pretrained_model_name_or_path if possible), or when it’s missing, by falling back to using pattern matching on pretrained_model_name_or_path:

The model is set in evaluation mode by default using model.eval() (so for instance, dropout modules are deactivated). To train the model, you should first set it back in training mode with model.train()

This is a generic model class that will be instantiated as one of the model classes of the library (with a time-series prediction head) when created with the from_pretrained() class method or the from_config() class method.

This class cannot be instantiated directly using __init__() (throws an error).

Instantiates one of the model classes of the library (with a time-series prediction head) from a configuration.

Note: Loading a model from its configuration file does not load the model weights. It only affects the model’s configuration. Use from_pretrained() to load the model weights.

( *model_args **kwargs )

This option can be used if you want to create a model from a pretrained configuration but load your own weights. In this case though, you should check if using save_pretrained() and from_pretrained() is not a simpler option.

Instantiate one of the model classes of the library (with a time-series prediction head) from a pretrained model.

The model class to instantiate is selected based on the model_type property of the config object (either passed as an argument or loaded from pretrained_model_name_or_path if possible), or when it’s missing, by falling back to using pattern matching on pretrained_model_name_or_path:

The model is set in evaluation mode by default using model.eval() (so for instance, dropout modules are deactivated). To train the model, you should first set it back in training mode with model.train()

---

## Transformers

**URL:** https://huggingface.co/docs/transformers/v5.0.0/en/main_classes/processors

**Contents:**
- Transformers
- Processors
- Multi-modal processors
  - class transformers.ProcessorMixin
    - apply_chat_template
    - batch_decode
    - check_argument_for_proper_class
    - decode
    - from_args_and_dict
    - from_pretrained

Transformers documentation

and get access to the augmented documentation experience

Processors can mean two different things in the Transformers library:

Any multi-modal model will require an object to encode or decode the data that groups several modalities (among text, vision and audio). This is handled by objects called processors, which group together two or more processing objects such as tokenizers (for the text modality), image processors (for vision) and feature extractors (for audio).

Those processors inherit from the following base class that implements the saving and loading functionality:

This is a mixin used to provide saving/loading functionality for all processor classes.

( conversation: list[dict[str, str]] | list[list[dict[str, str]]] chat_template: str | None = None **kwargs: typing_extensions.Unpack[transformers.processing_utils.AllKwargsForChatTemplate] )

Similar to the apply_chat_template method on tokenizers, this method applies a Jinja template to input conversations to turn them into a single tokenizable string.

The input is expected to be in the following format, where each message content is a list consisting of text and optionally image or video inputs. One can also provide an image, video, URL or local path which will be used to form pixel_values when return_dict=True. If not provided, one will get only the formatted text, optionally tokenized text.

conversation = [ { “role”: “user”, “content”: [ {“type”: “image”, “url”: “https://www.ilankelman.org/stopsigns/australia.jpg”}, {“type”: “text”, “text”: “Please describe this image in detail.”}, ], }, ]

This method forwards all its arguments to PreTrainedTokenizer’s batch_decode(). Please refer to the docstring of this method for more information.

( argument_name argument )

Checks the passed argument’s class against the expected transformers class. In case of an unexpected mismatch between expected and actual class, an error is raise. Otherwise, the proper retrieved class is returned.

This method forwards all its arguments to PreTrainedTokenizer’s decode(). Please refer to the docstring of this method for more information.

( args processor_dict: dict **kwargs ) → ~processing_utils.ProcessingMixin

~processing_utils.ProcessingMixin

The processor object instantiated from those parameters.

The processor object instantiated from those parameters.

Instantiates a type of ~processing_utils.ProcessingMixin from a Python dictionary of parameters.

( pretrained_model_name_or_path: str | os.PathLike cache_dir: str | os.PathLike | None = None force_download: bool = False local_files_only: bool = False token: str | bool | None = None revision: str = 'main' **kwargs )

Instantiate a processor associated with a pretrained model.

This class method is simply calling the feature extractor from_pretrained(), image processor ImageProcessingMixin and the tokenizer ~tokenization_utils_base.PreTrainedTokenizer.from_pretrained methods. Please refer to the docstrings of the methods above for more information.

( pretrained_model_name_or_path: str | os.PathLike **kwargs ) → tuple[Dict, Dict]

The dictionary(ies) that will be used to instantiate the processor object.

The dictionary(ies) that will be used to instantiate the processor object.

From a pretrained_model_name_or_path, resolve to a dictionary of parameters, to be used for instantiating a processor of type ~processing_utils.ProcessingMixin using from_args_and_dict.

( generated_outputs skip_special_tokens = True **kwargs ) → list[str]

Post-process the output of a vlm to decode the text.

( generated_outputs skip_special_tokens = True generation_mode = None **kwargs ) → list[str]

Post-process the output of a multimodal model to return the requested modality output. If the model cannot generated the requested modality, an error will be raised.

( repo_id: str commit_message: str | None = None commit_description: str | None = None private: bool | None = None token: bool | str | None = None revision: str | None = None create_pr: bool = False max_shard_size: int | str | None = '50GB' tags: list[str] | None = None )

Upload the processor files to the 🤗 Model Hub.

( auto_class = 'AutoProcessor' )

Register this class with a given auto class. This should only be used for custom feature extractors as the ones in the library are already mapped with AutoProcessor.

( save_directory push_to_hub: bool = False **kwargs )

Saves the attributes of this processor (feature extractor, tokenizer…) in the specified directory so that it can be reloaded using the from_pretrained() method.

This class method is simply calling save_pretrained() and save_pretrained(). Please refer to the docstrings of the methods above for more information.

Dictionary of all the attributes that make up this processor instance.

Dictionary of all the attributes that make up this processor instance.

Serializes this instance to a Python dictionary.

( json_file_path: str | os.PathLike )

Save this instance to a JSON file.

String containing all the attributes that make up this feature_extractor instance in JSON format.

String containing all the attributes that make up this feature_extractor instance in JSON format.

Serializes this instance to a JSON string.

All processors follow the same architecture which is that of the DataProcessor. The processor returns a list of InputExample. These InputExample can be converted to InputFeatures in order to be fed to the model.

Base class for data converters for sequence classification data sets.

Gets a collection of InputExample for the dev set.

Gets an example from a dict.

Gets the list of labels for this data set.

Gets a collection of InputExample for the test set.

Gets a collection of InputExample for the train set.

Some tensorflow_datasets datasets are not formatted the same way the GLUE datasets are. This method converts examples to the correct format.

( guid: str text_a: str text_b: str | None = None label: str | None = None )

A single training/test example for simple sequence classification.

Serializes this instance to a JSON string.

( input_ids: list attention_mask: list[int] | None = None token_type_ids: list[int] | None = None label: int | float | None = None )

A single set of features of data. Property names are the same names as the corresponding inputs to a model.

Serializes this instance to a JSON string.

General Language Understanding Evaluation (GLUE) is a benchmark that evaluates the performance of models across a diverse set of existing NLU tasks. It was released together with the paper GLUE: A multi-task benchmark and analysis platform for natural language understanding

This library hosts a total of 10 processors for the following tasks: MRPC, MNLI, MNLI (mismatched), CoLA, SST2, STSB, QQP, QNLI, RTE and WNLI.

Those processors are:

Additionally, the following method can be used to load values from a data file and convert them to a list of InputExample.

( examples: list tokenizer: PythonBackend max_length: int | None = None task = None label_list = None output_mode = None )

Loads a data file into a list of InputFeatures

The Cross-Lingual NLI Corpus (XNLI) is a benchmark that evaluates the quality of cross-lingual text representations. XNLI is crowd-sourced dataset based on MultiNLI: pairs of text are labeled with textual entailment annotations for 15 different languages (including both high-resource language such as English and low-resource languages such as Swahili).

It was released together with the paper XNLI: Evaluating Cross-lingual Sentence Representations

This library hosts the processor to load the XNLI data:

Please note that since the gold labels are available on the test set, evaluation is performed on the test set.

An example using these processors is given in the run_xnli.py script.

The Stanford Question Answering Dataset (SQuAD) is a benchmark that evaluates the performance of models on question answering. Two versions are available, v1.1 and v2.0. The first version (v1.1) was released together with the paper SQuAD: 100,000+ Questions for Machine Comprehension of Text. The second version (v2.0) was released alongside the paper Know What You Don’t Know: Unanswerable Questions for SQuAD.

This library hosts a processor for each of the two versions:

Those processors are:

They both inherit from the abstract class ~data.processors.utils.SquadProcessor

Processor for the SQuAD data set. overridden by SquadV1Processor and SquadV2Processor, used by the version 1.1 and version 2.0 of SQuAD, respectively.

( data_dir filename = None )

Returns the evaluation example from the data directory.

( dataset evaluate = False )

Creates a list of SquadExample using a TFDS dataset.

( data_dir filename = None )

Returns the training examples from the data directory.

Additionally, the following method can be used to convert SQuAD examples into ~data.processors.utils.SquadFeatures that can be used as model inputs.

( examples tokenizer max_seq_length doc_stride max_query_length is_training padding_strategy = 'max_length' return_dataset = False threads = 1 tqdm_enabled = True )

Converts a list of examples into a list of features that can be directly given as input to a model. It is model-dependant and takes advantage of many of the tokenizer’s features to create the model’s inputs.

These processors as well as the aforementioned method can be used with files containing the data as well as with the tensorflow_datasets package. Examples are given below.

Here is an example using the processors as well as the conversion method using data files:

Using tensorflow_datasets is as easy as using a data file:

Another example using these processors is given in the run_squad.py script.

---

## Transformers

**URL:** https://huggingface.co/docs/transformers/en/main_classes/text_generation

**Contents:**
- Transformers
- Generation
- GenerationConfig
  - class transformers.GenerationConfig
    - from_pretrained
    - from_model_config
    - save_pretrained
    - update
    - validate
    - get_generation_mode

Transformers documentation

and get access to the augmented documentation experience

Each framework has a generate method for text generation implemented in their respective GenerationMixin class:

You can parameterize the generate method with a GenerationConfig class instance. Please refer to this class for the complete list of generation parameters, which control the behavior of the generation method.

To learn how to inspect a model’s generation configuration, what are the defaults, how to change the parameters ad hoc, and how to create and save a customized generation configuration, refer to the text generation strategies guide. The guide also explains how to use related features, like token streaming.

Parameters that control the length of the output

Parameters that control the generation strategy used

Parameters that control the cache

If none is specified, we will use the default cache for the model (which is often DynamicCache). See our cache documentation for further information.

Parameters for manipulation of the model output logits

Parameters that define the output variables of generate

Special tokens that can be used at generation time

Generation parameters exclusive to encoder-decoder models

Generation parameters exclusive to assistant generation

Parameters related to performances and compilation

Class that holds a configuration for a generation task. A generate call supports the following generation methods for text-decoder, text-to-text, speech-to-text, and vision-to-text models:

To learn more about decoding strategies refer to the text generation strategies guide.

A large number of these flags control the logits or the stopping criteria of the generation. Make sure you check the generate-related classes for a full description of the possible manipulations, as well as examples of their usage.

Note: the configuration field that are still None will be overriden by GenerationConfig._get_default_generation_params() during the generation loop. If you want to use different values for these fields, make sure to explicitly set them in the generation config.

( pretrained_model_name: str | os.PathLike config_file_name: str | os.PathLike | None = None cache_dir: str | os.PathLike | None = None force_download: bool = False local_files_only: bool = False token: str | bool | None = None revision: str = 'main' **kwargs ) → GenerationConfig

To test a pull request you made on the Hub, you can pass revision="refs/pr/<pr_number>".

If True, then this functions returns a Tuple(config, unused_kwargs) where unused_kwargs is a dictionary consisting of the key/value pairs whose keys are not configuration attributes: i.e., the part of kwargs which has not been used to update config and is otherwise ignored.

The configuration object instantiated from this pretrained model.

The configuration object instantiated from this pretrained model.

Instantiate a GenerationConfig from a generation configuration file.

( model_config: typing.Union[ForwardRef('PreTrainedConfig'), dict] ) → GenerationConfig

The configuration object instantiated from those parameters.

The configuration object instantiated from those parameters.

Instantiates a GenerationConfig from a PreTrainedConfig. This function is useful to convert legacy PreTrainedConfig objects, which may contain generation parameters, into a stand-alone GenerationConfig.

( save_directory: str | os.PathLike config_file_name: str | os.PathLike | None = None push_to_hub: bool = False **kwargs )

Save a generation configuration object to the directory save_directory, so that it can be re-loaded using the from_pretrained() class method.

( defaults_only = False allow_custom_entries = False **kwargs ) → dict[str, Any]

Dictionary containing all the key-value pairs that were not used to update the instance.

Dictionary containing all the key-value pairs that were not used to update the instance.

Updates attributes of this class instance with attributes from kwargs if they match existing attributes, returning all the unused kwargs.

Validates the values of the attributes of the GenerationConfig instance. Raises exceptions in the presence of parameterization that can be detected as incorrect from the configuration instance alone.

Note that some parameters not validated here are best validated at generate runtime, as they may depend on other inputs and/or the model, such as parameters related to the generation length.

( assistant_model: typing.Optional[ForwardRef('PreTrainedModel')] = None ) → GenerationMode

The generation mode triggered by the instance.

The generation mode triggered by the instance.

Returns the generation mode triggered by the GenerationConfig instance.

A class containing all functions for auto-regressive text generation, to be used as a mixin in model classes. Inheriting from this class causes the model to have special generation-related behavior, such as loading a GenerationConfig at initialization time or ensuring generate-related tests are run in transformers CI.

A model class should inherit from GenerationMixin to enable calling methods like generate, or when it has defined a custom generate method that relies on GenerationMixin, directly or indirectly, which approximately shares the same interface to public methods like generate. Three examples:

The class exposes generate(), which can be used for:

To learn more about decoding strategies refer to the text generation strategies guide.

( inputs: torch.Tensor | None = None generation_config: transformers.generation.configuration_utils.GenerationConfig | None = None logits_processor: transformers.generation.logits_process.LogitsProcessorList | None = None stopping_criteria: transformers.generation.stopping_criteria.StoppingCriteriaList | None = None prefix_allowed_tokens_fn: collections.abc.Callable[[int, torch.Tensor], list[int]] | None = None synced_gpus: bool | None = None assistant_model: typing.Optional[ForwardRef('PreTrainedModel')] = None streamer: typing.Optional[ForwardRef('BaseStreamer')] = None negative_prompt_ids: torch.Tensor | None = None negative_prompt_attention_mask: torch.Tensor | None = None custom_generate: str | collections.abc.Callable | None = None **kwargs ) → ModelOutput or torch.LongTensor

ModelOutput or torch.LongTensor

A ModelOutput (if return_dict_in_generate=True or when config.return_dict_in_generate=True) or a torch.LongTensor. If the model is not an encoder-decoder model (model.config.is_encoder_decoder=False), the possible ModelOutput types are: GenerateDecoderOnlyOutput, GenerateBeamDecoderOnlyOutput If the model is an encoder-decoder model (model.config.is_encoder_decoder=True), the possible ModelOutput types are: GenerateEncoderDecoderOutput, GenerateBeamEncoderDecoderOutput

A ModelOutput (if return_dict_in_generate=True or when config.return_dict_in_generate=True) or a torch.LongTensor.

If the model is not an encoder-decoder model (model.config.is_encoder_decoder=False), the possible ModelOutput types are:

If the model is an encoder-decoder model (model.config.is_encoder_decoder=True), the possible ModelOutput types are:

Generates sequences of token ids for models with a language modeling head.

Most generation-controlling parameters are set in generation_config which, if not passed, will be set to the model’s default generation configuration. You can override any generation_config by passing the corresponding parameters to generate(), e.g. .generate(inputs, num_beams=4, do_sample=True).

For an overview of generation strategies and code examples, check out the following guide.

( sequences: Tensor scores: tuple beam_indices: torch.Tensor | None = None normalize_logits: bool = False ) → torch.Tensor

A torch.Tensor of shape (batch_size*num_return_sequences, sequence_length) containing the transition scores (logits)

A torch.Tensor of shape (batch_size*num_return_sequences, sequence_length) containing the transition scores (logits)

Computes the transition scores of sequences given the generation scores (and beam indices, if beam search was used). This is a convenient method to quickly obtain the scores of the selected tokens at generation time.

Mixin class for models to add continuous batching capabilities.

( inputs: list generation_config: transformers.generation.configuration_utils.GenerationConfig | None = None num_q_padding_intervals: int = 0 num_kv_padding_intervals: int = 0 allow_block_sharing: bool = True record_timestamps: bool = False progress_bar: bool = True **kwargs ) → dict[str, GenerationOutput]

dict[str, GenerationOutput]

a dictionary of request ids to GenerationOutput objects

a dictionary of request ids to GenerationOutput objects

Generate sequences for a batch of prompts using continuous batching.

( generation_config: transformers.generation.configuration_utils.GenerationConfig | None = None manual_eviction: bool = False max_queue_size: int = 0 num_q_padding_intervals: int = 0 num_kv_padding_intervals: int = 0 allow_block_sharing: bool = True ) → ContinuousBatchingManager

ContinuousBatchingManager

The manager instance to add requests and retrieve results.

The manager instance to add requests and retrieve results.

Initialize a manager for continuous batching inference.

( model: Module generation_config: GenerationConfig manual_eviction: bool = False max_queue_size: int = 0 num_q_padding_intervals: int = 0 num_kv_padding_intervals: int = 0 allow_block_sharing: bool = True )

Manager for handling continuous batching of generation requests.

This class provides the user interface for submitting generation requests, retrieving results, and managing the background generation thread.

( input_ids: list request_id: str | None = None max_new_tokens: int | None = None streaming: bool = False record_timestamps: bool = False ) → str

Add a new generation request to the queue.

Cancel a request by its ID.

Evict a request from the cache. It is assumed that the request is already finished.

( request_id: str | None = None timeout: float | None = None ) → Optional[GenerationOutput]

Optional[GenerationOutput]

The result data or None if timeout

The result data or None if timeout

Retrieve one result from the output queue.

Check if the background generation thread is running.

( stop_trigger_time: float timeout: float | None = None )

Wait for the background thread to finish.

Iterate over results matching a specific request id as they become available.

Start the background generation thread.

( block: bool = True timeout: float | None = None )

Signal the background thread to stop.

( cache: PagedAttentionCache retain_cache_on_finish: bool = False )

Abstract base class for scheduling requests in the continuous batch processor. Schedulers manage the lifecycle of requests from when they are added to the waiting queue to when they are scheduled for processing. Different schedulers implement different strategies for prioritizing and batching requests.

( state: RequestState )

Adds a request to the waiting list.

Remove all cancelled requests from active and waiting queues.

( request_id: str evict_from_cache: bool = True )

Completes processing of a request and optionally frees its allocated cache blocks. This method is called when a request has finished generation or encountered an error.

Gets generated tokens for an active request.

Checks if there are requests ready to be processed.

Checks if a request has been cancelled or removed.

( token_budget: int cache_budget: int )

Schedules requests for the next batch based on available token and cache budgets. This method selects which requests should be processed in the current batch, considering the budgets and the scheduler’s prioritization rules. The token_budget is the maximum number of tokens that can be processed in a batch, and the cache_budget is the maximum number of KV cache entries that can be read in a batch.

Marks a request for cancellation.

( cache: PagedAttentionCache retain_cache_on_finish: bool = False safety_margin: float = 0.2 )

This scheduler processes requests in the order they arrive, meaning decoding requests has priority over prefilling requests. Additionally, it includes a safety margin mechanism to prevent cache exhaustion. By default, when 80% of the cache is full, new requests will not be scheduled to prioritize decoding active requests.

( cache: PagedAttentionCache retain_cache_on_finish: bool = False )

Scheduler that prioritizes split prefill requests over decoding requests. This scheduler ensures that split prefill requests (which are continuations of partially processed prompts) are completed before processing new decoding requests.

---

## Transformers

**URL:** https://huggingface.co/docs/transformers/v5.0.0/en/model_doc/wav2vec2

**Contents:**
- Transformers
- Wav2Vec2
- Overview
- Usage tips
- Using Flash Attention 2
  - Installation
  - Usage
  - Expected speedups
- Resources
- Wav2Vec2Config

Transformers documentation

and get access to the augmented documentation experience

This model was released on 2020-06-20 and added to Hugging Face Transformers on 2021-02-02.

The Wav2Vec2 model was proposed in wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations by Alexei Baevski, Henry Zhou, Abdelrahman Mohamed, Michael Auli.

The abstract from the paper is the following:

We show for the first time that learning powerful representations from speech audio alone followed by fine-tuning on transcribed speech can outperform the best semi-supervised methods while being conceptually simpler. wav2vec 2.0 masks the speech input in the latent space and solves a contrastive task defined over a quantization of the latent representations which are jointly learned. Experiments using all labeled data of Librispeech achieve 1.8/3.3 WER on the clean/other test sets. When lowering the amount of labeled data to one hour, wav2vec 2.0 outperforms the previous state of the art on the 100 hour subset while using 100 times less labeled data. Using just ten minutes of labeled data and pre-training on 53k hours of unlabeled data still achieves 4.8/8.2 WER. This demonstrates the feasibility of speech recognition with limited amounts of labeled data.

This model was contributed by patrickvonplaten.

Note: Meta (FAIR) released a new version of Wav2Vec2-BERT 2.0 - it’s pretrained on 4.5M hours of audio. We especially recommend using it for fine-tuning tasks, e.g. as per this guide.

Flash Attention 2 is an faster, optimized version of the model.

First, check whether your hardware is compatible with Flash Attention 2. The latest list of compatible hardware can be found in the official documentation.

Next, install the latest version of Flash Attention 2:

To load a model using Flash Attention 2, we can pass the argument attn_implementation="flash_attention_2" to .from_pretrained. We’ll also load the model in half-precision (e.g. torch.float16), since it results in almost no degradation to audio quality but significantly lower memory usage and faster inference:

Below is an expected speedup diagram comparing the pure inference time between the native implementation in transformers of the facebook/wav2vec2-large-960h-lv60-self model and the flash-attention-2 and sdpa (scale-dot-product-attention) versions. . We show the average speedup obtained on the librispeech_asr clean validation split:

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with Wav2Vec2. If you’re interested in submitting a resource to be included here, please feel free to open a Pull Request and we’ll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.

( vocab_size = 32 hidden_size = 768 num_hidden_layers = 12 num_attention_heads = 12 intermediate_size = 3072 hidden_act = 'gelu' hidden_dropout = 0.1 activation_dropout = 0.1 attention_dropout = 0.1 feat_proj_dropout = 0.0 feat_quantizer_dropout = 0.0 final_dropout = 0.1 layerdrop = 0.1 initializer_range = 0.02 layer_norm_eps = 1e-05 feat_extract_norm = 'group' feat_extract_activation = 'gelu' conv_dim = (512, 512, 512, 512, 512, 512, 512) conv_stride = (5, 2, 2, 2, 2, 2, 2) conv_kernel = (10, 3, 3, 3, 3, 2, 2) conv_bias = False num_conv_pos_embeddings = 128 num_conv_pos_embedding_groups = 16 do_stable_layer_norm = False apply_spec_augment = True mask_time_prob = 0.05 mask_time_length = 10 mask_time_min_masks = 2 mask_feature_prob = 0.0 mask_feature_length = 10 mask_feature_min_masks = 0 num_codevectors_per_group = 320 num_codevector_groups = 2 contrastive_logits_temperature = 0.1 num_negatives = 100 codevector_dim = 256 proj_codevector_dim = 256 diversity_loss_weight = 0.1 ctc_loss_reduction = 'sum' ctc_zero_infinity = False use_weighted_layer_sum = False classifier_proj_size = 256 tdnn_dim = (512, 512, 512, 512, 1500) tdnn_kernel = (5, 3, 3, 1, 1) tdnn_dilation = (1, 2, 3, 1, 1) xvector_output_dim = 512 pad_token_id = 0 bos_token_id = 1 eos_token_id = 2 add_adapter = False adapter_kernel_size = 3 adapter_stride = 2 num_adapter_layers = 3 output_hidden_size = None adapter_attn_dim = None **kwargs )

This is the configuration class to store the configuration of a Wav2Vec2Model. It is used to instantiate an Wav2Vec2 model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the Wav2Vec2 facebook/wav2vec2-base-960h architecture.

Configuration objects inherit from PreTrainedConfig and can be used to control the model outputs. Read the documentation from PreTrainedConfig for more information.

( vocab_file bos_token = '<s>' eos_token = '</s>' unk_token = '<unk>' pad_token = '<pad>' word_delimiter_token = '|' replace_word_delimiter_char = ' ' do_lower_case = False target_lang = None **kwargs )

Constructs a Wav2Vec2CTC tokenizer.

This tokenizer inherits from PreTrainedTokenizer which contains some of the main methods. Users should refer to the superclass for more information regarding such methods.

( text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None text_pair: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None text_target: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None text_pair_target: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None add_special_tokens: bool = True padding: bool | str | PaddingStrategy = False truncation: bool | str | TruncationStrategy | None = None max_length: int | None = None stride: int = 0 is_split_into_words: bool = False pad_to_multiple_of: int | None = None padding_side: str | None = None return_tensors: str | TensorType | None = None return_token_type_ids: bool | None = None return_attention_mask: bool | None = None return_overflowing_tokens: bool = False return_special_tokens_mask: bool = False return_offsets_mapping: bool = False return_length: bool = False verbose: bool = True tokenizer_kwargs: dict[str, Any] | None = None **kwargs ) → BatchEncoding

If left unset or set to None, this will use the predefined model maximum length if a maximum length is required by one of the truncation/padding parameters. If the model has no specific maximum input length (like XLNet) truncation/padding to a maximum length will be deactivated.

What are token type IDs?

What are attention masks?

This is only available on fast tokenizers inheriting from PreTrainedTokenizerFast, if using Python’s tokenizer, this method will raise NotImplementedError.

A BatchEncoding with the following fields: input_ids — List of token ids to be fed to a model. What are input IDs? token_type_ids — List of token type ids to be fed to a model (when return_token_type_ids=True or if “token_type_ids” is in self.model_input_names). What are token type IDs? attention_mask — List of indices specifying which tokens should be attended to by the model (when return_attention_mask=True or if “attention_mask” is in self.model_input_names). What are attention masks? overflowing_tokens — List of overflowing tokens sequences (when a max_length is specified and return_overflowing_tokens=True). num_truncated_tokens — Number of tokens truncated (when a max_length is specified and return_overflowing_tokens=True). special_tokens_mask — List of 0s and 1s, with 1 specifying added special tokens and 0 specifying regular sequence tokens (when add_special_tokens=True and return_special_tokens_mask=True). length — The length of the inputs (when return_length=True)

A BatchEncoding with the following fields:

input_ids — List of token ids to be fed to a model.

token_type_ids — List of token type ids to be fed to a model (when return_token_type_ids=True or if “token_type_ids” is in self.model_input_names).

What are token type IDs?

attention_mask — List of indices specifying which tokens should be attended to by the model (when return_attention_mask=True or if “attention_mask” is in self.model_input_names).

What are attention masks?

overflowing_tokens — List of overflowing tokens sequences (when a max_length is specified and return_overflowing_tokens=True).

num_truncated_tokens — Number of tokens truncated (when a max_length is specified and return_overflowing_tokens=True).

special_tokens_mask — List of 0s and 1s, with 1 specifying added special tokens and 0 specifying regular sequence tokens (when add_special_tokens=True and return_special_tokens_mask=True).

length — The length of the inputs (when return_length=True)

Main method to tokenize and prepare for the model one or several sequence(s) or one or several pair(s) of sequences.

( save_directory: str filename_prefix: str | None = None )

( token_ids: typing.Union[int, list[int], numpy.ndarray, ForwardRef('torch.Tensor')] skip_special_tokens: bool = False clean_up_tokenization_spaces: bool | None = None output_char_offsets: bool = False output_word_offsets: bool = False **kwargs ) → str or Wav2Vec2CTCTokenizerOutput

Please take a look at the example below to better understand how to make use of output_char_offsets.

Please take a look at the example below to better understand how to make use of output_word_offsets.

str or Wav2Vec2CTCTokenizerOutput

The list of decoded sentences. Will be a Wav2Vec2CTCTokenizerOutput when output_char_offsets == True or output_word_offsets == True.

The list of decoded sentences. Will be a Wav2Vec2CTCTokenizerOutput when output_char_offsets == True or output_word_offsets == True.

Converts a sequence of ids in a string, using the tokenizer and vocabulary with options to remove special tokens and clean up tokenization spaces.

Similar to doing self.convert_tokens_to_string(self.convert_ids_to_tokens(token_ids)).

( sequences: typing.Union[list[int], list[list[int]], numpy.ndarray, ForwardRef('torch.Tensor')] skip_special_tokens: bool = False clean_up_tokenization_spaces: bool | None = None output_char_offsets: bool = False output_word_offsets: bool = False **kwargs ) → list[str] or Wav2Vec2CTCTokenizerOutput

Please take a look at the Example of decode() to better understand how to make use of output_char_offsets. batch_decode() works the same way with batched output.

Please take a look at the Example of decode() to better understand how to make use of output_word_offsets. batch_decode() works the same way with batched output.

list[str] or Wav2Vec2CTCTokenizerOutput

The list of decoded sentences. Will be a Wav2Vec2CTCTokenizerOutput when output_char_offsets == True or output_word_offsets == True.

The list of decoded sentences. Will be a Wav2Vec2CTCTokenizerOutput when output_char_offsets == True or output_word_offsets == True.

Convert a list of lists of token ids into a list of strings by calling decode.

Set the target language of a nested multi-lingual dictionary

( feature_size = 1 sampling_rate = 16000 padding_value = 0.0 return_attention_mask = False do_normalize = True **kwargs )

Wav2Vec2 models that have set config.feat_extract_norm == "group", such as wav2vec2-base, have not been trained using attention_mask. For such models, input_values should simply be padded with 0 and no attention_mask should be passed.

For Wav2Vec2 models that have set config.feat_extract_norm == "layer", such as wav2vec2-lv60, attention_mask should be passed for batched inference.

Constructs a Wav2Vec2 feature extractor.

This feature extractor inherits from SequenceFeatureExtractor which contains most of the main methods. Users should refer to this superclass for more information regarding those methods.

( raw_speech: numpy.ndarray | list[float] | list[numpy.ndarray] | list[list[float]] padding: bool | str | transformers.utils.generic.PaddingStrategy = False max_length: int | None = None truncation: bool = False pad_to_multiple_of: int | None = None return_attention_mask: bool | None = None return_tensors: str | transformers.utils.generic.TensorType | None = None sampling_rate: int | None = None **kwargs )

This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta), or on TPUs which benefit from having sequence lengths be a multiple of 128.

What are attention masks?

Wav2Vec2 models that have set config.feat_extract_norm == "group", such as wav2vec2-base, have not been trained using attention_mask. For such models, input_values should simply be padded with 0 and no attention_mask should be passed.

For Wav2Vec2 models that have set config.feat_extract_norm == "layer", such as wav2vec2-lv60, attention_mask should be passed for batched inference.

Main method to featurize and prepare for the model one or several sequence(s).

( feature_extractor tokenizer )

Constructs a Wav2Vec2Processor which wraps a feature extractor and a tokenizer into a single processor.

Wav2Vec2Processor offers all the functionalities of Wav2Vec2FeatureExtractor and Wav2Vec2CTCTokenizer. See the ~Wav2Vec2FeatureExtractor and ~Wav2Vec2CTCTokenizer for more information.

( audio: typing.Union[numpy.ndarray, ForwardRef('torch.Tensor'), list[numpy.ndarray], list['torch.Tensor'], NoneType] = None text: str | list[str] | None = None **kwargs: typing_extensions.Unpack[transformers.models.wav2vec2.processing_wav2vec2.Wav2Vec2ProcessorKwargs] )

This method operates on batches of extracted features and/or tokenized text. It forwards all arguments to Wav2Vec2FeatureExtractor.pad() and/or PreTrainedTokenizer.pad() depending on the input modality and returns their outputs. If both modalities are passed, Wav2Vec2FeatureExtractor.pad() and PreTrainedTokenizer.pad() are called.

( pretrained_model_name_or_path: str | os.PathLike cache_dir: str | os.PathLike | None = None force_download: bool = False local_files_only: bool = False token: str | bool | None = None revision: str = 'main' **kwargs )

Instantiate a processor associated with a pretrained model.

This class method is simply calling the feature extractor from_pretrained(), image processor ImageProcessingMixin and the tokenizer ~tokenization_utils_base.PreTrainedTokenizer.from_pretrained methods. Please refer to the docstrings of the methods above for more information.

( save_directory push_to_hub: bool = False **kwargs )

Saves the attributes of this processor (feature extractor, tokenizer…) in the specified directory so that it can be reloaded using the from_pretrained() method.

This class method is simply calling save_pretrained() and save_pretrained(). Please refer to the docstrings of the methods above for more information.

This method forwards all its arguments to PreTrainedTokenizer’s batch_decode(). Please refer to the docstring of this method for more information.

This method forwards all its arguments to PreTrainedTokenizer’s decode(). Please refer to the docstring of this method for more information.

( feature_extractor: FeatureExtractionMixin tokenizer: PreTrainedTokenizerBase decoder: BeamSearchDecoderCTC )

Constructs a Wav2Vec2ProcessorWithLM which wraps a feature extractor and a tokenizer into a single processor.

Wav2Vec2ProcessorWithLM offers all the functionalities of feature_extractor_class and tokenizer_class. See the ~feature_extractor_class and ~tokenizer_class for more information.

When used in normal mode, this method forwards all its arguments to the feature extractor’s ~FeatureExtractionMixin.pad and returns its output. If used in the context ~Wav2Vec2ProcessorWithLM.as_target_processor this method forwards all its arguments to Wav2Vec2CTCTokenizer’s pad(). Please refer to the docstring of the above two methods for more information.

( pretrained_model_name_or_path **kwargs )

Instantiate a Wav2Vec2ProcessorWithLM from a pretrained Wav2Vec2 processor.

This class method is simply calling the feature extractor’s from_pretrained(), Wav2Vec2CTCTokenizer’s from_pretrained(), and pyctcdecode.BeamSearchDecoderCTC.load_from_hf_hub.

Please refer to the docstrings of the methods above for more information.

( logits: ndarray pool: multiprocessing.pool.Pool | None = None num_processes: int | None = None beam_width: int | None = None beam_prune_logp: float | None = None token_min_logp: float | None = None hotwords: collections.abc.Iterable[str] | None = None hotword_weight: float | None = None alpha: float | None = None beta: float | None = None unk_score_offset: float | None = None lm_score_boundary: bool | None = None output_word_offsets: bool = False n_best: int = 1 )

Currently, only pools created with a ‘fork’ context can be used. If a ‘spawn’ pool is passed, it will be ignored and sequential decoding will be used instead.

Please take a look at the Example of decode() to better understand how to make use of output_word_offsets. batch_decode() works the same way with batched output.

Batch decode output logits to audio transcription with language model support.

This function makes use of Python’s multiprocessing. Currently, multiprocessing is available only on Unix systems (see this issue).

If you are decoding multiple batches, consider creating a Pool and passing it to batch_decode. Otherwise, batch_decode will be very slow since it will create a fresh Pool for each call. See usage example below.

Example: See Decoding multiple audios.

( logits: ndarray beam_width: int | None = None beam_prune_logp: float | None = None token_min_logp: float | None = None hotwords: collections.abc.Iterable[str] | None = None hotword_weight: float | None = None alpha: float | None = None beta: float | None = None unk_score_offset: float | None = None lm_score_boundary: bool | None = None output_word_offsets: bool = False n_best: int = 1 )

Please take a look at the example below to better understand how to make use of output_word_offsets.

Decode output logits to audio transcription with language model support.

If you are planning to decode multiple batches of audios, you should consider using batch_decode() and passing an instantiated multiprocessing.Pool. Otherwise, batch_decode() performance will be slower than calling decode() for each audio individually, as it internally instantiates a new Pool for every call. See the example below:

( text: list[list[str]] | list[str] | str logit_score: list[list[float]] | list[float] | float = None lm_score: list[list[float]] | list[float] | float = None word_offsets: list[list[list[dict[str, int | str]]]] | list[list[dict[str, int | str]]] | list[dict[str, int | str]] = None )

Output type of Wav2Vec2DecoderWithLM, with transcription.

( last_hidden_state: torch.FloatTensor | None = None extract_features: torch.FloatTensor | None = None hidden_states: tuple[torch.FloatTensor, ...] | None = None attentions: tuple[torch.FloatTensor, ...] | None = None )

Hidden-states of the model at the output of each layer plus the initial embedding outputs.

Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

Base class for models that have been trained with the Wav2Vec2 loss objective.

( loss: torch.FloatTensor | None = None projected_states: torch.FloatTensor | None = None projected_quantized_states: torch.FloatTensor | None = None codevector_perplexity: torch.FloatTensor | None = None hidden_states: tuple[torch.FloatTensor] | None = None attentions: tuple[torch.FloatTensor] | None = None contrastive_loss: torch.FloatTensor | None = None diversity_loss: torch.FloatTensor | None = None )

Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.

Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

Output type of Wav2Vec2ForPreTraining, with potential hidden states and attentions.

( config: Wav2Vec2Config )

The bare Wav2Vec2 Model outputting raw hidden-states without any specific head on top.

This model inherits from PreTrainedModel. Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch torch.nn.Module subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

( input_values: torch.Tensor | None attention_mask: torch.Tensor | None = None mask_time_indices: torch.FloatTensor | None = None output_attentions: bool | None = None output_hidden_states: bool | None = None return_dict: bool | None = None **kwargs ) → transformers.modeling_outputs.Wav2Vec2BaseModelOutput or tuple(torch.FloatTensor)

What are attention masks?

transformers.modeling_outputs.Wav2Vec2BaseModelOutput or tuple(torch.FloatTensor)

A transformers.modeling_outputs.Wav2Vec2BaseModelOutput or a tuple of torch.FloatTensor (if return_dict=False is passed or when config.return_dict=False) comprising various elements depending on the configuration (Wav2Vec2Config) and inputs. last_hidden_state (torch.FloatTensor of shape (batch_size, sequence_length, hidden_size)) — Sequence of hidden-states at the output of the last layer of the model. extract_features (torch.FloatTensor of shape (batch_size, sequence_length, conv_dim[-1])) — Sequence of extracted feature vectors of the last convolutional layer of the model. hidden_states (tuple(torch.FloatTensor), optional, returned when output_hidden_states=True is passed or when config.output_hidden_states=True) — Tuple of torch.FloatTensor (one for the output of the embeddings + one for the output of each layer) of shape (batch_size, sequence_length, hidden_size). Hidden-states of the model at the output of each layer plus the initial embedding outputs. attentions (tuple(torch.FloatTensor), optional, returned when output_attentions=True is passed or when config.output_attentions=True) — Tuple of torch.FloatTensor (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length). Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

A transformers.modeling_outputs.Wav2Vec2BaseModelOutput or a tuple of torch.FloatTensor (if return_dict=False is passed or when config.return_dict=False) comprising various elements depending on the configuration (Wav2Vec2Config) and inputs.

last_hidden_state (torch.FloatTensor of shape (batch_size, sequence_length, hidden_size)) — Sequence of hidden-states at the output of the last layer of the model.

extract_features (torch.FloatTensor of shape (batch_size, sequence_length, conv_dim[-1])) — Sequence of extracted feature vectors of the last convolutional layer of the model.

hidden_states (tuple(torch.FloatTensor), optional, returned when output_hidden_states=True is passed or when config.output_hidden_states=True) — Tuple of torch.FloatTensor (one for the output of the embeddings + one for the output of each layer) of shape (batch_size, sequence_length, hidden_size).

Hidden-states of the model at the output of each layer plus the initial embedding outputs.

attentions (tuple(torch.FloatTensor), optional, returned when output_attentions=True is passed or when config.output_attentions=True) — Tuple of torch.FloatTensor (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length).

Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

The Wav2Vec2Model forward method, overrides the __call__ special method.

Although the recipe for forward pass needs to be defined within this function, one should call the Module instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

( config target_lang: str | None = None )

Wav2Vec2 Model with a language modeling head on top for Connectionist Temporal Classification (CTC).

This model inherits from PreTrainedModel. Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch torch.nn.Module subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

( input_values: torch.Tensor | None attention_mask: torch.Tensor | None = None output_attentions: bool | None = None output_hidden_states: bool | None = None return_dict: bool | None = None labels: torch.Tensor | None = None **kwargs ) → transformers.modeling_outputs.CausalLMOutput or tuple(torch.FloatTensor)

What are attention masks?

transformers.modeling_outputs.CausalLMOutput or tuple(torch.FloatTensor)

A transformers.modeling_outputs.CausalLMOutput or a tuple of torch.FloatTensor (if return_dict=False is passed or when config.return_dict=False) comprising various elements depending on the configuration (Wav2Vec2Config) and inputs. loss (torch.FloatTensor of shape (1,), optional, returned when labels is provided) — Language modeling loss (for next-token prediction). logits (torch.FloatTensor of shape (batch_size, sequence_length, config.vocab_size)) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax). hidden_states (tuple(torch.FloatTensor), optional, returned when output_hidden_states=True is passed or when config.output_hidden_states=True) — Tuple of torch.FloatTensor (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape (batch_size, sequence_length, hidden_size). Hidden-states of the model at the output of each layer plus the optional initial embedding outputs. attentions (tuple(torch.FloatTensor), optional, returned when output_attentions=True is passed or when config.output_attentions=True) — Tuple of torch.FloatTensor (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length). Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

A transformers.modeling_outputs.CausalLMOutput or a tuple of torch.FloatTensor (if return_dict=False is passed or when config.return_dict=False) comprising various elements depending on the configuration (Wav2Vec2Config) and inputs.

loss (torch.FloatTensor of shape (1,), optional, returned when labels is provided) — Language modeling loss (for next-token prediction).

logits (torch.FloatTensor of shape (batch_size, sequence_length, config.vocab_size)) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).

hidden_states (tuple(torch.FloatTensor), optional, returned when output_hidden_states=True is passed or when config.output_hidden_states=True) — Tuple of torch.FloatTensor (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape (batch_size, sequence_length, hidden_size).

Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.

attentions (tuple(torch.FloatTensor), optional, returned when output_attentions=True is passed or when config.output_attentions=True) — Tuple of torch.FloatTensor (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length).

Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

The Wav2Vec2ForCTC forward method, overrides the __call__ special method.

Although the recipe for forward pass needs to be defined within this function, one should call the Module instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

( target_lang: str force_load = True **kwargs )

To test a pull request you made on the Hub, you can pass revision="refs/pr/<pr_number>".

Load a language adapter model from a pre-trained adapter model.

Activate the special “offline-mode” to use this method in a firewalled environment.

Wav2Vec2 Model with a sequence classification head on top (a linear layer over the pooled output) for tasks like SUPERB Keyword Spotting.

This model inherits from PreTrainedModel. Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch torch.nn.Module subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

( input_values: torch.Tensor | None attention_mask: torch.Tensor | None = None output_attentions: bool | None = None output_hidden_states: bool | None = None return_dict: bool | None = None labels: torch.Tensor | None = None **kwargs ) → transformers.modeling_outputs.SequenceClassifierOutput or tuple(torch.FloatTensor)

What are attention masks?

transformers.modeling_outputs.SequenceClassifierOutput or tuple(torch.FloatTensor)

A transformers.modeling_outputs.SequenceClassifierOutput or a tuple of torch.FloatTensor (if return_dict=False is passed or when config.return_dict=False) comprising various elements depending on the configuration (Wav2Vec2Config) and inputs. loss (torch.FloatTensor of shape (1,), optional, returned when labels is provided) — Classification (or regression if config.num_labels==1) loss. logits (torch.FloatTensor of shape (batch_size, config.num_labels)) — Classification (or regression if config.num_labels==1) scores (before SoftMax). hidden_states (tuple(torch.FloatTensor), optional, returned when output_hidden_states=True is passed or when config.output_hidden_states=True) — Tuple of torch.FloatTensor (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape (batch_size, sequence_length, hidden_size). Hidden-states of the model at the output of each layer plus the optional initial embedding outputs. attentions (tuple(torch.FloatTensor), optional, returned when output_attentions=True is passed or when config.output_attentions=True) — Tuple of torch.FloatTensor (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length). Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

A transformers.modeling_outputs.SequenceClassifierOutput or a tuple of torch.FloatTensor (if return_dict=False is passed or when config.return_dict=False) comprising various elements depending on the configuration (Wav2Vec2Config) and inputs.

loss (torch.FloatTensor of shape (1,), optional, returned when labels is provided) — Classification (or regression if config.num_labels==1) loss.

logits (torch.FloatTensor of shape (batch_size, config.num_labels)) — Classification (or regression if config.num_labels==1) scores (before SoftMax).

hidden_states (tuple(torch.FloatTensor), optional, returned when output_hidden_states=True is passed or when config.output_hidden_states=True) — Tuple of torch.FloatTensor (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape (batch_size, sequence_length, hidden_size).

Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.

attentions (tuple(torch.FloatTensor), optional, returned when output_attentions=True is passed or when config.output_attentions=True) — Tuple of torch.FloatTensor (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length).

Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

The Wav2Vec2ForSequenceClassification forward method, overrides the __call__ special method.

Although the recipe for forward pass needs to be defined within this function, one should call the Module instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example of single-label classification:

Example of multi-label classification:

The Wav2Vec2 Model with a frame classification head on top for tasks like Speaker Diarization.

This model inherits from PreTrainedModel. Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch torch.nn.Module subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

( input_values: torch.Tensor | None attention_mask: torch.Tensor | None = None labels: torch.Tensor | None = None output_attentions: bool | None = None output_hidden_states: bool | None = None return_dict: bool | None = None **kwargs ) → transformers.modeling_outputs.TokenClassifierOutput or tuple(torch.FloatTensor)

What are attention masks?

transformers.modeling_outputs.TokenClassifierOutput or tuple(torch.FloatTensor)

A transformers.modeling_outputs.TokenClassifierOutput or a tuple of torch.FloatTensor (if return_dict=False is passed or when config.return_dict=False) comprising various elements depending on the configuration (Wav2Vec2Config) and inputs. loss (torch.FloatTensor of shape (1,), optional, returned when labels is provided) — Classification loss. logits (torch.FloatTensor of shape (batch_size, sequence_length, config.num_labels)) — Classification scores (before SoftMax). hidden_states (tuple(torch.FloatTensor), optional, returned when output_hidden_states=True is passed or when config.output_hidden_states=True) — Tuple of torch.FloatTensor (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape (batch_size, sequence_length, hidden_size). Hidden-states of the model at the output of each layer plus the optional initial embedding outputs. attentions (tuple(torch.FloatTensor), optional, returned when output_attentions=True is passed or when config.output_attentions=True) — Tuple of torch.FloatTensor (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length). Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

A transformers.modeling_outputs.TokenClassifierOutput or a tuple of torch.FloatTensor (if return_dict=False is passed or when config.return_dict=False) comprising various elements depending on the configuration (Wav2Vec2Config) and inputs.

loss (torch.FloatTensor of shape (1,), optional, returned when labels is provided) — Classification loss.

logits (torch.FloatTensor of shape (batch_size, sequence_length, config.num_labels)) — Classification scores (before SoftMax).

hidden_states (tuple(torch.FloatTensor), optional, returned when output_hidden_states=True is passed or when config.output_hidden_states=True) — Tuple of torch.FloatTensor (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape (batch_size, sequence_length, hidden_size).

Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.

attentions (tuple(torch.FloatTensor), optional, returned when output_attentions=True is passed or when config.output_attentions=True) — Tuple of torch.FloatTensor (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length).

Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

The Wav2Vec2ForAudioFrameClassification forward method, overrides the __call__ special method.

Although the recipe for forward pass needs to be defined within this function, one should call the Module instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Wav2Vec2 Model with an XVector feature extraction head on top for tasks like Speaker Verification.

This model inherits from PreTrainedModel. Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch torch.nn.Module subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

( input_values: torch.Tensor | None attention_mask: torch.Tensor | None = None output_attentions: bool | None = None output_hidden_states: bool | None = None return_dict: bool | None = None labels: torch.Tensor | None = None **kwargs ) → transformers.modeling_outputs.XVectorOutput or tuple(torch.FloatTensor)

What are attention masks?

transformers.modeling_outputs.XVectorOutput or tuple(torch.FloatTensor)

A transformers.modeling_outputs.XVectorOutput or a tuple of torch.FloatTensor (if return_dict=False is passed or when config.return_dict=False) comprising various elements depending on the configuration (Wav2Vec2Config) and inputs. loss (torch.FloatTensor of shape (1,), optional, returned when labels is provided) — Classification loss. logits (torch.FloatTensor of shape (batch_size, config.xvector_output_dim)) — Classification hidden states before AMSoftmax. embeddings (torch.FloatTensor of shape (batch_size, config.xvector_output_dim)) — Utterance embeddings used for vector similarity-based retrieval. hidden_states (tuple(torch.FloatTensor), optional, returned when output_hidden_states=True is passed or when config.output_hidden_states=True) — Tuple of torch.FloatTensor (one for the output of the embeddings + one for the output of each layer) of shape (batch_size, sequence_length, hidden_size). Hidden-states of the model at the output of each layer plus the initial embedding outputs. attentions (tuple(torch.FloatTensor), optional, returned when output_attentions=True is passed or when config.output_attentions=True) — Tuple of torch.FloatTensor (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length). Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

A transformers.modeling_outputs.XVectorOutput or a tuple of torch.FloatTensor (if return_dict=False is passed or when config.return_dict=False) comprising various elements depending on the configuration (Wav2Vec2Config) and inputs.

loss (torch.FloatTensor of shape (1,), optional, returned when labels is provided) — Classification loss.

logits (torch.FloatTensor of shape (batch_size, config.xvector_output_dim)) — Classification hidden states before AMSoftmax.

embeddings (torch.FloatTensor of shape (batch_size, config.xvector_output_dim)) — Utterance embeddings used for vector similarity-based retrieval.

hidden_states (tuple(torch.FloatTensor), optional, returned when output_hidden_states=True is passed or when config.output_hidden_states=True) — Tuple of torch.FloatTensor (one for the output of the embeddings + one for the output of each layer) of shape (batch_size, sequence_length, hidden_size).

Hidden-states of the model at the output of each layer plus the initial embedding outputs.

attentions (tuple(torch.FloatTensor), optional, returned when output_attentions=True is passed or when config.output_attentions=True) — Tuple of torch.FloatTensor (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length).

Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

The Wav2Vec2ForXVector forward method, overrides the __call__ special method.

Although the recipe for forward pass needs to be defined within this function, one should call the Module instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

( config: Wav2Vec2Config )

Wav2Vec2 Model with a quantizer and VQ head on top.

This model inherits from PreTrainedModel. Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch torch.nn.Module subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

( input_values: torch.Tensor | None attention_mask: torch.Tensor | None = None mask_time_indices: torch.BoolTensor | None = None sampled_negative_indices: torch.BoolTensor | None = None output_attentions: bool | None = None output_hidden_states: bool | None = None return_dict: bool | None = None **kwargs ) → transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForPreTrainingOutput or tuple(torch.FloatTensor)

What are attention masks?

transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForPreTrainingOutput or tuple(torch.FloatTensor)

A transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForPreTrainingOutput or a tuple of torch.FloatTensor (if return_dict=False is passed or when config.return_dict=False) comprising various elements depending on the configuration (Wav2Vec2Config) and inputs. loss (*optional*, returned when sample_negative_indices are passed, torch.FloatTensor of shape (1,)) — Total loss as the sum of the contrastive loss (L_m) and the diversity loss (L_d) as stated in the official paper. projected_states (torch.FloatTensor of shape (batch_size, sequence_length, config.proj_codevector_dim)) — Hidden-states of the model projected to config.proj_codevector_dim that can be used to predict the masked projected quantized states. projected_quantized_states (torch.FloatTensor of shape (batch_size, sequence_length, config.proj_codevector_dim)) — Quantized extracted feature vectors projected to config.proj_codevector_dim representing the positive target vectors for contrastive loss. codevector_perplexity (torch.FloatTensor of shape (1,)) — The perplexity of the codevector distribution, used to measure the diversity of the codebook. hidden_states (tuple[torch.FloatTensor] | None.hidden_states, returned when output_hidden_states=True is passed or when config.output_hidden_states=True) — Tuple of torch.FloatTensor (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape (batch_size, sequence_length, hidden_size). Hidden-states of the model at the output of each layer plus the optional initial embedding outputs. attentions (tuple[torch.FloatTensor] | None.attentions, returned when output_attentions=True is passed or when config.output_attentions=True) — Tuple of torch.FloatTensor (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length). Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads. contrastive_loss (*optional*, returned when sample_negative_indices are passed, torch.FloatTensor of shape (1,)) — The contrastive loss (L_m) as stated in the official paper. diversity_loss (*optional*, returned when sample_negative_indices are passed, torch.FloatTensor of shape (1,)) — The diversity loss (L_d) as stated in the official paper.

A transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForPreTrainingOutput or a tuple of torch.FloatTensor (if return_dict=False is passed or when config.return_dict=False) comprising various elements depending on the configuration (Wav2Vec2Config) and inputs.

loss (*optional*, returned when sample_negative_indices are passed, torch.FloatTensor of shape (1,)) — Total loss as the sum of the contrastive loss (L_m) and the diversity loss (L_d) as stated in the official paper.

projected_states (torch.FloatTensor of shape (batch_size, sequence_length, config.proj_codevector_dim)) — Hidden-states of the model projected to config.proj_codevector_dim that can be used to predict the masked projected quantized states.

projected_quantized_states (torch.FloatTensor of shape (batch_size, sequence_length, config.proj_codevector_dim)) — Quantized extracted feature vectors projected to config.proj_codevector_dim representing the positive target vectors for contrastive loss.

codevector_perplexity (torch.FloatTensor of shape (1,)) — The perplexity of the codevector distribution, used to measure the diversity of the codebook.

hidden_states (tuple[torch.FloatTensor] | None.hidden_states, returned when output_hidden_states=True is passed or when config.output_hidden_states=True) — Tuple of torch.FloatTensor (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape (batch_size, sequence_length, hidden_size).

Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.

attentions (tuple[torch.FloatTensor] | None.attentions, returned when output_attentions=True is passed or when config.output_attentions=True) — Tuple of torch.FloatTensor (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length).

Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

contrastive_loss (*optional*, returned when sample_negative_indices are passed, torch.FloatTensor of shape (1,)) — The contrastive loss (L_m) as stated in the official paper.

diversity_loss (*optional*, returned when sample_negative_indices are passed, torch.FloatTensor of shape (1,)) — The diversity loss (L_d) as stated in the official paper.

The Wav2Vec2ForPreTraining forward method, overrides the __call__ special method.

Although the recipe for forward pass needs to be defined within this function, one should call the Module instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

---

## Transformers

**URL:** https://huggingface.co/docs/transformers/v5.0.0/en/installation

**Contents:**
- Transformers
- Installation
- Virtual environment
- Python
  - Source install
  - Editable install
- conda
- Set up
  - Cache directory
  - Offline mode

Transformers documentation

and get access to the augmented documentation experience

Transformers works with PyTorch. It has been tested on Python 3.9+ and PyTorch 2.2+.

uv is an extremely fast Rust-based Python package and project manager and requires a virtual environment by default to manage different projects and avoids compatibility issues between dependencies.

It can be used as a drop-in replacement for pip, but if you prefer to use pip, remove uv from the commands below.

Refer to the uv installation docs to install uv.

Create a virtual environment to install Transformers in.

Install Transformers with the following command.

uv is a fast Rust-based Python package and project manager.

For GPU acceleration, install the appropriate CUDA drivers for PyTorch.

Run the command below to check if your system detects an NVIDIA GPU.

To install a CPU-only version of Transformers, run the following command.

Test whether the install was successful with the following command. It should return a label and score for the provided text.

Installing from source installs the latest version rather than the stable version of the library. It ensures you have the most up-to-date changes in Transformers and it’s useful for experimenting with the latest features or fixing a bug that hasn’t been officially released in the stable version yet.

The downside is that the latest version may not always be stable. If you encounter any problems, please open a GitHub Issue so we can fix it as soon as possible.

Install from source with the following command.

Check if the install was successful with the command below. It should return a label and score for the provided text.

An editable install is useful if you’re developing locally with Transformers. It links your local copy of Transformers to the Transformers repository instead of copying the files. The files are added to Python’s import path.

You must keep the local Transformers folder to keep using it.

Update your local version of Transformers with the latest changes in the main repository with the following command.

conda is a language-agnostic package manager. Install Transformers from the conda-forge channel in your newly created virtual environment.

After installation, you can configure the Transformers cache location or set up the library for offline usage.

When you load a pretrained model with from_pretrained(), the model is downloaded from the Hub and locally cached.

Every time you load a model, it checks whether the cached model is up-to-date. If it’s the same, then the local model is loaded. If it’s not the same, the newer model is downloaded and cached.

The default directory given by the shell environment variable HF_HUB_CACHE is ~/.cache/huggingface/hub. On Windows, the default directory is C:\Users\username\.cache\huggingface\hub.

Cache a model in a different directory by changing the path in the following shell environment variables (listed by priority).

To use Transformers in an offline or firewalled environment requires the downloaded and cached files ahead of time. Download a model repository from the Hub with the snapshot_download method.

Refer to the Download files from the Hub guide for more options for downloading files from the Hub. You can download files from specific revisions, download from the CLI, and even filter which files to download from a repository.

Set the environment variable HF_HUB_OFFLINE=1 to prevent HTTP calls to the Hub when loading a model.

Another option for only loading cached files is to set local_files_only=True in from_pretrained().

---

## Transformers

**URL:** https://huggingface.co/docs/transformers/v5.0.0/en/main_classes/backbones

**Contents:**
- Transformers
- Backbone
- AutoBackbone
  - class transformers.AutoBackbone
- BackboneMixin
  - class transformers.utils.BackboneMixin
    - to_dict
- BackboneConfigMixin
  - class transformers.utils.BackboneConfigMixin
    - to_dict

Transformers documentation

and get access to the augmented documentation experience

A backbone is a model used for feature extraction for higher level computer vision tasks such as object detection and image classification. Transformers provides an AutoBackbone class for initializing a Transformers backbone from pretrained model weights, and two utility classes:

timm models are loaded with the TimmBackbone and TimmBackboneConfig classes.

Backbones are supported for the following models:

Serializes this instance to a Python dictionary. Override the default to_dict() from PreTrainedConfig to include the out_features and out_indices attributes.

A Mixin to support handling the out_features and out_indices attributes for the backbone configurations.

Serializes this instance to a Python dictionary. Override the default to_dict() from PreTrainedConfig to include the out_features and out_indices attributes.

Wrapper class for timm models to be used as backbones. This enables using the timm models interchangeably with the other models in the library keeping the same API.

( backbone = None num_channels = 3 features_only = True use_pretrained_backbone = True out_indices = None freeze_batch_norm_2d = False **kwargs )

This is the configuration class to store the configuration for a timm backbone TimmBackbone.

It is used to instantiate a timm backbone model according to the specified arguments, defining the model.

Configuration objects inherit from PreTrainedConfig and can be used to control the model outputs. Read the documentation from PreTrainedConfig for more information.

---

## Transformers

**URL:** https://huggingface.co/docs/transformers/v5.0.0/en/main_classes/executorch

**Contents:**
- Transformers
- ExecuTorch
- ExecuTorch Integration
  - class transformers.TorchExportableModuleWithStaticCache
    - forward
    - transformers.convert_and_export_with_cache

Transformers documentation

and get access to the augmented documentation experience

ExecuTorch is an end-to-end solution for enabling on-device inference capabilities across mobile and edge devices including wearables, embedded devices and microcontrollers. It is part of the PyTorch ecosystem and supports the deployment of PyTorch models with a focus on portability, productivity, and performance.

ExecuTorch introduces well defined entry points to perform model, device, and/or use-case specific optimizations such as backend delegation, user-defined compiler transformations, memory planning, and more. The first step in preparing a PyTorch model for execution on an edge device using ExecuTorch is to export the model. This is achieved through the use of a PyTorch API called torch.export.

An integration point is being developed to ensure that 🤗 Transformers can be exported using torch.export. The goal of this integration is not only to enable export but also to ensure that the exported artifact can be further lowered and optimized to run efficiently in ExecuTorch, particularly for mobile and edge use cases.

( model: PreTrainedModel batch_size: int | None = None max_cache_len: int | None = None device: torch.device | None = None )

A recipe module designed to make a PreTrainedModel exportable with torch.export, specifically for decoder-only LM to StaticCache. This module ensures that the exported model is compatible with further lowering and execution in ExecuTorch.

Note: This class is specifically designed to support export process using torch.export in a way that ensures the model can be further lowered and run efficiently in ExecuTorch.

( input_ids: torch.LongTensor | None = None inputs_embeds: torch.Tensor | None = None cache_position: torch.Tensor | None = None ) → torch.Tensor

Logits output from the model.

Logits output from the model.

Forward pass of the module, which is compatible with the ExecuTorch runtime.

This forward adapter serves two primary purposes:

Making the Model torch.export-Compatible: The adapter hides unsupported objects, such as the Cache, from the graph inputs and outputs, enabling the model to be exportable using torch.export without encountering issues.

Ensuring Compatibility with ExecuTorch runtime: The adapter matches the model’s forward signature with that in executorch/extension/llm/runner, ensuring that the exported model can be executed in ExecuTorch out-of-the-box.

( model: PreTrainedModel example_input_ids: torch.Tensor | None = None example_cache_position: torch.Tensor | None = None dynamic_shapes: dict | None = None strict: bool | None = None ) → Exported program (torch.export.ExportedProgram)

Exported program (torch.export.ExportedProgram)

The exported program generated via torch.export.

The exported program generated via torch.export.

Convert a PreTrainedModel into an exportable module and export it using torch.export, ensuring the exported model is compatible with ExecuTorch.

---

## Transformers

**URL:** https://huggingface.co/docs/transformers/v5.0.0/en/model_doc/afmoe

**Contents:**
- Transformers
- AFMoE
- Key Architecture Features
- Model Architecture Details
  - Expert Routing
  - Shared Experts
  - Attention Mechanism
- AfmoeConfig
  - class transformers.AfmoeConfig
- AfmoeModel

Transformers documentation

and get access to the augmented documentation experience

This model was released on {release_date} and added to Hugging Face Transformers on 2025-11-29.

AFMoE (Arcee Foundational Mixture of Experts) is a decoder-only transformer model that extends the Llama architecture with a sparse Mixture of Experts (MoE) approach. The model combines token-choice routing with shared experts and employs several architectural innovations for efficient inference and improved performance.

AFMoE introduces several key modifications to the standard transformer architecture:

The model supports extended context lengths with RoPE embeddings and includes all standard Transformers features including Flash Attention 2, SDPA, gradient checkpointing, and quantization support.

AFMoE is particularly well-suited for scenarios requiring efficient scaling through sparsity while maintaining strong performance. The shared experts provide a stable computation baseline while routed experts enable model capacity scaling.

The example below demonstrates how to generate text with AFMoE using Pipeline or the AutoModel.

AFMoE uses token-choice routing where each token independently selects top-k experts based on router logits. The routing mechanism includes:

Unlike standard MoE models, AFMoE includes shared experts that are always activated for every token, providing:

The hybrid attention pattern alternates between:

All attention layers include Q/K normalization and output gating for improved training dynamics.

( vocab_size: int | None = 200192 hidden_size: int | None = 2048 intermediate_size: int | None = 6144 moe_intermediate_size: int | None = 1408 num_hidden_layers: int | None = 32 num_dense_layers: int | None = 1 num_attention_heads: int | None = 16 num_key_value_heads: int | None = None head_dim: int | None = 128 hidden_act: str | None = 'silu' max_position_embeddings: int | None = 16384 initializer_range: float | None = 0.02 rms_norm_eps: float | None = 1e-05 use_cache: bool | None = True tie_word_embeddings: bool | None = False rope_theta: float | None = 10000.0 rope_parameters: transformers.modeling_rope_utils.RopeParameters | dict[str, transformers.modeling_rope_utils.RopeParameters] | None = None num_experts: int | None = 64 num_experts_per_tok: int | None = 6 num_shared_experts: int | None = 2 route_scale: float | None = 1.0 global_attn_every_n_layers: int | None = 4 sliding_window: int | None = 1024 layer_types: list | None = None attention_dropout: float | None = 0.0 mup_enabled: bool | None = False eos_token_id: bool | None = None pad_token_id: bool | None = None bos_token_id: bool | None = None **kwargs )

This is the configuration class to store the configuration of a AfmoeModel. It is used to instantiate an AFMoE model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of arcee-ai/Trinity-Mini.

AFMoE is an Adaptive Feedforward MoE (Mixture of Experts) model with token-choice routing, shared experts, and a hybrid attention mechanism combining sliding window and full attention patterns.

Configuration objects inherit from PreTrainedConfig and can be used to control the model outputs. Read the documentation from PreTrainedConfig for more information.

( config: AfmoeConfig )

The bare Afmoe Model outputting raw hidden-states without any specific head on top.

This model inherits from PreTrainedModel. Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch torch.nn.Module subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

( input_ids: torch.LongTensor | None = None attention_mask: torch.Tensor | None = None inputs_embeds: torch.FloatTensor | None = None position_ids: torch.LongTensor | None = None past_key_values: transformers.cache_utils.Cache | None = None cache_position: torch.LongTensor | None = None use_cache: bool | None = None **kwargs: typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs] ) → transformers.modeling_outputs.MoeModelOutputWithPast or tuple(torch.FloatTensor)

Indices can be obtained using AutoTokenizer. See PreTrainedTokenizer.encode() and PreTrainedTokenizer.call() for details.

What are attention masks?

What are position IDs?

Only Cache instance is allowed as input, see our kv cache guide. If no past_key_values are passed, DynamicCache will be initialized by default.

The model will output the same cache format that is fed as input.

If past_key_values are used, the user is expected to input only unprocessed input_ids (those that don’t have their past key value states given to this model) of shape (batch_size, unprocessed_length) instead of all input_ids of shape (batch_size, sequence_length).

transformers.modeling_outputs.MoeModelOutputWithPast or tuple(torch.FloatTensor)

A transformers.modeling_outputs.MoeModelOutputWithPast or a tuple of torch.FloatTensor (if return_dict=False is passed or when config.return_dict=False) comprising various elements depending on the configuration (None) and inputs. last_hidden_state (torch.FloatTensor of shape (batch_size, sequence_length, hidden_size)) — Sequence of hidden-states at the output of the last layer of the model. past_key_values (Cache, optional, returned when use_cache=True is passed or when config.use_cache=True) — It is a Cache instance. For more details, see our kv cache guide. Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if config.is_encoder_decoder=True in the cross-attention blocks) that can be used (see past_key_values input) to speed up sequential decoding. hidden_states (tuple(torch.FloatTensor), optional, returned when output_hidden_states=True is passed or when config.output_hidden_states=True) — Tuple of torch.FloatTensor (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape (batch_size, sequence_length, hidden_size). Hidden-states of the model at the output of each layer plus the optional initial embedding outputs. attentions (tuple(torch.FloatTensor), optional, returned when output_attentions=True is passed or when config.output_attentions=True) — Tuple of torch.FloatTensor (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length). Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads. router_logits (tuple(torch.FloatTensor), optional, returned when output_router_probs=True and config.add_router_probs=True is passed or when config.output_router_probs=True) — Tuple of torch.FloatTensor (one for each layer) of shape (batch_size, sequence_length, num_experts). Raw router logtis (post-softmax) that are computed by MoE routers, these terms are used to compute the auxiliary loss for Mixture of Experts models.

A transformers.modeling_outputs.MoeModelOutputWithPast or a tuple of torch.FloatTensor (if return_dict=False is passed or when config.return_dict=False) comprising various elements depending on the configuration (None) and inputs.

last_hidden_state (torch.FloatTensor of shape (batch_size, sequence_length, hidden_size)) — Sequence of hidden-states at the output of the last layer of the model.

past_key_values (Cache, optional, returned when use_cache=True is passed or when config.use_cache=True) — It is a Cache instance. For more details, see our kv cache guide.

Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if config.is_encoder_decoder=True in the cross-attention blocks) that can be used (see past_key_values input) to speed up sequential decoding.

hidden_states (tuple(torch.FloatTensor), optional, returned when output_hidden_states=True is passed or when config.output_hidden_states=True) — Tuple of torch.FloatTensor (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape (batch_size, sequence_length, hidden_size).

Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.

attentions (tuple(torch.FloatTensor), optional, returned when output_attentions=True is passed or when config.output_attentions=True) — Tuple of torch.FloatTensor (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length).

Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

router_logits (tuple(torch.FloatTensor), optional, returned when output_router_probs=True and config.add_router_probs=True is passed or when config.output_router_probs=True) — Tuple of torch.FloatTensor (one for each layer) of shape (batch_size, sequence_length, num_experts).

Raw router logtis (post-softmax) that are computed by MoE routers, these terms are used to compute the auxiliary loss for Mixture of Experts models.

The AfmoeModel forward method, overrides the __call__ special method.

Although the recipe for forward pass needs to be defined within this function, one should call the Module instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

The Afmoe Model for causal language modeling.

This model inherits from PreTrainedModel. Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch torch.nn.Module subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

( input_ids: torch.LongTensor | None = None attention_mask: torch.Tensor | None = None position_ids: torch.LongTensor | None = None past_key_values: transformers.cache_utils.Cache | None = None inputs_embeds: torch.FloatTensor | None = None labels: torch.LongTensor | None = None use_cache: bool | None = None cache_position: torch.LongTensor | None = None logits_to_keep: int | torch.Tensor = 0 **kwargs: typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs] ) → transformers.modeling_outputs.CausalLMOutputWithPast or tuple(torch.FloatTensor)

Indices can be obtained using AutoTokenizer. See PreTrainedTokenizer.encode() and PreTrainedTokenizer.call() for details.

What are attention masks?

What are position IDs?

Only Cache instance is allowed as input, see our kv cache guide. If no past_key_values are passed, DynamicCache will be initialized by default.

The model will output the same cache format that is fed as input.

If past_key_values are used, the user is expected to input only unprocessed input_ids (those that don’t have their past key value states given to this model) of shape (batch_size, unprocessed_length) instead of all input_ids of shape (batch_size, sequence_length).

transformers.modeling_outputs.CausalLMOutputWithPast or tuple(torch.FloatTensor)

A transformers.modeling_outputs.CausalLMOutputWithPast or a tuple of torch.FloatTensor (if return_dict=False is passed or when config.return_dict=False) comprising various elements depending on the configuration (AfmoeConfig) and inputs. loss (torch.FloatTensor of shape (1,), optional, returned when labels is provided) — Language modeling loss (for next-token prediction). logits (torch.FloatTensor of shape (batch_size, sequence_length, config.vocab_size)) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax). past_key_values (Cache, optional, returned when use_cache=True is passed or when config.use_cache=True) — It is a Cache instance. For more details, see our kv cache guide. Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see past_key_values input) to speed up sequential decoding. hidden_states (tuple(torch.FloatTensor), optional, returned when output_hidden_states=True is passed or when config.output_hidden_states=True) — Tuple of torch.FloatTensor (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape (batch_size, sequence_length, hidden_size). Hidden-states of the model at the output of each layer plus the optional initial embedding outputs. attentions (tuple(torch.FloatTensor), optional, returned when output_attentions=True is passed or when config.output_attentions=True) — Tuple of torch.FloatTensor (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length). Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

A transformers.modeling_outputs.CausalLMOutputWithPast or a tuple of torch.FloatTensor (if return_dict=False is passed or when config.return_dict=False) comprising various elements depending on the configuration (AfmoeConfig) and inputs.

loss (torch.FloatTensor of shape (1,), optional, returned when labels is provided) — Language modeling loss (for next-token prediction).

logits (torch.FloatTensor of shape (batch_size, sequence_length, config.vocab_size)) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).

past_key_values (Cache, optional, returned when use_cache=True is passed or when config.use_cache=True) — It is a Cache instance. For more details, see our kv cache guide.

Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see past_key_values input) to speed up sequential decoding.

hidden_states (tuple(torch.FloatTensor), optional, returned when output_hidden_states=True is passed or when config.output_hidden_states=True) — Tuple of torch.FloatTensor (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape (batch_size, sequence_length, hidden_size).

Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.

attentions (tuple(torch.FloatTensor), optional, returned when output_attentions=True is passed or when config.output_attentions=True) — Tuple of torch.FloatTensor (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length).

Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

The AfmoeForCausalLM forward method, overrides the __call__ special method.

Although the recipe for forward pass needs to be defined within this function, one should call the Module instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

---
