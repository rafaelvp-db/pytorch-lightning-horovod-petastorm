# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # Distributed Training on Databricks Platform using Pytorch Lightning, Petastorm and Horovod
# MAGIC 
# MAGIC This notebook walks through fine tuning a series of image classifiers, starting from a single device implementation all the way through to multi-node multi-device implementation capable of handling a large-scale model training. 
# MAGIC 
# MAGIC ***Disclaimer:** Main purpose of this notebook is to show how Pytorch Lightning can be used for distributed training on Databricks Platform. It is not about training a best model so we may not necessarily follow the best practices here, e.g. we keep learning rate the same even though we use larger batches in multi-GPU training.*

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Setup
# MAGIC 
# MAGIC First part of this notebook can be executed on a single node CPU cluster. A GPU cluster will be required to run a second part; minimum GPU requirements are clearly defined for each section, some cell can be run on a single GPU single node, some will work on a multi-GPU single node cluster and the rest will require multi-node GPU cluster)
# MAGIC 
# MAGIC This notebook was developed on a Databrick Runtime 10.1 with the foolowing libraries:
# MAGIC - torch: 1.9.1+cu111
# MAGIC - torchvision: 0.10.1+cu111
# MAGIC - pytorch_lightning: 1.5.8
# MAGIC - CUDA: 11.4 (`nvidia-smi`)
# MAGIC - Horovod: 0.23.0
# MAGIC 
# MAGIC ***We recommend running this notebook in your own workspace, not the Repos, even if it was cloned from GitHub (File -> Clone and save to your workspace). We've encountered errors when running this notebook directly from the Repos.***
# MAGIC   

# COMMAND ----------

import io
import numpy as np
from functools import partial
import datetime as dt
import logging

from PIL import Image

import torch, torchvision
from torch import nn
import torch.nn.functional as F
import torchmetrics.functional as FM
from torchmetrics import Accuracy
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks import TQDMProgressBar

from pyspark.sql.functions import col
from pyspark.sql.types import LongType

from petastorm import TransformSpec
from petastorm.spark import SparkDatasetConverter, make_spark_converter

print(f"Using:\n - torch: {torch.__version__}\n - torchvision: {torchvision.__version__}\n - pytorch_lightning: {pl.__version__}")

# COMMAND ----------

IS_DEV = True

DATA_DIR = "/databricks-datasets/flowers/delta"
GPU_COUNT = torch.cuda.device_count()
print(f"Found {GPU_COUNT if GPU_COUNT > 0 else 'no'} GPUs")

MAX_DEVICE_COUNT_TO_USE = 2

BATCH_SIZE = 64
MAX_EPOCH_COUNT = 15
STEPS_PER_EPOCH = 15

LR = 0.001
CLASS_COUNT = 5

SAMPLE_SIZE = 1000
print(f"Sample: size {SAMPLE_SIZE}")

EARLY_STOP_MIN_DELTA = 0.05
EARLY_STOP_PATIENCE = 3

# Set a cache directory on DBFS FUSE for intermediate data.
CACHE_DIR = "file:///dbfs/tmp/petastorm/cache"
spark.conf.set(SparkDatasetConverter.PARENT_CACHE_DIR_URL_CONF, CACHE_DIR)

# COMMAND ----------

# MAGIC %sh
# MAGIC rm -r /dbfs/tmp/petastorm/cache

# COMMAND ----------

def report_duration(action, start):
  end = dt.datetime.now()
  ds = (end - start).total_seconds()
  h, rem = divmod(ds, 3600)
  m, s = divmod(rem, 60)
  if h > 0:
      run_time = "{} hours {} minutes".format(int(h), int(m))
  elif m > 0:
      run_time = "{} minutes {} seconds".format(int(m), int(s))
  else:
      run_time = "{} seconds".format(int(s))

  msg = f"\n-- {action} completed in ***{run_time}*** at {end.strftime('%Y-%m-%d %H:%M:%S')}\n\n---------------------"
  print(msg)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Data - the flowers dataset
# MAGIC 
# MAGIC In this notebook we are using the flowers dataset from the TensorFlow team. Whereas the original dataset consists of flower photos stored under five sub-directories, one per class, here we are using a pre-processed dataset stored in Delta format.
# MAGIC 
# MAGIC This dataset contains ***3670*** images. To reduce the running time, we are using a smaller subset of the dataset for development and testing purposes in this notebook. 
# MAGIC 
# MAGIC ***Note:** The original/unprocced dataset can be also be used in the same ways, it is hosted under Databricks Datasets `dbfs:/databricks-datasets/flower_photos` for easy access.*

# COMMAND ----------

def get_data(sample_size=-1):
  df = spark.read.format("delta").load(DATA_DIR).select(["content", "label"])
  if sample_size > 0:
    df = df.limit(sample_size)
  classes = list(df.select("label").distinct().toPandas()["label"])

  assert CLASS_COUNT == len(classes)

  # Add a numeric class colunmn
  def to_class_index(class_name):
    return classes.index(class_name)

  class_index = udf(to_class_index, LongType())
  df = df.withColumn("cls_id", class_index(col("label"))).drop("label")

  train_df, val_df = df.randomSplit([0.8, 0.2], seed=12)

  # For distributed training data must have at least as many many partitions as the number of devices/processes
  train_df = train_df.repartition(MAX_DEVICE_COUNT_TO_USE)
  val_df = val_df.repartition(MAX_DEVICE_COUNT_TO_USE)

  print(f"Dataset size: {df.count()} (train: {train_df.count()}, val: {val_df.count()})")
  print(f"Labels ({CLASS_COUNT}): {classes}")
  if sample_size > 0:
    display(train_df.limit(10))
  
  return train_df, val_df

def create_spark_converters(use_sample=True):
  train_df, val_df = get_data(SAMPLE_SIZE if use_sample else -1)
  return make_spark_converter(train_df), make_spark_converter(val_df)


# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Dataframe and corresponding Petastorm wrappers need to be created here instead of inside the Pytorch Lightning model class. This is especially important for distributed training when model class instances will be created in other nodes where Spark context is not defined (Petastorm spark converter can be pickled) 

# COMMAND ----------

train_sample_converter, val_sample_converter = create_spark_converters(True)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### The Model
# MAGIC 
# MAGIC Everything has been included in the model class to show how the most part of the model logic can be encapsulated into a single class.
# MAGIC 
# MAGIC A special note about the value of parameter `num_epochs` used in `make_torch_dataloader` function. We set it to `None` (it is also a default value) to generate infinite batches of data to avoid handling the last incomplete batch. This is especially important for distributed training where we need to guarantee that the numbers of data records seen on all workers are identical per step. Given that the length of each data shard may not be identical, setting `num_epochs` to any specific number would fail to meet the guarantee and will likely result in an error.
# MAGIC 
# MAGIC Even though this is not really important for single device training, it determines the way we control epochs (training will run forever on infinite dataset which means there would be only 1 if other controls are not used), so we decided to introduce it from the beginning.
# MAGIC 
# MAGIC There is another place when this is important - the validation process. Pytorch Lightning Trainer, by default, will run a sanity validation check prior to any training, unless instructed otherwise (i.e. `num_sanity_val_steps` is set to be equal to `0`). That sanity check will initialise the validation data loader and will use those few initial batches defined by `num_sanity_val_steps`. However, because of doing so it will load the validation data loader before the first epoch but will not do so for the validation phase of the first epoch which will result in error (an attempt to read a second time from data loader which was not completed in the previous attempt). Possible workarounds is using a finite amount of epochs in `num_epochs` (e.g. `num_epochs=1` as there is no point in evaluating on repeated dataset), which is not good as it will likely result in a last batch being smaller than other batches. At the time of creating this notebook there is no way to set an equivalent of `drop_last` for the Data Loader created by `make_torch_dataloader`. The only way to work around this is to avoid doing any sanity checks (i.e. setting `num_sanity_val_steps=0`, setting it to anything else doesn't work) and using `limit_val_batches` parameter of the Trainer class.
# MAGIC 
# MAGIC A separate callback class can be used for sidecar operations like logging, etc but it is easier to keep all this within the model

# COMMAND ----------

class LitClassificationModel(pl.LightningModule):
  def __init__(self, train_converter, val_converter, class_count=CLASS_COUNT, lr=LR, logging_level=logging.INFO, device_id=0, device_count=1):
    super().__init__()
    self.lr = lr
    self.model = self.get_model(class_count, lr)
    self.train_converter = train_converter
    self.val_converter = val_converter
    self.train_dataloader_context = None
    self.val_dataloader_context = None
    self.state = {"epochs": 0}
    self.logging_level = logging_level
    self.device_id = device_id
    self.device_count = device_count

  def configure_optimizers(self):
    optimizer = torch.optim.SGD(self.model.classifier[1].parameters(), lr=self.lr, momentum=0.9)
    return optimizer

  def forward(self, x):
    x = self.model(x)
    return x
  
  def training_step(self, batch, batch_idx):
    X, y = batch["features"], batch["cls_id"]
    pred = self(X)
    loss = F.cross_entropy(pred, y)
    
    # Choosing to use step loss as a metric. Use the next (commented out) line instead if averaged epoch loss if preferred
    # self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
    self.log("train_loss", loss, prog_bar=True)
    
    if self.logging_level == logging.DEBUG:
      if batch_idx == 0:
        print(f" - [{self.device_id}] training batch size: {y.shape[0]}")
      print(f" - [{self.device_id}] training batch: {batch_idx}, loss: {loss}")
      
    return loss

  def training_step_end(self, training_step_outputs):
    if self.logging_level == logging.DEBUG:
      print(f" - [{self.device_id}] training step output: {training_step_outputs.item()}")
  
  def training_epoch_end(self, training_step_outputs):
    if self.logging_level == logging.DEBUG:
      print(f" - [{self.device_id}] training epoch output: {training_step_outputs}")
    
  def on_train_epoch_start(self):
    # No need to re-load data here as `train_dataloader` will be called on each epoch
    if self.logging_level in (logging.DEBUG, logging.INFO):
      print(f"++ [{self.device_id}] Epoch: {self.state['epochs']}")
    self.state["epochs"] += 1

  def train_dataloader(self):
    return self.get_dataloader_context(is_train=True).__enter__()
    
  def validation_step(self, batch, batch_idx):
    X, y = batch["features"], batch["cls_id"]
    pred = self(X)
    loss = F.cross_entropy(pred, y)
    acc = FM.accuracy(pred, y)

    # Roll validation up to epoch level
    self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
    self.log("val_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
    
    if self.logging_level == logging.DEBUG:
      print(f" - [{self.device_id}] val batch: {batch_idx}, size: {y.shape[0]}, loss: {loss}, acc: {acc}")

    return {"loss": loss, "acc": acc}

  def val_dataloader(self):
    return self.get_dataloader_context(is_train=False).__enter__()

  def on_train_end(self):
    # Close all readers as a best practice (plus actually helps avoid failure of the distributed training)
    self.train_dataloader_context.__exit__(None, None, None)
    self.val_dataloader_context.__exit__(None, None, None)
    
  def get_dataloader_context(self, is_train=True):
    if self.logging_level == logging.DEBUG:
      print(f" - [{self.device_id}] get_dataloader_context({'Train' if is_train else 'Val'})")
    
    # To improve performance, do the data transformation in a TransformSpec function in petastorm instead of Spark Dataframe
    if is_train:
      if self.train_dataloader_context:
        self.train_dataloader_context.__exit__(None, None, None)
      self.train_dataloader_context = self.train_converter.make_torch_dataloader(transform_spec=self._get_transform_spec(is_train=True), num_epochs=None,
                                                                                 cur_shard=self.device_id, shard_count=self.device_count, batch_size=BATCH_SIZE*self.device_count)
      return self.train_dataloader_context
    else:
      # https://petastorm.readthedocs.io/en/latest/_modules/petastorm/spark/spark_dataset_converter.html#SparkDatasetConverter.make_torch_dataloader
      if self.val_dataloader_context:
        self.val_dataloader_context.__exit__(None, None, None)
      self.val_dataloader_context = self.val_converter.make_torch_dataloader(transform_spec=self._get_transform_spec(is_train=False), num_epochs=None, 
                                                                             cur_shard=self.device_id, shard_count=self.device_count,  batch_size=BATCH_SIZE*self.device_count)
      return self.val_dataloader_context
  
  def get_model(self, class_count, lr):
    model = torchvision.models.mobilenet_v2(pretrained=True)

    # Freeze parameters in the feature extraction layers and replace the last layer
    for param in model.parameters():
      param.requires_grad = False

    # New modules have `requires_grad = True` by default
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, class_count)
    return model
    
  def _transform_row(self, is_train, batch):
    transformers = [transforms.Lambda(lambda x: Image.open(io.BytesIO(x)))]
    if is_train:
      transformers.extend([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
      ])
    else:
      transformers.extend([
        transforms.Resize(256),
        transforms.CenterCrop(224),
      ])
    transformers.extend([
      transforms.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    trans = transforms.Compose(transformers)

    batch["features"] = batch["content"].map(lambda x: trans(x).numpy())
    batch = batch.drop(labels=["content"], axis=1)
    return batch

  def _get_transform_spec(self, is_train=True):
    # The output shape of the `TransformSpec` is not automatically known by petastorm, so we need to specify the shape for new columns in `edit_fields`
    # and specify the order of the output columns in `selected_fields`
    return TransformSpec(partial(self._transform_row, is_train), 
                         edit_fields=[("features", np.float32, (3, 224, 224), False)], 
                         selected_fields=["features", "cls_id"])


# COMMAND ----------

def train(train_converter=train_sample_converter, val_converter=val_sample_converter, gpus=0, strategy=None, device_id=0, device_count=1, logging_level=logging.INFO):
  
  start = dt.datetime.now()

  if device_id == 0:
    print(f"Train on {str(max(gpus, device_count)) + ' GPU' + ('s' if gpus > 1 or device_count > 1 else '') if gpus > 0  else 'CPU'}:")
    print(f" - max epoch count: {MAX_EPOCH_COUNT}\n - batch size: {BATCH_SIZE*device_count}\n - steps per epoch: {STEPS_PER_EPOCH}\n - sample size: {SAMPLE_SIZE}")
    print(f" - start time: {start.strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n======================\n")
  
  # Use check_on_train_epoch_end=True to evaluate at the end of each epoch
  verbose = True if device_id == 0 else False
  stopper = EarlyStopping(monitor="val_loss", min_delta=EARLY_STOP_MIN_DELTA, patience=EARLY_STOP_PATIENCE, verbose=verbose, mode='min', check_on_train_epoch_end=True)
  callbacks = [stopper]
  
  # Add checkpointing if needed
  # checkpointer = ModelCheckpoint(monitor='val_loss', mode="min", save_top_k=1, verbose=verbose)
  # callbacks.append(checkpointer)

  # You could also use an additinal progress bar but default progress reporting was sufficient. Uncomment next line if desired
  # callbacks.append(TQDMProgressBar(refresh_rate=STEPS_PER_EPOCH, process_position=0))
  
  # We could use `on_train_batch_start` to control epoch sizes as shown in the link below but it's cleaner when done here with `limit_train_batches` parameter
  # https://pytorch-lightning.readthedocs.io/en/stable/_modules/pytorch_lightning/core/hooks.html#ModelHooks.on_train_batch_start
  trainer = pl.Trainer(
      accelerator = "gpu",
      devices=gpus,
      max_epochs=MAX_EPOCH_COUNT,
      limit_train_batches=STEPS_PER_EPOCH,  # this is the way to end the epoch, otherwise they will continue forever because the Train dataloader is producing infinite number of batches
      log_every_n_steps=1,
      val_check_interval=STEPS_PER_EPOCH,  # this value must be the same as `limit_train_batches`
      num_sanity_val_steps=0,  # this must be zero to prevent a Petastorm error about Data Loader not being read completely
      limit_val_batches=1,  # any value would work here but there is point in validating on repeated set of data so this should be smaller than total number of batches in Validation dataset
      reload_dataloaders_every_n_epochs=1,  # need to set this to 1
      strategy=strategy,
      callbacks=callbacks,
      default_root_dir='/tmp/lightning_logs'
  )

  model = LitClassificationModel(train_converter, val_converter, device_id=device_id, device_count=device_count, logging_level=logging_level)
  trainer.fit(model)

  if device_id == 0:
    report_duration(f"Training", start)
  
  return model.model if device_id == 0 else None

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Option 1: Use Horovod
# MAGIC 
# MAGIC ***Note:** This section will require a cluster with multiple GPUs, a single node cluster with multiple GPUs instace will be sufficient (e.g. p3.8xlarge on AWS or equivalent on other cloud providers)*

# COMMAND ----------

from pytorch_lightning.utilities import _HOROVOD_AVAILABLE
if _HOROVOD_AVAILABLE:
  print("Horovod is available")
  import horovod
  import horovod.torch as hvd
  from pytorch_lightning.plugins.training_type.horovod import HorovodPlugin
  print(f"Horovod: {horovod.__version__}")


# COMMAND ----------

def train_hvd():
  hvd.init()
  return train(gpus=4, strategy="horovod", device_id=hvd.rank(), device_count=hvd.size())

# Single node, 4 GPUs
hvd_model = horovod.run(train_hvd, np=1)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### 4 GPUs on Single Node Training Results
# MAGIC 
# MAGIC Train on 4 GPUs:
# MAGIC - max epoch count: 15
# MAGIC - batch size: 64
# MAGIC - steps per epoch: 15
# MAGIC - sample size: 1000
# MAGIC - start time: 2022-11-16 07:56:31
# MAGIC 
# MAGIC ======================
# MAGIC 
# MAGIC Epoch 14: 100% 16/16 [00:06<00:00,  2.38it/s, loss=0.415, v_num=0, train_loss=0.509, val_loss=0.474, val_acc=0.875][1,0]
# MAGIC 
# MAGIC -- Training completed in ***2 minutes 49 seconds*** at 2022-11-16 07:59:21
# MAGIC 
# MAGIC 
# MAGIC ***Observations:*** 
# MAGIC   - we get a better training time than single GPU training, however loss is decreasing slower
# MAGIC   - we are getting a better progress bar reporting in this training, must be Horovod's doing

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Clean Up

# COMMAND ----------

train_sample_converter.delete()
val_sample_converter.delete()
