import json
import time
from typing import Optional, Hashable, Any

import dask
import dask.config
import torch
import typer
import xbatcher
import xarray as xr
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from torch import multiprocessing
from typing_extensions import Annotated

from dask.cache import Cache

# comment these the next two lines out to disable Dask's cache
cache = Cache(1e10)  # 10gb cache
cache.register()


def print_json(obj):
    print(json.dumps(obj))


class XBatcherPyTorchDataset(TorchDataset):
    def __init__(self, batch_generator: xbatcher.BatchGenerator):
        self.bgen = batch_generator

    def __len__(self):
        return len(self.bgen)

    def __getitem__(self, idx):
        t0 = time.time()
        print_json(
            {
                "event": "get-batch start",
                "time": t0,
                "idx": idx,
                "pid": multiprocessing.current_process().pid,
            }
        )
        # load before stacking
        batch = self.bgen[idx].load()

        print(f"{batch=}")

        # Use to_stacked_array to stack without broadcasting,
        stacked = batch.to_stacked_array(
            new_dim="features", sample_dims=("time", "longitude", "latitude")
        ).transpose("time", "features", ...)

        print(f"{stacked=}")

        x = torch.tensor(stacked.data)
        t1 = time.time()
        print_json(
            {
                "event": "get-batch end",
                "time": t1,
                "idx": idx,
                "pid": multiprocessing.current_process().pid,
                "duration": t1 - t0,
            }
        )
        return x


def dataloader_from_gcs(variables: list | None = None, source: str ="gcs", input_steps: int = 2):
    """
    Set up a batch generator for data source `source` based on a list of input variables.

    """
    if source == "gcs":
        ds = xr.open_dataset(
            "gs://weatherbench2/datasets/era5/1959-2022-6h-128x64_equiangular_with_poles_conservative.zarr",
            engine="zarr",
            chunks={},
        )
    else:
        ds = xr.open_dataset(source, engine="zarr", chunks={})

    if variables is not None:
        ds = ds[variables]
    # some generic type hints to satisfy the signature of BatchGenerator 
    patch: dict[Hashable, int] = dict(
        time=input_steps,
    )
    overlap: dict[Hashable, int] = dict(time=1)

    bgen = xbatcher.BatchGenerator(
        ds,
        input_dims=patch,
        input_overlap=overlap,
        preload_batch=False,
    )

    dataset = XBatcherPyTorchDataset(bgen)

    return dataset


def main(
    source: str = "gcs",
    num_epochs: int = 2,
    num_batches: int = 3,
    batch_size: int = 16,
    shuffle: bool | None = None,
    num_workers: int | None = None,
    prefetch_factor: int | None = None,
    persistent_workers: bool | None = None,
    pin_memory: bool | None = None,
    train_step_time: float = 0.1,
    dask_threads: int | None = None
):
    _locals = {k: v for k, v in locals().items() if not k.startswith("_")}
    data_params: dict[str, Any] = {
        "batch_size": batch_size,
    }
    if shuffle is not None:
        data_params["shuffle"] = shuffle
    if num_workers is not None:
        data_params["num_workers"] = num_workers
        data_params["multiprocessing_context"] = "forkserver"
    if prefetch_factor is not None:
        data_params["prefetch_factor"] = prefetch_factor
    if persistent_workers is not None:
        data_params["persistent_workers"] = persistent_workers
    if pin_memory is not None:
        data_params["pin_memory"] = pin_memory
    if dask_threads is None or dask_threads <= 1:
        dask.config.set(scheduler="single-threaded")
    else:
        dask.config.set(scheduler="threads", num_workers=dask_threads)

    run_start_time = time.time()
    print_json(
        {
            "event": "run start",
            "time": run_start_time,
            "data_params": str(data_params),
            "locals": _locals,
        }
    )

    t0 = time.time()
    print_json({"event": "setup start", "time": t0})
    dataset = dataloader_from_gcs(source=source)
    training_generator = DataLoader(dataset, **data_params)
    _ = next(iter(training_generator))  # wait until dataloader is ready
    t1 = time.time()
    print_json({"event": "setup end", "time": t1, "duration": t1 - t0})

    for epoch in range(num_epochs):
        e0 = time.time()
        print_json({"event": "epoch start", "epoch": epoch, "time": e0})

        for i, sample in enumerate(training_generator):
            tt0 = time.time()
            print_json({"event": "training start", "batch": i, "time": tt0})
            #### MODEL UPDATE GOES HERE ####
            time.sleep(train_step_time)  # simulate model training
            ################################
            tt1 = time.time()
            print_json({"event": "training end", "batch": i, "time": tt1, "duration": tt1 - tt0})
            if i == num_batches - 1:
                break

        e1 = time.time()
        print_json({"event": "epoch end", "epoch": epoch, "time": e1, "duration": e1 - e0})

    run_finish_time = time.time()
    print_json(
        {"event": "run end", "time": run_finish_time, "duration": run_finish_time - run_start_time}
    )

