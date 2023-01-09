import argparse
import glob
import itertools
import logging
import math
import os
from collections import defaultdict, namedtuple
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import h5py
import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)


@dataclass
class TableLoader:
    """Loader class for individual tables."""

    file_or_path: Union[str, os.PathLike]
    column_names: List = field(init=False)
    num_rows: int = field(init=False)
    num_columns: int = field(init=False)
    table: pq.ParquetDataset = field(init=False, repr=False)
    name: str = field(init=False)

    def __post_init__(self):
        self.table = pq.ParquetDataset(self.file_or_path, use_legacy_dataset=False)
        self.num_rows = self.table._dataset.count_rows()
        self.column_names = self.table.schema.names
        self.num_columns = len(self.column_names)
        self.name = Path(self.file_or_path).name


class TableDataset(IterableDataset):
    """PyTorch iterable style dataset for single table."""

    def __init__(self, file_or_path: Union[str, os.PathLike], batch_size: int = 32):
        """Initialize dataset class.

        Args:
            file_or_path (Union[str, os.PathLike]): file name or folder containing parquet dataset.
            batch_size (int): number of rows to include in batch. Default 128.
        """
        self.batch_size = batch_size
        self.ds_table = TableLoader(file_or_path=file_or_path)
        self.table_name = Path(file_or_path).name
        self.reset()

    def reset(
        self,
    ):
        self.batches = self.ds_table.table._dataset.to_batches(batch_size=self.batch_size)
        self.batch_id = 0

    def __len__(self):
        return math.ceil(self.ds_table.num_rows / self.batch_size)

    def __iter__(
        self,
    ):
        return self

    def __next__(self):
        df_batch = next(self.batches).to_pandas()
        TableBatch = namedtuple("TableBatch", ("table_name", "batch_id", "data"))
        batch_data = {col_name: [str(c) for c in col_data.tolist()] for col_name, col_data in df_batch.items()}
        return TableBatch(table_name=self.ds_table.name, batch_id=self.batch_id + 1, data=batch_data)


@dataclass
class TableInput:
    """Wrapper class for table input data.

    Args:
        file_or_path: Union[str, os.PathLike]): File or directory containing table data.
        model_name_or_path (str): Model name or path to model weights
        num_rows_batch (int): Number of rows from a column used to produce
        average embeddings for the column. Default 1024
    """

    file_or_path: Union[str, os.PathLike]
    model_name_or_path: str
    num_rows_batch: int = 1024


@dataclass
class TableColumnEmbedOutput:
    """Base class for embedding outputs.

    Args:
        table_name (str): Name of table
        model_name (str): Name of model used to generate embeddings.
        num_rows_batch (int): Number of rows from a column used to produce average embeddings for the column.
        column_names (List[str]): Names of columns in table.
        column_name_embeddings (np.ndarray): Embeddings of column names.
        column_content_embeddings (Dict[str, np.ndarray]): Embeddings of column content
    """

    table_name: str
    model_name: str
    num_rows_batch: int
    column_names: List[str]
    column_name_embeddings: np.ndarray
    column_content_embeddings: Dict[str, np.ndarray]


class EmbedTableColumns:
    """Table Column Embedding Class."""

    def __init__(
        self,
        table_input: TableInput,
        stop_after_n: Optional[int] = None,
        model: Optional[Union[nn.Sequential, nn.Module, SentenceTransformer]] = None,
        verbose: bool = False,
    ) -> None:
        """Initialize table column embedding.

        Args:
            table_input (TableInput): Input wrapper class for table to embed.
            stop_after_n (Optional[int], optional): Number of batches to embed. Defaults to None.
            model (Optional[Union[nn.Sequential, nn.Module, SentenceTransformer]], optional):
                        Initialized SentenceTransformer model class. If None provided,
                        initializes the model from the name provided in `table_input`. Defaults to None.
            verbose (bool, optional): Boolean flag to output embedding steps.. Defaults to False.
        """

        dataset = TableDataset(file_or_path=Path(table_input.file_or_path), batch_size=table_input.num_rows_batch)
        num_columns = dataset.ds_table.num_columns
        table_name = dataset.ds_table.name
        model_name = table_input.model_name_or_path

        if Path(table_input.model_name_or_path).resolve().is_file():
            model_name = Path(table_input.model_name_or_path).name

        self.model = model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.model is None:
            embed_model_cls = SentenceTransformer
            logging.info(f"Initialized {model_name} with {embed_model_cls}")
            self.model = embed_model_cls(table_input.model_name_or_path, device=self.device)

        self.table_input = table_input
        self.model_name = model_name.replace("/", "-")
        self.table_name = table_name

        self.dataloader = DataLoader(dataset, batch_size=1, drop_last=False)
        self.stop_after_n = len(self.dataloader) if stop_after_n is None else stop_after_n
        self.verbose = verbose

        logging.info(
            f"Table Name: {table_name}, "
            f"Num rows: {dataset.ds_table.num_rows}, "
            f"Num columns: {num_columns}, "
            f"Num batches: {len(dataset)}, "
            f"Model: {model_name}"
        )

    def run(self) -> TableColumnEmbedOutput:

        self.embed_dim = self.model.get_sentence_embedding_dimension()
        column_content_embeddings = defaultdict(
            lambda: np.zeros(
                self.embed_dim,
            )
        )

        for i, batch in tqdm(
            enumerate(self.dataloader),
            desc="Per batch: Compute embeddings of each column content.",
            total=len(self.dataloader),
            disable=not self.verbose,
        ):
            for column, content in batch.data.items():
                column_contents = list(itertools.chain(*content))
                column_content_embeddings[column] += self.model.encode(column_contents, show_progress_bar=False).sum(0)

            if i + 1 > self.stop_after_n:
                break

        column_names = list(column_content_embeddings.keys())
        column_name_embeddings = self.model.encode(column_names, show_progress_bar=False)

        results = TableColumnEmbedOutput(
            table_name=self.table_name,
            model_name=self.model_name,
            num_rows_batch=self.table_input.num_rows_batch,
            column_names=column_names,
            column_name_embeddings=column_name_embeddings,
            column_content_embeddings=column_content_embeddings,
        )
        return results

    def save(self, output_dir: Union[str, os.PathLike], data: TableColumnEmbedOutput, save_fmt: str = "h5"):
        out_fname = Path(output_dir) / data.model_name / f"batch_nrows_{data.num_rows_batch}" / f"{data.table_name}"
        out_fname.parent.mkdir(exist_ok=True, parents=True)

        if "h5" in save_fmt:
            out_fname = out_fname.with_suffix(".h5")
            self.save_h5(out_fname, data)
        # else:
        #     out_fname = out_fname.with_suffix(".npz")
        #     self.save_npz(out_fname, data)

        logging.info(f"Saved to: {out_fname}")

    def save_h5(self, out_fname: Union[str, os.PathLike], results: TableColumnEmbedOutput):
        with h5py.File(str(out_fname), "w") as h5_file:
            for i, col_name in tqdm(
                enumerate(results.column_names),
                desc="Saving embeddings to hdf5.",
                total=len(results.column_names),
                disable=not self.verbose,
            ):
                h5_file.create_dataset(
                    name=col_name.replace("/", "_"),  # Replace backslash with another symbol. backslash for h5py creates new group.
                    data=np.stack(
                        [results.column_name_embeddings[i].squeeze(), results.column_content_embeddings[col_name]],
                        axis=0,
                    ),
                    compression="gzip",
                    chunks=(2, self.embed_dim),
                )


def main():
    """Create and save embeddings"""

    try:

        parser = argparse.ArgumentParser(description="Inputs and outputs")
        parser.add_argument("-i", "--input_file_or_path", type=str)
        parser.add_argument("-m", "--model_name", type=str, default="all-MiniLM-L6-v2")
        # parser.add_argument('-ms', '--models_str', type=str,
        #                     default='all-MiniLM-L6-v2 msmarco-MiniLM-L6-cos-v5 average_word_embeddings_glove.6B.300d',
        #                     help="Examples: -ms 'model1 model2 model3'")
        parser.add_argument("-n", "--num_rows_batch", type=int, default=1024)
        parser.add_argument("-s", "--stop_after_n", type=int, default=None)
        parser.add_argument("-o", "--output_dir", type=str, default="/opt/ml/processing/output")
        args = parser.parse_args()

        # models_list = args.models_str.split(' ')

        input_file_or_path = args.input_file_or_path
        model_name = args.model_name
        output_dir = args.output_dir

        directories = os.listdir(input_file_or_path)

        logging.info(f"Directoriesin {input_file_or_path} : {directories}")

        for directory in directories:

            logging.info(f"Creating and saving {model_name} embeddings for {directory}")

            table_input = TableInput(
                file_or_path=f"{input_file_or_path}/{directory}",
                model_name_or_path=model_name,
                num_rows_batch=args.num_rows_batch,
            )

            embedding_obj = EmbedTableColumns(table_input=table_input, stop_after_n=args.stop_after_n)

            result = embedding_obj.run()

            embedding_obj.save(output_dir=args.output_dir, data=result)

            logging.info(f"Success! {model_name} embeddings saved to {output_dir}.")

        logging.info("\nEmbedding creation complete.")
        logging.info(f"\nContents of {output_dir}:")
        for file in glob.glob(output_dir, recursive=True):
            logging.info(file)

    except Exception as e:
        logging.error(e, exc_info=True)


if __name__ == "__main__":
    main()
