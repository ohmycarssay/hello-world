from typing import Optional, cast

import torch
from torch.utils.data import IterableDataset
from torchvision.transforms import v2
import webdataset as wds

from .utils import expand_brace_patterns


def build_wds(
    shards: list[str],
    label: bool = True,
    batch_size: int = 128,
    shuffle_size: int = 10_000,
    transform: Optional[v2.Compose] = None,
) -> wds.compat.WebDataset:
    shards = expand_brace_patterns(shards)
    transform = transform or v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ])
        
    # pipeline = wds.pipeline.DataPipeline(
    #     wds.shardlists.ResampledShards(shards),
    #     wds.shardlists.split_by_worker,
    #     wds.tariterators.tarfile_to_samples(handler=wds.handlers.reraise_exception),
    #     wds.filters.shuffle(shuffle_size),
    #     wds.filters.decode("pil"),
    #     wds.filters.map(lambda x: {"images":transform(x["png"]), "labels":torch.tensor(label, dtype=torch.bool)}),
    #     wds.filters.batched(batch_size, partial=True)
    # # )
    # return pipeline

    dataset = wds.compat.WebDataset(shards, shardshuffle=False) \
          .shuffle(shuffle_size) \
          .decode("pil") \
          .map(lambda x: {"images": transform(x.get("png") or x.get("jpg")),
                          "labels": torch.tensor(label, dtype=torch.bool)}) \
          .batched(batch_size, partial=True)
    return cast(wds.compat.WebDataset, dataset)


class DiffusionFaceDataset(IterableDataset):
    def __init__(
        self,
        AIFaceDataset3000_shards: list[str],
        CelebA_shards: list[str],
        FFHQ_shards: list[str],
        SFHQ_shards: list[str],
        shuffle_size: int,
        batch_size: int,
        transform: Optional[v2.Compose] = None,
        concat_fn=None,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.AIFaceDataset3000_dataset = build_wds(AIFaceDataset3000_shards, False, batch_size//2, shuffle_size, transform)
        self.CelebA_dataset = build_wds(CelebA_shards, False, batch_size//2, shuffle_size, transform)
        self.FFHQ_dataset = build_wds(FFHQ_shards, False, batch_size//2, shuffle_size, transform)
        self.SFHQ_dataset = build_wds(SFHQ_shards, False, batch_size//2, shuffle_size, transform)
        self.concat_fn = concat_fn or self.default_concat


    def default_concat(self, a, b):

        real_images = torch.cat([x["images"] for x in a], dim=0)
        real_labels = torch.cat([x["labels"] for x in a], dim=0)
        fake_images = torch.cat([x["images"] for x in b], dim=0)
        fake_labels = torch.cat([x["labels"] for x in b], dim=0)

        images = torch.cat([real_images, fake_images], dim=0)
        labels = torch.cat([real_labels, fake_labels], dim=0)
        return images, labels


    def __iter__(self):
    # Real datasets
        real_datasets = [
            self.CelebA_dataset,
            self.FFHQ_dataset,
    ]

    # Fake datasets
        fake_datasets = [
            self.AIFaceDataset3000_dataset,
            self.SFHQ_dataset,
    ]

        real_iters = [iter(ds) for ds in real_datasets]
        fake_iters = [iter(ds) for ds in fake_datasets]

    # 두 그룹(real/fake)을 병렬로 순회
        for batch_group in zip(*real_iters, *fake_iters):
        # 앞부분은 real, 뒷부분은 fake
            num_real = len(real_iters)
            a = batch_group[:num_real]
            b = batch_group[num_real:]
            yield self.concat_fn(a, b)
