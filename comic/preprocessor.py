import os
import shutil
from os import mkdir
from pathlib import Path
import zipfile

from io import BytesIO
from collections.abc import Iterator
from PIL import Image, ImageFile

from itertools import chain
from typing import Iterable, Set, Tuple, Optional, Callable, List
from tempfile import TemporaryDirectory
from pprint import pprint
from icecream import ic
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

def parallel_resize_pad(sources: Iterable[Path], target_size: Tuple[int, int], output: Optional[Path] = None, bg_color: Tuple[int, int, int] = (0,0,0)):
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = {}
        for path in sources:
            futures[executor.submit(resize_and_pad, path, target_size)] = path


        for future in as_completed(futures):
            path = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"Exception when calling {path}: {e}")


def resize_and_pad(source: Path, target_size: Tuple[int, int], output: Optional[Path] = None, bg_color: Tuple[int, int, int] = (0,0,0)):
    try:
        image = Image.open(source)

        if image.size == target_size:
            return

        o_width, o_height = image.size
        target_width, target_height = target_size

        ratio = min(target_width / o_width, target_height / o_height)
        new_width = int(o_width * ratio)
        new_height = int(o_height * ratio)

        image = image.resize((new_width, new_height))

        canvas = Image.new('RGB', target_size, color=bg_color)

        paste_x = int((target_width - o_width) / 2)
        paste_y = int((target_height - o_height) / 2)

        canvas.paste(image, (paste_x, paste_y))

        if output:
            canvas.save(output)
        else:
            canvas.save(source)
    except OSError as e:
        print(f"[ERROR] File {source}: {e}")

def epub_image_generator(epub_path: Path) -> List[Path]:
    cname = epub_path.stem
    output = Path(epub_path).parent / ".split" / cname
    output.mkdir(parents=True, exist_ok=True)

    broken = []

    with (
        zipfile.ZipFile(epub_path) as archive,
        TemporaryDirectory() as tmp
    ):
        tmp_p = Path(tmp)

        for file in archive.namelist():
            try:
                archive.extract(file, path=tmp_p)
            except zipfile.BadZipFile as e:
                print(f"[Warning]: Skipping corrupted file {file} in {epub_path.name}: {e}")

                print(Path(file).name)
                broken.append(Path(file).name)
                # shutil.rmtree(next(tmp_p.glob(file)))
            except Exception as e:
                print(f"[Warning]: Skipping corrupted file {file} in {epub_path.name}: {e}")

        image_paths = chain(
            tmp_p.rglob("*.png"),
            tmp_p.rglob("*.jpg"),
            tmp_p.rglob("*.jpeg"),
            tmp_p.rglob("*.gif")
        )

        def filtering(image_name: Path):
            if image_name.name in broken:
                return None
            if not (output / image_name.name).exists():
                return image_name.rename(output / image_name.name)
            return output / image_name.name

        return list(filter(lambda x: x is not None, map(filtering, image_paths)))
        # return [im.rename(output / im.name) if not (output / im.name).exists() else output / im.name for im in image_paths ]


def find_max(source: Iterable[Path], collect: bool = False):
    dims: Set[Tuple[int, int]] = set()

    for image_path in source:
        image = Image.open(image_path)
        dims.add((image.width, image.height))

    if collect:
        return dims
    else:
        return max(dims, key=lambda i: i[0])[0], max(dims, key=lambda i: i[1])[1]



# with zipfile.ZipFile(next(epubs)) as archive:
#     for file in archive.namelist():
#         print(file)

if __name__ == "__main__":
    p = Path(r"C:\Users\ReWyn\global\warehouse\comics")
    epubs = p.glob("*.epub")

    collective: Set[Tuple[int, int]] = set()

    for epub in epubs:
        src = epub_image_generator(epub)
        d = find_max(src)
        collective |= d

    pprint(collective)
    print(max(collective, key=lambda i: i[0])[0], max(collective, key=lambda i: i[1])[1])
    # for item in data:
    #     print(item.width, item.height, item.split())