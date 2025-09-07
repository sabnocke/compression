from collections import namedtuple
from itertools import chain, tee, filterfalse
from pathlib import Path
from pprint import pprint
from typing import Iterable, Callable, Any, Dict, List, Union

import ffmpeg
from icecream import ic

from c_types import Collection2

Collection = namedtuple("Collection", ["path", "stream", "codec_name", "width", "height", "alive"])

type SegmentableCollection = Union[Iterable[Collection2], Dict[str, Iterable[Collection2]]]

#TODO make the stream/probe into its own dataclass/attr

def get_duration(stream: Dict[str, Any]) -> float:
    lg = float(stream.get("duration", 0))
    t = duration_string_to_seconds(stream.get("tags", {"DURATION-eng": "0:0:0.0"}).get("DURATION-eng"))
    t2 = duration_string_to_seconds(stream.get("tags", {"DURATION": "0:0:0.0"}).get("DURATION"))
    return lg + t + t2

def ffprobe(to_probe: Path, ignore_errors: bool = False):
    try:
        probe = ffmpeg.probe(to_probe)
        stream = next((stream for stream in probe["streams"] if stream["codec_type"] == "video"), None)
        return Collection2(
            to_probe,
            get_duration(stream),
            stream["coded_width"],
            stream["coded_height"],
            stream["codec_long_name"],
            stream["codec_name"],
            stream["codec_type"],
            True
        )
    except ffmpeg.Error as e:
        if not ignore_errors:
            ic(e.stderr.decode())
        return Collection2(
            to_probe, 0.0, 0, 0 , "", "", "", False
        )

def partition[T](predicate: Callable[[T], bool], iterable: Iterable[T]):
    """
    Use a predicate to partition entries into true entries and false entries.
    """
    t1, t2 = tee(iterable)
    return filter(predicate, t2), filterfalse(predicate, t1)

def seg_by_resolution_plus(collection: Iterable[Collection2]):
    index: Dict[str, List[Collection2]] = {}
    for each in collection:
        if not each:
            continue

        width = each.width
        if width in index:
            index[str(width)].append(each)
        else:
            index[str(width)] = [each]

    return index

def seg_by_codec_plus(collection: Iterable[Collection2]):
    index = {}
    for each in collection:
        if not each:
            continue

        if each.codec_long_name in index:
            index[each.codec_long_name].append(each)
        else:
            index[each.codec_long_name] = [each]
    return index


def duration_string_to_seconds(duration: str) -> float:
    if duration is None:
        return 0

    parts = duration.split(":")
    if len(parts) != 3:
        raise RuntimeError(f"Invalid duration: {duration}")

    hours, minutes, sms = parts
    seconds, milliseconds = sms.split('.')
    milliseconds = milliseconds.rstrip('0')

    total = (int(hours) * 3600 +
             int(minutes) * 60 +
             int(seconds) +
             int(milliseconds if milliseconds != '' else '0') / 1000)
    return total

def useful_stats(collection: Iterable[Collection]):
    count = 0
    duration = 0.0
    missing = 0
    for i in collection:
        if not i.alive:
            continue

        # print(i.width, end=' ')

        lg = float(i.stream.get("duration", 0))
        t = duration_string_to_seconds(i.stream.get("tags", {"DURATION-eng": "0:0:0.0"}).get("DURATION-eng"))
        t2 = duration_string_to_seconds(i.stream.get("tags", {"DURATION": "0:0:0.0"}).get("DURATION"))
        if not lg and not t and not t2:
            missing += 1
            # pprint(i.stream)
            # print(i.path, t2)
        else:
            count += 1
            duration += t + lg + t2
            # ic(lg, t , t2)

    print(f"\ntotal duration: {duration}s for {count} entries with {missing} missing entries")
    return count, missing, round(duration, 3), round(duration / (count + missing), 3) if (count + missing) > 0 else 0

def prepare(item: Collection):
    try:
        (
            ffmpeg
            .input(str(item.path))
            .filter("pad", 1920, 1080, 0, "(1080-ih)/2")
            .output(str(item.path))
            .run(overwrite_output=True)
        )
        return True
    except ffmpeg.Error as e:
        print(f"An error occurred while processing {item.path}.")
        print(f"stderr: {e.stderr.decode('utf8')}")
        return False

avi_pattern = "*.avi"
mp4_pattern = "*.mp4"
mkv_pattern = "*.mkv"
p = Path.home() / "Videos" / "Movies"
good_res = [1920, 3840]

mp4 = p.rglob(mp4_pattern)
avi = p.rglob(avi_pattern)
mkv = p.rglob(mkv_pattern)

complete, completeAlt = tee(chain(mp4, avi, mkv))
complete2, complete2alt = tee(map(lambda x: ffprobe(x, True), complete))

items = seg_by_resolution_plus(complete2)
for key, values in items.items():
    print(key)
    _r = seg_by_codec_plus(values)
    pprint(_r)




