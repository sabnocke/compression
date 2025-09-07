import codecs
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Collection2:
    path: Path
    duration: float
    width: int
    height: int
    codec_long_name: str
    codec_name: str
    codec_type: str
    alive: bool
    def is1920(self) -> bool:
        return self.width == 1920

    def is_x265(self) -> bool:
        return codecs.lookup(self.codec_long_name).name == "x265"



# more videos
# https://www.youtube.com/live/zpwGFZeeWxY?si=AfEe-hs9SQX-Q0mw
# https://youtu.be/KLuTLF3x9sA?si=8Z81s7hT-xu-7EWL
# https://youtu.be/UYmvFzDuO5k?si=SvbQY6hshysqm8RS
