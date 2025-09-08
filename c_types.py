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

    def is3840(self) -> bool:
        return self.width == 3840

    def is_x265(self) -> bool:
        return self.codec_long_name == "H.265 / HEVC (High Efficiency Video Coding)"

    def is_x264(self) -> bool:
        return self.codec_long_name == "H.264 / AVC / MPEG-4 AVC / MPEG-4 part 10"

    def is_vp9(self) -> bool:
        return self.codec_long_name == "Google VP9"

    def is_av1(self) -> bool:
        return self.codec_long_name == "AOMedia Video 1"

    def is_mpeg4(self) -> bool:
        return self.codec_long_name == "MPEG-4 Part 2"

    def is_mpeg2(self) -> bool:
        return self.codec_long_name == "MPEG-2 Video"

    def is_prores(self) -> bool:
        return self.codec_long_name == "Apple ProRes"

    @property
    def is_1920_x264(self) -> bool:
        return self.is1920() and self.is_x264()

    @property
    def is_3840_x265(self) -> bool:
        return self.is3840() and self.is_x265()

    @property
    def c_name(self) -> str:
        name = self.path.__str__().rsplit("\\", 1)[-1]
        return name

    def is_height(self, height: int) -> bool:
        return self.height == height

# more videos
# https://www.youtube.com/live/zpwGFZeeWxY?si=AfEe-hs9SQX-Q0mw
# https://youtu.be/KLuTLF3x9sA?si=8Z81s7hT-xu-7EWL
# https://youtu.be/UYmvFzDuO5k?si=SvbQY6hshysqm8RS
