"""Small source → sampler → detector orchestration; no tracking or temporal logic."""

from .event_schema import FrameResult
from .video_source import VideoSourceError


class VideoBehaviorPipeline:
    def __init__(self, source, sampler, adapter):
        self.source = source
        self.sampler = sampler
        self.adapter = adapter

    def results(self):
        try:
            with self.source:
                while True:
                    packet = self.source.read()
                    if packet is None:
                        return
                    if self.sampler.should_sample(packet.metadata.timestamp_ms):
                        yield self.adapter.process(packet.frame, packet.metadata)
        except VideoSourceError as exc:
            yield FrameResult(status="read_error", metadata=None, error=str(exc))
