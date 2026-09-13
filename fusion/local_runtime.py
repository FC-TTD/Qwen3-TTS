"""Both public entrypoints use this one process-local runtime."""
from fusion.config import settings
from fusion.runtime import FusionRuntime

runtime = FusionRuntime(idle_seconds=settings.idle_seconds,
                        max_pending=settings.max_pending,
                        queue_timeout=settings.queue_timeout)
