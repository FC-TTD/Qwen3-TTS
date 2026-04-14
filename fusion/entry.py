from __future__ import annotations

import uvicorn

from fusion.config import settings
from fusion.host import app


if __name__ == "__main__":
    uvicorn.run(app, host=settings.host, port=settings.port)
