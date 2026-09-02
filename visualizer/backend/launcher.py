# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Entrypoint for the packaged RF-DETR Visualizer backend."""

import uvicorn

from visualizer.backend.app import app


def main() -> None:
    """Run the local Visualizer API used by the Tauri application."""
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")


if __name__ == "__main__":
    main()
