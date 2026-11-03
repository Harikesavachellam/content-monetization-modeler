"""Execute every notebook cell with the Python environment running this script."""

import asyncio
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

project = Path(__file__).resolve().parents[1]
path = project / "notebooks" / "project_analysis.ipynb"
notebook = nbformat.read(path, as_version=4)
manager = KernelManager(kernel_name="python3")
manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
client = NotebookClient(notebook, km=manager, timeout=600,
                        resources={"metadata": {"path": str(project / "notebooks")}})
try:
    client.execute()
    nbformat.write(notebook, path)
    print(f"Completed all {len(notebook.cells)} cells without a cell error.")
finally:
    if manager.has_kernel:
        manager.shutdown_kernel(now=True)
