from pathlib import Path
import sys
import types

import clx.bootstrap as bs

PKG_DIR = Path(__file__).parent

bootstrap_context = bs.init_context({})
bs.load_file(bootstrap_context, PKG_DIR / "core.clj")
