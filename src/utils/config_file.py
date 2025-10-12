import os
from pathlib import Path
import yaml
from types import SimpleNamespace as NS

def _to_ns(d):
    return NS(**{k: _to_ns(v) if isinstance(v, dict) else v for k, v in d.items()})

# Permite sobreescribir ruta del yaml con la env var TFM_CONFIG (opcional)
_config_yaml = os.getenv("TFM_CONFIG", Path(__file__).with_name("application-config.yaml"))

with open(_config_yaml, "r", encoding="utf-8") as f:
    configYaml = _to_ns(yaml.safe_load(f))
