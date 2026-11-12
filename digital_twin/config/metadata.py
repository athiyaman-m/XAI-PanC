import json
import os
from .config import MODELS_DIR


class ModelMetadata:
    _instance = None
    _data: dict | None = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load(self):
        if self._data is not None:
            return self._data
        path = os.path.join(MODELS_DIR, "metadata.json")
        if os.path.isfile(path):
            with open(path) as f:
                self._data = json.load(f)
        else:
            self._data = {}
        return self._data

    @property
    def data(self) -> dict:
        if self._data is None:
            self.load()
        return self._data or {}

    def get(self, key, default=None):
        return self.data.get(key, default)
