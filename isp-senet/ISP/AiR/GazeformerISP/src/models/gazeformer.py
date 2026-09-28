from ._load_canonical import load_canonical

_module = load_canonical("gazeformer.py", "_isp_senet_air_gazeformer")
for _name, _value in vars(_module).items():
    if not _name.startswith("_"):
        globals()[_name] = _value

