import importlib
import os
import sys
from .base_attack import BaseAttack
from .base_model import BaseModel

def discover_plugins(plugin_folder: str):
    """
    Dynamically discover plugin classes that inherit from BaseAttack or BaseModel.
    Returns two dicts: 
       - attack_plugins = {attacker_name: attacker_instance, ...}
       - watermark_plugins = {model_name: model_instance, ...}
    """
    attack_plugins = {}
    watermark_plugins = {}

    # Optionally, ensure plugin_folder is on sys.path so imports work
    if plugin_folder not in sys.path:
        sys.path.append(plugin_folder)

    for root, dirs, files in os.walk(plugin_folder):
        for file in files:
            if file.endswith(".py"):
                module_name = file[:-3]  # strip .py
                module_path = os.path.join(root, file)
                # Convert to import path:
                # e.g. "plugins.audio_ddpm.attacker" if your plugin folder is "plugins/"
                rel_dir = os.path.relpath(root, plugin_folder)
                import_base = rel_dir.replace(os.path.sep, ".")
                if import_base == ".":
                    full_module_name = module_name
                else:
                    full_module_name = f"{import_base}.{module_name}"

                try:
                    module = importlib.import_module(full_module_name)
                    # Find classes deriving from BaseAttack
                    for attr_name in dir(module):
                        obj = getattr(module, attr_name)
                        if isinstance(obj, type) and issubclass(obj, BaseAttack) and obj is not BaseAttack:
                            instance = obj()  # instantiate
                            attack_plugins[instance.name] = instance

                        if isinstance(obj, type) and issubclass(obj, BaseModel) and obj is not BaseModel:
                            instance = obj()  # instantiate
                            watermark_plugins[instance.name] = instance

                except Exception as e:
                    print(f"Could not import plugin {full_module_name}: {e}")

    return attack_plugins, watermark_plugins