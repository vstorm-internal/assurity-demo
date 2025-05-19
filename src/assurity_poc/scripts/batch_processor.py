import argparse
import importlib
import logging
import pkgutil
from pathlib import Path
from typing import Dict, Callable
from logzero import logger

class BatchProcessor:
    def __init__(self):

        self.commands: Dict[str, Callable] = {}
        self._load_commands()

    def _load_commands(self):
        # Dynamically load all commands from the commands directory
        commands_path = Path(__file__).parent / 'commands'
        for module_info in pkgutil.iter_modules([str(commands_path)]):
            if not module_info.name.startswith('_'):
                try:
                    module = importlib.import_module(f'assurity_poc.scripts.commands.{module_info.name}')
                    if hasattr(module, 'run'):
                        self.commands[module_info.name] = module.run
                        logger.info(f"Loaded command: {module_info.name}")
                except Exception as e:
                    logger.error(f"Failed to load command {module_info.name}: {str(e)}")

    def run_command(self, command_name: str, **kwargs):
        if command_name not in self.commands:
            raise ValueError(f"Command '{command_name}' not found")
        
        logger.info(f"Running command: {command_name}")
        try:
            result = self.commands[command_name](**kwargs)
            logger.info(f"Command {command_name} completed successfully")
            return result
        except Exception as e:
            logger.error(f"Command {command_name} failed: {str(e)}")
            raise

def main():
    processor = BatchProcessor()
    
    parser = argparse.ArgumentParser(description='Batch processing command runner')
    parser.add_argument('command', choices=processor.commands.keys(),
                       help='The command to run')
    # Allow any additional arguments without validation
    args, unknown = parser.parse_known_args()
    
    # Convert the unknown arguments into kwargs
    kwargs = {}
    for arg in unknown:
        if arg.startswith('--'):
            key_value = arg[2:].split('=')
            if len(key_value) == 2:
                kwargs[key_value[0]] = key_value[1]
    
    processor.run_command(args.command, **kwargs)

if __name__ == '__main__':
    main() 