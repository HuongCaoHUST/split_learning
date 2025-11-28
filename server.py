import argparse
from src.Server import Server
import src.Log

parser = argparse.ArgumentParser(description="Split learning framework with controller.")

parser.add_argument(
    "-c", "--config",
    type=str,
    default="config.yaml",
    help="Path to the configuration YAML file"
)

args = parser.parse_args()

if __name__ == "__main__":
    server = Server(args.config)
    print("Training with configuration:", args.config)
    server.start()
    src.Log.print_with_color("Ok, ready!", "green")
