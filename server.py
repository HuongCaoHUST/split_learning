import argparse

from src.Server import Server
import src.Log

parser = argparse.ArgumentParser(description="Split learning framework with controller.")

args = parser.parse_args()

if __name__ == "__main__":
    server = Server('config.yaml')
    server.start()
    src.Log.print_with_color("Ok, ready!", "green")
