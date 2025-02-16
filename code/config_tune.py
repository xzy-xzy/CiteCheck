import argparse

parser = argparse.ArgumentParser( )
parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
parser.add_argument("--loc", default="")

config = parser.parse_args( )