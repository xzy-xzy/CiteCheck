import argparse

parser = argparse.ArgumentParser( )
parser.add_argument("--model", default="gpt-4o-2024-08-06")
parser.add_argument("--aim", choices=["dev", "test"], default="dev")
parser.add_argument("--api_key", default="none")
parser.add_argument("--base_url", default=None)

config = parser.parse_args( )